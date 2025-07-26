package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"time"
	"strings"
	"bufio"

	"github.com/gorilla/mux"
	"github.com/nats-io/nats.go"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/redis/go-redis/v9"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

// Phase 5.2 Required Metrics
var (
	// Required metric: scan_duration_seconds
	scanDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "scan_duration_seconds",
			Help:    "Duration of scan operations",
			Buckets: prometheus.ExponentialBuckets(1, 2, 10),
		},
		[]string{"scanner_type", "target_type"},
	)
	
	// Required metric: scan_exit_code_total
	scanExitCodeTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "scan_exit_code_total",
			Help: "Total scan operations by exit code",
		},
		[]string{"scanner_type", "exit_code"},
	)
	
	// Additional metrics for enhanced monitoring
	scansTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "xorb_scanner_scans_total",
			Help: "Total number of scans performed",
		},
		[]string{"status", "severity", "scanner_type"},
	)
	
	findingsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "xorb_scanner_findings_total",
			Help: "Total number of findings discovered",
		},
		[]string{"severity", "category", "scanner_type"},
	)
	
	activeScans = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "xorb_scanner_active_scans",
			Help: "Number of currently active scans",
		},
		[]string{"scanner_type"},
	)
	
	scannerVersionInfo = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "xorb_scanner_version_info",
			Help: "Scanner version information",
		},
		[]string{"scanner_type", "version", "fingerprint"},
	)
)

func init() {
	prometheus.MustRegister(scansTotal)
	prometheus.MustRegister(scanDuration)
	prometheus.MustRegister(scanExitCodeTotal)
	prometheus.MustRegister(findingsTotal)
	prometheus.MustRegister(activeScans)
	prometheus.MustRegister(scannerVersionInfo)
}

// ScanRequest represents a scan request
type ScanRequest struct {
	ID            string            `json:"id"`
	OrganizationID string           `json:"organization_id"`
	Targets       []string          `json:"targets"`
	Templates     []string          `json:"templates,omitempty"`
	Options       map[string]string `json:"options,omitempty"`
	Timeout       int               `json:"timeout,omitempty"`
	Priority      int               `json:"priority,omitempty"`
}

// ScanResult represents the result of a scan with Phase 5.2 enhancements
type ScanResult struct {
	ID            string    `json:"id"`
	ScanID        string    `json:"scan_id"`
	Target        string    `json:"target"`
	TemplateID    string    `json:"template_id"`
	Info          Info      `json:"info"`
	Severity      string    `json:"severity"`
	Description   string    `json:"description"`
	Timestamp     time.Time `json:"timestamp"`
	Raw           string    `json:"raw"`
	ScannerType   string    `json:"scanner_type"`   // Phase 5.2: Track scanner type
	ExitCode      int       `json:"exit_code"`      // Phase 5.2: Track exit codes
	Duration      float64   `json:"duration_seconds"` // Phase 5.2: Scan duration
	Version       ScannerVersionInfo `json:"version"` // Phase 5.2: Version fingerprint
}

// ScannerVersionInfo represents scanner version fingerprint (Phase 5.2)
type ScannerVersionInfo struct {
	ScannerType     string `json:"scanner_type"`
	ScannerVersion  string `json:"scanner_version"`
	ServiceVersion  string `json:"service_version"`
	Fingerprint     string `json:"fingerprint"`
	BuildTimestamp  string `json:"build_timestamp"`
}

// Info represents template information
type Info struct {
	Name        string   `json:"name"`
	Author      []string `json:"author"`
	Severity    string   `json:"severity"`
	Description string   `json:"description"`
	Tags        []string `json:"tags"`
}

// NucleiResult represents the JSON output from nuclei CLI
type NucleiResult struct {
	TemplateID   string `json:"template-id"`
	Info         struct {
		Name        string   `json:"name"`
		Author      []string `json:"author"`
		Severity    string   `json:"severity"`
		Description string   `json:"description"`
		Tags        []string `json:"tags"`
	} `json:"info"`
	Type      string `json:"type"`
	Host      string `json:"host"`
	Matched   string `json:"matched-at"`
	Timestamp string `json:"timestamp"`
	Raw       string `json:"raw,omitempty"`
}

// Scanner represents the nuclei scanner service
type Scanner struct {
	db          *gorm.DB
	redis       *redis.Client
	nats        *nats.Conn
	activeScans map[string]context.CancelFunc
}

// NewScanner creates a new scanner instance
func NewScanner() (*Scanner, error) {
	// Database connection
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		dbURL = "postgresql://xorb:xorb_secure_2024@postgres:5432/xorb_ptaas"
	}
	
	db, err := gorm.Open(postgres.Open(dbURL), &gorm.Config{})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %v", err)
	}
	
	// Redis connection
	redisURL := os.Getenv("REDIS_URL")
	if redisURL == "" {
		redisURL = "redis://redis:6379/0"
	}
	
	opt, err := redis.ParseURL(redisURL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse redis URL: %v", err)
	}
	rdb := redis.NewClient(opt)
	
	// NATS connection
	natsURL := os.Getenv("NATS_URL")
	if natsURL == "" {
		natsURL = "nats://nats:4222"
	}
	
	nc, err := nats.Connect(natsURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to NATS: %v", err)
	}
	
	scanner := &Scanner{
		db:          db,
		redis:       rdb,
		nats:        nc,
		activeScans: make(map[string]context.CancelFunc),
	}
	
	return scanner, nil
}

// Start starts the scanner service
func (s *Scanner) Start() error {
	log.Println("Starting Xorb Go Scanner Service...")
	
	// Subscribe to scan requests
	_, err := s.nats.Subscribe("scans.request", s.handleScanRequest)
	if err != nil {
		return fmt.Errorf("failed to subscribe to scan requests: %v", err)
	}
	
	// Set up HTTP server
	router := mux.NewRouter()
	
	// Health check endpoint
	router.HandleFunc("/health", s.healthCheck).Methods("GET")
	
	// Metrics endpoint
	router.Handle("/metrics", promhttp.Handler())
	
	// Scan management endpoints
	router.HandleFunc("/api/v1/scans", s.createScan).Methods("POST")
	router.HandleFunc("/api/v1/scans/{id}", s.getScan).Methods("GET")
	router.HandleFunc("/api/v1/scans/{id}/cancel", s.cancelScan).Methods("POST")
	router.HandleFunc("/api/v1/scans/{id}/results", s.getScanResults).Methods("GET")
	
	// Templates endpoint
	router.HandleFunc("/api/v1/templates", s.getTemplates).Methods("GET")
	
	port := os.Getenv("PORT")
	if port == "" {
		port = "8004"
	}
	
	log.Printf("Scanner service listening on port %s", port)
	return http.ListenAndServe(":"+port, router)
}

// handleScanRequest handles incoming scan requests from NATS
func (s *Scanner) handleScanRequest(msg *nats.Msg) {
	var req ScanRequest
	if err := json.Unmarshal(msg.Data, &req); err != nil {
		log.Printf("Failed to unmarshal scan request: %v", err)
		return
	}
	
	log.Printf("Received scan request: %s", req.ID)
	
	// Start scan in goroutine
	go s.performScan(req)
}

// performScan executes a nuclei scan using CLI
func (s *Scanner) performScan(req ScanRequest) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(req.Timeout)*time.Second)
	defer cancel()
	
	// Store cancel function for potential cancellation
	s.activeScans[req.ID] = cancel
	scannerType := "nuclei" // Default scanner type
	activeScans.WithLabelValues(scannerType).Inc()
	
	defer func() {
		delete(s.activeScans, req.ID)
		activeScans.WithLabelValues(scannerType).Dec()
	}()
	
	startTime := time.Now()
	
	// Build nuclei command
	args := []string{
		"-json",           // JSON output
		"-silent",         // Silent mode
		"-no-color",       // No color output
		"-no-meta",        // No metadata output
		"-rate-limit", "100", // Rate limiting
		"-threads", "25",     // Thread count
	}
	
	// Add targets
	for _, target := range req.Targets {
		args = append(args, "-target", target)
	}
	
	// Add templates if specified
	if len(req.Templates) > 0 {
		for _, template := range req.Templates {
			args = append(args, "-t", template)
		}
	} else {
		// Use default template categories
		args = append(args, "-t", "cves/", "-t", "vulnerabilities/", "-t", "exposures/")
	}
	
	// Apply custom options
	if req.Options != nil {
		if threads, ok := req.Options["threads"]; ok {
			args = append(args, "-threads", threads)
		}
		if rateLimit, ok := req.Options["rate_limit"]; ok {
			args = append(args, "-rate-limit", rateLimit)
		}
	}
	
	// Execute nuclei command
	cmd := exec.CommandContext(ctx, "nuclei", args...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Printf("Failed to create stdout pipe: %v", err)
		return
	}
	
	if err := cmd.Start(); err != nil {
		log.Printf("Failed to start nuclei command: %v", err)
		return
	}
	
	// Parse results line by line
	results := make([]ScanResult, 0)
	scanner := bufio.NewScanner(stdout)
	
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		
		var nucleiResult NucleiResult
		if err := json.Unmarshal([]byte(line), &nucleiResult); err != nil {
			log.Printf("Failed to parse nuclei result: %v", err)
			continue
		}
		
		// Convert to our format
		scanResult := ScanResult{
			ID:          fmt.Sprintf("%s_%d", req.ID, len(results)),
			ScanID:      req.ID,
			Target:      nucleiResult.Host,
			TemplateID:  nucleiResult.TemplateID,
			Severity:    nucleiResult.Info.Severity,
			Description: nucleiResult.Info.Description,
			Timestamp:   time.Now(),
			Raw:         nucleiResult.Raw,
			ScannerType: scannerType,
			Info: Info{
				Name:        nucleiResult.Info.Name,
				Author:      nucleiResult.Info.Author,
				Severity:    nucleiResult.Info.Severity,
				Description: nucleiResult.Info.Description,
				Tags:        nucleiResult.Info.Tags,
			},
		}
		
		results = append(results, scanResult)
		
		// Update findings metrics
		findingsTotal.WithLabelValues(scanResult.Severity, "vulnerability", scannerType).Inc()
	}
	
	// Wait for command to complete
	err = cmd.Wait()
	duration := time.Since(startTime)
	
	// Phase 5.2: Update required metrics with proper labels
	scanDuration.WithLabelValues(scannerType, "web").Observe(duration.Seconds())
	
	// Determine exit code and status
	exitCode := 0
	status := "success"
	if err != nil {
		if exitError, ok := err.(*exec.ExitError); ok {
			exitCode = exitError.ExitCode()
		} else {
			exitCode = 1
		}
		if exitCode != 0 && len(results) == 0 {
			// Only treat as error if no results and non-zero exit
			status = "error"
			log.Printf("Scan %s failed: %v", req.ID, err)
		}
	}
	
	// Phase 5.2: Track exit codes
	scanExitCodeTotal.WithLabelValues(scannerType, fmt.Sprintf("%d", exitCode)).Inc()
	
	// Update scan totals with scanner type
	scansTotal.WithLabelValues(status, "", scannerType).Inc()
	
	// Update all results with Phase 5.2 metadata
	for i := range results {
		results[i].ScannerType = scannerType
		results[i].ExitCode = exitCode
		results[i].Duration = duration.Seconds()
		results[i].Version = s.getScannerVersionInfo(scannerType)
	}
	
	// Store results in database and cache
	s.storeScanResults(req.ID, req.OrganizationID, results, status, err)
	
	// Publish results to NATS
	s.publishScanResults(req.ID, results, status, err)
	
	log.Printf("Scan %s completed with %d findings in %v", req.ID, len(results), duration)
}

// storeScanResults stores scan results in database and cache
func (s *Scanner) storeScanResults(scanID, orgID string, results []ScanResult, status string, scanErr error) {
	// Store in Redis cache for quick access
	resultsJSON, _ := json.Marshal(map[string]interface{}{
		"scan_id":         scanID,
		"organization_id": orgID,
		"status":          status,
		"results":         results,
		"timestamp":       time.Now(),
		"error":           scanErr,
	})
	
	key := fmt.Sprintf("scan_results:%s", scanID)
	s.redis.Set(context.Background(), key, resultsJSON, 24*time.Hour)
}

// publishScanResults publishes scan results to NATS with Phase 5.2 NDJSON format
func (s *Scanner) publishScanResults(scanID string, results []ScanResult, status string, scanErr error) {
	// Phase 5.2: Publish individual results as NDJSON to scan.result.* subjects
	for _, result := range results {
		// Publish to scanner-specific subject
		subject := fmt.Sprintf("scan.result.%s", result.ScannerType)
		resultJSON, _ := json.Marshal(result)
		
		if err := s.nats.Publish(subject, resultJSON); err != nil {
			log.Printf("Failed to publish result to %s: %v", subject, err)
		}
		
		// Also publish to general scan results
		s.nats.Publish("scan.results", resultJSON)
	}
	
	// Legacy batch result publication for backward compatibility
	message := map[string]interface{}{
		"scan_id":   scanID,
		"status":    status,
		"results":   results,
		"timestamp": time.Now(),
	}
	
	if scanErr != nil {
		message["error"] = scanErr.Error()
	}
	
	msgJSON, _ := json.Marshal(message)
	s.nats.Publish("scans.completed", msgJSON)
	
	log.Printf("Published %d scan results to NATS for scan %s", len(results), scanID)
}

// getScannerVersionInfo returns scanner version fingerprint (Phase 5.2)
func (s *Scanner) getScannerVersionInfo(scannerType string) ScannerVersionInfo {
	// Get nuclei version
	cmd := exec.Command("nuclei", "-version")
	output, err := cmd.Output()
	nucleiVersion := "unknown"
	if err == nil {
		nucleiVersion = strings.TrimSpace(string(output))
	}
	
	// Create deterministic fingerprint based on scanner versions
	fingerprint := fmt.Sprintf("%s_%s_%d", 
		scannerType, 
		nucleiVersion,
		time.Now().Unix()/3600) // Hourly rotation for cache busting
	
	version := ScannerVersionInfo{
		ScannerType:     scannerType,
		ScannerVersion:  nucleiVersion,
		ServiceVersion:  "5.2.0",
		Fingerprint:     fingerprint,
		BuildTimestamp:  time.Now().Format(time.RFC3339),
	}
	
	// Update Prometheus metric with version info
	scannerVersionInfo.WithLabelValues(
		scannerType, 
		version.ScannerVersion, 
		version.Fingerprint,
	).Set(1)
	
	return version
}

// HTTP Handlers
func (s *Scanner) healthCheck(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":    "healthy",
		"service":   "xorb-scanner-go",
		"version":   "1.0.0",
		"timestamp": time.Now().Format(time.RFC3339),
	})
}

func (s *Scanner) createScan(w http.ResponseWriter, r *http.Request) {
	var req ScanRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}
	
	// Generate scan ID if not provided
	if req.ID == "" {
		req.ID = fmt.Sprintf("scan_%d", time.Now().Unix())
	}
	
	// Set default timeout
	if req.Timeout == 0 {
		req.Timeout = 1800 // 30 minutes
	}
	
	// Start scan asynchronously
	go s.performScan(req)
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"scan_id": req.ID,
		"status":  "started",
		"message": "Scan initiated successfully",
	})
}

func (s *Scanner) getScan(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	scanID := vars["id"]
	
	// Get scan results from Redis
	key := fmt.Sprintf("scan_results:%s", scanID)
	result, err := s.redis.Get(context.Background(), key).Result()
	if err != nil {
		http.Error(w, "Scan not found", http.StatusNotFound)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(result))
}

func (s *Scanner) cancelScan(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	scanID := vars["id"]
	
	if cancel, exists := s.activeScans[scanID]; exists {
		cancel()
		delete(s.activeScans, scanID)
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"status":  "cancelled",
			"scan_id": scanID,
		})
	} else {
		http.Error(w, "Scan not found or already completed", http.StatusNotFound)
	}
}

func (s *Scanner) getScanResults(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	scanID := vars["id"]
	
	// Get results from cache
	key := fmt.Sprintf("scan_results:%s", scanID)
	result, err := s.redis.Get(context.Background(), key).Result()
	if err != nil {
		http.Error(w, "Scan results not found", http.StatusNotFound)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(result))
}

func (s *Scanner) getTemplates(w http.ResponseWriter, r *http.Request) {
	// Return available template categories/IDs
	templates := map[string][]string{
		"cves":              {"cves/"},
		"vulnerabilities":   {"vulnerabilities/"},
		"exposures":         {"exposures/"},
		"misconfigurations": {"misconfigurations/"},
		"technologies":      {"technologies/"},
		"workflows":         {"workflows/"},
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(templates)
}

func main() {
	scanner, err := NewScanner()
	if err != nil {
		log.Fatalf("Failed to create scanner: %v", err)
	}
	
	log.Fatal(scanner.Start())
}