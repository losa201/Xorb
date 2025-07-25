package api

import (
	"time"
)

// Asset represents a target asset
type Asset struct {
	ID          string    `json:"id"`
	Target      string    `json:"target"`
	Type        string    `json:"type"`
	Description string    `json:"description"`
	Tags        []string  `json:"tags"`
	Criticality string    `json:"criticality"`
	AllowedIPs  []string  `json:"allowed_ips"`
	BlockedIPs  []string  `json:"blocked_ips"`
	Scopes      []string  `json:"scopes"`
	Status      string    `json:"status"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

// AssetFilters for listing assets
type AssetFilters struct {
	Type        string   `json:"type,omitempty"`
	Criticality string   `json:"criticality,omitempty"`
	Tags        []string `json:"tags,omitempty"`
	Status      string   `json:"status,omitempty"`
	Limit       int      `json:"limit,omitempty"`
	Offset      int      `json:"offset,omitempty"`
}

// AssetUpdate for updating assets
type AssetUpdate struct {
	Description string   `json:"description,omitempty"`
	Tags        []string `json:"tags,omitempty"`
	Criticality string   `json:"criticality,omitempty"`
	AddTags     []string `json:"add_tags,omitempty"`
	RemoveTags  []string `json:"remove_tags,omitempty"`
}

// Scan represents a security scan
type Scan struct {
	ID             string            `json:"id"`
	AssetIDs       []string          `json:"asset_ids"`
	Type           string            `json:"type"`
	Template       string            `json:"template"`
	Profile        string            `json:"profile"`
	Agents         []string          `json:"agents"`
	Status         string            `json:"status"`
	Progress       int               `json:"progress"`
	Description    string            `json:"description"`
	Tags           []string          `json:"tags"`
	Timeout        int               `json:"timeout"`
	FindingsCount  int               `json:"findings_count"`
	ErrorMessage   string            `json:"error_message,omitempty"`
	StartedAt      *time.Time        `json:"started_at,omitempty"`
	CompletedAt    *time.Time        `json:"completed_at,omitempty"`
	CreatedAt      time.Time         `json:"created_at"`
	UpdatedAt      time.Time         `json:"updated_at"`
	Configuration  map[string]interface{} `json:"configuration,omitempty"`
}

// ScanRequest for starting scans
type ScanRequest struct {
	AssetIDs    []string `json:"asset_ids"`
	Type        string   `json:"type"`
	Template    string   `json:"template,omitempty"`
	Profile     string   `json:"profile"`
	Agents      []string `json:"agents,omitempty"`
	Timeout     int      `json:"timeout"`
	Description string   `json:"description,omitempty"`
	Tags        []string `json:"tags,omitempty"`
}

// ScanFilters for listing scans
type ScanFilters struct {
	Status    string   `json:"status,omitempty"`
	Type      string   `json:"type,omitempty"`
	AssetID   string   `json:"asset_id,omitempty"`
	Tags      []string `json:"tags,omitempty"`
	Limit     int      `json:"limit,omitempty"`
	Offset    int      `json:"offset,omitempty"`
}

// ScanTemplate represents scan templates
type ScanTemplate struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Type        string                 `json:"type"`
	Agents      []string               `json:"agents"`
	Config      map[string]interface{} `json:"config"`
	Tags        []string               `json:"tags"`
}

// ScanLog represents scan log entries
type ScanLog struct {
	ID        string    `json:"id"`
	ScanID    string    `json:"scan_id"`
	Level     string    `json:"level"`
	Message   string    `json:"message"`
	Agent     string    `json:"agent,omitempty"`
	Timestamp time.Time `json:"timestamp"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// Bounty represents a bug bounty program
type Bounty struct {
	ID          string     `json:"id"`
	Name        string     `json:"name"`
	Description string     `json:"description"`
	AssetIDs    []string   `json:"asset_ids"`
	Status      string     `json:"status"`
	StartDate   *time.Time `json:"start_date,omitempty"`
	EndDate     *time.Time `json:"end_date,omitempty"`
	MaxPayout   float64    `json:"max_payout"`
	PayoutRules []string   `json:"payout_rules"`
	Scopes      []string   `json:"scopes"`
	OutOfScope  []string   `json:"out_of_scope"`
	Tags        []string   `json:"tags"`
	Private     bool       `json:"private"`
	AllowedIPs  []string   `json:"allowed_ips"`
	Rules       string     `json:"rules"`
	Statistics  BountyStats `json:"statistics"`
	CreatedAt   time.Time  `json:"created_at"`
	UpdatedAt   time.Time  `json:"updated_at"`
}

// BountyStats contains bounty statistics
type BountyStats struct {
	TotalFindings    int     `json:"total_findings"`
	AcceptedFindings int     `json:"accepted_findings"`
	TotalPayout      float64 `json:"total_payout"`
	ParticipantCount int     `json:"participant_count"`
}

// BountyRequest for creating bounties
type BountyRequest struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	AssetIDs    []string   `json:"asset_ids"`
	StartDate   *time.Time `json:"start_date,omitempty"`
	EndDate     *time.Time `json:"end_date,omitempty"`
	MaxPayout   float64    `json:"max_payout"`
	PayoutRules []string   `json:"payout_rules,omitempty"`
	Scopes      []string   `json:"scopes,omitempty"`
	OutOfScope  []string   `json:"out_of_scope,omitempty"`
	Tags        []string   `json:"tags,omitempty"`
	Private     bool       `json:"private"`
	AllowedIPs  []string   `json:"allowed_ips,omitempty"`
	Rules       string     `json:"rules,omitempty"`
}

// BountyFilters for listing bounties
type BountyFilters struct {
	Status  string   `json:"status,omitempty"`
	Tags    []string `json:"tags,omitempty"`
	Private bool     `json:"private,omitempty"`
	Limit   int      `json:"limit,omitempty"`
	Offset  int      `json:"offset,omitempty"`
}

// BountyUpdate for updating bounties
type BountyUpdate struct {
	Name         string     `json:"name,omitempty"`
	Description  string     `json:"description,omitempty"`
	EndDate      *time.Time `json:"end_date,omitempty"`
	MaxPayout    *float64   `json:"max_payout,omitempty"`
	AddScopes    []string   `json:"add_scopes,omitempty"`
	RemoveScopes []string   `json:"remove_scopes,omitempty"`
	AddTags      []string   `json:"add_tags,omitempty"`
	RemoveTags   []string   `json:"remove_tags,omitempty"`
}

// Finding represents a security finding
type Finding struct {
	ID          string    `json:"id"`
	ScanID      string    `json:"scan_id,omitempty"`
	BountyID    string    `json:"bounty_id,omitempty"`
	AssetID     string    `json:"asset_id"`
	Title       string    `json:"title"`
	Description string    `json:"description"`
	Severity    string    `json:"severity"`
	Status      string    `json:"status"`
	CWEID       string    `json:"cwe_id,omitempty"`
	CVEID       string    `json:"cve_id,omitempty"`
	CVSSScore   float64   `json:"cvss_score,omitempty"`
	Impact      string    `json:"impact,omitempty"`
	PoC         string    `json:"poc,omitempty"`
	Remediation string    `json:"remediation,omitempty"`
	Tags        []string  `json:"tags"`
	Attachments []string  `json:"attachments,omitempty"`
	RawOutput   string    `json:"raw_output,omitempty"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

// FindingsFilters for listing findings
type FindingsFilters struct {
	ScanID    string   `json:"scan_id,omitempty"`
	BountyID  string   `json:"bounty_id,omitempty"`
	AssetID   string   `json:"asset_id,omitempty"`
	Severity  string   `json:"severity,omitempty"`
	Status    string   `json:"status,omitempty"`
	Tags      []string `json:"tags,omitempty"`
	Limit     int      `json:"limit,omitempty"`
	Offset    int      `json:"offset,omitempty"`
	SortBy    string   `json:"sort_by,omitempty"`
	SortOrder string   `json:"sort_order,omitempty"`
}

// VulnerabilitySubmission for bounty submissions
type VulnerabilitySubmission struct {
	BountyID    string   `json:"bounty_id"`
	Title       string   `json:"title"`
	Description string   `json:"description"`
	Severity    string   `json:"severity"`
	PoC         string   `json:"poc,omitempty"`
	Impact      string   `json:"impact,omitempty"`
	Attachments []string `json:"attachments,omitempty"`
	CWEID       string   `json:"cwe_id,omitempty"`
}

// FindingsExportRequest for exporting findings
type FindingsExportRequest struct {
	ScanID      string `json:"scan_id,omitempty"`
	BountyID    string `json:"bounty_id,omitempty"`
	AssetID     string `json:"asset_id,omitempty"`
	Severity    string `json:"severity,omitempty"`
	Status      string `json:"status,omitempty"`
	Format      string `json:"format"`
	IncludeRaw  bool   `json:"include_raw"`
	IncludePoC  bool   `json:"include_poc"`
	MinSeverity string `json:"min_severity,omitempty"`
}

// FindingsTemplateExportRequest for template-based exports
type FindingsTemplateExportRequest struct {
	Template   string   `json:"template"`
	Filters    []string `json:"filters,omitempty"`
	Format     string   `json:"format"`
	OutputFile string   `json:"output_file,omitempty"`
}

// FindingsStatsRequest for getting findings statistics
type FindingsStatsRequest struct {
	ScanID   string `json:"scan_id,omitempty"`
	BountyID string `json:"bounty_id,omitempty"`
	AssetID  string `json:"asset_id,omitempty"`
	Period   string `json:"period,omitempty"`
}

// FindingsStats contains findings statistics
type FindingsStats struct {
	Total          int                    `json:"total"`
	BySeverity     map[string]int         `json:"by_severity"`
	ByStatus       map[string]int         `json:"by_status"`
	ByAsset        map[string]int         `json:"by_asset"`
	TrendData      []FindingsTrendPoint   `json:"trend_data"`
	TopCWEs        []CWEStats             `json:"top_cwes"`
	ResolutionTime map[string]float64     `json:"resolution_time"`
}

// FindingsTrendPoint represents trend data
type FindingsTrendPoint struct {
	Date  time.Time `json:"date"`
	Count int       `json:"count"`
}

// CWEStats represents CWE statistics
type CWEStats struct {
	CWEID string `json:"cwe_id"`
	Count int    `json:"count"`
	Name  string `json:"name"`
}

// Payout represents bounty payouts
type Payout struct {
	ID          string    `json:"id"`
	BountyID    string    `json:"bounty_id"`
	FindingID   string    `json:"finding_id"`
	Researcher  string    `json:"researcher"`
	Amount      float64   `json:"amount"`
	Status      string    `json:"status"`
	Description string    `json:"description"`
	CreatedAt   time.Time `json:"created_at"`
	PaidAt      *time.Time `json:"paid_at,omitempty"`
}

// PayoutFilters for listing payouts
type PayoutFilters struct {
	BountyID string `json:"bounty_id,omitempty"`
	Status   string `json:"status,omitempty"`
	Limit    int    `json:"limit,omitempty"`
	Offset   int    `json:"offset,omitempty"`
}