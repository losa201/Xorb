package output

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/fatih/color"
	"github.com/olekukonko/tablewriter"
	"gopkg.in/yaml.v3"

	"github.com/xorb/xorbctl/internal/api"
	"github.com/xorb/xorbctl/internal/config"
	"github.com/xorb/xorbctl/internal/version"
)

var (
	// Color functions
	colorGreen   = color.New(color.FgGreen).SprintFunc()
	colorRed     = color.New(color.FgRed).SprintFunc()
	colorYellow  = color.New(color.FgYellow).SprintFunc()
	colorBlue    = color.New(color.FgBlue).SprintFunc()
	colorCyan    = color.New(color.FgCyan).SprintFunc()
	colorMagenta = color.New(color.FgMagenta).SprintFunc()
	colorBold    = color.New(color.Bold).SprintFunc()
)

// PrintSuccess prints a success message
func PrintSuccess(message string) {
	if config.IsColorOutputEnabled() {
		fmt.Printf("✓ %s\n", colorGreen(message))
	} else {
		fmt.Printf("✓ %s\n", message)
	}
}

// PrintError prints an error message
func PrintError(message string) {
	if config.IsColorOutputEnabled() {
		fmt.Fprintf(os.Stderr, "✗ %s\n", colorRed(message))
	} else {
		fmt.Fprintf(os.Stderr, "✗ %s\n", message)
	}
}

// PrintWarning prints a warning message
func PrintWarning(message string) {
	if config.IsColorOutputEnabled() {
		fmt.Printf("⚠ %s\n", colorYellow(message))
	} else {
		fmt.Printf("⚠ %s\n", message)
	}
}

// PrintInfo prints an info message
func PrintInfo(message string) {
	if config.IsColorOutputEnabled() {
		fmt.Printf("ℹ %s\n", colorBlue(message))
	} else {
		fmt.Printf("ℹ %s\n", message)
	}
}

// ConfirmAction prompts for user confirmation
func ConfirmAction(message string) (bool, error) {
	fmt.Printf("%s [y/N]: ", message)
	
	reader := bufio.NewReader(os.Stdin)
	response, err := reader.ReadString('\n')
	if err != nil {
		return false, fmt.Errorf("failed to read input: %w", err)
	}

	response = strings.ToLower(strings.TrimSpace(response))
	return response == "y" || response == "yes", nil
}

// formatOutput formats data according to the specified format
func formatOutput(data interface{}, format string) error {
	switch format {
	case "json":
		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		return encoder.Encode(data)
	case "yaml":
		encoder := yaml.NewEncoder(os.Stdout)
		return encoder.Encode(data)
	default:
		// Table format is handled by specific print functions
		return fmt.Errorf("unsupported format: %s", format)
	}
}

// PrintVersionInfo prints version information
func PrintVersionInfo(info *version.VersionInfo, format string) {
	if format != "table" {
		formatOutput(info, format)
		return
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Property", "Value"})
	table.SetBorder(false)
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)

	table.Append([]string{"Version", info.Version})
	table.Append([]string{"Git Commit", info.GitCommit})
	table.Append([]string{"Git Tag", info.GitTag})
	table.Append([]string{"Build Date", info.BuildDate})
	table.Append([]string{"Go Version", info.GoVersion})
	table.Append([]string{"Platform", info.Platform})
	table.Append([]string{"Architecture", info.Architecture})

	table.Render()
}

// PrintConfig prints configuration
func PrintConfig(cfg *config.Configuration, format string) {
	if format != "table" {
		formatOutput(cfg, format)
		return
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Key", "Value"})
	table.SetBorder(false)
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)

	table.Append([]string{"API Endpoint", cfg.APIEndpoint})
	table.Append([]string{"Default Output", cfg.DefaultOutput})
	table.Append([]string{"Verbose", strconv.FormatBool(cfg.Verbose)})
	table.Append([]string{"Timeout", strconv.Itoa(cfg.Timeout)})
	table.Append([]string{"Retry Attempts", strconv.Itoa(cfg.RetryAttempts)})
	table.Append([]string{"Color Output", strconv.FormatBool(cfg.ColorOutput)})
	table.Append([]string{"Auto Update", strconv.FormatBool(cfg.AutoUpdate)})
	table.Append([]string{"Telemetry Opt-in", strconv.FormatBool(cfg.TelemetryOpt)})

	table.Render()
}

// PrintConfigValue prints a single config value
func PrintConfigValue(key string, value interface{}, format string) {
	if format != "table" {
		data := map[string]interface{}{key: value}
		formatOutput(data, format)
		return
	}

	fmt.Printf("%s: %v\n", key, value)
}

// PrintAssets prints a list of assets
func PrintAssets(assets []*api.Asset, format string) {
	if format != "table" {
		formatOutput(assets, format)
		return
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"ID", "Target", "Type", "Criticality", "Status", "Created"})
	table.SetBorder(false)
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)

	for _, asset := range assets {
		table.Append([]string{
			asset.ID,
			asset.Target,
			asset.Type,
			asset.Criticality,
			asset.Status,
			asset.CreatedAt.Format("2006-01-02 15:04"),
		})
	}

	table.Render()
}

// PrintAsset prints a single asset
func PrintAsset(asset *api.Asset, format string) {
	if format != "table" {
		formatOutput(asset, format)
		return
	}

	PrintAssets([]*api.Asset{asset}, format)
}

// PrintAssetDetail prints detailed asset information
func PrintAssetDetail(asset *api.Asset, format string) {
	if format != "table" {
		formatOutput(asset, format)
		return
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Property", "Value"})
	table.SetBorder(false)
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)

	table.Append([]string{"ID", asset.ID})
	table.Append([]string{"Target", asset.Target})
	table.Append([]string{"Type", asset.Type})
	table.Append([]string{"Description", asset.Description})
	table.Append([]string{"Criticality", asset.Criticality})
	table.Append([]string{"Status", asset.Status})
	table.Append([]string{"Tags", strings.Join(asset.Tags, ", ")})
	table.Append([]string{"Scopes", strings.Join(asset.Scopes, ", ")})
	table.Append([]string{"Allowed IPs", strings.Join(asset.AllowedIPs, ", ")})
	table.Append([]string{"Blocked IPs", strings.Join(asset.BlockedIPs, ", ")})
	table.Append([]string{"Created", asset.CreatedAt.Format("2006-01-02 15:04:05")})
	table.Append([]string{"Updated", asset.UpdatedAt.Format("2006-01-02 15:04:05")})

	table.Render()
}

// PrintScans prints a list of scans
func PrintScans(scans []*api.Scan, format string) {
	if format != "table" {
		formatOutput(scans, format)
		return
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"ID", "Type", "Status", "Progress", "Findings", "Created"})
	table.SetBorder(false)
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)

	for _, scan := range scans {
		progress := fmt.Sprintf("%d%%", scan.Progress)
		if scan.Status == "completed" {
			progress = "100%"
		}

		table.Append([]string{
			scan.ID,
			scan.Type,
			scan.Status,
			progress,
			strconv.Itoa(scan.FindingsCount),
			scan.CreatedAt.Format("2006-01-02 15:04"),
		})
	}

	table.Render()
}

// PrintScan prints a single scan
func PrintScan(scan *api.Scan, format string) {
	if format != "table" {
		formatOutput(scan, format)
		return
	}

	PrintScans([]*api.Scan{scan}, format)
}

// PrintScanDetail prints detailed scan information
func PrintScanDetail(scan *api.Scan, format string) {
	if format != "table" {
		formatOutput(scan, format)
		return
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Property", "Value"})
	table.SetBorder(false)
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)

	table.Append([]string{"ID", scan.ID})
	table.Append([]string{"Type", scan.Type})
	table.Append([]string{"Status", scan.Status})
	table.Append([]string{"Progress", fmt.Sprintf("%d%%", scan.Progress)})
	table.Append([]string{"Description", scan.Description})
	table.Append([]string{"Template", scan.Template})
	table.Append([]string{"Profile", scan.Profile})
	table.Append([]string{"Agents", strings.Join(scan.Agents, ", ")})
	table.Append([]string{"Asset IDs", strings.Join(scan.AssetIDs, ", ")})
	table.Append([]string{"Tags", strings.Join(scan.Tags, ", ")})
	table.Append([]string{"Findings Count", strconv.Itoa(scan.FindingsCount)})
	table.Append([]string{"Timeout", strconv.Itoa(scan.Timeout)})
	table.Append([]string{"Created", scan.CreatedAt.Format("2006-01-02 15:04:05")})
	
	if scan.StartedAt != nil {
		table.Append([]string{"Started", scan.StartedAt.Format("2006-01-02 15:04:05")})
	}
	if scan.CompletedAt != nil {
		table.Append([]string{"Completed", scan.CompletedAt.Format("2006-01-02 15:04:05")})
	}
	if scan.ErrorMessage != "" {
		table.Append([]string{"Error", scan.ErrorMessage})
	}

	table.Render()
}

// PrintScanLogs prints scan logs
func PrintScanLogs(logs []*api.ScanLog, format string) {
	if format != "table" {
		formatOutput(logs, format)
		return
	}

	for _, log := range logs {
		timestamp := log.Timestamp.Format("15:04:05")
		level := strings.ToUpper(log.Level)
		
		if config.IsColorOutputEnabled() {
			switch log.Level {
			case "error":
				level = colorRed(level)
			case "warn":
				level = colorYellow(level)
			case "info":
				level = colorBlue(level)
			case "debug":
				level = colorCyan(level)
			}
		}

		agent := ""
		if log.Agent != "" {
			agent = "[" + log.Agent + "] "
		}

		fmt.Printf("%s [%s] %s%s\n", timestamp, level, agent, log.Message)
	}
}

// PrintScanTemplates prints scan templates
func PrintScanTemplates(templates []*api.ScanTemplate, format string) {
	if format != "table" {
		formatOutput(templates, format)
		return
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"ID", "Name", "Type", "Agents", "Description"})
	table.SetBorder(false)
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)

	for _, template := range templates {
		table.Append([]string{
			template.ID,
			template.Name,
			template.Type,
			strings.Join(template.Agents, ", "),
			template.Description,
		})
	}

	table.Render()
}

// PrintLogEntry prints a single log entry (for following logs)
func PrintLogEntry(log *api.ScanLog) {
	timestamp := log.Timestamp.Format("15:04:05")
	level := strings.ToUpper(log.Level)
	
	if config.IsColorOutputEnabled() {
		switch log.Level {
		case "error":
			level = colorRed(level)
		case "warn":
			level = colorYellow(level)
		case "info":
			level = colorBlue(level)
		case "debug":
			level = colorCyan(level)
		}
	}

	agent := ""
	if log.Agent != "" {
		agent = "[" + log.Agent + "] "
	}

	fmt.Printf("%s [%s] %s%s\n", timestamp, level, agent, log.Message)
}

// PrintBounties prints a list of bounties
func PrintBounties(bounties []*api.Bounty, format string) {
	if format != "table" {
		formatOutput(bounties, format)
		return
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"ID", "Name", "Status", "Max Payout", "Assets", "Findings", "Created"})
	table.SetBorder(false)
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)

	for _, bounty := range bounties {
		maxPayout := fmt.Sprintf("$%.2f", bounty.MaxPayout)
		assetCount := fmt.Sprintf("%d", len(bounty.AssetIDs))
		findingsCount := fmt.Sprintf("%d", bounty.Statistics.TotalFindings)

		table.Append([]string{
			bounty.ID,
			bounty.Name,
			bounty.Status,
			maxPayout,
			assetCount,
			findingsCount,
			bounty.CreatedAt.Format("2006-01-02 15:04"),
		})
	}

	table.Render()
}

// PrintBounty prints a single bounty
func PrintBounty(bounty *api.Bounty, format string) {
	if format != "table" {
		formatOutput(bounty, format)
		return
	}

	PrintBounties([]*api.Bounty{bounty}, format)
}

// PrintBountyDetail prints detailed bounty information
func PrintBountyDetail(bounty *api.Bounty, format string) {
	if format != "table" {
		formatOutput(bounty, format)
		return
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Property", "Value"})
	table.SetBorder(false)
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)

	table.Append([]string{"ID", bounty.ID})
	table.Append([]string{"Name", bounty.Name})
	table.Append([]string{"Description", bounty.Description})
	table.Append([]string{"Status", bounty.Status})
	table.Append([]string{"Max Payout", fmt.Sprintf("$%.2f", bounty.MaxPayout)})
	table.Append([]string{"Private", strconv.FormatBool(bounty.Private)})
	table.Append([]string{"Asset IDs", strings.Join(bounty.AssetIDs, ", ")})
	table.Append([]string{"Scopes", strings.Join(bounty.Scopes, ", ")})
	table.Append([]string{"Out of Scope", strings.Join(bounty.OutOfScope, ", ")})
	table.Append([]string{"Tags", strings.Join(bounty.Tags, ", ")})
	table.Append([]string{"Total Findings", strconv.Itoa(bounty.Statistics.TotalFindings)})
	table.Append([]string{"Accepted Findings", strconv.Itoa(bounty.Statistics.AcceptedFindings)})
	table.Append([]string{"Total Payout", fmt.Sprintf("$%.2f", bounty.Statistics.TotalPayout)})
	table.Append([]string{"Participants", strconv.Itoa(bounty.Statistics.ParticipantCount)})
	
	if bounty.StartDate != nil {
		table.Append([]string{"Start Date", bounty.StartDate.Format("2006-01-02 15:04:05")})
	}
	if bounty.EndDate != nil {
		table.Append([]string{"End Date", bounty.EndDate.Format("2006-01-02 15:04:05")})
	}
	
	table.Append([]string{"Created", bounty.CreatedAt.Format("2006-01-02 15:04:05")})
	table.Append([]string{"Updated", bounty.UpdatedAt.Format("2006-01-02 15:04:05")})

	table.Render()
}

// PrintPayouts prints a list of payouts
func PrintPayouts(payouts []*api.Payout, format string) {
	if format != "table" {
		formatOutput(payouts, format)
		return
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"ID", "Bounty", "Researcher", "Amount", "Status", "Created"})
	table.SetBorder(false)
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)

	for _, payout := range payouts {
		amount := fmt.Sprintf("$%.2f", payout.Amount)

		table.Append([]string{
			payout.ID,
			payout.BountyID,
			payout.Researcher,
			amount,
			payout.Status,
			payout.CreatedAt.Format("2006-01-02 15:04"),
		})
	}

	table.Render()
}

// PrintFindings prints a list of findings
func PrintFindings(findings []*api.Finding, format string) {
	if format != "table" {
		formatOutput(findings, format)
		return
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"ID", "Title", "Severity", "Status", "Asset", "Created"})
	table.SetBorder(false)
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)

	for _, finding := range findings {
		severity := finding.Severity
		if config.IsColorOutputEnabled() {
			switch finding.Severity {
			case "critical":
				severity = colorRed(severity)
			case "high":
				severity = colorMagenta(severity)
			case "medium":
				severity = colorYellow(severity)
			case "low":
				severity = colorBlue(severity)
			case "info":
				severity = colorCyan(severity)
			}
		}

		table.Append([]string{
			finding.ID,
			finding.Title,
			severity,
			finding.Status,
			finding.AssetID,
			finding.CreatedAt.Format("2006-01-02 15:04"),
		})
	}

	table.Render()
}

// PrintFinding prints a single finding
func PrintFinding(finding *api.Finding, format string) {
	if format != "table" {
		formatOutput(finding, format)
		return
	}

	PrintFindings([]*api.Finding{finding}, format)
}

// PrintFindingDetail prints detailed finding information
func PrintFindingDetail(finding *api.Finding, format string) {
	if format != "table" {
		formatOutput(finding, format)
		return
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Property", "Value"})
	table.SetBorder(false)
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)

	table.Append([]string{"ID", finding.ID})
	table.Append([]string{"Title", finding.Title})
	table.Append([]string{"Description", finding.Description})
	table.Append([]string{"Severity", finding.Severity})
	table.Append([]string{"Status", finding.Status})
	table.Append([]string{"Asset ID", finding.AssetID})
	
	if finding.ScanID != "" {
		table.Append([]string{"Scan ID", finding.ScanID})
	}
	if finding.BountyID != "" {
		table.Append([]string{"Bounty ID", finding.BountyID})
	}
	if finding.CWEID != "" {
		table.Append([]string{"CWE ID", finding.CWEID})
	}
	if finding.CVEID != "" {
		table.Append([]string{"CVE ID", finding.CVEID})
	}
	if finding.CVSSScore > 0 {
		table.Append([]string{"CVSS Score", fmt.Sprintf("%.1f", finding.CVSSScore)})
	}
	if finding.Impact != "" {
		table.Append([]string{"Impact", finding.Impact})
	}
	if finding.PoC != "" {
		table.Append([]string{"Proof of Concept", finding.PoC})
	}
	if finding.Remediation != "" {
		table.Append([]string{"Remediation", finding.Remediation})
	}
	
	table.Append([]string{"Tags", strings.Join(finding.Tags, ", ")})
	table.Append([]string{"Created", finding.CreatedAt.Format("2006-01-02 15:04:05")})
	table.Append([]string{"Updated", finding.UpdatedAt.Format("2006-01-02 15:04:05")})

	table.Render()
}

// PrintFindingsStats prints findings statistics
func PrintFindingsStats(stats *api.FindingsStats, format string) {
	if format != "table" {
		formatOutput(stats, format)
		return
	}

	// Main stats table
	fmt.Println(colorBold("Findings Statistics"))
	fmt.Println()

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Metric", "Value"})
	table.SetBorder(false)
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)

	table.Append([]string{"Total Findings", strconv.Itoa(stats.Total)})
	table.Render()

	// Severity breakdown
	if len(stats.BySeverity) > 0 {
		fmt.Println()
		fmt.Println(colorBold("By Severity"))
		
		table = tablewriter.NewWriter(os.Stdout)
		table.SetHeader([]string{"Severity", "Count"})
		table.SetBorder(false)
		
		for severity, count := range stats.BySeverity {
			table.Append([]string{severity, strconv.Itoa(count)})
		}
		table.Render()
	}

	// Status breakdown
	if len(stats.ByStatus) > 0 {
		fmt.Println()
		fmt.Println(colorBold("By Status"))
		
		table = tablewriter.NewWriter(os.Stdout)
		table.SetHeader([]string{"Status", "Count"})
		table.SetBorder(false)
		
		for status, count := range stats.ByStatus {
			table.Append([]string{status, strconv.Itoa(count)})
		}
		table.Render()
	}

	// Top CWEs
	if len(stats.TopCWEs) > 0 {
		fmt.Println()
		fmt.Println(colorBold("Top CWEs"))
		
		table = tablewriter.NewWriter(os.Stdout)
		table.SetHeader([]string{"CWE ID", "Name", "Count"})
		table.SetBorder(false)
		
		for _, cwe := range stats.TopCWEs {
			table.Append([]string{cwe.CWEID, cwe.Name, strconv.Itoa(cwe.Count)})
		}
		table.Render()
	}
}