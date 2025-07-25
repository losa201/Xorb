package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/xorb/xorbctl/internal/api"
	"github.com/xorb/xorbctl/internal/output"
)

func newFindingsCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:     "findings",
		Aliases: []string{"finding"},
		Short:   "Manage security findings",
		Long:    `List, export, and manage security findings from scans and bounty submissions.`,
	}

	cmd.AddCommand(
		newFindingsListCmd(),
		newFindingsShowCmd(),
		newFindingsPullCmd(),
		newFindingsExportCmd(),
		newFindingsStatsCmd(),
	)

	return cmd
}

func newFindingsListCmd() *cobra.Command {
	var (
		scanID    string
		bountyID  string
		assetID   string
		severity  string
		status    string
		tags      []string
		limit     int
		offset    int
		sortBy    string
		sortOrder string
	)

	cmd := &cobra.Command{
		Use:   "list",
		Short: "List security findings",
		Long: `List security findings with optional filtering and sorting.

Examples:
  xorbctl findings list --scan scan-123 --severity high
  xorbctl findings list --asset asset-456 --status open
  xorbctl findings list --bounty bounty-789 --sort-by severity --sort-order desc`,
		RunE: func(cmd *cobra.Command, args []string) error {
			filters := &api.FindingsFilters{
				ScanID:    scanID,
				BountyID:  bountyID,
				AssetID:   assetID,
				Severity:  severity,
				Status:    status,
				Tags:      tags,
				Limit:     limit,
				Offset:    offset,
				SortBy:    sortBy,
				SortOrder: sortOrder,
			}

			findings, err := api.ListFindings(filters)
			if err != nil {
				return fmt.Errorf("failed to list findings: %w", err)
			}

			output.PrintFindings(findings, outputFormat)

			return nil
		},
	}

	cmd.Flags().StringVar(&scanID, "scan", "", "Filter by scan ID")
	cmd.Flags().StringVar(&bountyID, "bounty", "", "Filter by bounty ID")
	cmd.Flags().StringVar(&assetID, "asset", "", "Filter by asset ID")
	cmd.Flags().StringVar(&severity, "severity", "", 
		"Filter by severity (info, low, medium, high, critical)")
	cmd.Flags().StringVar(&status, "status", "", 
		"Filter by status (open, triaged, resolved, false_positive)")
	cmd.Flags().StringSliceVar(&tags, "tags", nil, "Filter by tags")
	cmd.Flags().IntVar(&limit, "limit", 50, "Number of findings to return")
	cmd.Flags().IntVar(&offset, "offset", 0, "Offset for pagination")
	cmd.Flags().StringVar(&sortBy, "sort-by", "created_at", 
		"Sort by field (created_at, severity, status)")
	cmd.Flags().StringVar(&sortOrder, "sort-order", "desc", "Sort order (asc, desc)")

	return cmd
}

func newFindingsShowCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "show <finding-id>",
		Short: "Show finding details",
		Long:  `Display detailed information about a specific security finding.`,
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			findingID := args[0]

			finding, err := api.GetFinding(findingID)
			if err != nil {
				return fmt.Errorf("failed to get finding: %w", err)
			}

			output.PrintFindingDetail(finding, outputFormat)

			return nil
		},
	}

	return cmd
}

func newFindingsPullCmd() *cobra.Command {
	var (
		scanID      string
		bountyID    string
		assetID     string
		severity    string
		status      string
		format      string
		outputFile  string
		includeRaw  bool
		includePoc  bool
		minSeverity string
	)

	cmd := &cobra.Command{
		Use:   "pull",
		Short: "Pull findings in various formats",
		Long: `Pull and export findings in various formats including SARIF, JSON, CSV, and XML.

Examples:
  xorbctl findings pull --scan scan-123 --format sarif -o results.sarif
  xorbctl findings pull --asset asset-456 --format json --include-raw
  xorbctl findings pull --bounty bounty-789 --format csv -o report.csv`,
		RunE: func(cmd *cobra.Command, args []string) error {
			if scanID == "" && bountyID == "" && assetID == "" {
				return fmt.Errorf("must specify at least one of --scan, --bounty, or --asset")
			}

			exportRequest := &api.FindingsExportRequest{
				ScanID:      scanID,
				BountyID:    bountyID,
				AssetID:     assetID,
				Severity:    severity,
				Status:      status,
				Format:      format,
				IncludeRaw:  includeRaw,
				IncludePoC:  includePoc,
				MinSeverity: minSeverity,
			}

			data, err := api.ExportFindings(exportRequest)
			if err != nil {
				return fmt.Errorf("failed to export findings: %w", err)
			}

			if outputFile != "" {
				if err := os.WriteFile(outputFile, data, 0644); err != nil {
					return fmt.Errorf("failed to write output file: %w", err)
				}
				output.PrintSuccess(fmt.Sprintf("Findings exported to %s", outputFile))
			} else {
				fmt.Print(string(data))
			}

			return nil
		},
	}

	cmd.Flags().StringVar(&scanID, "scan", "", "Pull findings from scan")
	cmd.Flags().StringVar(&bountyID, "bounty", "", "Pull findings from bounty")
	cmd.Flags().StringVar(&assetID, "asset", "", "Pull findings from asset")
	cmd.Flags().StringVar(&severity, "severity", "", "Filter by severity")
	cmd.Flags().StringVar(&status, "status", "", "Filter by status")
	cmd.Flags().StringVar(&format, "format", "sarif", 
		"Export format (sarif, json, csv, xml, pdf)")
	cmd.Flags().StringVarP(&outputFile, "output", "o", "", "Output file path")
	cmd.Flags().BoolVar(&includeRaw, "include-raw", false, "Include raw tool output")
	cmd.Flags().BoolVar(&includePoc, "include-poc", false, "Include proof-of-concept data")
	cmd.Flags().StringVar(&minSeverity, "min-severity", "", 
		"Minimum severity level (info, low, medium, high, critical)")

	return cmd
}

func newFindingsExportCmd() *cobra.Command {
	var (
		template   string
		filters    []string
		outputFile string
		format     string
	)

	cmd := &cobra.Command{
		Use:   "export",
		Short: "Export findings using templates",
		Long: `Export findings using predefined templates for different formats and audiences.

Available templates:
  - executive-summary: High-level summary for executives
  - technical-report: Detailed technical report
  - compliance-report: Compliance-focused report (OWASP, NIST)
  - developer-report: Developer-focused remediation guide

Examples:
  xorbctl findings export --template executive-summary --format pdf -o summary.pdf
  xorbctl findings export --template technical-report --filters "severity=high,status=open"`,
		RunE: func(cmd *cobra.Command, args []string) error {
			exportRequest := &api.FindingsTemplateExportRequest{
				Template:   template,
				Filters:    filters,
				Format:     format,
				OutputFile: outputFile,
			}

			data, err := api.ExportFindingsWithTemplate(exportRequest)
			if err != nil {
				return fmt.Errorf("failed to export findings: %w", err)
			}

			if outputFile != "" {
				if err := os.WriteFile(outputFile, data, 0644); err != nil {
					return fmt.Errorf("failed to write output file: %w", err)
				}
				output.PrintSuccess(fmt.Sprintf("Report exported to %s", outputFile))
			} else {
				fmt.Print(string(data))
			}

			return nil
		},
	}

	cmd.Flags().StringVar(&template, "template", "", 
		"Export template (executive-summary, technical-report, compliance-report, developer-report)")
	cmd.Flags().StringSliceVar(&filters, "filters", nil, 
		"Export filters (key=value format)")
	cmd.Flags().StringVarP(&outputFile, "output", "o", "", "Output file path")
	cmd.Flags().StringVar(&format, "format", "pdf", 
		"Export format (pdf, html, docx, md)")

	cmd.MarkFlagRequired("template")

	return cmd
}

func newFindingsStatsCmd() *cobra.Command {
	var (
		scanID   string
		bountyID string
		assetID  string
		period   string
	)

	cmd := &cobra.Command{
		Use:   "stats",
		Short: "Show findings statistics",
		Long: `Display statistics and metrics about security findings.

Examples:
  xorbctl findings stats --scan scan-123
  xorbctl findings stats --asset asset-456 --period 30d
  xorbctl findings stats --bounty bounty-789`,
		RunE: func(cmd *cobra.Command, args []string) error {
			statsRequest := &api.FindingsStatsRequest{
				ScanID:   scanID,
				BountyID: bountyID,
				AssetID:  assetID,
				Period:   period,
			}

			stats, err := api.GetFindingsStats(statsRequest)
			if err != nil {
				return fmt.Errorf("failed to get findings stats: %w", err)
			}

			output.PrintFindingsStats(stats, outputFormat)

			return nil
		},
	}

	cmd.Flags().StringVar(&scanID, "scan", "", "Get stats for scan")
	cmd.Flags().StringVar(&bountyID, "bounty", "", "Get stats for bounty")
	cmd.Flags().StringVar(&assetID, "asset", "", "Get stats for asset")
	cmd.Flags().StringVar(&period, "period", "30d", 
		"Time period (7d, 30d, 90d, 1y)")

	return cmd
}