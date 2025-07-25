package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"github.com/xorb/xorbctl/internal/api"
	"github.com/xorb/xorbctl/internal/output"
)

func newScanCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "scan",
		Short: "Manage security scans",
		Long:  `Start, monitor, and manage security scans of target assets.`,
	}

	cmd.AddCommand(
		newScanRunCmd(),
		newScanListCmd(),
		newScanShowCmd(),
		newScanStopCmd(),
		newScanLogsCmd(),
		newScanTemplatesCmd(),
	)

	return cmd
}

func newScanRunCmd() *cobra.Command {
	var (
		assetIDs    []string
		scanType    string
		template    string
		profile     string
		agents      []string
		timeout     int
		description string
		tags        []string
		waitForRun  bool
		followLogs  bool
	)

	cmd := &cobra.Command{
		Use:   "run",
		Short: "Start a new security scan",
		Long: `Start a new security scan against one or more assets.

Examples:
  xorbctl scan run --assets asset-123 --type web --profile deep
  xorbctl scan run --assets asset-123,asset-456 --template nmap-full
  xorbctl scan run --assets asset-123 --agents nuclei,nmap --wait --follow`,
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(assetIDs) == 0 {
				return fmt.Errorf("at least one asset ID must be specified")
			}

			scanRequest := &api.ScanRequest{
				AssetIDs:    assetIDs,
				Type:        scanType,
				Template:    template,
				Profile:     profile,
				Agents:      agents,
				Timeout:     timeout,
				Description: description,
				Tags:        tags,
			}

			scan, err := api.StartScan(scanRequest)
			if err != nil {
				return fmt.Errorf("failed to start scan: %w", err)
			}

			output.PrintScan(scan, outputFormat)
			output.PrintSuccess(fmt.Sprintf("Scan %s started successfully", scan.ID))

			if waitForRun || followLogs {
				return waitForScanCompletion(scan.ID, followLogs)
			}

			return nil
		},
	}

	cmd.Flags().StringSliceVar(&assetIDs, "assets", nil, 
		"Asset IDs to scan (required, comma-separated)")
	cmd.Flags().StringVar(&scanType, "type", "web", 
		"Scan type (web, network, api, mobile)")
	cmd.Flags().StringVar(&template, "template", "", 
		"Scan template to use")
	cmd.Flags().StringVar(&profile, "profile", "standard", 
		"Scan profile (quick, standard, deep)")
	cmd.Flags().StringSliceVar(&agents, "agents", nil, 
		"Specific agents to use (comma-separated)")
	cmd.Flags().IntVar(&timeout, "timeout", 3600, 
		"Scan timeout in seconds")
	cmd.Flags().StringVar(&description, "description", "", 
		"Scan description")
	cmd.Flags().StringSliceVar(&tags, "tags", nil, 
		"Scan tags (comma-separated)")
	cmd.Flags().BoolVar(&waitForRun, "wait", false, 
		"Wait for scan to complete")
	cmd.Flags().BoolVar(&followLogs, "follow", false, 
		"Follow scan logs (implies --wait)")

	cmd.MarkFlagRequired("assets")

	return cmd
}

func newScanListCmd() *cobra.Command {
	var (
		status   string
		scanType string
		assetID  string
		tags     []string
		limit    int
		offset   int
	)

	cmd := &cobra.Command{
		Use:   "list",
		Short: "List security scans",
		Long:  `List security scans with optional filtering.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			filters := &api.ScanFilters{
				Status:   status,
				Type:     scanType,
				AssetID:  assetID,
				Tags:     tags,
				Limit:    limit,
				Offset:   offset,
			}

			scans, err := api.ListScans(filters)
			if err != nil {
				return fmt.Errorf("failed to list scans: %w", err)
			}

			output.PrintScans(scans, outputFormat)

			return nil
		},
	}

	cmd.Flags().StringVar(&status, "status", "", 
		"Filter by status (pending, running, completed, failed)")
	cmd.Flags().StringVar(&scanType, "type", "", "Filter by scan type")
	cmd.Flags().StringVar(&assetID, "asset", "", "Filter by asset ID")
	cmd.Flags().StringSliceVar(&tags, "tags", nil, "Filter by tags")
	cmd.Flags().IntVar(&limit, "limit", 50, "Number of scans to return")
	cmd.Flags().IntVar(&offset, "offset", 0, "Offset for pagination")

	return cmd
}

func newScanShowCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "show <scan-id>",
		Short: "Show scan details",
		Long:  `Display detailed information about a specific scan.`,
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			scanID := args[0]

			scan, err := api.GetScan(scanID)
			if err != nil {
				return fmt.Errorf("failed to get scan: %w", err)
			}

			output.PrintScanDetail(scan, outputFormat)

			return nil
		},
	}

	return cmd
}

func newScanStopCmd() *cobra.Command {
	var force bool

	cmd := &cobra.Command{
		Use:   "stop <scan-id>",
		Short: "Stop a running scan",
		Long:  `Stop a currently running scan.`,
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			scanID := args[0]

			if !force {
				confirmed, err := output.ConfirmAction(
					fmt.Sprintf("Are you sure you want to stop scan %s?", scanID))
				if err != nil {
					return err
				}
				if !confirmed {
					output.PrintInfo("Operation cancelled")
					return nil
				}
			}

			if err := api.StopScan(scanID); err != nil {
				return fmt.Errorf("failed to stop scan: %w", err)
			}

			output.PrintSuccess(fmt.Sprintf("Scan %s stopped successfully", scanID))

			return nil
		},
	}

	cmd.Flags().BoolVarP(&force, "force", "f", false, "Skip confirmation prompt")

	return cmd
}

func newScanLogsCmd() *cobra.Command {
	var (
		follow bool
		tail   int
	)

	cmd := &cobra.Command{
		Use:   "logs <scan-id>",
		Short: "View scan logs",
		Long:  `View logs from a security scan.`,
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			scanID := args[0]

			if follow {
				return followScanLogs(scanID)
			}

			logs, err := api.GetScanLogs(scanID, tail)
			if err != nil {
				return fmt.Errorf("failed to get scan logs: %w", err)
			}

			output.PrintScanLogs(logs, outputFormat)

			return nil
		},
	}

	cmd.Flags().BoolVarP(&follow, "follow", "f", false, "Follow log output")
	cmd.Flags().IntVar(&tail, "tail", 100, "Number of lines to show from end")

	return cmd
}

func newScanTemplatesCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "templates",
		Short: "List available scan templates",
		Long:  `List all available scan templates and profiles.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			templates, err := api.GetScanTemplates()
			if err != nil {
				return fmt.Errorf("failed to get scan templates: %w", err)
			}

			output.PrintScanTemplates(templates, outputFormat)

			return nil
		},
	}

	return cmd
}

func waitForScanCompletion(scanID string, followLogs bool) error {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	output.PrintInfo("Waiting for scan to complete...")

	if followLogs {
		go func() {
			if err := followScanLogs(scanID); err != nil {
				output.PrintWarning(fmt.Sprintf("Failed to follow logs: %v", err))
			}
		}()
	}

	for {
		select {
		case <-ticker.C:
			scan, err := api.GetScan(scanID)
			if err != nil {
				return fmt.Errorf("failed to check scan status: %w", err)
			}

			switch scan.Status {
			case "completed":
				output.PrintSuccess(fmt.Sprintf("Scan %s completed successfully", scanID))
				if scan.FindingsCount > 0 {
					output.PrintInfo(fmt.Sprintf("Found %d findings. Use 'xorbctl findings list --scan %s' to view them.", 
						scan.FindingsCount, scanID))
				}
				return nil
			case "failed":
				output.PrintError(fmt.Sprintf("Scan %s failed: %s", scanID, scan.ErrorMessage))
				return fmt.Errorf("scan failed")
			case "cancelled":
				output.PrintWarning(fmt.Sprintf("Scan %s was cancelled", scanID))
				return nil
			default:
				if !followLogs {
					output.PrintInfo(fmt.Sprintf("Scan status: %s (progress: %d%%)", 
						scan.Status, scan.Progress))
				}
			}
		}
	}
}

func followScanLogs(scanID string) error {
	// Implementation would use Server-Sent Events or WebSocket
	// to stream logs in real-time
	output.PrintInfo("Following scan logs...")
	
	// For now, poll for logs every few seconds
	ticker := time.NewTicker(3 * time.Second)
	defer ticker.Stop()

	lastLogTime := time.Now().Add(-1 * time.Hour) // Start from 1 hour ago

	for {
		select {
		case <-ticker.C:
			logs, err := api.GetScanLogsSince(scanID, lastLogTime)
			if err != nil {
				continue // Continue on error to not break log following
			}

			for _, log := range logs {
				output.PrintLogEntry(log)
				if log.Timestamp.After(lastLogTime) {
					lastLogTime = log.Timestamp
				}
			}
		}
	}
}