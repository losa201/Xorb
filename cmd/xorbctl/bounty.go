package main

import (
	"fmt"
	"strconv"
	"time"

	"github.com/spf13/cobra"
	"github.com/xorb/xorbctl/internal/api"
	"github.com/xorb/xorbctl/internal/output"
)

func newBountyCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "bounty",
		Short: "Manage bug bounty programs",
		Long:  `Create, manage, and participate in bug bounty programs.`,
	}

	cmd.AddCommand(
		newBountyOpenCmd(),
		newBountyListCmd(),
		newBountyShowCmd(),
		newBountyCloseCmd(),
		newBountyUpdateCmd(),
		newBountySubmitCmd(),
		newBountyPayoutsCmd(),
	)

	return cmd
}

func newBountyOpenCmd() *cobra.Command {
	var (
		name        string
		description string
		assetIDs    []string
		startDate   string
		endDate     string
		maxPayout   float64
		payoutRules []string
		scopes      []string
		outOfScope  []string
		tags        []string
		private     bool
		allowedIPs  []string
		rules       string
	)

	cmd := &cobra.Command{
		Use:   "open",
		Short: "Open a new bug bounty program",
		Long: `Open a new bug bounty program for specified assets.

Examples:
  xorbctl bounty open --name "Q1 2024 Program" --assets asset-123 --max-payout 5000
  xorbctl bounty open --name "API Security Bounty" --assets asset-123,asset-456 --private
  xorbctl bounty open --name "Mobile App Bounty" --start-date 2024-02-01 --end-date 2024-03-31`,
		RunE: func(cmd *cobra.Command, args []string) error {
			if name == "" {
				return fmt.Errorf("bounty name is required")
			}
			if len(assetIDs) == 0 {
				return fmt.Errorf("at least one asset ID must be specified")
			}

			var start, end *time.Time
			if startDate != "" {
				t, err := time.Parse("2006-01-02", startDate)
				if err != nil {
					return fmt.Errorf("invalid start date format (use YYYY-MM-DD): %w", err)
				}
				start = &t
			}
			if endDate != "" {
				t, err := time.Parse("2006-01-02", endDate)
				if err != nil {
					return fmt.Errorf("invalid end date format (use YYYY-MM-DD): %w", err)
				}
				end = &t
			}

			bountyRequest := &api.BountyRequest{
				Name:        name,
				Description: description,
				AssetIDs:    assetIDs,
				StartDate:   start,
				EndDate:     end,
				MaxPayout:   maxPayout,
				PayoutRules: payoutRules,
				Scopes:      scopes,
				OutOfScope:  outOfScope,
				Tags:        tags,
				Private:     private,
				AllowedIPs:  allowedIPs,
				Rules:       rules,
			}

			bounty, err := api.CreateBounty(bountyRequest)
			if err != nil {
				return fmt.Errorf("failed to open bounty: %w", err)
			}

			output.PrintBounty(bounty, outputFormat)
			output.PrintSuccess(fmt.Sprintf("Bounty program %s opened successfully", bounty.ID))

			return nil
		},
	}

	cmd.Flags().StringVar(&name, "name", "", "Bounty program name (required)")
	cmd.Flags().StringVar(&description, "description", "", "Bounty program description")
	cmd.Flags().StringSliceVar(&assetIDs, "assets", nil, "Asset IDs for bounty (required)")
	cmd.Flags().StringVar(&startDate, "start-date", "", "Start date (YYYY-MM-DD)")
	cmd.Flags().StringVar(&endDate, "end-date", "", "End date (YYYY-MM-DD)")
	cmd.Flags().Float64Var(&maxPayout, "max-payout", 1000.0, "Maximum payout amount")
	cmd.Flags().StringSliceVar(&payoutRules, "payout-rules", nil, "Payout rules")
	cmd.Flags().StringSliceVar(&scopes, "scopes", nil, "In-scope items")
	cmd.Flags().StringSliceVar(&outOfScope, "out-of-scope", nil, "Out-of-scope items")
	cmd.Flags().StringSliceVar(&tags, "tags", nil, "Bounty tags")
	cmd.Flags().BoolVar(&private, "private", false, "Private bounty program")
	cmd.Flags().StringSliceVar(&allowedIPs, "allowed-ips", nil, "Allowed IP ranges")
	cmd.Flags().StringVar(&rules, "rules", "", "Bounty rules and guidelines")

	cmd.MarkFlagRequired("name")
	cmd.MarkFlagRequired("assets")

	return cmd
}

func newBountyListCmd() *cobra.Command {
	var (
		status  string
		tags    []string
		private bool
		limit   int
		offset  int
	)

	cmd := &cobra.Command{
		Use:   "list",
		Short: "List bug bounty programs",
		Long:  `List all available bug bounty programs with optional filtering.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			filters := &api.BountyFilters{
				Status:  status,
				Tags:    tags,
				Private: private,
				Limit:   limit,
				Offset:  offset,
			}

			bounties, err := api.ListBounties(filters)
			if err != nil {
				return fmt.Errorf("failed to list bounties: %w", err)
			}

			output.PrintBounties(bounties, outputFormat)

			return nil
		},
	}

	cmd.Flags().StringVar(&status, "status", "", 
		"Filter by status (draft, active, paused, closed)")
	cmd.Flags().StringSliceVar(&tags, "tags", nil, "Filter by tags")
	cmd.Flags().BoolVar(&private, "private", false, "Show only private bounties")
	cmd.Flags().IntVar(&limit, "limit", 50, "Number of bounties to return")
	cmd.Flags().IntVar(&offset, "offset", 0, "Offset for pagination")

	return cmd
}

func newBountyShowCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "show <bounty-id>",
		Short: "Show bounty program details",
		Long:  `Display detailed information about a specific bounty program.`,
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			bountyID := args[0]

			bounty, err := api.GetBounty(bountyID)
			if err != nil {
				return fmt.Errorf("failed to get bounty: %w", err)
			}

			output.PrintBountyDetail(bounty, outputFormat)

			return nil
		},
	}

	return cmd
}

func newBountyCloseCmd() *cobra.Command {
	var (
		reason string
		force  bool
	)

	cmd := &cobra.Command{
		Use:   "close <bounty-id>",
		Short: "Close a bug bounty program",
		Long:  `Close an active bug bounty program.`,
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			bountyID := args[0]

			if !force {
				confirmed, err := output.ConfirmAction(
					fmt.Sprintf("Are you sure you want to close bounty %s?", bountyID))
				if err != nil {
					return err
				}
				if !confirmed {
					output.PrintInfo("Operation cancelled")
					return nil
				}
			}

			if err := api.CloseBounty(bountyID, reason); err != nil {
				return fmt.Errorf("failed to close bounty: %w", err)
			}

			output.PrintSuccess(fmt.Sprintf("Bounty %s closed successfully", bountyID))

			return nil
		},
	}

	cmd.Flags().StringVar(&reason, "reason", "", "Reason for closing")
	cmd.Flags().BoolVarP(&force, "force", "f", false, "Skip confirmation prompt")

	return cmd
}

func newBountyUpdateCmd() *cobra.Command {
	var (
		name        string
		description string
		maxPayout   string
		endDate     string
		addScopes   []string
		removeScopes []string
		addTags     []string
		removeTags  []string
	)

	cmd := &cobra.Command{
		Use:   "update <bounty-id>",
		Short: "Update bounty program",
		Long:  `Update properties of an existing bounty program.`,
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			bountyID := args[0]

			updates := &api.BountyUpdate{
				Name:         name,
				Description:  description,
				AddScopes:    addScopes,
				RemoveScopes: removeScopes,
				AddTags:      addTags,
				RemoveTags:   removeTags,
			}

			if maxPayout != "" {
				payout, err := strconv.ParseFloat(maxPayout, 64)
				if err != nil {
					return fmt.Errorf("invalid max payout value: %w", err)
				}
				updates.MaxPayout = &payout
			}

			if endDate != "" {
				t, err := time.Parse("2006-01-02", endDate)
				if err != nil {
					return fmt.Errorf("invalid end date format (use YYYY-MM-DD): %w", err)
				}
				updates.EndDate = &t
			}

			bounty, err := api.UpdateBounty(bountyID, updates)
			if err != nil {
				return fmt.Errorf("failed to update bounty: %w", err)
			}

			output.PrintBounty(bounty, outputFormat)
			output.PrintSuccess(fmt.Sprintf("Bounty %s updated successfully", bountyID))

			return nil
		},
	}

	cmd.Flags().StringVar(&name, "name", "", "Update bounty name")
	cmd.Flags().StringVar(&description, "description", "", "Update description")
	cmd.Flags().StringVar(&maxPayout, "max-payout", "", "Update maximum payout")
	cmd.Flags().StringVar(&endDate, "end-date", "", "Update end date (YYYY-MM-DD)")
	cmd.Flags().StringSliceVar(&addScopes, "add-scopes", nil, "Add scopes")
	cmd.Flags().StringSliceVar(&removeScopes, "remove-scopes", nil, "Remove scopes")
	cmd.Flags().StringSliceVar(&addTags, "add-tags", nil, "Add tags")
	cmd.Flags().StringSliceVar(&removeTags, "remove-tags", nil, "Remove tags")

	return cmd
}

func newBountySubmitCmd() *cobra.Command {
	var (
		bountyID    string
		title       string
		description string
		severity    string
		poc         string
		impact      string
		attachments []string
		cweID       string
	)

	cmd := &cobra.Command{
		Use:   "submit",
		Short: "Submit a vulnerability to a bounty program",
		Long: `Submit a vulnerability finding to an active bounty program.

Examples:
  xorbctl bounty submit --bounty bounty-123 --title "SQL Injection" --severity high
  xorbctl bounty submit --bounty bounty-123 --title "XSS" --poc "payload.txt" --attachments screen1.png,screen2.png`,
		RunE: func(cmd *cobra.Command, args []string) error {
			if bountyID == "" {
				return fmt.Errorf("bounty ID is required")
			}
			if title == "" {
				return fmt.Errorf("vulnerability title is required")
			}
			if description == "" {
				return fmt.Errorf("vulnerability description is required")
			}

			submission := &api.VulnerabilitySubmission{
				BountyID:    bountyID,
				Title:       title,
				Description: description,
				Severity:    severity,
				PoC:         poc,
				Impact:      impact,
				Attachments: attachments,
				CWEID:       cweID,
			}

			finding, err := api.SubmitVulnerability(submission)
			if err != nil {
				return fmt.Errorf("failed to submit vulnerability: %w", err)
			}

			output.PrintFinding(finding, outputFormat)
			output.PrintSuccess(fmt.Sprintf("Vulnerability %s submitted successfully", finding.ID))

			return nil
		},
	}

	cmd.Flags().StringVar(&bountyID, "bounty", "", "Target bounty program ID (required)")
	cmd.Flags().StringVar(&title, "title", "", "Vulnerability title (required)")
	cmd.Flags().StringVar(&description, "description", "", "Detailed description (required)")
	cmd.Flags().StringVar(&severity, "severity", "medium", "Severity level (info, low, medium, high, critical)")
	cmd.Flags().StringVar(&poc, "poc", "", "Proof of concept")
	cmd.Flags().StringVar(&impact, "impact", "", "Impact assessment")
	cmd.Flags().StringSliceVar(&attachments, "attachments", nil, "Attachment file paths")
	cmd.Flags().StringVar(&cweID, "cwe", "", "CWE ID (e.g., CWE-89)")

	cmd.MarkFlagRequired("bounty")
	cmd.MarkFlagRequired("title")
	cmd.MarkFlagRequired("description")

	return cmd
}

func newBountyPayoutsCmd() *cobra.Command {
	var (
		bountyID string
		status   string
		limit    int
		offset   int
	)

	cmd := &cobra.Command{
		Use:   "payouts",
		Short: "List bounty payouts",
		Long:  `List payouts from bounty programs.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			filters := &api.PayoutFilters{
				BountyID: bountyID,
				Status:   status,
				Limit:    limit,
				Offset:   offset,
			}

			payouts, err := api.ListPayouts(filters)
			if err != nil {
				return fmt.Errorf("failed to list payouts: %w", err)
			}

			output.PrintPayouts(payouts, outputFormat)

			return nil
		},
	}

	cmd.Flags().StringVar(&bountyID, "bounty", "", "Filter by bounty ID")
	cmd.Flags().StringVar(&status, "status", "", 
		"Filter by status (pending, approved, paid, rejected)")
	cmd.Flags().IntVar(&limit, "limit", 50, "Number of payouts to return")
	cmd.Flags().IntVar(&offset, "offset", 0, "Offset for pagination")

	return cmd
}