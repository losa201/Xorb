package main

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/spf13/cobra"
	"github.com/xorb/xorbctl/internal/api"
	"github.com/xorb/xorbctl/internal/output"
)

func newAssetCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "asset",
		Short: "Manage target assets",
		Long:  `Add, list, update, and remove target assets for scanning and bounty programs.`,
	}

	cmd.AddCommand(
		newAssetAddCmd(),
		newAssetListCmd(),
		newAssetUpdateCmd(),
		newAssetRemoveCmd(),
		newAssetShowCmd(),
	)

	return cmd
}

func newAssetAddCmd() *cobra.Command {
	var (
		assetType   string
		description string
		tags        []string
		criticality string
		allowedIPs  []string
		blockedIPs  []string
		scopes      []string
	)

	cmd := &cobra.Command{
		Use:   "add <target>",
		Short: "Add a new target asset",
		Long: `Add a new target asset for scanning and bounty programs.
The target can be a domain, IP address, or IP range.

Examples:
  xorbctl asset add example.com --type domain --criticality high
  xorbctl asset add 192.168.1.0/24 --type network --scopes web,api
  xorbctl asset add mobile-app --type mobile --description "iOS banking app"`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			target := args[0]

			asset := &api.Asset{
				Target:      target,
				Type:        assetType,
				Description: description,
				Tags:        tags,
				Criticality: criticality,
				AllowedIPs:  allowedIPs,
				BlockedIPs:  blockedIPs,
				Scopes:      scopes,
			}

			result, err := api.CreateAsset(asset)
			if err != nil {
				return fmt.Errorf("failed to add asset: %w", err)
			}

			output.PrintAsset(result, outputFormat)
			output.PrintSuccess(fmt.Sprintf("Asset %s added successfully", result.ID))

			return nil
		},
	}

	cmd.Flags().StringVar(&assetType, "type", "domain", 
		"Asset type (domain, ip, network, mobile, api)")
	cmd.Flags().StringVar(&description, "description", "", 
		"Asset description")
	cmd.Flags().StringSliceVar(&tags, "tags", nil, 
		"Asset tags (comma-separated)")
	cmd.Flags().StringVar(&criticality, "criticality", "medium", 
		"Asset criticality (low, medium, high, critical)")
	cmd.Flags().StringSliceVar(&allowedIPs, "allowed-ips", nil, 
		"Allowed IP ranges for testing (comma-separated)")
	cmd.Flags().StringSliceVar(&blockedIPs, "blocked-ips", nil, 
		"Blocked IP ranges (comma-separated)")
	cmd.Flags().StringSliceVar(&scopes, "scopes", nil, 
		"Testing scopes (web, api, mobile, network)")

	cmd.MarkFlagRequired("type")

	return cmd
}

func newAssetListCmd() *cobra.Command {
	var (
		assetType   string
		criticality string
		tags        []string
		limit       int
		offset      int
	)

	cmd := &cobra.Command{
		Use:   "list",
		Short: "List target assets",
		Long:  `List all target assets with optional filtering.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			filters := &api.AssetFilters{
				Type:        assetType,
				Criticality: criticality,
				Tags:        tags,
				Limit:       limit,
				Offset:      offset,
			}

			assets, err := api.ListAssets(filters)
			if err != nil {
				return fmt.Errorf("failed to list assets: %w", err)
			}

			output.PrintAssets(assets, outputFormat)

			return nil
		},
	}

	cmd.Flags().StringVar(&assetType, "type", "", "Filter by asset type")
	cmd.Flags().StringVar(&criticality, "criticality", "", "Filter by criticality")
	cmd.Flags().StringSliceVar(&tags, "tags", nil, "Filter by tags")
	cmd.Flags().IntVar(&limit, "limit", 50, "Number of assets to return")
	cmd.Flags().IntVar(&offset, "offset", 0, "Offset for pagination")

	return cmd
}

func newAssetUpdateCmd() *cobra.Command {
	var (
		description string
		tags        []string
		criticality string
		addTags     []string
		removeTags  []string
	)

	cmd := &cobra.Command{
		Use:   "update <asset-id>",
		Short: "Update target asset",
		Long:  `Update properties of an existing target asset.`,
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			assetID := args[0]

			updates := &api.AssetUpdate{
				Description: description,
				Tags:        tags,
				Criticality: criticality,
				AddTags:     addTags,
				RemoveTags:  removeTags,
			}

			asset, err := api.UpdateAsset(assetID, updates)
			if err != nil {
				return fmt.Errorf("failed to update asset: %w", err)
			}

			output.PrintAsset(asset, outputFormat)
			output.PrintSuccess(fmt.Sprintf("Asset %s updated successfully", assetID))

			return nil
		},
	}

	cmd.Flags().StringVar(&description, "description", "", "Update description")
	cmd.Flags().StringSliceVar(&tags, "tags", nil, "Replace all tags")
	cmd.Flags().StringVar(&criticality, "criticality", "", "Update criticality")  
	cmd.Flags().StringSliceVar(&addTags, "add-tags", nil, "Add tags")
	cmd.Flags().StringSliceVar(&removeTags, "remove-tags", nil, "Remove tags")

	return cmd
}

func newAssetRemoveCmd() *cobra.Command {
	var force bool

	cmd := &cobra.Command{
		Use:     "remove <asset-id>",
		Aliases: []string{"rm", "delete"},
		Short:   "Remove target asset",
		Long:    `Remove a target asset. This will also remove all associated scans and findings.`,
		Args:    cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			assetID := args[0]

			if !force {
				confirmed, err := output.ConfirmAction(
					fmt.Sprintf("Are you sure you want to remove asset %s?", assetID))
				if err != nil {
					return err
				}
				if !confirmed {
					output.PrintInfo("Operation cancelled")
					return nil
				}
			}

			if err := api.DeleteAsset(assetID); err != nil {
				return fmt.Errorf("failed to remove asset: %w", err)
			}

			output.PrintSuccess(fmt.Sprintf("Asset %s removed successfully", assetID))

			return nil
		},
	}

	cmd.Flags().BoolVarP(&force, "force", "f", false, "Skip confirmation prompt")

	return cmd
}

func newAssetShowCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "show <asset-id>",
		Short: "Show asset details",
		Long:  `Display detailed information about a specific asset.`,
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			assetID := args[0]

			asset, err := api.GetAsset(assetID)
			if err != nil {
				return fmt.Errorf("failed to get asset: %w", err)
			}

			output.PrintAssetDetail(asset, outputFormat)

			return nil
		},
	}

	return cmd
}