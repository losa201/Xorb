package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/xorb/xorbctl/internal/auth"
	"github.com/xorb/xorbctl/internal/config"
	"github.com/xorb/xorbctl/internal/version"
)

var (
	// Root command
	rootCmd = &cobra.Command{
		Use:   "xorbctl",
		Short: "Xorb PTaaS Command Line Interface",
		Long: `xorbctl is the official CLI for Xorb PTaaS (Penetration Testing as a Service).
Use this tool to manage assets, run scans, open bounties, and retrieve findings.`,
		SilenceUsage:  true,
		SilenceErrors: true,
		PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
			// Skip auth check for auth commands and version
			if cmd.Name() == "login" || cmd.Name() == "logout" || cmd.Name() == "version" {
				return nil
			}
			
			// Check if user is authenticated
			if !auth.IsAuthenticated() {
				return fmt.Errorf("not authenticated. Run 'xorbctl login' first")
			}
			
			return nil
		},
	}

	// Global flags
	apiEndpoint string
	outputFormat string
	verbose     bool
)

func init() {
	cobra.OnInitialize(initConfig)

	// Global flags
	rootCmd.PersistentFlags().StringVar(&apiEndpoint, "api-endpoint", 
		"https://api.xorb.io", "Xorb API endpoint")
	rootCmd.PersistentFlags().StringVarP(&outputFormat, "output", "o", 
		"table", "Output format (table, json, yaml)")
	rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", 
		false, "Enable verbose output")

	// Add all subcommands
	rootCmd.AddCommand(
		newLoginCmd(),
		newLogoutCmd(),
		newAssetCmd(),
		newScanCmd(),
		newBountyCmd(),
		newFindingsCmd(),
		newVersionCmd(),
		newConfigCmd(),
	)
}

func initConfig() {
	config.Initialize(apiEndpoint, verbose)
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}