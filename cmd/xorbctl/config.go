package main

import (
	"fmt"

	"github.com/spf13/cobra"
	"github.com/xorb/xorbctl/internal/config"
	"github.com/xorb/xorbctl/internal/output"
)

func newConfigCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "config",
		Short: "Manage xorbctl configuration",
		Long:  `View and modify xorbctl configuration settings.`,
	}

	cmd.AddCommand(
		newConfigGetCmd(),
		newConfigSetCmd(),
		newConfigListCmd(),
		newConfigResetCmd(),
	)

	return cmd
}

func newConfigGetCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "get <key>",
		Short: "Get configuration value",
		Long:  `Get the value of a specific configuration key.`,
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			key := args[0]
			
			value, err := config.Get(key)
			if err != nil {
				return fmt.Errorf("failed to get config: %w", err)
			}

			output.PrintConfigValue(key, value, outputFormat)
			return nil
		},
	}

	return cmd
}

func newConfigSetCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "set <key> <value>",
		Short: "Set configuration value",
		Long:  `Set the value of a specific configuration key.`,
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			key, value := args[0], args[1]
			
			if err := config.Set(key, value); err != nil {
				return fmt.Errorf("failed to set config: %w", err)
			}

			output.PrintSuccess(fmt.Sprintf("Configuration %s set to %s", key, value))
			return nil
		},
	}

	return cmd
}

func newConfigListCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "list",
		Short: "List all configuration",
		Long:  `List all configuration keys and values.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.GetAll()
			if err != nil {
				return fmt.Errorf("failed to get config: %w", err)
			}

			output.PrintConfig(cfg, outputFormat)
			return nil
		},
	}

	return cmd
}

func newConfigResetCmd() *cobra.Command {
	var force bool

	cmd := &cobra.Command{
		Use:   "reset",
		Short: "Reset configuration to defaults",
		Long:  `Reset all configuration to default values.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			if !force {
				confirmed, err := output.ConfirmAction(
					"Are you sure you want to reset all configuration to defaults?")
				if err != nil {
					return err
				}
				if !confirmed {
					output.PrintInfo("Operation cancelled")
					return nil
				}
			}

			if err := config.Reset(); err != nil {
				return fmt.Errorf("failed to reset config: %w", err)
			}

			output.PrintSuccess("Configuration reset to defaults")
			return nil
		},
	}

	cmd.Flags().BoolVarP(&force, "force", "f", false, "Skip confirmation prompt")

	return cmd
}