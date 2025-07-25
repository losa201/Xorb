package main

import (
	"github.com/spf13/cobra"
	"github.com/xorb/xorbctl/internal/output"
	"github.com/xorb/xorbctl/internal/version"
)

func newVersionCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "version",
		Short: "Show version information",
		Long:  `Display version, build, and platform information for xorbctl.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			info := version.GetVersionInfo()
			output.PrintVersionInfo(info, outputFormat)
			return nil
		},
	}

	return cmd
}