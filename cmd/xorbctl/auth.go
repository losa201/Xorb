package main

import (
	"context"
	"fmt"
	"time"

	"github.com/spf13/cobra"
	"github.com/xorb/xorbctl/internal/auth"
	"github.com/xorb/xorbctl/internal/output"
)

func newLoginCmd() *cobra.Command {
	var (
		deviceFlow bool
		timeout    int
	)

	cmd := &cobra.Command{
		Use:   "login",
		Short: "Authenticate with Xorb PTaaS",
		Long: `Authenticate with Xorb PTaaS using OAuth Device Flow.
This will open a browser window for authentication or provide a device code.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeout)*time.Second)
			defer cancel()

			if deviceFlow {
				return performDeviceFlowLogin(ctx)
			}

			// Default to device flow for now
			return performDeviceFlowLogin(ctx)
		},
	}

	cmd.Flags().BoolVar(&deviceFlow, "device-flow", true, "Use OAuth device flow (default)")
	cmd.Flags().IntVar(&timeout, "timeout", 300, "Authentication timeout in seconds")

	return cmd
}

func newLogoutCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "logout",
		Short: "Sign out of Xorb PTaaS",
		Long:  `Clear local authentication credentials and sign out.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			if err := auth.Logout(); err != nil {
				return fmt.Errorf("failed to logout: %w", err)
			}

			output.PrintSuccess("Successfully logged out")
			return nil
		},
	}

	return cmd
}

func performDeviceFlowLogin(ctx context.Context) error {
	output.PrintInfo("Starting OAuth device flow authentication...")

	// Start device flow
	deviceResponse, err := auth.StartDeviceFlow(ctx)
	if err != nil {
		return fmt.Errorf("failed to start device flow: %w", err)
	}

	// Display user instructions
	output.PrintInfo(fmt.Sprintf("Please visit: %s", deviceResponse.VerificationURIComplete))
	output.PrintInfo(fmt.Sprintf("And enter code: %s", deviceResponse.UserCode))
	output.PrintInfo("Waiting for authentication...")

	// Poll for token
	token, err := auth.PollForToken(ctx, deviceResponse)
	if err != nil {
		return fmt.Errorf("authentication failed: %w", err)
	}

	// Save token
	if err := auth.SaveToken(token); err != nil {
		return fmt.Errorf("failed to save authentication: %w", err)
	}

	// Get user info
	user, err := auth.GetUserInfo(token.AccessToken)
	if err != nil {
		output.PrintWarning("Authentication successful, but failed to get user info")
	} else {
		output.PrintSuccess(fmt.Sprintf("Successfully authenticated as %s (%s)", 
			user.Handle, user.Email))
	}

	return nil
}