package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/xorb/xorbctl/internal/config"
)

// OAuth Device Flow response structures
type DeviceAuthResponse struct {
	DeviceCode              string `json:"device_code"`
	UserCode                string `json:"user_code"`
	VerificationURI         string `json:"verification_uri"`
	VerificationURIComplete string `json:"verification_uri_complete"`
	ExpiresIn               int    `json:"expires_in"`
	Interval                int    `json:"interval"`
}

type TokenResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	TokenType    string `json:"token_type"`
	ExpiresIn    int    `json:"expires_in"`
	Scope        string `json:"scope"`
}

type UserInfo struct {
	ID           string `json:"id"`
	Handle       string `json:"handle"`
	Email        string `json:"email"`
	Name         string `json:"name"`
	Organization string `json:"organization"`
	Verified     bool   `json:"verified"`
}

// OAuth configuration
const (
	ClientID     = "xorbctl"
	AuthEndpoint = "/auth/device"
	TokenEndpoint = "/auth/token"
	UserEndpoint = "/auth/user"
)

// StartDeviceFlow initiates OAuth device flow
func StartDeviceFlow(ctx context.Context) (*DeviceAuthResponse, error) {
	apiEndpoint := config.GetAPIEndpoint()
	
	data := url.Values{}
	data.Set("client_id", ClientID)
	data.Set("scope", "read write")

	req, err := http.NewRequestWithContext(ctx, "POST", 
		apiEndpoint+AuthEndpoint, strings.NewReader(data.Encode()))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to start device flow: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("device flow failed with status %d", resp.StatusCode)
	}

	var deviceResponse DeviceAuthResponse
	if err := json.NewDecoder(resp.Body).Decode(&deviceResponse); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &deviceResponse, nil
}

// PollForToken polls for the access token after user authorization
func PollForToken(ctx context.Context, deviceAuth *DeviceAuthResponse) (*TokenResponse, error) {
	apiEndpoint := config.GetAPIEndpoint()
	
	ticker := time.NewTicker(time.Duration(deviceAuth.Interval) * time.Second)
	defer ticker.Stop()

	timeout := time.After(time.Duration(deviceAuth.ExpiresIn) * time.Second)

	for {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("context cancelled")
		case <-timeout:
			return nil, fmt.Errorf("device flow expired")
		case <-ticker.C:
			token, err := pollToken(ctx, apiEndpoint, deviceAuth.DeviceCode)
			if err != nil {
				// Check if it's a "pending" error, continue polling
				if strings.Contains(err.Error(), "authorization_pending") {
					continue
				}
				return nil, err
			}
			return token, nil
		}
	}
}

func pollToken(ctx context.Context, apiEndpoint, deviceCode string) (*TokenResponse, error) {
	data := url.Values{}
	data.Set("grant_type", "urn:ietf:params:oauth:grant-type:device_code")
	data.Set("device_code", deviceCode)
	data.Set("client_id", ClientID)

	req, err := http.NewRequestWithContext(ctx, "POST", 
		apiEndpoint+TokenEndpoint, strings.NewReader(data.Encode()))
	if err != nil {
		return nil, fmt.Errorf("failed to create token request: %w", err)
	}

	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to poll token: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		var tokenResponse TokenResponse
		if err := json.NewDecoder(resp.Body).Decode(&tokenResponse); err != nil {
			return nil, fmt.Errorf("failed to decode token response: %w", err)
		}
		return &tokenResponse, nil
	}

	// Handle error responses
	var errorResp struct {
		Error            string `json:"error"`
		ErrorDescription string `json:"error_description"`
	}
	
	if err := json.NewDecoder(resp.Body).Decode(&errorResp); err != nil {
		return nil, fmt.Errorf("unknown error with status %d", resp.StatusCode)
	}

	return nil, fmt.Errorf("%s: %s", errorResp.Error, errorResp.ErrorDescription)
}

// SaveToken saves the token to the config directory
func SaveToken(token *TokenResponse) error {
	configDir, err := getConfigDir()
	if err != nil {
		return fmt.Errorf("failed to get config directory: %w", err)
	}

	if err := os.MkdirAll(configDir, 0700); err != nil {
		return fmt.Errorf("failed to create config directory: %w", err)
	}

	tokenPath := filepath.Join(configDir, "token.json")
	
	// Add expiration time
	tokenData := struct {
		*TokenResponse
		ExpiresAt time.Time `json:"expires_at"`
	}{
		TokenResponse: token,
		ExpiresAt:     time.Now().Add(time.Duration(token.ExpiresIn) * time.Second),
	}

	data, err := json.MarshalIndent(tokenData, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal token: %w", err)
	}

	if err := os.WriteFile(tokenPath, data, 0600); err != nil {
		return fmt.Errorf("failed to save token: %w", err)
	}

	return nil
}

// GetToken retrieves the stored token
func GetToken() (*TokenResponse, error) {
	configDir, err := getConfigDir()
	if err != nil {
		return nil, fmt.Errorf("failed to get config directory: %w", err)
	}

	tokenPath := filepath.Join(configDir, "token.json")
	
	data, err := os.ReadFile(tokenPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("not authenticated")
		}
		return nil, fmt.Errorf("failed to read token: %w", err)
	}

	var tokenData struct {
		*TokenResponse
		ExpiresAt time.Time `json:"expires_at"`
	}

	if err := json.Unmarshal(data, &tokenData); err != nil {
		return nil, fmt.Errorf("failed to unmarshal token: %w", err)
	}

	// Check if token is expired
	if time.Now().After(tokenData.ExpiresAt) {
		return nil, fmt.Errorf("token expired")
	}

	return tokenData.TokenResponse, nil
}

// IsAuthenticated checks if the user is authenticated
func IsAuthenticated() bool {
	_, err := GetToken()
	return err == nil
}

// Logout clears the stored authentication
func Logout() error {
	configDir, err := getConfigDir()
	if err != nil {
		return fmt.Errorf("failed to get config directory: %w", err)
	}

	tokenPath := filepath.Join(configDir, "token.json")
	
	if err := os.Remove(tokenPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to remove token: %w", err)
	}

	return nil
}

// GetUserInfo retrieves user information using the access token
func GetUserInfo(accessToken string) (*UserInfo, error) {
	apiEndpoint := config.GetAPIEndpoint()
	
	req, err := http.NewRequest("GET", apiEndpoint+UserEndpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+accessToken)
	req.Header.Set("Accept", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to get user info: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to get user info with status %d", resp.StatusCode)
	}

	var userInfo UserInfo
	if err := json.NewDecoder(resp.Body).Decode(&userInfo); err != nil {
		return nil, fmt.Errorf("failed to decode user info: %w", err)
	}

	return &userInfo, nil
}

// getConfigDir returns the configuration directory
func getConfigDir() (string, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to get home directory: %w", err)
	}

	return filepath.Join(homeDir, ".config", "xorbctl"), nil
}