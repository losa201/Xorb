package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Configuration holds the CLI configuration
type Configuration struct {
	APIEndpoint    string `json:"api_endpoint"`
	DefaultOutput  string `json:"default_output"`
	Verbose        bool   `json:"verbose"`
	Timeout        int    `json:"timeout"`
	RetryAttempts  int    `json:"retry_attempts"`
	ColorOutput    bool   `json:"color_output"`
	AutoUpdate     bool   `json:"auto_update"`
	TelemetryOpt   bool   `json:"telemetry_opt_in"`
}

var (
	config        *Configuration
	defaultConfig = &Configuration{
		APIEndpoint:   "https://api.xorb.io",
		DefaultOutput: "table",
		Verbose:       false,
		Timeout:       300,
		RetryAttempts: 3,
		ColorOutput:   true,
		AutoUpdate:    true,
		TelemetryOpt:  false,
	}
)

// Initialize loads or creates the configuration
func Initialize(apiEndpoint string, verbose bool) {
	// Load config from file
	loadedConfig, err := loadConfig()
	if err != nil {
		// Use defaults if loading fails
		config = defaultConfig
	} else {
		config = loadedConfig
	}

	// Override with CLI flags if provided
	if apiEndpoint != "" {
		config.APIEndpoint = apiEndpoint
	}
	if verbose {
		config.Verbose = verbose
	}
}

// GetAPIEndpoint returns the configured API endpoint
func GetAPIEndpoint() string {
	if config == nil {
		return defaultConfig.APIEndpoint
	}
	return config.APIEndpoint
}

// IsVerbose returns whether verbose mode is enabled
func IsVerbose() bool {
	if config == nil {
		return defaultConfig.Verbose
	}
	return config.Verbose
}

// GetTimeout returns the configured timeout
func GetTimeout() int {
	if config == nil {
		return defaultConfig.Timeout
	}
	return config.Timeout
}

// GetRetryAttempts returns the configured retry attempts
func GetRetryAttempts() int {
	if config == nil {
		return defaultConfig.RetryAttempts
	}
	return config.RetryAttempts
}

// IsColorOutputEnabled returns whether color output is enabled
func IsColorOutputEnabled() bool {
	if config == nil {
		return defaultConfig.ColorOutput
	}
	return config.ColorOutput
}

// Get retrieves a configuration value by key
func Get(key string) (interface{}, error) {
	if config == nil {
		return nil, fmt.Errorf("configuration not initialized")
	}

	switch key {
	case "api_endpoint":
		return config.APIEndpoint, nil
	case "default_output":
		return config.DefaultOutput, nil
	case "verbose":
		return config.Verbose, nil
	case "timeout":
		return config.Timeout, nil
	case "retry_attempts":
		return config.RetryAttempts, nil
	case "color_output":
		return config.ColorOutput, nil
	case "auto_update":
		return config.AutoUpdate, nil
	case "telemetry_opt_in":
		return config.TelemetryOpt, nil
	default:
		return nil, fmt.Errorf("unknown configuration key: %s", key)
	}
}

// Set updates a configuration value by key
func Set(key string, value string) error {
	if config == nil {
		config = defaultConfig
	}

	switch key {
	case "api_endpoint":
		config.APIEndpoint = value
	case "default_output":
		if value != "table" && value != "json" && value != "yaml" {
			return fmt.Errorf("invalid output format: %s", value)
		}
		config.DefaultOutput = value
	case "verbose":
		config.Verbose = value == "true"
	case "color_output":
		config.ColorOutput = value == "true"
	case "auto_update":
		config.AutoUpdate = value == "true"
	case "telemetry_opt_in":
		config.TelemetryOpt = value == "true"
	default:
		return fmt.Errorf("unknown configuration key: %s", key)
	}

	return saveConfig(config)
}

// GetAll returns the entire configuration
func GetAll() (*Configuration, error) {
	if config == nil {
		return nil, fmt.Errorf("configuration not initialized")
	}
	return config, nil
}

// Reset resets configuration to defaults
func Reset() error {
	config = &Configuration{
		APIEndpoint:   defaultConfig.APIEndpoint,
		DefaultOutput: defaultConfig.DefaultOutput,
		Verbose:       defaultConfig.Verbose,
		Timeout:       defaultConfig.Timeout,
		RetryAttempts: defaultConfig.RetryAttempts,
		ColorOutput:   defaultConfig.ColorOutput,
		AutoUpdate:    defaultConfig.AutoUpdate,
		TelemetryOpt:  defaultConfig.TelemetryOpt,
	}

	return saveConfig(config)
}

// loadConfig loads configuration from file
func loadConfig() (*Configuration, error) {
	configDir, err := getConfigDir()
	if err != nil {
		return nil, fmt.Errorf("failed to get config directory: %w", err)
	}

	configPath := filepath.Join(configDir, "config.json")
	
	data, err := os.ReadFile(configPath)
	if err != nil {
		if os.IsNotExist(err) {
			return defaultConfig, nil
		}
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var cfg Configuration
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	return &cfg, nil
}

// saveConfig saves configuration to file
func saveConfig(cfg *Configuration) error {
	configDir, err := getConfigDir()
	if err != nil {
		return fmt.Errorf("failed to get config directory: %w", err)
	}

	if err := os.MkdirAll(configDir, 0700); err != nil {
		return fmt.Errorf("failed to create config directory: %w", err)
	}

	configPath := filepath.Join(configDir, "config.json")
	
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(configPath, data, 0600); err != nil {
		return fmt.Errorf("failed to save config: %w", err)
	}

	return nil
}

// getConfigDir returns the configuration directory
func getConfigDir() (string, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to get home directory: %w", err)
	}

	return filepath.Join(homeDir, ".config", "xorbctl"), nil
}