package version

import (
	"runtime"
	"time"
)

// Build-time variables (set via ldflags)
var (
	Version   = "dev"
	GitCommit = "unknown"
	GitTag    = "unknown"
	BuildDate = "unknown"
	GoVersion = runtime.Version()
)

// VersionInfo contains version and build information
type VersionInfo struct {
	Version    string    `json:"version"`
	GitCommit  string    `json:"git_commit"`
	GitTag     string    `json:"git_tag"`
	BuildDate  string    `json:"build_date"`
	GoVersion  string    `json:"go_version"`
	Platform   string    `json:"platform"`
	Architecture string  `json:"architecture"`
	BuildTime  time.Time `json:"build_time,omitempty"`
}

// GetVersionInfo returns comprehensive version information
func GetVersionInfo() *VersionInfo {
	info := &VersionInfo{
		Version:      Version,
		GitCommit:    GitCommit,
		GitTag:       GitTag,
		BuildDate:    BuildDate,
		GoVersion:    GoVersion,
		Platform:     runtime.GOOS,
		Architecture: runtime.GOARCH,
	}

	// Parse build date if available
	if BuildDate != "unknown" {
		if t, err := time.Parse(time.RFC3339, BuildDate); err == nil {
			info.BuildTime = t
		}
	}

	return info
}

// GetVersion returns the version string
func GetVersion() string {
	return Version
}

// GetUserAgent returns a user agent string for HTTP requests
func GetUserAgent() string {
	return "xorbctl/" + Version + " (" + runtime.GOOS + "/" + runtime.GOARCH + ")"
}