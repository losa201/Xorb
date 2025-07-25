# xorbctl - Xorb PTaaS Command Line Interface

The official CLI tool for Xorb PTaaS (Penetration Testing as a Service). Manage assets, run scans, open bounties, and retrieve findings from the command line.

## Features

- **Asset Management**: Add, list, update, and remove target assets
- **Security Scanning**: Start scans, monitor progress, and view results
- **Bug Bounty Programs**: Create and manage bounty programs
- **Finding Export**: Export findings in multiple formats (SARIF, JSON, CSV, PDF)
- **OAuth Authentication**: Secure OAuth Device Flow authentication
- **Cross-Platform**: Available for Linux, macOS, and Windows

## Installation

### Homebrew (macOS/Linux)

```bash
brew install xorb/tap/xorbctl
```

### Scoop (Windows)

```bash
scoop bucket add xorb https://github.com/xorb/scoop-bucket
scoop install xorbctl
```

### Manual Installation

1. Download the latest release from [GitHub Releases](https://github.com/xorb/xorb/releases)
2. Extract the binary to a directory in your PATH
3. Make it executable: `chmod +x xorbctl`

### Docker

```bash
docker run --rm ghcr.io/xorb/xorbctl:latest version
```

## Quick Start

### Authentication

First, authenticate with your Xorb PTaaS account:

```bash
xorbctl login
```

This will open a browser window for OAuth authentication or provide a device code.

### Asset Management

Add a target asset:

```bash
xorbctl asset add example.com --type domain --criticality high
```

List your assets:

```bash
xorbctl asset list
```

### Security Scanning

Start a security scan:

```bash
xorbctl scan run --assets asset-123 --type web --profile deep
```

Monitor scan progress:

```bash
xorbctl scan show scan-456
```

Follow scan logs in real-time:

```bash
xorbctl scan logs scan-456 --follow
```

### Finding Management

List findings:

```bash
xorbctl findings list --severity high --status open
```

Export findings to SARIF:

```bash
xorbctl findings pull --scan scan-456 --format sarif -o results.sarif
```

### Bug Bounty Programs

Open a bounty program:

```bash
xorbctl bounty open --name "Q1 2024 Program" --assets asset-123 --max-payout 5000
```

Submit a vulnerability:

```bash
xorbctl bounty submit --bounty bounty-789 --title "SQL Injection" --severity high
```

## Commands

### Global Flags

- `--api-endpoint`: Xorb API endpoint (default: https://api.xorb.io)
- `--output, -o`: Output format (table, json, yaml)
- `--verbose, -v`: Enable verbose output

### Authentication

- `xorbctl login`: Authenticate with Xorb PTaaS
- `xorbctl logout`: Sign out and clear credentials

### Asset Management

- `xorbctl asset add <target>`: Add a new target asset
- `xorbctl asset list`: List target assets
- `xorbctl asset show <asset-id>`: Show asset details
- `xorbctl asset update <asset-id>`: Update asset properties
- `xorbctl asset remove <asset-id>`: Remove target asset

### Security Scanning

- `xorbctl scan run`: Start a new security scan
- `xorbctl scan list`: List security scans
- `xorbctl scan show <scan-id>`: Show scan details
- `xorbctl scan stop <scan-id>`: Stop a running scan
- `xorbctl scan logs <scan-id>`: View scan logs
- `xorbctl scan templates`: List available scan templates

### Bug Bounty Programs

- `xorbctl bounty open`: Open a new bug bounty program
- `xorbctl bounty list`: List bug bounty programs
- `xorbctl bounty show <bounty-id>`: Show bounty details
- `xorbctl bounty update <bounty-id>`: Update bounty program
- `xorbctl bounty close <bounty-id>`: Close bounty program
- `xorbctl bounty submit`: Submit a vulnerability
- `xorbctl bounty payouts`: List bounty payouts

### Finding Management

- `xorbctl findings list`: List security findings
- `xorbctl findings show <finding-id>`: Show finding details
- `xorbctl findings pull`: Export findings in various formats
- `xorbctl findings export`: Export using templates
- `xorbctl findings stats`: Show findings statistics

### Configuration

- `xorbctl config get <key>`: Get configuration value
- `xorbctl config set <key> <value>`: Set configuration value
- `xorbctl config list`: List all configuration
- `xorbctl config reset`: Reset to defaults

### Utilities

- `xorbctl version`: Show version information

## Configuration

xorbctl stores configuration in `~/.config/xorbctl/config.json`. Available settings:

- `api_endpoint`: Xorb API endpoint
- `default_output`: Default output format (table, json, yaml)
- `timeout`: Request timeout in seconds
- `color_output`: Enable colored output
- `verbose`: Enable verbose logging

## Examples

### Complete Workflow

```bash
# Authenticate
xorbctl login

# Add a target asset
xorbctl asset add example.com --type domain --criticality high --tags "production,web"

# Start a comprehensive scan
xorbctl scan run --assets asset-123 --type web --profile deep --wait --follow

# List high severity findings
xorbctl findings list --severity high --status open

# Export findings for compliance
xorbctl findings export --template compliance-report --format pdf -o security-report.pdf

# Open a bug bounty program
xorbctl bounty open --name "Security Bounty" --assets asset-123 --max-payout 10000
```

### Asset Discovery

```bash
# Add multiple assets
xorbctl asset add "192.168.1.0/24" --type network --scopes web,api
xorbctl asset add mobile-app --type mobile --description "iOS banking app"

# List assets by criticality
xorbctl asset list --criticality critical

# Update asset tags
xorbctl asset update asset-123 --add-tags "api,production"
```

### Scan Management

```bash
# Run different scan types
xorbctl scan run --assets asset-123 --type web --agents nuclei,nmap
xorbctl scan run --assets asset-456 --template owasp-top10

# Monitor multiple scans
xorbctl scan list --status running

# Get scan statistics
xorbctl findings stats --scan scan-123 --period 30d
```

### Finding Analysis

```bash
# Filter findings by multiple criteria
xorbctl findings list --severity critical,high --asset asset-123

# Export findings with proof-of-concept
xorbctl findings pull --scan scan-456 --format json --include-poc

# Generate executive summary
xorbctl findings export --template executive-summary --format pdf
```

## Output Formats

xorbctl supports multiple output formats:

- **table** (default): Human-readable tabular output
- **json**: Machine-readable JSON output
- **yaml**: YAML output for configuration files

```bash
# Table output (default)
xorbctl asset list

# JSON output for scripting
xorbctl asset list -o json | jq '.[] | select(.criticality == "high")'

# YAML output for configuration
xorbctl config list -o yaml
```

## Authentication

xorbctl uses OAuth Device Flow for secure authentication. During login:

1. A device code is generated
2. You visit the verification URL in a browser
3. Enter the device code and authenticate
4. Access token is stored securely in `~/.config/xorbctl/token.json`

Tokens are automatically refreshed when needed.

## Environment Variables

- `XORB_API_ENDPOINT`: Override default API endpoint
- `XORB_CONFIG_DIR`: Override configuration directory
- `XORB_TOKEN_FILE`: Override token file location
- `NO_COLOR`: Disable colored output

## Troubleshooting

### Authentication Issues

```bash
# Clear stored credentials
xorbctl logout

# Re-authenticate
xorbctl login --timeout 600
```

### API Connection Issues

```bash
# Check API connectivity
xorbctl config get api_endpoint

# Use custom endpoint
xorbctl --api-endpoint https://custom.xorb.io asset list
```

### Verbose Logging

```bash
# Enable verbose output for debugging
xorbctl --verbose scan run --assets asset-123
```

## Shell Completion

Enable shell completion for better CLI experience:

### Bash

```bash
# Add to ~/.bashrc
source <(xorbctl completion bash)
```

### Zsh

```bash
# Add to ~/.zshrc
source <(xorbctl completion zsh)
```

### Fish

```bash
# Add to ~/.config/fish/config.fish
xorbctl completion fish | source
```

## Contributing

xorbctl is part of the Xorb PTaaS platform. For issues, feature requests, or contributions:

- GitHub: https://github.com/xorb/xorb
- Documentation: https://docs.xorb.io
- Support: support@xorb.io

## License

Copyright © 2024 Xorb Security. All rights reserved.

This software is proprietary and confidential. Unauthorized copying, distribution, or use is strictly prohibited.