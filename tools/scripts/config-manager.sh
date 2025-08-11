#!/bin/bash

# XORB Configuration Management Script
# Centralized configuration management for different environments

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." &> /dev/null && pwd)"
CONFIG_DIR="$PROJECT_ROOT/config"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="${XORB_ENV:-development}"
COMMAND=""
DRY_RUN=false
VERBOSE=false

usage() {
    cat << EOF
XORB Configuration Management Tool

Usage: $0 [OPTIONS] COMMAND

Commands:
  validate ENV             Validate configuration for environment
  deploy ENV              Deploy configuration for environment  
  switch ENV              Switch to different environment
  export ENV [FORMAT]     Export configuration (json|yaml|env)
  secrets ENV             Manage secrets for environment
  diff ENV1 ENV2          Compare configurations between environments
  template                Generate configuration templates
  status                  Show current configuration status
  reload                  Hot-reload configuration
  backup ENV              Backup current configuration
  restore ENV FILE        Restore configuration from backup

Options:
  -h, --help              Show this help message
  -v, --verbose           Verbose output
  -d, --dry-run          Show what would be done without executing
  --config-dir DIR        Use custom config directory (default: $CONFIG_DIR)

Environments:
  development, staging, production, test

Examples:
  $0 validate production
  $0 deploy staging --dry-run
  $0 export production json
  $0 switch development
  $0 diff staging production
  $0 secrets production

EOF
}

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        "INFO")
            echo -e "${GREEN}[INFO]${NC} ${timestamp} - $message" >&2
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} ${timestamp} - $message" >&2
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message" >&2
            ;;
        "DEBUG")
            if [[ "$VERBOSE" == "true" ]]; then
                echo -e "${BLUE}[DEBUG]${NC} ${timestamp} - $message" >&2
            fi
            ;;
    esac
}

check_dependencies() {
    local deps=("python3" "docker" "docker-compose" "jq")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [[ ${#missing_deps[@]} -ne 0 ]]; then
        log "ERROR" "Missing dependencies: ${missing_deps[*]}"
        log "INFO" "Please install missing dependencies before proceeding"
        exit 1
    fi
}

validate_environment() {
    local env="$1"
    local config_file="$CONFIG_DIR/$env.json"
    
    log "INFO" "Validating configuration for environment: $env"
    
    if [[ ! -f "$config_file" ]]; then
        log "ERROR" "Configuration file not found: $config_file"
        return 1
    fi
    
    # Validate JSON syntax
    if ! jq empty "$config_file" 2>/dev/null; then
        log "ERROR" "Invalid JSON syntax in $config_file"
        return 1
    fi
    
    # Run Python configuration validation
    if [[ "$DRY_RUN" == "false" ]]; then
        cd "$PROJECT_ROOT"
        python3 -c "
import sys
sys.path.append('src')
from common.config_manager import ConfigManager, Environment

try:
    manager = ConfigManager(environment='$env')
    config = manager.get_config()
    print('✅ Configuration validation passed')
except Exception as e:
    print(f'❌ Configuration validation failed: {e}')
    sys.exit(1)
"
    else
        log "INFO" "[DRY RUN] Would validate Python configuration"
    fi
    
    log "INFO" "Configuration validation completed for $env"
}

deploy_environment() {
    local env="$1"
    
    log "INFO" "Deploying configuration for environment: $env"
    
    # Validate first
    validate_environment "$env" || return 1
    
    # Check if secrets exist for production/staging
    if [[ "$env" == "production" || "$env" == "staging" ]]; then
        check_secrets "$env" || return 1
    fi
    
    # Deploy using docker-compose
    local compose_file="$PROJECT_ROOT/docker-compose.$env.yml"
    if [[ -f "$compose_file" ]]; then
        if [[ "$DRY_RUN" == "false" ]]; then
            log "INFO" "Starting services with docker-compose"
            cd "$PROJECT_ROOT"
            export XORB_ENV="$env"
            docker-compose -f "$compose_file" up -d
        else
            log "INFO" "[DRY RUN] Would run: docker-compose -f $compose_file up -d"
        fi
    else
        log "WARN" "Docker compose file not found: $compose_file"
    fi
    
    log "INFO" "Deployment completed for $env"
}

switch_environment() {
    local env="$1"
    
    log "INFO" "Switching to environment: $env"
    
    # Validate environment exists
    if [[ ! -f "$CONFIG_DIR/$env.json" ]]; then
        log "ERROR" "Environment configuration not found: $env"
        return 1
    fi
    
    # Update environment variable
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        # Update existing .env file
        if grep -q "XORB_ENV=" "$PROJECT_ROOT/.env"; then
            sed -i "s/XORB_ENV=.*/XORB_ENV=$env/" "$PROJECT_ROOT/.env"
        else
            echo "XORB_ENV=$env" >> "$PROJECT_ROOT/.env"
        fi
    else
        # Create new .env file from template
        cp "$PROJECT_ROOT/.env.template" "$PROJECT_ROOT/.env"
        sed -i "s/XORB_ENV=.*/XORB_ENV=$env/" "$PROJECT_ROOT/.env"
    fi
    
    export XORB_ENV="$env"
    log "INFO" "Switched to environment: $env"
}

export_configuration() {
    local env="$1"
    local format="${2:-json}"
    
    log "INFO" "Exporting configuration for $env in $format format"
    
    cd "$PROJECT_ROOT"
    python3 -c "
import sys
sys.path.append('src')
from common.config_manager import ConfigManager, ConfigFormat

try:
    manager = ConfigManager(environment='$env')
    config_format = ConfigFormat.${format^^}
    exported = manager.export_config(config_format, include_secrets=False)
    print(exported)
except Exception as e:
    print(f'Error exporting configuration: {e}', file=sys.stderr)
    sys.exit(1)
"
}

check_secrets() {
    local env="$1"
    local secrets_dir="$PROJECT_ROOT/secrets"
    local required_secrets=("db_password" "jwt_secret" "encryption_key")
    
    log "INFO" "Checking secrets for environment: $env"
    
    for secret in "${required_secrets[@]}"; do
        if [[ ! -f "$secrets_dir/$secret" ]]; then
            log "ERROR" "Required secret file missing: $secrets_dir/$secret"
            return 1
        fi
    done
    
    log "INFO" "All required secrets are present"
}

compare_configurations() {
    local env1="$1"
    local env2="$2"
    
    log "INFO" "Comparing configurations: $env1 vs $env2"
    
    local config1="$CONFIG_DIR/$env1.json"
    local config2="$CONFIG_DIR/$env2.json"
    
    if [[ ! -f "$config1" || ! -f "$config2" ]]; then
        log "ERROR" "One or both configuration files not found"
        return 1
    fi
    
    echo -e "${BLUE}Configuration differences between $env1 and $env2:${NC}"
    diff -u "$config1" "$config2" | head -50 || true
}

generate_templates() {
    log "INFO" "Generating configuration templates"
    
    # Generate environment-specific templates
    local environments=("development" "staging" "production" "test")
    
    for env in "${environments[@]}"; do
        local template_file="$CONFIG_DIR/${env}.template.json"
        if [[ "$DRY_RUN" == "false" ]]; then
            cd "$PROJECT_ROOT"
            python3 -c "
import sys
sys.path.append('src')
from common.config_manager import ConfigManager, ConfigFormat

try:
    manager = ConfigManager(environment='$env')
    exported = manager.export_config(ConfigFormat.JSON, include_secrets=False)
    with open('$template_file', 'w') as f:
        f.write(exported)
    print('Generated template: $template_file')
except Exception as e:
    print(f'Error generating template: {e}', file=sys.stderr)
"
        else
            log "INFO" "[DRY RUN] Would generate template: $template_file"
        fi
    done
}

show_status() {
    log "INFO" "XORB Configuration Status"
    
    echo -e "\n${BLUE}Current Environment:${NC} ${ENVIRONMENT}"
    echo -e "${BLUE}Config Directory:${NC} ${CONFIG_DIR}"
    
    echo -e "\n${BLUE}Available Configurations:${NC}"
    for config_file in "$CONFIG_DIR"/*.json; do
        if [[ -f "$config_file" ]]; then
            local env_name=$(basename "$config_file" .json)
            local status="✅"
            
            # Quick validation
            if ! jq empty "$config_file" 2>/dev/null; then
                status="❌"
            fi
            
            echo -e "  $status $env_name"
        fi
    done
    
    echo -e "\n${BLUE}Docker Compose Files:${NC}"
    for compose_file in "$PROJECT_ROOT"/docker-compose.*.yml; do
        if [[ -f "$compose_file" ]]; then
            echo -e "  ✅ $(basename "$compose_file")"
        fi
    done
}

hot_reload() {
    log "INFO" "Triggering configuration hot-reload"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        cd "$PROJECT_ROOT"
        python3 -c "
import sys
sys.path.append('src')
from common.config_manager import get_config_manager

try:
    manager = get_config_manager()
    manager.reload_config()
    print('✅ Configuration reloaded successfully')
except Exception as e:
    print(f'❌ Failed to reload configuration: {e}')
    sys.exit(1)
"
    else
        log "INFO" "[DRY RUN] Would reload configuration"
    fi
}

backup_configuration() {
    local env="$1"
    local backup_dir="$PROJECT_ROOT/backups/config"
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_file="$backup_dir/${env}_config_${timestamp}.json"
    
    log "INFO" "Backing up configuration for $env"
    
    mkdir -p "$backup_dir"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        cp "$CONFIG_DIR/$env.json" "$backup_file"
        log "INFO" "Configuration backed up to: $backup_file"
    else
        log "INFO" "[DRY RUN] Would backup to: $backup_file"
    fi
}

restore_configuration() {
    local env="$1"
    local backup_file="$2"
    
    log "INFO" "Restoring configuration for $env from $backup_file"
    
    if [[ ! -f "$backup_file" ]]; then
        log "ERROR" "Backup file not found: $backup_file"
        return 1
    fi
    
    # Validate backup file
    if ! jq empty "$backup_file" 2>/dev/null; then
        log "ERROR" "Invalid backup file format"
        return 1
    fi
    
    if [[ "$DRY_RUN" == "false" ]]; then
        # Backup current config first
        backup_configuration "$env"
        
        # Restore from backup
        cp "$backup_file" "$CONFIG_DIR/$env.json"
        log "INFO" "Configuration restored for $env"
    else
        log "INFO" "[DRY RUN] Would restore configuration from $backup_file"
    fi
}

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            --config-dir)
                CONFIG_DIR="$2"
                shift 2
                ;;
            validate|deploy|switch|export|secrets|diff|template|status|reload|backup|restore)
                COMMAND="$1"
                shift
                break
                ;;
            *)
                log "ERROR" "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    if [[ -z "$COMMAND" ]]; then
        log "ERROR" "No command specified"
        usage
        exit 1
    fi
    
    # Check dependencies
    check_dependencies
    
    # Create config directory if it doesn't exist
    mkdir -p "$CONFIG_DIR"
    
    log "DEBUG" "Command: $COMMAND, Environment: $ENVIRONMENT, DRY_RUN: $DRY_RUN"
    
    # Execute command
    case "$COMMAND" in
        validate)
            validate_environment "${1:-$ENVIRONMENT}"
            ;;
        deploy)
            deploy_environment "${1:-$ENVIRONMENT}"
            ;;
        switch)
            if [[ $# -eq 0 ]]; then
                log "ERROR" "Environment required for switch command"
                exit 1
            fi
            switch_environment "$1"
            ;;
        export)
            export_configuration "${1:-$ENVIRONMENT}" "${2:-json}"
            ;;
        secrets)
            check_secrets "${1:-$ENVIRONMENT}"
            ;;
        diff)
            if [[ $# -lt 2 ]]; then
                log "ERROR" "Two environments required for diff command"
                exit 1
            fi
            compare_configurations "$1" "$2"
            ;;
        template)
            generate_templates
            ;;
        status)
            show_status
            ;;
        reload)
            hot_reload
            ;;
        backup)
            backup_configuration "${1:-$ENVIRONMENT}"
            ;;
        restore)
            if [[ $# -lt 2 ]]; then
                log "ERROR" "Environment and backup file required for restore command"
                exit 1
            fi
            restore_configuration "$1" "$2"
            ;;
        *)
            log "ERROR" "Unknown command: $COMMAND"
            usage
            exit 1
            ;;
    esac
}

main "$@"