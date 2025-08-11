# XORB Directory Structure

This document describes the organized directory structure following best practices.

##  Root Directory Structure

```
/
├── .archive/                    # Archived/deprecated files
├── build/                       # Build outputs and artifacts
│   ├── dist/                   # Distribution files
│   └── artifacts/              # Build artifacts
├── config/                     # Configuration files
├── deploy/                     # Deployment configurations
│   ├── configs/               # Docker compose, K8s configs
│   └── environments/          # Environment-specific configs
├── documentation/              # All documentation
│   ├── api/                   # API documentation
│   ├── user-guides/           # User guides and READMEs
│   └── development/           # Development documentation
├── infra/                     # Infrastructure code
├── legacy/                    # Legacy code (for reference)
├── runtime/                   # Runtime files
│   ├── logs/                  # Application logs
│   ├── data/                  # Runtime data files
│   └── tmp/                   # Temporary files
├── secrets/                   # Secrets and certificates
├── src/                       # Source code
├── ssl/                       # SSL certificates
├── tests/                     # Test files
├── tools/                     # Development tools
│   ├── scripts/              # Build and deployment scripts
│   └── utilities/            # Utility scripts and tools
├── ptaas-backend/            # PTaaS backend service
├── ptaas-frontend/           # PTaaS frontend application
├── ptaas/                    # PTaaS core
├── xorb/                     # XORB core modules
└── [core files]              # Core project files (README, LICENSE, etc.)
```

##  Directory Purposes

###  Core Directories
- **src/**: Main application source code
- **tests/**: All test files
- **config/**: Configuration files and settings

###  Development & Build
- **tools/**: Development tools, scripts, and utilities
- **build/**: Build outputs, distributions, and artifacts
- **deploy/**: Deployment configurations and scripts

###  Documentation
- **documentation/**: All project documentation organized by type
  - API docs, user guides, development docs

###  Runtime
- **runtime/**: Files generated or used during application execution
  - Logs, temporary data, runtime artifacts

###  Infrastructure
- **infra/**: Infrastructure as code, Docker files, K8s manifests
- **secrets/**: Sensitive configuration files
- **ssl/**: SSL certificates and keys

###  Archive
- **.archive/**: Deprecated or old files kept for reference

##  Best Practices Applied

1. **Separation of Concerns**: Different types of files in different directories
2. **Clear Naming**: Descriptive directory names
3. **Consistent Structure**: Hierarchical organization
4. **Runtime Isolation**: Runtime files separated from source
5. **Documentation Centralization**: All docs in one place
6. **Tool Organization**: Scripts and utilities properly categorized
7. **Archive Strategy**: Old/deprecated content preserved but separated

##  Migration Notes

- Log files moved from root to `runtime/logs/`
- Documentation consolidated in `documentation/`
- Scripts organized under `tools/`
- Docker configs moved to `deploy/configs/`
- Build artifacts will go to `build/`
- Old frontend backups archived in `.archive/`