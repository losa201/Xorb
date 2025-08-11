#  XORB Centralized Configuration Management

##  Overview

The XORB platform now features a comprehensive centralized configuration management system that provides:

- **Environment-specific configurations** (development, staging, production, test)
- **Secret management integration** with HashiCorp Vault
- **Hot-reloading capabilities** for configuration changes
- **Feature flags** for gradual rollouts
- **Configuration validation** and error handling
- **Multi-format export** (JSON, YAML, Environment Variables)

##  Quick Start

###  1. Basic Usage

```python
from src.common.config_manager import get_config, get_feature_flag

#  Get current configuration
config = get_config()
print(f"Database URL: {config.database.get_url()}")
print(f"API Port: {config.api_service.port}")

#  Check feature flags
if get_feature_flag("advanced_analytics", False):
    print("Advanced analytics enabled")
```

###  2. Environment Switching

```bash
#  Switch to different environments
./tools/scripts/config-manager.sh switch development
./tools/scripts/config-manager.sh switch production

#  Validate configuration
./tools/scripts/config-manager.sh validate production

#  Deploy environment
./tools/scripts/config-manager.sh deploy staging --dry-run
```

###  3. Docker Compose Deployment

```bash
#  Development environment
docker-compose -f docker-compose.development.yml up -d

#  Production environment
docker-compose -f docker-compose.production.yml up -d

#  Check status
docker-compose ps
```

##  Configuration Structure

###  Environment Files

The system uses JSON configuration files for each environment:

- `config/development.json` - Development settings
- `config/staging.json` - Staging settings
- `config/production.json` - Production settings
- `config/test.json` - Test settings
- `config/default.json` - Base configuration

###  Configuration Schema

```python
@dataclass
class XORBConfig:
    environment: Environment
    app_name: str
    app_version: str
    debug: bool

    # Service configurations
    database: DatabaseConfig
    redis: RedisConfig
    security: SecurityConfig
    monitoring: MonitoringConfig
    api_service: ServiceConfig
    orchestrator_service: ServiceConfig
    intelligence_service: ServiceConfig

    # Feature flags
    feature_flags: Dict[str, bool]

    # External integrations
    integrations: Dict[str, Dict[str, Any]]
```

##  Environment-Specific Settings

###  Development Environment

```json
{
  "environment": "development",
  "debug": true,
  "database": {
    "name": "xorb_dev",
    "pool_size": 5
  },
  "feature_flags": {
    "debug_mode": true,
    "performance_profiling": true
  }
}
```

###  Production Environment

```json
{
  "environment": "production",
  "debug": false,
  "database": {
    "name": "xorb_prod",
    "ssl_mode": "require",
    "pool_size": 20
  },
  "security": {
    "jwt_expire_hours": 8,
    "password_min_length": 12
  }
}
```

##  Secret Management

###  Vault Integration

The system integrates with HashiCorp Vault for secure secret management:

```python
#  Secrets are automatically loaded from Vault
config = get_config()
db_url = config.database.get_url()  # Password loaded from Vault

#  Manual secret access
vault_client = VaultClient()
secret = vault_client.get_secret("xorb/config")
```

###  Environment Variables

Fallback to environment variables when Vault is unavailable:

```bash
export DATABASE_PASSWORD="secure_password"
export JWT_SECRET="very_long_secret_key"
export REDIS_PASSWORD="redis_password"
```

##  Feature Flags

###  Dynamic Feature Control

```python
from src.common.config_manager import get_config_manager

manager = get_config_manager()

#  Get feature flags
analytics_enabled = manager.get_feature_flag("advanced_analytics", False)
multi_tenant = manager.get_feature_flag("multi_tenant", False)

#  Set feature flags dynamically
manager.set_feature_flag("beta_features", True)
```

###  Available Feature Flags

- `advanced_analytics` - Advanced analytics and reporting
- `threat_hunting` - Threat hunting capabilities
- `compliance_automation` - Automated compliance checking
- `multi_tenant` - Multi-tenant support
- `sso_integration` - Single Sign-On integration
- `debug_mode` - Debug mode features
- `performance_profiling` - Performance profiling tools

##  Hot-Reloading

###  Automatic Reload

Configuration files are monitored for changes and automatically reloaded:

```python
#  Register callback for configuration changes
def on_config_changed(new_config):
    print(f"Configuration updated for {new_config.environment}")

manager = get_config_manager()
manager.register_reload_callback(on_config_changed)
```

###  Manual Reload

```python
from src.common.config_manager import reload_config

#  Trigger manual reload
reload_config()
```

##  Configuration Management CLI

###  Available Commands

```bash
#  Validate configuration
./tools/scripts/config-manager.sh validate production

#  Deploy environment
./tools/scripts/config-manager.sh deploy staging

#  Switch environments
./tools/scripts/config-manager.sh switch development

#  Export configuration
./tools/scripts/config-manager.sh export production json

#  Compare configurations
./tools/scripts/config-manager.sh diff staging production

#  Show status
./tools/scripts/config-manager.sh status

#  Hot-reload
./tools/scripts/config-manager.sh reload
```

###  Options

- `-v, --verbose` - Verbose output
- `-d, --dry-run` - Show what would be done without executing
- `--config-dir DIR` - Use custom config directory

##  Service-Specific Configuration

###  Database Configuration

```python
config = get_config()
db_config = config.database

print(f"Host: {db_config.host}")
print(f"Port: {db_config.port}")
print(f"Database: {db_config.name}")
print(f"Connection URL: {db_config.get_url()}")
```

###  Service Ports

```python
#  Get service-specific configuration
api_config = manager.get_service_config("api")
orchestrator_config = manager.get_service_config("orchestrator")

print(f"API: {api_config.host}:{api_config.port}")
print(f"Orchestrator: {orchestrator_config.host}:{orchestrator_config.port}")
```

##  Docker Integration

###  Environment Variables

```yaml
#  docker-compose.yml
services:
  api:
    environment:
      - XORB_ENV=production
      - DATABASE_HOST=postgres
      - REDIS_HOST=redis
    volumes:
      - ./config:/app/config:ro
```

###  Secrets Management

```yaml
#  Production secrets
secrets:
  jwt_secret:
    file: ./secrets/jwt_secret
  encryption_key:
    file: ./secrets/encryption_key
```

##  Validation and Error Handling

###  Configuration Validation

The system validates configuration automatically:

```python
#  Validation rules
- Production environments require JWT secrets
- Service ports must be unique
- Database connection parameters must be valid
- Feature flag values must be boolean
```

###  Error Handling

```python
try:
    config = get_config()
except ValidationError as e:
    print(f"Configuration validation failed: {e}")
except Exception as e:
    print(f"Configuration error: {e}")
    # Falls back to default configuration
```

##  Export and Import

###  Export Configuration

```bash
#  Export as JSON
./tools/scripts/config-manager.sh export production json

#  Export as YAML
./tools/scripts/config-manager.sh export production yaml

#  Export as environment variables
./tools/scripts/config-manager.sh export production env
```

###  Programmatic Export

```python
from src.common.config_manager import get_config_manager, ConfigFormat

manager = get_config_manager()

#  Export without secrets
json_config = manager.export_config(ConfigFormat.JSON, include_secrets=False)
yaml_config = manager.export_config(ConfigFormat.YAML, include_secrets=False)
env_config = manager.export_config(ConfigFormat.ENV, include_secrets=False)
```

##  Migration from Legacy Configuration

###  Step 1: Update Imports

```python
#  Old way
from src.common.config import get_settings
settings = get_settings()

#  New way
from src.common.config_manager import get_config
config = get_config()
```

###  Step 2: Update Configuration Access

```python
#  Old way
database_url = settings.database_url
api_port = settings.api_port

#  New way
database_url = config.database.get_url()
api_port = config.api_service.port
```

###  Step 3: Update Environment Variables

The new system uses structured environment variables:

```bash
#  Old
DATABASE_URL=postgresql://...

#  New
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=xorb_dev
```

##  Best Practices

###  1. Environment Separation

- Use environment-specific configuration files
- Never commit secrets to version control
- Use Vault or environment variables for sensitive data

###  2. Feature Flags

- Use feature flags for gradual rollouts
- Test feature flags in staging before production
- Monitor feature flag usage and performance impact

###  3. Configuration Validation

- Validate configuration in CI/CD pipelines
- Use dry-run mode for testing deployments
- Monitor configuration changes in production

###  4. Hot-Reloading

- Use hot-reloading for development environments
- Be cautious with hot-reloading in production
- Test configuration changes thoroughly

##  Troubleshooting

###  Common Issues

1. **Configuration File Not Found**
   ```bash
   ./tools/scripts/config-manager.sh validate development
   # Check if config/development.json exists
   ```

2. **Invalid JSON Syntax**
   ```bash
   # Validate JSON syntax
   jq empty config/production.json
   ```

3. **Missing Secrets**
   ```bash
   # Check required secrets exist
   ls -la secrets/
   ```

4. **Port Conflicts**
   ```bash
   # Check for port conflicts
   ./tools/scripts/config-manager.sh validate production
   ```

###  Debugging

```python
#  Enable verbose logging
import logging
logging.getLogger("config_manager").setLevel(logging.DEBUG)

#  Check configuration status
from src.common.config_manager import get_config_manager
manager = get_config_manager()
config = manager.get_config()
print(f"Current environment: {config.environment}")
print(f"Debug mode: {config.debug}")
```

##  Performance Considerations

- Configuration is cached for performance
- Hot-reloading has minimal performance impact
- Vault integration adds slight latency but improves security
- Feature flags have negligible runtime overhead

##  Security Considerations

- Secrets are never logged or exported by default
- Configuration files should have appropriate permissions
- Vault tokens should be rotated regularly
- Environment variables should be secured in production

---

For more information, see the [XORB Platform Architecture Guide](XORB_PLATFORM_ARCHITECTURE_GUIDE.md).