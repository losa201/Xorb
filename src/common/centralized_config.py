"""
XORB Centralized Configuration Management
Consolidates all configuration sources with Vault integration.

This replaces scattered configuration files and provides:
- Centralized configuration management
- Vault integration for secrets
- Environment-aware configuration
- Configuration validation
- Hot-reload capabilities
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import asyncio

import yaml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

from .vault_client import VaultClient
from .security_utils import validate_secret_format

logger = logging.getLogger(__name__)

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

# ============================================================================
# CONFIGURATION MODELS
# ============================================================================

class DatabaseConfig(BaseModel):
    """Database configuration with validation"""
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    name: str = Field(default="xorb", description="Database name")
    username: str = Field(default="xorb", description="Database username")
    password: str = Field(description="Database password")
    max_connections: int = Field(default=20, ge=1, le=100)
    ssl_mode: str = Field(default="prefer", regex="^(disable|allow|prefer|require|verify-ca|verify-full)$")

    @property
    def url(self) -> str:
        """Get database URL"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.name}"

class RedisConfig(BaseModel):
    """Redis configuration with validation"""
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    password: Optional[str] = Field(default=None, description="Redis password")
    database: int = Field(default=0, ge=0, le=15, description="Redis database number")
    max_connections: int = Field(default=10, ge=1, le=50)

    @property
    def url(self) -> str:
        """Get Redis URL"""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.database}"

class SecurityConfig(BaseModel):
    """Security configuration"""
    jwt_secret: str = Field(description="JWT signing secret")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiry_minutes: int = Field(default=30, ge=5, le=1440)
    password_min_length: int = Field(default=12, ge=8, le=128)
    max_login_attempts: int = Field(default=5, ge=1, le=20)
    lockout_duration_minutes: int = Field(default=15, ge=1, le=1440)
    require_mfa: bool = Field(default=False, description="Require MFA for all users")

    @validator('jwt_secret')
    def validate_jwt_secret(cls, v):
        if len(v) < 32:
            raise ValueError("JWT secret must be at least 32 characters")
        return v

class ServiceConfig(BaseModel):
    """Individual service configuration"""
    name: str
    host: str = Field(default="0.0.0.0")
    port: int = Field(ge=1024, le=65535)
    workers: int = Field(default=1, ge=1, le=10)
    timeout: int = Field(default=30, ge=5, le=300)
    enabled: bool = Field(default=True)

class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration"""
    prometheus_enabled: bool = Field(default=True)
    prometheus_port: int = Field(default=9090, ge=1024, le=65535)
    grafana_enabled: bool = Field(default=True)
    grafana_port: int = Field(default=3000, ge=1024, le=65535)
    log_level: LogLevel = Field(default=LogLevel.INFO)
    structured_logging: bool = Field(default=True)
    audit_logging: bool = Field(default=True)

class ExternalServicesConfig(BaseModel):
    """External service integrations"""
    nvidia_api_key: Optional[str] = Field(default=None, description="NVIDIA API key")
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API key")
    temporal_host: str = Field(default="temporal:7233", description="Temporal server")
    vault_url: str = Field(default="http://vault:8200", description="Vault server URL")
    vault_token: Optional[str] = Field(default=None, description="Vault token")

class XORBConfig(BaseSettings):
    """
    Centralized XORB platform configuration
    Loads from environment variables, config files, and Vault
    """

    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)

    # Core Services
    database: DatabaseConfig
    redis: RedisConfig
    security: SecurityConfig
    monitoring: MonitoringConfig
    external: ExternalServicesConfig

    # Service Definitions
    services: Dict[str, ServiceConfig] = Field(default_factory=dict)

    # Feature Flags
    features: Dict[str, bool] = Field(default_factory=lambda: {
        "advanced_analytics": True,
        "threat_hunting": True,
        "compliance_automation": True,
        "multi_tenant": False,
        "sso_integration": False,
        "ai_enhancement": True
    })

    # CORS and API settings
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    api_version: str = Field(default="v1")
    max_request_size: int = Field(default=10485760)  # 10MB

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False

        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            return (
                init_settings,
                env_settings,
                file_secret_settings,
            )

# ============================================================================
# CONFIGURATION MANAGER
# ============================================================================

class ConfigurationManager:
    """
    Centralized configuration manager with Vault integration
    """

    def __init__(self, config_dir: Path = None, vault_client: VaultClient = None):
        self.config_dir = config_dir or Path("config")
        self.vault_client = vault_client
        self._config: Optional[XORBConfig] = None
        self._config_cache: Dict[str, Any] = {}

    async def load_configuration(self, environment: Environment = None) -> XORBConfig:
        """Load and validate configuration from all sources"""
        try:
            # Determine environment
            env = environment or Environment(os.getenv("XORB_ENV", "development"))

            logger.info(f"Loading configuration for environment: {env.value}")

            # Load base configuration
            base_config = await self._load_base_config(env)

            # Load secrets from Vault
            vault_config = await self._load_vault_secrets() if self.vault_client else {}

            # Merge configurations (Vault secrets override base config)
            merged_config = self._merge_configs(base_config, vault_config)

            # Create and validate configuration
            self._config = XORBConfig(**merged_config)

            # Cache for quick access
            self._config_cache = merged_config

            logger.info("Configuration loaded successfully")
            return self._config

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    async def _load_base_config(self, env: Environment) -> Dict[str, Any]:
        """Load base configuration from files and environment"""
        config = {}

        # Load from config files
        config_files = [
            self.config_dir / "default.yaml",
            self.config_dir / f"{env.value}.yaml",
            self.config_dir / "local.yaml"  # Local overrides
        ]

        for config_file in config_files:
            if config_file.exists():
                with open(config_file) as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        config = self._deep_merge(config, file_config)
                        logger.debug(f"Loaded config from {config_file}")

        # Load from environment variables
        env_config = self._load_env_config()
        config = self._deep_merge(config, env_config)

        return config

    def _load_env_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {}

        # Database configuration
        if os.getenv("DATABASE_URL"):
            env_config["database"] = self._parse_database_url(os.getenv("DATABASE_URL"))
        else:
            env_config["database"] = {
                "host": os.getenv("DATABASE_HOST", "localhost"),
                "port": int(os.getenv("DATABASE_PORT", "5432")),
                "name": os.getenv("DATABASE_NAME", "xorb"),
                "username": os.getenv("DATABASE_USERNAME", "xorb"),
                "password": os.getenv("DATABASE_PASSWORD", ""),
            }

        # Redis configuration
        if os.getenv("REDIS_URL"):
            env_config["redis"] = self._parse_redis_url(os.getenv("REDIS_URL"))
        else:
            env_config["redis"] = {
                "host": os.getenv("REDIS_HOST", "localhost"),
                "port": int(os.getenv("REDIS_PORT", "6379")),
                "password": os.getenv("REDIS_PASSWORD"),
                "database": int(os.getenv("REDIS_DATABASE", "0")),
            }

        # Security configuration
        env_config["security"] = {
            "jwt_secret": os.getenv("JWT_SECRET", ""),
            "jwt_algorithm": os.getenv("JWT_ALGORITHM", "HS256"),
            "jwt_expiry_minutes": int(os.getenv("JWT_EXPIRY_MINUTES", "30")),
        }

        # External services
        env_config["external"] = {
            "nvidia_api_key": os.getenv("NVIDIA_API_KEY"),
            "openrouter_api_key": os.getenv("OPENROUTER_API_KEY"),
            "temporal_host": os.getenv("TEMPORAL_HOST", "temporal:7233"),
            "vault_url": os.getenv("VAULT_URL", "http://vault:8200"),
            "vault_token": os.getenv("VAULT_TOKEN"),
        }

        # Environment
        env_config["environment"] = os.getenv("XORB_ENV", "development")
        env_config["debug"] = os.getenv("DEBUG", "false").lower() == "true"

        return env_config

    async def _load_vault_secrets(self) -> Dict[str, Any]:
        """Load secrets from Vault"""
        if not self.vault_client:
            return {}

        try:
            vault_config = {}

            # Load main configuration secrets
            config_secrets = await self.vault_client.get_secret("xorb/config")
            if config_secrets:
                vault_config.update(config_secrets)

            # Load external service secrets
            external_secrets = await self.vault_client.get_secret("xorb/external")
            if external_secrets:
                vault_config["external"] = external_secrets

            # Get dynamic database credentials
            db_creds = await self.vault_client.get_database_credentials("xorb-app")
            if db_creds:
                vault_config.setdefault("database", {}).update(db_creds)

            logger.info("Loaded secrets from Vault")
            return vault_config

        except Exception as e:
            logger.warning(f"Failed to load secrets from Vault: {e}")
            return {}

    def _parse_database_url(self, url: str) -> Dict[str, str]:
        """Parse DATABASE_URL into components"""
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        return {
            "host": parsed.hostname,
            "port": parsed.port or 5432,
            "name": parsed.path[1:] if parsed.path else "xorb",
            "username": parsed.username,
            "password": parsed.password,
        }

    def _parse_redis_url(self, url: str) -> Dict[str, Any]:
        """Parse REDIS_URL into components"""
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        return {
            "host": parsed.hostname,
            "port": parsed.port or 6379,
            "password": parsed.password,
            "database": int(parsed.path[1:]) if parsed.path else 0,
        }

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _merge_configs(self, base: Dict, vault: Dict) -> Dict:
        """Merge base config with Vault secrets"""
        # Vault secrets take precedence
        return self._deep_merge(base, vault)

    def get_config(self) -> XORBConfig:
        """Get current configuration"""
        if not self._config:
            raise ValueError("Configuration not loaded. Call load_configuration() first.")
        return self._config

    def get_service_config(self, service_name: str) -> ServiceConfig:
        """Get configuration for specific service"""
        config = self.get_config()
        if service_name not in config.services:
            raise ValueError(f"Service '{service_name}' not found in configuration")
        return config.services[service_name]

    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature flag is enabled"""
        config = self.get_config()
        return config.features.get(feature_name, False)

    async def refresh_secrets(self):
        """Refresh secrets from Vault"""
        if not self.vault_client:
            logger.warning("No Vault client configured, cannot refresh secrets")
            return

        try:
            # Reload Vault secrets
            vault_config = await self._load_vault_secrets()

            # Merge with cached config
            merged_config = self._merge_configs(self._config_cache, vault_config)

            # Update configuration
            self._config = XORBConfig(**merged_config)
            self._config_cache = merged_config

            logger.info("Configuration secrets refreshed from Vault")

        except Exception as e:
            logger.error(f"Failed to refresh secrets: {e}")

    def validate_configuration(self) -> List[str]:
        """Validate current configuration and return any issues"""
        issues = []
        config = self.get_config()

        # Validate required secrets
        if not config.security.jwt_secret or len(config.security.jwt_secret) < 32:
            issues.append("JWT secret is missing or too short (minimum 32 characters)")

        if not config.database.password:
            issues.append("Database password is required")

        # Validate service configurations
        for service_name, service_config in config.services.items():
            if service_config.enabled and service_config.port < 1024:
                issues.append(f"Service '{service_name}' using privileged port {service_config.port}")

        # Production-specific validations
        if config.environment == Environment.PRODUCTION:
            if config.debug:
                issues.append("Debug mode should not be enabled in production")

            if "*" in config.cors_origins:
                issues.append("CORS should not allow all origins in production")

            if not config.security.require_mfa:
                issues.append("MFA should be required in production")

        return issues

# ============================================================================
# GLOBAL CONFIGURATION INSTANCE
# ============================================================================

# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None

async def initialize_config(
    config_dir: Path = None,
    vault_client: VaultClient = None,
    environment: Environment = None
) -> ConfigurationManager:
    """Initialize global configuration manager"""
    global _config_manager

    _config_manager = ConfigurationManager(config_dir, vault_client)
    await _config_manager.load_configuration(environment)

    # Validate configuration
    issues = _config_manager.validate_configuration()
    if issues:
        logger.warning("Configuration validation issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")

    return _config_manager

def get_config() -> XORBConfig:
    """Get global configuration instance"""
    if not _config_manager:
        raise ValueError("Configuration not initialized. Call initialize_config() first.")
    return _config_manager.get_config()

def get_config_manager() -> ConfigurationManager:
    """Get global configuration manager instance"""
    if not _config_manager:
        raise ValueError("Configuration not initialized. Call initialize_config() first.")
    return _config_manager
