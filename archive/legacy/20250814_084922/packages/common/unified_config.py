"""
Unified Configuration System for XORB Platform
Consolidates all configuration classes into a single, comprehensive system
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from functools import lru_cache
from enum import Enum

from .secret_manager import get_secret, get_database_url, get_jwt_secret


class Environment(Enum):
    """Deployment environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = field(default_factory=get_database_url)
    pool_size: int = field(default_factory=lambda: int(os.getenv("DB_POOL_SIZE", "20")))
    max_connections: int = field(default_factory=lambda: int(os.getenv("DB_MAX_CONNECTIONS", "50")))
    ssl_mode: str = field(default_factory=lambda: os.getenv("DB_SSL_MODE", "prefer"))
    echo: bool = field(default_factory=lambda: os.getenv("DB_ECHO", "false").lower() == "true")
    pool_pre_ping: bool = True
    pool_recycle: int = 3600  # 1 hour


@dataclass
class RedisConfig:
    """Redis configuration"""
    url: str = field(default_factory=lambda: get_secret("REDIS_URL", "redis://localhost:6379"))
    password: Optional[str] = field(default_factory=lambda: get_secret("REDIS_PASSWORD", None))
    db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    max_connections: int = field(default_factory=lambda: int(os.getenv("REDIS_MAX_CONNECTIONS", "20")))
    socket_timeout: int = field(default_factory=lambda: int(os.getenv("REDIS_SOCKET_TIMEOUT", "5")))
    socket_connect_timeout: int = field(default_factory=lambda: int(os.getenv("REDIS_CONNECT_TIMEOUT", "5")))


@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret: str = field(default_factory=get_jwt_secret)
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = field(default_factory=lambda: int(os.getenv("JWT_EXPIRE_MINUTES", "30")))
    refresh_token_expire_days: int = field(default_factory=lambda: int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7")))

    # Password security
    password_min_length: int = field(default_factory=lambda: int(os.getenv("PASSWORD_MIN_LENGTH", "12")))
    password_require_special_chars: bool = field(default_factory=lambda: os.getenv("PASSWORD_REQUIRE_SPECIAL", "true").lower() == "true")
    max_login_attempts: int = field(default_factory=lambda: int(os.getenv("MAX_LOGIN_ATTEMPTS", "5")))
    lockout_duration_minutes: int = field(default_factory=lambda: int(os.getenv("LOCKOUT_DURATION_MINUTES", "30")))

    # Argon2 configuration
    argon2_time_cost: int = field(default_factory=lambda: int(os.getenv("ARGON2_TIME_COST", "3")))
    argon2_memory_cost: int = field(default_factory=lambda: int(os.getenv("ARGON2_MEMORY_COST", "65536")))
    argon2_parallelism: int = field(default_factory=lambda: int(os.getenv("ARGON2_PARALLELISM", "2")))

    # Zero trust features
    device_fingerprint_required: bool = field(default_factory=lambda: os.getenv("DEVICE_FINGERPRINT_REQUIRED", "true").lower() == "true")
    geolocation_monitoring: bool = field(default_factory=lambda: os.getenv("GEOLOCATION_MONITORING", "true").lower() == "true")
    behavioral_analysis_required: bool = field(default_factory=lambda: os.getenv("BEHAVIORAL_ANALYSIS_REQUIRED", "true").lower() == "true")

    # MFA configuration
    mfa_required: bool = field(default_factory=lambda: os.getenv("MFA_REQUIRED", "true").lower() == "true")
    mfa_methods: List[str] = field(default_factory=lambda: os.getenv("MFA_METHODS", "totp,webauthn,sms").split(","))


@dataclass
class APIConfig:
    """API server configuration"""
    host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    workers: int = field(default_factory=lambda: int(os.getenv("API_WORKERS", "1")))
    cors_origins: List[str] = field(default_factory=lambda: [
        origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    ])
    max_request_size: int = field(default_factory=lambda: int(os.getenv("MAX_REQUEST_SIZE", "16777216")))  # 16MB
    request_timeout: int = field(default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "30")))


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_minute: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")))
    requests_per_hour: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_HOUR", "1000")))
    burst_capacity: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_BURST", "50")))
    window_seconds: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_WINDOW", "60")))


@dataclass
class SSOConfig:
    """Single Sign-On configuration"""
    # Azure AD
    azure_enabled: bool = field(default_factory=lambda: get_secret("AZURE_SSO_ENABLED", "false").lower() == "true")
    azure_client_id: str = field(default_factory=lambda: get_secret("AZURE_CLIENT_ID", ""))
    azure_client_secret: str = field(default_factory=lambda: get_secret("AZURE_CLIENT_SECRET", ""))
    azure_tenant: str = field(default_factory=lambda: get_secret("AZURE_TENANT", "common"))

    # Google
    google_enabled: bool = field(default_factory=lambda: get_secret("GOOGLE_SSO_ENABLED", "false").lower() == "true")
    google_client_id: str = field(default_factory=lambda: get_secret("GOOGLE_CLIENT_ID", ""))
    google_client_secret: str = field(default_factory=lambda: get_secret("GOOGLE_CLIENT_SECRET", ""))

    # Okta
    okta_enabled: bool = field(default_factory=lambda: get_secret("OKTA_SSO_ENABLED", "false").lower() == "true")
    okta_client_id: str = field(default_factory=lambda: get_secret("OKTA_CLIENT_ID", ""))
    okta_client_secret: str = field(default_factory=lambda: get_secret("OKTA_CLIENT_SECRET", ""))
    okta_domain: str = field(default_factory=lambda: get_secret("OKTA_DOMAIN", ""))

    # GitHub
    github_enabled: bool = field(default_factory=lambda: get_secret("GITHUB_SSO_ENABLED", "false").lower() == "true")
    github_client_id: str = field(default_factory=lambda: get_secret("GITHUB_CLIENT_ID", ""))
    github_client_secret: str = field(default_factory=lambda: get_secret("GITHUB_CLIENT_SECRET", ""))


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    metrics_enabled: bool = field(default_factory=lambda: os.getenv("METRICS_ENABLED", "true").lower() == "true")
    prometheus_port: int = field(default_factory=lambda: int(os.getenv("PROMETHEUS_PORT", "9090")))
    grafana_port: int = field(default_factory=lambda: int(os.getenv("GRAFANA_PORT", "3000")))
    tracing_enabled: bool = field(default_factory=lambda: os.getenv("TRACING_ENABLED", "true").lower() == "true")
    jaeger_endpoint: str = field(default_factory=lambda: os.getenv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces"))


@dataclass
class ServiceConfig:
    """External service configuration"""
    temporal_host: str = field(default_factory=lambda: os.getenv("TEMPORAL_HOST", "localhost:7233"))
    scanner_service_url: str = field(default_factory=lambda: os.getenv("SCANNER_SERVICE_URL", "http://localhost:8001"))
    compliance_service_url: str = field(default_factory=lambda: os.getenv("COMPLIANCE_SERVICE_URL", "http://localhost:8002"))
    notification_service_url: str = field(default_factory=lambda: os.getenv("NOTIFICATION_SERVICE_URL", "http://localhost:8003"))


@dataclass
class StorageConfig:
    """File storage configuration"""
    upload_dir: str = field(default_factory=lambda: os.getenv("UPLOAD_DIR", "./uploads"))
    max_file_size_mb: int = field(default_factory=lambda: int(os.getenv("MAX_FILE_SIZE_MB", "100")))
    allowed_extensions: List[str] = field(default_factory=lambda: [
        ext.strip() for ext in os.getenv("ALLOWED_FILE_EXTENSIONS", ".txt,.pdf,.doc,.docx,.json,.csv").split(",")
    ])

    # S3 configuration (optional)
    s3_enabled: bool = field(default_factory=lambda: os.getenv("S3_ENABLED", "false").lower() == "true")
    s3_bucket: str = field(default_factory=lambda: os.getenv("S3_BUCKET", ""))
    s3_region: str = field(default_factory=lambda: os.getenv("S3_REGION", "us-east-1"))
    s3_access_key: str = field(default_factory=lambda: get_secret("S3_ACCESS_KEY", ""))
    s3_secret_key: str = field(default_factory=lambda: get_secret("S3_SECRET_KEY", ""))


@dataclass
class EPYCConfig:
    """AMD EPYC optimization configuration"""
    numa_nodes: int = field(default_factory=lambda: int(os.getenv("NUMA_NODES", "2")))
    cpu_cores: int = field(default_factory=lambda: int(os.getenv("CPU_CORES", "16")))
    memory_gb: int = field(default_factory=lambda: int(os.getenv("MEMORY_GB", "32")))
    enable_numa_optimization: bool = field(default_factory=lambda: os.getenv("ENABLE_NUMA_OPTIMIZATION", "true").lower() == "true")
    cpu_affinity_enabled: bool = field(default_factory=lambda: os.getenv("CPU_AFFINITY_ENABLED", "false").lower() == "true")


@dataclass
class UnifiedConfig:
    """
    Unified Configuration System
    Consolidates all configuration classes into a single, comprehensive system
    """

    # Environment
    environment: Environment = field(default_factory=lambda: Environment(os.getenv("ENVIRONMENT", "development")))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    log_level: LogLevel = field(default_factory=lambda: LogLevel(os.getenv("LOG_LEVEL", "INFO")))

    # Application info
    app_name: str = "XORB PTaaS Platform"
    app_version: str = "1.0.0"
    app_description: str = "Penetration Testing as a Service Platform"

    # Configuration sections
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    api: APIConfig = field(default_factory=APIConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    sso: SSOConfig = field(default_factory=SSOConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    services: ServiceConfig = field(default_factory=ServiceConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    epyc: EPYCConfig = field(default_factory=EPYCConfig)

    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == Environment.PRODUCTION

    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == Environment.DEVELOPMENT

    def is_testing(self) -> bool:
        """Check if running in testing"""
        return self.environment == Environment.TESTING

    def get_sso_providers(self) -> List[str]:
        """Get list of enabled SSO providers"""
        providers = []
        if self.sso.azure_enabled and self.sso.azure_client_id:
            providers.append("azure")
        if self.sso.google_enabled and self.sso.google_client_id:
            providers.append("google")
        if self.sso.okta_enabled and self.sso.okta_client_id:
            providers.append("okta")
        if self.sso.github_enabled and self.sso.github_client_id:
            providers.append("github")
        return providers

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []

        # Production validation
        if self.is_production():
            if not self.security.jwt_secret or self.security.jwt_secret == "dev-secret-change-in-production":
                errors.append("JWT_SECRET must be set in production")

            if "localhost" in self.database.url:
                errors.append("Production should not use localhost database")

            if self.debug:
                errors.append("Debug mode should be disabled in production")

        # SSO validation
        if self.sso.azure_enabled and (not self.sso.azure_client_id or not self.sso.azure_client_secret):
            errors.append("Azure SSO enabled but missing client credentials")

        if self.sso.google_enabled and (not self.sso.google_client_id or not self.sso.google_client_secret):
            errors.append("Google SSO enabled but missing client credentials")

        if self.sso.okta_enabled and (not self.sso.okta_client_id or not self.sso.okta_client_secret or not self.sso.okta_domain):
            errors.append("Okta SSO enabled but missing client credentials or domain")

        if self.sso.github_enabled and (not self.sso.github_client_id or not self.sso.github_client_secret):
            errors.append("GitHub SSO enabled but missing client credentials")

        # Storage validation
        if self.storage.s3_enabled and (not self.storage.s3_bucket or not self.storage.s3_access_key):
            errors.append("S3 storage enabled but missing configuration")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (for serialization)"""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, '__dict__'):
                result[field_name] = field_value.__dict__
            else:
                result[field_name] = field_value
        return result


@lru_cache()
def get_config() -> UnifiedConfig:
    """Get cached unified configuration instance"""
    return UnifiedConfig()


def validate_config() -> bool:
    """Validate configuration and print results"""
    config = get_config()
    errors = config.validate()

    if errors:
        print("❌ Configuration validation failed:")
        for error in errors:
            print(f"   • {error}")
        return False

    print("✅ Configuration validation passed")

    # Print configuration summary
    print(f"   Environment: {config.environment.value}")
    print(f"   Debug Mode: {config.debug}")
    print(f"   Database: {config.database.url.split('@')[-1] if '@' in config.database.url else 'localhost'}")

    sso_providers = config.get_sso_providers()
    if sso_providers:
        print(f"   SSO Providers: {', '.join(sso_providers)}")
    else:
        print("   SSO Providers: None (local auth only)")

    return True


# Environment-specific configurations
def get_development_config() -> UnifiedConfig:
    """Get development-specific configuration"""
    config = UnifiedConfig()
    config.environment = Environment.DEVELOPMENT
    config.debug = True
    config.log_level = LogLevel.DEBUG
    config.api.workers = 1
    return config


def get_production_config() -> UnifiedConfig:
    """Get production-specific configuration"""
    config = UnifiedConfig()
    config.environment = Environment.PRODUCTION
    config.debug = False
    config.log_level = LogLevel.INFO
    config.api.workers = 4
    return config


def get_testing_config() -> UnifiedConfig:
    """Get testing-specific configuration"""
    config = UnifiedConfig()
    config.environment = Environment.TESTING
    config.debug = True
    config.log_level = LogLevel.DEBUG
    config.database.url = "sqlite:///test.db"
    config.redis.url = "redis://localhost:6379/1"
    return config
