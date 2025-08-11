"""
Production configuration management with environment-specific settings
"""

import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass, field

from pydantic import Field, validator
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseSettings
    # Create a fallback for SettingsConfigDict
    class SettingsConfigDict(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

from .logging import get_logger

logger = get_logger(__name__)


class AppSettings(BaseSettings):
    """Application configuration settings"""
    
    # Basic app settings
    app_name: str = Field(default="XORB Enterprise Cybersecurity Platform", env="APP_NAME")
    app_version: str = Field(default="3.0.0", env="APP_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # API settings
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    
    # Security settings
    jwt_secret_key: str = Field(default="dev-jwt-secret-key-change-in-production-12345678901234567890", env="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")  # Use HS256 for development
    jwt_expiration_minutes: int = Field(default=30, env="JWT_EXPIRATION_MINUTES")  # Longer for development
    jwt_refresh_expiration_days: int = Field(default=7, env="JWT_REFRESH_EXPIRATION_DAYS")  # Shorter refresh
    
    # Password policy
    min_password_length: int = Field(default=12, env="MIN_PASSWORD_LENGTH")
    require_mfa: bool = Field(default=True, env="REQUIRE_MFA")
    max_login_attempts: int = Field(default=5, env="MAX_LOGIN_ATTEMPTS")
    lockout_duration_minutes: int = Field(default=30, env="LOCKOUT_DURATION_MINUTES")
    
    # Database settings
    database_url: str = Field(default="postgresql://user:pass@localhost/xorb", env="DATABASE_URL")
    database_min_pool_size: int = Field(default=5, env="DB_MIN_POOL_SIZE")
    database_max_pool_size: int = Field(default=20, env="DB_MAX_POOL_SIZE")
    database_pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    database_command_timeout: int = Field(default=60, env="DB_COMMAND_TIMEOUT")
    
    # Redis settings
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    redis_socket_timeout: int = Field(default=5, env="REDIS_SOCKET_TIMEOUT")
    
    # Cache settings
    cache_backend: str = Field(default="hybrid", env="CACHE_BACKEND")
    cache_default_ttl: int = Field(default=3600, env="CACHE_DEFAULT_TTL")
    cache_max_size: int = Field(default=1000, env="CACHE_MAX_SIZE")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")
    rate_limit_per_day: int = Field(default=10000, env="RATE_LIMIT_PER_DAY")
    
    # CORS settings (use string to avoid JSON parsing issues)
    cors_allow_origins: str = Field(default="*", env="CORS_ALLOW_ORIGINS")
    cors_allow_methods: str = Field(default="GET,POST,PUT,DELETE,PATCH,OPTIONS", env="CORS_ALLOW_METHODS")
    cors_allow_headers: str = Field(default="*", env="CORS_ALLOW_HEADERS")
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_max_age: int = Field(default=3600, env="CORS_MAX_AGE")
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Monitoring settings
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    metrics_collection_interval: int = Field(default=60, env="METRICS_COLLECTION_INTERVAL")
    
    # External services
    temporal_host: str = Field(default="localhost:7233", env="TEMPORAL_HOST")
    temporal_namespace: str = Field(default="default", env="TEMPORAL_NAMESPACE")
    
    # API Keys for external services
    nvidia_api_key: Optional[str] = Field(default=None, env="NVIDIA_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, env="OPENROUTER_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # PTaaS settings
    ptaas_max_concurrent_scans: int = Field(default=10, env="PTAAS_MAX_CONCURRENT_SCANS")
    ptaas_scan_timeout_minutes: int = Field(default=60, env="PTAAS_SCAN_TIMEOUT_MINUTES")
    ptaas_result_retention_days: int = Field(default=90, env="PTAAS_RESULT_RETENTION_DAYS")
    
    # Security scanning tools
    nmap_path: str = Field(default="/usr/bin/nmap", env="NMAP_PATH")
    nuclei_path: str = Field(default="/usr/bin/nuclei", env="NUCLEI_PATH")
    nikto_path: str = Field(default="/usr/bin/nikto", env="NIKTO_PATH")
    
    # File upload settings
    max_file_upload_size_mb: int = Field(default=100, env="MAX_FILE_UPLOAD_SIZE_MB")
    allowed_file_extensions: str = Field(
        default=".pdf,.txt,.json,.xml,.csv",
        env="ALLOWED_FILE_EXTENSIONS"
    )
    
    # Feature flags
    enable_enterprise_features: bool = Field(default=True, env="ENABLE_ENTERPRISE_FEATURES")
    enable_ai_features: bool = Field(default=True, env="ENABLE_AI_FEATURES")
    enable_compliance_features: bool = Field(default=True, env="ENABLE_COMPLIANCE_FEATURES")
    enable_advanced_analytics: bool = Field(default=True, env="ENABLE_ADVANCED_ANALYTICS")
    
    # Development settings
    reload_on_change: bool = Field(default=False, env="RELOAD_ON_CHANGE")
    enable_debug_endpoints: bool = Field(default=False, env="ENABLE_DEBUG_ENDPOINTS")
    
    @validator("cors_allow_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list with robust error handling"""
        if isinstance(v, str):
            if not v.strip():  # Handle empty string
                return ["*"]
            if v.strip() == "*":
                return ["*"]
            origins = [origin.strip() for origin in v.split(",") if origin.strip()]
            # Validate origins format
            validated_origins = []
            for origin in origins:
                if origin == "*":
                    validated_origins.append(origin)
                elif origin.startswith(("http://", "https://")):
                    validated_origins.append(origin)
                elif origin:  # Local development
                    validated_origins.append(f"http://{origin}")
            return validated_origins if validated_origins else ["*"]
        elif isinstance(v, list):
            return v
        return ["*"]  # Default fallback
    
    @validator("cors_allow_methods", pre=True)
    def parse_cors_methods(cls, v):
        if isinstance(v, str):
            return [method.strip() for method in v.split(",")]
        return v
    
    @validator("cors_allow_headers", pre=True)
    def parse_cors_headers(cls, v):
        if isinstance(v, str):
            return [header.strip() for header in v.split(",")]
        return v
    

    
    @validator("environment")
    def validate_environment(cls, v):
        allowed_environments = ["development", "staging", "production", "test"]
        if v not in allowed_environments:
            raise ValueError(f"Environment must be one of: {allowed_environments}")
        return v
    
    @validator("jwt_secret_key")
    def validate_jwt_secret(cls, v, values):
        environment = values.get("environment", "development")
        
        # Require secure JWT secret in production
        if environment == "production":
            if not v or len(v) < 32:
                raise ValueError("JWT secret must be at least 32 characters in production")
            if v in ["change-me-in-production", "default", "secret", "key"]:
                raise ValueError("JWT secret cannot be a default value in production")
        
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of: {allowed_levels}")
        return v.upper()
    
    @validator("cache_backend")
    def validate_cache_backend(cls, v):
        allowed_backends = ["memory", "redis", "hybrid"]
        if v not in allowed_backends:
            raise ValueError(f"Cache backend must be one of: {allowed_backends}")
        return v
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


@dataclass
class SecurityConfig:
    """Security configuration derived from app settings"""
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60
    jwt_refresh_expiration_days: int = 30
    min_password_length: int = 12
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_numbers: bool = True
    require_special_chars: bool = True
    password_history_count: int = 5
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    rate_limit_window_seconds: int = 3600
    session_timeout_minutes: int = 480
    concurrent_sessions_limit: int = 3
    require_mfa: bool = True
    api_key_length: int = 32
    webhook_signature_tolerance_seconds: int = 300
    cors_max_age: int = 3600
    encryption_key_rotation_days: int = 90
    enable_field_level_encryption: bool = True
    max_file_upload_size_mb: int = 10
    allowed_file_extensions: List[str] = field(default_factory=list)
    enable_content_scanning: bool = True


@dataclass
class DatabaseConfig:
    """Database configuration derived from app settings"""
    database_url: str
    backend: str = "postgresql"
    min_pool_size: int = 5
    max_pool_size: int = 20
    pool_timeout: int = 30
    pool_recycle_seconds: int = 3600
    pool_pre_ping: bool = True
    command_timeout: int = 60
    statement_cache_size: int = 1024
    max_cached_statement_lifetime: int = 300
    enable_query_logging: bool = False
    slow_query_threshold_seconds: float = 1.0
    enable_query_metrics: bool = True
    enable_connection_metrics: bool = True
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 5
    enable_auto_migration: bool = False
    migration_timeout_seconds: int = 300


@dataclass
class CacheConfig:
    """Cache configuration derived from app settings"""
    backend: str = "hybrid"
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    redis_max_connections: int = 20
    redis_socket_timeout: int = 5
    redis_socket_connect_timeout: int = 5
    redis_retry_on_timeout: bool = True
    memory_max_size: int = 1000
    memory_ttl_seconds: int = 3600
    memory_strategy: str = "lru"
    default_ttl_seconds: int = 3600
    compression_threshold: int = 1024
    enable_metrics: bool = True
    key_prefix: str = "xorb:cache:"
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60


@dataclass
class MetricConfig:
    """Metrics configuration derived from app settings"""
    enable_prometheus: bool = True
    enable_custom_metrics: bool = True
    prometheus_port: int = 9090
    collection_interval: int = 60
    retention_days: int = 7
    enable_detailed_metrics: bool = True


class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self):
        self._app_settings: Optional[AppSettings] = None
        self._security_config: Optional[SecurityConfig] = None
        self._database_config: Optional[DatabaseConfig] = None
        self._cache_config: Optional[CacheConfig] = None
        self._metric_config: Optional[MetricConfig] = None
    
    @property
    def app_settings(self) -> AppSettings:
        """Get application settings"""
        if self._app_settings is None:
            self._app_settings = AppSettings()
            logger.info("Application settings loaded", environment=self._app_settings.environment)
        return self._app_settings
    
    @property
    def security_config(self) -> SecurityConfig:
        """Get security configuration"""
        if self._security_config is None:
            settings = self.app_settings
            self._security_config = SecurityConfig(
                jwt_secret_key=settings.jwt_secret_key,
                jwt_algorithm=settings.jwt_algorithm,
                jwt_expiration_minutes=settings.jwt_expiration_minutes,
                jwt_refresh_expiration_days=settings.jwt_refresh_expiration_days,
                min_password_length=settings.min_password_length,
                max_login_attempts=settings.max_login_attempts,
                lockout_duration_minutes=settings.lockout_duration_minutes,
                require_mfa=settings.require_mfa,
                max_file_upload_size_mb=settings.max_file_upload_size_mb,
                allowed_file_extensions=settings.get_allowed_file_extensions(),
                cors_max_age=settings.cors_max_age
            )
        return self._security_config
    
    @property
    def database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        if self._database_config is None:
            settings = self.app_settings
            self._database_config = DatabaseConfig(
                database_url=settings.database_url,
                min_pool_size=settings.database_min_pool_size,
                max_pool_size=settings.database_max_pool_size,
                pool_timeout=settings.database_pool_timeout,
                command_timeout=settings.database_command_timeout,
                enable_query_logging=settings.debug,
                enable_query_metrics=settings.enable_metrics,
                enable_connection_metrics=settings.enable_metrics
            )
        return self._database_config
    
    @property
    def cache_config(self) -> CacheConfig:
        """Get cache configuration"""
        if self._cache_config is None:
            settings = self.app_settings
            self._cache_config = CacheConfig(
                backend=settings.cache_backend,
                redis_url=settings.redis_url,
                redis_max_connections=settings.redis_max_connections,
                redis_socket_timeout=settings.redis_socket_timeout,
                memory_max_size=settings.cache_max_size,
                default_ttl_seconds=settings.cache_default_ttl,
                enable_metrics=settings.enable_metrics
            )
        return self._cache_config
    
    @property
    def metric_config(self) -> MetricConfig:
        """Get metrics configuration"""
        if self._metric_config is None:
            settings = self.app_settings
            self._metric_config = MetricConfig(
                enable_prometheus=settings.enable_metrics,
                enable_custom_metrics=settings.enable_metrics,
                prometheus_port=settings.prometheus_port,
                collection_interval=settings.metrics_collection_interval,
                enable_detailed_metrics=settings.debug
            )
        return self._metric_config
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get all feature flags"""
        settings = self.app_settings
        return {
            "enterprise_features": settings.enable_enterprise_features,
            "ai_features": settings.enable_ai_features,
            "compliance_features": settings.enable_compliance_features,
            "advanced_analytics": settings.enable_advanced_analytics,
            "debug_endpoints": settings.enable_debug_endpoints,
            "metrics": settings.enable_metrics,
            "rate_limiting": settings.rate_limit_enabled
        }
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.app_settings.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.app_settings.environment == "development"
    
    def is_testing(self) -> bool:
        """Check if running in test environment"""
        return self.app_settings.environment == "test"
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return any issues"""
        issues = []
        settings = self.app_settings
        
        # Security validation
        if self.is_production():
            if settings.jwt_secret_key == "change-me-in-production":
                issues.append("JWT secret key must be changed in production")
            
            if settings.debug:
                issues.append("Debug mode should be disabled in production")
            
            if settings.enable_debug_endpoints:
                issues.append("Debug endpoints should be disabled in production")
            
            if "*" in settings.cors_allow_origins:
                issues.append("CORS should not allow all origins in production")
        
        # Database validation
        if "localhost" in settings.database_url and self.is_production():
            issues.append("Database should not use localhost in production")
        
        # Required external services
        required_services = []
        if settings.enable_ai_features and not any([
            settings.nvidia_api_key,
            settings.openrouter_api_key,
            settings.openai_api_key,
            settings.anthropic_api_key
        ]):
            required_services.append("AI service API key")
        
        if required_services:
            issues.append(f"Missing required services: {', '.join(required_services)}")
        
        return issues
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get configuration summary for diagnostics"""
        settings = self.app_settings
        
        return {
            "app": {
                "name": settings.app_name,
                "version": settings.app_version,
                "environment": settings.environment,
                "debug": settings.debug
            },
            "api": {
                "host": settings.api_host,
                "port": settings.api_port,
                "prefix": settings.api_prefix,
                "workers": settings.api_workers
            },
            "database": {
                "url_masked": settings.database_url.split("@")[-1] if "@" in settings.database_url else "Not configured",
                "pool_size": f"{settings.database_min_pool_size}-{settings.database_max_pool_size}"
            },
            "cache": {
                "backend": settings.cache_backend,
                "max_size": settings.cache_max_size,
                "default_ttl": settings.cache_default_ttl
            },
            "features": self.get_feature_flags(),
            "security": {
                "mfa_required": settings.require_mfa,
                "min_password_length": settings.min_password_length,
                "max_login_attempts": settings.max_login_attempts
            }
        }


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


@lru_cache()
def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_settings() -> AppSettings:
    """Get application settings"""
    return get_config_manager().app_settings


def get_security_config() -> SecurityConfig:
    """Get security configuration"""
    return get_config_manager().security_config


def get_database_config() -> DatabaseConfig:
    """Get database configuration"""
    return get_config_manager().database_config


def get_cache_config() -> CacheConfig:
    """Get cache configuration"""
    return get_config_manager().cache_config


def get_metric_config() -> MetricConfig:
    """Get metrics configuration"""
    return get_config_manager().metric_config