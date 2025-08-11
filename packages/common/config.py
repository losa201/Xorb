"""
Configuration Management
Application settings and environment configuration with secure secret handling
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from functools import lru_cache

from .secret_manager import get_secret, get_database_url, get_jwt_secret


@dataclass
class Settings:
    """Application settings"""
    
    # Database configuration - using secure secret manager
    database_url: str = field(default_factory=get_database_url)
    redis_url: str = field(default_factory=lambda: get_secret("REDIS_URL", "redis://localhost:6379"))
    
    # Authentication - using secure secret manager
    jwt_secret: str = field(default_factory=get_jwt_secret)
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = 24
    
    # SSO Configuration - Azure AD (using secure secret manager)
    azure_sso_enabled: bool = field(default_factory=lambda: get_secret("AZURE_SSO_ENABLED", "false").lower() == "true")
    azure_client_id: str = field(default_factory=lambda: get_secret("AZURE_CLIENT_ID", ""))
    azure_client_secret: str = field(default_factory=lambda: get_secret("AZURE_CLIENT_SECRET", ""))
    azure_tenant: str = field(default_factory=lambda: get_secret("AZURE_TENANT", "common"))
    
    # SSO Configuration - Google (using secure secret manager)
    google_sso_enabled: bool = field(default_factory=lambda: get_secret("GOOGLE_SSO_ENABLED", "false").lower() == "true")
    google_client_id: str = field(default_factory=lambda: get_secret("GOOGLE_CLIENT_ID", ""))
    google_client_secret: str = field(default_factory=lambda: get_secret("GOOGLE_CLIENT_SECRET", ""))
    
    # SSO Configuration - Okta (using secure secret manager)
    okta_sso_enabled: bool = field(default_factory=lambda: get_secret("OKTA_SSO_ENABLED", "false").lower() == "true")
    okta_client_id: str = field(default_factory=lambda: get_secret("OKTA_CLIENT_ID", ""))
    okta_client_secret: str = field(default_factory=lambda: get_secret("OKTA_CLIENT_SECRET", ""))
    okta_domain: str = field(default_factory=lambda: get_secret("OKTA_DOMAIN", ""))
    
    # SSO Configuration - GitHub (using secure secret manager)
    github_sso_enabled: bool = field(default_factory=lambda: get_secret("GITHUB_SSO_ENABLED", "false").lower() == "true")
    github_client_id: str = field(default_factory=lambda: get_secret("GITHUB_CLIENT_ID", ""))
    github_client_secret: str = field(default_factory=lambda: get_secret("GITHUB_CLIENT_SECRET", ""))
    
    # Application settings
    app_name: str = "XORB PTaaS"
    app_version: str = "1.0.0"
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    
    # API settings
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    api_cors_origins: list = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*").split(","))
    
    # Security settings
    password_min_length: int = 8
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    
    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # External services
    temporal_host: str = field(default_factory=lambda: os.getenv("TEMPORAL_HOST", "localhost:7233"))
    
    # Service URLs
    scanner_service_url: str = field(default_factory=lambda: os.getenv("SCANNER_SERVICE_URL", "http://localhost:8001"))
    compliance_service_url: str = field(default_factory=lambda: os.getenv("COMPLIANCE_SERVICE_URL", "http://localhost:8002"))
    notification_service_url: str = field(default_factory=lambda: os.getenv("NOTIFICATION_SERVICE_URL", "http://localhost:8003"))
    
    # File storage
    upload_dir: str = field(default_factory=lambda: os.getenv("UPLOAD_DIR", "./uploads"))
    max_file_size_mb: int = 100
    
    # Email settings (for notifications)
    smtp_host: str = field(default_factory=lambda: os.getenv("SMTP_HOST", "localhost"))
    smtp_port: int = field(default_factory=lambda: int(os.getenv("SMTP_PORT", "587")))
    smtp_username: str = field(default_factory=lambda: os.getenv("SMTP_USERNAME", ""))
    smtp_password: str = field(default_factory=lambda: os.getenv("SMTP_PASSWORD", ""))
    smtp_use_tls: bool = field(default_factory=lambda: os.getenv("SMTP_USE_TLS", "true").lower() == "true")
    
    # Monitoring and metrics
    metrics_enabled: bool = field(default_factory=lambda: os.getenv("METRICS_ENABLED", "true").lower() == "true")
    prometheus_port: int = field(default_factory=lambda: int(os.getenv("PROMETHEUS_PORT", "9090")))
    
    def get_sso_config(self) -> Dict[str, Dict[str, Any]]:
        """Get complete SSO configuration"""
        return {
            "azure": {
                "enabled": self.azure_sso_enabled,
                "client_id": self.azure_client_id,
                "client_secret": self.azure_client_secret,
                "tenant": self.azure_tenant
            },
            "google": {
                "enabled": self.google_sso_enabled,
                "client_id": self.google_client_id,
                "client_secret": self.google_client_secret
            },
            "okta": {
                "enabled": self.okta_sso_enabled,
                "client_id": self.okta_client_id,
                "client_secret": self.okta_client_secret,
                "domain": self.okta_domain
            },
            "github": {
                "enabled": self.github_sso_enabled,
                "client_id": self.github_client_id,
                "client_secret": self.github_client_secret
            }
        }
    
    def get_service_urls(self) -> Dict[str, str]:
        """Get service URLs configuration"""
        return {
            "scanner": self.scanner_service_url,
            "compliance": self.compliance_service_url,
            "notifications": self.notification_service_url
        }
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment.lower() == "development"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


def load_settings_from_env() -> Settings:
    """Load settings from environment variables"""
    return Settings()


def validate_settings(settings: Settings) -> bool:
    """Validate settings configuration"""
    errors = []
    
    # Check required settings
    if not settings.jwt_secret or settings.jwt_secret == "your-secret-key-change-in-production":
        if settings.is_production():
            errors.append("JWT_SECRET must be set in production")
    
    # Validate SSO configurations
    sso_providers = []
    if settings.azure_sso_enabled:
        if not settings.azure_client_id or not settings.azure_client_secret:
            errors.append("Azure SSO enabled but missing client credentials")
        else:
            sso_providers.append("Azure AD")
    
    if settings.google_sso_enabled:
        if not settings.google_client_id or not settings.google_client_secret:
            errors.append("Google SSO enabled but missing client credentials")
        else:
            sso_providers.append("Google")
    
    if settings.okta_sso_enabled:
        if not settings.okta_client_id or not settings.okta_client_secret or not settings.okta_domain:
            errors.append("Okta SSO enabled but missing client credentials or domain")
        else:
            sso_providers.append("Okta")
    
    if settings.github_sso_enabled:
        if not settings.github_client_id or not settings.github_client_secret:
            errors.append("GitHub SSO enabled but missing client credentials")
        else:
            sso_providers.append("GitHub")
    
    # Report configuration
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    if sso_providers:
        print(f"SSO providers configured: {', '.join(sso_providers)}")
    else:
        print("No SSO providers configured - using local authentication only")
    
    return True


# Environment-specific configurations
def get_development_settings() -> Settings:
    """Get development-specific settings"""
    settings = Settings()
    settings.debug = True
    settings.log_level = "DEBUG"
    return settings


def get_production_settings() -> Settings:
    """Get production-specific settings"""
    settings = Settings()
    settings.debug = False
    settings.log_level = "INFO"
    return settings


def get_test_settings() -> Settings:
    """Get test-specific settings"""
    settings = Settings()
    settings.database_url = "sqlite:///test.db"
    settings.redis_url = "redis://localhost:6379/1"  # Different Redis DB for tests
    settings.debug = True
    settings.log_level = "DEBUG"
    return settings