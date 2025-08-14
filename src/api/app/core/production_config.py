"""
Production Configuration Management - Principal Auditor Enhanced
Advanced configuration for enterprise deployment with security hardening
"""

import os
import secrets
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import yaml
import json

logger = logging.getLogger(__name__)


class DeploymentEnvironment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ENTERPRISE = "enterprise"


class SecurityLevel(Enum):
    BASIC = "basic"
    ENHANCED = "enhanced"
    MAXIMUM = "maximum"
    QUANTUM_SAFE = "quantum_safe"


@dataclass
class DatabaseConfig:
    """Database configuration with connection pooling"""
    url: str
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    ssl_mode: str = "require"
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    ssl_ca: Optional[str] = None


@dataclass
class RedisConfig:
    """Redis configuration with clustering support"""
    url: str
    cluster_nodes: List[str] = field(default_factory=list)
    password: Optional[str] = None
    ssl: bool = True
    ssl_cert_reqs: str = "required"
    ssl_ca_certs: Optional[str] = None
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    health_check_interval: int = 30


@dataclass
class SecurityConfig:
    """Comprehensive security configuration"""
    secret_key: str
    jwt_algorithm: str = "RS256"
    jwt_access_token_expire_minutes: int = 15
    jwt_refresh_token_expire_days: int = 7
    password_hash_algorithm: str = "argon2"
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    cors_origins: List[str] = field(default_factory=list)
    trusted_hosts: List[str] = field(default_factory=list)
    security_headers: Dict[str, str] = field(default_factory=dict)
    tls_version: str = "1.3"
    cipher_suites: List[str] = field(default_factory=list)


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    grafana_enabled: bool = True
    grafana_port: int = 3000
    jaeger_enabled: bool = True
    jaeger_endpoint: str = "http://jaeger:14268/api/traces"
    log_level: str = "INFO"
    structured_logging: bool = True
    audit_logging: bool = True
    metrics_retention_days: int = 90


@dataclass
class PTaaSConfig:
    """PTaaS-specific configuration"""
    scanner_timeout: int = 1800  # 30 minutes
    max_concurrent_scans: int = 10
    scan_result_retention_days: int = 365
    compliance_frameworks: List[str] = field(default_factory=lambda: [
        "PCI-DSS", "HIPAA", "SOX", "ISO-27001", "GDPR", "NIST", "SOC2"
    ])
    scanner_tools: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    threat_intelligence_sources: List[str] = field(default_factory=list)
    ml_model_update_interval: int = 3600  # 1 hour


class ProductionConfigManager:
    """
    Production Configuration Manager - Principal Auditor Enhanced
    Manages enterprise-grade configuration with security best practices
    """
    
    def __init__(self, environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION):
        self.environment = environment
        self.config_path = Path(os.getenv("CONFIG_PATH", "/etc/xorb"))
        self.secrets_path = Path(os.getenv("SECRETS_PATH", "/run/secrets"))
        self._config_cache: Dict[str, Any] = {}
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from multiple sources with precedence"""
        try:
            # Load base configuration
            base_config = self._load_base_config()
            
            # Load environment-specific configuration
            env_config = self._load_environment_config()
            
            # Load secrets
            secrets_config = self._load_secrets()
            
            # Merge configurations with precedence: secrets > env > base
            self._config_cache = {**base_config, **env_config, **secrets_config}
            
            # Validate configuration
            self._validate_configuration()
            
            logger.info("Production configuration loaded successfully", 
                       environment=self.environment.value)
            
        except Exception as e:
            logger.error(f"Failed to load production configuration: {e}")
            raise
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration"""
        return {
            "app_name": "XORB Enterprise Cybersecurity Platform",
            "app_version": "2025.1.0",
            "api_host": "0.0.0.0",
            "api_port": 8000,
            "api_workers": 4,
            "debug": False,
            "testing": False,
            "log_level": "INFO",
            "environment": self.environment.value
        }
    
    def _load_environment_config(self) -> Dict[str, Any]:
        """Load environment-specific configuration"""
        config = {}
        
        # Load from environment variables
        env_mappings = {
            "DATABASE_URL": "database_url",
            "REDIS_URL": "redis_url",
            "SECRET_KEY": "secret_key",
            "JWT_SECRET": "jwt_secret",
            "API_HOST": "api_host",
            "API_PORT": "api_port",
            "API_WORKERS": "api_workers",
            "LOG_LEVEL": "log_level",
            "CORS_ORIGINS": "cors_origins",
            "TRUSTED_HOSTS": "trusted_hosts",
            "PROMETHEUS_ENABLED": "prometheus_enabled",
            "GRAFANA_ENABLED": "grafana_enabled"
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert types appropriately
                if config_key in ["api_port", "api_workers"]:
                    config[config_key] = int(value)
                elif config_key in ["prometheus_enabled", "grafana_enabled"]:
                    config[config_key] = value.lower() in ("true", "1", "yes")
                elif config_key in ["cors_origins", "trusted_hosts"]:
                    config[config_key] = [origin.strip() for origin in value.split(",")]
                else:
                    config[config_key] = value
        
        # Load from configuration file if exists
        config_file = self.config_path / f"{self.environment.value}.yaml"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    config.update(file_config)
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")
        
        return config
    
    def _load_secrets(self) -> Dict[str, Any]:
        """Load secrets from secure storage"""
        secrets = {}
        
        # Load from secrets files
        secret_files = {
            "secret_key": "SECRET_KEY",
            "jwt_secret": "JWT_SECRET",
            "database_password": "DATABASE_PASSWORD",
            "redis_password": "REDIS_PASSWORD",
            "nvidia_api_key": "NVIDIA_API_KEY",
            "openai_api_key": "OPENAI_API_KEY"
        }
        
        for config_key, secret_name in secret_files.items():
            secret_file = self.secrets_path / secret_name
            if secret_file.exists():
                try:
                    with open(secret_file, 'r') as f:
                        secrets[config_key] = f.read().strip()
                except Exception as e:
                    logger.warning(f"Failed to load secret {secret_name}: {e}")
        
        # Generate missing secrets
        if "secret_key" not in secrets:
            secrets["secret_key"] = secrets.token_urlsafe(64)
            logger.warning("Generated random secret key - store securely for production")
        
        return secrets
    
    def _validate_configuration(self):
        """Validate production configuration"""
        required_production_configs = [
            "secret_key", "database_url", "redis_url"
        ]
        
        missing_configs = []
        for config in required_production_configs:
            if config not in self._config_cache:
                missing_configs.append(config)
        
        if missing_configs and self.environment == DeploymentEnvironment.PRODUCTION:
            raise ValueError(f"Missing required production configs: {missing_configs}")
        
        # Validate security settings
        if self.environment == DeploymentEnvironment.PRODUCTION:
            self._validate_security_settings()
    
    def _validate_security_settings(self):
        """Validate security configuration for production"""
        security_checks = []
        
        # Check secret key strength
        secret_key = self._config_cache.get("secret_key", "")
        if len(secret_key) < 32:
            security_checks.append("Secret key too short (minimum 32 characters)")
        
        # Check CORS origins
        cors_origins = self._config_cache.get("cors_origins", [])
        if "*" in cors_origins:
            security_checks.append("Wildcard CORS origins not allowed in production")
        
        # Check debug mode
        if self._config_cache.get("debug", False):
            security_checks.append("Debug mode must be disabled in production")
        
        if security_checks:
            raise ValueError(f"Security validation failed: {security_checks}")
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        return DatabaseConfig(
            url=self._config_cache.get("database_url", 
                "postgresql://xorb:xorb@localhost:5432/xorb_prod"),
            pool_size=self._config_cache.get("db_pool_size", 20),
            max_overflow=self._config_cache.get("db_max_overflow", 30),
            ssl_mode=self._config_cache.get("db_ssl_mode", "require"),
            ssl_cert=self._config_cache.get("db_ssl_cert"),
            ssl_key=self._config_cache.get("db_ssl_key"),
            ssl_ca=self._config_cache.get("db_ssl_ca")
        )
    
    def get_redis_config(self) -> RedisConfig:
        """Get Redis configuration"""
        return RedisConfig(
            url=self._config_cache.get("redis_url", "redis://localhost:6379/0"),
            cluster_nodes=self._config_cache.get("redis_cluster_nodes", []),
            password=self._config_cache.get("redis_password"),
            ssl=self._config_cache.get("redis_ssl", True),
            ssl_cert_reqs=self._config_cache.get("redis_ssl_cert_reqs", "required"),
            ssl_ca_certs=self._config_cache.get("redis_ssl_ca_certs")
        )
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration"""
        return SecurityConfig(
            secret_key=self._config_cache.get("secret_key"),
            jwt_algorithm=self._config_cache.get("jwt_algorithm", "RS256"),
            jwt_access_token_expire_minutes=self._config_cache.get("jwt_access_token_expire_minutes", 15),
            rate_limit_per_minute=self._config_cache.get("rate_limit_per_minute", 60),
            rate_limit_per_hour=self._config_cache.get("rate_limit_per_hour", 1000),
            cors_origins=self._config_cache.get("cors_origins", []),
            trusted_hosts=self._config_cache.get("trusted_hosts", []),
            tls_version=self._config_cache.get("tls_version", "1.3")
        )
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        return MonitoringConfig(
            prometheus_enabled=self._config_cache.get("prometheus_enabled", True),
            prometheus_port=self._config_cache.get("prometheus_port", 9090),
            grafana_enabled=self._config_cache.get("grafana_enabled", True),
            grafana_port=self._config_cache.get("grafana_port", 3000),
            log_level=self._config_cache.get("log_level", "INFO"),
            structured_logging=self._config_cache.get("structured_logging", True),
            audit_logging=self._config_cache.get("audit_logging", True)
        )
    
    def get_ptaas_config(self) -> PTaaSConfig:
        """Get PTaaS configuration"""
        scanner_tools = {
            "nmap": {
                "executable": "/usr/bin/nmap",
                "timeout": 1800,
                "max_rate": 1000,
                "options": ["-sS", "-sV", "-O", "--script=default"]
            },
            "nuclei": {
                "executable": "/usr/bin/nuclei",
                "timeout": 3600,
                "templates_path": "/opt/nuclei-templates",
                "options": ["-c", "50", "-rl", "150"]
            },
            "nikto": {
                "executable": "/usr/bin/nikto",
                "timeout": 1800,
                "options": ["-C", "all", "-Format", "json"]
            },
            "sslscan": {
                "executable": "/usr/bin/sslscan",
                "timeout": 300,
                "options": ["--xml"]
            }
        }
        
        return PTaaSConfig(
            scanner_timeout=self._config_cache.get("scanner_timeout", 1800),
            max_concurrent_scans=self._config_cache.get("max_concurrent_scans", 10),
            scanner_tools=scanner_tools,
            threat_intelligence_sources=self._config_cache.get("threat_intel_sources", [
                "mitre_attack", "cve_database", "threat_feeds"
            ])
        )
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging"""
        return {
            "environment": self.environment.value,
            "app_version": self._config_cache.get("app_version"),
            "api_host": self._config_cache.get("api_host"),
            "api_port": self._config_cache.get("api_port"),
            "log_level": self._config_cache.get("log_level"),
            "prometheus_enabled": self._config_cache.get("prometheus_enabled"),
            "security_level": "enterprise",
            "ptaas_enabled": True,
            "compliance_frameworks": len(self.get_ptaas_config().compliance_frameworks)
        }
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment in [DeploymentEnvironment.PRODUCTION, DeploymentEnvironment.ENTERPRISE]
    
    def export_config(self, output_path: Path, include_secrets: bool = False) -> None:
        """Export configuration to file"""
        config_export = self._config_cache.copy()
        
        if not include_secrets:
            # Remove sensitive information
            sensitive_keys = ["secret_key", "jwt_secret", "database_password", "redis_password"]
            for key in sensitive_keys:
                if key in config_export:
                    config_export[key] = "[REDACTED]"
        
        with open(output_path, 'w') as f:
            yaml.dump(config_export, f, default_flow_style=False)
        
        logger.info(f"Configuration exported to {output_path}")


# Global production config manager instance
_production_config_manager = None


def get_production_config_manager(
    environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION
) -> ProductionConfigManager:
    """Get the global production configuration manager"""
    global _production_config_manager
    if _production_config_manager is None:
        _production_config_manager = ProductionConfigManager(environment)
    return _production_config_manager


def initialize_production_config(environment: str = "production") -> ProductionConfigManager:
    """Initialize production configuration"""
    env = DeploymentEnvironment(environment)
    config_manager = ProductionConfigManager(env)
    
    logger.info("Production configuration initialized",
               environment=env.value,
               config_summary=config_manager.get_config_summary())
    
    return config_manager