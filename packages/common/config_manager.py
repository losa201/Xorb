"""
Centralized Configuration Management System for XORB Platform
Enterprise-grade configuration with environment-specific settings, secret management, and hot-reloading
"""

import json
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import time

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None

from .vault_client import VaultClient


class Environment(Enum):
    """Supported environments"""
    DEVELOPMENT = "development"
    STAGING = "staging" 
    PRODUCTION = "production"
    TEST = "test"


class ConfigFormat(Enum):
    """Supported configuration formats"""
    JSON = "json"
    YAML = "yaml"
    ENV = "env"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    name: str = "xorb"
    username: str = "xorb_user"
    password: str = ""  # Will be loaded from secrets
    ssl_mode: str = "prefer"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    
    def get_url(self) -> str:
        """Get database connection URL"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.name}?sslmode={self.ssl_mode}"


@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""  # Will be loaded from secrets
    ssl: bool = False
    
    def get_url(self) -> str:
        """Get Redis connection URL"""
        protocol = "rediss" if self.ssl else "redis"
        auth = f":{self.password}@" if self.password else ""
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"


@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret: str = ""  # Will be loaded from secrets
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = 24
    password_min_length: int = 8
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    encryption_key: str = ""  # Will be loaded from secrets
    api_key_length: int = 32


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    enabled: bool = True
    prometheus_port: int = 9090
    grafana_port: int = 3000
    log_level: str = "INFO"
    tracing_enabled: bool = True
    metrics_retention_days: int = 30
    alert_webhook_url: str = ""


@dataclass
class ServiceConfig:
    """Service-specific configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 30
    max_connections: int = 1000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class XORBConfig:
    """Main XORB platform configuration"""
    environment: Environment = Environment.DEVELOPMENT
    app_name: str = "XORB Platform"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Service configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    api_service: ServiceConfig = field(default_factory=lambda: ServiceConfig(port=8000))
    orchestrator_service: ServiceConfig = field(default_factory=lambda: ServiceConfig(port=8001))
    intelligence_service: ServiceConfig = field(default_factory=lambda: ServiceConfig(port=8002))
    
    # Feature flags
    feature_flags: Dict[str, bool] = field(default_factory=lambda: {
        "advanced_analytics": True,
        "threat_hunting": True,
        "compliance_automation": True,
        "multi_tenant": True,
        "sso_integration": False
    })
    
    # External integrations
    integrations: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "vault": {
            "enabled": True,
            "url": "http://localhost:8200",
            "token_path": "/var/run/secrets/vault-token"
        },
        "temporal": {
            "host": "localhost:7233",
            "namespace": "xorb"
        }
    })


class ConfigFileWatcher:
    """File system watcher for configuration changes"""
    
    def __init__(self, config_manager):
        if WATCHDOG_AVAILABLE and FileSystemEventHandler:
            super().__init__()
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
    
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory and event.src_path in self.config_manager.watched_files:
            self.logger.info(f"Configuration file changed: {event.src_path}")
            self.config_manager.reload_config()

# Add FileSystemEventHandler inheritance only if available
if WATCHDOG_AVAILABLE and FileSystemEventHandler:
    ConfigFileWatcher.__bases__ = (FileSystemEventHandler,)


class ConfigManager:
    """
    Centralized configuration manager for XORB platform
    
    Features:
    - Environment-specific configurations
    - Secret management integration
    - Hot-reloading
    - Configuration validation
    - Feature flags
    """
    
    def __init__(self, 
                 config_dir: str = "/root/Xorb/config",
                 environment: Optional[str] = None,
                 enable_hot_reload: bool = True):
        self.config_dir = Path(config_dir)
        self.environment = Environment(environment or os.getenv("XORB_ENV", "development"))
        self.enable_hot_reload = enable_hot_reload
        self.logger = logging.getLogger(__name__)
        
        # Configuration state
        self.config: Optional[XORBConfig] = None
        self.watched_files: List[str] = []
        self.observers: List[Observer] = []
        self.reload_callbacks: List[callable] = []
        
        # Thread safety
        self.config_lock = threading.RLock()
        
        # Vault integration
        self.vault_client = None
        if os.getenv("VAULT_ENABLED", "true").lower() == "true":
            try:
                self.vault_client = VaultClient()
            except Exception as e:
                self.logger.warning(f"Failed to initialize Vault client: {e}")
        
        # Load initial configuration
        self.load_config()
        
        # Start file watchers if hot-reload is enabled
        if self.enable_hot_reload:
            self.start_file_watchers()
    
    def load_config(self) -> XORBConfig:
        """Load configuration from files and environment"""
        with self.config_lock:
            try:
                # Start with default configuration
                config = XORBConfig()
                config.environment = self.environment
                
                # Load base configuration
                base_config_path = self.config_dir / "default.json"
                if base_config_path.exists():
                    base_config = self._load_config_file(base_config_path)
                    config = self._merge_config(config, base_config)
                    self.watched_files.append(str(base_config_path))
                
                # Load environment-specific configuration
                env_config_path = self.config_dir / f"{self.environment.value}.json"
                if env_config_path.exists():
                    env_config = self._load_config_file(env_config_path)
                    config = self._merge_config(config, env_config)
                    self.watched_files.append(str(env_config_path))
                
                # Load secrets from Vault or environment
                config = self._load_secrets(config)
                
                # Apply environment variable overrides
                config = self._apply_env_overrides(config)
                
                # Validate configuration
                self._validate_config(config)
                
                self.config = config
                self.logger.info(f"Configuration loaded for environment: {self.environment.value}")
                
                # Trigger reload callbacks
                for callback in self.reload_callbacks:
                    try:
                        callback(config)
                    except Exception as e:
                        self.logger.error(f"Error in config reload callback: {e}")
                
                return config
                
            except Exception as e:
                self.logger.error(f"Failed to load configuration: {e}")
                if self.config is None:
                    # Return minimal working config if first load fails
                    self.config = XORBConfig()
                return self.config
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from a file"""
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config file {config_path}: {e}")
            return {}
    
    def _merge_config(self, base_config: XORBConfig, override_config: Dict[str, Any]) -> XORBConfig:
        """Merge configuration dictionaries into dataclass"""
        # Convert base config to dict
        config_dict = asdict(base_config)
        
        # Deep merge override config
        self._deep_merge_dict(config_dict, override_config)
        
        # Convert back to dataclass
        try:
            # Handle nested configs manually due to dataclass limitations
            if 'database' in config_dict:
                config_dict['database'] = DatabaseConfig(**config_dict['database'])
            if 'redis' in config_dict:
                config_dict['redis'] = RedisConfig(**config_dict['redis'])
            if 'security' in config_dict:
                config_dict['security'] = SecurityConfig(**config_dict['security'])
            if 'monitoring' in config_dict:
                config_dict['monitoring'] = MonitoringConfig(**config_dict['monitoring'])
            if 'api_service' in config_dict:
                config_dict['api_service'] = ServiceConfig(**config_dict['api_service'])
            if 'orchestrator_service' in config_dict:
                config_dict['orchestrator_service'] = ServiceConfig(**config_dict['orchestrator_service'])
            if 'intelligence_service' in config_dict:
                config_dict['intelligence_service'] = ServiceConfig(**config_dict['intelligence_service'])
            
            # Handle environment enum
            if 'environment' in config_dict and isinstance(config_dict['environment'], str):
                config_dict['environment'] = Environment(config_dict['environment'])
            
            return XORBConfig(**config_dict)
        except Exception as e:
            self.logger.error(f"Failed to merge config: {e}")
            return base_config
    
    def _deep_merge_dict(self, base_dict: Dict[str, Any], override_dict: Dict[str, Any]):
        """Deep merge two dictionaries"""
        for key, value in override_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge_dict(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _load_secrets(self, config: XORBConfig) -> XORBConfig:
        """Load secrets from Vault or environment variables"""
        try:
            if self.vault_client and self.vault_client.is_authenticated():
                # Load from Vault
                secrets = self.vault_client.get_secret("xorb/config")
                if secrets:
                    config.database.password = secrets.get("database_password", config.database.password)
                    config.redis.password = secrets.get("redis_password", config.redis.password)
                    config.security.jwt_secret = secrets.get("jwt_secret", config.security.jwt_secret)
                    config.security.encryption_key = secrets.get("encryption_key", config.security.encryption_key)
            else:
                # Fallback to environment variables
                config.database.password = os.getenv("DATABASE_PASSWORD", config.database.password)
                config.redis.password = os.getenv("REDIS_PASSWORD", config.redis.password)
                config.security.jwt_secret = os.getenv("JWT_SECRET", config.security.jwt_secret)
                config.security.encryption_key = os.getenv("ENCRYPTION_KEY", config.security.encryption_key)
                
        except Exception as e:
            self.logger.error(f"Failed to load secrets: {e}")
        
        return config
    
    def _apply_env_overrides(self, config: XORBConfig) -> XORBConfig:
        """Apply environment variable overrides"""
        # Database overrides
        config.database.host = os.getenv("DATABASE_HOST", config.database.host)
        config.database.port = int(os.getenv("DATABASE_PORT", config.database.port))
        config.database.name = os.getenv("DATABASE_NAME", config.database.name)
        
        # Redis overrides  
        config.redis.host = os.getenv("REDIS_HOST", config.redis.host)
        config.redis.port = int(os.getenv("REDIS_PORT", config.redis.port))
        
        # Service overrides
        config.api_service.host = os.getenv("API_HOST", config.api_service.host)
        config.api_service.port = int(os.getenv("API_PORT", config.api_service.port))
        
        # Debug and logging
        config.debug = os.getenv("DEBUG", str(config.debug)).lower() == "true"
        config.monitoring.log_level = os.getenv("LOG_LEVEL", config.monitoring.log_level)
        
        return config
    
    def _validate_config(self, config: XORBConfig):
        """Validate configuration"""
        errors = []
        
        # Production checks
        if config.environment == Environment.PRODUCTION:
            if not config.security.jwt_secret:
                errors.append("JWT secret must be set in production")
            if not config.security.encryption_key:
                errors.append("Encryption key must be set in production")
            if config.debug:
                errors.append("Debug mode should be disabled in production")
        
        # Database checks
        if not config.database.host:
            errors.append("Database host must be specified")
        
        # Port conflicts
        ports = [
            config.api_service.port,
            config.orchestrator_service.port,
            config.intelligence_service.port,
            config.monitoring.prometheus_port,
            config.monitoring.grafana_port
        ]
        if len(ports) != len(set(ports)):
            errors.append("Service ports must be unique")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            if config.environment == Environment.PRODUCTION:
                raise ValueError(error_msg)
            else:
                self.logger.warning(error_msg)
    
    def start_file_watchers(self):
        """Start file system watchers for hot-reloading"""
        if not self.enable_hot_reload or not WATCHDOG_AVAILABLE:
            if not WATCHDOG_AVAILABLE:
                self.logger.warning("Watchdog library not available - hot-reload disabled")
            return
            
        try:
            event_handler = ConfigFileWatcher(self)
            observer = Observer()
            observer.schedule(event_handler, str(self.config_dir), recursive=False)
            observer.start()
            self.observers.append(observer)
            self.logger.info("Configuration file watcher started")
        except Exception as e:
            self.logger.error(f"Failed to start file watcher: {e}")
    
    def stop_file_watchers(self):
        """Stop file system watchers"""
        for observer in self.observers:
            try:
                observer.stop()
                observer.join()
            except Exception as e:
                self.logger.error(f"Error stopping observer: {e}")
        self.observers.clear()
    
    def reload_config(self):
        """Reload configuration from files"""
        self.logger.info("Reloading configuration...")
        try:
            self.load_config()
            self.logger.info("Configuration reloaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {e}")
    
    def get_config(self) -> XORBConfig:
        """Get current configuration"""
        with self.config_lock:
            if self.config is None:
                self.load_config()
            return self.config
    
    def register_reload_callback(self, callback: callable):
        """Register callback for configuration reloads"""
        self.reload_callbacks.append(callback)
    
    def get_feature_flag(self, flag_name: str, default: bool = False) -> bool:
        """Get feature flag value"""
        config = self.get_config()
        return config.feature_flags.get(flag_name, default)
    
    def set_feature_flag(self, flag_name: str, enabled: bool):
        """Set feature flag value"""
        with self.config_lock:
            config = self.get_config()
            config.feature_flags[flag_name] = enabled
            self.logger.info(f"Feature flag '{flag_name}' set to {enabled}")
    
    def get_service_config(self, service_name: str) -> Optional[ServiceConfig]:
        """Get configuration for a specific service"""
        config = self.get_config()
        service_configs = {
            "api": config.api_service,
            "orchestrator": config.orchestrator_service,
            "intelligence": config.intelligence_service
        }
        return service_configs.get(service_name)
    
    def export_config(self, format: ConfigFormat = ConfigFormat.JSON, include_secrets: bool = False) -> str:
        """Export configuration to string"""
        config = self.get_config()
        config_dict = asdict(config)
        
        # Remove secrets if not including them
        if not include_secrets:
            if 'database' in config_dict and 'password' in config_dict['database']:
                config_dict['database']['password'] = "[REDACTED]"
            if 'redis' in config_dict and 'password' in config_dict['redis']:
                config_dict['redis']['password'] = "[REDACTED]"
            if 'security' in config_dict:
                config_dict['security']['jwt_secret'] = "[REDACTED]"
                config_dict['security']['encryption_key'] = "[REDACTED]"
        
        # Convert environment enum to string
        if 'environment' in config_dict:
            config_dict['environment'] = config_dict['environment'].value if hasattr(config_dict['environment'], 'value') else str(config_dict['environment'])
        
        if format == ConfigFormat.JSON:
            return json.dumps(config_dict, indent=2)
        elif format == ConfigFormat.YAML:
            return yaml.dump(config_dict, default_flow_style=False)
        else:
            # Environment variable format
            env_vars = []
            self._flatten_dict_to_env(config_dict, env_vars)
            return "\n".join(env_vars)
    
    def _flatten_dict_to_env(self, d: Dict[str, Any], env_vars: List[str], prefix: str = "XORB"):
        """Flatten dictionary to environment variables"""
        for key, value in d.items():
            env_key = f"{prefix}_{key.upper()}"
            if isinstance(value, dict):
                self._flatten_dict_to_env(value, env_vars, env_key)
            else:
                env_vars.append(f"{env_key}={value}")
    
    def __del__(self):
        """Cleanup when manager is destroyed"""
        self.stop_file_watchers()


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> XORBConfig:
    """Get current configuration"""
    return get_config_manager().get_config()


def reload_config():
    """Reload global configuration"""
    get_config_manager().reload_config()


def get_feature_flag(flag_name: str, default: bool = False) -> bool:
    """Get feature flag value"""
    return get_config_manager().get_feature_flag(flag_name, default)