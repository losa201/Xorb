"""
Production Configuration Management
Sophisticated configuration system with environment-specific settings, security, and validation
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import secrets

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


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
    url: str
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    ssl_mode: str = "prefer"
    connection_timeout: int = 30


@dataclass
class RedisConfig:
    """Redis configuration"""
    url: str
    password: Optional[str] = None
    database: int = 0
    max_connections: int = 100
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    decode_responses: bool = True


@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 1
    refresh_token_expiration_days: int = 7
    password_min_length: int = 12
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    require_mfa: bool = True
    allowed_origins: List[str] = field(default_factory=list)
    api_key_length: int = 32
    session_timeout_hours: int = 8


@dataclass
class ThreatHuntingConfig:
    """Threat hunting configuration"""
    max_concurrent_queries: int = 10
    query_timeout_seconds: int = 300
    max_hunting_hits: int = 10000
    ml_model_update_interval: int = 3600
    threat_intel_update_interval: int = 1800
    data_retention_days: int = 90
    enable_real_time_hunting: bool = True
    enable_ml_analysis: bool = True


@dataclass
class MitreConfig:
    """MITRE ATT&CK configuration"""
    framework_update_interval: int = 86400  # 24 hours
    confidence_threshold: float = 0.6
    max_techniques_per_analysis: int = 50
    enable_attribution: bool = True
    enable_technique_prediction: bool = True
    cache_framework_data: bool = True


@dataclass
class PTaaSConfig:
    """PTaaS configuration"""
    max_concurrent_scans: int = 15
    scan_timeout_minutes: int = 120
    max_targets_per_scan: int = 50
    enable_stealth_scanning: bool = True
    enable_compliance_scanning: bool = True
    scan_result_retention_days: int = 365
    enable_auto_orchestration: bool = True
    scan_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    enable_health_checks: bool = True
    health_check_interval: int = 30
    enable_performance_monitoring: bool = True
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"
    enable_distributed_tracing: bool = True


@dataclass
class ProductionConfig:
    """Complete production configuration"""
    environment: Environment
    debug: bool = False
    
    # Core services
    database: DatabaseConfig
    redis: RedisConfig
    security: SecurityConfig
    
    # Advanced services
    threat_hunting: ThreatHuntingConfig
    mitre: MitreConfig
    ptaas: PTaaSConfig
    monitoring: MonitoringConfig
    
    # Infrastructure
    worker_processes: int = 4
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: int = 300
    enable_gzip: bool = True
    
    # Feature flags
    enable_advanced_features: bool = True
    enable_ai_enhancement: bool = True
    enable_quantum_security: bool = False  # Future feature
    
    # External integrations
    external_apis: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration settings"""
        # Validate database URL
        if not self.database.url:
            raise ValueError("Database URL is required")
        
        # Validate Redis URL
        if not self.redis.url:
            raise ValueError("Redis URL is required")
        
        # Validate JWT secret
        if len(self.security.jwt_secret) < 32:
            raise ValueError("JWT secret must be at least 32 characters")
        
        # Validate scan limits
        if self.ptaas.max_concurrent_scans > 50:
            logger.warning("High concurrent scan limit may impact performance")
        
        # Environment-specific validations
        if self.environment == Environment.PRODUCTION:
            self._validate_production_config()
    
    def _validate_production_config(self):
        """Additional validation for production environment"""
        if self.debug:
            raise ValueError("Debug mode must be disabled in production")
        
        if not self.security.require_mfa:
            logger.warning("MFA is disabled in production - security risk")
        
        if self.monitoring.log_level == LogLevel.DEBUG:
            logger.warning("Debug logging enabled in production - may impact performance")


class ConfigurationManager:
    """Advanced configuration management with environment detection and validation"""
    
    def __init__(self):
        self._config: Optional[ProductionConfig] = None
        self._config_file_paths = [
            Path("config/production.json"),
            Path("config/production.yaml"),
            Path("/etc/xorb/config.json"),
            Path.home() / ".xorb" / "config.json"
        ]
    
    def load_configuration(self, config_file: Optional[str] = None) -> ProductionConfig:
        """Load configuration from environment variables and files"""
        try:
            # Detect environment
            environment = Environment(os.getenv("ENVIRONMENT", "development"))
            logger.info(f"Loading configuration for environment: {environment.value}")
            
            # Load base configuration from environment
            config_data = self._load_from_environment()
            
            # Override with file configuration if available
            if config_file:
                file_config = self._load_from_file(config_file)
                config_data.update(file_config)
            else:
                # Try default config file locations
                for config_path in self._config_file_paths:
                    if config_path.exists():
                        file_config = self._load_from_file(str(config_path))
                        config_data.update(file_config)
                        break
            
            # Build configuration objects
            self._config = self._build_config(environment, config_data)
            
            logger.info("Configuration loaded successfully")
            return self._config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        return {
            # Database configuration
            "database_url": os.getenv("DATABASE_URL", "postgresql://localhost:5432/xorb"),
            "database_pool_size": int(os.getenv("DATABASE_POOL_SIZE", "20")),
            "database_echo": os.getenv("DATABASE_ECHO", "false").lower() == "true",
            
            # Redis configuration
            "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
            "redis_password": os.getenv("REDIS_PASSWORD"),
            "redis_database": int(os.getenv("REDIS_DATABASE", "0")),
            
            # Security configuration
            "jwt_secret": os.getenv("JWT_SECRET", secrets.token_hex(32)),
            "jwt_expiration_hours": int(os.getenv("JWT_EXPIRATION_HOURS", "1")),
            "require_mfa": os.getenv("REQUIRE_MFA", "true").lower() == "true",
            "max_login_attempts": int(os.getenv("MAX_LOGIN_ATTEMPTS", "5")),
            
            # PTaaS configuration
            "max_concurrent_scans": int(os.getenv("MAX_CONCURRENT_SCANS", "15")),
            "scan_timeout_minutes": int(os.getenv("SCAN_TIMEOUT_MINUTES", "120")),
            "enable_stealth_scanning": os.getenv("ENABLE_STEALTH_SCANNING", "true").lower() == "true",
            
            # Threat hunting configuration
            "max_concurrent_queries": int(os.getenv("MAX_CONCURRENT_QUERIES", "10")),
            "query_timeout_seconds": int(os.getenv("QUERY_TIMEOUT_SECONDS", "300")),
            "enable_ml_analysis": os.getenv("ENABLE_ML_ANALYSIS", "true").lower() == "true",
            
            # MITRE configuration
            "mitre_confidence_threshold": float(os.getenv("MITRE_CONFIDENCE_THRESHOLD", "0.6")),
            "enable_mitre_attribution": os.getenv("ENABLE_MITRE_ATTRIBUTION", "true").lower() == "true",
            
            # Monitoring configuration
            "enable_prometheus": os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "enable_health_checks": os.getenv("ENABLE_HEALTH_CHECKS", "true").lower() == "true",
            
            # Feature flags
            "enable_advanced_features": os.getenv("ENABLE_ADVANCED_FEATURES", "true").lower() == "true",
            "enable_ai_enhancement": os.getenv("ENABLE_AI_ENHANCEMENT", "true").lower() == "true",
            "enable_quantum_security": os.getenv("ENABLE_QUANTUM_SECURITY", "false").lower() == "true",
            
            # External APIs
            "nvidia_api_key": os.getenv("NVIDIA_API_KEY"),
            "openrouter_api_key": os.getenv("OPENROUTER_API_KEY"),
            
            # CORS
            "cors_allow_origins": os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if os.getenv("CORS_ALLOW_ORIGINS") else []
        }
    
    def _load_from_file(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            config_path = Path(config_file)
            
            if not config_path.exists():
                logger.warning(f"Configuration file not found: {config_file}")
                return {}
            
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() == '.json':
                    return json.load(f)
                elif config_path.suffix.lower() in ['.yaml', '.yml']:
                    try:
                        import yaml
                        return yaml.safe_load(f)
                    except ImportError:
                        logger.warning("PyYAML not installed, cannot load YAML config")
                        return {}
                else:
                    logger.warning(f"Unsupported config file format: {config_path.suffix}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Failed to load config file {config_file}: {e}")
            return {}
    
    def _build_config(self, environment: Environment, config_data: Dict[str, Any]) -> ProductionConfig:
        """Build configuration objects from loaded data"""
        
        # Database configuration
        database_config = DatabaseConfig(
            url=config_data["database_url"],
            pool_size=config_data.get("database_pool_size", 20),
            echo=config_data.get("database_echo", False)
        )
        
        # Redis configuration
        redis_config = RedisConfig(
            url=config_data["redis_url"],
            password=config_data.get("redis_password"),
            database=config_data.get("redis_database", 0)
        )
        
        # Security configuration
        security_config = SecurityConfig(
            jwt_secret=config_data["jwt_secret"],
            jwt_expiration_hours=config_data.get("jwt_expiration_hours", 1),
            require_mfa=config_data.get("require_mfa", True),
            max_login_attempts=config_data.get("max_login_attempts", 5),
            allowed_origins=config_data.get("cors_allow_origins", [])
        )
        
        # Threat hunting configuration
        threat_hunting_config = ThreatHuntingConfig(
            max_concurrent_queries=config_data.get("max_concurrent_queries", 10),
            query_timeout_seconds=config_data.get("query_timeout_seconds", 300),
            enable_ml_analysis=config_data.get("enable_ml_analysis", True)
        )
        
        # MITRE configuration
        mitre_config = MitreConfig(
            confidence_threshold=config_data.get("mitre_confidence_threshold", 0.6),
            enable_attribution=config_data.get("enable_mitre_attribution", True)
        )
        
        # PTaaS configuration
        ptaas_config = PTaaSConfig(
            max_concurrent_scans=config_data.get("max_concurrent_scans", 15),
            scan_timeout_minutes=config_data.get("scan_timeout_minutes", 120),
            enable_stealth_scanning=config_data.get("enable_stealth_scanning", True)
        )
        
        # Monitoring configuration
        monitoring_config = MonitoringConfig(
            enable_prometheus=config_data.get("enable_prometheus", True),
            log_level=LogLevel(config_data.get("log_level", "INFO")),
            enable_health_checks=config_data.get("enable_health_checks", True)
        )
        
        # External APIs
        external_apis = {}
        if config_data.get("nvidia_api_key"):
            external_apis["nvidia"] = config_data["nvidia_api_key"]
        if config_data.get("openrouter_api_key"):
            external_apis["openrouter"] = config_data["openrouter_api_key"]
        
        return ProductionConfig(
            environment=environment,
            debug=environment in [Environment.DEVELOPMENT, Environment.TESTING],
            database=database_config,
            redis=redis_config,
            security=security_config,
            threat_hunting=threat_hunting_config,
            mitre=mitre_config,
            ptaas=ptaas_config,
            monitoring=monitoring_config,
            enable_advanced_features=config_data.get("enable_advanced_features", True),
            enable_ai_enhancement=config_data.get("enable_ai_enhancement", True),
            enable_quantum_security=config_data.get("enable_quantum_security", False),
            external_apis=external_apis
        )
    
    def get_config(self) -> ProductionConfig:
        """Get current configuration"""
        if not self._config:
            raise RuntimeError("Configuration not loaded. Call load_configuration() first.")
        return self._config
    
    def reload_configuration(self) -> ProductionConfig:
        """Reload configuration from sources"""
        logger.info("Reloading configuration...")
        return self.load_configuration()
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration and return validation results"""
        if not self._config:
            return {"valid": False, "error": "No configuration loaded"}
        
        try:
            # Re-run validation
            self._config._validate_config()
            
            return {
                "valid": True,
                "environment": self._config.environment.value,
                "advanced_features_enabled": self._config.enable_advanced_features,
                "ai_enhancement_enabled": self._config.enable_ai_enhancement,
                "services_configured": {
                    "database": bool(self._config.database.url),
                    "redis": bool(self._config.redis.url),
                    "threat_hunting": self._config.threat_hunting.enable_real_time_hunting,
                    "mitre": self._config.mitre.enable_attribution,
                    "ptaas": self._config.ptaas.enable_auto_orchestration,
                    "monitoring": self._config.monitoring.enable_health_checks
                }
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "environment": self._config.environment.value if self._config else "unknown"
            }


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None

def get_config_manager() -> ConfigurationManager:
    """Get global configuration manager instance"""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    
    return _config_manager

def get_config() -> ProductionConfig:
    """Get current production configuration"""
    return get_config_manager().get_config()

def load_config(config_file: Optional[str] = None) -> ProductionConfig:
    """Load production configuration"""
    return get_config_manager().load_configuration(config_file)