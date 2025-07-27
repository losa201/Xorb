"""
XORB Core Configuration Management

Centralized configuration for all XORB components with environment variable support.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from urllib.parse import urlparse

@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    postgres_host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    postgres_port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    postgres_db: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "xorb"))
    postgres_user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "xorb"))
    postgres_password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""))
    
    redis_host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    redis_port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    redis_password: str = field(default_factory=lambda: os.getenv("REDIS_PASSWORD", ""))
    
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    
    qdrant_host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    qdrant_port: int = field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))

    @property
    def postgres_url(self) -> str:
        """Construct PostgreSQL connection URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def redis_url(self) -> str:
        """Construct Redis connection URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}"
        return f"redis://{self.redis_host}:{self.redis_port}"

@dataclass
class AIConfig:
    """AI service configuration."""
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    nvidia_api_key: str = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    
    default_model: str = field(default_factory=lambda: os.getenv("DEFAULT_LLM_MODEL", "qwen/qwen-2.5-72b-instruct"))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "4096")))
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7")))

@dataclass
class SecurityConfig:
    """Security configuration."""
    jwt_secret_key: str = field(default_factory=lambda: os.getenv("JWT_SECRET_KEY", ""))
    encryption_key: str = field(default_factory=lambda: os.getenv("ENCRYPTION_KEY", ""))
    api_rate_limit: int = field(default_factory=lambda: int(os.getenv("API_RATE_LIMIT", "100")))
    
    # TLS Configuration
    tls_cert_path: str = field(default_factory=lambda: os.getenv("TLS_CERT_PATH", ""))
    tls_key_path: str = field(default_factory=lambda: os.getenv("TLS_KEY_PATH", ""))

@dataclass
class OrchestrationConfig:
    """Orchestration configuration."""
    max_concurrent_agents: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_AGENTS", "32")))
    agent_timeout: int = field(default_factory=lambda: int(os.getenv("AGENT_TIMEOUT", "300")))
    campaign_timeout: int = field(default_factory=lambda: int(os.getenv("CAMPAIGN_TIMEOUT", "3600")))
    
    # EPYC Optimization
    epyc_numa_nodes: int = field(default_factory=lambda: int(os.getenv("EPYC_NUMA_NODES", "2")))
    epyc_cores_per_node: int = field(default_factory=lambda: int(os.getenv("EPYC_CORES_PER_NODE", "32")))

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    prometheus_host: str = field(default_factory=lambda: os.getenv("PROMETHEUS_HOST", "localhost"))
    prometheus_port: int = field(default_factory=lambda: int(os.getenv("PROMETHEUS_PORT", "9090")))
    
    grafana_host: str = field(default_factory=lambda: os.getenv("GRAFANA_HOST", "localhost"))
    grafana_port: int = field(default_factory=lambda: int(os.getenv("GRAFANA_PORT", "3000")))
    
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "json"))

@dataclass
class XORBConfig:
    """Main XORB configuration."""
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    
    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Paths
    base_path: Path = field(default_factory=lambda: Path(os.getenv("XORB_BASE_PATH", "/root/Xorb")))
    data_path: Path = field(default_factory=lambda: Path(os.getenv("XORB_DATA_PATH", "/root/Xorb/data")))
    logs_path: Path = field(default_factory=lambda: Path(os.getenv("XORB_LOGS_PATH", "/root/Xorb/logs")))

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._ensure_paths()

    def _validate_config(self):
        """Validate critical configuration values."""
        if self.environment == "production":
            if not self.security.jwt_secret_key:
                raise ValueError("JWT_SECRET_KEY must be set in production")
            if not self.database.postgres_password:
                raise ValueError("POSTGRES_PASSWORD must be set in production")

    def _ensure_paths(self):
        """Ensure required paths exist."""
        for path in [self.data_path, self.logs_path]:
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "database": {
                "postgres_url": self.database.postgres_url,
                "redis_url": self.database.redis_url,
                "neo4j_uri": self.database.neo4j_uri,
            },
            "orchestration": {
                "max_concurrent_agents": self.orchestration.max_concurrent_agents,
                "agent_timeout": self.orchestration.agent_timeout,
            },
            "monitoring": {
                "log_level": self.monitoring.log_level,
                "log_format": self.monitoring.log_format,
            }
        }

# Global configuration instance
config = XORBConfig()