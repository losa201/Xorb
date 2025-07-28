#!/usr/bin/env python3
"""
XORB Pydantic Schema Validation System
Implements robust typing and validation across all services
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from pydantic import BaseModel, Field, root_validator, validator
    from pydantic.config import BaseConfig
except ImportError:
    print("Installing pydantic...")
    os.system("pip3 install pydantic")
    from pydantic import BaseModel, Field, root_validator, validator
    from pydantic.config import BaseConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XORBBaseConfig(BaseConfig):
    """Base configuration for all XORB Pydantic models"""
    validate_assignment = True
    extra = "forbid"
    use_enum_values = True
    anystr_strip_whitespace = True

class DatabaseConfig(BaseModel):
    """Database configuration with validation"""

    class Config(XORBBaseConfig):
        pass

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    database: str = Field(default="xorb", min_length=1, description="Database name")
    username: str = Field(default="xorb", min_length=1, description="Database username")
    password: str = Field(default="", min_length=8, description="Database password")
    pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, le=100, description="Max overflow connections")

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

class RedisConfig(BaseModel):
    """Redis configuration with validation"""

    class Config(XORBBaseConfig):
        pass

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    password: str | None = Field(default=None, description="Redis password")
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    max_connections: int = Field(default=50, ge=1, le=1000, description="Max connections")

class AIConfig(BaseModel):
    """AI service configuration with validation"""

    class Config(XORBBaseConfig):
        pass

    nvidia_api_key: str | None = Field(default=None, description="NVIDIA API key")
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    model_name: str = Field(default="qwen-2.5-72b-instruct", description="Default model")
    max_tokens: int = Field(default=4096, ge=1, le=32768, description="Max tokens per request")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Model temperature")
    timeout: int = Field(default=30, ge=1, le=300, description="Request timeout in seconds")

    @validator('nvidia_api_key')
    def validate_nvidia_key(cls, v):
        if v and len(v) < 10:
            raise ValueError('NVIDIA API key seems too short')
        return v

class SecurityConfig(BaseModel):
    """Security configuration with validation"""

    class Config(XORBBaseConfig):
        pass

    jwt_secret: str = Field(min_length=32, description="JWT secret key")
    encryption_key: str = Field(min_length=32, description="Encryption key")
    allowed_hosts: list[str] = Field(default=["localhost", "127.0.0.1"], description="Allowed hosts")
    cors_origins: list[str] = Field(default=["http://localhost:3000"], description="CORS origins")
    max_login_attempts: int = Field(default=5, ge=1, le=20, description="Max login attempts")
    session_timeout: int = Field(default=3600, ge=300, le=86400, description="Session timeout")

    @validator('jwt_secret')
    def validate_jwt_secret(cls, v):
        if len(v) < 32:
            raise ValueError('JWT secret must be at least 32 characters')
        return v

class OrchestrationConfig(BaseModel):
    """Orchestration service configuration"""

    class Config(XORBBaseConfig):
        pass

    max_concurrent_agents: int = Field(default=32, ge=1, le=128, description="Max concurrent agents")
    agent_timeout: int = Field(default=300, ge=30, le=3600, description="Agent timeout in seconds")
    retry_attempts: int = Field(default=3, ge=1, le=10, description="Retry attempts")
    circuit_breaker_threshold: int = Field(default=5, ge=1, le=20, description="Circuit breaker threshold")
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")

class MonitoringConfig(BaseModel):
    """Monitoring configuration with validation"""

    class Config(XORBBaseConfig):
        pass

    prometheus_port: int = Field(default=9090, ge=1000, le=65535, description="Prometheus port")
    grafana_port: int = Field(default=3000, ge=1000, le=65535, description="Grafana port")
    metrics_path: str = Field(default="/metrics", description="Metrics endpoint path")
    scrape_interval: int = Field(default=15, ge=5, le=300, description="Scrape interval in seconds")
    retention_days: int = Field(default=15, ge=1, le=365, description="Data retention in days")

class XORBConfig(BaseModel):
    """Main XORB configuration with all subsystems"""

    class Config(XORBBaseConfig):
        pass

    environment: str = Field(default="development", regex="^(development|staging|production)$")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    security: SecurityConfig = Field(default_factory=lambda: SecurityConfig(
        jwt_secret="default_jwt_secret_32_characters_min",
        encryption_key="default_encryption_key_32_chars_min"
    ))
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @root_validator
    def validate_config(cls, values):
        env = values.get('environment')
        if env == 'production':
            # Enforce stronger security in production
            security = values.get('security')
            if security and (security.jwt_secret.startswith('default') or
                           security.encryption_key.startswith('default')):
                raise ValueError('Production environment requires custom security keys')
        return values

class AgentConfig(BaseModel):
    """Agent configuration with validation"""

    class Config(XORBBaseConfig):
        pass

    name: str = Field(min_length=1, max_length=100, description="Agent name")
    type: str = Field(regex="^(scanner|analyzer|reporter|stealth)$", description="Agent type")
    enabled: bool = Field(default=True, description="Agent enabled")
    priority: int = Field(default=5, ge=1, le=10, description="Agent priority")
    max_concurrent: int = Field(default=5, ge=1, le=50, description="Max concurrent instances")
    timeout: int = Field(default=300, ge=30, le=3600, description="Execution timeout")
    capabilities: list[str] = Field(default_factory=list, description="Agent capabilities")

    @validator('capabilities')
    def validate_capabilities(cls, v):
        allowed_caps = {
            'web_scraping', 'port_scanning', 'vulnerability_assessment',
            'social_engineering', 'phishing', 'malware_analysis',
            'network_analysis', 'data_extraction', 'reporting'
        }
        for cap in v:
            if cap not in allowed_caps:
                raise ValueError(f'Unknown capability: {cap}')
        return v

class TaskConfig(BaseModel):
    """Task configuration with validation"""

    class Config(XORBBaseConfig):
        pass

    id: str = Field(min_length=1, description="Task ID")
    type: str = Field(min_length=1, description="Task type")
    priority: int = Field(default=5, ge=1, le=10, description="Task priority")
    timeout: int = Field(default=300, ge=30, le=3600, description="Task timeout")
    retry_count: int = Field(default=3, ge=0, le=10, description="Retry count")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Task parameters")

    @validator('parameters')
    def validate_parameters(cls, v):
        # Ensure no sensitive data in parameters
        sensitive_keys = {'password', 'secret', 'key', 'token'}
        for key in v.keys():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                raise ValueError(f'Sensitive parameter key detected: {key}')
        return v

class ValidationEngine:
    """Engine for validating and converting configurations"""

    def __init__(self):
        self.schemas = {
            'xorb_config': XORBConfig,
            'agent_config': AgentConfig,
            'task_config': TaskConfig,
            'database_config': DatabaseConfig,
            'ai_config': AIConfig,
            'security_config': SecurityConfig
        }

    def validate_file(self, file_path: Path, schema_name: str = None) -> dict[str, Any]:
        """Validate a configuration file"""
        try:
            with open(file_path) as f:
                if file_path.suffix == '.json':
                    data = json.load(f)
                elif file_path.suffix in ['.yml', '.yaml']:
                    import yaml
                    data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported file format: {file_path.suffix}")

            # Auto-detect schema if not specified
            if not schema_name:
                schema_name = self._detect_schema(data, file_path)

            if schema_name not in self.schemas:
                raise ValueError(f"Unknown schema: {schema_name}")

            schema_class = self.schemas[schema_name]
            validated_config = schema_class(**data)

            return {
                'valid': True,
                'schema': schema_name,
                'config': validated_config.dict(),
                'errors': []
            }

        except Exception as e:
            return {
                'valid': False,
                'schema': schema_name,
                'config': None,
                'errors': [str(e)]
            }

    def _detect_schema(self, data: dict[str, Any], file_path: Path) -> str:
        """Auto-detect configuration schema based on content"""
        if 'database' in data and 'ai' in data and 'security' in data:
            return 'xorb_config'
        elif 'name' in data and 'type' in data and 'capabilities' in data:
            return 'agent_config'
        elif 'id' in data and 'type' in data and 'parameters' in data:
            return 'task_config'
        elif 'host' in data and 'port' in data and 'database' in data:
            return 'database_config'
        elif 'nvidia_api_key' in data or 'openai_api_key' in data:
            return 'ai_config'
        elif 'jwt_secret' in data or 'encryption_key' in data:
            return 'security_config'
        else:
            # Default based on filename
            name = file_path.stem.lower()
            if 'agent' in name:
                return 'agent_config'
            elif 'task' in name:
                return 'task_config'
            elif 'database' in name or 'db' in name:
                return 'database_config'
            elif 'ai' in name or 'model' in name:
                return 'ai_config'
            elif 'security' in name or 'auth' in name:
                return 'security_config'
            else:
                return 'xorb_config'

    def validate_directory(self, directory: Path) -> dict[str, Any]:
        """Validate all configuration files in a directory"""
        results = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'files': {}
        }

        config_patterns = ['*.json', '*.yml', '*.yaml']
        config_files = []

        for pattern in config_patterns:
            config_files.extend(directory.glob(pattern))
            config_files.extend(directory.glob(f"**/{pattern}"))

        for file_path in config_files:
            if 'template' in file_path.name or file_path.name.startswith('.'):
                continue

            results['total_files'] += 1
            file_result = self.validate_file(file_path)

            if file_result['valid']:
                results['valid_files'] += 1
            else:
                results['invalid_files'] += 1

            results['files'][str(file_path)] = file_result

        return results

    def generate_validated_config(self, environment: str = "development") -> XORBConfig:
        """Generate a validated configuration for the specified environment"""
        config_data = {
            'environment': environment,
            'debug': environment != 'production',
            'log_level': 'DEBUG' if environment == 'development' else 'INFO',
            'database': {
                'host': os.getenv('POSTGRES_HOST', 'localhost'),
                'port': int(os.getenv('POSTGRES_PORT', '5432')),
                'database': os.getenv('POSTGRES_DB', 'xorb'),
                'username': os.getenv('POSTGRES_USER', 'xorb'),
                'password': os.getenv('POSTGRES_PASSWORD', 'xorb_postgres_secure_2024'),
            },
            'redis': {
                'host': os.getenv('REDIS_HOST', 'localhost'),
                'port': int(os.getenv('REDIS_PORT', '6379')),
                'password': os.getenv('REDIS_PASSWORD', 'xorb_redis_secure_2024'),
            },
            'ai': {
                'nvidia_api_key': os.getenv('NVIDIA_API_KEY'),
                'openai_api_key': os.getenv('OPENAI_API_KEY'),
                'model_name': os.getenv('AI_MODEL', 'qwen-2.5-72b-instruct'),
            },
            'security': {
                'jwt_secret': os.getenv('JWT_SECRET', 'xorb_jwt_secret_32_characters_minimum'),
                'encryption_key': os.getenv('ENCRYPTION_KEY', 'xorb_encryption_key_32_chars_minimum'),
                'allowed_hosts': os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(','),
            },
            'orchestration': {
                'max_concurrent_agents': int(os.getenv('MAX_CONCURRENT_AGENTS', '32')),
                'agent_timeout': int(os.getenv('AGENT_TIMEOUT', '300')),
            },
            'monitoring': {
                'prometheus_port': int(os.getenv('PROMETHEUS_PORT', '9090')),
                'grafana_port': int(os.getenv('GRAFANA_PORT', '3000')),
            }
        }

        return XORBConfig(**config_data)

def main():
    """Main validation execution"""
    print("🔧 XORB Pydantic Schema Validation")
    print("==================================")

    base_path = Path("/root/Xorb")
    validator = ValidationEngine()

    # Validate existing configuration files
    print("🔍 Validating existing configuration files...")
    results = validator.validate_directory(base_path)

    print(f"Files processed: {results['total_files']}")
    print(f"Valid: {results['valid_files']}")
    print(f"Invalid: {results['invalid_files']}")

    # Show validation errors
    for file_path, file_result in results['files'].items():
        if not file_result['valid']:
            print(f"\n❌ {file_path}:")
            for error in file_result['errors']:
                print(f"   - {error}")

    # Generate validated configurations for all environments
    print("\n🏗️ Generating validated configurations...")

    config_dir = base_path / "config"
    config_dir.mkdir(exist_ok=True)

    for env in ['development', 'staging', 'production']:
        try:
            config = validator.generate_validated_config(env)
            config_file = config_dir / f"xorb_config_{env}.json"

            with open(config_file, 'w') as f:
                json.dump(config.dict(), f, indent=2)

            print(f"✅ Generated: {config_file}")

        except Exception as e:
            print(f"❌ Failed to generate {env} config: {e}")

    # Generate schema documentation
    print("\n📋 Generating schema documentation...")
    schema_doc = {
        'schemas': {},
        'generated_at': datetime.now().isoformat(),
        'version': '2.0.0'
    }

    for name, schema_class in validator.schemas.items():
        schema_doc['schemas'][name] = schema_class.schema()

    doc_file = base_path / "docs" / "schemas.json"
    doc_file.parent.mkdir(exist_ok=True)

    with open(doc_file, 'w') as f:
        json.dump(schema_doc, f, indent=2)

    print(f"📄 Schema documentation: {doc_file}")

    # Calculate validation score
    if results['total_files'] > 0:
        validation_score = (results['valid_files'] / results['total_files']) * 100
    else:
        validation_score = 100

    print(f"\n🎯 Validation Score: {validation_score:.1f}%")

    if validation_score >= 95:
        print("✅ EXCELLENT - All configurations properly validated")
    elif validation_score >= 85:
        print("✅ GOOD - Minor validation issues")
    elif validation_score >= 70:
        print("⚠️ FAIR - Some validation improvements needed")
    else:
        print("❌ POOR - Significant validation issues found")

    return validation_score

if __name__ == "__main__":
    main()
