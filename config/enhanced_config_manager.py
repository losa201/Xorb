#!/usr/bin/env python3
"""
Enhanced Configuration Manager for XORB Supreme
Handles secure loading of API keys, budget controls, and system settings
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
from dotenv import load_dotenv
import secrets
from cryptography.fernet import Fernet
import base64

logger = logging.getLogger(__name__)

@dataclass
class LLMProviderConfig:
    """Configuration for LLM provider"""
    name: str
    api_key: str
    base_url: Optional[str] = None
    organization: Optional[str] = None
    models: List[str] = None
    daily_budget: float = 10.0
    per_request_limit: float = 2.0
    rate_limit_rpm: int = 60
    enabled: bool = True
    
    def __post_init__(self):
        if self.models is None:
            self.models = []

@dataclass
class BudgetConfig:
    """Budget and cost control configuration"""
    daily_limit: float
    monthly_limit: float
    per_request_limit: float
    alert_threshold: float
    warning_percentage: float
    auto_disable_on_limit: bool = True
    cost_tracking_enabled: bool = True

@dataclass
class SecurityConfig:
    """Security and authentication settings"""
    encryption_key: str
    session_secret: str
    api_key_rotation_days: int
    require_authorization: bool
    enable_responsible_disclosure: bool
    max_severity_level: str
    allowed_domains: List[str] = None
    
    def __post_init__(self):
        if self.allowed_domains is None:
            self.allowed_domains = []

@dataclass
class ScrapingConfig:
    """Web scraping configuration"""
    user_agent: str
    rate_limit_delay: int
    max_programs: int
    enable_screenshots: bool
    screenshot_quality: int
    timeout_seconds: int
    retry_attempts: int

@dataclass
class ReportingConfig:
    """Professional reporting settings"""
    default_format: str
    enable_pdf_export: bool
    retention_days: int
    company_name: str
    assessor_name: str
    assessor_credentials: str
    auto_generate_bounty_reports: bool
    enable_executive_reports: bool
    enable_compliance_reports: bool

@dataclass
class PerformanceConfig:
    """Performance and scaling settings"""
    max_concurrent_requests: int
    max_concurrent_targets: int
    worker_thread_count: int
    enable_caching: bool
    cache_size_mb: int
    cache_ttl_hours: int

class EnhancedConfigManager:
    """Comprehensive configuration management system"""
    
    def __init__(self, config_path: Optional[str] = None, env_file: Optional[str] = None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "..", "config.json")
        self.env_file = env_file or os.path.join(os.path.dirname(__file__), "..", ".env")
        
        # Load environment variables
        if os.path.exists(self.env_file):
            load_dotenv(self.env_file)
            logger.info(f"Loaded environment variables from {self.env_file}")
        
        # Initialize configuration
        self.config = self._load_configuration()
        self.cipher_suite = self._initialize_encryption()
        
        # Validate configuration
        self._validate_configuration()
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration from multiple sources"""
        config = {}
        
        # Load from JSON file if exists
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config.update(json.load(f))
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        
        # Override with environment variables
        env_config = self._load_from_environment()
        config.update(env_config)
        
        # Set defaults for missing values
        self._set_configuration_defaults(config)
        
        return config
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {}
        
        # LLM API Keys
        llm_providers = {}
        
        if os.getenv('OPENAI_API_KEY'):
            llm_providers['openai'] = {
                'api_key': os.getenv('OPENAI_API_KEY'),
                'organization': os.getenv('OPENAI_ORGANIZATION'),
                'models': ['gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo'],
                'base_url': 'https://api.openai.com/v1',
                'daily_budget': float(os.getenv('OPENAI_DAILY_BUDGET', '15.0')),
                'enabled': True
            }
        
        if os.getenv('OPENROUTER_API_KEY'):
            llm_providers['openrouter'] = {
                'api_key': os.getenv('OPENROUTER_API_KEY'),
                'base_url': 'https://openrouter.ai/api/v1',
                'models': [
                    'anthropic/claude-3-opus',
                    'anthropic/claude-3-sonnet', 
                    'google/gemini-pro-1.5',
                    'qwen/qwen3-235b-a22b-07-25:free'
                ],
                'daily_budget': float(os.getenv('OPENROUTER_DAILY_BUDGET', '20.0')),
                'enabled': True
            }
        
        if os.getenv('CLAUDE_API_KEY'):
            llm_providers['claude'] = {
                'api_key': os.getenv('CLAUDE_API_KEY'),
                'base_url': 'https://api.anthropic.com',
                'models': ['claude-3-opus-20240229', 'claude-3-sonnet-20240229'],
                'daily_budget': float(os.getenv('CLAUDE_DAILY_BUDGET', '10.0')),
                'enabled': True
            }
        
        if os.getenv('GEMINI_API_KEY'):
            llm_providers['gemini'] = {
                'api_key': os.getenv('GEMINI_API_KEY'),
                'models': ['gemini-1.5-pro', 'gemini-1.5-flash'],
                'daily_budget': float(os.getenv('GEMINI_DAILY_BUDGET', '8.0')),
                'enabled': True
            }
        
        if llm_providers:
            env_config['llm_providers'] = llm_providers
        
        # Budget Configuration
        env_config['budget'] = {
            'daily_limit': float(os.getenv('DAILY_BUDGET_LIMIT', '25.0')),
            'monthly_limit': float(os.getenv('MONTHLY_BUDGET_LIMIT', '200.0')),
            'per_request_limit': float(os.getenv('PER_REQUEST_LIMIT', '5.0')),
            'alert_threshold': float(os.getenv('COST_ALERT_THRESHOLD', '50.0')),
            'warning_percentage': float(os.getenv('BUDGET_WARNING_PERCENTAGE', '80.0'))
        }
        
        # Database Configuration
        env_config['database'] = {
            'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379'),
            'redis_password': os.getenv('REDIS_PASSWORD'),
            'redis_db': int(os.getenv('REDIS_DB', '0')),
            'database_url': os.getenv('DATABASE_URL', 'sqlite:///xorb_knowledge.db')
        }
        
        # Security Configuration
        env_config['security'] = {
            'encryption_key': os.getenv('ENCRYPTION_KEY', self._generate_encryption_key()),
            'session_secret': os.getenv('SESSION_SECRET', secrets.token_hex(32)),
            'api_key_rotation_days': int(os.getenv('API_KEY_ROTATION_DAYS', '90')),
            'require_authorization': os.getenv('REQUIRE_AUTHORIZATION_CONFIRMATION', 'true').lower() == 'true',
            'enable_responsible_disclosure': os.getenv('ENABLE_RESPONSIBLE_DISCLOSURE', 'true').lower() == 'true',
            'max_severity_level': os.getenv('MAX_VULNERABILITY_SEVERITY', 'critical'),
            'allowed_domains': [d.strip() for d in os.getenv('ALLOWED_DOMAINS', '').split(',') if d.strip()]
        }
        
        # Scraping Configuration
        env_config['scraping'] = {
            'user_agent': os.getenv('HACKERONE_USER_AGENT', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'),
            'rate_limit_delay': int(os.getenv('HACKERONE_RATE_LIMIT_DELAY', '2')),
            'max_programs': int(os.getenv('HACKERONE_MAX_PROGRAMS', '50')),
            'enable_screenshots': os.getenv('ENABLE_SCREENSHOTS', 'true').lower() == 'true',
            'screenshot_quality': int(os.getenv('SCREENSHOT_QUALITY', '80')),
            'timeout_seconds': int(os.getenv('SCRAPING_TIMEOUT', '30')),
            'retry_attempts': int(os.getenv('SCRAPING_RETRY_ATTEMPTS', '3'))
        }
        
        # Reporting Configuration
        env_config['reporting'] = {
            'default_format': os.getenv('DEFAULT_REPORT_FORMAT', 'markdown'),
            'enable_pdf_export': os.getenv('ENABLE_PDF_EXPORT', 'true').lower() == 'true',
            'retention_days': int(os.getenv('REPORT_RETENTION_DAYS', '180')),
            'company_name': os.getenv('COMPANY_NAME', 'XORB Security Solutions'),
            'assessor_name': os.getenv('ASSESSOR_NAME', 'Security Analyst'),
            'assessor_credentials': os.getenv('ASSESSOR_CREDENTIALS', 'Security Professional'),
            'auto_generate_bounty_reports': os.getenv('AUTO_GENERATE_BUG_BOUNTY_REPORTS', 'true').lower() == 'true',
            'enable_executive_reports': os.getenv('ENABLE_EXECUTIVE_REPORTS', 'true').lower() == 'true',
            'enable_compliance_reports': os.getenv('ENABLE_COMPLIANCE_REPORTS', 'true').lower() == 'true'
        }
        
        # Performance Configuration
        env_config['performance'] = {
            'max_concurrent_requests': int(os.getenv('MAX_CONCURRENT_REQUESTS', '5')),
            'max_concurrent_targets': int(os.getenv('MAX_CONCURRENT_TARGETS', '3')),
            'worker_thread_count': int(os.getenv('WORKER_THREAD_COUNT', '4')),
            'enable_caching': os.getenv('ENABLE_REQUEST_CACHING', 'true').lower() == 'true',
            'cache_size_mb': int(os.getenv('CACHE_SIZE_MB', '100')),
            'cache_ttl_hours': int(os.getenv('CACHE_TTL_HOURS', '24'))
        }
        
        # AI Enhancement Settings
        env_config['ai_enhancement'] = {
            'creativity_level': float(os.getenv('CREATIVITY_LEVEL', '0.8')),
            'enable_polyglot_payloads': os.getenv('ENABLE_POLYGLOT_PAYLOADS', 'true').lower() == 'true',
            'enable_chained_exploitation': os.getenv('ENABLE_CHAINED_EXPLOITATION', 'true').lower() == 'true',
            'enable_business_logic_analysis': os.getenv('ENABLE_BUSINESS_LOGIC_ANALYSIS', 'true').lower() == 'true',
            'industry_context_detection': os.getenv('INDUSTRY_CONTEXT_DETECTION', 'true').lower() == 'true'
        }
        
        return env_config
    
    def _set_configuration_defaults(self, config: Dict[str, Any]):
        """Set default values for missing configuration"""
        
        defaults = {
            'version': '3.0.0',
            'debug_mode': False,
            'log_level': 'INFO',
            'enable_free_tier_fallback': True,
            'fallback_model': 'qwen/qwen3-235b-a22b-07-25:free'
        }
        
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
    
    def _initialize_encryption(self) -> Optional[Fernet]:
        """Initialize encryption for sensitive data"""
        try:
            encryption_key = self.config.get('security', {}).get('encryption_key')
            if encryption_key:
                # Ensure key is properly formatted for Fernet
                if len(encryption_key) == 64:  # Hex string
                    key_bytes = bytes.fromhex(encryption_key)
                    key = base64.urlsafe_b64encode(key_bytes)
                else:
                    key = encryption_key.encode()
                
                return Fernet(key)
        except Exception as e:
            logger.warning(f"Failed to initialize encryption: {e}")
        
        return None
    
    def _generate_encryption_key(self) -> str:
        """Generate a new encryption key"""
        return secrets.token_hex(32)
    
    def _validate_configuration(self):
        """Validate configuration integrity"""
        required_sections = ['budget', 'security', 'performance']
        
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Missing required configuration section: {section}")
                raise ValueError(f"Configuration section '{section}' is required")
        
        # Validate LLM providers
        llm_providers = self.config.get('llm_providers', {})
        if not llm_providers:
            logger.warning("No LLM providers configured - system will use free tier only")
        
        # Validate budget limits
        budget = self.config.get('budget', {})
        if budget.get('daily_limit', 0) <= 0:
            logger.warning("Daily budget limit not set or invalid")
        
        logger.info("Configuration validation completed successfully")
    
    def get_llm_provider_config(self, provider_name: str) -> Optional[LLMProviderConfig]:
        """Get configuration for specific LLM provider"""
        providers = self.config.get('llm_providers', {})
        provider_data = providers.get(provider_name)
        
        if not provider_data:
            return None
        
        return LLMProviderConfig(
            name=provider_name,
            api_key=provider_data.get('api_key', ''),
            base_url=provider_data.get('base_url'),
            organization=provider_data.get('organization'),
            models=provider_data.get('models', []),
            daily_budget=provider_data.get('daily_budget', 10.0),
            per_request_limit=provider_data.get('per_request_limit', 2.0),
            rate_limit_rpm=provider_data.get('rate_limit_rpm', 60),
            enabled=provider_data.get('enabled', True)
        )
    
    def get_budget_config(self) -> BudgetConfig:
        """Get budget configuration"""
        budget_data = self.config.get('budget', {})
        
        return BudgetConfig(
            daily_limit=budget_data.get('daily_limit', 25.0),
            monthly_limit=budget_data.get('monthly_limit', 200.0),
            per_request_limit=budget_data.get('per_request_limit', 5.0),
            alert_threshold=budget_data.get('alert_threshold', 50.0),
            warning_percentage=budget_data.get('warning_percentage', 80.0),
            auto_disable_on_limit=budget_data.get('auto_disable_on_limit', True),
            cost_tracking_enabled=budget_data.get('cost_tracking_enabled', True)
        )
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration"""
        security_data = self.config.get('security', {})
        
        return SecurityConfig(
            encryption_key=security_data.get('encryption_key', ''),
            session_secret=security_data.get('session_secret', ''),
            api_key_rotation_days=security_data.get('api_key_rotation_days', 90),
            require_authorization=security_data.get('require_authorization', True),
            enable_responsible_disclosure=security_data.get('enable_responsible_disclosure', True),
            max_severity_level=security_data.get('max_severity_level', 'critical'),
            allowed_domains=security_data.get('allowed_domains', [])
        )
    
    def get_scraping_config(self) -> ScrapingConfig:
        """Get scraping configuration"""
        scraping_data = self.config.get('scraping', {})
        
        return ScrapingConfig(
            user_agent=scraping_data.get('user_agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'),
            rate_limit_delay=scraping_data.get('rate_limit_delay', 2),
            max_programs=scraping_data.get('max_programs', 50),
            enable_screenshots=scraping_data.get('enable_screenshots', True),
            screenshot_quality=scraping_data.get('screenshot_quality', 80),
            timeout_seconds=scraping_data.get('timeout_seconds', 30),
            retry_attempts=scraping_data.get('retry_attempts', 3)
        )
    
    def get_reporting_config(self) -> ReportingConfig:
        """Get reporting configuration"""
        reporting_data = self.config.get('reporting', {})
        
        return ReportingConfig(
            default_format=reporting_data.get('default_format', 'markdown'),
            enable_pdf_export=reporting_data.get('enable_pdf_export', True),
            retention_days=reporting_data.get('retention_days', 180),
            company_name=reporting_data.get('company_name', 'XORB Security Solutions'),
            assessor_name=reporting_data.get('assessor_name', 'Security Analyst'),
            assessor_credentials=reporting_data.get('assessor_credentials', 'Security Professional'),
            auto_generate_bounty_reports=reporting_data.get('auto_generate_bounty_reports', True),
            enable_executive_reports=reporting_data.get('enable_executive_reports', True),
            enable_compliance_reports=reporting_data.get('enable_compliance_reports', True)
        )
    
    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration"""
        performance_data = self.config.get('performance', {})
        
        return PerformanceConfig(
            max_concurrent_requests=performance_data.get('max_concurrent_requests', 5),
            max_concurrent_targets=performance_data.get('max_concurrent_targets', 3),
            worker_thread_count=performance_data.get('worker_thread_count', 4),
            enable_caching=performance_data.get('enable_caching', True),
            cache_size_mb=performance_data.get('cache_size_mb', 100),
            cache_ttl_hours=performance_data.get('cache_ttl_hours', 24)
        )
    
    def get_ai_enhancement_config(self) -> Dict[str, Any]:
        """Get AI enhancement configuration"""
        return self.config.get('ai_enhancement', {
            'creativity_level': 0.8,
            'enable_polyglot_payloads': True,
            'enable_chained_exploitation': True,
            'enable_business_logic_analysis': True,
            'industry_context_detection': True
        })
    
    def encrypt_sensitive_value(self, value: str) -> Optional[str]:
        """Encrypt sensitive configuration value"""
        if self.cipher_suite and value:
            try:
                encrypted = self.cipher_suite.encrypt(value.encode())
                return base64.urlsafe_b64encode(encrypted).decode()
            except Exception as e:
                logger.error(f"Failed to encrypt value: {e}")
        return None
    
    def decrypt_sensitive_value(self, encrypted_value: str) -> Optional[str]:
        """Decrypt sensitive configuration value"""
        if self.cipher_suite and encrypted_value:
            try:
                encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
                decrypted = self.cipher_suite.decrypt(encrypted_bytes)
                return decrypted.decode()
            except Exception as e:
                logger.error(f"Failed to decrypt value: {e}")
        return None
    
    def save_configuration(self, backup: bool = True):
        """Save current configuration to file"""
        try:
            if backup and os.path.exists(self.config_path):
                backup_path = f"{self.config_path}.backup"
                os.rename(self.config_path, backup_path)
                logger.info(f"Created configuration backup: {backup_path}")
            
            # Remove sensitive data before saving
            safe_config = self._sanitize_config_for_save(self.config.copy())
            
            with open(self.config_path, 'w') as f:
                json.dump(safe_config, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def _sanitize_config_for_save(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from configuration before saving"""
        sensitive_keys = ['api_key', 'password', 'secret', 'token', 'encryption_key']
        
        def sanitize_dict(d):
            if isinstance(d, dict):
                return {
                    k: '[REDACTED]' if any(sens in k.lower() for sens in sensitive_keys) 
                    else sanitize_dict(v) 
                    for k, v in d.items()
                }
            elif isinstance(d, list):
                return [sanitize_dict(item) for item in d]
            else:
                return d
        
        return sanitize_dict(config)
    
    def get_all_providers(self) -> List[str]:
        """Get list of all configured LLM providers"""
        return list(self.config.get('llm_providers', {}).keys())
    
    def is_provider_enabled(self, provider_name: str) -> bool:
        """Check if LLM provider is enabled"""
        provider_config = self.get_llm_provider_config(provider_name)
        return provider_config and provider_config.enabled and provider_config.api_key
    
    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled LLM providers"""
        return [name for name in self.get_all_providers() if self.is_provider_enabled(name)]
    
    def update_provider_config(self, provider_name: str, updates: Dict[str, Any]):
        """Update configuration for specific provider"""
        if 'llm_providers' not in self.config:
            self.config['llm_providers'] = {}
        
        if provider_name not in self.config['llm_providers']:
            self.config['llm_providers'][provider_name] = {}
        
        self.config['llm_providers'][provider_name].update(updates)
        logger.info(f"Updated configuration for provider: {provider_name}")
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate that API keys are properly formatted"""
        validation_results = {}
        
        providers = self.config.get('llm_providers', {})
        for provider_name, provider_config in providers.items():
            api_key = provider_config.get('api_key', '')
            
            # Basic validation - check if key looks valid
            is_valid = bool(api_key and len(api_key) > 10)
            
            # Provider-specific validation
            if provider_name == 'openai':
                is_valid = api_key.startswith('sk-')
            elif provider_name == 'openrouter':
                is_valid = api_key.startswith('sk-or-v1-')
            elif provider_name == 'claude':
                is_valid = api_key.startswith('sk-ant-')
            
            validation_results[provider_name] = is_valid
        
        return validation_results
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get high-level configuration summary"""
        enabled_providers = self.get_enabled_providers()
        budget_config = self.get_budget_config()
        
        return {
            'version': self.config.get('version', 'unknown'),
            'enabled_providers': enabled_providers,  
            'total_providers': len(self.get_all_providers()),
            'daily_budget_limit': budget_config.daily_limit,
            'monthly_budget_limit': budget_config.monthly_limit,
            'free_tier_fallback': self.config.get('enable_free_tier_fallback', False),
            'debug_mode': self.config.get('debug_mode', False),
            'configuration_file': self.config_path,
            'environment_file': self.env_file if os.path.exists(self.env_file) else None
        }