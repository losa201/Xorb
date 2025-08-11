"""
Secure configuration management with Vault integration
Replaces hardcoded secrets with secure retrieval
"""

import os
from typing import Optional
from .config import AppSettings
from ...common.vault_client_enhanced import get_secret_manager
from .logging import get_logger

logger = get_logger(__name__)

class SecureAppSettings(AppSettings):
    """Enhanced app settings with secure secret management"""
    
    def __init__(self, **kwargs):
        self._secret_manager = get_secret_manager()
        
        # Log secret manager status
        health = self._secret_manager.health_check()
        logger.info("Secret manager initialized", 
                   vault_available=health["vault_connected"],
                   fallback_mode=health["fallback_mode"],
                   sources=health["sources_available"])
        
        super().__init__(**kwargs)
    
    @property
    def jwt_secret_key(self) -> str:
        """Get JWT secret securely from Vault"""
        secret = self._secret_manager.get_secret('secret/xorb/jwt', 'secret_key')
        
        if not secret:
            # In development, generate temporary secret
            if self.environment == "development":
                logger.warning("Generating temporary JWT secret for development")
                secret = self._secret_manager.generate_secure_secret('jwt_secret')
            else:
                raise ValueError("JWT secret not found in secure storage")
        
        # Validate secret
        if not self._secret_manager.validate_secret('jwt_secret', secret):
            raise ValueError("JWT secret does not meet security requirements")
        
        return secret
    
    @property
    def database_url(self) -> str:
        """Get database URL with secure password"""
        # Get password from secure storage
        password = self._secret_manager.get_secret('secret/xorb/database', 'password')
        
        if not password:
            if self.environment == "development":
                logger.warning("Generating temporary database password for development")
                password = self._secret_manager.generate_secure_secret('db_password')
            else:
                raise ValueError("Database password not found in secure storage")
        
        # Validate password
        if not self._secret_manager.validate_secret('database_password', password):
            raise ValueError("Database password does not meet security requirements")
        
        # Construct URL from template
        template = os.getenv('DATABASE_URL_TEMPLATE', 
                           'postgresql://xorb_user:PASSWORD@localhost:5432/xorb_enterprise')
        
        # Replace placeholder safely
        return template.replace('PASSWORD', password)
    
    @property 
    def redis_url(self) -> str:
        """Get Redis URL with secure password"""
        password = self._secret_manager.get_secret('secret/xorb/redis', 'password')
        
        if not password:
            if self.environment == "development":
                logger.warning("Generating temporary Redis password for development")
                password = self._secret_manager.generate_secure_secret('redis_password')
            else:
                raise ValueError("Redis password not found in secure storage")
        
        if not self._secret_manager.validate_secret('redis_password', password):
            raise ValueError("Redis password does not meet security requirements")
        
        template = os.getenv('REDIS_URL_TEMPLATE',
                           'redis://:PASSWORD@localhost:6379/0')
        
        return template.replace('PASSWORD', password)

def get_secure_settings() -> SecureAppSettings:
    """Get secure application settings"""
    return SecureAppSettings()