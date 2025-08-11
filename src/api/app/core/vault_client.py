"""
HashiCorp Vault Client for Secure Secret Management

This module provides a secure interface to HashiCorp Vault for:
- JWT secret storage and retrieval
- Automatic authentication with AppRole
- Secret versioning and rollback capabilities
- Health monitoring and connection management
"""

import os
import time
from typing import Optional, Dict, Any

try:
    import hvac
except ImportError:
    hvac = None

from .logging import get_logger

logger = get_logger(__name__)


class VaultClient:
    """HashiCorp Vault client for secure secret management"""
    
    def __init__(self):
        """Initialize Vault client"""
        if not hvac:
            raise ImportError("hvac library required for Vault integration")
        
        self.vault_url = os.getenv('VAULT_URL', 'http://localhost:8200')
        self.client = hvac.Client(url=self.vault_url)
        self._authenticated = False
        self._auth_time = 0
        self._token_ttl = 3600  # 1 hour default
        
        # Authenticate on initialization
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Vault using AppRole"""
        try:
            # Try AppRole authentication first
            role_id = os.getenv('VAULT_ROLE_ID')
            secret_id = os.getenv('VAULT_SECRET_ID')
            
            if role_id and secret_id:
                auth_response = self.client.auth.approle.login(
                    role_id=role_id,
                    secret_id=secret_id
                )
                
                self._token_ttl = auth_response['auth']['lease_duration']
                self._authenticated = True
                self._auth_time = time.time()
                logger.info("Vault authentication successful via AppRole")
                return
            
            # Fallback to token authentication
            token = os.getenv('VAULT_TOKEN')
            if token:
                self.client.token = token
                
                # Verify token is valid
                if self.client.is_authenticated():
                    self._authenticated = True
                    self._auth_time = time.time()
                    logger.info("Vault authentication successful via token")
                    return
            
            raise Exception("No valid Vault authentication method found")
            
        except Exception as e:
            logger.error(f"Vault authentication failed: {e}")
            self._authenticated = False
            raise
    
    def is_authenticated(self) -> bool:
        """Check if client is authenticated and token is still valid"""
        if not self._authenticated:
            return False
        
        # Check if token has expired (with 5-minute buffer)
        if time.time() - self._auth_time > (self._token_ttl - 300):
            try:
                self._authenticate()
            except:
                return False
        
        return self.client.is_authenticated()
    
    def get_secret(self, path: str) -> Optional[Dict[str, Any]]:
        """Retrieve secret from Vault KV store"""
        if not self.is_authenticated():
            raise Exception("Vault client not authenticated")
        
        try:
            response = self.client.secrets.kv.v2.read_secret_version(path=path)
            return response['data']['data']
        except hvac.exceptions.InvalidPath:
            logger.warning(f"Secret not found at path: {path}")
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve secret from {path}: {e}")
            raise
    
    def store_secret(self, path: str, secret_data: Dict[str, Any]):
        """Store secret in Vault KV store"""
        if not self.is_authenticated():
            raise Exception("Vault client not authenticated")
        
        try:
            self.client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret=secret_data
            )
            logger.info(f"Secret stored successfully at path: {path}")
        except Exception as e:
            logger.error(f"Failed to store secret at {path}: {e}")
            raise