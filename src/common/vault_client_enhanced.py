"""
Enhanced Vault client with emergency secret management
Production-ready implementation with comprehensive fallback
"""

import os
import logging
from typing import Dict, Optional, Any
from pathlib import Path
import secrets
import base64

try:
    import hvac
    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False

logger = logging.getLogger(__name__)

class VaultSecretManager:
    """Production-ready secret management with Vault integration"""
    
    def __init__(self):
        self.vault_url = os.getenv('VAULT_URL', 'http://localhost:8200')
        self.vault_token = os.getenv('VAULT_TOKEN')
        self.client = None
        self.fallback_mode = False
        self._init_client()
    
    def _init_client(self):
        """Initialize Vault client with graceful fallback"""
        if not VAULT_AVAILABLE:
            logger.warning("⚠️ hvac library not available, using fallback mode")
            self.fallback_mode = True
            return
        
        if not self.vault_token:
            logger.warning("⚠️ VAULT_TOKEN not provided, using fallback mode")
            self.fallback_mode = True
            return
        
        try:
            self.client = hvac.Client(url=self.vault_url, token=self.vault_token)
            if self.client.is_authenticated():
                logger.info("✅ Vault client authenticated successfully")
                self.fallback_mode = False
            else:
                logger.warning("⚠️ Vault authentication failed, using fallback mode")
                self.fallback_mode = True
                self.client = None
        except Exception as e:
            logger.warning(f"⚠️ Vault connection failed: {e}, using fallback mode")
            self.fallback_mode = True
            self.client = None
    
    def get_secret(self, path: str, key: str) -> Optional[str]:
        """Get secret with comprehensive fallback chain"""
        
        # 1. Try Vault first
        if not self.fallback_mode and self.client:
            try:
                response = self.client.secrets.kv.v2.read_secret_version(path=path)
                secret_data = response['data']['data']
                value = secret_data.get(key)
                if value:
                    logger.info(f"✅ Retrieved secret {key} from Vault")
                    return value
            except Exception as e:
                logger.warning(f"⚠️ Vault secret retrieval failed: {e}")
        
        # 2. Try file-based secrets (Docker secrets)
        secret_file = f"/run/secrets/{key.lower()}"
        if Path(secret_file).exists():
            try:
                with open(secret_file, 'r') as f:
                    value = f.read().strip()
                logger.info(f"✅ Retrieved secret {key} from file")
                return value
            except Exception as e:
                logger.error(f"❌ File secret retrieval failed: {e}")
        
        # 3. Try environment file
        env_file = os.getenv(f"{key.upper()}_FILE")
        if env_file and Path(env_file).exists():
            try:
                with open(env_file, 'r') as f:
                    value = f.read().strip()
                logger.info(f"✅ Retrieved secret {key} from env file")
                return value
            except Exception as e:
                logger.error(f"❌ Env file secret retrieval failed: {e}")
        
        # 4. Final fallback to environment variable (NOT RECOMMENDED)
        env_value = os.getenv(key.upper())
        if env_value:
            logger.warning(f"⚠️ Using environment variable for {key} (SECURITY RISK)")
            return env_value
        
        logger.error(f"❌ Secret {key} not found in any source")
        return None
    
    def validate_secret(self, key: str, value: str) -> bool:
        """Validate secret meets security requirements"""
        if not value:
            return False
        
        # General requirements
        if len(value) < 16:
            logger.error(f"Secret {key} too short (minimum 16 characters)")
            return False
        
        # Specific validations
        if key.lower() == 'jwt_secret':
            if len(value) < 32:
                logger.error("JWT secret must be at least 32 characters")
                return False
        
        # Check for common weak values
        weak_patterns = [
            'password', 'secret', 'key', 'admin', 'test', 'dev',
            '123456', 'change-me', 'default', 'sample'
        ]
        
        if any(pattern in value.lower() for pattern in weak_patterns):
            logger.error(f"Secret {key} contains weak pattern")
            return False
        
        return True
    
    def generate_secure_secret(self, key: str, length: int = 32) -> str:
        """Generate cryptographically secure secret"""
        secret = secrets.token_urlsafe(length)
        logger.info(f"Generated secure secret for {key}")
        return secret
    
    def health_check(self) -> Dict[str, Any]:
        """Check secret management system health"""
        status = {
            "vault_available": VAULT_AVAILABLE,
            "vault_connected": False,
            "fallback_mode": self.fallback_mode,
            "sources_available": []
        }
        
        if self.client:
            try:
                status["vault_connected"] = self.client.is_authenticated()
            except:
                pass
        
        # Check available secret sources
        if status["vault_connected"]:
            status["sources_available"].append("vault")
        
        if Path("/run/secrets").exists():
            status["sources_available"].append("docker_secrets")
        
        status["sources_available"].append("environment")
        
        return status

# Global instance
_secret_manager = None

def get_secret_manager() -> VaultSecretManager:
    """Get global secret manager instance"""
    global _secret_manager
    if _secret_manager is None:
        _secret_manager = VaultSecretManager()
    return _secret_manager