"""
Secure secret management using HashiCorp Vault or environment variables
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class SecretManager:
    """
    Centralized secret management with multiple backends:
    1. HashiCorp Vault (production)
    2. Environment variables (development)
    3. Secure file (testing)
    """

    def __init__(self, environment: str = "development"):
        self.environment = environment
        self._vault_client = None
        self._secrets_cache = {}

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret by key with fallback mechanism"""

        # Try cache first
        if key in self._secrets_cache:
            return self._secrets_cache[key]

        # Try Vault in production
        if self.environment == "production":
            secret = self._get_from_vault(key)
            if secret:
                self._secrets_cache[key] = secret
                return secret

        # Fallback to environment variables
        secret = os.getenv(key, default)
        if secret:
            self._secrets_cache[key] = secret
            return secret

        logger.warning(f"Secret '{key}' not found in any backend")
        return default

    def get_database_url(self) -> str:
        """Get database URL with secure credential handling"""
        if self.environment == "production":
            # In production, use individual components
            db_host = self.get_secret("DB_HOST", "localhost")
            db_port = self.get_secret("DB_PORT", "5432")
            db_name = self.get_secret("DB_NAME", "xorb")
            db_user = self.get_secret("DB_USER")
            db_password = self.get_secret("DB_PASSWORD")

            if not db_user or not db_password:
                raise ValueError("Database credentials not found in secret store")

            return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        else:
            # Development fallback
            return self.get_secret("DATABASE_URL", "postgresql://xorb:changeme@localhost:5432/xorb")

    def get_jwt_secret(self) -> str:
        """Get JWT secret with validation"""
        secret = self.get_secret("JWT_SECRET")
        if not secret:
            if self.environment == "production":
                raise ValueError("JWT_SECRET is required in production")
            # Generate a temporary secret for development
            import secrets
            secret = secrets.token_urlsafe(64)
            logger.warning("Using temporary JWT secret for development")
        return secret

    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for external service"""
        key_name = f"{service.upper()}_API_KEY"
        return self.get_secret(key_name)

    def _get_from_vault(self, key: str) -> Optional[str]:
        """Get secret from HashiCorp Vault"""
        try:
            if not self._vault_client:
                self._init_vault_client()

            # Read from Vault KV store
            response = self._vault_client.secrets.kv.v2.read_secret_version(
                path="xorb/config",
                mount_point="secret"
            )

            secrets_data = response.get("data", {}).get("data", {})
            return secrets_data.get(key)

        except Exception as e:
            logger.error(f"Error reading from Vault: {e}")
            return None

    def _init_vault_client(self):
        """Initialize Vault client"""
        try:
            import hvac

            vault_url = os.getenv("VAULT_URL", "http://localhost:8200")
            vault_token = os.getenv("VAULT_TOKEN")

            if not vault_token:
                # Try to read from file
                token_file = Path("~/.vault-token").expanduser()
                if token_file.exists():
                    vault_token = token_file.read_text().strip()

            if not vault_token:
                raise ValueError("Vault token not found")

            self._vault_client = hvac.Client(url=vault_url, token=vault_token)

            if not self._vault_client.is_authenticated():
                raise ValueError("Vault authentication failed")

        except ImportError:
            logger.error("hvac library not installed - install with: pip install hvac")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Vault client: {e}")
            raise


# Global instance
secret_manager = SecretManager(environment=os.getenv("ENVIRONMENT", "development"))


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Convenience function to get secrets"""
    return secret_manager.get_secret(key, default)


def get_database_url() -> str:
    """Get secure database URL"""
    return secret_manager.get_database_url()


def get_jwt_secret() -> str:
    """Get JWT secret"""
    return secret_manager.get_jwt_secret()
