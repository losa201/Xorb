"""
HashiCorp Vault integration for secure secrets management
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

try:
    import hvac
    HVAC_AVAILABLE = True
except ImportError:
    HVAC_AVAILABLE = False
    hvac = None

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None


@dataclass
class DatabaseCredentials:
    """Database credentials from Vault"""
    username: str
    password: str
    expires_at: datetime

    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string"""
        host = os.getenv("POSTGRES_HOST", "postgres")
        port = os.getenv("POSTGRES_PORT", "5432")
        database = os.getenv("POSTGRES_DATABASE", "xorb_db")
        return f"postgresql://{self.username}:{self.password}@{host}:{port}/{database}"

    @property
    def is_expired(self) -> bool:
        """Check if credentials are expired"""
        return datetime.utcnow() >= self.expires_at


class VaultClient:
    """Production-ready Vault client with caching and error handling"""

    def __init__(
        self,
        vault_url: str = None,
        vault_token: str = None,
        vault_role: str = "xorb-app"
    ):
        self.vault_url = vault_url or os.getenv("VAULT_ADDR", "http://vault:8200")
        self.vault_token = vault_token or os.getenv("VAULT_TOKEN")
        self.vault_role = vault_role
        self.client = None
        self._db_credentials: Optional[DatabaseCredentials] = None
        self._secrets_cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, datetime] = {}

        if not HVAC_AVAILABLE:
            raise ImportError("hvac library required for Vault integration")

    async def initialize(self):
        """Initialize Vault client with AppRole authentication support"""
        try:
            self.client = hvac.Client(url=self.vault_url)

            # Try AppRole authentication first, then token
            if await self._try_approle_auth():
                print(f"Vault client initialized with AppRole: {self.vault_url}")
            elif self.vault_token:
                self.client.token = self.vault_token
                if not self.client.is_authenticated():
                    raise Exception("Vault token authentication failed")
                print(f"Vault client initialized with token: {self.vault_url}")
            else:
                raise Exception("No valid Vault authentication method available")

        except Exception as e:
            print(f"Failed to initialize Vault client: {e}")
            # Fall back to environment variables in development
            if os.getenv("ENVIRONMENT", "development") == "development":
                print("⚠️  Falling back to environment variables for development")
                self.client = None
            else:
                raise

    async def _try_approle_auth(self) -> bool:
        """Attempt AppRole authentication"""
        role_id = os.getenv("VAULT_ROLE_ID")
        secret_id = os.getenv("VAULT_SECRET_ID")

        if not role_id or not secret_id:
            return False

        try:
            response = self.client.auth.approle.login(
                role_id=role_id,
                secret_id=secret_id
            )

            if response and response.get("auth", {}).get("client_token"):
                self.client.token = response["auth"]["client_token"]
                return True

        except Exception as e:
            print(f"AppRole authentication failed: {e}")

        return False

    async def get_secret(self, path: str, cache_ttl: int = 300) -> Dict[str, Any]:
        """Get secret from Vault with caching"""
        cache_key = f"secret:{path}"

        # Check cache
        if (cache_key in self._secrets_cache and
            cache_key in self._cache_ttl and
            datetime.utcnow() < self._cache_ttl[cache_key]):
            return self._secrets_cache[cache_key]

        try:
            if self.client:
                # Vault KV v2 path format
                vault_path = f"secret/data/{path}"
                response = self.client.secrets.kv.v2.read_secret_version(path=path)
                secret_data = response["data"]["data"]
            else:
                # Fallback to environment variables
                secret_data = self._get_env_fallback(path)

            # Cache the result
            self._secrets_cache[cache_key] = secret_data
            self._cache_ttl[cache_key] = datetime.utcnow() + timedelta(seconds=cache_ttl)

            return secret_data

        except Exception as e:
            print(f"Failed to get secret {path}: {e}")
            # Return environment fallback
            return self._get_env_fallback(path)

    def _get_env_fallback(self, path: str) -> Dict[str, Any]:
        """Fallback to environment variables when Vault is unavailable"""
        fallback_map = {
            "xorb/config": {
                "JWT_SECRET": os.getenv("JWT_SECRET", "dev-secret-change-in-production"),
                "XORB_API_KEY": os.getenv("XORB_API_KEY", "dev-api-key"),
                "DB_PASSWORD": os.getenv("DB_PASSWORD", "password"),
                "DB_USER": os.getenv("DB_USER", "xorb_user"),
                "DB_NAME": os.getenv("DB_NAME", "xorb_secure"),
                "DB_HOST": os.getenv("DB_HOST", "localhost"),
                "DB_PORT": os.getenv("DB_PORT", "5432"),
                "REDIS_PASSWORD": os.getenv("REDIS_PASSWORD", "")
            },
            "xorb/external": {
                "NVIDIA_API_KEY": os.getenv("NVIDIA_API_KEY", ""),
                "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
                "AZURE_CLIENT_SECRET": os.getenv("AZURE_CLIENT_SECRET", ""),
                "GOOGLE_CLIENT_SECRET": os.getenv("GOOGLE_CLIENT_SECRET", ""),
                "GITHUB_CLIENT_SECRET": os.getenv("GITHUB_CLIENT_SECRET", "")
            }
        }

        return fallback_map.get(path, {})

    async def get_database_credentials(self, force_refresh: bool = False) -> DatabaseCredentials:
        """Get dynamic database credentials from Vault"""
        if (self._db_credentials and
            not self._db_credentials.is_expired and
            not force_refresh):
            return self._db_credentials

        try:
            if self.client:
                # Get dynamic database credentials
                response = self.client.secrets.database.generate_credentials(name="xorb-app")

                credentials = DatabaseCredentials(
                    username=response["data"]["username"],
                    password=response["data"]["password"],
                    expires_at=datetime.utcnow() + timedelta(hours=1)  # Default 1h TTL
                )
            else:
                # Fallback to static credentials
                credentials = DatabaseCredentials(
                    username=os.getenv("POSTGRES_USER", "xorb_user"),
                    password=os.getenv("POSTGRES_PASSWORD", "password"),
                    expires_at=datetime.utcnow() + timedelta(days=1)  # Static creds don't expire
                )

            self._db_credentials = credentials
            return credentials

        except Exception as e:
            print(f"Failed to get database credentials: {e}")
            # Return static fallback
            return DatabaseCredentials(
                username=os.getenv("POSTGRES_USER", "xorb_user"),
                password=os.getenv("POSTGRES_PASSWORD", "password"),
                expires_at=datetime.utcnow() + timedelta(days=1)
            )

    async def encrypt_data(self, plaintext: str, key_name: str = "jwt-signing") -> str:
        """Encrypt data using Vault transit engine"""
        try:
            if self.client:
                response = self.client.secrets.transit.encrypt_data(
                    name=key_name,
                    plaintext=plaintext
                )
                return response["data"]["ciphertext"]
            else:
                # Fallback: return base64 encoded (not secure, for dev only)
                import base64
                return base64.b64encode(plaintext.encode()).decode()

        except Exception as e:
            print(f"Failed to encrypt data: {e}")
            import base64
            return base64.b64encode(plaintext.encode()).decode()

    async def decrypt_data(self, ciphertext: str, key_name: str = "jwt-signing") -> str:
        """Decrypt data using Vault transit engine"""
        try:
            if self.client:
                response = self.client.secrets.transit.decrypt_data(
                    name=key_name,
                    ciphertext=ciphertext
                )
                return response["data"]["plaintext"]
            else:
                # Fallback: base64 decode (not secure, for dev only)
                import base64
                return base64.b64decode(ciphertext).decode()

        except Exception as e:
            print(f"Failed to decrypt data: {e}")
            import base64
            return base64.b64decode(ciphertext).decode()

    async def sign_jwt_payload(self, payload: Dict[str, Any], key_name: str = "jwt-signing") -> str:
        """Sign JWT payload using Vault transit engine"""
        try:
            if self.client:
                # Convert payload to base64
                import base64
                import json
                payload_b64 = base64.b64encode(json.dumps(payload).encode()).decode()

                response = self.client.secrets.transit.sign_data(
                    name=key_name,
                    hash_input=payload_b64
                )
                return response["data"]["signature"]
            else:
                # Fallback to local JWT signing
                return await self._fallback_jwt_sign(payload)

        except Exception as e:
            print(f"Failed to sign JWT: {e}")
            return await self._fallback_jwt_sign(payload)

    async def _fallback_jwt_sign(self, payload: Dict[str, Any]) -> str:
        """Fallback JWT signing for development"""
        try:
            import jwt
            secret = os.getenv("JWT_SECRET", "dev-secret")
            return jwt.encode(payload, secret, algorithm="HS256")
        except ImportError:
            # Return a simple token for testing
            import base64
            import json
            return base64.b64encode(json.dumps(payload).encode()).decode()

    async def refresh_credentials(self):
        """Refresh all cached credentials"""
        self._secrets_cache.clear()
        self._cache_ttl.clear()
        await self.get_database_credentials(force_refresh=True)

    async def health_check(self) -> Dict[str, Any]:
        """Check Vault health and connection status"""
        try:
            if self.client:
                health = self.client.sys.read_health_status()
                return {
                    "status": "healthy" if health else "unhealthy",
                    "vault_url": self.vault_url,
                    "authenticated": self.client.is_authenticated(),
                    "fallback_mode": False
                }
            else:
                return {
                    "status": "fallback",
                    "vault_url": self.vault_url,
                    "authenticated": False,
                    "fallback_mode": True,
                    "message": "Using environment variables"
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "fallback_mode": True
            }

    async def rotate_jwt_key(self) -> bool:
        """Rotate JWT signing key in Vault transit engine"""
        try:
            if self.client:
                self.client.secrets.transit.rotate_key(name="jwt-signing")
                print("✅ JWT signing key rotated successfully")
                return True
            else:
                print("⚠️  Vault not available, skipping key rotation")
                return False

        except Exception as e:
            print(f"❌ JWT key rotation failed: {e}")
            return False

    async def get_secret_version(self, path: str, version: int = None) -> Dict[str, Any]:
        """Get specific version of a secret from Vault KV v2"""
        try:
            if self.client:
                response = self.client.secrets.kv.v2.read_secret_version(
                    path=path,
                    version=version
                )
                return response["data"]["data"]
            else:
                return self._get_env_fallback(path)

        except Exception as e:
            print(f"Failed to get secret version {path}@{version}: {e}")
            return self._get_env_fallback(path)

    async def list_secrets(self, path: str = "") -> Dict[str, Any]:
        """List available secrets at a given path"""
        try:
            if self.client:
                response = self.client.secrets.kv.v2.list_secrets(path=path or "xorb")
                return response["data"]
            else:
                return {"keys": ["config", "external"]}

        except Exception as e:
            print(f"Failed to list secrets: {e}")
            return {"keys": []}

    async def backup_secrets(self, output_file: str) -> bool:
        """Export all secrets for backup (development only)"""
        if os.getenv("ENVIRONMENT", "development") != "development":
            print("❌ Secret backup only allowed in development environment")
            return False

        try:
            secrets_backup = {}

            # Get main XORB config secrets
            config_secrets = await self.get_secret("xorb/config")
            if config_secrets:
                secrets_backup["xorb/config"] = config_secrets

            # Get external API secrets
            external_secrets = await self.get_secret("xorb/external")
            if external_secrets:
                secrets_backup["xorb/external"] = external_secrets

            # Write to file with timestamp
            backup_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "environment": "development",
                "secrets": secrets_backup
            }

            with open(output_file, 'w') as f:
                json.dump(backup_data, f, indent=2)

            print(f"✅ Secrets backed up to {output_file}")
            return True

        except Exception as e:
            print(f"❌ Secret backup failed: {e}")
            return False


# Global Vault client instance
vault_client = VaultClient()


async def get_vault_client() -> VaultClient:
    """Dependency injection for Vault client"""
    if vault_client.client is None:
        await vault_client.initialize()
    return vault_client


# Convenience functions for common operations - Updated for existing infrastructure
async def get_jwt_secret() -> str:
    """Get JWT signing secret from Vault KV store"""
    client = await get_vault_client()
    secrets = await client.get_secret("xorb/config")
    return secrets.get("JWT_SECRET", os.getenv("JWT_SECRET", "dev-secret"))


async def get_database_url() -> str:
    """Get database connection URL with dynamic credentials"""
    client = await get_vault_client()
    credentials = await client.get_database_credentials()
    return credentials.connection_string


async def get_database_config() -> Dict[str, Any]:
    """Get database configuration from Vault secrets"""
    client = await get_vault_client()
    secrets = await client.get_secret("xorb/config")

    return {
        "host": secrets.get("DB_HOST", os.getenv("DB_HOST", "localhost")),
        "port": int(secrets.get("DB_PORT", os.getenv("DB_PORT", "5432"))),
        "database": secrets.get("DB_NAME", os.getenv("DB_NAME", "xorb_secure")),
        "username": secrets.get("DB_USER", os.getenv("DB_USER", "xorb_user")),
        "password": secrets.get("DB_PASSWORD", os.getenv("DB_PASSWORD", "password"))
    }


async def get_redis_config() -> Dict[str, Any]:
    """Get Redis configuration from Vault or environment"""
    client = await get_vault_client()
    secrets = await client.get_secret("xorb/config")

    return {
        "url": os.getenv("REDIS_URL", "redis://redis:6379/0"),
        "password": secrets.get("REDIS_PASSWORD", ""),
        "max_connections": 20,
        "retry_on_timeout": True
    }


async def get_external_api_key(service: str) -> str:
    """Get external API key (NVIDIA, OpenRouter, etc.)"""
    client = await get_vault_client()
    secrets = await client.get_secret("xorb/external")

    # Map service names to secret keys
    service_key_map = {
        "nvidia": "NVIDIA_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "azure": "AZURE_CLIENT_SECRET",
        "google": "GOOGLE_CLIENT_SECRET",
        "github": "GITHUB_CLIENT_SECRET"
    }

    key = service_key_map.get(service.lower(), f"{service.upper()}_API_KEY")
    return secrets.get(key, os.getenv(key, ""))


async def get_xorb_api_key() -> str:
    """Get XORB internal API key for service-to-service communication"""
    client = await get_vault_client()
    secrets = await client.get_secret("xorb/config")
    return secrets.get("XORB_API_KEY", os.getenv("XORB_API_KEY", "dev-api-key"))
