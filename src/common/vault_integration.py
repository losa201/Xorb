#!/usr/bin/env python3
"""
Enhanced Vault Integration for XORB Platform
Secure secret management with automatic rotation and fallback
"""

import os
import asyncio
import logging
import json
import aiohttp
import base64
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class VaultConfig:
    """Vault configuration settings"""
    vault_addr: str
    vault_namespace: str = "xorb"
    vault_role: str = "xorb-api"
    auth_method: str = "approle"  # approle, kubernetes, jwt
    mount_point: str = "secret"
    transit_mount: str = "transit"
    max_retries: int = 3
    timeout: int = 30
    token_renew_threshold: float = 0.2  # Renew when 20% TTL remaining


class VaultError(Exception):
    """Vault operation error"""
    pass


class VaultClient:
    """Production-ready Vault client with security best practices"""
    
    def __init__(self, config: VaultConfig):
        self.config = config
        self.token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.authenticated = False
        
        # Create session with security headers
        self._session_config = {
            "timeout": aiohttp.ClientTimeout(total=config.timeout),
            "headers": {
                "X-Vault-Namespace": config.vault_namespace,
                "Content-Type": "application/json"
            },
            "connector": aiohttp.TCPConnector(
                ssl=True,
                limit=10,
                limit_per_host=5
            )
        }
    
    async def initialize(self):
        """Initialize Vault client and authenticate"""
        try:
            self.session = aiohttp.ClientSession(**self._session_config)
            
            # Authenticate with Vault
            await self._authenticate()
            
            logger.info("Vault client initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize Vault client", error=str(e))
            raise VaultError(f"Vault initialization failed: {e}")
    
    async def _authenticate(self):
        """Authenticate with Vault using configured method"""
        if self.config.auth_method == "approle":
            await self._authenticate_approle()
        elif self.config.auth_method == "kubernetes":
            await self._authenticate_kubernetes()
        elif self.config.auth_method == "jwt":
            await self._authenticate_jwt()
        else:
            raise VaultError(f"Unsupported auth method: {self.config.auth_method}")
    
    async def _authenticate_approle(self):
        """Authenticate using AppRole method"""
        role_id = os.getenv("VAULT_ROLE_ID")
        secret_id = os.getenv("VAULT_SECRET_ID")
        
        if not role_id or not secret_id:
            raise VaultError("VAULT_ROLE_ID and VAULT_SECRET_ID must be set for AppRole auth")
        
        auth_data = {
            "role_id": role_id,
            "secret_id": secret_id
        }
        
        url = f"{self.config.vault_addr}/v1/auth/approle/login"
        
        async with self.session.post(url, json=auth_data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise VaultError(f"AppRole authentication failed: {error_text}")
            
            auth_response = await response.json()
            auth_data = auth_response.get("auth", {})
            
            self.token = auth_data.get("client_token")
            lease_duration = auth_data.get("lease_duration", 3600)
            
            self.token_expires_at = datetime.utcnow() + timedelta(seconds=lease_duration)
            self.session.headers["X-Vault-Token"] = self.token
            self.authenticated = True
            
            logger.info("Successfully authenticated with Vault using AppRole")
    
    async def _authenticate_kubernetes(self):
        """Authenticate using Kubernetes service account"""
        try:
            # Read Kubernetes service account token
            token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
            with open(token_path, 'r') as f:
                jwt_token = f.read().strip()
            
            auth_data = {
                "jwt": jwt_token,
                "role": self.config.vault_role
            }
            
            url = f"{self.config.vault_addr}/v1/auth/kubernetes/login"
            
            async with self.session.post(url, json=auth_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise VaultError(f"Kubernetes authentication failed: {error_text}")
                
                auth_response = await response.json()
                auth_data = auth_response.get("auth", {})
                
                self.token = auth_data.get("client_token")
                lease_duration = auth_data.get("lease_duration", 3600)
                
                self.token_expires_at = datetime.utcnow() + timedelta(seconds=lease_duration)
                self.session.headers["X-Vault-Token"] = self.token
                self.authenticated = True
                
                logger.info("Successfully authenticated with Vault using Kubernetes")
                
        except FileNotFoundError:
            raise VaultError("Kubernetes service account token not found")
        except Exception as e:
            raise VaultError(f"Kubernetes authentication failed: {e}")
    
    async def get_secret(self, secret_path: str, version: Optional[int] = None) -> Dict[str, Any]:
        """Get secret from Vault with automatic token renewal"""
        await self._ensure_authenticated()
        
        # Build URL for KV v2
        url = f"{self.config.vault_addr}/v1/{self.config.mount_point}/data/{secret_path}"
        
        params = {}
        if version:
            params["version"] = version
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("data", {}).get("data", {})
                    elif response.status == 404:
                        raise VaultError(f"Secret not found: {secret_path}")
                    elif response.status == 403:
                        # Try to re-authenticate
                        await self._authenticate()
                        continue
                    else:
                        error_text = await response.text()
                        raise VaultError(f"Failed to get secret: {error_text}")
                        
            except aiohttp.ClientError as e:
                if attempt == self.config.max_retries - 1:
                    raise VaultError(f"Failed to get secret after {self.config.max_retries} attempts: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise VaultError(f"Failed to get secret: {secret_path}")
    
    async def put_secret(self, secret_path: str, secret_data: Dict[str, Any]) -> bool:
        """Store secret in Vault"""
        await self._ensure_authenticated()
        
        url = f"{self.config.vault_addr}/v1/{self.config.mount_point}/data/{secret_path}"
        
        payload = {
            "data": secret_data,
            "options": {
                "cas": 0  # Check-and-set
            }
        }
        
        async with self.session.post(url, json=payload) as response:
            if response.status in [200, 204]:
                logger.info("Successfully stored secret", path=secret_path)
                return True
            else:
                error_text = await response.text()
                raise VaultError(f"Failed to store secret: {error_text}")
    
    async def encrypt_data(self, key_name: str, plaintext: str) -> str:
        """Encrypt data using Vault transit engine"""
        await self._ensure_authenticated()
        
        url = f"{self.config.vault_addr}/v1/{self.config.transit_mount}/encrypt/{key_name}"
        
        # Base64 encode the plaintext
        encoded_plaintext = base64.b64encode(plaintext.encode()).decode()
        
        payload = {
            "plaintext": encoded_plaintext
        }
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("data", {}).get("ciphertext", "")
            else:
                error_text = await response.text()
                raise VaultError(f"Failed to encrypt data: {error_text}")
    
    async def decrypt_data(self, key_name: str, ciphertext: str) -> str:
        """Decrypt data using Vault transit engine"""
        await self._ensure_authenticated()
        
        url = f"{self.config.vault_addr}/v1/{self.config.transit_mount}/decrypt/{key_name}"
        
        payload = {
            "ciphertext": ciphertext
        }
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                encoded_plaintext = result.get("data", {}).get("plaintext", "")
                return base64.b64decode(encoded_plaintext).decode()
            else:
                error_text = await response.text()
                raise VaultError(f"Failed to decrypt data: {error_text}")
    
    async def rotate_key(self, key_name: str) -> bool:
        """Rotate encryption key in transit engine"""
        await self._ensure_authenticated()
        
        url = f"{self.config.vault_addr}/v1/{self.config.transit_mount}/keys/{key_name}/rotate"
        
        async with self.session.post(url) as response:
            if response.status in [200, 204]:
                logger.info("Successfully rotated key", key_name=key_name)
                return True
            else:
                error_text = await response.text()
                raise VaultError(f"Failed to rotate key: {error_text}")
    
    async def _ensure_authenticated(self):
        """Ensure client is authenticated and token is valid"""
        if not self.authenticated:
            await self._authenticate()
            return
        
        # Check if token needs renewal
        if self.token_expires_at:
            time_until_expiry = (self.token_expires_at - datetime.utcnow()).total_seconds()
            if time_until_expiry < (3600 * self.config.token_renew_threshold):
                await self._renew_token()
    
    async def _renew_token(self):
        """Renew Vault token"""
        url = f"{self.config.vault_addr}/v1/auth/token/renew-self"
        
        async with self.session.post(url) as response:
            if response.status == 200:
                result = await response.json()
                auth_data = result.get("auth", {})
                lease_duration = auth_data.get("lease_duration", 3600)
                self.token_expires_at = datetime.utcnow() + timedelta(seconds=lease_duration)
                logger.info("Successfully renewed Vault token")
            else:
                # Token renewal failed, re-authenticate
                await self._authenticate()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Vault health and connectivity"""
        try:
            url = f"{self.config.vault_addr}/v1/sys/health"
            
            async with self.session.get(url) as response:
                health_data = await response.json()
                
                return {
                    "status": "healthy" if response.status == 200 else "unhealthy",
                    "authenticated": self.authenticated,
                    "token_expires_at": self.token_expires_at.isoformat() if self.token_expires_at else None,
                    "vault_status": health_data
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def close(self):
        """Close Vault client and cleanup"""
        if self.session:
            await self.session.close()
        logger.info("Vault client closed")


class SecretManager:
    """High-level secret management with caching and fallback"""
    
    def __init__(self, vault_client: VaultClient):
        self.vault_client = vault_client
        self.secret_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = timedelta(minutes=15)
    
    async def get_secret(self, secret_path: str, key: Optional[str] = None, 
                        fallback_env: Optional[str] = None) -> Union[str, Dict[str, Any]]:
        """Get secret with caching and environment variable fallback"""
        
        # Check cache first
        cache_key = f"{secret_path}:{key}" if key else secret_path
        cached_secret = self.secret_cache.get(cache_key)
        
        if cached_secret and self._is_cache_valid(cached_secret):
            secret_data = cached_secret["data"]
            return secret_data.get(key) if key else secret_data
        
        try:
            # Try to get from Vault
            secret_data = await self.vault_client.get_secret(secret_path)
            
            # Cache the secret
            self.secret_cache[cache_key] = {
                "data": secret_data,
                "cached_at": datetime.utcnow()
            }
            
            result = secret_data.get(key) if key else secret_data
            
            if result:
                return result
                
        except VaultError as e:
            logger.warning("Failed to get secret from Vault", error=str(e), path=secret_path)
        
        # Fallback to environment variable
        if fallback_env:
            env_value = os.getenv(fallback_env)
            if env_value:
                logger.info("Using environment variable fallback", env_var=fallback_env)
                return env_value
        
        raise VaultError(f"Secret not found: {secret_path}/{key}")
    
    def _is_cache_valid(self, cached_secret: Dict[str, Any]) -> bool:
        """Check if cached secret is still valid"""
        cached_at = cached_secret.get("cached_at")
        if not cached_at:
            return False
        
        return (datetime.utcnow() - cached_at) < self.cache_ttl
    
    async def clear_cache(self):
        """Clear secret cache"""
        self.secret_cache.clear()
        logger.info("Secret cache cleared")


# Global instances
_vault_client: Optional[VaultClient] = None
_secret_manager: Optional[SecretManager] = None


async def get_vault_client() -> VaultClient:
    """Get global Vault client instance"""
    global _vault_client
    
    if _vault_client is None:
        vault_config = VaultConfig(
            vault_addr=os.getenv("VAULT_ADDR", "https://vault.xorb.internal:8200"),
            vault_namespace=os.getenv("VAULT_NAMESPACE", "xorb"),
            vault_role=os.getenv("VAULT_ROLE", "xorb-api"),
            auth_method=os.getenv("VAULT_AUTH_METHOD", "approle")
        )
        
        _vault_client = VaultClient(vault_config)
        await _vault_client.initialize()
    
    return _vault_client


async def get_secret_manager() -> SecretManager:
    """Get global secret manager instance"""
    global _secret_manager
    
    if _secret_manager is None:
        vault_client = await get_vault_client()
        _secret_manager = SecretManager(vault_client)
    
    return _secret_manager


async def get_secret(secret_path: str, key: Optional[str] = None, 
                    fallback_env: Optional[str] = None) -> Union[str, Dict[str, Any]]:
    """Convenience function to get secret"""
    secret_manager = await get_secret_manager()
    return await secret_manager.get_secret(secret_path, key, fallback_env)


# Environment-aware secret loading
def load_secrets_from_vault_or_env() -> Dict[str, str]:
    """Load secrets from Vault with environment variable fallbacks"""
    secrets = {}
    
    # Define secret mappings
    secret_mappings = {
        "DATABASE_URL": ("xorb/database", "url", "DATABASE_URL"),
        "REDIS_URL": ("xorb/redis", "url", "REDIS_URL"),
        "JWT_SECRET": ("xorb/security", "jwt_secret", "JWT_SECRET"),
        "NVIDIA_API_KEY": ("xorb/external", "nvidia_api_key", "NVIDIA_API_KEY"),
        "OPENROUTER_API_KEY": ("xorb/external", "openrouter_api_key", "OPENROUTER_API_KEY"),
    }
    
    for env_key, (vault_path, vault_key, fallback_env) in secret_mappings.items():
        try:
            # This would be called in an async context in real usage
            # For now, return environment variable
            secrets[env_key] = os.getenv(fallback_env, "")
        except Exception as e:
            logger.error("Failed to load secret", secret=env_key, error=str(e))
            secrets[env_key] = os.getenv(fallback_env, "")
    
    return secrets