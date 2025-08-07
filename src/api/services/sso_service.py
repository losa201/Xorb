"""
Enterprise SSO Service
Handles integration with major identity providers (Azure AD, Google, Okta, SAML)
"""

import json
import secrets
import hashlib
import base64
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from urllib.parse import urlencode, quote
import httpx
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import aioredis

from src.common.models.auth import SSOProvider, SSOConfig, User
from src.common.config import get_settings


class SSOService:
    """Enterprise SSO integration service"""
    
    def __init__(self, redis_client: aioredis.Redis, config: Dict[str, Any]):
        self.redis = redis_client
        self.config = config
        self.settings = get_settings()
        
        # SSO Provider configurations
        self.providers = {
            "azure": {
                "name": "Microsoft Azure AD",
                "auth_endpoint": "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize",
                "token_endpoint": "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
                "userinfo_endpoint": "https://graph.microsoft.com/v1.0/me",
                "scope": "openid email profile User.Read",
                "response_type": "code",
                "grant_type": "authorization_code"
            },
            "google": {
                "name": "Google Workspace",
                "auth_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
                "token_endpoint": "https://oauth2.googleapis.com/token",
                "userinfo_endpoint": "https://www.googleapis.com/oauth2/v1/userinfo",
                "scope": "openid email profile",
                "response_type": "code",
                "grant_type": "authorization_code"
            },
            "okta": {
                "name": "Okta",
                "auth_endpoint": "https://{domain}/oauth2/v1/authorize",
                "token_endpoint": "https://{domain}/oauth2/v1/token",
                "userinfo_endpoint": "https://{domain}/oauth2/v1/userinfo",
                "scope": "openid email profile",
                "response_type": "code",
                "grant_type": "authorization_code"
            },
            "github": {
                "name": "GitHub Enterprise",
                "auth_endpoint": "https://github.com/login/oauth/authorize",
                "token_endpoint": "https://github.com/login/oauth/access_token",
                "userinfo_endpoint": "https://api.github.com/user",
                "scope": "read:user user:email",
                "response_type": "code",
                "grant_type": "authorization_code"
            }
        }
        
    async def get_available_providers(self) -> List[Dict[str, Any]]:
        """Get list of configured SSO providers"""
        available_providers = []
        
        for provider_id, provider_info in self.providers.items():
            config = self.config.get(f"sso_{provider_id}")
            if config and config.get("enabled", False):
                available_providers.append({
                    "id": provider_id,
                    "name": provider_info["name"],
                    "enabled": True,
                    "icon": f"/assets/icons/{provider_id}.svg"
                })
                
        return available_providers
        
    async def get_provider_config(self, provider: str) -> Optional[Dict[str, Any]]:
        """Get configuration for specific provider"""
        if provider not in self.providers:
            return None
            
        config = self.config.get(f"sso_{provider}")
        if not config or not config.get("enabled", False):
            return None
            
        return {
            **self.providers[provider],
            **config
        }
        
    async def generate_auth_url(self, provider: str, state: str, redirect_uri: str) -> str:
        """Generate OAuth authorization URL"""
        provider_config = await self.get_provider_config(provider)
        if not provider_config:
            raise ValueError(f"Provider {provider} not configured")
            
        # Base parameters
        params = {
            "client_id": provider_config["client_id"],
            "response_type": provider_config["response_type"],
            "scope": provider_config["scope"],
            "state": state,
            "redirect_uri": redirect_uri
        }
        
        # Provider-specific parameters
        if provider == "azure":
            params["response_mode"] = "query"
            auth_endpoint = provider_config["auth_endpoint"].format(
                tenant=provider_config.get("tenant", "common")
            )
        elif provider == "okta":
            auth_endpoint = provider_config["auth_endpoint"].format(
                domain=provider_config["domain"]
            )
        else:
            auth_endpoint = provider_config["auth_endpoint"]
            
        # PKCE for enhanced security
        if provider_config.get("use_pkce", True):
            code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
            code_challenge = base64.urlsafe_b64encode(
                hashlib.sha256(code_verifier.encode('utf-8')).digest()
            ).decode('utf-8').rstrip('=')
            
            params.update({
                "code_challenge": code_challenge,
                "code_challenge_method": "S256"
            })
            
            # Store code verifier for later use
            await self.redis.setex(
                f"pkce:{state}",
                600,  # 10 minutes
                code_verifier
            )
            
        return f"{auth_endpoint}?{urlencode(params)}"
        
    async def exchange_code_for_tokens(self, provider: str, code: str, redirect_uri: str, state: str = None) -> Dict[str, Any]:
        """Exchange authorization code for access tokens"""
        provider_config = await self.get_provider_config(provider)
        if not provider_config:
            raise ValueError(f"Provider {provider} not configured")
            
        # Prepare token request
        token_data = {
            "client_id": provider_config["client_id"],
            "client_secret": provider_config["client_secret"],
            "code": code,
            "grant_type": provider_config["grant_type"],
            "redirect_uri": redirect_uri
        }
        
        # Add PKCE code verifier if used
        if state and provider_config.get("use_pkce", True):
            code_verifier = await self.redis.get(f"pkce:{state}")
            if code_verifier:
                token_data["code_verifier"] = code_verifier.decode('utf-8')
                await self.redis.delete(f"pkce:{state}")
                
        # Provider-specific token endpoint
        if provider == "azure":
            token_endpoint = provider_config["token_endpoint"].format(
                tenant=provider_config.get("tenant", "common")
            )
        elif provider == "okta":
            token_endpoint = provider_config["token_endpoint"].format(
                domain=provider_config["domain"]
            )
        else:
            token_endpoint = provider_config["token_endpoint"]
            
        # Make token request
        async with httpx.AsyncClient() as client:
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            
            if provider == "github":
                headers["Accept"] = "application/json"
                
            response = await client.post(
                token_endpoint,
                data=token_data,
                headers=headers,
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"Token exchange failed: {response.text}")
                
            token_response = response.json()
            
            # Validate response
            if "access_token" not in token_response:
                raise Exception("No access token in response")
                
            return token_response
            
    async def get_user_info(self, provider: str, access_token: str) -> Dict[str, Any]:
        """Get user information from provider"""
        provider_config = await self.get_provider_config(provider)
        if not provider_config:
            raise ValueError(f"Provider {provider} not configured")
            
        # Provider-specific userinfo endpoint
        if provider == "okta":
            userinfo_endpoint = provider_config["userinfo_endpoint"].format(
                domain=provider_config["domain"]
            )
        else:
            userinfo_endpoint = provider_config["userinfo_endpoint"]
            
        # Make userinfo request
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {access_token}"}
            
            response = await client.get(
                userinfo_endpoint,
                headers=headers,
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to get user info: {response.text}")
                
            user_info = response.json()
            
            # Normalize user information across providers
            return self._normalize_user_info(provider, user_info)
            
    def _normalize_user_info(self, provider: str, user_info: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize user information from different providers"""
        normalized = {
            "provider": provider,
            "raw_data": user_info
        }
        
        if provider == "azure":
            normalized.update({
                "id": user_info.get("id"),
                "email": user_info.get("mail") or user_info.get("userPrincipalName"),
                "name": user_info.get("displayName"),
                "given_name": user_info.get("givenName"),
                "family_name": user_info.get("surname"),
                "username": user_info.get("userPrincipalName", "").split("@")[0]
            })
        elif provider == "google":
            normalized.update({
                "id": user_info.get("id"),
                "email": user_info.get("email"),
                "name": user_info.get("name"),
                "given_name": user_info.get("given_name"),
                "family_name": user_info.get("family_name"),
                "username": user_info.get("email", "").split("@")[0],
                "picture": user_info.get("picture")
            })
        elif provider == "okta":
            normalized.update({
                "id": user_info.get("sub"),
                "email": user_info.get("email"),
                "name": user_info.get("name"),
                "given_name": user_info.get("given_name"),
                "family_name": user_info.get("family_name"),
                "username": user_info.get("preferred_username", "").split("@")[0]
            })
        elif provider == "github":
            normalized.update({
                "id": str(user_info.get("id")),
                "email": user_info.get("email"),
                "name": user_info.get("name"),
                "username": user_info.get("login"),
                "avatar_url": user_info.get("avatar_url")
            })
            
        return normalized
        
    async def refresh_tokens(self, provider: str, refresh_token: str) -> Dict[str, Any]:
        """Refresh access tokens"""
        provider_config = await self.get_provider_config(provider)
        if not provider_config:
            raise ValueError(f"Provider {provider} not configured")
            
        token_data = {
            "client_id": provider_config["client_id"],
            "client_secret": provider_config["client_secret"],
            "refresh_token": refresh_token,
            "grant_type": "refresh_token"
        }
        
        # Provider-specific token endpoint
        if provider == "azure":
            token_endpoint = provider_config["token_endpoint"].format(
                tenant=provider_config.get("tenant", "common")
            )
        elif provider == "okta":
            token_endpoint = provider_config["token_endpoint"].format(
                domain=provider_config["domain"]
            )
        else:
            token_endpoint = provider_config["token_endpoint"]
            
        async with httpx.AsyncClient() as client:
            response = await client.post(
                token_endpoint,
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"Token refresh failed: {response.text}")
                
            return response.json()
            
    async def revoke_user_tokens(self, provider: str, user_id: str):
        """Revoke user tokens with provider"""
        try:
            # Get stored tokens
            tokens = await self.get_user_tokens(user_id)
            if not tokens:
                return
                
            provider_config = await self.get_provider_config(provider)
            if not provider_config:
                return
                
            # Provider-specific revocation
            if provider == "azure":
                # Azure AD doesn't have a standard revocation endpoint
                # Tokens expire automatically
                pass
            elif provider == "google":
                async with httpx.AsyncClient() as client:
                    await client.post(
                        "https://oauth2.googleapis.com/revoke",
                        data={"token": tokens.get("access_token")},
                        timeout=10.0
                    )
            elif provider == "okta":
                revoke_endpoint = f"https://{provider_config['domain']}/oauth2/v1/revoke"
                async with httpx.AsyncClient() as client:
                    await client.post(
                        revoke_endpoint,
                        data={
                            "token": tokens.get("access_token"),
                            "client_id": provider_config["client_id"]
                        },
                        timeout=10.0
                    )
                    
            # Clean up stored tokens
            await self.delete_user_tokens(user_id)
            
        except Exception as e:
            # Log error but don't fail the logout process
            print(f"Failed to revoke tokens for user {user_id}: {e}")
            
    async def store_sso_state(self, state: str, data: Dict[str, Any]):
        """Store SSO state for CSRF protection"""
        await self.redis.setex(
            f"sso_state:{state}",
            600,  # 10 minutes
            json.dumps(data)
        )
        
    async def get_sso_state(self, state: str) -> Optional[Dict[str, Any]]:
        """Retrieve SSO state"""
        data = await self.redis.get(f"sso_state:{state}")
        if data:
            return json.loads(data.decode('utf-8'))
        return None
        
    async def delete_sso_state(self, state: str):
        """Delete SSO state"""
        await self.redis.delete(f"sso_state:{state}")
        
    async def store_user_tokens(self, user_id: str, tokens: Dict[str, Any]):
        """Store user tokens securely"""
        await self.redis.setex(
            f"sso_tokens:{user_id}",
            86400 * 7,  # 7 days
            json.dumps(tokens)
        )
        
    async def get_user_tokens(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get stored user tokens"""
        data = await self.redis.get(f"sso_tokens:{user_id}")
        if data:
            return json.loads(data.decode('utf-8'))
        return None
        
    async def get_user_refresh_token(self, user_id: str) -> Optional[str]:
        """Get user refresh token"""
        tokens = await self.get_user_tokens(user_id)
        if tokens:
            return tokens.get("refresh_token")
        return None
        
    async def update_user_tokens(self, user_id: str, tokens: Dict[str, Any]):
        """Update stored user tokens"""
        await self.store_user_tokens(user_id, tokens)
        
    async def delete_user_tokens(self, user_id: str):
        """Delete stored user tokens"""
        await self.redis.delete(f"sso_tokens:{user_id}")
        
    async def get_frontend_config(self) -> Dict[str, Any]:
        """Get SSO configuration for frontend"""
        providers = await self.get_available_providers()
        
        return {
            "providers": providers,
            "enabled": len(providers) > 0,
            "default_redirect": "/dashboard",
            "login_url": "/api/v1/sso/{provider}/login",
            "logout_url": "/api/v1/sso/logout"
        }