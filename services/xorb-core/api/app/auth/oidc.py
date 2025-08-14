"""OIDC authentication provider."""
import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional
from uuid import UUID

import httpx
from authlib.integrations.httpx_client import AsyncOAuth2Client
from authlib.jose import jwt
from authlib.oidc.core import CodeIDToken
from fastapi import HTTPException, status
from pydantic import ValidationError

from .models import OIDCConfig, TokenData, UserClaims, Role
from ..infrastructure.cache import get_cache


class OIDCProvider:
    """OIDC authentication provider with caching."""

    def __init__(self, config: OIDCConfig):
        self.config = config
        self.client = AsyncOAuth2Client(
            client_id=config.client_id,
            client_secret=config.client_secret,
            scope=" ".join(config.scopes)
        )
        self._discovery_cache: Optional[Dict] = None
        self._jwks_cache: Optional[Dict] = None

    async def get_discovery_document(self) -> Dict:
        """Get OIDC discovery document with caching."""
        if self._discovery_cache:
            return self._discovery_cache

        cache = get_cache()
        cache_key = f"oidc:discovery:{self.config.issuer}"

        # Try cache first
        cached = await cache.get(cache_key)
        if cached:
            self._discovery_cache = cached
            return cached

        # Fetch from provider
        discovery_url = f"{self.config.issuer.rstrip('/')}/.well-known/openid-configuration"
        async with httpx.AsyncClient() as client:
            response = await client.get(discovery_url)
            response.raise_for_status()

        discovery = response.json()

        # Cache for 1 hour
        await cache.set(cache_key, discovery, expire=3600)
        self._discovery_cache = discovery
        return discovery

    async def get_jwks(self) -> Dict:
        """Get JWKS with caching."""
        if self._jwks_cache:
            return self._jwks_cache

        cache = get_cache()
        cache_key = f"oidc:jwks:{self.config.issuer}"

        # Try cache first
        cached = await cache.get(cache_key)
        if cached:
            self._jwks_cache = cached
            return cached

        # Get JWKS URI from discovery
        discovery = await self.get_discovery_document()
        jwks_uri = discovery["jwks_uri"]

        async with httpx.AsyncClient() as client:
            response = await client.get(jwks_uri)
            response.raise_for_status()

        jwks = response.json()

        # Cache for 1 hour
        await cache.set(cache_key, jwks, expire=3600)
        self._jwks_cache = jwks
        return jwks

    def get_authorization_url(self, state: str, nonce: str) -> str:
        """Generate authorization URL."""
        return self.client.create_authorization_url(
            self.config.redirect_uri,
            state=state,
            nonce=nonce
        )[0]

    async def exchange_code(self, code: str, state: str, nonce: str) -> TokenData:
        """Exchange authorization code for tokens."""
        discovery = await self.get_discovery_document()

        response = await self.client.fetch_token(
            discovery["token_endpoint"],
            code=code,
            redirect_uri=self.config.redirect_uri
        )

        # Validate ID token if present
        if "id_token" in response:
            await self._validate_id_token(response["id_token"], nonce)

        return TokenData(
            access_token=response["access_token"],
            token_type=response.get("token_type", "Bearer"),
            expires_in=response.get("expires_in", 3600),
            refresh_token=response.get("refresh_token")
        )

    async def _validate_id_token(self, id_token: str, nonce: str) -> Dict:
        """Validate ID token."""
        jwks = await self.get_jwks()
        discovery = await self.get_discovery_document()

        try:
            claims = jwt.decode(
                id_token,
                jwks,
                claims_options={
                    "iss": {"essential": True, "value": self.config.issuer},
                    "aud": {"essential": True, "value": self.config.client_id},
                    "nonce": {"essential": True, "value": nonce}
                }
            )
            return claims
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid ID token: {e}"
            )

    async def validate_access_token(self, access_token: str) -> UserClaims:
        """Validate access token and extract user claims."""
        jwks = await self.get_jwks()

        try:
            # Decode without verification first to get header
            unverified = jwt.decode(access_token, options={"verify_signature": False})

            # Now verify
            claims = jwt.decode(access_token, jwks)

            # Extract and map claims
            user_claims = self._map_claims(claims)
            return user_claims

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid access token: {e}"
            )

    def _map_claims(self, claims: Dict) -> UserClaims:
        """Map OIDC claims to UserClaims model."""
        try:
            # Extract tenant ID
            tenant_id_claim = claims.get(self.config.tenant_claim)
            if not tenant_id_claim:
                raise ValueError(f"Missing {self.config.tenant_claim} claim")

            tenant_id = UUID(str(tenant_id_claim))

            # Extract roles
            roles_claim = claims.get(self.config.roles_claim, [])
            roles = []
            for role_str in roles_claim:
                try:
                    roles.append(Role(role_str))
                except ValueError:
                    # Skip unknown roles
                    continue

            return UserClaims(
                sub=claims["sub"],
                email=claims.get(self.config.email_claim, ""),
                name=claims.get(self.config.name_claim),
                tenant_id=tenant_id,
                roles=roles,
                exp=datetime.fromtimestamp(claims["exp"], tz=timezone.utc),
                iat=datetime.fromtimestamp(claims["iat"], tz=timezone.utc)
            )

        except (KeyError, ValueError, ValidationError) as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token claims: {e}"
            )

    async def refresh_token(self, refresh_token: str) -> TokenData:
        """Refresh access token."""
        discovery = await self.get_discovery_document()

        response = await self.client.refresh_token(
            discovery["token_endpoint"],
            refresh_token=refresh_token
        )

        return TokenData(
            access_token=response["access_token"],
            token_type=response.get("token_type", "Bearer"),
            expires_in=response.get("expires_in", 3600),
            refresh_token=response.get("refresh_token", refresh_token)
        )


# Global OIDC provider instance
_oidc_provider: Optional[OIDCProvider] = None


def get_oidc_provider() -> OIDCProvider:
    """Get the global OIDC provider instance."""
    global _oidc_provider
    if not _oidc_provider:
        raise RuntimeError("OIDC provider not initialized")
    return _oidc_provider


def init_oidc_provider(config: OIDCConfig) -> None:
    """Initialize the global OIDC provider."""
    global _oidc_provider
    _oidc_provider = OIDCProvider(config)
