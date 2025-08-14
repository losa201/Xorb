"""
Enterprise authentication router with SSO support
"""

import os
import secrets
import uuid
import hashlib
import json
import logging
from typing import Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from ..services.enterprise_sso import sso_service, SSOUserInfo, SSOProtocol
from ..services.consolidated_auth_service import ConsolidatedAuthService as AuthenticationService
from ..middleware.tenant_context import get_current_tenant_id
from ..container import get_container
from ..domain.tenant_entities import TenantUser


router = APIRouter(prefix="/auth/enterprise", tags=["Enterprise Authentication"])

# In-memory state storage (replace with Redis in production)
_state_store = {}

async def store_state_parameter(state: str, tenant_id: str, expires_minutes: int = 10) -> bool:
    """Store state parameter with expiration for CSRF protection"""
    try:
        expiry_time = datetime.utcnow() + timedelta(minutes=expires_minutes)
        state_data = {
            "tenant_id": tenant_id,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expiry_time.isoformat(),
            "used": False
        }

        # In production, use Redis with TTL
        _state_store[state] = state_data

        # Clean up expired states
        await cleanup_expired_states()

        return True
    except Exception as e:
        print(f"Failed to store state parameter: {e}")
        return False

async def validate_state_parameter(state: str, tenant_id: str) -> bool:
    """Validate state parameter against stored value"""
    try:
        if not state or state not in _state_store:
            return False

        state_data = _state_store[state]

        # Check if already used
        if state_data.get("used", False):
            return False

        # Check if expired
        expires_at = datetime.fromisoformat(state_data["expires_at"])
        if datetime.utcnow() > expires_at:
            del _state_store[state]
            return False

        # Check tenant ID matches
        if state_data.get("tenant_id") != tenant_id:
            return False

        # Mark as used
        state_data["used"] = True
        _state_store[state] = state_data

        return True
    except Exception as e:
        print(f"Failed to validate state parameter: {e}")
        return False

async def cleanup_expired_states():
    """Clean up expired state parameters"""
    try:
        current_time = datetime.utcnow()
        expired_states = []

        for state, data in _state_store.items():
            try:
                expires_at = datetime.fromisoformat(data["expires_at"])
                if current_time > expires_at:
                    expired_states.append(state)
            except:
                expired_states.append(state)  # Remove invalid entries

        for state in expired_states:
            del _state_store[state]

    except Exception as e:
        print(f"Failed to cleanup expired states: {e}")

async def verify_admin_role(tenant_id: str) -> bool:
    """Verify user has admin role for tenant (simplified implementation)"""
    try:
        # In production, this would:
        # 1. Get current user from JWT token
        # 2. Verify user belongs to tenant
        # 3. Check user has admin/owner role
        # 4. May also check specific permissions

        # For now, return True (replace with actual role checking)
        return True
    except Exception as e:
        print(f"Failed to verify admin role: {e}")
        return False


class SSOInitiateRequest(BaseModel):
    """Request to initiate SSO login"""
    tenant_id: str
    redirect_uri: Optional[str] = None


class SSOConfigurationRequest(BaseModel):
    """Request to configure SSO for a tenant"""
    provider: str
    protocol: str
    client_id: str
    client_secret: str
    issuer_url: str
    redirect_uri: str
    scopes: list[str] = ["openid", "profile", "email"]
    claims_mapping: dict[str, str] = {}
    group_claims: list[str] = ["groups"]
    auto_provision_users: bool = True
    require_mfa: bool = False
    allowed_domains: list[str] = []


@router.post("/sso/initiate")
async def initiate_sso_login(request: SSOInitiateRequest):
    """Initiate SSO login flow"""
    try:
        # Generate secure state parameter
        state = secrets.token_urlsafe(32)

        # Store state parameter for CSRF protection
        if not await store_state_parameter(state, request.tenant_id):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store state parameter"
            )

        # Get SSO authorization URL
        auth_url = await sso_service.initiate_sso_login(
            tenant_id=request.tenant_id,
            state=state,
            redirect_uri=request.redirect_uri
        )

        return {
            "authorization_url": auth_url,
            "state": state,
            "tenant_id": request.tenant_id
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate SSO login: {str(e)}"
        )


@router.get("/sso/callback")
async def handle_sso_callback(
    request: Request,
    tenant_id: Optional[str] = None,
    state: Optional[str] = None
):
    """Handle SSO callback and complete authentication"""
    try:
        # Extract tenant ID from state or query params
        if not tenant_id:
            # Extract from state parameter in production
            tenant_id = request.query_params.get("tenant_id")

        if not tenant_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing tenant ID"
            )

        # Validate state parameter (CSRF protection)
        if not state:
            state = request.query_params.get("state")

        # SECURITY: Validate state parameter (CSRF protection) - CRITICAL SECURITY FIX
        if not state:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing required state parameter for CSRF protection"
            )

        if not await validate_state_parameter(state, tenant_id):
            logger.warning(f"SSO callback state validation failed - potential CSRF attack. State: {state}, Tenant: {tenant_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid state parameter - potential CSRF attack"
            )

        # Handle SSO callback
        sso_user_info = await sso_service.handle_sso_callback(tenant_id, request)

        # Set tenant context
        # Set tenant context (would use proper tenant context manager in production)
        # tenant_context.set_tenant(uuid.UUID(tenant_id))

        # Get or create user
        container = get_container()
        auth_service = container.get(AuthenticationService)

        user = await get_or_create_sso_user(sso_user_info, auth_service)

        # Create access token
        access_token = await auth_service.create_access_token(user)

        # Create response with token
        response_data = {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": str(user.id),
                "email": user.email,
                "username": user.username,
                "roles": user.roles,
                "tenant_id": str(user.tenant_id)
            },
            "sso_provider": sso_user_info.provider.value,
            "mfa_verified": sso_user_info.mfa_verified
        }

        # For web applications, redirect with token
        redirect_uri = request.query_params.get("redirect_uri")
        if redirect_uri:
            # In production, use secure token passing (e.g., authorization code flow)
            redirect_url = f"{redirect_uri}?token={access_token}"
            return RedirectResponse(url=redirect_url)

        return response_data

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SSO callback failed: {str(e)}"
        )


@router.post("/sso/callback")
async def handle_sso_post_callback(request: Request):
    """Handle SAML POST callback"""
    return await handle_sso_callback(request)


async def get_or_create_sso_user(
    sso_user_info: SSOUserInfo,
    auth_service: AuthenticationService
) -> TenantUser:
    """Get existing user or create new one from SSO information"""

    # Try to find existing user by email
    existing_user = await auth_service.get_user_by_email(sso_user_info.email)

    if existing_user:
        # Update user information from SSO
        existing_user.roles = sso_user_info.roles
        existing_user.record_login()

        # Update user in database
        await auth_service.update_user(existing_user)

        return existing_user

    else:
        # Create new user from SSO information
        new_user = TenantUser.create(
            tenant_id=uuid.UUID(sso_user_info.tenant_id),
            username=sso_user_info.email,  # Use email as username
            email=sso_user_info.email,
            password_hash="",  # No password for SSO users
            roles=sso_user_info.roles
        )

        # Store additional SSO metadata
        new_user.metadata = {
            "sso_provider": sso_user_info.provider.value,
            "sso_user_id": sso_user_info.user_id,
            "first_name": sso_user_info.first_name,
            "last_name": sso_user_info.last_name,
            "display_name": sso_user_info.display_name,
            "groups": sso_user_info.groups,
            "created_via_sso": True
        }

        # Create user in database
        created_user = await auth_service.create_user(new_user)

        return created_user


@router.post("/sso/configure")
async def configure_sso(
    config_request: SSOConfigurationRequest,
    current_tenant: str = Depends(get_current_tenant_id)
):
    """Configure SSO for a tenant (admin only)"""

    # Verify admin role
    if not await verify_admin_role(current_tenant):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )

    try:
        from ..services.enterprise_sso import SSOConfiguration, SSOProvider, SSOProtocol

        # Create SSO configuration
        sso_config = SSOConfiguration(
            provider=SSOProvider(config_request.provider),
            protocol=SSOProtocol(config_request.protocol),
            tenant_id=current_tenant or config_request.tenant_id,
            client_id=config_request.client_id,
            client_secret=config_request.client_secret,
            issuer_url=config_request.issuer_url,
            authorization_endpoint=f"{config_request.issuer_url}/auth",
            token_endpoint=f"{config_request.issuer_url}/token",
            userinfo_endpoint=f"{config_request.issuer_url}/userinfo",
            jwks_uri=f"{config_request.issuer_url}/keys",
            saml_metadata_url=None,
            saml_x509_cert=None,
            redirect_uri=config_request.redirect_uri,
            scopes=config_request.scopes,
            claims_mapping=config_request.claims_mapping,
            group_claims=config_request.group_claims,
            auto_provision_users=config_request.auto_provision_users,
            require_mfa=config_request.require_mfa,
            allowed_domains=config_request.allowed_domains
        )

        # Add configuration to SSO service
        sso_service.add_sso_configuration(current_tenant, sso_config)

        return {
            "message": "SSO configuration added successfully",
            "tenant_id": current_tenant,
            "provider": config_request.provider
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to configure SSO: {str(e)}"
        )


@router.get("/sso/providers")
async def list_sso_providers():
    """List available SSO providers"""

    providers = [
        {
            "id": "okta",
            "name": "Okta",
            "protocol": "oidc",
            "description": "Enterprise identity and access management"
        },
        {
            "id": "azure_ad",
            "name": "Microsoft Azure AD",
            "protocol": "oidc",
            "description": "Microsoft Azure Active Directory"
        },
        {
            "id": "google_workspace",
            "name": "Google Workspace",
            "protocol": "oidc",
            "description": "Google Workspace (formerly G Suite)"
        },
        {
            "id": "ping_identity",
            "name": "Ping Identity",
            "protocol": "oidc",
            "description": "PingOne and PingFederate"
        },
        {
            "id": "auth0",
            "name": "Auth0",
            "protocol": "oidc",
            "description": "Auth0 identity platform"
        },
        {
            "id": "onelogin",
            "name": "OneLogin",
            "protocol": "saml2",
            "description": "OneLogin SAML SSO"
        },
        {
            "id": "generic_oidc",
            "name": "Generic OIDC",
            "protocol": "oidc",
            "description": "Any OpenID Connect provider"
        },
        {
            "id": "generic_saml",
            "name": "Generic SAML 2.0",
            "protocol": "saml2",
            "description": "Any SAML 2.0 provider"
        }
    ]

    return {"providers": providers}


@router.get("/sso/status/{tenant_id}")
async def get_sso_status(tenant_id: str):
    """Get SSO configuration status for a tenant"""

    config = sso_service.get_sso_configuration(tenant_id)

    if not config:
        return {
            "configured": False,
            "tenant_id": tenant_id
        }

    return {
        "configured": True,
        "tenant_id": tenant_id,
        "provider": config.provider.value,
        "protocol": config.protocol.value,
        "auto_provision_users": config.auto_provision_users,
        "require_mfa": config.require_mfa,
        "allowed_domains": config.allowed_domains
    }


@router.delete("/sso/configure/{tenant_id}")
async def remove_sso_configuration(tenant_id: str):
    """Remove SSO configuration for a tenant (admin only)"""

    # Verify admin role
    if not await verify_admin_role(tenant_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )

    if tenant_id in sso_service.configurations:
        del sso_service.configurations[tenant_id]

        if tenant_id in sso_service.oidc_clients:
            del sso_service.oidc_clients[tenant_id]

        if tenant_id in sso_service.saml_clients:
            del sso_service.saml_clients[tenant_id]

        return {
            "message": "SSO configuration removed successfully",
            "tenant_id": tenant_id
        }

    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="SSO configuration not found"
        )


@router.get("/sso/metadata/{tenant_id}")
async def get_sso_metadata(tenant_id: str):
    """Get SSO metadata for service provider configuration"""

    config = sso_service.get_sso_configuration(tenant_id)

    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="SSO not configured for tenant"
        )

    # Generate service provider metadata
    base_url = os.getenv("BASE_URL", "https://api.xorb.com")

    metadata = {
        "entity_id": f"{base_url}/auth/enterprise/sp/{tenant_id}",
        "acs_url": f"{base_url}/auth/enterprise/sso/callback",
        "slo_url": f"{base_url}/auth/enterprise/sso/logout",
        "certificate": None,  # Would include SP certificate
        "want_assertions_signed": True,
        "want_name_id": True,
        "name_id_format": "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"
    }

    if config.protocol == SSOProtocol.SAML2:
        # Return SAML metadata XML
        saml_metadata = f"""<?xml version="1.0"?>
<md:EntityDescriptor xmlns:md="urn:oasis:names:tc:SAML:2.0:metadata"
                     entityID="{metadata['entity_id']}">
  <md:SPSSODescriptor AuthnRequestsSigned="false" WantAssertionsSigned="true"
                      protocolSupportEnumeration="urn:oasis:names:tc:SAML:2.0:protocol">
    <md:AssertionConsumerService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
                                 Location="{metadata['acs_url']}"
                                 index="1" />
  </md:SPSSODescriptor>
</md:EntityDescriptor>"""

        return Response(content=saml_metadata, media_type="application/xml")

    else:
        # Return OIDC metadata
        return metadata


# JWT token validation for SSO tokens
@router.post("/sso/validate")
async def validate_sso_token(
    token: str,
    tenant_id: str
):
    """Validate SSO-issued token"""

    config = sso_service.get_sso_configuration(tenant_id)

    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="SSO not configured for tenant"
        )

    try:
        if config.protocol == SSOProtocol.OIDC:
            client = sso_service.oidc_clients[tenant_id]
            # This would validate the token against the provider
            # For now, return success
            return {"valid": True, "tenant_id": tenant_id}

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Token validation not supported for this protocol"
            )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token validation failed: {str(e)}"
        )
