"""
SSO Router
Enterprise Single Sign-On authentication endpoints
"""

from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import RedirectResponse
from typing import Dict, Any

from src.api.controllers.sso_controller import SSOController, get_sso_controller
from src.api.dependencies import get_current_user
from src.common.models.auth import User

router = APIRouter(prefix="/api/v1/sso", tags=["sso"])


@router.get("/providers")
async def get_sso_providers(
    sso_controller: SSOController = Depends(get_sso_controller)
) -> Dict[str, Any]:
    """
    Get available SSO providers
    
    Returns list of configured SSO providers with their details
    """
    return await sso_controller.get_sso_providers()


@router.get("/config")
async def get_sso_configuration(
    sso_controller: SSOController = Depends(get_sso_controller)
) -> Dict[str, Any]:
    """
    Get SSO configuration for frontend
    
    Returns SSO configuration including available providers and URLs
    """
    return await sso_controller.get_sso_configuration()


@router.get("/{provider}/login")
async def initiate_sso_login(
    provider: str,
    request: Request,
    sso_controller: SSOController = Depends(get_sso_controller)
) -> RedirectResponse:
    """
    Initiate SSO login flow
    
    Redirects user to the SSO provider's authorization endpoint
    
    Args:
        provider: SSO provider identifier (azure, google, okta, github)
        request: FastAPI request object
        
    Returns:
        RedirectResponse to the provider's authorization endpoint
    """
    return await sso_controller.initiate_sso_login(provider, request)


@router.get("/{provider}/callback")
async def handle_sso_callback(
    provider: str,
    request: Request,
    sso_controller: SSOController = Depends(get_sso_controller)
) -> Dict[str, Any]:
    """
    Handle SSO callback
    
    Processes the callback from SSO provider and completes authentication
    
    Args:
        provider: SSO provider identifier
        request: FastAPI request object containing callback parameters
        
    Returns:
        Authentication response with user details and JWT token
    """
    return await sso_controller.handle_sso_callback(provider, request)


@router.post("/refresh")
async def refresh_sso_token(
    request: Request,
    current_user: User = Depends(get_current_user),
    sso_controller: SSOController = Depends(get_sso_controller)
) -> Dict[str, Any]:
    """
    Refresh SSO token
    
    Refreshes the user's SSO tokens and generates a new JWT
    
    Args:
        request: FastAPI request object
        current_user: Currently authenticated user
        
    Returns:
        New JWT token and expiration information
    """
    return await sso_controller.refresh_sso_token(request, current_user)


@router.post("/logout")
async def logout_sso_user(
    request: Request,
    current_user: User = Depends(get_current_user),
    sso_controller: SSOController = Depends(get_sso_controller)
) -> Dict[str, Any]:
    """
    Logout SSO user
    
    Revokes SSO tokens and invalidates the user session
    
    Args:
        request: FastAPI request object
        current_user: Currently authenticated user
        
    Returns:
        Success message
    """
    return await sso_controller.logout_sso_user(request, current_user)


@router.get("/user/profile")
async def get_sso_user_profile(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get SSO user profile
    
    Returns the current user's profile information
    
    Args:
        current_user: Currently authenticated user
        
    Returns:
        User profile information
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
        
    return {
        "success": True,
        "data": {
            "user": {
                "id": current_user.id,
                "username": current_user.username,
                "email": current_user.email,
                "role": current_user.role,
                "sso_provider": getattr(current_user, 'sso_provider', None),
                "last_login": getattr(current_user, 'last_login', None),
                "permissions": getattr(current_user, 'permissions', [])
            }
        }
    }