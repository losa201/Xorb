"""
Enterprise SSO Integration Controller
Provides Single Sign-On authentication with major identity providers
"""

from typing import Dict, Any, Optional
from fastapi import HTTPException, Request, Response, Depends
from fastapi.responses import RedirectResponse
from urllib.parse import urlencode, quote
import jwt
import httpx
import secrets
from datetime import datetime, timedelta
import hashlib
import base64

from src.api.services.sso_service import SSOService
from src.api.services.auth_service import AuthService
from src.common.models.auth import User, SSOProvider, SSOConfig
from src.api.dependencies import get_current_user, get_sso_service, get_auth_service


class SSOController:
    """Enterprise SSO authentication controller"""
    
    def __init__(self, sso_service: SSOService, auth_service: AuthService):
        self.sso_service = sso_service
        self.auth_service = auth_service
        
    async def get_sso_providers(self) -> Dict[str, Any]:
        """Get available SSO providers"""
        try:
            providers = await self.sso_service.get_available_providers()
            return {
                "success": True,
                "data": {
                    "providers": providers,
                    "count": len(providers)
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get SSO providers: {str(e)}")
            
    async def initiate_sso_login(self, provider: str, request: Request) -> RedirectResponse:
        """Initiate SSO login flow"""
        try:
            # Generate state parameter for CSRF protection
            state = secrets.token_urlsafe(32)
            
            # Store state in session/cache
            await self.sso_service.store_sso_state(state, {
                "provider": provider,
                "redirect_uri": request.url_for("sso_callback", provider=provider),
                "created_at": datetime.utcnow().isoformat()
            })
            
            # Get provider configuration
            provider_config = await self.sso_service.get_provider_config(provider)
            if not provider_config:
                raise HTTPException(status_code=404, detail=f"SSO provider {provider} not configured")
                
            # Generate authorization URL
            auth_url = await self.sso_service.generate_auth_url(
                provider=provider,
                state=state,
                redirect_uri=str(request.url_for("sso_callback", provider=provider))
            )
            
            return RedirectResponse(url=auth_url, status_code=302)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initiate SSO login: {str(e)}")
            
    async def handle_sso_callback(self, provider: str, request: Request) -> Dict[str, Any]:
        """Handle SSO callback and complete authentication"""
        try:
            # Extract callback parameters
            params = dict(request.query_params)
            
            # Validate state parameter
            state = params.get("state")
            if not state:
                raise HTTPException(status_code=400, detail="Missing state parameter")
                
            stored_state = await self.sso_service.get_sso_state(state)
            if not stored_state or stored_state.get("provider") != provider:
                raise HTTPException(status_code=400, detail="Invalid state parameter")
                
            # Handle authorization code
            auth_code = params.get("code")
            if not auth_code:
                error = params.get("error", "Unknown error")
                error_description = params.get("error_description", "")
                raise HTTPException(
                    status_code=400, 
                    detail=f"SSO authentication failed: {error} - {error_description}"
                )
                
            # Exchange code for tokens
            token_response = await self.sso_service.exchange_code_for_tokens(
                provider=provider,
                code=auth_code,
                redirect_uri=stored_state.get("redirect_uri")
            )
            
            # Get user information from provider
            user_info = await self.sso_service.get_user_info(
                provider=provider,
                access_token=token_response["access_token"]
            )
            
            # Create or update user account
            user = await self.auth_service.create_or_update_sso_user(
                provider=provider,
                user_info=user_info,
                tokens=token_response
            )
            
            # Generate application JWT token
            jwt_token = await self.auth_service.generate_jwt_token(user)
            
            # Clean up state
            await self.sso_service.delete_sso_state(state)
            
            return {
                "success": True,
                "data": {
                    "user": {
                        "id": user.id,
                        "username": user.username,
                        "email": user.email,
                        "role": user.role,
                        "sso_provider": provider
                    },
                    "token": jwt_token,
                    "expires_in": 3600,
                    "token_type": "Bearer"
                }
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"SSO callback failed: {str(e)}")
            
    async def refresh_sso_token(self, request: Request, current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
        """Refresh SSO token"""
        try:
            if not current_user.sso_provider:
                raise HTTPException(status_code=400, detail="User not authenticated via SSO")
                
            # Get stored refresh token
            refresh_token = await self.sso_service.get_user_refresh_token(current_user.id)
            if not refresh_token:
                raise HTTPException(status_code=400, detail="No refresh token available")
                
            # Refresh tokens with provider
            token_response = await self.sso_service.refresh_tokens(
                provider=current_user.sso_provider,
                refresh_token=refresh_token
            )
            
            # Update stored tokens
            await self.sso_service.update_user_tokens(
                user_id=current_user.id,
                tokens=token_response
            )
            
            # Generate new JWT token
            jwt_token = await self.auth_service.generate_jwt_token(current_user)
            
            return {
                "success": True,
                "data": {
                    "token": jwt_token,
                    "expires_in": 3600,
                    "token_type": "Bearer"
                }
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Token refresh failed: {str(e)}")
            
    async def logout_sso_user(self, request: Request, current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
        """Logout SSO user and revoke tokens"""
        try:
            if current_user.sso_provider:
                # Revoke tokens with provider
                await self.sso_service.revoke_user_tokens(
                    provider=current_user.sso_provider,
                    user_id=current_user.id
                )
                
            # Invalidate local session
            await self.auth_service.invalidate_user_session(current_user.id)
            
            return {
                "success": True,
                "message": "Successfully logged out"
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Logout failed: {str(e)}")
            
    async def get_sso_configuration(self) -> Dict[str, Any]:
        """Get SSO configuration for frontend"""
        try:
            config = await self.sso_service.get_frontend_config()
            return {
                "success": True,
                "data": config
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get SSO configuration: {str(e)}")


# Dependency injection
def get_sso_controller(
    sso_service: SSOService = Depends(get_sso_service),
    auth_service: AuthService = Depends(get_auth_service)
) -> SSOController:
    return SSOController(sso_service, auth_service)