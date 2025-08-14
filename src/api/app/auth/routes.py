"""Authentication routes."""
import secrets
from typing import Dict
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse

from .dependencies import get_current_user, require_auth
from .models import TokenData, UserClaims
from .oidc import get_oidc_provider
from ..infrastructure.cache import get_cache


router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.get("/login")
async def login(request: Request, redirect_uri: str = "/") -> RedirectResponse:
    """Initiate OIDC login flow."""
    oidc = get_oidc_provider()
    cache = get_cache()

    # Generate state and nonce for CSRF protection
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)

    # Store state and redirect URI in cache
    await cache.set(f"auth:state:{state}", {
        "nonce": nonce,
        "redirect_uri": redirect_uri
    }, expire=600)  # 10 minutes

    auth_url = oidc.get_authorization_url(state, nonce)
    return RedirectResponse(auth_url)


@router.get("/callback")
async def callback(
    request: Request,
    code: str,
    state: str,
    error: str = None
) -> RedirectResponse:
    """Handle OIDC callback."""
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OIDC error: {error}"
        )

    cache = get_cache()

    # Validate state
    cache_key = f"auth:state:{state}"
    state_data = await cache.get(cache_key)
    if not state_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired state"
        )

    # Clean up state
    await cache.delete(cache_key)

    try:
        oidc = get_oidc_provider()
        token_data = await oidc.exchange_code(code, state, state_data["nonce"])

        # Store token in secure cookie or session
        redirect_uri = state_data.get("redirect_uri", "/")
        response = RedirectResponse(redirect_uri)

        # Set secure HTTP-only cookie with token
        response.set_cookie(
            key="access_token",
            value=token_data.access_token,
            max_age=token_data.expires_in,
            httponly=True,
            secure=True,
            samesite="lax"
        )

        if token_data.refresh_token:
            response.set_cookie(
                key="refresh_token",
                value=token_data.refresh_token,
                max_age=7 * 24 * 3600,  # 7 days
                httponly=True,
                secure=True,
                samesite="lax"
            )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Token exchange failed: {e}"
        )


@router.post("/logout")
async def logout(response: Response) -> Dict[str, str]:
    """Logout current user."""
    response.delete_cookie("access_token")
    response.delete_cookie("refresh_token")
    return {"message": "Logged out successfully"}


@router.get("/me")
async def get_current_user_info(
    current_user: UserClaims = Depends(require_auth)
) -> UserClaims:
    """Get current user information."""
    return current_user


@router.post("/refresh")
async def refresh_access_token(request: Request) -> TokenData:
    """Refresh access token using refresh token."""
    refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token required"
        )

    try:
        oidc = get_oidc_provider()
        token_data = await oidc.refresh_token(refresh_token)
        return token_data

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token refresh failed: {e}"
        )


@router.get("/roles")
async def get_available_roles() -> Dict[str, list]:
    """Get available roles and their permissions."""
    from .models import ROLE_PERMISSIONS, Role, Permission

    return {
        "roles": [role.value for role in Role],
        "permissions": [perm.value for perm in Permission],
        "role_permissions": {
            role.value: [perm.value for perm in perms]
            for role, perms in ROLE_PERMISSIONS.items()
        }
    }
