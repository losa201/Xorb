from fastapi import APIRouter, Depends, HTTPException, status
import os
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from ..container import get_container
from ..services.interfaces import AuthenticationService
from ..domain.exceptions import DomainException
from ..security.auth import authenticator, Role

router = APIRouter()


class Token(BaseModel):
    access_token: str
    token_type: str


@router.post("/auth/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and return access token"""
    try:
        container = get_container()
        auth_service = container.get(AuthenticationService)
        
        # Authenticate user
        user = await auth_service.authenticate_user(
            username=form_data.username,
            password=form_data.password
        )
        
        # Create access token
        access_token = await auth_service.create_access_token(user)
        
        return Token(access_token=access_token, token_type="bearer")
        
    except DomainException as e:
        if "Invalid" in str(e) or "credentials" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication service error"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


def get_current_token():
    """Dependency to extract current token - simplified for this example"""
    # In a real implementation, this would extract the token from the Authorization header
    return "dummy_token"


@router.post("/auth/logout")
async def logout(token: str = Depends(get_current_token)):
    """Logout user by revoking token"""
    try:
        container = get_container()
        auth_service = container.get(AuthenticationService)
        
        success = await auth_service.revoke_token(token)
        
        return {
            "message": "Successfully logged out" if success else "Token not found",
            "revoked": success
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.post("/auth/dev-token", response_model=Token)
async def create_dev_token(username: str = "dev", role: str = "admin"):
    """Create a development JWT for local testing (enabled only when DEV_MODE=true).

    Parameters:
    - username: identifier for the token subject
    - role: one of ['admin','orchestrator','analyst','agent','readonly']
    """
    if os.getenv("DEV_MODE", "false").lower() != "true":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

    try:
        # map role string to Role enum (default to READONLY on invalid)
        try:
            selected_role = Role(role)
        except Exception:
            selected_role = Role.READONLY

        token_str = authenticator.generate_jwt(user_id=username, client_id=f"dev-{username}", roles=[selected_role])
        return Token(access_token=token_str, token_type="bearer")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create dev token: {str(e)}")
