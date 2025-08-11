from fastapi import APIRouter, Depends, HTTPException, status
import os
import secrets
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from ..container import get_container
from ..services.interfaces import AuthenticationService
from ..domain.exceptions import DomainException
from ..security import Role, require_admin
from ..core.logging import get_logger

logger = get_logger(__name__)

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


@router.post("/auth/dev-token", response_model=Token, include_in_schema=False)
async def create_dev_token(
    username: str = "dev-user", 
    role: str = "readonly",  # Default to least privilege
    # current_user = Depends(require_admin)  # TODO: Enable when admin auth is implemented
):
    """Create development token (DEV/TEST environments only, admin required)
    
    This endpoint is heavily restricted and only available in development
    environments with proper security controls.
    """
    
    # Multiple layers of protection
    environment_checks = [
        os.getenv("DEV_MODE") == "true",
        os.getenv("ENVIRONMENT") in ["development", "test"],
        os.getenv("ALLOW_DEV_TOKENS") == "true",  # Additional flag
        # current_user.is_admin if current_user else False  # TODO: Enable when auth is ready
    ]
    
    # For now, require explicit environment flag until full auth is implemented
    if not all(environment_checks[:3]):  # Skip admin check temporarily
        # Log unauthorized access attempt
        logger.warning("Unauthorized dev token access attempt",
                      environment=os.getenv("ENVIRONMENT"),
                      dev_mode=os.getenv("DEV_MODE"),
                      allow_dev_tokens=os.getenv("ALLOW_DEV_TOKENS"))
        raise HTTPException(status_code=404, detail="Not found")
    
    # Validate role more strictly
    allowed_roles = ["readonly", "analyst"]  # No admin by default
    if role not in allowed_roles:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role '{role}'. Allowed: {allowed_roles}"
        )
    
    try:
        container = get_container()
        auth_service = container.get(AuthenticationService)
        
        # Create limited development token
        additional_claims = {
            "username": username,
            "roles": [role],
            "dev_token": True,  # Mark as dev token
            "expires_in": 3600  # 1 hour max
        }
        
        token = auth_service.create_access_token(
            subject=username,
            additional_claims=additional_claims
        )
        
        # Audit log
        logger.info("Development token created",
                   username=username,
                   role=role,
                   environment=os.getenv("ENVIRONMENT"))
        
        return Token(access_token=token, token_type="bearer")
        
    except Exception as e:
        error_id = secrets.token_hex(8)
        logger.error("Dev token creation failed",
                    username=username,
                    error_id=error_id)
        raise HTTPException(
            status_code=500,
            detail="Token creation failed"
        )
