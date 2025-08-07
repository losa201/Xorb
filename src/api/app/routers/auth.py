from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from ..container import get_container
from ..services.interfaces import AuthenticationService
from ..domain.exceptions import DomainException

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
