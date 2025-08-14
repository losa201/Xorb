"""
Authentication controller - Handles auth-related HTTP requests
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from ..container import get_container
from ..services.interfaces import AuthenticationService
from ..domain.exceptions import DomainException
from .base import BaseController


class Token(BaseModel):
    access_token: str
    token_type: str


class AuthController(BaseController):
    """Authentication controller"""

    def __init__(self):
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self):
        """Setup authentication routes"""

        @self.router.post("/auth/token", response_model=Token)
        async def login_for_access_token(
            form_data: OAuth2PasswordRequestForm = Depends()
        ):
            return await self.login(form_data)

        @self.router.post("/auth/logout")
        async def logout_endpoint(
            token: str = Depends(get_current_token)
        ):
            return await self.logout(token)

    async def login(self, form_data: OAuth2PasswordRequestForm) -> Token:
        """Handle user login"""

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
            raise self.handle_domain_exception(e)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )

    async def logout(self, token: str) -> dict:
        """Handle user logout"""

        try:
            container = get_container()
            auth_service = container.get(AuthenticationService)

            # Revoke token
            success = await auth_service.revoke_token(token)

            return {
                "message": "Successfully logged out" if success else "Token not found",
                "revoked": success
            }

        except DomainException as e:
            raise self.handle_domain_exception(e)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )


def get_current_token():
    """Dependency to extract current token - simplified for this example"""
    # In a real implementation, this would extract the token from the Authorization header
    # For now, we'll just return a placeholder
    return "dummy_token"


# Create controller instance and export router
auth_controller = AuthController()
router = auth_controller.router
