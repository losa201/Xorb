"""
Base controller with common functionality
"""

from typing import Any, Dict, Optional
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse

from ..domain.exceptions import (
    DomainException, ValidationError, BusinessRuleViolation,
    ResourceNotFound, UnauthorizedAccess, RateLimitExceeded,
    InvalidCredentials, TokenExpired, InsufficientPermissions,
    EmbeddingGenerationFailed, WorkflowExecutionFailed,
    InvalidInput, ServiceUnavailable, ConfigurationError,
    ResourceLimitExceeded
)


class BaseController:
    """Base controller with common functionality"""

    def handle_domain_exception(self, e: DomainException) -> HTTPException:
        """Convert domain exceptions to HTTP exceptions"""

        if isinstance(e, ValidationError) or isinstance(e, InvalidInput):
            return HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=e.message
            )

        elif isinstance(e, UnauthorizedAccess) or isinstance(e, InvalidCredentials):
            return HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=e.message,
                headers={"WWW-Authenticate": "Bearer"}
            )

        elif isinstance(e, TokenExpired):
            return HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=e.message,
                headers={"WWW-Authenticate": "Bearer"}
            )

        elif isinstance(e, InsufficientPermissions):
            return HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=e.message
            )

        elif isinstance(e, ResourceNotFound):
            return HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=e.message
            )

        elif isinstance(e, BusinessRuleViolation):
            return HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=e.message
            )

        elif isinstance(e, RateLimitExceeded):
            headers = {
                "X-RateLimit-Limit": str(e.limit),
                "X-RateLimit-Remaining": str(e.remaining),
                "X-RateLimit-Reset": str(e.reset_time),
                "Retry-After": "60"
            }
            return HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=e.message,
                headers=headers
            )

        elif isinstance(e, ResourceLimitExceeded):
            return HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=e.message
            )

        elif isinstance(e, (EmbeddingGenerationFailed, WorkflowExecutionFailed, ServiceUnavailable)):
            return HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=e.message
            )

        elif isinstance(e, ConfigurationError):
            return HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal configuration error"
            )

        else:
            # Generic domain exception
            return HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=e.message
            )

    def success_response(
        self,
        data: Any,
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None
    ) -> JSONResponse:
        """Create a success response"""

        return JSONResponse(
            content=data,
            status_code=status_code,
            headers=headers or {}
        )

    def error_response(
        self,
        message: str,
        status_code: int = 400,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> JSONResponse:
        """Create an error response"""

        content = {
            "error": message,
            "status_code": status_code
        }

        if error_code:
            content["error_code"] = error_code

        if details:
            content["details"] = details

        return JSONResponse(
            content=content,
            status_code=status_code
        )

    def paginated_response(
        self,
        data: Any,
        page: int,
        size: int,
        total: int,
        headers: Optional[Dict[str, str]] = None
    ) -> JSONResponse:
        """Create a paginated response"""

        total_pages = (total + size - 1) // size

        content = {
            "data": data,
            "pagination": {
                "page": page,
                "size": size,
                "total": total,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_previous": page > 1
            }
        }

        return JSONResponse(
            content=content,
            status_code=200,
            headers=headers or {}
        )
