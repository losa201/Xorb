"""Global error handling middleware with structured error responses."""
import logging
import traceback
from typing import Dict, Any, Optional

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from redis.exceptions import RedisError
import structlog

from ..infrastructure.observability import add_trace_context


logger = structlog.get_logger("error_handler")


class ErrorDetail:
    """Structured error detail."""
    
    def __init__(
        self,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        field: Optional[str] = None
    ):
        self.code = code
        self.message = message
        self.details = details or {}
        self.field = field
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "code": self.code,
            "message": self.message
        }
        
        if self.field:
            result["field"] = self.field
        
        if self.details:
            result["details"] = self.details
        
        return result


class XorbError(Exception):
    """Base exception for Xorb application errors."""
    
    def __init__(
        self,
        message: str,
        code: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class ValidationError(XorbError):
    """Validation error with field-specific details."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details
        )
        self.field = field


class AuthenticationError(XorbError):
    """Authentication-related errors."""
    
    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            message=message,
            code="AUTHENTICATION_ERROR",
            status_code=status.HTTP_401_UNAUTHORIZED
        )


class AuthorizationError(XorbError):
    """Authorization-related errors."""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            message=message,
            code="AUTHORIZATION_ERROR", 
            status_code=status.HTTP_403_FORBIDDEN
        )


class TenantError(XorbError):
    """Tenant-related errors."""
    
    def __init__(self, message: str):
        super().__init__(
            message=message,
            code="TENANT_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST
        )


class ResourceNotFoundError(XorbError):
    """Resource not found errors."""
    
    def __init__(self, resource: str, identifier: str):
        super().__init__(
            message=f"{resource} not found: {identifier}",
            code="RESOURCE_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"resource": resource, "identifier": identifier}
        )


class StorageError(XorbError):
    """Storage-related errors."""
    
    def __init__(self, message: str, operation: str):
        super().__init__(
            message=message,
            code="STORAGE_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"operation": operation}
        )


class RateLimitError(XorbError):
    """Rate limiting errors."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        super().__init__(
            message=message,
            code="RATE_LIMIT_EXCEEDED",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details={"retry_after": retry_after} if retry_after else {}
        )


class GlobalErrorHandler(BaseHTTPMiddleware):
    """Global error handling middleware."""
    
    def __init__(self, app, debug: bool = False):
        super().__init__(app)
        self.debug = debug
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Handle errors and return structured responses."""
        try:
            response = await call_next(request)
            return response
        
        except HTTPException as e:
            return await self._handle_http_exception(request, e)
        
        except XorbError as e:
            return await self._handle_xorb_error(request, e)
        
        except ValidationError as e:
            return await self._handle_validation_error(request, e)
        
        except IntegrityError as e:
            return await self._handle_database_error(request, e)
        
        except SQLAlchemyError as e:
            return await self._handle_database_error(request, e)
        
        except RedisError as e:
            return await self._handle_redis_error(request, e)
        
        except Exception as e:
            return await self._handle_unexpected_error(request, e)
    
    async def _handle_http_exception(self, request: Request, exc: HTTPException) -> JSONResponse:
        """Handle FastAPI HTTP exceptions."""
        error_detail = ErrorDetail(
            code="HTTP_EXCEPTION",
            message=exc.detail if isinstance(exc.detail, str) else "HTTP error",
            details={"status_code": exc.status_code}
        )
        
        # Log error
        logger.warning(
            "HTTP exception",
            status_code=exc.status_code,
            detail=exc.detail,
            path=request.url.path,
            method=request.method
        )
        
        # Add trace context
        add_trace_context(
            error_type="http_exception",
            error_code=exc.status_code,
            error_detail=str(exc.detail)
        )
        
        return self._create_error_response(exc.status_code, [error_detail])
    
    async def _handle_xorb_error(self, request: Request, exc: XorbError) -> JSONResponse:
        """Handle application-specific errors."""
        error_detail = ErrorDetail(
            code=exc.code,
            message=exc.message,
            details=exc.details
        )
        
        # Log based on severity
        if exc.status_code >= 500:
            logger.error(
                "Application error",
                error_code=exc.code,
                error_message=exc.message,
                details=exc.details,
                path=request.url.path,
                method=request.method
            )
        else:
            logger.warning(
                "Client error",
                error_code=exc.code,
                error_message=exc.message,
                path=request.url.path,
                method=request.method
            )
        
        # Add trace context
        add_trace_context(
            error_type="application_error",
            error_code=exc.code,
            error_message=exc.message
        )
        
        return self._create_error_response(exc.status_code, [error_detail])
    
    async def _handle_validation_error(self, request: Request, exc: ValidationError) -> JSONResponse:
        """Handle validation errors."""
        error_detail = ErrorDetail(
            code=exc.code,
            message=exc.message,
            details=exc.details,
            field=exc.field
        )
        
        logger.warning(
            "Validation error",
            error_message=exc.message,
            field=exc.field,
            path=request.url.path,
            method=request.method
        )
        
        add_trace_context(
            error_type="validation_error",
            error_field=exc.field or "unknown"
        )
        
        return self._create_error_response(exc.status_code, [error_detail])
    
    async def _handle_database_error(self, request: Request, exc: SQLAlchemyError) -> JSONResponse:
        """Handle database errors."""
        error_detail = ErrorDetail(
            code="DATABASE_ERROR",
            message="Database operation failed"
        )
        
        if isinstance(exc, IntegrityError):
            error_detail.code = "INTEGRITY_ERROR"
            error_detail.message = "Data integrity constraint violated"
        
        logger.error(
            "Database error",
            error_type=type(exc).__name__,
            error_message=str(exc),
            path=request.url.path,
            method=request.method
        )
        
        add_trace_context(
            error_type="database_error",
            error_class=type(exc).__name__
        )
        
        return self._create_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            [error_detail]
        )
    
    async def _handle_redis_error(self, request: Request, exc: RedisError) -> JSONResponse:
        """Handle Redis errors."""
        error_detail = ErrorDetail(
            code="CACHE_ERROR",
            message="Cache operation failed"
        )
        
        logger.error(
            "Redis error",
            error_type=type(exc).__name__,
            error_message=str(exc),
            path=request.url.path,
            method=request.method
        )
        
        add_trace_context(
            error_type="redis_error",
            error_class=type(exc).__name__
        )
        
        return self._create_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            [error_detail]
        )
    
    async def _handle_unexpected_error(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected errors."""
        error_detail = ErrorDetail(
            code="INTERNAL_ERROR",
            message="An unexpected error occurred"
        )
        
        # Include traceback in debug mode
        if self.debug:
            error_detail.details["traceback"] = traceback.format_exc()
        
        logger.error(
            "Unexpected error",
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback.format_exc(),
            path=request.url.path,
            method=request.method
        )
        
        add_trace_context(
            error_type="unexpected_error",
            error_class=type(exc).__name__,
            error_message=str(exc)
        )
        
        return self._create_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            [error_detail]
        )
    
    def _create_error_response(
        self,
        status_code: int,
        errors: list[ErrorDetail]
    ) -> JSONResponse:
        """Create standardized error response."""
        
        response_data = {
            "success": False,
            "errors": [error.to_dict() for error in errors],
            "timestamp": logger._context.get("timestamp")
        }
        
        # Add request ID if available
        if hasattr(logger._context, "request_id"):
            response_data["request_id"] = logger._context["request_id"]
        
        return JSONResponse(
            status_code=status_code,
            content=response_data
        )


def handle_startup_errors():
    """Error handler for application startup."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    "Startup error",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    traceback=traceback.format_exc()
                )
                raise
        return wrapper
    return decorator


# Error response helpers
def create_validation_error_response(field: str, message: str) -> JSONResponse:
    """Create validation error response."""
    error = ErrorDetail(
        code="VALIDATION_ERROR",
        message=message,
        field=field
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "errors": [error.to_dict()]
        }
    )


def create_not_found_response(resource: str, identifier: str) -> JSONResponse:
    """Create not found error response."""
    error = ErrorDetail(
        code="RESOURCE_NOT_FOUND",
        message=f"{resource} not found",
        details={"resource": resource, "identifier": identifier}
    )
    
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "success": False,
            "errors": [error.to_dict()]
        }
    )