"""
Domain-specific exceptions
"""


class DomainException(Exception):
    """Base domain exception"""
    pass


class ValidationError(DomainException):
    """Validation error"""
    pass


class NotFoundError(DomainException):
    """Entity not found error"""
    pass


class UnauthorizedError(DomainException):
    """Unauthorized access error"""
    pass


class ForbiddenError(DomainException):
    """Forbidden access error"""
    pass


class ConflictError(DomainException):
    """Resource conflict error"""
    pass


class DatabaseConnectionError(DomainException):
    """Database connection error"""
    pass


class TenantNotFoundError(NotFoundError):
    """Tenant not found error"""
    pass


class TenantAccessDeniedError(ForbiddenError):
    """Tenant access denied error"""
    pass


class RateLimitExceededError(DomainException):
    """Rate limit exceeded error"""
    pass


class SecurityViolationError(DomainException):
    """Security violation error"""
    pass


class InvalidCredentials(UnauthorizedError):
    """Invalid credentials error"""
    pass


class UserNotFound(NotFoundError):
    """User not found error"""
    pass


class InvalidToken(UnauthorizedError):
    """Invalid token error"""
    pass


class TokenExpired(UnauthorizedError):
    """Token expired error"""
    pass


class AccountLocked(ForbiddenError):
    """Account locked error"""
    pass


class SecurityViolation(SecurityViolationError):
    """Security violation alias"""
    pass


class EmbeddingGenerationFailed(DomainException):
    """Embedding generation failed error"""
    pass


class ExternalServiceError(DomainException):
    """External service error"""
    pass


class WorkflowExecutionError(DomainException):
    """Workflow execution error"""
    pass


class ServiceUnavailable(DomainException):
    """Service unavailable error"""
    pass


class ResourceLimitExceeded(DomainException):
    """Resource limit exceeded error"""
    pass


class RateLimitExceeded(RateLimitExceededError):
    """Rate limit exceeded alias"""
    pass


class MFARequired(UnauthorizedError):
    """MFA required error"""
    pass


class InvalidMFACode(UnauthorizedError):
    """Invalid MFA code error"""
    pass


class WorkflowExecutionFailed(WorkflowExecutionError):
    """Workflow execution failed alias"""
    pass


class ResourceNotFound(NotFoundError):
    """Resource not found alias"""
    pass