"""
Domain exceptions - Business logic exceptions that express domain concepts.
"""


class DomainException(Exception):
    """Base domain exception"""
    def __init__(self, message: str, code: str = None):
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__


class ValidationError(DomainException):
    """Raised when domain validation fails"""
    pass


class BusinessRuleViolation(DomainException):
    """Raised when a business rule is violated"""
    pass


class ResourceNotFound(DomainException):
    """Raised when a requested resource is not found"""
    pass


class UnauthorizedAccess(DomainException):
    """Raised when access is unauthorized"""
    pass


class RateLimitExceeded(DomainException):
    """Raised when rate limit is exceeded"""
    def __init__(self, message: str, limit: int, remaining: int, reset_time: int):
        super().__init__(message)
        self.limit = limit
        self.remaining = remaining
        self.reset_time = reset_time


class InvalidCredentials(DomainException):
    """Raised when authentication credentials are invalid"""
    pass


class TokenExpired(DomainException):
    """Raised when an auth token has expired"""
    pass


class InsufficientPermissions(DomainException):
    """Raised when user lacks required permissions"""
    def __init__(self, message: str, required_role: str = None):
        super().__init__(message)
        self.required_role = required_role


class EmbeddingGenerationFailed(DomainException):
    """Raised when embedding generation fails"""
    def __init__(self, message: str, model: str = None):
        super().__init__(message)
        self.model = model


class WorkflowExecutionFailed(DomainException):
    """Raised when workflow execution fails"""
    def __init__(self, message: str, workflow_id: str = None):
        super().__init__(message)
        self.workflow_id = workflow_id


class InvalidInput(DomainException):
    """Raised when input data is invalid"""
    def __init__(self, message: str, field: str = None):
        super().__init__(message)
        self.field = field


class ServiceUnavailable(DomainException):
    """Raised when an external service is unavailable"""
    def __init__(self, message: str, service: str = None):
        super().__init__(message)
        self.service = service


class ConfigurationError(DomainException):
    """Raised when there's a configuration issue"""
    pass


class ResourceLimitExceeded(DomainException):
    """Raised when resource limits are exceeded"""
    def __init__(self, message: str, resource_type: str = None, limit: int = None):
        super().__init__(message)
        self.resource_type = resource_type
        self.limit = limit


class AccountLocked(DomainException):
    """Raised when user account is temporarily locked"""
    def __init__(self, message: str, lockout_duration: int = None):
        super().__init__(message)
        self.lockout_duration = lockout_duration


class SecurityViolation(DomainException):
    """Raised when a security violation is detected"""
    def __init__(self, message: str, violation_type: str = None):
        super().__init__(message)
        self.violation_type = violation_type


class MFARequired(DomainException):
    """Raised when multi-factor authentication is required"""
    def __init__(self, message: str, mfa_methods: list = None):
        super().__init__(message)
        self.mfa_methods = mfa_methods or []


class PasswordExpired(DomainException):
    """Raised when user password has expired"""
    def __init__(self, message: str, expired_days: int = None):
        super().__init__(message)
        self.expired_days = expired_days