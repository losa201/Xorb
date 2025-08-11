"""Comprehensive input validation and sanitization."""
import re
import html
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from datetime import datetime

from pydantic import BaseModel, validator, Field
from fastapi import HTTPException, status
import bleach

from ..middleware.error_handling import ValidationError


class BaseValidator:
    """Base class for input validators."""
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            raise ValidationError("Value must be a string")
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Limit length
        if len(value) > max_length:
            raise ValidationError(f"String too long (max {max_length} characters)")
        
        # HTML escape
        value = html.escape(value)
        
        return value.strip()
    
    @staticmethod
    def validate_uuid(value: Union[str, UUID]) -> UUID:
        """Validate UUID format."""
        if isinstance(value, UUID):
            return value
        
        try:
            return UUID(str(value))
        except (ValueError, TypeError):
            raise ValidationError(f"Invalid UUID format: {value}")
    
    @staticmethod
    def validate_email(email: str) -> str:
        """Validate email format."""
        email = BaseValidator.sanitize_string(email, 254)  # RFC 5321 limit
        
        # Basic email regex (RFC 5322 compliant)
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(pattern, email):
            raise ValidationError("Invalid email format")
        
        return email.lower()
    
    @staticmethod
    def validate_filename(filename: str) -> str:
        """Validate and sanitize filename."""
        filename = BaseValidator.sanitize_string(filename, 255)
        
        # Remove path separators
        filename = filename.replace('/', '').replace('\\', '')
        
        # Remove dangerous characters
        dangerous_chars = '<>:"|?*'
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Prevent reserved names (Windows)
        reserved_names = [
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        ]
        
        name_part = filename.split('.')[0].upper()
        if name_part in reserved_names:
            raise ValidationError(f"Reserved filename: {filename}")
        
        if not filename or filename in ['.', '..']:
            raise ValidationError("Invalid filename")
        
        return filename
    
    @staticmethod
    def validate_content_type(content_type: str, allowed_types: Optional[List[str]] = None) -> str:
        """Validate MIME content type."""
        content_type = BaseValidator.sanitize_string(content_type, 100)
        
        # Basic MIME type format validation
        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9!#$&\-\^_]*\/[a-zA-Z0-9][a-zA-Z0-9!#$&\-\^_.]*$', content_type):
            raise ValidationError("Invalid content type format")
        
        if allowed_types and content_type not in allowed_types:
            raise ValidationError(f"Content type not allowed: {content_type}")
        
        return content_type
    
    @staticmethod
    def validate_file_size(size: int, max_size: int = 100 * 1024 * 1024) -> int:
        """Validate file size."""
        if not isinstance(size, int) or size < 0:
            raise ValidationError("File size must be a positive integer")
        
        if size > max_size:
            raise ValidationError(f"File size exceeds maximum: {max_size} bytes")
        
        return size


class TextSanitizer:
    """Advanced text sanitization."""
    
    # Allowed HTML tags for rich text (if needed)
    ALLOWED_TAGS = [
        'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'code', 'pre'
    ]
    
    ALLOWED_ATTRIBUTES = {
        '*': ['class'],
        'a': ['href', 'title'],
        'img': ['src', 'alt', 'title', 'width', 'height']
    }
    
    @classmethod
    def sanitize_html(cls, text: str, strip_tags: bool = True) -> str:
        """Sanitize HTML content."""
        if strip_tags:
            # Strip all HTML tags
            return bleach.clean(text, tags=[], strip=True)
        else:
            # Clean but preserve allowed tags
            return bleach.clean(
                text,
                tags=cls.ALLOWED_TAGS,
                attributes=cls.ALLOWED_ATTRIBUTES,
                strip=True
            )
    
    @classmethod
    def sanitize_search_query(cls, query: str) -> str:
        """Sanitize search query input."""
        query = BaseValidator.sanitize_string(query, 500)
        
        # Remove dangerous search operators
        dangerous_patterns = [
            r'[;\'"\\]',  # SQL injection attempts
            r'<script',   # XSS attempts
            r'javascript:', # JavaScript injection
            r'data:',     # Data URLs
        ]
        
        for pattern in dangerous_patterns:
            query = re.sub(pattern, '', query, flags=re.IGNORECASE)
        
        return query.strip()
    
    @classmethod
    def sanitize_json_field(cls, value: Any) -> Any:
        """Sanitize JSON field values."""
        if isinstance(value, str):
            return cls.sanitize_html(value, strip_tags=True)
        elif isinstance(value, dict):
            return {k: cls.sanitize_json_field(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [cls.sanitize_json_field(item) for item in value]
        else:
            return value


class SecurityValidator:
    """Security-focused validation."""
    
    @staticmethod
    def validate_sql_injection(value: str) -> str:
        """Check for potential SQL injection patterns."""
        dangerous_patterns = [
            r"(?i)(union|select|insert|update|delete|drop|create|alter|exec|execute)",
            r"(?i)(script|javascript|vbscript|onload|onerror)",
            r"(?i)(or\s+1\s*=\s*1|and\s+1\s*=\s*1)",
            r"[;'\"\\]"
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, value):
                raise ValidationError("Potentially dangerous input detected")
        
        return value
    
    @staticmethod
    def validate_xss_attempt(value: str) -> str:
        """Check for potential XSS patterns."""
        xss_patterns = [
            r"(?i)<script",
            r"(?i)javascript:",
            r"(?i)vbscript:",
            r"(?i)onload\s*=",
            r"(?i)onerror\s*=",
            r"(?i)onclick\s*=",
            r"(?i)data:text/html"
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, value):
                raise ValidationError("Potentially dangerous input detected")
        
        return value
    
    @staticmethod
    def validate_path_traversal(path: str) -> str:
        """Check for path traversal attempts."""
        dangerous_patterns = [
            r"\.\./",
            r"\.\.\\",
            r"%2e%2e%2f",
            r"%2e%2e%5c"
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, path, re.IGNORECASE):
                raise ValidationError("Path traversal attempt detected")
        
        return path


# Pydantic models with validation
class ValidatedUUID(BaseModel):
    """UUID with validation."""
    value: UUID
    
    @validator('value', pre=True)
    def validate_uuid_format(cls, v):
        return BaseValidator.validate_uuid(v)


class ValidatedEmail(BaseModel):
    """Email with validation."""
    value: str
    
    @validator('value', pre=True)
    def validate_email_format(cls, v):
        return BaseValidator.validate_email(v)


class ValidatedFilename(BaseModel):
    """Filename with validation."""
    value: str
    
    @validator('value', pre=True)
    def validate_filename_format(cls, v):
        return BaseValidator.validate_filename(v)


class SafeString(BaseModel):
    """String with security validation."""
    value: str = Field(..., max_length=1000)
    
    @validator('value', pre=True)
    def validate_safe_string(cls, v):
        if not isinstance(v, str):
            raise ValueError("Value must be a string")
        
        # Apply security validation
        v = SecurityValidator.validate_sql_injection(v)
        v = SecurityValidator.validate_xss_attempt(v)
        
        # Sanitize
        return BaseValidator.sanitize_string(v)


class ValidatedSearchQuery(BaseModel):
    """Search query with validation."""
    query: str = Field(..., min_length=1, max_length=500)
    
    @validator('query', pre=True)
    def validate_search_query(cls, v):
        return TextSanitizer.sanitize_search_query(v)


class FileUploadValidation(BaseModel):
    """File upload validation model."""
    filename: str
    content_type: str
    size: int
    
    @validator('filename', pre=True)
    def validate_filename(cls, v):
        return BaseValidator.validate_filename(v)
    
    @validator('content_type', pre=True)
    def validate_content_type(cls, v):
        allowed_types = [
            'application/pdf', 'text/plain', 'text/csv',
            'image/jpeg', 'image/png', 'image/gif',
            'application/zip', 'application/json'
        ]
        return BaseValidator.validate_content_type(v, allowed_types)
    
    @validator('size', pre=True)
    def validate_file_size(cls, v):
        return BaseValidator.validate_file_size(v, max_size=100 * 1024 * 1024)


# Validation decorators
def validate_input(validation_class: BaseModel):
    """Decorator to validate input using Pydantic model."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # This would be used with FastAPI dependency injection
            # The actual validation happens in FastAPI route parameters
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Request validation helpers
def validate_pagination(
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    max_limit: int = 100
) -> tuple[int, int]:
    """Validate pagination parameters."""
    if limit is None:
        limit = 20
    elif limit < 1 or limit > max_limit:
        raise ValidationError(f"Limit must be between 1 and {max_limit}")
    
    if offset is None:
        offset = 0
    elif offset < 0:
        raise ValidationError("Offset must be non-negative")
    
    return limit, offset


def validate_date_range(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    max_range_days: int = 365
) -> tuple[Optional[datetime], Optional[datetime]]:
    """Validate date range parameters."""
    if start_date and end_date:
        if start_date >= end_date:
            raise ValidationError("Start date must be before end date")
        
        range_days = (end_date - start_date).days
        if range_days > max_range_days:
            raise ValidationError(f"Date range cannot exceed {max_range_days} days")
    
    return start_date, end_date


def validate_sort_params(
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = None,
    allowed_fields: Optional[List[str]] = None
) -> tuple[Optional[str], str]:
    """Validate sorting parameters."""
    if sort_order is None:
        sort_order = "desc"
    elif sort_order.lower() not in ["asc", "desc"]:
        raise ValidationError("Sort order must be 'asc' or 'desc'")
    
    if sort_by and allowed_fields and sort_by not in allowed_fields:
        raise ValidationError(f"Sort field must be one of: {', '.join(allowed_fields)}")
    
    return sort_by, sort_order.lower()