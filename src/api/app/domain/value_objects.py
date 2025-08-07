"""
Domain value objects - Immutable objects that describe characteristics of entities.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum


class PlanType(Enum):
    """Available subscription plans"""
    GROWTH = "Growth"
    PRO = "Pro"
    ENTERPRISE = "Enterprise"


class UserRole(Enum):
    """Available user roles"""
    ADMIN = "admin"
    USER = "user"
    READER = "reader"


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"


class EmbeddingModel(Enum):
    """Available embedding models"""
    NVIDIA_EMBED_QA_4 = "nvidia/embed-qa-4"


class InputType(Enum):
    """Input types for embeddings"""
    QUERY = "query"
    PASSAGE = "passage"
    CLASSIFICATION = "classification"


@dataclass(frozen=True)
class Email:
    """Email value object"""
    value: str
    
    def __post_init__(self):
        if not self._is_valid_email(self.value):
            raise ValueError(f"Invalid email format: {self.value}")
    
    def _is_valid_email(self, email: str) -> bool:
        """Basic email validation"""
        return "@" in email and "." in email.split("@")[1]


@dataclass(frozen=True)
class Username:
    """Username value object"""
    value: str
    
    def __post_init__(self):
        if not (3 <= len(self.value) <= 50):
            raise ValueError("Username must be between 3 and 50 characters")
        
        if not self.value.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Username can only contain letters, numbers, hyphens and underscores")


@dataclass(frozen=True)
class Domain:
    """Domain name value object"""
    value: str
    
    def __post_init__(self):
        if not self._is_valid_domain(self.value):
            raise ValueError(f"Invalid domain format: {self.value}")
    
    def _is_valid_domain(self, domain: str) -> bool:
        """Basic domain validation"""
        return "." in domain and len(domain) > 3 and not domain.startswith(".")


@dataclass(frozen=True)
class EmbeddingVector:
    """Embedding vector value object"""
    values: List[float]
    dimension: int
    
    def __post_init__(self):
        if len(self.values) != self.dimension:
            raise ValueError(f"Vector length {len(self.values)} doesn't match dimension {self.dimension}")
        
        if not all(isinstance(v, (int, float)) for v in self.values):
            raise ValueError("All vector values must be numeric")


@dataclass(frozen=True)
class UsageStats:
    """Usage statistics value object"""
    prompt_tokens: int
    total_tokens: int
    processing_time_ms: int
    
    def __post_init__(self):
        if self.prompt_tokens < 0 or self.total_tokens < 0:
            raise ValueError("Token counts cannot be negative")
        
        if self.processing_time_ms < 0:
            raise ValueError("Processing time cannot be negative")


@dataclass(frozen=True)
class RateLimitInfo:
    """Rate limit information value object"""
    limit: int
    remaining: int
    reset_time: int
    resource_type: str
    
    def __post_init__(self):
        if self.limit < 0 or self.remaining < 0:
            raise ValueError("Limit values cannot be negative")
        
        if self.remaining > self.limit:
            raise ValueError("Remaining cannot be greater than limit")


@dataclass(frozen=True)
class ApiResponse:
    """Generic API response value object"""
    data: Any
    status_code: int = 200
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.headers is None:
            object.__setattr__(self, 'headers', {})
        
        if not (100 <= self.status_code <= 599):
            raise ValueError("Invalid HTTP status code")


@dataclass(frozen=True)
class ErrorDetails:
    """Error details value object"""
    code: str
    message: str
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            object.__setattr__(self, 'details', {})


@dataclass(frozen=True)
class PaginationInfo:
    """Pagination information value object"""
    page: int
    size: int
    total: int
    
    def __post_init__(self):
        if self.page < 1:
            raise ValueError("Page must be >= 1")
        
        if self.size < 1:
            raise ValueError("Size must be >= 1")
        
        if self.total < 0:
            raise ValueError("Total cannot be negative")
    
    @property
    def total_pages(self) -> int:
        """Calculate total pages"""
        return (self.total + self.size - 1) // self.size
    
    @property
    def has_next(self) -> bool:
        """Check if there are more pages"""
        return self.page < self.total_pages
    
    @property
    def has_previous(self) -> bool:
        """Check if there are previous pages"""
        return self.page > 1