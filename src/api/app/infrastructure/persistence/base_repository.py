"""
Base Repository - Clean Architecture Infrastructure
Abstract base repository implementing repository pattern
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Dict, Any
from uuid import UUID

T = TypeVar('T')


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base repository implementing repository pattern.
    Provides common CRUD operations with clean architecture boundaries.
    """
    
    @abstractmethod
    async def create(self, entity: T) -> T:
        """Create a new entity"""
        pass
    
    @abstractmethod
    async def get_by_id(self, entity_id: UUID) -> Optional[T]:
        """Get entity by ID"""
        pass
    
    @abstractmethod
    async def update(self, entity: T) -> T:
        """Update existing entity"""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: UUID) -> bool:
        """Delete entity by ID"""
        pass
    
    @abstractmethod
    async def list_all(
        self, 
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[T]:
        """List entities with optional pagination and filtering"""
        pass
    
    @abstractmethod
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities with optional filtering"""
        pass
    
    @abstractmethod
    async def exists(self, entity_id: UUID) -> bool:
        """Check if entity exists"""
        pass


class AsyncReadOnlyRepository(ABC, Generic[T]):
    """
    Read-only repository interface for query-only access patterns.
    Useful for reporting and analytics use cases.
    """
    
    @abstractmethod
    async def get_by_id(self, entity_id: UUID) -> Optional[T]:
        """Get entity by ID"""
        pass
    
    @abstractmethod
    async def find_by_criteria(self, criteria: Dict[str, Any]) -> List[T]:
        """Find entities by criteria"""
        pass
    
    @abstractmethod
    async def search(
        self, 
        query: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[T]:
        """Search entities by text query"""
        pass


class UnitOfWork(ABC):
    """
    Unit of Work pattern for managing transactions across multiple repositories.
    Ensures ACID properties and maintains consistency boundaries.
    """
    
    @abstractmethod
    async def __aenter__(self):
        """Enter async context - begin transaction"""
        pass
    
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context - commit or rollback"""
        pass
    
    @abstractmethod
    async def commit(self):
        """Commit the transaction"""
        pass
    
    @abstractmethod
    async def rollback(self):
        """Rollback the transaction"""
        pass
    
    @abstractmethod
    def get_repository(self, repository_type: type) -> BaseRepository:
        """Get repository instance within transaction context"""
        pass