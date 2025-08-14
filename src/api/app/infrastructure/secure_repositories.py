"""
Secure Repository Base Classes
Production-grade repository implementations with mandatory tenant isolation
"""

import logging
from typing import List, Optional, Dict, Any, Type, TypeVar, Generic
from uuid import UUID
from abc import ABC, abstractmethod
from dataclasses import dataclass

from sqlalchemy import select, update, delete, and_, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.orm import declarative_base

from ..core.secure_tenant_context import TenantContext, SecureTenantContextManager
from ..core.logging import get_logger
from ..domain.tenant_entities import Tenant

logger = get_logger(__name__)

# Type variables for generic repositories
ModelType = TypeVar('ModelType')
EntityType = TypeVar('EntityType')


@dataclass
class QuerySecurityContext:
    """Security context for database queries"""
    tenant_id: UUID
    user_id: str
    operation: str  # 'read', 'write', 'delete'
    table_name: str
    additional_filters: Dict[str, Any] = None


class SecureRepositoryBase(Generic[ModelType, EntityType], ABC):
    """
    Base class for tenant-aware repositories with security enforcement
    
    All repositories MUST inherit from this class to ensure tenant isolation
    """
    
    def __init__(
        self, 
        session: AsyncSession,
        tenant_context: TenantContext,
        model_class: Type[ModelType]
    ):
        self.session = session
        self.tenant_context = tenant_context
        self.model_class = model_class
        self.table_name = model_class.__tablename__
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    
    def _validate_tenant_context(self) -> None:
        """Validate that tenant context is valid and current"""
        if not self.tenant_context.is_valid():
            raise ValueError("Tenant context expired or invalid")
    
    def _create_security_context(self, operation: str) -> QuerySecurityContext:
        """Create security context for query"""
        return QuerySecurityContext(
            tenant_id=self.tenant_context.tenant_id,
            user_id=self.tenant_context.user_id,
            operation=operation,
            table_name=self.table_name
        )
    
    async def _ensure_rls_context(self) -> None:
        """Ensure Row Level Security context is properly set"""
        try:
            # Verify tenant context is set in database session
            result = await self.session.execute(
                text("SELECT current_setting('app.tenant_id', true)")
            )
            current_tenant = result.scalar()
            
            if not current_tenant or current_tenant != str(self.tenant_context.tenant_id):
                # Re-establish context
                await self.session.execute(
                    text("SELECT set_config('app.tenant_id', :tenant_id, true)"),
                    {"tenant_id": str(self.tenant_context.tenant_id)}
                )
                
                # Log context reset
                self.logger.warning(
                    f"Database tenant context reset for table {self.table_name}"
                )
                
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to ensure RLS context: {e}")
            raise ValueError("Database security context verification failed")
    
    def _build_tenant_filter(self):
        """Build tenant filter for queries"""
        return self.model_class.tenant_id == self.tenant_context.tenant_id
    
    async def _log_data_access(
        self, 
        operation: str, 
        record_count: int = 1, 
        additional_info: Dict[str, Any] = None
    ) -> None:
        """Log data access for audit trail"""
        log_data = {
            "operation": operation,
            "table": self.table_name,
            "tenant_id": str(self.tenant_context.tenant_id),
            "user_id": self.tenant_context.user_id,
            "record_count": record_count,
            "timestamp": self.tenant_context.validated_at.isoformat()
        }
        
        if additional_info:
            log_data.update(additional_info)
        
        self.logger.info(f"Data access: {log_data}")
    
    async def get_by_id(self, entity_id: UUID) -> Optional[EntityType]:
        """Get entity by ID with tenant isolation"""
        self._validate_tenant_context()
        await self._ensure_rls_context()
        
        try:
            stmt = select(self.model_class).where(
                and_(
                    self.model_class.id == entity_id,
                    self._build_tenant_filter()
                )
            )
            
            result = await self.session.execute(stmt)
            model = result.scalar_one_or_none()
            
            if model:
                await self._log_data_access("read", 1, {"entity_id": str(entity_id)})
                return self._model_to_entity(model)
            
            return None
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error in get_by_id: {e}")
            raise
    
    async def list_entities(
        self, 
        filters: Dict[str, Any] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[EntityType]:
        """List entities with tenant isolation and filters"""
        self._validate_tenant_context()
        await self._ensure_rls_context()
        
        try:
            # Start with tenant filter
            conditions = [self._build_tenant_filter()]
            
            # Add additional filters
            if filters:
                for field, value in filters.items():
                    if hasattr(self.model_class, field):
                        conditions.append(getattr(self.model_class, field) == value)
            
            stmt = (
                select(self.model_class)
                .where(and_(*conditions))
                .limit(limit)
                .offset(offset)
            )
            
            result = await self.session.execute(stmt)
            models = result.scalars().all()
            
            entities = [self._model_to_entity(model) for model in models]
            
            await self._log_data_access(
                "read", 
                len(entities), 
                {"filters": filters, "limit": limit, "offset": offset}
            )
            
            return entities
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error in list_entities: {e}")
            raise
    
    async def create(self, entity_data: Dict[str, Any]) -> EntityType:
        """Create new entity with tenant isolation"""
        self._validate_tenant_context()
        await self._ensure_rls_context()
        
        try:
            # Ensure tenant_id is set
            entity_data['tenant_id'] = self.tenant_context.tenant_id
            
            # Create model instance
            model = self.model_class(**entity_data)
            
            self.session.add(model)
            await self.session.flush()  # Get ID without committing
            
            await self._log_data_access(
                "create", 
                1, 
                {"entity_id": str(model.id) if hasattr(model, 'id') else None}
            )
            
            return self._model_to_entity(model)
            
        except IntegrityError as e:
            await self.session.rollback()
            self.logger.error(f"Integrity error creating entity: {e}")
            raise ValueError("Entity creation failed - constraint violation")
        except SQLAlchemyError as e:
            await self.session.rollback()
            self.logger.error(f"Database error creating entity: {e}")
            raise
    
    async def update(
        self, 
        entity_id: UUID, 
        update_data: Dict[str, Any]
    ) -> Optional[EntityType]:
        """Update entity with tenant isolation"""
        self._validate_tenant_context()
        await self._ensure_rls_context()
        
        try:
            # Prevent tenant_id modification
            if 'tenant_id' in update_data:
                del update_data['tenant_id']
                self.logger.warning(
                    f"Attempted tenant_id modification blocked for entity {entity_id}"
                )
            
            # Use parameterized update to prevent injection
            stmt = (
                update(self.model_class)
                .where(
                    and_(
                        self.model_class.id == entity_id,
                        self._build_tenant_filter()
                    )
                )
                .values(**update_data)
                .returning(self.model_class)
            )
            
            result = await self.session.execute(stmt)
            model = result.scalar_one_or_none()
            
            if model:
                await self._log_data_access(
                    "update", 
                    1, 
                    {"entity_id": str(entity_id), "fields": list(update_data.keys())}
                )
                return self._model_to_entity(model)
            
            return None
            
        except SQLAlchemyError as e:
            await self.session.rollback()
            self.logger.error(f"Database error updating entity: {e}")
            raise
    
    async def delete(self, entity_id: UUID) -> bool:
        """Delete entity with tenant isolation"""
        self._validate_tenant_context()
        await self._ensure_rls_context()
        
        try:
            # Use parameterized delete to prevent injection
            stmt = delete(self.model_class).where(
                and_(
                    self.model_class.id == entity_id,
                    self._build_tenant_filter()
                )
            )
            
            result = await self.session.execute(stmt)
            
            if result.rowcount > 0:
                await self._log_data_access(
                    "delete", 
                    1, 
                    {"entity_id": str(entity_id)}
                )
                return True
            
            return False
            
        except SQLAlchemyError as e:
            await self.session.rollback()
            self.logger.error(f"Database error deleting entity: {e}")
            raise
    
    async def count(self, filters: Dict[str, Any] = None) -> int:
        """Count entities with tenant isolation"""
        self._validate_tenant_context()
        await self._ensure_rls_context()
        
        try:
            # Start with tenant filter
            conditions = [self._build_tenant_filter()]
            
            # Add additional filters
            if filters:
                for field, value in filters.items():
                    if hasattr(self.model_class, field):
                        conditions.append(getattr(self.model_class, field) == value)
            
            stmt = select(self.model_class.id).where(and_(*conditions))
            result = await self.session.execute(stmt)
            
            count = len(result.all())
            
            await self._log_data_access(
                "count", 
                count, 
                {"filters": filters}
            )
            
            return count
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error counting entities: {e}")
            raise
    
    async def execute_secure_query(
        self, 
        query: str, 
        params: Dict[str, Any],
        operation: str = "read"
    ) -> Any:
        """
        Execute raw SQL query with security validation
        
        IMPORTANT: Only use this for complex queries that cannot be expressed
        with the ORM. All queries MUST include tenant isolation.
        """
        self._validate_tenant_context()
        await self._ensure_rls_context()
        
        # Validate query contains tenant isolation
        if "tenant_id" not in query.lower():
            raise ValueError("Raw queries must include tenant_id filtering")
        
        # Ensure tenant_id parameter is set
        params['tenant_id'] = str(self.tenant_context.tenant_id)
        
        try:
            result = await self.session.execute(text(query), params)
            
            await self._log_data_access(
                f"raw_{operation}",
                1,
                {"query_hash": hash(query), "param_count": len(params)}
            )
            
            return result
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error in secure query: {e}")
            raise
    
    @abstractmethod
    def _model_to_entity(self, model: ModelType) -> EntityType:
        """Convert database model to domain entity"""
        pass
    
    @abstractmethod
    def _entity_to_model(self, entity: EntityType) -> ModelType:
        """Convert domain entity to database model"""
        pass


class SecureTenantRepository(SecureRepositoryBase[Tenant, Tenant]):
    """Secure repository for tenant management"""
    
    def __init__(self, session: AsyncSession, tenant_context: TenantContext):
        from ..domain.tenant_entities import Tenant as TenantModel
        super().__init__(session, tenant_context, TenantModel)
    
    async def get_by_slug(self, slug: str) -> Optional[Tenant]:
        """Get tenant by slug with validation"""
        self._validate_tenant_context()
        
        try:
            stmt = select(self.model_class).where(
                and_(
                    self.model_class.slug == slug,
                    self._build_tenant_filter()
                )
            )
            
            result = await self.session.execute(stmt)
            model = result.scalar_one_or_none()
            
            if model:
                await self._log_data_access("read", 1, {"slug": slug})
                return self._model_to_entity(model)
            
            return None
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error getting tenant by slug: {e}")
            raise
    
    def _model_to_entity(self, model: Tenant) -> Tenant:
        """Convert model to entity (same type in this case)"""
        return model
    
    def _entity_to_model(self, entity: Tenant) -> Tenant:
        """Convert entity to model (same type in this case)"""
        return entity


def create_secure_repository(
    repository_class: Type,
    session: AsyncSession,
    tenant_context: TenantContext
):
    """
    Factory function to create secure repository instances
    
    Args:
        repository_class: Repository class to instantiate
        session: Database session with tenant context
        tenant_context: Validated tenant context
        
    Returns:
        Repository instance with security enforced
    """
    if not issubclass(repository_class, SecureRepositoryBase):
        raise ValueError(
            f"Repository {repository_class.__name__} must inherit from SecureRepositoryBase"
        )
    
    return repository_class(session, tenant_context)