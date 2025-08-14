"""
Enhanced PostgreSQL and Redis Repository Implementations
Production-ready implementations replacing stubs with full functionality
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_, func
from sqlalchemy.orm import selectinload
from sqlalchemy.exc import IntegrityError, NoResultFound

# Redis imports with graceful fallback
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    try:
        import redis as redis_sync
        aioredis = None
        REDIS_AVAILABLE = False
    except ImportError:
        REDIS_AVAILABLE = False

from ..domain.entities import (
    User, Organization, EmbeddingRequest, EmbeddingResult,
    DiscoveryWorkflow, AuthToken
)
from ..domain.repositories import (
    UserRepository, OrganizationRepository, EmbeddingRepository,
    DiscoveryRepository, AuthTokenRepository, CacheRepository,
    ScanSessionRepository, TenantRepository
)
from .database_models import (
    UserModel, OrganizationModel, EmbeddingRequestModel, EmbeddingResultModel,
    DiscoveryWorkflowModel, AuthTokenModel, UserOrganizationModel,
    TenantModel, ScanSessionModel, SecurityFindingModel
)

logger = logging.getLogger(__name__)


class EnhancedPostgreSQLScanSessionRepository(ScanSessionRepository):
    """Enhanced PostgreSQL-backed scan session repository with full functionality"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new scan session with full metadata support"""
        try:
            session_model = ScanSessionModel(
                session_id=session_data.get("session_id", str(uuid.uuid4())),
                user_id=session_data.get("user_id"),
                tenant_id=session_data.get("tenant_id"),
                scan_type=session_data.get("scan_type", "comprehensive"),
                status=session_data.get("status", "queued"),
                targets=json.dumps(session_data.get("targets", [])),
                scan_config=json.dumps(session_data.get("scan_config", {})),
                metadata=json.dumps(session_data.get("metadata", {})),
                priority=session_data.get("priority", "medium"),
                estimated_duration=session_data.get("estimated_duration", 1800),
                created_at=datetime.utcnow()
            )

            self.session.add(session_model)
            await self.session.commit()
            await self.session.refresh(session_model)

            return self._model_to_dict(session_model)

        except IntegrityError as e:
            await self.session.rollback()
            logger.error(f"Scan session creation failed - integrity error: {e}")
            raise ValueError("Scan session with this ID already exists")
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error creating scan session: {e}")
            raise

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get scan session by ID with full details"""
        try:
            stmt = select(ScanSessionModel).where(ScanSessionModel.session_id == session_id)
            result = await self.session.execute(stmt)
            session_model = result.scalar_one_or_none()

            if session_model:
                return self._model_to_dict(session_model)
            return None

        except Exception as e:
            logger.error(f"Error getting scan session {session_id}: {e}")
            return None

    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update scan session with comprehensive update support"""
        try:
            # Prepare update data
            update_data = {}

            # Handle status updates
            if 'status' in updates:
                update_data['status'] = updates['status']
                if updates['status'] == 'running' and 'started_at' not in updates:
                    update_data['started_at'] = datetime.utcnow()
                elif updates['status'] in ['completed', 'failed', 'cancelled']:
                    update_data['completed_at'] = datetime.utcnow()

            # Handle progress updates
            if 'progress' in updates:
                update_data['progress'] = updates['progress']

            # Handle results
            if 'results' in updates:
                update_data['results'] = json.dumps(updates['results'])

            # Handle error information
            if 'error_message' in updates:
                update_data['error_message'] = updates['error_message']

            # Handle metadata updates
            if 'metadata' in updates:
                existing_session = await self.get_session(session_id)
                if existing_session:
                    existing_metadata = existing_session.get('metadata', {})
                    existing_metadata.update(updates['metadata'])
                    update_data['metadata'] = json.dumps(existing_metadata)
                else:
                    update_data['metadata'] = json.dumps(updates['metadata'])

            update_data['updated_at'] = datetime.utcnow()

            stmt = update(ScanSessionModel).where(
                ScanSessionModel.session_id == session_id
            ).values(**update_data)

            result = await self.session.execute(stmt)
            await self.session.commit()

            return result.rowcount > 0

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error updating scan session {session_id}: {e}")
            return False

    async def get_user_sessions(
        self,
        user_id: UUID,
        limit: int = 50,
        offset: int = 0,
        status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get scan sessions for a user with filtering and pagination"""
        try:
            stmt = select(ScanSessionModel).where(ScanSessionModel.user_id == user_id)

            if status_filter:
                stmt = stmt.where(ScanSessionModel.status == status_filter)

            stmt = stmt.order_by(ScanSessionModel.created_at.desc()).offset(offset).limit(limit)

            result = await self.session.execute(stmt)
            sessions = result.scalars().all()

            return [self._model_to_dict(session) for session in sessions]

        except Exception as e:
            logger.error(f"Error getting user sessions for {user_id}: {e}")
            return []

    async def get_active_sessions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all active scan sessions"""
        try:
            stmt = select(ScanSessionModel).where(
                ScanSessionModel.status.in_(["pending", "running", "queued"])
            )
            if limit:
                stmt = stmt.limit(limit)

            result = await self.session.execute(stmt)
            sessions = result.scalars().all()

            return [self._model_to_dict(session) for session in sessions]

        except Exception as e:
            logger.error(f"Error getting active sessions: {e}")
            return []

    async def get_scan_statistics(self, tenant_id: Optional[UUID] = None) -> Dict[str, Any]:
        """Get comprehensive scan statistics"""
        try:
            base_query = select(ScanSessionModel)

            if tenant_id:
                base_query = base_query.where(ScanSessionModel.tenant_id == tenant_id)

            # Total scans
            total_result = await self.session.execute(
                select(func.count()).select_from(base_query.subquery())
            )
            total_scans = total_result.scalar()

            # Status breakdown
            status_query = select(
                ScanSessionModel.status,
                func.count(ScanSessionModel.status)
            )
            if tenant_id:
                status_query = status_query.where(ScanSessionModel.tenant_id == tenant_id)

            status_result = await self.session.execute(
                status_query.group_by(ScanSessionModel.status)
            )
            status_counts = dict(status_result.all())

            # Recent activity (last 24 hours)
            recent_cutoff = datetime.utcnow() - timedelta(hours=24)
            recent_query = select(func.count())
            if tenant_id:
                recent_query = recent_query.where(
                    and_(
                        ScanSessionModel.tenant_id == tenant_id,
                        ScanSessionModel.created_at >= recent_cutoff
                    )
                )
            else:
                recent_query = recent_query.where(
                    ScanSessionModel.created_at >= recent_cutoff
                )

            recent_result = await self.session.execute(recent_query)
            recent_scans = recent_result.scalar()

            # Average duration for completed scans
            duration_query = select(
                func.avg(
                    func.extract('epoch', ScanSessionModel.completed_at - ScanSessionModel.started_at)
                )
            ).where(
                and_(
                    ScanSessionModel.status == 'completed',
                    ScanSessionModel.started_at.isnot(None),
                    ScanSessionModel.completed_at.isnot(None)
                )
            )

            if tenant_id:
                duration_query = duration_query.where(ScanSessionModel.tenant_id == tenant_id)

            duration_result = await self.session.execute(duration_query)
            avg_duration_seconds = duration_result.scalar() or 0

            return {
                "total_scans": total_scans,
                "status_breakdown": status_counts,
                "recent_activity_24h": recent_scans,
                "avg_duration_minutes": round(avg_duration_seconds / 60, 2),
                "active_scans": status_counts.get("running", 0) + status_counts.get("queued", 0),
                "completed_scans": status_counts.get("completed", 0),
                "failed_scans": status_counts.get("failed", 0),
                "generated_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting scan statistics: {e}")
            return {
                "total_scans": 0,
                "status_breakdown": {},
                "recent_activity_24h": 0,
                "avg_duration_minutes": 0,
                "generated_at": datetime.utcnow().isoformat(),
                "error": str(e)
            }

    async def delete_scan_session(self, session_id: str) -> bool:
        """Delete a scan session and associated data"""
        try:
            # Delete associated findings first
            await self.session.execute(
                delete(SecurityFindingModel).where(
                    SecurityFindingModel.scan_session_id == session_id
                )
            )

            # Delete the session
            stmt = delete(ScanSessionModel).where(ScanSessionModel.session_id == session_id)
            result = await self.session.execute(stmt)
            await self.session.commit()

            return result.rowcount > 0

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error deleting scan session {session_id}: {e}")
            return False

    async def get_sessions_by_status(
        self,
        status: str,
        tenant_id: Optional[UUID] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get sessions by status with optional tenant filtering"""
        try:
            stmt = select(ScanSessionModel).where(ScanSessionModel.status == status)

            if tenant_id:
                stmt = stmt.where(ScanSessionModel.tenant_id == tenant_id)

            stmt = stmt.order_by(ScanSessionModel.created_at.desc()).limit(limit)

            result = await self.session.execute(stmt)
            sessions = result.scalars().all()

            return [self._model_to_dict(session) for session in sessions]

        except Exception as e:
            logger.error(f"Error getting sessions by status {status}: {e}")
            return []

    def _model_to_dict(self, model: ScanSessionModel) -> Dict[str, Any]:
        """Convert SQLAlchemy model to dictionary with full metadata"""
        return {
            "session_id": model.session_id,
            "user_id": str(model.user_id) if model.user_id else None,
            "tenant_id": str(model.tenant_id) if model.tenant_id else None,
            "scan_type": model.scan_type,
            "status": model.status,
            "priority": model.priority,
            "progress": model.progress,
            "targets": json.loads(model.targets) if model.targets else [],
            "scan_config": json.loads(model.scan_config) if model.scan_config else {},
            "results": json.loads(model.results) if model.results else {},
            "metadata": json.loads(model.metadata) if model.metadata else {},
            "error_message": model.error_message,
            "estimated_duration": model.estimated_duration,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "started_at": model.started_at.isoformat() if model.started_at else None,
            "completed_at": model.completed_at.isoformat() if model.completed_at else None,
            "updated_at": model.updated_at.isoformat() if model.updated_at else None
        }


class EnhancedPostgreSQLTenantRepository(TenantRepository):
    """Enhanced PostgreSQL-backed tenant repository with comprehensive multi-tenancy support"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_tenant(self, tenant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new tenant with full configuration support"""
        try:
            tenant_model = TenantModel(
                id=tenant_data.get("id", uuid.uuid4()),
                name=tenant_data["name"],
                slug=tenant_data.get("slug", tenant_data["name"].lower().replace(" ", "-")),
                plan=tenant_data.get("plan", "free"),
                status=tenant_data.get("status", "pending"),
                settings=json.dumps(tenant_data.get("settings", {})),
                limits=json.dumps(tenant_data.get("limits", {
                    "max_users": 10,
                    "max_scans_per_month": 100,
                    "max_targets": 50,
                    "max_concurrent_scans": 3
                })),
                contact_email=tenant_data.get("contact_email"),
                billing_email=tenant_data.get("billing_email"),
                created_at=datetime.utcnow()
            )

            self.session.add(tenant_model)
            await self.session.commit()
            await self.session.refresh(tenant_model)

            return self._model_to_dict(tenant_model)

        except IntegrityError as e:
            await self.session.rollback()
            logger.error(f"Tenant creation failed - integrity error: {e}")
            raise ValueError("Tenant with this name or slug already exists")
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error creating tenant: {e}")
            raise

    async def get_tenant(self, tenant_id: UUID) -> Optional[Dict[str, Any]]:
        """Get tenant by ID with full details"""
        try:
            stmt = select(TenantModel).where(TenantModel.id == tenant_id)
            result = await self.session.execute(stmt)
            tenant_model = result.scalar_one_or_none()

            if tenant_model:
                return self._model_to_dict(tenant_model)
            return None

        except Exception as e:
            logger.error(f"Error getting tenant by ID {tenant_id}: {e}")
            return None

    async def get_tenant_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get tenant by slug"""
        try:
            stmt = select(TenantModel).where(TenantModel.slug == slug)
            result = await self.session.execute(stmt)
            tenant_model = result.scalar_one_or_none()

            if tenant_model:
                return self._model_to_dict(tenant_model)
            return None

        except Exception as e:
            logger.error(f"Error getting tenant by slug {slug}: {e}")
            return None

    async def update_tenant(self, tenant_id: UUID, updates: Dict[str, Any]) -> bool:
        """Update tenant with comprehensive field support"""
        try:
            update_data = {}

            # Handle direct field updates
            for field in ['name', 'plan', 'status', 'contact_email', 'billing_email']:
                if field in updates:
                    update_data[field] = updates[field]

            # Handle JSON field updates
            if 'settings' in updates:
                existing_tenant = await self.get_tenant(tenant_id)
                if existing_tenant:
                    existing_settings = existing_tenant.get('settings', {})
                    existing_settings.update(updates['settings'])
                    update_data['settings'] = json.dumps(existing_settings)
                else:
                    update_data['settings'] = json.dumps(updates['settings'])

            if 'limits' in updates:
                existing_tenant = await self.get_tenant(tenant_id)
                if existing_tenant:
                    existing_limits = existing_tenant.get('limits', {})
                    existing_limits.update(updates['limits'])
                    update_data['limits'] = json.dumps(existing_limits)
                else:
                    update_data['limits'] = json.dumps(updates['limits'])

            update_data['updated_at'] = datetime.utcnow()

            stmt = update(TenantModel).where(
                TenantModel.id == tenant_id
            ).values(**update_data)

            result = await self.session.execute(stmt)
            await self.session.commit()

            return result.rowcount > 0

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error updating tenant {tenant_id}: {e}")
            return False

    async def get_tenant_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get tenant by name"""
        try:
            stmt = select(TenantModel).where(TenantModel.name == name)
            result = await self.session.execute(stmt)
            tenant_model = result.scalar_one_or_none()

            if tenant_model:
                return self._model_to_dict(tenant_model)
            return None

        except Exception as e:
            logger.error(f"Error getting tenant by name {name}: {e}")
            return None

    async def list_tenants(
        self,
        status: Optional[str] = None,
        plan: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List tenants with filtering and pagination"""
        try:
            stmt = select(TenantModel)

            if status:
                stmt = stmt.where(TenantModel.status == status)
            if plan:
                stmt = stmt.where(TenantModel.plan == plan)

            stmt = stmt.order_by(TenantModel.created_at.desc()).offset(offset).limit(limit)

            result = await self.session.execute(stmt)
            tenants = result.scalars().all()

            return [self._model_to_dict(tenant) for tenant in tenants]

        except Exception as e:
            logger.error(f"Error listing tenants: {e}")
            return []

    async def deactivate_tenant(self, tenant_id: UUID) -> bool:
        """Deactivate a tenant and its associated resources"""
        try:
            # Update tenant status
            await self.update_tenant(tenant_id, {
                'status': 'inactive',
                'deactivated_at': datetime.utcnow()
            })

            # Cancel active scans for this tenant
            await self.session.execute(
                update(ScanSessionModel).where(
                    and_(
                        ScanSessionModel.tenant_id == tenant_id,
                        ScanSessionModel.status.in_(['queued', 'running'])
                    )
                ).values(status='cancelled', updated_at=datetime.utcnow())
            )

            await self.session.commit()
            return True

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error deactivating tenant {tenant_id}: {e}")
            return False

    def _model_to_dict(self, model: TenantModel) -> Dict[str, Any]:
        """Convert SQLAlchemy model to dictionary"""
        return {
            "id": str(model.id),
            "name": model.name,
            "slug": model.slug,
            "plan": model.plan,
            "status": model.status,
            "settings": json.loads(model.settings) if model.settings else {},
            "limits": json.loads(model.limits) if model.limits else {},
            "contact_email": model.contact_email,
            "billing_email": model.billing_email,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "updated_at": model.updated_at.isoformat() if model.updated_at else None
        }


class EnhancedRedisCacheRepository(CacheRepository):
    """Enhanced Redis-backed cache repository with advanced features and failover"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0", pool_size: int = 10):
        self._cache: Dict[str, Any] = {}  # Fallback in-memory cache
        self._redis_client = None
        self._redis_url = redis_url
        self._redis_available = False
        self._pool_size = pool_size
        self._ttl_cache: Dict[str, float] = {}  # For in-memory TTL tracking

    async def initialize(self) -> bool:
        """Initialize Redis connection with connection pooling"""
        try:
            if REDIS_AVAILABLE and aioredis:
                self._redis_client = await aioredis.from_url(
                    self._redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_timeout=5.0,
                    socket_connect_timeout=5.0,
                    max_connections=self._pool_size,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                # Test connection
                await self._redis_client.ping()
                self._redis_available = True
                logger.info(f"Enhanced Redis cache initialized successfully with pool size {self._pool_size}")

                # Start background cleanup for in-memory fallback
                asyncio.create_task(self._cleanup_expired_keys())
                return True
        except Exception as e:
            logger.warning(f"Redis unavailable, falling back to in-memory cache: {e}")

        self._redis_available = False
        logger.info("Using enhanced in-memory cache as fallback")
        asyncio.create_task(self._cleanup_expired_keys())
        return True

    async def _cleanup_expired_keys(self):
        """Background task to clean up expired keys in in-memory cache"""
        while True:
            try:
                current_time = asyncio.get_event_loop().time()
                expired_keys = [
                    key for key, expire_time in self._ttl_cache.items()
                    if expire_time <= current_time
                ]

                for key in expired_keys:
                    self._cache.pop(key, None)
                    self._ttl_cache.pop(key, None)

                await asyncio.sleep(60)  # Clean up every minute
            except Exception as e:
                logger.error(f"Error in cache cleanup task: {e}")
                await asyncio.sleep(60)

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with automatic JSON deserialization"""
        try:
            if self._redis_available and self._redis_client:
                value = await self._redis_client.get(key)
                if value:
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        return value
                return None
            else:
                # Check TTL for in-memory cache
                if key in self._ttl_cache:
                    if self._ttl_cache[key] <= asyncio.get_event_loop().time():
                        self._cache.pop(key, None)
                        self._ttl_cache.pop(key, None)
                        return None

                return self._cache.get(key)
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return self._cache.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL and automatic JSON serialization"""
        try:
            if self._redis_available and self._redis_client:
                serialized_value = json.dumps(value) if not isinstance(value, str) else value
                if ttl:
                    await self._redis_client.setex(key, ttl, serialized_value)
                else:
                    await self._redis_client.set(key, serialized_value)
                return True
            else:
                self._cache[key] = value
                if ttl:
                    expire_time = asyncio.get_event_loop().time() + ttl
                    self._ttl_cache[key] = expire_time
                return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            self._cache[key] = value
            if ttl:
                expire_time = asyncio.get_event_loop().time() + ttl
                self._ttl_cache[key] = expire_time
            return True

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if self._redis_available and self._redis_client:
                result = await self._redis_client.delete(key)
                return result > 0
            else:
                deleted = self._cache.pop(key, None) is not None
                self._ttl_cache.pop(key, None)
                return deleted
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            deleted = self._cache.pop(key, None) is not None
            self._ttl_cache.pop(key, None)
            return deleted

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            if self._redis_available and self._redis_client:
                return await self._redis_client.exists(key) > 0
            else:
                # Check TTL for in-memory cache
                if key in self._ttl_cache:
                    if self._ttl_cache[key] <= asyncio.get_event_loop().time():
                        self._cache.pop(key, None)
                        self._ttl_cache.pop(key, None)
                        return False

                return key in self._cache
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return key in self._cache

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment numeric value in cache"""
        try:
            if self._redis_available and self._redis_client:
                return await self._redis_client.incrby(key, amount)
            else:
                current = self._cache.get(key, 0)
                new_value = current + amount
                self._cache[key] = new_value
                return new_value
        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {e}")
            current = self._cache.get(key, 0)
            new_value = current + amount
            self._cache[key] = new_value
            return new_value

    async def decrement(self, key: str, amount: int = 1) -> int:
        """Decrement numeric value in cache"""
        return await self.increment(key, -amount)

    async def set_hash(self, key: str, field: str, value: Any) -> bool:
        """Set hash field in cache"""
        try:
            if self._redis_available and self._redis_client:
                serialized_value = json.dumps(value) if not isinstance(value, str) else value
                await self._redis_client.hset(key, field, serialized_value)
                return True
            else:
                if key not in self._cache:
                    self._cache[key] = {}
                self._cache[key][field] = value
                return True
        except Exception as e:
            logger.error(f"Cache hset error for key {key}, field {field}: {e}")
            if key not in self._cache:
                self._cache[key] = {}
            self._cache[key][field] = value
            return True

    async def get_hash(self, key: str, field: str) -> Optional[Any]:
        """Get hash field from cache"""
        try:
            if self._redis_available and self._redis_client:
                value = await self._redis_client.hget(key, field)
                if value:
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        return value
                return None
            else:
                hash_data = self._cache.get(key, {})
                return hash_data.get(field)
        except Exception as e:
            logger.error(f"Cache hget error for key {key}, field {field}: {e}")
            hash_data = self._cache.get(key, {})
            return hash_data.get(field)

    async def get_all_hash(self, key: str) -> Dict[str, Any]:
        """Get all hash fields from cache"""
        try:
            if self._redis_available and self._redis_client:
                hash_data = await self._redis_client.hgetall(key)
                result = {}
                for field, value in hash_data.items():
                    try:
                        result[field] = json.loads(value)
                    except json.JSONDecodeError:
                        result[field] = value
                return result
            else:
                return self._cache.get(key, {})
        except Exception as e:
            logger.error(f"Cache hgetall error for key {key}: {e}")
            return self._cache.get(key, {})

    async def delete_hash_field(self, key: str, field: str) -> bool:
        """Delete hash field from cache"""
        try:
            if self._redis_available and self._redis_client:
                result = await self._redis_client.hdel(key, field)
                return result > 0
            else:
                hash_data = self._cache.get(key, {})
                if field in hash_data:
                    del hash_data[field]
                    return True
                return False
        except Exception as e:
            logger.error(f"Cache hdel error for key {key}, field {field}: {e}")
            hash_data = self._cache.get(key, {})
            if field in hash_data:
                del hash_data[field]
                return True
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key"""
        try:
            if self._redis_available and self._redis_client:
                return await self._redis_client.expire(key, ttl)
            else:
                if key in self._cache:
                    expire_time = asyncio.get_event_loop().time() + ttl
                    self._ttl_cache[key] = expire_time
                    return True
                return False
        except Exception as e:
            logger.error(f"Cache expire error for key {key}: {e}")
            if key in self._cache:
                expire_time = asyncio.get_event_loop().time() + ttl
                self._ttl_cache[key] = expire_time
                return True
            return False

    async def get_keys_pattern(self, pattern: str) -> List[str]:
        """Get keys matching pattern"""
        try:
            if self._redis_available and self._redis_client:
                return await self._redis_client.keys(pattern)
            else:
                import fnmatch
                return [key for key in self._cache.keys() if fnmatch.fnmatch(key, pattern)]
        except Exception as e:
            logger.error(f"Cache keys pattern error for pattern {pattern}: {e}")
            import fnmatch
            return [key for key in self._cache.keys() if fnmatch.fnmatch(key, pattern)]

    async def flush_all(self) -> bool:
        """Flush all cache data"""
        try:
            if self._redis_available and self._redis_client:
                await self._redis_client.flushdb()
                return True
            else:
                self._cache.clear()
                self._ttl_cache.clear()
                return True
        except Exception as e:
            logger.error(f"Cache flush error: {e}")
            self._cache.clear()
            self._ttl_cache.clear()
            return True

    async def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information and statistics"""
        try:
            if self._redis_available and self._redis_client:
                info = await self._redis_client.info()
                return {
                    "type": "redis",
                    "connected": True,
                    "used_memory": info.get('used_memory_human', 'unknown'),
                    "connected_clients": info.get('connected_clients', 'unknown'),
                    "total_commands_processed": info.get('total_commands_processed', 'unknown'),
                    "keyspace_hits": info.get('keyspace_hits', 'unknown'),
                    "keyspace_misses": info.get('keyspace_misses', 'unknown')
                }
            else:
                return {
                    "type": "in_memory",
                    "connected": True,
                    "total_keys": len(self._cache),
                    "keys_with_ttl": len(self._ttl_cache),
                    "memory_usage": "unknown"
                }
        except Exception as e:
            logger.error(f"Cache info error: {e}")
            return {
                "type": "unknown",
                "connected": False,
                "error": str(e)
            }


# Factory functions for dependency injection
async def create_enhanced_scan_session_repository(session: AsyncSession) -> EnhancedPostgreSQLScanSessionRepository:
    """Factory function for enhanced scan session repository"""
    return EnhancedPostgreSQLScanSessionRepository(session)

async def create_enhanced_tenant_repository(session: AsyncSession) -> EnhancedPostgreSQLTenantRepository:
    """Factory function for enhanced tenant repository"""
    return EnhancedPostgreSQLTenantRepository(session)

async def create_enhanced_cache_repository(redis_url: str = "redis://localhost:6379/0") -> EnhancedRedisCacheRepository:
    """Factory function for enhanced cache repository"""
    cache_repo = EnhancedRedisCacheRepository(redis_url)
    await cache_repo.initialize()
    return cache_repo
