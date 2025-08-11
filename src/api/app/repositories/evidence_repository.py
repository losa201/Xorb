"""Evidence repository implementation."""
from typing import Dict, List, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_
from sqlalchemy.orm import selectinload

from ..domain.tenant_entities import Evidence
from ..infrastructure.database import get_async_session


class EvidenceRepository:
    """Repository for Evidence operations with tenant isolation."""
    
    def __init__(self, db_session: Optional[AsyncSession] = None):
        self.db_session = db_session
    
    def _get_session(self):
        """Get database session."""
        if self.db_session:
            return self.db_session
        return get_async_session()
    
    async def create(
        self,
        tenant_id: UUID,
        filename: str,
        content_type: str,
        size_bytes: int,
        sha256_hash: str,
        storage_path: str,
        storage_backend: str,
        uploaded_by: str
    ) -> Evidence:
        """Create new evidence record."""
        async with self._get_session() as session:
            evidence = Evidence(
                tenant_id=tenant_id,
                filename=filename,
                content_type=content_type,
                size_bytes=str(size_bytes),
                sha256_hash=sha256_hash,
                storage_path=storage_path,
                storage_backend=storage_backend,
                uploaded_by=uploaded_by,
                status="uploaded"
            )
            
            session.add(evidence)
            await session.commit()
            await session.refresh(evidence)
            return evidence
    
    async def get_by_id(self, evidence_id: UUID, tenant_id: UUID) -> Optional[Evidence]:
        """Get evidence by ID with tenant isolation."""
        async with self._get_session() as session:
            result = await session.execute(
                select(Evidence).where(
                    and_(
                        Evidence.id == evidence_id,
                        Evidence.tenant_id == tenant_id
                    )
                )
            )
            return result.scalar_one_or_none()
    
    async def get_by_storage_path(self, storage_path: str, tenant_id: UUID) -> Optional[Evidence]:
        """Get evidence by storage path with tenant isolation."""
        async with self._get_session() as session:
            result = await session.execute(
                select(Evidence).where(
                    and_(
                        Evidence.storage_path == storage_path,
                        Evidence.tenant_id == tenant_id
                    )
                )
            )
            return result.scalar_one_or_none()
    
    async def get_by_hash(self, sha256_hash: str, tenant_id: UUID) -> Optional[Evidence]:
        """Get evidence by hash with tenant isolation (deduplication)."""
        async with self._get_session() as session:
            result = await session.execute(
                select(Evidence).where(
                    and_(
                        Evidence.sha256_hash == sha256_hash,
                        Evidence.tenant_id == tenant_id
                    )
                )
            )
            return result.scalar_one_or_none()
    
    async def list_by_tenant(
        self,
        tenant_id: UUID,
        limit: int = 50,
        offset: int = 0,
        status_filter: Optional[str] = None
    ) -> List[Evidence]:
        """List evidence for tenant with pagination."""
        async with self._get_session() as session:
            query = select(Evidence).where(Evidence.tenant_id == tenant_id)
            
            if status_filter:
                query = query.where(Evidence.status == status_filter)
            
            query = query.offset(offset).limit(limit).order_by(Evidence.created_at.desc())
            
            result = await session.execute(query)
            return list(result.scalars().all())
    
    async def update_status(
        self,
        evidence_id: UUID,
        tenant_id: UUID,
        status: str,
        processed_at: Optional[str] = None
    ) -> Optional[Evidence]:
        """Update evidence processing status."""
        async with self._get_session() as session:
            query = (
                update(Evidence)
                .where(
                    and_(
                        Evidence.id == evidence_id,
                        Evidence.tenant_id == tenant_id
                    )
                )
                .values(status=status)
            )
            
            if processed_at:
                query = query.values(processed_at=processed_at)
            
            await session.execute(query)
            await session.commit()
            
            # Return updated record
            return await self.get_by_id(evidence_id, tenant_id)
    
    async def delete(self, evidence_id: UUID, tenant_id: UUID) -> bool:
        """Delete evidence with tenant isolation."""
        async with self._get_session() as session:
            result = await session.execute(
                delete(Evidence).where(
                    and_(
                        Evidence.id == evidence_id,
                        Evidence.tenant_id == tenant_id
                    )
                )
            )
            await session.commit()
            return result.rowcount > 0
    
    async def get_storage_stats(self, tenant_id: UUID) -> Dict[str, int]:
        """Get storage statistics for tenant."""
        async with self._get_session() as session:
            # Count total files
            total_result = await session.execute(
                select(Evidence).where(Evidence.tenant_id == tenant_id)
            )
            total_files = len(list(total_result.scalars().all()))
            
            # Calculate total size (would use SUM in production SQL)
            all_evidence = await session.execute(
                select(Evidence.size_bytes).where(Evidence.tenant_id == tenant_id)
            )
            
            total_size = 0
            for size_str in all_evidence.scalars():
                try:
                    total_size += int(size_str) if size_str else 0
                except (ValueError, TypeError):
                    continue
            
            return {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "status_counts": await self._get_status_counts(tenant_id, session)
            }
    
    async def _get_status_counts(self, tenant_id: UUID, session: AsyncSession) -> Dict[str, int]:
        """Get count of files by status."""
        result = await session.execute(
            select(Evidence.status).where(Evidence.tenant_id == tenant_id)
        )
        
        status_counts = {}
        for status in result.scalars():
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return status_counts