"""Storage service for evidence and file management."""
import hashlib
import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

from ..domain.tenant_entities import Evidence
from ..repositories.evidence_repository import EvidenceRepository
from ..storage.interface import (
    StorageDriver, FileMetadata, PresignedUrlRequest,
    DownloadUrlRequest, StorageBackend, FileStatus
)
from ..storage.validation import get_file_validator


logger = logging.getLogger(__name__)


class StorageService:
    """Service for managing evidence storage and uploads."""

    def __init__(self, storage_driver: StorageDriver):
        self.storage_driver = storage_driver
        self.file_validator = get_file_validator()
        self.evidence_repo = EvidenceRepository()

    async def create_upload_url(
        self,
        filename: str,
        content_type: str,
        size_bytes: int,
        tenant_id: UUID,
        uploaded_by: str,
        tags: Optional[List[str]] = None,
        custom_metadata: Optional[dict] = None
    ) -> dict:
        """Create presigned upload URL for evidence."""
        # Validate request parameters
        if size_bytes > self.storage_driver.config.max_file_size:
            raise ValueError(f"File size {size_bytes} exceeds maximum {self.storage_driver.config.max_file_size}")

        # Check allowed MIME types
        if (self.storage_driver.config.allowed_mime_types and
            content_type not in self.storage_driver.config.allowed_mime_types):
            raise ValueError(f"MIME type {content_type} not allowed")

        # Create presigned URL request
        request = PresignedUrlRequest(
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            tenant_id=tenant_id,
            uploaded_by=uploaded_by,
            tags=tags or [],
            custom_metadata=custom_metadata or {}
        )

        # Generate presigned URL
        response = await self.storage_driver.generate_presigned_upload_url(request)

        # Store evidence record in database using repository
        file_path = self.storage_driver.generate_file_path(tenant_id, filename, response.file_id)

        # Generate SHA256 hash placeholder (will be updated when file is uploaded)
        temp_hash = hashlib.sha256(f"{response.file_id}{filename}".encode()).hexdigest()

        evidence = await self.evidence_repo.create(
            tenant_id=tenant_id,
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            sha256_hash=temp_hash,
            storage_path=file_path,
            storage_backend=self.storage_driver.config.backend.value,
            uploaded_by=uploaded_by
        )

        return {
            "upload_url": response.upload_url,
            "file_id": str(response.file_id),
            "expires_at": response.expires_at.isoformat(),
            "required_headers": response.required_headers
        }

    async def complete_upload(
        self,
        file_id: UUID,
        tenant_id: UUID,
        validate_file: bool = True
    ) -> dict:
        """Complete file upload and validate."""

        # Get evidence record using repository
        evidence = await self.evidence_repo.get_by_id(file_id, tenant_id)

        if not evidence:
            raise ValueError(f"Evidence {file_id} not found")

        if evidence.status != "uploaded":  # Evidence created with "uploaded" status
            logger.info(f"Evidence {file_id} already processed, status: {evidence.status}")

        try:
            # Verify file exists in storage
            if not await self.storage_driver.file_exists(evidence.storage_path):
                raise ValueError("File not found in storage")

            # Get file metadata from storage
            storage_metadata = await self.storage_driver.get_file_metadata(evidence.storage_path)

            if storage_metadata:
                # Update evidence with actual metadata
                actual_size = storage_metadata.get('size_bytes', int(evidence.size_bytes))

                # Calculate actual SHA256 hash if possible
                actual_hash = storage_metadata.get('sha256_hash') or evidence.sha256_hash

                # Validate file if requested
                validation_result = None
                new_status = "processed"

                if validate_file:
                    try:
                        # Download file for validation
                        file_data = b""
                        async for chunk in self.storage_driver.download_file(evidence.storage_path):
                            file_data += chunk

                        # Validate file content
                        validation_result = await self.file_validator.validate_file(
                            file_data, evidence.filename, evidence.content_type
                        )

                        # Calculate actual hash
                        actual_hash = hashlib.sha256(file_data).hexdigest()

                        # Check validation result
                        if not validation_result.is_valid:
                            new_status = "error"
                            await self.evidence_repo.update_status(file_id, tenant_id, new_status)
                            return {
                                "status": "error",
                                "errors": validation_result.errors,
                                "warnings": validation_result.warnings
                            }

                        # Check malware scan (if implemented)
                        if hasattr(validation_result, 'malware_scan_result') and validation_result.malware_scan_result == "INFECTED":
                            new_status = "quarantined"
                            await self.evidence_repo.update_status(file_id, tenant_id, new_status)
                            return {
                                "status": "quarantined",
                                "reason": "Malware detected",
                                "scan_details": getattr(validation_result, 'malware_scan_details', {})
                            }

                    except Exception as e:
                        logger.error(f"File validation failed: {e}")
                        new_status = "error"

                # Update evidence status to processed
                updated_evidence = await self.evidence_repo.update_status(
                    file_id,
                    tenant_id,
                    new_status,
                    processed_at=datetime.utcnow().isoformat()
                )

                logger.info(f"Completed upload for evidence {file_id}")

                result = {
                    "status": new_status,
                    "file_id": str(file_id),
                    "filename": evidence.filename,
                    "size_bytes": actual_size,
                    "sha256_hash": actual_hash
                }

                if validation_result:
                    result["validation"] = {
                        "warnings": getattr(validation_result, 'warnings', []),
                        "detected_type": getattr(validation_result, 'detected_type', None),
                        "malware_scan": validation_result.malware_scan_result
                    }

                return result

        except Exception as e:
            # Mark as error using repository
            await self.evidence_repo.update_status(file_id, tenant_id, FileStatus.ERROR.value)
            logger.error(f"Failed to complete upload for {file_id}: {e}")
            raise

    async def create_download_url(
        self,
        file_id: UUID,
        tenant_id: UUID,
        expires_in: int = 3600,
        content_disposition: str = "attachment",
        custom_filename: Optional[str] = None
    ) -> dict:
        """Create presigned download URL for evidence."""
        # Get evidence record using repository
        evidence = await self.evidence_repo.get_by_id(file_id, tenant_id)

        if not evidence:
            raise ValueError(f"Evidence {file_id} not found")

        if evidence.status not in [FileStatus.UPLOADED.value, FileStatus.PROCESSED.value]:
            raise ValueError(f"Evidence {file_id} is not available for download")

        # Create download request
        request = DownloadUrlRequest(
            file_id=file_id,
            expires_in=expires_in,
            content_disposition=content_disposition,
            custom_filename=custom_filename
        )

        # Generate presigned URL
        response = await self.storage_driver.generate_presigned_download_url(request)

        return {
            "download_url": response.download_url,
            "filename": response.filename,
            "content_type": response.content_type,
            "size_bytes": response.size_bytes,
            "expires_at": response.expires_at.isoformat()
        }

    async def list_evidence(
        self,
        tenant_id: UUID,
        status: Optional[FileStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[dict]:
        """List evidence files for tenant."""
        # Use repository for listing
        status_filter = status.value if status else None
        evidence_list = await self.evidence_repo.list_by_tenant(
            tenant_id=tenant_id,
            limit=limit,
            offset=offset,
            status_filter=status_filter
        )

        return [
            {
                "id": str(evidence.id),
                "filename": evidence.filename,
                "content_type": evidence.content_type,
                "size_bytes": evidence.size_bytes,
                "status": evidence.status,
                "uploaded_by": evidence.uploaded_by,
                "created_at": evidence.created_at.isoformat(),
                "sha256_hash": evidence.sha256_hash
            }
            for evidence in evidence_list
        ]

    async def delete_evidence(
        self,
        file_id: UUID,
        tenant_id: UUID,
        permanent: bool = False
    ) -> bool:
        """Delete evidence file."""
        # Get evidence record using repository
        evidence = await self.evidence_repo.get_by_id(file_id, tenant_id)

        if not evidence:
            return False

        try:
            if permanent:
                # Delete from storage
                await self.storage_driver.delete_file(evidence.storage_path)

                # Delete from database using repository
                await self.evidence_repo.delete(file_id, tenant_id)
            else:
                # Soft delete - mark as deleted using repository
                await self.evidence_repo.update_status(
                    file_id,
                    tenant_id,
                    FileStatus.DELETED.value
                )

            logger.info(f"Deleted evidence {file_id} (permanent: {permanent})")
            return True

        except Exception as e:
            logger.error(f"Failed to delete evidence {file_id}: {e}")
            raise

    async def get_evidence_metadata(
        self,
        file_id: UUID,
        tenant_id: UUID
    ) -> Optional[dict]:
        """Get evidence metadata."""
        # Use repository to get evidence
        evidence = await self.evidence_repo.get_by_id(file_id, tenant_id)

        if not evidence:
            return None

        return {
            "id": str(evidence.id),
            "filename": evidence.filename,
            "content_type": evidence.content_type,
            "size_bytes": evidence.size_bytes,
            "sha256_hash": evidence.sha256_hash,
            "storage_path": evidence.storage_path,
            "storage_backend": evidence.storage_backend,
            "status": evidence.status,
            "uploaded_by": evidence.uploaded_by,
            "created_at": evidence.created_at.isoformat(),
            "updated_at": evidence.updated_at.isoformat() if evidence.updated_at else None,
            "processed_at": evidence.processed_at.isoformat() if evidence.processed_at else None
        }
