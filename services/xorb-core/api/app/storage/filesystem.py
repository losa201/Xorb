"""Filesystem storage driver implementation."""
import asyncio
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Tuple
from uuid import uuid4

import aiofiles
import aiofiles.os

from .interface import (
    StorageDriver, FilesystemConfig, FileMetadata,
    PresignedUrlRequest, PresignedUrlResponse,
    DownloadUrlRequest, DownloadUrlResponse,
    StorageBackend, FileStatus
)


logger = logging.getLogger(__name__)


class FilesystemStorageDriver(StorageDriver):
    """Filesystem-based storage driver."""

    def __init__(self, config: FilesystemConfig):
        super().__init__(config)
        self.config: FilesystemConfig = config
        self.storage_root = Path(config.storage_root)

        # Create storage root if it doesn't exist
        if config.create_directories:
            self.storage_root.mkdir(parents=True, exist_ok=True)

    async def generate_presigned_upload_url(
        self,
        request: PresignedUrlRequest
    ) -> PresignedUrlResponse:
        """Generate presigned URL for file upload.

        For filesystem storage, this creates a temporary upload token
        and returns a URL that the application can use to handle the upload.
        """
        file_id = uuid4()
        upload_token = uuid4()

        # Store upload metadata in memory or cache
        # In production, use Redis or database
        upload_metadata = {
            'file_id': str(file_id),
            'filename': request.filename,
            'content_type': request.content_type,
            'size_bytes': request.size_bytes,
            'tenant_id': str(request.tenant_id),
            'uploaded_by': request.uploaded_by,
            'tags': request.tags,
            'custom_metadata': request.custom_metadata,
            'expires_at': datetime.utcnow() + timedelta(seconds=request.expires_in)
        }

        # Store in global cache (would be Redis in production)
        await self._store_upload_token(str(upload_token), upload_metadata)

        return PresignedUrlResponse(
            upload_url=f"/api/storage/upload/{upload_token}",
            file_id=file_id,
            expires_at=upload_metadata['expires_at'],
            completion_url=f"/api/storage/complete/{file_id}"
        )

    async def generate_presigned_download_url(
        self,
        request: DownloadUrlRequest
    ) -> DownloadUrlResponse:
        """Generate presigned URL for file download."""
        # Get file metadata from database
        file_metadata = await self._get_file_metadata_from_db(request.file_id)
        if not file_metadata:
            raise FileNotFoundError(f"File {request.file_id} not found")

        download_token = uuid4()

        # Store download metadata
        download_metadata = {
            'file_id': str(request.file_id),
            'file_path': file_metadata['storage_path'],
            'content_type': file_metadata['content_type'],
            'filename': request.custom_filename or file_metadata['filename'],
            'size_bytes': file_metadata['size_bytes'],
            'content_disposition': request.content_disposition or 'attachment',
            'expires_at': datetime.utcnow() + timedelta(seconds=request.expires_in)
        }

        await self._store_download_token(str(download_token), download_metadata)

        return DownloadUrlResponse(
            download_url=f"/api/storage/download/{download_token}",
            filename=download_metadata['filename'],
            content_type=file_metadata['content_type'],
            size_bytes=file_metadata['size_bytes'],
            expires_at=download_metadata['expires_at']
        )

    async def upload_file(
        self,
        file_data: bytes,
        metadata: FileMetadata
    ) -> FileMetadata:
        """Upload file directly to filesystem."""
        file_path = self.storage_root / metadata.storage_path

        # Create directory structure
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Write file
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_data)

            # Set file permissions
            await aiofiles.os.chmod(file_path, self.config.permissions)

            # Update metadata
            metadata.status = FileStatus.UPLOADED
            metadata.updated_at = datetime.utcnow()

            logger.info(f"Uploaded file to {file_path}")
            return metadata

        except Exception as e:
            logger.error(f"Failed to upload file {metadata.filename}: {e}")
            metadata.status = FileStatus.ERROR
            raise

    async def download_file(self, file_path: str) -> AsyncIterator[bytes]:
        """Download file as async iterator."""
        full_path = self.storage_root / file_path

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            async with aiofiles.open(full_path, 'rb') as f:
                chunk_size = 64 * 1024  # 64KB chunks
                while True:
                    chunk = await f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        except Exception as e:
            logger.error(f"Failed to download file {file_path}: {e}")
            raise

    async def delete_file(self, file_path: str) -> bool:
        """Delete file from filesystem."""
        full_path = self.storage_root / file_path

        try:
            if full_path.exists():
                await aiofiles.os.remove(full_path)
                logger.info(f"Deleted file: {file_path}")

                # Clean up empty directories
                await self._cleanup_empty_directories(full_path.parent)
                return True
            else:
                logger.warning(f"File not found for deletion: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False

    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists."""
        full_path = self.storage_root / file_path
        return full_path.exists()

    async def get_file_metadata(self, file_path: str) -> Optional[Dict]:
        """Get file metadata from filesystem."""
        full_path = self.storage_root / file_path

        try:
            if not full_path.exists():
                return None

            stat = full_path.stat()
            return {
                'size_bytes': stat.st_size,
                'created_at': datetime.fromtimestamp(stat.st_ctime),
                'modified_at': datetime.fromtimestamp(stat.st_mtime),
                'permissions': oct(stat.st_mode)[-3:],
                'is_file': full_path.is_file(),
                'is_directory': full_path.is_dir()
            }
        except Exception as e:
            logger.error(f"Failed to get metadata for {file_path}: {e}")
            return None

    async def list_files(
        self,
        prefix: str,
        limit: int = 100,
        continuation_token: Optional[str] = None
    ) -> Tuple[List[str], Optional[str]]:
        """List files with prefix."""
        prefix_path = self.storage_root / prefix
        files = []

        try:
            if prefix_path.exists() and prefix_path.is_dir():
                # Get all files recursively
                all_files = []
                for root, dirs, filenames in os.walk(prefix_path):
                    for filename in filenames:
                        file_path = Path(root) / filename
                        relative_path = file_path.relative_to(self.storage_root)
                        all_files.append(str(relative_path))

                # Sort for consistent ordering
                all_files.sort()

                # Handle pagination
                start_index = 0
                if continuation_token:
                    try:
                        start_index = int(continuation_token)
                    except ValueError:
                        start_index = 0

                end_index = start_index + limit
                files = all_files[start_index:end_index]

                # Set continuation token if there are more files
                next_token = None
                if end_index < len(all_files):
                    next_token = str(end_index)

                return files, next_token

        except Exception as e:
            logger.error(f"Failed to list files with prefix {prefix}: {e}")

        return files, None

    async def _cleanup_empty_directories(self, directory: Path) -> None:
        """Clean up empty directories recursively."""
        try:
            if directory.exists() and directory.is_dir():
                # Check if directory is empty
                if not any(directory.iterdir()):
                    # Don't delete the storage root
                    if directory != self.storage_root:
                        await aiofiles.os.rmdir(directory)
                        # Recursively clean parent if it becomes empty
                        await self._cleanup_empty_directories(directory.parent)
        except Exception as e:
            logger.debug(f"Could not clean up directory {directory}: {e}")

    async def _store_upload_token(self, token: str, metadata: Dict) -> None:
        """Store upload token metadata."""
        # In production, this would use Redis
        # For now, store in a global dict or use the cache backend
        from ..infrastructure.cache import get_cache

        try:
            cache = get_cache()
            await cache.set(f"upload_token:{token}", metadata, expire=3600)
        except Exception:
            # Fallback to in-memory storage (not production-ready)
            if not hasattr(self, '_upload_tokens'):
                self._upload_tokens = {}
            self._upload_tokens[token] = metadata

    async def _store_download_token(self, token: str, metadata: Dict) -> None:
        """Store download token metadata."""
        from ..infrastructure.cache import get_cache

        try:
            cache = get_cache()
            await cache.set(f"download_token:{token}", metadata, expire=3600)
        except Exception:
            # Fallback to in-memory storage (not production-ready)
            if not hasattr(self, '_download_tokens'):
                self._download_tokens = {}
            self._download_tokens[token] = metadata

    async def get_upload_metadata(self, token: str) -> Optional[Dict]:
        """Get upload metadata by token."""
        from ..infrastructure.cache import get_cache

        try:
            cache = get_cache()
            return await cache.get(f"upload_token:{token}")
        except Exception:
            # Fallback
            return getattr(self, '_upload_tokens', {}).get(token)

    async def get_download_metadata(self, token: str) -> Optional[Dict]:
        """Get download metadata by token."""
        from ..infrastructure.cache import get_cache

        try:
            cache = get_cache()
            return await cache.get(f"download_token:{token}")
        except Exception:
            # Fallback
            return getattr(self, '_download_tokens', {}).get(token)

    async def _get_file_metadata_from_db(self, file_id, tenant_id: Optional[str] = None) -> Optional[Dict]:
        """Get file metadata from database."""
        from ..repositories.evidence_repository import EvidenceRepository
        from uuid import UUID

        try:
            if not tenant_id:
                logger.warning("No tenant_id provided for file metadata lookup")
                return None

            # Convert tenant_id to UUID if it's a string
            if isinstance(tenant_id, str):
                tenant_uuid = UUID(tenant_id)
            else:
                tenant_uuid = tenant_id

            # Query evidence from database
            repo = EvidenceRepository()
            evidence = await repo.get_by_id(file_id, tenant_uuid)

            if evidence:
                return {
                    'storage_path': evidence.storage_path,
                    'content_type': evidence.content_type,
                    'filename': evidence.filename,
                    'size_bytes': int(evidence.size_bytes) if evidence.size_bytes else 0
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get file metadata for {file_id}: {e}")
            return None


# Register filesystem driver
from .interface import StorageDriverFactory, StorageBackend
StorageDriverFactory.register_driver(StorageBackend.FILESYSTEM, FilesystemStorageDriver)
