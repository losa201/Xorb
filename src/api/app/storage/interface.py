"""Storage driver interface and models."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, AsyncIterator
from uuid import UUID

from pydantic import BaseModel, Field


class StorageBackend(str, Enum):
    """Available storage backends."""
    FILESYSTEM = "filesystem"
    S3 = "s3"
    AZURE_BLOB = "azure_blob"
    GCS = "gcs"


class FileStatus(str, Enum):
    """File processing status."""
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    QUARANTINED = "quarantined"
    DELETED = "deleted"
    ERROR = "error"


@dataclass
class StorageConfig:
    """Base storage configuration."""
    backend: StorageBackend
    base_path: str
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_mime_types: Optional[List[str]] = None
    enable_encryption: bool = True
    enable_versioning: bool = False


@dataclass
class S3Config(StorageConfig):
    """S3-specific configuration."""
    bucket_name: str = ""
    region: str = ""
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    endpoint_url: Optional[str] = None  # For S3-compatible services
    use_ssl: bool = True
    presigned_url_ttl: int = 3600  # 1 hour


@dataclass
class FilesystemConfig(StorageConfig):
    """Filesystem-specific configuration."""
    storage_root: str = ""
    create_directories: bool = True
    permissions: int = 0o644


class FileMetadata(BaseModel):
    """File metadata model."""
    filename: str
    content_type: str
    size_bytes: int
    sha256_hash: str
    uploaded_by: str
    tenant_id: UUID
    storage_path: str
    storage_backend: StorageBackend
    status: FileStatus = FileStatus.UPLOADED
    
    # Optional metadata
    original_filename: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    custom_metadata: Dict[str, str] = Field(default_factory=dict)
    
    # Timestamps
    created_at: datetime
    updated_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Security metadata
    scan_result: Optional[str] = None  # Clean, Infected, Error
    scan_details: Optional[Dict] = None
    quarantine_reason: Optional[str] = None


class PresignedUrlRequest(BaseModel):
    """Request for presigned URL generation."""
    filename: str
    content_type: str
    size_bytes: int
    expires_in: int = 3600  # 1 hour
    tenant_id: UUID
    uploaded_by: str
    
    # Optional metadata to store with file
    tags: List[str] = Field(default_factory=list)
    custom_metadata: Dict[str, str] = Field(default_factory=dict)


class PresignedUrlResponse(BaseModel):
    """Response containing presigned URL and metadata."""
    upload_url: str
    file_id: UUID
    expires_at: datetime
    required_headers: Dict[str, str] = Field(default_factory=dict)
    
    # For tracking upload completion
    completion_url: Optional[str] = None


class DownloadUrlRequest(BaseModel):
    """Request for download URL generation."""
    file_id: UUID
    expires_in: int = 3600  # 1 hour
    content_disposition: Optional[str] = None  # attachment, inline
    custom_filename: Optional[str] = None


class DownloadUrlResponse(BaseModel):
    """Response containing download URL."""
    download_url: str
    filename: str
    content_type: str
    size_bytes: int
    expires_at: datetime


class FileValidationResult(BaseModel):
    """Result of file validation."""
    is_valid: bool
    mime_type: str
    detected_type: Optional[str] = None  # From python-magic
    size_bytes: int
    sha256_hash: str
    
    # Validation details
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Security scan results
    malware_scan_result: Optional[str] = None
    malware_scan_details: Optional[Dict] = None


class StorageDriver(ABC):
    """Abstract base class for storage drivers."""
    
    def __init__(self, config: StorageConfig):
        self.config = config
    
    @abstractmethod
    async def generate_presigned_upload_url(
        self, 
        request: PresignedUrlRequest
    ) -> PresignedUrlResponse:
        """Generate presigned URL for file upload."""
        pass
    
    @abstractmethod
    async def generate_presigned_download_url(
        self, 
        request: DownloadUrlRequest
    ) -> DownloadUrlResponse:
        """Generate presigned URL for file download."""
        pass
    
    @abstractmethod
    async def upload_file(
        self, 
        file_data: bytes, 
        metadata: FileMetadata
    ) -> FileMetadata:
        """Upload file directly (for server-side uploads)."""
        pass
    
    @abstractmethod
    async def download_file(self, file_path: str) -> AsyncIterator[bytes]:
        """Download file as async iterator."""
        pass
    
    @abstractmethod
    async def delete_file(self, file_path: str) -> bool:
        """Delete file from storage."""
        pass
    
    @abstractmethod
    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists in storage."""
        pass
    
    @abstractmethod
    async def get_file_metadata(self, file_path: str) -> Optional[Dict]:
        """Get file metadata from storage."""
        pass
    
    @abstractmethod
    async def list_files(
        self, 
        prefix: str, 
        limit: int = 100,
        continuation_token: Optional[str] = None
    ) -> Tuple[List[str], Optional[str]]:
        """List files with optional pagination."""
        pass
    
    async def validate_file_type(
        self, 
        file_data: bytes, 
        filename: str,
        declared_content_type: str
    ) -> FileValidationResult:
        """Validate file type and content."""
        # This will be implemented in the base class with python-magic
        pass
    
    def generate_file_path(
        self, 
        tenant_id: UUID, 
        filename: str,
        file_id: Optional[UUID] = None
    ) -> str:
        """Generate storage path for file."""
        # Default implementation - can be overridden
        from uuid import uuid4
        
        if not file_id:
            file_id = uuid4()
        
        # Structure: tenant_id/year/month/file_id/filename
        now = datetime.utcnow()
        return f"{tenant_id}/{now.year:04d}/{now.month:02d}/{file_id}/{filename}"
    
    def get_mime_type_category(self, mime_type: str) -> str:
        """Get broad category for MIME type."""
        if mime_type.startswith('image/'):
            return 'image'
        elif mime_type.startswith('video/'):
            return 'video'
        elif mime_type.startswith('audio/'):
            return 'audio'
        elif mime_type.startswith('text/'):
            return 'text'
        elif mime_type in ['application/pdf']:
            return 'document'
        elif mime_type.startswith('application/'):
            return 'application'
        else:
            return 'unknown'


class StorageDriverFactory:
    """Factory for creating storage drivers."""
    
    _drivers: Dict[StorageBackend, type] = {}
    
    @classmethod
    def register_driver(cls, backend: StorageBackend, driver_class: type):
        """Register a storage driver."""
        cls._drivers[backend] = driver_class
    
    @classmethod
    def create_driver(cls, config: StorageConfig) -> StorageDriver:
        """Create storage driver instance."""
        driver_class = cls._drivers.get(config.backend)
        if not driver_class:
            raise ValueError(f"Unsupported storage backend: {config.backend}")
        
        return driver_class(config)
    
    @classmethod
    def list_backends(cls) -> List[StorageBackend]:
        """List available storage backends."""
        return list(cls._drivers.keys())