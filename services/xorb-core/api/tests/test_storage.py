"""Tests for storage system and file validation."""
import asyncio
import pytest
from datetime import datetime, timedelta
from io import BytesIO
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.storage.interface import (
    StorageDriverFactory, StorageBackend, FilesystemConfig, S3Config,
    PresignedUrlRequest, DownloadUrlRequest, FileStatus
)
from app.storage.filesystem import FilesystemStorageDriver
from app.storage.s3 import S3StorageDriver
from app.storage.validation import FileValidator, DEFAULT_ALLOWED_MIME_TYPES
from app.services.storage_service import StorageService


@pytest.fixture
def filesystem_config():
    """Filesystem storage configuration for testing."""
    return FilesystemConfig(
        backend=StorageBackend.FILESYSTEM,
        base_path="/tmp/test_storage",
        storage_root="/tmp/test_storage",
        max_file_size=10 * 1024 * 1024,  # 10MB
        create_directories=True
    )


@pytest.fixture
def s3_config():
    """S3 storage configuration for testing."""
    return S3Config(
        backend=StorageBackend.S3,
        base_path="evidence",
        bucket_name="test-bucket",
        region="us-east-1",
        access_key_id="test_key",
        secret_access_key="test_secret",
        max_file_size=100 * 1024 * 1024  # 100MB
    )


@pytest.fixture
def sample_file_data():
    """Sample file data for testing."""
    return b"This is a test file content for validation and upload testing."


@pytest.fixture
def presigned_request():
    """Sample presigned URL request."""
    return PresignedUrlRequest(
        filename="test_document.pdf",
        content_type="application/pdf",
        size_bytes=1024,
        tenant_id=uuid4(),
        uploaded_by="user123",
        tags=["evidence", "test"],
        custom_metadata={"source": "manual_upload"}
    )


class TestFileValidator:
    """Test file validation functionality."""

    def test_validator_initialization(self):
        """Test validator initialization with default settings."""
        validator = FileValidator()

        assert validator.allowed_mime_types == DEFAULT_ALLOWED_MIME_TYPES
        assert len(validator.blocked_mime_types) > 0
        assert 'application/x-executable' in validator.blocked_mime_types

    @pytest.mark.asyncio
    async def test_validate_allowed_file_type(self, sample_file_data):
        """Test validation of allowed file types."""
        validator = FileValidator()

        result = await validator.validate_file(
            sample_file_data,
            "document.txt",
            "text/plain"
        )

        assert result.is_valid
        assert result.mime_type == "text/plain"
        assert result.size_bytes == len(sample_file_data)
        assert len(result.sha256_hash) == 64  # SHA256 hex length
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_blocked_file_type(self, sample_file_data):
        """Test validation blocks dangerous file types."""
        validator = FileValidator()

        result = await validator.validate_file(
            sample_file_data,
            "malware.exe",
            "application/x-executable"
        )

        assert not result.is_valid
        assert len(result.errors) > 0
        assert any("not allowed" in error for error in result.errors)

    @pytest.mark.asyncio
    async def test_validate_file_size_limit(self):
        """Test file size validation."""
        validator = FileValidator(max_file_sizes={'text': 100})  # 100 bytes limit

        large_file = b"x" * 200  # 200 bytes

        result = await validator.validate_file(
            large_file,
            "large.txt",
            "text/plain"
        )

        assert not result.is_valid
        assert any("exceeds maximum" in error for error in result.errors)

    def test_filename_validation(self):
        """Test filename security validation."""
        validator = FileValidator()

        # Test path traversal
        errors = validator._validate_filename("../../../etc/passwd")
        assert len(errors) > 0
        assert any("path traversal" in error for error in errors)

        # Test suspicious extension
        errors = validator._validate_filename("script.exe")
        assert len(errors) > 0
        assert any("Suspicious file extension" in error for error in errors)

        # Test valid filename
        errors = validator._validate_filename("document.pdf")
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_content_scanning(self):
        """Test content-based validation."""
        validator = FileValidator(enable_content_scanning=True)

        # Test suspicious script content
        malicious_content = b"<script>alert('xss')</script>"

        result = await validator.validate_file(
            malicious_content,
            "file.txt",
            "text/plain"
        )

        # Should detect suspicious patterns
        assert len(result.errors) > 0 or len(result.warnings) > 0

    @pytest.mark.asyncio
    async def test_malware_scanning_unavailable(self, sample_file_data):
        """Test handling when ClamAV is not available."""
        validator = FileValidator()

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("ClamAV not found")

            result = await validator.validate_file(
                sample_file_data,
                "test.txt",
                "text/plain"
            )

            assert result.malware_scan_result is None
            assert any("Malware scan unavailable" in warning for warning in result.warnings)


class TestFilesystemStorage:
    """Test filesystem storage driver."""

    def test_driver_initialization(self, filesystem_config):
        """Test filesystem driver initialization."""
        driver = FilesystemStorageDriver(filesystem_config)

        assert driver.config == filesystem_config
        assert driver.storage_root.name == "test_storage"

    @pytest.mark.asyncio
    async def test_generate_presigned_upload_url(self, filesystem_config, presigned_request):
        """Test presigned upload URL generation."""
        driver = FilesystemStorageDriver(filesystem_config)

        with patch.object(driver, '_store_upload_token') as mock_store:
            mock_store.return_value = None

            response = await driver.generate_presigned_upload_url(presigned_request)

            assert response.upload_url.startswith("/api/storage/upload/")
            assert response.file_id is not None
            assert response.expires_at > datetime.utcnow()
            mock_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_file_direct(self, filesystem_config, sample_file_data):
        """Test direct file upload."""
        from app.storage.interface import FileMetadata

        driver = FilesystemStorageDriver(filesystem_config)

        metadata = FileMetadata(
            filename="test.txt",
            content_type="text/plain",
            size_bytes=len(sample_file_data),
            sha256_hash="test_hash",
            uploaded_by="user123",
            tenant_id=uuid4(),
            storage_path="tenant/2024/08/file.txt",
            storage_backend=StorageBackend.FILESYSTEM,
            created_at=datetime.utcnow()
        )

        # Mock file operations
        with patch('aiofiles.open', create=True) as mock_open:
            mock_file = AsyncMock()
            mock_open.return_value.__aenter__.return_value = mock_file
            mock_file.write = AsyncMock()

            with patch('aiofiles.os.chmod') as mock_chmod:
                mock_chmod.return_value = None

                result = await driver.upload_file(sample_file_data, metadata)

                assert result.status == FileStatus.UPLOADED
                assert result.updated_at is not None
                mock_file.write.assert_called_once_with(sample_file_data)

    @pytest.mark.asyncio
    async def test_file_exists_check(self, filesystem_config):
        """Test file existence check."""
        driver = FilesystemStorageDriver(filesystem_config)

        with patch.object(driver.storage_root, 'exists') as mock_exists:
            mock_exists.return_value = True

            # Create a mock Path object for the file
            with patch('pathlib.Path.__truediv__') as mock_truediv:
                mock_path = Mock()
                mock_path.exists.return_value = True
                mock_truediv.return_value = mock_path

                exists = await driver.file_exists("test/file.txt")
                assert exists is True


class TestS3Storage:
    """Test S3 storage driver."""

    def test_driver_initialization(self, s3_config):
        """Test S3 driver initialization."""
        with patch('boto3.client') as mock_boto3:
            mock_client = Mock()
            mock_boto3.return_value = mock_client

            driver = S3StorageDriver(s3_config)

            assert driver.config == s3_config
            assert driver.s3_client == mock_client
            mock_boto3.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_presigned_upload_url(self, s3_config, presigned_request):
        """Test S3 presigned upload URL generation."""
        with patch('boto3.client') as mock_boto3:
            mock_client = Mock()
            mock_boto3.return_value = mock_client

            # Mock presigned POST response
            mock_client.generate_presigned_post.return_value = {
                'url': 'https://test-bucket.s3.amazonaws.com/',
                'fields': {
                    'key': 'tenant/file.pdf',
                    'Content-Type': 'application/pdf'
                }
            }

            driver = S3StorageDriver(s3_config)

            with patch('asyncio.get_event_loop') as mock_loop:
                mock_executor = Mock()
                mock_executor.return_value = mock_client.generate_presigned_post.return_value
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_executor.return_value)

                response = await driver.generate_presigned_upload_url(presigned_request)

                assert response.upload_url == 'https://test-bucket.s3.amazonaws.com/'
                assert response.file_id is not None
                assert 'key' in response.required_headers

    @pytest.mark.asyncio
    async def test_upload_file_direct(self, s3_config, sample_file_data):
        """Test direct S3 file upload."""
        from app.storage.interface import FileMetadata

        with patch('boto3.client') as mock_boto3:
            mock_client = Mock()
            mock_boto3.return_value = mock_client

            driver = S3StorageDriver(s3_config)

            metadata = FileMetadata(
                filename="test.txt",
                content_type="text/plain",
                size_bytes=len(sample_file_data),
                sha256_hash="test_hash",
                uploaded_by="user123",
                tenant_id=uuid4(),
                storage_path="tenant/2024/08/file.txt",
                storage_backend=StorageBackend.S3,
                created_at=datetime.utcnow()
            )

            # Mock S3 put_object
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=None)

                result = await driver.upload_file(sample_file_data, metadata)

                assert result.status == FileStatus.UPLOADED
                assert result.updated_at is not None


class TestStorageService:
    """Test storage service functionality."""

    @pytest.fixture
    def mock_storage_driver(self, filesystem_config):
        """Mock storage driver for testing."""
        driver = Mock(spec=FilesystemStorageDriver)
        driver.config = filesystem_config
        return driver

    @pytest.fixture
    def storage_service(self, mock_storage_driver):
        """Storage service with mocked dependencies."""
        with patch('app.services.storage_service.get_file_validator') as mock_validator:
            mock_validator.return_value = Mock()
            service = StorageService(mock_storage_driver)
            return service

    @pytest.mark.asyncio
    async def test_create_upload_url(self, storage_service, mock_storage_driver):
        """Test upload URL creation."""
        tenant_id = uuid4()

        # Mock presigned URL response
        from app.storage.interface import PresignedUrlResponse
        mock_response = PresignedUrlResponse(
            upload_url="https://example.com/upload",
            file_id=uuid4(),
            expires_at=datetime.utcnow() + timedelta(hours=1),
            required_headers={}
        )

        mock_storage_driver.generate_presigned_upload_url = AsyncMock(return_value=mock_response)
        mock_storage_driver.generate_file_path.return_value = "tenant/file.pdf"

        # Mock database session
        with patch.object(storage_service, 'session_factory') as mock_session_factory:
            mock_session = AsyncMock()
            mock_session_factory.return_value.__aenter__.return_value = mock_session
            mock_session_factory.return_value.__aexit__.return_value = None

            result = await storage_service.create_upload_url(
                filename="test.pdf",
                content_type="application/pdf",
                size_bytes=1024,
                tenant_id=tenant_id,
                uploaded_by="user123"
            )

            assert "upload_url" in result
            assert "file_id" in result
            assert "expires_at" in result
            mock_storage_driver.generate_presigned_upload_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_upload_with_validation(self, storage_service, mock_storage_driver):
        """Test upload completion with file validation."""
        file_id = uuid4()
        tenant_id = uuid4()

        # Mock evidence from database
        from app.domain.tenant_entities import Evidence
        mock_evidence = Evidence(
            id=file_id,
            tenant_id=tenant_id,
            filename="test.pdf",
            content_type="application/pdf",
            size_bytes=1024,
            storage_path="tenant/file.pdf",
            storage_backend="filesystem",
            status=FileStatus.UPLOADING.value,
            uploaded_by="user123",
            created_at=datetime.utcnow()
        )

        # Mock storage operations
        mock_storage_driver.file_exists = AsyncMock(return_value=True)
        mock_storage_driver.get_file_metadata = AsyncMock(return_value={'size_bytes': 1024})
        mock_storage_driver.download_file = AsyncMock()
        mock_storage_driver.download_file.return_value.__aiter__ = AsyncMock(return_value=iter([b"test data"]))

        # Mock file validation
        from app.storage.validation import FileValidationResult
        mock_validation = FileValidationResult(
            is_valid=True,
            mime_type="application/pdf",
            size_bytes=1024,
            sha256_hash="test_hash",
            errors=[],
            warnings=[],
            malware_scan_result="CLEAN"
        )
        storage_service.file_validator.validate_file = AsyncMock(return_value=mock_validation)

        # Mock database session
        with patch.object(storage_service, 'session_factory') as mock_session_factory:
            mock_session = AsyncMock()
            mock_session_factory.return_value.__aenter__.return_value = mock_session
            mock_session_factory.return_value.__aexit__.return_value = None

            # Mock database query
            mock_result = Mock()
            mock_result.scalar_one_or_none.return_value = mock_evidence
            mock_session.execute.return_value = mock_result

            result = await storage_service.complete_upload(
                file_id=file_id,
                tenant_id=tenant_id,
                validate_file=True
            )

            assert result["status"] == "uploaded"
            assert result["file_id"] == str(file_id)
            assert "validation" in result
            storage_service.file_validator.validate_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_upload_malware_detected(self, storage_service, mock_storage_driver):
        """Test upload completion when malware is detected."""
        file_id = uuid4()
        tenant_id = uuid4()

        # Mock evidence
        from app.domain.tenant_entities import Evidence
        mock_evidence = Evidence(
            id=file_id,
            tenant_id=tenant_id,
            filename="malware.exe",
            content_type="application/octet-stream",
            status=FileStatus.UPLOADING.value,
            uploaded_by="user123",
            created_at=datetime.utcnow()
        )

        # Mock storage operations
        mock_storage_driver.file_exists = AsyncMock(return_value=True)
        mock_storage_driver.get_file_metadata = AsyncMock(return_value={'size_bytes': 1024})
        mock_storage_driver.download_file = AsyncMock()
        mock_storage_driver.download_file.return_value.__aiter__ = AsyncMock(return_value=iter([b"malware"]))

        # Mock validation with malware detection
        from app.storage.validation import FileValidationResult
        mock_validation = FileValidationResult(
            is_valid=True,
            mime_type="application/octet-stream",
            size_bytes=1024,
            sha256_hash="malware_hash",
            errors=[],
            warnings=[],
            malware_scan_result="INFECTED",
            malware_scan_details={"threat": "Win32.Trojan"}
        )
        storage_service.file_validator.validate_file = AsyncMock(return_value=mock_validation)

        # Mock database session
        with patch.object(storage_service, 'session_factory') as mock_session_factory:
            mock_session = AsyncMock()
            mock_session_factory.return_value.__aenter__.return_value = mock_session
            mock_session_factory.return_value.__aexit__.return_value = None

            mock_result = Mock()
            mock_result.scalar_one_or_none.return_value = mock_evidence
            mock_session.execute.return_value = mock_result

            result = await storage_service.complete_upload(
                file_id=file_id,
                tenant_id=tenant_id,
                validate_file=True
            )

            assert result["status"] == "quarantined"
            assert result["reason"] == "Malware detected"
            assert "scan_details" in result


class TestStorageIntegration:
    """Integration tests for storage system."""

    @pytest.mark.asyncio
    async def test_upload_download_roundtrip(self):
        """Test complete upload and download flow."""
        # This would be a full integration test:
        # 1. Create presigned upload URL
        # 2. Upload file using the URL
        # 3. Complete upload with validation
        # 4. Create download URL
        # 5. Download file and verify content
        pass

    @pytest.mark.asyncio
    async def test_tenant_isolation_in_storage(self):
        """Test that storage respects tenant isolation."""
        # This would test:
        # 1. Upload file as tenant A
        # 2. Try to access file as tenant B
        # 3. Verify access is denied
        pass

    @pytest.mark.asyncio
    async def test_file_validation_end_to_end(self):
        """Test file validation in real upload scenario."""
        # This would test:
        # 1. Upload various file types
        # 2. Verify validation results
        # 3. Test malware scanning if available
        # 4. Verify quarantine functionality
        pass


def test_storage_driver_factory():
    """Test storage driver factory registration and creation."""
    # Test factory has registered drivers
    backends = StorageDriverFactory.list_backends()
    assert StorageBackend.FILESYSTEM in backends
    assert StorageBackend.S3 in backends

    # Test driver creation
    fs_config = FilesystemConfig(
        backend=StorageBackend.FILESYSTEM,
        base_path="/tmp/test",
        storage_root="/tmp/test"
    )

    driver = StorageDriverFactory.create_driver(fs_config)
    assert isinstance(driver, FilesystemStorageDriver)

    # Test unsupported backend
    with pytest.raises(ValueError):
        bad_config = Mock()
        bad_config.backend = "unsupported"
        StorageDriverFactory.create_driver(bad_config)
