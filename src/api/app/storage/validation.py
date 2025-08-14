"""File validation utilities."""
import hashlib
import logging
import magic
import subprocess
from io import BytesIO
from typing import Dict, List, Optional, Set

from .interface import FileValidationResult


logger = logging.getLogger(__name__)


# Default allowed MIME types for evidence files
DEFAULT_ALLOWED_MIME_TYPES = {
    # Images
    'image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/tiff',
    'image/webp', 'image/svg+xml',

    # Documents
    'application/pdf', 'text/plain', 'text/csv', 'text/xml',
    'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-powerpoint', 'application/vnd.openxmlformats-officedocument.presentationml.presentation',

    # Archives
    'application/zip', 'application/x-tar', 'application/gzip',
    'application/x-7z-compressed', 'application/x-rar-compressed',

    # Programming files
    'text/x-python', 'text/x-java-source', 'text/x-c', 'text/x-shellscript',
    'application/json', 'application/xml', 'text/html', 'text/css',
    'application/javascript', 'text/x-sql',

    # Log files
    'text/x-log', 'application/x-ndjson',

    # Security-specific formats
    'application/x-pcap', 'application/vnd.tcpdump.pcap',
    'application/x-suricata-eve', 'application/x-zeek-log',
}

# MIME types that should be blocked for security reasons
BLOCKED_MIME_TYPES = {
    'application/x-executable', 'application/x-msdownload',
    'application/vnd.microsoft.portable-executable',
    'application/x-sharedlib', 'application/x-mach-binary',
    'application/x-dosexec', 'application/octet-stream',
}

# Maximum file sizes by category (in bytes)
MAX_FILE_SIZES = {
    'image': 50 * 1024 * 1024,      # 50MB
    'document': 100 * 1024 * 1024,  # 100MB
    'archive': 500 * 1024 * 1024,   # 500MB
    'text': 10 * 1024 * 1024,       # 10MB
    'default': 100 * 1024 * 1024,   # 100MB
}


class FileValidator:
    """File validation and security scanner."""

    def __init__(
        self,
        allowed_mime_types: Optional[Set[str]] = None,
        blocked_mime_types: Optional[Set[str]] = None,
        max_file_sizes: Optional[Dict[str, int]] = None,
        enable_content_scanning: bool = True
    ):
        self.allowed_mime_types = allowed_mime_types or DEFAULT_ALLOWED_MIME_TYPES
        self.blocked_mime_types = blocked_mime_types or BLOCKED_MIME_TYPES
        self.max_file_sizes = max_file_sizes or MAX_FILE_SIZES
        self.enable_content_scanning = enable_content_scanning

        # Initialize python-magic
        try:
            self.magic_mime = magic.Magic(mime=True)
            self.magic_desc = magic.Magic()
        except Exception as e:
            logger.warning(f"Failed to initialize python-magic: {e}")
            self.magic_mime = None
            self.magic_desc = None

    async def validate_file(
        self,
        file_data: bytes,
        filename: str,
        declared_content_type: str
    ) -> FileValidationResult:
        """Validate file content and metadata."""
        errors = []
        warnings = []

        # Calculate file hash
        sha256_hash = hashlib.sha256(file_data).hexdigest()
        size_bytes = len(file_data)

        # Detect actual MIME type
        detected_type = None
        if self.magic_mime:
            try:
                detected_type = self.magic_mime.from_buffer(file_data)
            except Exception as e:
                logger.warning(f"Failed to detect MIME type: {e}")
                warnings.append("Could not detect file type")

        # Use detected type if available, otherwise trust declared type
        actual_mime_type = detected_type or declared_content_type

        # Validate MIME type
        if actual_mime_type in self.blocked_mime_types:
            errors.append(f"File type {actual_mime_type} is not allowed")
        elif actual_mime_type not in self.allowed_mime_types:
            errors.append(f"File type {actual_mime_type} is not in allowed types")

        # Check if declared type matches detected type
        if detected_type and detected_type != declared_content_type:
            warnings.append(
                f"Declared type {declared_content_type} doesn't match detected type {detected_type}"
            )

        # Validate file size
        category = self._get_mime_category(actual_mime_type)
        max_size = self.max_file_sizes.get(category, self.max_file_sizes['default'])
        if size_bytes > max_size:
            errors.append(f"File size {size_bytes} exceeds maximum {max_size} for {category} files")

        # Validate filename
        filename_errors = self._validate_filename(filename)
        errors.extend(filename_errors)

        # Content-based validation
        if self.enable_content_scanning:
            content_errors = await self._scan_file_content(file_data, actual_mime_type)
            errors.extend(content_errors)

        # Malware scanning (if ClamAV is available)
        malware_result = None
        malware_details = None
        try:
            malware_result, malware_details = await self._scan_malware(file_data)
            if malware_result == "INFECTED":
                errors.append("File contains malware")
        except Exception as e:
            logger.warning(f"Malware scan failed: {e}")
            warnings.append("Malware scan unavailable")

        return FileValidationResult(
            is_valid=len(errors) == 0,
            mime_type=actual_mime_type,
            detected_type=detected_type,
            size_bytes=size_bytes,
            sha256_hash=sha256_hash,
            errors=errors,
            warnings=warnings,
            malware_scan_result=malware_result,
            malware_scan_details=malware_details
        )

    def _get_mime_category(self, mime_type: str) -> str:
        """Get category for MIME type."""
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
        elif 'zip' in mime_type or 'tar' in mime_type or 'compressed' in mime_type:
            return 'archive'
        else:
            return 'default'

    def _validate_filename(self, filename: str) -> List[str]:
        """Validate filename for security issues."""
        errors = []

        # Check for path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            errors.append("Filename contains path traversal characters")

        # Check for null bytes
        if '\x00' in filename:
            errors.append("Filename contains null bytes")

        # Check length
        if len(filename) > 255:
            errors.append("Filename too long (max 255 characters)")

        # Check for empty filename
        if not filename.strip():
            errors.append("Filename cannot be empty")

        # Check for suspicious extensions
        suspicious_extensions = [
            '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js',
            '.jar', '.sh', '.ps1', '.php', '.asp', '.jsp'
        ]

        for ext in suspicious_extensions:
            if filename.lower().endswith(ext):
                errors.append(f"Suspicious file extension: {ext}")

        return errors

    async def _scan_file_content(self, file_data: bytes, mime_type: str) -> List[str]:
        """Scan file content for suspicious patterns."""
        errors = []

        try:
            # For text files, check for suspicious content
            if mime_type.startswith('text/') or mime_type == 'application/json':
                try:
                    content = file_data.decode('utf-8', errors='ignore')

                    # Check for potential code injection
                    suspicious_patterns = [
                        '<script', 'javascript:', 'vbscript:', 'data:',
                        'eval(', 'exec(', 'system(', 'shell_exec(',
                        'passthru(', 'file_get_contents(', 'fopen(',
                        'include(', 'require(', 'import os', 'import subprocess'
                    ]

                    for pattern in suspicious_patterns:
                        if pattern.lower() in content.lower():
                            errors.append(f"Suspicious content pattern detected: {pattern}")
                            break  # Don't spam with multiple detections

                except UnicodeDecodeError:
                    # Binary data in text file
                    errors.append("Text file contains binary data")

            # Check for embedded executables in any file type
            if b'MZ' in file_data[:1024] or b'\x7fELF' in file_data[:1024]:
                errors.append("File appears to contain embedded executable")

        except Exception as e:
            logger.warning(f"Content scan failed: {e}")

        return errors

    async def _scan_malware(self, file_data: bytes) -> tuple[Optional[str], Optional[Dict]]:
        """Scan file for malware using ClamAV."""
        try:
            # Check if clamdscan is available
            result = subprocess.run(
                ['clamdscan', '--version'],
                capture_output=True,
                timeout=5
            )

            if result.returncode != 0:
                return None, {"error": "ClamAV not available"}

            # Scan file data via stdin
            result = subprocess.run(
                ['clamdscan', '--fdpass', '-'],
                input=file_data,
                capture_output=True,
                timeout=30
            )

            output = result.stdout.decode('utf-8', errors='ignore')

            if 'FOUND' in output:
                return "INFECTED", {"output": output}
            elif 'OK' in output:
                return "CLEAN", {"output": output}
            else:
                return "ERROR", {"output": output}

        except subprocess.TimeoutExpired:
            return "ERROR", {"error": "Scan timeout"}
        except FileNotFoundError:
            return None, {"error": "ClamAV not installed"}
        except Exception as e:
            logger.warning(f"Malware scan error: {e}")
            return "ERROR", {"error": str(e)}


# Global validator instance
_file_validator: Optional[FileValidator] = None


def init_file_validator(
    allowed_mime_types: Optional[Set[str]] = None,
    blocked_mime_types: Optional[Set[str]] = None,
    max_file_sizes: Optional[Dict[str, int]] = None,
    enable_content_scanning: bool = True
) -> None:
    """Initialize global file validator."""
    global _file_validator
    _file_validator = FileValidator(
        allowed_mime_types=allowed_mime_types,
        blocked_mime_types=blocked_mime_types,
        max_file_sizes=max_file_sizes,
        enable_content_scanning=enable_content_scanning
    )


def get_file_validator() -> FileValidator:
    """Get global file validator instance."""
    if _file_validator is None:
        init_file_validator()
    return _file_validator
