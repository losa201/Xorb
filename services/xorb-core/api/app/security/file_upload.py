"""
Secure file upload handling with validation and virus scanning
"""

import os
import magic
import hashlib
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, BinaryIO
from dataclasses import dataclass
from enum import Enum

from fastapi import UploadFile, HTTPException, status


class FileType(Enum):
    """Allowed file types"""
    IMAGE = "image"
    DOCUMENT = "document"
    ARCHIVE = "archive"
    TEXT = "text"
    SCRIPT = "script"  # For security testing files only


@dataclass
class FileValidationResult:
    """Result of file validation"""
    is_valid: bool
    file_type: Optional[FileType]
    mime_type: str
    file_size: int
    file_hash: str
    errors: List[str]
    warnings: List[str]
    sanitized_filename: str


class SecureFileHandler:
    """Secure file upload and validation handler"""

    def __init__(self):
        # Define allowed file types and extensions
        self.allowed_types = {
            FileType.IMAGE: {
                'extensions': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'],
                'mime_types': [
                    'image/jpeg', 'image/png', 'image/gif',
                    'image/bmp', 'image/webp'
                ],
                'max_size': 10 * 1024 * 1024  # 10MB
            },
            FileType.DOCUMENT: {
                'extensions': ['.pdf', '.doc', '.docx', '.txt', '.rtf'],
                'mime_types': [
                    'application/pdf', 'application/msword',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    'text/plain', 'application/rtf'
                ],
                'max_size': 50 * 1024 * 1024  # 50MB
            },
            FileType.ARCHIVE: {
                'extensions': ['.zip', '.tar', '.gz', '.7z'],
                'mime_types': [
                    'application/zip', 'application/x-tar',
                    'application/gzip', 'application/x-7z-compressed'
                ],
                'max_size': 100 * 1024 * 1024  # 100MB
            },
            FileType.TEXT: {
                'extensions': ['.txt', '.csv', '.json', '.xml', '.yaml', '.yml'],
                'mime_types': [
                    'text/plain', 'text/csv', 'application/json',
                    'application/xml', 'text/xml', 'application/x-yaml'
                ],
                'max_size': 10 * 1024 * 1024  # 10MB
            },
            FileType.SCRIPT: {
                'extensions': ['.py', '.sh', '.js', '.sql'],
                'mime_types': [
                    'text/x-python', 'application/x-sh',
                    'application/javascript', 'application/sql'
                ],
                'max_size': 1 * 1024 * 1024  # 1MB
            }
        }

        # Dangerous file extensions to always block
        self.blocked_extensions = [
            '.exe', '.scr', '.bat', '.cmd', '.com', '.pif', '.vbs', '.js',
            '.jar', '.app', '.deb', '.pkg', '.dmg', '.msi', '.run',
            '.php', '.asp', '.jsp', '.cgi', '.pl'
        ]

        # Magic number signatures for common file types
        self.file_signatures = {
            b'\xFF\xD8\xFF': 'image/jpeg',
            b'\x89PNG\r\n\x1a\n': 'image/png',
            b'GIF87a': 'image/gif',
            b'GIF89a': 'image/gif',
            b'%PDF-': 'application/pdf',
            b'PK\x03\x04': 'application/zip',
            b'PK\x05\x06': 'application/zip',
            b'PK\x07\x08': 'application/zip',
        }

    async def validate_file(
        self,
        file: UploadFile,
        allowed_types: List[FileType] = None,
        max_size: int = None
    ) -> FileValidationResult:
        """Comprehensive file validation"""
        errors = []
        warnings = []

        # Reset file pointer
        await file.seek(0)

        # Read file content for analysis
        content = await file.read()
        await file.seek(0)

        # Basic file info
        file_size = len(content)
        file_hash = hashlib.sha256(content).hexdigest()
        original_filename = file.filename or "unknown"

        # Sanitize filename
        sanitized_filename = self._sanitize_filename(original_filename)

        # Extension validation
        file_ext = Path(sanitized_filename).suffix.lower()
        if file_ext in self.blocked_extensions:
            errors.append(f"File extension {file_ext} is not allowed")

        # Detect file type by content (magic number)
        detected_mime = self._detect_mime_type(content)
        declared_mime = file.content_type or "unknown"

        # Validate MIME type consistency
        if detected_mime != declared_mime and detected_mime != "unknown":
            warnings.append(f"MIME type mismatch: declared {declared_mime}, detected {detected_mime}")

        # Determine file type
        file_type = self._determine_file_type(file_ext, detected_mime)

        # Check if file type is allowed
        if allowed_types and file_type not in allowed_types:
            errors.append(f"File type {file_type.value if file_type else 'unknown'} is not allowed")

        # Size validation
        if file_type and file_type in self.allowed_types:
            type_config = self.allowed_types[file_type]
            max_allowed_size = max_size or type_config['max_size']

            if file_size > max_allowed_size:
                errors.append(f"File size {file_size} exceeds maximum {max_allowed_size} bytes")

        # Content validation
        content_errors = await self._validate_file_content(content, file_type)
        errors.extend(content_errors)

        # Virus scanning (if ClamAV is available)
        virus_result = await self._scan_for_viruses(content)
        if virus_result:
            errors.append(f"Virus detected: {virus_result}")

        is_valid = len(errors) == 0

        return FileValidationResult(
            is_valid=is_valid,
            file_type=file_type,
            mime_type=detected_mime or declared_mime,
            file_size=file_size,
            file_hash=file_hash,
            errors=errors,
            warnings=warnings,
            sanitized_filename=sanitized_filename
        )

    async def save_file_securely(
        self,
        file: UploadFile,
        upload_dir: str,
        allowed_types: List[FileType] = None,
        max_size: int = None
    ) -> Dict[str, Any]:
        """Securely save uploaded file"""

        # Validate file first
        validation_result = await self.validate_file(file, allowed_types, max_size)

        if not validation_result.is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "File validation failed",
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings
                }
            )

        # Create secure upload directory
        upload_path = Path(upload_dir)
        upload_path.mkdir(parents=True, exist_ok=True)

        # Generate secure filename
        secure_filename = self._generate_secure_filename(
            validation_result.sanitized_filename,
            validation_result.file_hash
        )

        file_path = upload_path / secure_filename

        # Save file with restricted permissions
        await file.seek(0)
        content = await file.read()

        with open(file_path, 'wb') as f:
            f.write(content)

        # Set restrictive file permissions
        os.chmod(file_path, 0o644)

        return {
            "filename": secure_filename,
            "original_filename": file.filename,
            "file_path": str(file_path),
            "file_size": validation_result.file_size,
            "file_hash": validation_result.file_hash,
            "file_type": validation_result.file_type.value if validation_result.file_type else None,
            "mime_type": validation_result.mime_type,
            "warnings": validation_result.warnings
        }

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for security"""
        # Remove path components
        filename = Path(filename).name

        # Remove dangerous characters
        dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\0']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')

        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext

        # Ensure filename doesn't start with dot or dash
        if filename.startswith(('.', '-')):
            filename = 'file_' + filename

        return filename

    def _detect_mime_type(self, content: bytes) -> str:
        """Detect MIME type from file content"""
        try:
            # Check magic number signatures first
            for signature, mime_type in self.file_signatures.items():
                if content.startswith(signature):
                    return mime_type

            # Use python-magic if available
            if hasattr(magic, 'from_buffer'):
                return magic.from_buffer(content, mime=True)

        except Exception:
            pass

        return "unknown"

    def _determine_file_type(self, extension: str, mime_type: str) -> Optional[FileType]:
        """Determine file type from extension and MIME type"""
        for file_type, config in self.allowed_types.items():
            if (extension in config['extensions'] or
                mime_type in config['mime_types']):
                return file_type

        return None

    async def _validate_file_content(self, content: bytes, file_type: Optional[FileType]) -> List[str]:
        """Validate file content for security issues"""
        errors = []

        # Check for embedded executables
        if b'\x4D\x5A' in content[:1024]:  # MZ header (PE executable)
            errors.append("Embedded executable detected")

        # Check for script injection in text files
        if file_type in [FileType.TEXT, FileType.DOCUMENT]:
            text_content = content.decode('utf-8', errors='ignore')

            dangerous_patterns = [
                '<script', 'javascript:', 'vbscript:',
                'eval(', 'exec(', 'system(', 'shell_exec('
            ]

            for pattern in dangerous_patterns:
                if pattern.lower() in text_content.lower():
                    errors.append(f"Potentially dangerous content detected: {pattern}")

        # Check for ZIP bombs (nested archives)
        if file_type == FileType.ARCHIVE:
            if self._check_zip_bomb(content):
                errors.append("Potential zip bomb detected")

        return errors

    def _check_zip_bomb(self, content: bytes) -> bool:
        """Check for zip bomb patterns"""
        try:
            import zipfile
            import io

            # Check compression ratio
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                total_compressed = 0
                total_uncompressed = 0

                for info in zf.infolist():
                    total_compressed += info.compress_size
                    total_uncompressed += info.file_size

                    # Individual file check
                    if info.file_size > 100 * 1024 * 1024:  # 100MB
                        return True

                # Overall compression ratio check
                if total_compressed > 0:
                    ratio = total_uncompressed / total_compressed
                    if ratio > 100:  # 100:1 compression ratio
                        return True

        except Exception:
            pass

        return False

    async def _scan_for_viruses(self, content: bytes) -> Optional[str]:
        """Scan file content for viruses using ClamAV"""
        try:
            import pyclamd

            # Initialize ClamAV daemon connection
            cd = pyclamd.ClamdAgentNetwork()

            if cd.ping():
                # Scan content
                result = cd.scan_stream(content)
                if result:
                    return result.get('stream', 'Unknown virus')

        except ImportError:
            # ClamAV not available, skip virus scanning
            pass
        except Exception as e:
            # Log error but don't fail upload
            print(f"Virus scanning failed: {e}")

        return None

    def _generate_secure_filename(self, original_filename: str, file_hash: str) -> str:
        """Generate secure filename with hash"""
        name, ext = os.path.splitext(original_filename)

        # Use first 8 characters of hash for uniqueness
        hash_prefix = file_hash[:8]

        # Create secure filename
        secure_name = f"{name}_{hash_prefix}{ext}"

        return self._sanitize_filename(secure_name)


# Global file handler instance
secure_file_handler = SecureFileHandler()


# FastAPI dependencies
async def validate_upload_file(
    file: UploadFile,
    allowed_types: List[FileType] = None,
    max_size: int = None
) -> FileValidationResult:
    """Dependency for file upload validation"""
    return await secure_file_handler.validate_file(file, allowed_types, max_size)


# Example usage in routes
"""
from fastapi import Depends, File, UploadFile

@app.post("/upload/image")
async def upload_image(
    file: UploadFile = File(...),
    validation: FileValidationResult = Depends(
        lambda f: validate_upload_file(f, [FileType.IMAGE], 5*1024*1024)
    )
):
    if not validation.is_valid:
        raise HTTPException(400, detail=validation.errors)

    # Save file securely
    result = await secure_file_handler.save_file_securely(
        file, "uploads/images", [FileType.IMAGE]
    )

    return result
"""
