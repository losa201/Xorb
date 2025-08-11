"""S3 storage driver implementation."""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import AsyncIterator, Dict, List, Optional, Tuple
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config

from .interface import (
    StorageDriver, S3Config, FileMetadata,
    PresignedUrlRequest, PresignedUrlResponse,
    DownloadUrlRequest, DownloadUrlResponse,
    StorageBackend, FileStatus
)


logger = logging.getLogger(__name__)


class S3StorageDriver(StorageDriver):
    """S3-compatible storage driver."""
    
    def __init__(self, config: S3Config):
        super().__init__(config)
        self.config: S3Config = config
        
        # Configure boto3 client
        session_config = Config(
            region_name=config.region,
            use_ssl=config.use_ssl,
            retries={
                'max_attempts': 3,
                'mode': 'adaptive'
            }
        )
        
        # Set up credentials
        credentials = {}
        if config.access_key_id and config.secret_access_key:
            credentials = {
                'aws_access_key_id': config.access_key_id,
                'aws_secret_access_key': config.secret_access_key
            }
        
        # Create S3 client
        self.s3_client = boto3.client(
            's3',
            endpoint_url=config.endpoint_url,
            config=session_config,
            **credentials
        )
        
        # Verify bucket access on initialization
        asyncio.create_task(self._verify_bucket_access())
    
    async def _verify_bucket_access(self) -> None:
        """Verify S3 bucket access and permissions."""
        try:
            # Check if bucket exists and is accessible
            await asyncio.get_event_loop().run_in_executor(
                None, 
                self.s3_client.head_bucket,
                {'Bucket': self.config.bucket_name}
            )
            logger.info(f"S3 bucket {self.config.bucket_name} is accessible")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"S3 bucket {self.config.bucket_name} does not exist")
            elif error_code == '403':
                logger.error(f"Access denied to S3 bucket {self.config.bucket_name}")
            else:
                logger.error(f"S3 bucket access error: {e}")
        except NoCredentialsError:
            logger.error("S3 credentials not found")
        except Exception as e:
            logger.error(f"S3 bucket verification failed: {e}")
    
    async def generate_presigned_upload_url(
        self, 
        request: PresignedUrlRequest
    ) -> PresignedUrlResponse:
        """Generate presigned URL for S3 upload."""
        file_id = uuid4()
        
        # Generate S3 object key
        object_key = self.generate_file_path(
            request.tenant_id, 
            request.filename,
            file_id
        )
        
        # Prepare presigned POST parameters
        conditions = [
            {'bucket': self.config.bucket_name},
            {'key': object_key},
            ['content-length-range', 1, request.size_bytes],
            {'Content-Type': request.content_type}
        ]
        
        # Add custom metadata conditions
        fields = {
            'Content-Type': request.content_type,
            'x-amz-meta-tenant-id': str(request.tenant_id),
            'x-amz-meta-uploaded-by': request.uploaded_by,
            'x-amz-meta-file-id': str(file_id),
            'x-amz-meta-original-filename': request.filename
        }
        
        # Add tags if specified
        if request.tags:
            fields['x-amz-tagging'] = '&'.join([f'tag{i}={tag}' for i, tag in enumerate(request.tags)])
        
        # Add custom metadata
        for key, value in request.custom_metadata.items():
            fields[f'x-amz-meta-{key}'] = value
        
        try:
            # Generate presigned POST
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.generate_presigned_post(
                    Bucket=self.config.bucket_name,
                    Key=object_key,
                    Fields=fields,
                    Conditions=conditions,
                    ExpiresIn=request.expires_in
                )
            )
            
            return PresignedUrlResponse(
                upload_url=response['url'],
                file_id=file_id,
                expires_at=datetime.utcnow() + timedelta(seconds=request.expires_in),
                required_headers=response['fields']
            )
            
        except Exception as e:
            logger.error(f"Failed to generate S3 presigned upload URL: {e}")
            raise
    
    async def generate_presigned_download_url(
        self, 
        request: DownloadUrlRequest
    ) -> DownloadUrlResponse:
        """Generate presigned URL for S3 download."""
        # Get file metadata from database
        file_metadata = await self._get_file_metadata_from_db(request.file_id)
        if not file_metadata:
            raise FileNotFoundError(f"File {request.file_id} not found")
        
        object_key = file_metadata['storage_path']
        
        # Prepare download parameters
        params = {
            'Bucket': self.config.bucket_name,
            'Key': object_key
        }
        
        # Set content disposition if specified
        if request.content_disposition:
            filename = request.custom_filename or file_metadata['filename']
            params['ResponseContentDisposition'] = f'{request.content_disposition}; filename="{filename}"'
        
        try:
            # Generate presigned URL
            url = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.generate_presigned_url(
                    'get_object',
                    Params=params,
                    ExpiresIn=request.expires_in
                )
            )
            
            return DownloadUrlResponse(
                download_url=url,
                filename=file_metadata['filename'],
                content_type=file_metadata['content_type'],
                size_bytes=file_metadata['size_bytes'],
                expires_at=datetime.utcnow() + timedelta(seconds=request.expires_in)
            )
            
        except Exception as e:
            logger.error(f"Failed to generate S3 presigned download URL: {e}")
            raise
    
    async def upload_file(
        self, 
        file_data: bytes, 
        metadata: FileMetadata
    ) -> FileMetadata:
        """Upload file directly to S3."""
        object_key = metadata.storage_path
        
        # Prepare metadata for S3
        s3_metadata = {
            'tenant-id': str(metadata.tenant_id),
            'uploaded-by': metadata.uploaded_by,
            'file-id': str(metadata.id) if hasattr(metadata, 'id') else str(uuid4()),
            'original-filename': metadata.original_filename or metadata.filename,
            'sha256-hash': metadata.sha256_hash
        }
        
        # Add custom metadata
        s3_metadata.update(metadata.custom_metadata)
        
        # Prepare tags
        tags = []
        for tag in metadata.tags:
            tags.append(f'tag={tag}')
        tag_string = '&'.join(tags) if tags else None
        
        try:
            # Upload to S3
            put_args = {
                'Bucket': self.config.bucket_name,
                'Key': object_key,
                'Body': file_data,
                'ContentType': metadata.content_type,
                'Metadata': s3_metadata
            }
            
            if tag_string:
                put_args['Tagging'] = tag_string
            
            # Add server-side encryption if enabled
            if self.config.enable_encryption:
                put_args['ServerSideEncryption'] = 'AES256'
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.put_object(**put_args)
            )
            
            # Update metadata
            metadata.status = FileStatus.UPLOADED
            metadata.updated_at = datetime.utcnow()
            
            logger.info(f"Uploaded file to S3: {object_key}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to upload file to S3 {object_key}: {e}")
            metadata.status = FileStatus.ERROR
            raise
    
    async def download_file(self, file_path: str) -> AsyncIterator[bytes]:
        """Download file from S3 as async iterator."""
        try:
            # Get object
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.get_object(
                    Bucket=self.config.bucket_name,
                    Key=file_path
                )
            )
            
            # Stream the body
            body = response['Body']
            chunk_size = 64 * 1024  # 64KB chunks
            
            while True:
                chunk = body.read(chunk_size)
                if not chunk:
                    break
                yield chunk
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"File not found in S3: {file_path}")
            else:
                logger.error(f"Failed to download file from S3 {file_path}: {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to download file from S3 {file_path}: {e}")
            raise
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete file from S3."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.delete_object(
                    Bucket=self.config.bucket_name,
                    Key=file_path
                )
            )
            
            logger.info(f"Deleted file from S3: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file from S3 {file_path}: {e}")
            return False
    
    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists in S3."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.head_object(
                    Bucket=self.config.bucket_name,
                    Key=file_path
                )
            )
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                logger.error(f"Error checking S3 file existence {file_path}: {e}")
                return False
        except Exception as e:
            logger.error(f"Error checking S3 file existence {file_path}: {e}")
            return False
    
    async def get_file_metadata(self, file_path: str) -> Optional[Dict]:
        """Get file metadata from S3."""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.head_object(
                    Bucket=self.config.bucket_name,
                    Key=file_path
                )
            )
            
            return {
                'size_bytes': response['ContentLength'],
                'content_type': response['ContentType'],
                'last_modified': response['LastModified'],
                'etag': response['ETag'].strip('"'),
                'metadata': response.get('Metadata', {}),
                'storage_class': response.get('StorageClass', 'STANDARD'),
                'server_side_encryption': response.get('ServerSideEncryption')
            }
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return None
            else:
                logger.error(f"Error getting S3 file metadata {file_path}: {e}")
                return None
        except Exception as e:
            logger.error(f"Error getting S3 file metadata {file_path}: {e}")
            return None
    
    async def list_files(
        self, 
        prefix: str, 
        limit: int = 100,
        continuation_token: Optional[str] = None
    ) -> Tuple[List[str], Optional[str]]:
        """List files in S3 with prefix."""
        try:
            params = {
                'Bucket': self.config.bucket_name,
                'Prefix': prefix,
                'MaxKeys': limit
            }
            
            if continuation_token:
                params['ContinuationToken'] = continuation_token
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.list_objects_v2(**params)
            )
            
            files = []
            if 'Contents' in response:
                files = [obj['Key'] for obj in response['Contents']]
            
            next_token = response.get('NextContinuationToken')
            
            return files, next_token
            
        except Exception as e:
            logger.error(f"Failed to list S3 files with prefix {prefix}: {e}")
            return [], None
    
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


# Register S3 driver
from .interface import StorageDriverFactory, StorageBackend
StorageDriverFactory.register_driver(StorageBackend.S3, S3StorageDriver)