"""Storage and evidence management API routes."""
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form, status
from fastapi.responses import RedirectResponse

from ..auth.dependencies import require_auth, require_permissions
from ..auth.models import Permission, UserClaims
from ..middleware.tenant_context import require_tenant_context
from ..services.storage_service import StorageService
from ..storage.interface import StorageDriverFactory, FilesystemConfig, StorageBackend
# Import to register drivers
from ..storage import filesystem, s3
from ..security.input_validation import FileUploadValidation, validate_pagination
from ..infrastructure.observability import get_metrics_collector, add_trace_context
import structlog

logger = structlog.get_logger("storage_api")
router = APIRouter(prefix="/api/storage", tags=["Storage"])

# Initialize storage service (would be dependency injected in production)
storage_config = FilesystemConfig(
    backend=StorageBackend.FILESYSTEM,
    storage_root="/app/data/evidence",
    base_path="evidence"
)
storage_driver = StorageDriverFactory.create_driver(storage_config)
storage_service = StorageService(storage_driver)


@router.post("/upload-url", response_model=Dict)
async def create_upload_url(
    request: Request,
    filename: str = Form(...),
    content_type: str = Form(...),
    size_bytes: int = Form(...),
    tags: Optional[List[str]] = Form(default=[]),
    current_user: UserClaims = Depends(require_permissions(Permission.EVIDENCE_WRITE))
):
    """Create presigned upload URL for evidence files."""
    tenant_id = require_tenant_context(request)
    
    # Validate file upload parameters
    validation = FileUploadValidation(
        filename=filename,
        content_type=content_type,
        size=size_bytes
    )
    
    try:
        upload_info = await storage_service.create_upload_url(
            filename=validation.filename,
            content_type=validation.content_type,
            size_bytes=validation.size,
            tenant_id=tenant_id,
            uploaded_by=current_user.sub,
            tags=tags or []
        )
        
        # Record metrics
        metrics = get_metrics_collector()
        metrics.record_evidence_upload(str(tenant_id), size_bytes, True)
        
        # Add trace context
        add_trace_context(
            operation="create_upload_url",
            tenant_id=str(tenant_id),
            file_size=size_bytes,
            content_type=content_type
        )
        
        logger.info(
            "Created upload URL",
            tenant_id=str(tenant_id),
            filename=filename,
            size_bytes=size_bytes,
            user_id=current_user.sub
        )
        
        return upload_info
        
    except Exception as e:
        logger.error(
            "Failed to create upload URL",
            tenant_id=str(tenant_id),
            filename=filename,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create upload URL"
        )


@router.post("/complete/{file_id}")
async def complete_upload(
    request: Request,
    file_id: UUID,
    validate_file: bool = True,
    current_user: UserClaims = Depends(require_permissions(Permission.EVIDENCE_WRITE))
):
    """Complete file upload and validate."""
    tenant_id = require_tenant_context(request)
    
    try:
        result = await storage_service.complete_upload(
            file_id=file_id,
            tenant_id=tenant_id,
            validate_file=validate_file
        )
        
        logger.info(
            "Completed upload",
            tenant_id=str(tenant_id),
            file_id=str(file_id),
            status=result["status"],
            user_id=current_user.sub
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Failed to complete upload",
            tenant_id=str(tenant_id),
            file_id=str(file_id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete upload"
        )


@router.get("/download/{file_id}")
async def create_download_url(
    request: Request,
    file_id: UUID,
    expires_in: int = 3600,
    content_disposition: str = "attachment",
    custom_filename: Optional[str] = None,
    current_user: UserClaims = Depends(require_permissions(Permission.EVIDENCE_READ))
):
    """Create presigned download URL for evidence file."""
    tenant_id = require_tenant_context(request)
    
    try:
        download_info = await storage_service.create_download_url(
            file_id=file_id,
            tenant_id=tenant_id,
            expires_in=expires_in,
            content_disposition=content_disposition,
            custom_filename=custom_filename
        )
        
        logger.info(
            "Created download URL",
            tenant_id=str(tenant_id),
            file_id=str(file_id),
            user_id=current_user.sub
        )
        
        return download_info
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Failed to create download URL",
            tenant_id=str(tenant_id),
            file_id=str(file_id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create download URL"
        )


@router.get("/evidence", response_model=List[Dict])
async def list_evidence(
    request: Request,
    limit: int = 20,
    offset: int = 0,
    status_filter: Optional[str] = None,
    current_user: UserClaims = Depends(require_permissions(Permission.EVIDENCE_READ))
):
    """List evidence files for tenant."""
    tenant_id = require_tenant_context(request)
    
    # Validate pagination
    limit, offset = validate_pagination(limit, offset, max_limit=100)
    
    try:
        evidence_list = await storage_service.list_evidence(
            tenant_id=tenant_id,
            status=status_filter,
            limit=limit,
            offset=offset
        )
        
        logger.info(
            "Listed evidence",
            tenant_id=str(tenant_id),
            count=len(evidence_list),
            user_id=current_user.sub
        )
        
        return evidence_list
        
    except Exception as e:
        logger.error(
            "Failed to list evidence",
            tenant_id=str(tenant_id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list evidence"
        )


@router.get("/evidence/{file_id}")
async def get_evidence_metadata(
    request: Request,
    file_id: UUID,
    current_user: UserClaims = Depends(require_permissions(Permission.EVIDENCE_READ))
):
    """Get evidence metadata."""
    tenant_id = require_tenant_context(request)
    
    try:
        metadata = await storage_service.get_evidence_metadata(
            file_id=file_id,
            tenant_id=tenant_id
        )
        
        if not metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Evidence not found"
            )
        
        return metadata
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get evidence metadata",
            tenant_id=str(tenant_id),
            file_id=str(file_id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get evidence metadata"
        )


@router.delete("/evidence/{file_id}")
async def delete_evidence(
    request: Request,
    file_id: UUID,
    permanent: bool = False,
    current_user: UserClaims = Depends(require_permissions(Permission.EVIDENCE_DELETE))
):
    """Delete evidence file."""
    tenant_id = require_tenant_context(request)
    
    try:
        deleted = await storage_service.delete_evidence(
            file_id=file_id,
            tenant_id=tenant_id,
            permanent=permanent
        )
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Evidence not found"
            )
        
        logger.info(
            "Deleted evidence",
            tenant_id=str(tenant_id),
            file_id=str(file_id),
            permanent=permanent,
            user_id=current_user.sub
        )
        
        return {"message": "Evidence deleted successfully", "permanent": permanent}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to delete evidence",
            tenant_id=str(tenant_id),
            file_id=str(file_id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete evidence"
        )


# Direct upload endpoint (alternative to presigned URLs)
@router.post("/upload")
async def upload_file_direct(
    request: Request,
    file: UploadFile = File(...),
    tags: Optional[str] = Form(default=""),
    current_user: UserClaims = Depends(require_permissions(Permission.EVIDENCE_WRITE))
):
    """Direct file upload endpoint."""
    tenant_id = require_tenant_context(request)
    
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )
    
    # Read file data
    file_data = await file.read()
    
    # Validate file upload
    validation = FileUploadValidation(
        filename=file.filename,
        content_type=file.content_type or "application/octet-stream",
        size=len(file_data)
    )
    
    try:
        # Create upload URL first
        upload_info = await storage_service.create_upload_url(
            filename=validation.filename,
            content_type=validation.content_type,
            size_bytes=validation.size,
            tenant_id=tenant_id,
            uploaded_by=current_user.sub,
            tags=tags.split(",") if tags else []
        )
        
        # Complete upload
        result = await storage_service.complete_upload(
            file_id=UUID(upload_info["file_id"]),
            tenant_id=tenant_id,
            validate_file=True
        )
        
        logger.info(
            "Direct upload completed",
            tenant_id=str(tenant_id),
            filename=file.filename,
            size_bytes=len(file_data),
            user_id=current_user.sub
        )
        
        return result
        
    except Exception as e:
        logger.error(
            "Direct upload failed",
            tenant_id=str(tenant_id),
            filename=file.filename,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Upload failed"
        )