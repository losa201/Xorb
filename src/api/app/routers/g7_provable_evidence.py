"""
XORB Phase G7 Provable Evidence API Router
REST API endpoints for cryptographically signed evidence with trusted timestamps.
"""

import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import io
import json

from ..services.g7_provable_evidence_service import (
    ProvableEvidenceService,
    get_provable_evidence_service,
    EvidenceType,
    EvidenceFormat,
    ProvableEvidence,
    EvidenceMetadata,
    ChainOfCustodyEntry
)


router = APIRouter(
    prefix="/provable-evidence",
    tags=["G7 Provable Evidence"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Evidence not found"},
    }
)


# Pydantic models for API
class CreateEvidenceRequest(BaseModel):
    """Request to create new provable evidence."""
    evidence_type: str = Field(..., description="Type of evidence (scan_result, vulnerability_report, etc.)")
    format: str = Field(..., description="Evidence format (json, xml, pcap, etc.)")
    title: str = Field(..., max_length=200, description="Evidence title")
    description: str = Field(..., max_length=1000, description="Evidence description")
    source_system: str = Field(..., max_length=100, description="Source system that generated evidence")
    source_user: str = Field(..., max_length=100, description="User who created evidence")
    tags: Optional[List[str]] = Field(None, description="Evidence tags for categorization")


class EvidenceResponse(BaseModel):
    """Response containing evidence metadata and verification info."""
    evidence_id: str
    tenant_id: str
    evidence_type: str
    format: str
    title: str
    description: str
    source_system: str
    source_user: str
    created_at: datetime
    size_bytes: int
    content_hash: str
    signature_valid: Optional[bool] = None
    timestamp_valid: Optional[bool] = None
    storage_references: Dict[str, str] = {}
    tags: List[str] = []


class VerificationResponse(BaseModel):
    """Response from evidence verification."""
    evidence_id: str
    tenant_id: str
    verified_at: datetime
    overall_valid: bool
    checks: Dict[str, Any]


class ChainOfCustodyResponse(BaseModel):
    """Chain of custody entry response."""
    timestamp: datetime
    action: str
    actor: str
    actor_type: str
    details: str
    signature: Optional[str] = None


# Dependency to get current tenant (mock implementation)
async def get_current_tenant() -> str:
    """Get current tenant ID from authentication context."""
    # In production, this would extract from JWT token
    return "t-enterprise"  # Mock tenant for demo


@router.post("/create", response_model=EvidenceResponse)
async def create_evidence(
    request: CreateEvidenceRequest,
    content_file: UploadFile = File(..., description="Evidence content file"),
    background_tasks: BackgroundTasks,
    tenant_id: str = Depends(get_current_tenant),
    evidence_service: ProvableEvidenceService = Depends(get_provable_evidence_service)
):
    """Create new cryptographically signed provable evidence."""

    try:
        # Read uploaded content
        content = await content_file.read()

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Evidence content cannot be empty")

        if len(content) > 100 * 1024 * 1024:  # 100MB limit
            raise HTTPException(status_code=400, detail="Evidence content too large (max 100MB)")

        # Validate evidence type and format
        try:
            evidence_type = EvidenceType(request.evidence_type)
        except ValueError:
            valid_types = [t.value for t in EvidenceType]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid evidence_type. Must be one of: {valid_types}"
            )

        try:
            format = EvidenceFormat(request.format)
        except ValueError:
            valid_formats = [f.value for f in EvidenceFormat]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format. Must be one of: {valid_formats}"
            )

        # Create provable evidence
        evidence = await evidence_service.create_evidence(
            tenant_id=tenant_id,
            evidence_type=evidence_type,
            format=format,
            content=content,
            title=request.title,
            description=request.description,
            source_system=request.source_system,
            source_user=request.source_user,
            tags=request.tags or []
        )

        # Schedule background verification
        background_tasks.add_task(
            _background_verification,
            evidence_service,
            evidence.metadata.evidence_id,
            tenant_id
        )

        # Return response
        return EvidenceResponse(
            evidence_id=evidence.metadata.evidence_id,
            tenant_id=evidence.metadata.tenant_id,
            evidence_type=evidence.metadata.evidence_type.value,
            format=evidence.metadata.format.value,
            title=evidence.metadata.title,
            description=evidence.metadata.description,
            source_system=evidence.metadata.source_system,
            source_user=evidence.metadata.source_user,
            created_at=evidence.metadata.created_at,
            size_bytes=evidence.metadata.size_bytes,
            content_hash=evidence.metadata.content_hash,
            storage_references=evidence.storage_references,
            tags=evidence.metadata.tags
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create evidence: {str(e)}")


@router.get("/{evidence_id}", response_model=EvidenceResponse)
async def get_evidence(
    evidence_id: str,
    tenant_id: str = Depends(get_current_tenant),
    evidence_service: ProvableEvidenceService = Depends(get_provable_evidence_service)
):
    """Get evidence metadata and verification status."""

    evidence = await evidence_service.get_evidence(tenant_id, evidence_id)
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")

    # Verify evidence integrity
    verification = await evidence_service.verify_evidence(evidence)

    return EvidenceResponse(
        evidence_id=evidence.metadata.evidence_id,
        tenant_id=evidence.metadata.tenant_id,
        evidence_type=evidence.metadata.evidence_type.value,
        format=evidence.metadata.format.value,
        title=evidence.metadata.title,
        description=evidence.metadata.description,
        source_system=evidence.metadata.source_system,
        source_user=evidence.metadata.source_user,
        created_at=evidence.metadata.created_at,
        size_bytes=evidence.metadata.size_bytes,
        content_hash=evidence.metadata.content_hash,
        signature_valid=verification["checks"].get("signature", {}).get("valid"),
        timestamp_valid=verification["checks"].get("trusted_timestamp", {}).get("valid"),
        storage_references=evidence.storage_references,
        tags=evidence.metadata.tags
    )


@router.get("/{evidence_id}/content")
async def download_evidence_content(
    evidence_id: str,
    tenant_id: str = Depends(get_current_tenant),
    evidence_service: ProvableEvidenceService = Depends(get_provable_evidence_service)
):
    """Download the actual evidence content."""

    evidence = await evidence_service.get_evidence(tenant_id, evidence_id)
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")

    # Create file-like object for streaming
    content_stream = io.BytesIO(evidence.content)

    # Determine media type based on format
    media_type_map = {
        EvidenceFormat.JSON: "application/json",
        EvidenceFormat.XML: "application/xml",
        EvidenceFormat.PCAP: "application/vnd.tcpdump.pcap",
        EvidenceFormat.TAR_GZ: "application/gzip",
        EvidenceFormat.PDF: "application/pdf",
        EvidenceFormat.BINARY: "application/octet-stream"
    }

    media_type = media_type_map.get(evidence.metadata.format, "application/octet-stream")
    filename = f"{evidence_id}.{evidence.metadata.format.value}"

    return StreamingResponse(
        io.BytesIO(evidence.content),
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Length": str(len(evidence.content)),
            "X-Evidence-Hash": evidence.metadata.content_hash,
            "X-Evidence-Signature": evidence.signature.signature.hex()[:32]  # First 32 chars
        }
    )


@router.post("/{evidence_id}/verify", response_model=VerificationResponse)
async def verify_evidence(
    evidence_id: str,
    tenant_id: str = Depends(get_current_tenant),
    evidence_service: ProvableEvidenceService = Depends(get_provable_evidence_service)
):
    """Verify cryptographic integrity of evidence."""

    evidence = await evidence_service.get_evidence(tenant_id, evidence_id)
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")

    verification = await evidence_service.verify_evidence(evidence)

    return VerificationResponse(
        evidence_id=verification["evidence_id"],
        tenant_id=verification["tenant_id"],
        verified_at=datetime.fromisoformat(verification["verified_at"]),
        overall_valid=verification["overall_valid"],
        checks=verification["checks"]
    )


@router.get("/{evidence_id}/chain-of-custody", response_model=List[ChainOfCustodyResponse])
async def get_chain_of_custody(
    evidence_id: str,
    tenant_id: str = Depends(get_current_tenant),
    evidence_service: ProvableEvidenceService = Depends(get_provable_evidence_service)
):
    """Get complete chain of custody for evidence."""

    evidence = await evidence_service.get_evidence(tenant_id, evidence_id)
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")

    return [
        ChainOfCustodyResponse(
            timestamp=entry.timestamp,
            action=entry.action,
            actor=entry.actor,
            actor_type=entry.actor_type,
            details=entry.details,
            signature=entry.signature
        )
        for entry in evidence.chain_of_custody
    ]


@router.get("/{evidence_id}/export-package")
async def export_evidence_package(
    evidence_id: str,
    include_content: bool = True,
    tenant_id: str = Depends(get_current_tenant),
    evidence_service: ProvableEvidenceService = Depends(get_provable_evidence_service)
):
    """Export complete evidence package for legal proceedings."""

    evidence = await evidence_service.get_evidence(tenant_id, evidence_id)
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")

    # Create exportable package
    package = evidence.to_dict()

    if include_content:
        # Include base64-encoded content for complete package
        import base64
        package["content_base64"] = base64.b64encode(evidence.content).decode('utf-8')

    # Add verification at export time
    verification = await evidence_service.verify_evidence(evidence)
    package["export_verification"] = verification
    package["exported_at"] = datetime.utcnow().isoformat()
    package["exported_by"] = f"tenant_{tenant_id}"

    # Convert to JSON
    package_json = json.dumps(package, indent=2, default=str)

    return StreamingResponse(
        io.BytesIO(package_json.encode('utf-8')),
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename={evidence_id}_package.json",
            "Content-Length": str(len(package_json))
        }
    )


@router.get("/tenant/{tenant_id}/list")
async def list_tenant_evidence(
    tenant_id: str,
    skip: int = 0,
    limit: int = 50,
    evidence_type: Optional[str] = None,
    current_tenant: str = Depends(get_current_tenant),
    evidence_service: ProvableEvidenceService = Depends(get_provable_evidence_service)
):
    """List all evidence for a tenant."""

    # Authorization check - can only list own evidence
    if tenant_id != current_tenant:
        raise HTTPException(status_code=403, detail="Cannot access other tenant's evidence")

    # In a real implementation, this would query a database
    # For now, scan filesystem (not efficient for production)
    import os
    from pathlib import Path

    evidence_dir = evidence_service.storage_path / tenant_id
    if not evidence_dir.exists():
        return {"evidence": [], "total": 0}

    evidence_files = list(evidence_dir.glob("*_evidence.json"))

    # Apply filtering and pagination (basic implementation)
    evidence_list = []
    for evidence_file in evidence_files[skip:skip+limit]:
        try:
            with open(evidence_file, 'r') as f:
                evidence_data = json.load(f)

            metadata = evidence_data.get("metadata", {})

            # Filter by evidence type if specified
            if evidence_type and metadata.get("evidence_type") != evidence_type:
                continue

            evidence_list.append({
                "evidence_id": metadata.get("evidence_id"),
                "evidence_type": metadata.get("evidence_type"),
                "title": metadata.get("title"),
                "created_at": metadata.get("created_at"),
                "size_bytes": metadata.get("size_bytes"),
                "source_system": metadata.get("source_system")
            })

        except Exception:
            continue

    return {
        "evidence": evidence_list,
        "total": len(evidence_files),
        "skip": skip,
        "limit": limit
    }


# Background task functions
async def _background_verification(
    evidence_service: ProvableEvidenceService,
    evidence_id: str,
    tenant_id: str
):
    """Background task to verify evidence after creation."""
    try:
        evidence = await evidence_service.get_evidence(tenant_id, evidence_id)
        if evidence:
            verification = await evidence_service.verify_evidence(evidence)
            print(f"✅ Background verification for {evidence_id}: {verification['overall_valid']}")
    except Exception as e:
        print(f"❌ Background verification failed for {evidence_id}: {e}")


# Health check endpoint
@router.get("/health")
async def provable_evidence_health():
    """Health check for provable evidence service."""
    try:
        service = get_provable_evidence_service()

        # Basic service checks
        checks = {
            "service_initialized": service is not None,
            "key_manager_available": service.key_manager is not None,
            "timestamp_service_available": service.timestamp_service is not None,
            "storage_path_exists": service.storage_path.exists(),
            "ipfs_available": service.ipfs_client is not None
        }

        all_healthy = all(checks.values())

        return {
            "status": "healthy" if all_healthy else "degraded",
            "service": "G7 Provable Evidence",
            "version": "g7.1.0",
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
