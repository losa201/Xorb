"""
Compliance API Routes
Implements FastAPI endpoints for compliance validation system
"""
from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Optional
from uuid import UUID

from xorb.shared.fastapi_models import (
    ComplianceValidationRequest,
    ComplianceValidationResponse,
    ComplianceStatusResponse,
    ComplianceReportResponse
)
from xorb.shared.redis_client import get_redis_client
from xorb.core.compliance_orchestrator import ComplianceOrchestrator
from xorb.security.auth import verify_token

# Create router
router = APIRouter(
    prefix="/compliance",
    tags=["compliance"],
    dependencies=[Depends(verify_token)]
)

# Global orchestrator instance
orchestrator = ComplianceOrchestrator()

@router.post("/validate", status_code=status.HTTP_202_ACCEPTED)
async def start_compliance_validation(
    request: ComplianceValidationRequest
) -> ComplianceValidationResponse:
    """
    Start a new compliance validation process
    """
    try:
        # Run the async orchestration in a sync context
        result = await orchestrator.run_compliance_validation(
            standard=request.standard,
            targets=request.targets,
            scan_profile=request.scan_profile,
            policy=request.policy,
            metadata=request.metadata
        )

        return ComplianceValidationResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start compliance validation: {str(e)}"
        )

@router.get("/status/{validation_id}", response_model=ComplianceStatusResponse)
async def get_validation_status(validation_id: UUID):
    """
    Get the status of a compliance validation process
    """
    try:
        result = await orchestrator.get_validation_status(str(validation_id))

        if result.get("error"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Validation not found"
            )

        return ComplianceStatusResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get validation status: {str(e)}"
        )

@router.get("/report/{validation_id}", response_model=ComplianceReportResponse)
async def get_compliance_report(validation_id: UUID):
    """
    Get the compliance report for a completed validation
    """
    try:
        # Get validation status to check if completed
        status_result = await orchestrator.get_validation_status(str(validation_id))

        if status_result.get("error"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Validation not found"
            )

        if status_result.get("status") not in ["completed", "error"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Validation not completed yet"
            )

        # Get the compliance report from Redis
        redis_client = get_redis_client()
        report_data = await redis_client.hget(
            f"compliance:{validation_id}:reports",
            "compliance_report"
        )

        if not report_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Report not found"
            )

        return ComplianceReportResponse(**json.loads(report_data))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get compliance report: {str(e)}"
        )

@router.post("/cancel/{validation_id}", response_model=dict)
async def cancel_validation(validation_id: UUID):
    """
    Cancel a running compliance validation process
    """
    try:
        success = await orchestrator.cancel_validation(str(validation_id))

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to cancel validation"
            )

        return {"status": "cancelled", "validation_id": validation_id}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel validation: {str(e)}"
        )

# Additional endpoints could be added here for:
# - Listing recent validations
# - Exporting reports in different formats
# - Managing compliance templates
# - Configuring scan profiles
# - Webhook management for validation completion

__all__ = ["router"]

"""
This module implements FastAPI endpoints for the compliance validation system.

The API provides the following endpoints:

1. POST /compliance/validate
   - Start a new compliance validation process
   - Request body: ComplianceValidationRequest
   - Response: ComplianceValidationResponse
   - Status code: 202 Accepted

2. GET /compliance/status/{validation_id}
   - Get the status of a compliance validation process
   - Response: ComplianceStatusResponse

3. GET /compliance/report/{validation_id}
   - Get the compliance report for a completed validation
   - Response: ComplianceReportResponse

4. POST /compliance/cancel/{validation_id}
   - Cancel a running compliance validation process
   - Response: {"status": "cancelled", "validation_id": UUID}

All endpoints require authentication via the verify_token dependency.
"""
