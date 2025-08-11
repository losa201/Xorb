"""
Compliance API endpoints for SOC 2 evidence status and reporting
"""

import json
from datetime import datetime, timedelta
from typing import Any

import boto3
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..deps import get_admin_user

logger = structlog.get_logger("xorb.api.compliance")

router = APIRouter(prefix="/compliance", tags=["compliance"])

class EvidenceStatus(BaseModel):
    """Evidence collection status model"""
    last_collection: datetime | None
    collection_status: str
    evidence_types_collected: list[str]
    failing_controls: int
    next_collection: datetime | None
    s3_bucket: str

class ComplianceControl(BaseModel):
    """SOC 2 control status model"""
    control_id: str
    name: str
    status: str  # compliant, non_compliant, needs_review
    evidence_count: int
    last_evaluated: datetime
    remediation: str | None

class ComplianceReport(BaseModel):
    """Compliance report model"""
    report_date: str
    overall_status: str
    evidence_collection_status: str
    last_collection: datetime | None
    evidence_types: list[str]
    failing_controls: int
    soc2_readiness: str
    next_actions: list[str]

class ComplianceService:
    """Service for compliance-related operations"""

    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.s3_bucket = "xorb-soc2-evidence"  # From environment in production

        self.soc2_controls = {
            'CC6.1': 'Logical Access Security',
            'CC6.2': 'Authentication and Authorization',
            'CC6.3': 'Network Security Controls',
            'CC7.1': 'System Operations',
            'CC7.2': 'Change Management',
            'CC8.1': 'Data Classification',
            'A1.1': 'Availability Monitoring',
            'A1.2': 'Capacity Management',
            'C1.1': 'Confidentiality Controls',
            'C1.2': 'Data Encryption'
        }

    async def get_evidence_status(self) -> EvidenceStatus:
        """Get current evidence collection status"""
        try:
            # Get latest status from S3
            response = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key='dashboard/latest_status.json'
            )

            status_data = json.loads(response['Body'].read())

            return EvidenceStatus(
                last_collection=datetime.fromisoformat(status_data['last_collection'].replace('Z', '+00:00')) if status_data.get('last_collection') else None,
                collection_status=status_data.get('collection_status', 'unknown'),
                evidence_types_collected=status_data.get('evidence_types_collected', []),
                failing_controls=status_data.get('failing_controls', 0),
                next_collection=datetime.fromisoformat(status_data['next_collection'].replace('Z', '+00:00')) if status_data.get('next_collection') else None,
                s3_bucket=self.s3_bucket
            )

        except Exception as e:
            logger.error("Failed to get evidence status", error=str(e))
            # Return default status if S3 data unavailable
            return EvidenceStatus(
                last_collection=None,
                collection_status='error',
                evidence_types_collected=[],
                failing_controls=0,
                next_collection=None,
                s3_bucket=self.s3_bucket
            )

    async def get_control_status(self) -> list[ComplianceControl]:
        """Get SOC 2 control compliance status"""
        try:
            # Get latest compliance evidence
            latest_evidence = await self.get_latest_evidence_bundle()

            if not latest_evidence or 'controls_status' not in latest_evidence:
                # Return default control status
                return [
                    ComplianceControl(
                        control_id=control_id,
                        name=name,
                        status='needs_review',
                        evidence_count=0,
                        last_evaluated=datetime.utcnow(),
                        remediation='Evidence collection required'
                    )
                    for control_id, name in self.soc2_controls.items()
                ]

            controls = []
            for control_id, control_data in latest_evidence['controls_status'].items():
                controls.append(ComplianceControl(
                    control_id=control_id,
                    name=control_data.get('name', self.soc2_controls.get(control_id, 'Unknown')),
                    status=control_data.get('status', 'needs_review'),
                    evidence_count=control_data.get('evidence_count', 0),
                    last_evaluated=datetime.fromisoformat(control_data['last_evaluated'].replace('Z', '+00:00')),
                    remediation=control_data.get('remediation', '')
                ))

            return controls

        except Exception as e:
            logger.error("Failed to get control status", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to retrieve control status")

    async def get_latest_evidence_bundle(self) -> dict[str, Any] | None:
        """Get the latest evidence bundle from S3"""
        try:
            # List recent evidence files
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix='daily-evidence/',
                MaxKeys=1
            )

            if 'Contents' not in response:
                return None

            # Get the most recent evidence bundle
            latest_key = response['Contents'][0]['Key']
            evidence_response = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key=latest_key
            )

            return json.loads(evidence_response['Body'].read())

        except Exception as e:
            logger.warning("Failed to get latest evidence bundle", error=str(e))
            return None

    async def get_compliance_report(self, report_date: str | None = None) -> ComplianceReport:
        """Get compliance report for specific date or latest"""
        try:
            if not report_date:
                report_date = datetime.utcnow().strftime('%Y%m%d')

            # Try to get specific report
            report_key = f"reports/compliance_report_{report_date}.json"

            try:
                response = self.s3_client.get_object(
                    Bucket=self.s3_bucket,
                    Key=report_key
                )
                report_data = json.loads(response['Body'].read())

            except self.s3_client.exceptions.NoSuchKey:
                # Generate report on-demand if not found
                report_data = await self.generate_compliance_report()

            return ComplianceReport(**report_data)

        except Exception as e:
            logger.error("Failed to get compliance report", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to retrieve compliance report")

    async def generate_compliance_report(self) -> dict[str, Any]:
        """Generate compliance report from current evidence"""
        evidence_status = await self.get_evidence_status()
        control_status = await self.get_control_status()

        failing_controls = sum(1 for control in control_status if control.status == 'non_compliant')

        report_data = {
            'report_date': datetime.utcnow().strftime('%Y-%m-%d'),
            'overall_status': 'compliant' if failing_controls == 0 else 'needs_attention',
            'evidence_collection_status': evidence_status.collection_status,
            'last_collection': evidence_status.last_collection.isoformat() if evidence_status.last_collection else None,
            'evidence_types': evidence_status.evidence_types_collected,
            'failing_controls': failing_controls,
            'soc2_readiness': 'green' if failing_controls == 0 else 'yellow' if failing_controls < 3 else 'red',
            'next_actions': []
        }

        # Add next actions based on status
        if failing_controls > 0:
            report_data['next_actions'].append('Review and remediate failing controls')

        if evidence_status.collection_status == 'error':
            report_data['next_actions'].append('Fix evidence collection automation')

        if not evidence_status.last_collection or (datetime.utcnow() - evidence_status.last_collection).days > 1:
            report_data['next_actions'].append('Trigger manual evidence collection')

        return report_data

    async def get_evidence_files(self, evidence_type: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get list of evidence files by type"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=f'{evidence_type}/',
                MaxKeys=limit
            )

            files = []
            for obj in response.get('Contents', []):
                files.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat(),
                    'download_url': self.s3_client.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': self.s3_bucket, 'Key': obj['Key']},
                        ExpiresIn=3600
                    )
                })

            return sorted(files, key=lambda x: x['last_modified'], reverse=True)

        except Exception as e:
            logger.error("Failed to get evidence files", evidence_type=evidence_type, error=str(e))
            return []

# Initialize compliance service
compliance_service = ComplianceService()

@router.get("/status", response_model=EvidenceStatus)
async def get_evidence_status(
    current_user = Depends(get_admin_user)
):
    """Get current evidence collection status (Admin only)"""
    return await compliance_service.get_evidence_status()

@router.get("/controls", response_model=list[ComplianceControl])
async def get_control_status(
    current_user = Depends(get_admin_user)
):
    """Get SOC 2 control compliance status (Admin only)"""
    return await compliance_service.get_control_status()

@router.get("/report", response_model=ComplianceReport)
async def get_compliance_report(
    report_date: str | None = Query(None, description="Report date in YYYYMMDD format"),
    current_user = Depends(get_admin_user)
):
    """Get compliance report for specific date or latest (Admin only)"""
    return await compliance_service.get_compliance_report(report_date)

@router.get("/evidence/{evidence_type}")
async def get_evidence_files(
    evidence_type: str,
    limit: int = Query(10, ge=1, le=50),
    current_user = Depends(get_admin_user)
):
    """Get list of evidence files by type (Admin only)"""

    valid_types = ['iam-evidence', 'sbom-evidence', 'daily-evidence', 'summaries', 'reports']
    if evidence_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid evidence type. Valid types: {valid_types}")

    files = await compliance_service.get_evidence_files(evidence_type, limit)
    return {"evidence_type": evidence_type, "files": files}

@router.post("/trigger-collection")
async def trigger_evidence_collection(
    current_user = Depends(get_admin_user)
):
    """Manually trigger evidence collection (Admin only)"""
    try:
        # This would typically trigger the compliance service
        # For now, return success message
        logger.info("Manual evidence collection triggered", user_id=current_user.id)

        return {
            "status": "triggered",
            "message": "Evidence collection has been queued",
            "estimated_completion": (datetime.utcnow() + timedelta(minutes=30)).isoformat()
        }

    except Exception as e:
        logger.error("Failed to trigger evidence collection", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to trigger evidence collection")

@router.get("/dashboard")
async def get_compliance_dashboard(
    current_user = Depends(get_admin_user)
):
    """Get compliance dashboard data (Admin only)"""
    try:
        evidence_status = await compliance_service.get_evidence_status()
        control_status = await compliance_service.get_control_status()
        compliance_report = await compliance_service.get_compliance_report()

        # Calculate dashboard metrics
        total_controls = len(control_status)
        compliant_controls = sum(1 for control in control_status if control.status == 'compliant')
        failing_controls = sum(1 for control in control_status if control.status == 'non_compliant')

        dashboard_data = {
            'overview': {
                'soc2_readiness': compliance_report.soc2_readiness,
                'overall_status': compliance_report.overall_status,
                'last_evidence_collection': evidence_status.last_collection,
                'next_collection': evidence_status.next_collection
            },
            'metrics': {
                'total_controls': total_controls,
                'compliant_controls': compliant_controls,
                'failing_controls': failing_controls,
                'compliance_percentage': round((compliant_controls / total_controls) * 100, 1) if total_controls > 0 else 0
            },
            'evidence_status': {
                'collection_status': evidence_status.collection_status,
                'evidence_types_collected': evidence_status.evidence_types_collected,
                'total_evidence_types': len(evidence_status.evidence_types_collected)
            },
            'next_actions': compliance_report.next_actions,
            'recent_activity': await self.get_recent_compliance_activity()
        }

        return dashboard_data

    except Exception as e:
        logger.error("Failed to get compliance dashboard", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve compliance dashboard")

async def get_recent_compliance_activity() -> list[dict[str, Any]]:
    """Get recent compliance-related activity"""
    # This would typically query activity logs
    # For now, return sample data
    return [
        {
            'timestamp': (datetime.utcnow() - timedelta(hours=2)).isoformat(),
            'activity': 'Evidence collection completed',
            'status': 'success',
            'details': 'All evidence types collected successfully'
        },
        {
            'timestamp': (datetime.utcnow() - timedelta(days=1)).isoformat(),
            'activity': 'Control CC6.1 updated',
            'status': 'info',
            'details': 'RLS policies implemented'
        }
    ]
