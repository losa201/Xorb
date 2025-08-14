"""
Enhanced Analytics API Router
Provides business intelligence, advanced analytics, and visualization endpoints
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..services.enhanced_analytics_service import (
    EnhancedAnalyticsService, 
    MetricType, 
    TimeRange, 
    AnalyticsQuery,
    AnalyticsResult
)
from ..core.database import get_database_session
from ..core.auth import get_current_user
from ..core.logging import get_logger

logger = get_logger(__name__)

# Router configuration
router = APIRouter(prefix="/analytics", tags=["Enhanced Analytics"])

# Pydantic models
class AnalyticsQueryRequest(BaseModel):
    """Request model for analytics queries"""
    metric_type: MetricType
    time_range: TimeRange = TimeRange.DAY
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    aggregation: str = "avg"
    group_by: Optional[str] = None

class AnalyticsResponse(BaseModel):
    """Response model for analytics queries"""
    query_id: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    visualization: Optional[str] = None

class ExecutiveReportResponse(BaseModel):
    """Response model for executive reports"""
    report_id: str
    generated_at: datetime
    time_range: str
    key_metrics: Dict[str, Any]
    security_summary: Dict[str, Any]
    performance_summary: Dict[str, Any]
    business_summary: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    trend_analysis: Dict[str, Any]

class AnomalyDetectionResponse(BaseModel):
    """Response model for anomaly detection"""
    anomalies: List[Dict[str, Any]]
    time_range: str
    metric_type: str
    total_data_points: int
    anomaly_count: int

class ComplianceReportResponse(BaseModel):
    """Response model for compliance analytics"""
    query_id: str
    timestamp: datetime
    framework: str
    overall_score: float
    data: Dict[str, Any]
    visualization: Optional[str] = None


async def get_analytics_service(
    db: AsyncSession = Depends(get_database_session)
) -> EnhancedAnalyticsService:
    """Dependency for analytics service"""
    return EnhancedAnalyticsService(db)


@router.get("/security", response_model=AnalyticsResponse)
async def get_security_analytics(
    time_range: TimeRange = Query(TimeRange.DAY, description="Time range for analytics"),
    tenant_filter: Optional[str] = Query(None, description="Filter by tenant ID"),
    service_filter: Optional[str] = Query(None, description="Filter by service name"),
    analytics_service: EnhancedAnalyticsService = Depends(get_analytics_service),
    current_user: dict = Depends(get_current_user)
):
    """
    Get comprehensive security analytics and metrics
    
    Returns detailed security metrics including:
    - Threat detection statistics
    - Attack vector analysis
    - Security response times
    - Vulnerability assessments
    """
    
    try:
        logger.info(f"Generating security analytics for user {current_user.get('user_id')}")
        
        # Build filters
        filters = {}
        if tenant_filter:
            filters["tenant_id"] = tenant_filter
        if service_filter:
            filters["service_name"] = service_filter
        
        # Get analytics
        result = await analytics_service.get_security_analytics(
            time_range=time_range,
            filters=filters
        )
        
        return AnalyticsResponse(
            query_id=result.query_id,
            timestamp=result.timestamp,
            data=result.data,
            metadata=result.metadata,
            visualization=result.visualization
        )
        
    except Exception as e:
        logger.error(f"Error generating security analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate security analytics")


@router.get("/performance", response_model=AnalyticsResponse)
async def get_performance_analytics(
    time_range: TimeRange = Query(TimeRange.DAY, description="Time range for analytics"),
    service_filter: Optional[str] = Query(None, description="Filter by service name"),
    analytics_service: EnhancedAnalyticsService = Depends(get_analytics_service),
    current_user: dict = Depends(get_current_user)
):
    """
    Get comprehensive performance analytics and metrics
    
    Returns detailed performance metrics including:
    - Response times and throughput
    - System resource utilization
    - Error rates and uptime
    - Database performance
    """
    
    try:
        logger.info(f"Generating performance analytics for user {current_user.get('user_id')}")
        
        # Build filters
        filters = {}
        if service_filter:
            filters["service_name"] = service_filter
        
        # Get analytics
        result = await analytics_service.get_performance_analytics(
            time_range=time_range,
            filters=filters
        )
        
        return AnalyticsResponse(
            query_id=result.query_id,
            timestamp=result.timestamp,
            data=result.data,
            metadata=result.metadata,
            visualization=result.visualization
        )
        
    except Exception as e:
        logger.error(f"Error generating performance analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate performance analytics")


@router.get("/business", response_model=AnalyticsResponse)
async def get_business_intelligence(
    time_range: TimeRange = Query(TimeRange.MONTH, description="Time range for analytics"),
    analytics_service: EnhancedAnalyticsService = Depends(get_analytics_service),
    current_user: dict = Depends(get_current_user)
):
    """
    Get comprehensive business intelligence analytics
    
    Returns business metrics including:
    - Customer metrics and satisfaction
    - Revenue and growth analytics
    - Feature adoption rates
    - Support ticket analysis
    """
    
    try:
        logger.info(f"Generating business intelligence for user {current_user.get('user_id')}")
        
        # Get analytics
        result = await analytics_service.get_business_intelligence(
            time_range=time_range
        )
        
        return AnalyticsResponse(
            query_id=result.query_id,
            timestamp=result.timestamp,
            data=result.data,
            metadata=result.metadata,
            visualization=result.visualization
        )
        
    except Exception as e:
        logger.error(f"Error generating business intelligence: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate business intelligence")


@router.get("/compliance/{framework}", response_model=ComplianceReportResponse)
async def get_compliance_analytics(
    framework: str = "SOC2",
    time_range: TimeRange = Query(TimeRange.QUARTER, description="Time range for analytics"),
    analytics_service: EnhancedAnalyticsService = Depends(get_analytics_service),
    current_user: dict = Depends(get_current_user)
):
    """
    Get compliance analytics for specific frameworks
    
    Supported frameworks:
    - SOC2: SOC 2 Type II compliance
    - GDPR: General Data Protection Regulation
    - HIPAA: Health Insurance Portability and Accountability Act
    - PCI-DSS: Payment Card Industry Data Security Standard
    """
    
    try:
        logger.info(f"Generating compliance analytics for {framework}")
        
        # Validate framework
        supported_frameworks = ["SOC2", "GDPR", "HIPAA", "PCI-DSS", "ISO27001", "NIST"]
        if framework.upper() not in supported_frameworks:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported framework. Supported: {', '.join(supported_frameworks)}"
            )
        
        # Get analytics
        result = await analytics_service.get_compliance_analytics(
            framework=framework.upper(),
            time_range=time_range
        )
        
        return ComplianceReportResponse(
            query_id=result.query_id,
            timestamp=result.timestamp,
            framework=framework.upper(),
            overall_score=result.data.get("overall_score", 0.0),
            data=result.data,
            visualization=result.visualization
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating compliance analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate compliance analytics")


@router.post("/anomaly-detection", response_model=AnomalyDetectionResponse)
async def perform_anomaly_detection(
    request: AnalyticsQueryRequest,
    analytics_service: EnhancedAnalyticsService = Depends(get_analytics_service),
    current_user: dict = Depends(get_current_user)
):
    """
    Perform anomaly detection on metrics using machine learning
    
    Detects unusual patterns and outliers in:
    - Security metrics
    - Performance metrics
    - Business metrics
    - Operational metrics
    """
    
    try:
        logger.info(f"Performing anomaly detection on {request.metric_type.value}")
        
        # Perform anomaly detection
        result = await analytics_service.perform_anomaly_detection(
            metric_type=request.metric_type,
            time_range=request.time_range
        )
        
        return AnomalyDetectionResponse(**result)
        
    except Exception as e:
        logger.error(f"Error performing anomaly detection: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to perform anomaly detection")


@router.get("/executive-report", response_model=ExecutiveReportResponse)
async def generate_executive_report(
    time_range: TimeRange = Query(TimeRange.MONTH, description="Time range for report"),
    analytics_service: EnhancedAnalyticsService = Depends(get_analytics_service),
    current_user: dict = Depends(get_current_user)
):
    """
    Generate comprehensive executive summary report
    
    Provides high-level overview including:
    - Key performance indicators
    - Security posture summary
    - Business health metrics
    - Strategic recommendations
    - Trend analysis
    """
    
    try:
        logger.info(f"Generating executive report for user {current_user.get('user_id')}")
        
        # Generate report
        report = await analytics_service.generate_executive_report(time_range=time_range)
        
        return ExecutiveReportResponse(**report)
        
    except Exception as e:
        logger.error(f"Error generating executive report: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate executive report")


@router.get("/dashboards/security")
async def get_security_dashboard(
    time_range: TimeRange = Query(TimeRange.DAY, description="Time range for dashboard"),
    analytics_service: EnhancedAnalyticsService = Depends(get_analytics_service),
    current_user: dict = Depends(get_current_user)
):
    """
    Get security analytics dashboard data optimized for visualization
    """
    
    try:
        result = await analytics_service.get_security_analytics(time_range=time_range)
        
        # Return dashboard-optimized format
        return {
            "dashboard_type": "security",
            "time_range": time_range.value,
            "last_updated": datetime.now().isoformat(),
            "widgets": {
                "threats_overview": {
                    "total_detected": result.data.get("threats_detected", 0),
                    "total_blocked": result.data.get("threats_blocked", 0),
                    "block_rate": f"{(result.data.get('threats_blocked', 0) / max(result.data.get('threats_detected', 1), 1)) * 100:.1f}%"
                },
                "attack_vectors": result.data.get("attack_vectors", {}),
                "response_times": result.data.get("response_times", {}),
                "vulnerability_score": result.data.get("vulnerability_score", 0)
            },
            "charts": result.visualization
        }
        
    except Exception as e:
        logger.error(f"Error generating security dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate security dashboard")


@router.get("/dashboards/performance")
async def get_performance_dashboard(
    time_range: TimeRange = Query(TimeRange.DAY, description="Time range for dashboard"),
    analytics_service: EnhancedAnalyticsService = Depends(get_analytics_service),
    current_user: dict = Depends(get_current_user)
):
    """
    Get performance analytics dashboard data optimized for visualization
    """
    
    try:
        result = await analytics_service.get_performance_analytics(time_range=time_range)
        
        # Return dashboard-optimized format
        return {
            "dashboard_type": "performance",
            "time_range": time_range.value,
            "last_updated": datetime.now().isoformat(),
            "widgets": {
                "system_health": {
                    "uptime": f"{result.data.get('uptime_percentage', 0):.2f}%",
                    "response_time": f"{result.data.get('avg_response_time', 0):.1f}ms",
                    "error_rate": f"{result.data.get('error_rate', 0):.2f}%",
                    "throughput": result.data.get("throughput", 0)
                },
                "resource_utilization": {
                    "cpu": result.data.get("cpu_utilization", 0),
                    "memory": result.data.get("memory_utilization", 0),
                    "disk": result.data.get("disk_utilization", 0)
                },
                "database_performance": result.data.get("database_performance", {})
            },
            "charts": result.visualization
        }
        
    except Exception as e:
        logger.error(f"Error generating performance dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate performance dashboard")


@router.get("/dashboards/business")
async def get_business_dashboard(
    time_range: TimeRange = Query(TimeRange.MONTH, description="Time range for dashboard"),
    analytics_service: EnhancedAnalyticsService = Depends(get_analytics_service),
    current_user: dict = Depends(get_current_user)
):
    """
    Get business intelligence dashboard data optimized for visualization
    """
    
    try:
        result = await analytics_service.get_business_intelligence(time_range=time_range)
        
        # Return dashboard-optimized format
        return {
            "dashboard_type": "business",
            "time_range": time_range.value,
            "last_updated": datetime.now().isoformat(),
            "widgets": {
                "customer_metrics": {
                    "total_customers": result.data.get("total_customers", 0),
                    "active_users": result.data.get("active_users", 0),
                    "satisfaction": result.data.get("customer_satisfaction", 0),
                    "retention": result.data.get("customer_retention", 0)
                },
                "business_growth": {
                    "revenue_growth": result.data.get("revenue_growth", 0)
                },
                "support_metrics": result.data.get("support_tickets", {}),
                "feature_adoption": result.data.get("feature_adoption", {})
            },
            "charts": result.visualization
        }
        
    except Exception as e:
        logger.error(f"Error generating business dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate business dashboard")


@router.get("/metrics/summary")
async def get_metrics_summary(
    analytics_service: EnhancedAnalyticsService = Depends(get_analytics_service),
    current_user: dict = Depends(get_current_user)
):
    """
    Get high-level metrics summary for overview displays
    """
    
    try:
        # Get quick analytics for multiple areas
        security_result = await analytics_service.get_security_analytics(TimeRange.DAY)
        performance_result = await analytics_service.get_performance_analytics(TimeRange.DAY)
        business_result = await analytics_service.get_business_intelligence(TimeRange.MONTH)
        
        return {
            "summary": {
                "security_score": analytics_service._calculate_security_score(security_result.data),
                "performance_score": analytics_service._calculate_performance_score(performance_result.data),
                "business_health": analytics_service._calculate_business_health(business_result.data),
                "threats_blocked_24h": security_result.data.get("threats_blocked", 0),
                "avg_response_time": performance_result.data.get("avg_response_time", 0),
                "active_customers": business_result.data.get("active_users", 0),
                "uptime_percentage": performance_result.data.get("uptime_percentage", 0)
            },
            "status": "healthy" if all([
                analytics_service._calculate_security_score(security_result.data) > 80,
                analytics_service._calculate_performance_score(performance_result.data) > 80,
                analytics_service._calculate_business_health(business_result.data) > 70
            ]) else "attention_required",
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating metrics summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate metrics summary")


@router.get("/health")
async def analytics_health_check():
    """Health check endpoint for analytics service"""
    return {
        "status": "healthy",
        "service": "enhanced-analytics",
        "timestamp": datetime.now().isoformat(),
        "capabilities": {
            "security_analytics": True,
            "performance_analytics": True,
            "business_intelligence": True,
            "anomaly_detection": True,
            "visualization": True,
            "executive_reporting": True
        }
    }