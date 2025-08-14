#!/usr/bin/env python3
"""
Anomaly Detection API Router
ML-powered anomaly detection endpoints for XORB Platform
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..services.ml_anomaly_detection_service import (
    get_anomaly_detection_service,
    MLAnomalyDetectionService,
    AnomalyType,
    AnomalySeverity,
    AnomalyEvent,
    AnomalyDetectionConfig
)
from ..core.auth import get_current_user
from ..core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/anomaly-detection", tags=["Anomaly Detection"])


# Pydantic models for API
class AnomalyDetectionRequest(BaseModel):
    """Request model for anomaly detection"""
    anomaly_type: AnomalyType
    data: List[Dict[str, Any]]
    real_time: bool = False


class TrainingRequest(BaseModel):
    """Request model for model training"""
    training_data: Dict[AnomalyType, List[Dict[str, Any]]]
    retrain_existing: bool = False


class AnomalyEventResponse(BaseModel):
    """Response model for anomaly events"""
    event_id: str
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    confidence: float
    risk_score: float
    description: str
    features: Dict[str, Any]
    model_version: str
    mitigation_suggestions: List[str]


class AnomalyDetectionResponse(BaseModel):
    """Response model for anomaly detection results"""
    total_events: int
    anomalies_detected: int
    anomaly_rate: float
    anomalies: List[AnomalyEventResponse]
    processing_time_ms: int


class ModelStatusResponse(BaseModel):
    """Response model for model status"""
    service_initialized: bool
    redis_connected: bool
    models: Dict[str, Dict[str, Any]]


class AnomalyStatisticsResponse(BaseModel):
    """Response model for anomaly statistics"""
    time_window_hours: int
    anomaly_counts: Dict[str, int]
    total_anomalies: int
    anomaly_trends: Optional[Dict[str, List[int]]] = None


@router.post("/detect", response_model=AnomalyDetectionResponse)
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    background_tasks: BackgroundTasks,
    service: MLAnomalyDetectionService = Depends(get_anomaly_detection_service),
    current_user = Depends(get_current_user)
):
    """
    Detect anomalies in provided data using ML models
    
    **Features:**
    - Real-time and batch anomaly detection
    - Multiple ML algorithms (Isolation Forest, DBSCAN, Random Forest, LSTM)
    - Feature engineering for different data types
    - Risk scoring and severity classification
    - Mitigation suggestions
    
    **Supported Anomaly Types:**
    - user_behavior: User activity patterns
    - network_traffic: Network communication patterns
    - api_usage: API endpoint usage patterns
    - authentication: Login/logout patterns
    - system_performance: System metrics patterns
    """
    start_time = datetime.utcnow()
    
    try:
        logger.info(
            f"Anomaly detection request received",
            user_id=getattr(current_user, 'id', 'unknown'),
            anomaly_type=request.anomaly_type.value,
            data_size=len(request.data),
            real_time=request.real_time
        )
        
        if not request.data:
            raise HTTPException(status_code=400, detail="No data provided for anomaly detection")
        
        if request.real_time and len(request.data) != 1:
            raise HTTPException(status_code=400, detail="Real-time detection requires exactly one data point")
        
        # Perform anomaly detection
        if request.real_time:
            # Real-time detection
            anomaly = await service.detect_real_time(request.anomaly_type, request.data[0])
            anomalies = [anomaly] if anomaly else []
        else:
            # Batch detection
            detection_data = {request.anomaly_type: request.data}
            results = await service.detect_anomalies_batch(detection_data)
            anomalies = results.get(request.anomaly_type, [])
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Convert anomaly events to response format
        anomaly_responses = []
        for anomaly in anomalies:
            anomaly_responses.append(AnomalyEventResponse(
                event_id=anomaly.event_id,
                timestamp=anomaly.timestamp,
                anomaly_type=anomaly.anomaly_type,
                severity=anomaly.severity,
                confidence=anomaly.confidence,
                risk_score=anomaly.risk_score,
                description=anomaly.description,
                features=anomaly.features,
                model_version=anomaly.model_version,
                mitigation_suggestions=anomaly.mitigation_suggestions
            ))
        
        # Log high-severity anomalies
        for anomaly in anomalies:
            if anomaly.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]:
                logger.warning(
                    f"High-severity anomaly detected",
                    event_id=anomaly.event_id,
                    severity=anomaly.severity.value,
                    confidence=anomaly.confidence,
                    risk_score=anomaly.risk_score
                )
        
        # Background task for additional processing
        background_tasks.add_task(
            _process_anomaly_results,
            anomalies,
            request.anomaly_type,
            getattr(current_user, 'id', 'unknown')
        )
        
        response = AnomalyDetectionResponse(
            total_events=len(request.data),
            anomalies_detected=len(anomalies),
            anomaly_rate=len(anomalies) / len(request.data) if request.data else 0.0,
            anomalies=anomaly_responses,
            processing_time_ms=int(processing_time)
        )
        
        logger.info(
            f"Anomaly detection completed",
            total_events=response.total_events,
            anomalies_detected=response.anomalies_detected,
            processing_time_ms=response.processing_time_ms
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Anomaly detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")


@router.post("/train")
async def train_models(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    service: MLAnomalyDetectionService = Depends(get_anomaly_detection_service),
    current_user = Depends(get_current_user)
):
    """
    Train anomaly detection models with provided data
    
    **Features:**
    - Multi-model training (supervised and unsupervised)
    - Feature engineering and scaling
    - Model validation and evaluation
    - Incremental and full retraining
    
    **Requirements:**
    - Admin or security analyst role
    - Minimum data samples per anomaly type
    - Proper data format and structure
    """
    try:
        # Check user permissions (admin or security analyst)
        if not hasattr(current_user, 'role') or current_user.role not in ['admin', 'security_analyst']:
            raise HTTPException(status_code=403, detail="Insufficient permissions for model training")
        
        logger.info(
            f"Model training request received",
            user_id=getattr(current_user, 'id', 'unknown'),
            anomaly_types=list(request.training_data.keys()),
            total_samples=sum(len(data) for data in request.training_data.values())
        )
        
        if not request.training_data:
            raise HTTPException(status_code=400, detail="No training data provided")
        
        # Validate training data
        for anomaly_type, data in request.training_data.items():
            if len(data) < 50:  # Minimum samples required
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient training data for {anomaly_type.value}: {len(data)} < 50 samples"
                )
        
        # Start training in background
        background_tasks.add_task(
            _train_models_background,
            service,
            request.training_data,
            getattr(current_user, 'id', 'unknown')
        )
        
        return JSONResponse(
            status_code=202,
            content={
                "message": "Model training started",
                "anomaly_types": [at.value for at in request.training_data.keys()],
                "total_samples": sum(len(data) for data in request.training_data.values()),
                "status": "training_in_progress"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model training request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training request failed: {str(e)}")


@router.get("/status", response_model=ModelStatusResponse)
async def get_model_status(
    service: MLAnomalyDetectionService = Depends(get_anomaly_detection_service),
    current_user = Depends(get_current_user)
):
    """
    Get status of anomaly detection models
    
    **Returns:**
    - Service initialization status
    - Model training status for each anomaly type
    - Last training timestamps
    - Model versions
    - Redis connection status
    """
    try:
        status = await service.get_model_status()
        return ModelStatusResponse(**status)
        
    except Exception as e:
        logger.error(f"Failed to get model status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")


@router.get("/statistics", response_model=AnomalyStatisticsResponse)
async def get_anomaly_statistics(
    hours: int = Query(24, ge=1, le=168, description="Time window in hours (1-168)"),
    include_trends: bool = Query(False, description="Include hourly trends"),
    service: MLAnomalyDetectionService = Depends(get_anomaly_detection_service),
    current_user = Depends(get_current_user)
):
    """
    Get anomaly detection statistics
    
    **Features:**
    - Anomaly counts by type and time window
    - Trends and patterns analysis
    - Performance metrics
    - Historical data comparison
    """
    try:
        stats = await service.get_anomaly_statistics(hours)
        
        # Add trends if requested
        trends = None
        if include_trends:
            # This would fetch hourly trends from the database/cache
            trends = await _get_anomaly_trends(hours)
        
        return AnomalyStatisticsResponse(
            time_window_hours=stats.get("time_window_hours", hours),
            anomaly_counts=stats.get("anomaly_counts", {}),
            total_anomalies=stats.get("total_anomalies", 0),
            anomaly_trends=trends
        )
        
    except Exception as e:
        logger.error(f"Failed to get anomaly statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.post("/config")
async def update_detection_config(
    config: Dict[str, Any] = Body(...),
    service: MLAnomalyDetectionService = Depends(get_anomaly_detection_service),
    current_user = Depends(get_current_user)
):
    """
    Update anomaly detection configuration
    
    **Parameters:**
    - contamination_rate: Expected percentage of anomalies (0.01-0.5)
    - sensitivity: Detection sensitivity (0.1-1.0)
    - anomaly_threshold: Threshold for anomaly classification
    - critical_threshold: Threshold for critical anomalies
    
    **Requires:** Admin role
    """
    try:
        # Check admin permissions
        if not hasattr(current_user, 'role') or current_user.role != 'admin':
            raise HTTPException(status_code=403, detail="Admin role required")
        
        # Validate configuration parameters
        valid_params = {
            'contamination_rate', 'sensitivity', 'anomaly_threshold', 
            'critical_threshold', 'min_samples', 'batch_size'
        }
        
        invalid_params = set(config.keys()) - valid_params
        if invalid_params:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid configuration parameters: {invalid_params}"
            )
        
        # Apply configuration updates
        for param, value in config.items():
            if hasattr(service.config, param):
                setattr(service.config, param, value)
        
        logger.info(
            f"Anomaly detection configuration updated",
            user_id=getattr(current_user, 'id', 'unknown'),
            updates=config
        )
        
        return JSONResponse(
            content={
                "message": "Configuration updated successfully",
                "updated_parameters": list(config.keys()),
                "current_config": {
                    "contamination_rate": service.config.contamination_rate,
                    "sensitivity": service.config.sensitivity,
                    "anomaly_threshold": service.config.anomaly_threshold,
                    "critical_threshold": service.config.critical_threshold
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Configuration update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")


@router.get("/types")
async def get_anomaly_types():
    """
    Get available anomaly detection types and their descriptions
    """
    types_info = {
        AnomalyType.USER_BEHAVIOR.value: {
            "description": "Detects unusual user activity patterns",
            "features": ["login patterns", "access times", "endpoint usage", "geographic patterns"]
        },
        AnomalyType.NETWORK_TRAFFIC.value: {
            "description": "Identifies abnormal network traffic patterns",
            "features": ["traffic volume", "connection patterns", "protocol usage", "destination analysis"]
        },
        AnomalyType.API_USAGE.value: {
            "description": "Finds irregular API usage patterns",
            "features": ["request rates", "endpoint diversity", "HTTP methods", "response patterns"]
        },
        AnomalyType.AUTHENTICATION.value: {
            "description": "Detects suspicious authentication patterns",
            "features": ["login success/failure", "timing patterns", "geographic locations", "device patterns"]
        },
        AnomalyType.SYSTEM_PERFORMANCE.value: {
            "description": "Identifies system performance anomalies",
            "features": ["resource usage", "response times", "error rates", "throughput patterns"]
        }
    }
    
    return JSONResponse(content={
        "anomaly_types": types_info,
        "total_types": len(types_info)
    })


# Background task functions
async def _process_anomaly_results(
    anomalies: List[AnomalyEvent],
    anomaly_type: AnomalyType,
    user_id: str
):
    """Process anomaly detection results in background"""
    try:
        # Store anomalies in database
        # Send notifications for high-severity anomalies
        # Update metrics and statistics
        # Trigger automated responses if configured
        
        high_severity_count = sum(
            1 for anomaly in anomalies 
            if anomaly.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]
        )
        
        if high_severity_count > 0:
            logger.warning(
                f"High-severity anomalies detected in background processing",
                anomaly_type=anomaly_type.value,
                count=high_severity_count,
                user_id=user_id
            )
            
            # Here you would integrate with alerting systems
            # await send_security_alert(anomalies, user_id)
            
    except Exception as e:
        logger.error(f"Background anomaly processing failed: {str(e)}")


async def _train_models_background(
    service: MLAnomalyDetectionService,
    training_data: Dict[AnomalyType, List[Dict[str, Any]]],
    user_id: str
):
    """Train models in background"""
    try:
        logger.info(f"Starting background model training for user {user_id}")
        
        await service.train_models(training_data)
        
        logger.info(f"Background model training completed for user {user_id}")
        
        # Send training completion notification
        # await send_training_notification(user_id, "success")
        
    except Exception as e:
        logger.error(f"Background model training failed: {str(e)}")
        # await send_training_notification(user_id, "failed", str(e))


async def _get_anomaly_trends(hours: int) -> Dict[str, List[int]]:
    """Get anomaly trends for visualization"""
    # This would fetch historical data from database
    # For now, return mock data
    trends = {}
    for anomaly_type in AnomalyType:
        # Mock hourly data
        trends[anomaly_type.value] = [0] * min(hours, 24)
    
    return trends