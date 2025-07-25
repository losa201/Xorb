"""
REST API Data Transfer Objects (DTOs)

Pydantic models for request/response serialization.
These handle conversion between JSON and internal domain objects.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, validator

__all__ = [
    "CreateCampaignRequest",
    "StartCampaignRequest",
    "CampaignResponse",
    "FindingResponse", 
    "TriageFindingRequest",
    "EmbeddingRequest",
    "KnowledgeAtomResponse",
    "ErrorResponse",
    "HealthResponse"
]


class CreateCampaignRequest(BaseModel):
    """Request to create a new campaign"""
    
    name: str = Field(..., min_length=1, max_length=255)
    target_id: UUID
    max_cost_usd: Decimal = Field(..., gt=0, le=10000)
    max_duration_hours: int = Field(..., gt=0, le=168)  # Max 1 week
    max_api_calls: int = Field(..., gt=0, le=100000)
    required_capabilities: List[str] = Field(..., min_items=1)
    max_agents: int = Field(default=5, gt=0, le=20)
    auto_start: bool = Field(default=False)
    
    @validator('required_capabilities')
    def validate_capabilities(cls, v):
        """Validate agent capabilities"""
        valid_capabilities = {
            "web_scanning", "subdomain_enumeration", "port_scanning",
            "vulnerability_assessment", "content_discovery", "ssl_analysis",
            "dns_enumeration", "network_mapping", "payload_generation"
        }
        
        for capability in v:
            if capability not in valid_capabilities:
                raise ValueError(f"Invalid capability: {capability}")
        
        return v

    class Config:
        schema_extra = {
            "example": {
                "name": "HackerOne Bug Bounty Campaign",
                "target_id": "123e4567-e89b-12d3-a456-426614174000",
                "max_cost_usd": "500.00",
                "max_duration_hours": 24,
                "max_api_calls": 10000,
                "required_capabilities": ["web_scanning", "subdomain_enumeration"],
                "max_agents": 5,
                "auto_start": False
            }
        }


class StartCampaignRequest(BaseModel):
    """Request to start an existing campaign"""
    
    campaign_id: UUID
    force_start: bool = Field(default=False)
    
    class Config:
        schema_extra = {
            "example": {
                "campaign_id": "123e4567-e89b-12d3-a456-426614174000",
                "force_start": False
            }
        }


class CampaignResponse(BaseModel):
    """Campaign information response"""
    
    id: UUID
    name: str
    target_name: str
    status: str
    budget_used_percentage: float = Field(..., ge=0, le=100)
    agents_completed: int
    agents_total: int
    findings_discovered: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    class Config:
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "HackerOne Bug Bounty Campaign",
                "target_name": "example.com",
                "status": "running",
                "budget_used_percentage": 45.2,
                "agents_completed": 3,
                "agents_total": 5,
                "findings_discovered": 12,
                "created_at": "2024-01-15T10:30:00Z",
                "started_at": "2024-01-15T10:35:00Z",
                "completed_at": None,
                "estimated_completion": "2024-01-16T10:35:00Z"
            }
        }


class FindingResponse(BaseModel):
    """Security finding response"""
    
    id: UUID
    campaign_id: UUID
    agent_id: UUID
    title: str
    description: str
    severity: str
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    evidence: Dict[str, Any] = Field(default_factory=dict)
    similar_findings_count: Optional[int] = None
    confidence_score: Optional[float] = Field(None, ge=0, le=1)
    
    class Config:
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "campaign_id": "456e7890-e89b-12d3-a456-426614174000",
                "agent_id": "789e0123-e89b-12d3-a456-426614174000",
                "title": "SQL Injection Vulnerability",
                "description": "SQL injection found in login form parameter 'username'",
                "severity": "high",
                "status": "new",
                "created_at": "2024-01-15T12:45:00Z",
                "updated_at": "2024-01-15T12:45:00Z",
                "evidence": {
                    "url": "https://example.com/login",
                    "parameter": "username",
                    "payload": "' OR 1=1--",
                    "response_time": 2.5
                },
                "similar_findings_count": 2,
                "confidence_score": 0.95
            }
        }


class TriageFindingRequest(BaseModel):
    """Request to triage a finding"""
    
    finding_id: UUID
    new_status: str
    reasoning: Optional[str] = Field(None, max_length=1000)
    
    @validator('new_status')
    def validate_status(cls, v):
        """Validate finding status"""
        valid_statuses = {"new", "confirmed", "false_positive", "duplicate", "resolved"}
        if v not in valid_statuses:
            raise ValueError(f"Invalid status: {v}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "finding_id": "123e4567-e89b-12d3-a456-426614174000",
                "new_status": "confirmed",
                "reasoning": "Verified SQL injection vulnerability through manual testing"
            }
        }


class EmbeddingRequest(BaseModel):
    """Request to generate embeddings"""
    
    text: str = Field(..., min_length=1, max_length=10000)
    model: str = Field(default="nvidia/embed-qa-4")
    input_type: str = Field(default="query")
    
    @validator('input_type')
    def validate_input_type(cls, v):
        """Validate input type"""
        valid_types = {"query", "passage", "classification"}
        if v not in valid_types:
            raise ValueError(f"Invalid input_type: {v}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "text": "SQL injection vulnerability in web application",
                "model": "nvidia/embed-qa-4",
                "input_type": "passage"
            }
        }


class KnowledgeAtomResponse(BaseModel):
    """Knowledge atom response"""
    
    id: UUID
    content: str
    atom_type: str
    confidence: float = Field(..., ge=0, le=1)
    tags: List[str]
    source: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    embedding_dimension: Optional[int] = None
    related_atoms_count: Optional[int] = None
    
    class Config:
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "content": "SQL injection vulnerability identified in authentication bypass",
                "atom_type": "vulnerability",
                "confidence": 0.92,
                "tags": ["sql-injection", "authentication", "high-severity"],
                "source": "finding:456e7890-e89b-12d3-a456-426614174000",
                "created_at": "2024-01-15T14:20:00Z",
                "updated_at": "2024-01-15T14:20:00Z",
                "embedding_dimension": 1024,
                "related_atoms_count": 5
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid campaign configuration",
                "details": {
                    "field": "max_cost_usd",
                    "reason": "Must be greater than 0"
                },
                "timestamp": "2024-01-15T15:30:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str
    version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())
    dependencies: Dict[str, str] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "2.0.0",
                "timestamp": "2024-01-15T16:00:00Z",
                "dependencies": {
                    "database": "healthy",
                    "redis": "healthy",
                    "nats": "healthy",
                    "nvidia_api": "healthy"
                }
            }
        }