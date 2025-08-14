from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4

from pydantic import BaseModel, Field
from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.mutable import MutableList, MutableDict
from sqlalchemy.dialects.postgresql import ARRAY

from xorb.database.database import Base
from .enums import CampaignStatus, AgentType, ThreatSeverity
from .execution_enums import ScanType, ExploitType, EvidenceType

class UnifiedUser(Base, BaseModel):
    __tablename__ = "users"

    id: str = Column(String, primary_key=True, default=lambda: str(uuid4()))
    username: str = Column(String, unique=True, index=True)
    email: str = Column(String, unique=True, index=True)
    roles: List[str] = Column(MutableList.as_mutable(ARRAY(String)), default=list)
    api_keys: List[str] = Column(MutableList.as_mutable(ARRAY(String)), default=list)
    permissions: Dict[str, bool] = Column(MutableDict.as_mutable(JSON), default=dict)
    created_at: datetime = Column(DateTime, default=datetime.utcnow)
    last_active: datetime = Column(DateTime, default=datetime.utcnow)

class UnifiedSession(Base, BaseModel):
    __tablename__ = "sessions"

    session_id: str = Column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id: str = Column(String, index=True)
    ip_address: str = Column(String)
    user_agent: str = Column(String)
    created_at: datetime = Column(DateTime, default=datetime.utcnow)
    expires_at: datetime = Column(DateTime)
    permissions: Dict[str, bool] = Column(MutableDict.as_mutable(JSON), default=dict)

class APIKeyModel(Base, BaseModel):
    __tablename__ = "api_keys"

    key_id: str = Column(String, primary_key=True, default=lambda: str(uuid4()))
    key_hash: str = Column(String, unique=True, index=True)
    user_id: str = Column(String, index=True)
    name: str = Column(String)
    scopes: List[str] = Column(MutableList.as_mutable(ARRAY(String)), default=list)
    rate_limit: int = Column(Integer, default=1000)
    created_at: datetime = Column(DateTime, default=datetime.utcnow)
    expires_at: Optional[datetime] = Column(DateTime, nullable=True)
    last_used: Optional[datetime] = Column(DateTime, nullable=True)

class UnifiedTarget(Base, BaseModel):
    __tablename__ = "targets"

    id: str = Column(String, primary_key=True, default=lambda: str(uuid4()))
    hostname: str = Column(String)
    ip_address: Optional[str] = Column(String, nullable=True)
    ports: List[int] = Column(MutableList.as_mutable(ARRAY(Integer)), default=list)
    services: List[str] = Column(MutableList.as_mutable(ARRAY(String)), default=list)
    vulnerabilities: List[Dict[str, Any]] = Column(MutableList.as_mutable(ARRAY(JSON)), default=list)
    scope: str = Column(String, default="in-scope")
    confidence: float = Column(Float, default=1.0)
    metadata: Dict[str, Any] = Column(MutableDict.as_mutable(JSON), default=dict)

class UnifiedAgent(Base, BaseModel):
    __tablename__ = "agents"

    id: str = Column(String, primary_key=True, default=lambda: str(uuid4()))
    name: str = Column(String)
    agent_type: AgentType = Column(String) # Stored as string
    capabilities: List[str] = Column(MutableList.as_mutable(ARRAY(String)), default=list)
    status: str = Column(String, default="idle")
    target_id: Optional[str] = Column(String, nullable=True)
    performance_metrics: Dict[str, float] = Column(MutableDict.as_mutable(JSON), default=dict)
    stealth_config: Dict[str, Any] = Column(MutableDict.as_mutable(JSON), default=dict)
    created_at: datetime = Column(DateTime, default=datetime.utcnow)
    last_active: datetime = Column(DateTime, default=datetime.utcnow)

class UnifiedCampaign(Base, BaseModel):
    __tablename__ = "campaigns"

    id: str = Column(String, primary_key=True, default=lambda: str(uuid4()))
    name: str = Column(String)
    description: str = Column(String, default="")
    status: CampaignStatus = Column(String) # Stored as string
    targets: List[UnifiedTarget] = Column(MutableList.as_mutable(ARRAY(JSON)), default=list)
    agents: List[str] = Column(MutableList.as_mutable(ARRAY(String)), default=list)
    priority: str = Column(String, default="medium")
    max_duration: int = Column(Integer, default=3600)
    rules_of_engagement: Dict[str, Any] = Column(MutableDict.as_mutable(JSON), default=dict)
    results: Dict[str, Any] = Column(MutableDict.as_mutable(JSON), default=dict)
    created_at: datetime = Column(DateTime, default=datetime.utcnow)
    started_at: Optional[datetime] = Column(DateTime, nullable=True)
    completed_at: Optional[datetime] = Column(DateTime, nullable=True)

class ThreatIntelligence(Base, BaseModel):
    __tablename__ = "threat_intelligence"

    id: str = Column(String, primary_key=True, default=lambda: str(uuid4()))
    threat_type: str = Column(String)
    severity: ThreatSeverity = Column(String) # Stored as string
    indicators: Dict[str, List[str]] = Column(MutableDict.as_mutable(JSON), default=dict)
    ttps: List[str] = Column(MutableList.as_mutable(ARRAY(String)), default=list)
    attribution: Optional[str] = Column(String, nullable=True)
    confidence: float = Column(Float, default=0.0)
    created_at: datetime = Column(DateTime, default=datetime.utcnow)

class ScanResultModel(Base, BaseModel):
    __tablename__ = "scan_results"

    id: str = Column(String, primary_key=True, default=lambda: str(uuid4()))
    target_id: str = Column(String, index=True)
    scan_type: str = Column(String)  # ScanType stored as string
    status: str = Column(String, default="pending")
    start_time: datetime = Column(DateTime, default=datetime.utcnow)
    end_time: Optional[datetime] = Column(DateTime, nullable=True)
    duration: Optional[float] = Column(Float, nullable=True)
    results: Dict[str, Any] = Column(MutableDict.as_mutable(JSON), default=dict)
    findings: List[Dict[str, Any]] = Column(MutableList.as_mutable(ARRAY(JSON)), default=list)
    metadata: Dict[str, Any] = Column(MutableDict.as_mutable(JSON), default=dict)

class ExploitResultModel(Base, BaseModel):
    __tablename__ = "exploit_results"

    id: str = Column(String, primary_key=True, default=lambda: str(uuid4()))
    target_id: str = Column(String, index=True)
    vulnerability_id: str = Column(String)
    exploit_type: str = Column(String)  # ExploitType stored as string
    status: str = Column(String, default="pending")
    success_probability: float = Column(Float, default=0.0)
    evidence: List[str] = Column(MutableList.as_mutable(ARRAY(String)), default=list)
    payload_used: Optional[str] = Column(String, nullable=True)
    response_received: Optional[str] = Column(String, nullable=True)
    start_time: datetime = Column(DateTime, default=datetime.utcnow)
    end_time: Optional[datetime] = Column(DateTime, nullable=True)

class EvidenceModel(Base, BaseModel):
    __tablename__ = "evidence"

    id: str = Column(String, primary_key=True, default=lambda: str(uuid4()))
    evidence_type: str = Column(String)  # EvidenceType stored as string
    target_id: str = Column(String, index=True)
    scan_id: Optional[str] = Column(String, nullable=True)
    exploit_id: Optional[str] = Column(String, nullable=True)
    file_path: Optional[str] = Column(String, nullable=True)
    content: Optional[str] = Column(String, nullable=True)
    metadata: Dict[str, Any] = Column(MutableDict.as_mutable(JSON), default=dict)
    timestamp: datetime = Column(DateTime, default=datetime.utcnow)
