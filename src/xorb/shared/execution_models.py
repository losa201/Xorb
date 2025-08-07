from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from xorb.shared.epyc_execution_config import EPYCExecutionConfig
from xorb.shared.execution_enums import ScanType, ExploitType, EvidenceType

class ScanResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    target_id: str
    scan_type: ScanType
    status: str = "pending"  # pending, running, completed, failed
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    results: Dict[str, Any] = Field(default_factory=dict)
    findings: List[Dict[str, Any>] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ExploitResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    target_id: str
    vulnerability_id: str
    exploit_type: ExploitType
    status: str = "pending"  # pending, running, successful, failed
    success_probability: float = 0.0
    evidence: List[str] = Field(default_factory=list)  # Evidence IDs
    payload_used: Optional[str] = None
    response_received: Optional[str] = None
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None

class Evidence(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    evidence_type: EvidenceType
    target_id: str
    scan_id: Optional[str] = None
    exploit_id: Optional[str] = None
    file_path: Optional[str] = None
    content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StealthConfig(BaseModel):
    mode: str = "normal"  # stealth, normal, aggressive
    delay_range: tuple[float, float] = (1.0, 3.0)
    user_agent: str = EPYCExecutionConfig.USER_AGENTS[0]
    proxy_rotation: bool = False
    request_headers: Dict[str, str] = Field(default_factory=dict)
    evasion_techniques: List[str] = Field(default_factory=list)