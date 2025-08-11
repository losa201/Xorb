"""
Domain entities - Core business objects with identity and behavior.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4


@dataclass
class User:
    """User domain entity"""
    id: UUID
    username: str
    email: str
    roles: List[str]
    created_at: datetime
    is_active: bool = True
    
    @classmethod
    def create(cls, username: str, email: str, roles: List[str]) -> "User":
        """Factory method to create a new user"""
        return cls(
            id=uuid4(),
            username=username,
            email=email,
            roles=roles,
            created_at=datetime.utcnow(),
            is_active=True
        )
    
    def has_role(self, role: str) -> bool:
        """Check if user has a specific role"""
        return role in self.roles or 'admin' in self.roles
    
    def add_role(self, role: str) -> None:
        """Add a role to the user"""
        if role not in self.roles:
            self.roles.append(role)
    
    def remove_role(self, role: str) -> None:
        """Remove a role from the user"""
        if role in self.roles:
            self.roles.remove(role)


@dataclass
class Organization:
    """Organization domain entity"""
    id: UUID
    name: str
    plan_type: str
    created_at: datetime
    is_active: bool = True
    
    @classmethod
    def create(cls, name: str, plan_type: str) -> "Organization":
        """Factory method to create a new organization"""
        return cls(
            id=uuid4(),
            name=name,
            plan_type=plan_type,
            created_at=datetime.utcnow(),
            is_active=True
        )


@dataclass
class EmbeddingRequest:
    """Embedding request domain entity"""
    id: UUID
    texts: List[str]
    model: str
    input_type: str
    user_id: UUID
    org_id: UUID
    created_at: datetime
    status: str = "pending"
    
    @classmethod
    def create(
        cls, 
        texts: List[str], 
        model: str, 
        input_type: str, 
        user_id: UUID, 
        org_id: UUID
    ) -> "EmbeddingRequest":
        """Factory method to create a new embedding request"""
        return cls(
            id=uuid4(),
            texts=texts,
            model=model,
            input_type=input_type,
            user_id=user_id,
            org_id=org_id,
            created_at=datetime.utcnow(),
            status="pending"
        )
    
    def validate(self) -> None:
        """Validate embedding request"""
        if not self.texts:
            raise ValueError("Input texts cannot be empty")
        
        if len(self.texts) > 100:
            raise ValueError("Maximum 100 texts per request")
        
        if any(not text.strip() for text in self.texts):
            raise ValueError("Input texts cannot be empty strings")
    
    def mark_completed(self) -> None:
        """Mark the request as completed"""
        self.status = "completed"
    
    def mark_failed(self) -> None:
        """Mark the request as failed"""
        self.status = "failed"


@dataclass
class EmbeddingResult:
    """Embedding result domain entity"""
    id: UUID
    request_id: UUID
    embeddings: List[List[float]]
    model: str
    usage_stats: dict
    created_at: datetime
    
    @classmethod
    def create(
        cls,
        request_id: UUID,
        embeddings: List[List[float]],
        model: str,
        usage_stats: dict
    ) -> "EmbeddingResult":
        """Factory method to create a new embedding result"""
        return cls(
            id=uuid4(),
            request_id=request_id,
            embeddings=embeddings,
            model=model,
            usage_stats=usage_stats,
            created_at=datetime.utcnow()
        )


@dataclass
class DiscoveryWorkflow:
    """Discovery workflow domain entity"""
    id: UUID
    domain: str
    status: str
    user_id: UUID
    org_id: UUID
    workflow_id: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    findings: List[dict] = None
    
    @classmethod
    def create(
        cls,
        domain: str,
        user_id: UUID,
        org_id: UUID,
        workflow_id: str
    ) -> "DiscoveryWorkflow":
        """Factory method to create a new discovery workflow"""
        return cls(
            id=uuid4(),
            domain=domain,
            status="started",
            user_id=user_id,
            org_id=org_id,
            workflow_id=workflow_id,
            created_at=datetime.utcnow(),
            findings=[]
        )
    
    def mark_completed(self, findings: List[dict]) -> None:
        """Mark the workflow as completed with findings"""
        self.status = "completed"
        self.completed_at = datetime.utcnow()
        self.findings = findings
    
    def mark_failed(self) -> None:
        """Mark the workflow as failed"""
        self.status = "failed"
        self.completed_at = datetime.utcnow()


@dataclass
class AuthToken:
    """Authentication token domain entity"""
    token: str
    user_id: UUID
    expires_at: datetime
    created_at: datetime
    is_revoked: bool = False
    
    @classmethod
    def create(cls, token: str, user_id: UUID, expires_at: datetime) -> "AuthToken":
        """Factory method to create a new auth token"""
        return cls(
            token=token,
            user_id=user_id,
            expires_at=expires_at,
            created_at=datetime.utcnow(),
            is_revoked=False
        )
    
    def is_valid(self) -> bool:
        """Check if token is valid (not expired and not revoked)"""
        return not self.is_revoked and datetime.utcnow() < self.expires_at
    
    def revoke(self) -> None:
        """Revoke the token"""
        self.is_revoked = True


@dataclass
class ScanSession:
    """PTaaS scan session entity"""
    id: UUID
    organization_id: UUID
    user_id: UUID
    targets: List[str]
    scan_type: str
    status: str  # "pending", "running", "completed", "failed", "cancelled"
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Optional[dict] = None
    metadata: Optional[dict] = None
    
    @classmethod
    def create(
        cls,
        organization_id: UUID,
        user_id: UUID,
        targets: List[str],
        scan_type: str,
        metadata: Optional[dict] = None
    ) -> "ScanSession":
        """Factory method to create a new scan session"""
        return cls(
            id=uuid4(),
            organization_id=organization_id,
            user_id=user_id,
            targets=targets,
            scan_type=scan_type,
            status="pending",
            created_at=datetime.utcnow(),
            metadata=metadata or {}
        )
    
    def start(self) -> None:
        """Mark scan as started"""
        self.status = "running"
        self.started_at = datetime.utcnow()
    
    def complete(self, results: dict) -> None:
        """Mark scan as completed with results"""
        self.status = "completed"
        self.completed_at = datetime.utcnow()
        self.results = results
    
    def fail(self, error: str) -> None:
        """Mark scan as failed"""
        self.status = "failed"
        self.completed_at = datetime.utcnow()
        self.results = {"error": error}
    
    def cancel(self) -> None:
        """Cancel the scan"""
        self.status = "cancelled"
        self.completed_at = datetime.utcnow()


@dataclass
class ThreatAlert:
    """Security threat alert entity"""
    id: UUID
    organization_id: UUID
    scan_session_id: Optional[UUID]
    alert_type: str
    severity: str  # "low", "medium", "high", "critical"
    title: str
    description: str
    source: str
    indicators: List[str]
    created_at: datetime
    updated_at: datetime
    status: str = "active"  # "active", "investigating", "resolved", "false_positive"
    assigned_to: Optional[UUID] = None
    metadata: Optional[dict] = None
    
    @classmethod
    def create(
        cls,
        organization_id: UUID,
        alert_type: str,
        severity: str,
        title: str,
        description: str,
        source: str,
        indicators: List[str],
        scan_session_id: Optional[UUID] = None,
        metadata: Optional[dict] = None
    ) -> "ThreatAlert":
        """Factory method to create a new threat alert"""
        return cls(
            id=uuid4(),
            organization_id=organization_id,
            scan_session_id=scan_session_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            description=description,
            source=source,
            indicators=indicators,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata=metadata or {}
        )
    
    def update_status(self, status: str, assigned_to: Optional[UUID] = None) -> None:
        """Update alert status"""
        self.status = status
        self.updated_at = datetime.utcnow()
        if assigned_to:
            self.assigned_to = assigned_to
    
    def is_critical(self) -> bool:
        """Check if alert is critical severity"""
        return self.severity == "critical"