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
