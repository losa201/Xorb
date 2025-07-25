#!/usr/bin/env python3

import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib


class AtomType(str, Enum):
    VULNERABILITY = "vulnerability"
    TECHNIQUE = "technique"
    PAYLOAD = "payload"
    TARGET_INFO = "target_info"
    INTELLIGENCE = "intelligence"
    EXPLOIT = "exploit"
    DEFENSIVE = "defensive"


class ConfidenceLevel(str, Enum):
    VERY_LOW = "very_low"      # 0.0 - 0.2
    LOW = "low"                # 0.2 - 0.4
    MEDIUM = "medium"          # 0.4 - 0.6
    HIGH = "high"              # 0.6 - 0.8
    VERY_HIGH = "very_high"    # 0.8 - 1.0


@dataclass
class Source:
    name: str
    type: str  # "llm", "agent", "manual", "import", "validation"
    version: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    reliability_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    is_valid: bool
    confidence_adjustment: float = 0.0
    validation_method: str = ""
    validation_timestamp: datetime = field(default_factory=datetime.utcnow)
    notes: str = ""


@dataclass 
class KnowledgeAtom:
    atom_type: AtomType
    title: str
    content: Dict[str, Any]
    id: Optional[str] = None
    confidence: float = 0.5
    predictive_score: float = 0.0
    tags: Set[str] = field(default_factory=set)
    sources: List[Source] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    validation_results: List[ValidationResult] = field(default_factory=list)
    related_atoms: Set[str] = field(default_factory=set)
    usage_count: int = 0
    success_rate: float = 0.0
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        
        if not self.expires_at and self.atom_type in [AtomType.TARGET_INFO, AtomType.INTELLIGENCE]:
            self.expires_at = self.created_at + timedelta(days=7)

    @property
    def confidence_level(self) -> ConfidenceLevel:
        if self.confidence <= 0.2:
            return ConfidenceLevel.VERY_LOW
        elif self.confidence <= 0.4:
            return ConfidenceLevel.LOW
        elif self.confidence <= 0.6:
            return ConfidenceLevel.MEDIUM
        elif self.confidence <= 0.8:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH

    @property
    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def content_hash(self) -> str:
        content_str = json.dumps(self.content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def add_source(self, source: Source):
        self.sources.append(source)
        self._recalculate_confidence()
        self.updated_at = datetime.now(timezone.utc)

    def add_validation(self, result: ValidationResult):
        self.validation_results.append(result)
        if result.confidence_adjustment != 0:
            self.confidence = max(0.0, min(1.0, self.confidence + result.confidence_adjustment))
        self.updated_at = datetime.now(timezone.utc)

    def add_related_atom(self, atom_id: str, bidirectional: bool = True):
        self.related_atoms.add(atom_id)
        self.updated_at = datetime.now(timezone.utc)

    def record_usage(self, success: bool = True):
        self.usage_count += 1
        
        if self.usage_count == 1:
            self.success_rate = 1.0 if success else 0.0
        else:
            success_count = int(self.success_rate * (self.usage_count - 1))
            if success:
                success_count += 1
            self.success_rate = success_count / self.usage_count
        
        self._recalculate_predictive_score()
        self.updated_at = datetime.now(timezone.utc)

    def update_content(self, new_content: Dict[str, Any], source: Optional[Source] = None):
        old_hash = self.content_hash
        self.content.update(new_content)
        new_hash = self.content_hash
        
        if old_hash != new_hash and source:
            self.add_source(source)
        
        self.updated_at = datetime.now(timezone.utc)

    def _recalculate_confidence(self):
        if not self.sources:
            self.confidence = 0.5
            return
        
        source_weights = {
            "validation": 0.4,
            "agent": 0.3,
            "llm": 0.2,
            "manual": 0.5,
            "import": 0.1
        }
        
        total_weight = 0
        weighted_confidence = 0
        
        for source in self.sources:
            weight = source_weights.get(source.type, 0.1)
            reliability = source.reliability_score
            
            total_weight += weight
            weighted_confidence += weight * reliability
        
        if total_weight > 0:
            base_confidence = weighted_confidence / total_weight
            
            validation_boost = len([v for v in self.validation_results if v.is_valid]) * 0.1
            validation_penalty = len([v for v in self.validation_results if not v.is_valid]) * 0.2
            
            self.confidence = max(0.0, min(1.0, base_confidence + validation_boost - validation_penalty))

    def _recalculate_predictive_score(self):
        base_score = self.confidence * 0.5
        
        usage_factor = min(1.0, self.usage_count / 10.0) * 0.3
        success_factor = self.success_rate * 0.2
        
        age_factor = 0.0
        if self.created_at:
            age_days = (datetime.now(timezone.utc) - self.created_at).days
            age_factor = max(0.0, 1.0 - (age_days / 30.0)) * 0.1
        
        relationship_factor = min(1.0, len(self.related_atoms) / 5.0) * 0.1
        
        self.predictive_score = base_score + usage_factor + success_factor + age_factor + relationship_factor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "atom_type": self.atom_type.value,
            "title": self.title,
            "content": self.content,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "predictive_score": self.predictive_score,
            "tags": list(self.tags),
            "sources": [
                {
                    "name": s.name,
                    "type": s.type,
                    "version": s.version,
                    "timestamp": s.timestamp.isoformat(),
                    "reliability_score": s.reliability_score,
                    "metadata": s.metadata
                } for s in self.sources
            ],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "validation_results": [
                {
                    "is_valid": v.is_valid,
                    "confidence_adjustment": v.confidence_adjustment,
                    "validation_method": v.validation_method,
                    "validation_timestamp": v.validation_timestamp.isoformat(),
                    "notes": v.notes
                } for v in self.validation_results
            ],
            "related_atoms": list(self.related_atoms),
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "is_expired": self.is_expired,
            "content_hash": self.content_hash
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeAtom':
        sources = []
        for source_data in data.get("sources", []):
            source = Source(
                name=source_data["name"],
                type=source_data["type"],
                version=source_data.get("version"),
                timestamp=datetime.fromisoformat(source_data["timestamp"]),
                reliability_score=source_data.get("reliability_score", 0.5),
                metadata=source_data.get("metadata", {})
            )
            sources.append(source)
        
        validation_results = []
        for validation_data in data.get("validation_results", []):
            validation = ValidationResult(
                is_valid=validation_data["is_valid"],
                confidence_adjustment=validation_data.get("confidence_adjustment", 0.0),
                validation_method=validation_data.get("validation_method", ""),
                validation_timestamp=datetime.fromisoformat(validation_data["validation_timestamp"]),
                notes=validation_data.get("notes", "")
            )
            validation_results.append(validation)
        
        atom = cls(
            id=data["id"],
            atom_type=AtomType(data["atom_type"]),
            title=data["title"],
            content=data["content"],
            confidence=data["confidence"],
            predictive_score=data.get("predictive_score", 0.0),
            tags=set(data.get("tags", [])),
            sources=sources,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            validation_results=validation_results,
            related_atoms=set(data.get("related_atoms", [])),
            usage_count=data.get("usage_count", 0),
            success_rate=data.get("success_rate", 0.0)
        )
        
        return atom

    def __str__(self) -> str:
        return f"KnowledgeAtom(id={self.id[:8]}, type={self.atom_type.value}, confidence={self.confidence:.2f}, title='{self.title}')"

    def __repr__(self) -> str:
        return self.__str__()