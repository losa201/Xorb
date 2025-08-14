#!/usr/bin/env python3
"""
Replay Store Event Schema Definitions
Comprehensive schemas for cyber range episode replay and analysis
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import json
import zstandard as zstd
import uuid

class EventType(Enum):
    """Event types for cyber range episodes"""
    ATTACK_ACTION = "attack_action"
    DEFENSE_ACTION = "defense_action"
    ENVIRONMENT_CHANGE = "environment_change"
    DETECTION = "detection"
    RESPONSE = "response"
    METRIC = "metric"
    SYSTEM_EVENT = "system_event"
    AI_DECISION = "ai_decision"
    SAFETY_VIOLATION = "safety_violation"

class ThreatLevel(Enum):
    """Threat severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ActorType(Enum):
    """Actor types in the cyber range"""
    RED_AGENT = "red_agent"
    BLUE_AGENT = "blue_agent"
    PURPLE_AGENT = "purple_agent"
    HUMAN_OPERATOR = "human_operator"
    SYSTEM = "system"
    AI_ORCHESTRATOR = "ai_orchestrator"

@dataclass
class BaseEvent:
    """Base event structure for all cyber range events"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    episode_id: str = ""
    round_id: int = 0
    event_type: EventType = EventType.SYSTEM_EVENT
    actor_type: ActorType = ActorType.SYSTEM
    actor_id: str = ""
    
    # Core event data
    action: str = ""
    target: str = ""
    success: bool = False
    
    # Contextual information
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    
    # Trace information
    parent_event_id: Optional[str] = None
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class AttackEvent(BaseEvent):
    """Red team attack event with MITRE ATT&CK mapping"""
    event_type: EventType = EventType.ATTACK_ACTION
    actor_type: ActorType = ActorType.RED_AGENT
    
    # MITRE ATT&CK framework mapping
    technique_id: str = ""  # e.g., "T1190"
    technique_name: str = ""  # e.g., "Exploit Public-Facing Application"
    tactic: str = ""  # e.g., "initial_access"
    
    # Attack specifics
    payload: Optional[str] = None
    vulnerability_exploited: Optional[str] = None
    credentials_used: List[str] = field(default_factory=list)
    persistence_mechanism: Optional[str] = None
    
    # Impact assessment
    confidentiality_impact: ThreatLevel = ThreatLevel.LOW
    integrity_impact: ThreatLevel = ThreatLevel.LOW
    availability_impact: ThreatLevel = ThreatLevel.LOW
    
    # Detection evasion
    stealth_techniques: List[str] = field(default_factory=list)
    detected: bool = False
    detection_timestamp: Optional[str] = None

@dataclass
class DefenseEvent(BaseEvent):
    """Blue team defense event"""
    event_type: EventType = EventType.DEFENSE_ACTION
    actor_type: ActorType = ActorType.BLUE_AGENT
    
    # Defense categorization
    defense_category: str = ""  # detection, prevention, response, deception, hunting
    technique: str = ""
    effectiveness: str = "medium"  # low, medium, high
    
    # Resource allocation
    cost: int = 0  # 1-10 scale
    deployment_time: int = 0  # seconds
    
    # Coverage and monitoring
    coverage_area: List[str] = field(default_factory=list)
    monitoring_enhancement: bool = False
    
    # Response to threats
    threat_addressed: Optional[str] = None
    countermeasure_type: str = ""
    
    # Metrics
    false_positive_risk: float = 0.0
    detection_confidence: float = 0.0

@dataclass
class DetectionEvent(BaseEvent):
    """Threat detection event from security tools"""
    event_type: EventType = EventType.DETECTION
    
    # Detection specifics
    detection_source: str = ""  # IDS, WAF, EDR, etc.
    confidence_score: float = 0.0
    threat_level: ThreatLevel = ThreatLevel.LOW
    
    # Threat intelligence
    indicator_type: str = ""  # IOC, behavior, signature
    indicator_value: str = ""
    threat_description: str = ""
    
    # Correlation
    related_events: List[str] = field(default_factory=list)
    attack_chain_position: Optional[int] = None
    
    # Response
    auto_response_triggered: bool = False
    analyst_review_required: bool = False
    escalation_level: int = 0

@dataclass
class EnvironmentEvent(BaseEvent):
    """Purple team environment change event"""
    event_type: EventType = EventType.ENVIRONMENT_CHANGE
    actor_type: ActorType = ActorType.PURPLE_AGENT
    
    # Change details
    change_type: str = ""  # configuration, vulnerability, infrastructure, policy
    change_description: str = ""
    impact_level: ThreatLevel = ThreatLevel.LOW
    
    # Affected systems
    affected_systems: List[str] = field(default_factory=list)
    affected_services: List[str] = field(default_factory=list)
    
    # Change management
    scheduled: bool = False
    rollback_possible: bool = True
    rollback_window: int = 3600  # seconds
    
    # Compliance impact
    compliance_frameworks_affected: List[str] = field(default_factory=list)
    regulatory_notification_required: bool = False

@dataclass
class AIDecisionEvent(BaseEvent):
    """AI orchestrator decision event"""
    event_type: EventType = EventType.AI_DECISION
    actor_type: ActorType = ActorType.AI_ORCHESTRATOR
    
    # Model information
    model_name: str = ""
    model_version: str = ""
    inference_time_ms: int = 0
    
    # Decision context
    decision_type: str = ""  # strategy, tactic, response, escalation
    input_features: Dict[str, Any] = field(default_factory=dict)
    output_decision: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    
    # Alternative considerations
    alternatives_considered: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    
    # Safety and validation
    safety_check_passed: bool = True
    verifier_consensus: bool = True
    human_override_available: bool = True

@dataclass
class SafetyViolationEvent(BaseEvent):
    """Safety violation or boundary breach event"""
    event_type: EventType = EventType.SAFETY_VIOLATION
    
    # Violation details
    violation_type: str = ""  # scope_breach, roe_violation, resource_limit
    severity: ThreatLevel = ThreatLevel.HIGH
    violation_description: str = ""
    
    # Safety systems
    detected_by: str = ""  # safety_critic, scope_enforcer, kill_switch
    automatic_mitigation: bool = False
    mitigation_actions: List[str] = field(default_factory=list)
    
    # Impact and response
    episode_terminated: bool = False
    human_notification_sent: bool = False
    investigation_required: bool = True

class EventEncoder:
    """High-performance event encoding for JSONL storage"""
    
    @staticmethod
    def encode_event(event: BaseEvent) -> bytes:
        """Encode event to compressed JSONL format"""
        # Convert dataclass to dict, handling enums
        event_dict = {}
        for field_name, field_value in event.__dict__.items():
            if isinstance(field_value, Enum):
                event_dict[field_name] = field_value.value
            elif isinstance(field_value, datetime):
                event_dict[field_name] = field_value.isoformat()
            else:
                event_dict[field_name] = field_value
        
        # Add schema version for forward compatibility
        event_dict['_schema_version'] = "1.0.0"
        event_dict['_event_class'] = event.__class__.__name__
        
        # Serialize to JSON
        json_bytes = json.dumps(event_dict, separators=(',', ':')).encode('utf-8')
        
        # Compress with zstandard
        compressor = zstd.ZstdCompressor(level=3)
        return compressor.compress(json_bytes)
    
    @staticmethod
    def decode_event(compressed_data: bytes) -> Dict[str, Any]:
        """Decode compressed JSONL event"""
        decompressor = zstd.ZstdDecompressor()
        json_bytes = decompressor.decompress(compressed_data)
        return json.loads(json_bytes.decode('utf-8'))

class EventStream:
    """High-throughput event stream manager"""
    
    def __init__(self, stream_path: str):
        self.stream_path = stream_path
        self.encoder = EventEncoder()
        self._buffer = []
        self._buffer_size = 0
        self.max_buffer_size = 1024 * 1024  # 1MB buffer
    
    def write_event(self, event: BaseEvent):
        """Write event to stream with buffering"""
        encoded = self.encoder.encode_event(event)
        self._buffer.append(encoded)
        self._buffer_size += len(encoded)
        
        # Flush if buffer is full
        if self._buffer_size >= self.max_buffer_size:
            self.flush()
    
    def flush(self):
        """Flush buffered events to disk"""
        if not self._buffer:
            return
        
        with open(self.stream_path, 'ab') as f:
            for encoded_event in self._buffer:
                f.write(encoded_event + b'\n')
        
        self._buffer.clear()
        self._buffer_size = 0
    
    def read_events(self) -> List[Dict[str, Any]]:
        """Read all events from stream"""
        events = []
        try:
            with open(self.stream_path, 'rb') as f:
                for line in f:
                    if line.strip():
                        events.append(self.encoder.decode_event(line.strip()))
        except FileNotFoundError:
            pass
        return events

# Event factory for creating specific event types
def create_attack_event(
    episode_id: str,
    round_id: int,
    actor_id: str,
    technique_id: str,
    technique_name: str,
    tactic: str,
    target: str,
    success: bool,
    **kwargs
) -> AttackEvent:
    """Factory function for creating attack events"""
    return AttackEvent(
        episode_id=episode_id,
        round_id=round_id,
        actor_id=actor_id,
        technique_id=technique_id,
        technique_name=technique_name,
        tactic=tactic,
        target=target,
        success=success,
        **kwargs
    )

def create_defense_event(
    episode_id: str,
    round_id: int,
    actor_id: str,
    defense_category: str,
    technique: str,
    target: str,
    effectiveness: str,
    cost: int,
    **kwargs
) -> DefenseEvent:
    """Factory function for creating defense events"""
    return DefenseEvent(
        episode_id=episode_id,
        round_id=round_id,
        actor_id=actor_id,
        defense_category=defense_category,
        technique=technique,
        target=target,
        effectiveness=effectiveness,
        cost=cost,
        **kwargs
    )

def create_detection_event(
    episode_id: str,
    round_id: int,
    detection_source: str,
    confidence_score: float,
    threat_level: ThreatLevel,
    indicator_type: str,
    indicator_value: str,
    **kwargs
) -> DetectionEvent:
    """Factory function for creating detection events"""
    return DetectionEvent(
        episode_id=episode_id,
        round_id=round_id,
        detection_source=detection_source,
        confidence_score=confidence_score,
        threat_level=threat_level,
        indicator_type=indicator_type,
        indicator_value=indicator_value,
        **kwargs
    )

if __name__ == "__main__":
    # Example usage and testing
    print("Testing event schema and encoding...")
    
    # Create sample events
    attack_event = create_attack_event(
        episode_id="ep_001",
        round_id=1,
        actor_id="red_agent_001",
        technique_id="T1190",
        technique_name="Exploit Public-Facing Application",
        tactic="initial_access",
        target="corporate_website",
        success=True,
        payload="POST /wp-admin/admin-ajax.php",
        detected=False
    )
    
    defense_event = create_defense_event(
        episode_id="ep_001",
        round_id=1,
        actor_id="blue_agent_001",
        defense_category="prevention",
        technique="Vulnerability Patching",
        target="wordpress_plugin",
        effectiveness="high",
        cost=3
    )
    
    # Test encoding/decoding
    encoder = EventEncoder()
    
    # Encode events
    encoded_attack = encoder.encode_event(attack_event)
    encoded_defense = encoder.encode_event(defense_event)
    
    print(f"Attack event compressed size: {len(encoded_attack)} bytes")
    print(f"Defense event compressed size: {len(encoded_defense)} bytes")
    
    # Decode events
    decoded_attack = encoder.decode_event(encoded_attack)
    decoded_defense = encoder.decode_event(encoded_defense)
    
    print("Successfully encoded and decoded events!")
    print(f"Attack event type: {decoded_attack['_event_class']}")
    print(f"Defense event type: {decoded_defense['_event_class']}")