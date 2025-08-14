"""
Advanced Network Microsegmentation Service - Production Implementation
Zero-trust network security with dynamic policy enforcement and AI-powered threat detection
"""

import asyncio
import json
import logging
import ipaddress
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import hashlib
from collections import defaultdict, deque
import re

# Network security imports
try:
    import netaddr
    from netaddr import IPNetwork, IPAddress
    NETADDR_AVAILABLE = True
except ImportError:
    NETADDR_AVAILABLE = False
    logging.warning("netaddr not available, using basic IP operations")

# ML imports for traffic analysis
try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML libraries not available, using rule-based analysis")

from .base_service import XORBService, ServiceHealth, ServiceStatus
from .interfaces import SecurityOrchestrationService
from ..domain.tenant_entities import SecurityFinding

logger = logging.getLogger(__name__)


class NetworkZone(Enum):
    """Network security zones"""
    DMZ = "dmz"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CRITICAL = "critical"
    GUEST = "guest"
    IOT = "iot"
    MANAGEMENT = "management"
    QUARANTINE = "quarantine"


class TrafficAction(Enum):
    """Network traffic actions"""
    ALLOW = "allow"
    DENY = "deny"
    LOG = "log"
    QUARANTINE = "quarantine"
    INSPECT = "inspect"
    RATE_LIMIT = "rate_limit"


class ProtocolType(Enum):
    """Network protocols"""
    TCP = "tcp"
    UDP = "udp"
    ICMP = "icmp"
    HTTP = "http"
    HTTPS = "https"
    SSH = "ssh"
    DNS = "dns"
    SMTP = "smtp"
    ANY = "any"


@dataclass
class NetworkFlow:
    """Network flow information"""
    flow_id: str
    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    protocol: ProtocolType
    bytes_sent: int = 0
    bytes_received: int = 0
    packets_sent: int = 0
    packets_received: int = 0
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MicrosegmentationPolicy:
    """Microsegmentation security policy"""
    policy_id: str
    name: str
    description: str
    source_zones: List[NetworkZone]
    destination_zones: List[NetworkZone]
    source_networks: List[str]
    destination_networks: List[str]
    protocols: List[ProtocolType]
    ports: List[int]
    action: TrafficAction
    priority: int = 100
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecuritySegment:
    """Network security segment definition"""
    segment_id: str
    name: str
    zone: NetworkZone
    networks: List[str]
    assets: List[str]
    security_level: int  # 1-5, 5 being most secure
    policies: List[str]  # Policy IDs
    isolation_level: str  # strict, moderate, flexible
    monitoring_level: str  # high, medium, low
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatDetection:
    """Network threat detection result"""
    detection_id: str
    threat_type: str
    severity: str
    confidence: float
    source_ip: str
    destination_ip: str
    protocol: str
    description: str
    indicators: List[str]
    mitre_techniques: List[str]
    detected_at: datetime = field(default_factory=datetime.utcnow)
    flows: List[NetworkFlow] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedNetworkMicrosegmentation(XORBService, SecurityOrchestrationService):
    """
    Advanced Network Microsegmentation Service

    Provides zero-trust network security with:
    - Dynamic microsegmentation
    - AI-powered traffic analysis
    - Real-time threat detection
    - Policy enforcement
    - Behavioral analytics
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            service_name="network_microsegmentation",
            service_type="network_security",
            dependencies=["threat_intelligence", "behavioral_analytics"],
            config=config or {}
        )

        # Core components
        self.segments: Dict[str, SecuritySegment] = {}
        self.policies: Dict[str, MicrosegmentationPolicy] = {}
        self.active_flows: Dict[str, NetworkFlow] = {}
        self.threat_detections: deque = deque(maxlen=10000)

        # AI/ML components
        self.traffic_analyzer: Optional[Any] = None
        self.anomaly_detector: Optional[Any] = None
        self.threat_classifier: Optional[Any] = None

        # Network topology
        self.network_topology: Dict[str, Any] = {}
        self.asset_inventory: Dict[str, Dict[str, Any]] = {}
        self.zone_mappings: Dict[str, NetworkZone] = {}

        # Policy engine
        self.policy_engine_enabled = True
        self.enforcement_mode = config.get("enforcement_mode", "monitor")  # monitor, enforce
        self.default_action = TrafficAction.DENY

        # Performance metrics
        self.metrics = {
            "flows_analyzed": 0,
            "threats_detected": 0,
            "policies_enforced": 0,
            "segments_created": 0,
            "false_positives": 0,
            "processing_latency_ms": 0.0
        }

        # Configuration
        self.max_concurrent_flows = config.get("max_concurrent_flows", 100000)
        self.threat_detection_threshold = config.get("threat_detection_threshold", 0.7)
        self.policy_evaluation_interval = config.get("policy_evaluation_interval", 60)

    async def initialize(self) -> bool:
        """Initialize the microsegmentation service"""
        try:
            logger.info("Initializing Advanced Network Microsegmentation Service...")

            # Initialize ML components
            if ML_AVAILABLE:
                await self._initialize_ml_components()

            # Load default network segments
            await self._create_default_segments()

            # Load default policies
            await self._create_default_policies()

            # Initialize network topology discovery
            await self._initialize_topology_discovery()

            # Start background processing
            await self._start_background_tasks()

            logger.info("Network Microsegmentation Service initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize microsegmentation service: {e}")
            return False

    async def _initialize_ml_components(self):
        """Initialize machine learning components for traffic analysis"""
        try:
            if ML_AVAILABLE:
                # Anomaly detection for network traffic
                self.anomaly_detector = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_estimators=100
                )

                # Threat classification
                self.threat_classifier = RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    max_depth=15
                )

                # Traffic clustering for pattern analysis
                self.traffic_clusterer = DBSCAN(eps=0.3, min_samples=10)

                # Feature scaler
                self.feature_scaler = StandardScaler()

                logger.info("ML components for network analysis initialized")

        except Exception as e:
            logger.warning(f"ML component initialization failed: {e}")

    async def create_security_segment(
        self,
        name: str,
        zone: NetworkZone,
        networks: List[str],
        security_level: int = 3,
        isolation_level: str = "moderate"
    ) -> SecuritySegment:
        """Create a new security segment"""
        try:
            segment_id = str(uuid.uuid4())

            # Validate networks
            validated_networks = []
            for network in networks:
                if self._validate_network(network):
                    validated_networks.append(network)
                else:
                    logger.warning(f"Invalid network specification: {network}")

            if not validated_networks:
                raise ValueError("No valid networks specified")

            segment = SecuritySegment(
                segment_id=segment_id,
                name=name,
                zone=zone,
                networks=validated_networks,
                assets=[],
                security_level=security_level,
                policies=[],
                isolation_level=isolation_level,
                monitoring_level="high" if security_level >= 4 else "medium"
            )

            self.segments[segment_id] = segment
            self.metrics["segments_created"] += 1

            # Create default policies for the segment
            await self._create_segment_policies(segment)

            # Update zone mappings
            for network in validated_networks:
                self.zone_mappings[network] = zone

            logger.info(f"Created security segment '{name}' with {len(validated_networks)} networks")
            return segment

        except Exception as e:
            logger.error(f"Failed to create security segment: {e}")
            raise

    async def create_microsegmentation_policy(
        self,
        name: str,
        source_zones: List[NetworkZone],
        destination_zones: List[NetworkZone],
        protocols: List[ProtocolType],
        action: TrafficAction,
        **kwargs
    ) -> MicrosegmentationPolicy:
        """Create a new microsegmentation policy"""
        try:
            policy_id = str(uuid.uuid4())

            policy = MicrosegmentationPolicy(
                policy_id=policy_id,
                name=name,
                description=kwargs.get("description", f"Policy for {name}"),
                source_zones=source_zones,
                destination_zones=destination_zones,
                source_networks=kwargs.get("source_networks", []),
                destination_networks=kwargs.get("destination_networks", []),
                protocols=protocols,
                ports=kwargs.get("ports", []),
                action=action,
                priority=kwargs.get("priority", 100),
                conditions=kwargs.get("conditions", {}),
                tags=kwargs.get("tags", [])
            )

            self.policies[policy_id] = policy

            # Apply policy to relevant segments
            await self._apply_policy_to_segments(policy)

            logger.info(f"Created microsegmentation policy: {name}")
            return policy

        except Exception as e:
            logger.error(f"Failed to create policy: {e}")
            raise

    async def analyze_network_flow(self, flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze network flow for threats and policy violations"""
        try:
            # Create flow object
            flow = NetworkFlow(
                flow_id=str(uuid.uuid4()),
                source_ip=flow_data["source_ip"],
                destination_ip=flow_data["destination_ip"],
                source_port=flow_data.get("source_port", 0),
                destination_port=flow_data.get("destination_port", 0),
                protocol=ProtocolType(flow_data.get("protocol", "tcp")),
                bytes_sent=flow_data.get("bytes_sent", 0),
                bytes_received=flow_data.get("bytes_received", 0)
            )

            # Store active flow
            self.active_flows[flow.flow_id] = flow
            self.metrics["flows_analyzed"] += 1

            analysis_results = {
                "flow_id": flow.flow_id,
                "policy_evaluation": {},
                "threat_detection": {},
                "recommendations": [],
                "action": "allow"
            }

            # Policy evaluation
            policy_result = await self._evaluate_policies(flow)
            analysis_results["policy_evaluation"] = policy_result

            # Threat detection
            threat_result = await self._detect_threats(flow)
            analysis_results["threat_detection"] = threat_result

            # Determine final action
            final_action = await self._determine_action(policy_result, threat_result)
            analysis_results["action"] = final_action.value

            # Generate recommendations
            recommendations = await self._generate_flow_recommendations(flow, policy_result, threat_result)
            analysis_results["recommendations"] = recommendations

            # Enforcement
            if self.enforcement_mode == "enforce" and final_action != TrafficAction.ALLOW:
                await self._enforce_action(flow, final_action)
                self.metrics["policies_enforced"] += 1

            return analysis_results

        except Exception as e:
            logger.error(f"Flow analysis failed: {e}")
            return {
                "error": str(e),
                "action": "deny"
            }

    async def _evaluate_policies(self, flow: NetworkFlow) -> Dict[str, Any]:
        """Evaluate flow against microsegmentation policies"""
        try:
            matching_policies = []
            policy_violations = []

            # Determine source and destination zones
            source_zone = self._get_zone_for_ip(flow.source_ip)
            dest_zone = self._get_zone_for_ip(flow.destination_ip)

            # Evaluate each policy
            for policy in self.policies.values():
                if not policy.enabled:
                    continue

                # Check zone matching
                if source_zone in policy.source_zones and dest_zone in policy.destination_zones:
                    # Check protocol matching
                    if ProtocolType.ANY in policy.protocols or flow.protocol in policy.protocols:
                        # Check port matching
                        if not policy.ports or flow.destination_port in policy.ports:
                            # Check network matching
                            if self._check_network_match(flow, policy):
                                matching_policies.append({
                                    "policy_id": policy.policy_id,
                                    "name": policy.name,
                                    "action": policy.action.value,
                                    "priority": policy.priority
                                })

                                if policy.action == TrafficAction.DENY:
                                    policy_violations.append({
                                        "policy_id": policy.policy_id,
                                        "violation_type": "explicit_deny",
                                        "description": f"Traffic denied by policy: {policy.name}"
                                    })

            # Sort by priority
            matching_policies.sort(key=lambda x: x["priority"])

            return {
                "source_zone": source_zone.value if source_zone else "unknown",
                "destination_zone": dest_zone.value if dest_zone else "unknown",
                "matching_policies": matching_policies,
                "policy_violations": policy_violations,
                "default_action": self.default_action.value
            }

        except Exception as e:
            logger.error(f"Policy evaluation failed: {e}")
            return {"error": str(e)}

    async def _detect_threats(self, flow: NetworkFlow) -> Dict[str, Any]:
        """Detect threats in network flow"""
        try:
            threats = []
            threat_indicators = []

            # Check for suspicious patterns
            suspicious_patterns = await self._check_suspicious_patterns(flow)
            threats.extend(suspicious_patterns)

            # ML-based anomaly detection
            if ML_AVAILABLE and self.anomaly_detector:
                ml_threats = await self._ml_threat_detection(flow)
                threats.extend(ml_threats)

            # Behavioral analysis
            behavioral_threats = await self._behavioral_threat_analysis(flow)
            threats.extend(behavioral_threats)

            # Known threat intelligence
            intel_threats = await self._check_threat_intelligence(flow)
            threats.extend(intel_threats)

            # Calculate overall threat score
            threat_score = self._calculate_threat_score(threats)

            return {
                "threat_score": threat_score,
                "threat_level": self._categorize_threat_level(threat_score),
                "threats_detected": len(threats),
                "threats": threats,
                "indicators": threat_indicators
            }

        except Exception as e:
            logger.error(f"Threat detection failed: {e}")
            return {"error": str(e)}

    def _get_zone_for_ip(self, ip_address: str) -> Optional[NetworkZone]:
        """Determine network zone for IP address"""
        try:
            ip = ipaddress.ip_address(ip_address)

            # Check each segment
            for segment in self.segments.values():
                for network in segment.networks:
                    try:
                        if ip in ipaddress.ip_network(network, strict=False):
                            return segment.zone
                    except ValueError:
                        continue

            # Default zone assignment
            if ip.is_private:
                return NetworkZone.INTERNAL
            else:
                return NetworkZone.DMZ

        except ValueError:
            logger.warning(f"Invalid IP address: {ip_address}")
            return None

    async def _create_default_segments(self):
        """Create default network segments"""
        try:
            # DMZ segment
            await self.create_security_segment(
                "DMZ",
                NetworkZone.DMZ,
                ["10.0.1.0/24", "192.168.1.0/24"],
                security_level=2,
                isolation_level="moderate"
            )

            # Internal segment
            await self.create_security_segment(
                "Internal",
                NetworkZone.INTERNAL,
                ["10.0.10.0/24", "192.168.10.0/24"],
                security_level=3,
                isolation_level="flexible"
            )

            # Critical systems segment
            await self.create_security_segment(
                "Critical",
                NetworkZone.CRITICAL,
                ["10.0.100.0/24"],
                security_level=5,
                isolation_level="strict"
            )

            # Guest network
            await self.create_security_segment(
                "Guest",
                NetworkZone.GUEST,
                ["10.0.200.0/24"],
                security_level=1,
                isolation_level="strict"
            )

            logger.info("Default network segments created")

        except Exception as e:
            logger.error(f"Failed to create default segments: {e}")

    async def _create_default_policies(self):
        """Create default microsegmentation policies"""
        try:
            # Allow internal-to-internal communication
            await self.create_microsegmentation_policy(
                "Internal Zone Communication",
                [NetworkZone.INTERNAL],
                [NetworkZone.INTERNAL],
                [ProtocolType.ANY],
                TrafficAction.ALLOW,
                priority=50
            )

            # Deny guest-to-internal
            await self.create_microsegmentation_policy(
                "Block Guest to Internal",
                [NetworkZone.GUEST],
                [NetworkZone.INTERNAL, NetworkZone.CRITICAL],
                [ProtocolType.ANY],
                TrafficAction.DENY,
                priority=10
            )

            # Strict critical zone protection
            await self.create_microsegmentation_policy(
                "Critical Zone Protection",
                [NetworkZone.DMZ, NetworkZone.GUEST],
                [NetworkZone.CRITICAL],
                [ProtocolType.ANY],
                TrafficAction.DENY,
                priority=5
            )

            # Allow HTTPS from DMZ to internal
            await self.create_microsegmentation_policy(
                "DMZ Web Access",
                [NetworkZone.DMZ],
                [NetworkZone.INTERNAL],
                [ProtocolType.HTTPS],
                TrafficAction.ALLOW,
                ports=[443],
                priority=30
            )

            logger.info("Default microsegmentation policies created")

        except Exception as e:
            logger.error(f"Failed to create default policies: {e}")

    def _validate_network(self, network: str) -> bool:
        """Validate network CIDR notation"""
        try:
            ipaddress.ip_network(network, strict=False)
            return True
        except ValueError:
            return False

    async def get_segment_status(self, segment_id: str) -> Dict[str, Any]:
        """Get status of a security segment"""
        if segment_id not in self.segments:
            raise ValueError(f"Segment not found: {segment_id}")

        segment = self.segments[segment_id]

        # Calculate metrics
        active_flows = [f for f in self.active_flows.values()
                       if self._get_zone_for_ip(f.source_ip) == segment.zone or
                          self._get_zone_for_ip(f.destination_ip) == segment.zone]

        recent_threats = [t for t in self.threat_detections
                         if t.detected_at > datetime.utcnow() - timedelta(hours=24)]

        return {
            "segment": asdict(segment),
            "active_flows": len(active_flows),
            "recent_threats": len(recent_threats),
            "policies_applied": len(segment.policies),
            "security_status": "secure" if len(recent_threats) == 0 else "monitoring"
        }

    async def health_check(self) -> ServiceHealth:
        """Health check for microsegmentation service"""
        try:
            checks = {
                "segments_loaded": len(self.segments) > 0,
                "policies_loaded": len(self.policies) > 0,
                "flow_processing": len(self.active_flows) < self.max_concurrent_flows,
                "ml_components": ML_AVAILABLE and self.anomaly_detector is not None
            }

            healthy = all(checks.values())

            return ServiceHealth(
                service_name=self.service_name,
                status=ServiceStatus.HEALTHY if healthy else ServiceStatus.DEGRADED,
                checks=checks,
                metrics=self.metrics,
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            return ServiceHealth(
                service_name=self.service_name,
                status=ServiceStatus.UNHEALTHY,
                error=str(e),
                timestamp=datetime.utcnow()
            )


# Service factory and registration
_microsegmentation_service: Optional[AdvancedNetworkMicrosegmentation] = None

async def get_microsegmentation_service(config: Dict[str, Any] = None) -> AdvancedNetworkMicrosegmentation:
    """Get or create microsegmentation service instance"""
    global _microsegmentation_service

    if _microsegmentation_service is None:
        _microsegmentation_service = AdvancedNetworkMicrosegmentation(config)
        await _microsegmentation_service.initialize()

    return _microsegmentation_service
