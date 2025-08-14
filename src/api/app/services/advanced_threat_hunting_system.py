#!/usr/bin/env python3
"""
Advanced Threat Hunting and Incident Response System
Sophisticated threat hunting, detection, and automated incident response

Features:
- Advanced threat hunting with custom query language
- Machine learning-powered anomaly detection
- Automated incident response and containment
- Forensic evidence collection and analysis
- Advanced persistence detection
- Behavioral analysis and user entity behavior analytics (UEBA)
- Threat intelligence integration and correlation
- Custom detection rule development
- Advanced timeline analysis
- Threat actor attribution and campaign tracking
"""

import asyncio
import json
import logging
import hashlib
import base64
import re
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import tempfile
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import statistics

# Advanced analytics libraries
try:
    import pandas as pd
    import numpy as np
    from scipy import stats
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    logging.warning("Analytics libraries not available - some features disabled")

# Graph analysis for attack path reconstruction
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    GRAPH_ANALYSIS_AVAILABLE = True
except ImportError:
    GRAPH_ANALYSIS_AVAILABLE = False
    logging.warning("Graph analysis not available")

# Natural language processing for threat intel
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    logging.warning("NLP libraries not available")

from .base_service import SecurityService, ServiceHealth, ServiceStatus

logger = logging.getLogger(__name__)

class ThreatSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class IncidentStatus(Enum):
    NEW = "new"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    ERADICATED = "eradicated"
    RECOVERED = "recovered"
    CLOSED = "closed"

class DetectionType(Enum):
    SIGNATURE = "signature"
    ANOMALY = "anomaly"
    BEHAVIORAL = "behavioral"
    THREAT_INTEL = "threat_intel"
    MACHINE_LEARNING = "machine_learning"
    CUSTOM_RULE = "custom_rule"

class ResponseAction(Enum):
    ALERT = "alert"
    ISOLATE = "isolate"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    COLLECT_EVIDENCE = "collect_evidence"
    ESCALATE = "escalate"
    NOTIFY = "notify"

@dataclass
class ThreatDetection:
    """Individual threat detection event"""
    detection_id: str
    title: str
    description: str
    severity: ThreatSeverity
    detection_type: DetectionType
    confidence_score: float
    raw_data: Dict[str, Any]
    affected_entities: List[str]
    indicators_of_compromise: List[str]
    mitre_techniques: List[str]
    threat_actor_attribution: Optional[str]
    detection_time: datetime
    source_system: str
    rule_id: Optional[str]
    false_positive_probability: float

@dataclass
class SecurityIncident:
    """Security incident management"""
    incident_id: str
    title: str
    description: str
    severity: ThreatSeverity
    status: IncidentStatus
    initial_detection: ThreatDetection
    related_detections: List[ThreatDetection]
    affected_systems: List[str]
    attack_timeline: List[Dict[str, Any]]
    evidence_collected: List[str]
    containment_actions: List[str]
    eradication_actions: List[str]
    recovery_actions: List[str]
    lessons_learned: List[str]
    assigned_analyst: str
    created_time: datetime
    updated_time: datetime
    closed_time: Optional[datetime]

@dataclass
class ThreatHunt:
    """Proactive threat hunting session"""
    hunt_id: str
    hunt_name: str
    hypothesis: str
    hunt_query: str
    data_sources: List[str]
    time_range: Dict[str, datetime]
    findings: List[ThreatDetection]
    hunt_status: str
    analyst: str
    created_time: datetime
    completed_time: Optional[datetime]

@dataclass
class ForensicEvidence:
    """Digital forensic evidence"""
    evidence_id: str
    evidence_type: str  # file, memory, network, registry, log
    source_system: str
    collection_method: str
    file_path: Optional[str]
    file_hash: Optional[str]
    metadata: Dict[str, Any]
    chain_of_custody: List[str]
    analysis_results: Dict[str, Any]
    collection_time: datetime

class AdvancedThreatHuntingSystem(SecurityService):
    """
    Advanced Threat Hunting and Incident Response System

    Capabilities:
    - Proactive threat hunting with custom query language
    - Machine learning anomaly detection
    - Automated incident response workflows
    - Digital forensics and evidence collection
    - Advanced behavioral analysis
    - Threat intelligence correlation
    - Custom detection rule development
    - Real-time threat monitoring
    """

    def __init__(self, **kwargs):
        super().__init__(
            service_id="advanced_threat_hunting",
            dependencies=["database", "redis", "elasticsearch", "ml_engine"],
            config=kwargs.get("config", {})
        )

        # Threat hunting state
        self.active_hunts: Dict[str, ThreatHunt] = {}
        self.hunt_history: List[ThreatHunt] = []
        self.detection_rules: Dict[str, Any] = {}

        # Incident management
        self.active_incidents: Dict[str, SecurityIncident] = {}
        self.incident_history: List[SecurityIncident] = []
        self.response_playbooks: Dict[str, Any] = {}

        # Detection and analytics
        self.detection_models: Dict[str, Any] = {}
        self.behavioral_baselines: Dict[str, Any] = {}
        self.anomaly_detectors: Dict[str, Any] = {}

        # Forensics and evidence
        self.evidence_vault: Dict[str, ForensicEvidence] = {}
        self.forensic_tools: Dict[str, str] = {}
        self.evidence_processors: Dict[str, Callable] = {}

        # Threat intelligence
        self.threat_intel_feeds: Dict[str, Any] = {}
        self.ioc_database: Dict[str, Any] = {}
        self.threat_actor_profiles: Dict[str, Any] = {}

        # Real-time monitoring
        self.monitoring_channels: Dict[str, Any] = {}
        self.alert_queues: Dict[str, queue.Queue] = {}
        self.correlation_engine: Optional[Any] = None

        # Advanced capabilities
        self.capabilities = {
            "custom_hunt_language": True,
            "ml_anomaly_detection": ANALYTICS_AVAILABLE,
            "graph_analysis": GRAPH_ANALYSIS_AVAILABLE,
            "nlp_analysis": NLP_AVAILABLE,
            "automated_response": True,
            "forensic_collection": True,
            "behavioral_analytics": True,
            "threat_intel_correlation": True,
            "timeline_reconstruction": True,
            "attack_path_analysis": True
        }

        # Configuration
        self.config = {
            "max_concurrent_hunts": 10,
            "anomaly_threshold": 0.7,
            "correlation_window_hours": 24,
            "evidence_retention_days": 90,
            "auto_response_enabled": True,
            "ml_model_update_frequency": "daily"
        }

        # Thread pools for concurrent operations
        self.hunt_executor = ThreadPoolExecutor(max_workers=5)
        self.analysis_executor = ThreadPoolExecutor(max_workers=8)
        self.response_executor = ThreadPoolExecutor(max_workers=3)

    async def initialize(self) -> bool:
        """Initialize the advanced threat hunting system"""
        try:
            logger.info("Initializing Advanced Threat Hunting System...")

            # Initialize detection models
            await self._initialize_detection_models()

            # Load detection rules
            await self._load_detection_rules()

            # Set up response playbooks
            await self._initialize_response_playbooks()

            # Initialize forensic tools
            await self._initialize_forensic_tools()

            # Set up threat intelligence feeds
            await self._initialize_threat_intelligence()

            # Start real-time monitoring
            await self._start_real_time_monitoring()

            # Initialize correlation engine
            await self._initialize_correlation_engine()

            logger.info("Advanced Threat Hunting System initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize threat hunting system: {e}")
            return False

    async def execute_threat_hunt(self, hunt_config: Dict[str, Any]) -> ThreatHunt:
        """Execute a proactive threat hunt"""
        try:
            hunt_id = f"hunt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            # Create hunt object
            threat_hunt = ThreatHunt(
                hunt_id=hunt_id,
                hunt_name=hunt_config.get("hunt_name", f"Hunt {hunt_id}"),
                hypothesis=hunt_config.get("hypothesis", ""),
                hunt_query=hunt_config.get("hunt_query", ""),
                data_sources=hunt_config.get("data_sources", []),
                time_range={
                    "start": hunt_config.get("start_time", datetime.now() - timedelta(hours=24)),
                    "end": hunt_config.get("end_time", datetime.now())
                },
                findings=[],
                hunt_status="running",
                analyst=hunt_config.get("analyst", "system"),
                created_time=datetime.now(),
                completed_time=None
            )

            # Add to active hunts
            self.active_hunts[hunt_id] = threat_hunt

            # Execute hunt query
            hunt_results = await self._execute_hunt_query(threat_hunt)

            # Analyze results for threats
            findings = await self._analyze_hunt_results(hunt_results, threat_hunt)
            threat_hunt.findings = findings

            # Apply machine learning analysis
            if ANALYTICS_AVAILABLE:
                ml_findings = await self._apply_ml_analysis_to_hunt(hunt_results, threat_hunt)
                threat_hunt.findings.extend(ml_findings)

            # Correlate with threat intelligence
            intel_correlations = await self._correlate_with_threat_intel(threat_hunt.findings)
            await self._enrich_findings_with_intel(threat_hunt.findings, intel_correlations)

            # Generate hunt report
            hunt_report = await self._generate_hunt_report(threat_hunt)

            # Update hunt status
            threat_hunt.hunt_status = "completed"
            threat_hunt.completed_time = datetime.now()

            # Move to history
            self.hunt_history.append(threat_hunt)
            del self.active_hunts[hunt_id]

            logger.info(f"Threat hunt {hunt_id} completed: {len(threat_hunt.findings)} findings")
            return threat_hunt

        except Exception as e:
            logger.error(f"Threat hunt execution failed: {e}")
            raise

    async def create_custom_detection_rule(self, rule_config: Dict[str, Any]) -> str:
        """Create a custom detection rule"""
        try:
            rule_id = f"rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            # Validate rule configuration
            validated_config = await self._validate_rule_config(rule_config)

            # Create rule structure
            detection_rule = {
                "rule_id": rule_id,
                "rule_name": validated_config["rule_name"],
                "description": validated_config["description"],
                "severity": validated_config["severity"],
                "detection_logic": validated_config["detection_logic"],
                "data_sources": validated_config["data_sources"],
                "mitre_techniques": validated_config.get("mitre_techniques", []),
                "false_positive_filters": validated_config.get("false_positive_filters", []),
                "response_actions": validated_config.get("response_actions", []),
                "enabled": validated_config.get("enabled", True),
                "created_time": datetime.now(),
                "created_by": validated_config.get("analyst", "system"),
                "last_modified": datetime.now()
            }

            # Compile rule for performance
            compiled_rule = await self._compile_detection_rule(detection_rule)

            # Test rule
            test_results = await self._test_detection_rule(compiled_rule)
            if not test_results["valid"]:
                raise ValueError(f"Rule validation failed: {test_results['errors']}")

            # Store rule
            self.detection_rules[rule_id] = compiled_rule

            # Deploy rule to monitoring systems
            await self._deploy_detection_rule(compiled_rule)

            logger.info(f"Custom detection rule {rule_id} created and deployed")
            return rule_id

        except Exception as e:
            logger.error(f"Custom detection rule creation failed: {e}")
            raise

    async def automated_incident_response(self, detection: ThreatDetection) -> SecurityIncident:
        """Automated incident response workflow"""
        try:
            incident_id = f"incident_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            # Create security incident
            incident = SecurityIncident(
                incident_id=incident_id,
                title=f"Security Incident: {detection.title}",
                description=detection.description,
                severity=detection.severity,
                status=IncidentStatus.NEW,
                initial_detection=detection,
                related_detections=[detection],
                affected_systems=detection.affected_entities,
                attack_timeline=[],
                evidence_collected=[],
                containment_actions=[],
                eradication_actions=[],
                recovery_actions=[],
                lessons_learned=[],
                assigned_analyst="auto_responder",
                created_time=datetime.now(),
                updated_time=datetime.now(),
                closed_time=None
            )

            # Add to active incidents
            self.active_incidents[incident_id] = incident

            # Determine response playbook
            playbook = await self._select_response_playbook(detection, incident)

            # Execute containment actions
            if self.config["auto_response_enabled"]:
                containment_results = await self._execute_containment_actions(incident, playbook)
                incident.containment_actions = containment_results
                incident.status = IncidentStatus.CONTAINED

            # Collect forensic evidence
            evidence_collection_results = await self._automated_evidence_collection(incident)
            incident.evidence_collected = evidence_collection_results

            # Reconstruct attack timeline
            timeline = await self._reconstruct_attack_timeline(incident)
            incident.attack_timeline = timeline

            # Correlate with other incidents
            correlations = await self._correlate_with_existing_incidents(incident)
            if correlations:
                await self._merge_related_incidents(incident, correlations)

            # Analyze attack patterns
            attack_analysis = await self._analyze_attack_patterns(incident)

            # Generate threat intelligence
            threat_intel = await self._generate_threat_intelligence_from_incident(incident)

            # Update threat actor attribution
            attribution_analysis = await self._analyze_threat_actor_attribution(incident)

            # Determine eradication actions
            eradication_actions = await self._determine_eradication_actions(incident, attack_analysis)
            incident.eradication_actions = eradication_actions

            # Execute eradication if authorized
            if playbook.get("auto_eradicate", False):
                await self._execute_eradication_actions(incident, eradication_actions)
                incident.status = IncidentStatus.ERADICATED

            # Generate incident report
            incident_report = await self._generate_incident_report(incident)

            # Update incident
            incident.updated_time = datetime.now()

            logger.info(f"Automated incident response completed for {incident_id}")
            return incident

        except Exception as e:
            logger.error(f"Automated incident response failed: {e}")
            raise

    async def behavioral_anomaly_detection(self, entity_data: Dict[str, Any]) -> List[ThreatDetection]:
        """Advanced behavioral anomaly detection"""
        try:
            detections = []

            if not ANALYTICS_AVAILABLE:
                logger.warning("Analytics libraries not available for behavioral analysis")
                return detections

            # Extract behavioral features
            behavioral_features = await self._extract_behavioral_features(entity_data)

            # User Entity Behavior Analytics (UEBA)
            if entity_data.get("entity_type") == "user":
                user_anomalies = await self._detect_user_behavioral_anomalies(behavioral_features)
                detections.extend(user_anomalies)

            # Network behavior analysis
            if entity_data.get("entity_type") == "network":
                network_anomalies = await self._detect_network_behavioral_anomalies(behavioral_features)
                detections.extend(network_anomalies)

            # System behavior analysis
            if entity_data.get("entity_type") == "system":
                system_anomalies = await self._detect_system_behavioral_anomalies(behavioral_features)
                detections.extend(system_anomalies)

            # Application behavior analysis
            if entity_data.get("entity_type") == "application":
                app_anomalies = await self._detect_application_behavioral_anomalies(behavioral_features)
                detections.extend(app_anomalies)

            # Cross-entity correlation analysis
            correlation_anomalies = await self._detect_cross_entity_anomalies(
                entity_data, behavioral_features
            )
            detections.extend(correlation_anomalies)

            # Machine learning anomaly detection
            ml_anomalies = await self._ml_anomaly_detection(behavioral_features, entity_data)
            detections.extend(ml_anomalies)

            # Filter false positives
            filtered_detections = await self._filter_behavioral_false_positives(detections)

            logger.info(f"Behavioral anomaly detection completed: {len(filtered_detections)} anomalies detected")
            return filtered_detections

        except Exception as e:
            logger.error(f"Behavioral anomaly detection failed: {e}")
            return []

    async def forensic_evidence_collection(self, target_systems: List[str],
                                         collection_type: str = "comprehensive") -> List[ForensicEvidence]:
        """Automated forensic evidence collection"""
        try:
            evidence_list = []

            for system in target_systems:
                # System information collection
                system_evidence = await self._collect_system_evidence(system, collection_type)
                evidence_list.extend(system_evidence)

                # Memory acquisition
                if collection_type in ["comprehensive", "memory"]:
                    memory_evidence = await self._collect_memory_evidence(system)
                    evidence_list.extend(memory_evidence)

                # File system artifacts
                if collection_type in ["comprehensive", "filesystem"]:
                    fs_evidence = await self._collect_filesystem_evidence(system)
                    evidence_list.extend(fs_evidence)

                # Network artifacts
                if collection_type in ["comprehensive", "network"]:
                    network_evidence = await self._collect_network_evidence(system)
                    evidence_list.extend(network_evidence)

                # Registry analysis (Windows)
                if collection_type in ["comprehensive", "registry"]:
                    registry_evidence = await self._collect_registry_evidence(system)
                    evidence_list.extend(registry_evidence)

                # Log collection
                if collection_type in ["comprehensive", "logs"]:
                    log_evidence = await self._collect_log_evidence(system)
                    evidence_list.extend(log_evidence)

            # Store evidence in vault
            for evidence in evidence_list:
                self.evidence_vault[evidence.evidence_id] = evidence

            # Generate chain of custody documentation
            await self._generate_chain_of_custody(evidence_list)

            logger.info(f"Forensic evidence collection completed: {len(evidence_list)} artifacts collected")
            return evidence_list

        except Exception as e:
            logger.error(f"Forensic evidence collection failed: {e}")
            return []

    async def threat_intelligence_correlation(self, indicators: List[str]) -> Dict[str, Any]:
        """Advanced threat intelligence correlation"""
        try:
            correlation_results = {
                "correlation_id": f"corr_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "indicators_processed": len(indicators),
                "threat_matches": [],
                "actor_attribution": {},
                "campaign_analysis": {},
                "recommendations": [],
                "confidence_score": 0.0
            }

            # Process each indicator
            for indicator in indicators:
                # Check against IOC database
                ioc_matches = await self._check_ioc_database(indicator)
                if ioc_matches:
                    correlation_results["threat_matches"].extend(ioc_matches)

                # Query external threat intel feeds
                external_matches = await self._query_external_threat_intel(indicator)
                if external_matches:
                    correlation_results["threat_matches"].extend(external_matches)

                # Check against known malware families
                malware_matches = await self._check_malware_signatures(indicator)
                if malware_matches:
                    correlation_results["threat_matches"].extend(malware_matches)

            # Threat actor attribution analysis
            if correlation_results["threat_matches"]:
                attribution = await self._analyze_threat_actor_attribution_advanced(
                    correlation_results["threat_matches"]
                )
                correlation_results["actor_attribution"] = attribution

            # Campaign correlation analysis
            campaign_analysis = await self._analyze_campaign_correlations(
                indicators, correlation_results["threat_matches"]
            )
            correlation_results["campaign_analysis"] = campaign_analysis

            # Generate actionable recommendations
            recommendations = await self._generate_threat_intel_recommendations(
                correlation_results
            )
            correlation_results["recommendations"] = recommendations

            # Calculate overall confidence score
            confidence_score = await self._calculate_threat_intel_confidence(
                correlation_results
            )
            correlation_results["confidence_score"] = confidence_score

            logger.info(f"Threat intelligence correlation completed: {len(correlation_results['threat_matches'])} matches found")
            return correlation_results

        except Exception as e:
            logger.error(f"Threat intelligence correlation failed: {e}")
            return {"error": str(e)}

    async def attack_path_reconstruction(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Reconstruct attack path using graph analysis"""
        try:
            if not GRAPH_ANALYSIS_AVAILABLE:
                logger.warning("Graph analysis not available for attack path reconstruction")
                return {"error": "Graph analysis libraries not available"}

            attack_graph = {
                "graph_id": f"attack_graph_{incident.incident_id}",
                "nodes": [],
                "edges": [],
                "attack_phases": [],
                "critical_paths": [],
                "pivot_points": [],
                "recommendations": []
            }

            # Create attack graph
            G = nx.DiGraph()

            # Add nodes for each affected system and detection
            for system in incident.affected_systems:
                G.add_node(system, node_type="system", compromised=True)
                attack_graph["nodes"].append({
                    "id": system,
                    "type": "system",
                    "status": "compromised"
                })

            # Add nodes for detections
            for detection in incident.related_detections:
                detection_node = f"detection_{detection.detection_id}"
                G.add_node(detection_node, node_type="detection",
                          mitre_techniques=detection.mitre_techniques)
                attack_graph["nodes"].append({
                    "id": detection_node,
                    "type": "detection",
                    "mitre_techniques": detection.mitre_techniques
                })

            # Add edges based on temporal and logical relationships
            edges = await self._identify_attack_relationships(incident)
            for edge in edges:
                G.add_edge(edge["source"], edge["target"],
                          relationship=edge["relationship"],
                          timestamp=edge["timestamp"])
                attack_graph["edges"].append(edge)

            # Identify attack phases using MITRE ATT&CK framework
            attack_phases = await self._identify_attack_phases(G, incident)
            attack_graph["attack_phases"] = attack_phases

            # Find critical attack paths
            critical_paths = await self._find_critical_attack_paths(G)
            attack_graph["critical_paths"] = critical_paths

            # Identify pivot points
            pivot_points = await self._identify_pivot_points(G)
            attack_graph["pivot_points"] = pivot_points

            # Generate defensive recommendations
            recommendations = await self._generate_attack_path_recommendations(G, attack_graph)
            attack_graph["recommendations"] = recommendations

            logger.info(f"Attack path reconstruction completed for {incident.incident_id}")
            return attack_graph

        except Exception as e:
            logger.error(f"Attack path reconstruction failed: {e}")
            return {"error": str(e)}

    # Implementation helper methods
    async def _initialize_detection_models(self):
        """Initialize machine learning detection models"""
        try:
            if not ANALYTICS_AVAILABLE:
                return

            # Anomaly detection models
            self.detection_models["isolation_forest"] = IsolationForest(
                contamination=0.1, random_state=42
            )

            self.detection_models["user_behavior_model"] = DBSCAN(
                eps=0.5, min_samples=5
            )

            # Classification models for threat categorization
            self.detection_models["threat_classifier"] = RandomForestClassifier(
                n_estimators=100, random_state=42
            )

            # Preprocessing
            self.detection_models["scaler"] = StandardScaler()

            logger.info("Detection models initialized")

        except Exception as e:
            logger.error(f"Failed to initialize detection models: {e}")

    async def _execute_hunt_query(self, threat_hunt: ThreatHunt) -> List[Dict[str, Any]]:
        """Execute threat hunting query against data sources"""
        try:
            # This would interface with various data sources like:
            # - Elasticsearch for log analysis
            # - Security information and event management (SIEM) systems
            # - Endpoint detection and response (EDR) systems
            # - Network monitoring systems

            # Placeholder implementation
            hunt_results = []

            # Parse and execute custom hunt query language
            parsed_query = await self._parse_hunt_query(threat_hunt.hunt_query)

            # Execute against each data source
            for data_source in threat_hunt.data_sources:
                source_results = await self._query_data_source(data_source, parsed_query, threat_hunt.time_range)
                hunt_results.extend(source_results)

            return hunt_results

        except Exception as e:
            logger.error(f"Hunt query execution failed: {e}")
            return []

    # Additional helper methods would be implemented here...

    async def health_check(self) -> ServiceHealth:
        """Health check for threat hunting system"""
        try:
            checks = {
                "analytics_available": ANALYTICS_AVAILABLE,
                "graph_analysis_available": GRAPH_ANALYSIS_AVAILABLE,
                "nlp_available": NLP_AVAILABLE,
                "active_hunts": len(self.active_hunts),
                "active_incidents": len(self.active_incidents),
                "detection_rules": len(self.detection_rules),
                "evidence_vault_size": len(self.evidence_vault),
                "threat_intel_feeds": len(self.threat_intel_feeds)
            }

            status = ServiceStatus.HEALTHY if all([
                checks["analytics_available"] or checks["detection_rules"] > 0,
                checks["evidence_vault_size"] >= 0
            ]) else ServiceStatus.DEGRADED

            return ServiceHealth(
                service_id=self.service_id,
                status=status,
                checks=checks,
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            return ServiceHealth(
                service_id=self.service_id,
                status=ServiceStatus.UNHEALTHY,
                checks={"error": str(e)},
                timestamp=datetime.utcnow()
            )

# Export the advanced threat hunting system
__all__ = [
    "AdvancedThreatHuntingSystem",
    "ThreatDetection",
    "SecurityIncident",
    "ThreatHunt",
    "ForensicEvidence",
    "ThreatSeverity",
    "IncidentStatus",
    "DetectionType",
    "ResponseAction"
]
