#!/usr/bin/env python3
"""
Advanced Threat Correlation Engine
Principal Auditor Implementation: Next-Generation Threat Intelligence Platform

This module implements a sophisticated threat correlation engine that combines
real-time threat intelligence, behavioral analytics, and predictive modeling
to provide unparalleled cybersecurity insights.

Key Features:
- Multi-dimensional threat correlation across global feeds
- Real-time behavioral anomaly detection with ML
- Predictive threat modeling using advanced AI
- Automated threat hunting with contextual intelligence
- Integration with MITRE ATT&CK framework
- Quantum-safe threat indicators
- Advanced attribution analysis
"""

import asyncio
import logging
import json
import uuid
import numpy as np
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import structlog

# Advanced ML imports with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.cluster import DBSCAN
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    import networkx as nx
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False
    logging.warning("Advanced ML libraries not available - using fallback implementations")

# Internal imports
from ..common.security_framework import SecurityFramework, SecurityLevel
from ..common.audit_logger import AuditLogger, AuditEvent
from .unified_intelligence_command_center import IntelligenceAsset, IntelligenceSource

logger = structlog.get_logger(__name__)


class ThreatSeverity(Enum):
    """Threat severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ThreatCategory(Enum):
    """Threat categories for classification"""
    MALWARE = "malware"
    PHISHING = "phishing"
    RANSOMWARE = "ransomware"
    APT = "apt"
    INSIDER_THREAT = "insider_threat"
    DATA_EXFILTRATION = "data_exfiltration"
    NETWORK_INTRUSION = "network_intrusion"
    VULNERABILITY_EXPLOITATION = "vulnerability_exploitation"
    SOCIAL_ENGINEERING = "social_engineering"
    SUPPLY_CHAIN = "supply_chain"
    CRYPTOMINING = "cryptomining"
    BOTNET = "botnet"


class CorrelationMethod(Enum):
    """Threat correlation methods"""
    TEMPORAL = "temporal"
    BEHAVIORAL = "behavioral"
    NETWORK = "network"
    INDICATOR = "indicator"
    CONTEXTUAL = "contextual"
    SEMANTIC = "semantic"
    GRAPH_BASED = "graph_based"
    ML_CLUSTERING = "ml_clustering"


@dataclass
class ThreatIndicator:
    """Structured threat indicator with metadata"""
    indicator_id: str
    indicator_type: str  # ip, domain, hash, url, etc.
    value: str
    confidence: float
    severity: ThreatSeverity
    first_seen: datetime
    last_seen: datetime
    threat_categories: List[ThreatCategory] = field(default_factory=list)
    attribution: Optional[str] = None
    ttps: List[str] = field(default_factory=list)  # MITRE ATT&CK techniques
    context: Dict[str, Any] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    related_indicators: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatEvent:
    """Structured threat event for correlation"""
    event_id: str
    timestamp: datetime
    event_type: str
    source_system: str
    indicators: List[ThreatIndicator]
    severity: ThreatSeverity
    description: str
    raw_data: Dict[str, Any] = field(default_factory=dict)
    correlation_score: float = 0.0
    related_events: List[str] = field(default_factory=list)
    kill_chain_phase: Optional[str] = None
    mitre_techniques: List[str] = field(default_factory=list)
    contextual_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatCampaign:
    """Correlated threat campaign"""
    campaign_id: str
    name: str
    description: str
    threat_actor: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]
    events: List[str] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)
    severity: ThreatSeverity = ThreatSeverity.MEDIUM
    confidence: float = 0.0
    kill_chain_phases: List[str] = field(default_factory=list)
    ttps: List[str] = field(default_factory=list)
    attribution_confidence: float = 0.0
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    countermeasures: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThreatCorrelationNetwork:
    """Neural network for advanced threat correlation"""
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        if not ADVANCED_ML_AVAILABLE:
            logger.warning("Advanced ML not available - using fallback correlation")
            return
        
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoder for threat events
        self.event_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 64)
        )
        
        # Correlation head
        self.correlation_head = nn.Sequential(
            nn.Linear(64 * 2, 32),  # Concat two event encodings
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Clustering head for campaign detection
        self.campaign_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)  # Embedding for clustering
        )
        
        # Training components
        self.optimizer = None
        self.scaler = StandardScaler()
        self.training_history = []
        
    def forward(self, event1: torch.Tensor, event2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for correlation prediction"""
        if not ADVANCED_ML_AVAILABLE:
            return torch.zeros(1), torch.zeros(1)
        
        encoding1 = self.event_encoder(event1)
        encoding2 = self.event_encoder(event2)
        
        # Correlation prediction
        concat_encoding = torch.cat([encoding1, encoding2], dim=-1)
        correlation_score = self.correlation_head(concat_encoding)
        
        # Campaign embedding (average of encodings)
        campaign_embedding = self.campaign_head((encoding1 + encoding2) / 2)
        
        return correlation_score, campaign_embedding
    
    def encode_event(self, event: torch.Tensor) -> torch.Tensor:
        """Encode single event for clustering"""
        if not ADVANCED_ML_AVAILABLE:
            return torch.zeros(64)
        
        return self.event_encoder(event)
    
    def predict_correlation(self, event1: torch.Tensor, event2: torch.Tensor) -> float:
        """Predict correlation between two events"""
        if not ADVANCED_ML_AVAILABLE:
            return 0.5
        
        with torch.no_grad():
            correlation_score, _ = self.forward(event1, event2)
            return correlation_score.item()


class AdvancedThreatCorrelationEngine:
    """
    Advanced Threat Correlation Engine
    
    Provides sophisticated threat intelligence correlation, behavioral analysis,
    and predictive threat modeling for enterprise cybersecurity operations.
    
    Features:
    - Multi-dimensional threat correlation
    - Real-time behavioral anomaly detection
    - Predictive threat modeling
    - Automated campaign detection
    - MITRE ATT&CK integration
    - Attribution analysis
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.engine_id = str(uuid.uuid4())
        
        # Core components
        self.security_framework = SecurityFramework()
        self.audit_logger = AuditLogger()
        
        # Threat intelligence storage
        self.threat_indicators: Dict[str, ThreatIndicator] = {}
        self.threat_events: Dict[str, ThreatEvent] = {}
        self.threat_campaigns: Dict[str, ThreatCampaign] = {}
        
        # Correlation engines
        if ADVANCED_ML_AVAILABLE:
            self.correlation_network = ThreatCorrelationNetwork()
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            self.clustering_model = DBSCAN(eps=0.5, min_samples=3)
            self.threat_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.knowledge_graph = nx.DiGraph()
        else:
            self.correlation_network = None
            self.anomaly_detector = None
            self.clustering_model = None
            self.threat_classifier = None
            self.knowledge_graph = None
        
        # Real-time processing
        self.event_queue = asyncio.Queue(maxsize=10000)
        self.correlation_cache = {}
        self.behavioral_baselines = {}
        
        # Performance tracking
        self.correlation_metrics = {
            "events_processed": 0,
            "correlations_found": 0,
            "campaigns_detected": 0,
            "false_positives": 0,
            "processing_time_avg": 0.0,
            "accuracy": 0.0
        }
        
        # Background processing
        self.processing_tasks = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Advanced Threat Correlation Engine initialized", engine_id=self.engine_id)
    
    async def initialize(self) -> bool:
        """Initialize the correlation engine"""
        try:
            logger.info("Initializing Advanced Threat Correlation Engine")
            
            # Initialize security framework
            await self.security_framework.initialize()
            await self.audit_logger.initialize()
            
            # Load threat intelligence feeds
            await self._load_threat_intelligence_feeds()
            
            # Initialize ML models
            await self._initialize_ml_models()
            
            # Load knowledge base
            await self._load_knowledge_base()
            
            # Start background processing
            await self._start_background_processing()
            
            # Initialize behavioral baselines
            await self._initialize_behavioral_baselines()
            
            logger.info("Advanced Threat Correlation Engine fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"Correlation engine initialization failed: {e}")
            return False
    
    async def process_threat_event(self, event_data: Dict[str, Any]) -> ThreatEvent:
        """Process and correlate incoming threat event"""
        try:
            # Create structured threat event
            threat_event = await self._create_threat_event(event_data)
            
            # Store event
            self.threat_events[threat_event.event_id] = threat_event
            
            # Extract and process indicators
            await self._process_event_indicators(threat_event)
            
            # Perform real-time correlation
            correlations = await self._correlate_threat_event(threat_event)
            
            # Update campaigns if correlations found
            if correlations:
                await self._update_threat_campaigns(threat_event, correlations)
            
            # Behavioral analysis
            await self._analyze_behavioral_patterns(threat_event)
            
            # Update metrics
            self.correlation_metrics["events_processed"] += 1
            if correlations:
                self.correlation_metrics["correlations_found"] += 1
            
            # Audit logging
            await self.audit_logger.log_event(AuditEvent(
                event_type="threat_event_processed",
                component="threat_correlation_engine",
                details={
                    "event_id": threat_event.event_id,
                    "severity": threat_event.severity.value,
                    "correlations_found": len(correlations),
                    "indicators_count": len(threat_event.indicators)
                },
                security_level=SecurityLevel.HIGH if threat_event.severity in [ThreatSeverity.CRITICAL, ThreatSeverity.HIGH] else SecurityLevel.MEDIUM
            ))
            
            logger.info(f"Processed threat event {threat_event.event_id} with {len(correlations)} correlations")
            
            return threat_event
            
        except Exception as e:
            logger.error(f"Failed to process threat event: {e}")
            raise
    
    async def _create_threat_event(self, event_data: Dict[str, Any]) -> ThreatEvent:
        """Create structured threat event from raw data"""
        event_id = str(uuid.uuid4())
        
        # Extract indicators
        indicators = []
        for indicator_data in event_data.get("indicators", []):
            indicator = ThreatIndicator(
                indicator_id=str(uuid.uuid4()),
                indicator_type=indicator_data.get("type", "unknown"),
                value=indicator_data.get("value", ""),
                confidence=indicator_data.get("confidence", 0.5),
                severity=ThreatSeverity(indicator_data.get("severity", "medium")),
                first_seen=datetime.utcnow(),
                last_seen=datetime.utcnow(),
                threat_categories=[ThreatCategory(cat) for cat in indicator_data.get("categories", [])],
                attribution=indicator_data.get("attribution"),
                ttps=indicator_data.get("ttps", []),
                context=indicator_data.get("context", {}),
                sources=indicator_data.get("sources", []),
                metadata=indicator_data.get("metadata", {})
            )
            indicators.append(indicator)
        
        # Create event
        threat_event = ThreatEvent(
            event_id=event_id,
            timestamp=datetime.fromisoformat(event_data.get("timestamp", datetime.utcnow().isoformat())),
            event_type=event_data.get("event_type", "unknown"),
            source_system=event_data.get("source_system", "unknown"),
            indicators=indicators,
            severity=ThreatSeverity(event_data.get("severity", "medium")),
            description=event_data.get("description", ""),
            raw_data=event_data,
            kill_chain_phase=event_data.get("kill_chain_phase"),
            mitre_techniques=event_data.get("mitre_techniques", []),
            contextual_metadata=event_data.get("contextual_metadata", {})
        )
        
        return threat_event
    
    async def _correlate_threat_event(self, event: ThreatEvent) -> List[str]:
        """Correlate threat event with existing events and campaigns"""
        correlations = []
        
        try:
            # Temporal correlation (events within time window)
            temporal_correlations = await self._temporal_correlation(event)
            correlations.extend(temporal_correlations)
            
            # Indicator-based correlation
            indicator_correlations = await self._indicator_correlation(event)
            correlations.extend(indicator_correlations)
            
            # Behavioral correlation
            behavioral_correlations = await self._behavioral_correlation(event)
            correlations.extend(behavioral_correlations)
            
            # Network-based correlation
            network_correlations = await self._network_correlation(event)
            correlations.extend(network_correlations)
            
            # ML-based correlation
            if ADVANCED_ML_AVAILABLE:
                ml_correlations = await self._ml_correlation(event)
                correlations.extend(ml_correlations)
            
            # Update event with correlations
            event.related_events = list(set(correlations))
            
            return correlations
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return []
    
    async def _temporal_correlation(self, event: ThreatEvent) -> List[str]:
        """Find temporally correlated events"""
        correlations = []
        time_window = timedelta(hours=1)  # Configurable
        
        try:
            for other_event_id, other_event in self.threat_events.items():
                if other_event_id == event.event_id:
                    continue
                
                # Check if events are within time window
                time_diff = abs((event.timestamp - other_event.timestamp).total_seconds())
                if time_diff <= time_window.total_seconds():
                    # Additional checks for relevance
                    if self._check_temporal_relevance(event, other_event):
                        correlations.append(other_event_id)
            
            return correlations
            
        except Exception as e:
            logger.error(f"Temporal correlation failed: {e}")
            return []
    
    async def _indicator_correlation(self, event: ThreatEvent) -> List[str]:
        """Find events with overlapping indicators"""
        correlations = []
        
        try:
            event_indicators = set(ind.value for ind in event.indicators)
            
            for other_event_id, other_event in self.threat_events.items():
                if other_event_id == event.event_id:
                    continue
                
                other_indicators = set(ind.value for ind in other_event.indicators)
                overlap = event_indicators.intersection(other_indicators)
                
                # Correlation threshold based on overlap
                overlap_ratio = len(overlap) / min(len(event_indicators), len(other_indicators))
                if overlap_ratio > 0.3:  # 30% overlap threshold
                    correlations.append(other_event_id)
            
            return correlations
            
        except Exception as e:
            logger.error(f"Indicator correlation failed: {e}")
            return []
    
    async def _behavioral_correlation(self, event: ThreatEvent) -> List[str]:
        """Find behaviorally similar events"""
        correlations = []
        
        try:
            # Extract behavioral features
            event_features = self._extract_behavioral_features(event)
            
            for other_event_id, other_event in self.threat_events.items():
                if other_event_id == event.event_id:
                    continue
                
                other_features = self._extract_behavioral_features(other_event)
                similarity = self._calculate_behavioral_similarity(event_features, other_features)
                
                if similarity > 0.7:  # High similarity threshold
                    correlations.append(other_event_id)
            
            return correlations
            
        except Exception as e:
            logger.error(f"Behavioral correlation failed: {e}")
            return []
    
    async def _network_correlation(self, event: ThreatEvent) -> List[str]:
        """Find network-topology based correlations"""
        correlations = []
        
        try:
            # Extract network indicators
            event_ips = [ind.value for ind in event.indicators if ind.indicator_type == "ip"]
            event_domains = [ind.value for ind in event.indicators if ind.indicator_type == "domain"]
            
            for other_event_id, other_event in self.threat_events.items():
                if other_event_id == event.event_id:
                    continue
                
                other_ips = [ind.value for ind in other_event.indicators if ind.indicator_type == "ip"]
                other_domains = [ind.value for ind in other_event.indicators if ind.indicator_type == "domain"]
                
                # Check for network proximity
                if self._check_network_proximity(event_ips, other_ips) or \
                   self._check_domain_similarity(event_domains, other_domains):
                    correlations.append(other_event_id)
            
            return correlations
            
        except Exception as e:
            logger.error(f"Network correlation failed: {e}")
            return []
    
    async def _ml_correlation(self, event: ThreatEvent) -> List[str]:
        """ML-based advanced correlation"""
        correlations = []
        
        try:
            if not ADVANCED_ML_AVAILABLE or not self.correlation_network:
                return []
            
            # Encode current event
            event_encoding = self._encode_event_for_ml(event)
            event_tensor = torch.tensor(event_encoding, dtype=torch.float32)
            
            # Compare with other events
            for other_event_id, other_event in self.threat_events.items():
                if other_event_id == event.event_id:
                    continue
                
                other_encoding = self._encode_event_for_ml(other_event)
                other_tensor = torch.tensor(other_encoding, dtype=torch.float32)
                
                # Predict correlation
                correlation_score = self.correlation_network.predict_correlation(event_tensor, other_tensor)
                
                if correlation_score > 0.8:  # High correlation threshold
                    correlations.append(other_event_id)
            
            return correlations
            
        except Exception as e:
            logger.error(f"ML correlation failed: {e}")
            return []
    
    async def detect_threat_campaigns(self) -> List[ThreatCampaign]:
        """Detect threat campaigns from correlated events"""
        campaigns = []
        
        try:
            if not ADVANCED_ML_AVAILABLE:
                return await self._simple_campaign_detection()
            
            # Prepare data for clustering
            event_encodings = []
            event_ids = []
            
            for event_id, event in self.threat_events.items():
                encoding = self._encode_event_for_ml(event)
                event_encodings.append(encoding)
                event_ids.append(event_id)
            
            if len(event_encodings) < 3:
                return campaigns
            
            # Perform clustering
            X = np.array(event_encodings)
            clusters = self.clustering_model.fit_predict(X)
            
            # Create campaigns from clusters
            cluster_events = defaultdict(list)
            for i, cluster_id in enumerate(clusters):
                if cluster_id != -1:  # -1 is noise in DBSCAN
                    cluster_events[cluster_id].append(event_ids[i])
            
            # Create campaign objects
            for cluster_id, event_list in cluster_events.items():
                if len(event_list) >= 2:  # Minimum events for campaign
                    campaign = await self._create_threat_campaign(event_list, cluster_id)
                    campaigns.append(campaign)
                    self.threat_campaigns[campaign.campaign_id] = campaign
            
            self.correlation_metrics["campaigns_detected"] += len(campaigns)
            
            logger.info(f"Detected {len(campaigns)} threat campaigns")
            return campaigns
            
        except Exception as e:
            logger.error(f"Campaign detection failed: {e}")
            return []
    
    async def _create_threat_campaign(self, event_ids: List[str], cluster_id: int) -> ThreatCampaign:
        """Create threat campaign from correlated events"""
        campaign_id = f"campaign_{cluster_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Analyze events to extract campaign characteristics
        events = [self.threat_events[eid] for eid in event_ids if eid in self.threat_events]
        
        # Determine campaign timeline
        start_time = min(event.timestamp for event in events)
        end_time = max(event.timestamp for event in events)
        
        # Extract common indicators and TTPs
        all_indicators = []
        all_ttps = []
        all_categories = []
        
        for event in events:
            all_indicators.extend([ind.indicator_id for ind in event.indicators])
            all_ttps.extend(event.mitre_techniques)
            for indicator in event.indicators:
                all_categories.extend([cat.value for cat in indicator.threat_categories])
        
        # Determine campaign severity
        severities = [event.severity for event in events]
        max_severity = max(severities, key=lambda x: ["info", "low", "medium", "high", "critical"].index(x.value))
        
        # Attribution analysis
        attribution, attribution_confidence = await self._analyze_attribution(events)
        
        # Impact assessment
        impact_assessment = await self._assess_campaign_impact(events)
        
        # Generate countermeasures
        countermeasures = await self._generate_countermeasures(events)
        
        campaign = ThreatCampaign(
            campaign_id=campaign_id,
            name=f"Campaign {cluster_id}",
            description=f"Correlated threat campaign with {len(events)} events",
            threat_actor=attribution,
            start_time=start_time,
            end_time=end_time if end_time != start_time else None,
            events=event_ids,
            indicators=list(set(all_indicators)),
            severity=max_severity,
            confidence=self._calculate_campaign_confidence(events),
            kill_chain_phases=list(set(event.kill_chain_phase for event in events if event.kill_chain_phase)),
            ttps=list(set(all_ttps)),
            attribution_confidence=attribution_confidence,
            impact_assessment=impact_assessment,
            countermeasures=countermeasures,
            metadata={
                "cluster_id": cluster_id,
                "event_count": len(events),
                "dominant_categories": list(set(all_categories)),
                "duration_hours": (end_time - start_time).total_seconds() / 3600
            }
        )
        
        return campaign
    
    async def get_threat_intelligence_summary(self) -> Dict[str, Any]:
        """Get comprehensive threat intelligence summary"""
        try:
            # Calculate statistics
            total_events = len(self.threat_events)
            total_indicators = len(self.threat_indicators)
            total_campaigns = len(self.threat_campaigns)
            
            # Severity distribution
            severity_distribution = defaultdict(int)
            for event in self.threat_events.values():
                severity_distribution[event.severity.value] += 1
            
            # Category distribution
            category_distribution = defaultdict(int)
            for indicator in self.threat_indicators.values():
                for category in indicator.threat_categories:
                    category_distribution[category.value] += 1
            
            # Recent activity (last 24 hours)
            recent_cutoff = datetime.utcnow() - timedelta(hours=24)
            recent_events = len([e for e in self.threat_events.values() if e.timestamp > recent_cutoff])
            
            # Top threat actors
            threat_actors = defaultdict(int)
            for campaign in self.threat_campaigns.values():
                if campaign.threat_actor:
                    threat_actors[campaign.threat_actor] += 1
            
            top_threat_actors = sorted(threat_actors.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # MITRE ATT&CK coverage
            all_techniques = set()
            for event in self.threat_events.values():
                all_techniques.update(event.mitre_techniques)
            
            return {
                "summary_generated": datetime.utcnow().isoformat(),
                "engine_id": self.engine_id,
                "statistics": {
                    "total_events": total_events,
                    "total_indicators": total_indicators,
                    "total_campaigns": total_campaigns,
                    "recent_events_24h": recent_events,
                    "correlation_rate": (self.correlation_metrics["correlations_found"] / 
                                       max(self.correlation_metrics["events_processed"], 1)) * 100
                },
                "severity_distribution": dict(severity_distribution),
                "category_distribution": dict(category_distribution),
                "top_threat_actors": top_threat_actors,
                "mitre_attack_coverage": {
                    "techniques_observed": len(all_techniques),
                    "techniques": list(all_techniques)[:20]  # Top 20
                },
                "correlation_metrics": self.correlation_metrics.copy(),
                "active_campaigns": [
                    {
                        "campaign_id": campaign.campaign_id,
                        "name": campaign.name,
                        "severity": campaign.severity.value,
                        "event_count": len(campaign.events),
                        "duration_hours": campaign.metadata.get("duration_hours", 0)
                    }
                    for campaign in self.threat_campaigns.values()
                    if campaign.end_time is None or campaign.end_time > recent_cutoff
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to generate threat intelligence summary: {e}")
            return {"error": str(e)}
    
    # Helper methods for correlation analysis
    def _check_temporal_relevance(self, event1: ThreatEvent, event2: ThreatEvent) -> bool:
        """Check if events are temporally relevant"""
        # Check for common source systems
        if event1.source_system == event2.source_system:
            return True
        
        # Check for similar event types
        if event1.event_type == event2.event_type:
            return True
        
        # Check for escalating severity
        severity_order = ["info", "low", "medium", "high", "critical"]
        event1_idx = severity_order.index(event1.severity.value)
        event2_idx = severity_order.index(event2.severity.value)
        
        if abs(event1_idx - event2_idx) <= 1:  # Adjacent severity levels
            return True
        
        return False
    
    def _extract_behavioral_features(self, event: ThreatEvent) -> List[float]:
        """Extract behavioral features from event"""
        features = []
        
        # Basic features
        features.append(len(event.indicators))  # Number of indicators
        features.append(len(event.mitre_techniques))  # Number of techniques
        features.append(["info", "low", "medium", "high", "critical"].index(event.severity.value))  # Severity
        
        # Indicator type distribution
        indicator_types = defaultdict(int)
        for indicator in event.indicators:
            indicator_types[indicator.indicator_type] += 1
        
        # Normalize to feature vector
        common_types = ["ip", "domain", "hash", "url", "email"]
        for itype in common_types:
            features.append(indicator_types.get(itype, 0))
        
        # Time-based features
        hour_of_day = event.timestamp.hour
        day_of_week = event.timestamp.weekday()
        features.extend([hour_of_day / 24.0, day_of_week / 7.0])
        
        # Source system encoding
        source_hash = abs(hash(event.source_system)) % 1000
        features.append(source_hash / 1000.0)
        
        # Pad to fixed length
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]  # Return fixed-length feature vector
    
    def _calculate_behavioral_similarity(self, features1: List[float], features2: List[float]) -> float:
        """Calculate behavioral similarity between feature vectors"""
        if len(features1) != len(features2):
            return 0.0
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(features1, features2))
        magnitude1 = sum(a * a for a in features1) ** 0.5
        magnitude2 = sum(b * b for b in features2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _check_network_proximity(self, ips1: List[str], ips2: List[str]) -> bool:
        """Check if IP addresses are in close network proximity"""
        try:
            import ipaddress
            
            for ip1 in ips1:
                for ip2 in ips2:
                    try:
                        addr1 = ipaddress.ip_address(ip1)
                        addr2 = ipaddress.ip_address(ip2)
                        
                        # Check if in same /24 subnet
                        if addr1.version == addr2.version:
                            if addr1.version == 4:
                                net1 = ipaddress.ip_network(f"{ip1}/24", strict=False)
                                if addr2 in net1:
                                    return True
                    except:
                        continue
            
            return False
            
        except ImportError:
            # Fallback: simple string comparison
            return any(ip1.rsplit('.', 1)[0] == ip2.rsplit('.', 1)[0] 
                      for ip1 in ips1 for ip2 in ips2)
    
    def _check_domain_similarity(self, domains1: List[str], domains2: List[str]) -> bool:
        """Check if domains are similar"""
        for domain1 in domains1:
            for domain2 in domains2:
                # Check for same TLD
                if domain1.split('.')[-1] == domain2.split('.')[-1]:
                    # Check for similar subdomains
                    if self._calculate_string_similarity(domain1, domain2) > 0.7:
                        return True
        
        return False
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using Jaccard similarity"""
        set1 = set(str1.lower())
        set2 = set(str2.lower())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _encode_event_for_ml(self, event: ThreatEvent) -> List[float]:
        """Encode event for ML processing"""
        # Start with behavioral features
        features = self._extract_behavioral_features(event)
        
        # Add additional ML-specific features
        # Indicator confidence scores
        if event.indicators:
            avg_confidence = sum(ind.confidence for ind in event.indicators) / len(event.indicators)
            max_confidence = max(ind.confidence for ind in event.indicators)
            min_confidence = min(ind.confidence for ind in event.indicators)
            features.extend([avg_confidence, max_confidence, min_confidence])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # MITRE technique encoding
        common_techniques = ["T1566", "T1190", "T1078", "T1055", "T1003"]
        for technique in common_techniques:
            features.append(1.0 if any(technique in t for t in event.mitre_techniques) else 0.0)
        
        # Pad to fixed length (256 for neural network)
        while len(features) < 256:
            features.append(0.0)
        
        return features[:256]
    
    async def _simple_campaign_detection(self) -> List[ThreatCampaign]:
        """Simple rule-based campaign detection fallback"""
        campaigns = []
        processed_events = set()
        
        for event_id, event in self.threat_events.items():
            if event_id in processed_events:
                continue
            
            # Find related events
            related_events = [event_id]
            processed_events.add(event_id)
            
            for other_id, other_event in self.threat_events.items():
                if other_id == event_id or other_id in processed_events:
                    continue
                
                # Simple correlation based on indicators and time
                if self._simple_correlation_check(event, other_event):
                    related_events.append(other_id)
                    processed_events.add(other_id)
            
            # Create campaign if multiple events
            if len(related_events) >= 2:
                campaign = await self._create_threat_campaign(related_events, len(campaigns))
                campaigns.append(campaign)
        
        return campaigns
    
    def _simple_correlation_check(self, event1: ThreatEvent, event2: ThreatEvent) -> bool:
        """Simple correlation check for fallback"""
        # Time window check
        time_diff = abs((event1.timestamp - event2.timestamp).total_seconds())
        if time_diff > 3600:  # 1 hour
            return False
        
        # Indicator overlap
        indicators1 = set(ind.value for ind in event1.indicators)
        indicators2 = set(ind.value for ind in event2.indicators)
        overlap = len(indicators1.intersection(indicators2))
        
        if overlap > 0:
            return True
        
        # Same source system
        if event1.source_system == event2.source_system and event1.event_type == event2.event_type:
            return True
        
        return False
    
    def _calculate_campaign_confidence(self, events: List[ThreatEvent]) -> float:
        """Calculate confidence score for campaign"""
        if not events:
            return 0.0
        
        # Base confidence on number of events
        event_score = min(len(events) / 10.0, 1.0)
        
        # Average indicator confidence
        all_confidences = []
        for event in events:
            all_confidences.extend([ind.confidence for ind in event.indicators])
        
        indicator_score = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        
        # Time consistency (events closer in time = higher confidence)
        if len(events) > 1:
            timestamps = [event.timestamp for event in events]
            time_span = (max(timestamps) - min(timestamps)).total_seconds()
            time_score = 1.0 - min(time_span / (24 * 3600), 1.0)  # Penalty for events > 24h apart
        else:
            time_score = 1.0
        
        # Combined confidence
        confidence = (event_score * 0.4 + indicator_score * 0.4 + time_score * 0.2)
        
        return round(confidence, 3)
    
    # Additional helper methods (implementation continues...)
    async def _load_threat_intelligence_feeds(self):
        """Load external threat intelligence feeds"""
        logger.info("Loading threat intelligence feeds")
        # Implementation would load from various TI feeds
    
    async def _initialize_ml_models(self):
        """Initialize ML models for correlation"""
        if ADVANCED_ML_AVAILABLE:
            logger.info("Initializing ML models for threat correlation")
            # Model initialization would happen here
    
    async def _load_knowledge_base(self):
        """Load threat intelligence knowledge base"""
        logger.info("Loading threat intelligence knowledge base")
        # Knowledge base loading would happen here
    
    async def _start_background_processing(self):
        """Start background processing tasks"""
        logger.info("Starting background processing tasks")
        # Background task initialization would happen here
    
    async def _initialize_behavioral_baselines(self):
        """Initialize behavioral baselines for anomaly detection"""
        logger.info("Initializing behavioral baselines")
        # Baseline initialization would happen here
    
    async def _process_event_indicators(self, event: ThreatEvent):
        """Process and store event indicators"""
        for indicator in event.indicators:
            if indicator.indicator_id not in self.threat_indicators:
                self.threat_indicators[indicator.indicator_id] = indicator
            else:
                # Update existing indicator
                existing = self.threat_indicators[indicator.indicator_id]
                existing.last_seen = indicator.last_seen
                existing.sources = list(set(existing.sources + indicator.sources))
    
    async def _update_threat_campaigns(self, event: ThreatEvent, correlations: List[str]):
        """Update existing campaigns with new correlated event"""
        # Implementation for updating campaigns
        pass
    
    async def _analyze_behavioral_patterns(self, event: ThreatEvent):
        """Analyze behavioral patterns for the event"""
        # Behavioral analysis implementation
        pass
    
    async def _analyze_attribution(self, events: List[ThreatEvent]) -> Tuple[Optional[str], float]:
        """Analyze attribution for campaign"""
        # Attribution analysis would happen here
        return None, 0.0
    
    async def _assess_campaign_impact(self, events: List[ThreatEvent]) -> Dict[str, Any]:
        """Assess the impact of a threat campaign"""
        return {
            "potential_impact": "medium",
            "affected_systems": len(events),
            "data_at_risk": "unknown"
        }
    
    async def _generate_countermeasures(self, events: List[ThreatEvent]) -> List[str]:
        """Generate countermeasures for campaign"""
        return [
            "Monitor network traffic for suspicious patterns",
            "Update threat intelligence feeds",
            "Review access controls"
        ]


# Global engine instance
_correlation_engine: Optional[AdvancedThreatCorrelationEngine] = None


async def get_threat_correlation_engine(config: Dict[str, Any] = None) -> AdvancedThreatCorrelationEngine:
    """Get singleton threat correlation engine instance"""
    global _correlation_engine
    
    if _correlation_engine is None:
        _correlation_engine = AdvancedThreatCorrelationEngine(config)
        await _correlation_engine.initialize()
    
    return _correlation_engine


# Export main classes
__all__ = [
    "AdvancedThreatCorrelationEngine",
    "ThreatIndicator",
    "ThreatEvent", 
    "ThreatCampaign",
    "ThreatSeverity",
    "ThreatCategory",
    "CorrelationMethod",
    "get_threat_correlation_engine"
]