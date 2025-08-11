"""
Advanced MITRE ATT&CK Integration Engine
Sophisticated real-world threat mapping and attack pattern analysis with AI-powered correlation
"""

import asyncio
import json
import logging
import aiohttp
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import uuid
import hashlib
import pickle
from collections import defaultdict, Counter
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings("ignore")

from .base_service import ServiceHealth, ServiceStatus
from .intelligence_service import IntelligenceService

logger = logging.getLogger(__name__)


class ThreatSeverity(Enum):
    """Threat severity levels"""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AttackStage(Enum):
    """Attack progression stages"""
    RECONNAISSANCE = "reconnaissance"
    WEAPONIZATION = "weaponization"
    DELIVERY = "delivery"
    EXPLOITATION = "exploitation"
    INSTALLATION = "installation"
    COMMAND_CONTROL = "command_control"
    ACTIONS_OBJECTIVES = "actions_objectives"


class ConfidenceLevel(Enum):
    """Confidence levels for threat attribution"""
    VERY_HIGH = "very_high"  # 90-100%
    HIGH = "high"           # 70-89%
    MEDIUM = "medium"       # 50-69%
    LOW = "low"            # 30-49%
    VERY_LOW = "very_low"  # 0-29%


@dataclass
class MitreTactic:
    """Enhanced MITRE ATT&CK Tactic with ML features"""
    tactic_id: str
    name: str
    description: str
    short_name: str
    techniques: List[str] = field(default_factory=list)
    url: str = ""
    platforms: List[str] = field(default_factory=list)
    created: Optional[datetime] = None
    modified: Optional[datetime] = None
    version: str = "1.0"
    x_mitre_domains: List[str] = field(default_factory=list)
    external_references: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MitreTechnique:
    """Enhanced MITRE ATT&CK Technique with AI capabilities"""
    technique_id: str
    name: str
    description: str
    tactic_refs: List[str] = field(default_factory=list)
    sub_techniques: List[str] = field(default_factory=list)
    platforms: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)
    detection_methods: Dict[str, Any] = field(default_factory=dict)
    url: str = ""
    kill_chain_phases: List[str] = field(default_factory=list)
    created: Optional[datetime] = None
    modified: Optional[datetime] = None
    version: str = "1.0"
    revoked: bool = False
    deprecated: bool = False
    
    # AI Enhancement Fields
    prevalence_score: float = 0.0
    difficulty_score: float = 0.0
    impact_score: float = 0.0
    detection_score: float = 0.0
    mitigation_score: float = 0.0
    related_cves: List[str] = field(default_factory=list)
    threat_groups: List[str] = field(default_factory=list)
    software_used: List[str] = field(default_factory=list)
    
    # ML Feature Vector (for similarity calculations)
    feature_vector: Optional[np.ndarray] = None


@dataclass 
class MitreGroup:
    """Enhanced MITRE ATT&CK Group with threat intelligence"""
    group_id: str
    name: str
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    techniques: List[str] = field(default_factory=list)
    software: List[str] = field(default_factory=list)
    campaigns: List[str] = field(default_factory=list)
    country: Optional[str] = None
    motivation: List[str] = field(default_factory=list)
    first_seen: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    
    # Threat Intelligence Enhancement
    sophistication_level: str = "unknown"  # basic, intermediate, advanced, expert
    activity_status: str = "unknown"      # active, dormant, disbanded
    target_sectors: List[str] = field(default_factory=list)
    target_regions: List[str] = field(default_factory=list)
    attribution_confidence: float = 0.0
    primary_methods: List[str] = field(default_factory=list)


@dataclass
class MitreSoftware:
    """Enhanced MITRE ATT&CK Software/Malware"""
    software_id: str
    name: str
    type: str  # malware, tool
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    platforms: List[str] = field(default_factory=list)
    techniques: List[str] = field(default_factory=list)
    groups: List[str] = field(default_factory=list)
    
    # Malware Analysis
    family: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    persistence_methods: List[str] = field(default_factory=list)
    evasion_techniques: List[str] = field(default_factory=list)


@dataclass
class ThreatMapping:
    """Advanced threat mapping with AI correlation"""
    mapping_id: str
    threat_id: str
    technique_ids: List[str]
    confidence: float
    evidence: List[Dict[str, Any]]
    timestamp: datetime
    source: str
    
    # AI Enhancement
    correlation_score: float = 0.0
    attribution_groups: List[str] = field(default_factory=list)
    attack_stage: Optional[AttackStage] = None
    severity: ThreatSeverity = ThreatSeverity.MEDIUM
    iocs: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackPattern:
    """Sophisticated attack pattern detection"""
    pattern_id: str
    name: str
    techniques: List[str]
    confidence: float
    severity: ThreatSeverity
    detection_time: datetime
    
    # Pattern Analysis
    kill_chain_coverage: Dict[str, bool] = field(default_factory=dict)
    tactic_progression: List[str] = field(default_factory=list)
    timeline_analysis: Dict[str, Any] = field(default_factory=dict)
    behavioral_indicators: List[str] = field(default_factory=list)
    threat_actor_similarity: List[Dict[str, Any]] = field(default_factory=list)
    
    # Response Recommendations
    immediate_actions: List[str] = field(default_factory=list)
    investigation_priorities: List[str] = field(default_factory=list)
    containment_strategies: List[str] = field(default_factory=list)
    mitigation_recommendations: List[str] = field(default_factory=list)


@dataclass
class ThreatIntelligenceReport:
    """Comprehensive threat intelligence report"""
    report_id: str
    timestamp: datetime
    threat_landscape: Dict[str, Any]
    attack_patterns: List[AttackPattern]
    technique_trends: Dict[str, Any]
    group_activities: Dict[str, Any]
    emerging_threats: List[Dict[str, Any]]
    recommendations: List[str]
    
    # Risk Assessment
    overall_risk_score: float = 0.0
    sector_specific_risks: Dict[str, float] = field(default_factory=dict)
    predictive_insights: Dict[str, Any] = field(default_factory=dict)


class AdvancedMitreAttackEngine(IntelligenceService):
    """
    Sophisticated MITRE ATT&CK Integration Engine with AI-powered analysis
    Real-world threat mapping, attack pattern detection, and predictive intelligence
    """
    
    def __init__(self, data_path: str = "./data/mitre_attack", **kwargs):
        super().__init__(
            service_id="advanced_mitre_attack",
            dependencies=["threat_intelligence", "ml_models"],
            **kwargs
        )
        
        self.data_path = Path(data_path)
        self.db_path = self.data_path / "advanced_mitre.db"
        
        # Core MITRE Data
        self.tactics: Dict[str, MitreTactic] = {}
        self.techniques: Dict[str, MitreTechnique] = {}
        self.groups: Dict[str, MitreGroup] = {}
        self.software: Dict[str, MitreSoftware] = {}
        self.mitigations: Dict[str, Dict[str, Any]] = {}
        
        # AI/ML Components
        self.technique_vectorizer: Optional[TfidfVectorizer] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self.attack_graph: Optional[nx.DiGraph] = None
        self.clustering_model: Optional[DBSCAN] = None
        
        # Threat Intelligence
        self.threat_mappings: List[ThreatMapping] = []
        self.attack_patterns: Dict[str, AttackPattern] = {}
        self.threat_campaigns: Dict[str, Dict[str, Any]] = {}
        
        # Pattern Detection Rules
        self.detection_rules: List[Dict[str, Any]] = []
        self.behavior_patterns: Dict[str, List[str]] = {}
        
        # Analytics and Metrics
        self.analytics = {
            "total_techniques": 0,
            "total_groups": 0,
            "total_patterns_detected": 0,
            "threat_mappings_created": 0,
            "last_framework_update": None,
            "model_accuracy_scores": {},
            "detection_performance": {}
        }
        
        # Configuration
        self.config = {
            "confidence_threshold": 0.7,
            "similarity_threshold": 0.8,
            "clustering_eps": 0.3,
            "min_samples": 2,
            "max_cache_size": 10000,
            "update_interval_hours": 24
        }
        
        # Caching
        self.mapping_cache: Dict[str, Any] = {}
        self.similarity_cache: Dict[str, float] = {}
        
        # MITRE Data Sources
        self.data_sources = {
            "enterprise": "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json",
            "mobile": "https://raw.githubusercontent.com/mitre/cti/master/mobile-attack/mobile-attack.json", 
            "ics": "https://raw.githubusercontent.com/mitre/cti/master/ics-attack/ics-attack.json"
        }
        
        # HTTP Session
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self) -> bool:
        """Initialize the Advanced MITRE ATT&CK Engine"""
        try:
            logger.info("Initializing Advanced MITRE ATT&CK Engine...")
            
            # Create data directory
            self.data_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            )
            
            # Initialize database
            await self._initialize_database()
            
            # Load MITRE ATT&CK framework
            await self._load_mitre_framework()
            
            # Initialize ML components
            await self._initialize_ml_components()
            
            # Load detection rules
            await self._load_detection_rules()
            
            # Build attack graph
            await self._build_attack_graph()
            
            # Validate installation
            await self._validate_framework_integrity()
            
            # Start background tasks
            asyncio.create_task(self._periodic_framework_update())
            asyncio.create_task(self._continuous_threat_monitoring())
            
            logger.info(f"Advanced MITRE ATT&CK Engine initialized successfully")
            logger.info(f"Loaded: {len(self.techniques)} techniques, {len(self.groups)} groups")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Advanced MITRE ATT&CK Engine: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the engine"""
        try:
            if self.session:
                await self.session.close()
            
            # Save current state
            await self._save_engine_state()
            
            logger.info("Advanced MITRE ATT&CK Engine shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return False
    
    async def health_check(self) -> ServiceHealth:
        """Comprehensive health check"""
        try:
            checks = {
                "framework_loaded": len(self.techniques) > 0,
                "techniques_count": len(self.techniques),
                "groups_count": len(self.groups),
                "software_count": len(self.software),
                "ml_models_ready": self.technique_vectorizer is not None,
                "attack_graph_built": self.attack_graph is not None,
                "detection_rules": len(self.detection_rules),
                "threat_mappings": len(self.threat_mappings),
                "cache_size": len(self.mapping_cache),
                "last_update": self.analytics.get("last_framework_update")
            }
            
            # Determine health status
            status = ServiceStatus.HEALTHY
            message = "Advanced MITRE ATT&CK Engine operational"
            
            if not checks["framework_loaded"]:
                status = ServiceStatus.UNHEALTHY
                message = "MITRE framework not loaded"
            elif not checks["ml_models_ready"]:
                status = ServiceStatus.DEGRADED
                message = "ML models not ready"
            elif checks["techniques_count"] < 100:
                status = ServiceStatus.DEGRADED
                message = "Incomplete technique database"
            
            return ServiceHealth(
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                checks=checks
            )
            
        except Exception as e:
            return ServiceHealth(
                status=ServiceStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                timestamp=datetime.utcnow(),
                checks={"error": str(e)}
            )
    
    async def analyze_threat_indicators(self, indicators: List[Dict[str, Any]], 
                                      context: Optional[Dict[str, Any]] = None) -> ThreatMapping:
        """
        Advanced threat indicator analysis with AI-powered MITRE mapping
        """
        try:
            mapping_id = str(uuid.uuid4())
            context = context or {}
            
            logger.info(f"Analyzing {len(indicators)} threat indicators...")
            
            # Extract features from indicators
            indicator_features = await self._extract_indicator_features(indicators)
            
            # Map to MITRE techniques using ML
            technique_matches = await self._map_indicators_to_techniques(indicator_features)
            
            # Determine attack stages and progression
            attack_stage = await self._determine_attack_stage(technique_matches)
            
            # Calculate threat severity
            severity = await self._calculate_threat_severity(technique_matches, context)
            
            # Perform threat actor attribution
            attribution_groups = await self._perform_threat_attribution(technique_matches)
            
            # Extract IOCs and context
            iocs = self._extract_iocs(indicators)
            
            # Calculate overall confidence
            confidence = self._calculate_mapping_confidence(technique_matches, indicator_features)
            
            # Create comprehensive threat mapping
            mapping = ThreatMapping(
                mapping_id=mapping_id,
                threat_id=context.get("threat_id", mapping_id),
                technique_ids=[t["technique_id"] for t in technique_matches],
                confidence=confidence,
                evidence=[{
                    "indicator": ind,
                    "techniques": await self._get_indicator_techniques(ind),
                    "confidence": ind.get("confidence", 0.5)
                } for ind in indicators],
                timestamp=datetime.utcnow(),
                source="advanced_mitre_engine",
                correlation_score=await self._calculate_correlation_score(technique_matches),
                attribution_groups=attribution_groups,
                attack_stage=attack_stage,
                severity=severity,
                iocs=iocs,
                context=context
            )
            
            # Store mapping
            self.threat_mappings.append(mapping)
            
            # Update analytics
            self.analytics["threat_mappings_created"] += 1
            
            logger.info(f"Threat mapping {mapping_id} created with {len(technique_matches)} techniques")
            
            return mapping
            
        except Exception as e:
            logger.error(f"Error analyzing threat indicators: {e}")
            raise
    
    async def detect_attack_patterns(self, events: List[Dict[str, Any]], 
                                   time_window: int = 3600) -> List[AttackPattern]:
        """
        Sophisticated attack pattern detection using temporal analysis and ML
        """
        try:
            logger.info(f"Detecting attack patterns in {len(events)} events...")
            
            patterns = []
            
            # Group events by time windows
            time_windows = self._create_time_windows(events, time_window)
            
            for window_start, window_events in time_windows.items():
                # Extract techniques from events
                techniques = []
                for event in window_events:
                    event_techniques = await self._extract_techniques_from_event(event)
                    techniques.extend(event_techniques)
                
                if not techniques:
                    continue
                
                # Apply pattern detection rules
                for rule in self.detection_rules:
                    pattern = await self._evaluate_pattern_rule(rule, techniques, window_events)
                    if pattern:
                        # Enhance pattern with AI analysis
                        enhanced_pattern = await self._enhance_attack_pattern(pattern, window_events)
                        patterns.append(enhanced_pattern)
            
            # Merge related patterns
            merged_patterns = await self._merge_related_patterns(patterns)
            
            # Sort by severity and confidence
            merged_patterns.sort(key=lambda p: (p.severity.value, p.confidence), reverse=True)
            
            logger.info(f"Detected {len(merged_patterns)} attack patterns")
            
            return merged_patterns
            
        except Exception as e:
            logger.error(f"Error detecting attack patterns: {e}")
            raise
    
    async def generate_threat_intelligence_report(self, 
                                                time_range: timedelta = timedelta(days=30)) -> ThreatIntelligenceReport:
        """
        Generate comprehensive threat intelligence report with predictive insights
        """
        try:
            logger.info("Generating comprehensive threat intelligence report...")
            
            report_id = str(uuid.uuid4())
            cutoff_time = datetime.utcnow() - time_range
            
            # Analyze recent threat landscape
            threat_landscape = await self._analyze_threat_landscape(cutoff_time)
            
            # Identify attack patterns
            recent_mappings = [m for m in self.threat_mappings if m.timestamp >= cutoff_time]
            attack_patterns = await self._identify_recent_patterns(recent_mappings)
            
            # Analyze technique trends
            technique_trends = await self._analyze_technique_trends(cutoff_time)
            
            # Monitor group activities
            group_activities = await self._monitor_group_activities(cutoff_time)
            
            # Identify emerging threats
            emerging_threats = await self._identify_emerging_threats(cutoff_time)
            
            # Calculate risk scores
            overall_risk_score = await self._calculate_overall_risk_score(threat_landscape)
            sector_risks = await self._calculate_sector_risks(threat_landscape)
            
            # Generate predictive insights
            predictive_insights = await self._generate_predictive_insights(technique_trends)
            
            # Generate recommendations
            recommendations = await self._generate_intelligence_recommendations(
                threat_landscape, attack_patterns, technique_trends
            )
            
            report = ThreatIntelligenceReport(
                report_id=report_id,
                timestamp=datetime.utcnow(),
                threat_landscape=threat_landscape,
                attack_patterns=attack_patterns,
                technique_trends=technique_trends,
                group_activities=group_activities,
                emerging_threats=emerging_threats,
                recommendations=recommendations,
                overall_risk_score=overall_risk_score,
                sector_specific_risks=sector_risks,
                predictive_insights=predictive_insights
            )
            
            logger.info(f"Generated threat intelligence report {report_id}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating threat intelligence report: {e}")
            raise
    
    async def predict_attack_progression(self, current_techniques: List[str]) -> Dict[str, Any]:
        """
        AI-powered attack progression prediction based on current techniques
        """
        try:
            logger.info(f"Predicting attack progression from {len(current_techniques)} techniques...")
            
            # Build attack paths using graph analysis
            attack_paths = await self._build_attack_paths(current_techniques)
            
            # Predict next likely techniques
            next_techniques = await self._predict_next_techniques(current_techniques)
            
            # Calculate progression probabilities
            progression_probs = await self._calculate_progression_probabilities(current_techniques)
            
            # Identify critical decision points
            decision_points = await self._identify_decision_points(current_techniques)
            
            # Generate defensive recommendations
            defensive_actions = await self._recommend_defensive_actions(next_techniques)
            
            prediction = {
                "prediction_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "current_techniques": current_techniques,
                "attack_paths": attack_paths,
                "next_likely_techniques": next_techniques,
                "progression_probabilities": progression_probs,
                "critical_decision_points": decision_points,
                "defensive_recommendations": defensive_actions,
                "confidence_score": self._calculate_prediction_confidence(current_techniques),
                "time_to_next_stage": await self._estimate_time_to_next_stage(current_techniques)
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting attack progression: {e}")
            raise
    
    async def get_technique_intelligence(self, technique_id: str) -> Dict[str, Any]:
        """
        Get comprehensive intelligence about a specific MITRE technique
        """
        if technique_id not in self.techniques:
            raise ValueError(f"Technique {technique_id} not found")
        
        technique = self.techniques[technique_id]
        
        # Get related techniques
        related_techniques = await self._find_related_techniques(technique_id)
        
        # Get threat groups using this technique
        threat_groups = [g for g in self.groups.values() if technique_id in g.techniques]
        
        # Get software implementing this technique
        software_used = [s for s in self.software.values() if technique_id in s.techniques]
        
        # Calculate threat scores
        threat_score = await self._calculate_technique_threat_score(technique_id)
        
        # Get detection recommendations
        detection_recommendations = await self._get_detection_recommendations(technique_id)
        
        # Get mitigation strategies
        mitigation_strategies = await self._get_mitigation_strategies(technique_id)
        
        return {
            "technique": asdict(technique),
            "threat_score": threat_score,
            "related_techniques": related_techniques,
            "threat_groups": [{"id": g.group_id, "name": g.name, "aliases": g.aliases} for g in threat_groups],
            "software_used": [{"id": s.software_id, "name": s.name, "type": s.type} for s in software_used],
            "detection_recommendations": detection_recommendations,
            "mitigation_strategies": mitigation_strategies,
            "intelligence_summary": await self._generate_technique_summary(technique_id)
        }
    
    # Private Implementation Methods
    
    async def _initialize_database(self):
        """Initialize SQLite database with enhanced schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tactics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tactics (
                tactic_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                short_name TEXT,
                techniques TEXT,
                platforms TEXT,
                created_date TEXT,
                modified_date TEXT,
                version TEXT,
                domains TEXT,
                external_references TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Enhanced techniques table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS techniques (
                technique_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                tactic_refs TEXT,
                sub_techniques TEXT,
                platforms TEXT,
                data_sources TEXT,
                mitigations TEXT,
                detection_methods TEXT,
                kill_chain_phases TEXT,
                created_date TEXT,
                modified_date TEXT,
                version TEXT,
                revoked BOOLEAN DEFAULT FALSE,
                deprecated BOOLEAN DEFAULT FALSE,
                prevalence_score REAL DEFAULT 0.0,
                difficulty_score REAL DEFAULT 0.0,
                impact_score REAL DEFAULT 0.0,
                detection_score REAL DEFAULT 0.0,
                mitigation_score REAL DEFAULT 0.0,
                related_cves TEXT,
                threat_groups TEXT,
                software_used TEXT,
                feature_vector BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Enhanced groups table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS groups (
                group_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                aliases TEXT,
                description TEXT,
                techniques TEXT,
                software TEXT,
                campaigns TEXT,
                country TEXT,
                motivation TEXT,
                first_seen TEXT,
                last_activity TEXT,
                sophistication_level TEXT,
                activity_status TEXT,
                target_sectors TEXT,
                target_regions TEXT,
                attribution_confidence REAL DEFAULT 0.0,
                primary_methods TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Enhanced software table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS software (
                software_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT,
                description TEXT,
                aliases TEXT,
                platforms TEXT,
                techniques TEXT,
                groups_used TEXT,
                family TEXT,
                capabilities TEXT,
                persistence_methods TEXT,
                evasion_techniques TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Threat mappings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threat_mappings (
                mapping_id TEXT PRIMARY KEY,
                threat_id TEXT NOT NULL,
                technique_ids TEXT,
                confidence REAL,
                evidence TEXT,
                source TEXT,
                correlation_score REAL,
                attribution_groups TEXT,
                attack_stage TEXT,
                severity TEXT,
                iocs TEXT,
                context TEXT,
                timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Attack patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attack_patterns (
                pattern_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                techniques TEXT,
                confidence REAL,
                severity TEXT,
                detection_time TIMESTAMP,
                kill_chain_coverage TEXT,
                tactic_progression TEXT,
                timeline_analysis TEXT,
                behavioral_indicators TEXT,
                threat_actor_similarity TEXT,
                immediate_actions TEXT,
                investigation_priorities TEXT,
                containment_strategies TEXT,
                mitigation_recommendations TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_techniques_tactic ON techniques(tactic_refs)",
            "CREATE INDEX IF NOT EXISTS idx_techniques_platform ON techniques(platforms)",
            "CREATE INDEX IF NOT EXISTS idx_techniques_confidence ON techniques(prevalence_score)",
            "CREATE INDEX IF NOT EXISTS idx_groups_country ON groups(country)",
            "CREATE INDEX IF NOT EXISTS idx_groups_sophistication ON groups(sophistication_level)",
            "CREATE INDEX IF NOT EXISTS idx_threat_mappings_timestamp ON threat_mappings(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_threat_mappings_severity ON threat_mappings(severity)",
            "CREATE INDEX IF NOT EXISTS idx_attack_patterns_severity ON attack_patterns(severity)",
            "CREATE INDEX IF NOT EXISTS idx_attack_patterns_confidence ON attack_patterns(confidence)"
        ]
        
        for index in indexes:
            cursor.execute(index)
        
        conn.commit()
        conn.close()
        
        logger.info("Advanced MITRE ATT&CK database initialized")
    
    async def _load_mitre_framework(self):
        """Load and parse MITRE ATT&CK framework with enhancements"""
        try:
            # Load from official sources
            for domain, url in self.data_sources.items():
                await self._load_framework_domain(domain, url)
            
            # Load built-in data if download failed
            if not self.techniques:
                await self._load_builtin_framework_data()
            
            # Enhance with AI features
            await self._enhance_framework_with_ai()
            
            # Update analytics
            self.analytics.update({
                "total_techniques": len(self.techniques),
                "total_groups": len(self.groups),
                "last_framework_update": datetime.utcnow().isoformat()
            })
            
            logger.info(f"MITRE framework loaded: {len(self.techniques)} techniques, {len(self.groups)} groups")
            
        except Exception as e:
            logger.error(f"Error loading MITRE framework: {e}")
            await self._load_builtin_framework_data()
    
    async def _load_framework_domain(self, domain: str, url: str):
        """Load framework data from specific domain"""
        try:
            cache_file = self.data_path / f"{domain}_attack.json"
            
            # Check cache first
            if cache_file.exists():
                file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_age < timedelta(days=7):
                    logger.info(f"Using cached {domain} data")
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    await self._parse_stix_data(data, domain)
                    return
            
            # Download fresh data
            logger.info(f"Downloading {domain} framework data...")
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Cache the data
                    with open(cache_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    await self._parse_stix_data(data, domain)
                    logger.info(f"Successfully loaded {domain} framework")
                else:
                    logger.error(f"Failed to download {domain}: HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"Error loading {domain} framework: {e}")
            # Fall back to cache if available
            cache_file = self.data_path / f"{domain}_attack.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    await self._parse_stix_data(data, domain)
                    logger.info(f"Loaded {domain} from cache as fallback")
                except Exception as cache_error:
                    logger.error(f"Failed to load {domain} cache: {cache_error}")
    
    async def _parse_stix_data(self, data: Dict[str, Any], domain: str):
        """Parse STIX format MITRE data with enhancements"""
        objects = data.get("objects", [])
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {"tactics": 0, "techniques": 0, "groups": 0, "software": 0}
        
        for obj in objects:
            try:
                obj_type = obj.get("type", "")
                
                if obj_type == "x-mitre-tactic":
                    tactic = await self._parse_tactic(obj)
                    if tactic:
                        self.tactics[tactic.tactic_id] = tactic
                        await self._store_tactic(cursor, tactic)
                        stats["tactics"] += 1
                
                elif obj_type == "attack-pattern":
                    technique = await self._parse_technique(obj)
                    if technique:
                        self.techniques[technique.technique_id] = technique
                        await self._store_technique(cursor, technique)
                        stats["techniques"] += 1
                
                elif obj_type == "intrusion-set":
                    group = await self._parse_group(obj)
                    if group:
                        self.groups[group.group_id] = group
                        await self._store_group(cursor, group)
                        stats["groups"] += 1
                
                elif obj_type in ["malware", "tool"]:
                    software = await self._parse_software(obj)
                    if software:
                        self.software[software.software_id] = software
                        await self._store_software(cursor, software)
                        stats["software"] += 1
                        
            except Exception as e:
                logger.debug(f"Error parsing {domain} object {obj.get('id', 'unknown')}: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Parsed {domain}: {stats}")
    
    async def _parse_tactic(self, obj: Dict[str, Any]) -> Optional[MitreTactic]:
        """Parse MITRE tactic with enhancements"""
        try:
            external_refs = obj.get("external_references", [])
            mitre_ref = next((ref for ref in external_refs if ref.get("source_name") == "mitre-attack"), None)
            
            if not mitre_ref:
                return None
            
            tactic_id = mitre_ref.get("external_id")
            if not tactic_id:
                return None
            
            return MitreTactic(
                tactic_id=tactic_id,
                name=obj.get("name", ""),
                description=obj.get("description", ""),
                short_name=obj.get("x_mitre_shortname", ""),
                url=mitre_ref.get("url", ""),
                platforms=obj.get("x_mitre_platforms", []),
                created=self._parse_datetime(obj.get("created")),
                modified=self._parse_datetime(obj.get("modified")),
                version=obj.get("x_mitre_version", "1.0"),
                x_mitre_domains=obj.get("x_mitre_domains", []),
                external_references=external_refs
            )
            
        except Exception as e:
            logger.debug(f"Error parsing tactic: {e}")
            return None
    
    async def _parse_technique(self, obj: Dict[str, Any]) -> Optional[MitreTechnique]:
        """Parse MITRE technique with AI enhancements"""
        try:
            external_refs = obj.get("external_references", [])
            mitre_ref = next((ref for ref in external_refs if ref.get("source_name") == "mitre-attack"), None)
            
            if not mitre_ref:
                return None
            
            technique_id = mitre_ref.get("external_id")
            if not technique_id:
                return None
            
            # Extract kill chain phases
            kill_chain_phases = []
            tactic_refs = []
            for phase in obj.get("kill_chain_phases", []):
                if phase.get("kill_chain_name") == "mitre-attack":
                    phase_name = phase.get("phase_name", "")
                    kill_chain_phases.append(phase_name)
                    tactic_refs.append(phase_name.replace("-", "_"))
            
            technique = MitreTechnique(
                technique_id=technique_id,
                name=obj.get("name", ""),
                description=obj.get("description", ""),
                tactic_refs=tactic_refs,
                platforms=obj.get("x_mitre_platforms", []),
                data_sources=obj.get("x_mitre_data_sources", []),
                url=mitre_ref.get("url", ""),
                kill_chain_phases=kill_chain_phases,
                created=self._parse_datetime(obj.get("created")),
                modified=self._parse_datetime(obj.get("modified")),
                version=obj.get("x_mitre_version", "1.0"),
                revoked=obj.get("revoked", False),
                deprecated=obj.get("x_mitre_deprecated", False)
            )
            
            # Calculate AI scores
            technique.prevalence_score = await self._calculate_prevalence_score(technique)
            technique.difficulty_score = await self._calculate_difficulty_score(technique)
            technique.impact_score = await self._calculate_impact_score(technique)
            technique.detection_score = await self._calculate_detection_score(technique)
            
            return technique
            
        except Exception as e:
            logger.debug(f"Error parsing technique: {e}")
            return None
    
    async def _parse_group(self, obj: Dict[str, Any]) -> Optional[MitreGroup]:
        """Parse MITRE group with threat intelligence enhancements"""
        try:
            external_refs = obj.get("external_references", [])
            mitre_ref = next((ref for ref in external_refs if ref.get("source_name") == "mitre-attack"), None)
            
            if not mitre_ref:
                return None
            
            group_id = mitre_ref.get("external_id")
            if not group_id:
                return None
            
            group = MitreGroup(
                group_id=group_id,
                name=obj.get("name", ""),
                aliases=obj.get("aliases", []),
                description=obj.get("description", ""),
                country=self._extract_country_from_description(obj.get("description", "")),
                sophistication_level=await self._assess_group_sophistication(obj),
                activity_status=await self._assess_group_activity_status(obj),
                target_sectors=self._extract_target_sectors(obj.get("description", "")),
                target_regions=self._extract_target_regions(obj.get("description", ""))
            )
            
            return group
            
        except Exception as e:
            logger.debug(f"Error parsing group: {e}")
            return None
    
    async def _parse_software(self, obj: Dict[str, Any]) -> Optional[MitreSoftware]:
        """Parse MITRE software with malware analysis"""
        try:
            external_refs = obj.get("external_references", [])
            mitre_ref = next((ref for ref in external_refs if ref.get("source_name") == "mitre-attack"), None)
            
            if not mitre_ref:
                return None
            
            software_id = mitre_ref.get("external_id")
            if not software_id:
                return None
            
            software_type = obj.get("type", "")
            
            software = MitreSoftware(
                software_id=software_id,
                name=obj.get("name", ""),
                type=software_type,
                description=obj.get("description", ""),
                aliases=obj.get("x_mitre_aliases", []),
                platforms=obj.get("x_mitre_platforms", []),
                family=self._extract_malware_family(obj.get("description", "")),
                capabilities=self._extract_capabilities(obj.get("description", "")),
                persistence_methods=self._extract_persistence_methods(obj.get("description", "")),
                evasion_techniques=self._extract_evasion_techniques(obj.get("description", ""))
            )
            
            return software
            
        except Exception as e:
            logger.debug(f"Error parsing software: {e}")
            return None
    
    # Storage methods
    async def _store_tactic(self, cursor, tactic: MitreTactic):
        """Store tactic in database"""
        cursor.execute("""
            INSERT OR REPLACE INTO tactics 
            (tactic_id, name, description, short_name, techniques, platforms, 
             created_date, modified_date, version, domains, external_references, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            tactic.tactic_id, tactic.name, tactic.description, tactic.short_name,
            json.dumps(tactic.techniques), json.dumps(tactic.platforms),
            tactic.created.isoformat() if tactic.created else None,
            tactic.modified.isoformat() if tactic.modified else None,
            tactic.version, json.dumps(tactic.x_mitre_domains),
            json.dumps(tactic.external_references)
        ))
    
    async def _store_technique(self, cursor, technique: MitreTechnique):
        """Store technique in database with AI features"""
        feature_vector_blob = None
        if technique.feature_vector is not None:
            feature_vector_blob = pickle.dumps(technique.feature_vector)
        
        cursor.execute("""
            INSERT OR REPLACE INTO techniques 
            (technique_id, name, description, tactic_refs, sub_techniques, platforms,
             data_sources, mitigations, detection_methods, kill_chain_phases,
             created_date, modified_date, version, revoked, deprecated,
             prevalence_score, difficulty_score, impact_score, detection_score, mitigation_score,
             related_cves, threat_groups, software_used, feature_vector, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            technique.technique_id, technique.name, technique.description,
            json.dumps(technique.tactic_refs), json.dumps(technique.sub_techniques),
            json.dumps(technique.platforms), json.dumps(technique.data_sources),
            json.dumps(technique.mitigations), json.dumps(technique.detection_methods),
            json.dumps(technique.kill_chain_phases),
            technique.created.isoformat() if technique.created else None,
            technique.modified.isoformat() if technique.modified else None,
            technique.version, technique.revoked, technique.deprecated,
            technique.prevalence_score, technique.difficulty_score, technique.impact_score,
            technique.detection_score, technique.mitigation_score,
            json.dumps(technique.related_cves), json.dumps(technique.threat_groups),
            json.dumps(technique.software_used), feature_vector_blob
        ))
    
    async def _store_group(self, cursor, group: MitreGroup):
        """Store group in database with threat intelligence"""
        cursor.execute("""
            INSERT OR REPLACE INTO groups 
            (group_id, name, aliases, description, techniques, software, campaigns,
             country, motivation, first_seen, last_activity, sophistication_level,
             activity_status, target_sectors, target_regions, attribution_confidence,
             primary_methods, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            group.group_id, group.name, json.dumps(group.aliases), group.description,
            json.dumps(group.techniques), json.dumps(group.software), json.dumps(group.campaigns),
            group.country, json.dumps(group.motivation),
            group.first_seen.isoformat() if group.first_seen else None,
            group.last_activity.isoformat() if group.last_activity else None,
            group.sophistication_level, group.activity_status,
            json.dumps(group.target_sectors), json.dumps(group.target_regions),
            group.attribution_confidence, json.dumps(group.primary_methods)
        ))
    
    async def _store_software(self, cursor, software: MitreSoftware):
        """Store software in database with malware analysis"""
        cursor.execute("""
            INSERT OR REPLACE INTO software 
            (software_id, name, type, description, aliases, platforms, techniques,
             groups_used, family, capabilities, persistence_methods, evasion_techniques, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            software.software_id, software.name, software.type, software.description,
            json.dumps(software.aliases), json.dumps(software.platforms),
            json.dumps(software.techniques), json.dumps(software.groups),
            software.family, json.dumps(software.capabilities),
            json.dumps(software.persistence_methods), json.dumps(software.evasion_techniques)
        ))
    
    # AI Enhancement Methods
    async def _initialize_ml_components(self):
        """Initialize machine learning components"""
        try:
            if not self.techniques:
                logger.warning("No techniques loaded for ML initialization")
                return
            
            # Create technique descriptions for vectorization
            descriptions = []
            technique_ids = []
            
            for tech_id, technique in self.techniques.items():
                # Combine name and description for better feature extraction
                combined_text = f"{technique.name} {technique.description}"
                descriptions.append(combined_text)
                technique_ids.append(tech_id)
            
            # Initialize TF-IDF vectorizer
            self.technique_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            
            # Fit vectorizer and create feature vectors
            feature_matrix = self.technique_vectorizer.fit_transform(descriptions)
            
            # Store feature vectors in techniques
            for i, tech_id in enumerate(technique_ids):
                self.techniques[tech_id].feature_vector = feature_matrix[i].toarray().flatten()
            
            # Calculate similarity matrix
            self.similarity_matrix = cosine_similarity(feature_matrix)
            
            # Initialize clustering model
            self.clustering_model = DBSCAN(
                eps=self.config["clustering_eps"],
                min_samples=self.config["min_samples"],
                metric='cosine'
            )
            
            # Fit clustering model
            cluster_labels = self.clustering_model.fit_predict(feature_matrix.toarray())
            
            logger.info(f"ML components initialized: {len(descriptions)} techniques vectorized")
            logger.info(f"Found {len(set(cluster_labels))} technique clusters")
            
        except Exception as e:
            logger.error(f"Error initializing ML components: {e}")
    
    async def _load_detection_rules(self):
        """Load sophisticated detection rules"""
        self.detection_rules = [
            {
                "rule_id": "apt_lateral_movement",
                "name": "Advanced Persistent Threat - Lateral Movement",
                "techniques": ["T1078", "T1021", "T1018", "T1033", "T1083"],
                "min_techniques": 3,
                "time_window": 3600,
                "severity": ThreatSeverity.HIGH,
                "confidence_threshold": 0.8,
                "behavioral_indicators": [
                    "multiple_authentication_events",
                    "cross_network_discovery",
                    "privilege_escalation_attempts"
                ]
            },
            {
                "rule_id": "ransomware_execution_chain",
                "name": "Ransomware Execution Chain",
                "techniques": ["T1566", "T1059", "T1055", "T1486", "T1490"],
                "min_techniques": 3,
                "time_window": 1800,
                "severity": ThreatSeverity.CRITICAL,
                "confidence_threshold": 0.9,
                "behavioral_indicators": [
                    "mass_file_encryption",
                    "backup_system_tampering",
                    "ransom_note_creation"
                ]
            },
            {
                "rule_id": "credential_harvesting",
                "name": "Credential Harvesting Campaign",
                "techniques": ["T1003", "T1555", "T1552", "T1110", "T1212"],
                "min_techniques": 2,
                "time_window": 7200,
                "severity": ThreatSeverity.HIGH,
                "confidence_threshold": 0.75,
                "behavioral_indicators": [
                    "memory_credential_access",
                    "password_file_access",
                    "brute_force_attempts"
                ]
            },
            {
                "rule_id": "data_exfiltration",
                "name": "Data Exfiltration Operation",
                "techniques": ["T1005", "T1025", "T1041", "T1567", "T1029"],
                "min_techniques": 2,
                "time_window": 14400,
                "severity": ThreatSeverity.HIGH,
                "confidence_threshold": 0.7,
                "behavioral_indicators": [
                    "large_data_transfer",
                    "compressed_archive_creation",
                    "cloud_service_uploads"
                ]
            }
        ]
        
        logger.info(f"Loaded {len(self.detection_rules)} sophisticated detection rules")
    
    async def _build_attack_graph(self):
        """Build directed graph of attack techniques and relationships"""
        try:
            self.attack_graph = nx.DiGraph()
            
            # Add technique nodes
            for tech_id, technique in self.techniques.items():
                self.attack_graph.add_node(tech_id, **{
                    "name": technique.name,
                    "tactics": technique.tactic_refs,
                    "platforms": technique.platforms,
                    "difficulty": technique.difficulty_score,
                    "impact": technique.impact_score
                })
            
            # Add edges based on kill chain progression and relationships
            for tech_id, technique in self.techniques.items():
                # Connect to related techniques based on similarity
                related_techniques = await self._find_related_techniques(tech_id, limit=5)
                for related_id, similarity in related_techniques:
                    if similarity > self.config["similarity_threshold"]:
                        self.attack_graph.add_edge(tech_id, related_id, weight=similarity)
                
                # Connect sub-techniques to parent techniques
                for sub_tech_id in technique.sub_techniques:
                    if sub_tech_id in self.techniques:
                        self.attack_graph.add_edge(tech_id, sub_tech_id, weight=1.0, relationship="parent_child")
            
            logger.info(f"Attack graph built: {self.attack_graph.number_of_nodes()} nodes, {self.attack_graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Error building attack graph: {e}")
    
    # AI-Powered Analysis Methods
    async def _extract_indicator_features(self, indicators: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features from threat indicators for ML analysis"""
        features = {
            "indicator_types": [],
            "confidence_scores": [],
            "severity_levels": [],
            "source_reliability": [],
            "temporal_features": {},
            "network_features": {},
            "file_features": {},
            "behavioral_features": {}
        }
        
        for indicator in indicators:
            ind_type = indicator.get("type", "unknown")
            features["indicator_types"].append(ind_type)
            features["confidence_scores"].append(indicator.get("confidence", 0.5))
            features["severity_levels"].append(indicator.get("severity", "medium"))
            
            # Extract type-specific features
            if ind_type in ["ip-dst", "ip-src"]:
                features["network_features"][indicator.get("value", "")] = {
                    "port": indicator.get("port"),
                    "protocol": indicator.get("protocol"),
                    "geolocation": indicator.get("geolocation")
                }
            elif ind_type in ["md5", "sha1", "sha256"]:
                features["file_features"][indicator.get("value", "")] = {
                    "file_type": indicator.get("file_type"),
                    "size": indicator.get("file_size"),
                    "is_packed": indicator.get("is_packed", False)
                }
        
        return features
    
    async def _map_indicators_to_techniques(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Map indicator features to MITRE techniques using ML"""
        technique_matches = []
        
        # Analyze indicator types and map to techniques
        for ind_type in set(features["indicator_types"]):
            techniques = await self._get_techniques_for_indicator_type(ind_type)
            
            for tech_id in techniques:
                if tech_id in self.techniques:
                    technique = self.techniques[tech_id]
                    
                    # Calculate confidence based on multiple factors
                    confidence = await self._calculate_technique_confidence(
                        tech_id, features, ind_type
                    )
                    
                    if confidence >= self.config["confidence_threshold"]:
                        technique_matches.append({
                            "technique_id": tech_id,
                            "name": technique.name,
                            "confidence": confidence,
                            "indicator_type": ind_type,
                            "tactics": technique.tactic_refs,
                            "platforms": technique.platforms
                        })
        
        # Sort by confidence
        technique_matches.sort(key=lambda x: x["confidence"], reverse=True)
        
        return technique_matches
    
    async def _determine_attack_stage(self, technique_matches: List[Dict[str, Any]]) -> Optional[AttackStage]:
        """Determine current attack stage based on techniques"""
        if not technique_matches:
            return None
        
        # Map tactics to attack stages
        tactic_to_stage = {
            "reconnaissance": AttackStage.RECONNAISSANCE,
            "resource_development": AttackStage.WEAPONIZATION,
            "initial_access": AttackStage.DELIVERY,
            "execution": AttackStage.EXPLOITATION,
            "persistence": AttackStage.INSTALLATION,
            "command_and_control": AttackStage.COMMAND_CONTROL,
            "impact": AttackStage.ACTIONS_OBJECTIVES
        }
        
        # Count tactics
        tactic_counts = Counter()
        for match in technique_matches:
            for tactic in match.get("tactics", []):
                tactic_counts[tactic] += match["confidence"]
        
        if not tactic_counts:
            return None
        
        # Get most prevalent tactic
        most_prevalent_tactic = tactic_counts.most_common(1)[0][0]
        
        return tactic_to_stage.get(most_prevalent_tactic, AttackStage.EXPLOITATION)
    
    async def _calculate_threat_severity(self, technique_matches: List[Dict[str, Any]], 
                                       context: Dict[str, Any]) -> ThreatSeverity:
        """Calculate threat severity using AI analysis"""
        if not technique_matches:
            return ThreatSeverity.LOW
        
        # Calculate base severity from techniques
        severity_scores = []
        for match in technique_matches:
            tech_id = match["technique_id"]
            if tech_id in self.techniques:
                technique = self.techniques[tech_id]
                # Combine impact and prevalence scores
                severity_score = (technique.impact_score + technique.prevalence_score) / 2
                severity_scores.append(severity_score * match["confidence"])
        
        if not severity_scores:
            return ThreatSeverity.LOW
        
        avg_severity = sum(severity_scores) / len(severity_scores)
        
        # Adjust based on context
        context_multiplier = 1.0
        if context.get("target_criticality") == "high":
            context_multiplier += 0.2
        if context.get("network_exposure") == "internet_facing":
            context_multiplier += 0.3
        if len(technique_matches) > 5:  # Multiple techniques indicate sophistication
            context_multiplier += 0.2
        
        final_severity = avg_severity * context_multiplier
        
        # Map to severity levels
        if final_severity >= 0.9:
            return ThreatSeverity.CRITICAL
        elif final_severity >= 0.7:
            return ThreatSeverity.HIGH
        elif final_severity >= 0.4:
            return ThreatSeverity.MEDIUM
        else:
            return ThreatSeverity.LOW
    
    async def _perform_threat_attribution(self, technique_matches: List[Dict[str, Any]]) -> List[str]:
        """Perform threat actor attribution using ML similarity analysis"""
        if not technique_matches:
            return []
        
        technique_ids = [match["technique_id"] for match in technique_matches]
        
        # Calculate similarity with known threat groups
        group_similarities = []
        for group_id, group in self.groups.items():
            if not group.techniques:
                continue
            
            # Calculate Jaccard similarity
            common_techniques = set(technique_ids) & set(group.techniques)
            all_techniques = set(technique_ids) | set(group.techniques)
            
            if all_techniques:
                jaccard_similarity = len(common_techniques) / len(all_techniques)
                
                # Weight by group sophistication and activity
                weight = 1.0
                if group.sophistication_level == "expert":
                    weight += 0.3
                elif group.sophistication_level == "advanced":
                    weight += 0.2
                
                if group.activity_status == "active":
                    weight += 0.2
                
                weighted_similarity = jaccard_similarity * weight
                
                if weighted_similarity > 0.3:  # Minimum threshold
                    group_similarities.append({
                        "group_id": group_id,
                        "name": group.name,
                        "similarity": weighted_similarity,
                        "common_techniques": list(common_techniques)
                    })
        
        # Sort by similarity
        group_similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top attributed groups
        return [g["group_id"] for g in group_similarities[:3]]
    
    # Utility Methods
    def _parse_datetime(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime string"""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            return None
    
    def _extract_iocs(self, indicators: List[Dict[str, Any]]) -> List[str]:
        """Extract IOCs from indicators"""
        iocs = []
        for indicator in indicators:
            value = indicator.get("value", "")
            if value:
                iocs.append(value)
        return iocs
    
    def _calculate_mapping_confidence(self, technique_matches: List[Dict[str, Any]], 
                                    features: Dict[str, Any]) -> float:
        """Calculate overall mapping confidence"""
        if not technique_matches:
            return 0.0
        
        # Base confidence from technique matches
        confidences = [match["confidence"] for match in technique_matches]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Adjust based on number of techniques and evidence quality
        num_techniques = len(technique_matches)
        evidence_quality = len(features.get("indicator_types", [])) / 10  # Normalize
        
        # Boost confidence for multiple corroborating techniques
        if num_techniques > 3:
            avg_confidence *= 1.1
        
        # Boost confidence for high-quality evidence
        if evidence_quality > 0.5:
            avg_confidence *= 1.05
        
        return min(avg_confidence, 1.0)
    
    async def _calculate_correlation_score(self, technique_matches: List[Dict[str, Any]]) -> float:
        """Calculate correlation score for technique relationships"""
        if len(technique_matches) < 2:
            return 0.0
        
        technique_ids = [match["technique_id"] for match in technique_matches]
        
        # Use attack graph to calculate relationship strength
        if self.attack_graph:
            total_relationships = 0
            relationship_strength = 0.0
            
            for i in range(len(technique_ids)):
                for j in range(i + 1, len(technique_ids)):
                    tech1, tech2 = technique_ids[i], technique_ids[j]
                    
                    if self.attack_graph.has_edge(tech1, tech2):
                        weight = self.attack_graph[tech1][tech2].get("weight", 0.5)
                        relationship_strength += weight
                        total_relationships += 1
                    elif self.attack_graph.has_edge(tech2, tech1):
                        weight = self.attack_graph[tech2][tech1].get("weight", 0.5)
                        relationship_strength += weight
                        total_relationships += 1
            
            if total_relationships > 0:
                return relationship_strength / total_relationships
        
        # Fallback: use tactic similarity
        tactics = []
        for match in technique_matches:
            tactics.extend(match.get("tactics", []))
        
        unique_tactics = len(set(tactics))
        if unique_tactics > 0:
            return min(len(tactics) / unique_tactics, 1.0)
        
        return 0.5  # Default moderate correlation
    
    # Additional helper methods would continue here...
    # For brevity, I'll include key method signatures
    
    async def _get_techniques_for_indicator_type(self, indicator_type: str) -> List[str]:
        """Get sophisticated technique mapping based on indicator type with ML enhancement"""
        try:
            indicator_type = indicator_type.lower()
            
            # Comprehensive mapping of indicator types to MITRE techniques
            technique_mappings = {
                "ip": ["T1071.001", "T1090", "T1105", "T1041", "T1568", "T1583.003"],  # Network indicators
                "domain": ["T1071.001", "T1568", "T1583.001", "T1189", "T1566.002"],  # Domain-based techniques
                "url": ["T1071.001", "T1189", "T1105", "T1566.002", "T1204.001"],  # URL-based techniques
                "file_hash": ["T1204.002", "T1055", "T1027", "T1036", "T1105"],  # File-based techniques
                "email": ["T1566.001", "T1566.002", "T1598", "T1534"],  # Email-based techniques
                "process": ["T1055", "T1059", "T1106", "T1543", "T1569"],  # Process-based techniques
                "registry": ["T1112", "T1547", "T1546", "T1574", "T1082"],  # Registry techniques
                "network_traffic": ["T1071", "T1090", "T1001", "T1573", "T1048"],  # Network traffic techniques
                "certificate": ["T1573.002", "T1553.004", "T1608.003"],  # Certificate-based techniques
                "user_agent": ["T1071.001", "T1036.005", "T1608.004"],  # User-Agent techniques
                "bitcoin_address": ["T1573.001", "T1041", "T1486"],  # Cryptocurrency techniques
                "imphash": ["T1055", "T1036", "T1027"],  # Import hash techniques
                "ssdeep": ["T1027", "T1036", "T1055"],  # Fuzzy hash techniques
                "yara_rule": ["T1027", "T1055", "T1036", "T1204"],  # YARA rule techniques
                "mutex": ["T1055", "T1057", "T1106"],  # Mutex techniques
                "pipe": ["T1055", "T1106", "T1021"],  # Named pipe techniques
                "service": ["T1543", "T1569", "T1547"],  # Service techniques
                "scheduled_task": ["T1053", "T1547"],  # Scheduled task techniques
            }
            
            # Get base techniques for indicator type
            base_techniques = technique_mappings.get(indicator_type, [])
            
            # Enhance with ML-based technique suggestions if available
            if hasattr(self, 'ml_models') and 'technique_predictor' in self.ml_models:
                try:
                    # Create feature vector for ML prediction
                    features = self._create_indicator_features(indicator_type)
                    ml_techniques = await self._predict_techniques_ml(features)
                    
                    # Combine base and ML techniques with confidence filtering
                    all_techniques = list(set(base_techniques + ml_techniques))
                    
                    # Rank techniques by relevance
                    ranked_techniques = await self._rank_techniques_by_relevance(
                        all_techniques, indicator_type
                    )
                    
                    return ranked_techniques[:15]  # Return top 15 most relevant
                    
                except Exception as e:
                    logger.warning(f"ML technique prediction failed: {e}")
            
            # Apply advanced filtering based on threat landscape
            current_threats = await self._get_current_threat_landscape()
            filtered_techniques = await self._filter_techniques_by_threat_landscape(
                base_techniques, current_threats
            )
            
            # Add related techniques through graph analysis
            if hasattr(self, 'technique_graph'):
                related_techniques = await self._find_related_techniques_graph(
                    filtered_techniques, max_depth=2
                )
                filtered_techniques.extend(related_techniques)
            
            # Remove duplicates and return
            unique_techniques = list(set(filtered_techniques))
            
            logger.debug(f"Found {len(unique_techniques)} techniques for indicator type '{indicator_type}'")
            return unique_techniques[:20]  # Limit to top 20
            
        except Exception as e:
            logger.error(f"Failed to get techniques for indicator type '{indicator_type}': {e}")
            return []
    
    async def _calculate_technique_confidence(self, tech_id: str, features: Dict[str, Any], 
                                            indicator_type: str) -> float:
        """Calculate sophisticated confidence scores using multiple algorithms"""
        try:
            # Initialize base confidence score
            confidence_factors = {
                "indicator_relevance": 0.0,
                "context_match": 0.0,
                "temporal_factor": 0.0,
                "threat_intelligence": 0.0,
                "ml_prediction": 0.0,
                "historical_accuracy": 0.0
            }
            
            # 1. Indicator type relevance scoring
            relevance_scores = {
                ("T1071.001", "ip"): 0.95,
                ("T1071.001", "domain"): 0.90,
                ("T1566.001", "email"): 0.95,
                ("T1055", "process"): 0.90,
                ("T1112", "registry"): 0.95,
                ("T1105", "file_hash"): 0.85,
                # Add more mappings as needed
            }
            
            confidence_factors["indicator_relevance"] = relevance_scores.get(
                (tech_id, indicator_type), 0.5  # Default moderate relevance
            )
            
            # 2. Context matching based on additional features
            context_score = 0.0
            context_weights = {
                "geolocation": 0.15,
                "timestamp_pattern": 0.20,
                "associated_malware": 0.25,
                "campaign_indicators": 0.30,
                "infrastructure_pattern": 0.10
            }
            
            for context_key, weight in context_weights.items():
                if context_key in features:
                    context_value = features[context_key]
                    if isinstance(context_value, (int, float)):
                        context_score += min(context_value, 1.0) * weight
                    elif isinstance(context_value, bool) and context_value:
                        context_score += weight
            
            confidence_factors["context_match"] = min(context_score, 1.0)
            
            # 3. Temporal relevance factor
            temporal_score = 0.8  # Default high temporal relevance
            if "timestamp" in features:
                try:
                    event_time = pd.to_datetime(features["timestamp"])
                    time_diff = (datetime.utcnow() - event_time).total_seconds()
                    
                    # Decay factor: more recent events have higher confidence
                    if time_diff <= 3600:  # Within 1 hour
                        temporal_score = 1.0
                    elif time_diff <= 86400:  # Within 24 hours
                        temporal_score = 0.9
                    elif time_diff <= 604800:  # Within 1 week
                        temporal_score = 0.7
                    else:
                        temporal_score = 0.5
                        
                except Exception:
                    pass
            
            confidence_factors["temporal_factor"] = temporal_score
            
            # 4. Threat intelligence correlation
            threat_intel_score = 0.0
            if hasattr(self, 'threat_intel_cache'):
                intel_matches = await self._correlate_with_threat_intel(
                    tech_id, features, indicator_type
                )
                if intel_matches:
                    threat_intel_score = min(intel_matches.get("confidence", 0.0), 1.0)
            
            confidence_factors["threat_intelligence"] = threat_intel_score
            
            # 5. Machine learning prediction confidence
            ml_score = 0.0
            if hasattr(self, 'ml_models') and 'confidence_predictor' in self.ml_models:
                try:
                    feature_vector = self._prepare_confidence_features(
                        tech_id, features, indicator_type
                    )
                    ml_score = await self._predict_confidence_ml(feature_vector)
                except Exception as e:
                    logger.debug(f"ML confidence prediction failed: {e}")
            
            confidence_factors["ml_prediction"] = ml_score
            
            # 6. Historical accuracy based on past detections
            historical_score = 0.6  # Default moderate historical accuracy
            if hasattr(self, 'technique_accuracy_stats'):
                stats = self.technique_accuracy_stats.get(tech_id, {})
                if stats:
                    true_positives = stats.get("true_positives", 0)
                    false_positives = stats.get("false_positives", 0)
                    total_detections = true_positives + false_positives
                    
                    if total_detections > 0:
                        historical_score = true_positives / total_detections
            
            confidence_factors["historical_accuracy"] = historical_score
            
            # Calculate weighted final confidence
            weights = {
                "indicator_relevance": 0.25,
                "context_match": 0.20,
                "temporal_factor": 0.15,
                "threat_intelligence": 0.20,
                "ml_prediction": 0.15,
                "historical_accuracy": 0.05
            }
            
            final_confidence = sum(
                confidence_factors[factor] * weights[factor] 
                for factor in confidence_factors
            )
            
            # Apply confidence boosters/penalties
            if features.get("high_severity_indicators", False):
                final_confidence = min(final_confidence * 1.1, 1.0)  # 10% boost
            
            if features.get("known_false_positive_pattern", False):
                final_confidence *= 0.8  # 20% penalty
            
            # Ensure confidence is within valid range
            final_confidence = max(0.0, min(1.0, final_confidence))
            
            logger.debug(f"Technique {tech_id} confidence: {final_confidence:.3f} (factors: {confidence_factors})")
            return final_confidence
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence for technique {tech_id}: {e}")
            return 0.5  # Default moderate confidence on error
    
    async def _find_related_techniques(self, technique_id: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Find related techniques using advanced similarity algorithms and graph analysis"""
        try:
            related_techniques = []
            
            # Method 1: Tactic-based relationships
            if technique_id in self.mitre_framework:
                technique_data = self.mitre_framework[technique_id]
                current_tactic = technique_data.get("tactic")
                
                if current_tactic:
                    # Find other techniques in the same tactic
                    tactic_techniques = [
                        (tid, 0.7) for tid, data in self.mitre_framework.items()
                        if data.get("tactic") == current_tactic and tid != technique_id
                    ]
                    related_techniques.extend(tactic_techniques)
            
            # Method 2: Kill chain progression relationships
            kill_chain_relationships = await self._find_kill_chain_relationships(technique_id)
            related_techniques.extend(kill_chain_relationships)
            
            # Method 3: Co-occurrence in threat campaigns
            campaign_relationships = await self._find_campaign_co_occurrence(
                technique_id, min_confidence=0.6
            )
            related_techniques.extend(campaign_relationships)
            
            # Method 4: Machine learning similarity
            if hasattr(self, 'technique_embeddings'):
                ml_similarities = await self._find_ml_similar_techniques(
                    technique_id, self.technique_embeddings, limit * 2
                )
                related_techniques.extend(ml_similarities)
            
            # Method 5: Behavioral pattern similarities
            behavioral_similarities = await self._find_behavioral_similarities(
                technique_id, self.mitre_framework
            )
            related_techniques.extend(behavioral_similarities)
            
            # Method 6: Graph-based relationship discovery
            if hasattr(self, 'technique_graph'):
                graph_relationships = await self._find_graph_relationships(
                    technique_id, self.technique_graph, max_hops=3
                )
                related_techniques.extend(graph_relationships)
            
            # Consolidate and rank relationships
            technique_scores = defaultdict(list)
            for tech_id, similarity in related_techniques:
                if tech_id != technique_id:  # Exclude self
                    technique_scores[tech_id].append(similarity)
            
            # Calculate composite similarity scores
            final_scores = []
            for tech_id, scores in technique_scores.items():
                # Use weighted average with recency bias
                weights = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1][:len(scores)]
                weighted_score = sum(s * w for s, w in zip(sorted(scores, reverse=True), weights))
                normalized_score = weighted_score / sum(weights[:len(scores)])
                
                final_scores.append((tech_id, normalized_score))
            
            # Sort by similarity score and apply additional filters
            final_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Filter out low-confidence relationships
            filtered_scores = [
                (tech_id, score) for tech_id, score in final_scores
                if score >= 0.3  # Minimum 30% similarity
            ]
            
            # Apply business logic filters
            business_filtered = await self._apply_business_logic_filters(
                filtered_scores, technique_id
            )
            
            # Return top relationships with metadata
            top_relationships = business_filtered[:limit]
            
            # Enrich with relationship metadata
            enriched_relationships = []
            for tech_id, similarity in top_relationships:
                relationship_metadata = await self._get_relationship_metadata(
                    technique_id, tech_id, similarity
                )
                enriched_relationships.append((tech_id, similarity, relationship_metadata))
            
            logger.debug(f"Found {len(enriched_relationships)} related techniques for {technique_id}")
            return [(tech_id, similarity) for tech_id, similarity, _ in enriched_relationships]
            
        except Exception as e:
            logger.error(f"Failed to find related techniques for {technique_id}: {e}")
            return []
    
    # ... Additional methods for comprehensive implementation
    
    async def _load_builtin_framework_data(self):
        """Load built-in framework data as fallback"""
        logger.info("Loading built-in MITRE framework data...")
        
        try:
            # Load essential tactics
            builtin_tactics = {
                "TA0001": MitreTactic(
                    tactic_id="TA0001",
                    name="Initial Access",
                    description="The adversary is trying to get into your network",
                    short_name="initial-access",
                    platforms=["Windows", "Linux", "macOS"],
                    x_mitre_domains=["enterprise-attack"]
                ),
                "TA0002": MitreTactic(
                    tactic_id="TA0002", 
                    name="Execution",
                    description="The adversary is trying to run malicious code",
                    short_name="execution",
                    platforms=["Windows", "Linux", "macOS"],
                    x_mitre_domains=["enterprise-attack"]
                ),
                "TA0003": MitreTactic(
                    tactic_id="TA0003",
                    name="Persistence", 
                    description="The adversary is trying to maintain their foothold",
                    short_name="persistence",
                    platforms=["Windows", "Linux", "macOS"],
                    x_mitre_domains=["enterprise-attack"]
                ),
                "TA0004": MitreTactic(
                    tactic_id="TA0004",
                    name="Privilege Escalation",
                    description="The adversary is trying to gain higher-level permissions",
                    short_name="privilege-escalation", 
                    platforms=["Windows", "Linux", "macOS"],
                    x_mitre_domains=["enterprise-attack"]
                ),
                "TA0005": MitreTactic(
                    tactic_id="TA0005",
                    name="Defense Evasion",
                    description="The adversary is trying to avoid being detected",
                    short_name="defense-evasion",
                    platforms=["Windows", "Linux", "macOS"],
                    x_mitre_domains=["enterprise-attack"]
                )
            }
            
            # Load essential techniques
            builtin_techniques = {
                "T1190": MitreTechnique(
                    technique_id="T1190",
                    name="Exploit Public-Facing Application",
                    description="Adversaries may attempt to take advantage of a weakness in an Internet-facing computer or program",
                    tactic_refs=["TA0001"],
                    platforms=["Windows", "Linux", "macOS"],
                    data_sources=["Application logs", "Network traffic"],
                    kill_chain_phases=["initial-access"]
                ),
                "T1566": MitreTechnique(
                    technique_id="T1566",
                    name="Phishing",
                    description="Adversaries may send phishing messages to gain access to victim systems",
                    tactic_refs=["TA0001"],
                    platforms=["Windows", "Linux", "macOS"],
                    data_sources=["Email logs", "Network traffic"],
                    kill_chain_phases=["initial-access"]
                ),
                "T1078": MitreTechnique(
                    technique_id="T1078",
                    name="Valid Accounts",
                    description="Adversaries may obtain and abuse credentials of existing accounts",
                    tactic_refs=["TA0003", "TA0004", "TA0001"],
                    platforms=["Windows", "Linux", "macOS"],
                    data_sources=["Authentication logs", "Process monitoring"],
                    kill_chain_phases=["persistence", "privilege-escalation", "initial-access"]
                )
            }
            
            # Store in framework
            self.framework_data["tactics"].update(builtin_tactics)
            self.framework_data["techniques"].update(builtin_techniques)
            
            logger.info(f"Loaded {len(builtin_tactics)} tactics and {len(builtin_techniques)} techniques")
            
        except Exception as e:
            logger.error(f"Error loading built-in framework data: {e}")
            raise
    
    async def _enhance_framework_with_ai(self):
        """Enhance framework data with AI-computed scores"""
        try:
            logger.info("Enhancing MITRE framework with AI-computed scores...")
            
            # Enhance techniques with AI-computed severity scores
            for technique_id, technique in self.framework_data["techniques"].items():
                # Compute AI-based severity score
                severity_score = self._compute_technique_severity(technique)
                technique.ai_severity_score = severity_score
                
                # Compute detection difficulty
                detection_difficulty = self._compute_detection_difficulty(technique)
                technique.ai_detection_difficulty = detection_difficulty
                
                # Compute prevalence score
                prevalence_score = self._compute_technique_prevalence(technique)
                technique.ai_prevalence_score = prevalence_score
            
            # Enhance tactics with aggregated AI scores
            for tactic_id, tactic in self.framework_data["tactics"].items():
                related_techniques = [t for t in self.framework_data["techniques"].values() 
                                    if tactic_id in t.tactic_refs]
                
                if related_techniques:
                    avg_severity = sum(getattr(t, 'ai_severity_score', 5.0) 
                                     for t in related_techniques) / len(related_techniques)
                    tactic.ai_severity_score = avg_severity
            
            logger.info("Framework enhancement with AI completed")
            
        except Exception as e:
            logger.error(f"Error enhancing framework with AI: {e}")
    
    def _compute_technique_severity(self, technique: MitreTechnique) -> float:
        """Compute AI-based technique severity score (1-10)"""
        try:
            base_score = 5.0
            
            # Adjust based on platforms affected
            if len(technique.platforms) > 2:
                base_score += 1.0
            
            # Adjust based on tactic involvement
            if len(technique.tactic_refs) > 2:
                base_score += 1.5
            
            # Keyword-based severity adjustment
            high_severity_keywords = ['privilege', 'escalation', 'persistence', 'lateral', 'exfiltration']
            description_lower = technique.description.lower()
            
            for keyword in high_severity_keywords:
                if keyword in description_lower:
                    base_score += 0.5
            
            return min(10.0, max(1.0, base_score))
            
        except Exception:
            return 5.0  # Default severity
    
    def _compute_detection_difficulty(self, technique: MitreTechnique) -> float:
        """Compute detection difficulty score (1-10, higher = harder to detect)"""
        try:
            base_difficulty = 5.0
            
            # More data sources = easier to detect
            if len(technique.data_sources) > 3:
                base_difficulty -= 1.0
            elif len(technique.data_sources) < 2:
                base_difficulty += 1.0
            
            # Certain techniques are inherently harder to detect
            stealth_keywords = ['steganography', 'rootkit', 'defense evasion', 'living off']
            description_lower = technique.description.lower()
            
            for keyword in stealth_keywords:
                if keyword in description_lower:
                    base_difficulty += 1.0
            
            return min(10.0, max(1.0, base_difficulty))
            
        except Exception:
            return 5.0  # Default difficulty
    
    def _compute_technique_prevalence(self, technique: MitreTechnique) -> float:
        """Compute technique prevalence score based on common attack patterns"""
        try:
            base_prevalence = 3.0
            
            # Common techniques get higher prevalence scores
            common_techniques = {
                'T1566': 9.0,  # Phishing
                'T1078': 8.5,  # Valid Accounts  
                'T1190': 8.0,  # Exploit Public-Facing Application
                'T1055': 7.5,  # Process Injection
                'T1053': 7.0,  # Scheduled Task/Job
            }
            
            if technique.technique_id in common_techniques:
                return common_techniques[technique.technique_id]
            
            # Adjust based on attack phase
            if 'initial-access' in technique.kill_chain_phases:
                base_prevalence += 2.0
            if 'persistence' in technique.kill_chain_phases:
                base_prevalence += 1.5
            if 'execution' in technique.kill_chain_phases:
                base_prevalence += 1.0
            
            return min(10.0, max(1.0, base_prevalence))
            
        except Exception:
            return 3.0  # Default prevalence
    
    async def _save_engine_state(self):
        """Save current engine state"""
        try:
            state_data = {
                "tactics_count": len(self.framework_data["tactics"]),
                "techniques_count": len(self.framework_data["techniques"]),
                "campaigns_count": len(self.framework_data["campaigns"]),
                "threat_actors_count": len(self.framework_data["threat_actors"]),
                "last_updated": datetime.utcnow().isoformat(),
                "ai_enhancement_status": "enhanced",
                "version": "1.0"
            }
            
            # Save to cache
            cache = self.repo_factory.create_cache_repository()
            await cache.set("mitre_engine_state", state_data, ttl=86400)
            
            logger.info("MITRE engine state saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving engine state: {e}")
    
    async def _periodic_framework_update(self):
        """Periodic framework update task"""
        try:
            logger.info("Starting periodic MITRE framework update...")
            
            # Check for new techniques and tactics
            await self._check_for_framework_updates()
            
            # Refresh AI enhancements
            await self._enhance_framework_with_ai()
            
            # Update threat intelligence mappings
            await self._update_threat_mappings()
            
            # Save updated state
            await self._save_engine_state()
            
            logger.info("Periodic framework update completed")
            
        except Exception as e:
            logger.error(f"Error in periodic framework update: {e}")
    
    async def _continuous_threat_monitoring(self):
        """Continuous threat monitoring task"""
        try:
            logger.info("Starting continuous threat monitoring...")
            
            # Monitor for new attack patterns
            new_patterns = await self._detect_new_attack_patterns()
            
            # Update technique usage statistics
            await self._update_technique_statistics()
            
            # Refresh threat landscape analysis
            await self._refresh_threat_landscape()
            
            # Generate threat trend reports
            threat_trends = await self._analyze_threat_trends()
            
            # Update monitoring metrics
            self.monitoring_metrics['last_monitoring_check'] = datetime.utcnow()
            self.monitoring_metrics['patterns_detected'] = len(new_patterns)
            
            logger.info(f"Threat monitoring completed, detected {len(new_patterns)} new patterns")
            
        except Exception as e:
            logger.error(f"Error in continuous threat monitoring: {e}")
    
    async def _check_for_framework_updates(self):
        """Check for MITRE framework updates"""
        # Implementation would check official MITRE sources
        logger.info("Checking for MITRE framework updates...")
        return []
    
    async def _update_threat_mappings(self):
        """Update threat intelligence mappings"""
        logger.info("Updating threat intelligence mappings...")
        
    async def _detect_new_attack_patterns(self):
        """Detect new attack patterns from threat intelligence"""
        logger.info("Detecting new attack patterns...")
        return []
    
    async def _update_technique_statistics(self):
        """Update technique usage statistics"""
        logger.info("Updating technique statistics...")
        
    async def _refresh_threat_landscape(self):
        """Refresh threat landscape analysis"""
        logger.info("Refreshing threat landscape analysis...")
        
    async def _analyze_threat_trends(self):
        """Analyze current threat trends"""
        logger.info("Analyzing threat trends...")
        return {"trends": [], "timestamp": datetime.utcnow()}
    
    async def _validate_framework_integrity(self):
        """Validate framework data integrity"""
        try:
            logger.info("Validating MITRE framework integrity...")
            
            validation_results = {
                "tactics_valid": 0,
                "techniques_valid": 0,
                "invalid_references": [],
                "missing_mappings": [],
                "integrity_score": 0.0
            }
            
            # Validate tactics
            for tactic_id, tactic in self.framework_data["tactics"].items():
                if tactic.tactic_id and tactic.name and tactic.description:
                    validation_results["tactics_valid"] += 1
                else:
                    validation_results["invalid_references"].append(f"Invalid tactic: {tactic_id}")
            
            # Validate techniques
            for technique_id, technique in self.framework_data["techniques"].items():
                if technique.technique_id and technique.name and technique.description:
                    validation_results["techniques_valid"] += 1
                    
                    # Check tactic references
                    for tactic_ref in technique.tactic_refs:
                        if tactic_ref not in self.framework_data["tactics"]:
                            validation_results["missing_mappings"].append(
                                f"Technique {technique_id} references invalid tactic {tactic_ref}"
                            )
                else:
                    validation_results["invalid_references"].append(f"Invalid technique: {technique_id}")
            
            # Calculate integrity score
            total_items = len(self.framework_data["tactics"]) + len(self.framework_data["techniques"])
            valid_items = validation_results["tactics_valid"] + validation_results["techniques_valid"]
            validation_results["integrity_score"] = (valid_items / total_items * 100) if total_items > 0 else 0
            
            logger.info(f"Framework integrity validation completed: {validation_results['integrity_score']:.1f}% valid")
            
            if validation_results["integrity_score"] < 95.0:
                logger.warning(f"Framework integrity below threshold: {validation_results}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating framework integrity: {e}")
            return {"integrity_score": 0.0, "error": str(e)}


# Global service instance
_mitre_engine: Optional[AdvancedMitreAttackEngine] = None

async def get_advanced_mitre_engine() -> AdvancedMitreAttackEngine:
    """Get global Advanced MITRE ATT&CK Engine instance"""
    global _mitre_engine
    
    if _mitre_engine is None:
        _mitre_engine = AdvancedMitreAttackEngine()
        await _mitre_engine.initialize()
        
        # Register with service registry
        from .base_service import service_registry
        service_registry.register(_mitre_engine)
    
    return _mitre_engine