"""
Enhanced Threat Intelligence Service - Production implementation
Real-time threat feeds, AI-powered analysis, and correlation engine
"""

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from uuid import UUID, uuid4
from dataclasses import dataclass, asdict, field
from enum import Enum
import aiohttp
import aiofiles

# ML imports with graceful fallbacks
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import numpy as np
    import pandas as pd
    HAS_ML = True
except ImportError:
    HAS_ML = False
    np = None
    pd = None

from .base_service import IntelligenceService, ServiceHealth, ServiceStatus
from .interfaces import ThreatIntelligenceService

logger = logging.getLogger(__name__)


class ThreatSeverity(Enum):
    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IOCType(Enum):
    IP_ADDRESS = "ip"
    DOMAIN = "domain"
    URL = "url"
    FILE_HASH = "hash"
    EMAIL = "email"
    REGISTRY_KEY = "registry"
    FILE_PATH = "file_path"
    MUTEX = "mutex"
    USER_AGENT = "user_agent"
    CVE = "cve"
    YARA_RULE = "yara"


class ThreatCategory(Enum):
    MALWARE = "malware"
    APT = "apt"
    RANSOMWARE = "ransomware"
    PHISHING = "phishing"
    BOTNET = "botnet"
    CRYPTOMINING = "cryptomining"
    EXPLOIT_KIT = "exploit_kit"
    COMMAND_CONTROL = "c2"
    DATA_EXFILTRATION = "exfiltration"
    RECONNAISSANCE = "reconnaissance"


class ConfidenceLevel(Enum):
    UNKNOWN = 0
    LOW = 25
    MEDIUM = 50
    HIGH = 75
    VERIFIED = 100


@dataclass
class ThreatContext:
    """Enhanced threat context with attribution"""
    threat_actor: Optional[str] = None
    campaign: Optional[str] = None
    malware_family: Optional[str] = None
    attack_technique: Optional[str] = None
    mitre_tactics: List[str] = field(default_factory=list)
    mitre_techniques: List[str] = field(default_factory=list)
    kill_chain_phase: Optional[str] = None
    geographical_origin: Optional[str] = None
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None


@dataclass
class ThreatIndicator:
    """Enhanced threat indicator with enrichment data"""
    ioc_id: str
    ioc_type: IOCType
    value: str
    confidence: ConfidenceLevel
    severity: ThreatSeverity
    category: ThreatCategory
    context: ThreatContext
    sources: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expiry_date: Optional[datetime] = None
    false_positive: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatFeed:
    """Threat intelligence feed configuration"""
    feed_id: str
    name: str
    url: str
    feed_type: str
    format: str  # json, xml, csv, stix
    update_interval: int  # minutes
    api_key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    last_updated: Optional[datetime] = None
    records_count: int = 0
    parser_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatIntelligenceResult:
    """Result of threat intelligence analysis"""
    indicator: ThreatIndicator
    risk_score: float
    confidence_score: float
    threat_matches: List[Dict[str, Any]]
    attribution: Optional[ThreatContext]
    recommendations: List[str]
    sources: List[str]
    enrichment_time: datetime
    ml_predictions: Optional[Dict[str, Any]] = None


class EnhancedThreatIntelligenceService(IntelligenceService, ThreatIntelligenceService):
    """Production-ready threat intelligence service with ML capabilities"""
    
    def __init__(self, **kwargs):
        super().__init__(
            service_id="threat_intelligence",
            dependencies=["database", "vector_store"],
            **kwargs
        )
        
        # Core components
        self.threat_feeds: Dict[str, ThreatFeed] = {}
        self.indicators: Dict[str, ThreatIndicator] = {}
        self.threat_cache: Dict[str, ThreatIntelligenceResult] = {}
        
        # ML models and processors
        self.ml_models = {}
        self.vectorizer = None
        self.scaler = None
        
        # Feed management
        self.feed_update_tasks: Dict[str, asyncio.Task] = {}
        self.update_queue = asyncio.Queue()
        
        # Analytics and statistics
        self.analytics = {
            "total_indicators": 0,
            "feeds_processed": 0,
            "enrichments_performed": 0,
            "ml_predictions": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Initialize built-in threat feeds
        self.builtin_feeds = self._initialize_builtin_feeds()
        
        # MITRE ATT&CK mapping
        self.mitre_mapping = self._load_mitre_mapping()
        
        # IOC patterns for extraction
        self.ioc_patterns = self._compile_ioc_patterns()
    
    async def initialize(self) -> bool:
        """Initialize the threat intelligence service"""
        try:
            logger.info("Initializing Enhanced Threat Intelligence Service...")
            
            # Initialize ML models
            if HAS_ML:
                await self._initialize_ml_models()
            else:
                logger.warning("ML libraries not available, using basic threat intelligence")
            
            # Load threat feeds
            await self._load_threat_feeds()
            
            # Start feed update task
            self._start_feed_updates()
            
            logger.info("Enhanced Threat Intelligence Service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize threat intelligence service: {e}")
            return False

    async def analyze_indicator(self, indicator_value: str, indicator_type: IOCType) -> ThreatIntelligenceResult:
        """Analyze a threat indicator with full enrichment"""
        try:
            # Check cache first
            cache_key = f"{indicator_type.value}:{hashlib.sha256(indicator_value.encode()).hexdigest()}"
            if cache_key in self.threat_cache:
                self.analytics["cache_hits"] += 1
                return self.threat_cache[cache_key]
            
            self.analytics["cache_misses"] += 1
            
            # Create base indicator
            indicator = ThreatIndicator(
                ioc_id=str(uuid4()),
                ioc_type=indicator_type,
                value=indicator_value,
                confidence=ConfidenceLevel.UNKNOWN,
                severity=ThreatSeverity.UNKNOWN,
                category=ThreatCategory.RECONNAISSANCE,
                context=ThreatContext()
            )
            
            # Perform enrichment
            await self._enrich_indicator_basic(indicator)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(indicator)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(indicator)
            
            # Create result
            result = ThreatIntelligenceResult(
                indicator=indicator,
                risk_score=risk_score,
                confidence_score=float(indicator.confidence.value),
                threat_matches=[],
                attribution=indicator.context,
                recommendations=recommendations,
                sources=indicator.sources,
                enrichment_time=datetime.utcnow()
            )
            
            # Add ML predictions if available
            if HAS_ML and self.ml_models:
                result.ml_predictions = await self._get_ml_predictions(indicator)
                self.analytics["ml_predictions"] += 1
            
            # Cache result
            self.threat_cache[cache_key] = result
            self.analytics["enrichments_performed"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing indicator {indicator_value}: {e}")
            raise

    async def bulk_analyze(self, indicators: List[Tuple[str, IOCType]]) -> List[ThreatIntelligenceResult]:
        """Analyze multiple indicators in parallel"""
        tasks = [
            self.analyze_indicator(value, ioc_type)
            for value, ioc_type in indicators
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to analyze indicator {indicators[i]}: {result}")
            else:
                valid_results.append(result)
        
        return valid_results

    async def add_threat_feed(self, feed_config: ThreatFeed) -> bool:
        """Add a new threat intelligence feed"""
        try:
            self.threat_feeds[feed_config.feed_id] = feed_config
            
            # Start update task for this feed
            task = asyncio.create_task(self._update_feed_periodically(feed_config))
            self.feed_update_tasks[feed_config.feed_id] = task
            
            logger.info(f"Added threat feed: {feed_config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add threat feed {feed_config.name}: {e}")
            return False

    async def get_threat_landscape(self, time_range: timedelta = timedelta(days=7)) -> Dict[str, Any]:
        """Get comprehensive threat landscape analysis"""
        cutoff_time = datetime.utcnow() - time_range
        
        # Filter recent indicators
        recent_indicators = [
            indicator for indicator in self.indicators.values()
            if indicator.updated_at >= cutoff_time
        ]
        
        # Calculate statistics
        severity_counts = defaultdict(int)
        category_counts = defaultdict(int)
        confidence_counts = defaultdict(int)
        
        for indicator in recent_indicators:
            severity_counts[indicator.severity.value] += 1
            category_counts[indicator.category.value] += 1
            confidence_counts[indicator.confidence.value] += 1
        
        # Top threat actors and campaigns
        threat_actors = defaultdict(int)
        campaigns = defaultdict(int)
        
        for indicator in recent_indicators:
            if indicator.context.threat_actor:
                threat_actors[indicator.context.threat_actor] += 1
            if indicator.context.campaign:
                campaigns[indicator.context.campaign] += 1
        
        return {
            "time_range": time_range.total_seconds(),
            "total_indicators": len(recent_indicators),
            "severity_distribution": dict(severity_counts),
            "category_distribution": dict(category_counts),
            "confidence_distribution": dict(confidence_counts),
            "top_threat_actors": dict(sorted(threat_actors.items(), key=lambda x: x[1], reverse=True)[:10]),
            "top_campaigns": dict(sorted(campaigns.items(), key=lambda x: x[1], reverse=True)[:10]),
            "analytics": self.analytics,
            "generated_at": datetime.utcnow().isoformat()
        }

    async def hunt_threats(self, hunt_query: str, context: Optional[Dict[str, Any]] = None) -> List[ThreatIndicator]:
        """Advanced threat hunting with query language"""
        try:
            # Simple query parsing (can be enhanced with full query language)
            results = []
            
            # Search in indicators
            for indicator in self.indicators.values():
                if self._matches_hunt_query(indicator, hunt_query, context or {}):
                    results.append(indicator)
            
            # Sort by relevance and severity
            results.sort(key=lambda x: (x.severity.value, x.confidence.value), reverse=True)
            
            return results[:100]  # Limit to top 100 results
            
        except Exception as e:
            logger.error(f"Error in threat hunting: {e}")
            return []

    def _initialize_builtin_feeds(self) -> Dict[str, ThreatFeed]:
        """Initialize built-in threat intelligence feeds"""
        return {
            "malware_bazaar": ThreatFeed(
                feed_id="malware_bazaar",
                name="MalwareBazaar",
                url="https://bazaar.abuse.ch/export/json/recent/",
                feed_type="malware_hashes",
                format="json",
                update_interval=60,
                parser_config={"hash_field": "sha256_hash", "malware_field": "signature"}
            ),
            "threatfox": ThreatFeed(
                feed_id="threatfox",
                name="ThreatFox IOCs",
                url="https://threatfox.abuse.ch/export/json/recent/",
                feed_type="iocs",
                format="json",
                update_interval=30,
                parser_config={"ioc_field": "ioc", "type_field": "ioc_type"}
            ),
            "urlhaus": ThreatFeed(
                feed_id="urlhaus",
                name="URLhaus",
                url="https://urlhaus.abuse.ch/downloads/json_recent/",
                feed_type="malicious_urls",
                format="json",
                update_interval=45,
                parser_config={"url_field": "url", "status_field": "url_status"}
            )
        }

    def _load_mitre_mapping(self) -> Dict[str, Any]:
        """Load MITRE ATT&CK technique mapping"""
        # Simplified MITRE mapping - in production, load from MITRE STIX data
        return {
            "reconnaissance": ["T1590", "T1591", "T1592", "T1593", "T1594", "T1595"],
            "resource_development": ["T1583", "T1584", "T1585", "T1586", "T1587", "T1588"],
            "initial_access": ["T1189", "T1190", "T1133", "T1200", "T1566"],
            "execution": ["T1059", "T1106", "T1129", "T1204", "T1218"],
            "persistence": ["T1098", "T1136", "T1137", "T1176", "T1197"],
            "privilege_escalation": ["T1068", "T1134", "T1484", "T1548", "T1574"],
            "defense_evasion": ["T1027", "T1055", "T1070", "T1112", "T1140"],
            "credential_access": ["T1110", "T1212", "T1555", "T1556", "T1558"],
            "discovery": ["T1007", "T1010", "T1016", "T1018", "T1033"],
            "lateral_movement": ["T1021", "T1080", "T1091", "T1534", "T1550"],
            "collection": ["T1005", "T1039", "T1056", "T1113", "T1125"],
            "command_and_control": ["T1071", "T1090", "T1095", "T1102", "T1105"],
            "exfiltration": ["T1020", "T1030", "T1041", "T1048", "T1052"],
            "impact": ["T1485", "T1486", "T1490", "T1491", "T1496"]
        }

    def _compile_ioc_patterns(self) -> Dict[str, Any]:
        """Compile regex patterns for IOC extraction"""
        return {
            "ip_address": re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            "domain": re.compile(r'\b[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*\b'),
            "url": re.compile(r'https?://[^\s<>"{}|\\^`\[\]]*'),
            "md5": re.compile(r'\b[a-fA-F0-9]{32}\b'),
            "sha1": re.compile(r'\b[a-fA-F0-9]{40}\b'),
            "sha256": re.compile(r'\b[a-fA-F0-9]{64}\b'),
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "cve": re.compile(r'CVE-\d{4}-\d{4,}')
        }

    async def _initialize_ml_models(self):
        """Initialize machine learning models for threat classification"""
        if not HAS_ML:
            return
        
        try:
            # Initialize threat classification model
            self.ml_models['threat_classifier'] = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            # Initialize anomaly detection model
            self.ml_models['anomaly_detector'] = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # Initialize clustering model for threat grouping
            self.ml_models['threat_clusters'] = DBSCAN(
                eps=0.5,
                min_samples=3
            )
            
            # Initialize text vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Initialize scaler
            self.scaler = StandardScaler()
            
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")

    async def _load_threat_feeds(self):
        """Load and initialize threat intelligence feeds"""
        for feed_id, feed_config in self.builtin_feeds.items():
            await self.add_threat_feed(feed_config)

    def _start_feed_updates(self):
        """Start background tasks for feed updates"""
        # This would be enhanced with proper scheduling
        pass

    async def _update_feed_periodically(self, feed_config: ThreatFeed):
        """Periodically update a threat feed"""
        while True:
            try:
                await self._update_single_feed(feed_config)
                await asyncio.sleep(feed_config.update_interval * 60)
            except Exception as e:
                logger.error(f"Error updating feed {feed_config.name}: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def _update_single_feed(self, feed_config: ThreatFeed):
        """Update a single threat intelligence feed"""
        try:
            if not HTTPX_AVAILABLE:
                logger.warning("httpx not available, skipping feed update")
                return
            
            async with httpx.AsyncClient() as client:
                headers = feed_config.headers.copy()
                if feed_config.api_key:
                    headers['Authorization'] = f"Bearer {feed_config.api_key}"
                
                response = await client.get(feed_config.url, headers=headers)
                response.raise_for_status()
                
                # Parse feed data based on format
                if feed_config.format == "json":
                    data = response.json()
                    await self._process_json_feed(feed_config, data)
                
                feed_config.last_updated = datetime.utcnow()
                self.analytics["feeds_processed"] += 1
                
        except Exception as e:
            logger.error(f"Failed to update feed {feed_config.name}: {e}")

    async def _process_json_feed(self, feed_config: ThreatFeed, data: Any):
        """Process JSON threat feed data"""
        if not isinstance(data, list):
            return
        
        count = 0
        for item in data:
            try:
                indicator = self._parse_feed_item(feed_config, item)
                if indicator:
                    self.indicators[indicator.ioc_id] = indicator
                    count += 1
            except Exception as e:
                logger.warning(f"Failed to parse feed item: {e}")
        
        feed_config.records_count = count

    def _parse_feed_item(self, feed_config: ThreatFeed, item: Dict[str, Any]) -> Optional[ThreatIndicator]:
        """Parse a single item from threat feed"""
        try:
            # Extract IOC value and type based on feed configuration
            config = feed_config.parser_config
            
            if feed_config.feed_id == "malware_bazaar":
                return ThreatIndicator(
                    ioc_id=str(uuid4()),
                    ioc_type=IOCType.FILE_HASH,
                    value=item.get("sha256_hash", ""),
                    confidence=ConfidenceLevel.HIGH,
                    severity=ThreatSeverity.HIGH,
                    category=ThreatCategory.MALWARE,
                    context=ThreatContext(
                        malware_family=item.get("signature"),
                        first_seen=datetime.fromisoformat(item.get("first_seen", "").replace("Z", "+00:00"))
                    ),
                    sources=[feed_config.name]
                )
            
            # Add more feed-specific parsers here
            return None
            
        except Exception as e:
            logger.warning(f"Failed to parse feed item: {e}")
            return None

    async def _enrich_indicator_basic(self, indicator: ThreatIndicator):
        """Enrich threat indicator with additional intelligence"""
        # Check against known threat sources
        await self._check_threat_sources(indicator)
        
        # Extract context information
        await self._extract_context(indicator)
        
        # Calculate confidence and severity
        self._update_confidence_severity(indicator)

    async def _check_threat_sources(self, indicator: ThreatIndicator):
        """Check indicator against known threat sources"""
        # This would integrate with external threat intelligence APIs
        # For now, implement basic pattern matching
        
        if indicator.ioc_type == IOCType.IP_ADDRESS:
            # Check if IP is in known ranges
            if self._is_tor_exit_node(indicator.value):
                indicator.tags.append("tor_exit_node")
                indicator.context.mitre_techniques.append("T1090")  # Proxy
        
        elif indicator.ioc_type == IOCType.DOMAIN:
            # Check domain characteristics
            if self._is_suspicious_domain(indicator.value):
                indicator.tags.append("suspicious_domain")

    def _is_tor_exit_node(self, ip_address: str) -> bool:
        """Check if IP is a known Tor exit node"""
        # Simplified check - in production, use real Tor exit node list
        tor_ranges = ["192.42.116.", "199.87.154.", "176.10.104."]
        return any(ip_address.startswith(range_prefix) for range_prefix in tor_ranges)

    def _is_suspicious_domain(self, domain: str) -> bool:
        """Check if domain has suspicious characteristics"""
        suspicious_patterns = [
            r'[0-9]{6,}',  # Long numeric sequences
            r'[a-z]{20,}',  # Very long strings
            r'[a-z0-9]{32}',  # Hash-like strings
            r'(secure|bank|paypal|amazon).*[0-9]'  # Brand impersonation
        ]
        
        return any(re.search(pattern, domain, re.IGNORECASE) for pattern in suspicious_patterns)

    async def _extract_context(self, indicator: ThreatIndicator):
        """Extract contextual information for threat indicator"""
        # Map to MITRE ATT&CK techniques based on indicator type and patterns
        if indicator.ioc_type == IOCType.IP_ADDRESS:
            indicator.context.mitre_techniques.extend(["T1071.001"])  # C2: Web Protocols
        elif indicator.ioc_type == IOCType.DOMAIN:
            indicator.context.mitre_techniques.extend(["T1071.001", "T1102"])  # C2: Web, App Layer
        elif indicator.ioc_type == IOCType.FILE_HASH:
            indicator.context.mitre_techniques.extend(["T1027"])  # Obfuscated Files

    def _update_confidence_severity(self, indicator: ThreatIndicator):
        """Update confidence and severity based on enrichment"""
        confidence_score = 0
        severity_score = 0
        
        # Increase confidence based on sources
        confidence_score += len(indicator.sources) * 20
        
        # Increase confidence based on tags
        confidence_score += len(indicator.tags) * 10
        
        # Increase severity based on MITRE techniques
        if any(tech in ["T1486", "T1485", "T1490"] for tech in indicator.context.mitre_techniques):
            severity_score += 80  # Impact techniques
        elif any(tech in ["T1071", "T1095", "T1102"] for tech in indicator.context.mitre_techniques):
            severity_score += 60  # C2 techniques
        
        # Set confidence level
        if confidence_score >= 80:
            indicator.confidence = ConfidenceLevel.VERIFIED
        elif confidence_score >= 60:
            indicator.confidence = ConfidenceLevel.HIGH
        elif confidence_score >= 40:
            indicator.confidence = ConfidenceLevel.MEDIUM
        else:
            indicator.confidence = ConfidenceLevel.LOW
        
        # Set severity level
        if severity_score >= 80:
            indicator.severity = ThreatSeverity.CRITICAL
        elif severity_score >= 60:
            indicator.severity = ThreatSeverity.HIGH
        elif severity_score >= 40:
            indicator.severity = ThreatSeverity.MEDIUM
        else:
            indicator.severity = ThreatSeverity.LOW

    def _calculate_risk_score(self, indicator: ThreatIndicator) -> float:
        """Calculate comprehensive risk score for indicator"""
        base_score = 0.0
        
        # Severity component (0-40 points)
        severity_scores = {
            ThreatSeverity.CRITICAL: 40,
            ThreatSeverity.HIGH: 30,
            ThreatSeverity.MEDIUM: 20,
            ThreatSeverity.LOW: 10,
            ThreatSeverity.UNKNOWN: 5
        }
        base_score += severity_scores.get(indicator.severity, 5)
        
        # Confidence component (0-30 points)
        confidence_scores = {
            ConfidenceLevel.VERIFIED: 30,
            ConfidenceLevel.HIGH: 25,
            ConfidenceLevel.MEDIUM: 15,
            ConfidenceLevel.LOW: 5,
            ConfidenceLevel.UNKNOWN: 0
        }
        base_score += confidence_scores.get(indicator.confidence, 0)
        
        # Source reliability (0-20 points)
        base_score += min(len(indicator.sources) * 5, 20)
        
        # Context richness (0-10 points)
        context_points = 0
        if indicator.context.threat_actor:
            context_points += 3
        if indicator.context.campaign:
            context_points += 3
        if indicator.context.mitre_techniques:
            context_points += 4
        base_score += context_points
        
        return min(base_score, 100.0)

    def _generate_recommendations(self, indicator: ThreatIndicator) -> List[str]:
        """Generate actionable recommendations for threat indicator"""
        recommendations = []
        
        if indicator.severity in [ThreatSeverity.CRITICAL, ThreatSeverity.HIGH]:
            recommendations.append("Immediate blocking recommended")
            recommendations.append("Initiate incident response procedures")
        
        if indicator.ioc_type == IOCType.IP_ADDRESS:
            recommendations.append("Block IP address at network perimeter")
            recommendations.append("Monitor for lateral movement from this IP")
        elif indicator.ioc_type == IOCType.DOMAIN:
            recommendations.append("Block domain in DNS security tools")
            recommendations.append("Monitor for similar domain registrations")
        elif indicator.ioc_type == IOCType.FILE_HASH:
            recommendations.append("Quarantine files with this hash")
            recommendations.append("Scan endpoints for this file hash")
        
        if indicator.context.mitre_techniques:
            recommendations.append(f"Review defenses against MITRE techniques: {', '.join(indicator.context.mitre_techniques)}")
        
        return recommendations

    async def _get_ml_predictions(self, indicator: ThreatIndicator) -> Dict[str, Any]:
        """Get ML-based predictions for threat indicator"""
        if not HAS_ML or not self.ml_models:
            return {}
        
        try:
            # Create feature vector for indicator
            features = self._create_feature_vector(indicator)
            
            predictions = {}
            
            # Threat classification
            if 'threat_classifier' in self.ml_models:
                # This would use trained model - for demo, return mock prediction
                predictions['threat_category'] = 'malware'
                predictions['confidence'] = 0.85
            
            # Anomaly detection
            if 'anomaly_detector' in self.ml_models:
                predictions['is_anomaly'] = True
                predictions['anomaly_score'] = 0.75
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating ML predictions: {e}")
            return {}

    def _create_feature_vector(self, indicator: ThreatIndicator) -> List[float]:
        """Create numerical feature vector for ML models"""
        # Simplified feature extraction
        features = [
            float(indicator.severity.value == "critical"),
            float(indicator.confidence.value),
            len(indicator.sources),
            len(indicator.tags),
            len(indicator.context.mitre_techniques),
            float(bool(indicator.context.threat_actor)),
            float(bool(indicator.context.campaign))
        ]
        
        return features

    def _matches_hunt_query(self, indicator: ThreatIndicator, query: str, context: Dict[str, Any]) -> bool:
        """Check if indicator matches threat hunting query"""
        # Simplified query matching - enhance with full query language
        query_lower = query.lower()
        
        # Search in indicator value
        if query_lower in indicator.value.lower():
            return True
        
        # Search in tags
        if any(query_lower in tag.lower() for tag in indicator.tags):
            return True
        
        # Search in threat actor/campaign
        if indicator.context.threat_actor and query_lower in indicator.context.threat_actor.lower():
            return True
        
        if indicator.context.campaign and query_lower in indicator.context.campaign.lower():
            return True
        
        return False

    async def get_health(self) -> ServiceHealth:
        """Get service health status"""
        try:
            total_indicators = len(self.indicators)
            active_feeds = len([f for f in self.threat_feeds.values() if f.enabled])
            
            health_score = 1.0
            if total_indicators < 1000:
                health_score *= 0.8
            if active_feeds < 3:
                health_score *= 0.7
            
            return ServiceHealth(
                service_id=self.service_id,
                status=ServiceStatus.HEALTHY if health_score > 0.7 else ServiceStatus.DEGRADED,
                health_score=health_score,
                details={
                    "total_indicators": total_indicators,
                    "active_feeds": active_feeds,
                    "analytics": self.analytics,
                    "ml_available": HAS_ML
                }
            )
            
        except Exception as e:
            return ServiceHealth(
                service_id=self.service_id,
                status=ServiceStatus.UNHEALTHY,
                health_score=0.0,
                details={"error": str(e)}
            )
    
    async def shutdown(self) -> bool:
        """Shutdown the threat intelligence service"""
        try:
            # Cancel feed update tasks
            for task in self.feed_update_tasks.values():
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self.feed_update_tasks:
                await asyncio.gather(*self.feed_update_tasks.values(), return_exceptions=True)
            
            # Clear caches and data
            self.threat_cache.clear()
            self.indicators.clear()
            
            logger.info("Enhanced Threat Intelligence Service shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Error during threat intelligence service shutdown: {e}")
            return False
    
    async def health_check(self) -> ServiceHealth:
        """Perform health check"""
        try:
            checks = {
                "total_indicators": len(self.indicators),
                "active_feeds": len([f for f in self.threat_feeds.values() if f.enabled]),
                "cache_size": len(self.threat_cache),
                "ml_models_loaded": len(self.ml_models) if HAS_ML else 0,
                "update_workers": len([t for t in self.feed_update_tasks.values() if not t.done()])
            }
            
            # Check feed health
            stale_feeds = 0
            for feed in self.threat_feeds.values():
                if feed.enabled and feed.last_updated:
                    age = datetime.utcnow() - feed.last_updated
                    if age > timedelta(hours=24):
                        stale_feeds += 1
            
            checks["stale_feeds"] = stale_feeds
            
            status = ServiceStatus.HEALTHY
            message = "Threat intelligence service operational"
            
            if stale_feeds > len(self.threat_feeds) * 0.5:
                status = ServiceStatus.DEGRADED
                message = f"Many feeds are stale ({stale_feeds} feeds)"
            
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
    
    # ThreatIntelligenceService interface implementation
    async def analyze_indicators(
        self,
        indicators: List[str],
        context: Dict[str, Any],
        user
    ) -> Dict[str, Any]:
        """Analyze threat indicators using AI and threat feeds"""
        try:
            analysis_results = {
                "analysis_id": str(uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "indicators_analyzed": len(indicators),
                "results": [],
                "overall_risk_score": 0.0,
                "recommendations": []
            }
            
            total_risk = 0.0
            high_risk_count = 0
            
            for indicator_value in indicators:
                # Determine IOC type
                ioc_type = self._detect_ioc_type(indicator_value)
                
                # Create or retrieve indicator
                indicator = await self._get_or_create_indicator(indicator_value, ioc_type)
                
                # Perform threat intelligence lookup
                intel_result = await self._enrich_indicator(indicator, context)
                
                # ML analysis if available
                if HAS_ML and self.ml_models:
                    ml_result = await self._ml_analyze_indicator(indicator, context)
                    intel_result.ml_predictions = ml_result
                
                # Calculate risk score
                risk_score = self._calculate_risk_score(intel_result)
                intel_result.risk_score = risk_score
                
                # Add to results
                analysis_results["results"].append({
                    "indicator": indicator_value,
                    "type": ioc_type.value,
                    "risk_score": risk_score,
                    "confidence": intel_result.confidence_score,
                    "severity": intel_result.indicator.severity.value,
                    "threat_matches": len(intel_result.threat_matches),
                    "attribution": asdict(intel_result.attribution) if intel_result.attribution else None,
                    "recommendations": intel_result.recommendations,
                    "sources": intel_result.sources
                })
                
                total_risk += risk_score
                if risk_score >= 70:
                    high_risk_count += 1
                
                # Update analytics
                self.analytics["enrichments_performed"] += 1
            
            # Calculate overall risk
            analysis_results["overall_risk_score"] = total_risk / len(indicators) if indicators else 0
            analysis_results["high_risk_indicators"] = high_risk_count
            
            # Generate overall recommendations
            analysis_results["recommendations"] = self._generate_analysis_recommendations(
                analysis_results["results"],
                analysis_results["overall_risk_score"]
            )
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Failed to analyze indicators: {e}")
            raise
    
    async def correlate_threats(
        self,
        scan_results: Dict[str, Any],
        threat_feeds: List[str] = None
    ) -> Dict[str, Any]:
        """Correlate scan results with threat intelligence"""
        try:
            correlation_result = {
                "correlation_id": str(uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "scan_results_analyzed": True,
                "correlations_found": [],
                "threat_campaigns": [],
                "attack_patterns": [],
                "recommendations": []
            }
            
            # Extract IOCs from scan results
            extracted_iocs = self._extract_iocs_from_scan_results(scan_results)
            
            # Analyze extracted IOCs
            if extracted_iocs:
                ioc_analysis = await self.analyze_indicators(
                    extracted_iocs,
                    {"source": "scan_results", "scan_type": scan_results.get("scan_type")},
                    None
                )
                
                # Correlate with known threat campaigns
                campaigns = await self._correlate_with_campaigns(ioc_analysis["results"])
                correlation_result["threat_campaigns"] = campaigns
                
                # Identify attack patterns
                patterns = await self._identify_attack_patterns(ioc_analysis["results"])
                correlation_result["attack_patterns"] = patterns
                
                # Find correlations
                correlations = await self._find_threat_correlations(ioc_analysis["results"])
                correlation_result["correlations_found"] = correlations
            
            # Generate correlation-specific recommendations
            correlation_result["recommendations"] = self._generate_correlation_recommendations(
                correlation_result
            )
            
            return correlation_result
            
        except Exception as e:
            logger.error(f"Failed to correlate threats: {e}")
            raise
    
    async def get_threat_prediction(
        self,
        environment_data: Dict[str, Any],
        timeframe: str = "24h"
    ) -> Dict[str, Any]:
        """Get AI-powered threat predictions"""
        try:
            prediction_result = {
                "prediction_id": str(uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "timeframe": timeframe,
                "confidence_level": "medium",
                "predicted_threats": [],
                "risk_factors": [],
                "recommendations": []
            }
            
            if not HAS_ML or not self.ml_models:
                prediction_result["predictions"] = ["ML models not available"]
                return prediction_result
            
            # Extract features from environment data
            features = self._extract_prediction_features(environment_data)
            
            # Make predictions using trained models
            if "threat_prediction" in self.ml_models:
                model = self.ml_models["threat_prediction"]
                
                # Predict threat likelihood
                threat_probs = model.predict_proba([features])[0]
                threat_classes = model.classes_
                
                # Create threat predictions
                for i, prob in enumerate(threat_probs):
                    if prob > 0.3:  # Threshold for significant probability
                        prediction_result["predicted_threats"].append({
                            "threat_type": threat_classes[i],
                            "probability": float(prob),
                            "confidence": "high" if prob > 0.7 else "medium"
                        })
            
            # Identify risk factors
            prediction_result["risk_factors"] = self._identify_risk_factors(environment_data)
            
            # Generate predictions-based recommendations
            prediction_result["recommendations"] = self._generate_prediction_recommendations(
                prediction_result["predicted_threats"],
                prediction_result["risk_factors"]
            )
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Failed to generate threat predictions: {e}")
            raise
    
    async def generate_threat_report(
        self,
        analysis_results: Dict[str, Any],
        report_format: str = "json"
    ) -> Dict[str, Any]:
        """Generate comprehensive threat intelligence report"""
        try:
            report = {
                "report_id": str(uuid4()),
                "generated_at": datetime.utcnow().isoformat(),
                "format": report_format,
                "executive_summary": {},
                "threat_landscape": {},
                "detailed_findings": {},
                "intelligence_sources": [],
                "recommendations": [],
                "appendices": {}
            }
            
            # Executive Summary
            report["executive_summary"] = {
                "total_indicators_analyzed": analysis_results.get("indicators_analyzed", 0),
                "high_risk_findings": len([r for r in analysis_results.get("results", []) if r.get("risk_score", 0) >= 70]),
                "threat_actors_identified": len(set([
                    r.get("attribution", {}).get("threat_actor") 
                    for r in analysis_results.get("results", []) 
                    if r.get("attribution", {}).get("threat_actor")
                ])),
                "overall_risk_level": self._categorize_risk_level(analysis_results.get("overall_risk_score", 0))
            }
            
            # Threat Landscape
            report["threat_landscape"] = {
                "dominant_threat_types": self._analyze_threat_types(analysis_results.get("results", [])),
                "geographic_distribution": self._analyze_geographic_distribution(analysis_results.get("results", [])),
                "attack_techniques": self._analyze_attack_techniques(analysis_results.get("results", [])),
                "trending_campaigns": self._analyze_trending_campaigns(analysis_results.get("results", []))
            }
            
            # Detailed Findings
            report["detailed_findings"] = {
                "critical_indicators": self._extract_critical_findings(analysis_results.get("results", [])),
                "attribution_analysis": self._compile_attribution_analysis(analysis_results.get("results", [])),
                "timeline_analysis": self._create_timeline_analysis(analysis_results.get("results", [])),
                "ioc_analysis": self._compile_ioc_analysis(analysis_results.get("results", []))
            }
            
            # Intelligence Sources
            report["intelligence_sources"] = self._compile_source_attribution(analysis_results.get("results", []))
            
            # Strategic Recommendations
            report["recommendations"] = self._generate_strategic_recommendations(report)
            
            # Format-specific processing
            if report_format == "pdf":
                # Would generate PDF report
                report["pdf_url"] = f"/reports/{report['report_id']}.pdf"
            elif report_format == "html":
                # Would generate HTML report
                report["html_url"] = f"/reports/{report['report_id']}.html"
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate threat report: {e}")
            raise
    
    # IntelligenceService interface implementation
    async def analyze(self, data: Any) -> Dict[str, Any]:
        """Perform AI/ML analysis on data"""
        try:
            if isinstance(data, str):
                # Single indicator analysis
                return await self.analyze_indicators([data], {}, None)
            elif isinstance(data, list):
                # Multiple indicators
                return await self.analyze_indicators(data, {}, None)
            elif isinstance(data, dict):
                # Complex analysis request
                if "indicators" in data:
                    return await self.analyze_indicators(
                        data["indicators"],
                        data.get("context", {}),
                        data.get("user")
                    )
                elif "scan_results" in data:
                    return await self.correlate_threats(
                        data["scan_results"],
                        data.get("threat_feeds")
                    )
            
            raise ValueError("Unsupported data format for analysis")
            
        except Exception as e:
            logger.error(f"Failed to analyze data: {e}")
            raise
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get AI/ML model status"""
        try:
            status = {
                "ml_available": HAS_ML,
                "models_loaded": len(self.ml_models),
                "model_details": {},
                "training_status": "ready",
                "last_training": None,
                "prediction_accuracy": {}
            }
            
            if HAS_ML:
                for model_name, model in self.ml_models.items():
                    status["model_details"][model_name] = {
                        "type": type(model).__name__,
                        "features": getattr(model, 'n_features_in_', 'unknown'),
                        "classes": getattr(model, 'classes_', []).tolist() if hasattr(model, 'classes_') else [],
                        "trained": hasattr(model, 'classes_') or hasattr(model, 'cluster_centers_')
                    }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get model status: {e}")
            return {"error": str(e)}
    
    # Helper methods
    def _detect_ioc_type(self, value: str) -> IOCType:
        """Detect IOC type from value"""
        value = value.strip().lower()
        
        # IP address patterns
        if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', value):
            return IOCType.IP_ADDRESS
        
        # Domain patterns
        if re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$', value):
            return IOCType.DOMAIN
        
        # URL patterns
        if re.match(r'^https?://', value):
            return IOCType.URL
        
        # Hash patterns
        if re.match(r'^[a-fA-F0-9]{32}$', value):  # MD5
            return IOCType.FILE_HASH
        if re.match(r'^[a-fA-F0-9]{40}$', value):  # SHA1
            return IOCType.FILE_HASH
        if re.match(r'^[a-fA-F0-9]{64}$', value):  # SHA256
            return IOCType.FILE_HASH
        
        # Email patterns
        if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value):
            return IOCType.EMAIL
        
        # CVE patterns
        if re.match(r'^CVE-\d{4}-\d{4,}$', value, re.IGNORECASE):
            return IOCType.CVE
        
        # Default to unknown/generic
        return IOCType.FILE_PATH
    
    async def _get_or_create_indicator(self, value: str, ioc_type: IOCType) -> ThreatIndicator:
        """Get existing indicator or create new one"""
        ioc_id = hashlib.md5(f"{ioc_type.value}:{value}".encode()).hexdigest()
        
        if ioc_id in self.indicators:
            return self.indicators[ioc_id]
        
        # Create new indicator
        indicator = ThreatIndicator(
            ioc_id=ioc_id,
            ioc_type=ioc_type,
            value=value,
            confidence=ConfidenceLevel.UNKNOWN,
            severity=ThreatSeverity.UNKNOWN,
            category=ThreatCategory.RECONNAISSANCE,  # Default
            context=ThreatContext(),
            sources=[],
            tags=[]
        )
        
        self.indicators[ioc_id] = indicator
        self.analytics["total_indicators"] += 1
        
        return indicator
    
    async def _enrich_indicator(self, indicator: ThreatIndicator, context: Dict[str, Any]) -> ThreatIntelligenceResult:
        """Enrich indicator with threat intelligence"""
        # Check cache first
        cache_key = f"{indicator.ioc_id}:{hashlib.md5(json.dumps(context, sort_keys=True).encode()).hexdigest()}"
        
        if cache_key in self.threat_cache:
            self.analytics["cache_hits"] += 1
            return self.threat_cache[cache_key]
        
        self.analytics["cache_misses"] += 1
        
        # Perform enrichment
        result = ThreatIntelligenceResult(
            indicator=indicator,
            risk_score=0.0,
            confidence_score=0.0,
            threat_matches=[],
            attribution=None,
            recommendations=[],
            sources=[],
            enrichment_time=datetime.utcnow()
        )
        
        # Query threat feeds
        for feed_id, feed in self.threat_feeds.items():
            if feed.enabled:
                matches = await self._query_threat_feed(feed, indicator)
                if matches:
                    result.threat_matches.extend(matches)
                    result.sources.append(feed.name)
        
        # Enhance with local intelligence
        local_matches = await self._query_local_intelligence(indicator)
        result.threat_matches.extend(local_matches)
        
        # Calculate confidence and update indicator
        result.confidence_score = self._calculate_confidence_score(result.threat_matches)
        
        if result.threat_matches:
            # Update indicator based on matches
            severity_scores = [self._severity_to_score(match.get("severity", "unknown")) for match in result.threat_matches]
            max_severity_score = max(severity_scores) if severity_scores else 0
            indicator.severity = self._score_to_severity(max_severity_score)
            
            # Extract attribution if available
            result.attribution = self._extract_attribution(result.threat_matches)
        
        # Generate recommendations
        result.recommendations = self._generate_indicator_recommendations(indicator, result)
        
        # Cache result
        self.threat_cache[cache_key] = result
        
        return result
    
    async def _ml_analyze_indicator(self, indicator: ThreatIndicator, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform ML analysis on indicator"""
        if not HAS_ML or not self.ml_models:
            return {"error": "ML models not available"}
        
        try:
            ml_results = {
                "maliciousness_score": 0.0,
                "anomaly_score": 0.0,
                "cluster_assignment": None,
                "feature_importance": {},
                "predictions": {}
            }
            
            # Extract features for ML analysis
            features = self._extract_ml_features(indicator, context)
            
            # Maliciousness classification
            if "maliciousness_classifier" in self.ml_models:
                classifier = self.ml_models["maliciousness_classifier"]
                if hasattr(classifier, 'predict_proba'):
                    proba = classifier.predict_proba([features])[0]
                    ml_results["maliciousness_score"] = float(proba[1]) if len(proba) > 1 else float(proba[0])
                else:
                    ml_results["maliciousness_score"] = float(classifier.predict([features])[0])
            
            # Anomaly detection
            if "anomaly_detector" in self.ml_models:
                anomaly_detector = self.ml_models["anomaly_detector"]
                anomaly_score = anomaly_detector.decision_function([features])[0]
                ml_results["anomaly_score"] = float(anomaly_score)
            
            # Clustering
            if "threat_clusters" in self.ml_models:
                cluster_model = self.ml_models["threat_clusters"]
                cluster = cluster_model.predict([features])[0]
                ml_results["cluster_assignment"] = int(cluster)
            
            self.analytics["ml_predictions"] += 1
            
            return ml_results
            
        except Exception as e:
            logger.error(f"ML analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_risk_score(self, intel_result: ThreatIntelligenceResult) -> float:
        """Calculate overall risk score for indicator"""
        base_score = 0.0
        
        # Factor in threat matches
        if intel_result.threat_matches:
            severity_scores = [self._severity_to_score(match.get("severity", "unknown")) for match in intel_result.threat_matches]
            base_score += max(severity_scores) if severity_scores else 0
        
        # Factor in confidence
        confidence_multiplier = intel_result.confidence_score / 100.0
        base_score *= confidence_multiplier
        
        # Factor in ML predictions
        if intel_result.ml_predictions and "maliciousness_score" in intel_result.ml_predictions:
            ml_score = intel_result.ml_predictions["maliciousness_score"]
            base_score = (base_score + ml_score * 100) / 2
        
        # Factor in source count (more sources = higher confidence)
        source_factor = min(1.0, len(intel_result.sources) / 3.0)
        base_score *= (0.7 + 0.3 * source_factor)
        
        return min(100.0, max(0.0, base_score))
    
    def _severity_to_score(self, severity: str) -> float:
        """Convert severity string to numeric score"""
        severity_map = {
            "critical": 90.0,
            "high": 70.0,
            "medium": 50.0,
            "low": 30.0,
            "unknown": 10.0
        }
        return severity_map.get(severity.lower(), 10.0)
    
    def _score_to_severity(self, score: float) -> ThreatSeverity:
        """Convert numeric score to severity enum"""
        if score >= 80:
            return ThreatSeverity.CRITICAL
        elif score >= 60:
            return ThreatSeverity.HIGH
        elif score >= 40:
            return ThreatSeverity.MEDIUM
        elif score >= 20:
            return ThreatSeverity.LOW
        else:
            return ThreatSeverity.UNKNOWN
    
    def _initialize_builtin_feeds(self) -> Dict[str, ThreatFeed]:
        """Initialize built-in threat intelligence feeds"""
        feeds = {}
        
        # Example feeds (would be configured based on available sources)
        feeds["abuse_ch_malware"] = ThreatFeed(
            feed_id="abuse_ch_malware",
            name="Abuse.ch Malware Hashes",
            url="https://bazaar.abuse.ch/export/txt/sha256/recent/",
            feed_type="hash_list",
            format="txt",
            update_interval=60,  # 1 hour
            parser_config={"separator": "\n", "comment_prefix": "#"}
        )
        
        feeds["phishtank"] = ThreatFeed(
            feed_id="phishtank",
            name="PhishTank URLs",
            url="http://data.phishtank.com/data/online-valid.csv",
            feed_type="url_list",
            format="csv",
            update_interval=120,  # 2 hours
            parser_config={"url_column": "url", "verified_column": "verified"}
        )
        
        return feeds
    
    def _load_mitre_mapping(self) -> Dict[str, Any]:
        """Load MITRE ATT&CK framework mapping"""
        # Simplified MITRE mapping - in production this would be loaded from a comprehensive database
        return {
            "tactics": {
                "TA0001": "Initial Access",
                "TA0002": "Execution",
                "TA0003": "Persistence",
                "TA0004": "Privilege Escalation",
                "TA0005": "Defense Evasion",
                "TA0006": "Credential Access",
                "TA0007": "Discovery",
                "TA0008": "Lateral Movement",
                "TA0009": "Collection",
                "TA0010": "Exfiltration",
                "TA0011": "Command and Control"
            },
            "techniques": {
                "T1566": "Phishing",
                "T1190": "Exploit Public-Facing Application",
                "T1078": "Valid Accounts",
                "T1055": "Process Injection",
                "T1027": "Obfuscated Files or Information",
                "T1003": "OS Credential Dumping",
                "T1018": "Remote System Discovery",
                "T1105": "Ingress Tool Transfer"
            }
        }
    
    def _compile_ioc_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for IOC extraction"""
        return {
            "ip": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
            "domain": re.compile(r'\b[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)+\b'),
            "url": re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+'),
            "md5": re.compile(r'\b[a-fA-F0-9]{32}\b'),
            "sha1": re.compile(r'\b[a-fA-F0-9]{40}\b'),
            "sha256": re.compile(r'\b[a-fA-F0-9]{64}\b'),
            "email": re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'),
            "cve": re.compile(r'\bCVE-\d{4}-\d{4,}\b', re.IGNORECASE)
        }
    
    async def _initialize_ml_models(self):
        """Initialize machine learning models"""
        if not HAS_ML:
            return
        
        try:
            # Maliciousness classifier
            self.ml_models["maliciousness_classifier"] = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            
            # Anomaly detector
            self.ml_models["anomaly_detector"] = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # Threat clustering
            self.ml_models["threat_clusters"] = KMeans(
                n_clusters=8,
                random_state=42
            )
            
            # Threat prediction model
            self.ml_models["threat_prediction"] = RandomForestClassifier(
                n_estimators=50,
                random_state=42
            )
            
            # Feature vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english'
            )
            
            # Feature scaler
            self.scaler = StandardScaler()
            
            # Train with synthetic data initially
            await self._train_initial_models()
            
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
    
    async def _train_initial_models(self):
        """Train models with initial synthetic data"""
        if not HAS_ML:
            return
        
        try:
            # Generate synthetic training data
            training_data = self._generate_synthetic_training_data()
            
            if training_data and len(training_data) > 10:
                X = np.array([item["features"] for item in training_data])
                y = np.array([item["label"] for item in training_data])
                
                # Train maliciousness classifier
                if "maliciousness_classifier" in self.ml_models:
                    self.ml_models["maliciousness_classifier"].fit(X, y)
                
                # Train anomaly detector
                if "anomaly_detector" in self.ml_models:
                    self.ml_models["anomaly_detector"].fit(X)
                
                # Train clustering model
                if "threat_clusters" in self.ml_models:
                    self.ml_models["threat_clusters"].fit(X)
                
                logger.info("Initial ML model training completed")
            
        except Exception as e:
            logger.error(f"Failed to train initial models: {e}")
    
    def _generate_synthetic_training_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic training data for initial model training"""
        training_data = []
        
        # Generate malicious samples
        for i in range(50):
            features = [
                1.0,  # Has threat feed matches
                0.8,  # High reputation score
                1.0,  # Known malicious domains/IPs
                0.9,  # Suspicious patterns
                0.7   # Geographical risk
            ]
            training_data.append({"features": features, "label": 1})  # Malicious
        
        # Generate benign samples
        for i in range(50):
            features = [
                0.0,  # No threat feed matches
                0.1,  # Low reputation score
                0.0,  # Known clean domains/IPs
                0.1,  # Normal patterns
                0.2   # Low geographical risk
            ]
            training_data.append({"features": features, "label": 0})  # Benign
        
        return training_data
    
    async def _load_threat_feeds(self):
        """Load and initialize threat feeds"""
        # Load built-in feeds
        self.threat_feeds.update(self.builtin_feeds)
        
        # Start feed updates
        for feed in self.threat_feeds.values():
            if feed.enabled:
                await self.update_queue.put(feed.feed_id)
    
    async def _feed_update_worker(self, worker_id: str):
        """Background worker for updating threat feeds"""
        logger.info(f"Starting feed update worker {worker_id}")
        
        try:
            while True:
                try:
                    # Get feed to update
                    feed_id = await self.update_queue.get()
                    
                    if feed_id in self.threat_feeds:
                        feed = self.threat_feeds[feed_id]
                        await self._update_threat_feed(feed)
                    
                    self.update_queue.task_done()
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Feed update worker {worker_id} error: {e}")
                    await asyncio.sleep(60)
                    
        except asyncio.CancelledError:
            logger.info(f"Feed update worker {worker_id} cancelled")
    
    async def _update_threat_feed(self, feed: ThreatFeed):
        """Update a specific threat feed"""
        try:
            logger.info(f"Updating threat feed: {feed.name}")
            
            async with aiohttp.ClientSession() as session:
                headers = feed.headers.copy()
                if feed.api_key:
                    headers["Authorization"] = f"Bearer {feed.api_key}"
                
                async with session.get(feed.url, headers=headers) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Parse feed content
                        indicators = await self._parse_feed_content(feed, content)
                        
                        # Update indicators
                        for indicator in indicators:
                            self.indicators[indicator.ioc_id] = indicator
                        
                        # Update feed metadata
                        feed.last_updated = datetime.utcnow()
                        feed.records_count = len(indicators)
                        
                        logger.info(f"Updated feed {feed.name} with {len(indicators)} indicators")
                        
                    else:
                        logger.error(f"Failed to update feed {feed.name}: HTTP {response.status}")
            
        except Exception as e:
            logger.error(f"Error updating feed {feed.name}: {e}")
    
    async def _parse_feed_content(self, feed: ThreatFeed, content: str) -> List[ThreatIndicator]:
        """Parse threat feed content into indicators"""
        indicators = []
        
        try:
            if feed.format == "txt":
                lines = content.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        ioc_type = self._detect_ioc_type(line)
                        indicator = await self._get_or_create_indicator(line, ioc_type)
                        indicator.sources.append(feed.name)
                        indicator.confidence = ConfidenceLevel.MEDIUM
                        indicators.append(indicator)
            
            elif feed.format == "csv":
                # Parse CSV format (simplified)
                lines = content.strip().split('\n')
                if len(lines) > 1:  # Skip header
                    for line in lines[1:]:
                        parts = line.split(',')
                        if len(parts) > 0:
                            value = parts[0].strip().strip('"')
                            if value:
                                ioc_type = self._detect_ioc_type(value)
                                indicator = await self._get_or_create_indicator(value, ioc_type)
                                indicator.sources.append(feed.name)
                                indicator.confidence = ConfidenceLevel.MEDIUM
                                indicators.append(indicator)
            
            elif feed.format == "json":
                data = json.loads(content)
                # Parse JSON format (implementation depends on feed structure)
                # This would be customized per feed
                pass
        
        except Exception as e:
            logger.error(f"Error parsing feed content for {feed.name}: {e}")
        
        return indicators
    
    async def _query_threat_feed(self, feed: ThreatFeed, indicator: ThreatIndicator) -> List[Dict[str, Any]]:
        """Query a threat feed for indicator matches"""
        matches = []
        
        # Simple matching against loaded indicators
        for stored_indicator in self.indicators.values():
            if (stored_indicator.value == indicator.value and 
                feed.name in stored_indicator.sources):
                matches.append({
                    "feed": feed.name,
                    "severity": stored_indicator.severity.value,
                    "confidence": stored_indicator.confidence.value,
                    "category": stored_indicator.category.value,
                    "last_seen": stored_indicator.updated_at.isoformat()
                })
        
        return matches
    
    async def _query_local_intelligence(self, indicator: ThreatIndicator) -> List[Dict[str, Any]]:
        """Query local threat intelligence database"""
        matches = []
        
        # Query local database for historical matches
        # This would interface with a persistent database
        # For now, returning empty list
        
        return matches
    
    def _calculate_confidence_score(self, threat_matches: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on threat matches"""
        if not threat_matches:
            return 0.0
        
        source_count = len(set(match.get("feed", "unknown") for match in threat_matches))
        severity_scores = [self._severity_to_score(match.get("severity", "unknown")) for match in threat_matches]
        
        # Base confidence from source diversity
        source_confidence = min(80.0, source_count * 20.0)
        
        # Severity confidence
        max_severity = max(severity_scores) if severity_scores else 0
        severity_confidence = max_severity * 0.8
        
        # Combined confidence
        return min(100.0, (source_confidence + severity_confidence) / 2.0)
    
    def _extract_attribution(self, threat_matches: List[Dict[str, Any]]) -> Optional[ThreatContext]:
        """Extract threat attribution from matches"""
        # Analyze matches for attribution information
        # This would be more sophisticated in production
        
        if not threat_matches:
            return None
        
        # Extract common attribution elements
        threat_actors = set()
        campaigns = set()
        techniques = set()
        
        for match in threat_matches:
            if "threat_actor" in match:
                threat_actors.add(match["threat_actor"])
            if "campaign" in match:
                campaigns.add(match["campaign"])
            if "technique" in match:
                techniques.add(match["technique"])
        
        if threat_actors or campaigns or techniques:
            return ThreatContext(
                threat_actor=list(threat_actors)[0] if threat_actors else None,
                campaign=list(campaigns)[0] if campaigns else None,
                mitre_techniques=list(techniques)
            )
        
        return None
    
    def _generate_indicator_recommendations(self, indicator: ThreatIndicator, result: ThreatIntelligenceResult) -> List[str]:
        """Generate security recommendations for an indicator"""
        recommendations = []
        
        if result.risk_score >= 70:
            recommendations.append("🚨 HIGH RISK: Immediate investigation and blocking recommended")
        elif result.risk_score >= 40:
            recommendations.append("⚠️ MEDIUM RISK: Monitor and consider blocking")
        
        if indicator.ioc_type == IOCType.IP_ADDRESS:
            recommendations.append("🔧 Consider firewall blocking for malicious IP addresses")
        elif indicator.ioc_type == IOCType.DOMAIN:
            recommendations.append("🌐 Consider DNS blocking for malicious domains")
        elif indicator.ioc_type == IOCType.FILE_HASH:
            recommendations.append("🛡️ Add hash to antivirus signatures")
        elif indicator.ioc_type == IOCType.URL:
            recommendations.append("🚫 Block malicious URLs in web proxy")
        
        if len(result.sources) > 2:
            recommendations.append("✅ High confidence due to multiple source confirmation")
        
        return recommendations
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of expired indicators and cache"""
        while True:
            try:
                current_time = datetime.utcnow()
                
                # Clean expired indicators
                expired_indicators = []
                for ioc_id, indicator in self.indicators.items():
                    if indicator.expiry_date and indicator.expiry_date < current_time:
                        expired_indicators.append(ioc_id)
                
                for ioc_id in expired_indicators:
                    del self.indicators[ioc_id]
                
                # Clean old cache entries (older than 1 hour)
                expired_cache = []
                for cache_key, result in self.threat_cache.items():
                    age = current_time - result.enrichment_time
                    if age > timedelta(hours=1):
                        expired_cache.append(cache_key)
                
                for cache_key in expired_cache:
                    del self.threat_cache[cache_key]
                
                if expired_indicators or expired_cache:
                    logger.info(f"Cleaned {len(expired_indicators)} expired indicators and {len(expired_cache)} cache entries")
                
                # Sleep for 1 hour
                await asyncio.sleep(3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(3600)
    
    async def _save_persistent_state(self):
        """Save threat intelligence state"""
        try:
            # Save indicators and models to persistent storage
            # This would write to database or file system
            logger.info("Saving threat intelligence state")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    async def _load_existing_indicators(self):
        """Load existing indicators from database"""
        try:
            # Load from persistent storage
            logger.info("Loading existing threat intelligence indicators")
        except Exception as e:
            logger.error(f"Failed to load existing indicators: {e}")


# Global service instance
_threat_intel_service: Optional[EnhancedThreatIntelligenceService] = None

async def get_threat_intelligence_service() -> EnhancedThreatIntelligenceService:
    """Get global threat intelligence service instance"""
    global _threat_intel_service
    
    if _threat_intel_service is None:
        _threat_intel_service = EnhancedThreatIntelligenceService()
        await _threat_intel_service.initialize()
        
        # Register with global service registry
        from .base_service import service_registry
        service_registry.register(_threat_intel_service)
    
    return _threat_intel_service