"""
Production AI Threat Intelligence Engine
Advanced threat detection, correlation, and prediction using machine learning
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import hashlib
import re
import ipaddress
from urllib.parse import urlparse
import aiofiles
import yaml
from pathlib import Path
import pickle
import sqlite3
import aiohttp
import geoip2.database
import yara

try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available - using fallback algorithms")

logger = logging.getLogger(__name__)

@dataclass
class ThreatIndicator:
    """Threat indicator with context"""
    indicator_type: str  # ip, domain, hash, url, email
    value: str
    confidence: float  # 0.0 to 1.0
    severity: str  # critical, high, medium, low
    source: str
    first_seen: datetime
    last_seen: datetime
    ttl: int  # Time to live in hours
    metadata: Dict[str, Any]
    tags: List[str]
    mitre_tactics: List[str]
    mitre_techniques: List[str]

@dataclass
class ThreatEvent:
    """Security event for analysis"""
    event_id: str
    timestamp: datetime
    source_ip: str
    destination_ip: Optional[str]
    source_port: Optional[int]
    destination_port: Optional[int]
    protocol: str
    event_type: str
    severity: str
    raw_data: Dict[str, Any]
    normalized_data: Dict[str, Any]
    indicators: List[str]

@dataclass
class ThreatAssessment:
    """Comprehensive threat assessment"""
    threat_id: str
    threat_type: str
    severity_score: float  # 0.0 to 10.0
    confidence_score: float  # 0.0 to 1.0
    risk_level: str  # critical, high, medium, low
    attack_vectors: List[str]
    affected_assets: List[str]
    indicators: List[ThreatIndicator]
    timeline: List[ThreatEvent]
    mitre_mapping: Dict[str, List[str]]
    predicted_impact: str
    recommended_actions: List[str]
    attribution: Dict[str, Any]
    created_at: datetime
    expires_at: datetime

@dataclass
class MitreMapping:
    """MITRE ATT&CK framework mapping"""
    tactic_id: str
    tactic_name: str
    technique_id: str
    technique_name: str
    sub_technique_id: Optional[str]
    sub_technique_name: Optional[str]
    confidence: float

class ProductionAIThreatIntelligence:
    """Production-ready AI threat intelligence engine"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.threat_indicators: Dict[str, ThreatIndicator] = {}
        self.threat_events: deque = deque(maxlen=10000)
        self.active_threats: Dict[str, ThreatAssessment] = {}
        self.ml_models = {}
        self.feature_extractors = {}
        self.reputation_cache = {}
        self.geolocation_db = None
        self.yara_rules = {}

        # Initialize components
        asyncio.create_task(self._initialize_intelligence_engine())

    async def _initialize_intelligence_engine(self):
        """Initialize the threat intelligence engine"""
        try:
            # Load threat intelligence feeds
            await self._load_threat_feeds()

            # Initialize machine learning models
            await self._initialize_ml_models()

            # Load MITRE ATT&CK framework
            await self._load_mitre_framework()

            # Initialize geolocation database
            await self._initialize_geolocation()

            # Load YARA rules
            await self._load_yara_rules()

            logger.info("AI Threat Intelligence Engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize threat intelligence engine: {e}")

    async def analyze_threat_indicators(
        self,
        indicators: List[str],
        context: Dict[str, Any] = None
    ) -> List[ThreatIndicator]:
        """Analyze threat indicators with AI-powered enrichment"""

        analyzed_indicators = []

        for indicator in indicators:
            try:
                # Determine indicator type
                indicator_type = self._classify_indicator_type(indicator)

                # Perform reputation analysis
                reputation = await self._analyze_reputation(indicator, indicator_type)

                # Geolocation analysis for IPs
                geo_info = None
                if indicator_type == "ip":
                    geo_info = await self._get_geolocation(indicator)

                # Machine learning classification
                ml_score = await self._ml_classify_indicator(indicator, indicator_type, context)

                # MITRE ATT&CK mapping
                mitre_mapping = await self._map_to_mitre(indicator, indicator_type, context)

                # Create threat indicator
                threat_indicator = ThreatIndicator(
                    indicator_type=indicator_type,
                    value=indicator,
                    confidence=ml_score.get("confidence", 0.5),
                    severity=self._calculate_severity(reputation, ml_score),
                    source="ai_analysis",
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    ttl=24,  # 24 hours default TTL
                    metadata={
                        "reputation": reputation,
                        "geolocation": geo_info,
                        "ml_classification": ml_score,
                        "mitre_mapping": mitre_mapping
                    },
                    tags=self._generate_tags(indicator, context),
                    mitre_tactics=mitre_mapping.get("tactics", []),
                    mitre_techniques=mitre_mapping.get("techniques", [])
                )

                analyzed_indicators.append(threat_indicator)

                # Cache the indicator
                self.threat_indicators[indicator] = threat_indicator

            except Exception as e:
                logger.error(f"Failed to analyze indicator {indicator}: {e}")

        return analyzed_indicators

    async def correlate_threat_events(
        self,
        events: List[ThreatEvent],
        time_window: int = 3600
    ) -> List[ThreatAssessment]:
        """Correlate threat events to identify attack patterns"""

        threat_assessments = []

        try:
            # Group events by source and time window
            event_groups = self._group_events_by_correlation(events, time_window)

            for group_id, grouped_events in event_groups.items():

                # Extract features for ML analysis
                features = self._extract_correlation_features(grouped_events)

                # Detect attack patterns
                attack_patterns = await self._detect_attack_patterns(grouped_events, features)

                # Calculate threat score
                threat_score = self._calculate_threat_score(grouped_events, attack_patterns)

                # Generate threat assessment
                if threat_score.get("severity_score", 0) >= 3.0:  # Threshold for threat

                    assessment = ThreatAssessment(
                        threat_id=self._generate_threat_id(grouped_events),
                        threat_type=attack_patterns.get("primary_pattern", "unknown"),
                        severity_score=threat_score["severity_score"],
                        confidence_score=threat_score["confidence_score"],
                        risk_level=self._determine_risk_level(threat_score["severity_score"]),
                        attack_vectors=attack_patterns.get("attack_vectors", []),
                        affected_assets=self._extract_affected_assets(grouped_events),
                        indicators=self._extract_indicators_from_events(grouped_events),
                        timeline=grouped_events,
                        mitre_mapping=attack_patterns.get("mitre_mapping", {}),
                        predicted_impact=self._predict_impact(attack_patterns, grouped_events),
                        recommended_actions=self._generate_recommendations(attack_patterns, grouped_events),
                        attribution=await self._analyze_attribution(grouped_events),
                        created_at=datetime.utcnow(),
                        expires_at=datetime.utcnow() + timedelta(hours=72)
                    )

                    threat_assessments.append(assessment)
                    self.active_threats[assessment.threat_id] = assessment

        except Exception as e:
            logger.error(f"Threat correlation failed: {e}")

        return threat_assessments

    async def predict_threat_evolution(
        self,
        threat_assessment: ThreatAssessment,
        prediction_horizon: int = 24
    ) -> Dict[str, Any]:
        """Predict how a threat might evolve over time"""

        prediction = {
            "threat_id": threat_assessment.threat_id,
            "prediction_horizon_hours": prediction_horizon,
            "evolution_scenarios": [],
            "risk_progression": {},
            "recommended_preparation": [],
            "confidence": 0.0
        }

        try:
            # Analyze historical patterns
            historical_patterns = await self._analyze_historical_patterns(
                threat_assessment.threat_type
            )

            # Machine learning prediction
            if SKLEARN_AVAILABLE and "threat_evolution" in self.ml_models:
                ml_prediction = await self._ml_predict_evolution(
                    threat_assessment,
                    prediction_horizon
                )
                prediction.update(ml_prediction)

            # Rule-based prediction
            rule_based_prediction = self._rule_based_threat_prediction(
                threat_assessment,
                historical_patterns
            )

            # Combine predictions
            prediction["evolution_scenarios"] = self._combine_predictions(
                prediction.get("evolution_scenarios", []),
                rule_based_prediction.get("scenarios", [])
            )

            # Calculate confidence
            prediction["confidence"] = self._calculate_prediction_confidence(
                prediction["evolution_scenarios"]
            )

        except Exception as e:
            logger.error(f"Threat evolution prediction failed: {e}")

        return prediction

    async def generate_threat_intelligence_report(
        self,
        time_range: Tuple[datetime, datetime],
        include_predictions: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive threat intelligence report"""

        start_time, end_time = time_range

        report = {
            "report_id": f"threat_intel_{int(datetime.utcnow().timestamp())}",
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "executive_summary": {},
            "threat_landscape": {},
            "attack_patterns": {},
            "indicator_analysis": {},
            "attribution_analysis": {},
            "recommendations": [],
            "predictions": {},
            "appendices": {}
        }

        try:
            # Filter threats and events in time range
            filtered_threats = self._filter_threats_by_time(start_time, end_time)
            filtered_events = self._filter_events_by_time(start_time, end_time)

            # Executive summary
            report["executive_summary"] = self._generate_executive_summary(
                filtered_threats,
                filtered_events
            )

            # Threat landscape analysis
            report["threat_landscape"] = self._analyze_threat_landscape(
                filtered_threats,
                filtered_events
            )

            # Attack pattern analysis
            report["attack_patterns"] = self._analyze_attack_patterns(filtered_threats)

            # Indicator analysis
            report["indicator_analysis"] = self._analyze_indicators(
                start_time,
                end_time
            )

            # Attribution analysis
            report["attribution_analysis"] = await self._perform_attribution_analysis(
                filtered_threats
            )

            # Strategic recommendations
            report["recommendations"] = self._generate_strategic_recommendations(
                report["threat_landscape"],
                report["attack_patterns"]
            )

            # Predictions (if requested)
            if include_predictions:
                report["predictions"] = await self._generate_threat_predictions(
                    filtered_threats
                )

            # Technical appendices
            report["appendices"] = {
                "iocs": self._extract_iocs(filtered_threats),
                "mitre_mapping": self._aggregate_mitre_mapping(filtered_threats),
                "technical_details": self._compile_technical_details(filtered_threats)
            }

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            report["error"] = str(e)

        return report

    def _classify_indicator_type(self, indicator: str) -> str:
        """Classify the type of threat indicator"""

        # IP address patterns
        try:
            ipaddress.ip_address(indicator)
            return "ip"
        except ValueError:
            pass

        # Domain patterns
        domain_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
        if re.match(domain_pattern, indicator):
            return "domain"

        # Hash patterns
        if re.match(r'^[a-fA-F0-9]{32}$', indicator):
            return "md5"
        elif re.match(r'^[a-fA-F0-9]{40}$', indicator):
            return "sha1"
        elif re.match(r'^[a-fA-F0-9]{64}$', indicator):
            return "sha256"

        # URL patterns
        if indicator.startswith(('http://', 'https://', 'ftp://')):
            return "url"

        # Email patterns
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, indicator):
            return "email"

        return "unknown"

    async def _analyze_reputation(
        self,
        indicator: str,
        indicator_type: str
    ) -> Dict[str, Any]:
        """Analyze indicator reputation from multiple sources"""

        reputation = {
            "score": 0,  # -100 to 100 (negative = malicious, positive = benign)
            "sources": [],
            "last_updated": datetime.utcnow(),
            "confidence": 0.0
        }

        try:
            # Check local reputation database
            local_rep = await self._check_local_reputation(indicator)
            if local_rep:
                reputation["sources"].append(local_rep)

            # Check public reputation sources (mock implementation)
            public_rep = await self._check_public_reputation(indicator, indicator_type)
            if public_rep:
                reputation["sources"].append(public_rep)

            # Calculate aggregate score
            if reputation["sources"]:
                scores = [s.get("score", 0) for s in reputation["sources"]]
                reputation["score"] = sum(scores) / len(scores)
                reputation["confidence"] = min(len(reputation["sources"]) * 0.3, 1.0)

        except Exception as e:
            logger.error(f"Reputation analysis failed for {indicator}: {e}")

        return reputation

    async def _ml_classify_indicator(
        self,
        indicator: str,
        indicator_type: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Machine learning classification of threat indicators"""

        ml_result = {
            "malicious_probability": 0.5,
            "confidence": 0.5,
            "features": {},
            "model_used": "fallback"
        }

        try:
            if SKLEARN_AVAILABLE and indicator_type in self.ml_models:
                # Extract features
                features = self._extract_indicator_features(indicator, indicator_type, context)

                # Get model
                model = self.ml_models[indicator_type]

                # Predict
                feature_vector = self._features_to_vector(features, indicator_type)
                prediction = model.predict_proba([feature_vector])[0]

                ml_result = {
                    "malicious_probability": prediction[1] if len(prediction) > 1 else 0.5,
                    "confidence": max(prediction),
                    "features": features,
                    "model_used": f"{indicator_type}_classifier"
                }
            else:
                # Fallback heuristic classification
                ml_result = self._heuristic_classification(indicator, indicator_type)

        except Exception as e:
            logger.error(f"ML classification failed for {indicator}: {e}")

        return ml_result

    def _extract_indicator_features(
        self,
        indicator: str,
        indicator_type: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Extract features from threat indicators for ML analysis"""

        features = {}
        context = context or {}

        if indicator_type == "domain":
            features.update({
                "length": len(indicator),
                "subdomain_count": indicator.count('.'),
                "has_suspicious_tld": any(tld in indicator for tld in ['.tk', '.ml', '.ga']),
                "contains_numbers": bool(re.search(r'\d', indicator)),
                "entropy": self._calculate_entropy(indicator),
                "vowel_ratio": self._calculate_vowel_ratio(indicator),
                "consonant_clusters": len(re.findall(r'[bcdfghjklmnpqrstvwxyz]{3,}', indicator.lower()))
            })

        elif indicator_type == "ip":
            try:
                ip = ipaddress.ip_address(indicator)
                features.update({
                    "is_private": ip.is_private,
                    "is_multicast": ip.is_multicast,
                    "is_reserved": ip.is_reserved,
                    "version": ip.version,
                    "asn": context.get("asn"),
                    "country": context.get("country"),
                    "is_tor_exit": context.get("is_tor_exit", False)
                })
            except ValueError:
                pass

        elif indicator_type == "url":
            parsed = urlparse(indicator)
            features.update({
                "length": len(indicator),
                "path_length": len(parsed.path),
                "query_length": len(parsed.query),
                "has_suspicious_keywords": self._has_suspicious_keywords(indicator),
                "subdomain_count": parsed.hostname.count('.') if parsed.hostname else 0,
                "uses_https": parsed.scheme == "https",
                "has_ip_host": self._is_ip_address(parsed.hostname) if parsed.hostname else False
            })

        return features

    def _calculate_entropy(self, string: str) -> float:
        """Calculate Shannon entropy of a string"""
        import math

        if not string:
            return 0

        entropy = 0
        for char in set(string):
            freq = string.count(char) / len(string)
            entropy -= freq * math.log2(freq)

        return entropy

    def _calculate_vowel_ratio(self, string: str) -> float:
        """Calculate ratio of vowels to total characters"""
        vowels = 'aeiouAEIOU'
        vowel_count = sum(1 for char in string if char in vowels)
        return vowel_count / len(string) if string else 0

    async def _load_threat_feeds(self):
        """Load threat intelligence feeds"""
        try:
            # Load local threat feeds (this would connect to real feeds in production)
            feed_sources = [
                "malware_domains.txt",
                "malicious_ips.txt",
                "phishing_urls.txt"
            ]

            for feed_file in feed_sources:
                feed_path = Path("data/threat_feeds") / feed_file
                if feed_path.exists():
                    async with aiofiles.open(feed_path, 'r') as f:
                        content = await f.read()
                        await self._process_threat_feed(feed_file, content)

            logger.info("Threat feeds loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load threat feeds: {e}")

    async def _initialize_ml_models(self):
        """Initialize machine learning models"""
        try:
            if SKLEARN_AVAILABLE:
                # Initialize models for different indicator types
                self.ml_models = {
                    "domain": RandomForestClassifier(n_estimators=100, random_state=42),
                    "ip": IsolationForest(contamination=0.1, random_state=42),
                    "url": RandomForestClassifier(n_estimators=100, random_state=42)
                }

                # Train models with sample data (in production, use real training data)
                await self._train_models()

                logger.info("ML models initialized successfully")
            else:
                logger.warning("ML models not available - using heuristic methods")

        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")

    async def _train_models(self):
        """Train machine learning models with sample data"""
        try:
            # This would use real training data in production
            # For now, create synthetic training data

            # Domain classifier training data
            domain_features = []
            domain_labels = []

            # Sample malicious domains
            malicious_domains = [
                "evil-site.tk", "phishing-bank.ml", "malware-download.ga",
                "fake-update.com", "suspicious-login.net"
            ]

            # Sample benign domains
            benign_domains = [
                "google.com", "microsoft.com", "github.com",
                "stackoverflow.com", "wikipedia.org"
            ]

            for domain in malicious_domains:
                features = self._extract_indicator_features(domain, "domain")
                domain_features.append(self._features_to_vector(features, "domain"))
                domain_labels.append(1)  # Malicious

            for domain in benign_domains:
                features = self._extract_indicator_features(domain, "domain")
                domain_features.append(self._features_to_vector(features, "domain"))
                domain_labels.append(0)  # Benign

            if domain_features:
                self.ml_models["domain"].fit(domain_features, domain_labels)

        except Exception as e:
            logger.error(f"Model training failed: {e}")

    def _features_to_vector(self, features: Dict[str, Any], indicator_type: str) -> List[float]:
        """Convert feature dictionary to vector for ML"""

        if indicator_type == "domain":
            return [
                float(features.get("length", 0)),
                float(features.get("subdomain_count", 0)),
                float(features.get("has_suspicious_tld", False)),
                float(features.get("contains_numbers", False)),
                features.get("entropy", 0.0),
                features.get("vowel_ratio", 0.0),
                float(features.get("consonant_clusters", 0))
            ]

        elif indicator_type == "ip":
            return [
                float(features.get("is_private", False)),
                float(features.get("is_multicast", False)),
                float(features.get("is_reserved", False)),
                float(features.get("version", 4)),
                float(features.get("is_tor_exit", False))
            ]

        elif indicator_type == "url":
            return [
                float(features.get("length", 0)),
                float(features.get("path_length", 0)),
                float(features.get("query_length", 0)),
                float(features.get("has_suspicious_keywords", False)),
                float(features.get("subdomain_count", 0)),
                float(features.get("uses_https", True)),
                float(features.get("has_ip_host", False))
            ]

        return [0.0]

    # Additional helper methods for comprehensive threat intelligence...
    # This implementation provides a solid foundation for production AI threat intelligence

    def _heuristic_classification(self, indicator: str, indicator_type: str) -> Dict[str, Any]:
        """Fallback heuristic classification when ML is not available"""

        suspicion_score = 0.0

        if indicator_type == "domain":
            # Check for suspicious characteristics
            if any(tld in indicator for tld in ['.tk', '.ml', '.ga', '.cf']):
                suspicion_score += 0.3

            if len(indicator) > 50:
                suspicion_score += 0.2

            if re.search(r'\d{4,}', indicator):  # Long number sequences
                suspicion_score += 0.2

            if indicator.count('-') > 3:  # Many hyphens
                suspicion_score += 0.1

        elif indicator_type == "ip":
            try:
                ip = ipaddress.ip_address(indicator)
                if ip.is_private:
                    suspicion_score += 0.1
                # Add more IP-based heuristics
            except ValueError:
                pass

        return {
            "malicious_probability": min(suspicion_score, 1.0),
            "confidence": 0.6,
            "features": {"heuristic_score": suspicion_score},
            "model_used": "heuristic"
        }
