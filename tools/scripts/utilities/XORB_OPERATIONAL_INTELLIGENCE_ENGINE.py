#!/usr/bin/env python3
"""
🧠 XORB Operational Intelligence Engine
Advanced tactical intelligence analysis and decision-making system

This engine provides sophisticated intelligence analysis, predictive threat modeling,
and automated operational decision-making for the XORB platform.
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import pickle
from pathlib import Path
import hashlib
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EXTREME = "extreme"

class IntelligenceType(Enum):
    TACTICAL = "tactical"
    STRATEGIC = "strategic"
    OPERATIONAL = "operational"
    TECHNICAL = "technical"
    PREDICTIVE = "predictive"

class DecisionConfidence(Enum):
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.8
    MAXIMUM = 0.95

@dataclass
class ThreatIntelligence:
    id: str
    type: IntelligenceType
    threat_level: ThreatLevel
    confidence: float
    source: str
    timestamp: datetime
    data: Dict[str, Any]
    indicators: List[str]
    attribution: Optional[str] = None
    ttps: List[str] = None
    prediction_horizon: Optional[int] = None

@dataclass
class OperationalDecision:
    decision_id: str
    decision_type: str
    action: str
    priority: int
    confidence: DecisionConfidence
    reasoning: str
    required_resources: List[str]
    estimated_impact: float
    timestamp: datetime
    execution_window: timedelta
    dependencies: List[str] = None

@dataclass
class IntelligenceCorrelation:
    correlation_id: str
    related_intel: List[str]
    correlation_strength: float
    pattern_type: str
    insights: List[str]
    recommended_actions: List[str]
    timestamp: datetime

class XORBOperationalIntelligenceEngine:
    """Advanced operational intelligence and decision-making engine"""

    def __init__(self):
        self.engine_id = f"INTEL-ENGINE-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.intelligence_db = {}
        self.decision_history = []
        self.correlation_cache = {}
        self.predictive_models = {}
        self.threat_patterns = {}

        # Intelligence analysis parameters
        self.correlation_threshold = 0.7
        self.prediction_accuracy_target = 0.85
        self.decision_timeout = 300  # 5 minutes

        # Operational metrics
        self.metrics = {
            "intelligence_processed": 0,
            "decisions_made": 0,
            "correlations_identified": 0,
            "predictions_generated": 0,
            "accuracy_score": 0.0
        }

        logger.info(f"🧠 XORB Operational Intelligence Engine initialized - ID: {self.engine_id}")

    async def process_threat_intelligence(self, raw_intelligence: Dict[str, Any]) -> ThreatIntelligence:
        """Process and enrich raw threat intelligence"""
        try:
            # Extract and validate intelligence data
            intel_id = self._generate_intel_id(raw_intelligence)

            # Classify intelligence type
            intel_type = await self._classify_intelligence_type(raw_intelligence)

            # Assess threat level
            threat_level = await self._assess_threat_level(raw_intelligence)

            # Calculate confidence score
            confidence = await self._calculate_confidence(raw_intelligence)

            # Extract indicators
            indicators = await self._extract_indicators(raw_intelligence)

            # Perform attribution analysis
            attribution = await self._perform_attribution(raw_intelligence)

            # Map to MITRE ATT&CK TTPs
            ttps = await self._map_to_attack_framework(raw_intelligence)

            # Create intelligence object
            intelligence = ThreatIntelligence(
                id=intel_id,
                type=intel_type,
                threat_level=threat_level,
                confidence=confidence,
                source=raw_intelligence.get("source", "unknown"),
                timestamp=datetime.now(),
                data=raw_intelligence,
                indicators=indicators,
                attribution=attribution,
                ttps=ttps,
                prediction_horizon=await self._calculate_prediction_horizon(raw_intelligence)
            )

            # Store intelligence
            self.intelligence_db[intel_id] = intelligence

            # Update metrics
            self.metrics["intelligence_processed"] += 1

            logger.info(f"🧠 Processed intelligence: {intel_id} | Type: {intel_type.value} | Threat: {threat_level.value}")

            return intelligence

        except Exception as e:
            logger.error(f"❌ Intelligence processing error: {e}")
            raise

    async def correlate_intelligence(self, intelligence_ids: List[str]) -> List[IntelligenceCorrelation]:
        """Correlate multiple intelligence items to identify patterns"""
        try:
            correlations = []

            for i, intel_id_1 in enumerate(intelligence_ids):
                for intel_id_2 in intelligence_ids[i+1:]:
                    correlation = await self._analyze_correlation(intel_id_1, intel_id_2)
                    if correlation and correlation.correlation_strength >= self.correlation_threshold:
                        correlations.append(correlation)

            # Identify complex patterns
            complex_correlations = await self._identify_complex_patterns(correlations)
            correlations.extend(complex_correlations)

            # Update metrics
            self.metrics["correlations_identified"] += len(correlations)

            logger.info(f"🔗 Identified {len(correlations)} intelligence correlations")

            return correlations

        except Exception as e:
            logger.error(f"❌ Intelligence correlation error: {e}")
            return []

    async def generate_operational_decision(self, context: Dict[str, Any]) -> OperationalDecision:
        """Generate tactical operational decisions based on intelligence analysis"""
        try:
            # Analyze current threat landscape
            threat_assessment = await self._analyze_threat_landscape(context)

            # Evaluate available resources
            resource_assessment = await self._evaluate_resources(context)

            # Generate decision options
            decision_options = await self._generate_decision_options(threat_assessment, resource_assessment)

            # Select optimal decision
            optimal_decision = await self._select_optimal_decision(decision_options, context)

            # Calculate decision confidence
            confidence = await self._calculate_decision_confidence(optimal_decision, context)

            # Create decision object
            decision = OperationalDecision(
                decision_id=f"DECISION-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                decision_type=optimal_decision["type"],
                action=optimal_decision["action"],
                priority=optimal_decision["priority"],
                confidence=confidence,
                reasoning=optimal_decision["reasoning"],
                required_resources=optimal_decision["resources"],
                estimated_impact=optimal_decision["impact"],
                timestamp=datetime.now(),
                execution_window=timedelta(seconds=optimal_decision["execution_time"]),
                dependencies=optimal_decision.get("dependencies", [])
            )

            # Store decision
            self.decision_history.append(decision)

            # Update metrics
            self.metrics["decisions_made"] += 1

            logger.info(f"🎯 Generated operational decision: {decision.decision_id} | Action: {decision.action}")

            return decision

        except Exception as e:
            logger.error(f"❌ Decision generation error: {e}")
            raise

    async def predict_threat_evolution(self, intelligence_data: List[ThreatIntelligence]) -> Dict[str, Any]:
        """Predict how threats will evolve based on current intelligence"""
        try:
            # Prepare prediction data
            prediction_data = await self._prepare_prediction_data(intelligence_data)

            # Apply predictive models
            predictions = {}

            # Short-term predictions (1-7 days)
            predictions["short_term"] = await self._predict_short_term_threats(prediction_data)

            # Medium-term predictions (1-4 weeks)
            predictions["medium_term"] = await self._predict_medium_term_threats(prediction_data)

            # Long-term predictions (1-3 months)
            predictions["long_term"] = await self._predict_long_term_threats(prediction_data)

            # Campaign evolution predictions
            predictions["campaign_evolution"] = await self._predict_campaign_evolution(prediction_data)

            # Threat actor behavior predictions
            predictions["actor_behavior"] = await self._predict_actor_behavior(prediction_data)

            # Update metrics
            self.metrics["predictions_generated"] += 1

            logger.info(f"🔮 Generated threat evolution predictions with {len(predictions)} categories")

            return predictions

        except Exception as e:
            logger.error(f"❌ Threat prediction error: {e}")
            return {}

    async def optimize_defensive_posture(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize defensive configurations based on intelligence analysis"""
        try:
            # Analyze current defensive posture
            posture_analysis = await self._analyze_defensive_posture(current_state)

            # Identify vulnerabilities and gaps
            vulnerabilities = await self._identify_defensive_gaps(posture_analysis)

            # Generate optimization recommendations
            optimizations = await self._generate_optimization_recommendations(vulnerabilities)

            # Prioritize optimizations
            prioritized_optimizations = await self._prioritize_optimizations(optimizations)

            # Calculate implementation impact
            impact_analysis = await self._calculate_implementation_impact(prioritized_optimizations)

            optimization_plan = {
                "analysis_timestamp": datetime.now().isoformat(),
                "current_posture_score": posture_analysis["score"],
                "identified_gaps": len(vulnerabilities),
                "optimization_recommendations": prioritized_optimizations,
                "implementation_timeline": impact_analysis["timeline"],
                "expected_improvement": impact_analysis["improvement"],
                "resource_requirements": impact_analysis["resources"]
            }

            logger.info(f"🛡️ Generated defensive optimization plan with {len(prioritized_optimizations)} recommendations")

            return optimization_plan

        except Exception as e:
            logger.error(f"❌ Defensive optimization error: {e}")
            return {}

    async def _classify_intelligence_type(self, raw_intel: Dict[str, Any]) -> IntelligenceType:
        """Classify the type of intelligence"""
        content = str(raw_intel).lower()

        # Technical intelligence indicators
        if any(term in content for term in ["hash", "ip", "domain", "malware", "vulnerability"]):
            return IntelligenceType.TECHNICAL

        # Tactical intelligence indicators
        if any(term in content for term in ["ttp", "technique", "tactic", "attack"]):
            return IntelligenceType.TACTICAL

        # Strategic intelligence indicators
        if any(term in content for term in ["campaign", "apt", "group", "nation-state"]):
            return IntelligenceType.STRATEGIC

        # Predictive intelligence indicators
        if any(term in content for term in ["trend", "forecast", "prediction", "emerging"]):
            return IntelligenceType.PREDICTIVE

        return IntelligenceType.OPERATIONAL

    async def _assess_threat_level(self, raw_intel: Dict[str, Any]) -> ThreatLevel:
        """Assess the threat level of intelligence"""
        threat_score = 0
        content = str(raw_intel).lower()

        # Critical indicators
        critical_terms = ["zero-day", "apt", "nation-state", "critical", "emergency"]
        threat_score += sum(5 for term in critical_terms if term in content)

        # High severity indicators
        high_terms = ["exploit", "backdoor", "ransomware", "breach", "compromise"]
        threat_score += sum(3 for term in high_terms if term in content)

        # Medium severity indicators
        medium_terms = ["vulnerability", "malware", "phishing", "suspicious"]
        threat_score += sum(2 for term in medium_terms if term in content)

        # Low severity indicators
        low_terms = ["scan", "reconnaissance", "probe"]
        threat_score += sum(1 for term in low_terms if term in content)

        if threat_score >= 15:
            return ThreatLevel.EXTREME
        elif threat_score >= 10:
            return ThreatLevel.CRITICAL
        elif threat_score >= 6:
            return ThreatLevel.HIGH
        elif threat_score >= 3:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW

    async def _calculate_confidence(self, raw_intel: Dict[str, Any]) -> float:
        """Calculate confidence score for intelligence"""
        confidence_score = 0.5  # Base confidence

        # Source reliability
        source = raw_intel.get("source", "").lower()
        if "government" in source or "official" in source:
            confidence_score += 0.3
        elif "commercial" in source or "vendor" in source:
            confidence_score += 0.2
        elif "community" in source or "osint" in source:
            confidence_score += 0.1

        # Data completeness
        data_fields = len([v for v in raw_intel.values() if v])
        if data_fields >= 10:
            confidence_score += 0.15
        elif data_fields >= 5:
            confidence_score += 0.1

        # Corroboration
        if raw_intel.get("corroborated", False):
            confidence_score += 0.2

        return min(confidence_score, 1.0)

    async def _extract_indicators(self, raw_intel: Dict[str, Any]) -> List[str]:
        """Extract indicators of compromise from intelligence"""
        indicators = []
        content = str(raw_intel)

        # IP address patterns
        import re
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        indicators.extend(re.findall(ip_pattern, content))

        # Domain patterns
        domain_pattern = r'\b[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}\b'
        indicators.extend(re.findall(domain_pattern, content))

        # Hash patterns (MD5, SHA1, SHA256)
        hash_patterns = [
            r'\b[a-fA-F0-9]{32}\b',  # MD5
            r'\b[a-fA-F0-9]{40}\b',  # SHA1
            r'\b[a-fA-F0-9]{64}\b'   # SHA256
        ]
        for pattern in hash_patterns:
            indicators.extend(re.findall(pattern, content))

        return list(set(indicators))  # Remove duplicates

    async def _perform_attribution(self, raw_intel: Dict[str, Any]) -> Optional[str]:
        """Perform threat actor attribution analysis"""
        content = str(raw_intel).lower()

        # Known APT groups
        apt_groups = {
            "apt28": ["fancy bear", "pawn storm", "strontium"],
            "apt29": ["cozy bear", "the dukes", "yttrium"],
            "lazarus": ["hidden cobra", "zinc"],
            "carbanak": ["fin7", "carbanak"],
            "apt1": ["comment crew", "comment group"],
            "apt40": ["leviathan", "mudcarp"]
        }

        for group, aliases in apt_groups.items():
            if group in content or any(alias in content for alias in aliases):
                return group.upper()

        return None

    async def _map_to_attack_framework(self, raw_intel: Dict[str, Any]) -> List[str]:
        """Map intelligence to MITRE ATT&CK techniques"""
        content = str(raw_intel).lower()
        ttps = []

        # Common technique mappings
        technique_mappings = {
            "spear phishing": "T1566.001",
            "credential dumping": "T1003",
            "lateral movement": "TA0008",
            "privilege escalation": "TA0004",
            "persistence": "TA0003",
            "defense evasion": "TA0005",
            "command and control": "TA0011",
            "exfiltration": "TA0010",
            "powershell": "T1059.001",
            "remote desktop": "T1021.001",
            "scheduled task": "T1053.005"
        }

        for technique, technique_id in technique_mappings.items():
            if technique in content:
                ttps.append(technique_id)

        return ttps

    async def _calculate_prediction_horizon(self, raw_intel: Dict[str, Any]) -> int:
        """Calculate prediction horizon in hours"""
        threat_level = await self._assess_threat_level(raw_intel)
        intel_type = await self._classify_intelligence_type(raw_intel)

        # Base horizon by threat level
        horizon_map = {
            ThreatLevel.EXTREME: 6,
            ThreatLevel.CRITICAL: 24,
            ThreatLevel.HIGH: 72,
            ThreatLevel.MEDIUM: 168,  # 1 week
            ThreatLevel.LOW: 720     # 1 month
        }

        base_horizon = horizon_map.get(threat_level, 168)

        # Adjust by intelligence type
        if intel_type == IntelligenceType.PREDICTIVE:
            base_horizon *= 2
        elif intel_type == IntelligenceType.TACTICAL:
            base_horizon //= 2

        return base_horizon

    async def _analyze_correlation(self, intel_id_1: str, intel_id_2: str) -> Optional[IntelligenceCorrelation]:
        """Analyze correlation between two intelligence items"""
        if intel_id_1 not in self.intelligence_db or intel_id_2 not in self.intelligence_db:
            return None

        intel_1 = self.intelligence_db[intel_id_1]
        intel_2 = self.intelligence_db[intel_id_2]

        correlation_strength = 0.0
        pattern_type = "unknown"
        insights = []

        # Temporal correlation
        time_diff = abs((intel_1.timestamp - intel_2.timestamp).total_seconds())
        if time_diff < 3600:  # Within 1 hour
            correlation_strength += 0.3
            pattern_type = "temporal"
            insights.append("Intelligence items occurred within close time proximity")

        # Indicator overlap
        common_indicators = set(intel_1.indicators) & set(intel_2.indicators)
        if common_indicators:
            correlation_strength += 0.4 * (len(common_indicators) / max(len(intel_1.indicators), len(intel_2.indicators)))
            pattern_type = "indicator_overlap"
            insights.append(f"Shared {len(common_indicators)} common indicators")

        # Attribution correlation
        if intel_1.attribution and intel_2.attribution and intel_1.attribution == intel_2.attribution:
            correlation_strength += 0.5
            pattern_type = "attribution"
            insights.append(f"Same threat actor attribution: {intel_1.attribution}")

        # TTP correlation
        if intel_1.ttps and intel_2.ttps:
            common_ttps = set(intel_1.ttps) & set(intel_2.ttps)
            if common_ttps:
                correlation_strength += 0.3 * (len(common_ttps) / max(len(intel_1.ttps), len(intel_2.ttps)))
                pattern_type = "ttp_similarity"
                insights.append(f"Shared {len(common_ttps)} TTPs")

        if correlation_strength >= self.correlation_threshold:
            correlation = IntelligenceCorrelation(
                correlation_id=f"CORR-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                related_intel=[intel_id_1, intel_id_2],
                correlation_strength=correlation_strength,
                pattern_type=pattern_type,
                insights=insights,
                recommended_actions=await self._generate_correlation_actions(intel_1, intel_2, correlation_strength),
                timestamp=datetime.now()
            )

            return correlation

        return None

    async def _generate_correlation_actions(self, intel_1: ThreatIntelligence, intel_2: ThreatIntelligence, strength: float) -> List[str]:
        """Generate recommended actions based on intelligence correlation"""
        actions = []

        if strength >= 0.8:
            actions.append("Initiate immediate threat hunting campaign")
            actions.append("Escalate to security operations center")
            actions.append("Implement emergency defensive measures")
        elif strength >= 0.6:
            actions.append("Enhance monitoring for related indicators")
            actions.append("Update threat detection rules")
            actions.append("Conduct focused threat assessment")
        else:
            actions.append("Monitor for additional corroborating evidence")
            actions.append("Update threat intelligence database")

        return actions

    def _generate_intel_id(self, raw_intel: Dict[str, Any]) -> str:
        """Generate unique intelligence ID"""
        content_hash = hashlib.sha256(str(raw_intel).encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"INTEL-{timestamp}-{content_hash}"

    async def get_intelligence_summary(self) -> Dict[str, Any]:
        """Get comprehensive intelligence summary"""
        summary = {
            "engine_id": self.engine_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics,
            "intelligence_count": len(self.intelligence_db),
            "decision_count": len(self.decision_history),
            "threat_level_distribution": self._calculate_threat_distribution(),
            "intelligence_type_distribution": self._calculate_intel_type_distribution(),
            "recent_decisions": [asdict(d) for d in self.decision_history[-5:]],
            "system_health": await self._assess_system_health()
        }

        return summary

    def _calculate_threat_distribution(self) -> Dict[str, int]:
        """Calculate distribution of threat levels"""
        distribution = {level.value: 0 for level in ThreatLevel}
        for intel in self.intelligence_db.values():
            distribution[intel.threat_level.value] += 1
        return distribution

    def _calculate_intel_type_distribution(self) -> Dict[str, int]:
        """Calculate distribution of intelligence types"""
        distribution = {intel_type.value: 0 for intel_type in IntelligenceType}
        for intel in self.intelligence_db.values():
            distribution[intel.type.value] += 1
        return distribution

    async def _assess_system_health(self) -> Dict[str, Any]:
        """Assess system health and performance"""
        health = {
            "status": "operational",
            "processing_rate": self.metrics["intelligence_processed"] / max(1, (datetime.now().hour + 1)),
            "decision_effectiveness": self._calculate_decision_effectiveness(),
            "prediction_accuracy": self.metrics["accuracy_score"],
            "memory_usage": len(self.intelligence_db) * 1024,  # Approximate
            "correlation_efficiency": self.metrics["correlations_identified"] / max(1, self.metrics["intelligence_processed"])
        }

        return health

    def _calculate_decision_effectiveness(self) -> float:
        """Calculate effectiveness of operational decisions"""
        if not self.decision_history:
            return 0.0

        # Simple effectiveness calculation based on confidence scores
        total_confidence = sum(d.confidence.value for d in self.decision_history)
        return total_confidence / len(self.decision_history)

async def main():
    """Demonstrate XORB Operational Intelligence Engine"""
    logger.info("🧠 Starting XORB Operational Intelligence Engine demonstration")

    engine = XORBOperationalIntelligenceEngine()

    # Sample intelligence data
    sample_intelligence = [
        {
            "source": "government_feed",
            "content": "APT29 using spear phishing with PowerShell payloads",
            "indicators": ["192.168.1.100", "malicious-domain.com", "abc123def456"],
            "timestamp": datetime.now().isoformat(),
            "corroborated": True
        },
        {
            "source": "commercial_threat_intel",
            "content": "Zero-day exploit targeting critical infrastructure",
            "indicators": ["10.0.0.50", "evil-site.net"],
            "timestamp": datetime.now().isoformat(),
            "severity": "critical"
        }
    ]

    # Process intelligence
    processed_intel = []
    for intel_data in sample_intelligence:
        intel = await engine.process_threat_intelligence(intel_data)
        processed_intel.append(intel)

    # Correlate intelligence
    intel_ids = [intel.id for intel in processed_intel]
    correlations = await engine.correlate_intelligence(intel_ids)

    # Generate operational decision
    context = {
        "current_threat_level": "high",
        "available_resources": ["agents", "monitoring", "response_team"],
        "time_constraints": "immediate"
    }
    decision = await engine.generate_operational_decision(context)

    # Predict threat evolution
    predictions = await engine.predict_threat_evolution(processed_intel)

    # Get summary
    summary = await engine.get_intelligence_summary()

    logger.info("🧠 Operational Intelligence Engine demonstration complete")
    logger.info(f"📊 Processed {len(processed_intel)} intelligence items")
    logger.info(f"🔗 Found {len(correlations)} correlations")
    logger.info(f"🎯 Generated decision: {decision.action}")

    return {
        "engine_id": engine.engine_id,
        "processed_intelligence": [asdict(intel) for intel in processed_intel],
        "correlations": [asdict(corr) for corr in correlations],
        "operational_decision": asdict(decision),
        "threat_predictions": predictions,
        "system_summary": summary
    }

if __name__ == "__main__":
    asyncio.run(main())
