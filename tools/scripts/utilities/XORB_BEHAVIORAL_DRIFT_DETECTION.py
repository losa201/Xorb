#!/usr/bin/env python3
"""
🛡️ XORB Behavioral Drift Detection Loop
Real-time behavioral anomaly detection for autonomous agents

This module continuously monitors XORB agents for behavioral drift that could
indicate compromise, manipulation, or malfunction. Pure tactical monitoring
without quantum mysticism - just solid statistical anomaly detection.
"""

import asyncio
import json
import logging
import time
import uuid
import numpy as np
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BehaviorType(Enum):
    EXECUTION_PATTERN = "execution_pattern"
    RESOURCE_USAGE = "resource_usage"
    NETWORK_BEHAVIOR = "network_behavior"
    DATA_ACCESS = "data_access"
    COMMAND_FREQUENCY = "command_frequency"
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    AUTHENTICATION = "authentication"

class AnomalyType(Enum):
    STATISTICAL_OUTLIER = "statistical_outlier"
    PATTERN_DEVIATION = "pattern_deviation"
    FREQUENCY_ANOMALY = "frequency_anomaly"
    THRESHOLD_BREACH = "threshold_breach"
    BEHAVIORAL_SHIFT = "behavioral_shift"
    CORRELATION_ANOMALY = "correlation_anomaly"

class SeverityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class BehaviorMetric:
    metric_id: str
    agent_id: str
    behavior_type: BehaviorType
    timestamp: datetime
    value: float
    metadata: Dict[str, Any]
    normalized_value: float = 0.0

@dataclass
class BehaviorBaseline:
    agent_id: str
    behavior_type: BehaviorType
    mean: float
    std_dev: float
    min_value: float
    max_value: float
    sample_count: int
    last_updated: datetime
    confidence_interval: Tuple[float, float]

@dataclass
class AnomalyDetection:
    detection_id: str
    agent_id: str
    behavior_type: BehaviorType
    anomaly_type: AnomalyType
    severity: SeverityLevel
    timestamp: datetime
    baseline_value: float
    observed_value: float
    deviation_score: float
    confidence: float
    evidence: List[str]
    recommended_actions: List[str]

@dataclass
class AgentBehaviorProfile:
    agent_id: str
    agent_type: str
    creation_time: datetime
    last_activity: datetime
    behavior_baselines: Dict[BehaviorType, BehaviorBaseline]
    total_anomalies: int
    recent_anomalies: int
    trust_score: float
    risk_level: SeverityLevel

class XORBBehavioralDriftDetection:
    """XORB Behavioral Drift Detection System"""
    
    def __init__(self):
        self.detector_id = f"BDD-{uuid.uuid4().hex[:8]}"
        self.initialization_time = datetime.now()
        
        # Behavior monitoring configuration
        self.monitoring_config = {
            "baseline_window_hours": 24,
            "anomaly_threshold_std": 2.5,
            "critical_threshold_std": 4.0,
            "min_samples_for_baseline": 50,
            "correlation_window_minutes": 30,
            "trust_score_decay_rate": 0.95
        }
        
        # Agent profiles and baselines
        self.agent_profiles: Dict[str, AgentBehaviorProfile] = {}
        self.behavior_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Real-time metrics buffer
        self.metrics_buffer: deque = deque(maxlen=10000)
        
        # Anomaly tracking
        self.detected_anomalies: List[AnomalyDetection] = []
        self.active_investigations: Dict[str, List[AnomalyDetection]] = {}
        
        # Detection statistics
        self.detection_stats = {
            "total_metrics_processed": 0,
            "anomalies_detected": 0,
            "false_positives": 0,
            "true_positives": 0,
            "agents_monitored": 0,
            "baselines_established": 0,
            "high_severity_alerts": 0,
            "trust_score_degradations": 0
        }
        
        # Simulated XORB agents for testing
        self.xorb_agents = self._initialize_agent_simulation()
        
        logger.info(f"🛡️ XORB Behavioral Drift Detection initialized - ID: {self.detector_id}")
        logger.info("📊 Behavioral monitoring: TACTICAL ANOMALY DETECTION")
    
    def _initialize_agent_simulation(self) -> Dict[str, Dict[str, Any]]:
        """Initialize simulated XORB agent environment"""
        agents = {}
        
        agent_types = [
            "reconnaissance_agent",
            "penetration_testing_agent", 
            "vulnerability_scanner",
            "threat_intelligence_collector",
            "network_monitor",
            "malware_analyzer",
            "incident_responder",
            "compliance_auditor"
        ]
        
        for i in range(12):  # 12 agents for monitoring
            agent_id = f"AGENT-{agent_types[i % len(agent_types)].upper()}-{uuid.uuid4().hex[:4]}"
            agents[agent_id] = {
                "type": agent_types[i % len(agent_types)],
                "status": "active",
                "last_activity": datetime.now(),
                "normal_behavior": {
                    "execution_frequency": np.random.normal(5.0, 1.0),  # executions per minute
                    "cpu_usage": np.random.normal(25.0, 5.0),  # percentage
                    "memory_usage": np.random.normal(512.0, 100.0),  # MB
                    "network_requests": np.random.normal(10.0, 2.0),  # requests per minute
                    "response_time": np.random.normal(2.5, 0.5),  # seconds
                    "error_rate": np.random.normal(0.02, 0.01)  # percentage
                }
            }
        
        return agents
    
    async def collect_agent_metrics(self, agent_id: str) -> List[BehaviorMetric]:
        """Collect behavioral metrics from agent"""
        if agent_id not in self.xorb_agents:
            return []
        
        agent = self.xorb_agents[agent_id]
        current_time = datetime.now()
        metrics = []
        
        # Simulate normal behavior with occasional anomalies
        anomaly_probability = 0.05  # 5% chance of anomalous behavior
        is_anomalous = np.random.random() < anomaly_probability
        
        normal_behavior = agent["normal_behavior"]
        
        # Generate behavioral metrics
        behavior_data = {
            BehaviorType.EXECUTION_PATTERN: {
                "base": normal_behavior["execution_frequency"],
                "noise": 0.2,
                "anomaly_multiplier": 3.0 if is_anomalous else 1.0
            },
            BehaviorType.RESOURCE_USAGE: {
                "base": normal_behavior["cpu_usage"],
                "noise": 5.0,
                "anomaly_multiplier": 2.5 if is_anomalous else 1.0
            },
            BehaviorType.NETWORK_BEHAVIOR: {
                "base": normal_behavior["network_requests"],
                "noise": 1.0,
                "anomaly_multiplier": 4.0 if is_anomalous else 1.0
            },
            BehaviorType.RESPONSE_TIME: {
                "base": normal_behavior["response_time"],
                "noise": 0.3,
                "anomaly_multiplier": 3.0 if is_anomalous else 1.0
            },
            BehaviorType.ERROR_RATE: {
                "base": normal_behavior["error_rate"],
                "noise": 0.005,
                "anomaly_multiplier": 10.0 if is_anomalous else 1.0
            }
        }
        
        for behavior_type, params in behavior_data.items():
            # Generate metric value
            base_value = params["base"]
            noise = np.random.normal(0, params["noise"])
            anomaly_factor = params["anomaly_multiplier"]
            
            if is_anomalous and behavior_type == BehaviorType.EXECUTION_PATTERN:
                # Simulate compromised agent - erratic execution pattern
                value = base_value * anomaly_factor + noise
            elif is_anomalous and behavior_type == BehaviorType.NETWORK_BEHAVIOR:
                # Simulate data exfiltration - high network activity
                value = base_value * anomaly_factor + noise
            else:
                value = base_value + noise
            
            value = max(0, value)  # Ensure non-negative values
            
            metric = BehaviorMetric(
                metric_id=f"METRIC-{uuid.uuid4().hex[:8]}",
                agent_id=agent_id,
                behavior_type=behavior_type,
                timestamp=current_time,
                value=value,
                metadata={
                    "agent_type": agent["type"],
                    "is_simulated_anomaly": is_anomalous,
                    "collection_method": "direct_monitoring"
                }
            )
            
            metrics.append(metric)
        
        return metrics
    
    async def establish_behavior_baseline(self, agent_id: str, behavior_type: BehaviorType) -> Optional[BehaviorBaseline]:
        """Establish behavioral baseline for agent"""
        history_key = f"{agent_id}_{behavior_type.value}"
        
        if history_key not in self.behavior_history:
            return None
        
        history = list(self.behavior_history[history_key])
        if len(history) < self.monitoring_config["min_samples_for_baseline"]:
            return None
        
        values = [metric.value for metric in history]
        
        mean_value = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
        min_value = min(values)
        max_value = max(values)
        
        # Calculate confidence interval (95%)
        margin = 1.96 * std_dev / np.sqrt(len(values))
        confidence_interval = (mean_value - margin, mean_value + margin)
        
        baseline = BehaviorBaseline(
            agent_id=agent_id,
            behavior_type=behavior_type,
            mean=mean_value,
            std_dev=std_dev,
            min_value=min_value,
            max_value=max_value,
            sample_count=len(values),
            last_updated=datetime.now(),
            confidence_interval=confidence_interval
        )
        
        # Update agent profile
        if agent_id not in self.agent_profiles:
            self.agent_profiles[agent_id] = AgentBehaviorProfile(
                agent_id=agent_id,
                agent_type=self.xorb_agents.get(agent_id, {}).get("type", "unknown"),
                creation_time=datetime.now(),
                last_activity=datetime.now(),
                behavior_baselines={},
                total_anomalies=0,
                recent_anomalies=0,
                trust_score=1.0,
                risk_level=SeverityLevel.LOW
            )
        
        self.agent_profiles[agent_id].behavior_baselines[behavior_type] = baseline
        self.detection_stats["baselines_established"] += 1
        
        return baseline
    
    async def detect_anomalies(self, metric: BehaviorMetric) -> List[AnomalyDetection]:
        """Detect behavioral anomalies in metric"""
        anomalies = []
        
        # Get agent profile and baseline
        if metric.agent_id not in self.agent_profiles:
            return anomalies
        
        profile = self.agent_profiles[metric.agent_id]
        
        if metric.behavior_type not in profile.behavior_baselines:
            return anomalies
        
        baseline = profile.behavior_baselines[metric.behavior_type]
        
        # Calculate deviation score
        if baseline.std_dev > 0:
            deviation_score = abs(metric.value - baseline.mean) / baseline.std_dev
        else:
            deviation_score = 0.0
        
        # Determine severity
        severity = SeverityLevel.LOW
        anomaly_type = AnomalyType.STATISTICAL_OUTLIER
        
        if deviation_score > self.monitoring_config["critical_threshold_std"]:
            severity = SeverityLevel.CRITICAL
        elif deviation_score > self.monitoring_config["anomaly_threshold_std"]:
            severity = SeverityLevel.HIGH
        elif deviation_score > 2.0:
            severity = SeverityLevel.MEDIUM
        
        # Check for threshold breaches
        if metric.value > baseline.max_value * 1.5 or metric.value < baseline.min_value * 0.5:
            anomaly_type = AnomalyType.THRESHOLD_BREACH
            severity = max(severity, SeverityLevel.MEDIUM)
        
        # Only report significant anomalies
        if deviation_score > 2.0:
            # Generate evidence
            evidence = [
                f"Deviation score: {deviation_score:.2f} standard deviations",
                f"Baseline mean: {baseline.mean:.2f}, observed: {metric.value:.2f}",
                f"Baseline range: [{baseline.min_value:.2f}, {baseline.max_value:.2f}]"
            ]
            
            # Generate recommendations
            recommendations = await self._generate_anomaly_recommendations(metric, baseline, deviation_score)
            
            anomaly = AnomalyDetection(
                detection_id=f"ANOMALY-{uuid.uuid4().hex[:8]}",
                agent_id=metric.agent_id,
                behavior_type=metric.behavior_type,
                anomaly_type=anomaly_type,
                severity=severity,
                timestamp=metric.timestamp,
                baseline_value=baseline.mean,
                observed_value=metric.value,
                deviation_score=deviation_score,
                confidence=min(0.99, deviation_score / 5.0),
                evidence=evidence,
                recommended_actions=recommendations
            )
            
            anomalies.append(anomaly)
            
            # Update agent profile
            profile.total_anomalies += 1
            profile.recent_anomalies += 1
            
            # Degrade trust score
            trust_degradation = min(0.1, deviation_score / 20.0)
            profile.trust_score = max(0.0, profile.trust_score - trust_degradation)
            
            if profile.trust_score < 0.7:
                profile.risk_level = SeverityLevel.HIGH
                self.detection_stats["trust_score_degradations"] += 1
            elif profile.trust_score < 0.85:
                profile.risk_level = SeverityLevel.MEDIUM
            
            # Update statistics
            self.detection_stats["anomalies_detected"] += 1
            if severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
                self.detection_stats["high_severity_alerts"] += 1
        
        return anomalies
    
    async def _generate_anomaly_recommendations(self, metric: BehaviorMetric, baseline: BehaviorBaseline, deviation_score: float) -> List[str]:
        """Generate recommendations for handling anomaly"""
        recommendations = []
        
        if metric.behavior_type == BehaviorType.EXECUTION_PATTERN:
            if metric.value > baseline.mean * 2:
                recommendations.extend([
                    "Investigate agent for potential compromise",
                    "Check for unauthorized command execution",
                    "Review agent execution logs"
                ])
            else:
                recommendations.append("Monitor agent execution frequency")
        
        elif metric.behavior_type == BehaviorType.RESOURCE_USAGE:
            if metric.value > baseline.mean * 2:
                recommendations.extend([
                    "Check for resource-intensive malware",
                    "Investigate cryptocurrency mining activity",
                    "Review system performance logs"
                ])
            else:
                recommendations.append("Monitor resource consumption patterns")
        
        elif metric.behavior_type == BehaviorType.NETWORK_BEHAVIOR:
            if metric.value > baseline.mean * 3:
                recommendations.extend([
                    "Investigate potential data exfiltration",
                    "Check for command and control communication",
                    "Review network traffic logs"
                ])
            else:
                recommendations.append("Monitor network activity patterns")
        
        elif metric.behavior_type == BehaviorType.ERROR_RATE:
            if metric.value > baseline.mean * 5:
                recommendations.extend([
                    "Investigate agent stability issues",
                    "Check for system tampering",
                    "Review error logs for attack patterns"
                ])
        
        # General recommendations
        if deviation_score > 4.0:
            recommendations.extend([
                "Consider agent quarantine",
                "Increase monitoring frequency",
                "Escalate to security team"
            ])
        
        return recommendations
    
    async def process_metrics_batch(self, metrics: List[BehaviorMetric]) -> List[AnomalyDetection]:
        """Process batch of metrics for anomaly detection"""
        all_anomalies = []
        
        for metric in metrics:
            # Normalize metric value
            history_key = f"{metric.agent_id}_{metric.behavior_type.value}"
            self.behavior_history[history_key].append(metric)
            self.metrics_buffer.append(metric)
            
            # Update statistics
            self.detection_stats["total_metrics_processed"] += 1
            
            # Establish baseline if enough data
            await self.establish_behavior_baseline(metric.agent_id, metric.behavior_type)
            
            # Detect anomalies
            anomalies = await self.detect_anomalies(metric)
            all_anomalies.extend(anomalies)
            
            # Store anomalies
            self.detected_anomalies.extend(anomalies)
        
        return all_anomalies
    
    async def behavioral_monitoring_cycle(self) -> Dict[str, Any]:
        """Execute behavioral monitoring cycle"""
        logger.info("📊 Starting behavioral monitoring cycle")
        
        all_metrics = []
        all_anomalies = []
        
        # Collect metrics from all agents
        for agent_id in self.xorb_agents.keys():
            metrics = await self.collect_agent_metrics(agent_id)
            all_metrics.extend(metrics)
        
        # Process metrics for anomaly detection
        anomalies = await self.process_metrics_batch(all_metrics)
        all_anomalies.extend(anomalies)
        
        # Update agent activity
        for agent_id in self.xorb_agents.keys():
            if agent_id in self.agent_profiles:
                self.agent_profiles[agent_id].last_activity = datetime.now()
        
        # Calculate cycle statistics
        high_risk_agents = len([p for p in self.agent_profiles.values() if p.risk_level in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]])
        avg_trust_score = statistics.mean([p.trust_score for p in self.agent_profiles.values()]) if self.agent_profiles else 1.0
        
        cycle_results = {
            "cycle_timestamp": datetime.now().isoformat(),
            "metrics_collected": len(all_metrics),
            "anomalies_detected": len(all_anomalies),
            "high_severity_anomalies": len([a for a in all_anomalies if a.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]]),
            "agents_monitored": len(self.xorb_agents),
            "high_risk_agents": high_risk_agents,
            "average_trust_score": avg_trust_score,
            "detection_statistics": self.detection_stats,
            "anomaly_details": [asdict(a) for a in all_anomalies],
            "agent_profiles": {aid: asdict(profile) for aid, profile in self.agent_profiles.items()}
        }
        
        return cycle_results

async def main():
    """Main behavioral drift detection execution"""
    logger.info("📊 Starting XORB Behavioral Drift Detection")
    
    # Initialize detector
    bdd = XORBBehavioralDriftDetection()
    
    # Execute continuous monitoring cycles
    session_duration = 3  # 3 minutes for demonstration
    cycles_completed = 0
    
    start_time = time.time()
    end_time = start_time + (session_duration * 60)
    
    while time.time() < end_time:
        try:
            # Execute behavioral monitoring cycle
            cycle_results = await bdd.behavioral_monitoring_cycle()
            cycles_completed += 1
            
            # Log progress
            logger.info(f"📊 Monitoring Cycle #{cycles_completed} completed")
            logger.info(f"📈 Metrics collected: {cycle_results['metrics_collected']}")
            logger.info(f"🚨 Anomalies detected: {cycle_results['anomalies_detected']}")
            logger.info(f"⚠️ High severity: {cycle_results['high_severity_anomalies']}")
            logger.info(f"🔒 Avg trust score: {cycle_results['average_trust_score']:.3f}")
            logger.info(f"🎯 High risk agents: {cycle_results['high_risk_agents']}")
            
            await asyncio.sleep(15.0)  # 15-second cycles
            
        except Exception as e:
            logger.error(f"Error in behavioral monitoring: {e}")
            await asyncio.sleep(10.0)
    
    # Final results
    final_results = {
        "session_id": f"BDD-SESSION-{int(start_time)}",
        "cycles_completed": cycles_completed,
        "detection_statistics": bdd.detection_stats,
        "total_anomalies": len(bdd.detected_anomalies),
        "agents_profiles": len(bdd.agent_profiles),
        "average_trust_score": statistics.mean([p.trust_score for p in bdd.agent_profiles.values()]) if bdd.agent_profiles else 1.0,
        "high_risk_agents": len([p for p in bdd.agent_profiles.values() if p.risk_level in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]]),
        "detection_effectiveness": bdd.detection_stats["anomalies_detected"] / max(1, bdd.detection_stats["total_metrics_processed"])
    }
    
    # Save results
    results_filename = f"xorb_behavioral_drift_results_{int(time.time())}.json"
    with open(results_filename, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    logger.info(f"💾 Behavioral drift results saved: {results_filename}")
    logger.info("🏆 XORB Behavioral Drift Detection completed!")
    
    # Display final summary
    logger.info("📊 Behavioral Monitoring Summary:")
    logger.info(f"  • Cycles completed: {cycles_completed}")
    logger.info(f"  • Metrics processed: {bdd.detection_stats['total_metrics_processed']}")
    logger.info(f"  • Anomalies detected: {bdd.detection_stats['anomalies_detected']}")
    logger.info(f"  • High severity alerts: {bdd.detection_stats['high_severity_alerts']}")
    logger.info(f"  • Trust score degradations: {bdd.detection_stats['trust_score_degradations']}")
    logger.info(f"  • Detection effectiveness: {final_results['detection_effectiveness']:.1%}")
    logger.info(f"  • Agents monitored: {len(bdd.agent_profiles)}")
    logger.info(f"  • High risk agents: {final_results['high_risk_agents']}")
    
    return final_results

if __name__ == "__main__":
    # Execute behavioral drift detection
    asyncio.run(main())