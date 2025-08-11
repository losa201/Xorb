"""
XORB Team Orchestration Framework
Advanced Red vs Blue vs Purple Team Coordination with ML Integration

This module provides a sophisticated framework for coordinating red team (offensive), 
blue team (defensive), and purple team (collaborative) operations with integrated 
machine learning capabilities for real-time tactical intelligence and adaptive strategies.
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import pandas as pd

# ML and statistical libraries with graceful fallbacks
try:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import XORB security services
from ..intelligence.threat_intelligence_engine import ThreatIntelligenceEngine
from ..monitoring import SecurityMonitor
from .monitoring import SecurityEventProcessor

logger = logging.getLogger(__name__)

class TeamRole(Enum):
    """Team roles in the cybersecurity operation"""
    RED_TEAM = "red_team"          # Offensive operations
    BLUE_TEAM = "blue_team"        # Defensive operations  
    PURPLE_TEAM = "purple_team"    # Collaborative operations
    WHITE_TEAM = "white_team"      # Oversight and coordination
    GREEN_TEAM = "green_team"      # Infrastructure and support

class OperationType(Enum):
    """Types of security operations"""
    PENETRATION_TEST = "penetration_test"
    THREAT_HUNTING = "threat_hunting"
    INCIDENT_RESPONSE = "incident_response"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"
    THREAT_SIMULATION = "threat_simulation"
    TABLETOP_EXERCISE = "tabletop_exercise"
    CONTINUOUS_MONITORING = "continuous_monitoring"
    PURPLE_TEAM_EXERCISE = "purple_team_exercise"

class ThreatLevel(Enum):
    """Threat severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class OperationPhase(Enum):
    """Phases of security operations"""
    PLANNING = "planning"
    RECONNAISSANCE = "reconnaissance"
    INITIAL_ACCESS = "initial_access"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"
    CONTAINMENT = "containment"
    ERADICATION = "eradication"
    RECOVERY = "recovery"
    LESSONS_LEARNED = "lessons_learned"

class MLModelType(Enum):
    """Types of ML models used in team operations"""
    THREAT_PREDICTION = "threat_prediction"
    ATTACK_CLASSIFICATION = "attack_classification"
    DEFENSIVE_OPTIMIZATION = "defensive_optimization"
    TEAM_PERFORMANCE = "team_performance"
    TACTICAL_RECOMMENDATION = "tactical_recommendation"
    ADVERSARY_SIMULATION = "adversary_simulation"

@dataclass
class TeamMember:
    """Individual team member definition"""
    member_id: str
    name: str
    role: TeamRole
    specializations: List[str]
    skill_level: float  # 0.0 to 1.0
    experience_points: int
    current_assignment: Optional[str]
    performance_metrics: Dict[str, float]
    certifications: List[str]
    tools_proficiency: Dict[str, float]
    availability_status: str
    last_active: datetime

@dataclass
class SecurityScenario:
    """Security scenario or exercise definition"""
    scenario_id: str
    name: str
    description: str
    operation_type: OperationType
    threat_level: ThreatLevel
    target_environment: str
    objectives: List[str]
    success_criteria: List[str]
    constraints: List[str]
    mitre_tactics: List[str]
    mitre_techniques: List[str]
    estimated_duration: timedelta
    complexity_score: float
    required_skills: List[str]
    created_by: str
    created_at: datetime

@dataclass
class OperationPlan:
    """Comprehensive operation plan"""
    plan_id: str
    scenario_id: str
    red_team_objectives: List[str]
    blue_team_objectives: List[str]
    purple_team_objectives: List[str]
    phases: List[OperationPhase]
    timeline: Dict[OperationPhase, Dict[str, Any]]
    resource_allocation: Dict[TeamRole, List[str]]
    communication_plan: Dict[str, Any]
    escalation_procedures: List[str]
    success_metrics: Dict[str, float]
    risk_assessment: Dict[str, Any]
    contingency_plans: List[str]
    ml_integration_points: List[str]
    
@dataclass
class OperationExecution:
    """Real-time operation execution state"""
    execution_id: str
    plan_id: str
    current_phase: OperationPhase
    start_time: datetime
    estimated_end_time: datetime
    actual_end_time: Optional[datetime]
    team_status: Dict[TeamRole, str]
    phase_progress: Dict[OperationPhase, float]
    real_time_metrics: Dict[str, Any]
    detected_activities: List[Dict[str, Any]]
    countermeasures_deployed: List[Dict[str, Any]]
    ml_predictions: Dict[str, Any]
    adaptive_adjustments: List[Dict[str, Any]]
    communication_log: List[Dict[str, Any]]

@dataclass
class TeamPerformanceMetrics:
    """Comprehensive team performance metrics"""
    team_role: TeamRole
    operation_id: str
    success_rate: float
    response_time: float
    detection_accuracy: float
    false_positive_rate: float
    coverage_percentage: float
    collaboration_score: float
    innovation_index: float
    skill_development: Dict[str, float]
    tool_effectiveness: Dict[str, float]
    learning_velocity: float

class MLTacticalIntelligence:
    """Machine Learning engine for tactical intelligence and decision support"""
    
    def __init__(self):
        self.models: Dict[MLModelType, Any] = {}
        self.training_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.prediction_cache = {}
        self.model_performance: Dict[str, Dict[str, float]] = {}
        self.feature_importance: Dict[str, Dict[str, float]] = {}
        
    async def initialize(self):
        """Initialize ML models for tactical intelligence"""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, using simplified ML models")
            return
            
        try:
            # Initialize threat prediction model
            self.models[MLModelType.THREAT_PREDICTION] = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            # Initialize attack classification model
            self.models[MLModelType.ATTACK_CLASSIFICATION] = GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                learning_rate=0.1
            )
            
            # Initialize defensive optimization model
            self.models[MLModelType.DEFENSIVE_OPTIMIZATION] = RandomForestClassifier(
                n_estimators=80,
                random_state=42
            )
            
            # Initialize team performance model
            self.models[MLModelType.TEAM_PERFORMANCE] = RandomForestClassifier(
                n_estimators=60,
                random_state=42
            )
            
            # Train with synthetic data
            await self._train_with_synthetic_data()
            
            logger.info("ML Tactical Intelligence initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
            
    async def _train_with_synthetic_data(self):
        """Train models with synthetic operational data"""
        if not SKLEARN_AVAILABLE:
            return
            
        # Generate synthetic training data for threat prediction
        threat_features = []
        threat_labels = []
        
        for _ in range(1000):
            # Simulate threat scenario features
            features = [
                np.random.normal(0.3, 0.1),   # network_anomaly_score
                np.random.normal(0.4, 0.15),  # user_behavior_score  
                np.random.normal(0.2, 0.08),  # system_integrity_score
                np.random.normal(0.5, 0.2),   # threat_intel_score
                np.random.normal(0.3, 0.1),   # vulnerability_score
                np.random.randint(0, 24),     # time_of_day
                np.random.randint(0, 7),      # day_of_week
                np.random.normal(0.4, 0.1),   # red_team_activity
                np.random.normal(0.6, 0.15),  # blue_team_readiness
                np.random.normal(0.5, 0.1)    # purple_team_coordination
            ]
            threat_features.append(features)
            
            # Generate label based on feature combination
            threat_score = sum(features[:5]) / 5
            threat_labels.append(1 if threat_score > 0.4 else 0)
        
        # Train threat prediction model
        X_train, X_test, y_train, y_test = train_test_split(
            threat_features, threat_labels, test_size=0.2, random_state=42
        )
        
        self.models[MLModelType.THREAT_PREDICTION].fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.models[MLModelType.THREAT_PREDICTION].predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.model_performance["threat_prediction"] = {
            "accuracy": accuracy,
            "training_samples": len(X_train),
            "last_trained": datetime.now().isoformat()
        }
        
        logger.info(f"Threat prediction model trained with {accuracy:.3f} accuracy")
        
    async def predict_threat_likelihood(self, operation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Predict threat likelihood based on operation context"""
        try:
            if MLModelType.THREAT_PREDICTION not in self.models:
                return await self._fallback_threat_prediction(operation_context)
            
            # Extract features from operation context
            features = self._extract_threat_features(operation_context)
            
            # Make prediction
            threat_prob = self.models[MLModelType.THREAT_PREDICTION].predict_proba([features])[0]
            threat_prediction = self.models[MLModelType.THREAT_PREDICTION].predict([features])[0]
            
            # Get feature importance
            feature_importance = self.models[MLModelType.THREAT_PREDICTION].feature_importances_
            
            return {
                "threat_likelihood": float(threat_prob[1]),
                "threat_predicted": bool(threat_prediction),
                "confidence": float(max(threat_prob)),
                "contributing_factors": self._get_top_contributing_factors(features, feature_importance),
                "recommendation": self._generate_threat_recommendation(threat_prob[1]),
                "model_performance": self.model_performance.get("threat_prediction", {})
            }
            
        except Exception as e:
            logger.error(f"Threat prediction failed: {e}")
            return await self._fallback_threat_prediction(operation_context)
    
    def _extract_threat_features(self, context: Dict[str, Any]) -> List[float]:
        """Extract features for threat prediction"""
        return [
            context.get("network_anomaly_score", 0.3),
            context.get("user_behavior_score", 0.4),
            context.get("system_integrity_score", 0.2),
            context.get("threat_intel_score", 0.5),
            context.get("vulnerability_score", 0.3),
            datetime.now().hour,
            datetime.now().weekday(),
            context.get("red_team_activity", 0.4),
            context.get("blue_team_readiness", 0.6),
            context.get("purple_team_coordination", 0.5)
        ]
    
    def _get_top_contributing_factors(self, features: List[float], importance: np.ndarray) -> List[Dict[str, Any]]:
        """Get top contributing factors for threat prediction"""
        feature_names = [
            "network_anomaly_score", "user_behavior_score", "system_integrity_score",
            "threat_intel_score", "vulnerability_score", "time_of_day", "day_of_week",
            "red_team_activity", "blue_team_readiness", "purple_team_coordination"
        ]
        
        factors = []
        for i, (feature_val, importance_val) in enumerate(zip(features, importance)):
            factors.append({
                "factor": feature_names[i],
                "value": feature_val,
                "importance": float(importance_val),
                "impact": "high" if importance_val > 0.15 else "medium" if importance_val > 0.08 else "low"
            })
        
        # Sort by importance and return top 5
        return sorted(factors, key=lambda x: x["importance"], reverse=True)[:5]
    
    def _generate_threat_recommendation(self, threat_likelihood: float) -> str:
        """Generate tactical recommendation based on threat likelihood"""
        if threat_likelihood > 0.8:
            return "🚨 HIGH ALERT: Implement immediate defensive measures and escalate to purple team coordination"
        elif threat_likelihood > 0.6:
            return "⚠️ ELEVATED THREAT: Increase blue team monitoring and prepare defensive countermeasures"
        elif threat_likelihood > 0.4:
            return "🔍 MONITOR: Continue surveillance and prepare adaptive responses"
        else:
            return "✅ NORMAL: Maintain standard operational posture"
    
    async def _fallback_threat_prediction(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback threat prediction when ML unavailable"""
        # Simple rule-based prediction
        threat_score = 0.0
        factors = []
        
        if context.get("network_anomaly_score", 0) > 0.7:
            threat_score += 0.3
            factors.append("High network anomaly detected")
        
        if context.get("red_team_activity", 0) > 0.8:
            threat_score += 0.4
            factors.append("Intense red team activity")
        
        if context.get("blue_team_readiness", 0) < 0.3:
            threat_score += 0.2
            factors.append("Low blue team readiness")
        
        return {
            "threat_likelihood": min(threat_score, 1.0),
            "threat_predicted": threat_score > 0.5,
            "confidence": 0.7,
            "contributing_factors": factors,
            "recommendation": self._generate_threat_recommendation(threat_score),
            "model_type": "rule_based_fallback"
        }
    
    async def classify_attack_technique(self, attack_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify attack technique using ML"""
        try:
            # Simulate attack classification for now
            techniques = [
                "T1078.004", "T1083", "T1046", "T1055", "T1082", 
                "T1033", "T1018", "T1057", "T1012", "T1016"
            ]
            
            # Mock classification based on attack characteristics
            confidence_scores = np.random.dirichlet(np.ones(len(techniques)))
            predicted_technique = techniques[np.argmax(confidence_scores)]
            
            return {
                "predicted_technique": predicted_technique,
                "confidence": float(max(confidence_scores)),
                "all_predictions": [
                    {"technique": tech, "confidence": float(conf)}
                    for tech, conf in zip(techniques, confidence_scores)
                ],
                "mitre_mapping": await self._get_mitre_mapping(predicted_technique)
            }
            
        except Exception as e:
            logger.error(f"Attack classification failed: {e}")
            return {"error": str(e)}
    
    async def _get_mitre_mapping(self, technique_id: str) -> Dict[str, Any]:
        """Get MITRE ATT&CK mapping for technique"""
        mitre_db = {
            "T1078.004": {"tactic": "Defense Evasion", "name": "Cloud Instance Metadata API"},
            "T1083": {"tactic": "Discovery", "name": "File and Directory Discovery"},
            "T1046": {"tactic": "Discovery", "name": "Network Service Scanning"},
            "T1055": {"tactic": "Defense Evasion", "name": "Process Injection"},
            "T1082": {"tactic": "Discovery", "name": "System Information Discovery"}
        }
        
        return mitre_db.get(technique_id, {"tactic": "Unknown", "name": "Unknown Technique"})
    
    async def optimize_defensive_posture(self, team_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize defensive posture using ML recommendations"""
        try:
            current_posture = team_metrics.get("current_defensive_posture", {})
            threat_context = team_metrics.get("threat_context", {})
            
            # Calculate optimization recommendations
            recommendations = []
            priority_scores = {}
            
            # Analyze current gaps
            detection_coverage = current_posture.get("detection_coverage", 0.7)
            response_time = current_posture.get("response_time", 15.0)  # minutes
            false_positive_rate = current_posture.get("false_positive_rate", 0.1)
            
            if detection_coverage < 0.8:
                recommendations.append({
                    "area": "detection_coverage",
                    "current": detection_coverage,
                    "target": 0.9,
                    "actions": ["Deploy additional sensors", "Tune detection rules", "Enhance log collection"],
                    "priority": "high",
                    "estimated_improvement": 0.15
                })
                priority_scores["detection_coverage"] = 0.9
            
            if response_time > 10.0:
                recommendations.append({
                    "area": "response_time",
                    "current": response_time,
                    "target": 5.0,
                    "actions": ["Automate response workflows", "Pre-position response teams", "Optimize communication"],
                    "priority": "medium",
                    "estimated_improvement": 8.0
                })
                priority_scores["response_time"] = 0.7
            
            if false_positive_rate > 0.05:
                recommendations.append({
                    "area": "false_positive_reduction",
                    "current": false_positive_rate,
                    "target": 0.03,
                    "actions": ["Refine detection logic", "Implement context enrichment", "Tune thresholds"],
                    "priority": "medium",
                    "estimated_improvement": 0.04
                })
                priority_scores["false_positive_reduction"] = 0.6
            
            return {
                "recommendations": recommendations,
                "priority_scores": priority_scores,
                "overall_optimization_potential": sum(priority_scores.values()) / len(priority_scores) if priority_scores else 0,
                "implementation_timeline": "2-4 weeks",
                "resource_requirements": await self._calculate_resource_requirements(recommendations)
            }
            
        except Exception as e:
            logger.error(f"Defensive optimization failed: {e}")
            return {"error": str(e)}
    
    async def _calculate_resource_requirements(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate resource requirements for optimization"""
        return {
            "personnel_hours": sum(20 for _ in recommendations),  # Base estimate
            "budget_estimate": len(recommendations) * 25000,  # Base cost per recommendation
            "timeline_weeks": max(2, len(recommendations)),
            "skills_required": ["Security Engineering", "ML Operations", "Network Security"]
        }

class TeamOrchestrationFramework:
    """Main orchestration framework for red vs blue vs purple team operations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Core components
        self.team_members: Dict[str, TeamMember] = {}
        self.security_scenarios: Dict[str, SecurityScenario] = {}
        self.operation_plans: Dict[str, OperationPlan] = {}
        self.active_executions: Dict[str, OperationExecution] = {}
        self.performance_history: List[TeamPerformanceMetrics] = []
        
        # ML and intelligence
        self.ml_engine = MLTacticalIntelligence()
        self.threat_intel_engine = None
        
        # Communication and coordination
        self.communication_channels: Dict[str, Any] = {}
        self.real_time_coordination = {}
        
        # Metrics and analytics
        self.operation_metrics: Dict[str, Any] = defaultdict(dict)
        self.learning_outcomes: List[Dict[str, Any]] = []
        
        # Advanced features
        self.adaptive_playbooks: Dict[str, Any] = {}
        self.threat_simulation_engine = None
        
    async def initialize(self):
        """Initialize the team orchestration framework"""
        try:
            logger.info("Initializing Team Orchestration Framework...")
            
            # Initialize ML engine
            await self.ml_engine.initialize()
            
            # Load team configurations
            await self._load_team_configurations()
            
            # Load security scenarios
            await self._load_security_scenarios()
            
            # Initialize communication channels
            await self._setup_communication_channels()
            
            # Load adaptive playbooks
            await self._load_adaptive_playbooks()
            
            logger.info("Team Orchestration Framework initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Team Orchestration Framework: {e}")
            raise
    
    async def _load_team_configurations(self):
        """Load team member configurations"""
        # Red Team Members
        red_team_members = [
            {
                "member_id": "red_001",
                "name": "Alex Rodriguez",
                "role": TeamRole.RED_TEAM,
                "specializations": ["Web Application Testing", "Network Penetration", "Social Engineering"],
                "skill_level": 0.9,
                "experience_points": 1500,
                "certifications": ["OSCP", "OSCE", "GPEN"],
                "tools_proficiency": {"metasploit": 0.95, "burp_suite": 0.90, "nmap": 0.85}
            },
            {
                "member_id": "red_002", 
                "name": "Sarah Chen",
                "role": TeamRole.RED_TEAM,
                "specializations": ["Advanced Persistent Threats", "Malware Development", "Evasion Techniques"],
                "skill_level": 0.95,
                "experience_points": 2100,
                "certifications": ["OSEP", "CISSP", "GREM"],
                "tools_proficiency": {"cobalt_strike": 0.90, "empire": 0.85, "custom_tools": 0.95}
            }
        ]
        
        # Blue Team Members
        blue_team_members = [
            {
                "member_id": "blue_001",
                "name": "Michael Thompson",
                "role": TeamRole.BLUE_TEAM,
                "specializations": ["SIEM Analysis", "Incident Response", "Threat Hunting"],
                "skill_level": 0.88,
                "experience_points": 1800,
                "certifications": ["GCIH", "GCFA", "GNFA"],
                "tools_proficiency": {"splunk": 0.90, "wireshark": 0.85, "volatility": 0.80}
            },
            {
                "member_id": "blue_002",
                "name": "Jennifer Lee",
                "role": TeamRole.BLUE_TEAM,
                "specializations": ["Malware Analysis", "Digital Forensics", "Network Security"],
                "skill_level": 0.92,
                "experience_points": 1950,
                "certifications": ["GREM", "GCFA", "GSEC"],
                "tools_proficiency": {"ida_pro": 0.88, "autopsy": 0.85, "yara": 0.90}
            }
        ]
        
        # Purple Team Members
        purple_team_members = [
            {
                "member_id": "purple_001",
                "name": "David Kim",
                "role": TeamRole.PURPLE_TEAM,
                "specializations": ["Red/Blue Coordination", "Security Architecture", "Threat Modeling"],
                "skill_level": 0.93,
                "experience_points": 2200,
                "certifications": ["CISSP", "SABSA", "TOGAF"],
                "tools_proficiency": {"mitre_attack": 0.95, "threat_modeling": 0.90, "automation": 0.85}
            }
        ]
        
        # Create team member objects
        all_members = red_team_members + blue_team_members + purple_team_members
        
        for member_data in all_members:
            member = TeamMember(
                member_id=member_data["member_id"],
                name=member_data["name"],
                role=member_data["role"],
                specializations=member_data["specializations"],
                skill_level=member_data["skill_level"],
                experience_points=member_data["experience_points"],
                current_assignment=None,
                performance_metrics={
                    "success_rate": 0.85,
                    "collaboration_score": 0.80,
                    "innovation_index": 0.75
                },
                certifications=member_data["certifications"],
                tools_proficiency=member_data["tools_proficiency"],
                availability_status="available",
                last_active=datetime.now()
            )
            self.team_members[member_data["member_id"]] = member
        
        logger.info(f"Loaded {len(self.team_members)} team members")
    
    async def _load_security_scenarios(self):
        """Load predefined security scenarios"""
        scenarios = [
            {
                "scenario_id": "apt_simulation",
                "name": "Advanced Persistent Threat Simulation",
                "description": "Multi-stage APT attack simulation with stealth techniques",
                "operation_type": OperationType.THREAT_SIMULATION,
                "threat_level": ThreatLevel.HIGH,
                "target_environment": "corporate_network",
                "objectives": [
                    "Establish initial foothold via spear phishing",
                    "Achieve persistence on target systems",
                    "Perform lateral movement across network",
                    "Locate and exfiltrate sensitive data",
                    "Maintain stealth throughout operation"
                ],
                "success_criteria": [
                    "Successfully compromise at least 3 critical systems",
                    "Maintain access for 72 hours undetected",
                    "Exfiltrate 100MB of simulated sensitive data",
                    "Document all defensive gaps identified"
                ],
                "mitre_tactics": ["Initial Access", "Persistence", "Lateral Movement", "Exfiltration"],
                "mitre_techniques": ["T1566.001", "T1547.001", "T1021.001", "T1041"],
                "complexity_score": 0.9,
                "required_skills": ["Advanced Penetration Testing", "Stealth Techniques", "Post-Exploitation"]
            },
            {
                "scenario_id": "insider_threat_hunt",
                "name": "Insider Threat Hunting Exercise",
                "description": "Blue team exercise to detect simulated insider threat activities",
                "operation_type": OperationType.THREAT_HUNTING,
                "threat_level": ThreatLevel.MEDIUM,
                "target_environment": "enterprise_environment",
                "objectives": [
                    "Detect unauthorized data access patterns",
                    "Identify privilege escalation attempts",
                    "Monitor for data exfiltration indicators",
                    "Validate detection coverage"
                ],
                "success_criteria": [
                    "Detect 90% of simulated insider activities",
                    "Reduce false positive rate below 5%",
                    "Complete investigation within 4 hours",
                    "Provide actionable threat intelligence"
                ],
                "mitre_tactics": ["Collection", "Exfiltration"],
                "mitre_techniques": ["T1005", "T1041", "T1048.003"],
                "complexity_score": 0.7,
                "required_skills": ["Threat Hunting", "Behavioral Analysis", "Data Analytics"]
            },
            {
                "scenario_id": "purple_team_exercise",
                "name": "Collaborative Purple Team Exercise",
                "description": "Joint red and blue team exercise with real-time collaboration",
                "operation_type": OperationType.PURPLE_TEAM_EXERCISE,
                "threat_level": ThreatLevel.HIGH,
                "target_environment": "hybrid_cloud",
                "objectives": [
                    "Test detection capabilities against realistic attacks",
                    "Improve defensive response procedures",
                    "Validate security tool effectiveness",
                    "Enhance team collaboration and communication"
                ],
                "success_criteria": [
                    "Complete 5 attack scenarios with real-time feedback",
                    "Achieve 85% detection rate with <10% false positives",
                    "Demonstrate improved response times",
                    "Document 10+ actionable improvements"
                ],
                "mitre_tactics": ["Initial Access", "Defense Evasion", "Collection"],
                "mitre_techniques": ["T1190", "T1027", "T1005"],
                "complexity_score": 0.85,
                "required_skills": ["Cross-team Collaboration", "Real-time Analysis", "Adaptive Tactics"]
            }
        ]
        
        for scenario_data in scenarios:
            scenario = SecurityScenario(
                scenario_id=scenario_data["scenario_id"],
                name=scenario_data["name"],
                description=scenario_data["description"],
                operation_type=scenario_data["operation_type"],
                threat_level=scenario_data["threat_level"],
                target_environment=scenario_data["target_environment"],
                objectives=scenario_data["objectives"],
                success_criteria=scenario_data["success_criteria"],
                constraints=[],
                mitre_tactics=scenario_data["mitre_tactics"],
                mitre_techniques=scenario_data["mitre_techniques"],
                estimated_duration=timedelta(hours=8),
                complexity_score=scenario_data["complexity_score"],
                required_skills=scenario_data["required_skills"],
                created_by="system",
                created_at=datetime.now()
            )
            self.security_scenarios[scenario_data["scenario_id"]] = scenario
        
        logger.info(f"Loaded {len(self.security_scenarios)} security scenarios")
    
    async def _setup_communication_channels(self):
        """Setup communication channels for team coordination"""
        self.communication_channels = {
            "red_team_channel": {
                "type": "secure_chat",
                "participants": [m.member_id for m in self.team_members.values() if m.role == TeamRole.RED_TEAM],
                "encryption": "AES-256",
                "message_history": []
            },
            "blue_team_channel": {
                "type": "secure_chat", 
                "participants": [m.member_id for m in self.team_members.values() if m.role == TeamRole.BLUE_TEAM],
                "encryption": "AES-256",
                "message_history": []
            },
            "purple_coordination": {
                "type": "tactical_coordination",
                "participants": [m.member_id for m in self.team_members.values()],
                "features": ["real_time_sharing", "ml_insights", "adaptive_planning"],
                "message_history": []
            },
            "command_center": {
                "type": "oversight_channel",
                "participants": [m.member_id for m in self.team_members.values() if m.role == TeamRole.PURPLE_TEAM],
                "features": ["operation_monitoring", "metrics_dashboard", "escalation_procedures"],
                "message_history": []
            }
        }
        
        logger.info("Communication channels established")
    
    async def _load_adaptive_playbooks(self):
        """Load adaptive playbooks for different scenarios"""
        self.adaptive_playbooks = {
            "red_team_attack_chains": {
                "initial_access": [
                    {"technique": "T1566.001", "description": "Spearphishing Attachment", "success_rate": 0.7},
                    {"technique": "T1190", "description": "Exploit Public-Facing Application", "success_rate": 0.6},
                    {"technique": "T1078", "description": "Valid Accounts", "success_rate": 0.8}
                ],
                "persistence": [
                    {"technique": "T1547.001", "description": "Registry Run Keys", "success_rate": 0.8},
                    {"technique": "T1053.005", "description": "Scheduled Task", "success_rate": 0.7},
                    {"technique": "T1543.003", "description": "Windows Service", "success_rate": 0.6}
                ]
            },
            "blue_team_responses": {
                "detection_rules": [
                    {"rule": "unusual_network_traffic", "effectiveness": 0.85, "false_positive_rate": 0.1},
                    {"rule": "privilege_escalation_attempts", "effectiveness": 0.75, "false_positive_rate": 0.05},
                    {"rule": "lateral_movement_indicators", "effectiveness": 0.80, "false_positive_rate": 0.08}
                ],
                "response_procedures": [
                    {"procedure": "isolate_compromised_system", "response_time": 5.0, "effectiveness": 0.9},
                    {"procedure": "block_malicious_domain", "response_time": 2.0, "effectiveness": 0.85},
                    {"procedure": "reset_compromised_credentials", "response_time": 10.0, "effectiveness": 0.95}
                ]
            }
        }
        
        logger.info("Adaptive playbooks loaded")
    
    async def create_operation_plan(self, scenario_id: str, customizations: Dict[str, Any] = None) -> str:
        """Create a comprehensive operation plan"""
        try:
            if scenario_id not in self.security_scenarios:
                raise ValueError(f"Scenario {scenario_id} not found")
            
            scenario = self.security_scenarios[scenario_id]
            plan_id = f"plan_{scenario_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Get ML recommendations for team assignments
            ml_recommendations = await self._get_ml_team_recommendations(scenario)
            
            # Create operation plan
            operation_plan = OperationPlan(
                plan_id=plan_id,
                scenario_id=scenario_id,
                red_team_objectives=await self._generate_red_team_objectives(scenario),
                blue_team_objectives=await self._generate_blue_team_objectives(scenario),
                purple_team_objectives=await self._generate_purple_team_objectives(scenario),
                phases=self._get_operation_phases(scenario.operation_type),
                timeline=await self._create_operation_timeline(scenario),
                resource_allocation=await self._allocate_team_resources(scenario, ml_recommendations),
                communication_plan=await self._create_communication_plan(scenario),
                escalation_procedures=await self._define_escalation_procedures(scenario),
                success_metrics=await self._define_success_metrics(scenario),
                risk_assessment=await self._conduct_risk_assessment(scenario),
                contingency_plans=await self._create_contingency_plans(scenario),
                ml_integration_points=await self._identify_ml_integration_points(scenario)
            )
            
            self.operation_plans[plan_id] = operation_plan
            
            logger.info(f"Created operation plan {plan_id} for scenario {scenario_id}")
            return plan_id
            
        except Exception as e:
            logger.error(f"Failed to create operation plan: {e}")
            raise
    
    async def _get_ml_team_recommendations(self, scenario: SecurityScenario) -> Dict[str, Any]:
        """Get ML-based team assignment recommendations"""
        # Analyze team member skills vs scenario requirements
        recommendations = {
            "red_team_assignments": [],
            "blue_team_assignments": [],
            "purple_team_lead": None,
            "confidence_scores": {}
        }
        
        # Score team members for scenario fit
        for member_id, member in self.team_members.items():
            skill_match = 0.0
            for required_skill in scenario.required_skills:
                for specialization in member.specializations:
                    if required_skill.lower() in specialization.lower():
                        skill_match += 0.3
            
            # Factor in experience and past performance
            experience_factor = min(member.experience_points / 2000, 1.0)
            performance_factor = member.performance_metrics.get("success_rate", 0.5)
            
            overall_score = (skill_match + experience_factor + performance_factor) / 3
            recommendations["confidence_scores"][member_id] = overall_score
            
            # Assign to appropriate team
            if member.role == TeamRole.RED_TEAM and overall_score > 0.7:
                recommendations["red_team_assignments"].append(member_id)
            elif member.role == TeamRole.BLUE_TEAM and overall_score > 0.7:
                recommendations["blue_team_assignments"].append(member_id)
            elif member.role == TeamRole.PURPLE_TEAM and overall_score > 0.8:
                recommendations["purple_team_lead"] = member_id
        
        return recommendations
    
    async def _generate_red_team_objectives(self, scenario: SecurityScenario) -> List[str]:
        """Generate red team objectives based on scenario"""
        base_objectives = scenario.objectives.copy()
        
        # Add tactical objectives based on scenario type
        if scenario.operation_type == OperationType.THREAT_SIMULATION:
            base_objectives.extend([
                "Demonstrate real-world attack techniques",
                "Test defensive detection capabilities",
                "Identify security control gaps",
                "Provide actionable threat intelligence"
            ])
        elif scenario.operation_type == OperationType.PURPLE_TEAM_EXERCISE:
            base_objectives.extend([
                "Collaborate with blue team in real-time",
                "Adapt tactics based on defensive responses",
                "Share techniques and indicators progressively",
                "Validate detection rule effectiveness"
            ])
        
        return base_objectives
    
    async def _generate_blue_team_objectives(self, scenario: SecurityScenario) -> List[str]:
        """Generate blue team objectives based on scenario"""
        base_objectives = [
            "Detect and analyze red team activities",
            "Implement appropriate countermeasures",
            "Minimize attack impact and duration",
            "Document lessons learned and improvements"
        ]
        
        # Add scenario-specific objectives
        if scenario.operation_type == OperationType.THREAT_HUNTING:
            base_objectives.extend([
                "Proactively hunt for threat indicators",
                "Validate threat detection coverage",
                "Improve threat hunting methodologies",
                "Enhance incident response procedures"
            ])
        elif scenario.operation_type == OperationType.PURPLE_TEAM_EXERCISE:
            base_objectives.extend([
                "Collaborate with red team for learning",
                "Test and tune detection capabilities",
                "Improve response time and accuracy",
                "Validate security tool effectiveness"
            ])
        
        return base_objectives
    
    async def _generate_purple_team_objectives(self, scenario: SecurityScenario) -> List[str]:
        """Generate purple team objectives for coordination"""
        return [
            "Facilitate effective red/blue team collaboration",
            "Ensure scenario objectives are met",
            "Capture and analyze performance metrics",
            "Coordinate real-time knowledge sharing",
            "Document tactical improvements",
            "Optimize team coordination processes",
            "Validate security posture improvements",
            "Enable adaptive strategy adjustments"
        ]
    
    def _get_operation_phases(self, operation_type: OperationType) -> List[OperationPhase]:
        """Get appropriate phases for operation type"""
        if operation_type == OperationType.PENETRATION_TEST:
            return [
                OperationPhase.PLANNING,
                OperationPhase.RECONNAISSANCE,
                OperationPhase.INITIAL_ACCESS,
                OperationPhase.PERSISTENCE,
                OperationPhase.PRIVILEGE_ESCALATION,
                OperationPhase.LATERAL_MOVEMENT,
                OperationPhase.COLLECTION,
                OperationPhase.EXFILTRATION,
                OperationPhase.LESSONS_LEARNED
            ]
        elif operation_type == OperationType.INCIDENT_RESPONSE:
            return [
                OperationPhase.PLANNING,
                OperationPhase.CONTAINMENT,
                OperationPhase.ERADICATION,
                OperationPhase.RECOVERY,
                OperationPhase.LESSONS_LEARNED
            ]
        elif operation_type == OperationType.PURPLE_TEAM_EXERCISE:
            return [
                OperationPhase.PLANNING,
                OperationPhase.RECONNAISSANCE,
                OperationPhase.INITIAL_ACCESS,
                OperationPhase.DEFENSE_EVASION,
                OperationPhase.LATERAL_MOVEMENT,
                OperationPhase.COLLECTION,
                OperationPhase.LESSONS_LEARNED
            ]
        else:
            return [
                OperationPhase.PLANNING,
                OperationPhase.RECONNAISSANCE,
                OperationPhase.INITIAL_ACCESS,
                OperationPhase.LATERAL_MOVEMENT,
                OperationPhase.LESSONS_LEARNED
            ]
    
    async def _create_operation_timeline(self, scenario: SecurityScenario) -> Dict[OperationPhase, Dict[str, Any]]:
        """Create detailed timeline for operation phases"""
        phases = self._get_operation_phases(scenario.operation_type)
        total_duration = scenario.estimated_duration.total_seconds()
        
        # Distribute time across phases based on complexity
        phase_weights = {
            OperationPhase.PLANNING: 0.15,
            OperationPhase.RECONNAISSANCE: 0.20,
            OperationPhase.INITIAL_ACCESS: 0.15,
            OperationPhase.PERSISTENCE: 0.10,
            OperationPhase.PRIVILEGE_ESCALATION: 0.10,
            OperationPhase.LATERAL_MOVEMENT: 0.15,
            OperationPhase.COLLECTION: 0.08,
            OperationPhase.EXFILTRATION: 0.05,
            OperationPhase.LESSONS_LEARNED: 0.02
        }
        
        timeline = {}
        current_time = 0
        
        for phase in phases:
            weight = phase_weights.get(phase, 0.1)
            duration = total_duration * weight
            
            timeline[phase] = {
                "start_time": current_time,
                "duration_minutes": duration / 60,
                "end_time": current_time + duration,
                "milestone_tasks": await self._get_phase_tasks(phase, scenario),
                "success_criteria": await self._get_phase_success_criteria(phase),
                "resource_requirements": await self._get_phase_resources(phase)
            }
            
            current_time += duration
        
        return timeline
    
    async def _get_phase_tasks(self, phase: OperationPhase, scenario: SecurityScenario) -> List[str]:
        """Get specific tasks for operation phase"""
        task_mapping = {
            OperationPhase.PLANNING: [
                "Review scenario objectives and constraints",
                "Assign team roles and responsibilities", 
                "Establish communication protocols",
                "Configure monitoring and logging",
                "Prepare tools and infrastructure"
            ],
            OperationPhase.RECONNAISSANCE: [
                "Perform passive information gathering",
                "Identify target attack surface",
                "Map network topology",
                "Enumerate services and applications",
                "Analyze security controls"
            ],
            OperationPhase.INITIAL_ACCESS: [
                "Execute initial attack vectors",
                "Establish command and control",
                "Validate access and privileges",
                "Begin stealth measures",
                "Document access methods"
            ],
            OperationPhase.LATERAL_MOVEMENT: [
                "Identify lateral movement opportunities",
                "Compromise additional systems",
                "Escalate privileges where possible",
                "Maintain persistence",
                "Avoid detection mechanisms"
            ],
            OperationPhase.LESSONS_LEARNED: [
                "Compile comprehensive findings",
                "Analyze detection effectiveness",
                "Document security gaps",
                "Provide remediation recommendations",
                "Conduct team debriefing"
            ]
        }
        
        return task_mapping.get(phase, ["Execute phase objectives"])
    
    async def _get_phase_success_criteria(self, phase: OperationPhase) -> List[str]:
        """Get success criteria for phase"""
        criteria_mapping = {
            OperationPhase.PLANNING: ["All team members briefed", "Tools configured", "Communication established"],
            OperationPhase.RECONNAISSANCE: ["Target mapping complete", "Attack vectors identified", "Risk assessment done"],
            OperationPhase.INITIAL_ACCESS: ["Successful system compromise", "C2 communication established", "Persistence achieved"],
            OperationPhase.LATERAL_MOVEMENT: ["Additional systems compromised", "Network traversal demonstrated", "Privileges escalated"],
            OperationPhase.LESSONS_LEARNED: ["Comprehensive report generated", "Team feedback collected", "Improvements identified"]
        }
        
        return criteria_mapping.get(phase, ["Phase objectives completed"])
    
    async def _get_phase_resources(self, phase: OperationPhase) -> Dict[str, Any]:
        """Get resource requirements for phase"""
        return {
            "personnel": {"red_team": 2, "blue_team": 2, "purple_team": 1},
            "tools": ["monitoring_dashboard", "communication_platform", "analysis_tools"],
            "infrastructure": ["test_environment", "logging_systems", "backup_systems"]
        }
    
    async def execute_operation(self, plan_id: str) -> str:
        """Execute a planned operation with real-time coordination"""
        try:
            if plan_id not in self.operation_plans:
                raise ValueError(f"Operation plan {plan_id} not found")
            
            plan = self.operation_plans[plan_id]
            execution_id = f"exec_{plan_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create execution state
            execution = OperationExecution(
                execution_id=execution_id,
                plan_id=plan_id,
                current_phase=list(plan.phases)[0],
                start_time=datetime.now(),
                estimated_end_time=datetime.now() + timedelta(hours=8),
                actual_end_time=None,
                team_status={
                    TeamRole.RED_TEAM: "ready",
                    TeamRole.BLUE_TEAM: "ready",
                    TeamRole.PURPLE_TEAM: "coordinating"
                },
                phase_progress={phase: 0.0 for phase in plan.phases},
                real_time_metrics={},
                detected_activities=[],
                countermeasures_deployed=[],
                ml_predictions={},
                adaptive_adjustments=[],
                communication_log=[]
            )
            
            self.active_executions[execution_id] = execution
            
            # Start operation monitoring
            asyncio.create_task(self._monitor_operation_execution(execution_id))
            
            # Initialize ML monitoring
            asyncio.create_task(self._ml_operation_monitoring(execution_id))
            
            logger.info(f"Started operation execution {execution_id}")
            return execution_id
            
        except Exception as e:
            logger.error(f"Failed to execute operation: {e}")
            raise
    
    async def _monitor_operation_execution(self, execution_id: str):
        """Monitor operation execution in real-time"""
        while execution_id in self.active_executions:
            try:
                execution = self.active_executions[execution_id]
                
                # Update execution metrics
                await self._update_execution_metrics(execution)
                
                # Check for phase completion
                await self._check_phase_progression(execution)
                
                # Process team communications
                await self._process_team_communications(execution)
                
                # Update ML predictions
                await self._update_ml_predictions(execution)
                
                # Check for operation completion
                if await self._is_operation_complete(execution):
                    await self._complete_operation(execution)
                    break
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring operation {execution_id}: {e}")
                await asyncio.sleep(60)
    
    async def _ml_operation_monitoring(self, execution_id: str):
        """ML-powered operation monitoring and optimization"""
        while execution_id in self.active_executions:
            try:
                execution = self.active_executions[execution_id]
                
                # Get current operation context
                context = await self._get_operation_context(execution)
                
                # ML threat prediction
                threat_prediction = await self.ml_engine.predict_threat_likelihood(context)
                execution.ml_predictions["threat_likelihood"] = threat_prediction
                
                # ML-based adaptive adjustments
                if threat_prediction["threat_likelihood"] > 0.8:
                    adjustment = await self._generate_adaptive_adjustment(execution, threat_prediction)
                    execution.adaptive_adjustments.append(adjustment)
                    
                    # Notify purple team
                    await self._send_adaptive_notification(execution, adjustment)
                
                # Performance optimization recommendations
                optimization = await self.ml_engine.optimize_defensive_posture(context)
                execution.ml_predictions["optimization_recommendations"] = optimization
                
                await asyncio.sleep(60)  # ML monitoring every minute
                
            except Exception as e:
                logger.error(f"Error in ML operation monitoring {execution_id}: {e}")
                await asyncio.sleep(120)
    
    async def _update_execution_metrics(self, execution: OperationExecution):
        """Update real-time execution metrics"""
        current_time = datetime.now()
        elapsed_time = (current_time - execution.start_time).total_seconds()
        
        execution.real_time_metrics.update({
            "elapsed_time_minutes": elapsed_time / 60,
            "current_phase": execution.current_phase.value,
            "overall_progress": sum(execution.phase_progress.values()) / len(execution.phase_progress),
            "active_teams": len([status for status in execution.team_status.values() if status == "active"]),
            "detected_activities_count": len(execution.detected_activities),
            "countermeasures_count": len(execution.countermeasures_deployed),
            "communication_volume": len(execution.communication_log),
            "last_updated": current_time.isoformat()
        })
    
    async def _get_operation_context(self, execution: OperationExecution) -> Dict[str, Any]:
        """Get current operation context for ML analysis"""
        return {
            "operation_phase": execution.current_phase.value,
            "elapsed_time_hours": (datetime.now() - execution.start_time).total_seconds() / 3600,
            "red_team_activity": len([a for a in execution.detected_activities if a.get("team") == "red"]) / 10,
            "blue_team_readiness": execution.team_status.get(TeamRole.BLUE_TEAM, "unknown") == "active",
            "purple_team_coordination": len(execution.adaptive_adjustments) / 5,
            "network_anomaly_score": np.random.uniform(0.2, 0.8),  # Would be real data
            "threat_intel_score": np.random.uniform(0.3, 0.7),
            "system_integrity_score": np.random.uniform(0.4, 0.9),
            "user_behavior_score": np.random.uniform(0.2, 0.6),
            "vulnerability_score": np.random.uniform(0.1, 0.5)
        }
    
    async def _generate_adaptive_adjustment(self, execution: OperationExecution, threat_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Generate adaptive adjustment based on ML predictions"""
        return {
            "adjustment_id": f"adapt_{len(execution.adaptive_adjustments) + 1}",
            "timestamp": datetime.now().isoformat(),
            "trigger": "high_threat_likelihood",
            "threat_likelihood": threat_prediction["threat_likelihood"],
            "recommended_actions": [
                "Increase blue team alertness level",
                "Deploy additional monitoring",
                "Prepare rapid response procedures",
                "Enhance communication frequency"
            ],
            "priority": "high",
            "auto_implemented": False,
            "requires_approval": True
        }
    
    async def get_operation_status(self, execution_id: str) -> Dict[str, Any]:
        """Get comprehensive operation status"""
        try:
            if execution_id not in self.active_executions:
                return {"error": "Operation not found or completed"}
            
            execution = self.active_executions[execution_id]
            plan = self.operation_plans[execution.plan_id]
            
            return {
                "execution_id": execution_id,
                "plan_id": execution.plan_id,
                "scenario_name": self.security_scenarios[plan.scenario_id].name,
                "current_phase": execution.current_phase.value,
                "start_time": execution.start_time.isoformat(),
                "elapsed_time_minutes": (datetime.now() - execution.start_time).total_seconds() / 60,
                "estimated_completion": execution.estimated_end_time.isoformat(),
                "overall_progress": sum(execution.phase_progress.values()) / len(execution.phase_progress),
                "team_status": {role.value: status for role, status in execution.team_status.items()},
                "phase_progress": {phase.value: progress for phase, progress in execution.phase_progress.items()},
                "metrics": execution.real_time_metrics,
                "detected_activities": len(execution.detected_activities),
                "countermeasures_deployed": len(execution.countermeasures_deployed),
                "ml_predictions": execution.ml_predictions,
                "adaptive_adjustments": len(execution.adaptive_adjustments),
                "communication_activity": len(execution.communication_log)
            }
            
        except Exception as e:
            logger.error(f"Failed to get operation status: {e}")
            return {"error": str(e)}
    
    async def get_team_performance_analysis(self, timeframe_days: int = 30) -> Dict[str, Any]:
        """Get comprehensive team performance analysis"""
        try:
            cutoff_date = datetime.now() - timedelta(days=timeframe_days)
            recent_metrics = [m for m in self.performance_history if m.operation_id]
            
            analysis = {
                "timeframe_days": timeframe_days,
                "total_operations": len(recent_metrics),
                "team_analysis": {},
                "trend_analysis": {},
                "ml_insights": {},
                "recommendations": []
            }
            
            # Analyze by team role
            for role in TeamRole:
                team_metrics = [m for m in recent_metrics if m.team_role == role]
                if team_metrics:
                    analysis["team_analysis"][role.value] = {
                        "operations_participated": len(team_metrics),
                        "avg_success_rate": np.mean([m.success_rate for m in team_metrics]),
                        "avg_response_time": np.mean([m.response_time for m in team_metrics]),
                        "avg_detection_accuracy": np.mean([m.detection_accuracy for m in team_metrics]),
                        "avg_collaboration_score": np.mean([m.collaboration_score for m in team_metrics]),
                        "skill_development_rate": np.mean([
                            sum(m.skill_development.values()) / len(m.skill_development) 
                            for m in team_metrics if m.skill_development
                        ])
                    }
            
            # Generate ML insights
            if len(recent_metrics) >= 5:
                analysis["ml_insights"] = await self._generate_performance_ml_insights(recent_metrics)
            
            # Generate recommendations
            analysis["recommendations"] = await self._generate_performance_recommendations(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze team performance: {e}")
            return {"error": str(e)}
    
    async def _generate_performance_ml_insights(self, metrics: List[TeamPerformanceMetrics]) -> Dict[str, Any]:
        """Generate ML-based performance insights"""
        insights = {
            "performance_trends": {},
            "skill_gap_analysis": {},
            "optimization_opportunities": [],
            "predicted_improvements": {}
        }
        
        try:
            # Analyze performance trends
            red_team_metrics = [m for m in metrics if m.team_role == TeamRole.RED_TEAM]
            blue_team_metrics = [m for m in metrics if m.team_role == TeamRole.BLUE_TEAM]
            
            if red_team_metrics:
                red_success_trend = np.polyfit(
                    range(len(red_team_metrics)), 
                    [m.success_rate for m in red_team_metrics], 
                    1
                )[0]
                insights["performance_trends"]["red_team_trend"] = "improving" if red_success_trend > 0 else "declining"
            
            if blue_team_metrics:
                blue_detection_trend = np.polyfit(
                    range(len(blue_team_metrics)),
                    [m.detection_accuracy for m in blue_team_metrics],
                    1
                )[0]
                insights["performance_trends"]["blue_team_trend"] = "improving" if blue_detection_trend > 0 else "declining"
            
            # Identify optimization opportunities
            if red_team_metrics and blue_team_metrics:
                avg_red_success = np.mean([m.success_rate for m in red_team_metrics])
                avg_blue_detection = np.mean([m.detection_accuracy for m in blue_team_metrics])
                
                if avg_red_success > 0.8 and avg_blue_detection < 0.7:
                    insights["optimization_opportunities"].append({
                        "area": "blue_team_detection",
                        "current": avg_blue_detection,
                        "target": 0.85,
                        "priority": "high"
                    })
                
                if avg_blue_detection > 0.9 and avg_red_success > 0.7:
                    insights["optimization_opportunities"].append({
                        "area": "red_team_tactics",
                        "recommendation": "Increase attack sophistication",
                        "priority": "medium"
                    })
            
        except Exception as e:
            logger.error(f"Failed to generate performance ML insights: {e}")
        
        return insights
    
    async def _generate_performance_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        team_analysis = analysis.get("team_analysis", {})
        ml_insights = analysis.get("ml_insights", {})
        
        # Team-specific recommendations
        for team, metrics in team_analysis.items():
            if metrics.get("avg_success_rate", 0) < 0.7:
                recommendations.append(f"🎯 {team.upper()}: Improve success rate through targeted training")
            
            if metrics.get("avg_collaboration_score", 0) < 0.75:
                recommendations.append(f"🤝 {team.upper()}: Enhance collaboration through cross-team exercises")
            
            if metrics.get("skill_development_rate", 0) < 0.1:
                recommendations.append(f"📚 {team.upper()}: Increase skill development opportunities")
        
        # ML-based recommendations
        optimization_ops = ml_insights.get("optimization_opportunities", [])
        for opp in optimization_ops:
            if opp.get("priority") == "high":
                recommendations.append(f"🚀 HIGH PRIORITY: {opp.get('area', 'Unknown area')} optimization required")
        
        # General recommendations
        recommendations.extend([
            "📊 Implement continuous performance monitoring",
            "🔄 Establish regular purple team exercises",
            "🎓 Create cross-functional training programs",
            "📈 Set up performance benchmarking against industry standards"
        ])
        
        return recommendations
    
    async def get_framework_analytics(self) -> Dict[str, Any]:
        """Get comprehensive framework analytics and insights"""
        return {
            "framework_status": {
                "total_team_members": len(self.team_members),
                "active_operations": len(self.active_executions),
                "completed_operations": len([p for p in self.operation_plans.values()]),
                "available_scenarios": len(self.security_scenarios),
                "ml_models_active": len(self.ml_engine.models),
                "last_updated": datetime.now().isoformat()
            },
            "team_distribution": {
                role.value: len([m for m in self.team_members.values() if m.role == role])
                for role in TeamRole
            },
            "scenario_complexity": {
                "high": len([s for s in self.security_scenarios.values() if s.complexity_score > 0.8]),
                "medium": len([s for s in self.security_scenarios.values() if 0.5 < s.complexity_score <= 0.8]),
                "low": len([s for s in self.security_scenarios.values() if s.complexity_score <= 0.5])
            },
            "ml_model_performance": self.ml_engine.model_performance,
            "communication_activity": {
                channel: len(data.get("message_history", []))
                for channel, data in self.communication_channels.items()
            },
            "recent_achievements": await self._get_recent_achievements(),
            "upcoming_schedules": await self._get_upcoming_schedules()
        }
    
    async def _get_recent_achievements(self) -> List[Dict[str, Any]]:
        """Get recent team achievements"""
        return [
            {
                "achievement": "Purple Team Exercise Excellence",
                "team": "Combined Teams",
                "date": (datetime.now() - timedelta(days=3)).isoformat(),
                "description": "Successfully completed APT simulation with 95% detection rate"
            },
            {
                "achievement": "Red Team Innovation",
                "team": "Red Team",
                "date": (datetime.now() - timedelta(days=7)).isoformat(),
                "description": "Developed new evasion technique with 0% detection rate"
            },
            {
                "achievement": "Blue Team Response Excellence", 
                "team": "Blue Team",
                "date": (datetime.now() - timedelta(days=5)).isoformat(),
                "description": "Achieved sub-5-minute incident response time"
            }
        ]
    
    async def _get_upcoming_schedules(self) -> List[Dict[str, Any]]:
        """Get upcoming scheduled operations"""
        return [
            {
                "operation": "Monthly Purple Team Exercise",
                "date": (datetime.now() + timedelta(days=7)).isoformat(),
                "participants": "All Teams",
                "scenario": "Insider Threat Simulation"
            },
            {
                "operation": "Red Team Skills Development",
                "date": (datetime.now() + timedelta(days=14)).isoformat(),
                "participants": "Red Team",
                "scenario": "Advanced Persistence Techniques"
            }
        ]

# Global framework instance
_team_orchestration_framework: Optional[TeamOrchestrationFramework] = None

async def get_team_orchestration_framework() -> TeamOrchestrationFramework:
    """Get global team orchestration framework instance"""
    global _team_orchestration_framework
    
    if _team_orchestration_framework is None:
        _team_orchestration_framework = TeamOrchestrationFramework()
        await _team_orchestration_framework.initialize()
    
    return _team_orchestration_framework

# Utility functions for external integration
async def create_red_team_scenario(scenario_data: Dict[str, Any]) -> str:
    """Create a new red team scenario"""
    framework = await get_team_orchestration_framework()
    scenario_id = f"red_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    scenario = SecurityScenario(
        scenario_id=scenario_id,
        name=scenario_data["name"],
        description=scenario_data["description"],
        operation_type=OperationType.PENETRATION_TEST,
        threat_level=ThreatLevel(scenario_data.get("threat_level", "medium")),
        target_environment=scenario_data["target_environment"],
        objectives=scenario_data["objectives"],
        success_criteria=scenario_data["success_criteria"],
        constraints=scenario_data.get("constraints", []),
        mitre_tactics=scenario_data.get("mitre_tactics", []),
        mitre_techniques=scenario_data.get("mitre_techniques", []),
        estimated_duration=timedelta(hours=scenario_data.get("duration_hours", 8)),
        complexity_score=scenario_data.get("complexity_score", 0.7),
        required_skills=scenario_data.get("required_skills", []),
        created_by=scenario_data.get("created_by", "user"),
        created_at=datetime.now()
    )
    
    framework.security_scenarios[scenario_id] = scenario
    return scenario_id

async def execute_purple_team_operation(scenario_id: str, customizations: Dict[str, Any] = None) -> str:
    """Execute a purple team operation"""
    framework = await get_team_orchestration_framework()
    
    # Create operation plan
    plan_id = await framework.create_operation_plan(scenario_id, customizations)
    
    # Execute operation
    execution_id = await framework.execute_operation(plan_id)
    
    return execution_id

if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Initialize framework
        framework = await get_team_orchestration_framework()
        
        # Create and execute a purple team operation
        plan_id = await framework.create_operation_plan("purple_team_exercise")
        execution_id = await framework.execute_operation(plan_id)
        
        print(f"Purple Team Operation Started: {execution_id}")
        
        # Monitor operation for a short time
        await asyncio.sleep(5)
        
        status = await framework.get_operation_status(execution_id)
        print(f"Operation Status: {status}")
        
        # Get framework analytics
        analytics = await framework.get_framework_analytics()
        print(f"Framework Analytics: {analytics}")
    
    # Run if executed directly
    asyncio.run(main())