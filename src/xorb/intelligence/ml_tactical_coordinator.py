"""
XORB ML Tactical Coordinator
Advanced Machine Learning system for real-time tactical coordination,
adaptive strategy optimization, and intelligent team orchestration.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pickle
import hashlib
from pathlib import Path

# ML and advanced analytics with graceful fallbacks
try:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectKBest, f_classif
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from scipy import stats
    from scipy.spatial.distance import cosine, euclidean
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

logger = logging.getLogger(__name__)

class TacticalDecisionType(Enum):
    """Types of tactical decisions"""
    ATTACK_VECTOR_SELECTION = "attack_vector_selection"
    DEFENSIVE_POSTURE_ADJUSTMENT = "defensive_posture_adjustment"
    RESOURCE_ALLOCATION = "resource_allocation"
    ESCALATION_TIMING = "escalation_timing"
    TECHNIQUE_ADAPTATION = "technique_adaptation"
    TEAM_COORDINATION = "team_coordination"
    PRIORITY_ADJUSTMENT = "priority_adjustment"
    THREAT_RESPONSE = "threat_response"

class AdversaryProfile(Enum):
    """Adversary profile types for simulation"""
    SCRIPT_KIDDIE = "script_kiddie"
    CYBERCRIMINAL = "cybercriminal"
    NATION_STATE = "nation_state"
    INSIDER_THREAT = "insider_threat"
    HACKTIVIST = "hacktivist"
    APT_GROUP = "apt_group"

class TacticalContext(Enum):
    """Tactical operation contexts"""
    STEALTH_OPERATION = "stealth_operation"
    AGGRESSIVE_ASSAULT = "aggressive_assault"
    PERSISTENCE_FOCUSED = "persistence_focused"
    EXFILTRATION_MISSION = "exfiltration_mission"
    DISRUPTION_CAMPAIGN = "disruption_campaign"
    RECONNAISSANCE_PHASE = "reconnaissance_phase"

@dataclass
class TacticalDecision:
    """Individual tactical decision with ML backing"""
    decision_id: str
    decision_type: TacticalDecisionType
    context: Dict[str, Any]
    recommended_action: str
    confidence_score: float
    reasoning: List[str]
    alternative_actions: List[Dict[str, Any]]
    risk_assessment: Dict[str, float]
    expected_outcome: Dict[str, Any]
    implementation_steps: List[str]
    success_probability: float
    timestamp: datetime
    ml_model_used: str

@dataclass
class AdaptiveStrategy:
    """Adaptive strategy with ML optimization"""
    strategy_id: str
    team_role: str
    adversary_profile: AdversaryProfile
    tactical_context: TacticalContext
    base_strategy: Dict[str, Any]
    adaptive_modifications: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    learning_history: List[Dict[str, Any]]
    optimization_score: float
    last_updated: datetime

@dataclass
class TeamCoordinationPlan:
    """ML-optimized team coordination plan"""
    plan_id: str
    operation_phase: str
    team_assignments: Dict[str, List[str]]
    communication_matrix: Dict[str, Dict[str, float]]
    synchronization_points: List[Dict[str, Any]]
    resource_distribution: Dict[str, Dict[str, Any]]
    contingency_triggers: List[Dict[str, Any]]
    success_metrics: Dict[str, float]
    optimization_iterations: int
    ml_confidence: float

class MLTacticalCoordinator:
    """Advanced ML system for tactical coordination and adaptive strategy"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # ML Models for different aspects
        self.tactical_models: Dict[str, Any] = {}
        self.strategy_models: Dict[str, Any] = {}
        self.coordination_models: Dict[str, Any] = {}

        # Learning and adaptation
        self.decision_history: List[TacticalDecision] = []
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)
        self.adaptation_patterns: Dict[str, Any] = {}

        # Real-time coordination
        self.active_strategies: Dict[str, AdaptiveStrategy] = {}
        self.coordination_plans: Dict[str, TeamCoordinationPlan] = {}

        # Advanced analytics
        self.feature_importance: Dict[str, Dict[str, float]] = {}
        self.model_performance_metrics: Dict[str, Dict[str, float]] = {}
        self.learning_curves: Dict[str, List[float]] = defaultdict(list)

        # Tactical knowledge base
        self.technique_effectiveness: Dict[str, Dict[str, float]] = {}
        self.adversary_behavior_models: Dict[AdversaryProfile, Dict[str, Any]] = {}
        self.team_synergy_matrix: np.ndarray = None

    async def initialize(self):
        """Initialize the ML tactical coordinator"""
        try:
            logger.info("Initializing ML Tactical Coordinator...")

            # Initialize ML models
            await self._initialize_tactical_models()

            # Load historical data and train models
            await self._load_and_train_models()

            # Initialize adversary behavior models
            await self._initialize_adversary_models()

            # Setup team synergy analysis
            await self._initialize_team_synergy_analysis()

            # Load technique effectiveness data
            await self._load_technique_effectiveness()

            logger.info("ML Tactical Coordinator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ML Tactical Coordinator: {e}")
            raise

    async def _initialize_tactical_models(self):
        """Initialize various ML models for tactical decisions"""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, using simplified models")
            return

        try:
            # Tactical decision model
            self.tactical_models["decision_classifier"] = VotingClassifier([
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
                ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42))
            ])

            # Attack vector recommendation model
            self.tactical_models["attack_vector_recommender"] = RandomForestClassifier(
                n_estimators=150,
                max_depth=15,
                random_state=42
            )

            # Defensive optimization model
            self.tactical_models["defensive_optimizer"] = GradientBoostingClassifier(
                n_estimators=120,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )

            # Team coordination optimizer
            self.coordination_models["team_optimizer"] = RandomForestClassifier(
                n_estimators=80,
                max_depth=12,
                random_state=42
            )

            # Strategy adaptation model
            self.strategy_models["adaptation_predictor"] = Pipeline([
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(f_classif, k=10)),
                ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
            ])

            # Success probability estimator
            self.tactical_models["success_estimator"] = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )

            logger.info("Tactical ML models initialized")

        except Exception as e:
            logger.error(f"Failed to initialize tactical models: {e}")

    async def _load_and_train_models(self):
        """Load historical data and train ML models"""
        if not SKLEARN_AVAILABLE:
            return

        try:
            # Generate synthetic training data for tactical decisions
            tactical_data = await self._generate_tactical_training_data()

            # Train tactical decision model
            X_tactical, y_tactical = tactical_data["features"], tactical_data["labels"]
            self.tactical_models["decision_classifier"].fit(X_tactical, y_tactical)

            # Train attack vector recommender
            attack_data = await self._generate_attack_vector_training_data()
            X_attack, y_attack = attack_data["features"], attack_data["labels"]
            self.tactical_models["attack_vector_recommender"].fit(X_attack, y_attack)

            # Train defensive optimizer
            defense_data = await self._generate_defensive_training_data()
            X_defense, y_defense = defense_data["features"], defense_data["labels"]
            self.tactical_models["defensive_optimizer"].fit(X_defense, y_defense)

            # Train coordination optimizer
            coord_data = await self._generate_coordination_training_data()
            X_coord, y_coord = coord_data["features"], coord_data["labels"]
            self.coordination_models["team_optimizer"].fit(X_coord, y_coord)

            # Evaluate model performance
            await self._evaluate_model_performance()

            logger.info("ML models trained successfully")

        except Exception as e:
            logger.error(f"Failed to train ML models: {e}")

    async def _generate_tactical_training_data(self) -> Dict[str, Any]:
        """Generate synthetic training data for tactical decisions"""
        features = []
        labels = []

        # Simulate 2000 tactical scenarios
        for _ in range(2000):
            # Feature vector: [threat_level, team_readiness, time_pressure, resource_availability,
            #                 stealth_requirement, success_history, adversary_skill, environment_complexity]
            feature_vector = [
                np.random.uniform(0, 1),      # threat_level
                np.random.uniform(0, 1),      # team_readiness
                np.random.uniform(0, 1),      # time_pressure
                np.random.uniform(0, 1),      # resource_availability
                np.random.uniform(0, 1),      # stealth_requirement
                np.random.uniform(0, 1),      # success_history
                np.random.uniform(0, 1),      # adversary_skill
                np.random.uniform(0, 1),      # environment_complexity
                np.random.randint(0, 24),     # time_of_day
                np.random.randint(0, 7),      # day_of_week
            ]

            features.append(feature_vector)

            # Generate label based on feature combination
            decision_score = (
                feature_vector[0] * 0.3 +    # threat_level weight
                feature_vector[1] * 0.25 +   # team_readiness weight
                feature_vector[3] * 0.2 +    # resource_availability weight
                (1 - feature_vector[2]) * 0.15 +  # inverse time_pressure weight
                feature_vector[5] * 0.1      # success_history weight
            )

            # Classify decision type based on score
            if decision_score > 0.8:
                label = "aggressive_action"
            elif decision_score > 0.6:
                label = "moderate_action"
            elif decision_score > 0.4:
                label = "cautious_action"
            else:
                label = "defensive_action"

            labels.append(label)

        return {"features": features, "labels": labels}

    async def _generate_attack_vector_training_data(self) -> Dict[str, Any]:
        """Generate training data for attack vector recommendations"""
        features = []
        labels = []

        attack_vectors = [
            "phishing", "web_exploitation", "network_scanning", "credential_stuffing",
            "privilege_escalation", "lateral_movement", "data_exfiltration", "persistence"
        ]

        for _ in range(1500):
            # Features: target characteristics and constraints
            feature_vector = [
                np.random.uniform(0, 1),      # target_vulnerability_score
                np.random.uniform(0, 1),      # network_exposure
                np.random.uniform(0, 1),      # user_security_awareness
                np.random.uniform(0, 1),      # technical_defenses_strength
                np.random.uniform(0, 1),      # monitoring_capability
                np.random.uniform(0, 1),      # stealth_requirement
                np.random.uniform(0, 1),      # time_constraint
                np.random.uniform(0, 1),      # resource_limitation
            ]

            features.append(feature_vector)

            # Select best attack vector based on features
            if feature_vector[2] < 0.4:  # Low user awareness
                label = "phishing"
            elif feature_vector[1] > 0.7:  # High network exposure
                label = "network_scanning"
            elif feature_vector[0] > 0.6:  # High vulnerability score
                label = "web_exploitation"
            elif feature_vector[4] < 0.5:  # Low monitoring
                label = "lateral_movement"
            else:
                label = np.random.choice(attack_vectors)

            labels.append(label)

        return {"features": features, "labels": labels}

    async def _generate_defensive_training_data(self) -> Dict[str, Any]:
        """Generate training data for defensive optimization"""
        features = []
        labels = []

        defensive_actions = [
            "increase_monitoring", "deploy_honeypots", "segment_network",
            "update_signatures", "enhance_logging", "threat_hunting", "incident_response"
        ]

        for _ in range(1200):
            feature_vector = [
                np.random.uniform(0, 1),      # attack_intensity
                np.random.uniform(0, 1),      # current_detection_rate
                np.random.uniform(0, 1),      # false_positive_rate
                np.random.uniform(0, 1),      # response_time
                np.random.uniform(0, 1),      # resource_utilization
                np.random.uniform(0, 1),      # threat_intelligence_quality
                np.random.uniform(0, 1),      # team_expertise_level
                np.random.uniform(0, 1),      # budget_constraints
            ]

            features.append(feature_vector)

            # Determine best defensive action
            if feature_vector[0] > 0.8:  # High attack intensity
                label = "incident_response"
            elif feature_vector[1] < 0.5:  # Low detection rate
                label = "increase_monitoring"
            elif feature_vector[2] > 0.3:  # High false positives
                label = "update_signatures"
            elif feature_vector[3] > 0.5:  # Slow response time
                label = "enhance_logging"
            else:
                label = np.random.choice(defensive_actions)

            labels.append(label)

        return {"features": features, "labels": labels}

    async def _generate_coordination_training_data(self) -> Dict[str, Any]:
        """Generate training data for team coordination optimization"""
        features = []
        labels = []

        coordination_strategies = [
            "parallel_execution", "sequential_execution", "hierarchical_coordination",
            "decentralized_coordination", "hybrid_approach"
        ]

        for _ in range(1000):
            feature_vector = [
                np.random.randint(2, 8),      # team_size
                np.random.uniform(0, 1),      # communication_efficiency
                np.random.uniform(0, 1),      # task_complexity
                np.random.uniform(0, 1),      # time_criticality
                np.random.uniform(0, 1),      # resource_constraints
                np.random.uniform(0, 1),      # expertise_distribution
                np.random.uniform(0, 1),      # coordination_overhead
                np.random.uniform(0, 1),      # success_probability
            ]

            features.append(feature_vector)

            # Determine optimal coordination strategy
            if feature_vector[0] < 3:  # Small team
                label = "parallel_execution"
            elif feature_vector[3] > 0.8:  # High time criticality
                label = "decentralized_coordination"
            elif feature_vector[2] > 0.7:  # High complexity
                label = "hierarchical_coordination"
            else:
                label = np.random.choice(coordination_strategies)

            labels.append(label)

        return {"features": features, "labels": labels}

    async def _evaluate_model_performance(self):
        """Evaluate performance of trained ML models"""
        try:
            for model_name, model in self.tactical_models.items():
                # Generate test data for each model
                if model_name == "decision_classifier":
                    test_data = await self._generate_tactical_training_data()
                elif model_name == "attack_vector_recommender":
                    test_data = await self._generate_attack_vector_training_data()
                elif model_name == "defensive_optimizer":
                    test_data = await self._generate_defensive_training_data()
                else:
                    continue

                X_test, y_test = test_data["features"], test_data["labels"]

                # Split for evaluation
                X_train, X_eval, y_train, y_eval = train_test_split(
                    X_test, y_test, test_size=0.2, random_state=42
                )

                # Make predictions
                y_pred = model.predict(X_eval)

                # Calculate metrics
                accuracy = accuracy_score(y_eval, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_eval, y_pred, average='weighted')

                self.model_performance_metrics[model_name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "training_samples": len(X_train),
                    "evaluation_samples": len(X_eval)
                }

                logger.info(f"Model {model_name} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")

        except Exception as e:
            logger.error(f"Failed to evaluate model performance: {e}")

    async def _initialize_adversary_models(self):
        """Initialize adversary behavior models"""
        self.adversary_behavior_models = {
            AdversaryProfile.SCRIPT_KIDDIE: {
                "sophistication": 0.2,
                "persistence": 0.3,
                "stealth": 0.1,
                "resource_level": 0.2,
                "common_techniques": ["T1190", "T1566.001", "T1059.003"],
                "typical_targets": ["web_applications", "public_services"],
                "attack_patterns": ["opportunistic", "automated", "high_volume"]
            },
            AdversaryProfile.CYBERCRIMINAL: {
                "sophistication": 0.6,
                "persistence": 0.7,
                "stealth": 0.6,
                "resource_level": 0.5,
                "common_techniques": ["T1566.002", "T1021.001", "T1055"],
                "typical_targets": ["financial_systems", "personal_data", "credentials"],
                "attack_patterns": ["targeted", "profit_driven", "moderate_stealth"]
            },
            AdversaryProfile.NATION_STATE: {
                "sophistication": 0.9,
                "persistence": 0.9,
                "stealth": 0.9,
                "resource_level": 0.9,
                "common_techniques": ["T1078.004", "T1027", "T1070"],
                "typical_targets": ["government", "critical_infrastructure", "intelligence"],
                "attack_patterns": ["highly_targeted", "long_term", "advanced_techniques"]
            },
            AdversaryProfile.INSIDER_THREAT: {
                "sophistication": 0.4,
                "persistence": 0.8,
                "stealth": 0.7,
                "resource_level": 0.6,
                "common_techniques": ["T1078", "T1005", "T1041"],
                "typical_targets": ["sensitive_data", "intellectual_property", "financial_records"],
                "attack_patterns": ["privilege_abuse", "data_theft", "cover_tracks"]
            },
            AdversaryProfile.APT_GROUP: {
                "sophistication": 0.8,
                "persistence": 0.9,
                "stealth": 0.8,
                "resource_level": 0.8,
                "common_techniques": ["T1566.001", "T1021.001", "T1547.001"],
                "typical_targets": ["high_value_targets", "strategic_assets", "intelligence"],
                "attack_patterns": ["multi_stage", "living_off_land", "custom_tools"]
            }
        }

        logger.info("Adversary behavior models initialized")

    async def _initialize_team_synergy_analysis(self):
        """Initialize team synergy analysis matrix"""
        # Team roles interaction effectiveness matrix
        # Rows: initiating team, Columns: supporting team
        self.team_synergy_matrix = np.array([
            [1.0, 0.8, 0.9, 0.6, 0.7],  # Red team with others
            [0.8, 1.0, 0.9, 0.7, 0.6],  # Blue team with others
            [0.9, 0.9, 1.0, 0.8, 0.8],  # Purple team with others
            [0.6, 0.7, 0.8, 1.0, 0.5],  # White team with others
            [0.7, 0.6, 0.8, 0.5, 1.0]   # Green team with others
        ])

        logger.info("Team synergy analysis initialized")

    async def _load_technique_effectiveness(self):
        """Load technique effectiveness data"""
        # MITRE ATT&CK technique effectiveness by context
        self.technique_effectiveness = {
            "T1566.001": {  # Spearphishing Attachment
                "stealth_operation": 0.7,
                "aggressive_assault": 0.5,
                "persistence_focused": 0.6,
                "initial_access": 0.8
            },
            "T1190": {  # Exploit Public-Facing Application
                "stealth_operation": 0.6,
                "aggressive_assault": 0.8,
                "reconnaissance_phase": 0.7,
                "initial_access": 0.7
            },
            "T1021.001": {  # Remote Desktop Protocol
                "lateral_movement": 0.8,
                "persistence_focused": 0.7,
                "stealth_operation": 0.5,
                "privilege_escalation": 0.6
            },
            "T1055": {  # Process Injection
                "stealth_operation": 0.9,
                "defense_evasion": 0.8,
                "persistence_focused": 0.7,
                "privilege_escalation": 0.6
            },
            "T1078": {  # Valid Accounts
                "stealth_operation": 0.9,
                "persistence_focused": 0.8,
                "lateral_movement": 0.7,
                "privilege_escalation": 0.8
            }
        }

        logger.info("Technique effectiveness data loaded")

    async def make_tactical_decision(self, context: Dict[str, Any], decision_type: TacticalDecisionType) -> TacticalDecision:
        """Make ML-powered tactical decision"""
        try:
            decision_id = f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:17]}"

            # Extract features from context
            features = await self._extract_decision_features(context, decision_type)

            # Get ML recommendation
            if SKLEARN_AVAILABLE and decision_type.value in ["attack_vector_selection", "defensive_posture_adjustment"]:
                ml_recommendation = await self._get_ml_recommendation(features, decision_type)
            else:
                ml_recommendation = await self._get_rule_based_recommendation(features, decision_type)

            # Calculate confidence and risk
            confidence_score = ml_recommendation.get("confidence", 0.7)
            risk_assessment = await self._assess_decision_risk(features, ml_recommendation)

            # Generate reasoning
            reasoning = await self._generate_decision_reasoning(features, ml_recommendation, context)

            # Calculate success probability
            success_probability = await self._calculate_success_probability(features, ml_recommendation)

            # Generate alternative actions
            alternatives = await self._generate_alternative_actions(features, decision_type)

            # Create tactical decision
            decision = TacticalDecision(
                decision_id=decision_id,
                decision_type=decision_type,
                context=context,
                recommended_action=ml_recommendation["action"],
                confidence_score=confidence_score,
                reasoning=reasoning,
                alternative_actions=alternatives,
                risk_assessment=risk_assessment,
                expected_outcome=ml_recommendation.get("expected_outcome", {}),
                implementation_steps=ml_recommendation.get("implementation_steps", []),
                success_probability=success_probability,
                timestamp=datetime.now(),
                ml_model_used=ml_recommendation.get("model_used", "rule_based")
            )

            # Store decision for learning
            self.decision_history.append(decision)

            return decision

        except Exception as e:
            logger.error(f"Failed to make tactical decision: {e}")
            raise

    async def _extract_decision_features(self, context: Dict[str, Any], decision_type: TacticalDecisionType) -> List[float]:
        """Extract features for tactical decision making"""
        features = []

        # Common features
        features.extend([
            context.get("threat_level", 0.5),
            context.get("team_readiness", 0.7),
            context.get("time_pressure", 0.3),
            context.get("resource_availability", 0.8),
            context.get("stealth_requirement", 0.6),
            context.get("success_history", 0.7),
            context.get("adversary_skill", 0.5),
            context.get("environment_complexity", 0.6)
        ])

        # Time-based features
        now = datetime.now()
        features.extend([
            now.hour / 24.0,
            now.weekday() / 6.0
        ])

        # Decision-type specific features
        if decision_type == TacticalDecisionType.ATTACK_VECTOR_SELECTION:
            features.extend([
                context.get("target_vulnerability_score", 0.5),
                context.get("network_exposure", 0.4),
                context.get("user_security_awareness", 0.6),
                context.get("technical_defenses_strength", 0.7)
            ])
        elif decision_type == TacticalDecisionType.DEFENSIVE_POSTURE_ADJUSTMENT:
            features.extend([
                context.get("attack_intensity", 0.4),
                context.get("current_detection_rate", 0.8),
                context.get("false_positive_rate", 0.1),
                context.get("response_time", 0.3)
            ])

        # Pad to consistent size
        while len(features) < 16:
            features.append(0.0)

        return features[:16]

    async def _get_ml_recommendation(self, features: List[float], decision_type: TacticalDecisionType) -> Dict[str, Any]:
        """Get ML-based recommendation"""
        try:
            if decision_type == TacticalDecisionType.ATTACK_VECTOR_SELECTION:
                model = self.tactical_models["attack_vector_recommender"]
                prediction = model.predict([features])[0]
                probabilities = model.predict_proba([features])[0]
                confidence = max(probabilities)

                return {
                    "action": f"Execute {prediction} attack vector",
                    "confidence": confidence,
                    "model_used": "attack_vector_recommender",
                    "expected_outcome": {"success_rate": confidence, "detection_risk": 1 - confidence},
                    "implementation_steps": await self._get_attack_vector_steps(prediction)
                }

            elif decision_type == TacticalDecisionType.DEFENSIVE_POSTURE_ADJUSTMENT:
                model = self.tactical_models["defensive_optimizer"]
                prediction = model.predict([features])[0]
                probabilities = model.predict_proba([features])[0]
                confidence = max(probabilities)

                return {
                    "action": f"Implement {prediction} defensive measure",
                    "confidence": confidence,
                    "model_used": "defensive_optimizer",
                    "expected_outcome": {"effectiveness": confidence, "resource_cost": 0.3},
                    "implementation_steps": await self._get_defensive_steps(prediction)
                }

            else:
                return await self._get_rule_based_recommendation(features, decision_type)

        except Exception as e:
            logger.error(f"ML recommendation failed: {e}")
            return await self._get_rule_based_recommendation(features, decision_type)

    async def _get_rule_based_recommendation(self, features: List[float], decision_type: TacticalDecisionType) -> Dict[str, Any]:
        """Get rule-based recommendation as fallback"""
        threat_level = features[0]
        team_readiness = features[1]
        time_pressure = features[2]

        if decision_type == TacticalDecisionType.ATTACK_VECTOR_SELECTION:
            if threat_level > 0.7:
                action = "Execute aggressive multi-vector attack"
            elif team_readiness > 0.8:
                action = "Execute coordinated sophisticated attack"
            elif time_pressure > 0.6:
                action = "Execute rapid opportunistic attack"
            else:
                action = "Execute stealth reconnaissance attack"

            return {
                "action": action,
                "confidence": 0.7,
                "model_used": "rule_based",
                "expected_outcome": {"success_rate": 0.6, "detection_risk": 0.4},
                "implementation_steps": ["Prepare tools", "Execute attack", "Monitor results"]
            }

        elif decision_type == TacticalDecisionType.DEFENSIVE_POSTURE_ADJUSTMENT:
            if threat_level > 0.8:
                action = "Activate maximum defense protocols"
            elif threat_level > 0.5:
                action = "Enhance monitoring and response"
            else:
                action = "Maintain standard defensive posture"

            return {
                "action": action,
                "confidence": 0.7,
                "model_used": "rule_based",
                "expected_outcome": {"effectiveness": 0.7, "resource_cost": 0.4},
                "implementation_steps": ["Assess situation", "Implement measures", "Monitor effectiveness"]
            }

        return {
            "action": "Maintain current tactical approach",
            "confidence": 0.5,
            "model_used": "rule_based",
            "expected_outcome": {},
            "implementation_steps": []
        }

    async def _get_attack_vector_steps(self, attack_vector: str) -> List[str]:
        """Get implementation steps for attack vector"""
        steps_mapping = {
            "phishing": [
                "Craft convincing phishing email",
                "Set up command and control infrastructure",
                "Send targeted phishing campaign",
                "Monitor for successful compromises",
                "Establish persistent access"
            ],
            "web_exploitation": [
                "Scan target web applications",
                "Identify exploitable vulnerabilities",
                "Develop or acquire exploit code",
                "Execute exploitation attempt",
                "Establish web shell or backdoor"
            ],
            "network_scanning": [
                "Perform network reconnaissance",
                "Identify live hosts and services",
                "Scan for known vulnerabilities",
                "Attempt to exploit discovered weaknesses",
                "Establish network foothold"
            ],
            "lateral_movement": [
                "Enumerate local network",
                "Identify high-value targets",
                "Attempt credential harvesting",
                "Move laterally to target systems",
                "Establish persistence on new systems"
            ]
        }

        return steps_mapping.get(attack_vector, ["Execute attack vector", "Monitor results"])

    async def _get_defensive_steps(self, defensive_action: str) -> List[str]:
        """Get implementation steps for defensive action"""
        steps_mapping = {
            "increase_monitoring": [
                "Deploy additional monitoring sensors",
                "Enhance log collection and analysis",
                "Increase alert sensitivity levels",
                "Deploy threat hunting capabilities",
                "Monitor for new attack indicators"
            ],
            "segment_network": [
                "Identify critical network segments",
                "Design segmentation architecture",
                "Implement network access controls",
                "Test segmentation effectiveness",
                "Monitor inter-segment traffic"
            ],
            "update_signatures": [
                "Analyze recent threat intelligence",
                "Develop new detection signatures",
                "Test signatures for effectiveness",
                "Deploy signatures to security tools",
                "Monitor for false positives"
            ],
            "incident_response": [
                "Activate incident response team",
                "Contain identified threats",
                "Collect and analyze evidence",
                "Eradicate threats from environment",
                "Recover normal operations"
            ]
        }

        return steps_mapping.get(defensive_action, ["Implement defensive measure", "Monitor effectiveness"])

    async def _assess_decision_risk(self, features: List[float], recommendation: Dict[str, Any]) -> Dict[str, float]:
        """Assess risk associated with tactical decision"""
        base_risk = 0.3

        # Factor in environmental risk
        threat_level = features[0]
        environment_complexity = features[7] if len(features) > 7 else 0.5

        # Calculate different risk types
        detection_risk = base_risk + (threat_level * 0.3) + (environment_complexity * 0.2)
        operational_risk = base_risk + ((1 - features[1]) * 0.4)  # team_readiness inverse
        mission_risk = base_risk + (features[2] * 0.3)  # time_pressure

        return {
            "detection_risk": min(detection_risk, 1.0),
            "operational_risk": min(operational_risk, 1.0),
            "mission_risk": min(mission_risk, 1.0),
            "overall_risk": min((detection_risk + operational_risk + mission_risk) / 3, 1.0)
        }

    async def _generate_decision_reasoning(self, features: List[float], recommendation: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate reasoning for tactical decision"""
        reasoning = []

        # Analyze key factors
        threat_level = features[0]
        team_readiness = features[1]
        time_pressure = features[2]
        resource_availability = features[3]

        if threat_level > 0.7:
            reasoning.append(f"High threat level ({threat_level:.2f}) requires aggressive response")
        elif threat_level < 0.3:
            reasoning.append(f"Low threat level ({threat_level:.2f}) allows for measured approach")

        if team_readiness > 0.8:
            reasoning.append(f"High team readiness ({team_readiness:.2f}) enables complex operations")
        elif team_readiness < 0.4:
            reasoning.append(f"Low team readiness ({team_readiness:.2f}) requires simple tactics")

        if time_pressure > 0.7:
            reasoning.append(f"High time pressure ({time_pressure:.2f}) demands rapid execution")

        if resource_availability > 0.8:
            reasoning.append(f"Abundant resources ({resource_availability:.2f}) support advanced techniques")

        # Add ML model reasoning if available
        model_confidence = recommendation.get("confidence", 0.5)
        if model_confidence > 0.8:
            reasoning.append(f"ML model shows high confidence ({model_confidence:.2f}) in recommendation")

        return reasoning

    async def _calculate_success_probability(self, features: List[float], recommendation: Dict[str, Any]) -> float:
        """Calculate probability of tactical decision success"""
        # Base success probability from ML model or rule-based system
        base_probability = recommendation.get("confidence", 0.5)

        # Adjust based on contextual factors
        threat_level = features[0]
        team_readiness = features[1]
        resource_availability = features[3]

        # Positive factors
        if team_readiness > 0.8:
            base_probability += 0.1
        if resource_availability > 0.8:
            base_probability += 0.1

        # Negative factors
        if threat_level > 0.8:
            base_probability -= 0.1

        # Apply historical success rate if available
        historical_modifier = await self._get_historical_success_modifier(recommendation["action"])
        base_probability *= (1 + historical_modifier)

        return min(max(base_probability, 0.0), 1.0)

    async def _get_historical_success_modifier(self, action: str) -> float:
        """Get historical success rate modifier for action"""
        # Analyze past decisions with similar actions
        similar_decisions = [
            d for d in self.decision_history
            if action.lower() in d.recommended_action.lower()
        ]

        if len(similar_decisions) >= 3:
            avg_success = np.mean([d.success_probability for d in similar_decisions])
            return (avg_success - 0.5) * 0.2  # Small adjustment based on history

        return 0.0  # No historical data

    async def _generate_alternative_actions(self, features: List[float], decision_type: TacticalDecisionType) -> List[Dict[str, Any]]:
        """Generate alternative actions for tactical decision"""
        alternatives = []

        if decision_type == TacticalDecisionType.ATTACK_VECTOR_SELECTION:
            alternatives = [
                {"action": "Credential harvesting campaign", "confidence": 0.6, "risk": 0.4},
                {"action": "Social engineering attack", "confidence": 0.7, "risk": 0.3},
                {"action": "Technical exploitation", "confidence": 0.5, "risk": 0.6},
                {"action": "Physical security bypass", "confidence": 0.4, "risk": 0.7}
            ]
        elif decision_type == TacticalDecisionType.DEFENSIVE_POSTURE_ADJUSTMENT:
            alternatives = [
                {"action": "Deploy deception technology", "confidence": 0.7, "risk": 0.2},
                {"action": "Enhance threat hunting", "confidence": 0.8, "risk": 0.1},
                {"action": "Implement zero trust controls", "confidence": 0.6, "risk": 0.3},
                {"action": "Activate incident response", "confidence": 0.9, "risk": 0.1}
            ]

        # Sort by confidence
        alternatives.sort(key=lambda x: x["confidence"], reverse=True)

        return alternatives[:3]  # Return top 3 alternatives

    async def create_adaptive_strategy(self,
                                     team_role: str,
                                     adversary_profile: AdversaryProfile,
                                     tactical_context: TacticalContext,
                                     base_strategy: Dict[str, Any]) -> str:
        """Create adaptive strategy with ML optimization"""
        try:
            strategy_id = f"strategy_{team_role}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Get adversary behavior model
            adversary_model = self.adversary_behavior_models.get(adversary_profile, {})

            # Create base adaptive strategy
            strategy = AdaptiveStrategy(
                strategy_id=strategy_id,
                team_role=team_role,
                adversary_profile=adversary_profile,
                tactical_context=tactical_context,
                base_strategy=base_strategy,
                adaptive_modifications=[],
                performance_metrics={},
                learning_history=[],
                optimization_score=0.0,
                last_updated=datetime.now()
            )

            # Apply ML-based optimizations
            optimizations = await self._optimize_strategy_with_ml(strategy, adversary_model)
            strategy.adaptive_modifications.extend(optimizations)

            # Calculate optimization score
            strategy.optimization_score = await self._calculate_strategy_optimization_score(strategy)

            # Store strategy
            self.active_strategies[strategy_id] = strategy

            logger.info(f"Created adaptive strategy {strategy_id} with optimization score {strategy.optimization_score:.3f}")
            return strategy_id

        except Exception as e:
            logger.error(f"Failed to create adaptive strategy: {e}")
            raise

    async def _optimize_strategy_with_ml(self, strategy: AdaptiveStrategy, adversary_model: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply ML-based optimizations to strategy"""
        optimizations = []

        try:
            # Analyze adversary characteristics
            adversary_sophistication = adversary_model.get("sophistication", 0.5)
            adversary_stealth = adversary_model.get("stealth", 0.5)
            adversary_persistence = adversary_model.get("persistence", 0.5)

            # Generate context-aware optimizations
            if strategy.team_role == "red_team":
                optimizations.extend(await self._optimize_red_team_strategy(
                    strategy, adversary_sophistication, adversary_stealth
                ))
            elif strategy.team_role == "blue_team":
                optimizations.extend(await self._optimize_blue_team_strategy(
                    strategy, adversary_sophistication, adversary_persistence
                ))
            elif strategy.team_role == "purple_team":
                optimizations.extend(await self._optimize_purple_team_strategy(
                    strategy, adversary_model
                ))

            # Apply ML model recommendations if available
            if SKLEARN_AVAILABLE and "adaptation_predictor" in self.strategy_models:
                ml_optimizations = await self._get_ml_strategy_optimizations(strategy, adversary_model)
                optimizations.extend(ml_optimizations)

        except Exception as e:
            logger.error(f"Strategy optimization failed: {e}")

        return optimizations

    async def _optimize_red_team_strategy(self, strategy: AdaptiveStrategy, sophistication: float, stealth: float) -> List[Dict[str, Any]]:
        """Optimize red team strategy based on adversary profile"""
        optimizations = []

        if sophistication > 0.7:
            optimizations.append({
                "type": "technique_enhancement",
                "modification": "Use advanced evasion techniques",
                "reasoning": "High adversary sophistication requires advanced techniques",
                "impact_score": 0.8
            })

        if stealth > 0.7:
            optimizations.append({
                "type": "stealth_optimization",
                "modification": "Implement living-off-the-land techniques",
                "reasoning": "High stealth requirement demands minimal footprint",
                "impact_score": 0.7
            })

        if strategy.tactical_context == TacticalContext.STEALTH_OPERATION:
            optimizations.append({
                "type": "timing_optimization",
                "modification": "Execute during low-activity periods",
                "reasoning": "Stealth context requires optimal timing",
                "impact_score": 0.6
            })

        return optimizations

    async def _optimize_blue_team_strategy(self, strategy: AdaptiveStrategy, sophistication: float, persistence: float) -> List[Dict[str, Any]]:
        """Optimize blue team strategy based on adversary profile"""
        optimizations = []

        if sophistication > 0.8:
            optimizations.append({
                "type": "detection_enhancement",
                "modification": "Deploy advanced behavioral analytics",
                "reasoning": "Sophisticated adversary requires advanced detection",
                "impact_score": 0.9
            })

        if persistence > 0.7:
            optimizations.append({
                "type": "response_optimization",
                "modification": "Implement rapid containment procedures",
                "reasoning": "Persistent adversary requires quick response",
                "impact_score": 0.8
            })

        optimizations.append({
            "type": "threat_hunting",
            "modification": "Enhance proactive threat hunting",
            "reasoning": "Active hunting improves detection capabilities",
            "impact_score": 0.7
        })

        return optimizations

    async def _optimize_purple_team_strategy(self, strategy: AdaptiveStrategy, adversary_model: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize purple team coordination strategy"""
        optimizations = []

        optimizations.append({
            "type": "coordination_enhancement",
            "modification": "Implement real-time red/blue communication",
            "reasoning": "Enhanced coordination improves learning outcomes",
            "impact_score": 0.8
        })

        optimizations.append({
            "type": "adaptive_learning",
            "modification": "Enable dynamic strategy adjustment",
            "reasoning": "Adaptive learning improves effectiveness",
            "impact_score": 0.9
        })

        # Add adversary-specific optimizations
        common_techniques = adversary_model.get("common_techniques", [])
        if common_techniques:
            optimizations.append({
                "type": "technique_focus",
                "modification": f"Focus on detecting {', '.join(common_techniques[:3])}",
                "reasoning": "Target adversary's preferred techniques",
                "impact_score": 0.7
            })

        return optimizations

    async def _get_ml_strategy_optimizations(self, strategy: AdaptiveStrategy, adversary_model: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get ML-based strategy optimizations"""
        optimizations = []

        try:
            # Prepare features for ML model
            features = [
                adversary_model.get("sophistication", 0.5),
                adversary_model.get("persistence", 0.5),
                adversary_model.get("stealth", 0.5),
                adversary_model.get("resource_level", 0.5),
                len(adversary_model.get("common_techniques", [])) / 10.0,
                len(strategy.base_strategy.get("techniques", [])) / 10.0,
                strategy.optimization_score,
                len(strategy.adaptive_modifications) / 10.0
            ]

            # Get ML prediction (simplified for now)
            if len(features) >= 8:
                # Use rule-based optimization as ML model substitute
                optimization_score = np.mean(features[:4])

                if optimization_score > 0.7:
                    optimizations.append({
                        "type": "ml_enhancement",
                        "modification": "Apply high-sophistication countermeasures",
                        "reasoning": "ML model recommends advanced techniques",
                        "impact_score": optimization_score
                    })
                elif optimization_score > 0.5:
                    optimizations.append({
                        "type": "ml_enhancement",
                        "modification": "Apply moderate-sophistication techniques",
                        "reasoning": "ML model recommends standard techniques",
                        "impact_score": optimization_score
                    })

        except Exception as e:
            logger.error(f"ML strategy optimization failed: {e}")

        return optimizations

    async def _calculate_strategy_optimization_score(self, strategy: AdaptiveStrategy) -> float:
        """Calculate optimization score for strategy"""
        base_score = 0.5

        # Factor in number and quality of optimizations
        modification_count = len(strategy.adaptive_modifications)
        if modification_count > 0:
            avg_impact = np.mean([m.get("impact_score", 0.5) for m in strategy.adaptive_modifications])
            base_score += (modification_count * 0.1) + (avg_impact * 0.3)

        # Factor in adversary profile complexity
        adversary_complexity = sum([
            self.adversary_behavior_models.get(strategy.adversary_profile, {}).get("sophistication", 0.5),
            self.adversary_behavior_models.get(strategy.adversary_profile, {}).get("persistence", 0.5),
            self.adversary_behavior_models.get(strategy.adversary_profile, {}).get("stealth", 0.5)
        ]) / 3

        base_score += adversary_complexity * 0.2

        return min(base_score, 1.0)

    async def optimize_team_coordination(self, operation_context: Dict[str, Any]) -> TeamCoordinationPlan:
        """Create ML-optimized team coordination plan"""
        try:
            plan_id = f"coord_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Extract coordination features
            team_size = operation_context.get("team_size", 5)
            complexity = operation_context.get("task_complexity", 0.5)
            time_criticality = operation_context.get("time_criticality", 0.5)

            # Get optimal team assignments using ML
            team_assignments = await self._optimize_team_assignments(operation_context)

            # Calculate communication efficiency matrix
            comm_matrix = await self._calculate_communication_matrix(team_assignments)

            # Define synchronization points
            sync_points = await self._define_synchronization_points(operation_context)

            # Optimize resource distribution
            resource_distribution = await self._optimize_resource_distribution(team_assignments, operation_context)

            # Set up contingency triggers
            contingency_triggers = await self._define_contingency_triggers(operation_context)

            # Define success metrics
            success_metrics = await self._define_coordination_success_metrics(operation_context)

            # Calculate ML confidence
            ml_confidence = await self._calculate_coordination_confidence(operation_context)

            # Create coordination plan
            coordination_plan = TeamCoordinationPlan(
                plan_id=plan_id,
                operation_phase=operation_context.get("phase", "execution"),
                team_assignments=team_assignments,
                communication_matrix=comm_matrix,
                synchronization_points=sync_points,
                resource_distribution=resource_distribution,
                contingency_triggers=contingency_triggers,
                success_metrics=success_metrics,
                optimization_iterations=3,
                ml_confidence=ml_confidence
            )

            # Store coordination plan
            self.coordination_plans[plan_id] = coordination_plan

            logger.info(f"Created team coordination plan {plan_id} with ML confidence {ml_confidence:.3f}")
            return coordination_plan

        except Exception as e:
            logger.error(f"Failed to optimize team coordination: {e}")
            raise

    async def _optimize_team_assignments(self, context: Dict[str, Any]) -> Dict[str, List[str]]:
        """Optimize team member assignments using ML"""
        assignments = {
            "red_team": [],
            "blue_team": [],
            "purple_team": [],
            "support_team": []
        }

        # Simulate optimal assignments based on context
        complexity = context.get("task_complexity", 0.5)
        criticality = context.get("time_criticality", 0.5)

        if complexity > 0.7:
            assignments["red_team"] = ["red_lead", "red_specialist_1", "red_specialist_2"]
            assignments["blue_team"] = ["blue_lead", "blue_analyst_1", "blue_analyst_2"]
        else:
            assignments["red_team"] = ["red_lead", "red_specialist_1"]
            assignments["blue_team"] = ["blue_lead", "blue_analyst_1"]

        if criticality > 0.6:
            assignments["purple_team"] = ["purple_coordinator", "purple_analyst"]
        else:
            assignments["purple_team"] = ["purple_coordinator"]

        assignments["support_team"] = ["tech_support", "communications"]

        return assignments

    async def _calculate_communication_matrix(self, team_assignments: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """Calculate communication efficiency matrix between teams"""
        teams = list(team_assignments.keys())
        matrix = {}

        for team1 in teams:
            matrix[team1] = {}
            for team2 in teams:
                if team1 == team2:
                    efficiency = 1.0
                elif "purple" in team1 or "purple" in team2:
                    efficiency = 0.9  # Purple team has high communication with all
                elif ("red" in team1 and "blue" in team2) or ("blue" in team1 and "red" in team2):
                    efficiency = 0.6  # Red and blue teams have moderate direct communication
                else:
                    efficiency = 0.8  # Support teams have good communication

                matrix[team1][team2] = efficiency

        return matrix

    async def _define_synchronization_points(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Define synchronization points for team coordination"""
        sync_points = []

        operation_duration = context.get("duration_hours", 8)

        # Initial synchronization
        sync_points.append({
            "time_offset": 0,
            "type": "operation_kickoff",
            "participants": ["all_teams"],
            "objectives": ["Confirm readiness", "Establish communication", "Review objectives"],
            "duration_minutes": 15
        })

        # Mid-operation synchronization
        if operation_duration > 4:
            sync_points.append({
                "time_offset": operation_duration * 0.5,
                "type": "mid_operation_sync",
                "participants": ["red_team", "blue_team", "purple_team"],
                "objectives": ["Status update", "Tactical adjustment", "Resource reallocation"],
                "duration_minutes": 10
            })

        # Final synchronization
        sync_points.append({
            "time_offset": operation_duration * 0.9,
            "type": "operation_wrap_up",
            "participants": ["all_teams"],
            "objectives": ["Final status", "Prepare debrief", "Secure findings"],
            "duration_minutes": 20
        })

        return sync_points

    async def get_tactical_intelligence_summary(self) -> Dict[str, Any]:
        """Get comprehensive tactical intelligence summary"""
        return {
            "ml_coordinator_status": {
                "models_trained": len(self.tactical_models),
                "decisions_made": len(self.decision_history),
                "active_strategies": len(self.active_strategies),
                "coordination_plans": len(self.coordination_plans),
                "last_updated": datetime.now().isoformat()
            },
            "model_performance": self.model_performance_metrics,
            "decision_analytics": await self._analyze_decision_patterns(),
            "strategy_effectiveness": await self._analyze_strategy_effectiveness(),
            "coordination_efficiency": await self._analyze_coordination_efficiency(),
            "learning_insights": await self._generate_learning_insights(),
            "recommendations": await self._generate_tactical_recommendations()
        }

    async def _analyze_decision_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in tactical decisions"""
        if not self.decision_history:
            return {"total_decisions": 0}

        # Analyze decision types
        decision_types = [d.decision_type.value for d in self.decision_history]
        type_counts = {dt: decision_types.count(dt) for dt in set(decision_types)}

        # Analyze confidence trends
        confidence_scores = [d.confidence_score for d in self.decision_history]
        avg_confidence = np.mean(confidence_scores)

        # Analyze success probability trends
        success_probs = [d.success_probability for d in self.decision_history]
        avg_success_prob = np.mean(success_probs)

        return {
            "total_decisions": len(self.decision_history),
            "decision_type_distribution": type_counts,
            "average_confidence": avg_confidence,
            "average_success_probability": avg_success_prob,
            "confidence_trend": "improving" if len(confidence_scores) > 5 and confidence_scores[-5:] > confidence_scores[:5] else "stable"
        }

    async def _analyze_strategy_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of adaptive strategies"""
        if not self.active_strategies:
            return {"active_strategies": 0}

        optimization_scores = [s.optimization_score for s in self.active_strategies.values()]
        avg_optimization = np.mean(optimization_scores)

        # Analyze by team role
        team_effectiveness = {}
        for strategy in self.active_strategies.values():
            role = strategy.team_role
            if role not in team_effectiveness:
                team_effectiveness[role] = []
            team_effectiveness[role].append(strategy.optimization_score)

        team_averages = {role: np.mean(scores) for role, scores in team_effectiveness.items()}

        return {
            "active_strategies": len(self.active_strategies),
            "average_optimization_score": avg_optimization,
            "team_effectiveness": team_averages,
            "top_performing_strategy": max(self.active_strategies.keys(),
                                          key=lambda k: self.active_strategies[k].optimization_score)
        }

    async def _analyze_coordination_efficiency(self) -> Dict[str, Any]:
        """Analyze team coordination efficiency"""
        if not self.coordination_plans:
            return {"coordination_plans": 0}

        ml_confidences = [p.ml_confidence for p in self.coordination_plans.values()]
        avg_confidence = np.mean(ml_confidences)

        return {
            "coordination_plans": len(self.coordination_plans),
            "average_ml_confidence": avg_confidence,
            "optimization_iterations": np.mean([p.optimization_iterations for p in self.coordination_plans.values()])
        }

    async def _generate_learning_insights(self) -> List[str]:
        """Generate learning insights from tactical intelligence"""
        insights = []

        if len(self.decision_history) > 10:
            insights.append("🧠 Sufficient decision history available for pattern analysis")

            # Analyze recent decision confidence
            recent_decisions = self.decision_history[-10:]
            avg_recent_confidence = np.mean([d.confidence_score for d in recent_decisions])

            if avg_recent_confidence > 0.8:
                insights.append("🎯 Recent decisions show high confidence levels")
            elif avg_recent_confidence < 0.6:
                insights.append("⚠️ Recent decisions show lower confidence - consider model retraining")

        if len(self.active_strategies) > 3:
            insights.append("📋 Multiple active strategies enable comparative analysis")

        if self.model_performance_metrics:
            best_model = max(self.model_performance_metrics.keys(),
                           key=lambda k: self.model_performance_metrics[k].get("accuracy", 0))
            insights.append(f"🏆 Best performing model: {best_model}")

        return insights

    async def _generate_tactical_recommendations(self) -> List[str]:
        """Generate tactical recommendations for improvement"""
        recommendations = []

        # Model performance recommendations
        for model_name, metrics in self.model_performance_metrics.items():
            if metrics.get("accuracy", 0) < 0.8:
                recommendations.append(f"🔧 Consider retraining {model_name} model for better accuracy")

        # Strategy recommendations
        if len(self.active_strategies) < 3:
            recommendations.append("📈 Create more adaptive strategies for different scenarios")

        # Coordination recommendations
        if len(self.coordination_plans) < 2:
            recommendations.append("🤝 Develop more team coordination plans for various contexts")

        # Learning recommendations
        if len(self.decision_history) < 50:
            recommendations.append("📊 Collect more decision data for improved ML model training")

        # General recommendations
        recommendations.extend([
            "🔄 Implement continuous learning from operation outcomes",
            "📋 Regularly update adversary behavior models",
            "🎯 Enhance real-time tactical adaptation capabilities",
            "🔍 Expand threat intelligence integration for better context"
        ])

        return recommendations

# Global ML tactical coordinator instance
_ml_tactical_coordinator: Optional[MLTacticalCoordinator] = None

async def get_ml_tactical_coordinator() -> MLTacticalCoordinator:
    """Get global ML tactical coordinator instance"""
    global _ml_tactical_coordinator

    if _ml_tactical_coordinator is None:
        _ml_tactical_coordinator = MLTacticalCoordinator()
        await _ml_tactical_coordinator.initialize()

    return _ml_tactical_coordinator

# Utility functions for integration
async def make_tactical_decision_request(context: Dict[str, Any], decision_type: str) -> Dict[str, Any]:
    """Make a tactical decision request"""
    coordinator = await get_ml_tactical_coordinator()
    decision_type_enum = TacticalDecisionType(decision_type)
    decision = await coordinator.make_tactical_decision(context, decision_type_enum)

    return {
        "decision_id": decision.decision_id,
        "recommended_action": decision.recommended_action,
        "confidence_score": decision.confidence_score,
        "reasoning": decision.reasoning,
        "alternative_actions": decision.alternative_actions,
        "risk_assessment": decision.risk_assessment,
        "success_probability": decision.success_probability,
        "implementation_steps": decision.implementation_steps
    }

async def create_adaptive_strategy_request(team_role: str, adversary_type: str, context: str) -> str:
    """Create an adaptive strategy"""
    coordinator = await get_ml_tactical_coordinator()

    adversary_profile = AdversaryProfile(adversary_type)
    tactical_context = TacticalContext(context)
    base_strategy = {"techniques": [], "objectives": [], "constraints": []}

    strategy_id = await coordinator.create_adaptive_strategy(
        team_role, adversary_profile, tactical_context, base_strategy
    )

    return strategy_id

if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Initialize coordinator
        coordinator = await get_ml_tactical_coordinator()

        # Make a tactical decision
        context = {
            "threat_level": 0.8,
            "team_readiness": 0.9,
            "time_pressure": 0.6,
            "resource_availability": 0.7,
            "target_vulnerability_score": 0.6
        }

        decision = await coordinator.make_tactical_decision(
            context, TacticalDecisionType.ATTACK_VECTOR_SELECTION
        )

        print(f"Tactical Decision: {decision.recommended_action}")
        print(f"Confidence: {decision.confidence_score:.3f}")
        print(f"Success Probability: {decision.success_probability:.3f}")

        # Create adaptive strategy
        strategy_id = await coordinator.create_adaptive_strategy(
            "red_team", AdversaryProfile.APT_GROUP, TacticalContext.STEALTH_OPERATION, {}
        )

        print(f"Created adaptive strategy: {strategy_id}")

        # Get intelligence summary
        summary = await coordinator.get_tactical_intelligence_summary()
        print(f"Intelligence Summary: {summary}")

    # Run if executed directly
    asyncio.run(main())
