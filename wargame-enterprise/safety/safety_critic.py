#!/usr/bin/env python3
"""
AI Safety Critic for Cyber Range
Real-time AI action monitoring and safety validation
"""

import time
import json
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import re
import logging
import numpy as np
from pathlib import Path

# Machine Learning for safety prediction
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. ML safety features disabled.")

# Large Language Model integration for safety reasoning
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class SafetyLevel(Enum):
    """Safety assessment levels"""
    SAFE = "safe"           # Action is completely safe
    CAUTION = "caution"     # Action requires monitoring
    WARNING = "warning"     # Action has potential risks
    DANGEROUS = "dangerous" # Action poses significant risk
    FORBIDDEN = "forbidden" # Action must be blocked

class ViolationType(Enum):
    """Types of safety violations"""
    SCOPE_VIOLATION = "scope_violation"
    DESTRUCTIVE_ACTION = "destructive_action"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_DESTRUCTION = "data_destruction"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    RESOURCE_ABUSE = "resource_abuse"
    MALICIOUS_PAYLOAD = "malicious_payload"
    SOCIAL_ENGINEERING = "social_engineering"
    COMPLIANCE_BREACH = "compliance_breach"
    ETHICAL_VIOLATION = "ethical_violation"

class ActionCategory(Enum):
    """Categories of actions for safety analysis"""
    RECONNAISSANCE = "reconnaissance"
    EXPLOITATION = "exploitation"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"
    DEFENSE = "defense"
    RESPONSE = "response"

@dataclass
class SafetyRule:
    """Individual safety rule for action validation"""
    rule_id: str
    name: str
    description: str
    violation_type: ViolationType
    severity: SafetyLevel
    
    # Rule matching
    action_patterns: List[str] = field(default_factory=list)
    target_patterns: List[str] = field(default_factory=list)
    payload_patterns: List[str] = field(default_factory=list)
    
    # Context requirements
    required_permissions: List[str] = field(default_factory=list)
    forbidden_targets: List[str] = field(default_factory=list)
    time_restrictions: Dict[str, Any] = field(default_factory=dict)
    
    # ML-based detection
    anomaly_threshold: float = 0.5
    behavioral_model: Optional[str] = None
    
    enabled: bool = True
    created_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

@dataclass
class ActionContext:
    """Context for an action being evaluated"""
    action_id: str
    timestamp: str
    episode_id: str
    agent_id: str
    agent_type: str
    
    # Action details
    action_category: ActionCategory
    action_type: str
    target: str
    payload: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Environment context
    current_permissions: List[str] = field(default_factory=list)
    network_location: str = ""
    system_state: Dict[str, Any] = field(default_factory=dict)
    
    # Historical context
    previous_actions: List[str] = field(default_factory=list)
    attack_chain_position: int = 0
    session_duration: float = 0.0
    
    # Compliance context
    compliance_frameworks: List[str] = field(default_factory=list)
    data_sensitivity: str = "none"  # none, low, medium, high, critical

@dataclass
class SafetyAssessment:
    """Result of safety evaluation"""
    action_id: str
    timestamp: str
    safety_level: SafetyLevel
    allowed: bool
    
    # Analysis results
    rule_violations: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    anomaly_score: float = 0.0
    confidence: float = 1.0
    
    # Reasoning
    reasoning: str = ""
    mitigation_suggestions: List[str] = field(default_factory=list)
    alternative_actions: List[str] = field(default_factory=list)
    
    # Context preservation
    action_context: Optional[ActionContext] = None
    evaluation_time_ms: float = 0.0

class BehavioralAnalyzer:
    """ML-based behavioral analysis for safety"""
    
    def __init__(self):
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.anomaly_detector = IsolationForest(contamination=0.1) if SKLEARN_AVAILABLE else None
        self.action_embeddings = {}
        self.behavioral_profiles = {}
        self.is_trained = False
    
    def extract_features(self, context: ActionContext) -> np.ndarray:
        """Extract numerical features for ML analysis"""
        if not SKLEARN_AVAILABLE:
            return np.array([0.0])
        
        features = []
        
        # Temporal features
        hour = datetime.fromisoformat(context.timestamp.replace('Z', '+00:00')).hour
        features.extend([
            hour / 24.0,  # Time of day
            context.session_duration / 3600.0,  # Session duration in hours
            context.attack_chain_position / 10.0,  # Chain position normalized
        ])
        
        # Action features
        features.extend([
            len(context.action_type) / 50.0,  # Action complexity
            len(context.target) / 100.0,  # Target complexity
            len(context.payload or "") / 1000.0,  # Payload size
            len(context.parameters) / 20.0,  # Parameter count
        ])
        
        # Permission features
        features.extend([
            len(context.current_permissions) / 10.0,
            1.0 if "admin" in context.current_permissions else 0.0,
            1.0 if "root" in context.current_permissions else 0.0,
        ])
        
        # Historical features
        features.extend([
            len(context.previous_actions) / 50.0,
            len(set(context.previous_actions)) / 20.0,  # Action diversity
        ])
        
        # Categorical features (one-hot encoded)
        categories = [cat.value for cat in ActionCategory]
        cat_features = [1.0 if context.action_category.value == cat else 0.0 for cat in categories]
        features.extend(cat_features)
        
        return np.array(features).reshape(1, -1)
    
    def train_on_historical_data(self, historical_contexts: List[ActionContext]):
        """Train behavioral models on historical data"""
        if not SKLEARN_AVAILABLE or not historical_contexts:
            return
        
        # Extract features from historical data
        feature_matrix = []
        for context in historical_contexts:
            features = self.extract_features(context)
            feature_matrix.append(features.flatten())
        
        feature_matrix = np.array(feature_matrix)
        
        # Fit scaler and anomaly detector
        self.scaler.fit(feature_matrix)
        scaled_features = self.scaler.transform(feature_matrix)
        self.anomaly_detector.fit(scaled_features)
        
        # Build agent behavioral profiles
        agent_profiles = {}
        for context in historical_contexts:
            agent_id = context.agent_id
            if agent_id not in agent_profiles:
                agent_profiles[agent_id] = []
            
            features = self.extract_features(context)
            agent_profiles[agent_id].append(features.flatten())
        
        # Calculate agent behavioral centroids
        for agent_id, agent_features in agent_profiles.items():
            if len(agent_features) > 1:
                centroid = np.mean(agent_features, axis=0)
                self.behavioral_profiles[agent_id] = centroid
        
        self.is_trained = True
        logging.info(f"Trained behavioral analyzer on {len(historical_contexts)} historical actions")
    
    def analyze_action(self, context: ActionContext) -> Tuple[float, Dict[str, float]]:
        """Analyze action for behavioral anomalies"""
        if not SKLEARN_AVAILABLE or not self.is_trained:
            return 0.0, {}
        
        features = self.extract_features(context)
        scaled_features = self.scaler.transform(features)
        
        # Global anomaly detection
        anomaly_score = self.anomaly_detector.decision_function(scaled_features)[0]
        anomaly_score = max(0, min(1, (anomaly_score + 0.5) * 2))  # Normalize to 0-1
        
        analysis_results = {
            "global_anomaly": anomaly_score,
            "agent_deviation": 0.0,
            "temporal_anomaly": 0.0,
            "action_novelty": 0.0
        }
        
        # Agent-specific behavioral deviation
        if context.agent_id in self.behavioral_profiles:
            agent_centroid = self.behavioral_profiles[context.agent_id]
            similarity = cosine_similarity(features, agent_centroid.reshape(1, -1))[0][0]
            analysis_results["agent_deviation"] = 1.0 - similarity
        
        # Temporal anomaly (actions outside normal hours)
        hour = datetime.fromisoformat(context.timestamp.replace('Z', '+00:00')).hour
        if hour < 6 or hour > 22:  # Outside normal hours
            analysis_results["temporal_anomaly"] = 0.7
        
        # Action novelty (new action types)
        action_key = f"{context.action_category.value}_{context.action_type}"
        if action_key not in self.action_embeddings:
            analysis_results["action_novelty"] = 0.8
            self.action_embeddings[action_key] = features.flatten()
        
        return anomaly_score, analysis_results

class SafetyCritic:
    """AI Safety Critic for real-time action monitoring"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.rules: Dict[str, SafetyRule] = {}
        self.assessments: List[SafetyAssessment] = []
        self.behavioral_analyzer = BehavioralAnalyzer()
        
        # Configuration
        self.config_file = config_file
        self.strict_mode = True
        self.learning_enabled = True
        self.llm_reasoning_enabled = OPENAI_AVAILABLE
        
        # Statistics
        self.stats = {
            "total_evaluations": 0,
            "actions_blocked": 0,
            "actions_allowed": 0,
            "violations_detected": 0,
            "anomalies_detected": 0
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Logging
        self._setup_logging()
        
        # Load configuration and initialize
        self._load_configuration()
        self._initialize_safety_rules()
    
    def _setup_logging(self):
        """Setup safety critic logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('SafetyCritic')
    
    def _load_configuration(self):
        """Load safety critic configuration"""
        if self.config_file and Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                self.strict_mode = config.get('strict_mode', True)
                self.learning_enabled = config.get('learning_enabled', True)
                self.llm_reasoning_enabled = config.get('llm_reasoning_enabled', OPENAI_AVAILABLE)
                
                # Load custom rules
                for rule_data in config.get('rules', []):
                    rule = SafetyRule(**rule_data)
                    self.rules[rule.rule_id] = rule
                
            except Exception as e:
                self.logger.error(f"Failed to load safety critic configuration: {e}")
    
    def _initialize_safety_rules(self):
        """Initialize essential safety rules"""
        # Destructive action rule
        destructive_rule = SafetyRule(
            rule_id="destructive_actions",
            name="Destructive Action Prevention",
            description="Prevent actions that could cause data loss or system damage",
            violation_type=ViolationType.DESTRUCTIVE_ACTION,
            severity=SafetyLevel.FORBIDDEN,
            action_patterns=[
                r"rm\s+-rf",
                r"format\s+c:",
                r"DROP\s+DATABASE",
                r"shutdown\s+",
                r"reboot\s+",
                r"mkfs\.",
                r"dd\s+.*of=/dev/"
            ],
            payload_patterns=[
                r">/dev/null",
                r"2>&1",
                r"\|\s*sh",
                r"eval\s*\(",
                r"exec\s*\("
            ]
        )
        self.add_rule(destructive_rule)
        
        # Privilege escalation rule
        privesc_rule = SafetyRule(
            rule_id="privilege_escalation",
            name="Privilege Escalation Detection",
            description="Detect and control privilege escalation attempts",
            violation_type=ViolationType.PRIVILEGE_ESCALATION,
            severity=SafetyLevel.DANGEROUS,
            action_patterns=[
                r"sudo\s+",
                r"su\s+",
                r"setuid",
                r"chmod\s+\+s",
                r"passwd\s+"
            ],
            target_patterns=[
                r"/etc/passwd",
                r"/etc/shadow",
                r"/etc/sudoers",
                r"/bin/su",
                r"/usr/bin/sudo"
            ]
        )
        self.add_rule(privesc_rule)
        
        # Data exfiltration rule
        exfil_rule = SafetyRule(
            rule_id="data_exfiltration",
            name="Data Exfiltration Prevention",
            description="Monitor and control data exfiltration attempts",
            violation_type=ViolationType.UNAUTHORIZED_ACCESS,
            severity=SafetyLevel.WARNING,
            action_patterns=[
                r"scp\s+",
                r"rsync\s+",
                r"curl\s+.*-d",
                r"wget\s+.*--post",
                r"nc\s+.*<"
            ],
            forbidden_targets=[
                "external_network",
                "unknown_host",
                "suspicious_domain"
            ]
        )
        self.add_rule(exfil_rule)
        
        # Malicious payload rule
        payload_rule = SafetyRule(
            rule_id="malicious_payload",
            name="Malicious Payload Detection",
            description="Detect potentially malicious payloads",
            violation_type=ViolationType.MALICIOUS_PAYLOAD,
            severity=SafetyLevel.DANGEROUS,
            payload_patterns=[
                r"<script.*>.*</script>",
                r"javascript:",
                r"vbscript:",
                r"onload\s*=",
                r"onerror\s*=",
                r"\\x[0-9a-fA-F]{2}",  # Hex encoding
                r"%[0-9a-fA-F]{2}",    # URL encoding
                r"\$\(.*\)",           # Command substitution
                r"`.*`"                # Backtick execution
            ]
        )
        self.add_rule(payload_rule)
        
        # Compliance violation rule
        compliance_rule = SafetyRule(
            rule_id="compliance_violation",
            name="Compliance Framework Violation",
            description="Ensure actions comply with regulatory frameworks",
            violation_type=ViolationType.COMPLIANCE_BREACH,
            severity=SafetyLevel.WARNING,
            forbidden_targets=[
                "pii_database",
                "financial_records",
                "health_records",
                "payment_systems"
            ]
        )
        self.add_rule(compliance_rule)
    
    def add_rule(self, rule: SafetyRule):
        """Add a safety rule"""
        with self._lock:
            self.rules[rule.rule_id] = rule
            self.logger.info(f"Added safety rule: {rule.name}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a safety rule"""
        with self._lock:
            if rule_id in self.rules:
                rule = self.rules.pop(rule_id)
                self.logger.info(f"Removed safety rule: {rule.name}")
                return True
            return False
    
    def evaluate_action(self, context: ActionContext) -> SafetyAssessment:
        """Evaluate an action for safety"""
        start_time = time.time()
        
        with self._lock:
            self.stats["total_evaluations"] += 1
        
        assessment = SafetyAssessment(
            action_id=context.action_id,
            timestamp=datetime.utcnow().isoformat(),
            safety_level=SafetyLevel.SAFE,
            allowed=True,
            action_context=context
        )
        
        try:
            # Rule-based evaluation
            rule_violations = self._evaluate_rules(context)
            assessment.rule_violations = rule_violations
            
            # ML-based behavioral analysis
            if self.behavioral_analyzer.is_trained:
                anomaly_score, analysis_details = self.behavioral_analyzer.analyze_action(context)
                assessment.anomaly_score = anomaly_score
                
                # Add anomaly-based risk factors
                for analysis_type, score in analysis_details.items():
                    if score > 0.7:
                        assessment.risk_factors.append(f"High {analysis_type}: {score:.2f}")
            
            # Determine overall safety level
            assessment.safety_level = self._calculate_safety_level(
                rule_violations, assessment.anomaly_score, context
            )
            
            # Make allow/deny decision
            assessment.allowed = self._make_decision(assessment.safety_level, context)
            
            # Generate reasoning and suggestions
            assessment.reasoning = self._generate_reasoning(assessment, context)
            assessment.mitigation_suggestions = self._generate_mitigations(assessment, context)
            assessment.alternative_actions = self._suggest_alternatives(assessment, context)
            
            # Calculate confidence
            assessment.confidence = self._calculate_confidence(assessment, context)
            
            # Update statistics
            with self._lock:
                if assessment.allowed:
                    self.stats["actions_allowed"] += 1
                else:
                    self.stats["actions_blocked"] += 1
                
                if rule_violations:
                    self.stats["violations_detected"] += len(rule_violations)
                
                if assessment.anomaly_score > 0.7:
                    self.stats["anomalies_detected"] += 1
            
            # Store assessment
            self.assessments.append(assessment)
            
            # Log significant events
            if not assessment.allowed:
                self.logger.warning(
                    f"BLOCKED ACTION: {context.action_type} by {context.agent_id} "
                    f"(Safety: {assessment.safety_level.value}, Violations: {len(rule_violations)})"
                )
            elif assessment.safety_level in [SafetyLevel.WARNING, SafetyLevel.DANGEROUS]:
                self.logger.info(
                    f"RISKY ACTION ALLOWED: {context.action_type} by {context.agent_id} "
                    f"(Safety: {assessment.safety_level.value})"
                )
        
        except Exception as e:
            self.logger.error(f"Error in safety evaluation: {e}")
            # Fail safe - block action on evaluation error
            assessment.safety_level = SafetyLevel.DANGEROUS
            assessment.allowed = False
            assessment.reasoning = f"Safety evaluation failed: {str(e)}"
        
        finally:
            assessment.evaluation_time_ms = (time.time() - start_time) * 1000
        
        return assessment
    
    def _evaluate_rules(self, context: ActionContext) -> List[str]:
        """Evaluate action against all safety rules"""
        violations = []
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # Check action patterns
            for pattern in rule.action_patterns:
                if re.search(pattern, context.action_type, re.IGNORECASE):
                    violations.append(rule.rule_id)
                    break
            
            # Check target patterns
            for pattern in rule.target_patterns:
                if re.search(pattern, context.target, re.IGNORECASE):
                    violations.append(rule.rule_id)
                    break
            
            # Check payload patterns
            if context.payload:
                for pattern in rule.payload_patterns:
                    if re.search(pattern, context.payload, re.IGNORECASE):
                        violations.append(rule.rule_id)
                        break
            
            # Check forbidden targets
            for forbidden in rule.forbidden_targets:
                if forbidden.lower() in context.target.lower():
                    violations.append(rule.rule_id)
                    break
            
            # Check required permissions
            if rule.required_permissions:
                missing_perms = set(rule.required_permissions) - set(context.current_permissions)
                if missing_perms:
                    violations.append(rule.rule_id)
        
        return list(set(violations))  # Remove duplicates
    
    def _calculate_safety_level(self, violations: List[str], anomaly_score: float, 
                              context: ActionContext) -> SafetyLevel:
        """Calculate overall safety level"""
        # Start with safe assumption
        level = SafetyLevel.SAFE
        
        # Check rule violations
        if violations:
            violation_severities = [
                self.rules[rule_id].severity for rule_id in violations 
                if rule_id in self.rules
            ]
            
            if SafetyLevel.FORBIDDEN in violation_severities:
                level = SafetyLevel.FORBIDDEN
            elif SafetyLevel.DANGEROUS in violation_severities:
                level = SafetyLevel.DANGEROUS
            elif SafetyLevel.WARNING in violation_severities:
                level = SafetyLevel.WARNING
            else:
                level = SafetyLevel.CAUTION
        
        # Factor in anomaly score
        if anomaly_score > 0.9:
            if level.value < SafetyLevel.DANGEROUS.value:
                level = SafetyLevel.DANGEROUS
        elif anomaly_score > 0.7:
            if level.value < SafetyLevel.WARNING.value:
                level = SafetyLevel.WARNING
        elif anomaly_score > 0.5:
            if level.value < SafetyLevel.CAUTION.value:
                level = SafetyLevel.CAUTION
        
        # Consider context factors
        if context.data_sensitivity in ["high", "critical"]:
            if level.value < SafetyLevel.WARNING.value:
                level = SafetyLevel.WARNING
        
        if "admin" in context.current_permissions or "root" in context.current_permissions:
            if level.value < SafetyLevel.CAUTION.value:
                level = SafetyLevel.CAUTION
        
        return level
    
    def _make_decision(self, safety_level: SafetyLevel, context: ActionContext) -> bool:
        """Make allow/deny decision based on safety level"""
        if safety_level == SafetyLevel.FORBIDDEN:
            return False
        
        if safety_level == SafetyLevel.DANGEROUS:
            if self.strict_mode:
                return False
            # In non-strict mode, allow with logging
            return True
        
        # All other levels are allowed
        return True
    
    def _generate_reasoning(self, assessment: SafetyAssessment, context: ActionContext) -> str:
        """Generate human-readable reasoning for the decision"""
        reasoning_parts = []
        
        # Rule violations
        if assessment.rule_violations:
            violated_rules = [
                self.rules[rule_id].name for rule_id in assessment.rule_violations 
                if rule_id in self.rules
            ]
            reasoning_parts.append(f"Rule violations: {', '.join(violated_rules)}")
        
        # Anomaly detection
        if assessment.anomaly_score > 0.5:
            reasoning_parts.append(f"Behavioral anomaly detected (score: {assessment.anomaly_score:.2f})")
        
        # Risk factors
        if assessment.risk_factors:
            reasoning_parts.append(f"Risk factors: {', '.join(assessment.risk_factors)}")
        
        # Context considerations
        if context.data_sensitivity in ["high", "critical"]:
            reasoning_parts.append(f"High data sensitivity: {context.data_sensitivity}")
        
        if not reasoning_parts:
            reasoning_parts.append("No significant safety concerns detected")
        
        return "; ".join(reasoning_parts)
    
    def _generate_mitigations(self, assessment: SafetyAssessment, context: ActionContext) -> List[str]:
        """Generate mitigation suggestions"""
        mitigations = []
        
        if assessment.rule_violations:
            mitigations.append("Review and modify action to comply with safety rules")
        
        if assessment.anomaly_score > 0.7:
            mitigations.append("Consider additional validation for this unusual action")
        
        if "admin" in context.current_permissions:
            mitigations.append("Use minimal required privileges instead of admin access")
        
        if context.data_sensitivity in ["high", "critical"]:
            mitigations.append("Implement additional data protection measures")
        
        return mitigations
    
    def _suggest_alternatives(self, assessment: SafetyAssessment, context: ActionContext) -> List[str]:
        """Suggest alternative safer actions"""
        alternatives = []
        
        # Common safe alternatives
        action_alternatives = {
            "file_deletion": ["Move to quarantine folder", "Create backup before deletion"],
            "system_modification": ["Test in sandbox environment", "Create system snapshot"],
            "privilege_escalation": ["Use service account", "Request specific permissions"],
            "data_access": ["Use read-only access", "Access through approved API"]
        }
        
        for category, alts in action_alternatives.items():
            if category in context.action_type.lower():
                alternatives.extend(alts)
        
        return alternatives
    
    def _calculate_confidence(self, assessment: SafetyAssessment, context: ActionContext) -> float:
        """Calculate confidence in the safety assessment"""
        confidence = 1.0
        
        # Reduce confidence for unusual actions
        if assessment.anomaly_score > 0.8:
            confidence -= 0.2
        
        # Reduce confidence for complex payloads
        if context.payload and len(context.payload) > 1000:
            confidence -= 0.1
        
        # Reduce confidence for new action types
        if context.action_type not in [a.action_context.action_type for a in self.assessments[-100:]]:
            confidence -= 0.1
        
        return max(0.5, confidence)  # Minimum 50% confidence
    
    def train_on_episode_data(self, episode_data: List[ActionContext]):
        """Train behavioral models on episode data"""
        if self.learning_enabled and episode_data:
            self.behavioral_analyzer.train_on_historical_data(episode_data)
            self.logger.info(f"Updated behavioral models with {len(episode_data)} actions")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get safety critic statistics"""
        recent_assessments = [
            a for a in self.assessments 
            if datetime.fromisoformat(a.timestamp.replace('Z', '+00:00')) > datetime.utcnow() - timedelta(hours=1)
        ]
        
        return {
            "total_rules": len(self.rules),
            "active_rules": len([r for r in self.rules.values() if r.enabled]),
            "strict_mode": self.strict_mode,
            "learning_enabled": self.learning_enabled,
            "behavioral_model_trained": self.behavioral_analyzer.is_trained,
            "statistics": self.stats.copy(),
            "recent_assessments": len(recent_assessments),
            "recent_blocks": len([a for a in recent_assessments if not a.allowed]),
            "average_evaluation_time_ms": np.mean([a.evaluation_time_ms for a in self.assessments[-100:]]) if self.assessments else 0
        }
    
    def get_recent_assessments(self, limit: int = 100) -> List[SafetyAssessment]:
        """Get recent safety assessments"""
        return sorted(self.assessments, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def export_safety_report(self, file_path: str):
        """Export comprehensive safety report"""
        report = {
            "generated_timestamp": datetime.utcnow().isoformat(),
            "statistics": self.get_statistics(),
            "rules": {rule_id: asdict(rule) for rule_id, rule in self.rules.items()},
            "recent_assessments": [asdict(a) for a in self.get_recent_assessments(50)]
        }
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

if __name__ == "__main__":
    # Example usage and testing
    print("Testing Safety Critic...")
    
    # Initialize safety critic
    critic = SafetyCritic()
    
    # Create test action contexts
    safe_context = ActionContext(
        action_id="test_001",
        timestamp=datetime.utcnow().isoformat(),
        episode_id="ep_001",
        agent_id="red_agent_001",
        agent_type="red_agent",
        action_category=ActionCategory.RECONNAISSANCE,
        action_type="port_scan",
        target="192.168.1.100",
        payload="nmap -sS 192.168.1.100"
    )
    
    dangerous_context = ActionContext(
        action_id="test_002",
        timestamp=datetime.utcnow().isoformat(),
        episode_id="ep_001",
        agent_id="red_agent_001",
        agent_type="red_agent",
        action_category=ActionCategory.IMPACT,
        action_type="file_deletion",
        target="/etc/passwd",
        payload="rm -rf /etc/passwd",
        current_permissions=["admin"]
    )
    
    # Test evaluations
    print("\nTesting safe action...")
    safe_assessment = critic.evaluate_action(safe_context)
    print(f"Safe action - Allowed: {safe_assessment.allowed}, Level: {safe_assessment.safety_level.value}")
    print(f"Reasoning: {safe_assessment.reasoning}")
    
    print("\nTesting dangerous action...")
    dangerous_assessment = critic.evaluate_action(dangerous_context)
    print(f"Dangerous action - Allowed: {dangerous_assessment.allowed}, Level: {dangerous_assessment.safety_level.value}")
    print(f"Reasoning: {dangerous_assessment.reasoning}")
    print(f"Violations: {dangerous_assessment.rule_violations}")
    
    # Get statistics
    stats = critic.get_statistics()
    print(f"\nSafety Critic Statistics:")
    print(f"- Total rules: {stats['total_rules']}")
    print(f"- Total evaluations: {stats['statistics']['total_evaluations']}")
    print(f"- Actions blocked: {stats['statistics']['actions_blocked']}")
    print(f"- Actions allowed: {stats['statistics']['actions_allowed']}")
    
    print("Safety Critic test completed!")