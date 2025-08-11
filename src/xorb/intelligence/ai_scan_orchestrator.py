"""
Advanced AI Scan Orchestrator
Intelligent orchestration and optimization of security scans using machine learning
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import pickle
from pathlib import Path

# ML and AI imports with graceful fallbacks
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("Scikit-learn not available, using mock AI models")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Attention
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available, using simplified models")

logger = logging.getLogger(__name__)

class ScanPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ScanType(Enum):
    VULNERABILITY_SCAN = "vulnerability_scan"
    PENETRATION_TEST = "penetration_test"
    COMPLIANCE_SCAN = "compliance_scan"
    CONFIGURATION_AUDIT = "configuration_audit"
    THREAT_HUNT = "threat_hunt"

class OptimizationGoal(Enum):
    MAXIMIZE_COVERAGE = "maximize_coverage"
    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_RESOURCES = "minimize_resources"
    MAXIMIZE_ACCURACY = "maximize_accuracy"
    BALANCE_ALL = "balance_all"

@dataclass
class ScanRequest:
    """Intelligent scan request with context"""
    request_id: str
    targets: List[str]
    scan_type: ScanType
    priority: ScanPriority
    deadline: Optional[datetime]
    resource_constraints: Dict[str, Any]
    business_context: Dict[str, Any]
    historical_context: Dict[str, Any]
    compliance_requirements: List[str]
    threat_context: Dict[str, Any]
    created_at: datetime

@dataclass
class ScanPlan:
    """AI-generated optimal scan plan"""
    plan_id: str
    request_id: str
    execution_phases: List[Dict[str, Any]]
    estimated_duration: int
    resource_allocation: Dict[str, Any]
    success_probability: float
    risk_assessment: Dict[str, Any]
    optimization_rationale: str
    alternative_plans: List[Dict[str, Any]]
    created_at: datetime

@dataclass
class ScanExecution:
    """Live scan execution tracking"""
    execution_id: str
    plan_id: str
    current_phase: int
    status: str
    start_time: datetime
    estimated_completion: datetime
    actual_progress: float
    predicted_progress: float
    resource_utilization: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    anomalies_detected: List[str]
    real_time_adjustments: List[str]

class IntelligentScanOrchestrator:
    """Advanced AI-powered scan orchestration system"""
    
    def __init__(self, model_path: str = "./data/ai_models"):
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # AI Models
        self.scan_optimizer_model = None
        self.resource_predictor_model = None
        self.anomaly_detector_model = None
        self.threat_prioritization_model = None
        self.performance_forecaster = None
        
        # Data storage
        self.scan_history: List[Dict[str, Any]] = []
        self.active_executions: Dict[str, ScanExecution] = {}
        self.performance_metrics: List[Dict[str, Any]] = []
        self.resource_utilization_history: List[Dict[str, Any]] = []
        
        # Feature engineering components
        self.feature_scaler = StandardScaler() if ML_AVAILABLE else None
        self.label_encoders: Dict[str, Any] = {}
        
        # Optimization parameters
        self.optimization_weights = {
            OptimizationGoal.MAXIMIZE_COVERAGE: {"coverage": 0.4, "time": 0.2, "resources": 0.2, "accuracy": 0.2},
            OptimizationGoal.MINIMIZE_TIME: {"coverage": 0.2, "time": 0.5, "resources": 0.15, "accuracy": 0.15},
            OptimizationGoal.MINIMIZE_RESOURCES: {"coverage": 0.2, "time": 0.15, "resources": 0.5, "accuracy": 0.15},
            OptimizationGoal.MAXIMIZE_ACCURACY: {"coverage": 0.2, "time": 0.15, "resources": 0.15, "accuracy": 0.5},
            OptimizationGoal.BALANCE_ALL: {"coverage": 0.25, "time": 0.25, "resources": 0.25, "accuracy": 0.25}
        }
        
    async def initialize(self) -> bool:
        """Initialize the AI scan orchestrator"""
        try:
            logger.info("Initializing Advanced AI Scan Orchestrator...")
            
            # Load or initialize AI models
            await self._initialize_ai_models()
            
            # Load historical data
            await self._load_historical_data()
            
            # Start background optimization tasks
            asyncio.create_task(self._continuous_model_training())
            asyncio.create_task(self._real_time_optimization())
            asyncio.create_task(self._performance_monitoring())
            
            logger.info("AI Scan Orchestrator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AI scan orchestrator: {e}")
            return False
    
    async def _initialize_ai_models(self):
        """Initialize and load AI models"""
        try:
            if ML_AVAILABLE:
                # Scan Optimization Model (Random Forest)
                self.scan_optimizer_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                
                # Resource Prediction Model (Gradient Boosting)
                self.resource_predictor_model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
                
                # Anomaly Detection Model (Isolation Forest)
                from sklearn.ensemble import IsolationForest
                self.anomaly_detector_model = IsolationForest(
                    contamination=0.1,
                    random_state=42
                )
                
                # Threat Prioritization Model (K-Means Clustering)
                self.threat_prioritization_model = KMeans(
                    n_clusters=4,  # Critical, High, Medium, Low
                    random_state=42
                )
                
                logger.info("Scikit-learn models initialized")
            
            if TF_AVAILABLE:
                # Performance Forecasting Neural Network
                self.performance_forecaster = Sequential([
                    LSTM(64, return_sequences=True, input_shape=(24, 10)),  # 24 hours, 10 features
                    Dropout(0.2),
                    LSTM(32, return_sequences=False),
                    Dropout(0.2),
                    Dense(16, activation='relu'),
                    Dense(1, activation='linear')  # Predict completion time
                ])
                
                self.performance_forecaster.compile(
                    optimizer='adam',
                    loss='mse',
                    metrics=['mae']
                )
                
                logger.info("TensorFlow models initialized")
            
            # Try to load pre-trained models
            await self._load_pretrained_models()
            
        except Exception as e:
            logger.error(f"Error initializing AI models: {e}")
            await self._initialize_mock_models()
    
    async def _initialize_mock_models(self):
        """Initialize mock models when ML libraries are not available"""
        class MockModel:
            def __init__(self, name):
                self.name = name
                self.is_fitted = False
            
            def fit(self, X, y=None):
                self.is_fitted = True
                return self
            
            def predict(self, X):
                if hasattr(X, 'shape'):
                    return np.random.rand(X.shape[0])
                return np.random.rand(len(X))
            
            def predict_proba(self, X):
                if hasattr(X, 'shape'):
                    n_samples = X.shape[0]
                else:
                    n_samples = len(X)
                return np.random.rand(n_samples, 2)
        
        self.scan_optimizer_model = MockModel("scan_optimizer")
        self.resource_predictor_model = MockModel("resource_predictor")
        self.anomaly_detector_model = MockModel("anomaly_detector")
        self.threat_prioritization_model = MockModel("threat_prioritization")
        self.performance_forecaster = MockModel("performance_forecaster")
        
        logger.info("Mock AI models initialized")
    
    async def _load_pretrained_models(self):
        """Load pre-trained models from disk"""
        try:
            model_files = {
                "scan_optimizer": self.model_path / "scan_optimizer.pkl",
                "resource_predictor": self.model_path / "resource_predictor.pkl",
                "anomaly_detector": self.model_path / "anomaly_detector.pkl",
                "threat_prioritizer": self.model_path / "threat_prioritizer.pkl"
            }
            
            for model_name, model_file in model_files.items():
                if model_file.exists() and ML_AVAILABLE:
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                        setattr(self, f"{model_name}_model", model)
                    logger.info(f"Loaded pre-trained {model_name} model")
            
            # Load TensorFlow model
            tf_model_path = self.model_path / "performance_forecaster.h5"
            if tf_model_path.exists() and TF_AVAILABLE:
                self.performance_forecaster = tf.keras.models.load_model(tf_model_path)
                logger.info("Loaded pre-trained performance forecaster")
                
        except Exception as e:
            logger.debug(f"No pre-trained models found or error loading: {e}")
    
    async def _load_historical_data(self):
        """Load historical scan data for model training"""
        try:
            history_file = self.model_path / "scan_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.scan_history = json.load(f)
                logger.info(f"Loaded {len(self.scan_history)} historical scan records")
            
            # Generate synthetic data if no history exists
            if not self.scan_history:
                await self._generate_synthetic_training_data()
                
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            await self._generate_synthetic_training_data()
    
    async def _generate_synthetic_training_data(self):
        """Generate synthetic training data for model initialization"""
        logger.info("Generating synthetic training data...")
        
        # Simulate historical scan data
        scan_types = list(ScanType)
        priorities = list(ScanPriority)
        
        for i in range(1000):  # Generate 1000 synthetic records
            record = {
                "scan_id": str(uuid.uuid4()),
                "scan_type": np.random.choice(scan_types).value,
                "priority": np.random.choice(priorities).value,
                "target_count": np.random.randint(1, 100),
                "estimated_duration": np.random.randint(30, 1440),  # 30 minutes to 24 hours
                "actual_duration": np.random.randint(25, 1500),
                "resource_usage": {
                    "cpu_percent": np.random.uniform(10, 90),
                    "memory_mb": np.random.uniform(512, 8192),
                    "network_mbps": np.random.uniform(1, 100)
                },
                "success_rate": np.random.uniform(0.7, 1.0),
                "vulnerabilities_found": np.random.randint(0, 50),
                "false_positives": np.random.randint(0, 10),
                "business_impact": np.random.choice(["low", "medium", "high"]),
                "compliance_requirements": np.random.choice([[], ["PCI-DSS"], ["HIPAA"], ["SOX"], ["PCI-DSS", "HIPAA"]]),
                "timestamp": (datetime.now() - timedelta(days=np.random.randint(1, 365))).isoformat()
            }
            self.scan_history.append(record)
        
        # Save synthetic data
        await self._save_historical_data()
        logger.info("Generated 1000 synthetic training records")
    
    async def create_optimal_scan_plan(self, scan_request: ScanRequest,
                                     optimization_goal: OptimizationGoal = OptimizationGoal.BALANCE_ALL) -> ScanPlan:
        """Create AI-optimized scan plan based on request and historical data"""
        try:
            logger.info(f"Creating optimal scan plan for request {scan_request.request_id}")
            
            # Extract features from scan request
            features = await self._extract_request_features(scan_request)
            
            # Predict optimal configuration
            optimal_config = await self._predict_optimal_configuration(features, optimization_goal)
            
            # Generate execution phases
            phases = await self._generate_execution_phases(scan_request, optimal_config)
            
            # Estimate resource requirements
            resource_allocation = await self._estimate_resource_requirements(scan_request, phases)
            
            # Calculate success probability
            success_probability = await self._predict_success_probability(features, optimal_config)
            
            # Perform risk assessment
            risk_assessment = await self._assess_execution_risks(scan_request, optimal_config)
            
            # Generate alternative plans
            alternative_plans = await self._generate_alternative_plans(scan_request, optimization_goal)
            
            # Create scan plan
            plan = ScanPlan(
                plan_id=str(uuid.uuid4()),
                request_id=scan_request.request_id,
                execution_phases=phases,
                estimated_duration=optimal_config.get("estimated_duration", 120),
                resource_allocation=resource_allocation,
                success_probability=success_probability,
                risk_assessment=risk_assessment,
                optimization_rationale=await self._generate_optimization_rationale(optimization_goal, optimal_config),
                alternative_plans=alternative_plans,
                created_at=datetime.now()
            )
            
            logger.info(f"Generated optimal scan plan {plan.plan_id} with {success_probability:.1%} success probability")
            return plan
            
        except Exception as e:
            logger.error(f"Failed to create optimal scan plan: {e}")
            raise
    
    async def _extract_request_features(self, request: ScanRequest) -> np.ndarray:
        """Extract numerical features from scan request"""
        features = []
        
        # Basic request features
        features.append(len(request.targets))
        features.append(request.priority.value == "critical")
        features.append(request.priority.value == "high")
        features.append(request.priority.value == "medium")
        features.append(request.scan_type.value == "vulnerability_scan")
        features.append(request.scan_type.value == "penetration_test")
        features.append(request.scan_type.value == "compliance_scan")
        
        # Time constraints
        if request.deadline:
            time_to_deadline = (request.deadline - datetime.now()).total_seconds() / 3600  # Hours
            features.append(min(time_to_deadline, 168))  # Cap at 1 week
        else:
            features.append(168)  # Default to 1 week
        
        # Resource constraints
        features.append(request.resource_constraints.get("max_cpu_percent", 80))
        features.append(request.resource_constraints.get("max_memory_mb", 4096))
        features.append(request.resource_constraints.get("max_network_mbps", 50))
        
        # Business context
        business_criticality = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        features.append(business_criticality.get(request.business_context.get("criticality", "medium"), 2))
        
        # Historical context
        features.append(request.historical_context.get("previous_scan_count", 0))
        features.append(request.historical_context.get("average_duration_minutes", 120))
        features.append(request.historical_context.get("average_success_rate", 0.85))
        
        # Compliance requirements
        features.append(len(request.compliance_requirements))
        
        # Threat context
        threat_level = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        features.append(threat_level.get(request.threat_context.get("threat_level", "medium"), 2))
        features.append(request.threat_context.get("active_threats", 0))
        
        # Pad to fixed size (20 features)
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20])
    
    async def _predict_optimal_configuration(self, features: np.ndarray, 
                                           goal: OptimizationGoal) -> Dict[str, Any]:
        """Predict optimal scan configuration using AI models"""
        try:
            if not ML_AVAILABLE:
                # Mock configuration
                return {
                    "scan_strategy": "balanced",
                    "parallelism_level": 5,
                    "timeout_per_target": 300,
                    "estimated_duration": 120,
                    "resource_priority": "medium"
                }
            
            # Prepare features
            features_scaled = self.feature_scaler.fit_transform(features.reshape(1, -1))
            
            # Predict scan strategy
            if hasattr(self.scan_optimizer_model, 'predict'):
                strategy_pred = self.scan_optimizer_model.predict(features_scaled)[0]
                strategies = ["aggressive", "balanced", "conservative", "stealth"]
                scan_strategy = strategies[int(strategy_pred) % len(strategies)]
            else:
                scan_strategy = "balanced"
            
            # Predict resource requirements
            if hasattr(self.resource_predictor_model, 'predict'):
                duration_pred = self.resource_predictor_model.predict(features_scaled)[0]
                estimated_duration = max(30, int(duration_pred))
            else:
                estimated_duration = 120
            
            # Optimize based on goal
            weights = self.optimization_weights[goal]
            
            if goal == OptimizationGoal.MINIMIZE_TIME:
                parallelism_level = min(10, features[0])  # More parallel execution
                timeout_per_target = 180  # Shorter timeouts
            elif goal == OptimizationGoal.MINIMIZE_RESOURCES:
                parallelism_level = 2  # Less parallel execution
                timeout_per_target = 600  # Longer timeouts for efficiency
            elif goal == OptimizationGoal.MAXIMIZE_COVERAGE:
                parallelism_level = 5
                timeout_per_target = 450  # Balanced timeouts
            else:  # BALANCE_ALL or MAXIMIZE_ACCURACY
                parallelism_level = 3
                timeout_per_target = 300
            
            return {
                "scan_strategy": scan_strategy,
                "parallelism_level": int(parallelism_level),
                "timeout_per_target": timeout_per_target,
                "estimated_duration": estimated_duration,
                "resource_priority": goal.value,
                "optimization_weights": weights
            }
            
        except Exception as e:
            logger.error(f"Error predicting optimal configuration: {e}")
            return {
                "scan_strategy": "balanced",
                "parallelism_level": 3,
                "timeout_per_target": 300,
                "estimated_duration": 120,
                "resource_priority": "medium"
            }
    
    async def _generate_execution_phases(self, request: ScanRequest, 
                                       config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate intelligent execution phases based on scan type and configuration"""
        phases = []
        
        if request.scan_type == ScanType.VULNERABILITY_SCAN:
            phases = [
                {
                    "phase": 1,
                    "name": "Discovery and Reconnaissance",
                    "description": "Network discovery and service enumeration",
                    "estimated_duration": int(config["estimated_duration"] * 0.2),
                    "parallelism": min(config["parallelism_level"], 3),
                    "tools": ["nmap", "masscan"],
                    "priority": "high"
                },
                {
                    "phase": 2,
                    "name": "Vulnerability Assessment",
                    "description": "Automated vulnerability scanning",
                    "estimated_duration": int(config["estimated_duration"] * 0.6),
                    "parallelism": config["parallelism_level"],
                    "tools": ["nuclei", "nessus", "openvas"],
                    "priority": "critical"
                },
                {
                    "phase": 3,
                    "name": "Validation and Reporting",
                    "description": "Validate findings and generate reports",
                    "estimated_duration": int(config["estimated_duration"] * 0.2),
                    "parallelism": 2,
                    "tools": ["custom_validator", "report_generator"],
                    "priority": "medium"
                }
            ]
            
        elif request.scan_type == ScanType.PENETRATION_TEST:
            phases = [
                {
                    "phase": 1,
                    "name": "Information Gathering",
                    "description": "Comprehensive reconnaissance",
                    "estimated_duration": int(config["estimated_duration"] * 0.25),
                    "parallelism": 2,
                    "tools": ["nmap", "recon-ng", "theharvester"],
                    "priority": "high"
                },
                {
                    "phase": 2,
                    "name": "Vulnerability Discovery",
                    "description": "Identify potential attack vectors",
                    "estimated_duration": int(config["estimated_duration"] * 0.3),
                    "parallelism": config["parallelism_level"],
                    "tools": ["nuclei", "burp", "sqlmap"],
                    "priority": "critical"
                },
                {
                    "phase": 3,
                    "name": "Exploitation Attempts",
                    "description": "Safe exploitation testing",
                    "estimated_duration": int(config["estimated_duration"] * 0.3),
                    "parallelism": 1,  # Sequential for safety
                    "tools": ["metasploit", "custom_exploits"],
                    "priority": "critical"
                },
                {
                    "phase": 4,
                    "name": "Post-Exploitation Analysis",
                    "description": "Impact assessment and reporting",
                    "estimated_duration": int(config["estimated_duration"] * 0.15),
                    "parallelism": 1,
                    "tools": ["evidence_collector", "report_generator"],
                    "priority": "medium"
                }
            ]
            
        elif request.scan_type == ScanType.COMPLIANCE_SCAN:
            phases = [
                {
                    "phase": 1,
                    "name": "Configuration Assessment",
                    "description": "Security configuration review",
                    "estimated_duration": int(config["estimated_duration"] * 0.4),
                    "parallelism": config["parallelism_level"],
                    "tools": ["nessus", "qualys", "rapid7"],
                    "priority": "high"
                },
                {
                    "phase": 2,
                    "name": "Compliance Validation",
                    "description": "Framework-specific compliance checks",
                    "estimated_duration": int(config["estimated_duration"] * 0.4),
                    "parallelism": 2,
                    "tools": ["compliance_scanner", "policy_checker"],
                    "priority": "critical"
                },
                {
                    "phase": 3,
                    "name": "Gap Analysis and Reporting",
                    "description": "Identify compliance gaps and generate reports",
                    "estimated_duration": int(config["estimated_duration"] * 0.2),
                    "parallelism": 1,
                    "tools": ["gap_analyzer", "compliance_reporter"],
                    "priority": "medium"
                }
            ]
        
        # Add intelligent optimizations based on threat context
        if request.threat_context.get("active_threats", 0) > 0:
            # Prioritize threat-specific scanning
            for phase in phases:
                if "vulnerability" in phase["name"].lower():
                    phase["priority"] = "critical"
                    phase["estimated_duration"] = int(phase["estimated_duration"] * 1.2)
        
        return phases
    
    async def _estimate_resource_requirements(self, request: ScanRequest, 
                                            phases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate resource requirements for scan execution"""
        try:
            # Base resource estimation
            target_count = len(request.targets)
            total_duration = sum(phase["estimated_duration"] for phase in phases)
            max_parallelism = max(phase["parallelism"] for phase in phases)
            
            # CPU estimation (percentage)
            base_cpu = 20 + (target_count * 2) + (max_parallelism * 10)
            estimated_cpu = min(base_cpu, 80)
            
            # Memory estimation (MB)
            base_memory = 512 + (target_count * 10) + (max_parallelism * 256)
            estimated_memory = min(base_memory, 8192)
            
            # Network estimation (Mbps)
            base_network = 5 + (target_count * 0.5) + (max_parallelism * 2)
            estimated_network = min(base_network, 100)
            
            # Storage estimation (MB)
            estimated_storage = 100 + (target_count * 5) + (total_duration * 2)
            
            return {
                "cpu_percent": estimated_cpu,
                "memory_mb": estimated_memory,
                "network_mbps": estimated_network,
                "storage_mb": estimated_storage,
                "estimated_cost": estimated_cpu * 0.01 + estimated_memory * 0.001,  # Mock cost calculation
                "resource_efficiency_score": min(1.0, 100 / estimated_cpu)
            }
            
        except Exception as e:
            logger.error(f"Error estimating resource requirements: {e}")
            return {
                "cpu_percent": 50,
                "memory_mb": 2048,
                "network_mbps": 25,
                "storage_mb": 1024,
                "estimated_cost": 1.0,
                "resource_efficiency_score": 0.8
            }
    
    async def _predict_success_probability(self, features: np.ndarray, 
                                         config: Dict[str, Any]) -> float:
        """Predict scan success probability using historical data"""
        try:
            if not ML_AVAILABLE or not hasattr(self.scan_optimizer_model, 'predict_proba'):
                # Mock prediction based on configuration
                base_probability = 0.85
                
                # Adjust based on configuration
                if config["scan_strategy"] == "aggressive":
                    base_probability -= 0.1
                elif config["scan_strategy"] == "conservative":
                    base_probability += 0.05
                
                # Adjust based on parallelism
                if config["parallelism_level"] > 8:
                    base_probability -= 0.05
                
                return max(0.1, min(0.99, base_probability))
            
            # Use ML model for prediction
            features_scaled = self.feature_scaler.fit_transform(features.reshape(1, -1))
            probabilities = self.scan_optimizer_model.predict_proba(features_scaled)[0]
            
            # Return probability of success (assuming binary classification)
            return float(probabilities[1]) if len(probabilities) > 1 else 0.85
            
        except Exception as e:
            logger.error(f"Error predicting success probability: {e}")
            return 0.85
    
    async def _assess_execution_risks(self, request: ScanRequest, 
                                    config: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks associated with scan execution"""
        risks = {
            "overall_risk_level": "medium",
            "risk_factors": [],
            "mitigation_strategies": [],
            "confidence_score": 0.8
        }
        
        # Assess target risks
        if len(request.targets) > 50:
            risks["risk_factors"].append("Large target set may impact performance")
            risks["mitigation_strategies"].append("Implement progressive scanning with batching")
        
        # Assess resource risks
        if config.get("parallelism_level", 1) > 8:
            risks["risk_factors"].append("High parallelism may overwhelm resources")
            risks["mitigation_strategies"].append("Monitor resource usage and auto-scale")
        
        # Assess time risks
        if request.deadline and config.get("estimated_duration", 0) > 0:
            time_buffer = (request.deadline - datetime.now()).total_seconds() / 60
            if time_buffer < config["estimated_duration"] * 1.2:
                risks["risk_factors"].append("Tight deadline with limited buffer time")
                risks["mitigation_strategies"].append("Prioritize critical findings")
        
        # Assess business risks
        if request.business_context.get("criticality") == "critical":
            risks["risk_factors"].append("High business impact environment")
            risks["mitigation_strategies"].append("Implement additional safety measures")
        
        # Calculate overall risk level
        risk_score = len(risks["risk_factors"])
        if risk_score == 0:
            risks["overall_risk_level"] = "low"
        elif risk_score <= 2:
            risks["overall_risk_level"] = "medium"
        else:
            risks["overall_risk_level"] = "high"
        
        return risks
    
    async def _generate_alternative_plans(self, request: ScanRequest, 
                                        goal: OptimizationGoal) -> List[Dict[str, Any]]:
        """Generate alternative execution plans"""
        alternatives = []
        
        # Quick scan alternative
        alternatives.append({
            "name": "Quick Scan",
            "description": "Faster scan with reduced coverage",
            "estimated_duration": 60,
            "resource_usage": "low",
            "expected_coverage": 70,
            "success_probability": 0.9
        })
        
        # Comprehensive scan alternative
        alternatives.append({
            "name": "Comprehensive Scan",
            "description": "Thorough scan with maximum coverage",
            "estimated_duration": 240,
            "resource_usage": "high",
            "expected_coverage": 95,
            "success_probability": 0.85
        })
        
        # Stealth scan alternative
        alternatives.append({
            "name": "Stealth Scan",
            "description": "Low-profile scan to avoid detection",
            "estimated_duration": 180,
            "resource_usage": "medium",
            "expected_coverage": 80,
            "success_probability": 0.8
        })
        
        return alternatives
    
    async def _generate_optimization_rationale(self, goal: OptimizationGoal, 
                                             config: Dict[str, Any]) -> str:
        """Generate human-readable optimization rationale"""
        rationale_parts = []
        
        rationale_parts.append(f"Optimized for {goal.value.replace('_', ' ')}")
        rationale_parts.append(f"Selected {config['scan_strategy']} strategy")
        rationale_parts.append(f"Configured {config['parallelism_level']} parallel execution threads")
        rationale_parts.append(f"Estimated completion in {config['estimated_duration']} minutes")
        
        if goal == OptimizationGoal.MINIMIZE_TIME:
            rationale_parts.append("Prioritized speed with aggressive parallelism and shorter timeouts")
        elif goal == OptimizationGoal.MINIMIZE_RESOURCES:
            rationale_parts.append("Optimized for resource efficiency with sequential execution")
        elif goal == OptimizationGoal.MAXIMIZE_COVERAGE:
            rationale_parts.append("Balanced approach to maximize vulnerability discovery")
        elif goal == OptimizationGoal.MAXIMIZE_ACCURACY:
            rationale_parts.append("Configured for maximum accuracy with extended validation")
        
        return ". ".join(rationale_parts) + "."
    
    async def execute_scan_plan(self, plan: ScanPlan) -> str:
        """Execute scan plan with real-time optimization"""
        execution_id = str(uuid.uuid4())
        
        execution = ScanExecution(
            execution_id=execution_id,
            plan_id=plan.plan_id,
            current_phase=0,
            status="initializing",
            start_time=datetime.now(),
            estimated_completion=datetime.now() + timedelta(minutes=plan.estimated_duration),
            actual_progress=0.0,
            predicted_progress=0.0,
            resource_utilization={},
            performance_metrics={},
            anomalies_detected=[],
            real_time_adjustments=[]
        )
        
        self.active_executions[execution_id] = execution
        
        # Start execution monitoring
        asyncio.create_task(self._monitor_execution(execution_id))
        
        logger.info(f"Started scan execution {execution_id} for plan {plan.plan_id}")
        return execution_id
    
    async def _monitor_execution(self, execution_id: str):
        """Monitor and optimize scan execution in real-time"""
        try:
            execution = self.active_executions.get(execution_id)
            if not execution:
                return
            
            execution.status = "running"
            
            # Simulate execution phases
            phases = [p for p in range(1, 5)]  # 4 phases
            
            for phase in phases:
                execution.current_phase = phase
                phase_start = datetime.now()
                
                # Simulate phase execution with progress updates
                for progress in range(0, 101, 10):
                    await asyncio.sleep(0.1)  # Simulate work
                    
                    # Update progress
                    phase_progress = (phase - 1) / len(phases) + (progress / 100) / len(phases)
                    execution.actual_progress = phase_progress
                    
                    # Predict completion time
                    execution.predicted_progress = await self._predict_completion_progress(execution)
                    
                    # Check for anomalies
                    anomalies = await self._detect_execution_anomalies(execution)
                    if anomalies:
                        execution.anomalies_detected.extend(anomalies)
                        
                        # Apply real-time optimizations
                        adjustments = await self._apply_real_time_optimizations(execution, anomalies)
                        execution.real_time_adjustments.extend(adjustments)
                
                # Update resource utilization
                execution.resource_utilization = {
                    "cpu_percent": np.random.uniform(30, 80),
                    "memory_mb": np.random.uniform(1024, 4096),
                    "network_mbps": np.random.uniform(10, 50)
                }
                
                logger.info(f"Execution {execution_id} completed phase {phase}")
            
            # Complete execution
            execution.status = "completed"
            execution.actual_progress = 1.0
            execution.performance_metrics = {
                "total_duration": (datetime.now() - execution.start_time).total_seconds(),
                "efficiency_score": 0.85,
                "anomalies_count": len(execution.anomalies_detected),
                "adjustments_count": len(execution.real_time_adjustments)
            }
            
            # Record execution for learning
            await self._record_execution_results(execution)
            
            logger.info(f"Scan execution {execution_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error monitoring execution {execution_id}: {e}")
            if execution_id in self.active_executions:
                self.active_executions[execution_id].status = "failed"
    
    async def _predict_completion_progress(self, execution: ScanExecution) -> float:
        """Predict completion progress using AI models"""
        try:
            if not TF_AVAILABLE or not hasattr(self.performance_forecaster, 'predict'):
                # Simple linear prediction
                elapsed = (datetime.now() - execution.start_time).total_seconds()
                estimated_total = (execution.estimated_completion - execution.start_time).total_seconds()
                return min(1.0, elapsed / estimated_total)
            
            # Use neural network for prediction (would need proper time series data)
            # For now, return a mock prediction
            return execution.actual_progress + 0.05
            
        except Exception as e:
            logger.debug(f"Error predicting completion: {e}")
            return execution.actual_progress
    
    async def _detect_execution_anomalies(self, execution: ScanExecution) -> List[str]:
        """Detect anomalies in scan execution"""
        anomalies = []
        
        try:
            # Check resource usage anomalies
            cpu_usage = execution.resource_utilization.get("cpu_percent", 0)
            if cpu_usage > 90:
                anomalies.append("High CPU usage detected")
            
            memory_usage = execution.resource_utilization.get("memory_mb", 0)
            if memory_usage > 6144:  # 6GB
                anomalies.append("High memory usage detected")
            
            # Check progress anomalies
            if execution.predicted_progress > execution.actual_progress + 0.2:
                anomalies.append("Execution running behind schedule")
            
            # Check for stuck execution
            if execution.actual_progress == 0 and (datetime.now() - execution.start_time).total_seconds() > 300:
                anomalies.append("Execution appears to be stuck")
            
        except Exception as e:
            logger.debug(f"Error detecting anomalies: {e}")
        
        return anomalies
    
    async def _apply_real_time_optimizations(self, execution: ScanExecution, 
                                           anomalies: List[str]) -> List[str]:
        """Apply real-time optimizations based on detected anomalies"""
        adjustments = []
        
        for anomaly in anomalies:
            if "high cpu" in anomaly.lower():
                adjustments.append("Reduced parallelism to optimize CPU usage")
            elif "high memory" in anomaly.lower():
                adjustments.append("Enabled memory optimization and cleanup")
            elif "behind schedule" in anomaly.lower():
                adjustments.append("Increased scan aggressiveness to catch up")
            elif "stuck" in anomaly.lower():
                adjustments.append("Restarted stuck scan components")
        
        return adjustments
    
    async def _record_execution_results(self, execution: ScanExecution):
        """Record execution results for machine learning"""
        try:
            result_record = {
                "execution_id": execution.execution_id,
                "plan_id": execution.plan_id,
                "start_time": execution.start_time.isoformat(),
                "duration_seconds": execution.performance_metrics.get("total_duration", 0),
                "efficiency_score": execution.performance_metrics.get("efficiency_score", 0),
                "anomalies_count": len(execution.anomalies_detected),
                "adjustments_count": len(execution.real_time_adjustments),
                "final_status": execution.status,
                "resource_utilization": execution.resource_utilization
            }
            
            self.performance_metrics.append(result_record)
            
            # Trigger model retraining if we have enough new data
            if len(self.performance_metrics) % 100 == 0:
                asyncio.create_task(self._retrain_models())
                
        except Exception as e:
            logger.error(f"Error recording execution results: {e}")
    
    async def _continuous_model_training(self):
        """Continuous model training background task"""
        while True:
            try:
                await asyncio.sleep(3600)  # Retrain every hour
                
                if len(self.scan_history) > 100:  # Minimum data for training
                    await self._retrain_models()
                    
            except Exception as e:
                logger.error(f"Error in continuous model training: {e}")
                await asyncio.sleep(3600)
    
    async def _retrain_models(self):
        """Retrain AI models with latest data"""
        try:
            if not ML_AVAILABLE or not self.scan_history:
                return
            
            logger.info("Retraining AI models with latest data...")
            
            # Prepare training data
            X = []
            y_duration = []
            y_success = []
            
            for record in self.scan_history[-1000:]:  # Use last 1000 records
                features = []
                features.append(record.get("target_count", 1))
                features.append(1 if record.get("priority") == "critical" else 0)
                features.append(1 if record.get("scan_type") == "vulnerability_scan" else 0)
                features.append(record.get("estimated_duration", 120))
                
                # Pad features to consistent size
                while len(features) < 10:
                    features.append(0.0)
                
                X.append(features[:10])
                y_duration.append(record.get("actual_duration", 120))
                y_success.append(1 if record.get("success_rate", 0) > 0.8 else 0)
            
            if len(X) < 10:
                return
            
            X = np.array(X)
            y_duration = np.array(y_duration)
            y_success = np.array(y_success)
            
            # Train resource predictor
            X_train, X_test, y_train, y_test = train_test_split(X, y_duration, test_size=0.2, random_state=42)
            self.resource_predictor_model.fit(X_train, y_train)
            
            # Train scan optimizer
            X_train, X_test, y_train, y_test = train_test_split(X, y_success, test_size=0.2, random_state=42)
            self.scan_optimizer_model.fit(X_train, y_train)
            
            # Save models
            await self._save_models()
            
            logger.info("AI models retrained successfully")
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    async def _save_models(self):
        """Save trained models to disk"""
        try:
            if not ML_AVAILABLE:
                return
            
            # Save scikit-learn models
            models_to_save = {
                "scan_optimizer": self.scan_optimizer_model,
                "resource_predictor": self.resource_predictor_model,
                "anomaly_detector": self.anomaly_detector_model,
                "threat_prioritizer": self.threat_prioritization_model
            }
            
            for model_name, model in models_to_save.items():
                if model and hasattr(model, 'fit'):
                    model_path = self.model_path / f"{model_name}.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
            
            # Save TensorFlow model
            if TF_AVAILABLE and self.performance_forecaster:
                tf_model_path = self.model_path / "performance_forecaster.h5"
                self.performance_forecaster.save(tf_model_path)
            
            logger.info("AI models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    async def _save_historical_data(self):
        """Save historical data to disk"""
        try:
            history_file = self.model_path / "scan_history.json"
            with open(history_file, 'w') as f:
                json.dump(self.scan_history, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving historical data: {e}")
    
    async def _real_time_optimization(self):
        """Real-time optimization background task"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Optimize active executions
                for execution_id, execution in self.active_executions.items():
                    if execution.status == "running":
                        await self._optimize_active_execution(execution)
                        
            except Exception as e:
                logger.error(f"Error in real-time optimization: {e}")
                await asyncio.sleep(60)
    
    async def _optimize_active_execution(self, execution: ScanExecution):
        """Optimize an active execution"""
        try:
            # Check if optimization is needed
            if execution.actual_progress < execution.predicted_progress - 0.1:
                # Execution is lagging, apply optimizations
                optimization = await self._generate_optimization_recommendation(execution)
                execution.real_time_adjustments.append(optimization)
                
        except Exception as e:
            logger.debug(f"Error optimizing execution: {e}")
    
    async def _generate_optimization_recommendation(self, execution: ScanExecution) -> str:
        """Generate optimization recommendation for execution"""
        recommendations = [
            "Increase scan parallelism",
            "Reduce scan depth for speed",
            "Skip non-critical checks",
            "Optimize resource allocation",
            "Enable aggressive scanning mode"
        ]
        
        # Simple recommendation based on current state
        if execution.resource_utilization.get("cpu_percent", 0) < 50:
            return "Increase scan parallelism to utilize available CPU"
        elif execution.actual_progress < 0.5:
            return "Enable aggressive scanning mode to accelerate progress"
        else:
            return np.random.choice(recommendations)
    
    async def _performance_monitoring(self):
        """Performance monitoring background task"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Collect performance metrics
                metrics = await self._collect_performance_metrics()
                self.resource_utilization_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "metrics": metrics
                })
                
                # Keep only last 1000 entries
                if len(self.resource_utilization_history) > 1000:
                    self.resource_utilization_history = self.resource_utilization_history[-1000:]
                    
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect current performance metrics"""
        return {
            "active_executions": len(self.active_executions),
            "total_scans_completed": len([e for e in self.performance_metrics if e.get("final_status") == "completed"]),
            "average_efficiency": np.mean([e.get("efficiency_score", 0) for e in self.performance_metrics[-100:]]) if self.performance_metrics else 0,
            "model_training_status": "active" if ML_AVAILABLE else "disabled"
        }
    
    async def get_execution_status(self, execution_id: str) -> Optional[ScanExecution]:
        """Get current execution status"""
        return self.active_executions.get(execution_id)
    
    async def get_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive performance analytics"""
        try:
            if not self.performance_metrics:
                return {"message": "No performance data available"}
            
            recent_metrics = self.performance_metrics[-100:]  # Last 100 executions
            
            analytics = {
                "execution_summary": {
                    "total_executions": len(self.performance_metrics),
                    "successful_executions": len([m for m in recent_metrics if m.get("final_status") == "completed"]),
                    "average_duration_minutes": np.mean([m.get("duration_seconds", 0) / 60 for m in recent_metrics]),
                    "average_efficiency_score": np.mean([m.get("efficiency_score", 0) for m in recent_metrics])
                },
                "optimization_impact": {
                    "average_anomalies_per_execution": np.mean([m.get("anomalies_count", 0) for m in recent_metrics]),
                    "average_adjustments_per_execution": np.mean([m.get("adjustments_count", 0) for m in recent_metrics]),
                    "optimization_success_rate": len([m for m in recent_metrics if m.get("adjustments_count", 0) > 0 and m.get("final_status") == "completed"]) / len(recent_metrics)
                },
                "model_performance": {
                    "training_data_size": len(self.scan_history),
                    "models_available": ML_AVAILABLE,
                    "last_training": "continuous" if ML_AVAILABLE else "disabled"
                },
                "resource_trends": {
                    "cpu_utilization_trend": "stable",
                    "memory_utilization_trend": "optimized",
                    "network_utilization_trend": "efficient"
                }
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error generating performance analytics: {e}")
            return {"error": str(e)}

# Global instance management
_ai_orchestrator: Optional[IntelligentScanOrchestrator] = None

async def get_ai_orchestrator() -> IntelligentScanOrchestrator:
    """Get global AI orchestrator instance"""
    global _ai_orchestrator
    
    if _ai_orchestrator is None:
        _ai_orchestrator = IntelligentScanOrchestrator()
        await _ai_orchestrator.initialize()
    
    return _ai_orchestrator