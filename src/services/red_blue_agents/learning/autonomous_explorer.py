"""
Autonomous Learning and Exploration Engine

Self-learning system that adapts agent behavior based on success patterns,
environmental feedback, and continuous exploration of new tactics and techniques.
"""

import asyncio
import logging
import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid

import redis.asyncio as redis
import asyncpg
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)


class ExplorationStrategy(Enum):
    """Exploration strategies for autonomous learning"""
    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "upper_confidence_bound"
    THOMPSON_SAMPLING = "thompson_sampling"
    ADAPTIVE = "adaptive"


class LearningMode(Enum):
    """Learning modes for different scenarios"""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"


@dataclass
class TechniquePerformance:
    """Performance metrics for a technique"""
    technique_id: str
    success_count: int = 0
    failure_count: int = 0
    total_executions: int = 0
    avg_execution_time: float = 0.0
    success_rate: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()
            
    def update_metrics(self):
        """Update calculated metrics"""
        self.total_executions = self.success_count + self.failure_count
        if self.total_executions > 0:
            self.success_rate = self.success_count / self.total_executions
        else:
            self.success_rate = 0.0
            
        # Calculate confidence interval (Wilson score interval)
        if self.total_executions > 0:
            z = 1.96  # 95% confidence
            p = self.success_rate
            n = self.total_executions
            
            denominator = 1 + z**2 / n
            centre = (p + z**2 / (2*n)) / denominator
            adjustment = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator
            
            self.confidence_interval = (
                max(0.0, centre - adjustment),
                min(1.0, centre + adjustment)
            )
        else:
            self.confidence_interval = (0.0, 1.0)


@dataclass
class LearningState:
    """Current learning state for the autonomous explorer"""
    model_version: str
    exploration_rate: float
    learning_mode: LearningMode
    technique_performances: Dict[str, TechniquePerformance]
    feature_importance: Dict[str, float]
    model_accuracy: float
    last_training: datetime
    exploration_budget: int
    exploitation_threshold: float
    
    def __post_init__(self):
        if not self.technique_performances:
            self.technique_performances = {}
        if not self.feature_importance:
            self.feature_importance = {}


class AutonomousExplorer:
    """
    Autonomous learning and exploration engine for red/blue team agents.
    
    Features:
    - Reinforcement learning for technique selection
    - Automated parameter optimization
    - Success prediction modeling
    - Adaptive exploration strategies
    - Continuous model improvement
    - Multi-armed bandit optimization
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None,
                 postgres_pool: Optional[asyncpg.Pool] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.redis_client = redis_client
        self.postgres_pool = postgres_pool
        self.config = config or {}
        
        # Configuration
        self.exploration_strategy = ExplorationStrategy(
            self.config.get("exploration_strategy", "epsilon_greedy")
        )
        self.initial_exploration_rate = self.config.get("initial_exploration_rate", 0.3)
        self.min_exploration_rate = self.config.get("min_exploration_rate", 0.05)
        self.exploration_decay = self.config.get("exploration_decay", 0.995)
        self.learning_rate = self.config.get("learning_rate", 0.01)
        self.model_update_interval = self.config.get("model_update_interval", 3600)  # 1 hour
        self.min_samples_for_training = self.config.get("min_samples_for_training", 100)
        
        # State
        self.learning_state = LearningState(
            model_version="1.0.0",
            exploration_rate=self.initial_exploration_rate,
            learning_mode=LearningMode.BALANCED,
            technique_performances={},
            feature_importance={},
            model_accuracy=0.0,
            last_training=datetime.utcnow(),
            exploration_budget=1000,
            exploitation_threshold=0.8
        )
        
        # ML Models
        self.success_predictor = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.execution_time_predictor = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        self.feature_scaler = StandardScaler()
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.is_running = False
        
    async def initialize(self):
        """Initialize the autonomous explorer"""
        logger.info("Initializing Autonomous Explorer...")
        
        # Load existing state if available
        await self._load_learning_state()
        
        # Initialize database schema
        if self.postgres_pool:
            await self._initialize_database_schema()
            
        # Load historical data for training
        await self._load_historical_data()
        
        # Start background learning tasks
        self.is_running = True
        
        # Model training task
        training_task = asyncio.create_task(self._continuous_learning_loop())
        self.background_tasks.append(training_task)
        
        # Exploration strategy adaptation task
        adaptation_task = asyncio.create_task(self._adapt_exploration_strategy())
        self.background_tasks.append(adaptation_task)
        
        # Performance monitoring task
        monitoring_task = asyncio.create_task(self._monitor_performance())
        self.background_tasks.append(monitoring_task)
        
        logger.info("Autonomous Explorer initialized successfully")
        
    async def _initialize_database_schema(self):
        """Initialize database schema for learning data"""
        schema_sql = """
        -- Learning state table
        CREATE TABLE IF NOT EXISTS learning_state (
            id SERIAL PRIMARY KEY,
            model_version VARCHAR(20) NOT NULL,
            exploration_rate FLOAT NOT NULL,
            learning_mode VARCHAR(20) NOT NULL,
            technique_performances JSONB DEFAULT '{}',
            feature_importance JSONB DEFAULT '{}',
            model_accuracy FLOAT DEFAULT 0.0,
            last_training TIMESTAMPTZ DEFAULT NOW(),
            exploration_budget INTEGER DEFAULT 1000,
            exploitation_threshold FLOAT DEFAULT 0.8,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Technique exploration history
        CREATE TABLE IF NOT EXISTS technique_exploration (
            id SERIAL PRIMARY KEY,
            technique_id VARCHAR(100) NOT NULL,
            environment VARCHAR(50) NOT NULL,
            target_type VARCHAR(50),
            parameters JSONB DEFAULT '{}',
            success BOOLEAN NOT NULL,
            execution_time FLOAT DEFAULT 0.0,
            error_message TEXT,
            features JSONB DEFAULT '{}',
            exploration_context JSONB DEFAULT '{}',
            timestamp TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Model performance tracking
        CREATE TABLE IF NOT EXISTS model_performance (
            id SERIAL PRIMARY KEY,
            model_type VARCHAR(50) NOT NULL,
            model_version VARCHAR(20) NOT NULL,
            accuracy FLOAT,
            precision_score FLOAT,
            recall SCORE FLOAT,
            f1_score FLOAT,
            training_samples INTEGER,
            validation_samples INTEGER,
            feature_count INTEGER,
            training_time FLOAT,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Feature importance tracking
        CREATE TABLE IF NOT EXISTS feature_importance_history (
            id SERIAL PRIMARY KEY,
            model_version VARCHAR(20) NOT NULL,
            feature_name VARCHAR(100) NOT NULL,
            importance_score FLOAT NOT NULL,
            rank INTEGER,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_technique_exploration_technique ON technique_exploration (technique_id);
        CREATE INDEX IF NOT EXISTS idx_technique_exploration_timestamp ON technique_exploration (timestamp);
        CREATE INDEX IF NOT EXISTS idx_model_performance_version ON model_performance (model_version);
        CREATE INDEX IF NOT EXISTS idx_feature_importance_version ON feature_importance_history (model_version);
        """
        
        async with self.postgres_pool.acquire() as conn:
            await conn.execute(schema_sql)
            
    async def suggest_technique(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest the best technique based on current learning state and context"""
        environment = context.get("environment", "development")
        target_type = context.get("target_type", "unknown")
        agent_type = context.get("agent_type", "unknown")
        available_techniques = context.get("available_techniques", [])
        
        if not available_techniques:
            return {"technique_id": None, "confidence": 0.0, "reason": "No techniques available"}
            
        # Extract features from context
        features = self._extract_features(context)
        
        # Get technique recommendations based on exploration strategy
        if self.exploration_strategy == ExplorationStrategy.EPSILON_GREEDY:
            recommendation = await self._epsilon_greedy_selection(available_techniques, features, context)
        elif self.exploration_strategy == ExplorationStrategy.UCB:
            recommendation = await self._ucb_selection(available_techniques, features, context)
        elif self.exploration_strategy == ExplorationStrategy.THOMPSON_SAMPLING:
            recommendation = await self._thompson_sampling_selection(available_techniques, features, context)
        else:  # ADAPTIVE
            recommendation = await self._adaptive_selection(available_techniques, features, context)
            
        # Log the suggestion for learning
        await self._log_technique_suggestion(recommendation, context)
        
        return recommendation
        
    async def _epsilon_greedy_selection(self, techniques: List[str], features: Dict[str, Any], 
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Epsilon-greedy technique selection"""
        import random
        
        # Decide between exploration and exploitation
        if random.random() < self.learning_state.exploration_rate:
            # Explore: select random technique
            technique_id = random.choice(techniques)
            return {
                "technique_id": technique_id,
                "confidence": 0.5,
                "reason": "exploration",
                "strategy": "epsilon_greedy"
            }
        else:
            # Exploit: select best known technique
            best_technique = await self._get_best_technique(techniques, features, context)
            return {
                "technique_id": best_technique["technique_id"],
                "confidence": best_technique["confidence"],
                "reason": "exploitation",
                "strategy": "epsilon_greedy"
            }
            
    async def _ucb_selection(self, techniques: List[str], features: Dict[str, Any],
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Upper Confidence Bound technique selection"""
        scores = {}
        total_attempts = sum(
            perf.total_executions 
            for perf in self.learning_state.technique_performances.values()
        )
        
        if total_attempts == 0:
            total_attempts = 1
            
        for technique_id in techniques:
            perf = self.learning_state.technique_performances.get(
                technique_id, 
                TechniquePerformance(technique_id=technique_id)
            )
            
            if perf.total_executions == 0:
                # Untried technique gets maximum score
                scores[technique_id] = float('inf')
            else:
                # UCB formula: mean + sqrt(2 * ln(total_attempts) / technique_attempts)
                confidence = np.sqrt(2 * np.log(total_attempts) / perf.total_executions)
                scores[technique_id] = perf.success_rate + confidence
                
        # Select technique with highest UCB score
        best_technique = max(scores.keys(), key=lambda k: scores[k])
        confidence = min(1.0, scores[best_technique]) if scores[best_technique] != float('inf') else 1.0
        
        return {
            "technique_id": best_technique,
            "confidence": confidence,
            "reason": "ucb_optimization",
            "strategy": "ucb",
            "ucb_score": scores[best_technique]
        }
        
    async def _thompson_sampling_selection(self, techniques: List[str], features: Dict[str, Any],
                                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Thompson Sampling technique selection"""
        samples = {}
        
        for technique_id in techniques:
            perf = self.learning_state.technique_performances.get(
                technique_id,
                TechniquePerformance(technique_id=technique_id)
            )
            
            # Beta distribution parameters (Bayesian approach)
            alpha = perf.success_count + 1  # Prior: alpha = 1
            beta = perf.failure_count + 1   # Prior: beta = 1
            
            # Sample from Beta distribution
            sample = np.random.beta(alpha, beta)
            samples[technique_id] = sample
            
        # Select technique with highest sample
        best_technique = max(samples.keys(), key=lambda k: samples[k])
        
        return {
            "technique_id": best_technique,
            "confidence": samples[best_technique],
            "reason": "thompson_sampling",
            "strategy": "thompson_sampling",
            "beta_sample": samples[best_technique]
        }
        
    async def _adaptive_selection(self, techniques: List[str], features: Dict[str, Any],
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptive technique selection combining multiple strategies"""
        # Use different strategies based on learning mode
        if self.learning_state.learning_mode == LearningMode.EXPLORATION:
            # Favor exploration
            self.learning_state.exploration_rate = min(0.5, self.learning_state.exploration_rate * 1.1)
            return await self._epsilon_greedy_selection(techniques, features, context)
        elif self.learning_state.learning_mode == LearningMode.EXPLOITATION:
            # Favor exploitation
            self.learning_state.exploration_rate = max(0.05, self.learning_state.exploration_rate * 0.9)
            return await self._get_best_technique(techniques, features, context)
        else:  # BALANCED or CONSERVATIVE
            # Use UCB for balanced approach
            return await self._ucb_selection(techniques, features, context)
            
    async def _get_best_technique(self, techniques: List[str], features: Dict[str, Any],
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Get the best technique based on learned performance"""
        if not techniques:
            return {"technique_id": None, "confidence": 0.0}
            
        # Use ML model if available and trained
        if hasattr(self.success_predictor, 'classes_'):
            try:
                predictions = await self._predict_technique_success(techniques, features)
                best_technique = max(predictions.keys(), key=lambda k: predictions[k])
                return {
                    "technique_id": best_technique,
                    "confidence": predictions[best_technique],
                    "reason": "ml_prediction",
                    "predictions": predictions
                }
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")
                
        # Fallback to historical performance
        best_score = -1
        best_technique = techniques[0]
        
        for technique_id in techniques:
            perf = self.learning_state.technique_performances.get(technique_id)
            if perf and perf.total_executions > 0:
                # Use lower bound of confidence interval for conservative estimate
                score = perf.confidence_interval[0]
                if score > best_score:
                    best_score = score
                    best_technique = technique_id
                    
        return {
            "technique_id": best_technique,
            "confidence": best_score,
            "reason": "historical_performance"
        }
        
    async def _predict_technique_success(self, techniques: List[str], 
                                       features: Dict[str, Any]) -> Dict[str, float]:
        """Predict success probability for each technique"""
        predictions = {}
        
        for technique_id in techniques:
            # Combine technique-specific features with context features
            combined_features = {**features, "technique_id": technique_id}
            feature_vector = self._features_to_vector(combined_features)
            
            try:
                # Predict success probability
                proba = self.success_predictor.predict_proba([feature_vector])[0]
                # Get probability of success (class 1)
                success_prob = proba[1] if len(proba) > 1 else proba[0]
                predictions[technique_id] = float(success_prob)
            except Exception as e:
                logger.warning(f"Prediction failed for {technique_id}: {e}")
                predictions[technique_id] = 0.5  # Default probability
                
        return predictions
        
    async def record_technique_result(self, technique_id: str, success: bool, 
                                    execution_time: float, context: Dict[str, Any]):
        """Record the result of a technique execution for learning"""
        # Update technique performance
        if technique_id not in self.learning_state.technique_performances:
            self.learning_state.technique_performances[technique_id] = TechniquePerformance(
                technique_id=technique_id
            )
            
        perf = self.learning_state.technique_performances[technique_id]
        
        if success:
            perf.success_count += 1
        else:
            perf.failure_count += 1
            
        # Update average execution time
        total_time = perf.avg_execution_time * (perf.total_executions - 1) + execution_time
        perf.avg_execution_time = total_time / perf.total_executions if perf.total_executions > 0 else execution_time
        
        perf.update_metrics()
        perf.last_updated = datetime.utcnow()
        
        # Store in database for model training
        if self.postgres_pool:
            await self._store_exploration_result(technique_id, success, execution_time, context)
            
        # Update learning state in Redis
        await self._save_learning_state()
        
        # Adjust exploration rate based on recent performance
        await self._adjust_exploration_rate(success)
        
    async def _store_exploration_result(self, technique_id: str, success: bool,
                                      execution_time: float, context: Dict[str, Any]):
        """Store exploration result in database"""
        try:
            features = self._extract_features(context)
            
            async with self.postgres_pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO technique_exploration 
                       (technique_id, environment, target_type, parameters, success, 
                        execution_time, error_message, features, exploration_context)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)""",
                    technique_id,
                    context.get("environment", "unknown"),
                    context.get("target_type", "unknown"),
                    json.dumps(context.get("parameters", {})),
                    success,
                    execution_time,
                    context.get("error_message"),
                    json.dumps(features),
                    json.dumps(context)
                )
        except Exception as e:
            logger.error(f"Failed to store exploration result: {e}")
            
    async def _adjust_exploration_rate(self, success: bool):
        """Adjust exploration rate based on recent results"""
        if success:
            # Decrease exploration rate slightly on success (favor exploitation)
            self.learning_state.exploration_rate *= self.exploration_decay
        else:
            # Increase exploration rate slightly on failure (explore more)
            self.learning_state.exploration_rate *= (2 - self.exploration_decay)
            
        # Keep within bounds
        self.learning_state.exploration_rate = max(
            self.min_exploration_rate,
            min(0.8, self.learning_state.exploration_rate)
        )
        
    def _extract_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from context for ML models"""
        features = {
            # Environment features
            "environment": context.get("environment", "unknown"),
            "target_type": context.get("target_type", "unknown"),
            "agent_type": context.get("agent_type", "unknown"),
            
            # Target features
            "target_os": context.get("target_os", "unknown"),
            "target_services": len(context.get("target_services", [])),
            "target_ports": len(context.get("target_ports", [])),
            "target_technologies": len(context.get("target_technologies", [])),
            
            # Mission features
            "mission_duration": context.get("mission_duration", 0),
            "mission_complexity": context.get("mission_complexity", "medium"),
            "concurrent_agents": context.get("concurrent_agents", 1),
            
            # Temporal features
            "hour_of_day": datetime.utcnow().hour,
            "day_of_week": datetime.utcnow().weekday(),
            "month": datetime.utcnow().month,
            
            # Historical features
            "recent_success_rate": self._get_recent_success_rate(),
            "technique_novelty": self._calculate_technique_novelty(context.get("technique_id")),
        }
        
        return features
        
    def _features_to_vector(self, features: Dict[str, Any]) -> List[float]:
        """Convert feature dictionary to numerical vector"""
        vector = []
        
        # Categorical features (one-hot encoded)
        categorical_features = ["environment", "target_type", "agent_type", "target_os", "mission_complexity"]
        categorical_values = {
            "environment": ["production", "staging", "development", "cyber_range"],
            "target_type": ["web_app", "network", "host", "cloud", "mobile"],
            "agent_type": ["red_recon", "red_exploit", "red_persistence", "red_evasion", "red_collection",
                          "blue_detect", "blue_hunt", "blue_analyze", "blue_respond"],
            "target_os": ["windows", "linux", "macos", "unknown"],
            "mission_complexity": ["low", "medium", "high"]
        }
        
        for feature in categorical_features:
            value = features.get(feature, "unknown")
            possible_values = categorical_values.get(feature, ["unknown"])
            for possible_value in possible_values:
                vector.append(1.0 if value == possible_value else 0.0)
                
        # Numerical features
        numerical_features = [
            "target_services", "target_ports", "target_technologies",
            "mission_duration", "concurrent_agents", "hour_of_day",
            "day_of_week", "month", "recent_success_rate", "technique_novelty"
        ]
        
        for feature in numerical_features:
            vector.append(float(features.get(feature, 0)))
            
        return vector
        
    def _get_recent_success_rate(self) -> float:
        """Calculate recent success rate across all techniques"""
        recent_successes = 0
        recent_total = 0
        
        for perf in self.learning_state.technique_performances.values():
            # Weight recent performance more heavily
            recent_successes += perf.success_count
            recent_total += perf.total_executions
            
        return recent_successes / recent_total if recent_total > 0 else 0.5
        
    def _calculate_technique_novelty(self, technique_id: Optional[str]) -> float:
        """Calculate novelty score for a technique (0=well-known, 1=novel)"""
        if not technique_id:
            return 0.5
            
        perf = self.learning_state.technique_performances.get(technique_id)
        if not perf or perf.total_executions == 0:
            return 1.0  # Completely novel
            
        # Novelty decreases with more executions
        max_executions = 100  # Consider technique well-known after 100 executions
        novelty = max(0.0, 1.0 - (perf.total_executions / max_executions))
        return novelty
        
    async def _continuous_learning_loop(self):
        """Continuously train and update ML models"""
        while self.is_running:
            try:
                await asyncio.sleep(self.model_update_interval)
                await self._train_models()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in continuous learning loop: {e}")
                await asyncio.sleep(60)
                
    async def _train_models(self):
        """Train ML models with latest data"""
        if not self.postgres_pool:
            return
            
        logger.info("Training ML models...")
        
        try:
            # Load training data
            training_data = await self._load_training_data()
            
            if len(training_data) < self.min_samples_for_training:
                logger.warning(f"Insufficient training data: {len(training_data)} samples")
                return
                
            # Prepare features and targets
            X, y_success, y_time = self._prepare_training_data(training_data)
            
            if len(X) == 0:
                logger.warning("No valid training samples")
                return
                
            # Split data
            X_train, X_test, y_success_train, y_success_test, y_time_train, y_time_test = train_test_split(
                X, y_success, y_time, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Train success predictor
            self.success_predictor.fit(X_train_scaled, y_success_train)
            y_pred_success = self.success_predictor.predict(X_test_scaled)
            success_accuracy = accuracy_score(y_success_test, y_pred_success)
            
            # Train execution time predictor
            self.execution_time_predictor.fit(X_train_scaled, y_time_train)
            y_pred_time = self.execution_time_predictor.predict(X_test_scaled)
            time_rmse = np.sqrt(mean_squared_error(y_time_test, y_pred_time))
            
            # Update learning state
            self.learning_state.model_accuracy = success_accuracy
            self.learning_state.last_training = datetime.utcnow()
            
            # Extract feature importance
            if hasattr(self.success_predictor, 'feature_importances_'):
                feature_names = self._get_feature_names()
                importance_dict = dict(zip(feature_names, self.success_predictor.feature_importances_))
                self.learning_state.feature_importance = importance_dict
                
            # Save models
            await self._save_models()
            
            # Store performance metrics
            await self._store_model_performance(success_accuracy, time_rmse, len(X_train), len(X_test))
            
            logger.info(f"Model training complete - Success accuracy: {success_accuracy:.3f}, Time RMSE: {time_rmse:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to train models: {e}")
            
    async def _load_training_data(self) -> List[Dict[str, Any]]:
        """Load training data from database"""
        # Load last 30 days of exploration data
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        
        query = """
        SELECT technique_id, environment, target_type, parameters, success,
               execution_time, error_message, features, exploration_context, timestamp
        FROM technique_exploration 
        WHERE timestamp >= $1
        ORDER BY timestamp DESC
        LIMIT 10000
        """
        
        training_data = []
        
        async with self.postgres_pool.acquire() as conn:
            rows = await conn.fetch(query, cutoff_date)
            
            for row in rows:
                try:
                    data = {
                        "technique_id": row["technique_id"],
                        "environment": row["environment"],
                        "target_type": row["target_type"],
                        "parameters": json.loads(row["parameters"] or "{}"),
                        "success": row["success"],
                        "execution_time": float(row["execution_time"]),
                        "error_message": row["error_message"],
                        "features": json.loads(row["features"] or "{}"),
                        "exploration_context": json.loads(row["exploration_context"] or "{}"),
                        "timestamp": row["timestamp"]
                    }
                    training_data.append(data)
                except Exception as e:
                    logger.warning(f"Failed to parse training data row: {e}")
                    
        return training_data
        
    def _prepare_training_data(self, training_data: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[int], List[float]]:
        """Prepare training data for ML models"""
        X = []
        y_success = []
        y_time = []
        
        for sample in training_data:
            try:
                # Extract features
                features = sample["features"]
                features["technique_id"] = sample["technique_id"]
                
                feature_vector = self._features_to_vector(features)
                
                X.append(feature_vector)
                y_success.append(1 if sample["success"] else 0)
                y_time.append(sample["execution_time"])
                
            except Exception as e:
                logger.warning(f"Failed to prepare training sample: {e}")
                
        return X, y_success, y_time
        
    def _get_feature_names(self) -> List[str]:
        """Get feature names for importance tracking"""
        names = []
        
        # Categorical feature names
        categorical_features = ["environment", "target_type", "agent_type", "target_os", "mission_complexity"]
        categorical_values = {
            "environment": ["production", "staging", "development", "cyber_range"],
            "target_type": ["web_app", "network", "host", "cloud", "mobile"],
            "agent_type": ["red_recon", "red_exploit", "red_persistence", "red_evasion", "red_collection",
                          "blue_detect", "blue_hunt", "blue_analyze", "blue_respond"],
            "target_os": ["windows", "linux", "macos", "unknown"],
            "mission_complexity": ["low", "medium", "high"]
        }
        
        for feature in categorical_features:
            possible_values = categorical_values.get(feature, ["unknown"])
            for value in possible_values:
                names.append(f"{feature}_{value}")
                
        # Numerical feature names
        numerical_features = [
            "target_services", "target_ports", "target_technologies",
            "mission_duration", "concurrent_agents", "hour_of_day",
            "day_of_week", "month", "recent_success_rate", "technique_novelty"
        ]
        
        names.extend(numerical_features)
        return names
        
    async def _save_models(self):
        """Save trained models to storage"""
        try:
            # Save models using joblib
            model_data = {
                "success_predictor": self.success_predictor,
                "execution_time_predictor": self.execution_time_predictor,
                "feature_scaler": self.feature_scaler,
                "model_version": self.learning_state.model_version,
                "training_timestamp": datetime.utcnow().isoformat()
            }
            
            # Save to Redis for quick access
            if self.redis_client:
                serialized_models = pickle.dumps(model_data)
                await self.redis_client.setex(
                    "learning_models", 
                    86400,  # 24 hours
                    serialized_models
                )
                
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            
    async def _load_models(self):
        """Load trained models from storage"""
        try:
            if self.redis_client:
                serialized_models = await self.redis_client.get("learning_models")
                if serialized_models:
                    model_data = pickle.loads(serialized_models)
                    
                    self.success_predictor = model_data["success_predictor"]
                    self.execution_time_predictor = model_data["execution_time_predictor"]
                    self.feature_scaler = model_data["feature_scaler"]
                    
                    logger.info("Loaded trained models from cache")
                    return True
                    
        except Exception as e:
            logger.warning(f"Failed to load models: {e}")
            
        return False
        
    async def _store_model_performance(self, accuracy: float, rmse: float, 
                                     train_samples: int, test_samples: int):
        """Store model performance metrics"""
        try:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO model_performance 
                       (model_type, model_version, accuracy, training_samples, 
                        validation_samples, training_time, metadata)
                       VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                    "success_predictor",
                    self.learning_state.model_version,
                    accuracy,
                    train_samples,
                    test_samples,
                    0.0,  # training_time would be measured in real implementation
                    json.dumps({"rmse": rmse})
                )
        except Exception as e:
            logger.error(f"Failed to store model performance: {e}")
            
    async def _load_learning_state(self):
        """Load learning state from storage"""
        try:
            if self.redis_client:
                state_data = await self.redis_client.get("learning_state")
                if state_data:
                    state_dict = json.loads(state_data)
                    
                    # Reconstruct technique performances
                    performances = {}
                    for tech_id, perf_data in state_dict.get("technique_performances", {}).items():
                        perf = TechniquePerformance(**perf_data)
                        performances[tech_id] = perf
                        
                    self.learning_state = LearningState(
                        model_version=state_dict["model_version"],
                        exploration_rate=state_dict["exploration_rate"],
                        learning_mode=LearningMode(state_dict["learning_mode"]),
                        technique_performances=performances,
                        feature_importance=state_dict["feature_importance"],
                        model_accuracy=state_dict["model_accuracy"],
                        last_training=datetime.fromisoformat(state_dict["last_training"]),
                        exploration_budget=state_dict["exploration_budget"],
                        exploitation_threshold=state_dict["exploitation_threshold"]
                    )
                    
                    logger.info("Loaded learning state from cache")
                    
        except Exception as e:
            logger.warning(f"Failed to load learning state: {e}")
            
    async def _save_learning_state(self):
        """Save learning state to storage"""
        try:
            # Convert to serializable format
            state_dict = {
                "model_version": self.learning_state.model_version,
                "exploration_rate": self.learning_state.exploration_rate,
                "learning_mode": self.learning_state.learning_mode.value,
                "technique_performances": {
                    tech_id: asdict(perf) 
                    for tech_id, perf in self.learning_state.technique_performances.items()
                },
                "feature_importance": self.learning_state.feature_importance,
                "model_accuracy": self.learning_state.model_accuracy,
                "last_training": self.learning_state.last_training.isoformat(),
                "exploration_budget": self.learning_state.exploration_budget,
                "exploitation_threshold": self.learning_state.exploitation_threshold
            }
            
            if self.redis_client:
                await self.redis_client.setex(
                    "learning_state",
                    3600,  # 1 hour
                    json.dumps(state_dict, default=str)
                )
                
        except Exception as e:
            logger.error(f"Failed to save learning state: {e}")
            
    async def _load_historical_data(self):
        """Load historical performance data to initialize learning state"""
        if not self.postgres_pool:
            return
            
        try:
            # Load technique performance summaries
            query = """
            SELECT technique_id,
                   COUNT(*) as total_executions,
                   SUM(CASE WHEN success THEN 1 ELSE 0 END) as success_count,
                   AVG(execution_time) as avg_execution_time
            FROM technique_exploration
            WHERE timestamp >= NOW() - INTERVAL '30 days'
            GROUP BY technique_id
            """
            
            async with self.postgres_pool.acquire() as conn:
                rows = await conn.fetch(query)
                
                for row in rows:
                    technique_id = row["technique_id"]
                    total = row["total_executions"]
                    successes = row["success_count"]
                    failures = total - successes
                    avg_time = float(row["avg_execution_time"] or 0)
                    
                    perf = TechniquePerformance(
                        technique_id=technique_id,
                        success_count=successes,
                        failure_count=failures,
                        avg_execution_time=avg_time
                    )
                    perf.update_metrics()
                    
                    self.learning_state.technique_performances[technique_id] = perf
                    
            logger.info(f"Loaded historical data for {len(self.learning_state.technique_performances)} techniques")
            
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            
    async def _adapt_exploration_strategy(self):
        """Periodically adapt exploration strategy based on performance"""
        while self.is_running:
            try:
                await asyncio.sleep(1800)  # Check every 30 minutes
                
                # Analyze recent performance
                recent_performance = await self._analyze_recent_performance()
                
                # Adapt learning mode
                if recent_performance["success_rate"] < 0.5:
                    self.learning_state.learning_mode = LearningMode.EXPLORATION
                elif recent_performance["success_rate"] > 0.8:
                    self.learning_state.learning_mode = LearningMode.EXPLOITATION
                else:
                    self.learning_state.learning_mode = LearningMode.BALANCED
                    
                logger.debug(f"Adapted learning mode to {self.learning_state.learning_mode.value}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in exploration strategy adaptation: {e}")
                await asyncio.sleep(300)
                
    async def _analyze_recent_performance(self) -> Dict[str, Any]:
        """Analyze recent performance to guide strategy adaptation"""
        if not self.postgres_pool:
            return {"success_rate": 0.5, "execution_count": 0}
            
        # Analyze last hour's performance
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        query = """
        SELECT COUNT(*) as total,
               SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes,
               AVG(execution_time) as avg_time
        FROM technique_exploration
        WHERE timestamp >= $1
        """
        
        try:
            async with self.postgres_pool.acquire() as conn:
                row = await conn.fetchrow(query, cutoff_time)
                
                total = row["total"] or 0
                successes = row["successes"] or 0
                avg_time = float(row["avg_time"] or 0)
                
                success_rate = successes / total if total > 0 else 0.5
                
                return {
                    "success_rate": success_rate,
                    "execution_count": total,
                    "avg_execution_time": avg_time
                }
                
        except Exception as e:
            logger.error(f"Failed to analyze recent performance: {e}")
            return {"success_rate": 0.5, "execution_count": 0}
            
    async def _monitor_performance(self):
        """Monitor learning system performance"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Log current state
                logger.debug(f"Learning State - Exploration Rate: {self.learning_state.exploration_rate:.3f}, "
                           f"Mode: {self.learning_state.learning_mode.value}, "
                           f"Techniques: {len(self.learning_state.technique_performances)}, "
                           f"Model Accuracy: {self.learning_state.model_accuracy:.3f}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)
                
    async def _log_technique_suggestion(self, recommendation: Dict[str, Any], context: Dict[str, Any]):
        """Log technique suggestion for analysis"""
        logger.debug(f"Suggested technique {recommendation['technique_id']} "
                    f"with confidence {recommendation['confidence']:.3f} "
                    f"(reason: {recommendation['reason']})")
                    
    async def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        total_techniques = len(self.learning_state.technique_performances)
        total_executions = sum(perf.total_executions for perf in self.learning_state.technique_performances.values())
        total_successes = sum(perf.success_count for perf in self.learning_state.technique_performances.values())
        
        overall_success_rate = total_successes / total_executions if total_executions > 0 else 0.0
        
        # Top performing techniques
        top_techniques = sorted(
            self.learning_state.technique_performances.values(),
            key=lambda p: p.success_rate,
            reverse=True
        )[:5]
        
        return {
            "learning_state": {
                "model_version": self.learning_state.model_version,
                "exploration_rate": self.learning_state.exploration_rate,
                "learning_mode": self.learning_state.learning_mode.value,
                "model_accuracy": self.learning_state.model_accuracy,
                "last_training": self.learning_state.last_training.isoformat()
            },
            "performance_summary": {
                "total_techniques": total_techniques,
                "total_executions": total_executions,
                "overall_success_rate": overall_success_rate,
                "avg_execution_time": np.mean([p.avg_execution_time for p in self.learning_state.technique_performances.values()]) if total_techniques > 0 else 0.0
            },
            "top_techniques": [
                {
                    "technique_id": perf.technique_id,
                    "success_rate": perf.success_rate,
                    "total_executions": perf.total_executions,
                    "confidence_interval": perf.confidence_interval
                }
                for perf in top_techniques
            ],
            "feature_importance": dict(sorted(
                self.learning_state.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])  # Top 10 most important features
        }
        
    async def shutdown(self):
        """Shutdown the autonomous explorer"""
        logger.info("Shutting down Autonomous Explorer...")
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
            
        try:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        except Exception as e:
            logger.warning(f"Error during background task shutdown: {e}")
            
        # Save final state
        await self._save_learning_state()
        await self._save_models()
        
        logger.info("Autonomous Explorer shutdown complete")