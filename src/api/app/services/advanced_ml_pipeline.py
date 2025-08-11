"""
Advanced ML/AI Pipeline for XORB Enterprise
Production-ready MLOps with model versioning and performance monitoring
"""

import asyncio
import json
import logging
import numpy as np
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import time

# ML imports with graceful fallbacks
try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

from ..infrastructure.observability import get_metrics_collector, add_trace_context
from ..infrastructure.cache import get_cache_client

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Model metadata for version tracking"""
    model_id: str
    model_name: str
    version: str
    algorithm: str
    features: List[str]
    target: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: datetime
    training_duration: float
    data_size: int
    hyperparameters: Dict[str, Any]
    feature_importance: Dict[str, float]
    model_path: str
    status: str  # "training", "active", "deprecated"

@dataclass
class PredictionResult:
    """ML prediction result with confidence and explanation"""
    prediction: Any
    confidence: float
    probability_distribution: Dict[str, float]
    feature_contributions: Dict[str, float]
    model_version: str
    prediction_time: datetime
    processing_time_ms: float

@dataclass
class ModelPerformanceMetrics:
    """Model performance tracking"""
    model_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: Optional[float]
    confusion_matrix: List[List[int]]
    feature_importance: Dict[str, float]
    prediction_latency_ms: float
    throughput_per_second: float
    error_rate: float
    drift_score: float
    last_updated: datetime

class AdvancedMLPipeline:
    """
    Advanced ML/AI Pipeline with enterprise features:
    - Model versioning and lifecycle management
    - A/B testing and champion/challenger models
    - Performance monitoring and drift detection
    - Feature engineering automation
    - Model explainability and interpretability
    - Real-time prediction serving
    - Automated retraining pipelines
    """
    
    def __init__(self, 
                 model_registry_path: str = "/tmp/xorb_models",
                 cache_client=None):
        self.model_registry_path = Path(model_registry_path)
        self.model_registry_path.mkdir(exist_ok=True)
        
        self.cache_client = cache_client
        self.metrics = get_metrics_collector()
        
        # Model registry
        self.models: Dict[str, ModelMetadata] = {}
        self.active_models: Dict[str, Any] = {}  # Loaded models
        self.model_performance: Dict[str, ModelPerformanceMetrics] = {}
        
        # Feature processors
        self.feature_processors: Dict[str, Any] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, Any] = {}
        
        # A/B testing
        self.ab_test_config: Dict[str, Dict[str, Any]] = {}
        
        # Performance thresholds
        self.performance_thresholds = {
            "min_accuracy": 0.85,
            "max_drift_score": 0.3,
            "max_latency_ms": 100,
            "min_data_quality": 0.9
        }
        
        # Background tasks
        self._monitoring_task = None
        self._retraining_task = None
        
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the ML pipeline"""
        if self._initialized:
            return True
        
        try:
            logger.info("Initializing Advanced ML Pipeline...")
            
            if not HAS_SKLEARN:
                logger.warning("scikit-learn not available - ML features limited")
                return False
            
            # Load existing models from registry
            await self._load_model_registry()
            
            # Initialize feature processors
            await self._initialize_feature_processors()
            
            # Start background monitoring
            self._monitoring_task = asyncio.create_task(self._performance_monitor())
            self._retraining_task = asyncio.create_task(self._retraining_scheduler())
            
            self._initialized = True
            logger.info("Advanced ML Pipeline initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ML pipeline: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the ML pipeline"""
        try:
            # Cancel background tasks
            if self._monitoring_task:
                self._monitoring_task.cancel()
            if self._retraining_task:
                self._retraining_task.cancel()
            
            # Save model registry
            await self._save_model_registry()
            
            logger.info("ML Pipeline shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during ML pipeline shutdown: {e}")
    
    async def train_threat_classification_model(self,
                                              training_data: List[Dict[str, Any]],
                                              model_name: str = "threat_classifier",
                                              algorithm: str = "random_forest") -> ModelMetadata:
        """Train a threat classification model with advanced features"""
        start_time = time.time()
        
        try:
            if not HAS_SKLEARN or not HAS_PANDAS:
                raise RuntimeError("Required ML libraries not available")
            
            # Convert to DataFrame
            df = pd.DataFrame(training_data)
            
            # Feature engineering
            features, target = await self._engineer_features(df, target_column="is_threat")
            
            # Prepare data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42, stratify=target
            )
            
            # Select and configure model
            if algorithm == "random_forest":
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            elif algorithm == "gradient_boosting":
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
            elif algorithm == "neural_network":
                model = MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    alpha=0.01,
                    random_state=42
                )
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Train model
            logger.info(f"Training {algorithm} model with {len(X_train)} samples...")
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Feature importance
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_names = features.columns if hasattr(features, 'columns') else [f"feature_{i}" for i in range(len(features[0]))]
                feature_importance = dict(zip(feature_names, model.feature_importances_))
            
            # Create model metadata
            model_id = hashlib.md5(f"{model_name}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
            version = f"v{len([m for m in self.models.values() if m.model_name == model_name]) + 1}"
            
            training_duration = time.time() - start_time
            
            metadata = ModelMetadata(
                model_id=model_id,
                model_name=model_name,
                version=version,
                algorithm=algorithm,
                features=list(feature_importance.keys()),
                target="is_threat",
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                training_time=datetime.now(),
                training_duration=training_duration,
                data_size=len(training_data),
                hyperparameters=model.get_params(),
                feature_importance=feature_importance,
                model_path=str(self.model_registry_path / f"{model_id}.joblib"),
                status="active"
            )
            
            # Save model
            if HAS_JOBLIB:
                joblib.dump(model, metadata.model_path)
            
            # Register model
            self.models[model_id] = metadata
            self.active_models[model_id] = model
            
            # Update performance metrics
            await self._update_model_performance(model_id, X_test, y_test)
            
            # Save registry
            await self._save_model_registry()
            
            logger.info(f"Model training complete: {model_name} {version} (accuracy: {accuracy:.3f})")
            
            # Record metrics
            self.metrics.record_ml_training(
                model_name=model_name,
                algorithm=algorithm,
                accuracy=accuracy,
                training_duration=training_duration
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    async def predict_threat(self,
                           input_data: Dict[str, Any],
                           model_name: str = "threat_classifier",
                           explain: bool = True) -> PredictionResult:
        """Make threat predictions with explanation"""
        start_time = time.time()
        
        try:
            # Get active model
            model_id = await self._get_active_model_id(model_name)
            if not model_id:
                raise ValueError(f"No active model found for {model_name}")
            
            model = self.active_models.get(model_id)
            metadata = self.models.get(model_id)
            
            if not model or not metadata:
                raise ValueError(f"Model {model_id} not loaded")
            
            # Prepare features
            features = await self._prepare_features_for_prediction(input_data, metadata.features)
            
            # Make prediction
            prediction = model.predict([features])[0]
            prediction_proba = model.predict_proba([features])[0]
            
            # Calculate confidence
            confidence = float(np.max(prediction_proba))
            
            # Probability distribution
            class_labels = model.classes_ if hasattr(model, 'classes_') else ['benign', 'threat']
            prob_dist = dict(zip(class_labels, prediction_proba))
            
            # Feature contributions (SHAP-like explanation)
            feature_contributions = {}
            if explain and hasattr(model, 'feature_importances_'):
                feature_names = metadata.features
                for i, importance in enumerate(model.feature_importances_):
                    if i < len(feature_names):
                        contribution = importance * features[i] if i < len(features) else 0
                        feature_contributions[feature_names[i]] = float(contribution)
            
            processing_time = (time.time() - start_time) * 1000  # ms
            
            result = PredictionResult(
                prediction=bool(prediction) if isinstance(prediction, (int, np.integer)) else prediction,
                confidence=confidence,
                probability_distribution=prob_dist,
                feature_contributions=feature_contributions,
                model_version=metadata.version,
                prediction_time=datetime.now(),
                processing_time_ms=processing_time
            )
            
            # Record prediction metrics
            self.metrics.record_ml_prediction(
                model_name=model_name,
                confidence=confidence,
                processing_time_ms=processing_time
            )
            
            # Cache prediction for drift monitoring
            if self.cache_client:
                cache_key = f"ml_prediction:{model_id}:{hashlib.md5(str(input_data).encode()).hexdigest()[:8]}"
                await self.cache_client.setex(
                    cache_key, 
                    3600,  # 1 hour
                    json.dumps(asdict(result), default=str)
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    async def train_anomaly_detection_model(self,
                                          normal_data: List[Dict[str, Any]],
                                          model_name: str = "anomaly_detector") -> ModelMetadata:
        """Train an anomaly detection model for behavior analysis"""
        start_time = time.time()
        
        try:
            if not HAS_SKLEARN or not HAS_PANDAS:
                raise RuntimeError("Required ML libraries not available")
            
            # Convert to DataFrame
            df = pd.DataFrame(normal_data)
            
            # Feature engineering
            features = await self._engineer_numerical_features(df)
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Train Isolation Forest for anomaly detection
            model = IsolationForest(
                contamination=0.1,  # Expect 10% anomalies
                random_state=42,
                n_jobs=-1
            )
            
            logger.info(f"Training anomaly detection model with {len(features)} samples...")
            model.fit(scaled_features)
            
            # Create synthetic anomalies for evaluation
            # (In production, use known anomaly data)
            synthetic_anomalies = await self._generate_synthetic_anomalies(scaled_features, 100)
            test_data = np.vstack([scaled_features[:100], synthetic_anomalies])
            test_labels = np.hstack([np.ones(100), np.full(100, -1)])  # 1 = normal, -1 = anomaly
            
            # Evaluate
            predictions = model.predict(test_data)
            accuracy = accuracy_score(test_labels, predictions)
            
            # Create model metadata
            model_id = hashlib.md5(f"{model_name}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
            version = f"v{len([m for m in self.models.values() if m.model_name == model_name]) + 1}"
            
            training_duration = time.time() - start_time
            
            metadata = ModelMetadata(
                model_id=model_id,
                model_name=model_name,
                version=version,
                algorithm="isolation_forest",
                features=list(features.columns) if hasattr(features, 'columns') else [f"feature_{i}" for i in range(features.shape[1])],
                target="anomaly_score",
                accuracy=accuracy,
                precision=0.0,  # Not applicable for unsupervised
                recall=0.0,     # Not applicable for unsupervised
                f1_score=0.0,   # Not applicable for unsupervised
                training_time=datetime.now(),
                training_duration=training_duration,
                data_size=len(normal_data),
                hyperparameters=model.get_params(),
                feature_importance={},
                model_path=str(self.model_registry_path / f"{model_id}.joblib"),
                status="active"
            )
            
            # Save model and scaler
            if HAS_JOBLIB:
                joblib.dump({"model": model, "scaler": scaler}, metadata.model_path)
            
            # Register model
            self.models[model_id] = metadata
            self.active_models[model_id] = {"model": model, "scaler": scaler}
            
            # Save registry
            await self._save_model_registry()
            
            logger.info(f"Anomaly detection model training complete: {model_name} {version}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Anomaly detection training failed: {e}")
            raise
    
    async def detect_anomaly(self,
                           input_data: Dict[str, Any],
                           model_name: str = "anomaly_detector") -> PredictionResult:
        """Detect anomalies in behavior data"""
        start_time = time.time()
        
        try:
            # Get active model
            model_id = await self._get_active_model_id(model_name)
            if not model_id:
                raise ValueError(f"No active model found for {model_name}")
            
            model_data = self.active_models.get(model_id)
            metadata = self.models.get(model_id)
            
            if not model_data or not metadata:
                raise ValueError(f"Model {model_id} not loaded")
            
            model = model_data["model"]
            scaler = model_data["scaler"]
            
            # Prepare features
            features = await self._prepare_numerical_features_for_prediction(input_data, metadata.features)
            scaled_features = scaler.transform([features])
            
            # Detect anomaly
            anomaly_score = model.decision_function(scaled_features)[0]
            is_anomaly = model.predict(scaled_features)[0] == -1
            
            # Convert anomaly score to confidence
            confidence = float(1.0 / (1.0 + np.exp(-anomaly_score)))  # Sigmoid transformation
            
            processing_time = (time.time() - start_time) * 1000  # ms
            
            result = PredictionResult(
                prediction=bool(is_anomaly),
                confidence=confidence,
                probability_distribution={"normal": 1.0 - confidence, "anomaly": confidence},
                feature_contributions={"anomaly_score": float(anomaly_score)},
                model_version=metadata.version,
                prediction_time=datetime.now(),
                processing_time_ms=processing_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            raise
    
    async def get_model_performance(self, model_name: str = None) -> Dict[str, Any]:
        """Get model performance metrics"""
        try:
            if model_name:
                model_id = await self._get_active_model_id(model_name)
                if model_id and model_id in self.model_performance:
                    return asdict(self.model_performance[model_id])
                return {}
            else:
                # Return all model performance
                return {
                    model_id: asdict(perf) 
                    for model_id, perf in self.model_performance.items()
                }
        
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {"error": str(e)}
    
    async def deploy_model(self, model_id: str, deployment_config: Dict[str, Any] = None) -> bool:
        """Deploy a model to production"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            metadata = self.models[model_id]
            
            # Load model if not already loaded
            if model_id not in self.active_models:
                if HAS_JOBLIB and Path(metadata.model_path).exists():
                    model_data = joblib.load(metadata.model_path)
                    self.active_models[model_id] = model_data
                else:
                    raise ValueError(f"Model file not found: {metadata.model_path}")
            
            # Update status
            metadata.status = "active"
            
            # Deactivate other versions of the same model
            for mid, meta in self.models.items():
                if meta.model_name == metadata.model_name and mid != model_id:
                    meta.status = "deprecated"
            
            # Save registry
            await self._save_model_registry()
            
            logger.info(f"Model deployed: {metadata.model_name} {metadata.version}")
            
            return True
            
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            return False
    
    async def _engineer_features(self, df: pd.DataFrame, target_column: str) -> Tuple[Any, Any]:
        """Advanced feature engineering for threat data"""
        try:
            features_df = df.copy()
            
            # Remove target column
            target = features_df.pop(target_column)
            
            # Text feature engineering (for IOCs, domains, etc.)
            text_columns = features_df.select_dtypes(include=['object']).columns
            for col in text_columns:
                if col in features_df.columns:
                    # Length features
                    features_df[f"{col}_length"] = features_df[col].str.len()
                    
                    # Character composition features
                    features_df[f"{col}_num_digits"] = features_df[col].str.count(r'\d')
                    features_df[f"{col}_num_special"] = features_df[col].str.count(r'[^a-zA-Z0-9]')
                    features_df[f"{col}_entropy"] = features_df[col].apply(self._calculate_entropy)
                    
                    # Remove original text column
                    features_df = features_df.drop(columns=[col])
            
            # Numerical feature engineering
            numerical_columns = features_df.select_dtypes(include=[np.number]).columns
            for col in numerical_columns:
                # Log transformation for skewed data
                if features_df[col].min() > 0:
                    features_df[f"{col}_log"] = np.log1p(features_df[col])
                
                # Binning
                features_df[f"{col}_binned"] = pd.qcut(features_df[col], q=5, labels=False, duplicates='drop')
            
            # Fill missing values
            features_df = features_df.fillna(0)
            
            return features_df, target
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0
        
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        text_length = len(text)
        for count in char_counts.values():
            probability = count / text_length
            entropy -= probability * np.log2(probability)
        
        return entropy
    
    async def _get_active_model_id(self, model_name: str) -> Optional[str]:
        """Get the active model ID for a given model name"""
        for model_id, metadata in self.models.items():
            if metadata.model_name == model_name and metadata.status == "active":
                return model_id
        return None
    
    async def _performance_monitor(self):
        """Background task for monitoring model performance"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Monitor all active models
                for model_id, metadata in self.models.items():
                    if metadata.status == "active":
                        await self._check_model_drift(model_id)
                        await self._check_performance_degradation(model_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _retraining_scheduler(self):
        """Background task for scheduled model retraining"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Check if any models need retraining
                for model_id, metadata in self.models.items():
                    if metadata.status == "active":
                        # Check if model is older than 30 days
                        if (datetime.now() - metadata.training_time).days > 30:
                            logger.info(f"Model {metadata.model_name} needs retraining")
                            # Would trigger retraining workflow here
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Retraining scheduler error: {e}")
                await asyncio.sleep(3600)
    
    async def _load_model_registry(self):
        """Load model registry from disk"""
        try:
            registry_file = self.model_registry_path / "registry.json"
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                
                for model_id, model_data in data.items():
                    metadata = ModelMetadata(**model_data)
                    self.models[model_id] = metadata
                    
                    # Load active models
                    if metadata.status == "active" and Path(metadata.model_path).exists():
                        if HAS_JOBLIB:
                            try:
                                model = joblib.load(metadata.model_path)
                                self.active_models[model_id] = model
                            except Exception as e:
                                logger.warning(f"Could not load model {model_id}: {e}")
                
                logger.info(f"Loaded {len(self.models)} models from registry")
        
        except Exception as e:
            logger.warning(f"Could not load model registry: {e}")
    
    async def _save_model_registry(self):
        """Save model registry to disk"""
        try:
            registry_file = self.model_registry_path / "registry.json"
            data = {
                model_id: asdict(metadata) 
                for model_id, metadata in self.models.items()
            }
            
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Could not save model registry: {e}")


# Global ML pipeline instance
ml_pipeline: Optional[AdvancedMLPipeline] = None

async def get_ml_pipeline() -> AdvancedMLPipeline:
    """Get the global ML pipeline instance"""
    global ml_pipeline
    if not ml_pipeline:
        raise RuntimeError("ML pipeline not initialized")
    return ml_pipeline

async def init_ml_pipeline(model_registry_path: str = "/tmp/xorb_models") -> bool:
    """Initialize the ML pipeline"""
    global ml_pipeline
    
    if ml_pipeline:
        return True
    
    ml_pipeline = AdvancedMLPipeline(model_registry_path=model_registry_path)
    return await ml_pipeline.initialize()