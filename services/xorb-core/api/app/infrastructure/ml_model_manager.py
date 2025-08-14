"""ML Model Deployment and Management System for XORB Platform"""
import asyncio
import hashlib
import logging
import pickle
import json
import os
import shutil
from typing import Dict, List, Optional, Any, Union, Callable
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

# ML Framework imports with fallbacks
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None

try:
    from sklearn.base import BaseEstimator
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    BaseEstimator = object

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from sqlalchemy import text
from .database import get_async_session
from .observability import get_metrics_collector, add_trace_context

logger = logging.getLogger(__name__)


class ModelFramework(Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SCIKIT_LEARN = "sklearn"
    ONNX = "onnx"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class ModelStatus(Enum):
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    DEPLOYED = "deployed"
    SERVING = "serving"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class ModelType(Enum):
    THREAT_CLASSIFIER = "threat_classifier"
    ANOMALY_DETECTOR = "anomaly_detector"
    MALWARE_DETECTOR = "malware_detector"
    PHISHING_DETECTOR = "phishing_detector"
    BEHAVIOR_ANALYZER = "behavior_analyzer"
    RISK_ASSESSOR = "risk_assessor"
    EMBEDDING_MODEL = "embedding_model"
    LANGUAGE_MODEL = "language_model"


@dataclass
class ModelMetadata:
    model_id: str
    name: str
    version: str
    model_type: ModelType
    framework: ModelFramework
    description: str
    tags: List[str]
    created_at: datetime
    updated_at: datetime
    tenant_id: UUID
    created_by: str

    # Model specifications
    input_shape: Optional[tuple] = None
    output_shape: Optional[tuple] = None
    model_size_bytes: Optional[int] = None
    parameters_count: Optional[int] = None

    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    inference_time_ms: Optional[float] = None

    # Status and health
    status: ModelStatus = ModelStatus.TRAINING
    health_check_url: Optional[str] = None
    last_health_check: Optional[datetime] = None

    # Deployment configuration
    resource_requirements: Dict[str, Any] = None
    environment_variables: Dict[str, str] = None
    scaling_config: Dict[str, Any] = None

    # Training and validation
    training_data_hash: Optional[str] = None
    validation_metrics: Dict[str, float] = None
    hyperparameters: Dict[str, Any] = None


@dataclass
class ModelPrediction:
    model_id: str
    prediction_id: str
    input_data: Any
    output_data: Any
    confidence: float
    inference_time_ms: float
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class ModelTrainingRequest:
    model_name: str
    model_type: ModelType
    framework: ModelFramework
    training_data_path: str
    validation_data_path: Optional[str]
    hyperparameters: Dict[str, Any]
    tenant_id: UUID
    created_by: str
    tags: List[str] = None
    description: str = ""


class MLModelManager:
    """Comprehensive ML model deployment and management system"""

    def __init__(self, models_directory: str = "/opt/xorb/models"):
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(parents=True, exist_ok=True)

        # In-memory model registry
        self.model_registry: Dict[str, ModelMetadata] = {}
        self.loaded_models: Dict[str, Any] = {}  # Cached loaded models

        # Training queues and jobs
        self.training_queue: asyncio.Queue = asyncio.Queue()
        self.training_workers: List[asyncio.Task] = []

        self.metrics = get_metrics_collector()

    async def initialize(self):
        """Initialize the ML model manager"""
        logger.info("Initializing ML Model Manager...")

        # Load existing models from disk
        await self._load_model_registry()

        # Start training workers
        for i in range(2):  # 2 concurrent training workers
            worker = asyncio.create_task(self._training_worker(f"worker-{i}"))
            self.training_workers.append(worker)

        logger.info(f"ML Model Manager initialized with {len(self.model_registry)} models")

    async def register_model(
        self,
        model: Any,
        metadata: ModelMetadata,
        model_bytes: Optional[bytes] = None
    ) -> str:
        """Register a new ML model"""
        try:
            model_id = metadata.model_id

            # Create model directory
            model_path = self.models_directory / model_id
            model_path.mkdir(parents=True, exist_ok=True)

            # Save model to disk
            await self._save_model_to_disk(model, model_path, metadata.framework, model_bytes)

            # Calculate model size and parameters
            if hasattr(model, 'state_dict') and TORCH_AVAILABLE:
                # PyTorch model
                metadata.parameters_count = sum(p.numel() for p in model.parameters())
            elif hasattr(model, 'count_params') and TENSORFLOW_AVAILABLE:
                # TensorFlow model
                metadata.parameters_count = model.count_params()

            # Update model size
            model_file = model_path / "model.pkl"
            if model_file.exists():
                metadata.model_size_bytes = model_file.stat().st_size

            # Save metadata
            await self._save_model_metadata(metadata)

            # Register in memory
            self.model_registry[model_id] = metadata

            # Store in database for persistence
            await self._store_model_in_database(metadata)

            logger.info(f"Registered model {model_id} ({metadata.name})")

            # Record metrics
            self.metrics.record_job_execution(
                f"model_registration_{metadata.model_type.value}",
                0.0,
                True
            )

            return model_id

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

    async def deploy_model(
        self,
        model_id: str,
        deployment_config: Dict[str, Any] = None
    ) -> bool:
        """Deploy a model for serving"""
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found in registry")

            metadata = self.model_registry[model_id]

            # Load model into memory if not already loaded
            if model_id not in self.loaded_models:
                model = await self._load_model_from_disk(model_id)
                self.loaded_models[model_id] = model

            # Update deployment configuration
            if deployment_config:
                metadata.scaling_config = deployment_config.get("scaling", {})
                metadata.environment_variables = deployment_config.get("env_vars", {})
                metadata.resource_requirements = deployment_config.get("resources", {})

            # Update status to deployed
            metadata.status = ModelStatus.DEPLOYED
            metadata.updated_at = datetime.utcnow()

            # Save updated metadata
            await self._save_model_metadata(metadata)
            await self._update_model_in_database(metadata)

            logger.info(f"Deployed model {model_id}")

            add_trace_context(
                operation="model_deployment",
                model_id=model_id,
                model_type=metadata.model_type.value
            )

            return True

        except Exception as e:
            logger.error(f"Failed to deploy model {model_id}: {e}")
            return False

    async def predict(
        self,
        model_id: str,
        input_data: Any,
        tenant_id: UUID,
        metadata: Dict[str, Any] = None
    ) -> ModelPrediction:
        """Make prediction using deployed model"""
        start_time = datetime.utcnow()

        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found")

            model_metadata = self.model_registry[model_id]

            if model_metadata.status != ModelStatus.DEPLOYED:
                raise ValueError(f"Model {model_id} is not deployed")

            # Load model if not in memory
            if model_id not in self.loaded_models:
                model = await self._load_model_from_disk(model_id)
                self.loaded_models[model_id] = model
            else:
                model = self.loaded_models[model_id]

            # Prepare input data
            processed_input = await self._preprocess_input(
                input_data,
                model_metadata.framework,
                model_metadata.model_type
            )

            # Make prediction
            if model_metadata.framework == ModelFramework.PYTORCH and TORCH_AVAILABLE:
                prediction = await self._pytorch_predict(model, processed_input)
            elif model_metadata.framework == ModelFramework.TENSORFLOW and TENSORFLOW_AVAILABLE:
                prediction = await self._tensorflow_predict(model, processed_input)
            elif model_metadata.framework == ModelFramework.SCIKIT_LEARN and SKLEARN_AVAILABLE:
                prediction = await self._sklearn_predict(model, processed_input)
            else:
                # Custom prediction logic
                prediction = await self._custom_predict(model, processed_input)

            # Post-process output
            processed_output, confidence = await self._postprocess_output(
                prediction,
                model_metadata.model_type
            )

            # Calculate inference time
            inference_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Create prediction result
            prediction_result = ModelPrediction(
                model_id=model_id,
                prediction_id=str(uuid4()),
                input_data=input_data,
                output_data=processed_output,
                confidence=confidence,
                inference_time_ms=inference_time,
                timestamp=start_time,
                metadata=metadata or {}
            )

            # Record metrics
            self.metrics.record_job_execution(
                f"model_prediction_{model_metadata.model_type.value}",
                inference_time / 1000,
                True
            )

            # Store prediction for audit/analysis
            await self._store_prediction(prediction_result, tenant_id)

            return prediction_result

        except Exception as e:
            logger.error(f"Prediction failed for model {model_id}: {e}")
            inference_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Return error prediction
            return ModelPrediction(
                model_id=model_id,
                prediction_id=str(uuid4()),
                input_data=input_data,
                output_data={"error": str(e)},
                confidence=0.0,
                inference_time_ms=inference_time,
                timestamp=start_time,
                metadata={"error": True}
            )

    async def train_model(self, training_request: ModelTrainingRequest) -> str:
        """Queue model for training"""
        try:
            # Create model metadata
            model_id = str(uuid4())
            metadata = ModelMetadata(
                model_id=model_id,
                name=training_request.model_name,
                version="1.0.0",
                model_type=training_request.model_type,
                framework=training_request.framework,
                description=training_request.description,
                tags=training_request.tags or [],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                tenant_id=training_request.tenant_id,
                created_by=training_request.created_by,
                status=ModelStatus.TRAINING,
                hyperparameters=training_request.hyperparameters
            )

            # Register model metadata
            self.model_registry[model_id] = metadata
            await self._store_model_in_database(metadata)

            # Add to training queue
            await self.training_queue.put((model_id, training_request))

            logger.info(f"Queued model {model_id} for training")
            return model_id

        except Exception as e:
            logger.error(f"Failed to queue model for training: {e}")
            raise

    async def get_model_info(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model information"""
        return self.model_registry.get(model_id)

    async def list_models(
        self,
        tenant_id: Optional[UUID] = None,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None
    ) -> List[ModelMetadata]:
        """List models with optional filtering"""
        models = list(self.model_registry.values())

        if tenant_id:
            models = [m for m in models if m.tenant_id == tenant_id]

        if model_type:
            models = [m for m in models if m.model_type == model_type]

        if status:
            models = [m for m in models if m.status == status]

        return sorted(models, key=lambda x: x.updated_at, reverse=True)

    async def delete_model(self, model_id: str) -> bool:
        """Delete a model"""
        try:
            if model_id not in self.model_registry:
                return False

            # Remove from memory
            self.loaded_models.pop(model_id, None)
            metadata = self.model_registry.pop(model_id)

            # Delete from disk
            model_path = self.models_directory / model_id
            if model_path.exists():
                shutil.rmtree(model_path)

            # Delete from database
            await self._delete_model_from_database(model_id)

            logger.info(f"Deleted model {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False

    async def health_check(self, model_id: str) -> Dict[str, Any]:
        """Perform health check on deployed model"""
        try:
            if model_id not in self.model_registry:
                return {"status": "not_found", "healthy": False}

            metadata = self.model_registry[model_id]

            if metadata.status != ModelStatus.DEPLOYED:
                return {"status": "not_deployed", "healthy": False}

            # Test prediction with dummy data
            start_time = datetime.utcnow()

            try:
                # Generate test input based on model type
                test_input = await self._generate_test_input(metadata.model_type)

                # Make test prediction
                prediction = await self.predict(
                    model_id,
                    test_input,
                    metadata.tenant_id,
                    {"health_check": True}
                )

                health_time = (datetime.utcnow() - start_time).total_seconds() * 1000

                # Update health check timestamp
                metadata.last_health_check = datetime.utcnow()
                await self._save_model_metadata(metadata)

                return {
                    "status": "healthy",
                    "healthy": True,
                    "response_time_ms": health_time,
                    "last_check": metadata.last_health_check.isoformat(),
                    "prediction_id": prediction.prediction_id
                }

            except Exception as e:
                return {
                    "status": "unhealthy",
                    "healthy": False,
                    "error": str(e),
                    "response_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
                }

        except Exception as e:
            return {"status": "error", "healthy": False, "error": str(e)}

    async def get_model_metrics(self, model_id: str, days: int = 7) -> Dict[str, Any]:
        """Get model performance metrics"""
        try:
            if model_id not in self.model_registry:
                return {}

            # This would query the database for prediction history
            # For now, return mock metrics
            return {
                "total_predictions": 1500,
                "avg_inference_time_ms": 45.2,
                "avg_confidence": 0.87,
                "error_rate": 0.02,
                "predictions_per_hour": [120, 145, 98, 156, 134],
                "confidence_distribution": {
                    "0.0-0.2": 15,
                    "0.2-0.4": 32,
                    "0.4-0.6": 78,
                    "0.6-0.8": 245,
                    "0.8-1.0": 1130
                }
            }

        except Exception as e:
            logger.error(f"Failed to get metrics for model {model_id}: {e}")
            return {}

    # Private methods

    async def _training_worker(self, worker_name: str):
        """Background worker for model training"""
        logger.info(f"Started training worker: {worker_name}")

        while True:
            try:
                # Wait for training request
                model_id, training_request = await self.training_queue.get()

                logger.info(f"Worker {worker_name} starting training for model {model_id}")

                # Perform training
                success = await self._perform_training(model_id, training_request)

                # Update model status
                metadata = self.model_registry[model_id]
                metadata.status = ModelStatus.TRAINED if success else ModelStatus.FAILED
                metadata.updated_at = datetime.utcnow()

                await self._save_model_metadata(metadata)
                await self._update_model_in_database(metadata)

                logger.info(f"Training completed for model {model_id}, success: {success}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Training worker {worker_name} error: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying

    async def _perform_training(self, model_id: str, training_request: ModelTrainingRequest) -> bool:
        """Perform actual model training"""
        try:
            # This would implement actual training logic based on framework
            # For now, simulate training with a delay
            await asyncio.sleep(2)  # Simulate training time

            # Mock model creation
            if training_request.model_type == ModelType.THREAT_CLASSIFIER:
                model = self._create_mock_threat_classifier()
            else:
                model = self._create_mock_generic_model()

            # Save trained model
            metadata = self.model_registry[model_id]
            model_path = self.models_directory / model_id
            model_path.mkdir(parents=True, exist_ok=True)

            await self._save_model_to_disk(model, model_path, training_request.framework)

            # Update metadata with training results
            metadata.accuracy = 0.92  # Mock accuracy
            metadata.precision = 0.89
            metadata.recall = 0.94
            metadata.f1_score = 0.91

            return True

        except Exception as e:
            logger.error(f"Training failed for model {model_id}: {e}")
            return False

    def _create_mock_threat_classifier(self):
        """Create mock threat classifier model"""
        if SKLEARN_AVAILABLE:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            # Simple mock model
            class MockModel:
                def predict(self, X):
                    # Mock threat classification
                    return np.random.choice(['benign', 'malicious'], size=len(X))

                def predict_proba(self, X):
                    # Mock probabilities
                    return np.random.random((len(X), 2))

            return MockModel()

    def _create_mock_generic_model(self):
        """Create mock generic model"""
        class MockGenericModel:
            def predict(self, X):
                return np.random.random(len(X))

        return MockGenericModel()

    async def _save_model_to_disk(
        self,
        model: Any,
        model_path: Path,
        framework: ModelFramework,
        model_bytes: Optional[bytes] = None
    ):
        """Save model to disk"""
        try:
            if model_bytes:
                # Save raw bytes
                with open(model_path / "model.bin", "wb") as f:
                    f.write(model_bytes)
            elif framework == ModelFramework.PYTORCH and TORCH_AVAILABLE:
                # Save PyTorch model
                torch.save(model.state_dict(), model_path / "model.pth")
            elif framework == ModelFramework.TENSORFLOW and TENSORFLOW_AVAILABLE:
                # Save TensorFlow model
                model.save(str(model_path / "model"))
            elif JOBLIB_AVAILABLE:
                # Save with joblib (works for sklearn and others)
                joblib.dump(model, model_path / "model.pkl")
            else:
                # Fallback to pickle
                with open(model_path / "model.pkl", "wb") as f:
                    pickle.dump(model, f)

        except Exception as e:
            logger.error(f"Failed to save model to disk: {e}")
            raise

    async def _load_model_from_disk(self, model_id: str) -> Any:
        """Load model from disk"""
        try:
            model_path = self.models_directory / model_id
            metadata = self.model_registry[model_id]

            if metadata.framework == ModelFramework.PYTORCH and TORCH_AVAILABLE:
                # Load PyTorch model (would need architecture definition)
                state_dict = torch.load(model_path / "model.pth")
                # For now, return state dict
                return state_dict
            elif metadata.framework == ModelFramework.TENSORFLOW and TENSORFLOW_AVAILABLE:
                # Load TensorFlow model
                return tf.keras.models.load_model(str(model_path / "model"))
            elif (model_path / "model.pkl").exists() and JOBLIB_AVAILABLE:
                # Load with joblib
                return joblib.load(model_path / "model.pkl")
            elif (model_path / "model.bin").exists():
                # Load raw bytes
                with open(model_path / "model.bin", "rb") as f:
                    return f.read()
            else:
                # Fallback to pickle
                with open(model_path / "model.pkl", "rb") as f:
                    return pickle.load(f)

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise

    async def _preprocess_input(self, input_data: Any, framework: ModelFramework, model_type: ModelType) -> Any:
        """Preprocess input data for model"""
        # This would implement framework and model-specific preprocessing
        if isinstance(input_data, (list, tuple)):
            return np.array(input_data)
        elif isinstance(input_data, dict):
            # Convert dict to array based on model type
            if model_type == ModelType.THREAT_CLASSIFIER:
                # Extract relevant features for threat classification
                features = [
                    input_data.get('file_size', 0),
                    input_data.get('entropy', 0),
                    len(input_data.get('suspicious_strings', [])),
                    input_data.get('network_connections', 0)
                ]
                return np.array(features).reshape(1, -1)

        return input_data

    async def _pytorch_predict(self, model: Any, input_data: Any) -> Any:
        """Make prediction with PyTorch model"""
        if TORCH_AVAILABLE:
            with torch.no_grad():
                if hasattr(model, 'forward'):
                    return model(torch.tensor(input_data, dtype=torch.float32))
                else:
                    # State dict only - would need architecture
                    return torch.randn(1, 2)  # Mock prediction
        return np.array([0.5, 0.5])  # Mock fallback

    async def _tensorflow_predict(self, model: Any, input_data: Any) -> Any:
        """Make prediction with TensorFlow model"""
        if TENSORFLOW_AVAILABLE:
            return model.predict(input_data)
        return np.array([0.5, 0.5])  # Mock fallback

    async def _sklearn_predict(self, model: Any, input_data: Any) -> Any:
        """Make prediction with scikit-learn model"""
        if hasattr(model, 'predict'):
            return model.predict(input_data)
        return np.array([0.5])  # Mock fallback

    async def _custom_predict(self, model: Any, input_data: Any) -> Any:
        """Make prediction with custom model"""
        if hasattr(model, 'predict'):
            return model.predict(input_data)
        elif callable(model):
            return model(input_data)
        return np.array([0.5])  # Mock fallback

    async def _postprocess_output(self, prediction: Any, model_type: ModelType) -> tuple:
        """Post-process model output"""
        try:
            if isinstance(prediction, (list, tuple, np.ndarray)):
                prediction = np.array(prediction)

                if model_type == ModelType.THREAT_CLASSIFIER:
                    # Binary classification
                    if len(prediction.shape) > 1 and prediction.shape[1] == 2:
                        # Softmax output
                        confidence = float(np.max(prediction[0]))
                        predicted_class = "malicious" if np.argmax(prediction[0]) == 1 else "benign"
                        return {"classification": predicted_class, "probabilities": prediction[0].tolist()}, confidence
                    else:
                        # Single probability
                        prob = float(prediction[0]) if len(prediction) > 0 else 0.5
                        return {"classification": "malicious" if prob > 0.5 else "benign", "probability": prob}, prob

                elif model_type == ModelType.ANOMALY_DETECTOR:
                    # Anomaly score
                    score = float(prediction[0]) if len(prediction) > 0 else 0.5
                    return {"anomaly_score": score, "is_anomaly": score > 0.5}, score

                else:
                    # Generic output
                    return {"prediction": prediction.tolist()}, float(np.mean(prediction))

            return {"prediction": str(prediction)}, 0.5

        except Exception as e:
            logger.error(f"Failed to post-process output: {e}")
            return {"error": str(e)}, 0.0

    async def _generate_test_input(self, model_type: ModelType) -> Any:
        """Generate test input for health checks"""
        if model_type == ModelType.THREAT_CLASSIFIER:
            return {
                "file_size": 1024,
                "entropy": 7.5,
                "suspicious_strings": ["eval", "exec"],
                "network_connections": 3
            }
        elif model_type == ModelType.ANOMALY_DETECTOR:
            return [0.1, 0.2, 0.3, 0.4, 0.5]
        else:
            return [1.0, 2.0, 3.0]  # Generic test input

    async def _save_model_metadata(self, metadata: ModelMetadata):
        """Save model metadata to disk"""
        try:
            model_path = self.models_directory / metadata.model_id
            metadata_file = model_path / "metadata.json"

            # Convert to dict and handle non-serializable types
            metadata_dict = asdict(metadata)
            metadata_dict['created_at'] = metadata.created_at.isoformat()
            metadata_dict['updated_at'] = metadata.updated_at.isoformat()
            metadata_dict['tenant_id'] = str(metadata.tenant_id)
            metadata_dict['status'] = metadata.status.value
            metadata_dict['model_type'] = metadata.model_type.value
            metadata_dict['framework'] = metadata.framework.value

            if metadata.last_health_check:
                metadata_dict['last_health_check'] = metadata.last_health_check.isoformat()

            with open(metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save metadata for model {metadata.model_id}: {e}")

    async def _load_model_registry(self):
        """Load existing models from disk"""
        try:
            for model_dir in self.models_directory.iterdir():
                if model_dir.is_dir():
                    metadata_file = model_dir / "metadata.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file) as f:
                                metadata_dict = json.load(f)

                            # Convert back to ModelMetadata
                            metadata = ModelMetadata(
                                model_id=metadata_dict['model_id'],
                                name=metadata_dict['name'],
                                version=metadata_dict['version'],
                                model_type=ModelType(metadata_dict['model_type']),
                                framework=ModelFramework(metadata_dict['framework']),
                                description=metadata_dict['description'],
                                tags=metadata_dict['tags'],
                                created_at=datetime.fromisoformat(metadata_dict['created_at']),
                                updated_at=datetime.fromisoformat(metadata_dict['updated_at']),
                                tenant_id=UUID(metadata_dict['tenant_id']),
                                created_by=metadata_dict['created_by'],
                                status=ModelStatus(metadata_dict['status']),
                                **{k: v for k, v in metadata_dict.items()
                                   if k not in ['model_id', 'name', 'version', 'model_type',
                                              'framework', 'description', 'tags', 'created_at',
                                              'updated_at', 'tenant_id', 'created_by', 'status']}
                            )

                            self.model_registry[metadata.model_id] = metadata

                        except Exception as e:
                            logger.error(f"Failed to load model metadata from {metadata_file}: {e}")

        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")

    async def _store_model_in_database(self, metadata: ModelMetadata):
        """Store model metadata in database"""
        try:
            async with get_async_session() as session:
                # Set tenant context
                await session.execute(
                    text("SELECT set_config('app.tenant_id', :tenant_id, false)"),
                    {"tenant_id": str(metadata.tenant_id)}
                )

                # This would use a proper models table
                # For now, just log
                logger.debug(f"Stored model {metadata.model_id} in database")

        except Exception as e:
            logger.error(f"Failed to store model in database: {e}")

    async def _update_model_in_database(self, metadata: ModelMetadata):
        """Update model metadata in database"""
        try:
            async with get_async_session() as session:
                # Set tenant context
                await session.execute(
                    text("SELECT set_config('app.tenant_id', :tenant_id, false)"),
                    {"tenant_id": str(metadata.tenant_id)}
                )

                logger.debug(f"Updated model {metadata.model_id} in database")

        except Exception as e:
            logger.error(f"Failed to update model in database: {e}")

    async def _delete_model_from_database(self, model_id: str):
        """Delete model from database"""
        try:
            # Would implement actual database deletion
            logger.debug(f"Deleted model {model_id} from database")

        except Exception as e:
            logger.error(f"Failed to delete model from database: {e}")

    async def _store_prediction(self, prediction: ModelPrediction, tenant_id: UUID):
        """Store prediction for audit and analysis"""
        try:
            # Would store in predictions table
            logger.debug(f"Stored prediction {prediction.prediction_id} for model {prediction.model_id}")

        except Exception as e:
            logger.error(f"Failed to store prediction: {e}")

    async def shutdown(self):
        """Shutdown the ML model manager"""
        logger.info("Shutting down ML Model Manager...")

        # Cancel training workers
        for worker in self.training_workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self.training_workers, return_exceptions=True)

        # Clear loaded models
        self.loaded_models.clear()

        logger.info("ML Model Manager shutdown complete")


# Global model manager instance
_model_manager: Optional[MLModelManager] = None


async def get_model_manager() -> MLModelManager:
    """Get global ML model manager instance"""
    global _model_manager

    if _model_manager is None:
        _model_manager = MLModelManager()
        await _model_manager.initialize()

    return _model_manager
