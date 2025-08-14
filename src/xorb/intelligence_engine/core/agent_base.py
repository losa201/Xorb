from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class UnifiedAgent(BaseModel):
    """
    Base class for all agents in the Xorb platform.
    Provides a standardized interface and common functionality.
    """
    id: str
    name: str
    agent_type: str
    capabilities: List[str]
    status: str = "idle"
    created_at: datetime
    last_active: datetime
    performance_metrics: Dict[str, Any] = {}
    config: Dict[str, Any] = {}

    # Machine learning components for adaptive behavior
    feature_scaler: Optional[Any] = None  # For normalizing input features
    anomaly_detector: Optional[Any] = None  # For detecting unusual patterns
    behavior_model: Optional[Any] = None  # For learning normal behavior patterns

    def update_last_active(self):
        """Update the last_active timestamp."""
        self.last_active = datetime.now()

    def update_performance_metrics(self, metrics: Dict[str, Any]):
        """Update performance metrics."""
        self.performance_metrics.update(metrics)

    def get_capabilities(self) -> List[str]:
        """Get the agent's capabilities."""
        return self.capabilities

    def get_status(self) -> str:
        """Get the agent's current status."""
        return self.status

    def initialize_ml_components(self, config: Dict[str, Any]) -> None:
        """
        Initialize machine learning components based on configuration.

        Args:
            config: Configuration dictionary containing ML parameters
        """
        # Initialize feature scaler
        self.feature_scaler = StandardScaler()

        # Initialize anomaly detector with configurable parameters
        self.anomaly_detector = IsolationForest(
            n_estimators=config.get('n_estimators', 100),
            contamination=config.get('contamination', 0.01),
            behaviour='new',
            random_state=config.get('random_state', 42)
        )

        # Initialize behavior model (could be different based on agent type)
        self.behavior_model = None  # To be initialized by subclasses

    def train_behavior_model(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train the behavior model on historical data.

        Args:
            dataset: List of behavioral data points with metrics

        Returns:
            Training results and metrics
        """
        # This implementation would be specific to each agent type
        # Should be overridden by subclasses
        return {
            'status': 'success',
            'message': 'Base behavior model trained successfully',
            'training_samples': len(dataset)
        }

    def analyze_behavior(self, entity_id: str, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze current behavior against established patterns.

        Args:
            entity_id: Identifier for the entity being analyzed
            current_metrics: Current behavioral metrics

        Returns:
            Analysis results including anomaly score
        """
        # This implementation would be specific to each agent type
        # Should be overridden by subclasses
        return {
            'status': 'success',
            'entity_id': entity_id,
            'anomaly_score': 0.0,
            'is_anomaly': False,
            'risk_level': 'Low',
            'analysis_timestamp': datetime.now().isoformat()
        }

    def detect_anomalies(self, features: List[float]) -> Dict[str, Any]:
        """
        Detect anomalies in the provided features.

        Args:
            features: List of numerical features to analyze

        Returns:
            Dictionary containing anomaly score and detection status
        """
        if not self.anomaly_detector:
            return {
                'status': 'error',
                'error': 'Anomaly detector not initialized'
            }

        try:
            # Scale features
            scaled_features = self.feature_scaler.transform([features])

            # Get anomaly score
            anomaly_score = -self.anomaly_detector.score_samples(scaled_features)[0]

            # Determine if anomaly
            is_anomaly = anomaly_score > self._calculate_anomaly_threshold()

            return {
                'status': 'success',
                'anomaly_score': float(anomaly_score),
                'is_anomaly': bool(is_anomaly)
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def _calculate_anomaly_threshold(self) -> float:
        """
        Calculate the anomaly score threshold based on training data.

        Returns:
            Threshold value for anomaly detection
        """
        # This would be more sophisticated in a real implementation
        # For now, using a simple threshold based on contamination parameter
        return 0.75

    def get_specialization(self) -> str:
        """
        Get the agent's specialization.

        Returns:
            Specialization description
        """
        return 'Base Agent Specialization'  # To be overridden by subclasses

    def get_threat_intel(self) -> Dict[str, Any]:
        """
        Get threat intelligence information.

        Returns:
            Dictionary containing threat intelligence
        """
        # Base implementation - to be extended by subclasses
        return {
            'threat_intel': [],
            'sources': [],
            'last_updated': datetime.now().isoformat()
        }

    def get_capabilities(self) -> List[str]:
        """
        Get the agent's capabilities.

        Returns:
            List of capabilities
        """
        return self.capabilities

    def get_status(self) -> str:
        """
        Get the agent's current status.

        Returns:
            Current status string
        """
        return self.status

    def update_intelligence(self) -> Dict[str, Any]:
        """
        Update threat intelligence feeds.

        Returns:
            Update status and metadata
        """
        # Base implementation - to be extended by subclasses
        return {
            'status': 'success',
            'feeds_updated': 0,
            'new_indicators': 0,
            'timestamp': datetime.now().isoformat()
        }

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report.

        Returns:
            Report with statistics and findings
        """
        # Base implementation - to be extended by subclasses
        return {
            'agent_name': self.name,
            'timestamp': datetime.now().isoformat(),
            'summary': 'Base agent report'
        }
