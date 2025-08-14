from typing import Dict, List, Any, Optional, Type, TypeVar, Generic
from datetime import datetime
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod
import logging
from enum import Enum

# Define type variables for better type hinting
T = TypeVar('T', bound='BaseAgent')


class AgentCapability(str, Enum):
    """Enumeration of standard agent capabilities."""
    THREAT_INTELLIGENCE = "threat_intelligence"
    VULNERABILITY_ANALYSIS = "vulnerability_analysis"
    ATTACK_SIMULATION = "attack_simulation"
    NETWORK_MONITORING = "network_monitoring"
    BEHAVIOR_ANALYSIS = "behavior_analysis"
    INCIDENT_RESPONSE = "incident_response"
    MALWARE_ANALYSIS = "malware_analysis"
    LOG_ANALYSIS = "log_analysis"
    CONFIGURATION_AUDIT = "configuration_audit"
    PENETRATION_TESTING = "penetration_testing"
    COMPLIANCE_MONITORING = "compliance_monitoring"
    AI_ANALYSIS = "ai_analysis"
    DATA_EXFILTRATION_DETECTION = "data_exfiltration_detection"
    CLOUD_SECURITY = "cloud_security"
    ENDPOINT_SECURITY = "endpoint_security"


class AgentSpecialization(str, Enum):
    """Enumeration of agent specializations."""
    SECURITY_ANALYST = "security_analyst"
    THREAT_HUNTER = "threat_hunter"
    PENETRATION_TESTER = "penetration_tester"
    COMPLIANCE_OFFICER = "compliance_officer"
    NETWORK_DEFENDER = "network_defender"
    MALWARE_ANALYST = "malware_analyst"
    INCIDENT_RESPONDER = "incident_responder"
    CLOUD_SECURITY_SPECIALIST = "cloud_security_specialist"
    AI_SECURITY_ANALYST = "ai_security_analyst"


class AgentStatus(str, Enum):
    """Enumeration of agent status states."""
    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    TRAINING = "training"
    ANALYZING = "analyzing"
    SIMULATING = "simulating"
    RESPONDING = "responding"


class AgentMetrics(BaseModel):
    """Standard metrics for agent performance monitoring."""
    requests_handled: int = 0
    analysis_performed: int = 0
    threats_detected: int = 0
    false_positives: int = 0
    response_time_ms: float = 0.0
    accuracy_score: float = 0.0
    last_activity: datetime


class UnifiedAgentConfig(BaseModel):
    """Configuration for a unified agent."""
    name: str
    agent_type: str
    capabilities: List[AgentCapability]
    specialization: AgentSpecialization
    ml_config: Dict[str, Any] = {}


class UnifiedAgentContext:
    """Context manager for agent operations."""
    def __init__(self, agent: 'BaseAgent'):
        self.agent = agent
        self.logger = logging.getLogger(f"{__name__}.{agent.name}")

    def __enter__(self):
        self.agent.update_status(AgentStatus.ACTIVE)
        self.logger.debug(f"Agent {self.agent.name} entering active state")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.agent.update_status(AgentStatus.IDLE)
        self.agent.update_last_active()
        self.logger.debug(f"Agent {self.agent.name} returning to idle state")

        if exc_type:
            self.logger.error(f"Error in agent {self.agent.name}: {exc_val}", exc_info=True)
            self.agent.handle_error(exc_val)
            return True  # Suppress exception


class BaseAgent(ABC, BaseModel):
    """
    Abstract base class for all agents in the Xorb platform.
    Provides a standardized interface and common functionality.
    """
    id: str
    name: str
    agent_type: str
    capabilities: List[AgentCapability]
    specialization: AgentSpecialization
    status: AgentStatus = AgentStatus.IDLE
    created_at: datetime
    last_active: datetime
    metrics: AgentMetrics
    config: UnifiedAgentConfig

    # Machine learning components for adaptive behavior
    feature_scaler: Optional[Any] = None  # For normalizing input features
    anomaly_detector: Optional[Any] = None  # For detecting unusual patterns
    behavior_model: Optional[Any] = None  # For learning normal behavior patterns

    def update_last_active(self):
        """Update the last_active timestamp."""
        self.last_active = datetime.now()

    def update_status(self, status: AgentStatus):
        """Update the agent's status."""
        self.status = status
        self.update_last_active()

    def get_capabilities(self) -> List[AgentCapability]:
        """Get the agent's capabilities."""
        return self.capabilities

    def get_specialization(self) -> AgentSpecialization:
        """Get the agent's specialization."""
        return self.specialization

    def get_status(self) -> AgentStatus:
        """Get the agent's current status."""
        return self.status

    def get_metrics(self) -> AgentMetrics:
        """Get the agent's performance metrics."""
        return self.metrics

    def update_metrics(self, metrics_update: Dict[str, Any]) -> None:
        """Update performance metrics."""
        # Update last activity time
        metrics_update['last_activity'] = datetime.now()

        # Update metrics
        self.metrics = self.metrics.copy(update=metrics_update)

    def initialize_ml_components(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize machine learning components based on configuration.

        Args:
            config: Configuration dictionary containing ML parameters
        """
        # Use provided config or fall back to agent's config
        ml_config = config or self.config.ml_config

        # Initialize feature scaler
        self.feature_scaler = StandardScaler()

        # Initialize anomaly detector with configurable parameters
        self.anomaly_detector = IsolationForest(
            n_estimators=ml_config.get('n_estimators', 100),
            contamination=ml_config.get('contamination', 0.01),
            behaviour='new',
            random_state=ml_config.get('random_state', 42)
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

    def handle_error(self, error: Exception) -> None:
        """
        Handle errors that occur during agent operations.

        Args:
            error: The exception that occurred
        """
        # Base implementation - to be extended by subclasses
        self.update_status(AgentStatus.ERROR)
        self.update_metrics({
            'errors_handled': self.metrics.get('errors_handled', 0) + 1
        })

    @abstractmethod
    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's core functionality.

        Returns:
            Execution results
        """
        pass

    def __enter__(self: T) -> T:
        """Context manager entry point."""
        self.update_status(AgentStatus.ACTIVE)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit point."""
        self.update_status(AgentStatus.IDLE)
        self.update_last_active()

        if exc_type:
            self.handle_error(exc_val)
            return True  # Suppress exception

        return False  # Don't suppress exception

    @classmethod
    def create(cls: Type[T], config: UnifiedAgentConfig) -> T:
        """
        Factory method to create an agent instance.

        Args:
            config: Configuration for the agent

        Returns:
            Instantiated agent
        """
        return cls(
            id=str(uuid4()),
            name=config.name,
            agent_type=config.agent_type,
            capabilities=config.capabilities,
            specialization=config.specialization,
            status=AgentStatus.IDLE,
            created_at=datetime.now(),
            last_active=datetime.now(),
            metrics=AgentMetrics(
                last_activity=datetime.now()
            ),
            config=config
        )

    def get_context(self) -> UnifiedAgentContext:
        """Get a context manager for agent operations."""
        return UnifiedAgentContext(self)
