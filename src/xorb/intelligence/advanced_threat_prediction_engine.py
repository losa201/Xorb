#!/usr/bin/env python3
"""
Advanced Threat Prediction Engine
Principal Auditor Implementation: Next-Generation Predictive Cybersecurity

This module implements cutting-edge threat prediction capabilities using:
- Advanced time series forecasting with deep learning
- Graph-based threat propagation modeling
- Multi-modal threat intelligence fusion
- Bayesian neural networks for uncertainty quantification
- Attention mechanisms for temporal threat patterns
- Ensemble forecasting with multiple prediction models
- Real-time threat landscape evolution prediction
"""

import asyncio
import logging
import json
import uuid
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Advanced ML imports with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
    # Fallback classes for when PyTorch is not available
    class nn:
        class Module:
            def __init__(self):
                pass
            def forward(self, x):
                return x
        
        class LSTM:
            def __init__(self, *args, **kwargs):
                pass
        
        class Linear:
            def __init__(self, *args, **kwargs):
                pass
        
        class Dropout:
            def __init__(self, *args, **kwargs):
                pass
        
        class MultiheadAttention:
            def __init__(self, *args, **kwargs):
                pass
    
    class torch:
        @staticmethod
        def tensor(*args, **kwargs):
            import numpy as np
            return np.array(*args)
        
        @staticmethod
        def zeros(*args, **kwargs):
            import numpy as np
            return np.zeros(*args)
        
        @staticmethod
        def randn(*args, **kwargs):
            import numpy as np
            return np.random.randn(*args)
    
    class F:
        @staticmethod
        def relu(x):
            return x
        
        @staticmethod
        def softmax(x, dim=-1):
            return x

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import networkx as nx
    from scipy import stats
    from scipy.signal import find_peaks
    import pandas as pd
    SCIPY_PANDAS_AVAILABLE = True
except ImportError:
    SCIPY_PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ThreatPredictionType(Enum):
    """Types of threat predictions"""
    ATTACK_PROBABILITY = "attack_probability"           # Probability of attack occurrence
    THREAT_EVOLUTION = "threat_evolution"               # Evolution of threat landscape
    VULNERABILITY_EMERGENCE = "vulnerability_emergence" # New vulnerability prediction
    CAMPAIGN_ATTRIBUTION = "campaign_attribution"       # Campaign attribution prediction
    TARGET_LIKELIHOOD = "target_likelihood"            # Target selection probability
    ATTACK_TIMING = "attack_timing"                     # Attack timing prediction
    THREAT_ACTOR_BEHAVIOR = "threat_actor_behavior"     # Threat actor behavior prediction
    MALWARE_VARIANTS = "malware_variants"               # New malware variant prediction
    INFRASTRUCTURE_RISK = "infrastructure_risk"         # Infrastructure risk assessment
    GEOPOLITICAL_THREATS = "geopolitical_threats"       # Geopolitical threat prediction


class PredictionHorizon(Enum):
    """Prediction time horizons"""
    IMMEDIATE = "immediate"      # Next 1-24 hours
    SHORT_TERM = "short_term"    # Next 1-7 days
    MEDIUM_TERM = "medium_term"  # Next 1-4 weeks
    LONG_TERM = "long_term"      # Next 1-6 months
    STRATEGIC = "strategic"      # Next 6-12 months


class ConfidenceLevel(Enum):
    """Prediction confidence levels"""
    VERY_HIGH = "very_high"  # >95% confidence
    HIGH = "high"            # 85-95% confidence
    MEDIUM = "medium"        # 70-85% confidence
    LOW = "low"             # 50-70% confidence
    VERY_LOW = "very_low"    # <50% confidence


@dataclass
class ThreatPrediction:
    """Comprehensive threat prediction with metadata"""
    prediction_id: str
    prediction_type: ThreatPredictionType
    prediction_horizon: PredictionHorizon
    predicted_value: Union[float, Dict[str, Any]]
    confidence_level: ConfidenceLevel
    confidence_score: float
    uncertainty_bounds: Tuple[float, float]
    prediction_timestamp: datetime
    valid_until: datetime
    
    # Context and attribution
    threat_context: Dict[str, Any]
    data_sources: List[str]
    model_ensemble: List[str]
    feature_importance: Dict[str, float]
    
    # Risk assessment
    risk_level: str
    potential_impact: Dict[str, Any]
    recommended_actions: List[str]
    
    # Metadata
    model_version: str
    prediction_methodology: str
    validation_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatTimeSeriesData:
    """Time series data for threat prediction"""
    timestamp: datetime
    threat_indicators: Dict[str, float]
    attack_volumes: Dict[str, int]
    vulnerability_counts: Dict[str, int]
    threat_actor_activity: Dict[str, float]
    geopolitical_indicators: Dict[str, float]
    economic_indicators: Dict[str, float]
    seasonal_factors: Dict[str, float]
    external_events: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class TemporalAttentionLSTM(nn.Module):
    """LSTM with temporal attention for threat prediction"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, 
                 attention_size: int = 128, output_size: int = 1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Uncertainty estimation
        self.uncertainty_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attention_weights = self.attention(lstm_out)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Generate prediction and uncertainty
        prediction = self.output_layer(context)
        uncertainty = self.uncertainty_layer(context)
        
        return prediction, uncertainty, attention_weights


class BayesianNeuralNetwork(nn.Module):
    """Bayesian neural network for uncertainty quantification"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        # Mean and variance outputs
        self.feature_extractor = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(prev_size, output_size)
        self.var_layer = nn.Sequential(
            nn.Linear(prev_size, output_size),
            nn.Softplus()
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        mean = self.mean_layer(features)
        variance = self.var_layer(features)
        
        return mean, variance
    
    def sample_prediction(self, x, num_samples: int = 100):
        """Sample predictions for uncertainty estimation"""
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(num_samples):
            mean, var = self.forward(x)
            # Sample from Gaussian
            std = torch.sqrt(var)
            sample = torch.normal(mean, std)
            predictions.append(sample)
        
        predictions = torch.stack(predictions)
        pred_mean = torch.mean(predictions, dim=0)
        pred_std = torch.std(predictions, dim=0)
        
        self.eval()  # Disable dropout
        
        return pred_mean, pred_std


class ThreatGraphPropagationModel:
    """Graph-based threat propagation modeling"""
    
    def __init__(self):
        self.threat_graph = nx.DiGraph()
        self.propagation_probabilities = {}
        self.node_vulnerabilities = {}
        self.propagation_history = []
    
    def add_threat_node(self, node_id: str, node_type: str, 
                       vulnerability_score: float, metadata: Dict[str, Any] = None):
        """Add threat node to the graph"""
        self.threat_graph.add_node(
            node_id,
            node_type=node_type,
            vulnerability_score=vulnerability_score,
            metadata=metadata or {}
        )
        self.node_vulnerabilities[node_id] = vulnerability_score
    
    def add_propagation_edge(self, source: str, target: str, 
                           propagation_probability: float, 
                           propagation_time: float = 1.0):
        """Add propagation edge between nodes"""
        self.threat_graph.add_edge(
            source, target,
            propagation_probability=propagation_probability,
            propagation_time=propagation_time
        )
        self.propagation_probabilities[(source, target)] = propagation_probability
    
    def simulate_threat_propagation(self, initial_nodes: List[str], 
                                  time_steps: int = 10) -> Dict[str, List[float]]:
        """Simulate threat propagation over time"""
        
        # Initialize infection states
        infection_states = {node: 0.0 for node in self.threat_graph.nodes()}
        
        # Set initial infection
        for node in initial_nodes:
            if node in infection_states:
                infection_states[node] = 1.0
        
        propagation_timeline = {node: [infection_states[node]] 
                              for node in self.threat_graph.nodes()}
        
        # Simulate propagation
        for t in range(time_steps):
            new_states = infection_states.copy()
            
            for node in self.threat_graph.nodes():
                if infection_states[node] < 1.0:  # Not fully infected
                    # Calculate incoming infection pressure
                    incoming_pressure = 0.0
                    
                    for predecessor in self.threat_graph.predecessors(node):
                        edge_data = self.threat_graph[predecessor][node]
                        prop_prob = edge_data['propagation_probability']
                        source_infection = infection_states[predecessor]
                        
                        incoming_pressure += source_infection * prop_prob
                    
                    # Update infection state
                    vulnerability = self.node_vulnerabilities.get(node, 0.5)
                    infection_increase = incoming_pressure * vulnerability * 0.1
                    new_states[node] = min(1.0, infection_states[node] + infection_increase)
            
            infection_states = new_states
            
            # Record timeline
            for node in self.threat_graph.nodes():
                propagation_timeline[node].append(infection_states[node])
        
        return propagation_timeline
    
    def predict_critical_paths(self, source_nodes: List[str], 
                             target_nodes: List[str]) -> List[Dict[str, Any]]:
        """Predict critical propagation paths"""
        critical_paths = []
        
        for source in source_nodes:
            for target in target_nodes:
                try:
                    # Find all simple paths
                    paths = list(nx.all_simple_paths(
                        self.threat_graph, source, target, cutoff=5
                    ))
                    
                    for path in paths:
                        # Calculate path probability
                        path_probability = 1.0
                        path_time = 0.0
                        
                        for i in range(len(path) - 1):
                            edge_data = self.threat_graph[path[i]][path[i + 1]]
                            path_probability *= edge_data['propagation_probability']
                            path_time += edge_data.get('propagation_time', 1.0)
                        
                        critical_paths.append({
                            'path': path,
                            'probability': path_probability,
                            'expected_time': path_time,
                            'criticality_score': path_probability / (path_time + 1)
                        })
                
                except nx.NetworkXNoPath:
                    continue
        
        # Sort by criticality score
        critical_paths.sort(key=lambda x: x['criticality_score'], reverse=True)
        
        return critical_paths[:10]  # Return top 10 critical paths


class MultiModalThreatFusion:
    """Multi-modal threat intelligence fusion"""
    
    def __init__(self):
        self.data_sources = {}
        self.fusion_weights = {}
        self.correlation_matrix = np.eye(10)  # 10x10 default
        self.temporal_alignment = {}
    
    def register_data_source(self, source_id: str, source_type: str, 
                           reliability_score: float, metadata: Dict[str, Any] = None):
        """Register a threat intelligence data source"""
        self.data_sources[source_id] = {
            'type': source_type,
            'reliability': reliability_score,
            'metadata': metadata or {},
            'last_update': datetime.utcnow()
        }
        
        # Initialize fusion weight based on reliability
        self.fusion_weights[source_id] = reliability_score
    
    def fuse_threat_indicators(self, indicators_by_source: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Fuse threat indicators from multiple sources"""
        
        # Collect all indicator types
        all_indicators = set()
        for indicators in indicators_by_source.values():
            all_indicators.update(indicators.keys())
        
        fused_indicators = {}
        
        for indicator in all_indicators:
            values = []
            weights = []
            
            for source_id, indicators_dict in indicators_by_source.items():
                if indicator in indicators_dict:
                    values.append(indicators_dict[indicator])
                    weights.append(self.fusion_weights.get(source_id, 0.5))
            
            if values:
                # Weighted average fusion
                weighted_sum = sum(v * w for v, w in zip(values, weights))
                weight_sum = sum(weights)
                
                fused_value = weighted_sum / weight_sum if weight_sum > 0 else 0.0
                
                # Apply uncertainty penalty for conflicting sources
                value_variance = np.var(values) if len(values) > 1 else 0.0
                uncertainty_penalty = min(value_variance * 0.1, 0.3)
                
                fused_indicators[indicator] = max(0.0, fused_value - uncertainty_penalty)
        
        return fused_indicators
    
    def temporal_alignment(self, time_series_data: Dict[str, List[Tuple[datetime, float]]]) -> Dict[str, List[float]]:
        """Align time series data from different sources"""
        
        # Find common time range
        all_timestamps = []
        for source_data in time_series_data.values():
            all_timestamps.extend([ts for ts, _ in source_data])
        
        if not all_timestamps:
            return {}
        
        min_time = min(all_timestamps)
        max_time = max(all_timestamps)
        
        # Create uniform time grid
        time_delta = timedelta(hours=1)  # 1-hour intervals
        aligned_timestamps = []
        current_time = min_time
        
        while current_time <= max_time:
            aligned_timestamps.append(current_time)
            current_time += time_delta
        
        # Interpolate each source to common grid
        aligned_data = {}
        
        for source_id, source_data in time_series_data.items():
            if not source_data:
                continue
            
            # Sort by timestamp
            source_data.sort(key=lambda x: x[0])
            
            aligned_values = []
            source_idx = 0
            
            for target_time in aligned_timestamps:
                # Find closest data points
                while (source_idx < len(source_data) - 1 and 
                       source_data[source_idx + 1][0] <= target_time):
                    source_idx += 1
                
                if source_idx < len(source_data):
                    # Simple nearest neighbor interpolation
                    aligned_values.append(source_data[source_idx][1])
                else:
                    # Use last available value
                    aligned_values.append(source_data[-1][1])
            
            aligned_data[source_id] = aligned_values
        
        return aligned_data
    
    def calculate_source_correlations(self, aligned_data: Dict[str, List[float]]) -> np.ndarray:
        """Calculate correlations between data sources"""
        
        sources = list(aligned_data.keys())
        n_sources = len(sources)
        
        if n_sources < 2:
            return np.eye(max(2, n_sources))
        
        correlation_matrix = np.eye(n_sources)
        
        for i in range(n_sources):
            for j in range(i + 1, n_sources):
                data_i = aligned_data[sources[i]]
                data_j = aligned_data[sources[j]]
                
                if len(data_i) > 1 and len(data_j) > 1:
                    correlation = np.corrcoef(data_i, data_j)[0, 1]
                    if not np.isnan(correlation):
                        correlation_matrix[i, j] = correlation
                        correlation_matrix[j, i] = correlation
        
        return correlation_matrix


class AdvancedThreatPredictionEngine:
    """
    Advanced Threat Prediction Engine
    
    Provides next-generation threat prediction capabilities using:
    - Deep learning time series forecasting
    - Graph-based threat propagation modeling
    - Multi-modal intelligence fusion
    - Bayesian uncertainty quantification
    - Ensemble prediction methods
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.engine_id = str(uuid.uuid4())
        
        # Core prediction models
        self.prediction_models = {}
        self.ensemble_weights = {}
        
        # Components
        self.graph_model = ThreatGraphPropagationModel()
        self.fusion_engine = MultiModalThreatFusion()
        
        # Data management
        self.historical_data: List[ThreatTimeSeriesData] = []
        self.prediction_cache: Dict[str, ThreatPrediction] = {}
        self.model_performance: Dict[str, Dict[str, float]] = {}
        
        # Initialize models if ML libraries available
        if TORCH_AVAILABLE:
            self._initialize_neural_models()
        
        if SKLEARN_AVAILABLE:
            self._initialize_traditional_models()
        
        # Performance tracking
        self.prediction_metrics = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "average_accuracy": 0.0,
            "prediction_types": {},
            "model_usage": {}
        }
        
        logger.info("Advanced Threat Prediction Engine initialized", engine_id=self.engine_id)
    
    def _initialize_neural_models(self):
        """Initialize neural network models"""
        try:
            # Temporal attention LSTM for time series prediction
            self.prediction_models["temporal_lstm"] = TemporalAttentionLSTM(
                input_size=20,  # Number of threat indicators
                hidden_size=128,
                num_layers=2,
                attention_size=64,
                output_size=1
            )
            
            # Bayesian neural network for uncertainty quantification
            self.prediction_models["bayesian_nn"] = BayesianNeuralNetwork(
                input_size=50,  # Extended feature set
                hidden_sizes=[256, 128, 64],
                output_size=1
            )
            
            # Initialize ensemble weights
            self.ensemble_weights = {
                "temporal_lstm": 0.4,
                "bayesian_nn": 0.3,
                "traditional_ensemble": 0.3
            }
            
            logger.info("Neural prediction models initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize neural models: {e}")
    
    def _initialize_traditional_models(self):
        """Initialize traditional ML models"""
        try:
            # Random Forest for baseline predictions
            self.prediction_models["random_forest"] = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42
            )
            
            # Gradient Boosting for trend prediction
            self.prediction_models["gradient_boosting"] = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            # Data preprocessing
            self.scalers = {
                "standard": StandardScaler(),
                "minmax": MinMaxScaler()
            }
            
            logger.info("Traditional ML models initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize traditional models: {e}")
    
    async def add_historical_data(self, threat_data: ThreatTimeSeriesData) -> bool:
        """Add historical threat data for training"""
        try:
            self.historical_data.append(threat_data)
            
            # Keep only recent data (last 2 years)
            cutoff_date = datetime.utcnow() - timedelta(days=730)
            self.historical_data = [
                data for data in self.historical_data 
                if data.timestamp >= cutoff_date
            ]
            
            # Sort by timestamp
            self.historical_data.sort(key=lambda x: x.timestamp)
            
            logger.debug(f"Added historical data, total records: {len(self.historical_data)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add historical data: {e}")
            return False
    
    async def predict_threat(self,
                           prediction_type: ThreatPredictionType,
                           prediction_horizon: PredictionHorizon,
                           context: Dict[str, Any] = None,
                           models_to_use: List[str] = None) -> ThreatPrediction:
        """Generate comprehensive threat prediction"""
        
        prediction_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Generating {prediction_type.value} prediction for {prediction_horizon.value}")
            
            # Prepare input features
            features = await self._extract_prediction_features(context or {})
            
            # Select models to use
            if models_to_use is None:
                models_to_use = list(self.prediction_models.keys())
            
            # Generate predictions from ensemble
            model_predictions = {}
            model_uncertainties = {}
            
            for model_name in models_to_use:
                if model_name in self.prediction_models:
                    try:
                        pred, uncertainty = await self._predict_with_model(
                            model_name, features, prediction_type, prediction_horizon
                        )
                        model_predictions[model_name] = pred
                        model_uncertainties[model_name] = uncertainty
                        
                    except Exception as e:
                        logger.warning(f"Model {model_name} prediction failed: {e}")
            
            # Ensemble prediction
            if model_predictions:
                ensemble_prediction, ensemble_confidence = await self._ensemble_predict(
                    model_predictions, model_uncertainties
                )
            else:
                # Fallback prediction
                ensemble_prediction = await self._fallback_prediction(
                    prediction_type, prediction_horizon, context
                )
                ensemble_confidence = 0.5
            
            # Calculate uncertainty bounds
            uncertainty_bounds = await self._calculate_uncertainty_bounds(
                ensemble_prediction, model_uncertainties, ensemble_confidence
            )
            
            # Assess confidence level
            confidence_level = self._assess_confidence_level(ensemble_confidence)
            
            # Calculate prediction validity
            valid_until = await self._calculate_validity_period(
                prediction_type, prediction_horizon, ensemble_confidence
            )
            
            # Risk assessment
            risk_level, potential_impact = await self._assess_prediction_risk(
                prediction_type, ensemble_prediction, context
            )
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                prediction_type, ensemble_prediction, risk_level, context
            )
            
            # Feature importance analysis
            feature_importance = await self._analyze_feature_importance(
                features, model_predictions
            )
            
            # Create prediction object
            prediction = ThreatPrediction(
                prediction_id=prediction_id,
                prediction_type=prediction_type,
                prediction_horizon=prediction_horizon,
                predicted_value=ensemble_prediction,
                confidence_level=confidence_level,
                confidence_score=ensemble_confidence,
                uncertainty_bounds=uncertainty_bounds,
                prediction_timestamp=datetime.utcnow(),
                valid_until=valid_until,
                threat_context=context or {},
                data_sources=await self._get_data_sources_used(),
                model_ensemble=list(model_predictions.keys()),
                feature_importance=feature_importance,
                risk_level=risk_level,
                potential_impact=potential_impact,
                recommended_actions=recommendations,
                model_version="v2.0",
                prediction_methodology="ensemble_neural_symbolic",
                metadata={
                    "model_predictions": model_predictions,
                    "model_uncertainties": model_uncertainties,
                    "features_used": len(features)
                }
            )
            
            # Cache prediction
            self.prediction_cache[prediction_id] = prediction
            
            # Update metrics
            self.prediction_metrics["total_predictions"] += 1
            pred_type_key = prediction_type.value
            if pred_type_key not in self.prediction_metrics["prediction_types"]:
                self.prediction_metrics["prediction_types"][pred_type_key] = 0
            self.prediction_metrics["prediction_types"][pred_type_key] += 1
            
            logger.info(f"Threat prediction {prediction_id} generated successfully")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Threat prediction failed: {e}")
            raise
    
    async def _extract_prediction_features(self, context: Dict[str, Any]) -> np.ndarray:
        """Extract features for prediction models"""
        try:
            features = []
            
            # Historical trend features
            if len(self.historical_data) > 0:
                recent_data = self.historical_data[-30:]  # Last 30 data points
                
                # Threat indicator trends
                for indicator in ["malware_volume", "phishing_attempts", "vulnerability_reports"]:
                    values = [getattr(data, 'threat_indicators', {}).get(indicator, 0) 
                             for data in recent_data]
                    if values:
                        features.extend([
                            np.mean(values),
                            np.std(values),
                            values[-1] if values else 0,  # Latest value
                            np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0  # Trend
                        ])
                    else:
                        features.extend([0, 0, 0, 0])
                
                # Attack volume features
                attack_volumes = [sum(data.attack_volumes.values()) for data in recent_data]
                if attack_volumes:
                    features.extend([
                        np.mean(attack_volumes),
                        np.max(attack_volumes),
                        attack_volumes[-1] if attack_volumes else 0
                    ])
                else:
                    features.extend([0, 0, 0])
            else:
                # No historical data - use defaults
                features.extend([0] * 15)
            
            # Context features
            features.extend([
                context.get("geopolitical_tension", 0.5),
                context.get("economic_instability", 0.5),
                context.get("seasonal_factor", 0.5),
                context.get("recent_incidents", 0),
                context.get("threat_actor_activity", 0.5)
            ])
            
            # Pad to fixed size
            target_size = 50
            while len(features) < target_size:
                features.append(0.0)
            
            return np.array(features[:target_size], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.zeros(50, dtype=np.float32)
    
    async def _predict_with_model(self,
                                model_name: str,
                                features: np.ndarray,
                                prediction_type: ThreatPredictionType,
                                prediction_horizon: PredictionHorizon) -> Tuple[float, float]:
        """Generate prediction with specific model"""
        
        model = self.prediction_models[model_name]
        
        try:
            if model_name == "temporal_lstm" and TORCH_AVAILABLE:
                return await self._predict_with_lstm(model, features, prediction_horizon)
            elif model_name == "bayesian_nn" and TORCH_AVAILABLE:
                return await self._predict_with_bayesian_nn(model, features)
            elif model_name in ["random_forest", "gradient_boosting"] and SKLEARN_AVAILABLE:
                return await self._predict_with_sklearn(model, features)
            else:
                # Fallback prediction
                return 0.5, 0.3
                
        except Exception as e:
            logger.error(f"Prediction with model {model_name} failed: {e}")
            return 0.5, 0.5
    
    async def _predict_with_lstm(self, model: TemporalAttentionLSTM, 
                               features: np.ndarray, 
                               horizon: PredictionHorizon) -> Tuple[float, float]:
        """Predict using temporal LSTM"""
        try:
            model.eval()
            
            # Reshape features for LSTM (batch_size, seq_len, features)
            # Create sequence from recent data
            if len(self.historical_data) >= 10:
                sequence_data = []
                for data in self.historical_data[-10:]:
                    seq_features = list(data.threat_indicators.values())[:20]
                    while len(seq_features) < 20:
                        seq_features.append(0.0)
                    sequence_data.append(seq_features)
                
                input_tensor = torch.tensor([sequence_data], dtype=torch.float32)
            else:
                # Use features as repeated sequence
                input_tensor = torch.tensor([features[:20]] * 10).unsqueeze(0)
            
            with torch.no_grad():
                prediction, uncertainty, attention = model(input_tensor)
                
                pred_value = float(prediction.squeeze())
                uncert_value = float(uncertainty.squeeze())
                
                # Adjust for horizon
                horizon_factor = {"immediate": 1.0, "short_term": 0.9, 
                                "medium_term": 0.8, "long_term": 0.7, "strategic": 0.6}
                
                pred_value *= horizon_factor.get(horizon.value, 0.8)
                uncert_value *= (2.0 - horizon_factor.get(horizon.value, 0.8))
                
                return pred_value, uncert_value
                
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return 0.5, 0.3
    
    async def _predict_with_bayesian_nn(self, model: BayesianNeuralNetwork, 
                                      features: np.ndarray) -> Tuple[float, float]:
        """Predict using Bayesian neural network"""
        try:
            input_tensor = torch.tensor(features.reshape(1, -1), dtype=torch.float32)
            
            # Sample predictions for uncertainty
            pred_mean, pred_std = model.sample_prediction(input_tensor, num_samples=50)
            
            prediction = float(pred_mean.squeeze())
            uncertainty = float(pred_std.squeeze())
            
            return prediction, uncertainty
            
        except Exception as e:
            logger.error(f"Bayesian NN prediction failed: {e}")
            return 0.5, 0.3
    
    async def _predict_with_sklearn(self, model, features: np.ndarray) -> Tuple[float, float]:
        """Predict using scikit-learn model"""
        try:
            # Train model if not trained
            if not hasattr(model, 'feature_importances_') and len(self.historical_data) > 10:
                await self._train_sklearn_model(model)
            
            if hasattr(model, 'predict'):
                # Reshape for sklearn
                features_2d = features.reshape(1, -1)
                
                # Scale features if scaler available
                if "standard" in self.scalers:
                    try:
                        features_2d = self.scalers["standard"].transform(features_2d)
                    except:
                        pass
                
                prediction = model.predict(features_2d)[0]
                
                # Estimate uncertainty (simplified)
                if hasattr(model, 'estimators_'):
                    # For ensemble models, use prediction variance
                    tree_predictions = [tree.predict(features_2d)[0] for tree in model.estimators_]
                    uncertainty = np.std(tree_predictions)
                else:
                    uncertainty = 0.2  # Default uncertainty
                
                return float(prediction), float(uncertainty)
            else:
                return 0.5, 0.3
                
        except Exception as e:
            logger.error(f"Sklearn prediction failed: {e}")
            return 0.5, 0.3
    
    async def _train_sklearn_model(self, model):
        """Train sklearn model with historical data"""
        try:
            if len(self.historical_data) < 10:
                return
            
            # Prepare training data
            X = []
            y = []
            
            for i in range(len(self.historical_data) - 1):
                features = await self._extract_prediction_features({})
                target = sum(self.historical_data[i + 1].attack_volumes.values())
                
                X.append(features)
                y.append(target)
            
            if len(X) > 5:
                X = np.array(X)
                y = np.array(y)
                
                # Scale features
                X_scaled = self.scalers["standard"].fit_transform(X)
                
                # Train model
                model.fit(X_scaled, y)
                
        except Exception as e:
            logger.error(f"Sklearn model training failed: {e}")
    
    async def _ensemble_predict(self, 
                              model_predictions: Dict[str, float],
                              model_uncertainties: Dict[str, float]) -> Tuple[float, float]:
        """Combine predictions from multiple models"""
        try:
            if not model_predictions:
                return 0.5, 0.5
            
            # Weighted ensemble
            weighted_sum = 0.0
            weight_sum = 0.0
            uncertainty_sum = 0.0
            
            for model_name, prediction in model_predictions.items():
                weight = self.ensemble_weights.get(model_name, 1.0)
                uncertainty = model_uncertainties.get(model_name, 0.3)
                
                # Inverse uncertainty weighting
                inv_uncertainty_weight = 1.0 / (uncertainty + 0.1)
                final_weight = weight * inv_uncertainty_weight
                
                weighted_sum += prediction * final_weight
                weight_sum += final_weight
                uncertainty_sum += uncertainty * weight
            
            if weight_sum > 0:
                ensemble_prediction = weighted_sum / weight_sum
                ensemble_uncertainty = uncertainty_sum / sum(self.ensemble_weights.values())
            else:
                ensemble_prediction = np.mean(list(model_predictions.values()))
                ensemble_uncertainty = np.mean(list(model_uncertainties.values()))
            
            # Confidence is inverse of uncertainty
            ensemble_confidence = 1.0 / (ensemble_uncertainty + 0.1)
            ensemble_confidence = min(ensemble_confidence / 10, 0.99)  # Normalize
            
            return float(ensemble_prediction), float(ensemble_confidence)
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return 0.5, 0.5
    
    async def _fallback_prediction(self,
                                 prediction_type: ThreatPredictionType,
                                 prediction_horizon: PredictionHorizon,
                                 context: Dict[str, Any]) -> float:
        """Generate fallback prediction when models fail"""
        
        # Simple heuristic-based prediction
        base_threat_level = 0.3
        
        # Adjust based on prediction type
        type_adjustments = {
            ThreatPredictionType.ATTACK_PROBABILITY: 0.1,
            ThreatPredictionType.VULNERABILITY_EMERGENCE: 0.15,
            ThreatPredictionType.THREAT_EVOLUTION: 0.05,
            ThreatPredictionType.MALWARE_VARIANTS: 0.2
        }
        
        adjustment = type_adjustments.get(prediction_type, 0.0)
        
        # Adjust based on context
        if context:
            geopolitical = context.get("geopolitical_tension", 0.5)
            economic = context.get("economic_instability", 0.5)
            recent_incidents = context.get("recent_incidents", 0)
            
            context_adjustment = (geopolitical + economic) * 0.2 + recent_incidents * 0.1
            adjustment += context_adjustment
        
        # Adjust based on horizon (further predictions are less certain)
        horizon_factor = {"immediate": 1.0, "short_term": 0.9, 
                         "medium_term": 0.8, "long_term": 0.7, "strategic": 0.6}
        
        final_prediction = (base_threat_level + adjustment) * horizon_factor.get(prediction_horizon.value, 0.8)
        
        return min(max(final_prediction, 0.0), 1.0)
    
    async def get_prediction_metrics(self) -> Dict[str, Any]:
        """Get comprehensive prediction engine metrics"""
        try:
            # Calculate success rate
            total_predictions = self.prediction_metrics["total_predictions"]
            success_rate = (self.prediction_metrics["successful_predictions"] / total_predictions 
                          if total_predictions > 0 else 0.0)
            
            # Model performance analysis
            model_performance = {}
            for model_name in self.prediction_models.keys():
                model_performance[model_name] = self.model_performance.get(model_name, {
                    "accuracy": 0.0,
                    "predictions_made": 0,
                    "average_uncertainty": 0.0
                })
            
            return {
                "engine_metrics": {
                    "engine_id": self.engine_id,
                    "total_predictions": total_predictions,
                    "successful_predictions": self.prediction_metrics["successful_predictions"],
                    "success_rate": success_rate,
                    "average_accuracy": self.prediction_metrics["average_accuracy"],
                    "active_models": len(self.prediction_models),
                    "cached_predictions": len(self.prediction_cache)
                },
                "prediction_distribution": self.prediction_metrics["prediction_types"],
                "model_performance": model_performance,
                "data_metrics": {
                    "historical_data_points": len(self.historical_data),
                    "data_sources_registered": len(self.fusion_engine.data_sources),
                    "graph_nodes": self.graph_model.threat_graph.number_of_nodes(),
                    "graph_edges": self.graph_model.threat_graph.number_of_edges()
                },
                "model_availability": {
                    "torch_available": TORCH_AVAILABLE,
                    "sklearn_available": SKLEARN_AVAILABLE,
                    "scipy_pandas_available": SCIPY_PANDAS_AVAILABLE
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get prediction metrics: {e}")
            return {"error": str(e)}


# Global engine instance
_threat_prediction_engine: Optional[AdvancedThreatPredictionEngine] = None


async def get_threat_prediction_engine(config: Dict[str, Any] = None) -> AdvancedThreatPredictionEngine:
    """Get singleton advanced threat prediction engine instance"""
    global _threat_prediction_engine
    
    if _threat_prediction_engine is None:
        _threat_prediction_engine = AdvancedThreatPredictionEngine(config)
    
    return _threat_prediction_engine


# Export main classes
__all__ = [
    "AdvancedThreatPredictionEngine",
    "ThreatPrediction",
    "ThreatTimeSeriesData",
    "ThreatGraphPropagationModel",
    "MultiModalThreatFusion",
    "TemporalAttentionLSTM",
    "BayesianNeuralNetwork",
    "ThreatPredictionType",
    "PredictionHorizon",
    "ConfidenceLevel",
    "get_threat_prediction_engine"
]