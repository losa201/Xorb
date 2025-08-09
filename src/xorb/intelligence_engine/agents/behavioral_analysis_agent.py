from typing import Dict, Any, List
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime

class BehavioralAnalysisAgent:
    """
    AI agent specialized in behavioral analysis for cybersecurity.
    Detects anomalies in user and system behavior patterns.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.01,
            behaviour='new',
            random_state=42
        )
        self.trained = False
        self.behavior_patterns = {}
        
    def train(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train the behavioral analysis model on historical data.
        
        Args:
            dataset: List of behavioral data points with metrics
            
        Returns:
            Training results and metrics
        """
        try:
            # Extract numerical features from dataset
            features = []
            identifiers = []
            
            for record in dataset:
                # Extract relevant behavioral metrics
                feature_vector = [
                    record.get('login_attempts', 0),
                    record.get('data_access_volume', 0),
                    record.get('system_resource_usage', 0),
                    record.get('access_pattern_complexity', 0),
                    record.get('geolocation_variance', 0),
                    record.get('time_based_activity', 0)
                ]
                features.append(feature_vector)
                identifiers.append(record.get('user_id') or record.get('system_id'))
                
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Train model
            self.model.fit(scaled_features)
            
            # Store patterns for reference
            for i, identifier in enumerate(identifiers):
                if identifier not in self.behavior_patterns:
                    self.behavior_patterns[identifier] = []
                self.behavior_patterns[identifier].append({
                    'features': features[i].tolist(),
                    'timestamp': datetime.now().isoformat()
                })
                
            self.trained = True
            
            return {
                'status': 'success',
                'anomaly_score_threshold': self._calculate_anomaly_threshold(),
                'patterns_analyzed': len(identifiers),
                'training_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
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
        if not self.trained:
            return {'status': 'error', 'error': 'Model not trained'}
            
        try:
            # Extract feature vector from current metrics
            feature_vector = [
                current_metrics.get('login_attempts', 0),
                current_metrics.get('data_access_volume', 0),
                current_metrics.get('system_resource_usage', 0),
                current_metrics.get('access_pattern_complexity', 0),
                current_metrics.get('geolocation_variance', 0),
                current_metrics.get('time_based_activity', 0)
            ]
            
            # Scale features
            scaled_features = self.scaler.transform([feature_vector])
            
            # Get anomaly score
            anomaly_score = -self.model.score_samples(scaled_features)[0]
            
            # Determine if anomaly
            is_anomaly = anomaly_score > self._calculate_anomaly_threshold()
            
            # Store for pattern learning
            if entity_id in self.behavior_patterns:
                self.behavior_patterns[entity_id].append({
                    'features': feature_vector,
                    'timestamp': datetime.now().isoformat(),
                    'anomaly_score': anomaly_score,
                    'is_anomaly': is_anomaly
                })
            else:
                self.behavior_patterns[entity_id] = [{
                    'features': feature_vector,
                    'timestamp': datetime.now().isoformat(),
                    'anomaly_score': anomaly_score,
                    'is_anomaly': is_anomaly
                }]
                
            return {
                'status': 'success',
                'entity_id': entity_id,
                'anomaly_score': float(anomaly_score),
                'is_anomaly': bool(is_anomaly),
                'risk_level': self._calculate_risk_level(anomaly_score),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def get_behavior_profile(self, entity_id: str) -> Dict[str, Any]:
        """
        Get the behavior profile for a specific entity.
        
        Args:
            entity_id: Identifier for the entity
            
        Returns:
            Complete behavior profile
        """
        if entity_id not in self.behavior_patterns:
            return {
                'status': 'error',
                'error': 'No behavior patterns found for entity'
            }
            
        return {
            'status': 'success',
            'entity_id': entity_id,
            'patterns': self.behavior_patterns[entity_id],
            'pattern_count': len(self.behavior_patterns[entity_id])
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
        
    def _calculate_risk_level(self, anomaly_score: float) -> str:
        """
        Calculate risk level based on anomaly score.
        
        Args:
            anomaly_score: Calculated anomaly score
            
        Returns:
            Risk level as string (Low/Medium/High/Critical)
        """
        if anomaly_score < 0.5:
            return 'Low'
        elif anomaly_score < 0.7:
            return 'Medium'
        elif anomaly_score < 0.9:
            return 'High'
        else:
            return 'Critical'
        
    def update_model(self, new_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update the model with new behavioral data.
        
        Args:
            new_data: New behavioral data points
            
        Returns:
            Update results
        """
        return self.train(new_data)
        
    def reset(self) -> None:
        """
        Reset the agent to its initial state.
        """
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.01,
            behaviour='new',
            random_state=42
        )
        self.scaler = StandardScaler()
        self.trained = False
        self.behavior_patterns = {}