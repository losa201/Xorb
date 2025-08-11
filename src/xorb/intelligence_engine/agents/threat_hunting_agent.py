from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from xorb.intelligence_engine.agents.base_agent import UnifiedAgent

class ThreatHuntingAgent(UnifiedAgent):
    """
    AI agent specialized in proactive threat hunting and detection.
    Analyzes patterns across multiple data sources to identify potential threats.
    """
    
    def __init__(self, id: str, name: str, config: Dict[str, Any] = {}):
        """
        Initialize the threat hunting agent.
        
        Args:
            id: Unique identifier for the agent
            name: Name of the agent
            config: Configuration dictionary
        """
        super().__init__(
            id=id,
            name=name,
            agent_type="threat_hunting",
            capabilities=[
                "log_analysis",
                "pattern_recognition",
                "threat_intelligence",
                "anomaly_detection",
                "behavioral_analysis"
            ],
            created_at=datetime.now(),
            config=config
        )
        
        # Initialize ML components
        self.feature_scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(
            n_estimators=100,
            contamination=0.01,
            behaviour='new',
            random_state=42
        )
        self.cluster_model = DBSCAN(
            eps=0.5,
            min_samples=5
        )
        self.behavior_model = None  # Will be initialized during training
        self.threat_intel = {}  # Threat intelligence cache
        self.logger = logging.getLogger(__name__)
        
    def get_specialization(self) -> str:
        """
        Get the agent's specialization.
        
        Returns:
            Specialization description
        """
        return 'Proactive Threat Hunting and Detection'
        
    def train_behavior_model(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train the behavior model on historical data.
        
        Args:
            dataset: List of behavioral data points with metrics
            
        Returns:
            Training results and metrics
        """
        try:
            # Extract features from dataset
            features = []
            identifiers = []
            
            for record in dataset:
                # Extract relevant threat hunting features
                feature_vector = [
                    record.get('suspicious_activity_score', 0),
                    record.get('lateral_movement_likelihood', 0),
                    record.get('data_exfiltration_risk', 0),
                    record.get('command_and_control_indicators', 0),
                    record.get('malicious_behavior_probability', 0),
                    record.get('anomaly_score', 0)
                ]
                features.append(feature_vector)
                identifiers.append(record.get('event_id'))
                
            # Scale features
            scaled_features = self.feature_scaler.fit_transform(features)
            
            # Train behavior model
            self.behavior_model = IsolationForest(
                n_estimators=self.config.get('n_estimators', 100),
                contamination=self.config.get('contamination', 0.01),
                behaviour='new',
                random_state=self.config.get('random_state', 42)
            )
            self.behavior_model.fit(scaled_features)
            
            # Cluster analysis
            clusters = self.cluster_model.fit_predict(scaled_features)
            unique_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            
            # Update performance metrics
            self.update_performance_metrics({
                'training_samples': len(features),
                'clusters_identified': unique_clusters,
                'feature_dimensionality': len(feature_vector) if feature_vector else 0,
                'training_timestamp': datetime.now().isoformat()
            })
            
            return {
                'status': 'success',
                'clusters_identified': unique_clusters,
                'training_samples': len(features),
                'feature_dimensionality': len(feature_vector) if feature_vector else 0,
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
        if not self.behavior_model:
            return {
                'status': 'error',
                'error': 'Behavior model not initialized'
            }
            
        try:
            # Extract feature vector from current metrics
            feature_vector = [
                current_metrics.get('suspicious_activity_score', 0),
                current_metrics.get('lateral_movement_likelihood', 0),
                current_metrics.get('data_exfiltration_risk', 0),
                current_metrics.get('command_and_control_indicators', 0),
                current_metrics.get('malicious_behavior_probability', 0),
                current_metrics.get('anomaly_score', 0)
            ]
            
            # Scale features
            scaled_features = self.feature_scaler.transform([feature_vector])
            
            # Get anomaly score
            anomaly_score = -self.behavior_model.score_samples(scaled_features)[0]
            
            # Determine if anomaly
            is_anomaly = anomaly_score > self._calculate_anomaly_threshold()
            
            # Cluster analysis
            cluster = self.cluster_model.predict(scaled_features)[0]
            
            # Store for pattern learning
            if entity_id in self.performance_metrics.get('behavior_patterns', {}):
                self.performance_metrics['behavior_patterns'][entity_id].append({
                    'features': feature_vector,
                    'timestamp': datetime.now().isoformat(),
                    'anomaly_score': anomaly_score,
                    'is_anomaly': is_anomaly,
                    'cluster': int(cluster) if cluster != -1 else 'noise'
                })
            else:
                self.performance_metrics['behavior_patterns'][entity_id] = [{
                    'features': feature_vector,
                    'timestamp': datetime.now().isoformat(),
                    'anomaly_score': anomaly_score,
                    'is_anomaly': is_anomaly,
                    'cluster': int(cluster) if cluster != -1 else 'noise'
                }]
                
            return {
                'status': 'success',
                'entity_id': entity_id,
                'anomaly_score': float(anomaly_score),
                'is_anomaly': bool(is_anomaly),
                'risk_level': self._calculate_risk_level(anomaly_score),
                'cluster': int(cluster) if cluster != -1 else 'noise',
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def update_intelligence(self) -> Dict[str, Any]:
        """
        Update threat intelligence feeds.
        
        Returns:
            Update status and metadata
        """
        try:
            # Simulate threat intelligence update
            # In a real implementation, this would connect to threat intel feeds
            new_indicators = self._fetch_new_indicators()
            
            # Update internal threat intel cache
            for source, indicators in new_indicators.items():
                if source not in self.threat_intel:
                    self.threat_intel[source] = []
                self.threat_intel[source].extend(indicators)
                
            # Update performance metrics
            self.update_performance_metrics({
                'feeds_updated': len(new_indicators),
                'new_indicators': sum(len(v) for v in new_indicators.values()),
                'last_update': datetime.now().isoformat()
            })
            
            return {
                'status': 'success',
                'feeds_updated': len(new_indicators),
                'new_indicators': sum(len(v) for v in new_indicators.values()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Threat intel update failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def _fetch_new_indicators(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch new threat indicators from intelligence sources.
        
        Returns:
            Dictionary of threat indicators by source
        """
        # This is a placeholder for actual threat intel integration
        # In a real implementation, this would connect to threat intel APIs
        return {
            'mitre_att&ck': [{
                'id': 'T1021',
                'name': 'Remote Services',
                'description': 'Use of remote services to access systems',
                'severity': 'high',
                'timestamp': datetime.now().isoformat()
            }],
            'custom_feeds': [{
                'id': 'XORB-2023-001',
                'name': 'Suspicious PowerShell Usage',
                'description': 'Unusual PowerShell command patterns',
                'severity': 'medium',
                'timestamp': datetime.now().isoformat()
            }]
        }
        
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive threat hunting report.
        
        Returns:
            Report with statistics and findings
        """
        try:
            # Generate threat hunting report
            report = {
                'agent_name': self.name,
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_anomalies': len([p for patterns in self.performance_metrics.get('behavior_patterns', {}).values() for p in patterns if p.get('is_anomaly')]),
                    'high_risk_anomalies': len([p for patterns in self.performance_metrics.get('behavior_patterns', {}).values() for p in patterns if p.get('risk_level') == 'High' or p.get('risk_level') == 'Critical']),
                    'clusters_identified': len(set(p.get('cluster') for patterns in self.performance_metrics.get('behavior_patterns', {}).values() for p in patterns)),
                    'threat_intel_sources': len(self.threat_intel),
                    'last_update': self.performance_metrics.get('last_update', 'N/A')
                },
                'anomalies': [],  # Would be populated with actual anomalies
                'clusters': [],   # Would be populated with cluster analysis
                'threat_intel': {  # Summary of threat intelligence
                    source: len(indicators) for source, indicators in self.threat_intel.items()
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def get_threat_intel(self) -> Dict[str, Any]:
        """
        Get threat intelligence information.
        
        Returns:
            Dictionary containing threat intelligence
        """
        return {
            'threat_intel': self.threat_intel,
            'sources': list(self.threat_intel.keys()),
            'last_updated': self.performance_metrics.get('last_update', 'N/A')
        }