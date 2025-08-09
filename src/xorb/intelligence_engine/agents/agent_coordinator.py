from typing import Dict, List, Any, Optional
from src.xorb.intelligence_engine.agents.threat_hunting_agent import ThreatHuntingAgent
from src.xorb.intelligence_engine.agents.behavioral_analysis_agent import BehavioralAnalysisAgent
from src.xorb.intelligence_engine.agents.incident_response_agent import IncidentResponseAgent
from src.xorb.intelligence_engine.agents.attack_prediction_agent import AttackPredictionAgent
from src.xorb.intelligence_engine.agents.compliance_monitoring_agent import ComplianceMonitoringAgent
from src.xorb.intelligence_engine.core.threat_intel import ThreatIntelProvider
import logging
from datetime import datetime

class AgentCoordinator:
    """
    Coordinates communication and collaboration between specialized AI agents.
    Manages threat intelligence sharing, joint analysis, and coordinated response.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Agent Coordinator.
        
        Args:
            config: Configuration parameters for all agents
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize threat intelligence provider
        self.threat_intel = ThreatIntelProvider()
        
        # Initialize specialized agents
        self.threat_hunting_agent = ThreatHuntingAgent(
            name="THA-001",
            config=config.get('threat_hunting', {})
        )
        
        self.behavioral_analysis_agent = BehavioralAnalysisAgent(
            config.get('behavioral_analysis', {})
        )
        
        self.incident_response_agent = IncidentResponseAgent(
            name="IRA-001",
            config=config.get('incident_response', {})
        )
        
        self.attack_prediction_agent = AttackPredictionAgent(
            name="APA-001",
            config=config.get('attack_prediction', {})
        )
        
        self.compliance_monitoring_agent = ComplianceMonitoringAgent(
            name="CMA-001",
            config=config.get('compliance_monitoring', {})
        )
        
        # Shared context for collaborative analysis
        self.shared_context = {
            'last_threat_intel_update': None,
            'active_incidents': [],
            'recent_threats': [],
            'compliance_alerts': [],
            'behavioral_anomalies': []
        }
        
    def update_threat_intelligence(self) -> Dict[str, Any]:
        """
        Update threat intelligence feeds for all agents.
        
        Returns:
            Update status and metadata
        """
        try:
            result = self.threat_intel.update_feeds()
            
            # Update shared context
            self.shared_context['last_threat_intel_update'] = datetime.now().isoformat()
            
            return {
                'status': 'success',
                'feeds_updated': result.get('total_feeds', 0),
                'new_indicators': result.get('new_indicators', 0),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Threat intelligence update failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
        
    def analyze_threats(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive threat analysis using all agents.
        
        Args:
            data: Input data to analyze (network logs, system events, etc.)
            
        Returns:
            Combined analysis results from all agents
        """
        try:
            # Update threat intelligence if needed
            if self._should_update_intel():
                self.update_threat_intelligence()
            
            # Run threat hunting analysis
            hunting_results = self.threat_hunting_agent.hunt_threats(data)
            
            # Run behavioral analysis
            behavioral_results = self._analyze_behavioral_data(data)
            
            # Run attack prediction
            prediction_results = self.attack_prediction_agent.predict_attacks(data)
            
            # Run compliance monitoring
            compliance_results = self.compliance_monitoring_agent.check_compliance(data)
            
            # Update shared context
            self._update_shared_context(hunting_results, behavioral_results, 
                                      prediction_results, compliance_results)
            
            # Generate combined report
            return self._generate_analysis_report(
                hunting_results, behavioral_results, 
                prediction_results, compliance_results
            )
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
        
    def _analyze_behavioral_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and analyze behavioral data from input.
        
        Args:
            data: Input data containing behavioral metrics
            
        Returns:
            Behavioral analysis results
        """
        # Extract behavioral metrics from data
        behavioral_metrics = self._extract_behavioral_metrics(data)
        
        results = {
            'anomalies': [],
            'risk_assessments': []
        }
        
        # Analyze each entity's behavior
        for entity_id, metrics in behavioral_metrics.items():
            analysis = self.behavioral_analysis_agent.analyze_behavior(entity_id, metrics)
            
            if analysis['status'] == 'success' and analysis['is_anomaly']:
                results['anomalies'].append({
                    'entity_id': entity_id,
                    'analysis': analysis
                })
                
            results['risk_assessments'].append({
                'entity_id': entity_id,
                'risk_level': analysis.get('risk_level', 'Unknown')
            })
            
        return results
        
    def _extract_behavioral_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract behavioral metrics from raw data.
        
        Args:
            data: Raw input data
            
        Returns:
            Dictionary of behavioral metrics by entity ID
        """
        # This would be more sophisticated in a real implementation
        # For demonstration, we'll create a simple extraction
        metrics = {}
        
        # Example: Extract user behavior from data
        if 'user_activity' in data:
            for activity in data['user_activity']:
                user_id = activity.get('user_id')
                if user_id:
                    if user_id not in metrics:
                        metrics[user_id] = {
                            'login_attempts': 0,
                            'data_access_volume': 0,
                            'system_resource_usage': 0,
                            'access_pattern_complexity': 0,
                            'geolocation_variance': 0,
                            'time_based_activity': 0
                        }
                        
                    # Update metrics based on activity
                    metrics[user_id]['login_attempts'] += 1
                    metrics[user_id]['data_access_volume'] += activity.get('data_access_count', 0)
                    metrics[user_id]['system_resource_usage'] = max(
                        metrics[user_id]['system_resource_usage'], 
                        activity.get('resource_usage', 0)
                    )
                    
        return metrics
        
    def _update_shared_context(self, hunting_results, behavioral_results, 
                              prediction_results, compliance_results):
        """
        Update shared context with latest analysis results.
        
        Args:
            hunting_results: Results from threat hunting agent
            behavioral_results: Results from behavioral analysis
            prediction_results: Results from attack prediction
            compliance_results: Results from compliance monitoring
        """
        # Update recent threats
        self.shared_context['recent_threats'] = [
            {'id': threat.get('id'), 'confidence': threat.get('confidence', 0)} 
            for threat in hunting_results[:10]
        ]
        
        # Update behavioral anomalies
        self.shared_context['behavioral_anomalies'] = [
            {'entity_id': anomaly['entity_id'], 'risk_level': anomaly['analysis'].get('risk_level')}
            for anomaly in behavioral_results.get('anomalies', [])[:5]
        ]
        
        # Update compliance alerts
        self.shared_context['compliance_alerts'] = [
            {'rule': alert.get('rule'), 'severity': alert.get('severity', 'medium')}
            for alert in compliance_results.get('violations', [])[:5]
        ]
        
        # Update active incidents if any high-risk findings
        if any(item.get('risk_level') in ['High', 'Critical'] 
               for item in behavioral_results.get('risk_assessments', [])):
            self.shared_context['active_incidents'].append({
                'id': f"INC-{datetime.now().strftime('%Y%m%d%H%M')}",
                'type': 'behavioral_anomaly',
                'timestamp': datetime.now().isoformat(),
                'status': 'active'
            })
        
    def _generate_analysis_report(self, hunting_results, behavioral_results, 
                                prediction_results, compliance_results) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report combining all results.
        
        Args:
            hunting_results: Results from threat hunting agent
            behavioral_results: Results from behavioral analysis
            prediction_results: Results from attack prediction
            compliance_results: Results from compliance monitoring
            
        Returns:
            Comprehensive analysis report
        """
        # Calculate overall risk score
        overall_risk = self._calculate_overall_risk(
            hunting_results, behavioral_results, 
            prediction_results, compliance_results
        )
        
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'overall_risk_score': overall_risk,
            'risk_level': self._risk_score_to_level(overall_risk),
            'threat_intel_update': self.shared_context['last_threat_intel_update'],
            'threat_hunting': {
                'total_threats': len(hunting_results),
                'threats': hunting_results[:5]  # Show top 5 threats
            },
            'behavioral_analysis': {
                'total_anomalies': len(behavioral_results.get('anomalies', [])),
                'risk_assessments': behavioral_results.get('risk_assessments', [])[:5]
            },
            'attack_prediction': prediction_results,
            'compliance_monitoring': {
                'total_violations': len(compliance_results.get('violations', [])),
                'violations': compliance_results.get('violations', [])[:3]
            },
            'recommendations': self._generate_recommendations(
                hunting_results, behavioral_results, 
                prediction_results, compliance_results
            ),
            'active_incidents': self.shared_context['active_incidents'][:3]
        }
        
    def _calculate_overall_risk(self, hunting_results, behavioral_results, 
                               prediction_results, compliance_results) -> float:
        """
        Calculate overall risk score based on all analysis results.
        
        Args:
            hunting_results: Results from threat hunting agent
            behavioral_results: Results from behavioral analysis
            prediction_results: Results from attack prediction
            compliance_results: Results from compliance monitoring
            
        Returns:
            Overall risk score (0-1)
        """
        # Simple risk calculation for demonstration
        # In a real implementation, this would be more sophisticated
        
        risk_components = []
        
        # Threat hunting risk (based on confidence of top threats)
        if hunting_results:
            top_threats = hunting_results[:3]
            threat_risk = sum(threat.get('confidence', 0) for threat in top_threats) / 300  # Normalize
            risk_components.append(threat_risk)
        
        # Behavioral analysis risk (based on anomalies)
        behavioral_anomalies = behavioral_results.get('anomalies', [])
        if behavioral_anomalies:
            behavioral_risk = 0
            for anomaly in behavioral_anomalies:
                risk_level = anomaly['analysis'].get('risk_level', 'Low')
                if risk_level == 'Critical':
                    behavioral_risk += 0.9
                elif risk_level == 'High':
                    behavioral_risk += 0.7
                elif risk_level == 'Medium':
                    behavioral_risk += 0.5
            behavioral_risk = min(behavioral_risk / len(behavioral_anomalies), 1.0)
            risk_components.append(behavioral_risk)
        
        # Attack prediction risk (based on likelihood of predicted attacks)
        attack_predictions = prediction_results.get('predictions', [])
        if attack_predictions:
            attack_risk = sum(pred.get('likelihood', 0) for pred in attack_predictions) / 100
            risk_components.append(attack_risk)
        
        # Compliance risk (based on violation severity)
        compliance_violations = compliance_results.get('violations', [])
        if compliance_violations:
            compliance_risk = 0
            for violation in compliance_violations:
                severity = violation.get('severity', 'medium')
                if severity == 'high':
                    compliance_risk += 0.8
                elif severity == 'medium':
                    compliance_risk += 0.5
                else:
                    compliance_risk += 0.3
            compliance_risk = min(compliance_risk / len(compliance_violations), 1.0)
            risk_components.append(compliance_risk)
        
        # Calculate overall risk
        if risk_components:
            return min(sum(risk_components) / len(risk_components), 1.0)
        return 0.1  # Default low risk
        
    def _risk_score_to_level(self, score: float) -> str:
        """
        Convert numerical risk score to risk level.
        
        Args:
            score: Numerical risk score (0-1)
            
        Returns:
            Risk level as string (Low/Medium/High/Critical)
        """
        if score < 0.3:
            return 'Low'
        elif score < 0.5:
            return 'Medium'
        elif score < 0.7:
            return 'High'
        else:
            return 'Critical'
        
    def _generate_recommendations(self, hunting_results, behavioral_results, 
                                prediction_results, compliance_results) -> List[str]:
        """
        Generate security recommendations based on analysis results.
        
        Args:
            hunting_results: Results from threat hunting agent
            behavioral_results: Results from behavioral analysis
            prediction_results: Results from attack prediction
            compliance_results: Results from compliance monitoring
            
        Returns:
            List of security recommendations
        """
        recommendations = []
        
        # Add recommendations based on threat hunting
        if hunting_results:
            high_confidence_threats = [t for t in hunting_results if t.get('confidence', 0) > 70]
            if high_confidence_threats:
                recommendations.append(
                    f"Investigate {len(high_confidence_threats)} high-confidence threats immediately"
                )
        
        # Add recommendations based on behavioral analysis
        behavioral_anomalies = behavioral_results.get('anomalies', [])
        if behavioral_anomalies:
            critical_anomalies = [a for a in behavioral_anomalies 
                                 if a['analysis'].get('risk_level') == 'Critical']
            if critical_anomalies:
                recommendations.append(
                    f"Review {len(critical_anomalies)} critical behavioral anomalies"
                )
        
        # Add recommendations based on attack predictions
        attack_predictions = prediction_results.get('predictions', [])
        if attack_predictions:
            high_likelihood_attacks = [p for p in attack_predictions 
                                      if p.get('likelihood', 0) > 70]
            if high_likelihood_attacks:
                recommendations.append(
                    f"Prepare for {len(high_likelihood_attacks)} high-likelihood attacks"
                )
        
        # Add recommendations based on compliance violations
        compliance_violations = compliance_results.get('violations', [])
        if compliance_violations:
            high_severity_violations = [v for v in compliance_violations 
                                       if v.get('severity') in ['high', 'critical']]
            if high_severity_violations:
                recommendations.append(
                    f"Address {len(high_severity_violations)} high-severity compliance violations"
                )
        
        # Add general recommendations
        if not recommendations:
            recommendations.append("No immediate security concerns detected")
        
        return recommendations
        
    def _should_update_intel(self) -> bool:
        """
        Determine if threat intelligence should be updated.
        
        Returns:
            Boolean indicating whether to update
        """
        if not self.shared_context['last_threat_intel_update']:
            return True
            
        # Update if last update was more than 24 hours ago
        # In a real implementation, this would use actual timestamps
        return True
        
    def get_capabilities(self) -> List[str]:
        """
        Get the capabilities of this coordinator.
        
        Returns:
            List of capabilities
        """
        return [
            'multi_agent_coordination',
            'comprehensive_threat_analysis',
            'risk_scoring',
            'recommendation_generation',
            'threat_intel_integration',
            'incident_management'
        ]
        
    def get_specialization(self) -> str:
        """
        Get the coordinator's specialization.
        
        Returns:
            Specialization description
        """
        return 'Multi-Agent Cybersecurity Analysis and Coordination'