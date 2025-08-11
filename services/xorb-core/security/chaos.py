import random
import string
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import logging
import asyncio
from .audit import SecurityAuditFramework
from .testing import SecurityTestingFramework
from .monitoring import SecurityMonitoringSystem

class SecurityChaosEngineeringFramework:
    """Comprehensive framework for applying chaos engineering principles to security validation."""
    
    def __init__(self, 
                 audit_framework: SecurityAuditFramework,
                 test_framework: SecurityTestingFramework,
                 monitoring_system: SecurityMonitoringSystem):
        self.audit_framework = audit_framework
        self.test_framework = test_framework
        self.monitoring_system = monitoring_system
        self.logger = logging.getLogger(__name__)
        self.active_experiments = {}
        self.experiment_history = []
        
    async def start_chaos_experiment(self, 
                                     experiment_id: str, 
                                     experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start a new chaos experiment for security validation.
        
        Args:
            experiment_id: Unique identifier for the experiment
            experiment_config: Configuration for the experiment
            {
                'attack_vectors': List[str],  # Types of attacks to simulate
                'targets': List[str],         # Components to target
                'duration': int,              # Duration in seconds
                'intensity': str,             # Intensity level (low/medium/high/critical)
                'monitoring': bool            # Whether to enable monitoring
            }
        
        Returns:
            Dict with experiment status and details
        """
        try:
            # Validate experiment configuration
            if experiment_id in self.active_experiments:
                return {
                    'status': 'error',
                    'message': f'Experiment {experiment_id} already exists'
                }
                
            required_fields = ['attack_vectors', 'targets', 'duration', 'intensity']
            if not all(field in experiment_config for field in required_fields):
                return {
                    'status': 'error',
                    'message': 'Missing required configuration fields'
                }
                
            # Initialize experiment
            experiment = {
                'id': experiment_id,
                'config': experiment_config,
                'status': 'initializing',
                'start_time': datetime.utcnow(),
                'end_time': None,
                'results': None,
                'metrics': {},
                'security_impact': {}
            }
            
            self.active_experiments[experiment_id] = experiment
            self.logger.info(f'Starting chaos experiment {experiment_id}')
            
            # Run the experiment
            await self._run_experiment(experiment)
            
            return {
                'status': 'success',
                'experiment_id': experiment_id,
                'message': 'Chaos experiment completed successfully'
            }
            
        except Exception as e:
            self.logger.error(f'Error starting chaos experiment: {str(e)}', exc_info=True)
            return {
                'status': 'error',
                'message': f'Error starting chaos experiment: {str(e)}'
            }
            
    async def _run_experiment(self, experiment: Dict[str, Any]) -> None:
        """Internal method to run a chaos experiment."""
        try:
            experiment['status'] = 'running'
            
            # Initialize metrics
            metrics = {
                'attack_attempts': 0,
                'successful_attacks': 0,
                'detected_attacks': 0,
                'blocked_attacks': 0,
                'security_events': [],
                'system_response_time': [],
                'resource_usage': []
            }
            
            # Start monitoring if enabled
            monitoring_enabled = experiment['config'].get('monitoring', True)
            if monitoring_enabled:
                await self.monitoring_system.start_monitoring(experiment['id'])
                
            # Simulate attack vectors
            attack_duration = experiment['config']['duration']
            attack_vectors = experiment['config']['attack_vectors']
            targets = experiment['config']['targets']
            intensity = experiment['config']['intensity'].lower()
            
            # Map intensity to attack frequency
            intensity_factors = {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.9,
                'critical': 1.0
            }
            
            intensity_factor = intensity_factors.get(intensity, 0.5)
            attack_interval = max(0.1, (1.0 - intensity_factor) * 0.5)  # Adjust interval based on intensity
            
            end_time = datetime.utcnow() + timedelta(seconds=attack_duration)
            
            while datetime.utcnow() < end_time:
                # Randomly select attack vector and target
                attack_vector = random.choice(attack_vectors)
                target = random.choice(targets)
                
                # Execute attack simulation
                attack_result = await self._simulate_attack(attack_vector, target)
                
                # Record metrics
                metrics['attack_attempts'] += 1
                if attack_result['success']:
                    metrics['successful_attacks'] += 1
                    
                if attack_result['detected']:
                    metrics['detected_attacks'] += 1
                    
                if attack_result['blocked']:
                    metrics['blocked_attacks'] += 1
                    
                # Record security event if attack was successful
                if attack_result['success'] and not attack_result['blocked']:
                    metrics['security_events'].append({
                        'timestamp': datetime.utcnow().isoformat(),
                        'attack_vector': attack_vector,
                        'target': target,
                        'impact': attack_result['impact']
                    })
                    
                # Record system metrics
                metrics['system_response_time'].append(attack_result['response_time'])
                metrics['resource_usage'].append(attack_result['resource_usage'])
                
                # Wait before next attack
                await asyncio.sleep(attack_interval * random.uniform(0.8, 1.2))
                
            # End monitoring
            if monitoring_enabled:
                monitoring_data = await self.monitoring_system.stop_monitoring(experiment['id'])
                metrics['monitoring_data'] = monitoring_data
                
            # Analyze results
            analysis = await self._analyze_experiment_results(metrics)
            
            # Update experiment with results
            experiment['status'] = 'completed'
            experiment['end_time'] = datetime.utcnow()
            experiment['results'] = {
                'metrics': metrics,
                'analysis': analysis
            }
            
            # Add to history
            self.experiment_history.append(experiment)
            
            # Clean up active experiments
            del self.active_experiments[experiment_id]
            
        except Exception as e:
            self.logger.error(f'Error running chaos experiment: {str(e)}', exc_info=True)
            experiment['status'] = 'failed'
            experiment['error'] = str(e)
            
    async def _simulate_attack(self, attack_vector: str, target: str) -> Dict[str, Any]:
        """Simulate a specific attack vector against a target."""
        # This would be implemented with actual attack simulations in a real system
        # For this example, we'll simulate with random results based on attack vector and intensity
        
        # Base success rates for different attack vectors
        attack_success_rates = {
            'sql_injection': 0.4,
            'xss': 0.35,
            'csrf': 0.3,
            'session_hijacking': 0.25,
            'brute_force': 0.5,
            'privilege_escalation': 0.2,
            'api_abuse': 0.45,
            'file_inclusion': 0.35,
            'command_injection': 0.25,
            'directory_traversal': 0.3
        }
        
        # Base detection rates
        detection_rates = {
            'sql_injection': 0.8,
            'xss': 0.75,
            'csrf': 0.7,
            'session_hijacking': 0.65,
            'brute_force': 0.9,
            'privilege_escalation': 0.6,
            'api_abuse': 0.75,
            'file_inclusion': 0.7,
            'command_injection': 0.65,
            'directory_traversal': 0.75
        }
        
        # Base blocking rates
        blocking_rates = {
            'sql_injection': 0.95,
            'xss': 0.9,
            'csrf': 0.85,
            'session_hijacking': 0.8,
            'brute_force': 0.98,
            'privilege_escalation': 0.75,
            'api_abuse': 0.9,
            'file_inclusion': 0.85,
            'command_injection': 0.8,
            'directory_traversal': 0.85
        }
        
        # Calculate success probability
        base_success_rate = attack_success_rates.get(attack_vector, 0.3)
        success = random.random() < base_success_rate
        
        # Calculate detection probability
        base_detection_rate = detection_rates.get(attack_vector, 0.7)
        detected = random.random() < base_detection_rate
        
        # Calculate blocking probability
        base_blocking_rate = blocking_rates.get(attack_vector, 0.8)
        blocked = random.random() < base_blocking_rate
        
        # If attack is blocked, it can't be successful
        if blocked:
            success = False
            
        # If attack is detected, it might be blocked
        if detected and not blocked:
            blocked = random.random() < 0.5  # 50% chance of blocking detected attacks
            
        # Calculate impact if attack was successful and not blocked
        impact = 0.0
        if success and not blocked:
            impact = random.uniform(0.5, 1.0)  # High impact for successful, undetected attacks
        elif success and blocked:
            impact = random.uniform(0.1, 0.4)  # Lower impact for detected/blocked attacks
            
        # Simulate response time and resource usage
        response_time = random.uniform(50, 1500)  # ms
        resource_usage = {
            'cpu': random.uniform(20, 90),  # %
            'memory': random.uniform(30, 85),  # %
            'network': random.uniform(10, 70)  # Mbps
        }
        
        return {
            'attack_vector': attack_vector,
            'target': target,
            'success': success,
            'detected': detected,
            'blocked': blocked,
            'impact': impact,
            'response_time': response_time,
            'resource_usage': resource_usage
        }
        
    async def _analyze_experiment_results(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the results of a chaos experiment."""
        analysis = {
            'summary': {},
            'security_gaps': [],
            'recommendations': [],
            'risk_assessment': {},
            'compliance_check': {}
        }
        
        # Calculate key metrics
        total_attacks = metrics['attack_attempts']
        if total_attacks > 0:
            success_rate = metrics['successful_attacks'] / total_attacks
            detection_rate = metrics['detected_attacks'] / total_attacks
            blocking_rate = metrics['blocked_attacks'] / total_attacks
            effective_blocking_rate = (metrics['detected_attacks'] + metrics['blocked_attacks']) / total_attacks
            
            analysis['summary'] = {
                'total_attacks': total_attacks,
                'success_rate': success_rate,
                'detection_rate': detection_rate,
                'blocking_rate': blocking_rate,
                'effective_blocking_rate': effective_blocking_rate,
                'security_events_count': len(metrics['security_events'])
            }
            
            # Identify security gaps
            if success_rate > 0.2:
                analysis['security_gaps'].append({
                    'issue': 'High attack success rate',
                    'severity': 'high' if success_rate > 0.3 else 'medium',
                    'description': f'Attack success rate of {success_rate:.2%} indicates potential security vulnerabilities'
                })
                
            if detection_rate < 0.7:
                analysis['security_gaps'].append({
                    'issue': 'Low attack detection rate',
                    'severity': 'high' if detection_rate < 0.5 else 'medium',
                    'description': f'Attack detection rate of {detection_rate:.2%} indicates gaps in monitoring capabilities'
                })
                
            if blocking_rate < 0.75:
                analysis['security_gaps'].append({
                    'issue': 'Low attack blocking rate',
                    'severity': 'high' if blocking_rate < 0.6 else 'medium',
                    'description': f'Attack blocking rate of {blocking_rate:.2%} indicates gaps in protection mechanisms'
                })
                
            # Generate recommendations
            if success_rate > 0.2:
                analysis['recommendations'].append(
                    'Implement additional security controls for vulnerable components'
                )
                
            if detection_rate < 0.7:
                analysis['recommendations'].append(
                    'Enhance monitoring and detection capabilities for better threat visibility'
                )
                
            if blocking_rate < 0.75:
                analysis['recommendations'].append(
                    'Improve blocking mechanisms and response automation'
                )
                
            # Risk assessment
            risk_score = min(1.0, success_rate * 1.5)
            risk_level = 'low'
            if risk_score > 0.5:
                risk_level = 'high'
            elif risk_score > 0.3:
                risk_level = 'medium'
                
            analysis['risk_assessment'] = {
                'score': risk_score,
                'level': risk_level,
                'description': f'Overall security risk level: {risk_level.upper()} (score: {risk_score:.2f}/1.0)'
            }
            
            # Compliance check
            compliance_score = 1.0 - risk_score
            compliance_status = 'pass' if compliance_score >= 0.7 else 'fail'
            
            analysis['compliance_check'] = {
                'score': compliance_score,
                'status': compliance_status,
                'description': f'Compliance status: {compliance_status.upper()} (score: {compliance_score:.2f}/1.0)'
            }
            
        return analysis
        
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get the status of a running experiment."""
        if experiment_id in self.active_experiments:
            experiment = self.active_experiments[experiment_id]
            return {
                'status': experiment['status'],
                'start_time': experiment['start_time'].isoformat(),
                'elapsed_time': (datetime.utcnow() - experiment['start_time']).total_seconds(),
                'config': experiment['config']
            }
        elif any(exp['id'] == experiment_id for exp in self.experiment_history):
            experiment = next(exp for exp in self.experiment_history if exp['id'] == experiment_id)
            return {
                'status': experiment['status'],
                'start_time': experiment['start_time'].isoformat(),
                'end_time': experiment['end_time'].isoformat() if experiment['end_time'] else None,
                'results': experiment.get('results')
            }
        else:
            return {
                'status': 'error',
                'message': f'Experiment {experiment_id} not found'
            }
            
    async def generate_security_report(self, experiment_id: str) -> Dict[str, Any]:
        """Generate a comprehensive security report for an experiment."""
        experiment = self.get_experiment_status(experiment_id)
        if experiment.get('status') == 'error':
            return experiment
            
        if experiment['status'] not in ['completed', 'failed']:
            return {
                'status': 'error',
                'message': f'Experiment {experiment_id} is not completed yet'
            }
            
        # Get the experiment from history
        experiment_data = next(exp for exp in self.experiment_history if exp['id'] == experiment_id)
        results = experiment_data.get('results', {})
        metrics = results.get('metrics', {})
        analysis = results.get('analysis', {})
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(analysis)
        
        # Generate detailed findings
        detailed_findings = self._generate_detailed_findings(metrics, analysis)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(analysis)
        
        # Generate compliance report
        compliance_report = self._generate_compliance_report(analysis)
        
        return {
            'status': 'success',
            'report': {
                'executive_summary': executive_summary,
                'detailed_findings': detailed_findings,
                'recommendations': recommendations,
                'compliance_report': compliance_report,
                'raw_metrics': metrics
            }
        }
        
    def _generate_executive_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an executive summary for the security report."""
        risk_assessment = analysis.get('risk_assessment', {})
        risk_level = risk_assessment.get('level', 'unknown')
        
        summary = {
            'risk_level': risk_level,
            'risk_description': risk_assessment.get('description', 'No risk assessment available'),
            'compliance_status': analysis.get('compliance_check', {}).get('status', 'unknown'),
            'key_findings': []
        }
        
        # Add key findings based on security gaps
        for gap in analysis.get('security_gaps', []):
            summary['key_findings'].append({
                'issue': gap['issue'],
                'severity': gap['severity'],
                'description': gap['description']
            })
            
        return summary
        
    def _generate_detailed_findings(self, 
                                    metrics: Dict[str, Any], 
                                    analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed findings for the security report."""
        findings = {
            'attack_statistics': {
                'total_attacks': metrics.get('attack_attempts', 0),
                'successful_attacks': metrics.get('successful_attacks', 0),
                'detected_attacks': metrics.get('detected_attacks', 0),
                'blocked_attacks': metrics.get('blocked_attacks', 0)
            },
            'security_events': metrics.get('security_events', []),
            'system_performance': {
                'average_response_time': sum(metrics.get('system_response_time', [0])) / len(metrics.get('system_response_time', [1])) if metrics.get('system_response_time') else 0,
                'peak_response_time': max(metrics.get('system_response_time', [0])) if metrics.get('system_response_time') else 0,
                'resource_usage': {
                    'cpu': {
                        'average': sum(r['cpu'] for r in metrics.get('resource_usage', [])) / len(metrics.get('resource_usage', [1])) if metrics.get('resource_usage') else 0,
                        'peak': max(r['cpu'] for r in metrics.get('resource_usage', [])) if metrics.get('resource_usage') else 0
                    },
                    'memory': {
                        'average': sum(r['memory'] for r in metrics.get('resource_usage', [])) / len(metrics.get('resource_usage', [1])) if metrics.get('resource_usage') else 0,
                        'peak': max(r['memory'] for r in metrics.get('resource_usage', [])) if metrics.get('resource_usage') else 0
                    },
                    'network': {
                        'average': sum(r['network'] for r in metrics.get('resource_usage', [])) / len(metrics.get('resource_usage', [1])) if metrics.get('resource_usage') else 0,
                        'peak': max(r['network'] for r in metrics.get('resource_usage', [])) if metrics.get('resource_usage') else 0
                    }
                }
            },
            'analysis_details': analysis
        }
        
        return findings
        
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on the analysis."""
        return analysis.get('recommendations', [])
        
    def _generate_compliance_report(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a compliance report based on the analysis."""
        compliance_check = analysis.get('compliance_check', {})
        
        return {
            'status': compliance_check.get('status', 'unknown'),
            'score': compliance_check.get('score', 0.0),
            'requirements': [
                'OWASP Top 10 compliance',
                'GDPR compliance',
                'ISO 27001 compliance',
                'NIST cybersecurity framework compliance'
            ],
            'recommendations': compliance_check.get('recommendations', [])
        }
        
    async def run_security_validation(self, 
                                     intensity: str = 'medium',
                                     duration: int = 300,
                                     monitoring: bool = True) -> Dict[str, Any]:
        """
        Run a comprehensive security validation using chaos engineering.
        
        Args:
            intensity: Intensity level (low/medium/high/critical)
            duration: Duration in seconds
            monitoring: Whether to enable monitoring
            
        Returns:
            Dict with validation results
        """
        # Generate a random experiment ID
        experiment_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        
        # Define attack vectors based on intensity
        attack_vectors = [
            'sql_injection',
            'xss',
            'csrf',
            'session_hijacking',
            'brute_force',
            'privilege_escalation',
            'api_abuse',
            'file_inclusion',
            'command_injection',
            'directory_traversal'
        ]
        
        # For low intensity, use fewer attack vectors
        if intensity.lower() == 'low':
            attack_vectors = random.sample(attack_vectors, 4)
            
        # Define targets
        targets = [
            'auth_system',
            'api_endpoints',
            'database_layer',
            'user_interface',
            'file_storage',
            'network_communication'
        ]
        
        # Create experiment configuration
        experiment_config = {
            'attack_vectors': attack_vectors,
            'targets': targets,
            'duration': duration,
            'intensity': intensity,
            'monitoring': monitoring
        }
        
        # Start the experiment
        result = await self.start_chaos_experiment(experiment_id, experiment_config)
        
        if result['status'] != 'success':
            return result
            
        # Generate report
        report_result = await self.generate_security_report(experiment_id)
        
        if report_result['status'] != 'success':
            return report_result
            
        return {
            'status': 'success',
            'experiment_id': experiment_id,
            'report': report_result['report']
        }
        
    async def schedule_periodic_validation(self, 
                                          intensity: str = 'medium',
                                          duration: int = 300,
                                          interval: int = 86400,
                                          monitoring: bool = True) -> Dict[str, Any]:
        """
        Schedule periodic security validation using chaos engineering.
        
        Args:
            intensity: Intensity level (low/medium/high/critical)
            duration: Duration in seconds
            interval: Interval in seconds between validations
            monitoring: Whether to enable monitoring
            
        Returns:
            Dict with scheduling results
        """
        # This would be implemented with a scheduler in a real system
        # For this example, we'll just return a confirmation
        return {
            'status': 'success',
            'message': f'Scheduled periodic security validation every {interval} seconds',
            'intensity': intensity,
            'duration': duration,
            'monitoring': monitoring
        }
        
    async def get_security_health(self) -> Dict[str, Any]:
        """Get the overall security health based on recent experiments."""
        if not self.experiment_history:
            return {
                'status': 'no_data',
                'message': 'No security experiments have been run yet'
            }
            
        # Get recent experiments (last 7 days)
        recent_experiments = [
            exp for exp in self.experiment_history
            if (datetime.utcnow() - exp['end_time']).days <= 7
        ]
        
        if not recent_experiments:
            return {
                'status': 'no_recent_data',
                'message': 'No security experiments in the last 7 days'
            }
            
        # Calculate average risk score
        total_risk_score = sum(
            exp.get('results', {}).get('analysis', {}).get('risk_assessment', {}).get('score', 0)
            for exp in recent_experiments
        )
        average_risk_score = total_risk_score / len(recent_experiments) if recent_experiments else 0
        
        # Determine overall security health
        if average_risk_score < 0.3:
            health_status = 'excellent'
        elif average_risk_score < 0.5:
            health_status = 'good'
        elif average_risk_score < 0.7:
            health_status = 'fair'
        else:
            health_status = 'poor'
            
        return {
            'status': 'success',
            'health_status': health_status,
            'average_risk_score': average_risk_score,
            'recent_experiments': len(recent_experiments),
            'last_experiment': recent_experiments[-1]['end_time'].isoformat()
        }
        
    async def validate_security_changes(self, 
                                     changes: List[Dict[str, Any]],
                                     intensity: str = 'medium') -> Dict[str, Any]:
        """
        Validate specific security changes using chaos engineering.
        
        Args:
            changes: List of security changes to validate
            intensity: Intensity level (low/medium/high/critical)
            
        Returns:
            Dict with validation results
        """
        # Generate a random experiment ID
        experiment_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        
        # Determine targets based on changes
        targets = []
        for change in changes:
            if 'component' in change:
                targets.append(change['component'])
            elif 'type' in change:
                targets.append(change['type'])
                
        if not targets:
            targets = ['security_changes_validation']
            
        # Define attack vectors based on the nature of changes
        attack_vectors = [
            'privilege_escalation',  # For access control changes
            'api_abuse',            # For API-related changes
            'session_hijacking',    # For authentication changes
            'brute_force',          # For password policy changes
            'xss',                  # For UI changes
            'csrf'                  # For form-related changes
        ]
        
        # Filter attack vectors based on the nature of changes
        filtered_attack_vectors = []
        for change in changes:
            if change.get('type') == 'access_control':
                filtered_attack_vectors.append('privilege_escalation')
            elif change.get('type') == 'authentication':
                filtered_attack_vectors.extend(['session_hijacking', 'brute_force'])
            elif change.get('type') == 'api':
                filtered_attack_vectors.append('api_abuse')
            elif change.get('type') == 'ui':
                filtered_attack_vectors.append('xss')
            elif change.get('type') == 'network':
                filtered_attack_vectors.append('csrf')
                
        if not filtered_attack_vectors:
            filtered_attack_vectors = attack_vectors
            
        # Create experiment configuration
        experiment_config = {
            'attack_vectors': list(set(filtered_attack_vectors)),  # Remove duplicates
            'targets': targets,
            'duration': 300,  # 5 minutes
            'intensity': intensity,
            'monitoring': True
        }
        
        # Start the experiment
        result = await self.start_chaos_experiment(experiment_id, experiment_config)
        
        if result['status'] != 'success':
            return result
            
        # Generate report
        report_result = await self.generate_security_report(experiment_id)
        
        if report_result['status'] != 'success':
            return report_result
            
        return {
            'status': 'success',
            'experiment_id': experiment_id,
            'report': report_result['report'],
            'changes_validated': changes
        }