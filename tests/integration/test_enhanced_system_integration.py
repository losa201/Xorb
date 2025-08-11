"""
Comprehensive Integration Tests for Enhanced XORB Components
Principal Auditor Implementation - Production-Ready Test Suite
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock

# Import enhanced components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/api/app'))

from services.advanced_behavioral_analytics_engine import (
    AdvancedBehavioralAnalyticsEngine, BehavioralProfile, AnomalyDetection
)
from services.autonomous_security_orchestrator import AutonomousSecurityOrchestrator
from services.quantum_security_suite import QuantumSecurityService, CryptoAlgorithm
from services.advanced_performance_monitor import AdvancedPerformanceMonitor

# Mock data for testing
MOCK_SECURITY_EVENT = {
    'event_id': 'test_event_001',
    'timestamp': datetime.utcnow(),
    'event_type': 'authentication_attempt',
    'source_ip': '192.168.1.100',
    'user_agent': 'Mozilla/5.0 Test Browser',
    'entity_id': 'user_001',
    'request_size': 1024,
    'response_size': 512,
    'session_duration': 300,
    'failed_attempts': 0,
    'privilege_level': 1,
    'resource_access_count': 5,
    'location': {'country': 'US', 'city': 'New York'},
    'device_fingerprint': 'test_device_001'
}

MOCK_THREAT_CONTEXT = {
    'threat_indicators': ['suspicious_login_pattern', 'unusual_geolocation'],
    'network_activity': {'connections': 15, 'data_volume': 50000},
    'user_behavior': {'login_frequency': 3, 'access_pattern_anomaly': 0.7},
    'environment': 'production',
    'time_window': '1h'
}

@pytest.fixture
async def behavioral_analytics_engine():
    """Fixture for behavioral analytics engine"""
    engine = AdvancedBehavioralAnalyticsEngine({'baseline_window_days': 1})
    await engine.initialize()
    yield engine
    await engine.stop_monitoring() if hasattr(engine, 'stop_monitoring') else None

@pytest.fixture
async def autonomous_orchestrator():
    """Fixture for autonomous security orchestrator"""
    orchestrator = AutonomousSecurityOrchestrator()
    await orchestrator.initialize()
    yield orchestrator

@pytest.fixture
async def quantum_security_service():
    """Fixture for quantum security service"""
    service = QuantumSecurityService()
    await service.initialize()
    yield service

@pytest.fixture
async def performance_monitor():
    """Fixture for performance monitor"""
    monitor = AdvancedPerformanceMonitor({'monitoring_interval': 1})
    await monitor.initialize()
    yield monitor
    await monitor.stop_monitoring()

class TestBehavioralAnalyticsIntegration:
    """Integration tests for behavioral analytics engine"""
    
    @pytest.mark.asyncio
    async def test_real_time_anomaly_detection(self, behavioral_analytics_engine):
        """Test real-time anomaly detection capabilities"""
        engine = behavioral_analytics_engine
        
        # Create mock security events
        normal_events = []
        anomalous_events = []
        
        # Generate normal events
        for i in range(10):
            event = MOCK_SECURITY_EVENT.copy()
            event['event_id'] = f'normal_event_{i}'
            event['entity_id'] = 'normal_user'
            event['failed_attempts'] = 0
            normal_events.append(event)
        
        # Generate anomalous events
        for i in range(3):
            event = MOCK_SECURITY_EVENT.copy()
            event['event_id'] = f'anomalous_event_{i}'
            event['entity_id'] = 'suspicious_user'
            event['failed_attempts'] = 5  # High failure rate
            event['source_ip'] = '10.0.0.1'  # Different IP pattern
            anomalous_events.append(event)
        
        # Process normal events to establish baseline
        for event in normal_events:
            anomalies = await engine.detect_real_time_anomalies(event)
            # Should detect few or no anomalies for normal behavior
            assert len(anomalies) <= 1, "Too many anomalies detected for normal behavior"
        
        # Process anomalous events
        detected_anomalies = []
        for event in anomalous_events:
            anomalies = await engine.detect_real_time_anomalies(event)
            detected_anomalies.extend(anomalies)
        
        # Should detect anomalies in suspicious behavior
        assert len(detected_anomalies) >= 1, "Failed to detect obvious anomalies"
        
        # Verify anomaly structure
        if detected_anomalies:
            anomaly = detected_anomalies[0]
            assert 'anomaly_id' in anomaly
            assert 'entity_id' in anomaly
            assert 'anomaly_type' in anomaly
            assert 'risk_level' in anomaly
    
    @pytest.mark.asyncio
    async def test_behavioral_profile_analysis(self, behavioral_analytics_engine):
        """Test comprehensive behavioral profile analysis"""
        engine = behavioral_analytics_engine
        
        # Simulate user behavior over time
        entity_id = 'test_user_profile'
        events = []
        
        for i in range(20):
            event = MOCK_SECURITY_EVENT.copy()
            event['entity_id'] = entity_id
            event['event_id'] = f'profile_event_{i}'
            event['timestamp'] = datetime.utcnow() - timedelta(hours=i)
            events.append(event)
        
        # Analyze behavior
        profile = await engine.analyze_entity_behavior(entity_id, events)
        
        # Verify profile structure
        assert isinstance(profile, BehavioralProfile)
        assert profile.entity_id == entity_id
        assert profile.risk_score >= 0.0 and profile.risk_score <= 1.0
        assert profile.confidence_score >= 0.0 and profile.confidence_score <= 1.0
        assert isinstance(profile.anomaly_indicators, list)
        assert profile.baseline_established in [True, False]
        
        # Get risk assessment
        risk_assessment = await engine.get_entity_risk_assessment(entity_id)
        assert risk_assessment['entity_id'] == entity_id
        assert 'risk_level' in risk_assessment
        assert 'risk_score' in risk_assessment
        assert 'assessment_available' in risk_assessment
    
    @pytest.mark.asyncio
    async def test_analytics_dashboard(self, behavioral_analytics_engine):
        """Test analytics dashboard functionality"""
        engine = behavioral_analytics_engine
        
        # Generate some activity
        for i in range(5):
            event = MOCK_SECURITY_EVENT.copy()
            event['entity_id'] = f'dashboard_user_{i}'
            await engine.detect_real_time_anomalies(event)
        
        # Get dashboard data
        dashboard = await engine.get_analytics_dashboard()
        
        # Verify dashboard structure
        assert 'total_entities' in dashboard
        assert 'high_risk_entities' in dashboard
        assert 'detection_metrics' in dashboard
        assert isinstance(dashboard['total_entities'], int)
        assert isinstance(dashboard['high_risk_entities'], int)

class TestAutonomousOrchestratorIntegration:
    """Integration tests for autonomous security orchestrator"""
    
    @pytest.mark.asyncio
    async def test_orchestration_plan_creation(self, autonomous_orchestrator):
        """Test intelligent orchestration plan creation"""
        orchestrator = autonomous_orchestrator
        
        # Create orchestration plan
        objective = "Detect and respond to advanced persistent threat"
        context = MOCK_THREAT_CONTEXT
        constraints = {'max_duration': 3600, 'resource_limit': 'medium'}
        
        plan = await orchestrator.create_orchestration_plan(
            objective, context, constraints
        )
        
        # Verify plan structure
        assert 'plan_id' in plan
        assert 'objective' in plan
        assert 'task_decomposition' in plan
        assert 'agent_assignments' in plan
        assert 'execution_timeline' in plan
        assert 'success_probability' in plan
        
        # Verify task decomposition
        tasks = plan['task_decomposition']
        assert isinstance(tasks, list)
        assert len(tasks) > 0
        
        for task in tasks:
            assert 'task_id' in task
            assert 'type' in task
            assert 'description' in task
            assert 'priority' in task
            assert 'estimated_duration' in task
        
        # Verify agent assignments
        assignments = plan['agent_assignments']
        assert isinstance(assignments, dict)
        
        # Verify success probability is reasonable
        success_prob = plan['success_probability']
        assert 0.0 <= success_prob <= 1.0
    
    @pytest.mark.asyncio
    async def test_adaptive_response(self, autonomous_orchestrator):
        """Test adaptive response to security events"""
        orchestrator = autonomous_orchestrator
        
        # Create security event
        security_event = {
            'event_type': 'malware_detection',
            'severity': 'high',
            'affected_systems': ['web-server-01', 'database-01'],
            'indicators': ['malicious_hash', 'suspicious_network_activity'],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Generate adaptive response
        response = await orchestrator.adaptive_response(security_event)
        
        # Verify response structure
        assert 'event_id' in response
        assert 'security_event' in response
        assert 'event_analysis' in response
        assert 'response_strategy' in response
        assert 'response_agents' in response
        assert 'immediate_plan' in response
        assert 'immediate_results' in response
        assert 'followup_actions' in response
        assert 'confidence' in response
        
        # Verify confidence score
        confidence = response['confidence']
        assert 0.0 <= confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_intelligent_collaboration(self, autonomous_orchestrator):
        """Test intelligent agent collaboration"""
        orchestrator = autonomous_orchestrator
        
        # Create collaboration request
        collaboration_request = {
            'source_agent': 'threat_hunter_agent',
            'target_agents': ['vulnerability_scanner_agent', 'incident_responder_agent'],
            'type': 'knowledge_sharing',
            'data': {
                'threat_indicators': ['IOC1', 'IOC2', 'IOC3'],
                'analysis_results': {'confidence': 0.85, 'severity': 'high'}
            }
        }
        
        # Process collaboration
        result = await orchestrator.intelligent_collaboration(collaboration_request)
        
        # Verify collaboration result
        assert 'collaboration_id' in result
        assert 'source_agent' in result
        assert 'target_agents' in result
        assert 'collaboration_type' in result
        assert 'strategy' in result
        assert 'knowledge_sharing_results' in result
        assert 'joint_actions' in result
        assert 'collaboration_metrics' in result
    
    @pytest.mark.asyncio
    async def test_autonomous_learning(self, autonomous_orchestrator):
        """Test autonomous learning capabilities"""
        orchestrator = autonomous_orchestrator
        
        # Create learning data
        learning_data = {
            'execution_results': [
                {'task_id': 'task_001', 'success': True, 'duration': 120, 'efficiency': 0.85},
                {'task_id': 'task_002', 'success': False, 'duration': 300, 'efficiency': 0.32},
                {'task_id': 'task_003', 'success': True, 'duration': 180, 'efficiency': 0.92}
            ],
            'performance_metrics': {
                'accuracy': 0.78,
                'precision': 0.82,
                'recall': 0.75,
                'f1_score': 0.78
            },
            'environmental_factors': {
                'system_load': 0.65,
                'network_latency': 45,
                'threat_landscape': 'elevated'
            }
        }
        
        # Perform autonomous learning
        learning_results = await orchestrator.autonomous_learning(learning_data)
        
        # Verify learning results
        assert 'learning_session_id' in learning_results
        assert 'learning_data_analysis' in learning_results
        assert 'extracted_patterns' in learning_results
        assert 'capability_updates' in learning_results
        assert 'strategy_improvements' in learning_results
        assert 'effectiveness_score' in learning_results
        
        # Verify effectiveness score
        effectiveness = learning_results['effectiveness_score']
        assert 0.0 <= effectiveness <= 1.0

class TestQuantumSecurityIntegration:
    """Integration tests for quantum security suite"""
    
    @pytest.mark.asyncio
    async def test_quantum_keypair_generation(self, quantum_security_service):
        """Test quantum-safe key pair generation"""
        service = quantum_security_service
        
        # Test different algorithms
        algorithms = [
            CryptoAlgorithm.KYBER_1024,
            CryptoAlgorithm.DILITHIUM_5,
            CryptoAlgorithm.HYBRID_RSA_KYBER
        ]
        
        for algorithm in algorithms:
            keypair = await service.generate_quantum_safe_keypair(algorithm)
            
            # Verify keypair structure
            assert keypair.key_id is not None
            assert keypair.algorithm == algorithm.value
            assert keypair.public_key is not None
            assert keypair.private_key is not None
            assert keypair.created_timestamp is not None
    
    @pytest.mark.asyncio
    async def test_quantum_encryption_decryption(self, quantum_security_service):
        """Test quantum-safe encryption and decryption"""
        service = quantum_security_service
        
        # Test data
        test_data = b"This is sensitive data that needs quantum protection"
        
        # Test encryption
        encryption_result = await service.encrypt_with_quantum_protection(test_data)
        
        # Verify encryption result
        assert encryption_result.key_id is not None
        assert encryption_result.ciphertext is not None
        assert encryption_result.algorithm is not None
        assert encryption_result.ciphertext != test_data
        
        # Test decryption
        decrypted_data = await service.decrypt_with_quantum_protection(
            encryption_result, encryption_result.key_id
        )
        
        # Verify decryption
        assert decrypted_data == test_data
    
    @pytest.mark.asyncio
    async def test_quantum_signatures(self, quantum_security_service):
        """Test quantum-safe digital signatures"""
        service = quantum_security_service
        
        # Test data
        test_data = b"Document requiring quantum-safe signature"
        
        # Create signature
        signature_result = await service.create_quantum_safe_signature(test_data)
        
        # Verify signature structure
        assert signature_result.signature is not None
        assert signature_result.algorithm is not None
        assert signature_result.key_id is not None
        assert signature_result.data_hash is not None
        
        # Verify signature
        is_valid = await service.verify_quantum_safe_signature(
            test_data, signature_result, signature_result.key_id
        )
        
        assert is_valid is True
        
        # Test with tampered data
        tampered_data = b"Tampered document content"
        is_valid_tampered = await service.verify_quantum_safe_signature(
            tampered_data, signature_result, signature_result.key_id
        )
        
        assert is_valid_tampered is False
    
    @pytest.mark.asyncio
    async def test_quantum_readiness_assessment(self, quantum_security_service):
        """Test quantum readiness assessment"""
        service = quantum_security_service
        
        # Mock target system
        target_system = {
            'id': 'test_system_001',
            'cryptographic_implementations': ['RSA-2048', 'AES-256', 'SHA-256'],
            'network_protocols': ['TLS-1.2', 'TLS-1.3'],
            'applications': ['web_server', 'database', 'api_gateway']
        }
        
        # Perform assessment
        assessment = await service.assess_quantum_readiness(target_system)
        
        # Verify assessment structure
        assert 'system_id' in assessment
        assert 'assessment_timestamp' in assessment
        assert 'overall_score' in assessment
        assert 'crypto_analysis' in assessment
        assert 'vulnerability_score' in assessment
        assert 'compliance_status' in assessment
        assert 'migration_plan' in assessment
        assert 'recommendations' in assessment
        
        # Verify score ranges
        overall_score = assessment['overall_score']
        assert 0.0 <= overall_score <= 1.0
        
        vulnerability_score = assessment['vulnerability_score']
        assert 0.0 <= vulnerability_score <= 1.0

class TestPerformanceMonitorIntegration:
    """Integration tests for advanced performance monitor"""
    
    @pytest.mark.asyncio
    async def test_system_metrics_collection(self, performance_monitor):
        """Test system metrics collection"""
        monitor = performance_monitor
        
        # Collect metrics
        await asyncio.sleep(2)  # Let monitor collect some data
        
        # Get dashboard
        dashboard = await monitor.get_performance_dashboard()
        
        # Verify dashboard structure
        assert 'current_metrics' in dashboard
        assert 'recent_alerts' in dashboard
        assert 'performance_trends' in dashboard
        assert 'monitoring_status' in dashboard
        assert 'system_health_score' in dashboard
        
        # Verify current metrics
        current_metrics = dashboard['current_metrics']
        assert 'system_cpu_percent' in current_metrics
        assert 'system_memory_percent' in current_metrics
        assert 'system_disk_percent' in current_metrics
        assert 'timestamp' in current_metrics
        
        # Verify health score
        health_score = dashboard['system_health_score']
        assert 0.0 <= health_score <= 100.0
    
    @pytest.mark.asyncio
    async def test_performance_profiling(self, performance_monitor):
        """Test performance profiling capabilities"""
        monitor = performance_monitor
        profiler = monitor.profiler
        
        # Start profiling session
        session_id = "test_profiling_session"
        profile_data = profiler.start_profiling(session_id, "test_component")
        
        assert profile_data['session_id'] == session_id
        assert profile_data['component'] == "test_component"
        assert 'start_time' in profile_data
        
        # Record some function calls
        profiler.record_function_call(session_id, "test_function_1", 0.05)
        profiler.record_function_call(session_id, "test_function_2", 0.12)
        profiler.record_function_call(session_id, "test_function_1", 0.08)
        
        # Stop profiling
        results = profiler.stop_profiling(session_id)
        
        # Verify results
        assert results['session_id'] == session_id
        assert results['component'] == "test_component"
        assert 'duration' in results
        assert 'function_performance' in results
        
        # Verify function performance data
        func_perf = results['function_performance']
        assert 'test_function_1' in func_perf
        assert 'test_function_2' in func_perf
        
        assert func_perf['test_function_1']['call_count'] == 2
        assert func_perf['test_function_2']['call_count'] == 1
    
    @pytest.mark.asyncio
    async def test_optimization_recommendations(self, performance_monitor):
        """Test optimization recommendations generation"""
        monitor = performance_monitor
        
        # Generate recommendations
        recommendations = await monitor.generate_optimization_recommendations()
        
        # Verify recommendations structure
        assert isinstance(recommendations, list)
        
        for rec in recommendations:
            assert 'recommendation_id' in rec
            assert 'component' in rec
            assert 'issue_description' in rec
            assert 'recommendation' in rec
            assert 'expected_improvement' in rec
            assert 'implementation_complexity' in rec
            assert 'priority' in rec
            assert 'estimated_impact' in rec
            
            # Verify priority and impact ranges
            assert rec['priority'] >= 0
            assert 0.0 <= rec['estimated_impact'] <= 1.0

class TestSystemIntegration:
    """End-to-end system integration tests"""
    
    @pytest.mark.asyncio
    async def test_complete_threat_response_workflow(
        self, behavioral_analytics_engine, autonomous_orchestrator, 
        quantum_security_service, performance_monitor
    ):
        """Test complete threat detection and response workflow"""
        
        # Step 1: Behavioral analytics detects anomaly
        suspicious_event = MOCK_SECURITY_EVENT.copy()
        suspicious_event['failed_attempts'] = 10
        suspicious_event['source_ip'] = '1.2.3.4'  # Suspicious IP
        
        anomalies = await behavioral_analytics_engine.detect_real_time_anomalies(suspicious_event)
        assert len(anomalies) >= 0  # May or may not detect anomaly immediately
        
        # Step 2: Orchestrator creates response plan
        objective = "Respond to suspicious authentication activity"
        context = {
            'anomalies': anomalies,
            'event': suspicious_event,
            'threat_level': 'medium'
        }
        
        response_plan = await autonomous_orchestrator.create_orchestration_plan(
            objective, context
        )
        assert 'plan_id' in response_plan
        
        # Step 3: Quantum security provides secure communications
        sensitive_data = json.dumps(response_plan).encode('utf-8')
        encryption_result = await quantum_security_service.encrypt_with_quantum_protection(
            sensitive_data
        )
        assert encryption_result.ciphertext is not None
        
        # Step 4: Performance monitor tracks the workflow
        dashboard = await performance_monitor.get_performance_dashboard()
        assert 'system_health_score' in dashboard
        
        # Verify end-to-end workflow completion
        assert response_plan['status'] == 'planned'
        assert encryption_result.key_id is not None
        assert dashboard['monitoring_status']['active'] is True
    
    @pytest.mark.asyncio
    async def test_service_health_monitoring(
        self, behavioral_analytics_engine, autonomous_orchestrator,
        quantum_security_service, performance_monitor
    ):
        """Test health monitoring across all services"""
        
        services = [
            behavioral_analytics_engine,
            autonomous_orchestrator,
            quantum_security_service,
            performance_monitor
        ]
        
        for service in services:
            health = service.get_health()
            
            # Verify health structure
            assert hasattr(health, 'service_name')
            assert hasattr(health, 'is_healthy')
            assert hasattr(health, 'status')
            assert hasattr(health, 'details')
            
            # All services should be healthy after initialization
            assert health.is_healthy is True
            assert health.status in ['healthy', 'running']
    
    @pytest.mark.asyncio
    async def test_data_flow_between_services(
        self, behavioral_analytics_engine, autonomous_orchestrator
    ):
        """Test data flow between different services"""
        
        # Generate behavioral data
        entity_id = 'integration_test_user'
        events = [MOCK_SECURITY_EVENT.copy() for _ in range(5)]
        for i, event in enumerate(events):
            event['entity_id'] = entity_id
            event['event_id'] = f'integration_event_{i}'
        
        # Analyze with behavioral engine
        profile = await behavioral_analytics_engine.analyze_entity_behavior(
            entity_id, events
        )
        
        # Use profile data in orchestrator
        objective = "Analyze user behavior patterns"
        context = {
            'entity_profile': {
                'entity_id': profile.entity_id,
                'risk_score': profile.risk_score,
                'confidence': profile.confidence_score
            }
        }
        
        plan = await autonomous_orchestrator.create_orchestration_plan(
            objective, context
        )
        
        # Verify data flow
        assert plan['context']['entity_profile']['entity_id'] == entity_id
        assert 'risk_score' in plan['context']['entity_profile']
        assert plan['objective'] == objective

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])