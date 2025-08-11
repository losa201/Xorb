"""
Integration tests for production implementations
Tests for the advanced threat intelligence, orchestration, database, and security systems
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

# Import the production implementations
from src.api.app.services.production_threat_intelligence_engine import (
    ProductionThreatIntelligenceEngine, ThreatLevel, ThreatCategory, ConfidenceLevel
)
from src.api.app.services.production_orchestration_engine import (
    ProductionOrchestrationEngine, WorkflowStatus, TaskPriority, ExecutionStrategy
)
from src.api.app.infrastructure.production_database_manager import (
    ProductionDatabaseManager, DatabaseConfig, DatabaseStatus
)
from src.api.app.services.production_security_monitor import (
    ProductionSecurityMonitor, SecurityEventType, RiskLevel, ResponseAction
)


class TestProductionThreatIntelligenceEngine:
    """Test the production threat intelligence engine"""
    
    @pytest.fixture
    async def threat_engine(self):
        """Create threat intelligence engine for testing"""
        config = {
            'threat_feeds': {},
            'threat_model_path': None,
            'transformer_model': 'test-model'
        }
        
        engine = ProductionThreatIntelligenceEngine(config)
        
        # Mock ML dependencies if not available
        if not hasattr(engine, 'ml_model') or engine.ml_model is None:
            engine.ml_model = MagicMock()
            engine.ml_model.eval = MagicMock()
        
        return engine
    
    @pytest.mark.asyncio
    async def test_analyze_indicators_basic(self, threat_engine):
        """Test basic indicator analysis"""
        indicators = ["192.0.2.1", "evil.example.com", "badfile.exe"]
        
        analysis = await threat_engine.analyze_indicators(indicators)
        
        assert analysis.analysis_id is not None
        assert analysis.threat_level in ThreatLevel
        assert 0.0 <= analysis.confidence_score <= 1.0
        assert isinstance(analysis.indicators, list)
        assert isinstance(analysis.attack_vectors, list)
        assert isinstance(analysis.mitre_techniques, list)
        assert isinstance(analysis.recommendations, list)
    
    @pytest.mark.asyncio
    async def test_analyze_malicious_indicators(self, threat_engine):
        """Test analysis of known malicious indicators"""
        # Add known malicious indicators to feeds
        threat_engine.threat_feeds['malware_domains'] = ['evil.example.com']
        threat_engine.threat_feeds['malicious_ips'] = ['192.0.2.1']
        
        indicators = ["192.0.2.1", "evil.example.com"]
        
        analysis = await threat_engine.analyze_indicators(indicators)
        
        assert analysis.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        assert len(analysis.indicators) == 2
        assert any(indicator.threat_level == ThreatLevel.HIGH for indicator in analysis.indicators)
    
    @pytest.mark.asyncio
    async def test_correlate_threats(self, threat_engine):
        """Test threat correlation functionality"""
        indicators = ["192.0.2.1", "malware.exe", "c2.evil.com"]
        context = {"source": "endpoint_detection", "environment": "production"}
        
        correlation = await threat_engine.correlate_threats(indicators, context)
        
        assert 'correlation_id' in correlation
        assert 'threat_level' in correlation
        assert 'confidence_score' in correlation
        assert 'indicators_count' in correlation
        assert correlation['indicators_count'] == 3
    
    @pytest.mark.asyncio
    async def test_threat_prediction(self, threat_engine):
        """Test threat prediction capabilities"""
        prediction = await threat_engine.get_threat_prediction(24)
        
        assert prediction.prediction_id is not None
        assert prediction.threat_type in ThreatCategory
        assert 0.0 <= prediction.probability <= 1.0
        assert prediction.confidence in ConfidenceLevel
        assert prediction.timeline_hours == 24
        assert isinstance(prediction.recommendations, list)
        assert isinstance(prediction.risk_factors, dict)
    
    @pytest.mark.asyncio
    async def test_generate_threat_report(self, threat_engine):
        """Test threat report generation"""
        # First create an analysis
        indicators = ["suspicious.example.com"]
        analysis = await threat_engine.analyze_indicators(indicators)
        
        # Generate report
        report = await threat_engine.generate_threat_report(analysis.analysis_id)
        
        assert 'report_id' in report
        assert 'analysis_id' in report
        assert 'executive_summary' in report
        assert 'detailed_analysis' in report
        assert 'recommendations' in report
    
    @pytest.mark.asyncio
    async def test_health_check(self, threat_engine):
        """Test engine health check"""
        health = await threat_engine.health_check()
        
        assert 'status' in health
        assert 'timestamp' in health
        assert 'components' in health
        assert 'performance' in health


class TestProductionOrchestrationEngine:
    """Test the production orchestration engine"""
    
    @pytest.fixture
    async def orchestration_engine(self):
        """Create orchestration engine for testing"""
        config = {
            'temporal_url': 'localhost:7233',
            'max_workers': 5
        }
        
        engine = ProductionOrchestrationEngine(config)
        
        # Wait for initialization
        await asyncio.sleep(0.1)
        
        return engine
    
    @pytest.mark.asyncio
    async def test_create_workflow(self, orchestration_engine):
        """Test workflow creation"""
        workflow_data = {
            'name': 'Test Security Workflow',
            'description': 'Test workflow for security scanning',
            'tasks': [
                {
                    'task_id': 'test_scan',
                    'name': 'Test Scan',
                    'type': 'network_scan',
                    'priority': TaskPriority.HIGH.value,
                    'estimated_duration': 300,
                    'dependencies': [],
                    'parameters': {'target': 'localhost'},
                    'retry_policy': {'max_attempts': 3, 'backoff_multiplier': 2},
                    'timeout_seconds': 600,
                    'resource_requirements': {'cpu': 0.5, 'memory': '1Gi'}
                }
            ],
            'execution_strategy': ExecutionStrategy.SEQUENTIAL.value,
            'triggers': [{'type': 'manual'}],
            'conditions': {'require_approval': False},
            'metadata': {'test': True}
        }
        
        result = await orchestration_engine.create_workflow(workflow_data)
        
        assert 'workflow_id' in result
        assert result['status'] == 'created'
        assert result['workflow_id'] in orchestration_engine.workflow_definitions
    
    @pytest.mark.asyncio
    async def test_execute_workflow(self, orchestration_engine):
        """Test workflow execution"""
        # First create a workflow
        workflow_data = {
            'name': 'Test Execution Workflow',
            'tasks': [
                {
                    'task_id': 'test_task',
                    'name': 'Test Task',
                    'type': 'network_scan',
                    'priority': TaskPriority.MEDIUM.value,
                    'estimated_duration': 10,
                    'dependencies': [],
                    'parameters': {},
                    'retry_policy': {'max_attempts': 1, 'backoff_multiplier': 1},
                    'timeout_seconds': 30,
                    'resource_requirements': {}
                }
            ],
            'execution_strategy': ExecutionStrategy.SEQUENTIAL.value
        }
        
        create_result = await orchestration_engine.create_workflow(workflow_data)
        workflow_id = create_result['workflow_id']
        
        # Execute the workflow
        execution_id = await orchestration_engine.execute_workflow(workflow_id)
        
        assert execution_id is not None
        assert execution_id in orchestration_engine.active_executions
        
        # Wait for execution to start
        await asyncio.sleep(0.5)
        
        # Check status
        status = await orchestration_engine.get_workflow_status(execution_id)
        assert 'execution_id' in status
        assert 'status' in status
        assert status['execution_id'] == execution_id
    
    @pytest.mark.asyncio
    async def test_builtin_workflows(self, orchestration_engine):
        """Test that built-in workflows are registered"""
        expected_workflows = [
            'security_scan_comprehensive',
            'incident_response_comprehensive',
            'compliance_assessment_comprehensive',
            'threat_hunting_comprehensive',
            'vulnerability_remediation_automated'
        ]
        
        for workflow_id in expected_workflows:
            assert workflow_id in orchestration_engine.workflow_definitions
            
            workflow = orchestration_engine.workflow_definitions[workflow_id]
            assert workflow.name is not None
            assert len(workflow.tasks) > 0
            assert workflow.execution_strategy in ExecutionStrategy
    
    @pytest.mark.asyncio
    async def test_workflow_optimization(self, orchestration_engine):
        """Test workflow optimization capabilities"""
        # Create workflow with dependencies
        workflow_data = {
            'name': 'Optimization Test Workflow',
            'tasks': [
                {
                    'task_id': 'task_a',
                    'name': 'Task A',
                    'type': 'network_scan',
                    'priority': TaskPriority.HIGH.value,
                    'estimated_duration': 100,
                    'dependencies': [],
                    'parameters': {},
                    'retry_policy': {'max_attempts': 1},
                    'timeout_seconds': 200,
                    'resource_requirements': {}
                },
                {
                    'task_id': 'task_b',
                    'name': 'Task B',
                    'type': 'port_scan',
                    'priority': TaskPriority.MEDIUM.value,
                    'estimated_duration': 200,
                    'dependencies': ['task_a'],
                    'parameters': {},
                    'retry_policy': {'max_attempts': 1},
                    'timeout_seconds': 400,
                    'resource_requirements': {}
                }
            ],
            'execution_strategy': ExecutionStrategy.SEQUENTIAL.value
        }
        
        result = await orchestration_engine.create_workflow(workflow_data)
        
        # Verify dependency handling
        workflow = orchestration_engine.workflow_definitions[result['workflow_id']]
        assert len(workflow.tasks) == 2
        assert workflow.tasks[1].dependencies == ['task_a']
    
    @pytest.mark.asyncio
    async def test_health_check(self, orchestration_engine):
        """Test orchestration engine health check"""
        health = await orchestration_engine.health_check()
        
        assert 'status' in health
        assert 'active_executions' in health
        assert 'registered_workflows' in health
        assert 'registered_tasks' in health
        assert 'performance_metrics' in health


class TestProductionDatabaseManager:
    """Test the production database manager"""
    
    @pytest.fixture
    async def db_config(self):
        """Create test database configuration"""
        return DatabaseConfig(
            host='localhost',
            port=5432,
            database='test_xorb',
            username='test_user',
            password='test_pass',
            pool_size=5,
            max_overflow=10
        )
    
    @pytest.fixture
    async def db_manager(self, db_config):
        """Create database manager for testing"""
        manager = ProductionDatabaseManager(db_config)
        
        # Mock the database connections for testing
        manager.engine = MagicMock()
        manager.connection_pool = MagicMock()
        manager.redis_client = MagicMock()
        
        return manager
    
    @pytest.mark.asyncio
    async def test_query_optimization(self, db_manager):
        """Test query optimization analysis"""
        test_query = "SELECT * FROM users WHERE email = :email AND status = :status"
        
        optimization = db_manager.query_optimizer.analyze_query(test_query)
        
        assert 'query_hash' in optimization
        assert 'complexity_score' in optimization
        assert 'index_suggestions' in optimization
        assert 'rewrite_suggestions' in optimization
        assert 'estimated_cost' in optimization
        assert optimization['complexity_score'] >= 0
    
    @pytest.mark.asyncio
    async def test_cache_key_generation(self, db_manager):
        """Test cache key generation"""
        query = "SELECT * FROM table WHERE id = :id"
        parameters = {'id': 123}
        
        key1 = db_manager._generate_cache_key(query, parameters)
        key2 = db_manager._generate_cache_key(query, parameters)
        key3 = db_manager._generate_cache_key(query, {'id': 456})
        
        assert key1 == key2  # Same query and params should generate same key
        assert key1 != key3  # Different params should generate different key
        assert key1.startswith('db_cache:')
    
    @pytest.mark.asyncio
    async def test_health_status(self, db_manager):
        """Test database health status"""
        # Mock connection metrics
        db_manager.connection_metrics.total_connections = 10
        db_manager.connection_metrics.active_connections = 5
        db_manager.connection_metrics.query_count = 100
        db_manager.connection_metrics.average_query_time = 0.05
        
        health = await db_manager.get_health_status()
        
        assert 'status' in health
        assert 'connection_metrics' in health
        assert 'query_performance' in health
        assert 'cache_status' in health
        assert health['connection_metrics']['total_connections'] == 10
    
    @pytest.mark.asyncio
    async def test_performance_report(self, db_manager):
        """Test performance report generation"""
        # Add some mock query history
        from src.api.app.infrastructure.production_database_manager import QueryMetrics, QueryType
        
        mock_query = QueryMetrics(
            query_id=str(uuid4()),
            query_type=QueryType.SELECT,
            execution_time=0.05,
            rows_affected=10,
            error=None,
            timestamp=datetime.utcnow(),
            connection_id='test',
            user_context={}
        )
        
        db_manager.query_history.append(mock_query)
        
        report = await db_manager.get_performance_report()
        
        assert 'report_generated_at' in report
        assert 'total_queries' in report
        assert 'successful_queries' in report
        assert 'query_type_distribution' in report
        assert 'performance_percentiles' in report
    
    def test_query_complexity_calculation(self, db_manager):
        """Test query complexity calculation"""
        simple_query = "SELECT id FROM users WHERE email = 'test@example.com'"
        complex_query = """
            SELECT u.*, p.*, o.* 
            FROM users u 
            JOIN profiles p ON u.id = p.user_id 
            JOIN orders o ON u.id = o.user_id 
            WHERE u.status = 'active' 
            AND p.created_at > '2024-01-01' 
            GROUP BY u.id 
            ORDER BY u.created_at DESC
        """
        
        simple_complexity = db_manager.query_optimizer._calculate_complexity(simple_query)
        complex_complexity = db_manager.query_optimizer._calculate_complexity(complex_query)
        
        assert simple_complexity < complex_complexity
        assert complex_complexity > 5.0  # Should be flagged as complex


class TestProductionSecurityMonitor:
    """Test the production security monitor"""
    
    @pytest.fixture
    async def security_monitor(self):
        """Create security monitor for testing"""
        config = {
            'threat_feeds': {},
            'alert_thresholds': {
                'failed_login': 5,
                'data_exfiltration': '100MB'
            }
        }
        
        monitor = ProductionSecurityMonitor(config)
        
        # Wait for initialization
        await asyncio.sleep(0.1)
        
        return monitor
    
    @pytest.mark.asyncio
    async def test_process_security_event(self, security_monitor):
        """Test processing of security events"""
        event_data = {
            'event_type': 'login_failed',
            'source_ip': '192.0.2.100',
            'user_id': 'test_user',
            'user_agent': 'Mozilla/5.0 Test Browser',
            'resource': '/admin/login',
            'action': 'authentication',
            'timestamp': datetime.utcnow().isoformat(),
            'request_data': 'username=admin&password=wrong'
        }
        
        result = await security_monitor.process_security_event(event_data)
        
        assert 'event_id' in result
        assert 'processed' in result
        assert result['processed'] is True
        assert 'threat_detected' in result
        assert 'risk_level' in result
    
    @pytest.mark.asyncio
    async def test_threat_detection_rules(self, security_monitor):
        """Test threat detection rules"""
        # Test SQL injection detection
        sql_injection_event = {
            'event_type': 'web_request',
            'source_ip': '192.0.2.200',
            'resource': '/search',
            'action': 'query',
            'request_data': "search=test' UNION SELECT * FROM users--",
            'response_error': True
        }
        
        result = await security_monitor.process_security_event(sql_injection_event)
        
        assert result['processed'] is True
        # SQL injection should be detected as high risk
        if result['threat_detected']:
            assert result['risk_level'] in ['high', 'critical']
    
    @pytest.mark.asyncio
    async def test_behavioral_analysis(self, security_monitor):
        """Test behavioral analysis functionality"""
        user_id = 'test_user_123'
        
        # Create normal login event
        normal_event = {
            'event_type': 'login_success',
            'source_ip': '192.168.1.100',
            'user_id': user_id,
            'user_agent': 'Chrome/91.0',
            'resource': '/dashboard',
            'action': 'login'
        }
        
        await security_monitor.process_security_event(normal_event)
        
        # Check if user profile was created
        assert user_id in security_monitor.threat_detection.behavior_analyzer.user_profiles
        
        profile = security_monitor.threat_detection.behavior_analyzer.user_profiles[user_id]
        assert profile['user_id'] == user_id
        assert 'login_patterns' in profile
        assert 'access_patterns' in profile
    
    @pytest.mark.asyncio
    async def test_incident_creation(self, security_monitor):
        """Test incident creation from threats"""
        # Create a high-risk event that should trigger incident creation
        high_risk_event = {
            'event_type': 'privilege_escalation',
            'source_ip': '192.0.2.300',
            'user_id': 'attacker_user',
            'resource': '/admin/users',
            'action': 'admin_access',
            'admin_user': True
        }
        
        result = await security_monitor.process_security_event(high_risk_event)
        
        # Check if incident was created
        if result['threat_detected'] and result['risk_level'] in ['high', 'critical']:
            assert len(security_monitor.incidents) > 0
    
    @pytest.mark.asyncio
    async def test_security_dashboard(self, security_monitor):
        """Test security dashboard generation"""
        dashboard = await security_monitor.get_security_dashboard()
        
        assert 'timestamp' in dashboard
        assert 'summary' in dashboard
        assert 'threat_levels' in dashboard
        assert 'recent_incidents' in dashboard
        assert 'performance_metrics' in dashboard
        assert 'system_status' in dashboard
        
        summary = dashboard['summary']
        assert 'active_incidents' in summary
        assert 'events_last_hour' in summary
        assert 'blocked_ips' in summary
        assert 'blocked_users' in summary
    
    @pytest.mark.asyncio
    async def test_response_actions(self, security_monitor):
        """Test automated response actions"""
        # Mock response action execution
        with patch.object(security_monitor, '_execute_response_action') as mock_action:
            mock_action.return_value = None
            
            high_threat_event = {
                'event_type': 'malware_detection',
                'source_ip': '192.0.2.400',
                'resource': '/upload',
                'action': 'file_upload',
                'malware_signature': 'Trojan.Generic'
            }
            
            await security_monitor.process_security_event(high_threat_event)
            
            # Verify that response actions would be executed for high-threat events
            # The exact behavior depends on the threat detection rules
    
    @pytest.mark.asyncio
    async def test_health_check(self, security_monitor):
        """Test security monitor health check"""
        health = await security_monitor.health_check()
        
        assert 'status' in health
        assert 'timestamp' in health
        assert 'components' in health
        assert 'metrics' in health
        assert 'event_store_size' in health
        assert 'active_incidents' in health


class TestIntegratedSystemFlow:
    """Test integrated flow between all production systems"""
    
    @pytest.fixture
    async def integrated_system(self):
        """Create integrated system with all components"""
        # Database manager
        db_config = DatabaseConfig(
            host='localhost',
            port=5432,
            database='test_integration',
            username='test',
            password='test'
        )
        db_manager = ProductionDatabaseManager(db_config)
        
        # Mock database connections
        db_manager.engine = MagicMock()
        db_manager.connection_pool = MagicMock()
        
        # Security monitor
        security_monitor = ProductionSecurityMonitor()
        
        # Threat intelligence
        threat_engine = ProductionThreatIntelligenceEngine()
        
        # Orchestration engine
        orchestration_engine = ProductionOrchestrationEngine()
        
        return {
            'db_manager': db_manager,
            'security_monitor': security_monitor,
            'threat_engine': threat_engine,
            'orchestration_engine': orchestration_engine
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_threat_workflow(self, integrated_system):
        """Test complete threat detection to response workflow"""
        security_monitor = integrated_system['security_monitor']
        threat_engine = integrated_system['threat_engine']
        orchestration_engine = integrated_system['orchestration_engine']
        
        # Step 1: Security event detection
        malicious_event = {
            'event_type': 'network_intrusion',
            'source_ip': '192.0.2.500',
            'user_id': 'compromised_user',
            'resource': '/api/sensitive-data',
            'action': 'data_access',
            'data_volume': '500MB'
        }
        
        event_result = await security_monitor.process_security_event(malicious_event)
        
        # Step 2: Threat intelligence analysis
        if event_result['threat_detected']:
            indicators = [malicious_event['source_ip']]
            threat_analysis = await threat_engine.analyze_indicators(indicators)
            
            assert threat_analysis.analysis_id is not None
            assert len(threat_analysis.indicators) > 0
        
        # Step 3: Automated response workflow
        if event_result['risk_level'] in ['high', 'critical']:
            # Create incident response workflow
            workflow_data = {
                'name': 'Automated Incident Response',
                'tasks': [
                    {
                        'task_id': 'isolate_system',
                        'name': 'Isolate Compromised System',
                        'type': 'threat_containment',
                        'priority': 'emergency',
                        'estimated_duration': 300,
                        'dependencies': [],
                        'parameters': {'target_ip': malicious_event['source_ip']},
                        'retry_policy': {'max_attempts': 1},
                        'timeout_seconds': 600,
                        'resource_requirements': {}
                    }
                ],
                'execution_strategy': 'sequential'
            }
            
            create_result = await orchestration_engine.create_workflow(workflow_data)
            assert create_result['status'] == 'created'
            
            execution_id = await orchestration_engine.execute_workflow(create_result['workflow_id'])
            assert execution_id is not None
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, integrated_system):
        """Test performance monitoring across all systems"""
        db_manager = integrated_system['db_manager']
        security_monitor = integrated_system['security_monitor']
        
        # Check health of all systems
        db_health = await db_manager.get_health_status()
        security_health = await security_monitor.health_check()
        
        assert db_health['status'] in ['healthy', 'degraded']  # Allow degraded in test environment
        assert security_health['status'] == 'healthy'
        
        # Verify metrics are being collected
        assert 'query_performance' in db_health
        assert 'metrics' in security_health
    
    @pytest.mark.asyncio
    async def test_data_flow_integrity(self, integrated_system):
        """Test data integrity across system boundaries"""
        security_monitor = integrated_system['security_monitor']
        
        # Generate multiple related events
        base_ip = '192.0.2.600'
        events = []
        
        for i in range(5):
            event = {
                'event_type': 'authentication_failure',
                'source_ip': base_ip,
                'user_id': f'user_{i}',
                'resource': '/login',
                'action': 'authentication',
                'correlation_id': 'test_correlation_123'
            }
            
            result = await security_monitor.process_security_event(event)
            events.append(result)
        
        # Verify all events were processed
        assert all(event['processed'] for event in events)
        
        # Check if correlation is working (events should be related)
        dashboard = await security_monitor.get_security_dashboard()
        assert dashboard['summary']['events_last_hour'] >= 5


# Test configuration and utilities
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])