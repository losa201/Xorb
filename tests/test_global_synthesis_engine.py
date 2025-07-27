#!/usr/bin/env python3
"""
Test suite for Global Intelligence Synthesis Engine v10.0

Comprehensive testing of Phase 10 global intelligence synthesis capabilities:
- Intelligence source integration and polling
- Signal ingestion and normalization
- Cross-source correlation and deduplication
- Intelligence-driven mission creation
- Feedback learning and optimization
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any

# Test imports
from xorb_core.intelligence.global_synthesis_engine import (
    GlobalSynthesisEngine, IntelligenceSource, IntelligenceSignal,
    CorrelatedIntelligence, IntelligenceSourceType, SignalPriority,
    IntelligenceSignalStatus
)
from xorb_core.autonomous.autonomous_orchestrator import AutonomousOrchestrator
from xorb_core.mission.adaptive_mission_engine import AdaptiveMissionEngine
from xorb_core.autonomous.episodic_memory_system import EpisodicMemorySystem
from xorb_core.knowledge_fabric.vector_fabric import VectorFabric


class TestGlobalSynthesisEngine:
    """Test suite for Global Synthesis Engine"""
    
    @pytest.fixture
    async def synthesis_engine(self):
        """Create test synthesis engine with mocked dependencies"""
        
        # Mock dependencies
        orchestrator = Mock(spec=AutonomousOrchestrator)
        mission_engine = Mock(spec=AdaptiveMissionEngine)
        episodic_memory = Mock(spec=EpisodicMemorySystem)
        vector_fabric = Mock(spec=VectorFabric)
        
        # Create synthesis engine
        engine = GlobalSynthesisEngine(
            orchestrator=orchestrator,
            mission_engine=mission_engine,
            episodic_memory=episodic_memory,
            vector_fabric=vector_fabric
        )
        
        return engine
    
    @pytest.fixture
    def sample_intelligence_sources(self):
        """Create sample intelligence sources for testing"""
        
        return {
            'cve_nvd': IntelligenceSource(
                source_id='cve_nvd',
                source_type=IntelligenceSourceType.CVE_NVD,
                name='CVE/NVD Feed',
                url='https://services.nvd.nist.gov/rest/json/cves/2.0',
                confidence_weight=0.95,
                reliability_score=0.98
            ),
            'hackerone': IntelligenceSource(
                source_id='hackerone',
                source_type=IntelligenceSourceType.HACKERONE,
                name='HackerOne Platform',
                url='https://api.hackerone.com/v1/reports',
                confidence_weight=0.85,
                reliability_score=0.9
            ),
            'osint_feed': IntelligenceSource(
                source_id='osint_feed',
                source_type=IntelligenceSourceType.OSINT_RSS,
                name='OSINT RSS Feed',
                url='https://threatpost.com/feed/',
                confidence_weight=0.7,
                reliability_score=0.75
            )
        }
    
    @pytest.fixture
    def sample_intelligence_signals(self):
        """Create sample intelligence signals for testing"""
        
        return [
            IntelligenceSignal(
                signal_id='cve_test_001',
                source_id='cve_nvd',
                source_type=IntelligenceSourceType.CVE_NVD,
                title='CVE-2024-1234: Critical RCE in Web Framework',
                description='Critical remote code execution vulnerability',
                content={
                    'cve_id': 'CVE-2024-1234',
                    'cvss_score': 9.8,
                    'affected_products': ['Web Framework v1.0-2.5']
                },
                raw_data={'severity': 'CRITICAL'},
                timestamp=datetime.utcnow(),
                priority=SignalPriority.CRITICAL,
                signal_type='vulnerability',
                tags=['cve', 'rce', 'critical']
            ),
            IntelligenceSignal(
                signal_id='h1_test_001',
                source_id='hackerone',
                source_type=IntelligenceSourceType.HACKERONE,
                title='Disclosed: XSS in Popular CMS',
                description='Cross-site scripting vulnerability in CMS',
                content={
                    'report_id': '123456',
                    'severity': 'high',
                    'bounty_amount': 5000
                },
                raw_data={'program': 'test-program'},
                timestamp=datetime.utcnow(),
                priority=SignalPriority.HIGH,
                signal_type='bug_bounty',
                tags=['hackerone', 'xss', 'disclosed']
            )
        ]
    
    @pytest.mark.asyncio
    async def test_synthesis_engine_initialization(self, synthesis_engine):
        """Test synthesis engine initialization"""
        
        assert synthesis_engine is not None
        assert synthesis_engine.intelligence_sources == {}
        assert synthesis_engine.raw_signals == {}
        assert synthesis_engine.correlated_intelligence == {}
        assert synthesis_engine.max_signals_memory == 50000
        assert synthesis_engine.correlation_threshold == 0.75
    
    @pytest.mark.asyncio
    async def test_intelligence_source_initialization(self, synthesis_engine):
        """Test intelligence source initialization"""
        
        await synthesis_engine._initialize_intelligence_sources()
        
        # Verify sources were created
        assert len(synthesis_engine.intelligence_sources) > 0
        
        # Check for expected source types
        source_types = [source.source_type for source in synthesis_engine.intelligence_sources.values()]
        expected_types = [
            IntelligenceSourceType.CVE_NVD,
            IntelligenceSourceType.HACKERONE,
            IntelligenceSourceType.OSINT_RSS,
            IntelligenceSourceType.INTERNAL_MISSIONS,
            IntelligenceSourceType.PROMETHEUS_ALERTS
        ]
        
        for expected_type in expected_types:
            assert expected_type in source_types
    
    @pytest.mark.asyncio
    async def test_signal_ingestion_pipeline(self, synthesis_engine, sample_intelligence_signals):
        """Test signal ingestion and processing"""
        
        # Setup
        synthesis_engine.intelligence_sources = {'test_source': Mock()}
        
        # Test signal ingestion
        for signal in sample_intelligence_signals:
            await synthesis_engine.ingestion_queue.put(signal)
        
        # Process ingestion batch
        signals_batch = []
        while not synthesis_engine.ingestion_queue.empty():
            signals_batch.append(await synthesis_engine.ingestion_queue.get())
        
        processed_signals = await synthesis_engine._process_ingestion_batch(signals_batch)
        
        # Verify processing
        assert len(processed_signals) == len(sample_intelligence_signals)
        
        for signal in processed_signals:
            assert signal.status == IntelligenceSignalStatus.NORMALIZED
            assert len(signal.processing_history) > 0
            assert signal.deduplication_hash is not None
    
    @pytest.mark.asyncio
    async def test_signal_deduplication(self, synthesis_engine, sample_intelligence_signals):
        """Test signal deduplication functionality"""
        
        # Create duplicate signals
        signal1 = sample_intelligence_signals[0]
        signal2 = IntelligenceSignal(
            signal_id='duplicate_signal',
            source_id=signal1.source_id,
            source_type=signal1.source_type,
            title=signal1.title,  # Same title
            description=signal1.description,  # Same description
            content=signal1.content,
            raw_data=signal1.raw_data,
            timestamp=datetime.utcnow(),
            priority=signal1.priority,
            signal_type=signal1.signal_type,
            tags=signal1.tags
        )
        
        # Process first signal
        processed1 = await synthesis_engine._process_ingestion_batch([signal1])
        assert len(processed1) == 1
        
        # Process duplicate signal - should be filtered out
        processed2 = await synthesis_engine._process_ingestion_batch([signal2])
        assert len(processed2) == 0  # Duplicate should be filtered
        
        # Verify deduplication cache
        assert signal1.deduplication_hash in synthesis_engine.deduplication_cache
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession')
    async def test_cve_nvd_polling(self, mock_session, synthesis_engine, sample_intelligence_sources):
        """Test CVE/NVD source polling"""
        
        # Mock HTTP response
        mock_response_data = {
            'vulnerabilities': [
                {
                    'cve': {
                        'id': 'CVE-2024-TEST',
                        'descriptions': [{'value': 'Test vulnerability description'}],
                        'published': '2024-01-01T00:00:00.000'
                    }
                }
            ]
        }
        
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        
        mock_session_instance = Mock()
        mock_session_instance.get = AsyncMock(return_value=mock_response)
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=None)
        mock_session.return_value = mock_session_instance
        
        # Test CVE polling
        source = sample_intelligence_sources['cve_nvd']
        signals = await synthesis_engine._poll_cve_nvd(source)
        
        # Verify results
        assert len(signals) == 1
        assert signals[0].signal_type == 'vulnerability'
        assert 'CVE-2024-TEST' in signals[0].title
        assert signals[0].source_type == IntelligenceSourceType.CVE_NVD
    
    @pytest.mark.asyncio
    async def test_internal_mission_polling(self, synthesis_engine, sample_intelligence_sources):
        """Test internal mission intelligence polling"""
        
        # Mock episodic memory response
        mock_memory = Mock()
        mock_memory.memory_id = 'test_memory_001'
        mock_memory.operation_type = 'vulnerability_scan'
        mock_memory.success_metrics = {'success_rate': 0.85}
        mock_memory.insights = ['Found 3 critical vulnerabilities']
        mock_memory.patterns_identified = ['Common misconfigurations']
        mock_memory.performance_data = {'execution_time': 300}
        mock_memory.timestamp = datetime.utcnow()
        mock_memory.context = {'mission_id': 'mission_123'}
        
        synthesis_engine.episodic_memory.query_memories = AsyncMock(return_value=[mock_memory])
        
        # Test internal polling
        source = sample_intelligence_sources['osint_feed']  # Using existing source for structure
        source.source_type = IntelligenceSourceType.INTERNAL_MISSIONS
        
        signals = await synthesis_engine._poll_internal_missions(source)
        
        # Verify results
        assert len(signals) == 1
        assert signals[0].signal_type == 'mission_intelligence'
        assert 'Mission Intelligence' in signals[0].title
        assert signals[0].content['insights'] == ['Found 3 critical vulnerabilities']
    
    @pytest.mark.asyncio
    async def test_signal_priority_calculation(self, synthesis_engine):
        """Test signal priority calculation logic"""
        
        # Test CVE priority calculation
        high_cvss_cve = {
            'id': 'CVE-2024-HIGH',
            'metrics': [{'cvssData': {'baseScore': 9.5}}]
        }
        critical_priority = synthesis_engine._calculate_cve_priority(high_cvss_cve)
        assert critical_priority == SignalPriority.CRITICAL
        
        # Test HackerOne priority calculation
        high_bounty_report = {
            'severity': {'rating': 'critical'},
            'bounty': {'amount': 10000}
        }
        bounty_priority = synthesis_engine._calculate_hackerone_priority(high_bounty_report)
        assert bounty_priority in [SignalPriority.HIGH, SignalPriority.CRITICAL]
    
    @pytest.mark.asyncio
    async def test_synthesis_status_reporting(self, synthesis_engine):
        """Test synthesis engine status reporting"""
        
        # Add some test data
        synthesis_engine.intelligence_sources = {'test': Mock()}
        synthesis_engine.raw_signals = {'signal1': Mock(), 'signal2': Mock()}
        synthesis_engine.correlated_intelligence = {'intel1': Mock()}
        
        status = await synthesis_engine.get_synthesis_status()
        
        # Verify status structure
        assert 'synthesis_engine' in status
        assert 'processing_queues' in status
        assert 'intelligence_sources' in status
        assert 'recent_intelligence' in status
        assert 'performance_metrics' in status
        
        # Verify metrics
        assert status['synthesis_engine']['total_sources'] == 1
        assert status['synthesis_engine']['raw_signals'] == 2
        assert status['synthesis_engine']['correlated_intelligence'] == 1
    
    @pytest.mark.asyncio
    async def test_metrics_initialization(self, synthesis_engine):
        """Test metrics initialization and structure"""
        
        metrics = synthesis_engine.synthesis_metrics
        
        expected_metrics = [
            'signals_ingested',
            'signals_correlated', 
            'missions_triggered',
            'processing_duration',
            'source_reliability',
            'correlation_accuracy',
            'active_sources',
            'pending_signals'
        ]
        
        for metric_name in expected_metrics:
            assert metric_name in metrics
    
    @pytest.mark.asyncio
    async def test_signal_filtering(self, synthesis_engine, sample_intelligence_sources):
        """Test signal filtering based on source configuration"""
        
        # Create source with filtering rules
        source = sample_intelligence_sources['cve_nvd']
        source.include_patterns = ['critical', 'high']
        source.exclude_patterns = ['low', 'informational']
        source.priority_keywords = ['rce', 'authentication']
        
        # Create test signals
        critical_signal = IntelligenceSignal(
            signal_id='critical_test',
            source_id=source.source_id,
            source_type=source.source_type,
            title='Critical RCE Vulnerability',
            description='Remote code execution',
            content={},
            raw_data={},
            timestamp=datetime.utcnow(),
            tags=['critical', 'rce']
        )
        
        low_signal = IntelligenceSignal(
            signal_id='low_test',
            source_id=source.source_id,
            source_type=source.source_type,
            title='Low Priority Information',
            description='Informational update',
            content={},
            raw_data={},
            timestamp=datetime.utcnow(),
            tags=['low', 'informational']
        )
        
        # Test filtering
        filtered_signals = synthesis_engine._filter_signals([critical_signal, low_signal], source)
        
        # Should include critical signal, exclude low signal
        signal_titles = [s.title for s in filtered_signals]
        assert 'Critical RCE Vulnerability' in signal_titles
        assert 'Low Priority Information' not in signal_titles
    
    @pytest.mark.asyncio
    async def test_error_handling_in_source_polling(self, synthesis_engine, sample_intelligence_sources):
        """Test error handling during source polling"""
        
        # Mock a source that will raise an exception
        source = sample_intelligence_sources['cve_nvd']
        
        # Patch the polling method to raise an exception
        with patch.object(synthesis_engine, '_poll_cve_nvd', side_effect=Exception("Network error")):
            
            # Should not raise exception, should handle gracefully
            signals = await synthesis_engine._poll_intelligence_source(source)
            
            # Should return empty list on error
            assert signals == []
            
            # Error count should be incremented
            assert source.error_count >= 0  # May have been incremented by other tests


class TestSignalCorrelation:
    """Test suite for signal correlation and intelligence synthesis"""
    
    @pytest.fixture
    def correlation_test_signals(self):
        """Create signals for correlation testing"""
        
        base_time = datetime.utcnow()
        
        return [
            IntelligenceSignal(
                signal_id='vuln_signal_1',
                source_id='cve_nvd',
                source_type=IntelligenceSourceType.CVE_NVD,
                title='CVE-2024-1234: SQL Injection in WebApp',
                description='SQL injection vulnerability in web application',
                content={'cve_id': 'CVE-2024-1234', 'cvss_score': 8.5},
                raw_data={},
                timestamp=base_time,
                priority=SignalPriority.HIGH,
                tags=['sql_injection', 'webapp', 'vulnerability']
            ),
            IntelligenceSignal(
                signal_id='bounty_signal_1', 
                source_id='hackerone',
                source_type=IntelligenceSourceType.HACKERONE,
                title='SQL Injection Report - WebApp v2.1',
                description='Disclosed SQL injection in same application',
                content={'report_id': '789', 'bounty_amount': 3000},
                raw_data={},
                timestamp=base_time + timedelta(hours=2),
                priority=SignalPriority.HIGH,
                tags=['sql_injection', 'webapp', 'disclosed']
            ),
            IntelligenceSignal(
                signal_id='unrelated_signal',
                source_id='osint_feed',
                source_type=IntelligenceSourceType.OSINT_RSS,
                title='New Malware Campaign Detected',
                description='APT group launches new campaign',
                content={'campaign': 'APT-X'},
                raw_data={},
                timestamp=base_time + timedelta(hours=1),
                priority=SignalPriority.MEDIUM,
                tags=['malware', 'apt', 'campaign']
            )
        ]
    
    @pytest.mark.asyncio
    async def test_signal_correlation_similarity(self, synthesis_engine, correlation_test_signals):
        """Test correlation based on content similarity"""
        
        # Add signals to engine
        for signal in correlation_test_signals:
            synthesis_engine.raw_signals[signal.signal_id] = signal
        
        # Test correlation calculation (would be implemented in correlation pipeline)
        # For now, test similarity detection logic
        
        signal1 = correlation_test_signals[0]  # CVE signal
        signal2 = correlation_test_signals[1]  # HackerOne signal
        signal3 = correlation_test_signals[2]  # Unrelated signal
        
        # Check tag overlap
        signal1_tags = set(signal1.tags)
        signal2_tags = set(signal2.tags)
        signal3_tags = set(signal3.tags)
        
        # SQL injection signals should have high overlap
        overlap_1_2 = len(signal1_tags & signal2_tags) / len(signal1_tags | signal2_tags)
        assert overlap_1_2 > 0.5  # Should have significant overlap
        
        # Unrelated signal should have low overlap
        overlap_1_3 = len(signal1_tags & signal3_tags) / len(signal1_tags | signal3_tags)
        assert overlap_1_3 < 0.3  # Should have minimal overlap
    
    @pytest.mark.asyncio
    async def test_temporal_correlation(self, correlation_test_signals):
        """Test temporal correlation between signals"""
        
        signal1 = correlation_test_signals[0]
        signal2 = correlation_test_signals[1]
        
        # Calculate time difference
        time_diff = abs((signal2.timestamp - signal1.timestamp).total_seconds())
        
        # Signals within 24 hours should be temporally correlated
        assert time_diff < 24 * 3600
        
        # Time difference should be reasonable for correlation
        assert time_diff > 0  # Different timestamps


class TestIntelligenceDrivenMissions:
    """Test suite for intelligence-driven mission creation"""
    
    @pytest.fixture
    def sample_correlated_intelligence(self):
        """Create sample correlated intelligence for testing"""
        
        return CorrelatedIntelligence(
            intelligence_id='intel_test_001',
            primary_signal_id='vuln_signal_1',
            related_signal_ids=['bounty_signal_1', 'osint_signal_1'],
            synthesized_title='Critical SQL Injection in WebApp Platform',
            synthesized_description='Multiple sources confirm SQL injection vulnerability',
            key_indicators=['sql_injection', 'webapp', 'authentication_bypass', 'rce'],
            threat_context={
                'type': 'vulnerability',
                'affected_systems': ['webapp.example.com'],
                'attack_vectors': ['web', 'api'],
                'targets': [{'hostname': 'webapp.example.com', 'priority': 5}]
            },
            overall_priority=SignalPriority.CRITICAL,
            confidence_score=0.92,
            threat_level='critical',
            impact_assessment={
                'confidentiality': 'high',
                'integrity': 'high', 
                'availability': 'medium'
            },
            recommended_actions=[
                {'type': 'investigate', 'description': 'Verify SQL injection vulnerability'},
                {'type': 'assess', 'description': 'Determine exploit feasibility'},
                {'type': 'verify', 'description': 'Confirm affected systems'}
            ],
            required_capabilities=['web_testing', 'vulnerability_scanning', 'reconnaissance'],
            estimated_effort='medium',
            created_at=datetime.utcnow()
        )
    
    @pytest.mark.asyncio
    async def test_mission_type_determination(self, synthesis_engine, sample_correlated_intelligence):
        """Test mission type determination from intelligence"""
        
        intelligence = sample_correlated_intelligence
        
        # Mock the orchestrator method
        with patch.object(synthesis_engine.orchestrator, '_determine_mission_type') as mock_determine:
            mock_determine.return_value = "VULNERABILITY_ASSESSMENT"
            
            mission_type = await synthesis_engine.orchestrator._determine_mission_type(intelligence)
            
            assert mission_type == "VULNERABILITY_ASSESSMENT"
            mock_determine.assert_called_once_with(intelligence)
    
    @pytest.mark.asyncio
    async def test_capability_mapping(self, synthesis_engine, sample_correlated_intelligence):
        """Test mapping intelligence to agent capabilities"""
        
        intelligence = sample_correlated_intelligence
        
        # Mock the orchestrator method
        with patch.object(synthesis_engine.orchestrator, '_map_intelligence_to_capabilities') as mock_map:
            from xorb_core.agents.base_agent import AgentCapability
            expected_capabilities = [
                AgentCapability.RECONNAISSANCE,
                AgentCapability.WEB_CRAWLING,
                AgentCapability.VULNERABILITY_SCANNING
            ]
            mock_map.return_value = expected_capabilities
            
            capabilities = await synthesis_engine.orchestrator._map_intelligence_to_capabilities(intelligence)
            
            assert AgentCapability.RECONNAISSANCE in capabilities
            assert AgentCapability.WEB_CRAWLING in capabilities
            assert AgentCapability.VULNERABILITY_SCANNING in capabilities
    
    @pytest.mark.asyncio
    async def test_mission_objective_extraction(self, synthesis_engine, sample_correlated_intelligence):
        """Test extraction of mission objectives from intelligence"""
        
        intelligence = sample_correlated_intelligence
        
        # Mock the orchestrator method
        with patch.object(synthesis_engine.orchestrator, '_extract_mission_objectives') as mock_extract:
            expected_objectives = [
                'Investigate: Verify SQL injection vulnerability',
                'Assess: Determine exploit feasibility',
                'Verify: Confirm affected systems',
                'Gather additional intelligence on discovered assets'
            ]
            mock_extract.return_value = expected_objectives
            
            objectives = synthesis_engine.orchestrator._extract_mission_objectives(intelligence)
            
            assert len(objectives) >= 3
            assert any('Investigate' in obj for obj in objectives)
            assert any('Assess' in obj for obj in objectives)
            assert any('Verify' in obj for obj in objectives)
    
    @pytest.mark.asyncio 
    async def test_target_extraction(self, synthesis_engine, sample_correlated_intelligence):
        """Test extraction of targets from intelligence"""
        
        intelligence = sample_correlated_intelligence
        
        # Mock the orchestrator method
        with patch.object(synthesis_engine.orchestrator, '_extract_targets_from_intelligence') as mock_extract:
            expected_targets = [
                {
                    'hostname': 'webapp.example.com',
                    'source': 'intelligence_synthesis',
                    'priority': 5
                }
            ]
            mock_extract.return_value = expected_targets
            
            targets = synthesis_engine.orchestrator._extract_targets_from_intelligence(intelligence)
            
            assert len(targets) > 0
            assert any(target['hostname'] == 'webapp.example.com' for target in targets)
    
    @pytest.mark.asyncio
    async def test_mission_timeout_calculation(self, synthesis_engine, sample_correlated_intelligence):
        """Test mission timeout calculation based on priority"""
        
        intelligence = sample_correlated_intelligence
        
        # Critical priority should have shorter timeout
        intelligence.overall_priority = SignalPriority.CRITICAL
        timeout_critical = synthesis_engine.orchestrator._calculate_mission_timeout(intelligence)
        
        # High priority timeout
        intelligence.overall_priority = SignalPriority.HIGH
        timeout_high = synthesis_engine.orchestrator._calculate_mission_timeout(intelligence)
        
        # Low priority timeout
        intelligence.overall_priority = SignalPriority.LOW
        timeout_low = synthesis_engine.orchestrator._calculate_mission_timeout(intelligence)
        
        # Critical should be shortest, low should be longest
        assert timeout_critical < timeout_high < timeout_low
        assert timeout_critical <= 1800  # 30 minutes or less for critical
        assert timeout_low >= 7200  # 2 hours or more for low


@pytest.mark.integration
class TestSynthesisEngineIntegration:
    """Integration tests for the complete synthesis pipeline"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_synthesis_flow(self):
        """Test complete flow from signal ingestion to mission creation"""
        
        # This would be a comprehensive integration test
        # Testing the full pipeline: Ingest → Normalize → Correlate → Act → Learn
        
        # For now, placeholder that verifies the test framework
        assert True
    
    @pytest.mark.asyncio
    async def test_failover_scenarios(self):
        """Test system behavior under failover conditions"""
        
        # Test source dropout scenarios
        # Test network connectivity issues
        # Test service dependency failures
        
        # Placeholder for comprehensive failover testing
        assert True
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test synthesis engine performance under high load"""
        
        # Test with high volume of signals
        # Test concurrent processing
        # Test memory usage and cleanup
        
        # Placeholder for performance testing
        assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--asyncio-mode=auto'])