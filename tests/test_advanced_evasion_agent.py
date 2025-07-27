"""
Comprehensive Integration Test Suite for Advanced Evasion Agent

This test suite validates all advanced evasion and stealth capabilities including:
- Timing evasion techniques
- Protocol obfuscation methods
- DNS tunneling capabilities
- Anti-forensics measures
- Stealth profile operations
- Detection evasion effectiveness
"""

import pytest
import asyncio
import time
import tempfile
import json
import base64
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

# Add the xorb_core package to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "xorb_core"))

from agents.stealth.advanced_evasion_agent import (
    AdvancedEvasionAgent,
    TimingEvasionTechnique,
    ProtocolObfuscationTechnique, 
    DNSTunnelingTechnique,
    AntiForensicsTechnique,
    EvasionTechnique,
    StealthLevel,
    EvasionConfig,
    StealthProfile
)


class TestTimingEvasionTechnique:
    """Test suite for timing evasion techniques."""
    
    @pytest.fixture
    def timing_technique(self):
        """Create timing evasion technique instance."""
        return TimingEvasionTechnique()
    
    @pytest.fixture
    def evasion_config(self):
        """Create test evasion configuration."""
        return EvasionConfig(
            techniques=[EvasionTechnique.TIMING_EVASION],
            stealth_level=StealthLevel.MEDIUM,
            timing_variance=0.2,
            max_operation_time=300
        )
    
    @pytest.mark.asyncio
    async def test_timing_pattern_selection(self, timing_technique):
        """Test timing pattern selection based on stealth level."""
        patterns = {
            StealthLevel.LOW: timing_technique._select_timing_pattern(StealthLevel.LOW),
            StealthLevel.MEDIUM: timing_technique._select_timing_pattern(StealthLevel.MEDIUM),
            StealthLevel.HIGH: timing_technique._select_timing_pattern(StealthLevel.HIGH),
            StealthLevel.MAXIMUM: timing_technique._select_timing_pattern(StealthLevel.MAXIMUM)
        }
        
        # Verify patterns are appropriate for each level
        assert patterns[StealthLevel.LOW] in ["exponential_backoff"]
        assert patterns[StealthLevel.MEDIUM] in ["gaussian_distribution", "human_simulation"]
        assert patterns[StealthLevel.HIGH] in ["periodic_burst", "fibonacci_sequence"]
        assert patterns[StealthLevel.MAXIMUM] in ["human_simulation", "periodic_burst"]
    
    @pytest.mark.asyncio
    async def test_timing_schedule_generation(self, timing_technique, evasion_config):
        """Test timing schedule generation for different patterns."""
        patterns = ["exponential_backoff", "gaussian_distribution", "human_simulation", "periodic_burst", "fibonacci_sequence"]
        
        for pattern in patterns:
            schedule = timing_technique._generate_timing_schedule(pattern, evasion_config)
            
            # Verify schedule is generated
            assert len(schedule) > 0
            
            # Verify schedule format
            for delay, chunk_size in schedule:
                assert isinstance(delay, (int, float))
                assert isinstance(chunk_size, int)
                assert delay >= 0
                assert chunk_size >= 0
    
    @pytest.mark.asyncio
    async def test_jitter_application(self, timing_technique):
        """Test timing jitter application."""
        base_delay = 5.0
        variance = 0.3
        
        jittered_delays = []
        for _ in range(100):
            jittered = timing_technique._apply_jitter(base_delay, variance)
            jittered_delays.append(jittered)
            
            # Verify jitter is within reasonable bounds
            assert jittered > 0.1  # Minimum delay
            assert abs(jittered - base_delay) <= base_delay * variance * 1.5  # Allow some tolerance
        
        # Verify jitter creates variance
        delay_variance = max(jittered_delays) - min(jittered_delays)
        assert delay_variance > 0.5  # Should have meaningful variance
    
    @pytest.mark.asyncio
    async def test_stealth_sleep(self, timing_technique):
        """Test stealth sleep with anti-detection measures."""
        start_time = time.time()
        
        # Test short delay
        await timing_technique._stealth_sleep(0.5)
        short_elapsed = time.time() - start_time
        assert 0.4 <= short_elapsed <= 1.0
        
        # Test longer delay (should be chunked)
        start_time = time.time()
        await timing_technique._stealth_sleep(2.0)
        long_elapsed = time.time() - start_time
        assert 1.8 <= long_elapsed <= 3.0
    
    @pytest.mark.asyncio
    async def test_innocent_operations(self, timing_technique):
        """Test innocent operations for masking intent."""
        operations = [
            timing_technique._dns_lookup,
            timing_technique._http_request,
            timing_technique._memory_access
        ]
        
        for operation in operations:
            start_time = time.time()
            await operation()
            elapsed = time.time() - start_time
            
            # Operations should complete quickly
            assert elapsed < 1.0
    
    @pytest.mark.asyncio
    async def test_timing_evasion_execution(self, timing_technique, evasion_config):
        """Test complete timing evasion execution."""
        target = "test.example.com"
        payload = b"test_payload_data_for_timing_evasion"
        
        start_time = time.time()
        success = await timing_technique.execute(target, payload, evasion_config)
        execution_time = time.time() - start_time
        
        # Verify execution
        assert isinstance(success, bool)
        assert execution_time > 0.1  # Should take some time due to timing evasion
        
        # Verify detection signature
        signature = timing_technique.get_detection_signature()
        assert signature["technique"] == "timing_evasion"
        assert "indicators" in signature
        assert "detection_methods" in signature
    
    def test_detection_probability_estimation(self, timing_technique):
        """Test detection probability estimation for different environments."""
        environments = ["basic", "corporate", "enterprise", "government", "unknown"]
        
        for env in environments:
            prob = timing_technique.estimate_detection_probability(env)
            assert 0.0 <= prob <= 1.0
            
        # Government should have highest detection probability
        gov_prob = timing_technique.estimate_detection_probability("government")
        basic_prob = timing_technique.estimate_detection_probability("basic")
        assert gov_prob > basic_prob


class TestProtocolObfuscationTechnique:
    """Test suite for protocol obfuscation techniques."""
    
    @pytest.fixture
    def obfuscation_technique(self):
        """Create protocol obfuscation technique instance."""
        return ProtocolObfuscationTechnique()
    
    @pytest.fixture
    def evasion_config(self):
        """Create test evasion configuration."""
        return EvasionConfig(
            techniques=[EvasionTechnique.PROTOCOL_OBFUSCATION],
            stealth_level=StealthLevel.HIGH,
            obfuscation_layers=3
        )
    
    @pytest.mark.asyncio
    async def test_http_header_obfuscation(self, obfuscation_technique):
        """Test HTTP header obfuscation method."""
        test_data = b"sensitive_data_to_obfuscate"
        
        obfuscated = await obfuscation_technique._http_header_obfuscation(test_data)
        
        # Verify obfuscated data looks like HTTP
        obfuscated_str = obfuscated.decode()
        assert "GET /api/data HTTP/1.1" in obfuscated_str
        assert "Host: legitimate-site.com" in obfuscated_str
        assert "X-Custom-" in obfuscated_str
        
        # Verify original data is encoded in headers
        assert base64.b64encode(test_data).decode() in obfuscated_str.replace("\n", "").replace(" ", "")
    
    @pytest.mark.asyncio
    async def test_custom_protocol_wrapper(self, obfuscation_technique):
        """Test custom protocol wrapper method."""
        test_data = b"test_protocol_data"
        
        wrapped = await obfuscation_technique._custom_protocol_wrapper(test_data)
        
        # Verify protocol header structure
        assert len(wrapped) > len(test_data) + 8  # Header + padding
        
        # Verify magic number is present
        magic_bytes = wrapped[4:6]
        magic_number = int.from_bytes(magic_bytes, byteorder='big')
        assert magic_number == 0x1337
    
    @pytest.mark.asyncio
    async def test_steganographic_encoding(self, obfuscation_technique):
        """Test steganographic encoding method."""
        test_data = b"steganographic_test_data"
        
        encoded = await obfuscation_technique._steganographic_encoding(test_data)
        
        # Verify PNG header is present
        assert encoded.startswith(b'\x89PNG\r\n\x1a\n')
        
        # Verify data is significantly larger (due to steganographic encoding)
        assert len(encoded) > len(test_data) * 4
    
    @pytest.mark.asyncio
    async def test_mime_type_spoofing(self, obfuscation_technique):
        """Test MIME type spoofing method."""
        test_data = b"mime_spoofing_test"
        
        spoofed = await obfuscation_technique._mime_type_spoofing(test_data)
        
        # Verify JPEG headers
        assert spoofed.startswith(b'\xff\xd8\xff\xe0\x00\x10JFIF')
        assert spoofed.endswith(b'\xff\xd9')
        
        # Verify original data is embedded
        assert test_data in spoofed
    
    @pytest.mark.asyncio
    async def test_compression_obfuscation(self, obfuscation_technique):
        """Test compression obfuscation method."""
        test_data = b"compression_test_data_that_should_compress_well" * 10
        
        compressed = await obfuscation_technique._compression_obfuscation(test_data)
        
        # Verify compression header
        assert compressed.startswith(b'COMP')
        
        # Verify data is compressed (should be smaller)
        assert len(compressed) < len(test_data)
    
    @pytest.mark.asyncio
    async def test_protocol_wrapping(self, obfuscation_technique, evasion_config):
        """Test protocol wrapping methods."""
        test_data = b"protocol_wrapping_test"
        target = "example.com"
        
        # Test HTTP wrapping
        http_wrapped = obfuscation_technique._wrap_in_http(test_data, target)
        http_str = http_wrapped.decode()
        assert "POST /upload HTTP/1.1" in http_str
        assert f"Host: {target}" in http_str
        assert "Content-Length:" in http_str
        
        # Test HTTPS wrapping
        https_wrapped = obfuscation_technique._wrap_in_https(test_data, target)
        assert https_wrapped.startswith(b'SSL_HANDSHAKE_SIMULATION')
        
        # Test DNS wrapping
        dns_wrapped = obfuscation_technique._wrap_in_dns(test_data, target)
        assert len(dns_wrapped) > 12  # DNS header is 12 bytes minimum
    
    @pytest.mark.asyncio
    async def test_multi_layer_obfuscation(self, obfuscation_technique, evasion_config):
        """Test multiple layers of obfuscation."""
        target = "test.example.com"
        payload = b"multi_layer_test_payload"
        
        success = await obfuscation_technique.execute(target, payload, evasion_config)
        
        # Verify execution
        assert isinstance(success, bool)
        
        # Verify detection signature
        signature = obfuscation_technique.get_detection_signature()
        assert signature["technique"] == "protocol_obfuscation"
        assert len(signature["indicators"]) > 0
        assert len(signature["detection_methods"]) > 0
    
    def test_detection_probability_estimation(self, obfuscation_technique):
        """Test detection probability estimation."""
        environments = ["basic", "corporate", "enterprise", "government"]
        
        probabilities = []
        for env in environments:
            prob = obfuscation_technique.estimate_detection_probability(env)
            probabilities.append(prob)
            assert 0.0 <= prob <= 1.0
        
        # Should have increasing detection probability
        assert probabilities == sorted(probabilities)


class TestDNSTunnelingTechnique:
    """Test suite for DNS tunneling techniques."""
    
    @pytest.fixture
    def dns_technique(self):
        """Create DNS tunneling technique instance."""
        return DNSTunnelingTechnique()
    
    @pytest.fixture
    def evasion_config(self):
        """Create test evasion configuration."""
        return EvasionConfig(
            techniques=[EvasionTechnique.DNS_TUNNELING],
            stealth_level=StealthLevel.HIGH,
            dns_tunnel_domains=["tunnel.example.com", "covert.test.org"]
        )
    
    def test_payload_encoding_methods(self, dns_technique):
        """Test different payload encoding methods."""
        test_payload = b"dns_tunneling_test_payload_data"
        
        # Test base32 encoding
        base32_encoded = dns_technique._encode_payload(test_payload, "base32")
        assert isinstance(base32_encoded, str)
        assert len(base32_encoded) > 0
        assert base32_encoded.islower()  # Base32 should be lowercase
        
        # Test base64 encoding
        base64_encoded = dns_technique._encode_payload(test_payload, "base64")
        assert isinstance(base64_encoded, str)
        assert len(base64_encoded) > 0
        
        # Test hex encoding
        hex_encoded = dns_technique._encode_payload(test_payload, "hex")
        assert isinstance(hex_encoded, str)
        assert len(hex_encoded) == len(test_payload) * 2
        assert all(c in "0123456789abcdef" for c in hex_encoded)
        
        # Test custom encoding
        custom_encoded = dns_technique._encode_payload(test_payload, "custom")
        assert isinstance(custom_encoded, str)
        assert len(custom_encoded) > 0
        assert all(c in "abcdefghijklmnopqrstuvwxyz" for c in custom_encoded)
    
    def test_custom_encoding_reversibility(self, dns_technique):
        """Test that custom encoding can be reversed."""
        test_payload = b"reversibility_test"
        
        # Encode
        encoded = dns_technique._custom_encoding(test_payload)
        
        # Decode (simulate reverse process)
        key = b"stealthkey123456"
        decoded_chars = []
        for char in encoded:
            byte_val = ord(char) - ord('a')
            decoded_chars.append(byte_val)
        
        # XOR back
        decoded = bytes(b ^ key[i % len(key)] for i, b in enumerate(decoded_chars))
        
        assert decoded == test_payload
    
    def test_query_fragmentation(self, dns_technique, evasion_config):
        """Test fragmentation of data into DNS queries."""
        # Test with different payload sizes
        payloads = [
            b"short",
            b"medium_length_payload_for_dns_tunneling_test",
            b"very_long_payload_that_should_be_fragmented_into_multiple_dns_queries_to_test_the_fragmentation_logic" * 5
        ]
        
        for payload in payloads:
            encoded = dns_technique._encode_payload(payload, "base32")
            queries = dns_technique._fragment_into_queries(encoded, evasion_config)
            
            # Verify queries are generated
            assert len(queries) > 0
            
            # Verify query structure
            for query in queries:
                assert "domain" in query
                assert "type" in query
                assert "sequence" in query
                assert "chunk" in query
                
                # Verify domain structure
                domain = query["domain"]
                assert isinstance(domain, str)
                assert "." in domain  # Should have domain structure
                
                # Verify domain length limits
                for label in domain.split("."):
                    assert len(label) <= 63  # DNS label length limit
                
                # Verify query type is valid
                assert query["type"] in dns_technique.query_types
    
    @pytest.mark.asyncio
    async def test_dns_query_sending(self, dns_technique, evasion_config):
        """Test DNS query sending with evasion measures."""
        test_query = {
            "domain": "test.tunnel.example.com",
            "type": "A",
            "sequence": 0,
            "chunk": "testdata"
        }
        
        # Mock legitimate DNS query
        with patch.object(dns_technique, '_send_legitimate_dns_query', new_callable=AsyncMock) as mock_legit:
            success = await dns_technique._send_dns_query(test_query, evasion_config)
            
            # Verify execution
            assert isinstance(success, bool)
            
            # Legitimate queries should be sent occasionally (mocked)
            # This tests the masking functionality
    
    @pytest.mark.asyncio
    async def test_complete_dns_tunneling(self, dns_technique, evasion_config):
        """Test complete DNS tunneling operation."""
        target = "tunnel.example.com"
        payload = b"complete_dns_tunneling_test_payload"
        
        success = await dns_technique.execute(target, payload, evasion_config)
        
        # Verify execution
        assert isinstance(success, bool)
        
        # Verify detection signature
        signature = dns_technique.get_detection_signature()
        assert signature["technique"] == "dns_tunneling"
        assert "unusual_domain_patterns" in signature["indicators"]
        assert "dns_traffic_analysis" in signature["detection_methods"]
    
    def test_detection_probability_estimation(self, dns_technique):
        """Test detection probability estimation for DNS tunneling."""
        environments = ["basic", "corporate", "enterprise", "government"]
        
        for env in environments:
            prob = dns_technique.estimate_detection_probability(env)
            assert 0.0 <= prob <= 1.0
        
        # Government should have higher detection probability than basic
        gov_prob = dns_technique.estimate_detection_probability("government")
        basic_prob = dns_technique.estimate_detection_probability("basic")
        assert gov_prob > basic_prob


class TestAntiForensicsTechnique:
    """Test suite for anti-forensics techniques."""
    
    @pytest.fixture
    def anti_forensics_technique(self):
        """Create anti-forensics technique instance."""
        return AntiForensicsTechnique()
    
    @pytest.fixture
    def evasion_config(self):
        """Create test evasion configuration."""
        return EvasionConfig(
            techniques=[EvasionTechnique.ANTI_FORENSICS],
            stealth_level=StealthLevel.MAXIMUM,
            anti_forensics_enabled=True,
            memory_cleaning=True
        )
    
    @pytest.mark.asyncio
    async def test_pre_operation_cleanup(self, anti_forensics_technique):
        """Test pre-operation cleanup procedures."""
        # Set some environment variables that should be cleared
        import os
        test_vars = {"HTTP_PROXY": "test", "HTTPS_PROXY": "test", "USER": "testuser"}
        for var, value in test_vars.items():
            os.environ[var] = value
        
        await anti_forensics_technique._pre_operation_cleanup()
        
        # Verify sensitive variables are cleared
        for var in test_vars:
            assert var not in os.environ or os.environ[var] != test_vars[var]
    
    @pytest.mark.asyncio
    async def test_memory_cleanup(self, anti_forensics_technique):
        """Test memory cleanup procedures."""
        # Allocate some memory that should be cleaned
        test_data = [b"sensitive_data" * 1000 for _ in range(10)]
        
        await anti_forensics_technique._memory_cleanup()
        
        # Clear test data
        del test_data
        
        # Memory cleanup should complete without errors
        assert True  # If we reach here, cleanup succeeded
    
    @pytest.mark.asyncio
    async def test_forensic_aware_execution(self, anti_forensics_technique, evasion_config):
        """Test execution with forensic awareness."""
        target = "test.example.com"
        payload = b"forensic_aware_test_payload"
        
        success = await anti_forensics_technique._execute_with_forensic_awareness(target, payload, evasion_config)
        
        # Verify execution
        assert isinstance(success, bool)
    
    @pytest.mark.asyncio
    async def test_cleanup_verification(self, anti_forensics_technique):
        """Test cleanup verification procedures."""
        cleanup_score = await anti_forensics_technique._verify_cleanup()
        
        # Verify cleanup score is between 0 and 1
        assert 0.0 <= cleanup_score <= 1.0
        
        # Test individual verification checks
        checks = [
            anti_forensics_technique._check_memory_residue(),
            anti_forensics_technique._check_temporary_files(),
            anti_forensics_technique._check_log_entries(),
            anti_forensics_technique._check_system_state(),
            anti_forensics_technique._check_network_artifacts()
        ]
        
        for check in checks:
            result = await check
            assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_complete_anti_forensics(self, anti_forensics_technique, evasion_config):
        """Test complete anti-forensics operation."""
        target = "test.example.com"
        payload = b"complete_anti_forensics_test"
        
        success = await anti_forensics_technique.execute(target, payload, evasion_config)
        
        # Verify execution
        assert isinstance(success, bool)
        
        # Verify detection signature
        signature = anti_forensics_technique.get_detection_signature()
        assert signature["technique"] == "anti_forensics"
        assert "log_file_modifications" in signature["indicators"]
        assert "log_integrity_monitoring" in signature["detection_methods"]
    
    def test_detection_probability_estimation(self, anti_forensics_technique):
        """Test detection probability estimation for anti-forensics."""
        environments = ["basic", "corporate", "enterprise", "government"]
        
        for env in environments:
            prob = anti_forensics_technique.estimate_detection_probability(env)
            assert 0.0 <= prob <= 1.0
        
        # Anti-forensics should have lower detection probability in basic environments
        basic_prob = anti_forensics_technique.estimate_detection_probability("basic")
        gov_prob = anti_forensics_technique.estimate_detection_probability("government")
        assert basic_prob < gov_prob


class TestAdvancedEvasionAgent:
    """Test suite for the complete Advanced Evasion Agent."""
    
    @pytest.fixture
    def agent(self):
        """Create advanced evasion agent instance."""
        return AdvancedEvasionAgent()
    
    @pytest.fixture
    def basic_task_config(self):
        """Create basic task configuration."""
        return {
            "target": "test.example.com",
            "payload": "test_payload_data",
            "stealth_profile": "corporate",
            "environment": "corporate"
        }
    
    def test_agent_initialization(self, agent):
        """Test agent initialization and configuration."""
        # Verify agent properties
        assert agent.agent_id == "advanced_evasion_agent"
        assert "Advanced Evasion & Stealth Agent" in agent.name
        assert len(agent.capabilities) > 0
        
        # Verify techniques are loaded
        assert len(agent.techniques) > 0
        assert EvasionTechnique.TIMING_EVASION in agent.techniques
        assert EvasionTechnique.PROTOCOL_OBFUSCATION in agent.techniques
        assert EvasionTechnique.DNS_TUNNELING in agent.techniques
        assert EvasionTechnique.ANTI_FORENSICS in agent.techniques
        
        # Verify stealth profiles are loaded
        assert len(agent.stealth_profiles) > 0
        assert "corporate" in agent.stealth_profiles
        assert "government" in agent.stealth_profiles
        assert "cloud" in agent.stealth_profiles
    
    def test_stealth_profile_retrieval(self, agent):
        """Test stealth profile retrieval and customization."""
        # Test default profile retrieval
        corporate_profile = agent._get_stealth_profile("corporate", [])
        assert corporate_profile.profile_name == "corporate"
        assert corporate_profile.target_environment == "corporate"
        
        # Test custom techniques
        custom_techniques = ["timing_evasion", "dns_tunneling"]
        custom_profile = agent._get_stealth_profile("corporate", custom_techniques)
        assert len(custom_profile.evasion_config.techniques) == 2
        assert EvasionTechnique.TIMING_EVASION in custom_profile.evasion_config.techniques
        assert EvasionTechnique.DNS_TUNNELING in custom_profile.evasion_config.techniques
        
        # Test unknown profile (should fallback to corporate)
        unknown_profile = agent._get_stealth_profile("unknown", [])
        assert unknown_profile.target_environment == "corporate"
    
    @pytest.mark.asyncio
    async def test_operational_window_check(self, agent):
        """Test operational window checking."""
        # Create profile with specific operational windows
        profile = StealthProfile(
            profile_name="test",
            target_environment="test",
            evasion_config=EvasionConfig(),
            operational_windows=[(9, 17), (19, 23)]  # 9-17 and 19-23 hours
        )
        
        # Mock current time to test different scenarios
        import datetime
        
        # Test time within window (assume 10 AM)
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value.hour = 10
            within_window = await agent._check_operational_window(profile)
            assert within_window is True
        
        # Test time outside window (assume 6 AM)
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value.hour = 6
            outside_window = await agent._check_operational_window(profile)
            assert outside_window is False
        
        # Test profile with no restrictions
        unrestricted_profile = StealthProfile(
            profile_name="unrestricted",
            target_environment="test", 
            evasion_config=EvasionConfig(),
            operational_windows=[]
        )
        
        unrestricted_result = await agent._check_operational_window(unrestricted_profile)
        assert unrestricted_result is True
    
    @pytest.mark.asyncio
    async def test_stealth_operation_execution(self, agent):
        """Test stealth operation execution with different profiles."""
        target = "test.example.com"
        payload = b"stealth_operation_test_payload"
        environment = "corporate"
        
        # Test with different stealth profiles
        for profile_name in ["corporate", "government", "cloud"]:
            profile = agent.stealth_profiles[profile_name]
            
            operation_result = await agent._execute_stealth_operation(target, payload, profile, environment)
            
            # Verify operation result structure
            assert "overall_success" in operation_result
            assert "technique_results" in operation_result
            assert "estimated_detection_probability" in operation_result
            assert "total_techniques" in operation_result
            assert "successful_techniques" in operation_result
            
            # Verify result values
            assert isinstance(operation_result["overall_success"], bool)
            assert isinstance(operation_result["estimated_detection_probability"], float)
            assert 0.0 <= operation_result["estimated_detection_probability"] <= 1.0
            assert operation_result["total_techniques"] == len(profile.evasion_config.techniques)
            assert 0 <= operation_result["successful_techniques"] <= operation_result["total_techniques"]
            
            # Verify technique results
            for technique in profile.evasion_config.techniques:
                assert technique.value in operation_result["technique_results"]
                technique_result = operation_result["technique_results"][technique.value]
                assert "success" in technique_result
                assert "detection_probability" in technique_result
                assert "signature" in technique_result
    
    def test_success_score_calculation(self, agent):
        """Test success score calculation."""
        # Test perfect operation
        perfect_result = {
            "successful_techniques": 4,
            "total_techniques": 4,
            "estimated_detection_probability": 0.1
        }
        perfect_score = agent._calculate_success_score(perfect_result)
        assert 0.8 <= perfect_score <= 1.0
        
        # Test failed operation
        failed_result = {
            "successful_techniques": 0,
            "total_techniques": 4,
            "estimated_detection_probability": 0.9
        }
        failed_score = agent._calculate_success_score(failed_result)
        assert 0.0 <= failed_score <= 0.2
        
        # Test partial success
        partial_result = {
            "successful_techniques": 2,
            "total_techniques": 4,
            "estimated_detection_probability": 0.5
        }
        partial_score = agent._calculate_success_score(partial_result)
        assert 0.2 <= partial_score <= 0.8
    
    def test_statistics_update(self, agent):
        """Test operation statistics updating."""
        initial_stats = agent.get_operation_statistics()
        initial_total = initial_stats["total_operations"]
        
        # Create test operation result
        operation_result = {
            "overall_success": True,
            "estimated_detection_probability": 0.3
        }
        
        profile = agent.stealth_profiles["corporate"]
        
        # Update statistics
        agent._update_statistics(operation_result, profile)
        
        # Verify statistics are updated
        updated_stats = agent.get_operation_statistics()
        assert updated_stats["total_operations"] == initial_total + 1
        assert updated_stats["successful_operations"] >= initial_stats["successful_operations"]
        
        # Verify technique usage is tracked
        for technique in profile.evasion_config.techniques:
            assert technique.value in updated_stats["techniques_used"]
    
    @pytest.mark.asyncio
    async def test_agent_execution_success_case(self, agent, basic_task_config):
        """Test successful agent execution."""
        # Mock operational window check to always pass
        with patch.object(agent, '_check_operational_window', return_value=True):
            result = await agent.execute(basic_task_config)
            
            # Verify result structure
            assert hasattr(result, 'success')
            assert hasattr(result, 'data')
            assert hasattr(result, 'execution_time')
            assert hasattr(result, 'metadata')
            
            # Verify result values
            assert isinstance(result.success, bool)
            assert isinstance(result.data, dict)
            assert isinstance(result.execution_time, (int, float))
            assert isinstance(result.metadata, dict)
            assert result.execution_time > 0
            
            # Verify metadata contains expected fields
            assert "stealth_profile" in result.metadata
            assert "techniques_used" in result.metadata
            assert "success_score" in result.metadata
            assert "detection_probability" in result.metadata
    
    @pytest.mark.asyncio
    async def test_agent_execution_timing_restriction(self, agent, basic_task_config):
        """Test agent execution with timing restrictions."""
        # Mock operational window check to fail
        with patch.object(agent, '_check_operational_window', return_value=False):
            result = await agent.execute(basic_task_config)
            
            # Should fail due to timing restriction
            assert result.success is False
            assert "timing_restriction" in result.metadata.get("reason", "")
    
    @pytest.mark.asyncio
    async def test_agent_execution_with_custom_techniques(self, agent):
        """Test agent execution with custom technique specification."""
        custom_config = {
            "target": "test.example.com",
            "payload": "custom_test_payload",
            "stealth_profile": "corporate",
            "techniques": ["timing_evasion", "anti_forensics"],
            "environment": "enterprise"
        }
        
        with patch.object(agent, '_check_operational_window', return_value=True):
            result = await agent.execute(custom_config)
            
            # Verify custom techniques are used
            assert "timing_evasion" in result.metadata["techniques_used"]
            assert "anti_forensics" in result.metadata["techniques_used"]
    
    def test_agent_capabilities(self, agent):
        """Test agent capabilities reporting."""
        capabilities = agent.get_capabilities()
        
        # Verify capabilities list
        assert isinstance(capabilities, list)
        assert len(capabilities) > 0
        
        # Verify expected capabilities are present
        expected_capabilities = [
            "timing_evasion",
            "protocol_obfuscation",
            "dns_tunneling",
            "anti_forensics"
        ]
        
        for capability in expected_capabilities:
            assert capability in capabilities
    
    def test_stealth_profiles_retrieval(self, agent):
        """Test stealth profiles retrieval."""
        profiles = agent.get_stealth_profiles()
        
        # Verify profiles structure
        assert isinstance(profiles, dict)
        assert len(profiles) > 0
        
        # Verify profile information
        for profile_name, profile_info in profiles.items():
            assert "target_environment" in profile_info
            assert "stealth_level" in profile_info
            assert "techniques" in profile_info
            assert "operational_windows" in profile_info
            
            assert isinstance(profile_info["techniques"], list)
            assert isinstance(profile_info["operational_windows"], list)
    
    def test_detection_signatures_retrieval(self, agent):
        """Test detection signatures retrieval."""
        signatures = agent.get_detection_signatures()
        
        # Verify signatures structure
        assert isinstance(signatures, dict)
        assert len(signatures) > 0
        
        # Verify signature information
        for technique_name, signature in signatures.items():
            assert "technique" in signature
            assert "indicators" in signature
            assert "detection_methods" in signature
            
            assert isinstance(signature["indicators"], list)
            assert isinstance(signature["detection_methods"], list)
            assert len(signature["indicators"]) > 0
            assert len(signature["detection_methods"]) > 0


@pytest.mark.asyncio
async def test_end_to_end_evasion_scenario():
    """Test complete end-to-end evasion scenario."""
    agent = AdvancedEvasionAgent()
    
    # Define comprehensive test scenario
    scenario_config = {
        "target": "target.example.com",
        "payload": "comprehensive_evasion_test_payload_with_sufficient_length",
        "stealth_profile": "government",  # Maximum stealth
        "environment": "enterprise",
        "techniques": ["timing_evasion", "protocol_obfuscation", "dns_tunneling", "anti_forensics"]
    }
    
    # Mock operational window to allow execution
    with patch.object(agent, '_check_operational_window', return_value=True):
        start_time = time.time()
        result = await agent.execute(scenario_config)
        total_time = time.time() - start_time
        
        # Verify comprehensive test results
        assert isinstance(result.success, bool)
        assert result.execution_time > 0
        assert total_time > result.execution_time * 0.8  # Allow for some overhead
        
        # Verify all requested techniques were attempted
        assert len(result.metadata["techniques_used"]) == 4
        
        # Verify success score is calculated
        assert "success_score" in result.metadata
        assert 0.0 <= result.metadata["success_score"] <= 1.0
        
        # Verify detection probability is estimated
        assert "detection_probability" in result.metadata
        assert 0.0 <= result.metadata["detection_probability"] <= 1.0


@pytest.mark.asyncio
async def test_performance_under_load():
    """Test agent performance under concurrent load."""
    agent = AdvancedEvasionAgent()
    
    # Create multiple concurrent tasks
    num_tasks = 10
    tasks = []
    
    for i in range(num_tasks):
        task_config = {
            "target": f"target{i}.example.com",
            "payload": f"load_test_payload_{i}",
            "stealth_profile": "corporate",
            "environment": "corporate"
        }
        tasks.append(agent.execute(task_config))
    
    # Mock operational window check
    with patch.object(agent, '_check_operational_window', return_value=True):
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Verify all tasks completed
        assert len(results) == num_tasks
        
        # Verify no exceptions occurred
        for result in results:
            assert not isinstance(result, Exception)
            assert hasattr(result, 'success')
            assert hasattr(result, 'execution_time')
        
        # Verify reasonable performance (should complete within reasonable time)
        average_time = total_time / num_tasks
        assert average_time < 5.0  # Should average less than 5 seconds per task
        
        # Verify statistics are properly maintained
        stats = agent.get_operation_statistics()
        assert stats["total_operations"] >= num_tasks


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])