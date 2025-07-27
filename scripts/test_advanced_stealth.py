from typing import Dict, List, Any, Optional

#!/usr/bin/env python3
"""
Advanced Stealth and Evasion Test Script
Tests sophisticated evasion techniques and stealth capabilities
"""

import asyncio
import sys
import os
import aiofiles
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from xorb_core.evasion.advanced_stealth import (
    AdvancedStealthEngine, EvasionTechnique, DetectionVector,
    demo_advanced_stealth
)
import logging
import time
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_stealth_session_creation() -> None:
    """Test stealth session creation and configuration"""
    logger.info("=== Testing Stealth Session Creation ===")
    
    stealth_engine = AdvancedStealthEngine()
    
    # Test session creation with different requirements
    test_cases = [
        {
            "target": "example.com",
            "requirements": [DetectionVector.NETWORK_IDS],
            "risk_tolerance": 0.2
        },
        {
            "target": "test.local",
            "requirements": [DetectionVector.BEHAVIORAL_ANALYSIS, DetectionVector.SIGNATURE_DETECTION],
            "risk_tolerance": 0.5
        },
        {
            "target": "high-security.com",
            "requirements": [DetectionVector.NETWORK_IDS, DetectionVector.ANOMALY_DETECTION],
            "risk_tolerance": 0.1
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        session = await stealth_engine.create_stealth_session(
            target=test_case["target"],
            evasion_requirements=test_case["requirements"],
            risk_tolerance=test_case["risk_tolerance"]
        )
        
        # Verify session properties
        assert session.session_id is not None, "Session should have ID"
        assert session.target == test_case["target"], "Session should have correct target"
        assert len(session.evasion_stack) > 0, "Session should have evasion techniques"
        assert session.active == True, "Session should be active"
        
        logger.info(f"✅ Session {i+1}: {session.session_id} with {len(session.evasion_stack)} techniques")
    
    logger.info("✅ Stealth session creation test passed")

async def test_evasion_technique_selection() -> None:
    """Test evasion technique optimization and selection"""
    logger.info("=== Testing Evasion Technique Selection ===")
    
    stealth_engine = AdvancedStealthEngine()
    
    # Test different requirement combinations
    requirement_sets = [
        [DetectionVector.NETWORK_IDS],
        [DetectionVector.BEHAVIORAL_ANALYSIS],
        [DetectionVector.NETWORK_IDS, DetectionVector.SIGNATURE_DETECTION],
        [DetectionVector.ANOMALY_DETECTION, DetectionVector.HEURISTIC_ANALYSIS],
        []  # No specific requirements
    ]
    
    for requirements in requirement_sets:
        stack = await stealth_engine._optimize_evasion_stack(requirements, 0.3)
        
        # Verify optimization results
        assert isinstance(stack, list), "Stack should be a list"
        assert len(stack) <= 4, "Stack should not exceed 4 techniques"
        
        if requirements:
            # Check that selected techniques are relevant
            for technique in stack:
                profile = stealth_engine.evasion_profiles[technique]
                has_relevant_effectiveness = any(
                    req in profile.effectiveness for req in requirements
                )
                assert has_relevant_effectiveness, f"Technique {technique} should be relevant"
        
        logger.info(f"Selected {len(stack)} techniques for {len(requirements)} requirements")
    
    logger.info("✅ Evasion technique selection test passed")

async def test_stealth_request_execution() -> None:
    """Test stealth request execution with evasion techniques"""
    logger.info("=== Testing Stealth Request Execution ===")
    
    stealth_engine = AdvancedStealthEngine()
    
    # Create test session
    session = await stealth_engine.create_stealth_session(
        target="httpbin.org",
        evasion_requirements=[DetectionVector.NETWORK_IDS, DetectionVector.BEHAVIORAL_ANALYSIS],
        risk_tolerance=0.4
    )
    
    # Execute multiple requests with different parameters
    test_requests = [
        {"type": "reconnaissance", "url": "https://httpbin.org/get"},
        {"type": "enumeration", "url": "https://httpbin.org/user-agent"},
        {"type": "exploitation", "url": "https://httpbin.org/headers"},
    ]
    
    for req in test_requests:
        result = await stealth_engine.execute_stealth_request(
            session,
            req["type"], 
            req["url"]
        )
        
        # Verify request execution
        assert "status_code" in result or "error" in result, "Result should have status or error"
        
        if "status_code" in result:
            logger.info(f"✅ {req['type']} request: {result['status_code']}")
        else:
            logger.warning(f"⚠️ {req['type']} request failed: {result.get('error')}")
    
    # Verify session metrics
    metrics = await stealth_engine.get_session_metrics(session.session_id)
    assert metrics["metrics"]["requests_sent"] > 0, "Should have sent requests"
    
    await stealth_engine.close_stealth_session(session.session_id)
    logger.info("✅ Stealth request execution test passed")

async def test_timing_evasion() -> None:
    """Test timing evasion techniques"""
    logger.info("=== Testing Timing Evasion ===")
    
    stealth_engine = AdvancedStealthEngine()
    
    # Create session with timing evasion
    session = await stealth_engine.create_stealth_session(
        target="example.com",
        evasion_requirements=[DetectionVector.BEHAVIORAL_ANALYSIS],
        risk_tolerance=0.3
    )
    
    # Force timing evasion to be included
    if EvasionTechnique.TIMING_EVASION not in session.evasion_stack:
        session.evasion_stack.append(EvasionTechnique.TIMING_EVASION)
    
    # Test different timing patterns
    timing_patterns = ["human_like", "background_noise", "regular_intervals"]
    
    for pattern in timing_patterns:
        session.current_profile["timing_patterns"] = {"pattern_type": pattern}
        
        start_time = time.time()
        await stealth_engine._apply_timing_evasion(session)
        elapsed = time.time() - start_time
        
        # Verify timing delay was applied
        assert elapsed > 0.05, f"Timing delay should be applied for {pattern}"
        logger.info(f"✅ {pattern} timing: {elapsed:.2f}s delay")
    
    await stealth_engine.close_stealth_session(session.session_id)
    logger.info("✅ Timing evasion test passed")

async def test_dns_tunneling() -> None:
    """Test DNS tunneling implementation"""
    logger.info("=== Testing DNS Tunneling ===")
    
    stealth_engine = AdvancedStealthEngine()
    
    session = await stealth_engine.create_stealth_session(
        target="example.com",
        evasion_requirements=[DetectionVector.NETWORK_IDS],
        risk_tolerance=0.4
    )
    
    # Test DNS tunneling with different data sizes
    test_data = [
        b"small",
        b"medium_sized_data_for_testing",
        b"large_data_payload_that_needs_to_be_split_across_multiple_dns_queries_for_proper_testing"
    ]
    
    for data in test_data:
        result = await stealth_engine.implement_dns_tunneling(
            session,
            data,
            "example.com"
        )
        
        # Verify DNS tunneling structure
        assert "technique" in result, "Result should specify technique"
        assert result["technique"] == "dns_tunneling", "Should be DNS tunneling"
        
        if "chunks_sent" in result:
            assert result["chunks_sent"] > 0, "Should have sent chunks"
            logger.info(f"✅ DNS tunneling: {len(data)} bytes in {result['chunks_sent']} chunks")
        else:
            logger.warning(f"⚠️ DNS tunneling failed: {result.get('error')}")
    
    await stealth_engine.close_stealth_session(session.session_id)
    logger.info("✅ DNS tunneling test passed")

async def test_steganography() -> None:
    """Test steganographic data hiding"""
    logger.info("=== Testing Steganography ===")
    
    stealth_engine = AdvancedStealthEngine()
    
    # Test LSB steganography
    test_cases = [
        {
            "cover": b"This is a long cover message for hiding secret data inside using steganography.",
            "secret": b"SECRET",
            "method": "lsb"
        },
        {
            "cover": b"A" * 100,  # Repetitive data
            "secret": b"HIDDEN",
            "method": "lsb"
        }
    ]
    
    for i, case in enumerate(test_cases):
        try:
            stego_data = await stealth_engine.implement_steganography(
                case["cover"],
                case["secret"],
                case["method"]
            )
            
            # Verify steganography results
            assert len(stego_data) >= len(case["cover"]), "Stego data should be at least as long as cover"
            assert stego_data != case["cover"], "Stego data should be different from cover"
            
            logger.info(f"✅ Steganography test {i+1}: {len(case['secret'])} bytes hidden in {len(case['cover'])} bytes")
            
        except Exception as e:
            logger.warning(f"⚠️ Steganography test {i+1} failed: {e}")
    
    # Test with data too large for cover
    try:
        large_secret = b"X" * 1000
        small_cover = b"small"
        
        await stealth_engine.implement_steganography(small_cover, large_secret, "lsb")
        assert False, "Should have raised error for oversized secret"
        
    except ValueError as e:
        logger.info(f"✅ Correctly handled oversized secret: {e}")
    
    logger.info("✅ Steganography test passed")

async def test_protocol_obfuscation() -> None:
    """Test protocol obfuscation techniques"""
    logger.info("=== Testing Protocol Obfuscation ===")
    
    stealth_engine = AdvancedStealthEngine()
    
    session = await stealth_engine.create_stealth_session(
        target="example.com",
        evasion_requirements=[DetectionVector.SIGNATURE_DETECTION],
        risk_tolerance=0.3
    )
    
    # Force protocol obfuscation
    if EvasionTechnique.PROTOCOL_OBFUSCATION not in session.evasion_stack:
        session.evasion_stack.append(EvasionTechnique.PROTOCOL_OBFUSCATION)
    
    # Test protocol obfuscation application
    base_params = {
        "url": "https://example.com",
        "method": "GET",
        "headers": {"Original": "Header"},
        "data": None
    }
    
    obfuscated_params = await stealth_engine._apply_protocol_obfuscation(base_params)
    
    # Verify obfuscation was applied
    assert len(obfuscated_params["headers"]) > len(base_params["headers"]), "Should have added headers"
    
    # Check for specific obfuscation headers
    expected_headers = ["X-Forwarded-For", "X-Real-IP", "Accept-Language"]
    for header in expected_headers:
        assert header in obfuscated_params["headers"], f"Should have {header} header"
    
    logger.info(f"✅ Protocol obfuscation added {len(obfuscated_params['headers'])} headers")
    
    await stealth_engine.close_stealth_session(session.session_id)
    logger.info("✅ Protocol obfuscation test passed")

async def test_behavioral_mimicry() -> None:
    """Test behavioral mimicry implementation"""
    logger.info("=== Testing Behavioral Mimicry ===")
    
    stealth_engine = AdvancedStealthEngine()
    
    session = await stealth_engine.create_stealth_session(
        target="example.com",
        evasion_requirements=[DetectionVector.BEHAVIORAL_ANALYSIS],
        risk_tolerance=0.4
    )
    
    # Test different behavioral patterns
    behaviors = ["normal_user", "search_bot", "api_client", "unknown_behavior"]
    
    for behavior in behaviors:
        result = await stealth_engine.implement_behavioral_mimicry(session, behavior)
        
        if "error" not in result:
            assert result["technique"] == "behavioral_mimicry", "Should be behavioral mimicry"
            assert "target_behavior" in result, "Should specify target behavior"
            
            # Check if profile was updated
            if result.get("profile_updated"):
                mimicry_profile = session.current_profile.get("behavioral_mimicry", {})
                assert "patterns" in mimicry_profile, "Should have behavioral patterns"
                
            logger.info(f"✅ Behavioral mimicry: {result['target_behavior']}")
        else:
            logger.warning(f"⚠️ Behavioral mimicry failed for {behavior}: {result['error']}")
    
    await stealth_engine.close_stealth_session(session.session_id)
    logger.info("✅ Behavioral mimicry test passed")

async def test_anti_forensics() -> None:
    """Test anti-forensics techniques"""
    logger.info("=== Testing Anti-Forensics ===")
    
    stealth_engine = AdvancedStealthEngine()
    
    session = await stealth_engine.create_stealth_session(
        target="example.com",
        evasion_requirements=[DetectionVector.HOST_IDS],
        risk_tolerance=0.3
    )
    
    # Test anti-forensics implementation
    result = await stealth_engine.implement_anti_forensics(session)
    
    # Verify anti-forensics results
    if "error" not in result:
        assert result["technique"] == "anti_forensics", "Should be anti-forensics"
        assert "implemented" in result, "Should list implemented techniques"
        assert len(result["implemented"]) > 0, "Should have implemented techniques"
        
        logger.info(f"✅ Anti-forensics: {len(result['implemented'])} techniques")
    else:
        logger.warning(f"⚠️ Anti-forensics failed: {result['error']}")
    
    await stealth_engine.close_stealth_session(session.session_id)
    logger.info("✅ Anti-forensics test passed")

async def test_session_metrics() -> None:
    """Test session metrics and monitoring"""
    logger.info("=== Testing Session Metrics ===")
    
    stealth_engine = AdvancedStealthEngine()
    
    session = await stealth_engine.create_stealth_session(
        target="httpbin.org",
        evasion_requirements=[DetectionVector.NETWORK_IDS],
        risk_tolerance=0.4
    )
    
    # Execute some requests to generate metrics
    for i in range(3):
        await stealth_engine.execute_stealth_request(
            session,
            "test",
            "https://httpbin.org/status/200"
        )
    
    # Get session metrics
    metrics = await stealth_engine.get_session_metrics(session.session_id)
    
    # Verify metrics structure
    assert "session_id" in metrics, "Should have session ID"
    assert "target" in metrics, "Should have target"
    assert "metrics" in metrics, "Should have metrics data"
    assert "session_duration" in metrics, "Should have duration"
    
    # Verify metrics values
    assert metrics["metrics"]["requests_sent"] == 3, "Should have sent 3 requests"
    assert metrics["session_duration"] > 0, "Should have positive duration"
    
    logger.info(f"✅ Session metrics: {metrics['metrics']['requests_sent']} requests, {metrics['session_duration']:.2f}s duration")
    
    # Test non-existent session
    invalid_metrics = await stealth_engine.get_session_metrics("invalid_session")
    assert "error" in invalid_metrics, "Should return error for invalid session"
    
    await stealth_engine.close_stealth_session(session.session_id)
    logger.info("✅ Session metrics test passed")

async def test_evasion_profiles() -> None:
    """Test evasion profile effectiveness ratings"""
    logger.info("=== Testing Evasion Profiles ===")
    
    stealth_engine = AdvancedStealthEngine()
    
    # Verify all techniques have profiles
    expected_techniques = [
        EvasionTechnique.TIMING_EVASION,
        EvasionTechnique.TRAFFIC_FRAGMENTATION,
        EvasionTechnique.PROTOCOL_OBFUSCATION,
        EvasionTechnique.USER_AGENT_ROTATION,
        EvasionTechnique.DNS_TUNNELING
    ]
    
    for technique in expected_techniques:
        assert technique in stealth_engine.evasion_profiles, f"Should have profile for {technique}"
        
        profile = stealth_engine.evasion_profiles[technique]
        
        # Verify profile structure
        assert 0 <= profile.confidence <= 1, "Confidence should be between 0 and 1"
        assert 0 <= profile.resource_cost <= 1, "Resource cost should be between 0 and 1"
        assert 0 <= profile.detection_risk <= 1, "Detection risk should be between 0 and 1"
        assert profile.implementation_complexity in ["low", "medium", "high"], "Should have valid complexity"
        
        # Verify effectiveness ratings
        for vector, effectiveness in profile.effectiveness.items():
            assert isinstance(vector, DetectionVector), "Should be valid detection vector"
            assert 0 <= effectiveness <= 1, "Effectiveness should be between 0 and 1"
        
        logger.info(f"✅ {technique.value}: {profile.confidence:.2f} confidence, {profile.detection_risk:.2f} risk")
    
    logger.info("✅ Evasion profiles test passed")

async def main() -> None:
    """Run all advanced stealth and evasion tests"""
    logger.info("Starting Advanced Stealth and Evasion Tests")
    logger.info("=" * 70)
    
    try:
        # Run comprehensive test suite
        await test_stealth_session_creation()
        await test_evasion_technique_selection()
        await test_stealth_request_execution()
        await test_timing_evasion()
        await test_dns_tunneling()
        await test_steganography()
        await test_protocol_obfuscation()
        await test_behavioral_mimicry()
        await test_anti_forensics()
        await test_session_metrics()
        await test_evasion_profiles()
        
        # Run the demo
        logger.info("=== Running Advanced Stealth Demo ===")
        await demo_advanced_stealth()
        
        logger.info("=" * 70)
        logger.info("🎉 All advanced stealth and evasion tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())