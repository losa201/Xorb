#!/usr/bin/env python3
"""
Test LLM Integration with XORB Supreme
Demonstrates AI-powered payload generation and knowledge enrichment
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from knowledge_fabric.llm_knowledge_fabric import LLMKnowledgeFabric
from llm.intelligent_client import IntelligentLLMClient, TaskType
from llm.payload_generator import (
    PayloadCategory,
    PayloadComplexity,
    PayloadGenerator,
    TargetContext,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_llm_integration():
    """Test the complete LLM integration pipeline"""

    # Load configuration
    config = {
        "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", "test-key-placeholder"),
        "redis_url": "redis://localhost:6379/0",
        "database_url": "sqlite+aiosqlite:///./xorb_enhanced.db"
    }

    logger.info("🚀 Starting XORB Supreme LLM Integration Test")

    # Initialize LLM client
    llm_client = IntelligentLLMClient(config)
    await llm_client.start()

    # Initialize payload generator
    payload_generator = PayloadGenerator(llm_client)

    # Initialize enhanced knowledge fabric
    knowledge_fabric = LLMKnowledgeFabric(
        redis_url=config["redis_url"],
        database_url=config["database_url"],
        llm_client=llm_client
    )
    await knowledge_fabric.initialize()

    try:
        logger.info("📋 Test 1: Basic LLM Payload Generation")

        # Test target context
        target_context = TargetContext(
            url="https://vulnerable-app.example.com",
            technology_stack=["PHP", "MySQL", "Apache"],
            operating_system="Linux",
            web_server="Apache 2.4",
            input_fields=["username", "password", "search", "comment"],
            parameters=["id", "page", "action"]
        )

        # Generate XSS payloads
        xss_payloads = await payload_generator.generate_contextual_payloads(
            category=PayloadCategory.XSS,
            target_context=target_context,
            complexity=PayloadComplexity.INTERMEDIATE,
            count=3
        )

        logger.info(f"✅ Generated {len(xss_payloads)} XSS payloads")
        for i, payload in enumerate(xss_payloads, 1):
            logger.info(f"  {i}. {payload.payload[:50]}... (confidence: {payload.success_probability:.2f})")

        logger.info("📋 Test 2: SQL Injection Payload Generation")

        # Generate SQL injection payloads
        sqli_payloads = await payload_generator.generate_contextual_payloads(
            category=PayloadCategory.SQL_INJECTION,
            target_context=target_context,
            complexity=PayloadComplexity.ADVANCED,
            count=3
        )

        logger.info(f"✅ Generated {len(sqli_payloads)} SQL injection payloads")
        for i, payload in enumerate(sqli_payloads, 1):
            logger.info(f"  {i}. {payload.payload[:50]}... (confidence: {payload.success_probability:.2f})")

        logger.info("📋 Test 3: Knowledge Fabric Integration")

        # Store payloads in knowledge fabric
        stored_atom_ids = []

        for payload in xss_payloads + sqli_payloads:
            # Create mock LLM response for provenance
            from datetime import datetime

            from llm.intelligent_client import LLMProvider, LLMResponse

            mock_response = LLMResponse(
                content=payload.payload,
                model_used="moonshotai/kimi-k2:free",
                provider=LLMProvider.OPENROUTER,
                tokens_used=150,
                cost_usd=0.0,
                confidence_score=payload.success_probability,
                generated_at=datetime.utcnow(),
                request_id=f"test_{int(datetime.utcnow().timestamp())}"
            )

            atom_id = await knowledge_fabric.store_llm_payload(
                payload=payload,
                llm_response=mock_response,
                context={
                    "test": True,
                    "target_url": target_context.url,
                    "technology": target_context.technology_stack
                }
            )
            stored_atom_ids.append(atom_id)

        logger.info(f"✅ Stored {len(stored_atom_ids)} payloads in knowledge fabric")

        logger.info("📋 Test 4: Knowledge Fabric Query")

        # Query stored payloads
        high_confidence_payloads = await knowledge_fabric.query_llm_atoms(
            category=PayloadCategory.XSS,
            min_confidence=0.5,
            limit=5
        )

        logger.info(f"✅ Found {len(high_confidence_payloads)} high-confidence XSS payloads")

        logger.info("📋 Test 5: LLM Analysis")

        # Analyze vulnerability
        sample_vuln = """
        SQL Injection vulnerability found in login.php parameter 'username'.
        The application directly concatenates user input into SQL query without sanitization.
        Payload: admin' OR '1'='1' -- 
        Response: Successfully logged in as admin user.
        """

        analysis = await knowledge_fabric.analyze_with_llm(
            content=sample_vuln,
            analysis_type="vulnerability_assessment"
        )

        logger.info("✅ LLM vulnerability analysis completed")
        logger.info(f"  Analysis keys: {list(analysis.keys())}")

        logger.info("📋 Test 6: System Statistics")

        # Get LLM client stats
        client_stats = llm_client.get_usage_stats()
        logger.info("✅ LLM Client Stats:")
        logger.info(f"  Total requests: {client_stats['total_requests']}")
        logger.info(f"  Total cost: ${client_stats['total_cost_usd']:.4f}")
        logger.info(f"  Average cost per request: ${client_stats['avg_cost_per_request']:.4f}")

        # Get knowledge fabric stats
        fabric_stats = await knowledge_fabric.get_llm_fabric_stats()
        logger.info("✅ Knowledge Fabric Stats:")
        logger.info(f"  Total LLM atoms: {fabric_stats['total_llm_atoms']}")
        logger.info(f"  Average confidence: {fabric_stats['avg_confidence']:.2f}")
        logger.info(f"  Total cost: ${fabric_stats['total_cost']:.4f}")

        logger.info("📋 Test 7: Advanced Features Demo")

        # Demonstrate batch generation
        batch_requests = []
        from llm.intelligent_client import LLMRequest

        for category in [PayloadCategory.SSRF, PayloadCategory.RCE]:
            request = LLMRequest(
                task_type=TaskType.PAYLOAD_GENERATION,
                prompt=f"Generate creative {category.value} payloads for Linux web applications",
                target_info={"os": "linux", "app_type": "web"},
                max_tokens=500,
                temperature=0.8
            )
            batch_requests.append(request)

        if batch_requests:
            batch_responses = await llm_client.batch_generate(batch_requests)
            logger.info(f"✅ Batch generated {len(batch_responses)} payload sets")

        logger.info("🎉 All LLM integration tests completed successfully!")

        # Summary report
        print("\n" + "="*60)
        print("XORB SUPREME LLM INTEGRATION SUMMARY")
        print("="*60)
        print(f"✅ Generated {len(xss_payloads + sqli_payloads)} AI-powered payloads")
        print(f"✅ Stored {len(stored_atom_ids)} payloads with full provenance")
        print("✅ AI analysis and enhancement capabilities working")
        print(f"✅ Cost tracking: ${client_stats['total_cost_usd']:.4f} spent")
        print(f"✅ Knowledge fabric contains {fabric_stats['total_llm_atoms']} AI atoms")
        print("\nREADY FOR PRODUCTION BUG BOUNTY CAMPAIGNS! 🚀")
        print("="*60)

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        await llm_client.close()
        logger.info("🧹 Cleanup completed")

if __name__ == "__main__":
    asyncio.run(test_llm_integration())
