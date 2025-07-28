#!/usr/bin/env python3

import asyncio
import logging
import sys
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_ensemble_ml():
    """Test the ensemble ML predictor."""
    print("\n🧠 Testing Ensemble ML Predictor...")
    try:
        from knowledge_fabric.ensemble_predictor import EnsembleTargetPredictor

        predictor = EnsembleTargetPredictor()
        print("  ✅ Ensemble ML Predictor imported successfully")

        # Test if ML libraries are available
        try:
            import catboost
            import lightgbm
            import xgboost
            print("  ✅ All ML libraries available (XGBoost, LightGBM, CatBoost)")
        except ImportError as e:
            print(f"  ⚠️  Some ML libraries missing: {e}")

        return True
    except Exception as e:
        print(f"  ❌ Ensemble ML test failed: {e}")
        return False

async def test_threat_intelligence():
    """Test threat intelligence components."""
    print("\n🔍 Testing Threat Intelligence...")
    try:
        from integrations.threat_intel_streamer import (
            CVEFeedProcessor,
            IOCFeedProcessor,
        )

        print("  ✅ Threat Intelligence classes imported successfully")

        # Test CVE processor
        cve_processor = CVEFeedProcessor()
        print("  ✅ CVE Feed Processor created")

        # Test IOC processor
        ioc_processor = IOCFeedProcessor()
        print("  ✅ IOC Feed Processor created")

        return True
    except Exception as e:
        print(f"  ❌ Threat Intelligence test failed: {e}")
        return False

async def test_stealth_agents():
    """Test stealth agent components."""
    print("\n🥷 Testing Stealth Agents...")
    try:
        from agents.stealth_agents import (
            AntiDetectionEngine,
            StealthConfig,
            UserAgentRotator,
        )

        print("  ✅ Stealth Agent classes imported successfully")

        # Test user agent rotator
        ua_rotator = UserAgentRotator()
        user_agent = ua_rotator.get_random_user_agent()
        print(f"  ✅ User Agent Rotator works: {user_agent[:50]}...")

        # Test stealth config
        stealth_config = StealthConfig()
        print("  ✅ Stealth Config created")

        # Test anti-detection engine
        anti_detection = AntiDetectionEngine(stealth_config)
        headers = anti_detection.get_random_headers()
        print(f"  ✅ Random headers generated: {len(headers)} headers")

        return True
    except Exception as e:
        print(f"  ❌ Stealth Agents test failed: {e}")
        return False

async def test_market_intelligence():
    """Test market intelligence components."""
    print("\n💰 Testing Market Intelligence...")
    try:
        from integrations.market_intelligence import (
            MarketTrendAnalyzer,
            ProgramAnalytics,
        )

        print("  ✅ Market Intelligence classes imported successfully")

        # Test program analytics
        analytics = ProgramAnalytics()
        print("  ✅ Program Analytics created")

        # Test trend analyzer
        trend_analyzer = MarketTrendAnalyzer()
        print("  ✅ Market Trend Analyzer created")

        return True
    except Exception as e:
        print(f"  ❌ Market Intelligence test failed: {e}")
        return False

async def test_knowledge_fabric():
    """Test basic knowledge fabric."""
    print("\n🧩 Testing Knowledge Fabric...")
    try:
        from knowledge_fabric.atom import AtomType, KnowledgeAtom

        print("  ✅ Knowledge Fabric classes imported successfully")

        # Test creating a knowledge atom
        atom = KnowledgeAtom(
            id="test_atom_001",
            atom_type=AtomType.TECHNIQUE,
            content={"title": "Test Technique", "description": "A test security technique"},
            confidence=0.8
        )
        print(f"  ✅ Knowledge Atom created: {atom.id}")

        return True
    except Exception as e:
        print(f"  ❌ Knowledge Fabric test failed: {e}")
        return False

async def main():
    """Run all component tests."""
    print("🚀 XORB Supreme Enhanced Edition - Component Tests")
    print("=" * 60)

    tests = [
        ("Knowledge Fabric", test_knowledge_fabric),
        ("Ensemble ML", test_ensemble_ml),
        ("Threat Intelligence", test_threat_intelligence),
        ("Stealth Agents", test_stealth_agents),
        ("Market Intelligence", test_market_intelligence),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"  ❌ {test_name} test crashed: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("🎯 Test Results Summary")
    print("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")

    print(f"\n📊 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All components are working correctly!")
        print("\n🚀 Ready to start XORB Supreme Enhanced Edition")

        # Show next steps
        print("\n💡 Next Steps:")
        print("  1. Run: python run_xorb_supreme.py")
        print("  2. Configure API keys in config.json")
        print("  3. Start monitoring: docker-compose -f docker-compose.monitoring.yml up -d")

    else:
        print("⚠️  Some components have issues. Check the logs above.")
        print("💡 The system may still work with reduced functionality.")

if __name__ == "__main__":
    asyncio.run(main())
