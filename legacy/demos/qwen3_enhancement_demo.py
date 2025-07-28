#!/usr/bin/env python3
"""
XORB Qwen3-Coder Enhanced Autonomous Enhancement Demo
Demonstrates the aggressive code improvement capabilities
"""

import asyncio
import logging
import time

from qwen3_autonomous_enhancement_orchestrator import (
    OpenRouterQwen3Client,
    Qwen3AutonomousEnhancementOrchestrator,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('QWEN3-DEMO')

async def demonstrate_enhanced_qwen3():
    """Demonstrate enhanced Qwen3-Coder capabilities."""

    print("\n🧠 QWEN3-CODER ENHANCED AUTONOMOUS ENHANCEMENT DEMO")
    print("🚀 Features: Aggressive Detection, Auto-Fixes, Modernization")
    print("⚡ Performance: 5-minute cycles, adaptive scheduling")
    print("🎯 Target: Complete XORB codebase enhancement")
    print("\n🔥 DEMONSTRATION STARTING...\n")

    # Initialize enhanced orchestrator
    orchestrator = Qwen3AutonomousEnhancementOrchestrator()

    try:
        # Run a single enhanced cycle for demonstration
        logger.info("🔄 Running single enhanced cycle for demonstration...")

        start_time = time.time()
        cycle_results = await orchestrator.run_enhancement_cycle()
        demo_duration = time.time() - start_time

        # Display enhanced results
        print("\n✅ ENHANCED CYCLE COMPLETED!")
        print(f"⏱️ Duration: {demo_duration:.1f} seconds")
        print(f"📁 Files Analyzed: {cycle_results['files_analyzed']}")
        print(f"🐛 Issues Found: {cycle_results['issues_found']}")
        print(f"✨ Enhancements Applied: {cycle_results['enhancements_applied']}")

        if "performance_metrics" in cycle_results:
            metrics = cycle_results["performance_metrics"]
            print("\n📊 PERFORMANCE METRICS:")
            print(f"   Enhancement Rate: {metrics['enhancement_rate']:.2f} per file")
            print(f"   Success Rate: {metrics['success_rate']:.1%}")

        # Show enhanced capabilities
        print("\n🎯 ENHANCED CAPABILITIES DEMONSTRATED:")
        print("   🔍 Aggressive issue detection (critical, high, medium)")
        print("   🤖 Auto-fix for common issues (async, security, errors)")
        print("   ✨ Modernization (f-strings, pathlib, dataclasses)")
        print("   ⚡ Performance optimization (list comprehensions, caching)")
        print("   🔒 Security improvements (subprocess safety, input validation)")
        print("   🧪 Error handling enhancements (logging, specific exceptions)")
        print("   🏗️ Maintainability improvements (type hints, context managers)")

        # Show test results if available
        if cycle_results.get("test_results"):
            test_results = cycle_results["test_results"]
            print("\n🧪 TEST VALIDATION:")
            print(f"   Tests Run: {test_results.get('tests_run', 0)}")
            print(f"   Tests Passed: {test_results.get('tests_passed', 0)}")
            print(f"   Tests Failed: {test_results.get('tests_failed', 0)}")

        # Show commit status
        if cycle_results.get("commit_success"):
            print("\n📝 GIT INTEGRATION:")
            print("   ✅ Changes committed automatically")
            print("   🏷️ Conventional commit format")
            print("   👥 Co-authored by Qwen3-Coder")

        # Show next steps
        print("\n🚀 NEXT STEPS:")
        print("   1. Run full autonomous mode: python3 qwen3_autonomous_enhancement_orchestrator.py")
        print("   2. Use convenient script: ./start_qwen3_enhancement.sh")
        print("   3. Monitor logs in: logs/qwen3_enhancement.log")
        print("   4. Check cycle results in: logs/enhancement_cycle_*.json")

        print("\n🧠 QWEN3-CODER ENHANCED DEMO COMPLETE!")

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        print(f"\n❌ Demo encountered an error: {e}")

async def show_analysis_sample():
    """Show sample of enhanced analysis capabilities."""

    print("\n🔍 SAMPLE ENHANCED CODE ANALYSIS")
    print("=" * 50)

    # Sample code with issues for analysis
    sample_code = '''
import os
import time
import requests

def process_data(data):
    result = ""
    for item in data:
        result += str(item)
    
    try:
        response = requests.get("http://example.com")
        time.sleep(1)  # This is in an async context elsewhere
    except:
        pass
    
    file = open("output.txt", "w")
    file.write(result)
    file.close()
    
    return result
'''

    # Analyze with enhanced Qwen3
    client = OpenRouterQwen3Client()
    analysis = await client.analyze_code(sample_code, "sample.py", "python")

    print("📄 Sample Code Issues:")
    for i, issue in enumerate(analysis.issues_found, 1):
        severity_icon = {"critical": "🚨", "high": "⚠️", "medium": "🔶", "low": "🔸"}.get(issue['severity'], "❓")
        print(f"   {i}. {severity_icon} {issue['type'].upper()}: {issue['description']}")
        print(f"      💡 Fix: {issue['recommendation']}")

    print("\n✨ Enhancement Opportunities:")
    for i, enhancement in enumerate(analysis.enhancements, 1):
        priority_icon = {"high": "🔥", "medium": "⚡", "low": "💡"}.get(enhancement['priority'], "❓")
        print(f"   {i}. {priority_icon} {enhancement['type'].upper()}: {enhancement['description']}")
        print(f"      📈 Benefit: {enhancement['expected_benefit']}")
        if 'estimated_impact' in enhancement:
            print(f"      📊 Impact: {enhancement['estimated_impact']}/10")

    print("\n📊 Code Quality Scores:")
    print(f"   🔧 Maintainability: {analysis.maintainability_score:.1f}/10")
    print(f"   🔒 Security: {analysis.security_score:.1f}/10")
    print(f"   ⚡ Performance: {analysis.performance_score:.1f}/10")
    print(f"   🏗️ Complexity: {analysis.complexity_score:.1f}/10")

async def main():
    """Main demo execution."""

    # Show sample analysis first
    await show_analysis_sample()

    # Then run enhanced cycle demo
    await demonstrate_enhanced_qwen3()

if __name__ == "__main__":
    asyncio.run(main())
