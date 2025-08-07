#!/usr/bin/env python3
"""
XORB Autonomous PTaaS Execution
Advanced penetration testing with AI-powered decision making
"""

import asyncio
import logging
import sys
import json
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from xorb.ptaas.autonomous_engine import (
    initialize_ptaas_engine, 
    Target, 
    TestPhase
)
from xorb.intelligence.llm_integration import initialize_llm_orchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_ptaas.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Execute autonomous penetration testing demonstration."""
    
    print("🎯 XORB Autonomous PTaaS - AI-Powered Penetration Testing")
    print("=" * 65)
    print()
    
    try:
        # Initialize systems
        logger.info("Initializing XORB Autonomous PTaaS Engine...")
        
        # Initialize LLM orchestrator first
        await initialize_llm_orchestrator()
        
        # Initialize PTaaS engine
        ptaas_engine = await initialize_ptaas_engine()
        
        print("✅ Autonomous PTaaS engine initialized successfully")
        print("🧠 AI-powered decision making active")
        print("⚡ EPYC-optimized parallel processing enabled")
        print()
        
        # Define demonstration targets
        targets = [
            Target(
                id="target_web_app",
                name="Corporate Web Application",
                ip_range="192.168.1.100-110",
                ports=[80, 443, 8080],
                services=["http", "https"],
                priority="high",
                authorized=True,
                constraints=["no_data_exfiltration", "business_hours_only"]
            ),
            Target(
                id="target_db_server",
                name="Database Server",
                ip_range="192.168.1.200",
                ports=[3306, 5432, 1433],
                services=["mysql", "postgresql"],
                priority="critical",
                authorized=True,
                constraints=["read_only_access"]
            ),
            Target(
                id="target_file_server",
                name="File Server",
                ip_range="192.168.1.150",
                ports=[445, 139, 22],
                services=["smb", "ssh"],
                priority="medium",
                authorized=True
            )
        ]
        
        print("🎯 DEMONSTRATION TARGETS:")
        print("-" * 30)
        for target in targets:
            print(f"   • {target.name}")
            print(f"     IP Range: {target.ip_range}")
            print(f"     Ports: {target.ports}")
            print(f"     Priority: {target.priority}")
            print(f"     Services: {target.services}")
            print()
        
        # Start autonomous penetration test
        logger.info("Starting autonomous penetration testing session...")
        session_id = await ptaas_engine.start_autonomous_pentest(
            targets=targets,
            session_config={
                "autonomous_mode": True,
                "safety_enabled": True,
                "max_exploitation_depth": 2
            }
        )
        
        print(f"🚀 Autonomous PTaaS session started: {session_id}")
        print("🤖 AI is now conducting comprehensive security assessment...")
        print()
        
        # Monitor session progress
        await monitor_session_progress(ptaas_engine, session_id)
        
        # Wait for session completion (with timeout)
        await asyncio.sleep(45)  # Allow time for full execution
        
        # Get final session report
        session_report = await ptaas_engine.get_session_report(session_id)
        
        if session_report:
            await display_comprehensive_report(session_report)
            
            # Save detailed report
            report_file = f"autonomous_ptaas_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(session_report, f, indent=2, default=str)
            
            print(f"\n📄 Detailed PTaaS report saved to: {report_file}")
        else:
            print("⏳ Session still in progress or report not available")
        
        print("\n✅ Autonomous PTaaS demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Autonomous PTaaS execution failed: {e}")
        print(f"\n❌ PTaaS execution failed: {e}")
        sys.exit(1)

async def monitor_session_progress(ptaas_engine, session_id):
    """Monitor and display session progress."""
    
    phases_completed = set()
    
    for _ in range(30):  # Monitor for up to 30 iterations
        status = await ptaas_engine.get_session_status(session_id)
        
        if status:
            current_phase = status["current_phase"]
            
            if current_phase not in phases_completed:
                phase_emoji = {
                    "reconnaissance": "🔍",
                    "vulnerability_discovery": "🔎", 
                    "exploitation": "💥",
                    "post_exploitation": "🕵️",
                    "reporting": "📊"
                }.get(current_phase, "⚙️")
                
                phase_name = current_phase.replace("_", " ").title()
                print(f"{phase_emoji} Phase: {phase_name}")
                
                if current_phase == "vulnerability_discovery":
                    print(f"   📋 Vulnerabilities found: {status['vulnerabilities_found']}")
                elif current_phase == "exploitation":
                    print(f"   🎯 Successful exploits: {status['successful_exploits']}")
                
                phases_completed.add(current_phase)
            
            if status["is_complete"]:
                print("✅ All phases completed")
                break
                
        await asyncio.sleep(1.5)

async def display_comprehensive_report(report):
    """Display comprehensive penetration testing report."""
    
    print("\n" + "="*70)
    print("📋 AUTONOMOUS PENETRATION TESTING REPORT")
    print("="*70)
    
    # Session metadata
    metadata = report["session_metadata"]
    print(f"🆔 Session ID: {metadata['session_id']}")
    print(f"⏱️  Duration: {metadata['duration_minutes']:.1f} minutes")
    print(f"🤖 Mode: {'Autonomous AI' if metadata['autonomous_mode'] else 'Manual'}")
    print()
    
    # Target summary
    target_summary = report["target_summary"]
    print("🎯 TARGET ANALYSIS:")
    print("-" * 20)
    print(f"   Total Targets: {target_summary['total_targets']}")
    print(f"   Targets Scanned: {target_summary['targets_scanned']}")
    print(f"   Services Discovered: {target_summary['services_discovered']}")
    print()
    
    # Vulnerability summary
    vuln_summary = report["vulnerability_summary"]
    print("🔍 VULNERABILITY ASSESSMENT:")
    print("-" * 30)
    print(f"   Total Vulnerabilities: {vuln_summary['total_vulnerabilities']}")
    print(f"   🔴 Critical: {vuln_summary['critical']}")
    print(f"   🟠 High: {vuln_summary['high']}")
    print(f"   🟡 Medium: {vuln_summary['medium']}")
    print(f"   🟢 Low: {vuln_summary['low']}")
    print()
    
    # Exploitation summary
    exploit_summary = report["exploitation_summary"]
    print("💥 EXPLOITATION RESULTS:")
    print("-" * 25)
    print(f"   Exploits Attempted: {exploit_summary['exploits_attempted']}")
    print(f"   Exploits Successful: {exploit_summary['exploits_successful']}")
    print(f"   Success Rate: {exploit_summary['success_rate']:.1f}%")
    print(f"   Highest Access: {exploit_summary['highest_access']}")
    print()
    
    # AI decisions
    ai_decisions = report.get("ai_decisions", [])
    if ai_decisions:
        print("🧠 AI DECISION SUMMARY:")
        print("-" * 22)
        for decision in ai_decisions:
            print(f"   Phase: {decision['phase']}")
            print(f"   Decision: {decision['decision']} (confidence: {decision['confidence']:.2f})")
            if decision.get('reasoning'):
                print(f"   Reasoning: {decision['reasoning'][0] if decision['reasoning'] else 'N/A'}")
            print()
    
    # Security recommendations
    recommendations = report.get("recommendations", [])
    print("📋 SECURITY RECOMMENDATIONS:")
    print("-" * 30)
    for i, rec in enumerate(recommendations[:8], 1):  # Show top 8 recommendations
        priority_emoji = "🔴" if i <= 2 else "🟠" if i <= 4 else "🟡"
        print(f"   {priority_emoji} {i}. {rec}")
    print()
    
    # Risk assessment
    critical_vulns = vuln_summary['critical']
    high_vulns = vuln_summary['high']
    successful_exploits = exploit_summary['exploits_successful']
    
    if critical_vulns > 0 or successful_exploits > 0:
        risk_level = "🔴 CRITICAL"
    elif high_vulns > 2:
        risk_level = "🟠 HIGH"
    elif vuln_summary['total_vulnerabilities'] > 3:
        risk_level = "🟡 MEDIUM"
    else:
        risk_level = "🟢 LOW"
    
    print(f"⚠️  OVERALL RISK ASSESSMENT: {risk_level}")
    print()
    
    # Compliance and business impact
    print("📊 BUSINESS IMPACT ASSESSMENT:")
    print("-" * 32)
    
    if successful_exploits > 0:
        print("   💼 Potential data breach risk identified")
        print("   📉 Reputation damage possible")
        print("   💰 Financial impact: HIGH")
    else:
        print("   ✅ No successful exploitations")
        print("   📈 Security posture: Improving")
        print("   💰 Financial impact: LOW")

def display_banner():
    """Display PTaaS banner."""
    banner = """
╔══════════════════════════════════════════════════════════╗
║           XORB AUTONOMOUS PENETRATION TESTING           ║
║              AI-Powered Security Assessment              ║
╠══════════════════════════════════════════════════════════╣
║  🤖 AI Decision Making      🔍 Advanced Reconnaissance  ║
║  💥 Intelligent Exploitation ⚡ EPYC Optimization      ║
║  🛡️  Safety Constraints     📊 Comprehensive Reporting ║
║  🎯 Autonomous Operation    🧠 LLM-Enhanced Analysis    ║
╚══════════════════════════════════════════════════════════╝
    """
    print(banner)

if __name__ == "__main__":
    display_banner()
    asyncio.run(main())