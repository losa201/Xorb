#!/usr/bin/env python3
"""
Enhanced PTaaS Agent Demonstration Script
==========================================

This script demonstrates the enhanced capabilities of the PTaaS Agent including:
- Compliance scanning for major frameworks (PCI-DSS, HIPAA, SOX, etc.)
- AI-powered threat analysis and recommendations
- Enhanced vulnerability assessment and prioritization
- Comprehensive reporting with compliance and AI insights

Usage:
    python demo_enhanced_ptaas_agent.py

Prerequisites:
    - XORB API service running (http://localhost:8000)
    - Valid API token
    - Python 3.8+
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from agents.ptaas_agent import PtaasAgent
except ImportError as e:
    print(f"Error importing PTaaS Agent: {e}")
    print("Please ensure you're running this script from the project root directory.")
    sys.exit(1)

# Demo configuration
DEMO_CONFIG = {
    "api_token": "demo_token_12345",  # Replace with actual token
    "api_base_url": "http://localhost:8000/api/v1",
    "demo_targets": [
        {
            "name": "Web Application Server",
            "ip": "192.168.1.100",
            "domain": "webapp.example.com",
            "description": "Production web application server"
        },
        {
            "name": "Database Server", 
            "ip": "192.168.1.200",
            "domain": "db.example.com",
            "description": "Customer database server (PCI-DSS scope)"
        },
        {
            "name": "Healthcare API",
            "ip": "192.168.1.150",
            "domain": "api.healthcare.com", 
            "description": "Healthcare API server (HIPAA scope)"
        }
    ],
    "compliance_scenarios": {
        "webapp.example.com": ["PCI-DSS", "SOX"],
        "db.example.com": ["PCI-DSS"],
        "api.healthcare.com": ["HIPAA", "GDPR"]
    }
}

def print_banner():
    """Print the demo banner."""
    print("=" * 80)
    print("🛡️  ENHANCED PTaaS AGENT DEMONSTRATION")
    print("=" * 80)
    print("📅 Demo Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("🤖 Agent Version: 2.0 - Enhanced with Compliance & AI")
    print("🔧 Capabilities:")
    print("   • Real-world scanner integration (Nmap, Nuclei, Nikto, SSLScan)")
    print("   • Compliance framework assessment (PCI-DSS, HIPAA, SOX, etc.)")
    print("   • AI-powered threat analysis and MITRE ATT&CK mapping")
    print("   • Enhanced vulnerability prioritization and exploitation")
    print("   • Comprehensive reporting with executive summaries")
    print("=" * 80)
    print()

def print_section_header(title, description=""):
    """Print a formatted section header."""
    print("\n" + "🔸" * 60)
    print(f"📋 {title}")
    if description:
        print(f"   {description}")
    print("🔸" * 60)

def demonstrate_agent_capabilities():
    """Demonstrate the enhanced PTaaS agent capabilities."""
    
    print_banner()
    
    # Initialize the enhanced agent
    print_section_header("AGENT INITIALIZATION", "Setting up the Enhanced PTaaS Agent")
    
    agent = PtaasAgent(
        id="demo_enhanced_ptaas_001",
        resource_level=1.0,
        api_token=DEMO_CONFIG["api_token"],
        api_base_url=DEMO_CONFIG["api_base_url"],
        skill_level=0.85
    )
    
    print(f"✅ Agent initialized: {agent.id}")
    print(f"🎯 Skill Level: {agent.skill_level}")
    print(f"🔧 Supported Frameworks: {', '.join(agent.supported_frameworks)}")
    print(f"⚙️ Configuration: {json.dumps(agent.config, indent=2)}")
    
    # Demonstrate telemetry
    print("\n📊 Agent Telemetry:")
    telemetry = agent.get_telemetry()
    print(json.dumps(telemetry, indent=2, default=str))
    
    # Run demonstrations for each target
    all_reports = []
    
    for i, target in enumerate(DEMO_CONFIG["demo_targets"], 1):
        print_section_header(
            f"DEMONSTRATION {i}/3: {target['name']}", 
            f"Target: {target['domain']} ({target['ip']})"
        )
        
        print(f"📝 Description: {target['description']}")
        
        # Get compliance frameworks for this target
        compliance_frameworks = DEMO_CONFIG["compliance_scenarios"].get(target["domain"], [])
        
        if compliance_frameworks:
            print(f"⚖️  Compliance Frameworks: {', '.join(compliance_frameworks)}")
        else:
            print("⚖️  Compliance Frameworks: None specified")
        
        # Prepare target information
        target_info = {
            "ip": target["ip"],
            "domain": target["domain"],
            "stealth_mode": False  # Demo mode - not stealthy
        }
        
        print(f"\n🚀 Starting enhanced penetration test...")
        print(f"🎯 Target: {target_info}")
        
        try:
            # Run the enhanced penetration test
            start_time = time.time()
            
            report = agent.run_pentest(
                target=target_info,
                compliance_frameworks=compliance_frameworks,
                enable_ai_analysis=True
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"\n✅ Penetration test completed in {duration:.1f} seconds")
            
            # Display key results
            print_test_summary(report)
            
            # Store report for later analysis
            all_reports.append({
                "target": target,
                "report": report,
                "duration": duration
            })
            
        except Exception as e:
            print(f"❌ Error during penetration test: {e}")
            continue
        
        # Pause between demonstrations
        if i < len(DEMO_CONFIG["demo_targets"]):
            print("\n⏸️  Pausing for 3 seconds before next demonstration...")
            time.sleep(3)
    
    # Final summary and analysis
    print_section_header("DEMONSTRATION SUMMARY", "Overall Results and Analysis")
    
    if all_reports:
        print_overall_summary(all_reports)
        
        # Save detailed reports
        save_demo_reports(all_reports)
        
    else:
        print("❌ No successful penetration tests completed.")
    
    print("\n🎉 Enhanced PTaaS Agent demonstration completed!")
    print("📁 Detailed reports saved to 'demo_reports/' directory")
    print("🔍 Review the reports for comprehensive security findings and recommendations")

def print_test_summary(report):
    """Print a summary of the penetration test results."""
    
    # Executive summary
    exec_summary = report.get('executive_summary', {})
    print(f"\n📊 EXECUTIVE SUMMARY:")
    print(f"   🎯 Overall Risk Level: {exec_summary.get('overall_risk_level', 'UNKNOWN')}")
    print(f"   🔍 Total Vulnerabilities: {exec_summary.get('total_vulnerabilities', 0)}")
    print(f"   🚨 Critical Issues: {exec_summary.get('critical_issues', 0)}")
    print(f"   ⚠️  High Issues: {exec_summary.get('high_issues', 0)}")
    print(f"   💥 Exploitation Success Rate: {exec_summary.get('exploitation_success_rate', 0):.1%}")
    
    # Vulnerability breakdown
    vuln_analysis = report.get('vulnerability_analysis', {})
    severity_breakdown = vuln_analysis.get('severity_breakdown', {})
    
    print(f"\n🔍 VULNERABILITY BREAKDOWN:")
    for severity, count in severity_breakdown.items():
        if count > 0:
            print(f"   📌 {severity.upper()}: {count}")
    
    # Compliance results
    compliance_assessment = report.get('compliance_assessment', {})
    if compliance_assessment and compliance_assessment.get('results'):
        print(f"\n⚖️  COMPLIANCE ASSESSMENT:")
        for framework, result in compliance_assessment['results'].items():
            status = result.get('status', 'UNKNOWN')
            score = result.get('score', 0)
            print(f"   📋 {framework}: {status} (Score: {score:.1f}%)")
    
    # AI Analysis
    ai_analysis = report.get('ai_analysis', {})
    if ai_analysis:
        print(f"\n🤖 AI THREAT ANALYSIS:")
        print(f"   🎯 Threat Level: {ai_analysis.get('threat_level', 'UNKNOWN')}")
        print(f"   🎲 Confidence: {ai_analysis.get('confidence', 0):.1%}")
        print(f"   📊 Risk Score: {ai_analysis.get('risk_score', 0)}/100")
        
        attack_patterns = ai_analysis.get('attack_patterns', [])
        if attack_patterns:
            print(f"   🎭 Attack Patterns: {', '.join(attack_patterns[:3])}")
        
        mitre_techniques = ai_analysis.get('mitre_techniques', [])
        if mitre_techniques:
            print(f"   🎖️  MITRE Techniques: {', '.join(mitre_techniques[:3])}")
    
    # Top recommendations
    recommendations = report.get('recommendations', [])
    if recommendations:
        print(f"\n💡 TOP RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"   {i}. {rec}")

def print_overall_summary(all_reports):
    """Print an overall summary of all demonstrations."""
    
    total_targets = len(all_reports)
    total_vulns = sum(r['report'].get('executive_summary', {}).get('total_vulnerabilities', 0) for r in all_reports)
    total_critical = sum(r['report'].get('executive_summary', {}).get('critical_issues', 0) for r in all_reports)
    total_high = sum(r['report'].get('executive_summary', {}).get('high_issues', 0) for r in all_reports)
    avg_duration = sum(r['duration'] for r in all_reports) / total_targets if total_targets else 0
    
    print(f"📈 OVERALL STATISTICS:")
    print(f"   🎯 Targets Assessed: {total_targets}")
    print(f"   🔍 Total Vulnerabilities Found: {total_vulns}")
    print(f"   🚨 Total Critical Issues: {total_critical}")
    print(f"   ⚠️  Total High Issues: {total_high}")
    print(f"   ⏱️  Average Scan Duration: {avg_duration:.1f} seconds")
    
    # Compliance summary
    compliance_frameworks = set()
    compliant_count = 0
    total_compliance_checks = 0
    
    for report_data in all_reports:
        compliance_results = report_data['report'].get('compliance_assessment', {}).get('results', {})
        for framework, result in compliance_results.items():
            compliance_frameworks.add(framework)
            total_compliance_checks += 1
            if result.get('status') == 'COMPLIANT':
                compliant_count += 1
    
    if total_compliance_checks > 0:
        compliance_rate = (compliant_count / total_compliance_checks) * 100
        print(f"\n⚖️  COMPLIANCE SUMMARY:")
        print(f"   📋 Frameworks Assessed: {', '.join(compliance_frameworks)}")
        print(f"   ✅ Compliance Rate: {compliance_rate:.1f}% ({compliant_count}/{total_compliance_checks})")
    
    # Risk assessment
    risk_levels = [r['report'].get('executive_summary', {}).get('overall_risk_level', 'UNKNOWN') for r in all_reports]
    critical_targets = sum(1 for level in risk_levels if level == 'CRITICAL')
    high_targets = sum(1 for level in risk_levels if level == 'HIGH')
    
    print(f"\n🎯 RISK ASSESSMENT:")
    print(f"   🚨 Critical Risk Targets: {critical_targets}")
    print(f"   ⚠️  High Risk Targets: {high_targets}")
    print(f"   📊 Medium/Low Risk Targets: {total_targets - critical_targets - high_targets}")

def save_demo_reports(all_reports):
    """Save detailed reports to files."""
    
    # Create reports directory
    reports_dir = Path("demo_reports")
    reports_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save individual reports
    for i, report_data in enumerate(all_reports, 1):
        target_name = report_data['target']['name'].replace(' ', '_').lower()
        filename = f"ptaas_report_{i}_{target_name}_{timestamp}.json"
        filepath = reports_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"📄 Saved report: {filepath}")
    
    # Save summary report
    summary_filename = f"ptaas_summary_{timestamp}.json"
    summary_filepath = reports_dir / summary_filename
    
    summary_data = {
        "demonstration_metadata": {
            "timestamp": datetime.now().isoformat(),
            "agent_version": "2.0_enhanced",
            "total_targets": len(all_reports),
            "demo_config": DEMO_CONFIG
        },
        "reports": all_reports
    }
    
    with open(summary_filepath, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    print(f"📊 Saved summary: {summary_filepath}")

def main():
    """Main function to run the demonstration."""
    
    # Check if we're in demo mode or real mode
    if len(sys.argv) > 1 and sys.argv[1] == "--real":
        print("🔴 REAL MODE: This will attempt to connect to actual XORB API")
        print("⚠️  Make sure the XORB API is running and you have valid credentials")
        
        # Get API token from environment or prompt
        api_token = os.getenv('XORB_API_TOKEN')
        if not api_token:
            api_token = input("🔑 Enter your XORB API token: ")
            if not api_token:
                print("❌ API token required for real mode")
                return
        
        DEMO_CONFIG["api_token"] = api_token
        
        # Confirm real targets
        print("🎯 Demo will test against the following targets:")
        for target in DEMO_CONFIG["demo_targets"]:
            print(f"   • {target['domain']} ({target['ip']})")
        
        confirm = input("\n⚠️  Proceed with real penetration testing? (yes/no): ")
        if confirm.lower() != 'yes':
            print("❌ Demonstration cancelled")
            return
    
    else:
        print("🟡 DEMO MODE: Using simulated API responses")
        print("💡 Use --real flag to test against actual XORB API")
        print("⚠️  Demo mode will simulate scan results for demonstration purposes")
        
        input("\n📋 Press Enter to start the demonstration...")
    
    try:
        demonstrate_agent_capabilities()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()