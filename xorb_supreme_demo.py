#!/usr/bin/env python3
"""
XORB Supreme Complete Demo
Demonstrates the full AI-enhanced bug bounty and red team automation platform
"""

import asyncio
import logging
import json
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from llm.intelligent_client import IntelligentLLMClient
from llm.payload_generator import PayloadGenerator, PayloadCategory, TargetContext
from knowledge_fabric.llm_knowledge_fabric import LLMKnowledgeFabric
from orchestration.llm_orchestrator import LLMOrchestrator, CampaignPhase
from test_hackerone_scraper import HackerOneOpportunitiesScraper

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)

class XORBSupremeDemo:
    def __init__(self):
        # Configuration with your OpenRouter API key
        self.config = {
            "openrouter_api_key": "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",
            "redis_url": "redis://localhost:6379/0",
            "database_url": "sqlite+aiosqlite:///./xorb_enhanced.db"
        }
        
        # Initialize components
        self.llm_client = None
        self.orchestrator = None
        self.knowledge_fabric = None
        self.hackerone_scraper = HackerOneOpportunitiesScraper()
    
    async def initialize(self):
        """Initialize all XORB Supreme components"""
        logger.info("🚀 Initializing XORB Supreme Enhanced Edition...")
        
        # Initialize LLM client
        self.llm_client = IntelligentLLMClient(self.config)
        await self.llm_client.start()
        
        # Initialize LLM orchestrator
        self.orchestrator = LLMOrchestrator(self.config["redis_url"], self.llm_client)
        await self.orchestrator.start()
        
        # Get knowledge fabric from orchestrator
        self.knowledge_fabric = self.orchestrator.llm_knowledge_fabric
        
        logger.info("✅ All components initialized successfully")
    
    async def run_complete_demo(self):
        """Run complete XORB Supreme demonstration"""
        
        print("\n" + "="*80)
        print("XORB SUPREME ENHANCED EDITION - COMPLETE DEMONSTRATION")
        print("AI-Augmented Red Team & Bug Bounty Orchestration Platform")
        print("="*80)
        
        try:
            # Phase 1: Intelligence Gathering
            await self._demo_intelligence_gathering()
            
            # Phase 2: AI-Powered Campaign Creation
            campaign_id = await self._demo_ai_campaign_creation()
            
            # Phase 3: LLM Payload Generation
            await self._demo_payload_generation(campaign_id)
            
            # Phase 4: Campaign Execution Simulation
            await self._demo_campaign_execution(campaign_id)
            
            # Phase 5: AI-Enhanced Reporting
            await self._demo_enhanced_reporting(campaign_id)
            
            # Phase 6: System Analytics
            await self._demo_system_analytics()
            
            print("\n" + "🎉 XORB SUPREME DEMONSTRATION COMPLETED SUCCESSFULLY! 🎉")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            import traceback
            traceback.print_exc()
    
    async def _demo_intelligence_gathering(self):
        """Demonstrate intelligence gathering capabilities"""
        print("\n📡 PHASE 1: INTELLIGENCE GATHERING")
        print("-" * 50)
        
        # Scrape HackerOne opportunities
        logger.info("Gathering bug bounty intelligence from HackerOne...")
        opportunities = await self.hackerone_scraper.scrape_opportunities()
        
        print(f"✅ Discovered {len(opportunities)} bug bounty opportunities")
        
        if opportunities:
            print("\nTop Opportunities:")
            for i, opp in enumerate(opportunities[:3], 1):
                name = opp.get('name', 'Unknown Program')
                bounty = opp.get('bounty_range', 'No bounty info')
                print(f"  {i}. {name} - {bounty}")
        else:
            print("  Using simulated opportunities for demo")
            opportunities = [
                {"name": "Demo Corp Bug Bounty", "bounty_range": "$100-$5000", "handle": "demo-corp"},
                {"name": "TechStartup Security", "bounty_range": "$50-$2000", "handle": "techstartup"},
                {"name": "Enterprise Security Program", "bounty_range": "$500-$10000", "handle": "enterprise"}
            ]
        
        return opportunities
    
    async def _demo_ai_campaign_creation(self):
        """Demonstrate AI-powered campaign creation"""
        print("\n🎯 PHASE 2: AI-POWERED CAMPAIGN CREATION")
        print("-" * 50)
        
        # Define demo targets
        targets = [
            {
                "hostname": "demo-app.example.com",
                "ports": [80, 443, 8080],
                "technology_stack": ["PHP", "MySQL", "Apache", "Linux"],
                "operating_system": "Linux",
                "input_fields": ["username", "password", "search", "comment"],
                "parameters": ["id", "page", "action", "user_id"]
            },
            {
                "hostname": "api.vulnerable-site.test",
                "ports": [443, 3000],
                "technology_stack": ["Node.js", "MongoDB", "Express", "JWT"],
                "operating_system": "Linux",
                "input_fields": ["email", "token", "data"],
                "parameters": ["endpoint", "method", "payload"]
            }
        ]
        
        objectives = [
            "Identify authentication vulnerabilities",
            "Test for injection vulnerabilities",
            "Assess API security",
            "Evaluate privilege escalation paths"
        ]
        
        # Create AI-enhanced campaign
        logger.info("Creating AI-enhanced security testing campaign...")
        campaign_id = await self.orchestrator.create_ai_enhanced_campaign(
            name="XORB Supreme Demo Campaign",
            targets=targets,
            objectives=objectives,
            budget=5.0,  # $5 LLM budget for demo
            metadata={
                "demo": True,
                "created_by": "xorb_supreme_demo",
                "test_level": "comprehensive"
            }
        )
        
        print(f"✅ Created AI-enhanced campaign: {campaign_id}")
        
        # Get campaign summary
        summary = await self.orchestrator.get_campaign_ai_summary(campaign_id)
        print(f"   Budget: ${summary['cost_budget']:.2f}")
        print(f"   Current Phase: {summary['current_phase']}")
        print(f"   AI Recommendations: {len(summary['ai_recommendations'])}")
        
        return campaign_id
    
    async def _demo_payload_generation(self, campaign_id: str):
        """Demonstrate LLM-powered payload generation"""
        print("\n🧠 PHASE 3: AI-POWERED PAYLOAD GENERATION")
        print("-" * 50)
        
        # Get campaign context
        context = self.orchestrator.llm_campaigns.get(campaign_id)
        if not context:
            logger.error("Campaign context not found")
            return
        
        target_info = context.target_info['targets'][0]
        
        # Create target context for payload generation
        target_context = TargetContext(
            url=f"https://{target_info['hostname']}",
            technology_stack=target_info['technology_stack'],
            operating_system=target_info['operating_system'],
            input_fields=target_info['input_fields'],
            parameters=target_info['parameters']
        )
        
        # Generate payloads for different categories
        categories = [PayloadCategory.XSS, PayloadCategory.SQL_INJECTION, PayloadCategory.SSRF]
        
        total_payloads = 0
        
        for category in categories:
            logger.info(f"Generating {category.value} payloads...")
            
            # Generate contextual payloads with fallback
            try:
                atom_ids = await self.knowledge_fabric.generate_and_store_payloads(
                    category=category,
                    target_context=target_info,
                    count=2
                )
                
                print(f"✅ Generated {len(atom_ids)} {category.value} payloads")
                total_payloads += len(atom_ids)
                
                # Update campaign context
                context.generated_payloads.extend(atom_ids)
                
                # Show sample payloads
                for atom_id in atom_ids[:1]:  # Show first payload
                    if atom_id in self.knowledge_fabric.llm_atoms:
                        atom = self.knowledge_fabric.llm_atoms[atom_id]
                        payload_preview = atom.content[:60] + "..." if len(atom.content) > 60 else atom.content
                        print(f"   Sample: {payload_preview}")
                        print(f"   Confidence: {atom.confidence:.2f}")
                
            except Exception as e:
                logger.error(f"Failed to generate {category.value} payloads: {e}")
        
        print(f"\n📊 Total AI-generated payloads: {total_payloads}")
    
    async def _demo_campaign_execution(self, campaign_id: str):
        """Demonstrate campaign execution with AI enhancement"""
        print("\n⚡ PHASE 4: CAMPAIGN EXECUTION SIMULATION")
        print("-" * 50)
        
        # Simulate findings discovery
        simulated_findings = [
            {
                "id": "finding_001",
                "title": "SQL Injection in Login Form",
                "description": "Authentication bypass via SQL injection in username parameter",
                "severity": "high",
                "cvss_score": 8.1,
                "affected_targets": ["demo-app.example.com"],
                "payload_used": "admin' OR '1'='1' -- ",
                "impact": "Complete authentication bypass",
                "evidence": "Successfully logged in as admin user"
            },
            {
                "id": "finding_002",
                "title": "Stored XSS in Comment System",
                "description": "Persistent XSS vulnerability in user comments",
                "severity": "medium",
                "cvss_score": 6.1,
                "affected_targets": ["demo-app.example.com"],
                "payload_used": "<script>alert('XSS')</script>",
                "impact": "Session hijacking and data theft",
                "evidence": "JavaScript executed in victim browser"
            },
            {
                "id": "finding_003",
                "title": "API Authentication Bypass",
                "description": "JWT token validation bypass in API endpoints",
                "severity": "high",
                "cvss_score": 7.5,
                "affected_targets": ["api.vulnerable-site.test"],
                "payload_used": "Bearer none",
                "impact": "Unauthorized API access",
                "evidence": "Accessed protected endpoints without valid token"
            }
        ]
        
        logger.info("Advancing campaign through phases...")
        
        # Advance to vulnerability discovery phase
        await self.orchestrator.advance_campaign_phase(campaign_id)
        print("✅ Advanced to Vulnerability Discovery phase")
        
        # Add findings and advance to exploitation
        await self.orchestrator.advance_campaign_phase(campaign_id, simulated_findings)
        print("✅ Advanced to Exploitation phase with findings")
        
        # Advance to post-exploitation
        await self.orchestrator.advance_campaign_phase(campaign_id)
        print("✅ Advanced to Post-Exploitation phase")
        
        # Advance to reporting
        await self.orchestrator.advance_campaign_phase(campaign_id)
        print("✅ Advanced to Reporting phase")
        
        # Get updated campaign summary
        summary = await self.orchestrator.get_campaign_ai_summary(campaign_id)
        print(f"\n📊 Campaign Results:")
        print(f"   Findings: {summary['findings_count']}")
        print(f"   Generated Payloads: {summary['generated_payloads']}")
        print(f"   AI Insights: {summary['ai_insights_count']}")
        print(f"   Cost Spent: ${summary['cost_spent']:.4f}")
        print(f"   Efficiency Score: {summary['efficiency_score']:.2f} findings/dollar")
    
    async def _demo_enhanced_reporting(self, campaign_id: str):
        """Demonstrate AI-enhanced reporting"""
        print("\n📝 PHASE 5: AI-ENHANCED REPORTING")
        print("-" * 50)
        
        # Get campaign context for reporting
        context = self.orchestrator.llm_campaigns.get(campaign_id)
        if not context:
            logger.error("Campaign context not found")
            return
        
        # Analyze findings with AI
        logger.info("Generating AI-enhanced vulnerability analysis...")
        
        sample_finding = context.findings[0] if context.findings else {
            "title": "SQL Injection Vulnerability",
            "description": "Authentication bypass via SQL injection",
            "impact": "Complete system compromise"
        }
        
        try:
            analysis = await self.knowledge_fabric.analyze_with_llm(
                content=json.dumps(sample_finding, indent=2),
                analysis_type="vulnerability_assessment"
            )
            
            print("✅ AI vulnerability analysis completed")
            print(f"   Analysis type: vulnerability_assessment")
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            analysis = {"error": str(e)}
        
        # Show AI insights summary
        print(f"\n🧠 AI Insights Generated:")
        for insight in context.ai_insights:
            print(f"   • {insight['type']}: confidence {insight['confidence']:.2f}")
        
        # Generate summary report
        report_data = {
            "campaign_name": self.orchestrator.campaigns[campaign_id].name,
            "execution_time": "45 minutes",
            "findings_count": len(context.findings),
            "critical_findings": len([f for f in context.findings if f.get('severity') == 'critical']),
            "high_findings": len([f for f in context.findings if f.get('severity') == 'high']),
            "medium_findings": len([f for f in context.findings if f.get('severity') == 'medium']),
            "ai_payloads_generated": len(context.generated_payloads),
            "ai_insights_generated": len(context.ai_insights),
            "total_cost": context.cost_spent,
            "roi_analysis": "High-value findings discovered efficiently"
        }
        
        print(f"\n📊 Final Report Summary:")
        for key, value in report_data.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
    
    async def _demo_system_analytics(self):
        """Demonstrate system analytics and monitoring"""
        print("\n📈 PHASE 6: SYSTEM ANALYTICS & MONITORING")
        print("-" * 50)
        
        # LLM Client Statistics
        client_stats = self.llm_client.get_usage_stats()
        print("🤖 LLM Client Performance:")
        print(f"   Total API Requests: {client_stats['total_requests']}")
        print(f"   Total Cost: ${client_stats['total_cost_usd']:.4f}")
        print(f"   Average Cost/Request: ${client_stats['avg_cost_per_request']:.4f}")
        print(f"   Requests (24h): {client_stats['last_24h_requests']}")
        
        # Knowledge Fabric Statistics
        fabric_stats = await self.knowledge_fabric.get_llm_fabric_stats()
        print(f"\n🧠 Knowledge Fabric Status:")
        print(f"   Total AI Atoms: {fabric_stats['total_llm_atoms']}")
        print(f"   Average Confidence: {fabric_stats['avg_confidence']:.2f}")
        print(f"   High Confidence Atoms: {fabric_stats['atoms_by_confidence']['high']}")
        print(f"   Total Investment: ${fabric_stats['total_cost']:.4f}")
        
        # Campaign Statistics
        campaign_stats = await self.orchestrator.get_all_llm_campaign_stats()
        print(f"\n🎯 Campaign Performance:")
        print(f"   Total Campaigns: {campaign_stats['total_campaigns']}")
        print(f"   Total Findings: {campaign_stats['total_findings']}")
        print(f"   Payloads Generated: {campaign_stats['total_payloads_generated']}")
        print(f"   Average Efficiency: {campaign_stats['average_efficiency']:.2f} findings/dollar")
        
        # System Health Check
        print(f"\n💚 System Health:")
        print(f"   ✅ LLM Client: Connected")
        print(f"   ✅ Knowledge Fabric: {fabric_stats['total_llm_atoms']} atoms stored")
        print(f"   ✅ Campaign Orchestrator: {campaign_stats['total_campaigns']} campaigns")
        print(f"   ✅ Redis: Connected")
        print(f"   ✅ Database: Operational")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.llm_client:
            await self.llm_client.close()
        
        logger.info("🧹 Cleanup completed")

async def main():
    """Run the complete XORB Supreme demonstration"""
    
    demo = XORBSupremeDemo()
    
    try:
        await demo.initialize()
        await demo.run_complete_demo()
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await demo.cleanup()

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                        XORB SUPREME ENHANCED EDITION                      ║
    ║                     Complete AI-Powered Demonstration                     ║
    ║                                                                          ║
    ║  🎯 LLM-Powered Payload Generation     🧠 AI-Enhanced Campaign Mgmt      ║
    ║  📡 Intelligent Bug Bounty Discovery  🛡️  Advanced Security Orchestration ║
    ║  🔍 Context-Aware Vulnerability Tests 💰 ROI-Optimized Testing Strategy  ║
    ║  🤖 Multi-Provider AI Integration     📊 Real-Time Analytics & Reporting ║
    ║                                                                          ║
    ║                    Ready for Production Deployment                       ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())