#!/usr/bin/env python3
"""
XORB Supreme Production System
Real-world AI-enhanced bug bounty and penetration testing platform
"""

import asyncio
import logging
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from llm.intelligent_client import IntelligentLLMClient
from llm.payload_generator import PayloadGenerator, PayloadCategory, TargetContext, PayloadComplexity
from knowledge_fabric.llm_knowledge_fabric import LLMKnowledgeFabric
from test_hackerone_scraper import HackerOneOpportunitiesScraper
from integrations.hackerone_client import HackerOneClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
logger = logging.getLogger(__name__)

class XORBSupremeProduction:
    """Production XORB Supreme system for real bug bounty operations"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config = self._load_config(config_file)
        
        # Core components
        self.llm_client = None
        self.payload_generator = None
        self.knowledge_fabric = None
        self.hackerone_client = None
        self.hackerone_scraper = None
        
        # Operation state
        self.active_campaigns = {}
        self.discovered_programs = []
        self.generated_payloads = {}
        
    def _load_config(self, config_file: str) -> dict:
        """Load configuration from file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Validate required keys
            required_keys = ["openrouter_api_key", "redis_url", "database_url"]
            for key in required_keys:
                if not config.get(key):
                    logger.warning(f"Missing or empty config key: {key}")
            
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._default_config()
    
    def _default_config(self) -> dict:
        """Default configuration"""
        return {
            "openrouter_api_key": "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",
            "hackerone_api_key": "",
            "redis_url": "redis://localhost:6379/0",
            "database_url": "sqlite+aiosqlite:///./xorb_production.db",
            "security_level": "production",
            "deployment_mode": "production",
            "llm_budget": 50.0,  # $50 monthly LLM budget
            "auto_submission": False,  # Require manual approval
            "rate_limit_delay": 2.0,  # Seconds between requests
            "max_concurrent_targets": 5
        }
    
    async def initialize(self):
        """Initialize all production components"""
        logger.info("Initializing XORB Supreme Production System...")
        
        # Initialize LLM client
        self.llm_client = IntelligentLLMClient(self.config)
        await self.llm_client.start()
        logger.info("✅ LLM client initialized")
        
        # Initialize payload generator
        self.payload_generator = PayloadGenerator(self.llm_client)
        logger.info("✅ Payload generator initialized")
        
        # Initialize knowledge fabric
        self.knowledge_fabric = LLMKnowledgeFabric(
            redis_url=self.config["redis_url"],
            database_url=self.config["database_url"],
            llm_client=self.llm_client
        )
        await self.knowledge_fabric.initialize()
        logger.info("✅ Knowledge fabric initialized")
        
        # Initialize HackerOne integrations
        if self.config.get("hackerone_api_key"):
            self.hackerone_client = HackerOneClient(
                api_key=self.config["hackerone_api_key"]
            )
            await self.hackerone_client.start()
            logger.info("✅ HackerOne API client initialized")
        else:
            logger.warning("HackerOne API key not configured - API features disabled")
        
        self.hackerone_scraper = HackerOneOpportunitiesScraper()
        logger.info("✅ HackerOne scraper initialized")
        
        logger.info("🚀 XORB Supreme Production System fully operational")
    
    async def discover_opportunities(self) -> list:
        """Discover and analyze bug bounty opportunities"""
        logger.info("Discovering bug bounty opportunities...")
        
        # Scrape HackerOne opportunities
        opportunities = await self.hackerone_scraper.scrape_opportunities()
        logger.info(f"Found {len(opportunities)} opportunities from web scraping")
        
        # Get programs via API if available
        if self.hackerone_client:
            try:
                api_programs = await self.hackerone_client.get_programs(eligible_only=True)
                logger.info(f"Retrieved {len(api_programs)} programs via API")
                
                # Merge API and scraped data
                for program in api_programs:
                    opportunities.append({
                        "name": program.name,
                        "handle": program.handle,
                        "url": program.url,
                        "bounty_enabled": program.bounty_enabled,
                        "average_bounty": program.average_bounty_lower_amount,
                        "source": "api"
                    })
            except Exception as e:
                logger.error(f"API program retrieval failed: {e}")
        
        # Filter and prioritize opportunities
        self.discovered_programs = self._prioritize_opportunities(opportunities)
        
        logger.info(f"Prioritized {len(self.discovered_programs)} high-value opportunities")
        return self.discovered_programs
    
    def _prioritize_opportunities(self, opportunities: list) -> list:
        """Prioritize opportunities based on value and feasibility"""
        prioritized = []
        
        for opp in opportunities:
            score = 0
            
            # Bounty value scoring
            if opp.get("min_bounty"):
                if opp["min_bounty"] >= 1000:
                    score += 10
                elif opp["min_bounty"] >= 500:
                    score += 7
                elif opp["min_bounty"] >= 100:
                    score += 5
                else:
                    score += 2
            
            # Program activity scoring
            if opp.get("bounty_enabled"):
                score += 5
            
            # Add priority score
            opp["priority_score"] = score
            prioritized.append(opp)
        
        # Sort by priority score
        prioritized.sort(key=lambda x: x.get("priority_score", 0), reverse=True)
        
        return prioritized[:20]  # Top 20 opportunities
    
    async def generate_target_payloads(self, target_url: str, target_context: dict) -> dict:
        """Generate AI-powered payloads for specific target"""
        logger.info(f"Generating payloads for target: {target_url}")
        
        # Create target context
        context = TargetContext(
            url=target_url,
            technology_stack=target_context.get("technology_stack", []),
            operating_system=target_context.get("operating_system"),
            web_server=target_context.get("web_server"),
            input_fields=target_context.get("input_fields", []),
            parameters=target_context.get("parameters", [])
        )
        
        payload_results = {}
        categories = [
            PayloadCategory.XSS,
            PayloadCategory.SQL_INJECTION,
            PayloadCategory.SSRF,
            PayloadCategory.RCE,
            PayloadCategory.LFI
        ]
        
        for category in categories:
            try:
                logger.info(f"Generating {category.value} payloads...")
                
                payloads = await self.payload_generator.generate_contextual_payloads(
                    category=category,
                    target_context=context,
                    complexity=PayloadComplexity.ADVANCED,
                    count=5
                )
                
                # Store in knowledge fabric
                atom_ids = []
                for payload in payloads:
                    from llm.intelligent_client import LLMResponse, LLMProvider
                    
                    mock_response = LLMResponse(
                        content=payload.payload,
                        model_used="production_generator",
                        provider=LLMProvider.OPENROUTER,
                        tokens_used=100,
                        cost_usd=0.001,
                        confidence_score=payload.success_probability,
                        generated_at=datetime.utcnow(),
                        request_id=f"prod_{int(datetime.utcnow().timestamp())}"
                    )
                    
                    atom_id = await self.knowledge_fabric.store_llm_payload(
                        payload=payload,
                        llm_response=mock_response,
                        context={"target": target_url, "production": True}
                    )
                    atom_ids.append(atom_id)
                
                payload_results[category.value] = {
                    "payloads": payloads,
                    "atom_ids": atom_ids,
                    "count": len(payloads)
                }
                
                logger.info(f"Generated {len(payloads)} {category.value} payloads")
                
            except Exception as e:
                logger.error(f"Failed to generate {category.value} payloads: {e}")
                payload_results[category.value] = {"error": str(e)}
        
        self.generated_payloads[target_url] = payload_results
        return payload_results
    
    async def analyze_target(self, target_url: str) -> dict:
        """Perform comprehensive target analysis"""
        logger.info(f"Analyzing target: {target_url}")
        
        analysis_results = {
            "target": target_url,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "technology_analysis": {},
            "vulnerability_assessment": {},
            "attack_surface": {},
            "recommendations": []
        }
        
        # AI-powered technology stack analysis
        tech_analysis_prompt = f"""
        Analyze this target for security testing:
        URL: {target_url}
        
        Provide structured analysis including:
        1. Likely technology stack
        2. Common vulnerability patterns
        3. Recommended testing approaches
        4. Potential attack vectors
        5. Risk assessment
        
        Focus on actionable intelligence for authorized penetration testing.
        """
        
        try:
            from llm.intelligent_client import LLMRequest
            request = LLMRequest(
                task_type="vulnerability_analysis",
                prompt=tech_analysis_prompt,
                max_tokens=2000,
                temperature=0.3
            )
            
            response = await self.llm_client.generate_payload(request)
            
            analysis_results["ai_analysis"] = {
                "content": response.content,
                "confidence": response.confidence_score,
                "model_used": response.model_used,
                "cost": response.cost_usd
            }
            
            logger.info(f"AI analysis completed for {target_url}")
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            analysis_results["ai_analysis"] = {"error": str(e)}
        
        return analysis_results
    
    async def execute_testing_campaign(self, target_config: dict) -> dict:
        """Execute comprehensive testing campaign"""
        target_url = target_config["url"]
        logger.info(f"Executing testing campaign for: {target_url}")
        
        campaign_results = {
            "target": target_url,
            "campaign_id": f"campaign_{int(datetime.utcnow().timestamp())}",
            "start_time": datetime.utcnow().isoformat(),
            "phases": {},
            "findings": [],
            "total_cost": 0.0
        }
        
        # Phase 1: Target Analysis
        logger.info("Phase 1: Target Analysis")
        analysis = await self.analyze_target(target_url)
        campaign_results["phases"]["analysis"] = analysis
        
        # Phase 2: Payload Generation
        logger.info("Phase 2: Payload Generation")
        payloads = await self.generate_target_payloads(target_url, target_config)
        campaign_results["phases"]["payload_generation"] = payloads
        
        # Phase 3: Vulnerability Assessment (simulated for safety)
        logger.info("Phase 3: Vulnerability Assessment")
        assessment_results = await self._simulate_vulnerability_testing(target_url, payloads)
        campaign_results["phases"]["vulnerability_testing"] = assessment_results
        campaign_results["findings"] = assessment_results.get("findings", [])
        
        # Phase 4: Report Generation
        logger.info("Phase 4: Report Generation")
        report = await self._generate_campaign_report(campaign_results)
        campaign_results["phases"]["reporting"] = report
        
        campaign_results["end_time"] = datetime.utcnow().isoformat()
        campaign_results["total_cost"] = self._calculate_campaign_cost(campaign_results)
        
        logger.info(f"Campaign completed for {target_url}")
        logger.info(f"Findings: {len(campaign_results['findings'])}")
        logger.info(f"Total cost: ${campaign_results['total_cost']:.4f}")
        
        return campaign_results
    
    async def _simulate_vulnerability_testing(self, target_url: str, payloads: dict) -> dict:
        """Simulate vulnerability testing (for safety - no actual testing)"""
        logger.info("Simulating vulnerability testing (no actual network requests)")
        
        # This would contain actual testing logic in a real deployment
        # For safety, we simulate findings based on payload analysis
        
        simulated_findings = []
        
        for category, payload_data in payloads.items():
            if "payloads" in payload_data:
                for payload in payload_data["payloads"][:2]:  # Simulate 2 findings per category
                    if payload.success_probability > 0.6:  # High confidence payloads
                        finding = {
                            "id": f"finding_{len(simulated_findings) + 1}",
                            "category": category,
                            "title": f"Potential {category.replace('_', ' ').title()} Vulnerability", 
                            "description": payload.description,
                            "severity": self._calculate_severity(payload.success_probability),
                            "payload": payload.payload,
                            "confidence": payload.success_probability,
                            "detection_difficulty": payload.detection_difficulty,
                            "remediation": payload.remediation,
                            "target_parameter": payload.target_parameter,
                            "expected_result": payload.expected_result
                        }
                        simulated_findings.append(finding)
        
        return {
            "method": "simulated",
            "findings_count": len(simulated_findings),
            "findings": simulated_findings,
            "high_severity": len([f for f in simulated_findings if f["severity"] == "high"]),
            "medium_severity": len([f for f in simulated_findings if f["severity"] == "medium"]),
            "low_severity": len([f for f in simulated_findings if f["severity"] == "low"])
        }
    
    def _calculate_severity(self, success_probability: float) -> str:
        """Calculate severity based on success probability"""
        if success_probability >= 0.8:
            return "high"
        elif success_probability >= 0.6:
            return "medium"
        else:
            return "low"
    
    async def _generate_campaign_report(self, campaign_results: dict) -> dict:
        """Generate comprehensive campaign report"""
        logger.info("Generating campaign report...")
        
        findings = campaign_results.get("findings", [])
        high_findings = [f for f in findings if f["severity"] == "high"]
        
        report_prompt = f"""
        Generate a professional security assessment report summary:
        
        Target: {campaign_results['target']}
        Campaign Duration: {campaign_results.get('start_time')} to {campaign_results.get('end_time')}
        Total Findings: {len(findings)}
        High Severity: {len(high_findings)}
        
        Key Findings:
        {json.dumps(findings[:3], indent=2)}
        
        Provide:
        1. Executive summary with business impact
        2. Technical risk assessment
        3. Prioritized remediation roadmap
        4. Strategic security recommendations
        5. Next steps and follow-up actions
        """
        
        try:
            from llm.intelligent_client import LLMRequest
            request = LLMRequest(
                task_type="report_enhancement",
                prompt=report_prompt,
                max_tokens=2500,
                temperature=0.4
            )
            
            response = await self.llm_client.generate_payload(request)
            
            return {
                "report_content": response.content,
                "confidence": response.confidence_score,
                "model_used": response.model_used,
                "cost": response.cost_usd,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {
                "error": str(e),
                "fallback_report": self._generate_fallback_report(campaign_results)
            }
    
    def _generate_fallback_report(self, campaign_results: dict) -> dict:
        """Generate fallback report without LLM"""
        findings = campaign_results.get("findings", [])
        
        return {
            "executive_summary": f"Security assessment completed for {campaign_results['target']}",
            "findings_summary": f"Identified {len(findings)} potential security issues",
            "high_priority_count": len([f for f in findings if f["severity"] == "high"]),
            "recommendations": ["Implement input validation", "Use parameterized queries", "Deploy WAF"],
            "next_steps": ["Validate findings", "Implement fixes", "Retest vulnerabilities"]
        }
    
    def _calculate_campaign_cost(self, campaign_results: dict) -> float:
        """Calculate total campaign cost"""
        total_cost = 0.0
        
        # Add LLM costs from each phase
        for phase_name, phase_data in campaign_results.get("phases", {}).items():
            if isinstance(phase_data, dict):
                if "ai_analysis" in phase_data and "cost" in phase_data["ai_analysis"]:
                    total_cost += phase_data["ai_analysis"]["cost"]
                if "cost" in phase_data:
                    total_cost += phase_data["cost"]
        
        return total_cost
    
    async def run_production_campaign(self, targets: list) -> dict:
        """Run complete production campaign"""
        logger.info(f"Starting production campaign with {len(targets)} targets")
        
        campaign_summary = {
            "campaign_start": datetime.utcnow().isoformat(),
            "targets_processed": 0,
            "total_findings": 0,
            "total_cost": 0.0,
            "target_results": {}
        }
        
        for target_config in targets:
            try:
                # Rate limiting
                await asyncio.sleep(self.config.get("rate_limit_delay", 2.0))
                
                # Execute campaign for target
                target_results = await self.execute_testing_campaign(target_config)
                
                # Update summary
                campaign_summary["targets_processed"] += 1
                campaign_summary["total_findings"] += len(target_results.get("findings", []))
                campaign_summary["total_cost"] += target_results.get("total_cost", 0.0)
                campaign_summary["target_results"][target_config["url"]] = target_results
                
                logger.info(f"Completed target {target_config['url']}")
                
                # Check budget limits
                if campaign_summary["total_cost"] >= self.config.get("llm_budget", 50.0):
                    logger.warning("LLM budget limit reached, stopping campaign")
                    break
                
            except Exception as e:
                logger.error(f"Failed to process target {target_config['url']}: {e}")
        
        campaign_summary["campaign_end"] = datetime.utcnow().isoformat()
        
        logger.info("Production campaign completed")
        logger.info(f"Targets processed: {campaign_summary['targets_processed']}")
        logger.info(f"Total findings: {campaign_summary['total_findings']}")
        logger.info(f"Total cost: ${campaign_summary['total_cost']:.4f}")
        
        return campaign_summary
    
    async def get_system_status(self) -> dict:
        """Get comprehensive system status"""
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_health": "operational",
            "components": {}
        }
        
        # LLM client status
        if self.llm_client:
            llm_stats = self.llm_client.get_usage_stats()
            status["components"]["llm_client"] = {
                "status": "active",
                "total_requests": llm_stats["total_requests"],
                "total_cost": llm_stats["total_cost_usd"],
                "budget_remaining": self.config.get("llm_budget", 50.0) - llm_stats["total_cost_usd"]
            }
        
        # Knowledge fabric status
        if self.knowledge_fabric:
            fabric_stats = await self.knowledge_fabric.get_llm_fabric_stats()
            status["components"]["knowledge_fabric"] = {
                "status": "active",
                "total_atoms": fabric_stats["total_llm_atoms"],
                "average_confidence": fabric_stats["avg_confidence"]
            }
        
        # HackerOne integration status
        status["components"]["hackerone"] = {
            "api_client": "active" if self.hackerone_client else "disabled",
            "scraper": "active" if self.hackerone_scraper else "disabled",
            "discovered_programs": len(self.discovered_programs)
        }
        
        return status
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down XORB Supreme...")
        
        if self.llm_client:
            await self.llm_client.close()
        
        if self.hackerone_client:
            await self.hackerone_client.close()
        
        logger.info("✅ XORB Supreme shutdown complete")

async def main():
    """Main production entry point"""
    parser = argparse.ArgumentParser(description="XORB Supreme Production System")
    parser.add_argument("--config", default="config.json", help="Configuration file")
    parser.add_argument("--targets", help="JSON file with target configurations")
    parser.add_argument("--discover", action="store_true", help="Discover bug bounty opportunities")
    parser.add_argument("--status", action="store_true", help="Show system status")
    
    args = parser.parse_args()
    
    # Initialize system
    xorb = XORBSupremeProduction(args.config)
    
    try:
        await xorb.initialize()
        
        if args.status:
            # Show system status
            status = await xorb.get_system_status()
            print(json.dumps(status, indent=2))
            
        elif args.discover:
            # Discover opportunities
            opportunities = await xorb.discover_opportunities()
            print(f"Discovered {len(opportunities)} opportunities")
            for opp in opportunities[:10]:
                print(f"- {opp.get('name', 'Unknown')}: Score {opp.get('priority_score', 0)}")
        
        elif args.targets:
            # Run campaign on specified targets
            with open(args.targets, 'r') as f:
                targets = json.load(f)
            
            results = await xorb.run_production_campaign(targets)
            
            # Save results
            output_file = f"campaign_results_{int(datetime.utcnow().timestamp())}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Campaign completed. Results saved to {output_file}")
        
        else:
            print("XORB Supreme Production System ready")
            print("Use --discover to find opportunities")
            print("Use --targets <file> to run campaigns")
            print("Use --status to check system health")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Production system error: {e}")
        raise
    finally:
        await xorb.shutdown()

if __name__ == "__main__":
    asyncio.run(main())