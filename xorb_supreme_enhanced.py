#!/usr/bin/env python3
"""
XORB Supreme Enhanced - AI-Powered Security Platform
Next-generation bug bounty and penetration testing platform with LLM enhancement
"""

import asyncio
import logging
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import enhanced components
from config.enhanced_config_manager import EnhancedConfigManager
from llm.enhanced_multi_provider_client import EnhancedMultiProviderClient, EnhancedLLMRequest, TaskComplexity
from llm.creative_payload_engine import CreativePayloadEngine, VulnerabilityCategory, TargetProfile, PayloadTechnique
from reporting.professional_report_engine import (
    ProfessionalReportEngine, VulnerabilityFinding, ReportMetadata, 
    ReportType, SeverityLevel, ComplianceFramework
)

# Import existing components
from test_hackerone_scraper import HackerOneOpportunitiesScraper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XORBSupremeEnhanced:
    """Next-generation AI-powered security platform"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Initialize configuration
        self.config_manager = EnhancedConfigManager(config_path)
        
        # Initialize core components
        self.llm_client = None
        self.payload_engine = None
        self.report_engine = None
        self.scraper = None
        
        # Runtime state
        self.session_stats = {
            'opportunities_scraped': 0,
            'payloads_generated': 0,
            'reports_created': 0,
            'total_cost': 0.0,
            'session_start': datetime.utcnow()
        }
        
    async def initialize(self):
        """Initialize all system components"""
        logger.info("🚀 Initializing XORB Supreme Enhanced...")
        
        # Initialize LLM client with enhanced config
        llm_config = self._build_llm_config()
        self.llm_client = EnhancedMultiProviderClient(llm_config)
        await self.llm_client.start()
        
        # Initialize specialized engines
        self.payload_engine = CreativePayloadEngine(self.llm_client)
        self.report_engine = ProfessionalReportEngine(self.llm_client)
        self.scraper = HackerOneOpportunitiesScraper()
        
        # Display system status
        await self._display_system_status()
        
        logger.info("✅ XORB Supreme Enhanced initialization complete")
    
    async def shutdown(self):
        """Gracefully shutdown all components"""
        logger.info("🔄 Shutting down XORB Supreme Enhanced...")
        
        if self.llm_client:
            await self.llm_client.close()
        
        # Save session statistics
        await self._save_session_stats()
        
        logger.info("✅ Shutdown complete")
    
    def _build_llm_config(self) -> Dict[str, Any]:
        """Build LLM configuration from enhanced config manager"""
        budget_config = self.config_manager.get_budget_config()
        
        config = {
            'daily_budget_limit': budget_config.daily_limit,
            'monthly_budget_limit': budget_config.monthly_limit,
            'per_request_limit': budget_config.per_request_limit
        }
        
        # Add API keys for enabled providers
        for provider_name in self.config_manager.get_enabled_providers():
            provider_config = self.config_manager.get_llm_provider_config(provider_name)
            if provider_config:
                config[f'{provider_name}_api_key'] = provider_config.api_key
                if provider_config.organization:
                    config[f'{provider_name}_organization'] = provider_config.organization
        
        return config
    
    async def discover_opportunities(self, max_programs: int = 50) -> List[Dict[str, Any]]:
        """Discover bug bounty opportunities using enhanced scraping"""
        logger.info(f"🕷️ Discovering bug bounty opportunities (max: {max_programs})")
        
        try:
            # Enhanced opportunity discovery
            opportunities = await self.scraper.scrape_opportunities()
            
            if opportunities:
                # Enhance opportunities with AI analysis
                enhanced_opportunities = await self._enhance_opportunities_with_ai(opportunities[:max_programs])
                
                self.session_stats['opportunities_scraped'] = len(enhanced_opportunities)
                
                # Save results
                output_file = f"enhanced_opportunities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(output_file, 'w') as f:
                    json.dump(enhanced_opportunities, f, indent=2)
                
                logger.info(f"✅ Discovered {len(enhanced_opportunities)} enhanced opportunities")
                logger.info(f"📄 Results saved to {output_file}")
                
                return enhanced_opportunities
            else:
                logger.warning("❌ No opportunities discovered")
                return []
                
        except Exception as e:
            logger.error(f"❌ Opportunity discovery failed: {e}")
            return []
    
    async def _enhance_opportunities_with_ai(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance scraped opportunities with AI analysis"""
        enhanced = []
        
        for opportunity in opportunities:
            try:
                # Create AI analysis request
                analysis_request = EnhancedLLMRequest(
                    task_type="vulnerability_analysis",
                    prompt=f"""
Analyze this bug bounty opportunity for strategic value:

PROGRAM: {opportunity.get('name', 'Unknown')}
BOUNTY: {opportunity.get('bounty_range', 'Not specified')}
URL: {opportunity.get('url', 'Not available')}

Provide strategic analysis including:
1. ROI potential (1-10 score)
2. Competition level estimate
3. Technical complexity assessment
4. Recommended time investment
5. Success probability estimate

Format as structured analysis for strategic decision making.
""",
                    complexity=TaskComplexity.MODERATE,
                    max_tokens=800,
                    temperature=0.4,
                    use_paid_api=True,
                    budget_limit_usd=0.20
                )
                
                response = await self.llm_client.generate_enhanced_payload(analysis_request)
                
                # Add AI enhancement to opportunity
                opportunity['ai_analysis'] = {
                    'strategic_assessment': response.content,
                    'model_used': response.model_used,
                    'analysis_cost': response.cost_usd,
                    'confidence': response.confidence_score,
                    'enhanced_at': datetime.utcnow().isoformat()
                }
                
                self.session_stats['total_cost'] += response.cost_usd
                
            except Exception as e:
                logger.warning(f"Failed to enhance opportunity {opportunity.get('name', 'Unknown')}: {e}")
                opportunity['ai_analysis'] = {'error': str(e)}
            
            enhanced.append(opportunity)
        
        return enhanced
    
    async def generate_creative_payloads(
        self,
        target_url: str,
        vulnerability_types: List[str],
        creativity_level: float = 0.8,
        count_per_type: int = 5
    ) -> Dict[str, Any]:
        """Generate creative payloads for target using AI enhancement"""
        logger.info(f"🛠️ Generating creative payloads for {target_url}")
        
        # Create target profile
        target_profile = TargetProfile(
            url=target_url,
            technology_stack=['Web Application'],  # Would be enhanced with reconnaissance
            web_server='Unknown',
            database_type='Unknown',
            framework='Unknown',
            language='Unknown',
            operating_system='Linux',
            cloud_provider=None,
            industry_sector=self._detect_industry_sector(target_url)
        )
        
        all_payloads = {}
        total_cost = 0.0
        
        for vuln_type in vulnerability_types:
            try:
                # Map string to enum
                category = self._map_vulnerability_type(vuln_type)
                
                logger.info(f"   Generating {count_per_type} {vuln_type} payloads...")
                
                # Generate creative payloads
                payloads = await self.payload_engine.generate_creative_payloads(
                    category=category,
                    target_profile=target_profile,
                    count=count_per_type,
                    creativity_level=creativity_level,
                    use_paid_api=True
                )
                
                # Convert to serializable format
                payload_data = []
                for payload in payloads:
                    payload_dict = {
                        'payload_content': payload.payload_content,
                        'technique': payload.technique.value,
                        'creativity_score': payload.creativity_score,
                        'bypass_mechanisms': payload.bypass_mechanisms,
                        'explanation': payload.explanation,
                        'evasion_methods': payload.evasion_methods,
                        'detection_difficulty': payload.detection_difficulty,
                        'business_impact': payload.business_impact,
                        'confidence_score': payload.confidence_score
                    }
                    payload_data.append(payload_dict)
                
                all_payloads[vuln_type] = {
                    'payloads': payload_data,
                    'count': len(payloads),
                    'average_creativity': sum(p.creativity_score for p in payloads) / len(payloads) if payloads else 0
                }
                
                self.session_stats['payloads_generated'] += len(payloads)
                
            except Exception as e:
                logger.error(f"Failed to generate {vuln_type} payloads: {e}")
                all_payloads[vuln_type] = {'error': str(e), 'payloads': []}
        
        # Create comprehensive results
        results = {
            'target_profile': {
                'url': target_profile.url,
                'technology_stack': target_profile.technology_stack,
                'industry_sector': target_profile.industry_sector
            },
            'generation_parameters': {
                'creativity_level': creativity_level,
                'count_per_type': count_per_type,
                'vulnerability_types': vulnerability_types
            },
            'payloads_by_category': all_payloads,
            'generation_metadata': {
                'generated_at': datetime.utcnow().isoformat(),
                'total_payloads': sum(cat.get('count', 0) for cat in all_payloads.values()),
                'generation_cost': total_cost,
                'session_id': id(self)
            }
        }
        
        # Save results
        output_file = f"creative_payloads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Generated {results['generation_metadata']['total_payloads']} creative payloads")
        logger.info(f"📄 Results saved to {output_file}")
        
        return results
    
    async def generate_exploitation_chain(
        self,
        initial_vulnerability: str,
        target_url: str,
        objective: str = "complete_compromise"
    ) -> Dict[str, Any]:
        """Generate sophisticated exploitation chain"""
        logger.info(f"⚡ Generating exploitation chain: {initial_vulnerability} → {objective}")
        
        # Create target profile
        target_profile = TargetProfile(
            url=target_url,
            technology_stack=['Web Application'],
            web_server='Unknown',
            database_type='Unknown',
            framework='Unknown',
            language='Unknown',
            operating_system='Linux',
            cloud_provider=None
        )
        
        # Map vulnerability type
        category = self._map_vulnerability_type(initial_vulnerability)
        
        try:
            # Generate exploitation chain
            chain = await self.payload_engine.generate_exploitation_chain(
                initial_vulnerability=category,
                target_profile=target_profile,
                objective=objective,
                use_paid_api=True
            )
            
            # Convert to serializable format
            chain_data = {
                'chain_name': chain.chain_name,
                'initial_vector': chain.initial_vector.value,
                'steps': chain.steps,
                'final_objective': chain.final_objective,
                'stealth_rating': chain.stealth_rating,
                'complexity_level': chain.complexity_level.value,
                'estimated_success_rate': chain.estimated_success_rate,
                'required_privileges': chain.required_privileges,
                'cleanup_steps': chain.cleanup_steps,
                'detection_points': chain.detection_points,
                'mitigation_bypasses': chain.mitigation_bypasses
            }
            
            results = {
                'exploitation_chain': chain_data,
                'target_profile': {
                    'url': target_profile.url,
                    'technology_stack': target_profile.technology_stack
                },
                'generation_metadata': {
                    'generated_at': datetime.utcnow().isoformat(),
                    'initial_vulnerability': initial_vulnerability,
                    'objective': objective
                }
            }
            
            # Save results
            output_file = f"exploitation_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"✅ Generated exploitation chain: {chain.chain_name}")
            logger.info(f"📄 Chain saved to {output_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Exploitation chain generation failed: {e}")
            return {'error': str(e)}
    
    async def generate_professional_reports(
        self,
        findings_data: List[Dict[str, Any]],
        organization: str,
        report_types: List[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive professional security reports"""
        logger.info(f"📊 Generating professional reports for {organization}")
        
        if report_types is None:
            report_types = ['executive', 'technical', 'bug_bounty']
        
        # Convert findings data to VulnerabilityFinding objects
        findings = self._convert_to_vulnerability_findings(findings_data)
        
        # Create report metadata
        metadata = ReportMetadata(
            report_id=f"XORB_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type=ReportType.VULNERABILITY_ASSESSMENT,
            target_organization=organization,
            target_systems=['Web Application'],
            assessment_period={
                'start': datetime.utcnow() - timedelta(days=7),
                'end': datetime.utcnow()
            },
            assessor_name=self.config_manager.get_reporting_config().assessor_name,
            client_contact="Security Team",
            compliance_frameworks=[ComplianceFramework.OWASP]
        )
        
        generated_reports = {}
        total_cost = 0.0
        
        # Generate requested report types
        for report_type in report_types:
            try:
                if report_type == 'executive':
                    logger.info("   Generating executive summary report...")
                    report = await self.report_engine.generate_executive_report(
                        findings=findings,
                        metadata=metadata,
                        use_paid_api=True
                    )
                    generated_reports['executive'] = report
                    total_cost += report.get('generation_cost', 0)
                
                elif report_type == 'technical':
                    logger.info("   Generating technical detailed report...")
                    report = await self.report_engine.generate_technical_report(
                        findings=findings,
                        metadata=metadata,
                        methodology="AI-Enhanced Security Assessment",
                        use_paid_api=True
                    )
                    generated_reports['technical'] = report
                    total_cost += report.get('generation_cost', 0)
                
                elif report_type == 'bug_bounty' and findings:
                    logger.info("   Generating bug bounty submission reports...")
                    bounty_reports = []
                    
                    # Generate reports for critical findings
                    critical_findings = [f for f in findings if f.severity == SeverityLevel.CRITICAL][:3]
                    
                    for finding in critical_findings:
                        bounty_report = await self.report_engine.generate_bug_bounty_report(
                            vulnerability=finding,
                            program_info={'program': organization, 'scope': 'comprehensive'},
                            use_paid_api=True
                        )
                        bounty_reports.append(bounty_report)
                        total_cost += bounty_report.get('generation_cost', 0)
                    
                    generated_reports['bug_bounty'] = bounty_reports
                
            except Exception as e:
                logger.error(f"Failed to generate {report_type} report: {e}")
                generated_reports[report_type] = {'error': str(e)}
        
        # Compile comprehensive results
        results = {
            'reports': generated_reports,
            'metadata': {
                'organization': organization,
                'report_types': report_types,
                'findings_count': len(findings),
                'generated_at': datetime.utcnow().isoformat(),
                'total_generation_cost': total_cost,
                'assessor': metadata.assessor_name
            }
        }
        
        self.session_stats['reports_created'] += len(generated_reports)
        self.session_stats['total_cost'] += total_cost
        
        # Save comprehensive report suite
        output_file = f"professional_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Generated {len(generated_reports)} professional reports")
        logger.info(f"💰 Total generation cost: ${total_cost:.4f}")
        logger.info(f"📄 Reports saved to {output_file}")
        
        return results
    
    async def run_comprehensive_assessment(
        self,
        target_url: str,
        organization: str = None,
        vulnerability_types: List[str] = None,
        creativity_level: float = 0.8
    ) -> Dict[str, Any]:
        """Run comprehensive security assessment with AI enhancement"""
        logger.info(f"🎯 Running comprehensive AI-enhanced assessment for {target_url}")
        
        if organization is None:
            organization = self._extract_organization_from_url(target_url)
        
        if vulnerability_types is None:
            vulnerability_types = ['xss', 'sql_injection', 'ssrf', 'rce']
        
        assessment_results = {
            'target_url': target_url,
            'organization': organization,
            'assessment_start': datetime.utcnow().isoformat()
        }
        
        try:
            # Phase 1: Opportunity Discovery & Analysis
            logger.info("📊 Phase 1: Market Intelligence & Opportunity Analysis")
            opportunities = await self.discover_opportunities()
            assessment_results['market_intelligence'] = {
                'opportunities_discovered': len(opportunities),
                'analysis_complete': True
            }
            
            # Phase 2: Creative Payload Generation
            logger.info("🛠️ Phase 2: AI-Powered Payload Generation")
            payloads = await self.generate_creative_payloads(
                target_url=target_url,
                vulnerability_types=vulnerability_types,
                creativity_level=creativity_level,
                count_per_type=3
            )
            assessment_results['payload_generation'] = payloads['generation_metadata']
            
            # Phase 3: Exploitation Chain Development
            logger.info("⚡ Phase 3: Advanced Exploitation Chain Analysis")
            if vulnerability_types:
                primary_vuln = vulnerability_types[0]
                chain = await self.generate_exploitation_chain(
                    initial_vulnerability=primary_vuln,
                    target_url=target_url,
                    objective="comprehensive_assessment"
                )
                assessment_results['exploitation_analysis'] = chain.get('generation_metadata', {})
            
            # Phase 4: Professional Report Generation
            logger.info("📊 Phase 4: Professional Report Generation")
            
            # Create sample findings for demonstration
            sample_findings = self._create_sample_findings(target_url, payloads)
            
            reports = await self.generate_professional_reports(
                findings_data=sample_findings,
                organization=organization,
                report_types=['executive', 'technical']
            )
            assessment_results['professional_reports'] = reports['metadata']
            
            # Assessment Summary
            assessment_results['assessment_summary'] = {
                'phases_completed': 4,
                'total_payloads_generated': self.session_stats['payloads_generated'],
                'reports_created': self.session_stats['reports_created'],
                'total_cost': self.session_stats['total_cost'],
                'assessment_duration': str(datetime.utcnow() - datetime.fromisoformat(assessment_results['assessment_start'])),
                'assessment_complete': True
            }
            
            # Save comprehensive assessment
            output_file = f"comprehensive_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(assessment_results, f, indent=2)
            
            logger.info("🎉 Comprehensive Assessment Complete!")
            logger.info(f"📄 Full results saved to {output_file}")
            
            return assessment_results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive assessment failed: {e}")
            assessment_results['error'] = str(e)
            return assessment_results
    
    async def _display_system_status(self):
        """Display comprehensive system status"""
        config_summary = self.config_manager.get_configuration_summary()
        budget_config = self.config_manager.get_budget_config()
        
        print("🚀 XORB Supreme Enhanced - System Status")
        print("=" * 60)
        print(f"Version: {config_summary['version']}")
        print(f"Configuration: {config_summary['configuration_file']}")
        print(f"Environment: {config_summary.get('environment_file', 'Not loaded')}")
        print()
        
        print("🧠 LLM PROVIDERS:")
        enabled_providers = config_summary['enabled_providers']
        if enabled_providers:
            for provider in enabled_providers:
                provider_config = self.config_manager.get_llm_provider_config(provider)
                print(f"   ✅ {provider.upper()}: {len(provider_config.models)} models, ${provider_config.daily_budget}/day budget")
        else:
            print("   ⚠️ No paid providers configured - using free tier only")
        
        print(f"   Free Tier Fallback: {'Enabled' if config_summary['free_tier_fallback'] else 'Disabled'}")
        print()
        
        print("💰 BUDGET CONTROLS:")
        print(f"   Daily Limit: ${budget_config.daily_limit}")
        print(f"   Monthly Limit: ${budget_config.monthly_limit}")
        print(f"   Per Request Limit: ${budget_config.per_request_limit}")
        print()
        
        print("🛠️ SYSTEM CAPABILITIES:")
        ai_config = self.config_manager.get_ai_enhancement_config()
        capabilities = [
            "🕷️ HackerOne Opportunity Scraping (API-free)",
            "🧠 AI-Powered Market Intelligence",
            "🛠️ Creative Payload Generation",
            "⚡ Exploitation Chain Development", 
            "📊 Professional Report Generation",
            "🎯 Business Logic Analysis" if ai_config.get('enable_business_logic_analysis') else None,
            "🔗 Polyglot Payload Creation" if ai_config.get('enable_polyglot_payloads') else None,
            "🌐 Multi-Provider LLM Routing"
        ]
        
        for capability in capabilities:
            if capability:
                print(f"   {capability}")
        
        print()
        print("✅ System Ready for Production Operations")
        print()
    
    def _map_vulnerability_type(self, vuln_type: str) -> VulnerabilityCategory:
        """Map string vulnerability type to enum"""
        mapping = {
            'xss': VulnerabilityCategory.XSS,
            'sql_injection': VulnerabilityCategory.SQL_INJECTION,
            'sqli': VulnerabilityCategory.SQL_INJECTION,
            'ssrf': VulnerabilityCategory.SSRF,
            'rce': VulnerabilityCategory.RCE,
            'lfi': VulnerabilityCategory.LFI,
            'idor': VulnerabilityCategory.IDOR,
            'csrf': VulnerabilityCategory.CSRF
        }
        return mapping.get(vuln_type.lower(), VulnerabilityCategory.XSS)
    
    def _detect_industry_sector(self, url: str) -> Optional[str]:
        """Detect industry sector from URL"""
        domain = url.lower()
        
        if any(keyword in domain for keyword in ['bank', 'finance', 'pay']):
            return 'Financial Services'
        elif any(keyword in domain for keyword in ['health', 'medical', 'hospital']):
            return 'Healthcare'
        elif any(keyword in domain for keyword in ['edu', 'school', 'university']):
            return 'Education'
        elif any(keyword in domain for keyword in ['gov', 'government']):
            return 'Government'
        else:
            return 'Technology'
    
    def _extract_organization_from_url(self, url: str) -> str:
        """Extract organization name from URL"""
        from urllib.parse import urlparse
        
        try:
            domain = urlparse(url).netloc or url
            # Remove www. and get main domain name
            domain = domain.replace('www.', '').split('.')[0]
            return domain.title()
        except:
            return 'Target Organization'
    
    def _convert_to_vulnerability_findings(self, findings_data: List[Dict[str, Any]]) -> List[VulnerabilityFinding]:
        """Convert findings data to VulnerabilityFinding objects"""
        findings = []
        
        for i, data in enumerate(findings_data, 1):
            finding = VulnerabilityFinding(
                id=f"XORB-{i:04d}",
                title=data.get('title', f'Vulnerability {i}'),
                severity=SeverityLevel.HIGH,  # Default severity
                cvss_score=data.get('cvss_score', 7.5),
                cvss_vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                description=data.get('description', 'Security vulnerability identified'),
                proof_of_concept=data.get('proof_of_concept', 'PoC available'),
                business_impact=data.get('business_impact', 'High business impact'),
                technical_impact="System compromise possible",
                affected_components=[data.get('component', 'Web Application')],
                remediation_steps=[data.get('remediation', 'Apply security patch')],
                remediation_priority=1,
                estimated_fix_time="1-3 days",
                retest_required=True,
                references=[],
                discovered_at=datetime.utcnow()
            )
            findings.append(finding)
        
        return findings
    
    def _create_sample_findings(self, target_url: str, payload_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create sample findings for demonstration"""
        findings = []
        
        # Extract payload categories that were generated
        payload_categories = payload_results.get('payloads_by_category', {})
        
        for category, data in payload_categories.items():
            if data.get('payloads'):
                sample_payload = data['payloads'][0]  # Use first payload as example
                
                finding = {
                    'title': f'{category.upper()} Vulnerability in {self._extract_organization_from_url(target_url)}',
                    'description': f'Advanced {category} vulnerability identified using AI-generated payload',
                    'cvss_score': min(9.0, 5.0 + sample_payload.get('creativity_score', 5) * 0.4),
                    'proof_of_concept': sample_payload.get('payload_content', 'PoC payload'),
                    'business_impact': sample_payload.get('business_impact', 'Significant security risk'),
                    'component': target_url,
                    'remediation': f'Implement input validation and {category}-specific protections'
                }
                findings.append(finding)
        
        return findings
    
    async def _save_session_stats(self):
        """Save session statistics"""
        stats_file = f"session_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        self.session_stats['session_end'] = datetime.utcnow().isoformat()
        self.session_stats['session_duration'] = str(
            datetime.utcnow() - self.session_stats['session_start']
        )
        
        with open(stats_file, 'w') as f:
            json.dump(self.session_stats, f, indent=2, default=str)
        
        logger.info(f"📊 Session statistics saved to {stats_file}")

async def main():
    """Main CLI interface for XORB Supreme Enhanced"""
    parser = argparse.ArgumentParser(description='XORB Supreme Enhanced - AI-Powered Security Platform')
    
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--status', action='store_true', help='Display system status')
    parser.add_argument('--discover', action='store_true', help='Discover bug bounty opportunities')
    parser.add_argument('--target', help='Target URL for assessment')
    parser.add_argument('--organization', help='Target organization name')
    parser.add_argument('--payloads', nargs='+', default=['xss', 'sqli', 'ssrf'], 
                       help='Vulnerability types for payload generation')
    parser.add_argument('--creativity', type=float, default=0.8, 
                       help='Creativity level for AI generation (0.0-1.0)')
    parser.add_argument('--comprehensive', action='store_true', 
                       help='Run comprehensive assessment')
    parser.add_argument('--reports', nargs='+', default=['executive', 'technical'],
                       help='Report types to generate')
    
    args = parser.parse_args()
    
    # Initialize XORB Supreme Enhanced
    xorb = XORBSupremeEnhanced(config_path=args.config)
    
    try:
        await xorb.initialize()
        
        if args.status:
            # Status already displayed during initialization
            return
        
        elif args.discover:
            await xorb.discover_opportunities()
        
        elif args.comprehensive and args.target:
            await xorb.run_comprehensive_assessment(
                target_url=args.target,
                organization=args.organization,
                vulnerability_types=args.payloads,
                creativity_level=args.creativity
            )
        
        elif args.target:
            # Generate payloads for specific target
            await xorb.generate_creative_payloads(
                target_url=args.target,
                vulnerability_types=args.payloads,
                creativity_level=args.creativity
            )
            
            # Generate reports if requested
            sample_findings = [
                {
                    'title': f'{vuln.upper()} Vulnerability',
                    'description': f'{vuln} vulnerability in target application',
                    'cvss_score': 7.5,
                    'business_impact': 'High'
                }
                for vuln in args.payloads
            ]
            
            await xorb.generate_professional_reports(
                findings_data=sample_findings,
                organization=args.organization or xorb._extract_organization_from_url(args.target),
                report_types=args.reports
            )
        
        else:
            print("🎯 XORB Supreme Enhanced - Ready for Operation")
            print("Use --help for available commands")
            print("Example: python xorb_supreme_enhanced.py --comprehensive --target https://example.com")
    
    finally:
        await xorb.shutdown()

if __name__ == "__main__":
    asyncio.run(main())