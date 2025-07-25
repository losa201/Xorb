#!/usr/bin/env python3
"""
LLM-Enhanced Orchestrator for XORB Supreme
Integrates AI-powered payload generation into campaign execution
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from orchestration.orchestrator import Orchestrator, Campaign, CampaignStatus
from llm.intelligent_client import IntelligentLLMClient, LLMRequest, TaskType
from llm.payload_generator import PayloadGenerator, PayloadCategory, TargetContext, PayloadComplexity
from knowledge_fabric.llm_knowledge_fabric import LLMKnowledgeFabric

logger = logging.getLogger(__name__)

class CampaignPhase(Enum):
    RECONNAISSANCE = "reconnaissance"
    VULNERABILITY_DISCOVERY = "vulnerability_discovery"  
    EXPLOITATION = "exploitation"
    POST_EXPLOITATION = "post_exploitation"
    REPORTING = "reporting"

@dataclass
class LLMCampaignContext:
    """Context for LLM-enhanced campaigns"""
    campaign_id: str
    target_info: Dict[str, Any]
    current_phase: CampaignPhase
    findings: List[Dict[str, Any]]
    generated_payloads: List[str]  # Atom IDs
    ai_insights: List[Dict[str, Any]]
    cost_budget: float = 10.0  # Max LLM cost in USD
    cost_spent: float = 0.0

class LLMOrchestrator(Orchestrator):
    """Enhanced orchestrator with LLM capabilities"""
    
    def __init__(self, redis_url: str, llm_client: IntelligentLLMClient):
        super().__init__(redis_url)
        self.redis_url = redis_url  # Store for LLM knowledge fabric
        self.llm_client = llm_client
        self.payload_generator = PayloadGenerator(llm_client)
        self.llm_knowledge_fabric: Optional[LLMKnowledgeFabric] = None
        
        # LLM campaign tracking
        self.llm_campaigns: Dict[str, LLMCampaignContext] = {}
        self.ai_recommendations: Dict[str, List[Dict[str, Any]]] = {}
        
    async def start(self):
        """Start the LLM orchestrator"""
        await super().start()
        await self.llm_client.start()
        
        # Initialize enhanced knowledge fabric
        if not self.llm_knowledge_fabric:
            self.llm_knowledge_fabric = LLMKnowledgeFabric(
                redis_url=self.redis_url,
                database_url="sqlite+aiosqlite:///./xorb_enhanced.db",
                llm_client=self.llm_client
            )
            await self.llm_knowledge_fabric.initialize()
        
        logger.info("LLM Orchestrator started with AI capabilities")
    
    async def create_ai_enhanced_campaign(
        self,
        name: str,
        targets: List[Dict[str, Any]],
        objectives: List[str],
        budget: float = 10.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create campaign with AI enhancement"""
        
        # Create base campaign
        campaign_id = await self.create_campaign(name, targets, metadata or {})
        
        # Initialize LLM context
        llm_context = LLMCampaignContext(
            campaign_id=campaign_id,
            target_info={"targets": targets, "objectives": objectives},
            current_phase=CampaignPhase.RECONNAISSANCE,
            findings=[],
            generated_payloads=[],
            ai_insights=[],
            cost_budget=budget
        )
        
        self.llm_campaigns[campaign_id] = llm_context
        
        # Generate initial AI reconnaissance plan
        await self._generate_campaign_strategy(campaign_id)
        
        logger.info(f"Created AI-enhanced campaign {campaign_id} with ${budget} budget")
        return campaign_id
    
    async def _generate_campaign_strategy(self, campaign_id: str) -> Dict[str, Any]:
        """Generate AI-powered campaign strategy"""
        
        if campaign_id not in self.llm_campaigns:
            raise ValueError(f"Campaign {campaign_id} not found")
        
        context = self.llm_campaigns[campaign_id]
        
        # Build strategy prompt
        strategy_prompt = f"""
        Create a comprehensive security testing strategy for this campaign:
        
        Campaign: {self.campaigns[campaign_id].name}
        Targets: {json.dumps(context.target_info['targets'], indent=2)}
        Objectives: {', '.join(context.target_info['objectives'])}
        
        Provide a phased approach with:
        1. Reconnaissance techniques and tools
        2. Vulnerability discovery priorities
        3. Exploitation strategy and payload types
        4. Post-exploitation objectives
        5. Success metrics and reporting requirements
        
        Focus on ethical, authorized testing within responsible disclosure principles.
        """
        
        request = LLMRequest(
            task_type=TaskType.EXPLOITATION_STRATEGY,
            prompt=strategy_prompt,
            target_info=context.target_info,
            max_tokens=2000,
            temperature=0.7
        )
        
        try:
            response = await self.llm_client.generate_payload(request)
            
            # Parse strategy
            strategy = self._parse_strategy_response(response)
            
            # Store AI insight
            context.ai_insights.append({
                "type": "campaign_strategy",
                "content": strategy,
                "generated_at": datetime.utcnow().isoformat(),
                "confidence": response.confidence_score,
                "cost": response.cost_usd
            })
            
            context.cost_spent += response.cost_usd
            
            # Store recommendations
            self.ai_recommendations[campaign_id] = strategy.get("recommendations", [])
            
            logger.info(f"Generated AI strategy for campaign {campaign_id}")
            return strategy
            
        except Exception as e:
            logger.error(f"Failed to generate campaign strategy: {e}")
            return {"error": str(e)}
    
    async def advance_campaign_phase(self, campaign_id: str, findings: List[Dict[str, Any]] = None):
        """Advance campaign to next phase with AI enhancement"""
        
        if campaign_id not in self.llm_campaigns:
            raise ValueError(f"Campaign {campaign_id} not found in LLM campaigns")
        
        context = self.llm_campaigns[campaign_id]
        
        # Add new findings
        if findings:
            context.findings.extend(findings)
        
        # Determine next phase
        phase_order = [
            CampaignPhase.RECONNAISSANCE,
            CampaignPhase.VULNERABILITY_DISCOVERY,
            CampaignPhase.EXPLOITATION,
            CampaignPhase.POST_EXPLOITATION,
            CampaignPhase.REPORTING
        ]
        
        current_index = phase_order.index(context.current_phase)
        if current_index < len(phase_order) - 1:
            next_phase = phase_order[current_index + 1]
            context.current_phase = next_phase
            
            # Generate phase-specific content
            await self._generate_phase_content(campaign_id, next_phase)
            
            logger.info(f"Advanced campaign {campaign_id} to phase {next_phase.value}")
        else:
            logger.info(f"Campaign {campaign_id} already in final phase")
    
    async def _generate_phase_content(self, campaign_id: str, phase: CampaignPhase):
        """Generate AI content for specific campaign phase"""
        
        context = self.llm_campaigns[campaign_id]
        
        if phase == CampaignPhase.VULNERABILITY_DISCOVERY:
            await self._generate_discovery_payloads(campaign_id)
        elif phase == CampaignPhase.EXPLOITATION:
            await self._generate_exploitation_strategy(campaign_id)
        elif phase == CampaignPhase.POST_EXPLOITATION:
            await self._generate_persistence_techniques(campaign_id)
        elif phase == CampaignPhase.REPORTING:
            await self._generate_report_enhancements(campaign_id)
    
    async def _generate_discovery_payloads(self, campaign_id: str):
        """Generate vulnerability discovery payloads"""
        
        context = self.llm_campaigns[campaign_id]
        
        # Determine payload categories based on targets
        categories = self._determine_payload_categories(context.target_info)
        
        # Generate payloads for each category
        for category in categories:
            if context.cost_spent >= context.cost_budget:
                logger.warning(f"Campaign {campaign_id} reached cost budget")
                break
            
            # Create target context
            target_context = TargetContext(
                url=context.target_info['targets'][0].get('hostname', 'unknown'),
                technology_stack=context.target_info['targets'][0].get('technology_stack', []),
                operating_system=context.target_info['targets'][0].get('operating_system'),
                parameters=context.target_info['targets'][0].get('parameters', [])
            )
            
            # Generate payloads
            payloads = await self.payload_generator.generate_contextual_payloads(
                category=category,
                target_context=target_context,
                complexity=PayloadComplexity.INTERMEDIATE,
                count=3
            )
            
            # Store payloads in knowledge fabric
            for payload in payloads:
                from llm.intelligent_client import LLMResponse, LLMProvider
                
                mock_response = LLMResponse(
                    content=payload.payload,
                    model_used="campaign_generator",
                    provider=LLMProvider.OPENROUTER,
                    tokens_used=100,
                    cost_usd=0.001,
                    confidence_score=payload.success_probability,
                    generated_at=datetime.utcnow(),
                    request_id=f"campaign_{campaign_id}_{int(datetime.utcnow().timestamp())}"
                )
                
                atom_id = await self.llm_knowledge_fabric.store_llm_payload(
                    payload=payload,
                    llm_response=mock_response,
                    context={"campaign_id": campaign_id, "phase": "discovery"}
                )
                
                context.generated_payloads.append(atom_id)
        
        logger.info(f"Generated {len(context.generated_payloads)} discovery payloads for campaign {campaign_id}")
    
    async def _generate_exploitation_strategy(self, campaign_id: str):
        """Generate exploitation strategy based on findings"""
        
        context = self.llm_campaigns[campaign_id]
        
        if not context.findings:
            logger.warning(f"No findings available for exploitation strategy in campaign {campaign_id}")
            return
        
        # Build exploitation prompt
        findings_summary = "\n".join([
            f"- {finding.get('title', 'Unknown')}: {finding.get('description', '')[:100]}..."
            for finding in context.findings[:5]  # Limit to top 5 findings
        ])
        
        strategy_prompt = f"""
        Based on these security findings, generate an advanced exploitation strategy:
        
        Findings:
        {findings_summary}
        
        Provide:
        1. Prioritized exploitation order
        2. Chained attack techniques
        3. Persistence mechanisms
        4. Data exfiltration methods
        5. Detection evasion techniques
        
        Focus on realistic attack paths and defensive countermeasures.
        """
        
        request = LLMRequest(
            task_type=TaskType.EXPLOITATION_STRATEGY,
            prompt=strategy_prompt,
            target_info={"findings": context.findings},
            max_tokens=2000,
            temperature=0.6
        )
        
        try:
            response = await self.llm_client.generate_payload(request)
            
            # Store exploitation strategy
            context.ai_insights.append({
                "type": "exploitation_strategy",
                "content": response.content,
                "generated_at": datetime.utcnow().isoformat(),
                "confidence": response.confidence_score,
                "cost": response.cost_usd
            })
            
            context.cost_spent += response.cost_usd
            
            logger.info(f"Generated exploitation strategy for campaign {campaign_id}")
            
        except Exception as e:
            logger.error(f"Failed to generate exploitation strategy: {e}")
    
    async def _generate_persistence_techniques(self, campaign_id: str):
        """Generate post-exploitation persistence techniques"""
        
        context = self.llm_campaigns[campaign_id]
        
        persistence_prompt = f"""
        Generate advanced persistence techniques for post-exploitation phase:
        
        Target Environment:
        - OS: {context.target_info['targets'][0].get('operating_system', 'Unknown')}
        - Services: {context.target_info['targets'][0].get('technology_stack', [])}
        
        Provide:
        1. Registry/configuration persistence
        2. Service/daemon persistence
        3. Scheduled task persistence
        4. Network-based persistence
        5. Detection evasion methods
        
        Include both Windows and Linux techniques where applicable.
        """
        
        request = LLMRequest(
            task_type=TaskType.TACTIC_SUGGESTION,
            prompt=persistence_prompt,
            max_tokens=1500,
            temperature=0.5
        )
        
        try:
            response = await self.llm_client.generate_payload(request)
            
            context.ai_insights.append({
                "type": "persistence_techniques",
                "content": response.content,
                "generated_at": datetime.utcnow().isoformat(),
                "confidence": response.confidence_score,
                "cost": response.cost_usd
            })
            
            context.cost_spent += response.cost_usd
            
            logger.info(f"Generated persistence techniques for campaign {campaign_id}")
            
        except Exception as e:
            logger.error(f"Failed to generate persistence techniques: {e}")
    
    async def _generate_report_enhancements(self, campaign_id: str):
        """Generate AI-enhanced report content"""
        
        context = self.llm_campaigns[campaign_id]
        
        # Summarize campaign results
        report_prompt = f"""
        Generate a comprehensive security assessment report summary:
        
        Campaign: {self.campaigns[campaign_id].name}
        Findings: {len(context.findings)} vulnerabilities discovered
        Payloads Generated: {len(context.generated_payloads)}
        AI Insights: {len(context.ai_insights)}
        
        Key Findings:
        {json.dumps(context.findings[:3], indent=2)}
        
        Provide:
        1. Executive summary
        2. Risk assessment and business impact
        3. Technical findings summary
        4. Remediation roadmap with priorities
        5. Strategic security recommendations
        
        Focus on actionable insights and business value.
        """
        
        request = LLMRequest(
            task_type=TaskType.REPORT_ENHANCEMENT,
            prompt=report_prompt,
            max_tokens=2500,
            temperature=0.4
        )
        
        try:
            response = await self.llm_client.generate_payload(request)
            
            # Store enhanced report
            context.ai_insights.append({
                "type": "report_enhancement",
                "content": response.content,
                "generated_at": datetime.utcnow().isoformat(),
                "confidence": response.confidence_score,
                "cost": response.cost_usd
            })
            
            context.cost_spent += response.cost_usd
            
            logger.info(f"Generated enhanced report for campaign {campaign_id}")
            
        except Exception as e:
            logger.error(f"Failed to generate report enhancement: {e}")
    
    def _determine_payload_categories(self, target_info: Dict[str, Any]) -> List[PayloadCategory]:
        """Determine relevant payload categories based on target info"""
        
        categories = [PayloadCategory.XSS, PayloadCategory.SQL_INJECTION]  # Default web app categories
        
        # Add categories based on technology stack
        for target in target_info.get('targets', []):
            tech_stack = target.get('technology_stack', [])
            
            if any('sql' in tech.lower() or 'mysql' in tech.lower() or 'postgres' in tech.lower() 
                   for tech in tech_stack):
                categories.append(PayloadCategory.SQL_INJECTION)
            
            if any('api' in tech.lower() or 'rest' in tech.lower() for tech in tech_stack):
                categories.append(PayloadCategory.SSRF)
            
            if any('linux' in str(target.get('operating_system', '')).lower()):
                categories.append(PayloadCategory.RCE)
            
            if any('file' in str(target.get('input_fields', [])).lower()):
                categories.append(PayloadCategory.LFI)
        
        return list(set(categories))  # Remove duplicates
    
    def _parse_strategy_response(self, response) -> Dict[str, Any]:
        """Parse LLM strategy response"""
        try:
            if response.content.strip().startswith('{'):
                return json.loads(response.content)
        except:
            pass
        
        # Fallback text parsing
        return {
            "raw_strategy": response.content,
            "recommendations": ["Implement comprehensive security testing", "Use AI-powered payload generation"],
            "phases": ["reconnaissance", "discovery", "exploitation", "reporting"]
        }
    
    async def get_campaign_ai_summary(self, campaign_id: str) -> Dict[str, Any]:
        """Get AI-enhanced campaign summary"""
        
        if campaign_id not in self.llm_campaigns:
            return {"error": "Campaign not found in LLM campaigns"}
        
        context = self.llm_campaigns[campaign_id]
        base_campaign = self.campaigns.get(campaign_id)
        
        return {
            "campaign_id": campaign_id,
            "name": base_campaign.name if base_campaign else "Unknown",
            "current_phase": context.current_phase.value,
            "findings_count": len(context.findings),
            "generated_payloads": len(context.generated_payloads),
            "ai_insights_count": len(context.ai_insights),
            "cost_budget": context.cost_budget,
            "cost_spent": context.cost_spent,
            "cost_remaining": context.cost_budget - context.cost_spent,
            "efficiency_score": len(context.findings) / max(context.cost_spent, 0.01),
            "ai_recommendations": self.ai_recommendations.get(campaign_id, [])
        }
    
    async def get_all_llm_campaign_stats(self) -> Dict[str, Any]:
        """Get statistics for all LLM campaigns"""
        
        stats = {
            "total_campaigns": len(self.llm_campaigns),
            "total_cost_spent": sum(ctx.cost_spent for ctx in self.llm_campaigns.values()),
            "total_payloads_generated": sum(len(ctx.generated_payloads) for ctx in self.llm_campaigns.values()),
            "total_findings": sum(len(ctx.findings) for ctx in self.llm_campaigns.values()),
            "campaigns_by_phase": {},
            "average_efficiency": 0.0
        }
        
        # Phase distribution
        for context in self.llm_campaigns.values():
            phase = context.current_phase.value
            stats["campaigns_by_phase"][phase] = stats["campaigns_by_phase"].get(phase, 0) + 1
        
        # Calculate average efficiency (findings per dollar)
        if stats["total_cost_spent"] > 0:
            stats["average_efficiency"] = stats["total_findings"] / stats["total_cost_spent"]
        
        return stats
    
    async def cleanup_completed_campaigns(self, max_age_days: int = 30):
        """Clean up old completed campaigns"""
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
        campaigns_to_remove = []
        
        for campaign_id, context in self.llm_campaigns.items():
            base_campaign = self.campaigns.get(campaign_id)
            
            if (base_campaign and 
                base_campaign.status == CampaignStatus.COMPLETED and
                base_campaign.created_at < cutoff_date):
                
                campaigns_to_remove.append(campaign_id)
        
        # Remove old campaigns
        for campaign_id in campaigns_to_remove:
            del self.llm_campaigns[campaign_id]
            if campaign_id in self.ai_recommendations:
                del self.ai_recommendations[campaign_id]
        
        logger.info(f"Cleaned up {len(campaigns_to_remove)} old campaigns")
        return len(campaigns_to_remove)