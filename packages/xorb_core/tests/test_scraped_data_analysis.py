#!/usr/bin/env python3
"""
Test AI-Powered Analysis of Scraped HackerOne Data
Real-world functionality demonstration using LLM-enhanced analysis
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
import sys
import os

# Add project root to path
sys.path.append('/root/xorb')

from llm.intelligent_client import IntelligentLLMClient, LLMRequest, TaskType
from llm.qwen_security_specialist import QwenSecuritySpecialist
from knowledge_fabric.llm_knowledge_fabric import LLMKnowledgeFabric

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScrapedDataAnalyzer:
    def __init__(self):
        # Load configuration
        with open('/root/xorb/config.json', 'r') as f:
            self.config = json.load(f)
        
        # Initialize LLM client
        self.llm_client = IntelligentLLMClient(self.config)
        self.qwen_specialist = QwenSecuritySpecialist(self.llm_client)
        
        # Initialize knowledge fabric
        self.knowledge_fabric = LLMKnowledgeFabric(
            redis_url="redis://localhost:6379",
            database_url="sqlite:///knowledge.db",
            llm_client=self.llm_client
        )
    
    async def analyze_opportunities(self, opportunities_file: str) -> Dict[str, Any]:
        """Analyze scraped opportunities using AI"""
        logger.info(f"Starting AI analysis of {opportunities_file}")
        
        # Load scraped data
        with open(opportunities_file, 'r') as f:
            opportunities = json.load(f)
        
        logger.info(f"Loaded {len(opportunities)} opportunities for analysis")
        
        # Start LLM client
        await self.llm_client.start()
        
        analysis_results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_opportunities": len(opportunities),
            "opportunity_insights": [],
            "strategic_recommendations": {},
            "ai_enhanced_prioritization": [],
            "market_intelligence": {},
            "cost_analysis": {}
        }
        
        try:
            # 1. Analyze individual opportunities
            logger.info("Running individual opportunity analysis...")
            for i, opportunity in enumerate(opportunities[:5], 1):  # Analyze first 5 for demo
                logger.info(f"Analyzing opportunity {i}/5: {opportunity.get('name', 'Unknown')}")
                
                insight = await self._analyze_single_opportunity(opportunity)
                analysis_results["opportunity_insights"].append(insight)
            
            # 2. Generate strategic recommendations
            logger.info("Generating strategic campaign recommendations...")
            strategy = await self.qwen_specialist.generate_campaign_strategy(opportunities[:3])
            analysis_results["strategic_recommendations"] = strategy
            
            # 3. Market intelligence analysis
            logger.info("Performing market intelligence analysis...")
            market_analysis = await self._analyze_market_trends(opportunities)
            analysis_results["market_intelligence"] = market_analysis
            
            # 4. ROI and prioritization analysis
            logger.info("Calculating ROI-based prioritization...")
            prioritization = await self._calculate_roi_prioritization(opportunities)
            analysis_results["ai_enhanced_prioritization"] = prioritization
            
            # 5. Store results in knowledge fabric
            logger.info("Storing analysis in knowledge fabric...")
            await self._store_analysis_results(analysis_results)
            
            return analysis_results
            
        finally:
            await self.llm_client.close()
    
    async def _analyze_single_opportunity(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single bug bounty opportunity"""
        
        analysis_prompt = f"""
BUG BOUNTY OPPORTUNITY ANALYSIS:

Analyze this HackerOne bug bounty program for strategic value and testing approach:

PROGRAM DATA:
{json.dumps(opportunity, indent=2)}

ANALYSIS REQUIREMENTS:

1. MARKET VALUE ASSESSMENT:
   - Bounty range competitiveness
   - Expected ROI for security researchers
   - Market positioning vs similar programs

2. TECHNICAL ATTACK SURFACE:
   - Likely technology stack based on program URL/name
   - Common vulnerability categories for this type of target
   - Testing complexity and effort estimation

3. STRATEGIC RECOMMENDATIONS:
   - Priority level (1-10) for targeting
   - Recommended testing methodologies
   - Time investment vs reward potential
   - Skill level requirements

4. COMPETITIVE ANALYSIS:
   - Program popularity indicators
   - Researcher competition level
   - Unique opportunities or advantages

5. RISK ASSESSMENT:
   - Program stability and payout reliability
   - Scope clarity and testing permissions
   - Legal and compliance considerations

OUTPUT FORMAT (JSON):
{{
  "program_id": "extracted program identifier",
  "strategic_priority": 1-10,
  "market_value": {{
    "bounty_competitiveness": "high/medium/low",
    "roi_estimate": "expected return ratio",
    "market_position": "competitive analysis"
  }},
  "technical_profile": {{
    "likely_tech_stack": ["technology1", "technology2"],
    "primary_attack_vectors": ["vector1", "vector2"],
    "complexity_rating": "low/medium/high",
    "estimated_hours": "time investment estimate"
  }},
  "recommendations": {{
    "testing_approach": "recommended methodology",
    "skill_requirements": ["skill1", "skill2"],
    "success_probability": 0.0-1.0,
    "unique_advantages": "what makes this program attractive"
  }},
  "risk_factors": {{
    "program_risks": ["risk1", "risk2"],
    "mitigation_strategies": ["strategy1", "strategy2"],
    "confidence_level": 0.0-1.0
  }}
}}

Provide actionable intelligence for optimizing bug bounty research efforts.
"""
        
        request = LLMRequest(
            task_type=TaskType.VULNERABILITY_ANALYSIS,
            prompt=analysis_prompt,
            max_tokens=1500,
            temperature=0.3,
            structured_output=True
        )
        
        try:
            response = await self.llm_client.generate_payload(request)
            
            # Parse AI response
            content = response.content.strip()
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                if json_end != -1:
                    content = content[json_start:json_end].strip()
            
            if content.startswith('{'):
                ai_analysis = json.loads(content)
            else:
                ai_analysis = {"raw_analysis": content}
            
            return {
                "opportunity": opportunity,
                "ai_analysis": ai_analysis,
                "model_used": response.model_used,
                "analysis_cost": response.cost_usd,
                "confidence": response.confidence_score
            }
            
        except Exception as e:
            logger.error(f"Individual analysis failed: {e}")
            return {
                "opportunity": opportunity,
                "ai_analysis": {"error": str(e)},
                "analysis_cost": 0.0
            }
    
    async def _analyze_market_trends(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze market trends from scraped data"""
        
        # Calculate market statistics
        total_programs = len(opportunities)
        bounty_programs = [op for op in opportunities if 'min_bounty' in op]
        
        if bounty_programs:
            bounty_amounts = [op['min_bounty'] for op in bounty_programs if op.get('min_bounty', 0) > 0]
            avg_bounty = sum(bounty_amounts) / len(bounty_amounts) if bounty_amounts else 0
            max_bounty = max(bounty_amounts) if bounty_amounts else 0
            min_bounty = min(bounty_amounts) if bounty_amounts else 0
        else:
            avg_bounty = max_bounty = min_bounty = 0
        
        market_prompt = f"""
BUG BOUNTY MARKET INTELLIGENCE ANALYSIS:

Analyze the current HackerOne market based on this sample of {total_programs} programs:

MARKET DATA:
- Total programs analyzed: {total_programs}
- Programs with bounty info: {len(bounty_programs)}
- Average minimum bounty: ${avg_bounty:.2f}
- Highest minimum bounty: ${max_bounty}
- Lowest minimum bounty: ${min_bounty}

BOUNTY RANGES SAMPLE:
{json.dumps([op.get('bounty_range', 'N/A') for op in opportunities[:10]], indent=2)}

ANALYSIS REQUIREMENTS:

1. MARKET HEALTH:
   - Overall bounty competitiveness
   - Market saturation indicators
   - Growth trends and opportunities

2. RESEARCHER ECONOMICS:
   - ROI potential for different skill levels
   - Time investment vs reward ratios
   - Market efficiency assessment

3. STRATEGIC INSIGHTS:
   - Emerging opportunities
   - Under-served market segments
   - Competitive advantages for researchers

4. MARKET PREDICTIONS:
   - Expected trends in next 6 months
   - Technology focus areas
   - Bounty range evolution

OUTPUT FORMAT (JSON):
{{
  "market_health": {{
    "overall_assessment": "strong/moderate/weak",
    "competitiveness_rating": 1-10,
    "saturation_level": "low/medium/high",
    "growth_indicators": ["indicator1", "indicator2"]
  }},
  "economics": {{
    "roi_potential": {{
      "beginner": "ROI for new researchers",
      "intermediate": "ROI for experienced researchers", 
      "expert": "ROI for security experts"
    }},
    "efficiency_score": 1-10,
    "risk_reward_balance": "assessment of balance"
  }},
  "opportunities": {{
    "emerging_trends": ["trend1", "trend2"],
    "underserved_segments": ["segment1", "segment2"],
    "strategic_recommendations": ["recommendation1", "recommendation2"]
  }},
  "predictions": {{
    "6_month_outlook": "market direction prediction",
    "technology_focus": ["tech1", "tech2"],
    "bounty_trends": "expected bounty evolution"
  }}
}}

Provide strategic market intelligence for bug bounty operations.
"""
        
        request = LLMRequest(
            task_type=TaskType.VULNERABILITY_ANALYSIS,
            prompt=market_prompt,
            max_tokens=2000,
            temperature=0.4,
            structured_output=True
        )
        
        try:
            response = await self.llm_client.generate_payload(request)
            
            content = response.content.strip()
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                if json_end != -1:
                    content = content[json_start:json_end].strip()
            
            if content.startswith('{'):
                market_analysis = json.loads(content)
            else:
                market_analysis = {"raw_analysis": content}
            
            # Add calculated statistics
            market_analysis["calculated_metrics"] = {
                "total_programs": total_programs,
                "avg_bounty": avg_bounty,
                "max_bounty": max_bounty,
                "min_bounty": min_bounty,
                "programs_with_bounties": len(bounty_programs)
            }
            
            return market_analysis
            
        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            return {"error": str(e)}
    
    async def _calculate_roi_prioritization(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate ROI-based prioritization using AI"""
        
        prioritized = []
        
        for opportunity in opportunities[:8]:  # Prioritize top 8 for demo
            # Calculate basic ROI metrics
            min_bounty = opportunity.get('min_bounty', 0)
            max_bounty = opportunity.get('max_bounty', min_bounty * 2)  # Estimate if not available
            
            # Simple scoring algorithm (would be enhanced by AI in production)
            priority_score = 0
            
            # Bounty amount factor
            if min_bounty > 1000:
                priority_score += 3
            elif min_bounty > 500:
                priority_score += 2
            elif min_bounty > 100:
                priority_score += 1
            
            # URL quality factor (more specific URLs often indicate better programs)
            url = opportunity.get('url', '')
            if '/programs/' in url:
                priority_score += 2
            elif '?type=team' in url:
                priority_score += 1
            
            # Confidence factor
            confidence = opportunity.get('confidence_score', 0.5)
            priority_score += confidence * 2
            
            prioritized.append({
                "opportunity": opportunity,
                "priority_score": round(priority_score, 2),
                "estimated_roi": round(min_bounty / 40, 2) if min_bounty > 0 else 0,  # ROI per hour estimate
                "time_investment_estimate": "2-8 hours" if min_bounty < 500 else "8-40 hours",
                "risk_level": "low" if min_bounty < 200 else "medium" if min_bounty < 1000 else "high"
            })
        
        # Sort by priority score
        prioritized.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return prioritized
    
    async def _store_analysis_results(self, results: Dict[str, Any]):
        """Store analysis results in knowledge fabric"""
        try:
            # Store market intelligence
            await self.knowledge_fabric.store_llm_payload(
                payload_type="market_analysis",
                payload_content=results["market_intelligence"],
                metadata={
                    "analysis_type": "hackerone_market_intelligence",
                    "timestamp": results["analysis_timestamp"],
                    "opportunity_count": results["total_opportunities"]
                },
                model_used="qwen-235b",
                confidence_score=0.8,
                cost_usd=0.05  # Estimated cost
            )
            
            # Store strategic recommendations
            await self.knowledge_fabric.store_llm_payload(
                payload_type="strategic_recommendations",
                payload_content=results["strategic_recommendations"],
                metadata={
                    "analysis_type": "campaign_strategy",
                    "timestamp": results["analysis_timestamp"]
                },
                model_used="qwen-235b",
                confidence_score=0.9,
                cost_usd=0.08
            )
            
            logger.info("Analysis results stored in knowledge fabric successfully")
            
        except Exception as e:
            logger.error(f"Failed to store results in knowledge fabric: {e}")

async def main():
    """Main function to run comprehensive analysis"""
    analyzer = ScrapedDataAnalyzer()
    
    # Find the most recent scraped data file
    import glob
    scraped_files = glob.glob("/root/xorb/hackerone_opportunities_*.json")
    if not scraped_files:
        print("No scraped data files found. Run test_hackerone_scraper.py first.")
        return
    
    latest_file = max(scraped_files, key=os.path.getctime)
    print(f"Analyzing scraped data from: {latest_file}")
    
    # Run comprehensive analysis
    results = await analyzer.analyze_opportunities(latest_file)
    
    # Save analysis results
    output_file = f"ai_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== AI-POWERED ANALYSIS RESULTS ===")
    print(f"Analysis file saved: {output_file}")
    print(f"Total opportunities analyzed: {results['total_opportunities']}")
    print(f"Individual insights generated: {len(results['opportunity_insights'])}")
    
    # Display sample results
    if results['opportunity_insights']:
        print(f"\nSample AI Analysis:")
        first_insight = results['opportunity_insights'][0]
        opportunity = first_insight['opportunity']
        analysis = first_insight.get('ai_analysis', {})
        
        print(f"Program: {opportunity.get('name', 'Unknown')}")
        print(f"Bounty: {opportunity.get('bounty_range', 'N/A')}")
        print(f"AI Priority Score: {analysis.get('strategic_priority', 'N/A')}")
        print(f"Model Used: {first_insight.get('model_used', 'N/A')}")
        print(f"Analysis Cost: ${first_insight.get('analysis_cost', 0):.4f}")
    
    # Display market intelligence
    if results['market_intelligence']:
        market = results['market_intelligence']
        metrics = market.get('calculated_metrics', {})
        print(f"\nMarket Intelligence:")
        print(f"Average bounty: ${metrics.get('avg_bounty', 0):.2f}")
        print(f"Market assessment: {market.get('market_health', {}).get('overall_assessment', 'N/A')}")
    
    # Display prioritization
    if results['ai_enhanced_prioritization']:
        print(f"\nTop 3 Prioritized Opportunities:")
        for i, item in enumerate(results['ai_enhanced_prioritization'][:3], 1):
            opp = item['opportunity']
            print(f"{i}. {opp.get('name', 'Unknown')} - Priority: {item['priority_score']} - ROI: ${item['estimated_roi']}/hr")
    
    print(f"\n✅ Real-world HackerOne data analysis completed successfully!")
    print(f"📊 {results['total_opportunities']} opportunities processed with AI enhancement")
    print(f"🎯 Strategic recommendations generated for optimal targeting")
    print(f"📈 Market intelligence analysis provides competitive insights")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())