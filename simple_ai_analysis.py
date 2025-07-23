#!/usr/bin/env python3
"""
Simplified AI Analysis of Scraped HackerOne Data
Real-world functionality demonstration without complex dependencies
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleHackerOneAnalyzer:
    def __init__(self):
        # Load configuration
        with open('/root/xorb/config.json', 'r') as f:
            self.config = json.load(f)
        
        # Initialize LLM client
        self.llm_client = IntelligentLLMClient(self.config)
    
    async def analyze_scraped_opportunities(self, opportunities_file: str):
        """Analyze scraped opportunities using AI"""
        logger.info(f"Starting AI analysis of {opportunities_file}")
        
        # Load scraped data
        with open(opportunities_file, 'r') as f:
            opportunities = json.load(f)
        
        logger.info(f"Loaded {len(opportunities)} opportunities for analysis")
        
        # Start LLM client
        await self.llm_client.start()
        
        try:
            # 1. Analyze market trends
            logger.info("Analyzing market trends...")
            market_analysis = await self._analyze_market_intelligence(opportunities)
            
            # 2. Prioritize opportunities
            logger.info("Generating AI-powered prioritization...")
            prioritization = await self._generate_smart_prioritization(opportunities[:5])
            
            # 3. Generate testing strategies
            logger.info("Creating testing strategies...")
            strategies = await self._generate_testing_strategies(opportunities[:3])
            
            # Compile results
            analysis_results = {
                "analysis_timestamp": datetime.now().isoformat(),
                "total_opportunities": len(opportunities),
                "market_intelligence": market_analysis,
                "ai_prioritization": prioritization,
                "testing_strategies": strategies,
                "llm_usage_stats": self.llm_client.get_usage_stats()
            }
            
            # Save results
            output_file = f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            
            logger.info(f"Analysis complete. Results saved to {output_file}")
            
            # Display summary
            await self._display_analysis_summary(analysis_results)
            
            return analysis_results
            
        finally:
            await self.llm_client.close()
    
    async def _analyze_market_intelligence(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate market intelligence using AI"""
        
        # Calculate basic statistics
        bounty_programs = [op for op in opportunities if 'min_bounty' in op and op['min_bounty'] > 0]
        avg_bounty = sum(op['min_bounty'] for op in bounty_programs) / len(bounty_programs) if bounty_programs else 0
        max_bounty = max(op['min_bounty'] for op in bounty_programs) if bounty_programs else 0
        
        # Sample bounty data for AI analysis
        sample_data = {
            "total_programs": len(opportunities),
            "programs_with_bounties": len(bounty_programs),
            "average_minimum_bounty": round(avg_bounty, 2),
            "highest_minimum_bounty": max_bounty,
            "sample_bounty_ranges": [op.get('bounty_range', 'N/A') for op in opportunities[:8]]
        }
        
        market_prompt = f"""
Analyze the HackerOne bug bounty market based on this real data:

MARKET DATA:
{json.dumps(sample_data, indent=2)}

As a cybersecurity market analyst, provide insights on:

1. MARKET ASSESSMENT:
   - Overall bounty competitiveness 
   - Market health indicators
   - Researcher opportunity quality

2. STRATEGIC INSIGHTS:
   - ROI potential for security researchers
   - Market trends and patterns
   - Competitive landscape analysis

3. RECOMMENDATIONS:
   - Optimal targeting strategies
   - Time investment guidance
   - Skill development priorities

Provide actionable intelligence for bug bounty research operations.
Format as structured analysis with clear recommendations.
"""
        
        request = LLMRequest(
            task_type=TaskType.VULNERABILITY_ANALYSIS,
            prompt=market_prompt,
            max_tokens=1200,
            temperature=0.3,
            structured_output=False
        )
        
        try:
            response = await self.llm_client.generate_payload(request)
            return {
                "ai_analysis": response.content,
                "model_used": response.model_used,
                "analysis_cost": response.cost_usd,
                "confidence": response.confidence_score,
                "market_data": sample_data
            }
        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            return {"error": str(e), "market_data": sample_data}
    
    async def _generate_smart_prioritization(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate AI-powered opportunity prioritization"""
        
        prioritization_prompt = f"""
As a bug bounty strategist, prioritize these {len(opportunities)} HackerOne opportunities:

OPPORTUNITIES:
{json.dumps(opportunities, indent=2)}

For each program, analyze:
1. ROI potential (time vs reward)
2. Competition level estimation
3. Technical complexity assessment 
4. Success probability

Provide prioritized ranking with justification for targeting decisions.
Include specific recommendations for approach and time investment.
"""
        
        request = LLMRequest(
            task_type=TaskType.EXPLOITATION_STRATEGY,
            prompt=prioritization_prompt,
            max_tokens=1500,
            temperature=0.4,
            structured_output=False
        )
        
        try:
            response = await self.llm_client.generate_payload(request)
            return {
                "ai_prioritization": response.content,
                "model_used": response.model_used,
                "analysis_cost": response.cost_usd,
                "confidence": response.confidence_score,
                "opportunities_analyzed": len(opportunities)
            }
        except Exception as e:
            logger.error(f"Prioritization failed: {e}")
            return {"error": str(e)}
    
    async def _generate_testing_strategies(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate specific testing strategies for top opportunities"""
        
        strategies = []
        
        for i, opportunity in enumerate(opportunities, 1):
            strategy_prompt = f"""
Create a detailed penetration testing strategy for this bug bounty program:

PROGRAM: {opportunity.get('name', 'Unknown')}
BOUNTY: {opportunity.get('bounty_range', 'Not specified')}
URL: {opportunity.get('url', 'Not available')}

As a penetration testing expert, provide:

1. RECONNAISSANCE PLAN:
   - Target analysis approach
   - Technology stack identification
   - Attack surface mapping

2. TESTING METHODOLOGY:
   - Vulnerability categories to focus on
   - Specific testing techniques
   - Tool recommendations

3. EXPLOITATION STRATEGY:
   - Attack vector prioritization
   - Payload development approach
   - Proof-of-concept planning

4. SUCCESS OPTIMIZATION:
   - Time allocation recommendations
   - Risk vs reward assessment
   - Competitive advantages

Provide actionable testing strategy for maximum success probability.
"""
            
            request = LLMRequest(
                task_type=TaskType.EXPLOITATION_STRATEGY,
                prompt=strategy_prompt,
                max_tokens=1000,
                temperature=0.5,
                structured_output=False
            )
            
            try:
                response = await self.llm_client.generate_payload(request)
                strategies.append({
                    "opportunity": opportunity,
                    "testing_strategy": response.content,
                    "model_used": response.model_used,
                    "strategy_cost": response.cost_usd,
                    "confidence": response.confidence_score
                })
            except Exception as e:
                logger.error(f"Strategy generation failed for opportunity {i}: {e}")
                strategies.append({
                    "opportunity": opportunity,
                    "error": str(e)
                })
        
        return strategies
    
    async def _display_analysis_summary(self, results: Dict[str, Any]):
        """Display analysis summary to console"""
        
        print(f"\n=== AI-POWERED HACKERONE ANALYSIS RESULTS ===")
        print(f"Analysis completed: {results['analysis_timestamp']}")
        print(f"Total opportunities analyzed: {results['total_opportunities']}")
        
        # Market intelligence
        market = results.get('market_intelligence', {})
        if 'market_data' in market:
            data = market['market_data']
            print(f"\n📊 MARKET INTELLIGENCE:")
            print(f"Programs with bounty info: {data['programs_with_bounties']}")
            print(f"Average minimum bounty: ${data['average_minimum_bounty']}")
            print(f"Highest minimum bounty: ${data['highest_minimum_bounty']}")
            print(f"AI Analysis Model: {market.get('model_used', 'N/A')}")
            print(f"Analysis Cost: ${market.get('analysis_cost', 0):.4f}")
        
        # Prioritization
        prioritization = results.get('ai_prioritization', {})
        if 'opportunities_analyzed' in prioritization:
            print(f"\n🎯 AI PRIORITIZATION:")
            print(f"Opportunities prioritized: {prioritization['opportunities_analyzed']}")
            print(f"Model used: {prioritization.get('model_used', 'N/A')}")
            print(f"Analysis cost: ${prioritization.get('analysis_cost', 0):.4f}")
        
        # Testing strategies
        strategies = results.get('testing_strategies', [])
        if strategies:
            print(f"\n⚡ TESTING STRATEGIES:")
            print(f"Custom strategies generated: {len(strategies)}")
            for i, strategy in enumerate(strategies[:3], 1):
                if 'opportunity' in strategy:
                    opp = strategy['opportunity']
                    print(f"{i}. {opp.get('name', 'Unknown')} - {opp.get('bounty_range', 'N/A')}")
        
        # LLM usage statistics
        usage = results.get('llm_usage_stats', {})
        if usage:
            print(f"\n💰 LLM USAGE STATISTICS:")
            print(f"Total API requests: {usage.get('total_requests', 0)}")
            print(f"Total cost: ${usage.get('total_cost_usd', 0):.4f}")
            print(f"Average cost per request: ${usage.get('avg_cost_per_request', 0):.4f}")
        
        print(f"\n✅ REAL-WORLD AI ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"🚀 {results['total_opportunities']} HackerOne opportunities processed with LLM intelligence")
        print(f"📈 Market intelligence, prioritization, and testing strategies generated")
        print(f"🎯 Ready for strategic bug bounty operations")

async def main():
    """Main analysis function"""
    analyzer = SimpleHackerOneAnalyzer()
    
    # Find the most recent scraped data file
    import glob
    scraped_files = glob.glob("/root/xorb/hackerone_opportunities_*.json")
    if not scraped_files:
        print("❌ No scraped data files found. Run test_hackerone_scraper.py first.")
        return
    
    latest_file = max(scraped_files, key=os.path.getctime)
    print(f"🔍 Analyzing scraped data from: {os.path.basename(latest_file)}")
    
    # Run comprehensive AI analysis
    results = await analyzer.analyze_scraped_opportunities(latest_file)
    
    return results

if __name__ == "__main__":
    asyncio.run(main())