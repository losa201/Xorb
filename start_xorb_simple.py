#!/usr/bin/env python3
"""
Simple XORB startup script for testing core functionality
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Simple XORB startup with basic components"""
    
    logger.info("🚀 Starting XORB Simple Edition...")
    
    try:
        # Test Redis connection
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        logger.info("✅ Redis connection successful")
        
        # Test HackerOne scraper
        from test_hackerone_scraper import HackerOneOpportunitiesScraper
        scraper = HackerOneOpportunitiesScraper()
        logger.info("✅ HackerOne scraper initialized")
        
        # Test HackerOne client (without API key for now)
        try:
            from integrations.hackerone_client import HackerOneClient
            client = HackerOneClient(api_key="test")
            logger.info("✅ HackerOne client initialized")
        except Exception as e:
            logger.warning(f"HackerOne client issue (expected without API key): {e}")
        
        # Create a simple workflow
        logger.info("🎯 Running basic XORB workflow...")
        
        # 1. Scrape opportunities
        logger.info("Step 1: Scraping HackerOne opportunities...")
        opportunities = await scraper.scrape_opportunities()
        if opportunities:
            logger.info(f"Found {len(opportunities)} opportunities")
            
            # Display first few
            for i, opp in enumerate(opportunities[:3], 1):
                name = opp.get('name', 'Unknown')
                bounty = opp.get('bounty_range', 'No bounty info')
                logger.info(f"  {i}. {name} - {bounty}")
        else:
            logger.info("No opportunities found, but scraper works")
        
        # 2. Demo target analysis (simulated)
        logger.info("Step 2: Analyzing targets...")
        demo_targets = [
            "demo-app.example.com",
            "vulnerable-site.test"
        ]
        
        for target in demo_targets:
            logger.info(f"  📍 Analyzing target: {target}")
            # In a real scenario, this would use the agent system
            await asyncio.sleep(0.5)  # Simulate work
        
        # 3. Generate summary report
        logger.info("Step 3: Generating report...")
        report_data = {
            "timestamp": asyncio.get_event_loop().time(),
            "opportunities_found": len(opportunities) if opportunities else 0,
            "targets_analyzed": len(demo_targets),
            "status": "success"
        }
        logger.info(f"Report: {report_data}")
        
        logger.info("🎉 XORB Simple workflow completed successfully!")
        logger.info("\n=== NEXT STEPS ===")
        logger.info("1. Get HackerOne API key for full functionality")
        logger.info("2. Configure targets for actual scanning")
        logger.info("3. Set up proper security testing scope")
        logger.info("4. Start full XORB system with: python main.py --demo")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())