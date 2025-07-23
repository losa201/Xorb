
#!/usr/bin/env python3

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import necessary XORB components
from integrations.bounty_intelligence import BountyIntelligenceEngine
from integrations.hackerone_client import HackerOneClient
from orchestration.ml_orchestrator import IntelligentOrchestrator, CampaignPriority

class HackerOneAutomator:
    """
    Automates the process of finding promising HackerOne programs,
    fetching their in-scope assets, and launching testing campaigns.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.h1_client = HackerOneClient(
            api_key=self.config.get("hackerone_api_key", ""),
            username=self.config.get("hackerone_username", "")
        )
        self.bounty_intel = BountyIntelligenceEngine(self.h1_client)
        self.orchestrator = IntelligentOrchestrator()

    async def initialize(self):
        """Initialize all necessary components."""
        logger.info("Initializing HackerOne Automator components...")
        await self.h1_client.start()
        await self.orchestrator.start()
        logger.info("Initialization complete.")

    async def shutdown(self):
        """Gracefully shut down all components."""
        logger.info("Shutting down HackerOne Automator components...")
        await self.h1_client.close()
        await self.orchestrator.shutdown()
        logger.info("Shutdown complete.")

    async def run_automation_workflow(self, num_programs: int = 3):
        """
        Executes the full automation workflow.
        1. Finds top bounty programs.
        2. Fetches their in-scope assets.
        3. Launches testing campaigns.
        """
        logger.info("Starting HackerOne automation workflow...")

        # 1. Find top bounty programs
        logger.info(f"Identifying top {num_programs} bug bounty programs...")
        try:
            all_programs = await self.h1_client.get_programs(eligible_only=True)
            program_handles = [p.handle for p in all_programs]
            
            prioritized_programs = await self.bounty_intel.prioritize_programs(
                program_handles, max_programs=num_programs
            )
            
            if not prioritized_programs:
                logger.warning("No promising programs found. Exiting.")
                return

            logger.info(f"Selected top {len(prioritized_programs)} programs:")
            for handle, analysis in prioritized_programs:
                logger.info(f"  - {handle} (ROI Score: {analysis.roi_score:.2f})")

        except Exception as e:
            logger.error(f"Failed to identify top programs: {e}")
            return

        # 2. Fetch scopes and launch campaigns
        for program_handle, analysis in prioritized_programs:
            try:
                logger.info(f"Processing program: {program_handle}")

                # Fetch in-scope assets
                scopes = await self.h1_client.get_program_scopes(program_handle)
                in_scope_assets = [
                    s for s in scopes if s.get("eligible_for_submission")
                ]

                if not in_scope_assets:
                    logger.warning(f"No in-scope assets found for {program_handle}. Skipping.")
                    continue

                logger.info(f"Found {len(in_scope_assets)} in-scope assets for {program_handle}.")

                # Prepare targets for the campaign
                targets = []
                for asset in in_scope_assets:
                    if asset.get("asset_type") in ["URL", "DOMAIN"]:
                        targets.append({
                            "hostname": asset.get("asset_identifier"),
                            "ports": [80, 443]  # Default ports for web assets
                        })

                if not targets:
                    logger.warning(f"No actionable web assets found for {program_handle}. Skipping.")
                    continue

                # 3. Launch campaign
                campaign_name = f"H1 Auto-Scan: {program_handle}"
                logger.info(f"Launching campaign: {campaign_name}")

                campaign_id = await self.orchestrator.create_intelligent_campaign(
                    name=campaign_name,
                    targets=targets,
                    priority=CampaignPriority.HIGH,
                    metadata={
                        "source": "hackerone_automator",
                        "program_handle": program_handle,
                        "roi_score": analysis.roi_score,
                        "asset_count": len(targets)
                    }
                )

                await self.orchestrator.start_campaign(campaign_id)
                logger.info(f"Successfully launched campaign {campaign_id} for {program_handle}.")

            except Exception as e:
                logger.error(f"Failed to process program {program_handle}: {e}")

        logger.info("HackerOne automation workflow finished.")


async def main():
    """Main entry point for the script."""
    # Load configuration
    config_path = Path("config.json")
    if not config_path.exists():
        logger.error("config.json not found! Please create it from the example.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    # Check for HackerOne API key
    if not config.get("hackerone_api_key") or not config.get("hackerone_username"):
        logger.error(
            "HackerOne API key and username are not set in config.json. "
            "Please add them to run this script."
        )
        return

    automator = HackerOneAutomator(config)
    try:
        await automator.initialize()
        await automator.run_automation_workflow()
    finally:
        await automator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
