#!/usr/bin/env python3
"""
XORB Unified Intelligence Engine
Consolidated Orchestration + ML + LLM + Agent Management
Optimized for AMD EPYC deployment
"""

import asyncio
import logging
import os

from xorb.shared.epyc_config import EPYCConfig
from .core import UnifiedIntelligenceEngine

# Application factory
async def create_app():
    """Create and configure the intelligence engine application."""
    engine = UnifiedIntelligenceEngine()
    await engine.initialize()
    return engine.app

if __name__ == '__main__':
    
    # EPYC optimization environment variables
    os.environ['OMP_NUM_THREADS'] = str(EPYCConfig.OMP_NUM_THREADS)
    os.environ['MKL_NUM_THREADS'] = str(EPYCConfig.MKL_NUM_THREADS)
    os.environ['TORCH_NUM_THREADS'] = str(EPYCConfig.TORCH_NUM_THREADS)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def main():
        app = await create_app()
        web.run_app(app, host='0.0.0.0', port=8001)
    
    asyncio.run(main())