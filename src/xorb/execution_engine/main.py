#!/usr/bin/env python3
"""
XORB Unified Execution Engine
Consolidated Scanning + Exploitation + Stealth + Evidence Collection
Optimized for AMD EPYC deployment
"""

import asyncio
import logging
import os

from xorb.shared.epyc_execution_config import EPYCExecutionConfig
from .core import UnifiedExecutionEngine

# Application factory
async def create_app():
    """Create and configure the execution engine application."""
    engine = UnifiedExecutionEngine()
    await engine.initialize()
    return engine.app

if __name__ == '__main__':

    # EPYC optimization
    os.environ['OMP_NUM_THREADS'] = str(EPYCExecutionConfig.CPU_CORES)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    async def main():
        app = await create_app()
        web.run_app(app, host='0.0.0.0', port=8002)

    asyncio.run(main())
