#!/usr/bin/env python3
"""
XORB Unified Core Platform
Consolidated API Gateway + Authentication + Service Mesh
Optimized for AMD EPYC deployment
"""

import asyncio
import logging
import os

from xorb.shared.config import PlatformConfig
from .core import UnifiedCorePlatform

# Main application factory
async def create_app():
    """Create and configure the unified platform application."""
    platform = UnifiedCorePlatform()
    await platform.init_services()
    return platform.app

# Entry point for development
if __name__ == '__main__':
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the application
    async def main():
        app = await create_app()
        web.run_app(app, host='0.0.0.0', port=8000)
    
    asyncio.run(main())