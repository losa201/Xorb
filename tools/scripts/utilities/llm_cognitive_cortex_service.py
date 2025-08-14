#!/usr/bin/env python3
"""
XORB LLM Cognitive Cortex Service

Standalone service for running the LLM cognitive cortex with multi-model routing,
adaptive learning, and comprehensive telemetry.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# Add xorb_core to Python path
sys.path.insert(0, '/root/Xorb')
sys.path.insert(0, '/root/Xorb/xorb_core')

from xorb_core.llm.llm_api_service import app
import uvicorn


def setup_logging():
    """Setup comprehensive logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/llm_cortex_service.log')
        ]
    )

    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)

    logger = logging.getLogger("xorb.llm.service")
    logger.info("XORB LLM Cognitive Cortex Service starting...")
    return logger


def validate_environment():
    """Validate required environment variables"""
    required_vars = [
        'OPENROUTER_API_KEY',
        'NVIDIA_API_KEY'
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"WARNING: Missing environment variables: {missing_vars}")
        print("Some LLM providers may not be available.")

    # Set default audit key if not provided
    if not os.getenv('XORB_AUDIT_KEY'):
        os.environ['XORB_AUDIT_KEY'] = 'xorb-default-key-2025'
        print("Using default audit encryption key. Set XORB_AUDIT_KEY for production.")


async def health_check_loop():
    """Background health check loop"""
    import aiohttp

    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8009/llm/health') as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"Health check OK: {data['status']} - Models: {data['models_available']}")
                    else:
                        print(f"Health check failed: {response.status}")
        except Exception as e:
            print(f"Health check error: {e}")

        await asyncio.sleep(60)  # Check every minute


def main():
    """Main service entry point"""
    print("🧠 XORB LLM Cognitive Cortex Service")
    print("=" * 50)
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup logging
    logger = setup_logging()

    # Validate environment
    validate_environment()

    # Start background health checks
    # asyncio.create_task(health_check_loop())

    # Configuration
    config = {
        "host": "0.0.0.0",
        "port": 8009,
        "reload": False,  # Disable reload in production
        "log_level": "info",
        "access_log": True,
        "workers": 1  # Single worker for now to maintain state
    }

    logger.info(f"Starting LLM Cognitive Cortex service on {config['host']}:{config['port']}")

    try:
        # Run the service
        uvicorn.run(app, **config)
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
