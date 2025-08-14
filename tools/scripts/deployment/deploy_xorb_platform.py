#!/usr/bin/env python3
"""
Full XORB Platform Deployment Script
"""

import asyncio
import logging
import sys
import json
from pathlib import Path
from xorb_resilience_unified_deployment import XORBUnifiedDeployment, DeploymentConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('deployment.log')
    ]
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print deployment banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║          🚀 XORB RESILIENCE & SCALABILITY PLATFORM          ║
║                     DEPLOYMENT SYSTEM                       ║
║                        Version 2.0.0                        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

async def deploy_xorb_platform():
    """Deploy the complete XORB platform"""
    try:
        print_banner()
        logger.info("🎯 Initializing XORB Platform Deployment...")

        # Create deployment configuration
        config = DeploymentConfig(
            deployment_id="xorb-production-deployment",
            platform_version="2.0.0",
            environment="production",
            namespace="xorb_platform",
            enable_monitoring=True,
            enable_security=True,
            enable_backup=True,
            enable_scaling=True
        )

        print(f"📋 Deployment Configuration:")
        print(f"   ID: {config.deployment_id}")
        print(f"   Version: {config.platform_version}")
        print(f"   Environment: {config.environment}")
        print(f"   Namespace: {config.namespace}")
        print()

        # Initialize deployment system
        deployment = XORBUnifiedDeployment(config)

        print("🏗️  Starting full platform deployment...")
        print("   This will deploy all XORB resilience components")
        print("   Estimated time: 10-15 minutes")
        print()

        # Execute full deployment
        logger.info("Starting complete deployment execution...")
        success = await deployment.execute_deployment()

        if success:
            print("\n" + "="*60)
            print("🎉 XORB PLATFORM DEPLOYMENT SUCCESSFUL!")
            print("="*60)

            # Get deployment status
            status = deployment.get_deployment_status()
            print(f"📊 Deployment Statistics:")
            print(f"   Total Steps: {len(deployment.deployment_steps)}")
            print(f"   Completed: {len([s for s in deployment.deployment_steps.values() if s.status.value == 'completed'])}")
            print(f"   Failed: {len([s for s in deployment.deployment_steps.values() if s.status.value == 'failed'])}")
            print(f"   Services Deployed: {len(deployment.deployed_services)}")

            print(f"\n🌟 Deployed Services:")
            for service_name, service_info in deployment.deployed_services.items():
                status_emoji = "✅" if service_info.get('status') == 'running' else "⚠️"
                port = service_info.get('port', 'N/A')
                print(f"   {status_emoji} {service_name:20} → Port {port}")

            print(f"\n🔗 Access Points:")
            print(f"   📊 Grafana Dashboard:  http://localhost:3000")
            print(f"   📈 Prometheus Metrics: http://localhost:9090")
            print(f"   🧠 Neural Orchestrator: http://localhost:8003")
            print(f"   🎓 Learning Service:    http://localhost:8004")
            print(f"   🛡️  Threat Detection:   http://localhost:8005")

            print(f"\n📂 Deployment Artifacts:")
            print(f"   📋 Deployment Root: {deployment.deployment_root}")
            print(f"   📝 Configuration:   {deployment.config_dir}")
            print(f"   📜 Scripts:         {deployment.scripts_dir}")
            print(f"   📄 Logs:           {deployment.logs_dir}")

            print(f"\n✅ XORB Platform is now ready for operation!")
            return True

        else:
            print("\n" + "="*60)
            print("❌ XORB PLATFORM DEPLOYMENT FAILED!")
            print("="*60)

            # Show failed steps
            failed_steps = [s for s in deployment.deployment_steps.values() if s.status.value == 'failed']
            if failed_steps:
                print(f"\n🔍 Failed Steps ({len(failed_steps)}):")
                for step in failed_steps:
                    print(f"   ❌ {step.name}")
                    if step.error_message:
                        print(f"      Error: {step.error_message}")

            print(f"\n📋 Check deployment logs for detailed error information:")
            print(f"   📄 Main Log: deployment.log")
            print(f"   📄 System Log: {deployment.logs_dir}/deployment_{config.deployment_id}.log")

            return False

    except Exception as e:
        logger.error(f"Deployment execution failed: {e}")
        print(f"\n💥 Deployment Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main execution function"""
    try:
        success = await deploy_xorb_platform()

        if success:
            print(f"\n🎊 Deployment completed successfully!")
            print(f"   Run health checks: python3 -c \"import requests; print('Health:', requests.get('http://localhost:8003/health').status_code)\"")
            print(f"   Run tests: python3 xorb_resilience_testing_suite.py")
        else:
            print(f"\n😞 Deployment failed. Check logs for details.")

        return success

    except KeyboardInterrupt:
        print(f"\n🛑 Deployment interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
