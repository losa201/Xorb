#!/usr/bin/env python3
"""
XORB Supreme - AI-Augmented Red Team & Bug Bounty Orchestration System
Enhanced Edition with ML-powered decision making and production hardening

Main application entry point with full system integration
"""

import asyncio
import logging
import signal
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Core XORB components
from orchestration.ml_orchestrator import IntelligentOrchestrator
from knowledge_fabric.vector_fabric import VectorKnowledgeFabric
from agents.multi_engine_agents import MultiEngineAgentManager
from llm.hybrid_client import HybridLLMClient, PromptContext, LLMRequest
from integrations.bounty_intelligence import BountyIntelligenceEngine
from security.hardening import XORBSecurityManager, SecurityLevel
from deployment.optimizer import XORBDeploymentOptimizer, DeploymentMode
from orchestration.event_system import EventBus
from monitoring.dashboard import TerminalDashboard


class XORBSystem:
    """Main XORB system coordinator"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # System components
        self.orchestrator: Optional[IntelligentOrchestrator] = None
        self.knowledge_fabric: Optional[VectorKnowledgeFabric] = None
        self.agent_manager: Optional[MultiEngineAgentManager] = None
        self.llm_client: Optional[HybridLLMClient] = None
        self.bounty_intelligence: Optional[BountyIntelligenceEngine] = None
        self.security_manager: Optional[XORBSecurityManager] = None
        self.deployment_optimizer: Optional[XORBDeploymentOptimizer] = None
        self.event_bus: Optional[EventBus] = None
        self.dashboard: Optional[TerminalDashboard] = None
        
        # System state
        self.running = False
        self.startup_complete = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load system configuration"""
        default_config = {
            "redis_url": "redis://localhost:6379/0",
            "database_url": "sqlite+aiosqlite:///./xorb_enhanced.db",
            "openrouter_api_key": "your_api_key_here",
            "security_level": "production",
            "deployment_mode": "production",
            "enable_monitoring": True,
            "enable_ml": True,
            "enable_vector_search": True,
            "enable_bounty_intelligence": True,
            "log_level": "INFO",
            "components": {
                "orchestrator": {"enabled": True, "ml_enabled": True},
                "knowledge_fabric": {"enabled": True, "vector_enabled": True},
                "agent_manager": {"enabled": True, "multi_engine": True},
                "llm_client": {"enabled": True, "hybrid_mode": True},
                "bounty_intelligence": {"enabled": True},
                "security_manager": {"enabled": True},
                "deployment_optimizer": {"enabled": True},
                "event_bus": {"enabled": True},
                "dashboard": {"enabled": True}
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Failed to load config from {config_path}: {e}")
        
        return default_config

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.get("log_level", "INFO"))
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        logger = logging.getLogger("XORB")
        logger.info("XORB Supreme Enhanced Edition initializing...")
        
        return logger

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        if self.running:
            asyncio.create_task(self.shutdown())

    async def initialize(self):
        """Initialize all XORB components"""
        self.logger.info("Initializing XORB system components...")
        
        try:
            # Initialize event bus first (other components may use it)
            if self.config["components"]["event_bus"]["enabled"]:
                self.event_bus = EventBus(self.config["redis_url"])
                await self.event_bus.start()
                self.logger.info("✅ Event bus initialized")

            # Initialize security manager
            if self.config["components"]["security_manager"]["enabled"]:
                security_level = SecurityLevel(self.config["security_level"])
                self.security_manager = XORBSecurityManager(
                    self.config["redis_url"], 
                    security_level
                )
                self.logger.info("✅ Security manager initialized")

            # Initialize deployment optimizer
            if self.config["components"]["deployment_optimizer"]["enabled"]:
                deployment_mode = DeploymentMode(self.config["deployment_mode"])
                self.deployment_optimizer = XORBDeploymentOptimizer(deployment_mode)
                await self.deployment_optimizer.start_optimization()
                self.logger.info("✅ Deployment optimizer initialized")

            # Initialize knowledge fabric
            if self.config["components"]["knowledge_fabric"]["enabled"]:
                if self.config["enable_vector_search"]:
                    self.knowledge_fabric = VectorKnowledgeFabric(
                        self.config["redis_url"],
                        self.config["database_url"]
                    )
                else:
                    from knowledge_fabric.core import KnowledgeFabric
                    self.knowledge_fabric = KnowledgeFabric(
                        self.config["redis_url"],
                        self.config["database_url"]
                    )
                
                await self.knowledge_fabric.initialize()
                self.logger.info("✅ Knowledge fabric initialized")

            # Initialize LLM client
            if self.config["components"]["llm_client"]["enabled"]:
                if self.config["components"]["llm_client"].get("hybrid_mode", True):
                    self.llm_client = HybridLLMClient(
                        self.config["openrouter_api_key"],
                        enable_local_fallback=True
                    )
                else:
                    from llm.client import OpenRouterClient
                    self.llm_client = OpenRouterClient(self.config["openrouter_api_key"])
                
                self.logger.info("✅ LLM client initialized")

            # Initialize agent manager
            if self.config["components"]["agent_manager"]["enabled"]:
                self.agent_manager = MultiEngineAgentManager()
                await self.agent_manager.initialize()
                self.logger.info("✅ Agent manager initialized")

            # Initialize orchestrator
            if self.config["components"]["orchestrator"]["enabled"]:
                self.orchestrator = IntelligentOrchestrator(self.config["redis_url"])
                
                # Set dependencies
                if self.knowledge_fabric:
                    self.orchestrator.knowledge_fabric = self.knowledge_fabric
                if self.agent_manager:
                    self.orchestrator.agent_manager = self.agent_manager
                if self.llm_client:
                    self.orchestrator.llm_client = self.llm_client
                
                await self.orchestrator.start()
                self.logger.info("✅ Orchestrator initialized")

            # Initialize bounty intelligence
            if self.config["components"]["bounty_intelligence"]["enabled"]:
                self.bounty_intelligence = BountyIntelligenceEngine()
                await self.bounty_intelligence.initialize()
                self.logger.info("✅ Bounty intelligence initialized")

            # Initialize dashboard
            if self.config["components"]["dashboard"]["enabled"] and self.config["enable_monitoring"]:
                self.dashboard = TerminalDashboard()
                
                # Connect components to dashboard
                if self.orchestrator:
                    self.dashboard.orchestrator = self.orchestrator
                if self.deployment_optimizer:
                    self.dashboard.deployment_optimizer = self.deployment_optimizer
                if self.security_manager:
                    self.dashboard.security_manager = self.security_manager
                
                await self.dashboard.start()
                self.logger.info("✅ Monitoring dashboard initialized")

            self.startup_complete = True
            self.logger.info("🚀 XORB Supreme Enhanced Edition fully initialized!")
            
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            raise

    async def run(self):
        """Run the main XORB system"""
        if not self.startup_complete:
            await self.initialize()
        
        self.running = True
        self.logger.info("🎯 XORB Supreme is now operational!")
        
        # Print system status
        await self._print_system_status()
        
        # Main operation loop
        try:
            while self.running:
                # Periodic system health checks
                await self._system_health_check()
                
                # Update dashboard if enabled
                if self.dashboard:
                    await self.dashboard.update_display()
                
                await asyncio.sleep(5)  # Main loop interval
                
        except Exception as e:
            self.logger.error(f"Main loop error: {e}")
        
        self.logger.info("XORB system operation ended")

    async def _print_system_status(self):
        """Print comprehensive system status"""
        self.logger.info("=== XORB SYSTEM STATUS ===")
        
        if self.orchestrator:
            stats = await self.orchestrator.get_ml_orchestrator_stats()
            self.logger.info(f"Orchestrator: {len(self.orchestrator.campaigns)} campaigns, ML enabled: {stats['ml_enabled']}")
        
        if self.knowledge_fabric:
            if hasattr(self.knowledge_fabric, 'get_vector_fabric_stats'):
                stats = await self.knowledge_fabric.get_vector_fabric_stats()
                self.logger.info(f"Knowledge Fabric: Vector search enabled: {stats['semantic_search_enabled']}")
            else:
                self.logger.info("Knowledge Fabric: Basic mode (vector search disabled)")
        
        if self.agent_manager:
            status = await self.agent_manager.get_agent_status()
            self.logger.info(f"Agent Manager: {len(status)} agents registered")
        
        if self.security_manager:
            status = await self.security_manager.get_security_status()
            self.logger.info(f"Security: Level {status['security_level']}, {status['active_sessions']} active sessions")
        
        if self.deployment_optimizer:
            status = await self.deployment_optimizer.get_deployment_status()
            self.logger.info(f"Deployment: Mode {status['deployment_mode']}, CPU {status['system_resources']['cpu_usage']:.1f}%, Memory {status['system_resources']['memory_usage']:.1f}%")
        
        self.logger.info("==========================")

    async def _system_health_check(self):
        """Perform periodic system health checks"""
        try:
            # Check critical components
            if self.orchestrator and not hasattr(self.orchestrator, 'running'):
                self.logger.warning("Orchestrator appears to be stopped")
            
            if self.security_manager and self.event_bus:
                # Publish health check event
                await self.event_bus.publish(
                    "system.health.check",
                    "xorb_main",
                    {
                        "status": "healthy",
                        "components_running": self._count_running_components(),
                        "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds() if hasattr(self, '_start_time') else 0
                    }
                )
        
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")

    def _count_running_components(self) -> int:
        """Count running components"""
        count = 0
        components = [
            self.orchestrator, self.knowledge_fabric, self.agent_manager,
            self.llm_client, self.security_manager, self.deployment_optimizer,
            self.event_bus, self.dashboard
        ]
        
        for component in components:
            if component is not None:
                count += 1
        
        return count

    async def shutdown(self):
        """Graceful shutdown of all components"""
        self.logger.info("🛑 Initiating XORB system shutdown...")
        self.running = False
        
        # Shutdown components in reverse order
        shutdown_tasks = []
        
        if self.dashboard:
            shutdown_tasks.append(self.dashboard.stop())
        
        if self.bounty_intelligence:
            shutdown_tasks.append(self.bounty_intelligence.shutdown())
        
        if self.orchestrator:
            shutdown_tasks.append(self.orchestrator.shutdown())
        
        if self.agent_manager:
            shutdown_tasks.append(self.agent_manager.shutdown())
        
        if self.llm_client and hasattr(self.llm_client, 'shutdown'):
            shutdown_tasks.append(self.llm_client.shutdown())
        
        if self.knowledge_fabric:
            shutdown_tasks.append(self.knowledge_fabric.shutdown())
        
        if self.deployment_optimizer:
            shutdown_tasks.append(self.deployment_optimizer.stop_optimization())
        
        if self.security_manager:
            shutdown_tasks.append(self.security_manager.shutdown())
        
        if self.event_bus:
            shutdown_tasks.append(self.event_bus.stop())
        
        # Execute shutdowns concurrently with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*shutdown_tasks, return_exceptions=True),
                timeout=30
            )
        except asyncio.TimeoutError:
            self.logger.warning("Some components took too long to shutdown")
        
        self.logger.info("✅ XORB system shutdown complete")

    async def create_demo_campaign(self) -> Optional[str]:
        """Create a demonstration campaign"""
        if not self.orchestrator:
            self.logger.error("Orchestrator not available for demo campaign")
            return None
        
        demo_targets = [
            {
                "hostname": "demo-app.example.com",
                "ports": [80, 443, 8080],
                "subdomains": ["api.demo-app.example.com", "admin.demo-app.example.com"],
                "technology_stack": ["nginx", "php", "mysql"]
            },
            {
                "hostname": "vulnerable-site.test",
                "ports": [80, 443, 3306, 22],
                "subdomains": ["www.vulnerable-site.test", "staging.vulnerable-site.test"],
                "technology_stack": ["apache", "wordpress", "mysql"]
            }
        ]
        
        try:
            campaign_id = await self.orchestrator.create_intelligent_campaign(
                name="XORB Demo Campaign",
                targets=demo_targets,
                metadata={
                    "demo": True,
                    "created_by": "xorb_main",
                    "description": "Demonstration campaign showcasing XORB capabilities"
                }
            )
            
            self.logger.info(f"✅ Created demo campaign: {campaign_id}")
            return campaign_id
            
        except Exception as e:
            self.logger.error(f"Failed to create demo campaign: {e}")
            return None


async def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="XORB Supreme Enhanced Edition")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--demo", action="store_true", help="Run with demo campaign")
    parser.add_argument("--mode", choices=["development", "staging", "production"], 
                       default="production", help="Deployment mode")
    parser.add_argument("--security", choices=["development", "staging", "production", "high_security"],
                       default="production", help="Security level")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable monitoring dashboard")
    
    args = parser.parse_args()
    
    # Create system instance
    xorb = XORBSystem(args.config)
    
    # Override config from command line args
    xorb.config["deployment_mode"] = args.mode
    xorb.config["security_level"] = args.security
    if args.no_dashboard:
        xorb.config["components"]["dashboard"]["enabled"] = False
    
    try:
        # Initialize and run
        xorb._start_time = datetime.utcnow()
        await xorb.initialize()
        
        # Create demo campaign if requested
        if args.demo:
            await xorb.create_demo_campaign()
        
        # Run main system
        await xorb.run()
        
    except KeyboardInterrupt:
        xorb.logger.info("Received interrupt signal")
    except Exception as e:
        xorb.logger.error(f"Fatal error: {e}")
        raise
    finally:
        await xorb.shutdown()


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                        XORB SUPREME ENHANCED EDITION                      ║
    ║                 AI-Augmented Red Team & Bug Bounty Platform              ║
    ║                                                                          ║
    ║  🎯 ML-Powered Target Prioritization    🧠 Hybrid LLM Architecture      ║
    ║  📡 Event-Driven Architecture          🛡️  Production Security Hardening ║
    ║  🔍 Vector-Enhanced Knowledge Fabric   💰 ROI-Optimized Bounty Engine   ║
    ║  🤖 Multi-Engine Agent System          📊 Real-Time Resource Optimization ║
    ║                                                                          ║
    ║                          Ready for Enterprise Deployment                 ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())