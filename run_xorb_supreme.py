#!/usr/bin/env python3

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def main():
    """Main entry point for XORB Supreme Enhanced Edition."""
    
    print("🚀 Starting XORB Supreme Enhanced Edition...")
    print("=" * 60)
    
    # Load configuration
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        logger.info("Configuration loaded successfully")
    except FileNotFoundError:
        logger.error("config.json not found! Please create a configuration file.")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config.json: {e}")
        return
    
    # Initialize components based on configuration
    components = []
    
    # 1. Initialize Knowledge Fabric with Enhanced Features
    print("\n🧠 Initializing Enhanced Knowledge Fabric...")
    try:
        from knowledge_fabric.core import KnowledgeFabric
        
        # Check if ensemble ML is enabled
        if config.get("enhanced_features", {}).get("ensemble_ml", False):
            try:
                from knowledge_fabric.ensemble_predictor import EnsembleTargetPredictor
                print("  ✅ Ensemble ML Predictor available")
            except ImportError as e:
                print(f"  ⚠️  Ensemble ML not available: {e}")
        
        knowledge_fabric = KnowledgeFabric()
        await knowledge_fabric.initialize()
        components.append(("Knowledge Fabric", knowledge_fabric))
        print("  ✅ Knowledge Fabric initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize Knowledge Fabric: {e}")
        return
    
    # 2. Initialize Threat Intelligence Streaming
    if config.get("enhanced_features", {}).get("threat_intelligence", False):
        print("\n🔍 Initializing Threat Intelligence Streaming...")
        try:
            from integrations.threat_intel_streamer import ThreatIntelStreamer
            
            threat_streamer = ThreatIntelStreamer(knowledge_fabric)
            await threat_streamer.initialize()
            components.append(("Threat Intelligence", threat_streamer))
            print("  ✅ Threat Intelligence Streamer initialized")
            
            # Start streaming in background
            asyncio.create_task(threat_streamer.start_streaming(
                update_interval_minutes=config.get("threat_intelligence", {}).get("update_interval_minutes", 60)
            ))
            print("  🔄 Background streaming started")
            
        except Exception as e:
            logger.error(f"Failed to initialize Threat Intelligence: {e}")
    
    # 3. Initialize Stealth Agents
    if config.get("enhanced_features", {}).get("stealth_agents", False):
        print("\n🥷 Initializing Stealth Agents...")
        try:
            from agents.stealth_agents import StealthPlaywrightAgent, StealthConfig
            
            stealth_config = StealthConfig(
                user_agent_rotation=config.get("stealth_config", {}).get("user_agent_rotation", True),
                proxy_rotation=config.get("stealth_config", {}).get("proxy_rotation", False),
                request_delay_min=config.get("stealth_config", {}).get("request_delay_min", 1.0),
                request_delay_max=config.get("stealth_config", {}).get("request_delay_max", 3.0),
                fingerprint_randomization=config.get("stealth_config", {}).get("fingerprint_randomization", True)
            )
            
            # Create a stealth agent for demonstration
            stealth_agent = StealthPlaywrightAgent("stealth_demo", stealth_config)
            await stealth_agent.initialize()
            components.append(("Stealth Agent", stealth_agent))
            print("  ✅ Stealth Agents initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Stealth Agents: {e}")
            print(f"  ⚠️  Stealth agents not available: {e}")
    
    # 4. Initialize Market Intelligence
    if config.get("enhanced_features", {}).get("market_intelligence", False):
        print("\n💰 Initializing Market Intelligence...")
        try:
            from integrations.market_intelligence import MarketIntelligenceEngine
            from integrations.hackerone_client import HackerOneClient
            
            # Create a mock HackerOne client (would need real API key for full functionality)
            h1_client = HackerOneClient()
            market_engine = MarketIntelligenceEngine(h1_client)
            await market_engine.initialize()
            components.append(("Market Intelligence", market_engine))
            print("  ✅ Market Intelligence Engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Market Intelligence: {e}")
    
    # 5. Initialize ML Orchestrator
    print("\n🤖 Initializing ML-Enhanced Orchestrator...")
    try:
        from orchestration.ml_orchestrator import IntelligentOrchestrator
        
        orchestrator = IntelligentOrchestrator()
        await orchestrator.start()
        components.append(("ML Orchestrator", orchestrator))
        print("  ✅ ML-Enhanced Orchestrator initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize ML Orchestrator: {e}")
    
    # 6. Start Enhanced Dashboard
    print("\n📊 Starting Enhanced Dashboard...")
    try:
        from monitoring.dashboard import TerminalDashboard
        
        dashboard = TerminalDashboard()
        # Don't await this - let it run in background
        asyncio.create_task(dashboard.start_monitoring())
        print("  ✅ Enhanced Dashboard started")
        
    except Exception as e:
        logger.error(f"Failed to start Dashboard: {e}")
    
    # Display system status
    print("\n" + "=" * 60)
    print("🎉 XORB Supreme Enhanced Edition - READY!")
    print("=" * 60)
    print(f"✅ {len(components)} components initialized successfully")
    print("\n📋 Active Components:")
    
    for name, component in components:
        status = "🟢 Running" if hasattr(component, 'running') and getattr(component, 'running', True) else "🟢 Ready"
        print(f"  • {name}: {status}")
    
    print("\n🌟 Enhanced Features Active:")
    enhanced_features = config.get("enhanced_features", {})
    for feature, enabled in enhanced_features.items():
        status = "🟢 Enabled" if enabled else "⚪ Disabled"
        feature_name = feature.replace("_", " ").title()
        print(f"  • {feature_name}: {status}")
    
    print("\n🚀 System Capabilities:")
    print("  • AI-Powered Target Prediction with Ensemble ML")
    print("  • Real-Time Threat Intelligence Integration")
    print("  • Advanced Stealth Browser Automation")
    print("  • Market Intelligence and ROI Optimization")
    print("  • Comprehensive Security Monitoring")
    
    print("\n💡 Next Steps:")
    print("  1. Configure API keys in config.json for full functionality")
    print("  2. Start monitoring stack: docker-compose -f docker-compose.monitoring.yml up -d")
    print("  3. Access Grafana dashboard: http://localhost:3000")
    print("  4. Create your first enhanced campaign")
    
    print("\n🔧 Demo Commands:")
    print("  • Test ensemble ML: python -c \"from knowledge_fabric.ensemble_predictor import *\"")
    print("  • Check threat intel: python -c \"from integrations.threat_intel_streamer import *\"")
    print("  • Test stealth agents: python -c \"from agents.stealth_agents import *\"")
    
    # Keep the system running
    print("\n⏳ System running... Press Ctrl+C to stop")
    try:
        while True:
            await asyncio.sleep(30)
            # Periodic health check
            logger.info(f"XORB Supreme running with {len(components)} active components")
    except KeyboardInterrupt:
        print("\n🛑 Shutting down XORB Supreme...")
        
        # Cleanup components
        for name, component in components:
            try:
                if hasattr(component, 'close'):
                    await component.close()
                elif hasattr(component, 'cleanup'):
                    await component.cleanup()
                print(f"  ✅ {name} shut down cleanly")
            except Exception as e:
                logger.error(f"Error shutting down {name}: {e}")
        
        print("👋 XORB Supreme Enhanced Edition stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)