from dataclasses import dataclass

from typing import Dict, List, Any, Optional

#!/usr/bin/env python3
"""
Advanced XORB Integration Demonstration

This script demonstrates the integration of all advanced features:
- Agent discovery and registration
- Dynamic resource allocation
- Advanced metrics and monitoring
- Comprehensive logging and audit trails
- Security hardening features
"""

import asyncio
import json
import time
from pathlib import Path
import sys

# Add the xorb_core to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from xorb_core.agents.advanced_discovery import (
    AdvancedAgentRegistry, 
    initialize_agent_discovery,
    AgentCapability,
    AgentMetadata
)
from xorb_core.orchestration.dynamic_resource_manager import (
    DynamicResourceManager,
    LocalResourceProvider,
    create_development_policy,
    ResourceQuota
)
from xorb_core.monitoring.advanced_metrics import (
    AdvancedMetricsManager,
    initialize_metrics
)
from xorb_core.logging.audit_system import (
    AdvancedLoggingSystem,
    initialize_logging,
    AuditEventType,
    LogLevel,
    ComplianceFramework
)

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class XORBIntegrationDemo:
    """Comprehensive integration demonstration."""
    
    def __init__(self) -> None:
        self.agent_registry = None
        self.resource_manager = None
        self.metrics_manager = None
        self.logging_system = None
        
    async def initialize_systems(self) -> None:
        """Initialize all XORB advanced systems."""
        print("🚀 Initializing XORB Advanced Systems...")
        
        # Initialize agent discovery
        print("📡 Starting agent discovery system...")
        from xorb_core.agents.advanced_discovery import agent_registry
        self.agent_registry = agent_registry
        
        # Initialize resource management
        print("⚙️ Starting dynamic resource management...")
        resource_provider = LocalResourceProvider(
            ResourceQuota(cpu_cores=8, memory_gb=16, disk_gb=100, max_agents=10)
        )
        scaling_policy = create_development_policy()
        self.resource_manager = DynamicResourceManager(resource_provider, scaling_policy)
        
        # Initialize metrics
        print("📊 Starting advanced metrics collection...")
        from xorb_core.monitoring.advanced_metrics import metrics_manager
        self.metrics_manager = metrics_manager
        
        # Initialize logging
        print("📝 Starting comprehensive logging system...")
        from xorb_core.logging.audit_system import logging_system
        self.logging_system = logging_system
        
        print("✅ All systems initialized successfully!")
        
    async def demo_agent_discovery(self) -> None:
        """Demonstrate agent discovery capabilities."""
        print("\n🔍 AGENT DISCOVERY DEMONSTRATION")
        print("=" * 50)
        
        # Trigger agent discovery
        await self.agent_registry.discover_agents()
        
        # Get statistics
        stats = self.agent_registry.get_agent_statistics()
        print(f"📊 Discovery Statistics:")
        print(f"   • Total agents: {stats['total_agents']}")
        print(f"   • Active agents: {stats['active_agents']}")
        print(f"   • Capabilities: {len(stats['capabilities'])}")
        
        # Display agents by capability
        for capability, info in stats['capabilities'].items():
            if info['active'] > 0:
                print(f"   • {capability}: {info['active']} active agents")
        
        # Log discovery event
        await self.logging_system.audit(
            AuditEventType.SYSTEM_START,
            "agent_discovery_completed",
            details=stats
        )
        
    async def demo_resource_management(self) -> None:
        """Demonstrate dynamic resource management."""
        print("\n⚙️ RESOURCE MANAGEMENT DEMONSTRATION")
        print("=" * 50)
        
        # Start resource management (background task)
        management_task = asyncio.create_task(
            self.resource_manager.start_resource_management()
        )
        
        # Wait a bit for initial metrics collection
        await asyncio.sleep(5)
        
        # Simulate campaign resource allocation
        print("📋 Simulating campaign resource allocation...")
        
        campaign_requirements = {
            'max_agents': 3,
            'complexity_multiplier': 1.5,
            'estimated_duration': 1800  # 30 minutes
        }
        
        quota = await self.resource_manager.allocate_campaign_resources(
            "demo_campaign_001", 
            campaign_requirements
        )
        
        if quota:
            print(f"✅ Resources allocated:")
            print(f"   • CPU cores: {quota.cpu_cores}")
            print(f"   • Memory: {quota.memory_gb} GB")
            print(f"   • Max agents: {quota.max_agents}")
            
            # Record metrics
            self.metrics_manager.record_business_metric(
                "xorb_campaign_resource_allocation",
                1.0,
                {"campaign_id": "demo_campaign_001", "status": "success"}
            )
            
            # Audit the allocation
            await self.logging_system.audit(
                AuditEventType.CAMPAIGN_START,
                "resource_allocation",
                campaign_id="demo_campaign_001",
                details={"quota": quota.__dict__, "requirements": campaign_requirements},
                compliance_frameworks=[ComplianceFramework.SOC2]
            )
        else:
            print("❌ Resource allocation failed")
        
        # Get resource statistics
        resource_stats = self.resource_manager.get_resource_statistics()
        print(f"\n📊 Resource Statistics:")
        print(f"   • Current instances: {resource_stats['current_instances']}")
        print(f"   • Allocated campaigns: {resource_stats['allocated_campaigns']}")
        
        if 'current_usage' in resource_stats:
            usage = resource_stats['current_usage']
            print(f"   • CPU usage: {usage['cpu_percent']:.1f}%")
            print(f"   • Memory usage: {usage['memory_percent']:.1f}%")
        
        # Stop resource management
        await self.resource_manager.stop_resource_management()
        management_task.cancel()
        
        try:
            await management_task
        except asyncio.CancelledError:
            pass
        
    async def demo_advanced_metrics(self) -> None:
        """Demonstrate advanced metrics collection."""
        print("\n📊 ADVANCED METRICS DEMONSTRATION")
        print("=" * 50)
        
        # Start metrics collection (background task)
        metrics_task = asyncio.create_task(
            self.metrics_manager.start_metrics_collection()
        )
        
        # Wait for some metrics collection
        await asyncio.sleep(3)
        
        # Record some business metrics
        print("📈 Recording business metrics...")
        
        business_metrics = [
            ("xorb_agent_health_score", 95.0, {"agent_name": "discovery", "capability": "scanning"}),
            ("xorb_campaign_success_rate", 87.5, {"campaign_type": "vulnerability_scan"}),
            ("xorb_vulnerability_detection_rate", 1.0, {"severity": "high", "category": "web"}),
            ("xorb_knowledge_graph_nodes", 1250, {"node_type": "vulnerability"}),
            ("xorb_compliance_score", 92.0, {"framework": "soc2", "category": "access_control"})
        ]
        
        for metric_name, value, labels in business_metrics:
            self.metrics_manager.record_business_metric(metric_name, value, labels)
            print(f"   • {metric_name}: {value} {labels}")
        
        # Get metrics summary
        metrics_summary = self.metrics_manager.get_metrics_summary()
        print(f"\n📊 Metrics Summary:")
        print(f"   • Collectors: {metrics_summary['collectors']}")
        print(f"   • Exporters: {metrics_summary['exporters']}")
        print(f"   • Active alerts: {metrics_summary['active_alerts']}")
        print(f"   • Metric definitions: {metrics_summary['metric_definitions']}")
        
        # Get Prometheus metrics
        prometheus_metrics = self.metrics_manager.get_prometheus_metrics()
        if prometheus_metrics:
            print(f"   • Prometheus metrics available: {len(prometheus_metrics.split('\\n'))} lines")
        
        # Log metrics collection
        await self.logging_system.audit(
            AuditEventType.SYSTEM_START,
            "metrics_collection_demo",
            details=metrics_summary
        )
        
        # Stop metrics collection
        await self.metrics_manager.stop_metrics_collection()
        metrics_task.cancel()
        
        try:
            await metrics_task
        except asyncio.CancelledError:
            pass
        
    async def demo_logging_and_audit(self) -> None:
        """Demonstrate comprehensive logging and audit trails."""
        print("\n📝 LOGGING & AUDIT DEMONSTRATION")
        print("=" * 50)
        
        # Start logging system (background task)
        logging_task = asyncio.create_task(
            self.logging_system.start_logging_system()
        )
        
        # Wait for system to start
        await asyncio.sleep(2)
        
        # Demonstrate different log levels
        print("📋 Generating log entries...")
        
        await self.logging_system.log(LogLevel.INFO, "System initialization completed")
        await self.logging_system.log(LogLevel.DEBUG, "Debug information", module="demo", extra={"debug_level": 3})
        await self.logging_system.log(LogLevel.WARNING, "Resource threshold exceeded", threshold=80, current=85)
        await self.logging_system.log(LogLevel.ERROR, "Connection timeout", error_code="CONN_TIMEOUT")
        
        # Demonstrate audit events
        print("🔍 Generating audit events...")
        
        audit_events = [
            (AuditEventType.AUTHENTICATION, "user_login", {
                "user_id": "admin_001",
                "source_ip": "192.168.1.100",
                "user_agent": "XORB-Client/2.0",
                "outcome": "success"
            }),
            (AuditEventType.CAMPAIGN_START, "vulnerability_scan_initiated", {
                "campaign_id": "vscan_20250726_001",
                "target_id": "web_app_prod",
                "user_id": "analyst_001",
                "compliance_frameworks": [ComplianceFramework.SOC2, ComplianceFramework.ISO27001]
            }),
            (AuditEventType.DATA_ACCESS, "sensitive_data_accessed", {
                "resource": "customer_database",
                "user_id": "dba_001",
                "sensitive_data": True,
                "compliance_frameworks": [ComplianceFramework.GDPR]
            }),
            (AuditEventType.SECURITY_VIOLATION, "failed_authentication_attempt", {
                "user_id": "unknown",
                "source_ip": "203.0.113.1",
                "attempt_count": 5,
                "severity": LogLevel.CRITICAL
            })
        ]
        
        for event_type, action, kwargs in audit_events:
            await self.logging_system.audit(event_type, action, **kwargs)
            print(f"   • {event_type.value}: {action}")
        
        # Wait for processing
        await asyncio.sleep(3)
        
        # Get logging statistics
        logging_stats = self.logging_system.get_system_statistics()
        print(f"\n📊 Logging Statistics:")
        print(f"   • Storage backends: {logging_stats['storage_backends']}")
        print(f"   • Logs processed: {logging_stats['logs_processed']}")
        print(f"   • Audit events processed: {logging_stats['audit_events_processed']}")
        print(f"   • Storage errors: {logging_stats['storage_errors']}")
        
        # Generate compliance report
        print(f"\n🔒 Compliance Information:")
        print(f"   • SOC2 compliance tracking enabled")
        print(f"   • GDPR compliance tracking enabled")
        print(f"   • Audit trail integrity verification active")
        print(f"   • Encrypted storage for sensitive data")
        
        # Stop logging system
        await self.logging_system.stop_logging_system()
        logging_task.cancel()
        
        try:
            await logging_task
        except asyncio.CancelledError:
            pass
        
    async def demo_integration_scenarios(self) -> None:
        """Demonstrate integrated scenarios."""
        print("\n🔄 INTEGRATION SCENARIOS")
        print("=" * 50)
        
        print("📋 Scenario 1: High-load campaign with auto-scaling")
        
        # Simulate high load triggering scaling
        self.metrics_manager.record_business_metric("xorb_cpu_utilization", 92.0)
        self.metrics_manager.record_business_metric("xorb_memory_utilization", 88.0)
        
        # Log the scaling event
        await self.logging_system.audit(
            AuditEventType.CONFIGURATION_CHANGE,
            "auto_scaling_triggered",
            details={"trigger": "high_resource_utilization", "direction": "scale_up"},
            compliance_frameworks=[ComplianceFramework.SOC2]
        )
        
        print("   • High resource utilization detected")
        print("   • Auto-scaling triggered")
        print("   • Event logged for compliance")
        
        print("\n📋 Scenario 2: Security violation with immediate response")
        
        # Simulate security violation
        await self.logging_system.audit(
            AuditEventType.SECURITY_VIOLATION,
            "unauthorized_access_attempt",
            severity=LogLevel.CRITICAL,
            user_id="suspicious_user",
            source_ip="198.51.100.1",
            details={"blocked": True, "alert_sent": True},
            compliance_frameworks=[ComplianceFramework.SOC2, ComplianceFramework.ISO27001]
        )
        
        # Record security metrics
        self.metrics_manager.record_business_metric(
            "xorb_security_violations_total", 
            1.0, 
            {"severity": "critical", "blocked": "true"}
        )
        
        print("   • Security violation detected")
        print("   • Immediate blocking activated")
        print("   • Compliance audit trail created")
        print("   • Security metrics updated")
        
        print("\n📋 Scenario 3: Agent failure with automatic recovery")
        
        # Simulate agent failure
        await self.logging_system.audit(
            AuditEventType.AGENT_EXECUTION,
            "agent_failure_recovery",
            agent_id="scanner_001",
            outcome="failure",
            details={"error": "connection_timeout", "recovery_action": "restart", "recovered": True}
        )
        
        # Update agent health metrics
        self.metrics_manager.record_business_metric(
            "xorb_agent_health_score", 
            75.0, 
            {"agent_name": "scanner_001", "status": "recovering"}
        )
        
        print("   • Agent failure detected")
        print("   • Automatic recovery initiated")
        print("   • Health metrics updated")
        print("   • Recovery audit logged")
        
    async def generate_summary_report(self) -> None:
        """Generate comprehensive summary report."""
        print("\n📊 COMPREHENSIVE SUMMARY REPORT")
        print("=" * 50)
        
        # Agent statistics
        if self.agent_registry:
            agent_stats = self.agent_registry.get_agent_statistics()
            print(f"🤖 Agent System:")
            print(f"   • Total agents discovered: {agent_stats['total_agents']}")
            print(f"   • Active agents: {agent_stats['active_agents']}")
            print(f"   • Capabilities available: {len(agent_stats['capabilities'])}")
        
        # Resource statistics
        if self.resource_manager:
            resource_stats = self.resource_manager.get_resource_statistics()
            print(f"\n⚙️ Resource Management:")
            print(f"   • Current instances: {resource_stats['current_instances']}")
            print(f"   • Predictive scaling: {'Enabled' if resource_stats['scaling_policy']['predictive_enabled'] else 'Disabled'}")
        
        # Metrics statistics
        if self.metrics_manager:
            metrics_stats = self.metrics_manager.get_metrics_summary()
            print(f"\n📊 Metrics Collection:")
            print(f"   • Active collectors: {metrics_stats['collectors']}")
            print(f"   • Active exporters: {metrics_stats['exporters']}")
            print(f"   • Metric definitions: {metrics_stats['metric_definitions']}")
        
        # Logging statistics
        if self.logging_system:
            logging_stats = self.logging_system.get_system_statistics()
            print(f"\n📝 Logging & Audit:")
            print(f"   • Storage backends: {logging_stats['storage_backends']}")
            print(f"   • Logs processed: {logging_stats['logs_processed']}")
            print(f"   • Audit events: {logging_stats['audit_events_processed']}")
        
        print(f"\n✅ DEMONSTRATION COMPLETE")
        print(f"All advanced XORB systems successfully demonstrated!")
        
        # Final audit event
        await self.logging_system.audit(
            AuditEventType.SYSTEM_STOP,
            "integration_demo_completed",
            details={
                "demo_duration": "complete",
                "systems_tested": ["agent_discovery", "resource_management", "metrics", "logging"],
                "status": "success"
            }
        )


async def main() -> None:
    """Main demonstration function."""
    print("🎯 XORB Advanced Integration Demonstration")
    print("=" * 60)
    print("This demo showcases the advanced features implemented in XORB:")
    print("• Advanced Agent Discovery & Registration")
    print("• Dynamic Resource Allocation & Scaling")
    print("• Comprehensive Metrics & Monitoring")
    print("• Enterprise Logging & Audit Trails")
    print("• Security Hardening & Compliance")
    print("=" * 60)
    
    demo = XORBIntegrationDemo()
    
    try:
        # Initialize all systems
        await demo.initialize_systems()
        
        # Run demonstrations
        await demo.demo_agent_discovery()
        await demo.demo_resource_management()
        await demo.demo_advanced_metrics()
        await demo.demo_logging_and_audit()
        await demo.demo_integration_scenarios()
        
        # Generate final report
        await demo.generate_summary_report()
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Thank you for exploring XORB Advanced Features!")


if __name__ == "__main__":
    asyncio.run(main())