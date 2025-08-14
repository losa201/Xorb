"""
XORB Unified Orchestrator - Production Implementation
Integrates PTaaS scanning, workflow automation, and threat intelligence
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4

# Add the API path to enable imports
sys.path.append(str(Path(__file__).parent.parent / "api"))

from app.services.ptaas_orchestrator_service import get_orchestration_service, PTaaSOrchestrationService
from app.services.ptaas_scanner_service import get_scanner_service, SecurityScannerService
from app.services.intelligence_service import IntelligenceService
from app.domain.tenant_entities import ScanTarget
from core.workflow_engine import ProductionWorkflowOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/xorb/unified_orchestrator.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


class UnifiedOrchestrator:
    """Production-ready unified orchestrator for XORB platform"""

    def __init__(self):
        self.running = False
        self.ptaas_service: Optional[PTaaSOrchestrationService] = None
        self.scanner_service: Optional[SecurityScannerService] = None
        self.intelligence_service: Optional[IntelligenceService] = None
        self.workflow_orchestrator: Optional[ProductionWorkflowOrchestrator] = None

        # Platform statistics
        self.stats = {
            "uptime_start": datetime.utcnow(),
            "scans_completed": 0,
            "workflows_executed": 0,
            "threats_analyzed": 0,
            "compliance_checks": 0,
            "last_error": None
        }

        # Active tasks tracking
        self.active_scans: Dict[str, asyncio.Task] = {}
        self.active_workflows: Dict[str, asyncio.Task] = {}

        # Default tenant for demo purposes
        self.demo_tenant_id = UUID("00000000-0000-0000-0000-000000000001")

    async def initialize(self) -> bool:
        """Initialize all orchestrator services"""
        try:
            logger.info("Initializing XORB Unified Orchestrator...")

            # Initialize PTaaS orchestration service
            self.ptaas_service = await get_orchestration_service()
            logger.info("PTaaS orchestration service initialized")

            # Initialize scanner service
            self.scanner_service = await get_scanner_service()
            logger.info("Scanner service initialized")

            # Initialize intelligence service
            self.intelligence_service = IntelligenceService()
            await self.intelligence_service.initialize()
            logger.info("Intelligence service initialized")

            # Initialize workflow orchestrator
            self.workflow_orchestrator = ProductionWorkflowOrchestrator()
            await self.workflow_orchestrator.initialize()
            logger.info("Workflow orchestrator initialized")

            # Schedule initial demo workflows
            await self._schedule_initial_workflows()

            logger.info("XORB Unified Orchestrator initialization complete")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize unified orchestrator: {e}")
            return False

    async def _schedule_initial_workflows(self):
        """Schedule initial demo workflows for immediate execution"""
        try:
            # Define demo targets
            demo_targets = [
                ScanTarget(
                    host="scanme.nmap.org",
                    ports=[22, 80, 443],
                    scan_type="comprehensive"
                ),
                ScanTarget(
                    host="testphp.vulnweb.com",
                    ports=[80, 443],
                    scan_type="web_focused"
                )
            ]

            # Execute comprehensive security assessment
            logger.info("Scheduling demo comprehensive security assessment...")
            assessment_id = await self.ptaas_service.execute_workflow(
                "comprehensive_assessment",
                demo_targets,
                triggered_by="unified_orchestrator",
                tenant_id=self.demo_tenant_id
            )

            logger.info(f"Demo assessment scheduled: {assessment_id}")

            # Execute PCI-DSS compliance scan
            logger.info("Scheduling demo PCI-DSS compliance scan...")
            compliance_id = await self.ptaas_service.execute_workflow(
                "pci_dss_compliance",
                demo_targets[:1],  # Single target for compliance
                triggered_by="unified_orchestrator",
                tenant_id=self.demo_tenant_id
            )

            logger.info(f"Demo compliance scan scheduled: {compliance_id}")

            # Start monitoring these executions
            asyncio.create_task(self._monitor_execution(assessment_id, "Comprehensive Assessment"))
            asyncio.create_task(self._monitor_execution(compliance_id, "PCI-DSS Compliance"))

        except Exception as e:
            logger.error(f"Failed to schedule initial workflows: {e}")

    async def start(self):
        """Start the unified orchestrator"""
        if not await self.initialize():
            logger.error("Failed to initialize unified orchestrator")
            return

        self.running = True
        logger.info("XORB Unified Orchestrator started successfully")

        # Start background monitoring tasks
        tasks = [
            asyncio.create_task(self._platform_monitor(), name="platform_monitor"),
            asyncio.create_task(self._metrics_collector(), name="metrics_collector"),
            asyncio.create_task(self._continuous_scanning(), name="continuous_scanning"),
            asyncio.create_task(self._threat_intelligence_processor(), name="threat_intel"),
            asyncio.create_task(self._compliance_monitor(), name="compliance_monitor")
        ]

        try:
            # Wait for shutdown signal
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Unified orchestrator error: {e}")
        finally:
            await self.shutdown()

    async def _platform_monitor(self):
        """Monitor overall platform health and performance"""
        logger.info("Platform monitor started")

        while self.running:
            try:
                # Check service health
                services_status = {
                    "ptaas_service": self.ptaas_service is not None,
                    "scanner_service": self.scanner_service is not None,
                    "intelligence_service": self.intelligence_service is not None,
                    "workflow_orchestrator": self.workflow_orchestrator is not None
                }

                # Check active operations
                active_operations = {
                    "active_scans": len(self.active_scans),
                    "active_workflows": len(self.active_workflows),
                    "total_operations": len(self.active_scans) + len(self.active_workflows)
                }

                # Log platform status
                if all(services_status.values()):
                    logger.debug(f"Platform healthy: {active_operations['total_operations']} active operations")
                else:
                    logger.warning(f"Platform issues detected: {services_status}")

                await asyncio.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                logger.error(f"Platform monitor error: {e}")
                await asyncio.sleep(30)

    async def _metrics_collector(self):
        """Collect and log platform metrics"""
        logger.info("Metrics collector started")

        while self.running:
            try:
                uptime = datetime.utcnow() - self.stats["uptime_start"]

                # Collect metrics from services
                scanner_health = None
                if self.scanner_service:
                    scanner_health = await self.scanner_service.health_check()

                platform_metrics = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "uptime_hours": uptime.total_seconds() / 3600,
                    "stats": self.stats,
                    "scanner_health": scanner_health.checks if scanner_health else None,
                    "active_operations": {
                        "scans": len(self.active_scans),
                        "workflows": len(self.active_workflows)
                    }
                }

                logger.info(f"Platform metrics: {json.dumps(platform_metrics, default=str)}")

                await asyncio.sleep(300)  # Collect every 5 minutes

            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(300)

    async def _continuous_scanning(self):
        """Continuous security scanning workflow"""
        logger.info("Continuous scanning monitor started")

        while self.running:
            try:
                # Check if we should run automated scans
                current_hour = datetime.utcnow().hour

                # Run comprehensive scans every 6 hours
                if current_hour % 6 == 0:
                    await self._execute_automated_scan("comprehensive")

                # Run quick scans every 2 hours
                elif current_hour % 2 == 0:
                    await self._execute_automated_scan("quick")

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Continuous scanning error: {e}")
                await asyncio.sleep(3600)

    async def _execute_automated_scan(self, scan_type: str):
        """Execute automated security scan"""
        try:
            if not self.scanner_service:
                return

            # Define targets for automated scanning
            targets = [
                ScanTarget(
                    host="scanme.nmap.org",
                    ports=[22, 80, 443],
                    scan_type=scan_type
                )
            ]

            logger.info(f"Starting automated {scan_type} scan")

            for target in targets:
                scan_id = str(uuid4())
                scan_task = asyncio.create_task(
                    self._run_single_scan(target, scan_id, scan_type)
                )
                self.active_scans[scan_id] = scan_task

        except Exception as e:
            logger.error(f"Failed to execute automated scan: {e}")

    async def _run_single_scan(self, target: ScanTarget, scan_id: str, scan_type: str):
        """Run a single security scan"""
        try:
            logger.info(f"Starting {scan_type} scan {scan_id} for {target.host}")

            scan_result = await self.scanner_service.comprehensive_scan(target)

            self.stats["scans_completed"] += 1

            # Process scan results with intelligence service
            if self.intelligence_service and scan_result.vulnerabilities:
                await self._process_scan_intelligence(scan_result)

            logger.info(f"Scan {scan_id} completed: {len(scan_result.vulnerabilities)} vulnerabilities found")

        except Exception as e:
            logger.error(f"Scan {scan_id} failed: {e}")
        finally:
            # Cleanup
            self.active_scans.pop(scan_id, None)

    async def _process_scan_intelligence(self, scan_result):
        """Process scan results with threat intelligence"""
        try:
            # Analyze vulnerabilities for threat intelligence
            for vuln in scan_result.vulnerabilities[:5]:  # Process top 5
                threat_analysis = await self.intelligence_service.process_decision_request(
                    type("MockRequest", (), {
                        "decision_type": "THREAT_CLASSIFICATION",
                        "context": type("MockContext", (), {
                            "scenario": f"Vulnerability: {vuln.get('name', 'Unknown')}",
                            "available_data": {
                                "severity": vuln.get("severity", "unknown"),
                                "scanner": vuln.get("scanner", "unknown"),
                                "description": vuln.get("description", "")
                            },
                            "constraints": [],
                            "historical_context": [],
                            "urgency_level": "normal",
                            "confidence_threshold": 0.7
                        })(),
                        "model_preferences": []
                    })(),
                    self.demo_tenant_id
                )

                if threat_analysis:
                    self.stats["threats_analyzed"] += 1
                    logger.info(f"Threat analysis: {threat_analysis.recommendation}")

        except Exception as e:
            logger.error(f"Intelligence processing failed: {e}")

    async def _threat_intelligence_processor(self):
        """Process threat intelligence continuously"""
        logger.info("Threat intelligence processor started")

        while self.running:
            try:
                # Simulate threat intelligence processing
                if self.intelligence_service:
                    # Process any pending threat analysis
                    pass

                await asyncio.sleep(600)  # Process every 10 minutes

            except Exception as e:
                logger.error(f"Threat intelligence processing error: {e}")
                await asyncio.sleep(600)

    async def _compliance_monitor(self):
        """Monitor compliance status continuously"""
        logger.info("Compliance monitor started")

        while self.running:
            try:
                # Run compliance checks periodically
                current_time = datetime.utcnow()

                # Daily compliance checks
                if current_time.hour == 2:  # Run at 2 AM daily
                    await self._run_compliance_check("daily")

                # Weekly compliance checks
                if current_time.weekday() == 0 and current_time.hour == 3:  # Monday 3 AM
                    await self._run_compliance_check("weekly")

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Compliance monitor error: {e}")
                await asyncio.sleep(3600)

    async def _run_compliance_check(self, check_type: str):
        """Run compliance checking workflow"""
        try:
            if not self.ptaas_service:
                return

            logger.info(f"Running {check_type} compliance check")

            # Define compliance targets
            targets = [
                ScanTarget(
                    host="testphp.vulnweb.com",
                    ports=[80, 443],
                    scan_type="compliance"
                )
            ]

            # Execute PCI-DSS compliance workflow
            execution_id = await self.ptaas_service.execute_workflow(
                "pci_dss_compliance",
                targets,
                triggered_by=f"compliance_monitor_{check_type}",
                tenant_id=self.demo_tenant_id
            )

            self.stats["compliance_checks"] += 1

            # Monitor execution
            asyncio.create_task(self._monitor_execution(execution_id, f"Compliance Check ({check_type})"))

        except Exception as e:
            logger.error(f"Compliance check failed: {e}")

    async def _monitor_execution(self, execution_id: str, execution_name: str):
        """Monitor workflow execution"""
        try:
            logger.info(f"Monitoring execution: {execution_name} ({execution_id})")

            while True:
                if not self.ptaas_service:
                    break

                execution = await self.ptaas_service.get_execution_status(execution_id)

                if execution is None:
                    logger.warning(f"Execution {execution_id} not found")
                    break

                if execution.status.value in ["completed", "failed", "cancelled"]:
                    logger.info(f"Execution {execution_name} completed: {execution.status.value}")

                    if execution.status.value == "completed":
                        self.stats["workflows_executed"] += 1

                        # Log execution summary
                        if execution.findings_summary:
                            summary = execution.findings_summary
                            logger.info(f"Execution results: {summary.get('total_vulnerabilities', 0)} vulnerabilities, "
                                      f"compliance score: {execution.compliance_score or 'N/A'}")

                    break

                await asyncio.sleep(15)  # Check every 15 seconds

        except Exception as e:
            logger.error(f"Error monitoring execution {execution_id}: {e}")
        finally:
            # Cleanup
            self.active_workflows.pop(execution_id, None)

    async def execute_manual_scan(self, targets: List[ScanTarget], scan_type: str = "comprehensive") -> str:
        """Execute manual security scan"""
        try:
            if not self.ptaas_service:
                raise Exception("PTaaS service not available")

            workflow_map = {
                "comprehensive": "comprehensive_assessment",
                "compliance": "pci_dss_compliance",
                "threat_simulation": "apt_threat_simulation"
            }

            workflow_id = workflow_map.get(scan_type, "comprehensive_assessment")

            execution_id = await self.ptaas_service.execute_workflow(
                workflow_id,
                targets,
                triggered_by="manual_request",
                tenant_id=self.demo_tenant_id
            )

            # Monitor execution
            asyncio.create_task(self._monitor_execution(execution_id, f"Manual {scan_type} scan"))

            return execution_id

        except Exception as e:
            logger.error(f"Manual scan execution failed: {e}")
            raise

    async def get_platform_status(self) -> Dict[str, Any]:
        """Get comprehensive platform status"""
        uptime = datetime.utcnow() - self.stats["uptime_start"]

        # Get service statuses
        service_statuses = {}

        if self.scanner_service:
            scanner_health = await self.scanner_service.health_check()
            service_statuses["scanner"] = {
                "status": scanner_health.status.value if scanner_health else "unknown",
                "available_scanners": scanner_health.checks.get("available_scanners", 0) if scanner_health else 0
            }

        if self.ptaas_service:
            workflows = await self.ptaas_service.list_workflows()
            executions = await self.ptaas_service.list_executions()
            service_statuses["ptaas"] = {
                "workflows_available": len(workflows),
                "total_executions": len(executions),
                "recent_executions": len([e for e in executions if
                                        (datetime.utcnow() - e.started_at).days < 1])
            }

        return {
            "platform": "XORB Unified Security Platform",
            "status": "operational" if self.running else "stopped",
            "uptime_hours": uptime.total_seconds() / 3600,
            "statistics": self.stats,
            "services": service_statuses,
            "active_operations": {
                "scans": len(self.active_scans),
                "workflows": len(self.active_workflows)
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    async def shutdown(self):
        """Gracefully shutdown the unified orchestrator"""
        logger.info("Shutting down XORB Unified Orchestrator...")

        self.running = False

        # Cancel active scans
        for scan_id, task in self.active_scans.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled scan: {scan_id}")

        # Cancel active workflows
        for workflow_id, task in self.active_workflows.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled workflow: {workflow_id}")

        # Wait for cancellations
        all_tasks = list(self.active_scans.values()) + list(self.active_workflows.values())
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)

        # Shutdown services
        if self.ptaas_service:
            await self.ptaas_service.shutdown()

        if self.scanner_service:
            await self.scanner_service.shutdown()

        if self.workflow_orchestrator:
            await self.workflow_orchestrator.shutdown()

        logger.info("XORB Unified Orchestrator shutdown complete")


# Global orchestrator instance
unified_orchestrator = UnifiedOrchestrator()


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    asyncio.create_task(unified_orchestrator.shutdown())


async def main():
    """Main entry point for unified orchestrator"""
    try:
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start unified orchestrator
        await unified_orchestrator.start()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unified orchestrator failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("XORB Unified Orchestrator stopped")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
