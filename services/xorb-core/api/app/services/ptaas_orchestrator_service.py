"""
PTaaS Orchestrator Service - Real-world integration service
Coordinates all PTaaS components with production-ready implementations
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from ..domain.tenant_entities import Finding
from ..infrastructure.database import get_async_session
from ..infrastructure.observability import get_metrics_collector, add_trace_context
from .base_service import XORBService, ServiceHealth, ServiceStatus
from .threat_intelligence_service import get_threat_intelligence_engine
from .intelligence_service import IntelligenceService

logger = logging.getLogger(__name__)

@dataclass
class PTaaSTarget:
    """PTaaS scan target definition"""
    target_id: str
    host: str
    ports: List[int]
    scan_profile: str = "comprehensive"
    constraints: List[str] = None
    authorized: bool = True

@dataclass
class PTaaSSession:
    """PTaaS session management"""
    session_id: str
    tenant_id: UUID
    targets: List[PTaaSTarget]
    scan_type: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

@dataclass
class ScanResult:
    """Scan result summary"""
    target_id: str
    vulnerabilities_found: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    services_discovered: List[Dict[str, Any]]
    scan_duration: float
    tools_used: List[str]
    raw_data: Dict[str, Any]

class PTaaSOrchestrator(XORBService):
    """Production-ready PTaaS orchestration service"""

    def __init__(self, intelligence_service: IntelligenceService):
        super().__init__()
        self.intelligence_service = intelligence_service
        self.threat_intel_engine = None
        self.active_sessions: Dict[str, PTaaSSession] = {}
        self.scan_queue = asyncio.Queue()
        self.worker_tasks: List[asyncio.Task] = []
        self.metrics = get_metrics_collector()

        # Scanner integrations
        self.available_scanners = {}
        self.scan_profiles = {
            "quick": {
                "description": "Fast network scan with basic service detection",
                "timeout": 300,
                "tools": ["nmap_basic"],
                "max_ports": 100
            },
            "comprehensive": {
                "description": "Full security assessment with vulnerability scanning",
                "timeout": 1800,
                "tools": ["nmap_full", "nuclei", "custom_checks"],
                "max_ports": 1000
            },
            "stealth": {
                "description": "Low-profile scanning to avoid detection",
                "timeout": 3600,
                "tools": ["nmap_stealth", "custom_passive"],
                "max_ports": 500
            },
            "web_focused": {
                "description": "Specialized web application security testing",
                "timeout": 1200,
                "tools": ["nmap_web", "nikto", "dirb", "nuclei_web"],
                "max_ports": 50
            }
        }

    async def initialize(self) -> bool:
        """Initialize PTaaS orchestrator"""
        try:
            logger.info("Initializing PTaaS Orchestrator...")

            # Initialize threat intelligence engine
            from .job_service import get_job_service
            job_service = get_job_service()
            self.threat_intel_engine = get_threat_intelligence_engine(job_service)

            # Initialize scanner integrations
            await self._initialize_scanners()

            # Start worker tasks
            for i in range(3):  # 3 concurrent scan workers
                worker = asyncio.create_task(self._scan_worker(f"worker-{i}"))
                self.worker_tasks.append(worker)

            logger.info(f"PTaaS Orchestrator initialized with {len(self.available_scanners)} scanners")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize PTaaS Orchestrator: {e}")
            return False

    async def shutdown(self) -> bool:
        """Shutdown PTaaS orchestrator"""
        try:
            logger.info("Shutting down PTaaS Orchestrator...")

            # Cancel active scans
            for session_id in list(self.active_sessions.keys()):
                await self.cancel_scan_session(session_id)

            # Cancel worker tasks
            for worker in self.worker_tasks:
                worker.cancel()

            await asyncio.gather(*self.worker_tasks, return_exceptions=True)

            logger.info("PTaaS Orchestrator shutdown complete")
            return True

        except Exception as e:
            logger.error(f"Failed to shutdown PTaaS Orchestrator: {e}")
            return False

    async def _initialize_scanners(self):
        """Initialize available security scanners"""
        try:
            # Try to import and initialize real-world scanner
            import sys
            import os
            ptaas_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'ptaas')
            if ptaas_path not in sys.path:
                sys.path.append(ptaas_path)

            try:
                from scanning.real_world_scanner import get_scanner
                scanner = get_scanner()
                self.available_scanners["real_world"] = scanner
                logger.info("Real-world scanner integration available")
            except ImportError as e:
                logger.warning(f"Real-world scanner not available: {e}")
                # Fallback to mock scanner
                self.available_scanners["mock"] = self._create_mock_scanner()

            # Try to import threat hunting engine
            try:
                from threat_hunting_engine import ThreatHuntingEngine
                hunting_engine = ThreatHuntingEngine()
                await hunting_engine.initialize()
                self.available_scanners["threat_hunting"] = hunting_engine
                logger.info("Threat hunting engine available")
            except ImportError:
                logger.debug("Threat hunting engine not available")

            # Try to import behavioral analytics
            try:
                from behavioral_analytics import BehavioralAnalyticsEngine
                analytics_engine = BehavioralAnalyticsEngine()
                await analytics_engine.initialize()
                self.available_scanners["behavioral_analytics"] = analytics_engine
                logger.info("Behavioral analytics engine available")
            except ImportError:
                logger.debug("Behavioral analytics engine not available")

        except Exception as e:
            logger.error(f"Failed to initialize scanners: {e}")
            self.available_scanners["mock"] = self._create_mock_scanner()

    def _create_mock_scanner(self):
        """Create mock scanner for fallback"""
        class MockScanner:
            async def comprehensive_scan(self, target):
                await asyncio.sleep(2)  # Simulate scan time
                return {
                    "target": target.host,
                    "vulnerabilities": [
                        {
                            "name": "Mock Vulnerability",
                            "severity": "medium",
                            "description": "Simulated vulnerability for testing",
                            "port": 80
                        }
                    ],
                    "services": [
                        {"port": 80, "name": "http", "version": "unknown"}
                    ],
                    "scan_duration": 2.0
                }

        return MockScanner()

    async def create_scan_session(self,
                                targets: List[PTaaSTarget],
                                scan_type: str,
                                tenant_id: UUID,
                                metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create new PTaaS scan session"""
        try:
            session_id = str(uuid4())

            # Validate scan profile
            if scan_type not in self.scan_profiles:
                raise ValueError(f"Invalid scan type: {scan_type}")

            # Validate targets
            for target in targets:
                if not target.authorized:
                    raise ValueError(f"Target {target.host} not authorized for scanning")

            session = PTaaSSession(
                session_id=session_id,
                tenant_id=tenant_id,
                targets=targets,
                scan_type=scan_type,
                status="created",
                created_at=datetime.utcnow(),
                metadata=metadata or {}
            )

            self.active_sessions[session_id] = session

            logger.info(f"Created PTaaS session {session_id} with {len(targets)} targets")
            return session_id

        except Exception as e:
            logger.error(f"Failed to create scan session: {e}")
            raise

    async def start_scan_session(self, session_id: str) -> bool:
        """Start PTaaS scan session"""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")

            session = self.active_sessions[session_id]
            if session.status != "created":
                raise ValueError(f"Session {session_id} cannot be started (status: {session.status})")

            session.status = "queued"
            session.started_at = datetime.utcnow()

            # Add to scan queue
            await self.scan_queue.put(session)

            logger.info(f"Started PTaaS session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to start scan session {session_id}: {e}")
            return False

    async def _scan_worker(self, worker_name: str):
        """Background worker for processing scans"""
        logger.info(f"PTaaS scan worker {worker_name} started")

        while True:
            try:
                # Get next session to process
                session = await self.scan_queue.get()

                logger.info(f"Worker {worker_name} processing session {session.session_id}")

                # Process the scan session
                await self._process_scan_session(session)

                # Mark task done
                self.scan_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scan worker {worker_name} error: {e}")
                await asyncio.sleep(1)

    async def _process_scan_session(self, session: PTaaSSession):
        """Process individual scan session"""
        try:
            session.status = "running"
            scan_results = []

            # Get scan profile configuration
            profile = self.scan_profiles[session.scan_type]

            # Process each target
            for target in session.targets:
                try:
                    logger.info(f"Scanning target {target.host}")

                    # Execute scan based on available scanners
                    if "real_world" in self.available_scanners:
                        result = await self._run_real_world_scan(target, profile)
                    else:
                        result = await self._run_mock_scan(target, profile)

                    # Enrich with threat intelligence
                    enriched_result = await self._enrich_with_threat_intel(result, session.tenant_id)

                    # Store findings in database
                    await self._store_scan_findings(enriched_result, session)

                    scan_results.append(enriched_result)

                except Exception as e:
                    logger.error(f"Failed to scan target {target.host}: {e}")
                    scan_results.append({
                        "target_id": target.target_id,
                        "error": str(e),
                        "status": "failed"
                    })

            # Generate session summary
            session.results = {
                "scan_results": scan_results,
                "summary": self._generate_session_summary(scan_results),
                "completed_at": datetime.utcnow().isoformat()
            }

            session.status = "completed"
            session.completed_at = datetime.utcnow()

            # Record metrics
            duration = (session.completed_at - session.started_at).total_seconds()
            self.metrics.record_job_execution(f"ptaas_scan_{session.scan_type}", duration, True)

            add_trace_context(
                operation="ptaas_scan_completed",
                session_id=session.session_id,
                targets_scanned=len(session.targets),
                vulnerabilities_found=session.results["summary"]["total_vulnerabilities"]
            )

            logger.info(f"PTaaS session {session.session_id} completed successfully")

        except Exception as e:
            logger.error(f"PTaaS session {session.session_id} failed: {e}")
            session.status = "failed"
            session.completed_at = datetime.utcnow()
            session.results = {"error": str(e)}

    async def _run_real_world_scan(self, target: PTaaSTarget, profile: Dict[str, Any]) -> ScanResult:
        """Run scan using real-world scanner integration"""
        try:
            scanner = self.available_scanners["real_world"]

            # Convert PTaaS target to scanner target format
            from scanning.real_world_scanner import ScanTarget
            scan_target = ScanTarget(
                host=target.host,
                ports=target.ports,
                scan_type=profile.get("description", "comprehensive"),
                timeout=profile.get("timeout", 300),
                stealth_mode="stealth" in profile.get("tools", [])
            )

            # Execute comprehensive scan
            scan_result = await scanner.comprehensive_scan(scan_target)

            # Convert to PTaaS result format
            return ScanResult(
                target_id=target.target_id,
                vulnerabilities_found=len(scan_result.vulnerabilities),
                critical_issues=len([v for v in scan_result.vulnerabilities if v.get("severity") == "critical"]),
                high_issues=len([v for v in scan_result.vulnerabilities if v.get("severity") == "high"]),
                medium_issues=len([v for v in scan_result.vulnerabilities if v.get("severity") == "medium"]),
                low_issues=len([v for v in scan_result.vulnerabilities if v.get("severity") == "low"]),
                services_discovered=scan_result.services,
                scan_duration=scan_result.scan_statistics.get("duration_seconds", 0),
                tools_used=scan_result.scan_statistics.get("tools_used", []),
                raw_data=asdict(scan_result)
            )

        except Exception as e:
            logger.error(f"Real-world scan failed: {e}")
            raise

    async def _run_mock_scan(self, target: PTaaSTarget, profile: Dict[str, Any]) -> ScanResult:
        """Run mock scan for testing/fallback"""
        try:
            scanner = self.available_scanners["mock"]

            # Create mock target
            mock_target = type('MockTarget', (), {
                'host': target.host,
                'ports': target.ports
            })()

            result = await scanner.comprehensive_scan(mock_target)

            return ScanResult(
                target_id=target.target_id,
                vulnerabilities_found=len(result.get("vulnerabilities", [])),
                critical_issues=0,
                high_issues=0,
                medium_issues=len(result.get("vulnerabilities", [])),
                low_issues=0,
                services_discovered=result.get("services", []),
                scan_duration=result.get("scan_duration", 2.0),
                tools_used=["mock_scanner"],
                raw_data=result
            )

        except Exception as e:
            logger.error(f"Mock scan failed: {e}")
            raise

    async def _enrich_with_threat_intel(self, scan_result: ScanResult, tenant_id: UUID) -> ScanResult:
        """Enrich scan results with threat intelligence"""
        try:
            if not self.threat_intel_engine:
                return scan_result

            # Process each discovered service
            for service in scan_result.services_discovered:
                service_name = service.get("name", "")
                service_version = service.get("version", "")

                # Check for known vulnerabilities
                if service_name and service_version:
                    enrichment = await self.threat_intel_engine.enrich_evidence(
                        f"{service_name}:{service_version}",
                        tenant_id
                    )

                    if enrichment.get("threat_matches"):
                        service["threat_intelligence"] = enrichment

            return scan_result

        except Exception as e:
            logger.error(f"Threat intelligence enrichment failed: {e}")
            return scan_result

    async def _store_scan_findings(self, scan_result: ScanResult, session: PTaaSSession):
        """Store scan findings in database"""
        try:
            async with get_async_session() as db_session:
                # Set tenant context
                await db_session.execute(
                    "SELECT set_config('app.tenant_id', :tenant_id, false)",
                    {"tenant_id": str(session.tenant_id)}
                )

                # Create findings for vulnerabilities
                for i, vuln in enumerate(scan_result.raw_data.get("vulnerabilities", [])):
                    finding = Finding(
                        tenant_id=session.tenant_id,
                        title=f"PTaaS: {vuln.get('name', 'Security Issue')}",
                        description=vuln.get('description', 'Vulnerability discovered during PTaaS scan'),
                        severity=vuln.get('severity', 'medium'),
                        category="ptaas_vulnerability",
                        tags=["ptaas", "automated_scan", session.scan_type],
                        custom_metadata={
                            "session_id": session.session_id,
                            "target_id": scan_result.target_id,
                            "scanner": vuln.get('scanner', 'unknown'),
                            "port": vuln.get('port'),
                            "service": vuln.get('service'),
                            "raw_vulnerability": vuln
                        },
                        created_by="ptaas_orchestrator"
                    )

                    db_session.add(finding)

                await db_session.commit()

        except Exception as e:
            logger.error(f"Failed to store scan findings: {e}")

    def _generate_session_summary(self, scan_results: List[ScanResult]) -> Dict[str, Any]:
        """Generate summary of scan session results"""
        try:
            total_vulnerabilities = sum(r.vulnerabilities_found for r in scan_results if hasattr(r, 'vulnerabilities_found'))
            total_critical = sum(r.critical_issues for r in scan_results if hasattr(r, 'critical_issues'))
            total_high = sum(r.high_issues for r in scan_results if hasattr(r, 'high_issues'))
            total_medium = sum(r.medium_issues for r in scan_results if hasattr(r, 'medium_issues'))
            total_low = sum(r.low_issues for r in scan_results if hasattr(r, 'low_issues'))

            all_tools = set()
            for result in scan_results:
                if hasattr(result, 'tools_used'):
                    all_tools.update(result.tools_used)

            return {
                "targets_scanned": len(scan_results),
                "successful_scans": len([r for r in scan_results if not hasattr(r, 'error')]),
                "failed_scans": len([r for r in scan_results if hasattr(r, 'error')]),
                "total_vulnerabilities": total_vulnerabilities,
                "critical_vulnerabilities": total_critical,
                "high_vulnerabilities": total_high,
                "medium_vulnerabilities": total_medium,
                "low_vulnerabilities": total_low,
                "tools_used": list(all_tools),
                "risk_level": "critical" if total_critical > 0 else "high" if total_high > 0 else "medium" if total_medium > 0 else "low"
            }

        except Exception as e:
            logger.error(f"Failed to generate session summary: {e}")
            return {"error": str(e)}

    async def get_scan_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of scan session"""
        try:
            if session_id not in self.active_sessions:
                return None

            session = self.active_sessions[session_id]

            return {
                "session_id": session_id,
                "status": session.status,
                "scan_type": session.scan_type,
                "targets_count": len(session.targets),
                "created_at": session.created_at.isoformat(),
                "started_at": session.started_at.isoformat() if session.started_at else None,
                "completed_at": session.completed_at.isoformat() if session.completed_at else None,
                "results": session.results
            }

        except Exception as e:
            logger.error(f"Failed to get session status: {e}")
            return None

    async def cancel_scan_session(self, session_id: str) -> bool:
        """Cancel active scan session"""
        try:
            if session_id not in self.active_sessions:
                return False

            session = self.active_sessions[session_id]
            if session.status in ["completed", "failed", "cancelled"]:
                return False

            session.status = "cancelled"
            session.completed_at = datetime.utcnow()

            logger.info(f"Cancelled PTaaS session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel session {session_id}: {e}")
            return False

    async def get_available_scan_profiles(self) -> Dict[str, Any]:
        """Get available scan profiles"""
        return {
            "profiles": self.scan_profiles,
            "available_scanners": list(self.available_scanners.keys())
        }

    async def health_check(self) -> ServiceHealth:
        """Perform health check"""
        try:
            checks = {
                "active_sessions": len(self.active_sessions),
                "queue_size": self.scan_queue.qsize(),
                "available_scanners": len(self.available_scanners),
                "worker_tasks": len([t for t in self.worker_tasks if not t.done()])
            }

            status = ServiceStatus.HEALTHY
            message = "PTaaS Orchestrator is operational"

            # Check for issues
            if len(self.available_scanners) == 0:
                status = ServiceStatus.DEGRADED
                message = "No scanners available"
            elif self.scan_queue.qsize() > 10:
                status = ServiceStatus.DEGRADED
                message = "High queue load"

            return ServiceHealth(
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                checks=checks
            )

        except Exception as e:
            return ServiceHealth(
                status=ServiceStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                timestamp=datetime.utcnow(),
                checks={"error": str(e)}
            )

# Global orchestrator instance
_ptaas_orchestrator: Optional[PTaaSOrchestrator] = None

async def get_ptaas_orchestrator(intelligence_service: IntelligenceService) -> PTaaSOrchestrator:
    """Get global PTaaS orchestrator instance"""
    global _ptaas_orchestrator

    if _ptaas_orchestrator is None:
        _ptaas_orchestrator = PTaaSOrchestrator(intelligence_service)
        await _ptaas_orchestrator.initialize()

    return _ptaas_orchestrator
