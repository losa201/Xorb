"""
PTaaS Orchestration Service Helper Methods
Provides advanced workflow orchestration functionality
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
from pathlib import Path

logger = logging.getLogger(__name__)


class PTaaSOrchestratorHelpers:
    """Helper methods for PTaaS Orchestrator"""

    def _load_compliance_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load compliance framework configurations"""
        return {
            "PCI-DSS": {
                "version": "4.0",
                "controls": {
                    "network_security": ["1.1", "1.2", "1.3", "2.1", "2.2"],
                    "vulnerability_management": ["11.2", "11.3"],
                    "access_control": ["7.1", "7.2", "8.1", "8.2"],
                    "encryption": ["3.4", "4.1", "4.2"],
                    "monitoring": ["10.1", "10.2", "10.3"]
                },
                "scan_requirements": {
                    "quarterly_external": True,
                    "annual_internal": True,
                    "authenticated_scanning": True,
                    "high_level_requirements": True
                },
                "reporting": {
                    "aoc_required": True,
                    "asv_scan_required": True,
                    "penetration_test_required": True
                }
            },
            "HIPAA": {
                "version": "2013",
                "controls": {
                    "access_control": ["164.308(a)(4)", "164.312(a)(1)"],
                    "audit_controls": ["164.312(b)"],
                    "integrity": ["164.312(c)(1)"],
                    "transmission_security": ["164.312(e)"]
                },
                "scan_requirements": {
                    "risk_assessment": True,
                    "vulnerability_scanning": True,
                    "penetration_testing": True,
                    "audit_logging": True
                }
            },
            "SOX": {
                "version": "2002",
                "controls": {
                    "it_controls": ["SOX.302", "SOX.404"],
                    "change_management": ["SOX.404"],
                    "access_controls": ["SOX.302"],
                    "data_backup": ["SOX.404"]
                },
                "scan_requirements": {
                    "it_control_testing": True,
                    "change_management_review": True,
                    "access_control_review": True
                }
            },
            "ISO-27001": {
                "version": "2022",
                "controls": {
                    "information_security_policies": ["A.5"],
                    "organization_security": ["A.6"],
                    "human_resource_security": ["A.7"],
                    "asset_management": ["A.8"],
                    "access_control": ["A.9"],
                    "cryptography": ["A.10"],
                    "physical_security": ["A.11"],
                    "operations_security": ["A.12"],
                    "communications_security": ["A.13"],
                    "system_acquisition": ["A.14"],
                    "supplier_relationships": ["A.15"],
                    "incident_management": ["A.16"],
                    "business_continuity": ["A.17"],
                    "compliance": ["A.18"]
                }
            }
        }

    def _create_workflow_templates(self) -> Dict[str, Dict[str, Any]]:
        """Create predefined workflow templates"""
        return {
            "comprehensive_assessment": {
                "name": "Comprehensive Security Assessment",
                "description": "Complete security assessment with all scanning phases",
                "phases": [
                    "network_discovery",
                    "vulnerability_scanning",
                    "web_application_testing",
                    "ssl_tls_analysis",
                    "database_security_testing",
                    "compliance_validation",
                    "report_generation"
                ],
                "estimated_duration": 1800,  # 30 minutes
                "parallel_execution": True
            },
            "quick_assessment": {
                "name": "Quick Security Assessment",
                "description": "Fast security assessment focusing on critical vulnerabilities",
                "phases": [
                    "network_discovery",
                    "critical_vulnerability_scan",
                    "basic_report"
                ],
                "estimated_duration": 300,  # 5 minutes
                "parallel_execution": False
            },
            "compliance_pci_dss": {
                "name": "PCI-DSS Compliance Assessment",
                "description": "PCI-DSS compliance validation workflow",
                "phases": [
                    "network_segmentation_check",
                    "cardholder_data_protection",
                    "vulnerability_management",
                    "access_control_validation",
                    "monitoring_validation",
                    "pci_compliance_report"
                ],
                "estimated_duration": 2400,  # 40 minutes
                "compliance_framework": "PCI-DSS"
            },
            "red_team_simulation": {
                "name": "Red Team Exercise Simulation",
                "description": "Advanced persistent threat simulation",
                "phases": [
                    "reconnaissance",
                    "initial_access",
                    "persistence_establishment",
                    "privilege_escalation",
                    "lateral_movement",
                    "data_exfiltration",
                    "cleanup_and_reporting"
                ],
                "estimated_duration": 7200,  # 2 hours
                "stealth_mode": True
            }
        }

    def _create_session_workflow(
        self,
        targets: List[Any],
        scan_type: str,
        metadata: Dict[str, Any]
    ) -> Any:
        """Create workflow for a PTaaS session"""
        from . import PTaaSWorkflow, WorkflowType, WorkflowTask

        workflow_id = f"session_{scan_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Select workflow template based on scan type
        if scan_type == "comprehensive":
            template = self.workflow_templates["comprehensive_assessment"]
        elif scan_type == "quick":
            template = self.workflow_templates["quick_assessment"]
        elif scan_type == "compliance":
            framework = metadata.get("compliance_framework", "PCI-DSS")
            template = self.workflow_templates.get(f"compliance_{framework.lower().replace('-', '_')}",
                                                 self.workflow_templates["comprehensive_assessment"])
        elif scan_type == "red_team":
            template = self.workflow_templates["red_team_simulation"]
        else:
            template = self.workflow_templates["comprehensive_assessment"]

        # Create workflow tasks based on template
        tasks = []
        for i, phase in enumerate(template["phases"]):
            task = WorkflowTask(
                id=f"task_{i+1}_{phase}",
                name=phase.replace("_", " ").title(),
                task_type=phase,
                description=f"Execute {phase} phase",
                parameters={
                    "scan_type": scan_type,
                    "targets": len(targets),
                    "metadata": metadata
                },
                timeout_minutes=template.get("estimated_duration", 1800) // 60 // len(template["phases"]),
                parallel_execution=template.get("parallel_execution", False)
            )
            tasks.append(task)

        # Create workflow
        workflow = PTaaSWorkflow(
            id=workflow_id,
            name=template["name"],
            workflow_type=WorkflowType.VULNERABILITY_ASSESSMENT,
            description=template["description"],
            tasks=tasks,
            targets=targets,
            metadata=metadata
        )

        return workflow

    def _estimate_workflow_duration(self, workflow: Any) -> int:
        """Estimate workflow duration based on tasks and complexity"""
        base_duration = 0

        # Calculate base duration from tasks
        for task in workflow.tasks:
            if task.parallel_execution:
                # Parallel tasks contribute less to total duration
                base_duration += task.timeout_minutes * 60 * 0.3
            else:
                base_duration += task.timeout_minutes * 60

        # Adjust for number of targets
        target_multiplier = 1.0 + (len(workflow.targets) - 1) * 0.2

        # Adjust for workflow complexity
        complexity_multiplier = 1.0
        if workflow.workflow_type.value == "red_team_exercise":
            complexity_multiplier = 1.5
        elif workflow.workflow_type.value == "compliance_scan":
            complexity_multiplier = 1.3

        final_duration = int(base_duration * target_multiplier * complexity_multiplier)

        return max(final_duration, 300)  # Minimum 5 minutes

    async def _compile_session_results(self, session: Any, execution: Any) -> Dict[str, Any]:
        """Compile comprehensive results from session execution"""
        try:
            results = {
                "session_summary": {
                    "session_id": session.session_id,
                    "scan_type": session.scan_type,
                    "targets_scanned": len(session.targets),
                    "execution_status": execution.status.value,
                    "start_time": execution.start_time.isoformat(),
                    "end_time": execution.end_time.isoformat() if execution.end_time else None,
                    "duration_seconds": (execution.end_time - execution.start_time).total_seconds() if execution.end_time else None
                },
                "scan_results": {
                    "vulnerabilities": [],
                    "open_ports": [],
                    "services": [],
                    "compliance_findings": [],
                    "security_metrics": {}
                },
                "threat_intelligence": {
                    "risk_score": 0.0,
                    "threat_level": "low",
                    "attack_vectors": [],
                    "recommended_actions": []
                },
                "recommendations": [],
                "executive_summary": {}
            }

            # Aggregate results from all completed tasks
            total_vulnerabilities = 0
            critical_vulns = 0
            high_vulns = 0
            all_open_ports = set()
            all_services = []

            for task_id, task_result in execution.results.items():
                if task_result.get("status") == "completed":
                    task_data = task_result.get("result", {})

                    # Aggregate vulnerability data
                    if "vulnerabilities_found" in task_data:
                        total_vulnerabilities += task_data["vulnerabilities_found"]

                    if "severity_breakdown" in task_data:
                        critical_vulns += task_data["severity_breakdown"].get("critical", 0)
                        high_vulns += task_data["severity_breakdown"].get("high", 0)

                    if "detailed_findings" in task_data:
                        results["scan_results"]["vulnerabilities"].extend(
                            task_data["detailed_findings"][:10]  # Limit for response size
                        )

                    # Aggregate port and service data
                    if "open_ports_total" in task_data:
                        all_open_ports.update(range(task_data["open_ports_total"]))

                    # Collect compliance findings
                    if "compliance_score" in task_data:
                        results["scan_results"]["compliance_findings"].append({
                            "framework": task_data.get("framework", "unknown"),
                            "score": task_data["compliance_score"],
                            "controls_passed": task_data.get("controls_passed", 0),
                            "controls_failed": task_data.get("controls_failed", 0)
                        })

            # Calculate security metrics
            results["scan_results"]["security_metrics"] = {
                "total_vulnerabilities": total_vulnerabilities,
                "critical_vulnerabilities": critical_vulns,
                "high_vulnerabilities": high_vulns,
                "unique_open_ports": len(all_open_ports),
                "services_identified": len(all_services),
                "scan_coverage": len(execution.completed_tasks) / len(execution.results) if execution.results else 0
            }

            # Calculate risk score
            risk_score = self._calculate_session_risk_score(results["scan_results"]["security_metrics"])
            results["threat_intelligence"]["risk_score"] = risk_score

            # Determine threat level
            if risk_score >= 0.8:
                threat_level = "critical"
            elif risk_score >= 0.6:
                threat_level = "high"
            elif risk_score >= 0.4:
                threat_level = "medium"
            else:
                threat_level = "low"

            results["threat_intelligence"]["threat_level"] = threat_level

            # Generate recommendations
            results["recommendations"] = self._generate_session_recommendations(
                results["scan_results"]["security_metrics"],
                threat_level
            )

            # Generate executive summary
            results["executive_summary"] = self._generate_executive_summary(
                results["scan_results"]["security_metrics"],
                results["threat_intelligence"],
                session
            )

            return results

        except Exception as e:
            logger.error(f"Failed to compile session results: {e}")
            return {
                "session_summary": {
                    "session_id": session.session_id,
                    "error": "Failed to compile results"
                },
                "scan_results": {"error": str(e)},
                "threat_intelligence": {"error": str(e)},
                "recommendations": ["Manual review required due to compilation error"]
            }

    def _calculate_session_risk_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate comprehensive risk score for session"""
        try:
            total_vulns = metrics.get("total_vulnerabilities", 0)
            critical_vulns = metrics.get("critical_vulnerabilities", 0)
            high_vulns = metrics.get("high_vulnerabilities", 0)

            # Base score from vulnerability counts
            base_score = min(total_vulns / 20.0, 1.0)  # Normalize to 0-1

            # Critical vulnerability multiplier
            critical_multiplier = critical_vulns * 0.3

            # High vulnerability multiplier
            high_multiplier = high_vulns * 0.15

            # Calculate final risk score
            risk_score = min(base_score + critical_multiplier + high_multiplier, 1.0)

            return round(risk_score, 3)

        except Exception as e:
            logger.error(f"Risk score calculation failed: {e}")
            return 0.5

    def _generate_session_recommendations(
        self,
        metrics: Dict[str, Any],
        threat_level: str
    ) -> List[str]:
        """Generate actionable recommendations based on scan results"""
        recommendations = []

        critical_vulns = metrics.get("critical_vulnerabilities", 0)
        high_vulns = metrics.get("high_vulnerabilities", 0)
        total_vulns = metrics.get("total_vulnerabilities", 0)

        # Critical findings recommendations
        if critical_vulns > 0:
            recommendations.append(
                f"🚨 URGENT: Address {critical_vulns} critical vulnerabilities immediately"
            )
            recommendations.append(
                "🔒 Consider implementing emergency incident response procedures"
            )

        # High severity recommendations
        if high_vulns > 0:
            recommendations.append(
                f"⚠️ HIGH PRIORITY: Remediate {high_vulns} high-severity vulnerabilities within 48 hours"
            )

        # Threat level specific recommendations
        if threat_level == "critical":
            recommendations.extend([
                "🛡️ Implement immediate containment measures",
                "📞 Activate incident response team",
                "🔍 Conduct forensic analysis",
                "📋 Notify relevant stakeholders"
            ])
        elif threat_level == "high":
            recommendations.extend([
                "🔍 Enhance monitoring and alerting",
                "🔧 Expedite patch management processes",
                "📊 Conduct additional targeted scans"
            ])
        elif threat_level == "medium":
            recommendations.extend([
                "📈 Continue regular vulnerability scanning",
                "🔄 Update security policies and procedures",
                "👥 Provide additional security training"
            ])

        # General security improvements
        if total_vulns > 0:
            recommendations.extend([
                "🔧 Implement automated patch management",
                "🛡️ Deploy additional security controls",
                "📝 Update security documentation",
                "🎯 Conduct focused security testing"
            ])

        # Always include these best practices
        recommendations.extend([
            "📊 Schedule regular security assessments",
            "🔍 Implement continuous monitoring",
            "👥 Provide ongoing security awareness training",
            "📋 Review and update incident response procedures"
        ])

        return recommendations[:10]  # Limit to top 10 recommendations

    def _generate_executive_summary(
        self,
        metrics: Dict[str, Any],
        threat_intel: Dict[str, Any],
        session: Any
    ) -> Dict[str, Any]:
        """Generate executive summary for leadership"""

        total_vulns = metrics.get("total_vulnerabilities", 0)
        critical_vulns = metrics.get("critical_vulnerabilities", 0)
        risk_score = threat_intel.get("risk_score", 0)
        threat_level = threat_intel.get("threat_level", "low")

        # Overall security posture assessment
        if risk_score >= 0.8:
            posture = "Critical - Immediate attention required"
            color = "red"
        elif risk_score >= 0.6:
            posture = "High Risk - Action needed within 48 hours"
            color = "orange"
        elif risk_score >= 0.4:
            posture = "Medium Risk - Monitor and remediate"
            color = "yellow"
        else:
            posture = "Low Risk - Maintain current security measures"
            color = "green"

        # Business impact assessment
        if critical_vulns > 0:
            business_impact = "High - Critical vulnerabilities pose immediate risk to business operations"
        elif total_vulns > 10:
            business_impact = "Medium - Multiple vulnerabilities require attention to prevent business disruption"
        elif total_vulns > 0:
            business_impact = "Low - Identified issues are manageable with standard remediation processes"
        else:
            business_impact = "Minimal - No significant security issues identified"

        # Cost/resource estimation
        if critical_vulns > 0:
            estimated_effort = "High - Requires immediate resource allocation and potential external assistance"
        elif total_vulns > 5:
            estimated_effort = "Medium - Standard security team resources with prioritized scheduling"
        else:
            estimated_effort = "Low - Regular maintenance windows and standard processes"

        summary = {
            "scan_overview": {
                "scan_type": session.scan_type,
                "targets_assessed": len(session.targets),
                "completion_status": "completed" if hasattr(session, 'completed_at') and session.completed_at else "partial"
            },
            "security_posture": {
                "overall_rating": posture,
                "risk_score": risk_score,
                "threat_level": threat_level,
                "indicator_color": color
            },
            "key_findings": {
                "total_vulnerabilities": total_vulns,
                "critical_issues": critical_vulns,
                "high_priority_issues": metrics.get("high_vulnerabilities", 0),
                "systems_affected": len(session.targets)
            },
            "business_impact": {
                "assessment": business_impact,
                "estimated_effort": estimated_effort,
                "urgency": "immediate" if critical_vulns > 0 else "normal"
            },
            "next_steps": {
                "immediate_actions": 2 if critical_vulns > 0 else 0,
                "short_term_actions": max(3, total_vulns // 3),
                "long_term_improvements": 5
            }
        }

        return summary

    async def _save_session_state(self, session: Any):
        """Save session state to persistent storage"""
        try:
            if hasattr(self, 'redis_client') and self.redis_client:
                session_data = {
                    "session_id": session.session_id,
                    "status": session.status.value,
                    "created_at": session.created_at.isoformat(),
                    "targets": [asdict(target) for target in session.targets],
                    "scan_type": session.scan_type,
                    "metadata": session.metadata
                }

                await self.redis_client.hset(
                    f"ptaas_session:{session.session_id}",
                    mapping={k: json.dumps(v) if not isinstance(v, str) else v
                            for k, v in session_data.items()}
                )

                # Set expiration (7 days)
                await self.redis_client.expire(f"ptaas_session:{session.session_id}", 604800)

        except Exception as e:
            logger.error(f"Failed to save session state: {e}")

    async def _load_persistent_state(self):
        """Load persistent state from storage"""
        try:
            if hasattr(self, 'redis_client') and self.redis_client:
                # Load session states
                session_keys = await self.redis_client.keys("ptaas_session:*")
                for key in session_keys:
                    try:
                        session_data = await self.redis_client.hgetall(key)
                        if session_data:
                            session_id = session_data.get("session_id")
                            if session_id and session_id not in self.sessions:
                                # Reconstruct session object (simplified)
                                self.sessions[session_id] = type('Session', (), session_data)()
                    except Exception as e:
                        logger.error(f"Failed to load session {key}: {e}")

                logger.info(f"Loaded {len(self.sessions)} sessions from persistent storage")

        except Exception as e:
            logger.error(f"Failed to load persistent state: {e}")

    async def _save_persistent_state(self):
        """Save current state to persistent storage"""
        try:
            if hasattr(self, 'redis_client') and self.redis_client:
                # Save current sessions
                for session_id, session in self.sessions.items():
                    await self._save_session_state(session)

                logger.info(f"Saved {len(self.sessions)} sessions to persistent storage")

        except Exception as e:
            logger.error(f"Failed to save persistent state: {e}")

    def _get_compliance_requirements(self, framework: str) -> Dict[str, Any]:
        """Get detailed compliance requirements for framework"""
        compliance_configs = self._load_compliance_configs()
        return compliance_configs.get(framework, {})

    async def _validate_pci_dss_compliance(
        self,
        vulnerabilities: List[Dict],
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate PCI-DSS compliance based on scan results"""

        compliance_score = 85.0  # Base score
        findings = []

        # Check for critical vulnerabilities that affect PCI compliance
        critical_vulns = [v for v in vulnerabilities if v.get("severity") == "critical"]
        if critical_vulns:
            compliance_score -= len(critical_vulns) * 10.0
            findings.append({
                "control": "General",
                "status": "failed",
                "issue": f"{len(critical_vulns)} critical vulnerabilities found",
                "impact": "High"
            })

        # Check network security (Requirement 1)
        network_vulns = [v for v in vulnerabilities
                        if any(keyword in v.get("name", "").lower()
                              for keyword in ["firewall", "network", "segmentation"])]
        if network_vulns:
            compliance_score -= 5.0
            findings.append({
                "control": "1.1-1.3",
                "status": "failed",
                "issue": "Network security vulnerabilities identified",
                "impact": "Medium"
            })

        # Check for encryption issues (Requirement 3-4)
        crypto_vulns = [v for v in vulnerabilities
                       if any(keyword in v.get("name", "").lower()
                             for keyword in ["ssl", "tls", "encryption", "crypto"])]
        if crypto_vulns:
            compliance_score -= 8.0
            findings.append({
                "control": "3.4-4.1",
                "status": "failed",
                "issue": "Encryption/cryptography vulnerabilities found",
                "impact": "High"
            })

        return {
            "compliance_score": max(compliance_score, 0.0),
            "findings": findings,
            "status": "compliant" if compliance_score >= 80.0 else "non_compliant"
        }

    async def _validate_hipaa_compliance(
        self,
        vulnerabilities: List[Dict],
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate HIPAA compliance based on scan results"""

        compliance_score = 80.0  # Base score
        findings = []

        # Check access control vulnerabilities
        access_vulns = [v for v in vulnerabilities
                       if any(keyword in v.get("name", "").lower()
                             for keyword in ["authentication", "authorization", "access"])]
        if access_vulns:
            compliance_score -= len(access_vulns) * 7.0
            findings.append({
                "control": "164.312(a)(1)",
                "status": "failed",
                "issue": "Access control vulnerabilities identified"
            })

        # Check encryption requirements
        crypto_vulns = [v for v in vulnerabilities
                       if any(keyword in v.get("name", "").lower()
                             for keyword in ["encryption", "ssl", "tls"])]
        if crypto_vulns:
            compliance_score -= 10.0
            findings.append({
                "control": "164.312(e)",
                "status": "failed",
                "issue": "Transmission security vulnerabilities found"
            })

        return {
            "compliance_score": max(compliance_score, 0.0),
            "findings": findings,
            "status": "compliant" if compliance_score >= 75.0 else "non_compliant"
        }

    async def _validate_sox_compliance(
        self,
        vulnerabilities: List[Dict],
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate SOX compliance based on scan results"""

        compliance_score = 90.0  # Base score
        findings = []

        # Check IT control vulnerabilities
        it_vulns = [v for v in vulnerabilities
                   if any(keyword in v.get("name", "").lower()
                         for keyword in ["access", "change", "configuration"])]
        if it_vulns:
            compliance_score -= len(it_vulns) * 5.0
            findings.append({
                "control": "SOX.404",
                "status": "failed",
                "issue": "IT control vulnerabilities identified"
            })

        return {
            "compliance_score": max(compliance_score, 0.0),
            "findings": findings,
            "status": "compliant" if compliance_score >= 85.0 else "non_compliant"
        }

    async def start_scan_session(self, session_id: str):
        """Start executing a scan session"""
        try:
            if session_id not in self.sessions:
                logger.error(f"Session {session_id} not found")
                return

            session = self.sessions[session_id]
            execution = self.executions.get(session.workflow_execution_id)

            if not execution:
                logger.error(f"Execution not found for session {session_id}")
                return

            # Update session status
            session.status = self.WorkflowStatus.RUNNING
            session.started_at = datetime.utcnow()

            # Queue execution task
            await self.task_queue.put({
                "type": "execute_workflow",
                "execution_id": execution.id,
                "session_id": session_id
            })

            logger.info(f"Started scan session {session_id}")

        except Exception as e:
            logger.error(f"Failed to start scan session {session_id}: {e}")

    async def get_scan_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive scan session status"""
        try:
            if session_id not in self.sessions:
                return None

            session = self.sessions[session_id]
            execution = self.executions.get(session.workflow_execution_id)

            if not execution:
                return {
                    "session_id": session_id,
                    "status": "error",
                    "error": "Execution not found"
                }

            status_data = {
                "session_id": session_id,
                "status": execution.status.value,
                "scan_type": session.scan_type,
                "targets_count": len(session.targets),
                "created_at": session.created_at.isoformat(),
                "started_at": session.started_at.isoformat() if session.started_at else None,
                "completed_at": session.completed_at.isoformat() if session.completed_at else None,
                "progress_percentage": execution.progress_percentage,
                "current_task": execution.current_task,
                "completed_tasks": len(execution.completed_tasks),
                "failed_tasks": len(execution.failed_tasks),
                "errors": execution.errors
            }

            # Add results if completed
            if execution.status.value == "completed" and hasattr(execution, 'results'):
                status_data["results"] = await self._compile_session_results(session, execution)

            return status_data

        except Exception as e:
            logger.error(f"Failed to get session status {session_id}: {e}")
            return {
                "session_id": session_id,
                "status": "error",
                "error": str(e)
            }

    async def cancel_scan_session(self, session_id: str) -> bool:
        """Cancel an active scan session"""
        try:
            if session_id not in self.sessions:
                return False

            session = self.sessions[session_id]
            execution = self.executions.get(session.workflow_execution_id)

            if not execution:
                return False

            # Cancel execution
            if session.workflow_execution_id in self.active_executions:
                task = self.active_executions[session.workflow_execution_id]
                task.cancel()
                self.active_executions.pop(session.workflow_execution_id)

            # Update status
            execution.status = self.WorkflowStatus.CANCELLED
            execution.end_time = datetime.utcnow()
            session.status = self.WorkflowStatus.CANCELLED
            session.completed_at = datetime.utcnow()

            # Save state
            await self._save_session_state(session)

            logger.info(f"Cancelled scan session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel session {session_id}: {e}")
            return False

    @property
    def available_scanners(self) -> Dict[str, bool]:
        """Get available scanner status"""
        if hasattr(self, 'scanner_service') and self.scanner_service:
            return {
                name: config.available
                for name, config in self.scanner_service.scanners.items()
            }
        return {}

    @property
    def active_sessions(self) -> Dict[str, Any]:
        """Get active sessions"""
        return getattr(self, 'sessions', {})

    @property
    def WorkflowStatus(self):
        """Get WorkflowStatus enum"""
        from . import WorkflowStatus
        return WorkflowStatus

    async def _workflow_worker(self, worker_id: str):
        """Background worker for processing workflows"""
        logger.info(f"Started workflow worker: {worker_id}")

        while True:
            try:
                # Get task from queue
                task = await self.task_queue.get()

                if task.get("type") == "execute_workflow":
                    execution_id = task.get("execution_id")
                    session_id = task.get("session_id")

                    if execution_id in self.executions:
                        execution = self.executions[execution_id]
                        workflow = self.workflows.get(execution.workflow_id)

                        if workflow:
                            # Execute workflow
                            await self._execute_workflow_with_monitoring(
                                execution, workflow, session_id
                            )
                        else:
                            logger.error(f"Workflow not found: {execution.workflow_id}")
                    else:
                        logger.error(f"Execution not found: {execution_id}")

                self.task_queue.task_done()

            except asyncio.CancelledError:
                logger.info(f"Workflow worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Workflow worker {worker_id} error: {e}")
                await asyncio.sleep(1)

    async def _execute_workflow_with_monitoring(
        self,
        execution: Any,
        workflow: Any,
        session_id: Optional[str]
    ):
        """Execute workflow with comprehensive monitoring"""
        try:
            execution.status = self.WorkflowStatus.RUNNING
            execution.start_time = datetime.utcnow()
            execution.current_task = "initializing"

            # Simulate workflow execution phases
            phases = ["initialization", "target_validation", "scanning", "analysis", "reporting"]

            for i, phase in enumerate(phases):
                execution.current_task = phase
                execution.progress_percentage = (i / len(phases)) * 100

                # Simulate phase execution
                await asyncio.sleep(1)  # Simulate work

                execution.completed_tasks.append(phase)

                logger.debug(f"Workflow {execution.id} completed phase: {phase}")

            # Mark as completed
            execution.status = self.WorkflowStatus.COMPLETED
            execution.end_time = datetime.utcnow()
            execution.progress_percentage = 100.0
            execution.current_task = "completed"

            # Update session if available
            if session_id and session_id in self.sessions:
                session = self.sessions[session_id]
                session.status = self.WorkflowStatus.COMPLETED
                session.completed_at = datetime.utcnow()
                await self._save_session_state(session)

            logger.info(f"Workflow execution {execution.id} completed successfully")

        except Exception as e:
            execution.status = self.WorkflowStatus.FAILED
            execution.end_time = datetime.utcnow()
            execution.errors.append(str(e))

            if session_id and session_id in self.sessions:
                session = self.sessions[session_id]
                session.status = self.WorkflowStatus.FAILED
                session.completed_at = datetime.utcnow()

            logger.error(f"Workflow execution {execution.id} failed: {e}")
