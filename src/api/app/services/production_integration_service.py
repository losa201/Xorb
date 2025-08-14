"""
Production Integration Service
Integrates all production services into a cohesive security platform
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from .production_service_implementations import (
    ProductionAuthenticationService,
    ProductionPTaaSService,
    ProductionThreatIntelligenceService,
    ProductionNotificationService,
    ProductionHealthCheckService
)
from .production_orchestration_engine import ProductionOrchestrationEngine
from .production_ai_threat_intelligence_engine import ProductionAIThreatIntelligence
from .production_ptaas_scanner_implementation import ProductionPTaaSScanner

logger = logging.getLogger(__name__)

@dataclass
class PlatformConfig:
    """Configuration for the integrated security platform"""
    jwt_secret: str
    database_url: str
    redis_url: str
    smtp_config: Dict[str, Any]
    notification_config: Dict[str, Any]
    scanner_config: Dict[str, Any]
    ai_config: Dict[str, Any]
    orchestration_config: Dict[str, Any]

class ProductionSecurityPlatform:
    """Integrated production security platform"""

    def __init__(self, config: PlatformConfig):
        self.config = config
        self.services = {}
        self.health_status = {"status": "initializing"}

        # Initialize all services
        asyncio.create_task(self._initialize_platform())

    async def _initialize_platform(self):
        """Initialize all platform services"""

        try:
            logger.info("🚀 Starting XORB Production Security Platform initialization...")

            # Initialize authentication service
            logger.info("📝 Initializing authentication service...")
            self.services["authentication"] = ProductionAuthenticationService({
                "jwt_secret": self.config.jwt_secret,
                "token_expiry": 3600
            })

            # Initialize PTaaS service
            logger.info("🔒 Initializing PTaaS service...")
            self.services["ptaas"] = ProductionPTaaSService({
                "scanner_timeout": 300,
                "max_concurrent_scans": 5
            })

            # Initialize threat intelligence service
            logger.info("🧠 Initializing threat intelligence service...")
            self.services["threat_intelligence"] = ProductionThreatIntelligenceService({
                "update_interval": 3600,
                "correlation_window": 1800
            })

            # Initialize AI threat intelligence engine
            logger.info("🤖 Initializing AI threat intelligence engine...")
            self.services["ai_threat_intel"] = ProductionAIThreatIntelligence(
                self.config.ai_config
            )

            # Initialize notification service
            logger.info("📧 Initializing notification service...")
            self.services["notifications"] = ProductionNotificationService(
                self.config.notification_config
            )

            # Initialize orchestration engine
            logger.info("⚙️ Initializing orchestration engine...")
            self.services["orchestration"] = ProductionOrchestrationEngine(
                self.config.orchestration_config
            )

            # Initialize health check service
            logger.info("🏥 Initializing health check service...")
            self.services["health"] = ProductionHealthCheckService({
                "dependencies": ["database", "redis", "external_apis"]
            })

            # Initialize PTaaS scanner
            logger.info("🔍 Initializing production scanner...")
            self.services["scanner"] = ProductionPTaaSScanner(
                self.config.scanner_config
            )

            # Perform initial health check
            await self._perform_health_check()

            # Setup inter-service integrations
            await self._setup_service_integrations()

            # Register default workflows
            await self._register_default_workflows()

            self.health_status = {"status": "healthy", "initialized_at": datetime.utcnow()}
            logger.info("✅ XORB Production Security Platform initialized successfully!")

        except Exception as e:
            logger.error(f"❌ Platform initialization failed: {e}")
            self.health_status = {"status": "failed", "error": str(e)}
            raise

    async def _perform_health_check(self):
        """Perform comprehensive health check of all services"""

        logger.info("🏥 Performing platform health check...")

        health_service = self.services["health"]
        overall_health = await health_service.get_system_health()

        logger.info(f"Health check result: {overall_health['overall_status']}")

        for service_name, health_info in overall_health.get("services", {}).items():
            status = health_info.get("status", "unknown")
            if status == "healthy":
                logger.info(f"✅ {service_name}: {status}")
            else:
                logger.warning(f"⚠️ {service_name}: {status}")

    async def _setup_service_integrations(self):
        """Setup integrations between services"""

        logger.info("🔗 Setting up service integrations...")

        # Integrate PTaaS with threat intelligence
        await self._integrate_ptaas_with_threat_intel()

        # Integrate orchestration with notifications
        await self._integrate_orchestration_with_notifications()

        # Setup AI-powered scan enhancement
        await self._setup_ai_scan_enhancement()

        logger.info("✅ Service integrations configured")

    async def _integrate_ptaas_with_threat_intel(self):
        """Integrate PTaaS scans with threat intelligence"""

        ptaas_service = self.services["ptaas"]
        threat_intel = self.services["ai_threat_intel"]

        # Register post-scan analysis callback
        async def analyze_scan_results(session_id: str, results: Dict[str, Any]):
            try:
                # Extract indicators from scan results
                indicators = self._extract_indicators_from_scan(results)

                if indicators:
                    # Analyze with AI threat intelligence
                    threat_analysis = await threat_intel.analyze_threat_indicators(indicators)

                    # Enhance scan results with threat intelligence
                    results["threat_intelligence"] = {
                        "indicators_analyzed": len(indicators),
                        "threats_identified": len(threat_analysis),
                        "analysis_timestamp": datetime.utcnow().isoformat(),
                        "threat_indicators": [asdict(ti) for ti in threat_analysis]
                    }

                    logger.info(f"Enhanced scan {session_id} with threat intelligence")

            except Exception as e:
                logger.error(f"Threat intelligence analysis failed for scan {session_id}: {e}")

        # Store the callback for use in PTaaS scans
        ptaas_service.post_scan_callback = analyze_scan_results

    async def _integrate_orchestration_with_notifications(self):
        """Integrate orchestration engine with notification service"""

        orchestration = self.services["orchestration"]
        notifications = self.services["notifications"]

        # Register notification task handler
        async def send_notification_task(parameters: Dict[str, Any], context: Dict[str, Any]):
            recipient = parameters.get("recipient")
            message = parameters.get("message")
            channel = parameters.get("channel", "email")
            priority = parameters.get("priority", "normal")

            if recipient and message:
                success = await notifications.send_notification(recipient, message, channel, priority)
                return {"notification_sent": success, "recipient": recipient}
            else:
                raise ValueError("Missing recipient or message")

        orchestration.register_task_handler("send_notification", send_notification_task)

        # Register webhook task handler
        async def send_webhook_task(parameters: Dict[str, Any], context: Dict[str, Any]):
            url = parameters.get("url")
            payload = parameters.get("payload", {})

            if url:
                success = await notifications.send_webhook(url, payload)
                return {"webhook_sent": success, "url": url}
            else:
                raise ValueError("Missing webhook URL")

        orchestration.register_task_handler("send_webhook", send_webhook_task)

    async def _setup_ai_scan_enhancement(self):
        """Setup AI-powered scan enhancement"""

        scanner = self.services["scanner"]
        ai_threat_intel = self.services["ai_threat_intel"]

        # Enhance scanner with AI capabilities
        original_execute_scan = scanner.execute_comprehensive_scan

        async def enhanced_execute_scan(config):
            # Execute original scan
            results = await original_execute_scan(config)

            # Enhance with AI analysis
            try:
                indicators = self._extract_indicators_from_scan(results)
                if indicators:
                    threat_analysis = await ai_threat_intel.analyze_threat_indicators(indicators)

                    # Add AI insights to results
                    results["ai_enhancement"] = {
                        "threat_score": self._calculate_ai_threat_score(threat_analysis),
                        "risk_prediction": await ai_threat_intel.predict_threat_evolution(
                            self._create_threat_assessment(threat_analysis, results),
                            24  # 24-hour prediction
                        ),
                        "enhanced_recommendations": self._generate_ai_recommendations(threat_analysis)
                    }

            except Exception as e:
                logger.error(f"AI scan enhancement failed: {e}")

            return results

        # Replace the original method
        scanner.execute_comprehensive_scan = enhanced_execute_scan

    async def _register_default_workflows(self):
        """Register default security workflows"""

        orchestration = self.services["orchestration"]

        # Continuous monitoring workflow
        continuous_monitoring_workflow = {
            "name": "Continuous Security Monitoring",
            "description": "Automated continuous security monitoring and alerting",
            "tasks": [
                {
                    "task_id": "health_check",
                    "name": "System Health Check",
                    "task_type": "health_check",
                    "parameters": {"check_all_services": True}
                },
                {
                    "task_id": "threat_scan",
                    "name": "Threat Intelligence Scan",
                    "task_type": "threat_scan",
                    "parameters": {"scan_feeds": True}
                },
                {
                    "task_id": "alert_analysis",
                    "name": "Security Alert Analysis",
                    "task_type": "alert_analysis",
                    "dependencies": ["threat_scan"]
                },
                {
                    "task_id": "send_alerts",
                    "name": "Send Critical Alerts",
                    "task_type": "conditional_notification",
                    "dependencies": ["alert_analysis"],
                    "parameters": {
                        "condition": "critical_threats_found",
                        "recipients": ["security-team@company.com"]
                    }
                }
            ],
            "triggers": [
                {
                    "type": "scheduled",
                    "schedule": "0 * * * *",  # Every hour
                    "enabled": True
                }
            ]
        }

        await orchestration.create_workflow(continuous_monitoring_workflow)

        # Incident response workflow
        incident_response_workflow = {
            "name": "Automated Incident Response",
            "description": "Automated response to security incidents",
            "tasks": [
                {
                    "task_id": "analyze_incident",
                    "name": "Analyze Security Incident",
                    "task_type": "incident_analysis",
                    "parameters": {"deep_analysis": True}
                },
                {
                    "task_id": "containment",
                    "name": "Implement Containment",
                    "task_type": "incident_containment",
                    "dependencies": ["analyze_incident"]
                },
                {
                    "task_id": "notify_team",
                    "name": "Notify Security Team",
                    "task_type": "send_notification",
                    "dependencies": ["containment"],
                    "parameters": {
                        "recipient": "security-incident@company.com",
                        "channel": "email",
                        "priority": "critical"
                    }
                },
                {
                    "task_id": "generate_report",
                    "name": "Generate Incident Report",
                    "task_type": "report_generation",
                    "dependencies": ["notify_team"]
                }
            ]
        }

        await orchestration.create_workflow(incident_response_workflow)

        # Compliance monitoring workflow
        compliance_workflow = {
            "name": "Automated Compliance Monitoring",
            "description": "Continuous compliance monitoring and reporting",
            "tasks": [
                {
                    "task_id": "compliance_scan",
                    "name": "Run Compliance Scans",
                    "task_type": "compliance_scan",
                    "parameters": {
                        "frameworks": ["PCI-DSS", "HIPAA", "SOX", "ISO-27001"]
                    }
                },
                {
                    "task_id": "analyze_gaps",
                    "name": "Analyze Compliance Gaps",
                    "task_type": "compliance_analysis",
                    "dependencies": ["compliance_scan"]
                },
                {
                    "task_id": "generate_compliance_report",
                    "name": "Generate Compliance Report",
                    "task_type": "compliance_reporting",
                    "dependencies": ["analyze_gaps"]
                },
                {
                    "task_id": "notify_compliance_team",
                    "name": "Notify Compliance Team",
                    "task_type": "send_notification",
                    "dependencies": ["generate_compliance_report"],
                    "parameters": {
                        "recipient": "compliance@company.com",
                        "channel": "email"
                    }
                }
            ],
            "triggers": [
                {
                    "type": "scheduled",
                    "schedule": "0 6 * * 1",  # Every Monday at 6 AM
                    "enabled": True
                }
            ]
        }

        await orchestration.create_workflow(compliance_workflow)

        logger.info("✅ Default security workflows registered")

    def _extract_indicators_from_scan(self, scan_results: Dict[str, Any]) -> List[str]:
        """Extract threat indicators from scan results"""

        indicators = []

        # Extract from vulnerabilities
        for vuln in scan_results.get("vulnerabilities", []):
            if isinstance(vuln, dict):
                component = vuln.get("affected_component", "")
                if component and "." in component:
                    indicators.append(component)

        # Extract from services
        for service in scan_results.get("services", []):
            if isinstance(service, dict):
                host = service.get("host") or service.get("ip")
                if host:
                    indicators.append(host)

        # Extract from network analysis
        network_info = scan_results.get("network_analysis", {})
        if "external_connections" in network_info:
            indicators.extend(network_info["external_connections"])

        return list(set(indicators))  # Remove duplicates

    def _calculate_ai_threat_score(self, threat_analysis: List[Any]) -> float:
        """Calculate AI-based threat score"""

        if not threat_analysis:
            return 0.0

        total_score = 0.0
        for threat in threat_analysis:
            confidence = getattr(threat, 'confidence', 0.5)
            severity_weights = {
                "critical": 1.0,
                "high": 0.8,
                "medium": 0.5,
                "low": 0.2
            }
            severity = getattr(threat, 'severity', 'low').lower()
            weight = severity_weights.get(severity, 0.2)

            total_score += confidence * weight

        # Normalize to 0-10 scale
        normalized_score = (total_score / len(threat_analysis)) * 10
        return min(normalized_score, 10.0)

    def _create_threat_assessment(self, threat_analysis: List[Any], scan_results: Dict[str, Any]) -> Any:
        """Create threat assessment for prediction"""

        # This would create a proper ThreatAssessment object
        # For now, return a mock assessment
        class MockThreatAssessment:
            def __init__(self):
                self.threat_id = "mock_threat"
                self.threat_type = "security_scan_findings"
                self.severity_score = self._calculate_ai_threat_score(threat_analysis) if hasattr(self, '_calculate_ai_threat_score') else 5.0

        return MockThreatAssessment()

    def _generate_ai_recommendations(self, threat_analysis: List[Any]) -> List[str]:
        """Generate AI-powered security recommendations"""

        recommendations = []

        if not threat_analysis:
            return ["No specific threats detected - maintain current security posture"]

        # Analyze threat patterns
        critical_threats = [t for t in threat_analysis if getattr(t, 'severity', 'low').lower() == 'critical']
        high_threats = [t for t in threat_analysis if getattr(t, 'severity', 'low').lower() == 'high']

        if critical_threats:
            recommendations.append("⚠️ URGENT: Address critical security threats immediately")
            recommendations.append("🔒 Implement emergency access controls")
            recommendations.append("📧 Notify incident response team")

        if high_threats:
            recommendations.append("🎯 Prioritize remediation of high-severity vulnerabilities")
            recommendations.append("🔍 Increase monitoring frequency")

        if len(threat_analysis) > 5:
            recommendations.append("🔄 Consider comprehensive security audit")
            recommendations.append("📊 Review security policies and procedures")

        # AI-specific recommendations
        recommendations.extend([
            "🤖 Enable continuous AI-powered threat monitoring",
            "📈 Implement predictive threat modeling",
            "🔗 Integrate threat intelligence feeds",
            "🛡️ Deploy automated response mechanisms"
        ])

        return recommendations

    async def get_platform_status(self) -> Dict[str, Any]:
        """Get comprehensive platform status"""

        platform_status = {
            "platform": {
                "name": "XORB Production Security Platform",
                "version": "3.0.0",
                "status": self.health_status.get("status", "unknown"),
                "initialized_at": self.health_status.get("initialized_at", "unknown")
            },
            "services": {},
            "integrations": {
                "ptaas_threat_intel": True,
                "orchestration_notifications": True,
                "ai_scan_enhancement": True
            },
            "capabilities": [
                "Real-world security scanning",
                "AI-powered threat intelligence",
                "Automated workflow orchestration",
                "Compliance monitoring",
                "Incident response automation",
                "Predictive threat modeling"
            ]
        }

        # Get individual service status
        for service_name, service in self.services.items():
            try:
                if hasattr(service, 'get_status'):
                    status = await service.get_status()
                elif hasattr(service, 'health_check'):
                    status = await service.health_check()
                else:
                    status = {"status": "available"}

                platform_status["services"][service_name] = status

            except Exception as e:
                platform_status["services"][service_name] = {
                    "status": "error",
                    "error": str(e)
                }

        return platform_status

    async def execute_security_scan(
        self,
        targets: List[str],
        scan_type: str = "comprehensive",
        compliance_framework: Optional[str] = None
    ) -> str:
        """Execute comprehensive security scan with all platform capabilities"""

        try:
            # Create enhanced scan request
            scan_request = {
                "targets": [{"host": target} for target in targets],
                "scan_type": scan_type,
                "compliance_framework": compliance_framework,
                "ai_enhancement": True,
                "threat_intelligence": True
            }

            # Execute via PTaaS service
            ptaas_service = self.services["ptaas"]
            session_id = await ptaas_service.create_scan_session(
                targets=targets,
                scan_type=scan_type,
                tenant_id="platform",
                metadata=scan_request
            )

            logger.info(f"🔍 Initiated enhanced security scan {session_id}")
            return session_id

        except Exception as e:
            logger.error(f"Failed to execute security scan: {e}")
            raise

    async def analyze_threat_indicators(self, indicators: List[str]) -> Dict[str, Any]:
        """Analyze threat indicators with full platform capabilities"""

        try:
            # Use AI threat intelligence
            ai_threat_intel = self.services["ai_threat_intel"]
            threat_analysis = await ai_threat_intel.analyze_threat_indicators(indicators)

            # Use traditional threat intelligence
            threat_intel = self.services["threat_intelligence"]
            traditional_analysis = await threat_intel.analyze_indicators(indicators)

            # Combine results
            combined_analysis = {
                "ai_analysis": [asdict(analysis) for analysis in threat_analysis],
                "traditional_analysis": traditional_analysis,
                "combined_threat_score": self._calculate_ai_threat_score(threat_analysis),
                "recommendations": self._generate_ai_recommendations(threat_analysis),
                "analysis_timestamp": datetime.utcnow().isoformat()
            }

            logger.info(f"🧠 Analyzed {len(indicators)} threat indicators")
            return combined_analysis

        except Exception as e:
            logger.error(f"Threat indicator analysis failed: {e}")
            raise

    async def create_security_workflow(self, workflow_definition: Dict[str, Any]) -> str:
        """Create custom security workflow"""

        try:
            orchestration = self.services["orchestration"]
            workflow_id = await orchestration.create_workflow(workflow_definition)

            logger.info(f"⚙️ Created security workflow {workflow_id}")
            return workflow_id

        except Exception as e:
            logger.error(f"Failed to create security workflow: {e}")
            raise

    async def shutdown(self):
        """Gracefully shutdown the platform"""

        logger.info("🛑 Shutting down XORB Production Security Platform...")

        # Shutdown orchestration engine
        if "orchestration" in self.services:
            self.services["orchestration"].stop()

        # Close other service connections
        for service_name, service in self.services.items():
            try:
                if hasattr(service, 'shutdown'):
                    await service.shutdown()
                elif hasattr(service, 'close'):
                    await service.close()
            except Exception as e:
                logger.warning(f"Failed to shutdown {service_name}: {e}")

        logger.info("✅ Platform shutdown complete")

# Global platform instance
_platform_instance: Optional[ProductionSecurityPlatform] = None

async def get_production_platform(config: Optional[PlatformConfig] = None) -> ProductionSecurityPlatform:
    """Get or create the production security platform instance"""

    global _platform_instance

    if _platform_instance is None:
        if config is None:
            # Use default configuration
            config = PlatformConfig(
                jwt_secret="production-jwt-secret-change-me",
                database_url="postgresql://localhost/xorb",
                redis_url="redis://localhost:6379",
                smtp_config={
                    "host": "localhost",
                    "port": 587,
                    "username": "",
                    "password": ""
                },
                notification_config={
                    "default_channel": "email",
                    "webhook_timeout": 30
                },
                scanner_config={
                    "timeout": 300,
                    "max_parallel_scans": 5
                },
                ai_config={
                    "model_path": "models/",
                    "enable_ml": True
                },
                orchestration_config={
                    "max_workers": 10,
                    "workflow_timeout": 3600
                }
            )

        _platform_instance = ProductionSecurityPlatform(config)

    return _platform_instance

async def shutdown_production_platform():
    """Shutdown the production security platform"""

    global _platform_instance

    if _platform_instance:
        await _platform_instance.shutdown()
        _platform_instance = None
