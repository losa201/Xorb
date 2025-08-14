#!/usr/bin/env python3
"""
XORB Automated Incident Response Orchestration System
Real-time incident detection, classification, and automated response workflows
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import redis.asyncio as redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IncidentSeverity(Enum):
    """Incident severity levels"""
    P4_LOW = 4
    P3_MEDIUM = 3
    P2_HIGH = 2
    P1_CRITICAL = 1

class IncidentStatus(Enum):
    """Incident status"""
    OPEN = "open"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    RESOLVED = "resolved"
    CLOSED = "closed"

class ResponseAction(Enum):
    """Automated response actions"""
    ISOLATE_HOST = "isolate_host"
    BLOCK_IP = "block_ip"
    DISABLE_USER = "disable_user"
    QUARANTINE_FILE = "quarantine_file"
    RESET_PASSWORD = "reset_password"
    ESCALATE_TO_SOC = "escalate_to_soc"
    NOTIFY_ADMIN = "notify_admin"
    COLLECT_FORENSICS = "collect_forensics"
    UPDATE_SIGNATURES = "update_signatures"

@dataclass
class SecurityIncident:
    """Security incident data structure"""
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    source_system: str
    affected_assets: List[str]
    indicators: List[str]
    evidence: Dict[str, Any]
    assigned_analyst: Optional[str] = None
    response_actions: List[str] = None
    timeline: List[Dict] = None

    def __post_init__(self):
        if self.response_actions is None:
            self.response_actions = []
        if self.timeline is None:
            self.timeline = []

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['severity'] = self.severity.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data

@dataclass
class ResponsePlaybook:
    """Incident response playbook definition"""
    playbook_id: str
    name: str
    description: str
    trigger_conditions: Dict[str, Any]
    response_actions: List[Dict[str, Any]]
    escalation_rules: Dict[str, Any]
    sla_minutes: int
    approval_required: bool = False

class IncidentResponseOrchestrator:
    """Advanced incident response orchestration system"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = None
        self.redis_url = redis_url
        self.active_incidents: Dict[str, SecurityIncident] = {}
        self.response_playbooks: Dict[str, ResponsePlaybook] = {}
        self.response_handlers: Dict[ResponseAction, Callable] = {}
        self.notification_channels: Dict[str, Dict] = {}

        # Initialize response handlers
        self._register_response_handlers()

        # Load playbooks
        self._load_response_playbooks()

        # Setup notification channels
        self._setup_notification_channels()

    async def initialize(self):
        """Initialize incident response orchestrator"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()

            # Load existing incidents from Redis
            await self._load_active_incidents()

            logger.info("Incident Response Orchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize incident response orchestrator: {e}")
            raise

    def _register_response_handlers(self):
        """Register automated response action handlers"""
        self.response_handlers = {
            ResponseAction.ISOLATE_HOST: self._isolate_host,
            ResponseAction.BLOCK_IP: self._block_ip_address,
            ResponseAction.DISABLE_USER: self._disable_user_account,
            ResponseAction.QUARANTINE_FILE: self._quarantine_file,
            ResponseAction.RESET_PASSWORD: self._reset_user_password,
            ResponseAction.ESCALATE_TO_SOC: self._escalate_to_soc,
            ResponseAction.NOTIFY_ADMIN: self._notify_administrators,
            ResponseAction.COLLECT_FORENSICS: self._collect_forensic_evidence,
            ResponseAction.UPDATE_SIGNATURES: self._update_security_signatures
        }

    def _load_response_playbooks(self):
        """Load incident response playbooks"""
        playbooks = [
            ResponsePlaybook(
                playbook_id="malware_detection",
                name="Malware Detection Response",
                description="Automated response to malware detection events",
                trigger_conditions={
                    "threat_category": "malware",
                    "confidence": 0.8
                },
                response_actions=[
                    {"action": ResponseAction.ISOLATE_HOST.value, "priority": 1},
                    {"action": ResponseAction.QUARANTINE_FILE.value, "priority": 2},
                    {"action": ResponseAction.COLLECT_FORENSICS.value, "priority": 3},
                    {"action": ResponseAction.NOTIFY_ADMIN.value, "priority": 4}
                ],
                escalation_rules={
                    "auto_escalate_after_minutes": 30,
                    "escalate_to": "soc_analyst"
                },
                sla_minutes=15,
                approval_required=False
            ),
            ResponsePlaybook(
                playbook_id="brute_force_attack",
                name="Brute Force Attack Response",
                description="Response to brute force login attempts",
                trigger_conditions={
                    "threat_category": "intrusion",
                    "pattern": "multiple_failed_logins"
                },
                response_actions=[
                    {"action": ResponseAction.BLOCK_IP.value, "priority": 1},
                    {"action": ResponseAction.DISABLE_USER.value, "priority": 2},
                    {"action": ResponseAction.RESET_PASSWORD.value, "priority": 3},
                    {"action": ResponseAction.NOTIFY_ADMIN.value, "priority": 4}
                ],
                escalation_rules={
                    "auto_escalate_after_minutes": 20,
                    "escalate_to": "security_team"
                },
                sla_minutes=10
            ),
            ResponsePlaybook(
                playbook_id="data_exfiltration",
                name="Data Exfiltration Response",
                description="Response to suspected data exfiltration",
                trigger_conditions={
                    "threat_category": "data_exfiltration",
                    "severity": "high"
                },
                response_actions=[
                    {"action": ResponseAction.ISOLATE_HOST.value, "priority": 1},
                    {"action": ResponseAction.BLOCK_IP.value, "priority": 1},
                    {"action": ResponseAction.COLLECT_FORENSICS.value, "priority": 2},
                    {"action": ResponseAction.ESCALATE_TO_SOC.value, "priority": 3}
                ],
                escalation_rules={
                    "auto_escalate_after_minutes": 5,
                    "escalate_to": "incident_commander"
                },
                sla_minutes=5,
                approval_required=True
            ),
            ResponsePlaybook(
                playbook_id="privilege_escalation",
                name="Privilege Escalation Response",
                description="Response to privilege escalation attempts",
                trigger_conditions={
                    "threat_category": "privilege_escalation",
                    "confidence": 0.7
                },
                response_actions=[
                    {"action": ResponseAction.DISABLE_USER.value, "priority": 1},
                    {"action": ResponseAction.RESET_PASSWORD.value, "priority": 2},
                    {"action": ResponseAction.COLLECT_FORENSICS.value, "priority": 3},
                    {"action": ResponseAction.NOTIFY_ADMIN.value, "priority": 4}
                ],
                escalation_rules={
                    "auto_escalate_after_minutes": 15,
                    "escalate_to": "security_lead"
                },
                sla_minutes=10
            )
        ]

        for playbook in playbooks:
            self.response_playbooks[playbook.playbook_id] = playbook

        logger.info(f"Loaded {len(playbooks)} incident response playbooks")

    def _setup_notification_channels(self):
        """Setup notification channels for incident response"""
        self.notification_channels = {
            "email": {
                "smtp_server": "localhost",
                "smtp_port": 587,
                "username": "xorb-alerts@company.com",
                "password": "secure_password",
                "recipients": [
                    "soc-team@company.com",
                    "security-admin@company.com",
                    "incident-response@company.com"
                ]
            },
            "slack": {
                "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
                "channel": "#security-alerts"
            },
            "pagerduty": {
                "service_key": "your-pagerduty-service-key",
                "api_url": "https://events.pagerduty.com/v2/enqueue"
            }
        }

    async def create_incident(self,
                            title: str,
                            description: str,
                            severity: IncidentSeverity,
                            source_system: str,
                            affected_assets: List[str],
                            indicators: List[str],
                            evidence: Dict[str, Any]) -> SecurityIncident:
        """Create new security incident and trigger automated response"""
        try:
            incident_id = str(uuid.uuid4())

            incident = SecurityIncident(
                incident_id=incident_id,
                title=title,
                description=description,
                severity=severity,
                status=IncidentStatus.OPEN,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                source_system=source_system,
                affected_assets=affected_assets,
                indicators=indicators,
                evidence=evidence
            )

            # Add creation event to timeline
            incident.timeline.append({
                "timestamp": incident.created_at.isoformat(),
                "action": "incident_created",
                "description": f"Incident created by {source_system}",
                "severity": severity.name
            })

            self.active_incidents[incident_id] = incident

            # Store in Redis
            await self.redis_client.setex(
                f"incident:{incident_id}",
                86400 * 7,  # 7 days
                json.dumps(incident.to_dict())
            )

            # Trigger automated response
            await self._trigger_automated_response(incident)

            logger.info(f"Created incident {incident_id}: {title}")
            return incident

        except Exception as e:
            logger.error(f"Failed to create incident: {e}")
            raise

    async def _trigger_automated_response(self, incident: SecurityIncident):
        """Trigger automated response based on incident characteristics"""
        try:
            # Find matching playbook
            playbook = self._find_matching_playbook(incident)

            if not playbook:
                logger.warning(f"No matching playbook found for incident {incident.incident_id}")
                # Default response - escalate to human analyst
                await self._escalate_to_soc(incident.incident_id, {})
                return

            logger.info(f"Executing playbook '{playbook.name}' for incident {incident.incident_id}")

            # Update incident status
            incident.status = IncidentStatus.INVESTIGATING
            incident.updated_at = datetime.utcnow()

            # Add playbook execution to timeline
            incident.timeline.append({
                "timestamp": datetime.utcnow().isoformat(),
                "action": "playbook_triggered",
                "description": f"Executing playbook: {playbook.name}",
                "playbook_id": playbook.playbook_id
            })

            # Execute response actions in priority order
            sorted_actions = sorted(playbook.response_actions, key=lambda x: x.get("priority", 999))

            for action_config in sorted_actions:
                action_name = action_config["action"]
                try:
                    action_enum = ResponseAction(action_name)
                    handler = self.response_handlers.get(action_enum)

                    if handler:
                        success = await handler(incident.incident_id, action_config)

                        # Record action execution
                        incident.response_actions.append(action_name)
                        incident.timeline.append({
                            "timestamp": datetime.utcnow().isoformat(),
                            "action": f"executed_{action_name}",
                            "description": f"Executed response action: {action_name}",
                            "success": success
                        })

                        if success:
                            logger.info(f"Successfully executed {action_name} for incident {incident.incident_id}")
                        else:
                            logger.warning(f"Failed to execute {action_name} for incident {incident.incident_id}")
                    else:
                        logger.error(f"No handler found for action: {action_name}")

                except Exception as action_error:
                    logger.error(f"Error executing action {action_name}: {action_error}")

            # Check if containment was successful
            if self._is_incident_contained(incident):
                incident.status = IncidentStatus.CONTAINED
                incident.timeline.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": "incident_contained",
                    "description": "Automated response successfully contained the incident"
                })

            # Schedule escalation if configured
            if playbook.escalation_rules:
                escalation_delay = playbook.escalation_rules.get("auto_escalate_after_minutes", 30)
                asyncio.create_task(self._schedule_escalation(incident.incident_id, escalation_delay))

            # Update incident in storage
            await self._update_incident(incident)

        except Exception as e:
            logger.error(f"Automated response error for incident {incident.incident_id}: {e}")

    def _find_matching_playbook(self, incident: SecurityIncident) -> Optional[ResponsePlaybook]:
        """Find matching playbook for incident"""
        # Extract incident characteristics
        incident_data = {
            "severity": incident.severity.name.lower(),
            "affected_assets": incident.affected_assets,
            "indicators": incident.indicators,
            "evidence": incident.evidence
        }

        # Check each playbook for matching conditions
        for playbook in self.response_playbooks.values():
            if self._matches_trigger_conditions(incident_data, playbook.trigger_conditions):
                return playbook

        return None

    def _matches_trigger_conditions(self, incident_data: Dict, conditions: Dict) -> bool:
        """Check if incident matches playbook trigger conditions"""
        for key, expected_value in conditions.items():
            if key == "threat_category":
                # Check evidence for threat category
                threat_category = incident_data.get("evidence", {}).get("threat_category", "")
                if expected_value not in threat_category:
                    return False

            elif key == "confidence":
                # Check confidence level
                confidence = incident_data.get("evidence", {}).get("confidence", 0)
                if confidence < expected_value:
                    return False

            elif key == "severity":
                # Check severity level
                if incident_data.get("severity", "").lower() != expected_value:
                    return False

            elif key == "pattern":
                # Check for specific patterns in evidence
                pattern = incident_data.get("evidence", {}).get("pattern", "")
                if expected_value not in pattern:
                    return False

        return True

    # Response Action Handlers

    async def _isolate_host(self, incident_id: str, config: Dict) -> bool:
        """Isolate affected host from network"""
        try:
            incident = self.active_incidents.get(incident_id)
            if not incident:
                return False

            for asset in incident.affected_assets:
                if asset.startswith("host:"):
                    host_ip = asset.replace("host:", "")

                    # Simulate network isolation (integrate with your network security tools)
                    logger.info(f"Isolating host {host_ip} for incident {incident_id}")

                    # In production, this would call your firewall/network security API
                    # Example: await firewall_api.block_host(host_ip)

                    # For demo, we'll just log the action
                    await asyncio.sleep(1)  # Simulate API call

            return True

        except Exception as e:
            logger.error(f"Host isolation error: {e}")
            return False

    async def _block_ip_address(self, incident_id: str, config: Dict) -> bool:
        """Block malicious IP address"""
        try:
            incident = self.active_incidents.get(incident_id)
            if not incident:
                return False

            # Extract IP addresses from evidence
            source_ip = incident.evidence.get("source_ip")
            if source_ip:
                logger.info(f"Blocking IP {source_ip} for incident {incident_id}")

                # Integrate with firewall/WAF to block IP
                # await firewall_api.block_ip(source_ip)

                await asyncio.sleep(1)  # Simulate API call

            return True

        except Exception as e:
            logger.error(f"IP blocking error: {e}")
            return False

    async def _disable_user_account(self, incident_id: str, config: Dict) -> bool:
        """Disable compromised user account"""
        try:
            incident = self.active_incidents.get(incident_id)
            if not incident:
                return False

            # Extract user from evidence
            user_id = incident.evidence.get("user_id")
            if user_id:
                logger.info(f"Disabling user account {user_id} for incident {incident_id}")

                # Integrate with identity management system
                # await identity_api.disable_user(user_id)

                await asyncio.sleep(1)  # Simulate API call

            return True

        except Exception as e:
            logger.error(f"User disabling error: {e}")
            return False

    async def _quarantine_file(self, incident_id: str, config: Dict) -> bool:
        """Quarantine malicious file"""
        try:
            incident = self.active_incidents.get(incident_id)
            if not incident:
                return False

            # Extract file hash from evidence
            file_hash = incident.evidence.get("file_hash")
            if file_hash:
                logger.info(f"Quarantining file {file_hash} for incident {incident_id}")

                # Integrate with endpoint security solution
                # await endpoint_api.quarantine_file(file_hash)

                await asyncio.sleep(1)  # Simulate API call

            return True

        except Exception as e:
            logger.error(f"File quarantine error: {e}")
            return False

    async def _reset_user_password(self, incident_id: str, config: Dict) -> bool:
        """Reset compromised user password"""
        try:
            incident = self.active_incidents.get(incident_id)
            if not incident:
                return False

            user_id = incident.evidence.get("user_id")
            if user_id:
                logger.info(f"Resetting password for user {user_id} for incident {incident_id}")

                # Generate secure temporary password
                temp_password = str(uuid.uuid4())[:12]

                # Integrate with identity management
                # await identity_api.reset_password(user_id, temp_password)

                await asyncio.sleep(1)  # Simulate API call

            return True

        except Exception as e:
            logger.error(f"Password reset error: {e}")
            return False

    async def _escalate_to_soc(self, incident_id: str, config: Dict) -> bool:
        """Escalate incident to SOC analysts"""
        try:
            incident = self.active_incidents.get(incident_id)
            if not incident:
                return False

            logger.info(f"Escalating incident {incident_id} to SOC")

            # Send notification to SOC team
            await self._send_notification(
                "email",
                "SOC Escalation",
                f"Incident {incident_id} has been escalated to SOC.\n\n"
                f"Title: {incident.title}\n"
                f"Severity: {incident.severity.name}\n"
                f"Description: {incident.description}"
            )

            return True

        except Exception as e:
            logger.error(f"SOC escalation error: {e}")
            return False

    async def _notify_administrators(self, incident_id: str, config: Dict) -> bool:
        """Notify system administrators"""
        try:
            incident = self.active_incidents.get(incident_id)
            if not incident:
                return False

            logger.info(f"Notifying administrators about incident {incident_id}")

            await self._send_notification(
                "email",
                f"Security Incident: {incident.title}",
                f"A security incident has been detected and automated response initiated.\n\n"
                f"Incident ID: {incident_id}\n"
                f"Title: {incident.title}\n"
                f"Severity: {incident.severity.name}\n"
                f"Status: {incident.status.value}\n"
                f"Description: {incident.description}\n\n"
                f"Automated responses executed: {', '.join(incident.response_actions)}"
            )

            return True

        except Exception as e:
            logger.error(f"Admin notification error: {e}")
            return False

    async def _collect_forensic_evidence(self, incident_id: str, config: Dict) -> bool:
        """Collect forensic evidence"""
        try:
            incident = self.active_incidents.get(incident_id)
            if not incident:
                return False

            logger.info(f"Collecting forensic evidence for incident {incident_id}")

            # Simulate forensic data collection
            forensic_data = {
                "collected_at": datetime.utcnow().isoformat(),
                "network_logs": "simulated_network_logs.pcap",
                "system_logs": "simulated_system_logs.log",
                "memory_dump": "simulated_memory_dump.dmp",
                "disk_image": "simulated_disk_image.dd"
            }

            # Store forensic evidence in incident
            incident.evidence["forensic_data"] = forensic_data

            await asyncio.sleep(2)  # Simulate collection time

            return True

        except Exception as e:
            logger.error(f"Forensic collection error: {e}")
            return False

    async def _update_security_signatures(self, incident_id: str, config: Dict) -> bool:
        """Update security signatures based on incident"""
        try:
            incident = self.active_incidents.get(incident_id)
            if not incident:
                return False

            logger.info(f"Updating security signatures for incident {incident_id}")

            # Extract IOCs for signature creation
            file_hash = incident.evidence.get("file_hash")
            ip_address = incident.evidence.get("source_ip")

            if file_hash or ip_address:
                # Update security tools with new signatures
                # await security_tools_api.update_signatures({
                #     "file_hashes": [file_hash] if file_hash else [],
                #     "ip_addresses": [ip_address] if ip_address else []
                # })

                await asyncio.sleep(1)  # Simulate API call

            return True

        except Exception as e:
            logger.error(f"Signature update error: {e}")
            return False

    async def _send_notification(self, channel: str, subject: str, message: str):
        """Send notification through specified channel"""
        try:
            if channel == "email":
                await self._send_email_notification(subject, message)
            elif channel == "slack":
                await self._send_slack_notification(message)
            elif channel == "pagerduty":
                await self._send_pagerduty_alert(subject, message)

        except Exception as e:
            logger.error(f"Notification error: {e}")

    async def _send_email_notification(self, subject: str, message: str):
        """Send email notification"""
        # Simplified email sending (configure with real SMTP in production)
        logger.info(f"Email notification: {subject}")
        logger.debug(f"Email content: {message}")

    async def _send_slack_notification(self, message: str):
        """Send Slack notification"""
        logger.info(f"Slack notification: {message}")

    async def _send_pagerduty_alert(self, title: str, description: str):
        """Send PagerDuty alert"""
        logger.info(f"PagerDuty alert: {title}")

    def _is_incident_contained(self, incident: SecurityIncident) -> bool:
        """Check if incident has been contained"""
        # Determine containment based on executed actions
        critical_actions = [
            ResponseAction.ISOLATE_HOST.value,
            ResponseAction.BLOCK_IP.value,
            ResponseAction.QUARANTINE_FILE.value
        ]

        for action in critical_actions:
            if action in incident.response_actions:
                return True

        return False

    async def _schedule_escalation(self, incident_id: str, delay_minutes: int):
        """Schedule automatic escalation"""
        await asyncio.sleep(delay_minutes * 60)

        incident = self.active_incidents.get(incident_id)
        if incident and incident.status in [IncidentStatus.OPEN, IncidentStatus.INVESTIGATING]:
            logger.info(f"Auto-escalating incident {incident_id} after {delay_minutes} minutes")
            await self._escalate_to_soc(incident_id, {})

    async def _update_incident(self, incident: SecurityIncident):
        """Update incident in storage"""
        incident.updated_at = datetime.utcnow()

        await self.redis_client.setex(
            f"incident:{incident.incident_id}",
            86400 * 7,  # 7 days
            json.dumps(incident.to_dict())
        )

    async def _load_active_incidents(self):
        """Load active incidents from Redis"""
        try:
            keys = await self.redis_client.keys("incident:*")

            for key in keys:
                incident_data = await self.redis_client.get(key)
                if incident_data:
                    incident_dict = json.loads(incident_data)

                    # Reconstruct incident object
                    incident = SecurityIncident(
                        incident_id=incident_dict["incident_id"],
                        title=incident_dict["title"],
                        description=incident_dict["description"],
                        severity=IncidentSeverity(incident_dict["severity"]),
                        status=IncidentStatus(incident_dict["status"]),
                        created_at=datetime.fromisoformat(incident_dict["created_at"]),
                        updated_at=datetime.fromisoformat(incident_dict["updated_at"]),
                        source_system=incident_dict["source_system"],
                        affected_assets=incident_dict["affected_assets"],
                        indicators=incident_dict["indicators"],
                        evidence=incident_dict["evidence"],
                        assigned_analyst=incident_dict.get("assigned_analyst"),
                        response_actions=incident_dict.get("response_actions", []),
                        timeline=incident_dict.get("timeline", [])
                    )

                    self.active_incidents[incident.incident_id] = incident

            logger.info(f"Loaded {len(self.active_incidents)} active incidents from storage")

        except Exception as e:
            logger.error(f"Error loading incidents: {e}")

    async def get_incident_status(self, incident_id: str) -> Optional[Dict]:
        """Get current incident status"""
        incident = self.active_incidents.get(incident_id)
        if incident:
            return incident.to_dict()
        return None

    async def list_active_incidents(self) -> List[Dict]:
        """List all active incidents"""
        return [incident.to_dict() for incident in self.active_incidents.values()]

async def main():
    """Main function for testing incident response orchestrator"""
    orchestrator = IncidentResponseOrchestrator()
    await orchestrator.initialize()

    # Example incident creation
    incident = await orchestrator.create_incident(
        title="Malware Detection on Critical Server",
        description="Suspicious executable detected on production database server",
        severity=IncidentSeverity.P1_CRITICAL,
        source_system="endpoint_security",
        affected_assets=["host:10.0.0.100", "database:prod-db-01"],
        indicators=["file_hash:abc123", "process:malicious.exe"],
        evidence={
            "threat_category": "malware",
            "confidence": 0.95,
            "file_hash": "abc123def456",
            "source_ip": "192.168.1.100",
            "user_id": "compromised_user"
        }
    )

    print(f"Created incident: {incident.incident_id}")

    # Wait for automated response to complete
    await asyncio.sleep(5)

    # Check incident status
    status = await orchestrator.get_incident_status(incident.incident_id)
    print(f"Incident status: {status['status']}")
    print(f"Response actions executed: {status['response_actions']}")

    # List all active incidents
    active_incidents = await orchestrator.list_active_incidents()
    print(f"Total active incidents: {len(active_incidents)}")

if __name__ == "__main__":
    asyncio.run(main())
