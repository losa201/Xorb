"""
Automated Incident Response and Orchestration System
AI-powered incident detection, classification, and automated response capabilities
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import hashlib
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class IncidentSeverity(Enum):
    """Incident severity levels"""
    CRITICAL = "critical"        # System compromise, data breach
    HIGH = "high"               # Service disruption, security breach
    MEDIUM = "medium"           # Performance degradation, suspicious activity
    LOW = "low"                 # Minor issues, informational alerts
    INFORMATIONAL = "info"      # Monitoring alerts, routine events

class IncidentCategory(Enum):
    """Incident categories"""
    SECURITY_BREACH = "security_breach"
    MALWARE_INFECTION = "malware_infection"
    DATA_BREACH = "data_breach"
    DENIAL_OF_SERVICE = "denial_of_service"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SYSTEM_COMPROMISE = "system_compromise"
    PHISHING_ATTACK = "phishing_attack"
    INSIDER_THREAT = "insider_threat"
    COMPLIANCE_VIOLATION = "compliance_violation"
    SYSTEM_OUTAGE = "system_outage"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CONFIGURATION_ERROR = "configuration_error"

class IncidentStatus(Enum):
    """Incident lifecycle status"""
    NEW = "new"
    TRIAGED = "triaged"
    INVESTIGATING = "investigating"
    CONTAINMENT = "containment"
    ERADICATION = "eradication"
    RECOVERY = "recovery"
    POST_INCIDENT = "post_incident"
    CLOSED = "closed"

class ResponseAction(Enum):
    """Automated response actions"""
    BLOCK_IP = "block_ip"
    QUARANTINE_USER = "quarantine_user"
    ISOLATE_SYSTEM = "isolate_system"
    DISABLE_ACCOUNT = "disable_account"
    RESET_PASSWORD = "reset_password"
    REVOKE_ACCESS = "revoke_access"
    ALERT_TEAM = "alert_team"
    ESCALATE = "escalate"
    COLLECT_EVIDENCE = "collect_evidence"
    BACKUP_SYSTEM = "backup_system"
    PATCH_SYSTEM = "patch_system"
    SCAN_SYSTEM = "scan_system"

class AutomationLevel(Enum):
    """Automation levels for responses"""
    MANUAL = "manual"           # Manual intervention required
    SEMI_AUTOMATED = "semi"     # Automated with approval
    FULLY_AUTOMATED = "auto"    # Fully automated response

@dataclass
class IncidentEvidence:
    """Digital evidence associated with incident"""
    evidence_id: str
    evidence_type: str  # log, file, network_capture, memory_dump
    source_system: str
    collected_at: datetime
    file_path: Optional[str] = None
    hash_value: Optional[str] = None
    chain_of_custody: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.chain_of_custody is None:
            self.chain_of_custody = ["Automated Collection System"]
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ResponseActionResult:
    """Result of automated response action"""
    action_id: str
    action_type: ResponseAction
    status: str  # success, failed, pending
    executed_at: datetime
    duration_seconds: float
    output: str
    error_message: Optional[str] = None
    rollback_info: Optional[Dict[str, Any]] = None

@dataclass
class SecurityIncident:
    """Security incident data structure"""
    incident_id: str
    title: str
    description: str
    category: IncidentCategory
    severity: IncidentSeverity
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    detected_by: str
    assigned_to: Optional[str] = None
    source_events: List[str] = None
    affected_systems: List[str] = None
    affected_users: List[str] = None
    indicators: List[Dict[str, Any]] = None
    evidence: List[IncidentEvidence] = None
    response_actions: List[ResponseActionResult] = None
    timeline: List[Dict[str, Any]] = None
    estimated_impact: Dict[str, Any] = None
    remediation_steps: List[str] = None
    lessons_learned: List[str] = None
    
    def __post_init__(self):
        if self.source_events is None:
            self.source_events = []
        if self.affected_systems is None:
            self.affected_systems = []
        if self.affected_users is None:
            self.affected_users = []
        if self.indicators is None:
            self.indicators = []
        if self.evidence is None:
            self.evidence = []
        if self.response_actions is None:
            self.response_actions = []
        if self.timeline is None:
            self.timeline = []
        if self.estimated_impact is None:
            self.estimated_impact = {}
        if self.remediation_steps is None:
            self.remediation_steps = []
        if self.lessons_learned is None:
            self.lessons_learned = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['category'] = self.category.value
        data['severity'] = self.severity.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        
        # Convert evidence
        data['evidence'] = []
        for evidence in self.evidence:
            evidence_dict = asdict(evidence)
            evidence_dict['collected_at'] = evidence.collected_at.isoformat()
            data['evidence'].append(evidence_dict)
        
        # Convert response actions
        data['response_actions'] = []
        for action in self.response_actions:
            action_dict = asdict(action)
            action_dict['action_type'] = action.action_type.value
            action_dict['executed_at'] = action.executed_at.isoformat()
            data['response_actions'].append(action_dict)
        
        return data

@dataclass
class PlaybookRule:
    """Incident response playbook rule"""
    rule_id: str
    name: str
    description: str
    trigger_conditions: Dict[str, Any]
    response_actions: List[Dict[str, Any]]
    automation_level: AutomationLevel
    priority: int
    enabled: bool = True

class IncidentResponseOrchestrator:
    """Automated incident response and orchestration engine"""
    
    def __init__(self):
        self.active_incidents = {}
        self.incident_history = deque(maxlen=10000)
        self.response_playbooks = {}
        self.automation_rules = {}
        self.evidence_collectors = {}
        self.notification_channels = {}
        self.escalation_rules = {}
        
        # Initialize response capabilities
        self._initialize_playbooks()
        self._initialize_automation_rules()
        self._initialize_response_actions()
    
    def _initialize_playbooks(self):
        """Initialize incident response playbooks"""
        
        # Malware Infection Playbook
        self.response_playbooks["malware_infection"] = PlaybookRule(
            rule_id="pb_malware_001",
            name="Malware Infection Response",
            description="Automated response to malware detection",
            trigger_conditions={
                "category": IncidentCategory.MALWARE_INFECTION,
                "severity": [IncidentSeverity.HIGH, IncidentSeverity.CRITICAL]
            },
            response_actions=[
                {"action": ResponseAction.ISOLATE_SYSTEM, "delay": 0, "approval_required": False},
                {"action": ResponseAction.COLLECT_EVIDENCE, "delay": 60, "approval_required": False},
                {"action": ResponseAction.SCAN_SYSTEM, "delay": 300, "approval_required": False},
                {"action": ResponseAction.ALERT_TEAM, "delay": 0, "approval_required": False}
            ],
            automation_level=AutomationLevel.FULLY_AUTOMATED,
            priority=1
        )
        
        # Unauthorized Access Playbook
        self.response_playbooks["unauthorized_access"] = PlaybookRule(
            rule_id="pb_unauth_001",
            name="Unauthorized Access Response",
            description="Response to unauthorized access attempts",
            trigger_conditions={
                "category": IncidentCategory.UNAUTHORIZED_ACCESS,
                "severity": [IncidentSeverity.HIGH, IncidentSeverity.CRITICAL]
            },
            response_actions=[
                {"action": ResponseAction.DISABLE_ACCOUNT, "delay": 0, "approval_required": False},
                {"action": ResponseAction.RESET_PASSWORD, "delay": 30, "approval_required": True},
                {"action": ResponseAction.COLLECT_EVIDENCE, "delay": 60, "approval_required": False},
                {"action": ResponseAction.ALERT_TEAM, "delay": 0, "approval_required": False}
            ],
            automation_level=AutomationLevel.SEMI_AUTOMATED,
            priority=2
        )
        
        # Data Breach Playbook
        self.response_playbooks["data_breach"] = PlaybookRule(
            rule_id="pb_breach_001",
            name="Data Breach Response",
            description="Critical response to data breach incidents",
            trigger_conditions={
                "category": IncidentCategory.DATA_BREACH
            },
            response_actions=[
                {"action": ResponseAction.ISOLATE_SYSTEM, "delay": 0, "approval_required": False},
                {"action": ResponseAction.COLLECT_EVIDENCE, "delay": 60, "approval_required": False},
                {"action": ResponseAction.ALERT_TEAM, "delay": 0, "approval_required": False},
                {"action": ResponseAction.ESCALATE, "delay": 300, "approval_required": False}
            ],
            automation_level=AutomationLevel.SEMI_AUTOMATED,
            priority=1
        )
        
        # DDoS Attack Playbook
        self.response_playbooks["ddos_attack"] = PlaybookRule(
            rule_id="pb_ddos_001",
            name="DDoS Attack Response",
            description="Response to denial of service attacks",
            trigger_conditions={
                "category": IncidentCategory.DENIAL_OF_SERVICE
            },
            response_actions=[
                {"action": ResponseAction.BLOCK_IP, "delay": 0, "approval_required": False},
                {"action": ResponseAction.COLLECT_EVIDENCE, "delay": 120, "approval_required": False},
                {"action": ResponseAction.ALERT_TEAM, "delay": 60, "approval_required": False}
            ],
            automation_level=AutomationLevel.FULLY_AUTOMATED,
            priority=2
        )
    
    def _initialize_automation_rules(self):
        """Initialize automation rules for incident classification"""
        
        self.automation_rules = {
            # Security event patterns
            "multiple_failed_logins": {
                "pattern": "failed_login_count > 10 AND time_window < 300",
                "category": IncidentCategory.UNAUTHORIZED_ACCESS,
                "severity": IncidentSeverity.MEDIUM,
                "confidence": 0.8
            },
            "malware_signature_match": {
                "pattern": "malware_detected = True",
                "category": IncidentCategory.MALWARE_INFECTION,
                "severity": IncidentSeverity.HIGH,
                "confidence": 0.9
            },
            "suspicious_data_transfer": {
                "pattern": "data_transfer_volume > threshold AND destination = external",
                "category": IncidentCategory.DATA_BREACH,
                "severity": IncidentSeverity.CRITICAL,
                "confidence": 0.7
            },
            "privilege_escalation": {
                "pattern": "privilege_change = True AND user_risk_score > 0.7",
                "category": IncidentCategory.INSIDER_THREAT,
                "severity": IncidentSeverity.HIGH,
                "confidence": 0.8
            }
        }
    
    def _initialize_response_actions(self):
        """Initialize automated response action handlers"""
        
        self.response_handlers = {
            ResponseAction.BLOCK_IP: self._block_ip_address,
            ResponseAction.QUARANTINE_USER: self._quarantine_user_account,
            ResponseAction.ISOLATE_SYSTEM: self._isolate_system,
            ResponseAction.DISABLE_ACCOUNT: self._disable_user_account,
            ResponseAction.RESET_PASSWORD: self._reset_user_password,
            ResponseAction.REVOKE_ACCESS: self._revoke_user_access,
            ResponseAction.ALERT_TEAM: self._alert_security_team,
            ResponseAction.ESCALATE: self._escalate_incident,
            ResponseAction.COLLECT_EVIDENCE: self._collect_digital_evidence,
            ResponseAction.BACKUP_SYSTEM: self._backup_system_state,
            ResponseAction.PATCH_SYSTEM: self._apply_security_patches,
            ResponseAction.SCAN_SYSTEM: self._scan_affected_system
        }
    
    async def create_incident(self, 
                            title: str,
                            description: str,
                            category: IncidentCategory,
                            severity: IncidentSeverity,
                            source_events: List[str] = None,
                            affected_systems: List[str] = None,
                            detected_by: str = "Automated Detection") -> SecurityIncident:
        """Create new security incident"""
        
        try:
            incident_id = f"INC-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
            
            incident = SecurityIncident(
                incident_id=incident_id,
                title=title,
                description=description,
                category=category,
                severity=severity,
                status=IncidentStatus.NEW,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                detected_by=detected_by,
                source_events=source_events or [],
                affected_systems=affected_systems or []
            )
            
            # Add to timeline
            incident.timeline.append({
                "timestamp": datetime.utcnow().isoformat(),
                "event": "Incident Created",
                "details": f"Incident {incident_id} created by {detected_by}",
                "actor": detected_by
            })
            
            # Store incident
            self.active_incidents[incident_id] = incident
            
            # Trigger automated response
            await self._trigger_automated_response(incident)
            
            logger.info(f"Security incident created: {incident_id} - {title}")
            
            return incident
            
        except Exception as e:
            logger.error(f"Error creating incident: {e}")
            raise
    
    async def _trigger_automated_response(self, incident: SecurityIncident):
        """Trigger automated response based on incident characteristics"""
        
        try:
            # Find matching playbooks
            matching_playbooks = self._find_matching_playbooks(incident)
            
            if not matching_playbooks:
                logger.info(f"No automated playbooks match incident {incident.incident_id}")
                return
            
            # Execute highest priority playbook
            playbook = max(matching_playbooks, key=lambda p: p.priority)
            
            logger.info(f"Executing playbook {playbook.name} for incident {incident.incident_id}")
            
            # Update incident status
            incident.status = IncidentStatus.TRIAGED
            incident.updated_at = datetime.utcnow()
            incident.timeline.append({
                "timestamp": datetime.utcnow().isoformat(),
                "event": "Automated Response Triggered",
                "details": f"Executing playbook: {playbook.name}",
                "actor": "Automation Engine"
            })
            
            # Execute response actions
            for action_config in playbook.response_actions:
                action_type = action_config["action"]
                delay = action_config.get("delay", 0)
                approval_required = action_config.get("approval_required", False)
                
                if delay > 0:
                    await asyncio.sleep(delay)
                
                if approval_required and playbook.automation_level != AutomationLevel.FULLY_AUTOMATED:
                    # Queue for manual approval
                    await self._queue_for_approval(incident, action_type, action_config)
                else:
                    # Execute immediately
                    await self._execute_response_action(incident, action_type, action_config)
            
        except Exception as e:
            logger.error(f"Error in automated response: {e}")
    
    def _find_matching_playbooks(self, incident: SecurityIncident) -> List[PlaybookRule]:
        """Find playbooks that match incident characteristics"""
        
        matching_playbooks = []
        
        for playbook in self.response_playbooks.values():
            if not playbook.enabled:
                continue
            
            conditions = playbook.trigger_conditions
            
            # Check category match
            if "category" in conditions:
                if conditions["category"] != incident.category:
                    continue
            
            # Check severity match
            if "severity" in conditions:
                if isinstance(conditions["severity"], list):
                    if incident.severity not in conditions["severity"]:
                        continue
                else:
                    if conditions["severity"] != incident.severity:
                        continue
            
            # Check other conditions (could be extended)
            # For now, basic category and severity matching
            
            matching_playbooks.append(playbook)
        
        return matching_playbooks
    
    async def _execute_response_action(self, 
                                     incident: SecurityIncident, 
                                     action_type: ResponseAction, 
                                     config: Dict[str, Any]):
        """Execute automated response action"""
        
        try:
            action_id = f"ACT-{datetime.utcnow().strftime('%H%M%S')}-{uuid.uuid4().hex[:4]}"
            start_time = datetime.utcnow()
            
            logger.info(f"Executing response action {action_type.value} for incident {incident.incident_id}")
            
            # Get action handler
            handler = self.response_handlers.get(action_type)
            if not handler:
                raise ValueError(f"No handler found for action {action_type.value}")
            
            # Execute action
            result = await handler(incident, config)
            
            # Calculate duration
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Create action result
            action_result = ResponseActionResult(
                action_id=action_id,
                action_type=action_type,
                status="success" if result.get("success", False) else "failed",
                executed_at=start_time,
                duration_seconds=duration,
                output=result.get("output", ""),
                error_message=result.get("error"),
                rollback_info=result.get("rollback_info")
            )
            
            # Add to incident
            incident.response_actions.append(action_result)
            incident.updated_at = datetime.utcnow()
            incident.timeline.append({
                "timestamp": datetime.utcnow().isoformat(),
                "event": f"Response Action Executed",
                "details": f"{action_type.value}: {action_result.status}",
                "actor": "Automation Engine"
            })
            
            logger.info(f"Response action {action_type.value} completed with status: {action_result.status}")
            
        except Exception as e:
            logger.error(f"Error executing response action {action_type.value}: {e}")
            
            # Record failed action
            action_result = ResponseActionResult(
                action_id=f"ACT-{datetime.utcnow().strftime('%H%M%S')}-ERR",
                action_type=action_type,
                status="failed",
                executed_at=datetime.utcnow(),
                duration_seconds=0,
                output="",
                error_message=str(e)
            )
            
            incident.response_actions.append(action_result)
    
    async def _queue_for_approval(self, 
                                incident: SecurityIncident, 
                                action_type: ResponseAction, 
                                config: Dict[str, Any]):
        """Queue response action for manual approval"""
        
        logger.info(f"Queuing action {action_type.value} for approval on incident {incident.incident_id}")
        
        # In a real implementation, this would integrate with approval workflow system
        incident.timeline.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "Action Queued for Approval",
            "details": f"{action_type.value} requires manual approval",
            "actor": "Automation Engine"
        })
    
    # Response Action Handlers (Mock implementations)
    
    async def _block_ip_address(self, incident: SecurityIncident, config: Dict[str, Any]) -> Dict[str, Any]:
        """Block IP address at firewall"""
        
        # Extract IP addresses from incident indicators
        ip_addresses = []
        for indicator in incident.indicators:
            if indicator.get("type") == "ip_address":
                ip_addresses.append(indicator.get("value"))
        
        if not ip_addresses:
            # Extract from source events or use default
            ip_addresses = ["192.168.1.100"]  # Mock IP
        
        try:
            # Mock firewall API call
            blocked_ips = []
            for ip in ip_addresses:
                # Simulate firewall rule creation
                await asyncio.sleep(0.1)  # Simulate API call
                blocked_ips.append(ip)
                logger.info(f"Blocked IP address: {ip}")
            
            return {
                "success": True,
                "output": f"Successfully blocked {len(blocked_ips)} IP addresses: {', '.join(blocked_ips)}",
                "rollback_info": {
                    "action": "unblock_ips",
                    "blocked_ips": blocked_ips,
                    "rule_ids": [f"rule_{ip.replace('.', '_')}" for ip in blocked_ips]
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to block IP addresses: {str(e)}",
                "output": ""
            }
    
    async def _isolate_system(self, incident: SecurityIncident, config: Dict[str, Any]) -> Dict[str, Any]:
        """Isolate affected system from network"""
        
        systems = incident.affected_systems or ["unknown-system"]
        
        try:
            isolated_systems = []
            for system in systems:
                # Mock system isolation (would integrate with network management)
                await asyncio.sleep(0.2)
                isolated_systems.append(system)
                logger.info(f"Isolated system from network: {system}")
            
            return {
                "success": True,
                "output": f"Successfully isolated {len(isolated_systems)} systems: {', '.join(isolated_systems)}",
                "rollback_info": {
                    "action": "restore_network_access",
                    "isolated_systems": isolated_systems
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to isolate systems: {str(e)}",
                "output": ""
            }
    
    async def _disable_user_account(self, incident: SecurityIncident, config: Dict[str, Any]) -> Dict[str, Any]:
        """Disable user account"""
        
        users = incident.affected_users or ["unknown-user"]
        
        try:
            disabled_accounts = []
            for user in users:
                # Mock user account disabling (would integrate with identity management)
                await asyncio.sleep(0.1)
                disabled_accounts.append(user)
                logger.info(f"Disabled user account: {user}")
            
            return {
                "success": True,
                "output": f"Successfully disabled {len(disabled_accounts)} accounts: {', '.join(disabled_accounts)}",
                "rollback_info": {
                    "action": "enable_accounts",
                    "disabled_accounts": disabled_accounts
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to disable accounts: {str(e)}",
                "output": ""
            }
    
    async def _collect_digital_evidence(self, incident: SecurityIncident, config: Dict[str, Any]) -> Dict[str, Any]:
        """Collect digital evidence"""
        
        try:
            evidence_items = []
            
            # Collect system logs
            log_evidence = IncidentEvidence(
                evidence_id=f"LOG-{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                evidence_type="system_logs",
                source_system="security_monitoring",
                collected_at=datetime.utcnow(),
                file_path="/var/log/security.log",
                hash_value=hashlib.sha256(f"mock_log_data_{incident.incident_id}".encode()).hexdigest(),
                metadata={"log_entries": 1500, "time_range": "last_24_hours"}
            )
            evidence_items.append(log_evidence)
            
            # Collect network captures
            if incident.category in [IncidentCategory.DENIAL_OF_SERVICE, IncidentCategory.UNAUTHORIZED_ACCESS]:
                network_evidence = IncidentEvidence(
                    evidence_id=f"NET-{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    evidence_type="network_capture",
                    source_system="network_monitor",
                    collected_at=datetime.utcnow(),
                    file_path="/var/captures/incident_traffic.pcap",
                    hash_value=hashlib.sha256(f"mock_network_data_{incident.incident_id}".encode()).hexdigest(),
                    metadata={"packets": 25000, "duration": "30_minutes"}
                )
                evidence_items.append(network_evidence)
            
            # Add evidence to incident
            incident.evidence.extend(evidence_items)
            
            return {
                "success": True,
                "output": f"Collected {len(evidence_items)} pieces of digital evidence",
                "evidence_collected": len(evidence_items)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to collect evidence: {str(e)}",
                "output": ""
            }
    
    async def _alert_security_team(self, incident: SecurityIncident, config: Dict[str, Any]) -> Dict[str, Any]:
        """Alert security team"""
        
        try:
            # Mock notification system
            notification_channels = ["email", "slack", "sms"]
            
            alert_message = f"""
            SECURITY INCIDENT ALERT
            
            Incident ID: {incident.incident_id}
            Severity: {incident.severity.value.upper()}
            Category: {incident.category.value.replace('_', ' ').title()}
            
            Title: {incident.title}
            Description: {incident.description}
            
            Affected Systems: {', '.join(incident.affected_systems) if incident.affected_systems else 'Unknown'}
            
            Time: {incident.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
            
            Please review and take appropriate action.
            """
            
            sent_notifications = []
            for channel in notification_channels:
                await asyncio.sleep(0.1)  # Simulate notification sending
                sent_notifications.append(channel)
                logger.info(f"Sent security alert via {channel}")
            
            return {
                "success": True,
                "output": f"Security team alerted via {len(sent_notifications)} channels: {', '.join(sent_notifications)}",
                "notifications_sent": len(sent_notifications)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to alert security team: {str(e)}",
                "output": ""
            }
    
    # Additional response action handlers (mock implementations)
    
    async def _quarantine_user_account(self, incident, config):
        return {"success": True, "output": "User account quarantined"}
    
    async def _reset_user_password(self, incident, config):
        return {"success": True, "output": "User password reset and user notified"}
    
    async def _revoke_user_access(self, incident, config):
        return {"success": True, "output": "User access tokens revoked"}
    
    async def _escalate_incident(self, incident, config):
        incident.severity = IncidentSeverity.CRITICAL
        return {"success": True, "output": "Incident escalated to CRITICAL severity"}
    
    async def _backup_system_state(self, incident, config):
        return {"success": True, "output": "System state backed up successfully"}
    
    async def _apply_security_patches(self, incident, config):
        return {"success": True, "output": "Security patches applied to affected systems"}
    
    async def _scan_affected_system(self, incident, config):
        return {"success": True, "output": "Security scan completed on affected systems"}
    
    # Incident Management Methods
    
    async def get_incident(self, incident_id: str) -> Optional[SecurityIncident]:
        """Get incident by ID"""
        return self.active_incidents.get(incident_id)
    
    async def update_incident_status(self, incident_id: str, status: IncidentStatus, notes: str = None):
        """Update incident status"""
        
        incident = self.active_incidents.get(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")
        
        old_status = incident.status
        incident.status = status
        incident.updated_at = datetime.utcnow()
        
        incident.timeline.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "Status Changed",
            "details": f"Status changed from {old_status.value} to {status.value}",
            "notes": notes,
            "actor": "System"
        })
        
        # Move to history if closed
        if status == IncidentStatus.CLOSED:
            self.incident_history.append(incident)
            del self.active_incidents[incident_id]
        
        logger.info(f"Incident {incident_id} status updated to {status.value}")
    
    async def get_incident_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get incident statistics"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Include both active and historical incidents
        all_incidents = list(self.active_incidents.values()) + list(self.incident_history)
        recent_incidents = [i for i in all_incidents if i.created_at >= cutoff_date]
        
        if not recent_incidents:
            return {
                "total_incidents": 0,
                "by_severity": {},
                "by_category": {},
                "by_status": {},
                "average_resolution_time": 0,
                "incidents_per_day": 0
            }
        
        # Calculate statistics
        stats = {
            "total_incidents": len(recent_incidents),
            "by_severity": defaultdict(int),
            "by_category": defaultdict(int),
            "by_status": defaultdict(int),
            "incidents_per_day": len(recent_incidents) / days
        }
        
        resolution_times = []
        
        for incident in recent_incidents:
            stats["by_severity"][incident.severity.value] += 1
            stats["by_category"][incident.category.value] += 1
            stats["by_status"][incident.status.value] += 1
            
            # Calculate resolution time for closed incidents
            if incident.status == IncidentStatus.CLOSED:
                resolution_time = (incident.updated_at - incident.created_at).total_seconds() / 3600  # hours
                resolution_times.append(resolution_time)
        
        # Calculate average resolution time
        if resolution_times:
            stats["average_resolution_time"] = sum(resolution_times) / len(resolution_times)
        else:
            stats["average_resolution_time"] = 0
        
        # Convert defaultdicts to regular dicts
        stats["by_severity"] = dict(stats["by_severity"])
        stats["by_category"] = dict(stats["by_category"])
        stats["by_status"] = dict(stats["by_status"])
        
        return stats

# Global incident response orchestrator
incident_orchestrator = IncidentResponseOrchestrator()

async def get_incident_orchestrator() -> IncidentResponseOrchestrator:
    """Get incident response orchestrator instance"""
    return incident_orchestrator

async def create_security_incident(title: str, description: str, category: IncidentCategory, severity: IncidentSeverity) -> SecurityIncident:
    """Create new security incident"""
    orchestrator = await get_incident_orchestrator()
    return await orchestrator.create_incident(title, description, category, severity)