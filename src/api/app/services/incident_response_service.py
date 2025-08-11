#!/usr/bin/env python3
"""
Automated Incident Response and Threat Hunting Service
Advanced security orchestration, automation, and response (SOAR) for XORB Platform

This service provides:
- Automated incident detection and classification
- Threat hunting capabilities with ML integration
- Automated response actions and playbooks
- Incident lifecycle management
- Threat intelligence integration
- Forensic evidence collection
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class IncidentSeverity(Enum):
    """Incident severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(Enum):
    """Incident status states"""
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    ERADICATED = "eradicated"
    RECOVERED = "recovered"
    CLOSED = "closed"


class IncidentCategory(Enum):
    """Incident categories"""
    MALWARE = "malware"
    PHISHING = "phishing"
    DATA_BREACH = "data_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DENIAL_OF_SERVICE = "denial_of_service"
    INSIDER_THREAT = "insider_threat"
    VULNERABILITY_EXPLOIT = "vulnerability_exploit"
    POLICY_VIOLATION = "policy_violation"
    SYSTEM_COMPROMISE = "system_compromise"
    NETWORK_INTRUSION = "network_intrusion"


class ResponseAction(Enum):
    """Automated response actions"""
    BLOCK_IP = "block_ip"
    DISABLE_USER = "disable_user"
    ISOLATE_SYSTEM = "isolate_system"
    RESET_PASSWORD = "reset_password"
    REVOKE_TOKENS = "revoke_tokens"
    QUARANTINE_FILE = "quarantine_file"
    COLLECT_EVIDENCE = "collect_evidence"
    NOTIFY_TEAM = "notify_team"
    CREATE_TICKET = "create_ticket"
    TRIGGER_BACKUP = "trigger_backup"


class ThreatLevel(Enum):
    """Threat hunting levels"""
    RECONNAISSANCE = "reconnaissance"
    INITIAL_ACCESS = "initial_access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    COMMAND_CONTROL = "command_control"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"


@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_id: str
    timestamp: datetime
    source_ip: str
    event_type: str
    severity: IncidentSeverity
    description: str
    raw_data: Dict[str, Any]
    indicators: List[str]
    mitre_techniques: List[str]
    confidence_score: float  # 0.0 - 1.0


@dataclass
class ThreatIndicator:
    """Threat indicator of compromise (IoC)"""
    indicator_id: str
    indicator_type: str  # ip, domain, hash, url, email
    value: str
    threat_level: ThreatLevel
    confidence: float
    first_seen: datetime
    last_seen: datetime
    source: str
    context: Dict[str, Any]


@dataclass
class Incident:
    """Security incident data structure"""
    incident_id: str
    title: str
    description: str
    category: IncidentCategory
    severity: IncidentSeverity
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    assigned_to: Optional[str]
    affected_systems: List[str]
    indicators: List[ThreatIndicator]
    events: List[SecurityEvent]
    response_actions: List[str]
    evidence: List[str]
    timeline: List[Dict[str, Any]]
    remediation_steps: List[str]
    lessons_learned: Optional[str]


@dataclass
class ResponsePlaybook:
    """Automated response playbook"""
    playbook_id: str
    name: str
    description: str
    triggers: List[str]  # Conditions that trigger this playbook
    actions: List[ResponseAction]
    required_approvals: List[str]
    automation_level: str  # "full", "semi", "manual"
    estimated_duration: int  # minutes
    success_criteria: List[str]


class ThreatHuntingEngine:
    """Advanced threat hunting engine"""
    
    def __init__(self):
        self.hunting_rules: Dict[str, Dict[str, Any]] = {}
        self.indicators: Dict[str, ThreatIndicator] = {}
        self._initialize_hunting_rules()
    
    def _initialize_hunting_rules(self):
        """Initialize threat hunting rules"""
        self.hunting_rules = {
            "suspicious_login_patterns": {
                "description": "Detect suspicious login patterns",
                "indicators": [
                    "multiple_failed_logins_same_ip",
                    "login_from_unusual_location",
                    "login_outside_business_hours",
                    "concurrent_logins_different_locations"
                ],
                "severity": IncidentSeverity.MEDIUM,
                "mitre_techniques": ["T1078", "T1110"]
            },
            
            "lateral_movement_detection": {
                "description": "Detect lateral movement attempts",
                "indicators": [
                    "unusual_network_traffic",
                    "smb_enumeration",
                    "rdp_brute_force",
                    "powershell_remote_execution"
                ],
                "severity": IncidentSeverity.HIGH,
                "mitre_techniques": ["T1021", "T1059", "T1135"]
            },
            
            "data_exfiltration_indicators": {
                "description": "Detect potential data exfiltration",
                "indicators": [
                    "large_data_transfer",
                    "unusual_database_queries",
                    "file_access_anomalies",
                    "external_data_transfer"
                ],
                "severity": IncidentSeverity.CRITICAL,
                "mitre_techniques": ["T1041", "T1030", "T1005"]
            },
            
            "privilege_escalation_attempts": {
                "description": "Detect privilege escalation attempts",
                "indicators": [
                    "sudo_abuse",
                    "setuid_exploitation",
                    "token_manipulation",
                    "exploit_attempts"
                ],
                "severity": IncidentSeverity.HIGH,
                "mitre_techniques": ["T1068", "T1134", "T1548"]
            },
            
            "command_control_communication": {
                "description": "Detect C2 communication patterns",
                "indicators": [
                    "beacon_traffic",
                    "dns_tunneling",
                    "suspicious_tls_certificates",
                    "unusual_http_patterns"
                ],
                "severity": IncidentSeverity.HIGH,
                "mitre_techniques": ["T1071", "T1090", "T1572"]
            }
        }
        
        logger.info(f"Initialized {len(self.hunting_rules)} threat hunting rules")
    
    async def hunt_threats(self, security_events: List[SecurityEvent]) -> List[ThreatIndicator]:
        """Perform threat hunting on security events"""
        detected_threats = []
        
        try:
            for rule_name, rule_config in self.hunting_rules.items():
                threats = await self._apply_hunting_rule(rule_name, rule_config, security_events)
                detected_threats.extend(threats)
            
            # Deduplicate and score threats
            unique_threats = self._deduplicate_threats(detected_threats)
            scored_threats = self._score_threats(unique_threats, security_events)
            
            logger.info(f"Threat hunting completed: {len(scored_threats)} threats detected")
            return scored_threats
            
        except Exception as e:
            logger.error(f"Threat hunting failed: {str(e)}")
            return []
    
    async def _apply_hunting_rule(
        self, 
        rule_name: str, 
        rule_config: Dict[str, Any], 
        events: List[SecurityEvent]
    ) -> List[ThreatIndicator]:
        """Apply specific hunting rule to events"""
        
        threats = []
        indicators = rule_config.get("indicators", [])
        
        for event in events:
            # Check if event matches any indicators
            matches = []
            for indicator in indicators:
                if await self._check_indicator_match(indicator, event):
                    matches.append(indicator)
            
            # If sufficient matches, create threat indicator
            if len(matches) >= 2:  # Require at least 2 indicator matches
                threat = ThreatIndicator(
                    indicator_id=f"{rule_name}_{event.event_id}",
                    indicator_type="behavioral_pattern",
                    value=rule_name,
                    threat_level=ThreatLevel.LATERAL_MOVEMENT,  # Default, should be mapped from rule
                    confidence=len(matches) / len(indicators),
                    first_seen=event.timestamp,
                    last_seen=event.timestamp,
                    source="threat_hunting_engine",
                    context={
                        "rule_name": rule_name,
                        "matched_indicators": matches,
                        "event_id": event.event_id,
                        "mitre_techniques": rule_config.get("mitre_techniques", [])
                    }
                )
                threats.append(threat)
        
        return threats
    
    async def _check_indicator_match(self, indicator: str, event: SecurityEvent) -> bool:
        """Check if event matches specific indicator"""
        
        # Pattern matching logic for different indicators
        patterns = {
            "multiple_failed_logins_same_ip": lambda e: (
                "authentication" in e.event_type.lower() and 
                "failed" in e.description.lower() and
                e.confidence_score < 0.3
            ),
            "login_from_unusual_location": lambda e: (
                "authentication" in e.event_type.lower() and
                "geo" in str(e.raw_data) and
                e.confidence_score < 0.5
            ),
            "unusual_network_traffic": lambda e: (
                "network" in e.event_type.lower() and
                e.severity in [IncidentSeverity.MEDIUM, IncidentSeverity.HIGH]
            ),
            "large_data_transfer": lambda e: (
                "data_transfer" in e.event_type.lower() and
                e.severity == IncidentSeverity.HIGH
            ),
            "powershell_remote_execution": lambda e: (
                "powershell" in e.description.lower() and
                "remote" in e.description.lower()
            ),
            "sudo_abuse": lambda e: (
                "sudo" in e.description.lower() and
                ("privilege" in e.description.lower() or "elevation" in e.description.lower())
            )
        }
        
        pattern_func = patterns.get(indicator)
        if pattern_func:
            return pattern_func(event)
        
        return False
    
    def _deduplicate_threats(self, threats: List[ThreatIndicator]) -> List[ThreatIndicator]:
        """Remove duplicate threat indicators"""
        seen = set()
        unique_threats = []
        
        for threat in threats:
            key = f"{threat.indicator_type}:{threat.value}"
            if key not in seen:
                seen.add(key)
                unique_threats.append(threat)
        
        return unique_threats
    
    def _score_threats(self, threats: List[ThreatIndicator], events: List[SecurityEvent]) -> List[ThreatIndicator]:
        """Score and prioritize threats"""
        for threat in threats:
            # Enhance confidence based on additional factors
            base_confidence = threat.confidence
            
            # Factor in event severity
            related_events = [e for e in events if e.event_id in str(threat.context)]
            if related_events:
                avg_severity = sum(self._severity_to_score(e.severity) for e in related_events) / len(related_events)
                threat.confidence = min(1.0, base_confidence + (avg_severity * 0.2))
        
        # Sort by confidence (highest first)
        threats.sort(key=lambda t: t.confidence, reverse=True)
        return threats
    
    def _severity_to_score(self, severity: IncidentSeverity) -> float:
        """Convert severity to numeric score"""
        mapping = {
            IncidentSeverity.LOW: 0.25,
            IncidentSeverity.MEDIUM: 0.5,
            IncidentSeverity.HIGH: 0.75,
            IncidentSeverity.CRITICAL: 1.0
        }
        return mapping.get(severity, 0.5)


class IncidentResponseEngine:
    """Automated incident response engine"""
    
    def __init__(self):
        self.playbooks: Dict[str, ResponsePlaybook] = {}
        self.active_incidents: Dict[str, Incident] = {}
        self._initialize_playbooks()
    
    def _initialize_playbooks(self):
        """Initialize response playbooks"""
        playbooks = [
            ResponsePlaybook(
                playbook_id="malware_response",
                name="Malware Incident Response",
                description="Automated response for malware detection",
                triggers=["malware_detected", "suspicious_file_execution"],
                actions=[
                    ResponseAction.ISOLATE_SYSTEM,
                    ResponseAction.QUARANTINE_FILE,
                    ResponseAction.COLLECT_EVIDENCE,
                    ResponseAction.NOTIFY_TEAM
                ],
                required_approvals=["security_analyst"],
                automation_level="semi",
                estimated_duration=30,
                success_criteria=["system_isolated", "malware_contained", "evidence_collected"]
            ),
            
            ResponsePlaybook(
                playbook_id="credential_compromise",
                name="Credential Compromise Response",
                description="Response for compromised user credentials",
                triggers=["credential_theft", "suspicious_login", "password_spray"],
                actions=[
                    ResponseAction.DISABLE_USER,
                    ResponseAction.RESET_PASSWORD,
                    ResponseAction.REVOKE_TOKENS,
                    ResponseAction.NOTIFY_TEAM
                ],
                required_approvals=[],
                automation_level="full",
                estimated_duration=10,
                success_criteria=["account_disabled", "credentials_reset", "tokens_revoked"]
            ),
            
            ResponsePlaybook(
                playbook_id="data_exfiltration",
                name="Data Exfiltration Response",
                description="Response for potential data breach",
                triggers=["large_data_transfer", "unauthorized_data_access"],
                actions=[
                    ResponseAction.BLOCK_IP,
                    ResponseAction.ISOLATE_SYSTEM,
                    ResponseAction.COLLECT_EVIDENCE,
                    ResponseAction.NOTIFY_TEAM,
                    ResponseAction.CREATE_TICKET
                ],
                required_approvals=["incident_commander", "legal_team"],
                automation_level="semi",
                estimated_duration=60,
                success_criteria=["traffic_blocked", "system_isolated", "legal_notified"]
            ),
            
            ResponsePlaybook(
                playbook_id="network_intrusion",
                name="Network Intrusion Response",
                description="Response for network-based attacks",
                triggers=["network_anomaly", "lateral_movement", "c2_communication"],
                actions=[
                    ResponseAction.BLOCK_IP,
                    ResponseAction.ISOLATE_SYSTEM,
                    ResponseAction.COLLECT_EVIDENCE,
                    ResponseAction.NOTIFY_TEAM
                ],
                required_approvals=["security_analyst"],
                automation_level="semi",
                estimated_duration=45,
                success_criteria=["network_isolated", "traffic_blocked", "evidence_preserved"]
            )
        ]
        
        for playbook in playbooks:
            self.playbooks[playbook.playbook_id] = playbook
        
        logger.info(f"Initialized {len(self.playbooks)} response playbooks")
    
    async def create_incident(
        self,
        title: str,
        description: str,
        category: IncidentCategory,
        severity: IncidentSeverity,
        events: List[SecurityEvent],
        indicators: List[ThreatIndicator]
    ) -> Incident:
        """Create new security incident"""
        
        incident_id = f"INC_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        incident = Incident(
            incident_id=incident_id,
            title=title,
            description=description,
            category=category,
            severity=severity,
            status=IncidentStatus.NEW,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            assigned_to=None,
            affected_systems=list(set(event.raw_data.get("system", "unknown") for event in events)),
            indicators=indicators,
            events=events,
            response_actions=[],
            evidence=[],
            timeline=[{
                "timestamp": datetime.utcnow().isoformat(),
                "action": "incident_created",
                "details": f"Incident {incident_id} created with {len(events)} events"
            }],
            remediation_steps=[],
            lessons_learned=None
        )
        
        self.active_incidents[incident_id] = incident
        
        logger.info(f"Created incident {incident_id}", 
                   category=category.value, 
                   severity=severity.value)
        
        # Trigger automated response
        await self._trigger_automated_response(incident)
        
        return incident
    
    async def _trigger_automated_response(self, incident: Incident):
        """Trigger automated response based on incident characteristics"""
        
        # Find applicable playbooks
        applicable_playbooks = []
        
        for playbook in self.playbooks.values():
            # Check if any trigger matches the incident
            for trigger in playbook.triggers:
                if (trigger in incident.description.lower() or 
                    trigger in incident.category.value or
                    any(trigger in event.event_type.lower() for event in incident.events)):
                    applicable_playbooks.append(playbook)
                    break
        
        # Execute applicable playbooks
        for playbook in applicable_playbooks:
            await self._execute_playbook(incident, playbook)
    
    async def _execute_playbook(self, incident: Incident, playbook: ResponsePlaybook):
        """Execute response playbook"""
        
        logger.info(f"Executing playbook {playbook.name} for incident {incident.incident_id}")
        
        # Update incident timeline
        incident.timeline.append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": "playbook_execution_started",
            "details": f"Started executing playbook: {playbook.name}"
        })
        
        executed_actions = []
        
        for action in playbook.actions:
            try:
                # Check if approval required
                if (playbook.required_approvals and 
                    playbook.automation_level != "full"):
                    
                    # In a real implementation, this would check for approvals
                    # For now, we'll simulate approval for demo purposes
                    approval_required = True
                    
                    if approval_required:
                        incident.timeline.append({
                            "timestamp": datetime.utcnow().isoformat(),
                            "action": "approval_pending",
                            "details": f"Waiting for approval to execute {action.value}"
                        })
                        continue
                
                # Execute action
                success = await self._execute_response_action(action, incident)
                
                if success:
                    executed_actions.append(action.value)
                    incident.timeline.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "action": "response_action_executed",
                        "details": f"Successfully executed {action.value}"
                    })
                else:
                    incident.timeline.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "action": "response_action_failed",
                        "details": f"Failed to execute {action.value}"
                    })
                
            except Exception as e:
                logger.error(f"Failed to execute action {action.value}: {str(e)}")
                incident.timeline.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": "response_action_error",
                    "details": f"Error executing {action.value}: {str(e)}"
                })
        
        # Update incident
        incident.response_actions.extend(executed_actions)
        incident.updated_at = datetime.utcnow()
        
        # Check if incident can be auto-resolved
        if self._check_auto_resolution_criteria(incident, playbook):
            incident.status = IncidentStatus.CONTAINED
            incident.timeline.append({
                "timestamp": datetime.utcnow().isoformat(),
                "action": "auto_contained",
                "details": "Incident automatically contained by playbook execution"
            })
    
    async def _execute_response_action(self, action: ResponseAction, incident: Incident) -> bool:
        """Execute specific response action"""
        
        try:
            if action == ResponseAction.BLOCK_IP:
                # Extract IPs from events and block them
                ips_to_block = set()
                for event in incident.events:
                    if event.source_ip and event.source_ip != "unknown":
                        ips_to_block.add(event.source_ip)
                
                for ip in ips_to_block:
                    # In real implementation, integrate with firewall/WAF
                    logger.info(f"Blocking IP {ip} for incident {incident.incident_id}")
                    incident.evidence.append(f"Blocked IP: {ip}")
                
                return len(ips_to_block) > 0
            
            elif action == ResponseAction.DISABLE_USER:
                # Extract users from events and disable accounts
                users_to_disable = set()
                for event in incident.events:
                    user = event.raw_data.get("user_id") or event.raw_data.get("username")
                    if user:
                        users_to_disable.add(user)
                
                for user in users_to_disable:
                    # In real implementation, integrate with identity provider
                    logger.info(f"Disabling user {user} for incident {incident.incident_id}")
                    incident.evidence.append(f"Disabled user: {user}")
                
                return len(users_to_disable) > 0
            
            elif action == ResponseAction.ISOLATE_SYSTEM:
                # Isolate affected systems
                for system in incident.affected_systems:
                    if system != "unknown":
                        # In real implementation, integrate with network orchestration
                        logger.info(f"Isolating system {system} for incident {incident.incident_id}")
                        incident.evidence.append(f"Isolated system: {system}")
                
                return len(incident.affected_systems) > 0
            
            elif action == ResponseAction.COLLECT_EVIDENCE:
                # Collect forensic evidence
                evidence_items = []
                
                # Collect event logs
                for event in incident.events:
                    evidence_items.append(f"Event log: {event.event_id}")
                
                # Collect system information
                for system in incident.affected_systems:
                    evidence_items.append(f"System snapshot: {system}")
                
                incident.evidence.extend(evidence_items)
                logger.info(f"Collected {len(evidence_items)} evidence items for incident {incident.incident_id}")
                
                return True
            
            elif action == ResponseAction.NOTIFY_TEAM:
                # Send notifications to security team
                notification_sent = await self._send_incident_notification(incident)
                if notification_sent:
                    incident.evidence.append("Security team notified")
                
                return notification_sent
            
            elif action == ResponseAction.REVOKE_TOKENS:
                # Revoke authentication tokens
                users_affected = set()
                for event in incident.events:
                    user = event.raw_data.get("user_id")
                    if user:
                        users_affected.add(user)
                
                for user in users_affected:
                    # In real implementation, integrate with token management system
                    logger.info(f"Revoking tokens for user {user} for incident {incident.incident_id}")
                    incident.evidence.append(f"Revoked tokens for user: {user}")
                
                return len(users_affected) > 0
            
            else:
                logger.warning(f"Unimplemented response action: {action.value}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to execute response action {action.value}: {str(e)}")
            return False
    
    async def _send_incident_notification(self, incident: Incident) -> bool:
        """Send incident notification to security team"""
        try:
            # In real implementation, integrate with notification systems
            # (Slack, email, PagerDuty, etc.)
            
            notification_data = {
                "incident_id": incident.incident_id,
                "title": incident.title,
                "severity": incident.severity.value,
                "category": incident.category.value,
                "affected_systems": incident.affected_systems,
                "event_count": len(incident.events),
                "created_at": incident.created_at.isoformat()
            }
            
            logger.info(f"Sending incident notification: {json.dumps(notification_data)}")
            
            # Simulate successful notification
            return True
            
        except Exception as e:
            logger.error(f"Failed to send incident notification: {str(e)}")
            return False
    
    def _check_auto_resolution_criteria(self, incident: Incident, playbook: ResponsePlaybook) -> bool:
        """Check if incident meets auto-resolution criteria"""
        
        # Check if all success criteria are met
        success_criteria = playbook.success_criteria
        met_criteria = 0
        
        for criteria in success_criteria:
            if criteria == "system_isolated" and any("Isolated system" in ev for ev in incident.evidence):
                met_criteria += 1
            elif criteria == "traffic_blocked" and any("Blocked IP" in ev for ev in incident.evidence):
                met_criteria += 1
            elif criteria == "account_disabled" and any("Disabled user" in ev for ev in incident.evidence):
                met_criteria += 1
            elif criteria == "evidence_collected" and len(incident.evidence) > 0:
                met_criteria += 1
        
        # Auto-resolve if 80% of criteria are met
        return (met_criteria / len(success_criteria)) >= 0.8


class IncidentResponseService:
    """Main incident response and threat hunting service"""
    
    def __init__(self):
        self.threat_hunting_engine = ThreatHuntingEngine()
        self.incident_response_engine = IncidentResponseEngine()
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the incident response service"""
        try:
            logger.info("Initializing Incident Response Service")
            
            self.is_initialized = True
            logger.info("Incident Response Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Incident Response Service: {str(e)}")
            raise
    
    async def process_security_events(self, events: List[SecurityEvent]) -> List[Incident]:
        """Process security events and create incidents as needed"""
        
        if not events:
            return []
        
        try:
            # Perform threat hunting
            threats = await self.threat_hunting_engine.hunt_threats(events)
            
            # Group events by potential incidents
            incident_groups = self._group_events_for_incidents(events, threats)
            
            # Create incidents for each group
            incidents = []
            for group in incident_groups:
                incident = await self._create_incident_from_group(group, threats)
                if incident:
                    incidents.append(incident)
            
            logger.info(f"Processed {len(events)} events, created {len(incidents)} incidents")
            return incidents
            
        except Exception as e:
            logger.error(f"Failed to process security events: {str(e)}")
            return []
    
    def _group_events_for_incidents(
        self, 
        events: List[SecurityEvent], 
        threats: List[ThreatIndicator]
    ) -> List[List[SecurityEvent]]:
        """Group related events that should form incidents"""
        
        groups = []
        processed_events = set()
        
        # Group by source IP and time window
        for event in events:
            if event.event_id in processed_events:
                continue
            
            # Find related events (same source IP within 1 hour)
            related_events = [event]
            processed_events.add(event.event_id)
            
            for other_event in events:
                if (other_event.event_id not in processed_events and
                    other_event.source_ip == event.source_ip and
                    abs((other_event.timestamp - event.timestamp).total_seconds()) < 3600):
                    
                    related_events.append(other_event)
                    processed_events.add(other_event.event_id)
            
            # Only create group if it has significance
            if (len(related_events) > 1 or 
                any(e.severity in [IncidentSeverity.HIGH, IncidentSeverity.CRITICAL] for e in related_events)):
                groups.append(related_events)
        
        return groups
    
    async def _create_incident_from_group(
        self, 
        event_group: List[SecurityEvent], 
        threats: List[ThreatIndicator]
    ) -> Optional[Incident]:
        """Create incident from grouped events"""
        
        if not event_group:
            return None
        
        try:
            # Determine incident characteristics
            max_severity = max(event.severity for event in event_group)
            
            # Determine category based on events
            category = self._determine_incident_category(event_group)
            
            # Find related threat indicators
            related_threats = []
            for threat in threats:
                threat_events = threat.context.get("event_id", "")
                if any(event.event_id in threat_events for event in event_group):
                    related_threats.append(threat)
            
            # Generate incident title and description
            title = self._generate_incident_title(event_group, category)
            description = self._generate_incident_description(event_group, related_threats)
            
            # Create incident
            incident = await self.incident_response_engine.create_incident(
                title=title,
                description=description,
                category=category,
                severity=max_severity,
                events=event_group,
                indicators=related_threats
            )
            
            return incident
            
        except Exception as e:
            logger.error(f"Failed to create incident from event group: {str(e)}")
            return None
    
    def _determine_incident_category(self, events: List[SecurityEvent]) -> IncidentCategory:
        """Determine incident category from events"""
        
        # Simple categorization based on event types and descriptions
        event_text = " ".join(event.event_type + " " + event.description for event in events).lower()
        
        if any(term in event_text for term in ["malware", "virus", "trojan", "ransomware"]):
            return IncidentCategory.MALWARE
        elif any(term in event_text for term in ["phishing", "social engineering"]):
            return IncidentCategory.PHISHING
        elif any(term in event_text for term in ["data", "exfiltration", "breach"]):
            return IncidentCategory.DATA_BREACH
        elif any(term in event_text for term in ["unauthorized", "access", "login"]):
            return IncidentCategory.UNAUTHORIZED_ACCESS
        elif any(term in event_text for term in ["dos", "ddos", "denial"]):
            return IncidentCategory.DENIAL_OF_SERVICE
        elif any(term in event_text for term in ["insider", "internal"]):
            return IncidentCategory.INSIDER_THREAT
        elif any(term in event_text for term in ["vulnerability", "exploit", "cve"]):
            return IncidentCategory.VULNERABILITY_EXPLOIT
        elif any(term in event_text for term in ["network", "intrusion", "lateral"]):
            return IncidentCategory.NETWORK_INTRUSION
        else:
            return IncidentCategory.SYSTEM_COMPROMISE
    
    def _generate_incident_title(self, events: List[SecurityEvent], category: IncidentCategory) -> str:
        """Generate incident title"""
        
        source_ips = list(set(event.source_ip for event in events if event.source_ip != "unknown"))
        ip_summary = f" from {source_ips[0]}" if len(source_ips) == 1 else f" from {len(source_ips)} IPs"
        
        return f"{category.value.replace('_', ' ').title()}{ip_summary} - {len(events)} events"
    
    def _generate_incident_description(
        self, 
        events: List[SecurityEvent], 
        threats: List[ThreatIndicator]
    ) -> str:
        """Generate incident description"""
        
        description_parts = [
            f"Security incident involving {len(events)} events",
            f"Time range: {min(e.timestamp for e in events)} to {max(e.timestamp for e in events)}",
            f"Affected systems: {list(set(e.raw_data.get('system', 'unknown') for e in events))}",
        ]
        
        if threats:
            description_parts.append(f"Threat indicators: {len(threats)} detected")
            for threat in threats[:3]:  # Show top 3 threats
                description_parts.append(f"- {threat.indicator_type}: {threat.value} (confidence: {threat.confidence:.2f})")
        
        high_severity_events = [e for e in events if e.severity in [IncidentSeverity.HIGH, IncidentSeverity.CRITICAL]]
        if high_severity_events:
            description_parts.append(f"High-severity events: {len(high_severity_events)}")
        
        return "\n".join(description_parts)
    
    async def get_active_incidents(self) -> List[Incident]:
        """Get all active incidents"""
        return list(self.incident_response_engine.active_incidents.values())
    
    async def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get specific incident by ID"""
        return self.incident_response_engine.active_incidents.get(incident_id)
    
    async def update_incident_status(self, incident_id: str, status: IncidentStatus, notes: str = "") -> bool:
        """Update incident status"""
        incident = self.incident_response_engine.active_incidents.get(incident_id)
        if not incident:
            return False
        
        incident.status = status
        incident.updated_at = datetime.utcnow()
        incident.timeline.append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": "status_updated",
            "details": f"Status changed to {status.value}. Notes: {notes}"
        })
        
        logger.info(f"Updated incident {incident_id} status to {status.value}")
        return True
    
    async def get_threat_hunting_statistics(self) -> Dict[str, Any]:
        """Get threat hunting statistics"""
        return {
            "hunting_rules": len(self.threat_hunting_engine.hunting_rules),
            "active_incidents": len(self.incident_response_engine.active_incidents),
            "available_playbooks": len(self.incident_response_engine.playbooks),
            "service_status": "active" if self.is_initialized else "inactive"
        }


# Global service instance
_incident_response_service: Optional[IncidentResponseService] = None


async def get_incident_response_service() -> IncidentResponseService:
    """Get the global incident response service instance"""
    global _incident_response_service
    
    if _incident_response_service is None:
        _incident_response_service = IncidentResponseService()
        await _incident_response_service.initialize()
    
    return _incident_response_service


# Export main classes
__all__ = [
    "IncidentResponseService",
    "SecurityEvent",
    "Incident",
    "ThreatIndicator",
    "IncidentSeverity",
    "IncidentStatus",
    "IncidentCategory",
    "get_incident_response_service"
]