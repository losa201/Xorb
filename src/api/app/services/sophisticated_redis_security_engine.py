"""
XORB Sophisticated Redis Security Engine - Advanced Security for Redis Infrastructure
Implements comprehensive security monitoring, threat detection, and protection for Redis operations
"""

import asyncio
import logging
import json
import time
import hashlib
import hmac
import secrets
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import ipaddress
import re
from collections import defaultdict, deque
import statistics

from .interfaces import SecurityService
from .base_service import XORBService, ServiceType
from ..infrastructure.advanced_redis_orchestrator import AdvancedRedisOrchestrator, get_redis_orchestrator

logger = logging.getLogger(__name__)


class SecurityThreatLevel(Enum):
    """Security threat levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events"""
    AUTHENTICATION_FAILURE = "auth_failure"
    SUSPICIOUS_ACCESS_PATTERN = "suspicious_access"
    RATE_LIMIT_VIOLATION = "rate_limit_violation"
    INJECTION_ATTEMPT = "injection_attempt"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    UNAUTHORIZED_COMMAND = "unauthorized_command"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    BRUTE_FORCE_ATTACK = "brute_force"
    DDOS_ATTEMPT = "ddos_attempt"


class SecurityAction(Enum):
    """Security response actions"""
    LOG_ONLY = "log_only"
    RATE_LIMIT = "rate_limit"
    TEMPORARY_BLOCK = "temporary_block"
    PERMANENT_BLOCK = "permanent_block"
    ALERT_ADMIN = "alert_admin"
    EMERGENCY_LOCKDOWN = "emergency_lockdown"
    FORCE_DISCONNECT = "force_disconnect"
    ESCALATE_TO_SOC = "escalate_to_soc"


@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_id: str
    event_type: SecurityEventType
    threat_level: SecurityThreatLevel
    timestamp: float
    source_ip: str
    user_id: Optional[str]
    redis_command: Optional[str]
    affected_keys: List[str]
    details: Dict[str, Any]
    risk_score: float
    mitigation_actions: List[SecurityAction]
    investigation_status: str = "pending"


@dataclass
class SecurityRule:
    """Security rule configuration"""
    rule_id: str
    name: str
    description: str
    rule_type: str
    pattern: str
    threat_level: SecurityThreatLevel
    enabled: bool
    actions: List[SecurityAction]
    threshold: Optional[float] = None
    time_window_seconds: Optional[int] = None
    whitelist_ips: List[str] = None
    custom_logic: Optional[str] = None


@dataclass
class SecurityProfile:
    """Security profile for users/applications"""
    profile_id: str
    name: str
    allowed_commands: Set[str]
    denied_commands: Set[str]
    max_keys_per_command: int
    max_commands_per_minute: int
    max_data_size_mb: int
    allowed_key_patterns: List[str]
    denied_key_patterns: List[str]
    network_restrictions: Dict[str, Any]
    time_restrictions: Dict[str, Any]


@dataclass
class ThreatIntelligence:
    """Threat intelligence data"""
    ip_address: str
    threat_type: str
    confidence_score: float
    first_seen: float
    last_seen: float
    attack_patterns: List[str]
    geolocation: Dict[str, str]
    reputation_score: float
    sources: List[str]


class SophisticatedRedisSecurityEngine(SecurityService, XORBService):
    """Sophisticated security engine for Redis infrastructure protection"""

    def __init__(self, orchestrator: AdvancedRedisOrchestrator):
        super().__init__(service_type=ServiceType.SECURITY)
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(__name__)

        # Security state management
        self.security_events: deque = deque(maxlen=10000)
        self.active_threats: Dict[str, ThreatIntelligence] = {}
        self.blocked_ips: Set[str] = set()
        self.rate_limited_clients: Dict[str, float] = {}

        # Security rules and profiles
        self.security_rules: Dict[str, SecurityRule] = {}
        self.security_profiles: Dict[str, SecurityProfile] = {}
        self.default_profile: SecurityProfile = self._create_default_security_profile()

        # Monitoring and analysis
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.command_statistics: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.ip_statistics: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Security configuration
        self.config = {
            "enable_threat_detection": True,
            "enable_behavioral_analysis": True,
            "enable_real_time_blocking": True,
            "enable_forensic_logging": True,
            "anomaly_detection_sensitivity": 0.8,
            "threat_intelligence_update_interval": 3600,
            "security_scan_interval": 300,
            "max_login_attempts": 5,
            "lockout_duration_minutes": 30,
            "emergency_lockdown_threshold": 10,
            "data_loss_prevention": True,
            "command_audit_logging": True
        }

        # Initialize default security rules
        self._initialize_default_security_rules()

        # Monitoring state
        self.is_monitoring_active = False
        self.monitoring_tasks: List[asyncio.Task] = []

        # Encryption and security keys
        self.security_keys = self._generate_security_keys()

        # Threat intelligence feeds
        self.threat_feeds: Dict[str, Any] = {}

        # Forensic data retention
        self.forensic_buffer: deque = deque(maxlen=50000)

    async def initialize(self) -> bool:
        """Initialize Redis security engine"""
        try:
            self.logger.info("Initializing Redis Security Engine")

            # Load security configuration
            await self._load_security_configuration()

            # Initialize threat intelligence
            await self._initialize_threat_intelligence()

            # Start security monitoring
            await self._start_security_monitoring()

            # Set up command monitoring hooks
            await self._setup_command_monitoring()

            self.logger.info("Redis Security Engine initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Redis Security Engine: {e}")
            return False

    def _create_default_security_profile(self) -> SecurityProfile:
        """Create default security profile"""
        return SecurityProfile(
            profile_id="default",
            name="Default Security Profile",
            allowed_commands={
                "GET", "SET", "MGET", "MSET", "EXISTS", "DEL", "EXPIRE", "TTL",
                "INCR", "DECR", "LPUSH", "RPOP", "LLEN", "HGET", "HSET", "HGETALL"
            },
            denied_commands={
                "FLUSHDB", "FLUSHALL", "CONFIG", "EVAL", "SCRIPT", "DEBUG",
                "SHUTDOWN", "CLIENT", "MONITOR", "LATENCY"
            },
            max_keys_per_command=100,
            max_commands_per_minute=1000,
            max_data_size_mb=10,
            allowed_key_patterns=["user:*", "session:*", "cache:*", "temp:*"],
            denied_key_patterns=["admin:*", "system:*", "security:*"],
            network_restrictions={
                "allowed_networks": ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"],
                "blocked_countries": ["CN", "RU", "KP"],
                "require_vpn": False
            },
            time_restrictions={
                "allowed_hours": list(range(24)),  # 24/7 by default
                "timezone": "UTC",
                "maintenance_windows": []
            }
        )

    def _initialize_default_security_rules(self):
        """Initialize default security rules"""
        default_rules = [
            SecurityRule(
                rule_id="auth_failure_detector",
                name="Authentication Failure Detection",
                description="Detect multiple authentication failures",
                rule_type="threshold",
                pattern="auth_failure",
                threat_level=SecurityThreatLevel.MEDIUM,
                enabled=True,
                actions=[SecurityAction.RATE_LIMIT, SecurityAction.ALERT_ADMIN],
                threshold=5,
                time_window_seconds=300
            ),
            SecurityRule(
                rule_id="injection_detector",
                name="Redis Injection Detection",
                description="Detect Redis command injection attempts",
                rule_type="pattern",
                pattern=r"(EVAL|SCRIPT|LUA|CONFIG|DEBUG|SHUTDOWN)",
                threat_level=SecurityThreatLevel.HIGH,
                enabled=True,
                actions=[SecurityAction.TEMPORARY_BLOCK, SecurityAction.ALERT_ADMIN]
            ),
            SecurityRule(
                rule_id="rate_limit_violation",
                name="Rate Limit Violation",
                description="Detect rate limit violations",
                rule_type="threshold",
                pattern="rate_limit_exceeded",
                threat_level=SecurityThreatLevel.MEDIUM,
                enabled=True,
                actions=[SecurityAction.RATE_LIMIT],
                threshold=10,
                time_window_seconds=60
            ),
            SecurityRule(
                rule_id="suspicious_key_access",
                name="Suspicious Key Access",
                description="Detect access to sensitive key patterns",
                rule_type="pattern",
                pattern=r"(admin|system|security|config|password|secret)",
                threat_level=SecurityThreatLevel.HIGH,
                enabled=True,
                actions=[SecurityAction.LOG_ONLY, SecurityAction.ALERT_ADMIN]
            ),
            SecurityRule(
                rule_id="data_exfiltration_detector",
                name="Data Exfiltration Detection",
                description="Detect potential data exfiltration attempts",
                rule_type="threshold",
                pattern="large_data_access",
                threat_level=SecurityThreatLevel.HIGH,
                enabled=True,
                actions=[SecurityAction.TEMPORARY_BLOCK, SecurityAction.ESCALATE_TO_SOC],
                threshold=100,  # MB
                time_window_seconds=300
            ),
            SecurityRule(
                rule_id="brute_force_detector",
                name="Brute Force Attack Detection",
                description="Detect brute force attacks",
                rule_type="threshold",
                pattern="rapid_commands",
                threat_level=SecurityThreatLevel.HIGH,
                enabled=True,
                actions=[SecurityAction.TEMPORARY_BLOCK, SecurityAction.ALERT_ADMIN],
                threshold=1000,  # Commands per minute
                time_window_seconds=60
            )
        ]

        for rule in default_rules:
            self.security_rules[rule.rule_id] = rule

    def _generate_security_keys(self) -> Dict[str, str]:
        """Generate security keys for encryption and authentication"""
        return {
            "encryption_key": secrets.token_hex(32),
            "signing_key": secrets.token_hex(32),
            "session_key": secrets.token_hex(16),
            "audit_key": secrets.token_hex(24)
        }

    async def _load_security_configuration(self):
        """Load security configuration from Redis"""
        try:
            client = await self.orchestrator.get_optimal_client("read")

            # Load security rules
            rules_data = await client.get("security:rules")
            if rules_data:
                rules = json.loads(rules_data)
                for rule_data in rules:
                    rule = SecurityRule(**rule_data)
                    self.security_rules[rule.rule_id] = rule

            # Load security profiles
            profiles_data = await client.get("security:profiles")
            if profiles_data:
                profiles = json.loads(profiles_data)
                for profile_data in profiles:
                    profile = SecurityProfile(**profile_data)
                    self.security_profiles[profile.profile_id] = profile

            # Load blocked IPs
            blocked_ips_data = await client.get("security:blocked_ips")
            if blocked_ips_data:
                self.blocked_ips = set(json.loads(blocked_ips_data))

            self.logger.info(f"Loaded {len(self.security_rules)} security rules and {len(self.security_profiles)} profiles")

        except Exception as e:
            self.logger.error(f"Failed to load security configuration: {e}")

    async def _initialize_threat_intelligence(self):
        """Initialize threat intelligence feeds"""
        try:
            # Load threat intelligence data
            client = await self.orchestrator.get_optimal_client("read")
            threat_data = await client.get("security:threat_intelligence")

            if threat_data:
                threats = json.loads(threat_data)
                for threat_info in threats:
                    threat = ThreatIntelligence(**threat_info)
                    self.active_threats[threat.ip_address] = threat

            # Initialize threat feeds (placeholder for external feeds)
            self.threat_feeds = {
                "malicious_ips": set(),
                "tor_exit_nodes": set(),
                "known_botnets": set(),
                "suspicious_user_agents": set()
            }

            self.logger.info(f"Initialized threat intelligence with {len(self.active_threats)} known threats")

        except Exception as e:
            self.logger.error(f"Failed to initialize threat intelligence: {e}")

    async def _start_security_monitoring(self):
        """Start security monitoring tasks"""
        self.is_monitoring_active = True

        # Real-time threat detection
        threat_task = asyncio.create_task(self._threat_detection_loop())
        self.monitoring_tasks.append(threat_task)

        # Behavioral analysis
        behavior_task = asyncio.create_task(self._behavioral_analysis_loop())
        self.monitoring_tasks.append(behavior_task)

        # Security scanning
        scan_task = asyncio.create_task(self._security_scan_loop())
        self.monitoring_tasks.append(scan_task)

        # Threat intelligence updates
        intel_task = asyncio.create_task(self._threat_intelligence_update_loop())
        self.monitoring_tasks.append(intel_task)

        # Forensic data management
        forensic_task = asyncio.create_task(self._forensic_data_management_loop())
        self.monitoring_tasks.append(forensic_task)

        self.logger.info("Started security monitoring tasks")

    async def _setup_command_monitoring(self):
        """Setup command monitoring hooks"""
        try:
            # Subscribe to Redis command events (if monitoring is enabled)
            await self.orchestrator.pubsub_coordinator.subscribe(
                "redis_commands",
                self._handle_redis_command_event
            )

            self.logger.info("Command monitoring hooks established")

        except Exception as e:
            self.logger.error(f"Failed to setup command monitoring: {e}")

    async def _handle_redis_command_event(self, event_data: Dict[str, Any]):
        """Handle Redis command events for security analysis"""
        try:
            # Extract event information
            command = event_data.get("command", "").upper()
            client_ip = event_data.get("client_ip", "unknown")
            user_id = event_data.get("user_id")
            keys = event_data.get("keys", [])
            data_size = event_data.get("data_size", 0)
            timestamp = event_data.get("timestamp", time.time())

            # Update statistics
            self._update_command_statistics(client_ip, command, data_size)

            # Check security rules
            await self._evaluate_security_rules(
                command, client_ip, user_id, keys, data_size, timestamp
            )

            # Behavioral analysis
            await self._analyze_command_behavior(
                command, client_ip, user_id, timestamp
            )

        except Exception as e:
            self.logger.error(f"Error handling command event: {e}")

    def _update_command_statistics(self, client_ip: str, command: str, data_size: int):
        """Update command statistics for analysis"""
        current_minute = int(time.time() // 60)

        # Update IP statistics
        if client_ip not in self.ip_statistics:
            self.ip_statistics[client_ip] = {
                "command_count": defaultdict(int),
                "total_commands": 0,
                "data_transferred": 0,
                "first_seen": time.time(),
                "last_seen": time.time(),
                "minute_buckets": defaultdict(int)
            }

        stats = self.ip_statistics[client_ip]
        stats["command_count"][command] += 1
        stats["total_commands"] += 1
        stats["data_transferred"] += data_size
        stats["last_seen"] = time.time()
        stats["minute_buckets"][current_minute] += 1

        # Update global command statistics
        self.command_statistics[command]["total"] += 1
        self.command_statistics[command]["minute_buckets"][current_minute] += 1

    async def _evaluate_security_rules(
        self,
        command: str,
        client_ip: str,
        user_id: Optional[str],
        keys: List[str],
        data_size: int,
        timestamp: float
    ):
        """Evaluate security rules against command"""
        try:
            for rule in self.security_rules.values():
                if not rule.enabled:
                    continue

                # Check if rule matches
                if await self._rule_matches(rule, command, client_ip, user_id, keys, data_size):
                    # Generate security event
                    event = SecurityEvent(
                        event_id=f"evt_{int(timestamp)}_{secrets.token_hex(4)}",
                        event_type=self._get_event_type_from_rule(rule),
                        threat_level=rule.threat_level,
                        timestamp=timestamp,
                        source_ip=client_ip,
                        user_id=user_id,
                        redis_command=command,
                        affected_keys=keys,
                        details={
                            "rule_id": rule.rule_id,
                            "rule_name": rule.name,
                            "data_size": data_size,
                            "pattern_matched": rule.pattern
                        },
                        risk_score=self._calculate_risk_score(rule, client_ip, command),
                        mitigation_actions=rule.actions
                    )

                    # Process security event
                    await self._process_security_event(event)

        except Exception as e:
            self.logger.error(f"Error evaluating security rules: {e}")

    async def _rule_matches(
        self,
        rule: SecurityRule,
        command: str,
        client_ip: str,
        user_id: Optional[str],
        keys: List[str],
        data_size: int
    ) -> bool:
        """Check if security rule matches current command"""
        try:
            if rule.rule_type == "pattern":
                # Pattern-based rule
                if re.search(rule.pattern, command, re.IGNORECASE):
                    return True

                # Check key patterns
                for key in keys:
                    if re.search(rule.pattern, key, re.IGNORECASE):
                        return True

            elif rule.rule_type == "threshold":
                # Threshold-based rule
                return await self._check_threshold_rule(rule, client_ip, command, data_size)

            elif rule.rule_type == "custom":
                # Custom logic rule (placeholder for advanced rules)
                return await self._evaluate_custom_rule(rule, command, client_ip, user_id, keys, data_size)

            return False

        except Exception as e:
            self.logger.error(f"Error checking rule match: {e}")
            return False

    async def _check_threshold_rule(
        self,
        rule: SecurityRule,
        client_ip: str,
        command: str,
        data_size: int
    ) -> bool:
        """Check threshold-based security rule"""
        try:
            if not rule.threshold or not rule.time_window_seconds:
                return False

            current_time = time.time()
            window_start = current_time - rule.time_window_seconds

            if rule.pattern == "auth_failure":
                # Check authentication failures
                auth_failures = self._count_events_in_window(
                    client_ip,
                    SecurityEventType.AUTHENTICATION_FAILURE,
                    window_start
                )
                return auth_failures >= rule.threshold

            elif rule.pattern == "rate_limit_exceeded":
                # Check rate limit violations
                if client_ip in self.ip_statistics:
                    stats = self.ip_statistics[client_ip]
                    commands_in_window = sum(
                        count for minute, count in stats["minute_buckets"].items()
                        if minute * 60 >= window_start
                    )
                    return commands_in_window >= rule.threshold

            elif rule.pattern == "large_data_access":
                # Check data exfiltration (data size in MB)
                if client_ip in self.ip_statistics:
                    stats = self.ip_statistics[client_ip]
                    data_mb = stats["data_transferred"] / (1024 * 1024)
                    return data_mb >= rule.threshold

            elif rule.pattern == "rapid_commands":
                # Check for rapid command execution
                if client_ip in self.ip_statistics:
                    stats = self.ip_statistics[client_ip]
                    current_minute = int(current_time // 60)
                    commands_this_minute = stats["minute_buckets"][current_minute]
                    return commands_this_minute >= rule.threshold

            return False

        except Exception as e:
            self.logger.error(f"Error checking threshold rule: {e}")
            return False

    def _count_events_in_window(
        self,
        client_ip: str,
        event_type: SecurityEventType,
        window_start: float
    ) -> int:
        """Count security events in time window"""
        count = 0
        for event in self.security_events:
            if (event.source_ip == client_ip and
                event.event_type == event_type and
                event.timestamp >= window_start):
                count += 1
        return count

    async def _evaluate_custom_rule(
        self,
        rule: SecurityRule,
        command: str,
        client_ip: str,
        user_id: Optional[str],
        keys: List[str],
        data_size: int
    ) -> bool:
        """Evaluate custom security rule logic"""
        # Placeholder for custom rule evaluation
        # In practice, this would execute custom logic defined in rule.custom_logic
        return False

    def _get_event_type_from_rule(self, rule: SecurityRule) -> SecurityEventType:
        """Get security event type from rule"""
        if "auth" in rule.name.lower():
            return SecurityEventType.AUTHENTICATION_FAILURE
        elif "injection" in rule.name.lower():
            return SecurityEventType.INJECTION_ATTEMPT
        elif "rate" in rule.name.lower():
            return SecurityEventType.RATE_LIMIT_VIOLATION
        elif "exfiltration" in rule.name.lower():
            return SecurityEventType.DATA_EXFILTRATION
        elif "brute" in rule.name.lower():
            return SecurityEventType.BRUTE_FORCE_ATTACK
        else:
            return SecurityEventType.SUSPICIOUS_ACCESS_PATTERN

    def _calculate_risk_score(self, rule: SecurityRule, client_ip: str, command: str) -> float:
        """Calculate risk score for security event"""
        base_score = 0.5

        # Threat level modifier
        threat_modifiers = {
            SecurityThreatLevel.LOW: 0.2,
            SecurityThreatLevel.MEDIUM: 0.5,
            SecurityThreatLevel.HIGH: 0.8,
            SecurityThreatLevel.CRITICAL: 1.0
        }
        base_score *= threat_modifiers[rule.threat_level]

        # IP reputation modifier
        if client_ip in self.active_threats:
            threat = self.active_threats[client_ip]
            base_score *= (1 + threat.confidence_score)

        # Command risk modifier
        dangerous_commands = {"EVAL", "SCRIPT", "CONFIG", "DEBUG", "SHUTDOWN", "FLUSHDB", "FLUSHALL"}
        if command in dangerous_commands:
            base_score *= 1.5

        # Historical behavior modifier
        if client_ip in self.ip_statistics:
            stats = self.ip_statistics[client_ip]
            if stats["total_commands"] > 10000:  # High activity
                base_score *= 1.2

        return min(1.0, base_score)

    async def _process_security_event(self, event: SecurityEvent):
        """Process security event and execute mitigation actions"""
        try:
            # Add to event history
            self.security_events.append(event)

            # Add to forensic buffer
            self.forensic_buffer.append({
                "timestamp": event.timestamp,
                "event_type": event.event_type.value,
                "source_ip": event.source_ip,
                "command": event.redis_command,
                "keys": event.affected_keys,
                "risk_score": event.risk_score,
                "details": event.details
            })

            # Log security event
            self.logger.warning(f"Security event: {event.event_type.value} from {event.source_ip} "
                              f"(Risk: {event.risk_score:.2f})")

            # Execute mitigation actions
            await self._execute_mitigation_actions(event)

            # Send security alert
            await self._send_security_alert(event)

        except Exception as e:
            self.logger.error(f"Error processing security event: {e}")

    async def _execute_mitigation_actions(self, event: SecurityEvent):
        """Execute mitigation actions for security event"""
        try:
            for action in event.mitigation_actions:
                if action == SecurityAction.LOG_ONLY:
                    # Already logged
                    continue

                elif action == SecurityAction.RATE_LIMIT:
                    await self._apply_rate_limit(event.source_ip)

                elif action == SecurityAction.TEMPORARY_BLOCK:
                    await self._temporary_block_ip(event.source_ip, duration_minutes=30)

                elif action == SecurityAction.PERMANENT_BLOCK:
                    await self._permanent_block_ip(event.source_ip)

                elif action == SecurityAction.FORCE_DISCONNECT:
                    await self._force_disconnect_client(event.source_ip)

                elif action == SecurityAction.EMERGENCY_LOCKDOWN:
                    await self._emergency_lockdown()

                elif action == SecurityAction.ALERT_ADMIN:
                    await self._alert_administrator(event)

                elif action == SecurityAction.ESCALATE_TO_SOC:
                    await self._escalate_to_soc(event)

        except Exception as e:
            self.logger.error(f"Error executing mitigation actions: {e}")

    async def _apply_rate_limit(self, client_ip: str):
        """Apply rate limiting to client IP"""
        self.rate_limited_clients[client_ip] = time.time() + 300  # 5 minutes
        self.logger.info(f"Applied rate limiting to {client_ip}")

    async def _temporary_block_ip(self, client_ip: str, duration_minutes: int = 30):
        """Temporarily block IP address"""
        try:
            client = await self.orchestrator.get_optimal_client("write")

            # Add to temporary block list
            block_key = f"security:temp_block:{client_ip}"
            await client.setex(block_key, duration_minutes * 60, "blocked")

            # Add to blocked IPs set
            self.blocked_ips.add(client_ip)

            self.logger.warning(f"Temporarily blocked IP {client_ip} for {duration_minutes} minutes")

        except Exception as e:
            self.logger.error(f"Error blocking IP {client_ip}: {e}")

    async def _permanent_block_ip(self, client_ip: str):
        """Permanently block IP address"""
        try:
            client = await self.orchestrator.get_optimal_client("write")

            # Add to permanent block list
            self.blocked_ips.add(client_ip)
            await client.sadd("security:permanent_blocks", client_ip)

            self.logger.critical(f"Permanently blocked IP {client_ip}")

        except Exception as e:
            self.logger.error(f"Error permanently blocking IP {client_ip}: {e}")

    async def _force_disconnect_client(self, client_ip: str):
        """Force disconnect Redis client"""
        try:
            # This would require Redis client tracking and management
            # Implementation depends on Redis client library capabilities
            self.logger.info(f"Force disconnect requested for {client_ip}")

        except Exception as e:
            self.logger.error(f"Error forcing disconnect for {client_ip}: {e}")

    async def _emergency_lockdown(self):
        """Trigger emergency lockdown"""
        try:
            client = await self.orchestrator.get_optimal_client("write")

            # Set emergency lockdown flag
            await client.setex("security:emergency_lockdown", 3600, "active")

            # Block all new connections temporarily
            await client.config_set("maxclients", "0")

            self.logger.critical("EMERGENCY LOCKDOWN ACTIVATED")

            # Send critical alert
            await self.orchestrator.pubsub_coordinator.publish(
                "security_alerts",
                {
                    "type": "emergency_lockdown",
                    "timestamp": time.time(),
                    "message": "Emergency security lockdown activated"
                }
            )

        except Exception as e:
            self.logger.error(f"Error during emergency lockdown: {e}")

    async def _alert_administrator(self, event: SecurityEvent):
        """Send alert to administrator"""
        try:
            alert_data = {
                "type": "security_alert",
                "event_id": event.event_id,
                "threat_level": event.threat_level.value,
                "source_ip": event.source_ip,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp,
                "risk_score": event.risk_score,
                "details": event.details
            }

            await self.orchestrator.pubsub_coordinator.publish(
                "admin_alerts",
                alert_data
            )

        except Exception as e:
            self.logger.error(f"Error sending admin alert: {e}")

    async def _escalate_to_soc(self, event: SecurityEvent):
        """Escalate security event to SOC"""
        try:
            soc_alert = {
                "type": "soc_escalation",
                "event_id": event.event_id,
                "threat_level": event.threat_level.value,
                "source_ip": event.source_ip,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp,
                "risk_score": event.risk_score,
                "affected_systems": ["redis_cluster"],
                "investigation_priority": "high" if event.risk_score > 0.7 else "medium",
                "details": event.details,
                "recommended_actions": [action.value for action in event.mitigation_actions]
            }

            await self.orchestrator.pubsub_coordinator.publish(
                "soc_escalations",
                soc_alert
            )

            self.logger.critical(f"Escalated security event {event.event_id} to SOC")

        except Exception as e:
            self.logger.error(f"Error escalating to SOC: {e}")

    async def _send_security_alert(self, event: SecurityEvent):
        """Send security alert via pub/sub"""
        try:
            alert = {
                "event_id": event.event_id,
                "timestamp": event.timestamp,
                "threat_level": event.threat_level.value,
                "event_type": event.event_type.value,
                "source_ip": event.source_ip,
                "risk_score": event.risk_score,
                "summary": f"{event.event_type.value} detected from {event.source_ip}"
            }

            await self.orchestrator.pubsub_coordinator.publish(
                "security_events",
                alert
            )

        except Exception as e:
            self.logger.error(f"Error sending security alert: {e}")

    async def _threat_detection_loop(self):
        """Continuous threat detection loop"""
        while self.is_monitoring_active:
            try:
                # Analyze current threats
                await self._analyze_active_threats()

                # Update threat intelligence
                await self._update_threat_scores()

                # Check for new threat patterns
                await self._detect_new_threat_patterns()

                await asyncio.sleep(60)  # Run every minute

            except Exception as e:
                self.logger.error(f"Threat detection loop error: {e}")
                await asyncio.sleep(30)

    async def _behavioral_analysis_loop(self):
        """Continuous behavioral analysis loop"""
        while self.is_monitoring_active:
            try:
                # Analyze IP behavior patterns
                await self._analyze_ip_behavior()

                # Detect anomalous command patterns
                await self._detect_command_anomalies()

                # Update behavioral baselines
                await self._update_behavioral_baselines()

                await asyncio.sleep(300)  # Run every 5 minutes

            except Exception as e:
                self.logger.error(f"Behavioral analysis loop error: {e}")
                await asyncio.sleep(60)

    async def _security_scan_loop(self):
        """Continuous security scanning loop"""
        while self.is_monitoring_active:
            try:
                # Scan for security misconfigurations
                await self._scan_security_configuration()

                # Check for unauthorized access attempts
                await self._scan_unauthorized_access()

                # Validate security policies
                await self._validate_security_policies()

                await asyncio.sleep(self.config["security_scan_interval"])

            except Exception as e:
                self.logger.error(f"Security scan loop error: {e}")
                await asyncio.sleep(60)

    async def _threat_intelligence_update_loop(self):
        """Threat intelligence update loop"""
        while self.is_monitoring_active:
            try:
                # Update threat feeds (placeholder)
                await self._update_threat_feeds()

                # Correlate with external threat intelligence
                await self._correlate_external_threats()

                # Clean up old threat data
                await self._cleanup_old_threats()

                await asyncio.sleep(self.config["threat_intelligence_update_interval"])

            except Exception as e:
                self.logger.error(f"Threat intelligence update error: {e}")
                await asyncio.sleep(300)

    async def _forensic_data_management_loop(self):
        """Forensic data management loop"""
        while self.is_monitoring_active:
            try:
                # Archive old forensic data
                await self._archive_forensic_data()

                # Generate forensic reports
                await self._generate_forensic_reports()

                # Maintain chain of custody
                await self._maintain_chain_of_custody()

                await asyncio.sleep(3600)  # Run every hour

            except Exception as e:
                self.logger.error(f"Forensic data management error: {e}")
                await asyncio.sleep(600)

    async def _analyze_active_threats(self):
        """Analyze currently active threats"""
        try:
            current_time = time.time()

            for ip, threat in list(self.active_threats.items()):
                # Update threat activity
                if current_time - threat.last_seen > 3600:  # 1 hour
                    # Threat has been inactive, reduce confidence
                    threat.confidence_score *= 0.9

                    if threat.confidence_score < 0.1:
                        # Remove low-confidence old threats
                        del self.active_threats[ip]
                        self.logger.info(f"Removed inactive threat: {ip}")

                # Check for escalation patterns
                if ip in self.ip_statistics:
                    stats = self.ip_statistics[ip]
                    if stats["total_commands"] > 1000:  # High activity
                        threat.confidence_score = min(1.0, threat.confidence_score * 1.1)
                        self.logger.warning(f"Escalated threat confidence for {ip}: {threat.confidence_score}")

        except Exception as e:
            self.logger.error(f"Error analyzing active threats: {e}")

    async def _update_threat_scores(self):
        """Update threat scores based on behavior"""
        try:
            for ip, stats in self.ip_statistics.items():
                if ip not in self.active_threats:
                    continue

                threat = self.active_threats[ip]

                # Calculate behavior score
                behavior_score = 0.5

                # High command frequency increases score
                if stats["total_commands"] > 500:
                    behavior_score += 0.2

                # Dangerous commands increase score
                dangerous_count = sum(
                    stats["command_count"].get(cmd, 0)
                    for cmd in ["EVAL", "CONFIG", "DEBUG", "SCRIPT"]
                )
                if dangerous_count > 0:
                    behavior_score += 0.3

                # Update threat score
                threat.confidence_score = min(1.0, behavior_score)
                threat.last_seen = time.time()

        except Exception as e:
            self.logger.error(f"Error updating threat scores: {e}")

    async def _detect_new_threat_patterns(self):
        """Detect new threat patterns from behavior analysis"""
        try:
            # Analyze command patterns for anomalies
            for ip, stats in self.ip_statistics.items():
                if ip in self.active_threats:
                    continue  # Already known threat

                # Check for suspicious patterns
                suspicious_score = 0.0

                # Rapid command execution
                recent_activity = sum(
                    count for minute, count in stats["minute_buckets"].items()
                    if minute * 60 > time.time() - 300  # Last 5 minutes
                )
                if recent_activity > 100:
                    suspicious_score += 0.3

                # High data transfer
                if stats["data_transferred"] > 100 * 1024 * 1024:  # 100MB
                    suspicious_score += 0.2

                # Dangerous command usage
                dangerous_commands = stats["command_count"].get("EVAL", 0) + stats["command_count"].get("CONFIG", 0)
                if dangerous_commands > 0:
                    suspicious_score += 0.4

                # New threat detected
                if suspicious_score > 0.6:
                    new_threat = ThreatIntelligence(
                        ip_address=ip,
                        threat_type="behavioral_anomaly",
                        confidence_score=suspicious_score,
                        first_seen=stats["first_seen"],
                        last_seen=stats["last_seen"],
                        attack_patterns=list(stats["command_count"].keys()),
                        geolocation={},  # Would be populated from external service
                        reputation_score=0.0,
                        sources=["behavioral_analysis"]
                    )

                    self.active_threats[ip] = new_threat
                    self.logger.warning(f"New threat detected: {ip} (Score: {suspicious_score:.2f})")

        except Exception as e:
            self.logger.error(f"Error detecting new threat patterns: {e}")

    async def _analyze_ip_behavior(self):
        """Analyze IP address behavior patterns"""
        try:
            current_time = time.time()

            for ip, stats in self.ip_statistics.items():
                # Calculate behavior metrics
                activity_duration = current_time - stats["first_seen"]
                commands_per_hour = stats["total_commands"] / max(activity_duration / 3600, 1)

                # Detect rapid bursts
                recent_minutes = [
                    count for minute, count in stats["minute_buckets"].items()
                    if minute * 60 > current_time - 300  # Last 5 minutes
                ]

                if recent_minutes:
                    max_burst = max(recent_minutes)
                    avg_burst = sum(recent_minutes) / len(recent_minutes)

                    # Detect anomalous bursts
                    if max_burst > avg_burst * 5 and max_burst > 50:
                        await self._create_behavioral_alert(ip, "command_burst", {
                            "max_burst": max_burst,
                            "avg_burst": avg_burst,
                            "commands_per_hour": commands_per_hour
                        })

        except Exception as e:
            self.logger.error(f"Error analyzing IP behavior: {e}")

    async def _create_behavioral_alert(self, ip: str, pattern_type: str, metrics: Dict[str, Any]):
        """Create behavioral anomaly alert"""
        try:
            event = SecurityEvent(
                event_id=f"behavior_{int(time.time())}_{secrets.token_hex(4)}",
                event_type=SecurityEventType.ANOMALOUS_BEHAVIOR,
                threat_level=SecurityThreatLevel.MEDIUM,
                timestamp=time.time(),
                source_ip=ip,
                user_id=None,
                redis_command=None,
                affected_keys=[],
                details={
                    "pattern_type": pattern_type,
                    "metrics": metrics,
                    "analysis_type": "behavioral"
                },
                risk_score=0.6,
                mitigation_actions=[SecurityAction.LOG_ONLY, SecurityAction.ALERT_ADMIN]
            )

            await self._process_security_event(event)

        except Exception as e:
            self.logger.error(f"Error creating behavioral alert: {e}")

    async def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        try:
            current_time = time.time()

            # Recent security events
            recent_events = [
                event for event in self.security_events
                if current_time - event.timestamp < 3600  # Last hour
            ]

            # Threat summary
            threat_levels = defaultdict(int)
            for event in recent_events:
                threat_levels[event.threat_level.value] += 1

            # Active threats summary
            active_threat_count = len(self.active_threats)
            high_confidence_threats = sum(
                1 for threat in self.active_threats.values()
                if threat.confidence_score > 0.7
            )

            return {
                "timestamp": current_time,
                "security_status": "active",
                "monitoring_active": self.is_monitoring_active,
                "recent_events": {
                    "total": len(recent_events),
                    "by_threat_level": dict(threat_levels),
                    "critical_count": threat_levels["critical"],
                    "high_count": threat_levels["high"]
                },
                "active_threats": {
                    "total": active_threat_count,
                    "high_confidence": high_confidence_threats,
                    "blocked_ips": len(self.blocked_ips),
                    "rate_limited_clients": len(self.rate_limited_clients)
                },
                "security_rules": {
                    "total": len(self.security_rules),
                    "enabled": sum(1 for rule in self.security_rules.values() if rule.enabled),
                    "disabled": sum(1 for rule in self.security_rules.values() if not rule.enabled)
                },
                "monitoring_statistics": {
                    "unique_ips_tracked": len(self.ip_statistics),
                    "total_commands_monitored": sum(
                        stats["total_commands"] for stats in self.ip_statistics.values()
                    ),
                    "forensic_buffer_size": len(self.forensic_buffer)
                },
                "system_health": {
                    "emergency_lockdown": await self._check_emergency_lockdown_status(),
                    "monitoring_tasks": len([t for t in self.monitoring_tasks if not t.done()]),
                    "security_keys_initialized": len(self.security_keys) > 0
                }
            }

        except Exception as e:
            self.logger.error(f"Error getting security status: {e}")
            return {"error": str(e), "timestamp": time.time()}

    async def _check_emergency_lockdown_status(self) -> bool:
        """Check if emergency lockdown is active"""
        try:
            client = await self.orchestrator.get_optimal_client("read")
            lockdown_status = await client.get("security:emergency_lockdown")
            return lockdown_status == "active"
        except Exception:
            return False

    async def investigate_security_event(self, event_id: str) -> Dict[str, Any]:
        """Investigate specific security event"""
        try:
            # Find the event
            target_event = None
            for event in self.security_events:
                if event.event_id == event_id:
                    target_event = event
                    break

            if not target_event:
                return {"error": "Event not found"}

            # Gather investigation data
            investigation_data = {
                "event": asdict(target_event),
                "timeline": await self._build_event_timeline(target_event),
                "related_events": await self._find_related_events(target_event),
                "ip_analysis": await self._analyze_source_ip(target_event.source_ip),
                "forensic_data": await self._gather_forensic_data(target_event),
                "recommendations": await self._generate_investigation_recommendations(target_event)
            }

            return investigation_data

        except Exception as e:
            self.logger.error(f"Error investigating security event {event_id}: {e}")
            return {"error": str(e)}

    async def _build_event_timeline(self, event: SecurityEvent) -> List[Dict[str, Any]]:
        """Build timeline of events around target event"""
        timeline = []

        # Get events from same IP in surrounding time window
        time_window = 3600  # 1 hour
        start_time = event.timestamp - time_window
        end_time = event.timestamp + time_window

        for other_event in self.security_events:
            if (other_event.source_ip == event.source_ip and
                start_time <= other_event.timestamp <= end_time):
                timeline.append({
                    "timestamp": other_event.timestamp,
                    "event_type": other_event.event_type.value,
                    "threat_level": other_event.threat_level.value,
                    "command": other_event.redis_command,
                    "risk_score": other_event.risk_score
                })

        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"])

        return timeline

    async def _find_related_events(self, event: SecurityEvent) -> List[Dict[str, Any]]:
        """Find events related to target event"""
        related = []

        for other_event in self.security_events:
            if other_event.event_id == event.event_id:
                continue

            # Same IP
            if other_event.source_ip == event.source_ip:
                related.append({
                    "event_id": other_event.event_id,
                    "relationship": "same_ip",
                    "timestamp": other_event.timestamp,
                    "event_type": other_event.event_type.value
                })

            # Same user
            if (event.user_id and other_event.user_id and
                other_event.user_id == event.user_id):
                related.append({
                    "event_id": other_event.event_id,
                    "relationship": "same_user",
                    "timestamp": other_event.timestamp,
                    "event_type": other_event.event_type.value
                })

            # Similar keys accessed
            if event.affected_keys and other_event.affected_keys:
                common_keys = set(event.affected_keys) & set(other_event.affected_keys)
                if common_keys:
                    related.append({
                        "event_id": other_event.event_id,
                        "relationship": "common_keys",
                        "timestamp": other_event.timestamp,
                        "common_keys": list(common_keys)
                    })

        return related

    async def _analyze_source_ip(self, ip: str) -> Dict[str, Any]:
        """Analyze source IP address"""
        analysis = {
            "ip_address": ip,
            "threat_intelligence": None,
            "statistics": None,
            "geolocation": None,
            "reputation": "unknown"
        }

        # Get threat intelligence
        if ip in self.active_threats:
            threat = self.active_threats[ip]
            analysis["threat_intelligence"] = asdict(threat)

        # Get IP statistics
        if ip in self.ip_statistics:
            stats = self.ip_statistics[ip]
            analysis["statistics"] = {
                "total_commands": stats["total_commands"],
                "data_transferred_mb": stats["data_transferred"] / (1024 * 1024),
                "first_seen": stats["first_seen"],
                "last_seen": stats["last_seen"],
                "top_commands": dict(sorted(
                    stats["command_count"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5])
            }

        # Check IP reputation (placeholder)
        if self._is_private_ip(ip):
            analysis["reputation"] = "internal"
        elif ip in self.blocked_ips:
            analysis["reputation"] = "blocked"
        else:
            analysis["reputation"] = "unknown"

        return analysis

    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP address is private"""
        try:
            ip_obj = ipaddress.ip_address(ip)
            return ip_obj.is_private
        except ValueError:
            return False

    async def _gather_forensic_data(self, event: SecurityEvent) -> Dict[str, Any]:
        """Gather forensic data for event"""
        forensic_data = {
            "event_hash": hashlib.sha256(
                f"{event.event_id}{event.timestamp}{event.source_ip}".encode()
            ).hexdigest(),
            "chain_of_custody": [],
            "evidence_collected": [],
            "related_logs": []
        }

        # Collect related forensic entries
        for entry in self.forensic_buffer:
            if (entry["source_ip"] == event.source_ip and
                abs(entry["timestamp"] - event.timestamp) < 300):  # 5 minutes
                forensic_data["related_logs"].append(entry)

        # Add chain of custody entry
        forensic_data["chain_of_custody"].append({
            "timestamp": time.time(),
            "action": "forensic_analysis",
            "analyst": "redis_security_engine",
            "description": f"Forensic analysis performed for event {event.event_id}"
        })

        return forensic_data

    async def _generate_investigation_recommendations(self, event: SecurityEvent) -> List[str]:
        """Generate investigation recommendations"""
        recommendations = []

        if event.threat_level == SecurityThreatLevel.CRITICAL:
            recommendations.append("Immediately isolate affected systems")
            recommendations.append("Escalate to incident response team")
            recommendations.append("Preserve all forensic evidence")

        if event.source_ip not in self.active_threats:
            recommendations.append("Add source IP to threat intelligence database")

        if event.event_type == SecurityEventType.INJECTION_ATTEMPT:
            recommendations.append("Review and strengthen input validation")
            recommendations.append("Audit Redis configuration for security hardening")

        if event.event_type == SecurityEventType.DATA_EXFILTRATION:
            recommendations.append("Immediately review data access logs")
            recommendations.append("Check for compromised credentials")
            recommendations.append("Implement additional data loss prevention measures")

        recommendations.append("Monitor source IP for continued malicious activity")
        recommendations.append("Review security policies and update if necessary")

        return recommendations

    async def shutdown(self):
        """Shutdown Redis security engine"""
        self.logger.info("Shutting down Redis Security Engine")

        self.is_monitoring_active = False

        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()

        # Save security state
        try:
            await self._save_security_state()
        except Exception as e:
            self.logger.error(f"Error saving security state: {e}")

        self.logger.info("Redis Security Engine shutdown complete")

    async def _save_security_state(self):
        """Save security state to Redis"""
        try:
            client = await self.orchestrator.get_optimal_client("write")

            # Save blocked IPs
            await client.set(
                "security:blocked_ips",
                json.dumps(list(self.blocked_ips))
            )

            # Save threat intelligence
            threat_data = [asdict(threat) for threat in self.active_threats.values()]
            await client.setex(
                "security:threat_intelligence",
                86400 * 7,  # 7 days
                json.dumps(threat_data)
            )

            # Save security rules
            rules_data = [asdict(rule) for rule in self.security_rules.values()]
            await client.setex(
                "security:rules",
                86400 * 30,  # 30 days
                json.dumps(rules_data)
            )

            self.logger.info("Security state saved successfully")

        except Exception as e:
            self.logger.error(f"Error saving security state: {e}")


# Factory function for dependency injection
async def create_redis_security_engine() -> SophisticatedRedisSecurityEngine:
    """Create Redis security engine instance"""
    orchestrator = await get_redis_orchestrator()
    engine = SophisticatedRedisSecurityEngine(orchestrator)
    await engine.initialize()
    return engine
