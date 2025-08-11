"""
PTaaS Security Hardening Module
Implements comprehensive security controls for penetration testing operations
"""

import logging
import hashlib
import hmac
import secrets
import ipaddress
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from uuid import UUID
import re
import json
from dataclasses import dataclass, asdict

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

@dataclass
class SecurityPolicy:
    """Security policy configuration for PTaaS operations"""
    max_targets_per_session: int = 50
    max_ports_per_target: int = 1000
    max_session_duration_hours: int = 24
    require_authorization_confirmation: bool = True
    allowed_scan_profiles: List[str] = None
    blocked_networks: List[str] = None
    rate_limit_per_minute: int = 10
    rate_limit_per_hour: int = 100
    require_mfa_for_destructive: bool = True
    audit_all_operations: bool = True
    
    def __post_init__(self):
        if self.allowed_scan_profiles is None:
            self.allowed_scan_profiles = ["quick", "comprehensive", "stealth", "web_focused"]
        if self.blocked_networks is None:
            self.blocked_networks = [
                "127.0.0.0/8",      # Loopback
                "169.254.0.0/16",   # Link-local
                "224.0.0.0/4",      # Multicast
                "240.0.0.0/4"       # Reserved
            ]

@dataclass
class SecurityEvent:
    """Security event for audit logging"""
    event_id: str
    event_type: str
    user_id: str
    tenant_id: str
    source_ip: str
    timestamp: datetime
    resource: str
    action: str
    result: str
    details: Dict[str, Any]
    risk_level: str = "low"

class NetworkSecurityValidator:
    """Validates network targets for security compliance"""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.blocked_networks = [ipaddress.ip_network(net) for net in policy.blocked_networks]
        
        # Additional dangerous networks
        self.high_risk_networks = [
            ipaddress.ip_network("10.0.0.0/8"),      # Private Class A
            ipaddress.ip_network("172.16.0.0/12"),   # Private Class B
            ipaddress.ip_network("192.168.0.0/16"),  # Private Class C
        ]
        
        # Government and critical infrastructure blocks
        self.restricted_networks = [
            ipaddress.ip_network("3.0.0.0/8"),       # General Electric
            ipaddress.ip_network("6.0.0.0/8"),       # Army Information Systems Center
            ipaddress.ip_network("7.0.0.0/8"),       # Computer Sciences Corporation
            ipaddress.ip_network("8.8.8.0/24"),      # Google Public DNS
            ipaddress.ip_network("1.1.1.0/24"),      # Cloudflare DNS
        ]
    
    def validate_target_host(self, host: str, user_context: Dict[str, Any]) -> Tuple[bool, str, str]:
        """
        Validate if a target host is allowed for scanning
        
        Returns: (is_valid, risk_level, message)
        """
        try:
            # Resolve host to IP
            ip = self._resolve_host_to_ip(host)
            if not ip:
                return False, "high", f"Unable to resolve host: {host}"
            
            ip_obj = ipaddress.ip_address(ip)
            
            # Check against blocked networks
            for blocked_net in self.blocked_networks:
                if ip_obj in blocked_net:
                    return False, "critical", f"Target {host} ({ip}) is in blocked network: {blocked_net}"
            
            # Check against restricted networks (requires elevated permissions)
            for restricted_net in self.restricted_networks:
                if ip_obj in restricted_net:
                    if not user_context.get("elevated_permissions", False):
                        return False, "critical", f"Target {host} ({ip}) requires elevated permissions"
                    return True, "high", f"Elevated permission scan of restricted network: {restricted_net}"
            
            # Check private networks (warning but allowed)
            for private_net in self.high_risk_networks:
                if ip_obj in private_net:
                    return True, "medium", f"Scanning private network: {private_net}"
            
            # Check for localhost/loopback
            if ip_obj.is_loopback:
                return False, "high", "Loopback addresses are not allowed"
            
            # Check for multicast/broadcast
            if ip_obj.is_multicast:
                return False, "medium", "Multicast addresses are not allowed"
            
            # Public IP - allowed with low risk
            return True, "low", "Public IP address - scan permitted"
            
        except Exception as e:
            logger.error(f"Error validating target host {host}: {e}")
            return False, "high", f"Validation error for {host}: {str(e)}"
    
    def _resolve_host_to_ip(self, host: str) -> Optional[str]:
        """Resolve hostname to IP address"""
        import socket
        
        # If already an IP, validate and return
        try:
            ipaddress.ip_address(host)
            return host
        except ValueError:
            pass
        
        # Try to resolve hostname
        try:
            return socket.gethostbyname(host)
        except socket.gaierror:
            return None
    
    def validate_port_list(self, ports: List[int]) -> Tuple[bool, str]:
        """Validate port list for security"""
        if len(ports) > self.policy.max_ports_per_target:
            return False, f"Too many ports specified (max: {self.policy.max_ports_per_target})"
        
        # Check for dangerous ports
        dangerous_ports = {
            135: "Microsoft RPC",
            139: "NetBIOS Session",
            445: "Microsoft SMB",
            1433: "Microsoft SQL Server",
            3389: "Remote Desktop",
            5432: "PostgreSQL",
            3306: "MySQL",
            27017: "MongoDB"
        }
        
        high_risk_ports = [port for port in ports if port in dangerous_ports]
        if high_risk_ports:
            port_names = [f"{port} ({dangerous_ports[port]})" for port in high_risk_ports]
            return True, f"High-risk ports detected: {', '.join(port_names)}"
        
        # Check for privileged ports (require elevation)
        privileged_ports = [port for port in ports if port < 1024]
        if privileged_ports:
            return True, f"Privileged ports detected: {privileged_ports}"
        
        return True, "Port list validated"

class AuthorizationValidator:
    """Validates user authorization for PTaaS operations"""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
    
    def validate_scan_authorization(self, 
                                  user_context: Dict[str, Any], 
                                  targets: List[Dict[str, Any]],
                                  scan_type: str) -> Tuple[bool, List[str]]:
        """Validate user authorization for scan operation"""
        violations = []
        
        # Check scan profile permission
        if scan_type not in self.policy.allowed_scan_profiles:
            violations.append(f"Scan profile '{scan_type}' not permitted")
        
        # Check target count limits
        if len(targets) > self.policy.max_targets_per_session:
            violations.append(f"Too many targets ({len(targets)} > {self.policy.max_targets_per_session})")
        
        # Check explicit authorization for each target
        if self.policy.require_authorization_confirmation:
            for target in targets:
                if not target.get("authorized", False):
                    violations.append(f"Target {target.get('host', 'unknown')} not explicitly authorized")
        
        # Check MFA requirement for destructive scans
        if self.policy.require_mfa_for_destructive:
            destructive_profiles = ["comprehensive", "web_focused"]
            if scan_type in destructive_profiles and not user_context.get("mfa_verified", False):
                violations.append(f"MFA required for {scan_type} scans")
        
        # Check user permissions
        required_permissions = self._get_required_permissions(scan_type, targets)
        user_permissions = set(user_context.get("permissions", []))
        missing_permissions = required_permissions - user_permissions
        
        if missing_permissions:
            violations.append(f"Missing permissions: {', '.join(missing_permissions)}")
        
        return len(violations) == 0, violations
    
    def _get_required_permissions(self, scan_type: str, targets: List[Dict[str, Any]]) -> Set[str]:
        """Get required permissions for scan operation"""
        permissions = {"ptaas:execute"}
        
        # Add scan-type specific permissions
        if scan_type == "comprehensive":
            permissions.add("ptaas:comprehensive")
        elif scan_type == "stealth":
            permissions.add("ptaas:stealth")
        elif scan_type == "web_focused":
            permissions.add("ptaas:web_testing")
        
        # Add target-specific permissions
        for target in targets:
            host = target.get("host", "")
            if self._is_internal_network(host):
                permissions.add("ptaas:internal_network")
            if self._is_critical_system(host):
                permissions.add("ptaas:critical_systems")
        
        return permissions
    
    def _is_internal_network(self, host: str) -> bool:
        """Check if host is in internal network"""
        try:
            ip = ipaddress.ip_address(host)
            return ip.is_private
        except ValueError:
            return False
    
    def _is_critical_system(self, host: str) -> bool:
        """Check if host is a critical system"""
        critical_patterns = [
            r".*prod.*",
            r".*production.*",
            r".*db.*",
            r".*database.*",
            r".*dc.*",
            r".*domain.*controller.*"
        ]
        
        host_lower = host.lower()
        return any(re.match(pattern, host_lower) for pattern in critical_patterns)

class SessionSecurityManager:
    """Manages security for PTaaS sessions"""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_tokens: Dict[str, str] = {}
    
    def create_secure_session(self, 
                            session_id: str, 
                            user_context: Dict[str, Any],
                            targets: List[Dict[str, Any]]) -> Tuple[bool, str, Optional[str]]:
        """Create a secure session with validation"""
        
        # Generate session security token
        session_token = self._generate_session_token(session_id, user_context)
        
        # Create session security context
        security_context = {
            "session_id": session_id,
            "user_id": user_context["user_id"],
            "tenant_id": user_context["tenant_id"],
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(hours=self.policy.max_session_duration_hours),
            "targets": targets,
            "security_token": session_token,
            "permissions": user_context.get("permissions", []),
            "mfa_verified": user_context.get("mfa_verified", False),
            "source_ip": user_context.get("source_ip", "unknown"),
            "user_agent": user_context.get("user_agent", "unknown")
        }
        
        # Validate session security
        is_valid, message = self._validate_session_security(security_context)
        if not is_valid:
            return False, message, None
        
        # Store session
        self.active_sessions[session_id] = security_context
        self.session_tokens[session_token] = session_id
        
        return True, "Session created successfully", session_token
    
    def validate_session_token(self, session_id: str, provided_token: str) -> bool:
        """Validate session security token"""
        session = self.active_sessions.get(session_id)
        if not session:
            return False
        
        # Check token match
        if session.get("security_token") != provided_token:
            return False
        
        # Check expiration
        if datetime.utcnow() > session.get("expires_at", datetime.min):
            self._cleanup_session(session_id)
            return False
        
        return True
    
    def _generate_session_token(self, session_id: str, user_context: Dict[str, Any]) -> str:
        """Generate cryptographically secure session token"""
        token_data = f"{session_id}:{user_context['user_id']}:{datetime.utcnow().isoformat()}"
        token_bytes = token_data.encode() + secrets.token_bytes(32)
        return hashlib.sha256(token_bytes).hexdigest()
    
    def _validate_session_security(self, security_context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate session security requirements"""
        
        # Check session duration
        duration_hours = (security_context["expires_at"] - security_context["created_at"]).total_seconds() / 3600
        if duration_hours > self.policy.max_session_duration_hours:
            return False, f"Session duration exceeds maximum ({self.policy.max_session_duration_hours} hours)"
        
        # Check target limits
        if len(security_context["targets"]) > self.policy.max_targets_per_session:
            return False, f"Too many targets in session"
        
        return True, "Session security validated"
    
    def _cleanup_session(self, session_id: str):
        """Clean up expired or invalid session"""
        session = self.active_sessions.pop(session_id, None)
        if session and session.get("security_token"):
            self.session_tokens.pop(session.get("security_token"), None)

class SecurityAuditLogger:
    """Comprehensive security audit logging for PTaaS operations"""
    
    def __init__(self):
        self.events: List[SecurityEvent] = []
        self.high_risk_events: List[SecurityEvent] = []
    
    async def log_security_event(self, 
                                event_type: str,
                                user_id: str,
                                tenant_id: str,
                                source_ip: str,
                                resource: str,
                                action: str,
                                result: str,
                                details: Dict[str, Any],
                                risk_level: str = "low"):
        """Log comprehensive security event"""
        
        event = SecurityEvent(
            event_id=secrets.token_hex(16),
            event_type=event_type,
            user_id=user_id,
            tenant_id=tenant_id,
            source_ip=source_ip,
            timestamp=datetime.utcnow(),
            resource=resource,
            action=action,
            result=result,
            details=details,
            risk_level=risk_level
        )
        
        # Store event
        self.events.append(event)
        if risk_level in ["high", "critical"]:
            self.high_risk_events.append(event)
        
        # Log to application logger
        log_level = {
            "low": logging.INFO,
            "medium": logging.WARNING,
            "high": logging.ERROR,
            "critical": logging.CRITICAL
        }.get(risk_level, logging.INFO)
        
        logger.log(log_level, 
                  f"PTaaS Security Event: {event_type} | User: {user_id} | "
                  f"Action: {action} | Result: {result} | Risk: {risk_level}")
        
        # Store in database for persistent audit trail
        await self._store_audit_event(event)
    
    async def _store_audit_event(self, event: SecurityEvent):
        """Store security event in database"""
        try:
            # In a real implementation, store in dedicated audit table
            # For now, just ensure the event is logged
            audit_record = {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "user_id": event.user_id,
                "tenant_id": event.tenant_id,
                "timestamp": event.timestamp.isoformat(),
                "resource": event.resource,
                "action": event.action,
                "result": event.result,
                "risk_level": event.risk_level,
                "details": json.dumps(event.details)
            }
            
            # Store in audit log (implement actual database storage)
            logger.info(f"Audit event stored: {audit_record}")
            
        except Exception as e:
            logger.error(f"Failed to store audit event: {e}")

class PTaaSSecurityManager:
    """Main security manager for PTaaS operations"""
    
    def __init__(self, policy: Optional[SecurityPolicy] = None):
        self.policy = policy or SecurityPolicy()
        self.network_validator = NetworkSecurityValidator(self.policy)
        self.auth_validator = AuthorizationValidator(self.policy)
        self.session_manager = SessionSecurityManager(self.policy)
        self.audit_logger = SecurityAuditLogger()
    
    async def validate_scan_request(self, 
                                   user_context: Dict[str, Any],
                                   targets: List[Dict[str, Any]],
                                   scan_type: str) -> Tuple[bool, List[str], str]:
        """Comprehensive validation of scan request"""
        
        violations = []
        overall_risk = "low"
        
        # Validate authorization
        auth_valid, auth_violations = self.auth_validator.validate_scan_authorization(
            user_context, targets, scan_type
        )
        if not auth_valid:
            violations.extend(auth_violations)
        
        # Validate each target
        for target in targets:
            host = target.get("host", "")
            ports = target.get("ports", [])
            
            # Validate host
            host_valid, risk_level, message = self.network_validator.validate_target_host(
                host, user_context
            )
            if not host_valid:
                violations.append(f"Target {host}: {message}")
            else:
                # Update overall risk level
                if risk_level == "critical":
                    overall_risk = "critical"
                elif risk_level == "high" and overall_risk != "critical":
                    overall_risk = "high"
                elif risk_level == "medium" and overall_risk not in ["critical", "high"]:
                    overall_risk = "medium"
            
            # Validate ports
            ports_valid, ports_message = self.network_validator.validate_port_list(ports)
            if not ports_valid:
                violations.append(f"Target {host} ports: {ports_message}")
        
        # Log security event
        await self.audit_logger.log_security_event(
            event_type="ptaas_scan_validation",
            user_id=user_context["user_id"],
            tenant_id=user_context["tenant_id"],
            source_ip=user_context.get("source_ip", "unknown"),
            resource=f"scan_request_{scan_type}",
            action="validate",
            result="approved" if len(violations) == 0 else "rejected",
            details={
                "scan_type": scan_type,
                "target_count": len(targets),
                "violations": violations,
                "targets": [t.get("host") for t in targets]
            },
            risk_level=overall_risk
        )
        
        return len(violations) == 0, violations, overall_risk
    
    async def create_secure_session(self,
                                   session_id: str,
                                   user_context: Dict[str, Any],
                                   targets: List[Dict[str, Any]]) -> Tuple[bool, str, Optional[str]]:
        """Create secure PTaaS session"""
        
        success, message, token = self.session_manager.create_secure_session(
            session_id, user_context, targets
        )
        
        # Log session creation
        await self.audit_logger.log_security_event(
            event_type="ptaas_session_create",
            user_id=user_context["user_id"],
            tenant_id=user_context["tenant_id"],
            source_ip=user_context.get("source_ip", "unknown"),
            resource=f"session_{session_id}",
            action="create",
            result="success" if success else "failed",
            details={
                "session_id": session_id,
                "target_count": len(targets),
                "message": message
            },
            risk_level="medium" if success else "high"
        )
        
        return success, message, token
    
    async def validate_session_operation(self,
                                        session_id: str,
                                        session_token: str,
                                        user_context: Dict[str, Any],
                                        operation: str) -> bool:
        """Validate session operation with security token"""
        
        # Validate session token
        token_valid = self.session_manager.validate_session_token(session_id, session_token)
        
        # Log operation attempt
        await self.audit_logger.log_security_event(
            event_type="ptaas_session_operation",
            user_id=user_context["user_id"],
            tenant_id=user_context["tenant_id"],
            source_ip=user_context.get("source_ip", "unknown"),
            resource=f"session_{session_id}",
            action=operation,
            result="authorized" if token_valid else "unauthorized",
            details={
                "session_id": session_id,
                "operation": operation,
                "token_provided": session_token is not None
            },
            risk_level="high" if not token_valid else "low"
        )
        
        return token_valid
    
    async def get_security_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Get security metrics for monitoring"""
        
        # Filter events by tenant
        tenant_events = [e for e in self.audit_logger.events if e.tenant_id == tenant_id]
        recent_events = [e for e in tenant_events if e.timestamp > datetime.utcnow() - timedelta(hours=24)]
        
        return {
            "total_events": len(tenant_events),
            "recent_events": len(recent_events),
            "high_risk_events": len([e for e in recent_events if e.risk_level in ["high", "critical"]]),
            "active_sessions": len([s for s in self.session_manager.active_sessions.values() 
                                  if s["tenant_id"] == tenant_id]),
            "policy_violations": len([e for e in recent_events if e.result in ["rejected", "unauthorized"]]),
            "risk_distribution": {
                "low": len([e for e in recent_events if e.risk_level == "low"]),
                "medium": len([e for e in recent_events if e.risk_level == "medium"]),
                "high": len([e for e in recent_events if e.risk_level == "high"]),
                "critical": len([e for e in recent_events if e.risk_level == "critical"])
            }
        }

# Global security manager instance
_security_manager: Optional[PTaaSSecurityManager] = None

def get_ptaas_security_manager() -> PTaaSSecurityManager:
    """Get global PTaaS security manager instance"""
    global _security_manager
    if _security_manager is None:
        _security_manager = PTaaSSecurityManager()
    return _security_manager