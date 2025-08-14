#!/usr/bin/env python3
"""
XORB Resilience & Scalability Layer - Security Hardening & Compliance
Advanced security with mTLS, RBAC, audit trails, and compliance reporting
"""

import asyncio
import json
import time
import logging
import hashlib
import hmac
import base64
import secrets
import ssl
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import os
from pathlib import Path
import cryptography
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption
import jwt
from collections import defaultdict, deque
import ipaddress
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityRole(Enum):
    """Security roles for RBAC"""
    ADMIN = "admin"
    OPERATOR = "operator"
    ANALYST = "analyst"
    VIEWER = "viewer"
    SERVICE = "service"
    AGENT = "agent"

class Permission(Enum):
    """System permissions"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"
    MONITOR = "monitor"
    DEPLOY = "deploy"
    SCALE = "scale"

class AuditEventType(Enum):
    """Types of audit events"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_EVENT = "security_event"
    SYSTEM_EVENT = "system_event"
    SERVICE_EVENT = "service_event"

class SecurityThreatLevel(Enum):
    """Security threat levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityPrincipal:
    """Security principal (user or service)"""
    principal_id: str
    principal_type: str  # "user" or "service"
    name: str
    roles: Set[SecurityRole]
    permissions: Set[Permission] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = None
    active: bool = True
    mfa_enabled: bool = False

@dataclass
class SecurityToken:
    """Security token for authentication"""
    token_id: str
    principal_id: str
    token_type: str  # "access", "refresh", "service"
    token_hash: str
    expires_at: datetime
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    scopes: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AuditEvent:
    """Audit trail event"""
    event_id: str
    event_type: AuditEventType
    principal_id: Optional[str]
    resource: str
    action: str
    outcome: str  # "success", "failure", "denied"
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0

@dataclass
class SecurityPolicy:
    """Security policy definition"""
    policy_id: str
    name: str
    description: str
    rules: List[Dict[str, Any]]
    applies_to: List[str]  # Resources or services
    priority: int = 0
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class ComplianceReport:
    """Compliance reporting structure"""
    report_id: str
    report_type: str  # "SOC2", "ISO27001", "GDPR", etc.
    period_start: datetime
    period_end: datetime
    compliance_score: float
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    evidence: List[Dict[str, Any]]
    generated_at: datetime = field(default_factory=datetime.now)

class XORBSecurityManager:
    """Advanced security management system"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.security_id = str(uuid.uuid4())

        # Security components
        self.principals: Dict[str, SecurityPrincipal] = {}
        self.tokens: Dict[str, SecurityToken] = {}
        self.security_policies: Dict[str, SecurityPolicy] = {}

        # Audit and compliance
        self.audit_log: deque = deque(maxlen=10000)
        self.compliance_reports: List[ComplianceReport] = []

        # mTLS configuration
        self.ca_cert_path = self.config.get('ca_cert_path', '/tmp/xorb_ca.crt')
        self.ca_key_path = self.config.get('ca_key_path', '/tmp/xorb_ca.key')
        self.cert_validity_days = self.config.get('cert_validity_days', 365)

        # Security settings
        self.jwt_secret = self.config.get('jwt_secret', secrets.token_urlsafe(32))
        self.token_expiry_minutes = self.config.get('token_expiry_minutes', 60)
        self.max_failed_attempts = self.config.get('max_failed_attempts', 5)
        self.lockout_duration_minutes = self.config.get('lockout_duration_minutes', 15)

        # Rate limiting and security monitoring
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.suspicious_activities: List[Dict[str, Any]] = []

        # Initialize components
        self._initialize_default_roles_and_permissions()
        self._initialize_default_security_policies()
        self._setup_ca_certificate()

        logger.info(f"Security Manager initialized: {self.security_id}")

    def _initialize_default_roles_and_permissions(self):
        """Initialize default RBAC roles and permissions"""
        try:
            # Define role-permission mappings
            role_permissions = {
                SecurityRole.ADMIN: {Permission.READ, Permission.WRITE, Permission.DELETE,
                                   Permission.EXECUTE, Permission.ADMIN, Permission.MONITOR,
                                   Permission.DEPLOY, Permission.SCALE},
                SecurityRole.OPERATOR: {Permission.READ, Permission.WRITE, Permission.EXECUTE,
                                      Permission.MONITOR, Permission.DEPLOY, Permission.SCALE},
                SecurityRole.ANALYST: {Permission.READ, Permission.MONITOR},
                SecurityRole.VIEWER: {Permission.READ},
                SecurityRole.SERVICE: {Permission.READ, Permission.WRITE, Permission.EXECUTE},
                SecurityRole.AGENT: {Permission.READ, Permission.WRITE, Permission.EXECUTE, Permission.MONITOR}
            }

            self.role_permissions = role_permissions

            logger.info("Default RBAC roles and permissions initialized")

        except Exception as e:
            logger.error(f"Failed to initialize RBAC: {e}")

    def _initialize_default_security_policies(self):
        """Initialize default security policies"""
        try:
            default_policies = [
                {
                    'policy_id': 'password_policy',
                    'name': 'Password Policy',
                    'description': 'Enforce strong password requirements',
                    'rules': [
                        {'type': 'min_length', 'value': 12},
                        {'type': 'require_uppercase', 'value': True},
                        {'type': 'require_lowercase', 'value': True},
                        {'type': 'require_numbers', 'value': True},
                        {'type': 'require_symbols', 'value': True}
                    ],
                    'applies_to': ['user_authentication'],
                    'priority': 100
                },
                {
                    'policy_id': 'access_control_policy',
                    'name': 'Access Control Policy',
                    'description': 'Control access to sensitive resources',
                    'rules': [
                        {'type': 'require_mfa', 'resources': ['admin_panel', 'production_deploy']},
                        {'type': 'ip_whitelist', 'allowed_ips': ['10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16']},
                        {'type': 'time_restriction', 'allowed_hours': [9, 17]}
                    ],
                    'applies_to': ['sensitive_operations'],
                    'priority': 90
                },
                {
                    'policy_id': 'data_protection_policy',
                    'name': 'Data Protection Policy',
                    'description': 'Protect sensitive data at rest and in transit',
                    'rules': [
                        {'type': 'encryption_required', 'algorithm': 'AES-256'},
                        {'type': 'data_classification', 'levels': ['public', 'internal', 'confidential', 'restricted']},
                        {'type': 'retention_period', 'days': 2555}  # 7 years
                    ],
                    'applies_to': ['data_storage', 'data_transmission'],
                    'priority': 95
                }
            ]

            for policy_data in default_policies:
                policy = SecurityPolicy(
                    policy_id=policy_data['policy_id'],
                    name=policy_data['name'],
                    description=policy_data['description'],
                    rules=policy_data['rules'],
                    applies_to=policy_data['applies_to'],
                    priority=policy_data['priority']
                )
                self.security_policies[policy.policy_id] = policy

            logger.info(f"Initialized {len(default_policies)} default security policies")

        except Exception as e:
            logger.error(f"Failed to initialize security policies: {e}")

    def _setup_ca_certificate(self):
        """Setup Certificate Authority for mTLS"""
        try:
            ca_cert_path = Path(self.ca_cert_path)
            ca_key_path = Path(self.ca_key_path)

            # Create CA certificate if it doesn't exist
            if not ca_cert_path.exists() or not ca_key_path.exists():
                logger.info("Creating new CA certificate")

                # Generate CA private key
                ca_private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=4096
                )

                # Create CA certificate
                ca_name = x509.Name([
                    x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                    x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "XORB Platform"),
                    x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Security"),
                    x509.NameAttribute(NameOID.COMMON_NAME, "XORB CA"),
                ])

                ca_cert = x509.CertificateBuilder().subject_name(
                    ca_name
                ).issuer_name(
                    ca_name
                ).public_key(
                    ca_private_key.public_key()
                ).serial_number(
                    x509.random_serial_number()
                ).not_valid_before(
                    datetime.utcnow()
                ).not_valid_after(
                    datetime.utcnow() + timedelta(days=3650)  # 10 years
                ).add_extension(
                    x509.BasicConstraints(ca=True, path_length=None),
                    critical=True
                ).add_extension(
                    x509.KeyUsage(
                        key_cert_sign=True,
                        crl_sign=True,
                        digital_signature=False,
                        content_commitment=False,
                        key_encipherment=False,
                        data_encipherment=False,
                        key_agreement=False,
                        encipher_only=False,
                        decipher_only=False
                    ),
                    critical=True
                ).sign(ca_private_key, hashes.SHA256())

                # Write CA certificate
                with open(ca_cert_path, 'wb') as f:
                    f.write(ca_cert.public_bytes(Encoding.PEM))

                # Write CA private key
                with open(ca_key_path, 'wb') as f:
                    f.write(ca_private_key.private_bytes(
                        Encoding.PEM,
                        PrivateFormat.PKCS8,
                        NoEncryption()
                    ))

                # Set appropriate permissions
                os.chmod(ca_key_path, 0o600)
                os.chmod(ca_cert_path, 0o644)

                logger.info(f"CA certificate created: {ca_cert_path}")
            else:
                logger.info("Using existing CA certificate")

        except Exception as e:
            logger.error(f"Failed to setup CA certificate: {e}")

    async def create_principal(self, principal_id: str, principal_type: str, name: str,
                             roles: List[SecurityRole], attributes: Optional[Dict[str, Any]] = None) -> SecurityPrincipal:
        """Create a new security principal"""
        try:
            # Calculate permissions from roles
            permissions = set()
            for role in roles:
                if role in self.role_permissions:
                    permissions.update(self.role_permissions[role])

            principal = SecurityPrincipal(
                principal_id=principal_id,
                principal_type=principal_type,
                name=name,
                roles=set(roles),
                permissions=permissions,
                attributes=attributes or {}
            )

            self.principals[principal_id] = principal

            # Audit event
            await self._create_audit_event(
                event_type=AuditEventType.SYSTEM_EVENT,
                principal_id=None,  # System action
                resource="security_principal",
                action="create",
                outcome="success",
                details={'created_principal': principal_id, 'roles': [r.value for r in roles]}
            )

            logger.info(f"Created security principal: {principal_id} ({principal_type})")
            return principal

        except Exception as e:
            logger.error(f"Failed to create principal: {e}")
            raise e

    async def authenticate_principal(self, principal_id: str, credentials: Dict[str, Any],
                                   source_ip: Optional[str] = None) -> Optional[SecurityToken]:
        """Authenticate a principal and create access token"""
        try:
            # Check if principal exists
            if principal_id not in self.principals:
                await self._record_failed_attempt(principal_id, source_ip, "principal_not_found")
                return None

            principal = self.principals[principal_id]

            # Check if principal is active
            if not principal.active:
                await self._record_failed_attempt(principal_id, source_ip, "principal_inactive")
                return None

            # Check for account lockout
            if await self._is_locked_out(principal_id):
                await self._record_failed_attempt(principal_id, source_ip, "account_locked")
                return None

            # Verify credentials (simplified - in reality, would check password hash, etc.)
            if not await self._verify_credentials(principal, credentials):
                await self._record_failed_attempt(principal_id, source_ip, "invalid_credentials")
                return None

            # Clear failed attempts on successful authentication
            if principal_id in self.failed_attempts:
                del self.failed_attempts[principal_id]

            # Create access token
            token = await self._create_access_token(principal)

            # Update principal last accessed
            principal.last_accessed = datetime.now()

            # Audit event
            await self._create_audit_event(
                event_type=AuditEventType.AUTHENTICATION,
                principal_id=principal_id,
                resource="authentication",
                action="login",
                outcome="success",
                source_ip=source_ip,
                details={'token_id': token.token_id}
            )

            logger.info(f"Principal authenticated: {principal_id}")
            return token

        except Exception as e:
            logger.error(f"Authentication failed for {principal_id}: {e}")
            await self._record_failed_attempt(principal_id, source_ip, "system_error")
            return None

    async def _verify_credentials(self, principal: SecurityPrincipal, credentials: Dict[str, Any]) -> bool:
        """Verify principal credentials"""
        try:
            # This is a simplified implementation
            # In production, would verify password hashes, API keys, certificates, etc.

            if principal.principal_type == "service":
                # Service authentication via API key or certificate
                api_key = credentials.get('api_key')
                if api_key and api_key == principal.attributes.get('api_key'):
                    return True
            elif principal.principal_type == "user":
                # User authentication via password
                password = credentials.get('password')
                expected_password = principal.attributes.get('password_hash')
                if password and expected_password:
                    # In production, would use proper password hashing (bcrypt, etc.)
                    password_hash = hashlib.sha256(password.encode()).hexdigest()
                    return password_hash == expected_password

            return False

        except Exception as e:
            logger.error(f"Credential verification failed: {e}")
            return False

    async def _create_access_token(self, principal: SecurityPrincipal) -> SecurityToken:
        """Create access token for authenticated principal"""
        try:
            token_id = str(uuid.uuid4())
            expires_at = datetime.now() + timedelta(minutes=self.token_expiry_minutes)

            # Create JWT payload
            payload = {
                'token_id': token_id,
                'principal_id': principal.principal_id,
                'principal_type': principal.principal_type,
                'roles': [role.value for role in principal.roles],
                'permissions': [perm.value for perm in principal.permissions],
                'exp': expires_at.timestamp(),
                'iat': datetime.now().timestamp(),
                'iss': 'xorb-security-manager'
            }

            # Generate JWT token
            jwt_token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
            token_hash = hashlib.sha256(jwt_token.encode()).hexdigest()

            # Create security token
            token = SecurityToken(
                token_id=token_id,
                principal_id=principal.principal_id,
                token_type="access",
                token_hash=token_hash,
                expires_at=expires_at,
                scopes=set(perm.value for perm in principal.permissions),
                metadata={'jwt_token': jwt_token}
            )

            self.tokens[token_id] = token

            return token

        except Exception as e:
            logger.error(f"Failed to create access token: {e}")
            raise e

    async def validate_token(self, token_string: str) -> Optional[SecurityPrincipal]:
        """Validate access token and return principal"""
        try:
            # Decode JWT token
            payload = jwt.decode(token_string, self.jwt_secret, algorithms=['HS256'])

            token_id = payload.get('token_id')
            principal_id = payload.get('principal_id')

            # Check if token exists in our store
            if token_id not in self.tokens:
                return None

            token = self.tokens[token_id]

            # Check if token is expired
            if datetime.now() > token.expires_at:
                del self.tokens[token_id]
                return None

            # Check if principal still exists and is active
            if principal_id not in self.principals:
                return None

            principal = self.principals[principal_id]
            if not principal.active:
                return None

            # Update token last used
            token.last_used = datetime.now()

            return principal

        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return None

    async def authorize_action(self, principal: SecurityPrincipal, resource: str,
                             action: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Authorize principal action on resource"""
        try:
            # Convert action to permission
            action_permission_map = {
                'read': Permission.READ,
                'write': Permission.WRITE,
                'delete': Permission.DELETE,
                'execute': Permission.EXECUTE,
                'admin': Permission.ADMIN,
                'monitor': Permission.MONITOR,
                'deploy': Permission.DEPLOY,
                'scale': Permission.SCALE
            }

            required_permission = action_permission_map.get(action.lower())
            if not required_permission:
                return False

            # Check if principal has required permission
            has_permission = required_permission in principal.permissions

            # Apply security policies
            policy_result = await self._evaluate_security_policies(principal, resource, action, context)

            # Final authorization decision
            authorized = has_permission and policy_result

            # Audit event
            await self._create_audit_event(
                event_type=AuditEventType.AUTHORIZATION,
                principal_id=principal.principal_id,
                resource=resource,
                action=action,
                outcome="success" if authorized else "denied",
                source_ip=context.get('source_ip') if context else None,
                details={
                    'required_permission': required_permission.value,
                    'has_permission': has_permission,
                    'policy_result': policy_result
                }
            )

            if not authorized:
                logger.warning(f"Authorization denied: {principal.principal_id} -> {resource}:{action}")

            return authorized

        except Exception as e:
            logger.error(f"Authorization failed: {e}")
            return False

    async def _evaluate_security_policies(self, principal: SecurityPrincipal, resource: str,
                                        action: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Evaluate security policies for authorization"""
        try:
            # Get applicable policies
            applicable_policies = [
                policy for policy in self.security_policies.values()
                if policy.active and any(applies in resource for applies in policy.applies_to)
            ]

            # Sort by priority (higher priority first)
            applicable_policies.sort(key=lambda p: p.priority, reverse=True)

            for policy in applicable_policies:
                for rule in policy.rules:
                    rule_type = rule.get('type')

                    # IP whitelist check
                    if rule_type == 'ip_whitelist' and context and 'source_ip' in context:
                        source_ip = context['source_ip']
                        allowed_ips = rule.get('allowed_ips', [])

                        ip_allowed = False
                        for allowed_ip in allowed_ips:
                            try:
                                if ipaddress.ip_address(source_ip) in ipaddress.ip_network(allowed_ip):
                                    ip_allowed = True
                                    break
                            except:
                                pass

                        if not ip_allowed:
                            logger.warning(f"IP not whitelisted: {source_ip}")
                            return False

                    # MFA requirement check
                    elif rule_type == 'require_mfa':
                        required_resources = rule.get('resources', [])
                        if any(req_resource in resource for req_resource in required_resources):
                            if not principal.mfa_enabled:
                                logger.warning(f"MFA required for resource: {resource}")
                                return False

                    # Time restriction check
                    elif rule_type == 'time_restriction':
                        allowed_hours = rule.get('allowed_hours', [])
                        current_hour = datetime.now().hour
                        if allowed_hours and current_hour not in range(allowed_hours[0], allowed_hours[1]):
                            logger.warning(f"Access outside allowed hours: {current_hour}")
                            return False

            return True

        except Exception as e:
            logger.error(f"Policy evaluation failed: {e}")
            return False

    async def _record_failed_attempt(self, principal_id: str, source_ip: Optional[str], reason: str):
        """Record failed authentication attempt"""
        try:
            self.failed_attempts[principal_id].append(datetime.now())

            # Audit event
            await self._create_audit_event(
                event_type=AuditEventType.AUTHENTICATION,
                principal_id=principal_id,
                resource="authentication",
                action="login",
                outcome="failure",
                source_ip=source_ip,
                details={'failure_reason': reason}
            )

            # Check for suspicious activity
            if len(self.failed_attempts[principal_id]) >= self.max_failed_attempts:
                await self._flag_suspicious_activity(principal_id, source_ip, "excessive_failed_attempts")

        except Exception as e:
            logger.error(f"Failed to record failed attempt: {e}")

    async def _is_locked_out(self, principal_id: str) -> bool:
        """Check if principal is locked out due to failed attempts"""
        try:
            if principal_id not in self.failed_attempts:
                return False

            attempts = self.failed_attempts[principal_id]

            # Remove old attempts (outside lockout window)
            cutoff_time = datetime.now() - timedelta(minutes=self.lockout_duration_minutes)
            recent_attempts = [attempt for attempt in attempts if attempt > cutoff_time]
            self.failed_attempts[principal_id] = recent_attempts

            # Check if locked out
            return len(recent_attempts) >= self.max_failed_attempts

        except Exception as e:
            logger.error(f"Lockout check failed: {e}")
            return False

    async def _flag_suspicious_activity(self, principal_id: str, source_ip: Optional[str], activity_type: str):
        """Flag suspicious security activity"""
        try:
            activity = {
                'activity_id': str(uuid.uuid4()),
                'principal_id': principal_id,
                'source_ip': source_ip,
                'activity_type': activity_type,
                'timestamp': datetime.now(),
                'threat_level': SecurityThreatLevel.MEDIUM.value,
                'details': {
                    'failed_attempts': len(self.failed_attempts.get(principal_id, [])),
                    'lockout_triggered': await self._is_locked_out(principal_id)
                }
            }

            self.suspicious_activities.append(activity)

            # Keep only recent activities
            if len(self.suspicious_activities) > 1000:
                self.suspicious_activities = self.suspicious_activities[-1000:]

            # Create high-priority audit event
            await self._create_audit_event(
                event_type=AuditEventType.SECURITY_EVENT,
                principal_id=principal_id,
                resource="security_monitoring",
                action="suspicious_activity_detected",
                outcome="flagged",
                source_ip=source_ip,
                details=activity,
                risk_score=0.7
            )

            logger.warning(f"Suspicious activity flagged: {activity_type} for {principal_id}")

        except Exception as e:
            logger.error(f"Failed to flag suspicious activity: {e}")

    async def generate_service_certificate(self, service_name: str, hostnames: List[str]) -> Tuple[str, str]:
        """Generate mTLS certificate for service"""
        try:
            # Load CA certificate and key
            with open(self.ca_cert_path, 'rb') as f:
                ca_cert = x509.load_pem_x509_certificate(f.read())

            with open(self.ca_key_path, 'rb') as f:
                ca_private_key = serialization.load_pem_private_key(f.read(), password=None)

            # Generate service private key
            service_private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )

            # Create service certificate
            service_name_attrs = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "XORB Platform"),
                x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Services"),
                x509.NameAttribute(NameOID.COMMON_NAME, service_name),
            ])

            # Create SAN (Subject Alternative Names)
            san_names = [x509.DNSName(hostname) for hostname in hostnames]

            service_cert = x509.CertificateBuilder().subject_name(
                service_name_attrs
            ).issuer_name(
                ca_cert.subject
            ).public_key(
                service_private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=self.cert_validity_days)
            ).add_extension(
                x509.SubjectAlternativeName(san_names),
                critical=False
            ).add_extension(
                x509.KeyUsage(
                    key_cert_sign=False,
                    crl_sign=False,
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=True,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False
                ),
                critical=True
            ).add_extension(
                x509.ExtendedKeyUsage([
                    x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                    x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH
                ]),
                critical=True
            ).sign(ca_private_key, hashes.SHA256())

            # Convert to PEM format
            cert_pem = service_cert.public_bytes(Encoding.PEM).decode('utf-8')
            key_pem = service_private_key.private_bytes(
                Encoding.PEM,
                PrivateFormat.PKCS8,
                NoEncryption()
            ).decode('utf-8')

            logger.info(f"Generated mTLS certificate for service: {service_name}")
            return cert_pem, key_pem

        except Exception as e:
            logger.error(f"Failed to generate service certificate: {e}")
            raise e

    async def _create_audit_event(self, event_type: AuditEventType, principal_id: Optional[str],
                                resource: str, action: str, outcome: str,
                                source_ip: Optional[str] = None, user_agent: Optional[str] = None,
                                details: Optional[Dict[str, Any]] = None, risk_score: float = 0.0):
        """Create audit trail event"""
        try:
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                principal_id=principal_id,
                resource=resource,
                action=action,
                outcome=outcome,
                source_ip=source_ip,
                user_agent=user_agent,
                details=details or {},
                risk_score=risk_score
            )

            self.audit_log.append(event)

            # Log high-risk events
            if risk_score > 0.5:
                logger.warning(f"High-risk audit event: {event.event_id} - {event.action} on {event.resource}")

        except Exception as e:
            logger.error(f"Failed to create audit event: {e}")

    async def generate_compliance_report(self, report_type: str, period_start: datetime,
                                       period_end: datetime) -> ComplianceReport:
        """Generate compliance report"""
        try:
            report_id = f"{report_type}_{int(time.time())}"

            # Filter audit events for the period
            period_events = [
                event for event in self.audit_log
                if period_start <= event.timestamp <= period_end
            ]

            # Generate findings based on report type
            findings = []
            recommendations = []
            evidence = []
            compliance_score = 100.0

            if report_type.upper() == "SOC2":
                findings, recommendations, compliance_score = await self._generate_soc2_findings(period_events)
            elif report_type.upper() == "ISO27001":
                findings, recommendations, compliance_score = await self._generate_iso27001_findings(period_events)
            elif report_type.upper() == "GDPR":
                findings, recommendations, compliance_score = await self._generate_gdpr_findings(period_events)

            # Collect evidence
            evidence = [
                {
                    'type': 'audit_events',
                    'count': len(period_events),
                    'high_risk_events': len([e for e in period_events if e.risk_score > 0.5])
                },
                {
                    'type': 'security_policies',
                    'count': len([p for p in self.security_policies.values() if p.active]),
                    'policies': list(self.security_policies.keys())
                },
                {
                    'type': 'principals',
                    'count': len([p for p in self.principals.values() if p.active]),
                    'user_count': len([p for p in self.principals.values() if p.principal_type == 'user' and p.active]),
                    'service_count': len([p for p in self.principals.values() if p.principal_type == 'service' and p.active])
                }
            ]

            report = ComplianceReport(
                report_id=report_id,
                report_type=report_type,
                period_start=period_start,
                period_end=period_end,
                compliance_score=compliance_score,
                findings=findings,
                recommendations=recommendations,
                evidence=evidence
            )

            self.compliance_reports.append(report)

            logger.info(f"Generated compliance report: {report_id} (score: {compliance_score:.1f}%)")
            return report

        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            raise e

    async def _generate_soc2_findings(self, events: List[AuditEvent]) -> Tuple[List[Dict[str, Any]], List[str], float]:
        """Generate SOC2 compliance findings"""
        findings = []
        recommendations = []
        score = 100.0

        # Check authentication events
        auth_failures = len([e for e in events if e.event_type == AuditEventType.AUTHENTICATION and e.outcome == "failure"])
        if auth_failures > 50:  # Threshold
            findings.append({
                'category': 'Security',
                'severity': 'Medium',
                'description': f'High number of authentication failures: {auth_failures}',
                'recommendation': 'Review authentication security and implement stronger controls'
            })
            recommendations.append('Implement account lockout policies and monitoring')
            score -= 5.0

        # Check authorization events
        auth_denials = len([e for e in events if e.event_type == AuditEventType.AUTHORIZATION and e.outcome == "denied"])
        if auth_denials > 100:
            findings.append({
                'category': 'Security',
                'severity': 'Low',
                'description': f'High number of authorization denials: {auth_denials}',
                'recommendation': 'Review access control policies'
            })
            score -= 2.0

        # Check for high-risk events
        high_risk_events = len([e for e in events if e.risk_score > 0.7])
        if high_risk_events > 10:
            findings.append({
                'category': 'Security',
                'severity': 'High',
                'description': f'Multiple high-risk security events: {high_risk_events}',
                'recommendation': 'Investigate and remediate high-risk security events'
            })
            recommendations.append('Implement enhanced security monitoring and alerting')
            score -= 10.0

        return findings, recommendations, max(0.0, score)

    async def _generate_iso27001_findings(self, events: List[AuditEvent]) -> Tuple[List[Dict[str, Any]], List[str], float]:
        """Generate ISO27001 compliance findings"""
        findings = []
        recommendations = []
        score = 100.0

        # Similar structure to SOC2 but with ISO27001 specific controls
        # This is a simplified implementation

        return findings, recommendations, score

    async def _generate_gdpr_findings(self, events: List[AuditEvent]) -> Tuple[List[Dict[str, Any]], List[str], float]:
        """Generate GDPR compliance findings"""
        findings = []
        recommendations = []
        score = 100.0

        # Check data access events
        data_access_events = len([e for e in events if e.event_type == AuditEventType.DATA_ACCESS])
        if data_access_events == 0:
            findings.append({
                'category': 'Data Protection',
                'severity': 'Medium',
                'description': 'No data access events logged',
                'recommendation': 'Implement comprehensive data access logging'
            })
            score -= 15.0

        return findings, recommendations, max(0.0, score)

    async def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        try:
            # Count active principals by type
            user_count = len([p for p in self.principals.values() if p.principal_type == 'user' and p.active])
            service_count = len([p for p in self.principals.values() if p.principal_type == 'service' and p.active])

            # Active tokens
            active_tokens = len([t for t in self.tokens.values() if datetime.now() < t.expires_at])

            # Recent audit events
            recent_events = len([e for e in self.audit_log if (datetime.now() - e.timestamp).total_seconds() < 3600])

            # Security metrics
            auth_failures_24h = len([e for e in self.audit_log
                                   if e.event_type == AuditEventType.AUTHENTICATION
                                   and e.outcome == "failure"
                                   and (datetime.now() - e.timestamp).total_seconds() < 86400])

            suspicious_activities_24h = len([a for a in self.suspicious_activities
                                           if (datetime.now() - a['timestamp']).total_seconds() < 86400])

            return {
                'security_manager_id': self.security_id,
                'principals': {
                    'total': len(self.principals),
                    'active': len([p for p in self.principals.values() if p.active]),
                    'users': user_count,
                    'services': service_count,
                    'mfa_enabled': len([p for p in self.principals.values() if p.mfa_enabled])
                },
                'authentication': {
                    'active_tokens': active_tokens,
                    'total_tokens_issued': len(self.tokens),
                    'auth_failures_24h': auth_failures_24h,
                    'lockout_accounts': len([pid for pid in self.failed_attempts.keys() if await self._is_locked_out(pid)])
                },
                'security_policies': {
                    'total_policies': len(self.security_policies),
                    'active_policies': len([p for p in self.security_policies.values() if p.active])
                },
                'audit_trail': {
                    'total_events': len(self.audit_log),
                    'recent_events_1h': recent_events,
                    'high_risk_events_24h': len([e for e in self.audit_log
                                               if e.risk_score > 0.7
                                               and (datetime.now() - e.timestamp).total_seconds() < 86400])
                },
                'security_monitoring': {
                    'suspicious_activities_24h': suspicious_activities_24h,
                    'threat_detections': len(self.suspicious_activities)
                },
                'compliance': {
                    'reports_generated': len(self.compliance_reports),
                    'latest_report': self.compliance_reports[-1].report_id if self.compliance_reports else None
                },
                'mtls': {
                    'ca_certificate_exists': os.path.exists(self.ca_cert_path),
                    'ca_certificate_path': self.ca_cert_path
                },
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get security status: {e}")
            return {'error': str(e)}

# Example usage and testing
async def main():
    """Example usage of XORB Security Manager"""
    try:
        print("🔐 XORB Security Manager initializing...")

        # Initialize security manager
        security_manager = XORBSecurityManager({
            'jwt_secret': 'test_secret_key_for_demo',
            'token_expiry_minutes': 60,
            'max_failed_attempts': 3,
            'cert_validity_days': 365
        })

        print("✅ Security manager initialized")

        # Create test principals
        print("\n👤 Creating security principals...")

        # Create admin user
        admin_principal = await security_manager.create_principal(
            principal_id="admin_001",
            principal_type="user",
            name="System Administrator",
            roles=[SecurityRole.ADMIN],
            attributes={
                'password_hash': hashlib.sha256('admin_password'.encode()).hexdigest(),
                'email': 'admin@xorb.local'
            }
        )

        # Create service principal
        service_principal = await security_manager.create_principal(
            principal_id="neural_orchestrator",
            principal_type="service",
            name="Neural Orchestrator Service",
            roles=[SecurityRole.SERVICE],
            attributes={
                'api_key': 'service_api_key_12345'
            }
        )

        print("✅ Security principals created")

        # Test authentication
        print("\n🔑 Testing authentication...")

        # Authenticate admin
        admin_token = await security_manager.authenticate_principal(
            "admin_001",
            {'password': 'admin_password'},
            source_ip="192.168.1.100"
        )

        if admin_token:
            print(f"✅ Admin authenticated: {admin_token.token_id}")

            # Test authorization
            print("\n🛡️ Testing authorization...")

            # Validate token
            validated_principal = await security_manager.validate_token(admin_token.metadata['jwt_token'])
            if validated_principal:
                print("✅ Token validated")

                # Test authorization
                authorized = await security_manager.authorize_action(
                    validated_principal,
                    "system_configuration",
                    "write",
                    context={'source_ip': '192.168.1.100'}
                )

                if authorized:
                    print("✅ Authorization granted")
                else:
                    print("❌ Authorization denied")

        # Generate service certificate
        print("\n📜 Generating service certificate...")
        cert_pem, key_pem = await security_manager.generate_service_certificate(
            "neural_orchestrator",
            ["localhost", "neural-orchestrator.xorb.local"]
        )
        print("✅ Service certificate generated")

        # Generate compliance report
        print("\n📋 Generating compliance report...")
        report = await security_manager.generate_compliance_report(
            "SOC2",
            datetime.now() - timedelta(days=30),
            datetime.now()
        )
        print(f"✅ Compliance report generated: {report.report_id} (score: {report.compliance_score:.1f}%)")

        # Get security status
        status = await security_manager.get_security_status()
        print(f"\n📊 Security Status:")
        print(f"- Total Principals: {status['principals']['total']}")
        print(f"- Active Tokens: {status['authentication']['active_tokens']}")
        print(f"- Security Policies: {status['security_policies']['total_policies']}")
        print(f"- Audit Events: {status['audit_trail']['total_events']}")
        print(f"- CA Certificate: {'✅' if status['mtls']['ca_certificate_exists'] else '❌'}")

        print(f"\n✅ XORB Security Manager demonstration completed!")

    except Exception as e:
        logger.error(f"Main execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
