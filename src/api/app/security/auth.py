"""
XORB API Authentication and Authorization
Implements Zero Trust security with mTLS, JWT, and RBAC
"""
import os
import jwt
import time
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from enum import Enum
from dataclasses import dataclass
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


class Role(Enum):
    """Security roles with hierarchical permissions"""
    ADMIN = "admin"
    ORCHESTRATOR = "orchestrator"
    ANALYST = "analyst"
    AGENT = "agent"
    READONLY = "readonly"


class Permission(Enum):
    """Granular permissions for API operations"""
    # Agent Management
    AGENT_CREATE = "agent:create"
    AGENT_READ = "agent:read"
    AGENT_UPDATE = "agent:update"
    AGENT_DELETE = "agent:delete"
    
    # Task Orchestration
    TASK_SUBMIT = "task:submit"
    TASK_READ = "task:read"
    TASK_PRIORITY = "task:priority"
    TASK_CANCEL = "task:cancel"
    
    # Security Operations
    SECURITY_READ = "security:read"
    SECURITY_RESPOND = "security:respond"
    SECURITY_CONFIG = "security:config"
    
    # Configuration
    CONFIG_READ = "config:read"
    CONFIG_WRITE = "config:write"
    
    # Telemetry
    TELEMETRY_READ = "telemetry:read"
    TELEMETRY_WRITE = "telemetry:write"
    
    # System Administration
    SYSTEM_ADMIN = "system:admin"


@dataclass
class SecurityContext:
    """Security context for authenticated requests"""
    user_id: str
    client_id: str
    roles: List[Role]
    permissions: List[Permission]
    certificate_fingerprint: Optional[str]
    request_id: str
    timestamp: datetime


class RolePermissions:
    """Role-based permission mapping"""
    ROLE_PERMISSIONS = {
        Role.ADMIN: [
            Permission.AGENT_CREATE, Permission.AGENT_READ, Permission.AGENT_UPDATE, Permission.AGENT_DELETE,
            Permission.TASK_SUBMIT, Permission.TASK_READ, Permission.TASK_PRIORITY, Permission.TASK_CANCEL,
            Permission.SECURITY_READ, Permission.SECURITY_RESPOND, Permission.SECURITY_CONFIG,
            Permission.CONFIG_READ, Permission.CONFIG_WRITE,
            Permission.TELEMETRY_READ, Permission.TELEMETRY_WRITE,
            Permission.SYSTEM_ADMIN
        ],
        Role.ORCHESTRATOR: [
            Permission.AGENT_CREATE, Permission.AGENT_READ, Permission.AGENT_UPDATE,
            Permission.TASK_SUBMIT, Permission.TASK_READ, Permission.TASK_PRIORITY,
            Permission.SECURITY_READ, Permission.SECURITY_RESPOND,
            Permission.CONFIG_READ,
            Permission.TELEMETRY_READ, Permission.TELEMETRY_WRITE
        ],
        Role.ANALYST: [
            Permission.AGENT_READ,
            Permission.TASK_READ,
            Permission.SECURITY_READ,
            Permission.CONFIG_READ,
            Permission.TELEMETRY_READ
        ],
        Role.AGENT: [
            Permission.AGENT_READ,
            Permission.TASK_READ,
            Permission.TELEMETRY_WRITE
        ],
        Role.READONLY: [
            Permission.AGENT_READ,
            Permission.TASK_READ,
            Permission.SECURITY_READ,
            Permission.TELEMETRY_READ
        ]
    }

    @classmethod
    def get_permissions(cls, roles: List[Role]) -> List[Permission]:
        """Get all permissions for given roles"""
        permissions = set()
        for role in roles:
            permissions.update(cls.ROLE_PERMISSIONS.get(role, []))
        return list(permissions)


class XORBAuthenticator:
    """Main authentication and authorization service"""
    
    def __init__(self):
        self.jwt_secret = os.getenv("JWT_SECRET", "xorb-secure-key-change-in-production")
        self.jwt_algorithm = "HS256"
        self.jwt_expiry_hours = 24
        
        # Certificate validation settings
        self.require_mtls = os.getenv("REQUIRE_MTLS", "true").lower() == "true"
        self.trusted_ca_path = os.getenv("TRUSTED_CA_PATH", "/etc/xorb/ca.crt")
        
        # Rate limiting
        self.rate_limits = {
            Role.ADMIN: 10000,
            Role.ORCHESTRATOR: 5000,
            Role.ANALYST: 1000,
            Role.AGENT: 2000,
            Role.READONLY: 500
        }
        
        # Request tracking for rate limiting
        self.request_counts: Dict[str, Dict[str, int]] = {}
        
    def generate_jwt(self, user_id: str, client_id: str, roles: List[Role]) -> str:
        """Generate JWT token for authenticated user"""
        payload = {
            "user_id": user_id,
            "client_id": client_id,
            "roles": [role.value for role in roles],
            "iat": int(time.time()),
            "exp": int(time.time()) + (self.jwt_expiry_hours * 3600),
            "iss": "xorb-api",
            "aud": "xorb-platform"
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def verify_jwt(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token, 
                self.jwt_secret, 
                algorithms=[self.jwt_algorithm],
                audience="xorb-platform",
                issuer="xorb-api"
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def verify_client_certificate(self, request: Request) -> Optional[str]:
        """Verify client certificate for mTLS"""
        if not self.require_mtls:
            return None
            
        # Get client certificate from request headers (set by reverse proxy)
        cert_header = request.headers.get("X-Client-Cert")
        if not cert_header:
            raise HTTPException(status_code=401, detail="Client certificate required")
        
        try:
            # Decode and verify certificate
            cert_data = cert_header.encode('utf-8')
            cert = x509.load_pem_x509_certificate(cert_data)
            
            # Verify certificate chain and validity
            self._verify_certificate_chain(cert)
            
            # Return certificate fingerprint
            fingerprint = hashlib.sha256(cert.fingerprint(hashes.SHA256())).hexdigest()
            return fingerprint
            
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Certificate verification failed: {str(e)}")
    
    def _verify_certificate_chain(self, cert: x509.Certificate):
        """Verify certificate against trusted CA"""
        # Implementation would verify against trusted CA
        # This is a placeholder for the actual implementation
        if cert.not_valid_after < datetime.utcnow():
            raise ValueError("Certificate expired")
        if cert.not_valid_before > datetime.utcnow():
            raise ValueError("Certificate not yet valid")
    
    def check_rate_limit(self, client_id: str, role: Role) -> bool:
        """Check if client has exceeded rate limit"""
        current_minute = int(time.time() // 60)
        
        if client_id not in self.request_counts:
            self.request_counts[client_id] = {}
        
        client_requests = self.request_counts[client_id]
        
        # Clean old entries
        for minute in list(client_requests.keys()):
            if minute < current_minute - 5:  # Keep last 5 minutes
                del client_requests[minute]
        
        # Count requests in current minute
        current_count = client_requests.get(current_minute, 0)
        limit = self.rate_limits.get(role, 100)  # Default limit
        
        if current_count >= limit:
            return False
        
        # Increment counter
        client_requests[current_minute] = current_count + 1
        return True
    
    def authenticate_request(self, request: Request, token: str) -> SecurityContext:
        """Complete authentication and authorization"""
        # Verify client certificate if required
        cert_fingerprint = self.verify_client_certificate(request)
        
        # Verify JWT token
        payload = self.verify_jwt(token)
        
        # Extract user information
        user_id = payload["user_id"]
        client_id = payload["client_id"]
        role_names = payload["roles"]
        
        # Convert roles
        roles = [Role(role) for role in role_names]
        
        # Get permissions
        permissions = RolePermissions.get_permissions(roles)
        
        # Check rate limits
        primary_role = roles[0] if roles else Role.READONLY
        if not self.check_rate_limit(client_id, primary_role):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Generate request ID for tracing
        request_id = hashlib.md5(f"{user_id}{client_id}{time.time()}".encode()).hexdigest()
        
        return SecurityContext(
            user_id=user_id,
            client_id=client_id,
            roles=roles,
            permissions=permissions,
            certificate_fingerprint=cert_fingerprint,
            request_id=request_id,
            timestamp=datetime.utcnow()
        )


# Global authenticator instance
authenticator = XORBAuthenticator()

# FastAPI security scheme
security = HTTPBearer()


def get_security_context(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> SecurityContext:
    """FastAPI dependency for authentication"""
    return authenticator.authenticate_request(request, credentials.credentials)


def require_permission(permission: Permission):
    """Decorator to require specific permission"""
    def permission_checker(context: SecurityContext = Depends(get_security_context)):
        if permission not in context.permissions:
            raise HTTPException(
                status_code=403, 
                detail=f"Permission required: {permission.value}"
            )
        return context
    return permission_checker


def require_role(required_role: Role):
    """Decorator to require specific role"""
    def role_checker(context: SecurityContext = Depends(get_security_context)):
        if required_role not in context.roles:
            raise HTTPException(
                status_code=403, 
                detail=f"Role required: {required_role.value}"
            )
        return context
    return role_checker


# Common permission dependencies
require_admin = require_role(Role.ADMIN)
require_orchestrator = require_permission(Permission.TASK_SUBMIT)
require_agent_management = require_permission(Permission.AGENT_CREATE)
require_security_ops = require_permission(Permission.SECURITY_RESPOND)
require_config_access = require_permission(Permission.CONFIG_READ)