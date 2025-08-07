"""
Comprehensive audit logging middleware for security monitoring and compliance
"""

import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

from ..domain.exceptions import DomainException


class AuditEvent:
    """Represents an audit event"""
    
    def __init__(
        self,
        event_id: str,
        event_type: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        client_ip: str = "unknown",
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        outcome: str = "success",
        risk_level: str = "low",
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        self.event_id = event_id
        self.event_type = event_type
        self.user_id = user_id
        self.username = username
        self.client_ip = client_ip
        self.user_agent = user_agent
        self.resource = resource
        self.action = action
        self.outcome = outcome
        self.risk_level = risk_level
        self.details = details or {}
        self.timestamp = timestamp or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "user_id": self.user_id,
            "username": self.username,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "resource": self.resource,
            "action": self.action,
            "outcome": self.outcome,
            "risk_level": self.risk_level,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "compliance_fields": {
                "gdpr_relevant": self._is_gdpr_relevant(),
                "pci_relevant": self._is_pci_relevant(),
                "retention_period": self._get_retention_period()
            }
        }
    
    def _is_gdpr_relevant(self) -> bool:
        """Check if event is GDPR relevant"""
        gdpr_events = [
            "user_created", "user_updated", "user_deleted",
            "data_export", "data_deletion", "consent_changed"
        ]
        return self.event_type in gdpr_events or "personal_data" in self.details
    
    def _is_pci_relevant(self) -> bool:
        """Check if event is PCI DSS relevant"""
        pci_events = [
            "payment_processed", "card_data_accessed",
            "security_policy_change", "access_control_change"
        ]
        return self.event_type in pci_events
    
    def _get_retention_period(self) -> int:
        """Get retention period in days based on event type"""
        retention_map = {
            "authentication": 90,
            "authorization": 90,
            "data_access": 365,
            "security_event": 2555,  # 7 years
            "payment": 2555,
            "admin_action": 2555,
            "default": 365
        }
        
        for category, days in retention_map.items():
            if category in self.event_type or category in self.details.get("categories", []):
                return days
        
        return retention_map["default"]


class AuditLogger:
    """Centralized audit logging service"""
    
    def __init__(self, redis_client: redis.Redis, db_session: Optional[AsyncSession] = None):
        self.redis_client = redis_client
        self.db_session = db_session
        
        # Risk scoring weights
        self.risk_weights = {
            "failed_login": 2,
            "privilege_escalation": 5,
            "data_export": 3,
            "admin_action": 4,
            "security_violation": 5,
            "unusual_access_pattern": 3
        }
    
    async def log_event(self, event: AuditEvent):
        """Log audit event to multiple destinations"""
        event_dict = event.to_dict()
        
        # Store in Redis for real-time monitoring
        await self._store_in_redis(event_dict)
        
        # Store in database for long-term retention
        if self.db_session:
            await self._store_in_database(event_dict)
        
        # Check for security alerts
        await self._check_security_alerts(event)
        
        # Update user risk score
        if event.user_id:
            await self._update_user_risk_score(event)
    
    async def _store_in_redis(self, event_dict: Dict[str, Any]):
        """Store event in Redis for real-time access"""
        # Store in daily audit log
        date_key = datetime.utcnow().strftime('%Y-%m-%d')
        audit_key = f"audit_log:{date_key}"
        
        await self.redis_client.lpush(audit_key, json.dumps(event_dict))
        await self.redis_client.expire(audit_key, 86400 * 30)  # 30 days
        
        # Store in user-specific log
        if event_dict.get("user_id"):
            user_key = f"user_audit:{event_dict['user_id']}"
            await self.redis_client.lpush(user_key, json.dumps(event_dict))
            await self.redis_client.expire(user_key, 86400 * 90)  # 90 days
        
        # Store high-risk events separately
        if event_dict.get("risk_level") in ["high", "critical"]:
            risk_key = f"high_risk_events:{date_key}"
            await self.redis_client.lpush(risk_key, json.dumps(event_dict))
            await self.redis_client.expire(risk_key, 86400 * 365)  # 1 year
    
    async def _store_in_database(self, event_dict: Dict[str, Any]):
        """Store event in database for long-term retention"""
        # In a real implementation, this would insert into an audit_logs table
        # For now, we'll simulate this
        pass
    
    async def _check_security_alerts(self, event: AuditEvent):
        """Check if event should trigger security alerts"""
        alerts = []
        
        # Multiple failed logins
        if event.event_type == "authentication" and event.outcome == "failure":
            failed_count = await self._get_recent_failed_logins(event.user_id, event.client_ip)
            if failed_count >= 5:
                alerts.append({
                    "type": "multiple_failed_logins",
                    "severity": "high",
                    "message": f"Multiple failed login attempts detected",
                    "details": {"failed_count": failed_count, "user_id": event.user_id}
                })
        
        # Unusual access patterns
        if event.event_type == "data_access":
            if await self._is_unusual_access_pattern(event):
                alerts.append({
                    "type": "unusual_access_pattern",
                    "severity": "medium",
                    "message": "Unusual data access pattern detected",
                    "details": {"resource": event.resource, "user_id": event.user_id}
                })
        
        # Privilege escalation attempts
        if "admin" in event.details.get("action", "") and event.outcome == "failure":
            alerts.append({
                "type": "privilege_escalation_attempt",
                "severity": "high",
                "message": "Potential privilege escalation attempt",
                "details": {"user_id": event.user_id, "action": event.action}
            })
        
        # Send alerts
        for alert in alerts:
            await self._send_security_alert(alert, event)
    
    async def _get_recent_failed_logins(self, user_id: str, client_ip: str) -> int:
        """Get count of recent failed logins"""
        # Check last hour for failed logins from this IP or user
        key = f"failed_logins:{user_id}:{client_ip}"
        count = await self.redis_client.get(key)
        return int(count) if count else 0
    
    async def _is_unusual_access_pattern(self, event: AuditEvent) -> bool:
        """Detect unusual access patterns"""
        # Check for access outside normal hours
        hour = event.timestamp.hour
        if hour < 6 or hour > 22:  # Outside 6 AM - 10 PM
            return True
        
        # Check for rapid successive accesses
        if event.user_id:
            recent_key = f"recent_access:{event.user_id}"
            recent_count = await self.redis_client.llen(recent_key)
            if recent_count > 10:  # More than 10 accesses in recent window
                return True
        
        return False
    
    async def _update_user_risk_score(self, event: AuditEvent):
        """Update user risk score based on activity"""
        if not event.user_id:
            return
        
        risk_increase = self.risk_weights.get(event.event_type, 1)
        if event.outcome == "failure":
            risk_increase *= 2
        
        risk_key = f"user_risk:{event.user_id}"
        current_score = await self.redis_client.get(risk_key)
        current_score = int(current_score) if current_score else 0
        
        new_score = min(current_score + risk_increase, 100)  # Cap at 100
        
        await self.redis_client.setex(risk_key, 86400 * 7, new_score)  # 7 days
        
        # Log high risk users
        if new_score > 70:
            await self.log_event(AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type="high_risk_user",
                user_id=event.user_id,
                risk_level="high",
                details={"risk_score": new_score, "trigger_event": event.event_type}
            ))
    
    async def _send_security_alert(self, alert: Dict[str, Any], event: AuditEvent):
        """Send security alert to monitoring systems"""
        alert_data = {
            "alert": alert,
            "event": event.to_dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store alert in Redis
        alert_key = f"security_alerts:{datetime.utcnow().strftime('%Y-%m-%d')}"
        await self.redis_client.lpush(alert_key, json.dumps(alert_data))
        await self.redis_client.expire(alert_key, 86400 * 90)  # 90 days
        
        # In production, this would also:
        # - Send email/SMS notifications
        # - Post to Slack/Teams
        # - Send to SIEM system
        # - Trigger automated responses
    
    async def get_audit_trail(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit trail with filtering"""
        events = []
        
        if user_id:
            # Get user-specific events
            user_key = f"user_audit:{user_id}"
            raw_events = await self.redis_client.lrange(user_key, 0, limit - 1)
        else:
            # Get general audit log
            date_key = datetime.utcnow().strftime('%Y-%m-%d')
            audit_key = f"audit_log:{date_key}"
            raw_events = await self.redis_client.lrange(audit_key, 0, limit - 1)
        
        for raw_event in raw_events:
            try:
                event_dict = json.loads(raw_event)
                
                # Apply filters
                if event_type and event_dict.get("event_type") != event_type:
                    continue
                
                if start_date and datetime.fromisoformat(event_dict["timestamp"]) < start_date:
                    continue
                
                if end_date and datetime.fromisoformat(event_dict["timestamp"]) > end_date:
                    continue
                
                events.append(event_dict)
                
            except json.JSONDecodeError:
                continue
        
        return events


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all API requests and responses"""
    
    def __init__(self, app, redis_client: redis.Redis):
        super().__init__(app)
        self.audit_logger = AuditLogger(redis_client)
        
        # Sensitive endpoints that require special logging
        self.sensitive_endpoints = {
            "/auth/login": "authentication",
            "/auth/register": "user_creation",
            "/admin": "admin_action",
            "/users": "user_management",
            "/api/v1/embeddings": "data_processing"
        }
        
        # Fields to redact in logs
        self.redacted_fields = {
            "password", "token", "secret", "key", "credential",
            "authorization", "x-api-key", "cookie"
        }
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        event_id = str(uuid.uuid4())
        
        # Extract request information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Get user information if available
        user_id = None
        username = None
        try:
            # Try to extract user from request state (set by auth middleware)
            user = getattr(request.state, "user", None)
            if user:
                user_id = str(user.id)
                username = user.username
        except:
            pass
        
        # Determine event type
        event_type = self._determine_event_type(request)
        risk_level = self._assess_risk_level(request, event_type)
        
        try:
            # Process request
            response = await call_next(request)
            outcome = "success" if response.status_code < 400 else "failure"
            
            # Log the request/response
            await self._log_api_request(
                event_id=event_id,
                request=request,
                response=response,
                user_id=user_id,
                username=username,
                client_ip=client_ip,
                user_agent=user_agent,
                event_type=event_type,
                outcome=outcome,
                risk_level=risk_level,
                duration=time.time() - start_time
            )
            
            return response
            
        except Exception as e:
            # Log failed requests
            await self._log_api_request(
                event_id=event_id,
                request=request,
                response=None,
                user_id=user_id,
                username=username,
                client_ip=client_ip,
                user_agent=user_agent,
                event_type=event_type,
                outcome="error",
                risk_level="high",
                duration=time.time() - start_time,
                error=str(e)
            )
            raise
    
    async def _log_api_request(
        self,
        event_id: str,
        request: Request,
        response: Optional[Response],
        user_id: Optional[str],
        username: Optional[str],
        client_ip: str,
        user_agent: str,
        event_type: str,
        outcome: str,
        risk_level: str,
        duration: float,
        error: Optional[str] = None
    ):
        """Log API request details"""
        
        details = {
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": self._redact_sensitive_headers(dict(request.headers)),
            "duration_ms": round(duration * 1000, 2)
        }
        
        if response:
            details["status_code"] = response.status_code
            details["response_headers"] = self._redact_sensitive_headers(dict(response.headers))
        
        if error:
            details["error"] = error
        
        # Add request body for sensitive operations (with redaction)
        if event_type in ["authentication", "user_creation", "admin_action"]:
            body = await self._get_request_body(request)
            if body:
                details["request_body"] = self._redact_sensitive_data(body)
        
        event = AuditEvent(
            event_id=event_id,
            event_type=f"api_{event_type}",
            user_id=user_id,
            username=username,
            client_ip=client_ip,
            user_agent=user_agent,
            resource=request.url.path,
            action=request.method,
            outcome=outcome,
            risk_level=risk_level,
            details=details
        )
        
        await self.audit_logger.log_event(event)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _determine_event_type(self, request: Request) -> str:
        """Determine event type based on request path"""
        path = request.url.path
        
        for endpoint, event_type in self.sensitive_endpoints.items():
            if path.startswith(endpoint):
                return event_type
        
        # Default categorization
        if path.startswith("/admin"):
            return "admin_action"
        elif path.startswith("/auth"):
            return "authentication"
        elif path.startswith("/api"):
            return "api_access"
        else:
            return "web_access"
    
    def _assess_risk_level(self, request: Request, event_type: str) -> str:
        """Assess risk level of the request"""
        # High risk operations
        if event_type in ["admin_action", "user_creation"]:
            return "high"
        
        # Medium risk operations
        if event_type in ["authentication", "data_processing"]:
            return "medium"
        
        # Check for suspicious patterns
        if request.method in ["DELETE", "PUT"] and not request.url.path.startswith("/api/v1"):
            return "medium"
        
        return "low"
    
    def _redact_sensitive_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Redact sensitive information from headers"""
        redacted = {}
        for key, value in headers.items():
            key_lower = key.lower()
            if any(field in key_lower for field in self.redacted_fields):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = value
        return redacted
    
    def _redact_sensitive_data(self, data: Any) -> Any:
        """Redact sensitive information from request/response data"""
        if isinstance(data, dict):
            redacted = {}
            for key, value in data.items():
                key_lower = key.lower()
                if any(field in key_lower for field in self.redacted_fields):
                    redacted[key] = "[REDACTED]"
                else:
                    redacted[key] = self._redact_sensitive_data(value)
            return redacted
        elif isinstance(data, list):
            return [self._redact_sensitive_data(item) for item in data]
        else:
            return data
    
    async def _get_request_body(self, request: Request) -> Optional[Dict]:
        """Get request body if available"""
        try:
            if request.headers.get("content-type", "").startswith("application/json"):
                body = await request.json()
                return body
        except:
            pass
        return None