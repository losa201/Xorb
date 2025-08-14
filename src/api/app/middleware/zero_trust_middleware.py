"""
Zero Trust Security Middleware
Implements zero-trust security controls at the middleware level
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from uuid import uuid4
import hashlib
import ipaddress

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..core.zero_trust_engine import (
    ZeroTrustEngine, SecurityContext, TrustLevel, AccessDecision,
    get_zero_trust_engine, evaluate_request_trust
)
from ..core.logging import get_logger
from ..services.enhanced_ml_threat_intelligence import get_ml_threat_intelligence

logger = get_logger(__name__)

class ZeroTrustSecurityMiddleware(BaseHTTPMiddleware):
    """
    Zero Trust Security Middleware
    
    Implements comprehensive zero-trust security controls:
    - Continuous verification of all requests
    - Risk-based access control
    - Behavioral analysis
    - Threat detection
    - Adaptive security policies
    """
    
    def __init__(
        self,
        app: ASGIApp,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(app)
        self.config = config or {}
        self.engine = get_zero_trust_engine()
        self.ml_intelligence = get_ml_threat_intelligence()
        
        # Configuration
        self.enabled = self.config.get('enabled', True)
        self.bypass_paths = self.config.get('bypass_paths', [
            '/docs', '/redoc', '/openapi.json', '/health', '/readiness'
        ])
        self.challenge_paths = self.config.get('challenge_paths', [
            '/api/v1/admin', '/api/v1/sensitive'
        ])
        
        # Trust thresholds
        self.min_trust_level = TrustLevel(self.config.get('min_trust_level', TrustLevel.MEDIUM.value))
        self.challenge_threshold = self.config.get('challenge_threshold', 0.6)
        self.block_threshold = self.config.get('block_threshold', 0.3)
        
        # Rate limiting for security events
        self.security_event_cache: Dict[str, List[datetime]] = {}
        self.max_security_events_per_hour = self.config.get('max_security_events_per_hour', 10)
        
        # Request tracking
        self.request_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = timedelta(hours=1)
        
        logger.info("Zero Trust Security Middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through zero trust security pipeline"""
        if not self.enabled:
            return await call_next(request)
        
        start_time = time.time()
        
        try:
            # Extract security context
            security_context = await self._extract_security_context(request)
            
            # Check if path should bypass zero trust
            if self._should_bypass_path(request.url.path):
                logger.debug("Path bypassed zero trust evaluation", path=request.url.path)
                return await call_next(request)
            
            # Evaluate trust level
            trust_level, trust_score, access_decision = await self.engine.evaluate_trust(security_context)
            
            # Log security evaluation
            await self._log_security_evaluation(
                request, security_context, trust_level, trust_score, access_decision
            )
            
            # Apply access decision
            response = await self._apply_access_decision(
                request, call_next, security_context, access_decision, trust_score
            )
            
            # Add security headers
            await self._add_security_headers(response, trust_level, trust_score)
            
            # Update request tracking
            await self._update_request_tracking(security_context, trust_level, trust_score)
            
            # Process time tracking
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Trust-Level"] = trust_level.name
            response.headers["X-Trust-Score"] = f"{trust_score:.3f}"
            
            return response
            
        except Exception as e:
            logger.error("Zero trust middleware error", error=str(e), path=request.url.path)
            # In case of error, default to deny access
            return await self._create_security_error_response(
                "Security evaluation failed", 
                status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    async def _extract_security_context(self, request: Request) -> SecurityContext:
        """Extract security context from request"""
        # Get user information
        user_id = await self._extract_user_id(request)
        session_id = await self._extract_session_id(request)
        device_id = await self._extract_device_id(request)
        
        # Get network information
        ip_address = await self._get_client_ip(request)
        user_agent = request.headers.get('User-Agent', 'Unknown')
        
        # Get additional context
        location = await self._get_location_info(ip_address)
        mfa_verified = await self._check_mfa_status(request)
        device_trusted = await self._check_device_trust(device_id)
        network_trusted = await self._check_network_trust(ip_address)
        
        # Get behavioral score
        behavior_score = await self._calculate_behavior_score(user_id, request)
        
        # Get risk indicators
        risk_indicators = await self._get_risk_indicators(request)
        
        # Get previous sessions
        previous_sessions = await self._get_previous_sessions(user_id)
        
        return SecurityContext(
            user_id=user_id,
            session_id=session_id,
            device_id=device_id,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.utcnow(),
            location=location,
            mfa_verified=mfa_verified,
            device_trusted=device_trusted,
            network_trusted=network_trusted,
            behavior_score=behavior_score,
            risk_indicators=risk_indicators,
            previous_sessions=previous_sessions
        )
    
    async def _extract_user_id(self, request: Request) -> str:
        """Extract user ID from request"""
        # Try to get from JWT token
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            try:
                # This would typically decode JWT and extract user ID
                # For now, use a placeholder
                token = auth_header[7:]
                user_id = f"user_{hashlib.md5(token.encode()).hexdigest()[:8]}"
                return user_id
            except Exception:
                pass
        
        # Try to get from session
        session_user = request.state.__dict__.get('user_id')
        if session_user:
            return str(session_user)
        
        # Default to anonymous
        return "anonymous"
    
    async def _extract_session_id(self, request: Request) -> str:
        """Extract session ID from request"""
        # Try various session ID sources
        session_id = (
            request.headers.get('X-Session-ID') or
            request.cookies.get('session_id') or
            request.state.__dict__.get('session_id')
        )
        
        if session_id:
            return str(session_id)
        
        # Generate temporary session ID
        return f"temp_{uuid4().hex[:8]}"
    
    async def _extract_device_id(self, request: Request) -> str:
        """Extract device ID from request"""
        # Try device fingerprinting based on headers
        fingerprint_components = [
            request.headers.get('User-Agent', ''),
            request.headers.get('Accept-Language', ''),
            request.headers.get('Accept-Encoding', ''),
            request.headers.get('Accept', ''),
            str(request.headers.get('X-Forwarded-For', ''))
        ]
        
        fingerprint = '|'.join(fingerprint_components)
        device_id = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
        
        return f"device_{device_id}"
    
    async def _get_client_ip(self, request: Request) -> str:
        """Get client IP address with proxy support"""
        # Check for forwarded headers
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            # Get the first IP in the chain (original client)
            ip = forwarded_for.split(',')[0].strip()
            try:
                ipaddress.ip_address(ip)  # Validate IP
                return ip
            except ValueError:
                pass
        
        # Check for real IP header
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            try:
                ipaddress.ip_address(real_ip)
                return real_ip
            except ValueError:
                pass
        
        # Default to client host
        client_host = getattr(request.client, 'host', '127.0.0.1')
        return client_host
    
    async def _get_location_info(self, ip_address: str) -> Optional[Dict[str, str]]:
        """Get location information for IP address"""
        # This would typically use a GeoIP service
        # For now, return basic info for private/public IPs
        try:
            ip = ipaddress.ip_address(ip_address)
            if ip.is_private:
                return {'country': 'LOCAL', 'region': 'PRIVATE', 'city': 'PRIVATE'}
            else:
                # Would integrate with GeoIP service here
                return {'country': 'UNKNOWN', 'region': 'UNKNOWN', 'city': 'UNKNOWN'}
        except ValueError:
            return None
    
    async def _check_mfa_status(self, request: Request) -> bool:
        """Check if request has valid MFA verification"""
        # Check for MFA tokens/headers
        mfa_token = (
            request.headers.get('X-MFA-Token') or
            request.headers.get('X-TOTP-Token')
        )
        
        # Check session state
        mfa_verified = request.state.__dict__.get('mfa_verified', False)
        
        return bool(mfa_token) or mfa_verified
    
    async def _check_device_trust(self, device_id: str) -> bool:
        """Check if device is trusted"""
        # This would typically check against a device trust database
        # For now, use simple heuristics
        
        # Check if device has been seen before
        if device_id in self.request_cache:
            device_history = self.request_cache[device_id]
            return device_history.get('trusted', False)
        
        return False
    
    async def _check_network_trust(self, ip_address: str) -> bool:
        """Check if network/IP is trusted"""
        try:
            ip = ipaddress.ip_address(ip_address)
            
            # Trust private networks
            if ip.is_private:
                return True
            
            # Trust specific IP ranges (would be configurable)
            trusted_ranges = self.config.get('trusted_ip_ranges', [])
            for range_str in trusted_ranges:
                if ip in ipaddress.ip_network(range_str):
                    return True
            
            # Check against threat intelligence
            # This would integrate with threat feeds
            known_bad_ips = self.config.get('blocked_ips', set())
            if ip_address in known_bad_ips:
                return False
            
            return False  # Default to untrusted for public IPs
            
        except ValueError:
            return False
    
    async def _calculate_behavior_score(self, user_id: str, request: Request) -> float:
        """Calculate behavioral analysis score"""
        if user_id == "anonymous":
            return 0.3  # Low score for anonymous users
        
        score = 0.5  # Base score
        
        # Check request patterns
        if user_id in self.request_cache:
            user_history = self.request_cache[user_id]
            
            # Consistent access patterns increase score
            recent_paths = user_history.get('recent_paths', [])
            if request.url.path in recent_paths:
                score += 0.2
            
            # Reasonable request frequency
            last_request_time = user_history.get('last_request_time')
            if last_request_time:
                time_since_last = datetime.utcnow() - last_request_time
                if timedelta(seconds=5) < time_since_last < timedelta(minutes=30):
                    score += 0.1
            
            # No recent security events
            security_events = user_history.get('security_events', 0)
            if security_events == 0:
                score += 0.2
        
        return min(1.0, score)
    
    async def _get_risk_indicators(self, request: Request) -> List[str]:
        """Get risk indicators for request"""
        indicators = []
        
        # Check User-Agent for suspicious patterns
        user_agent = request.headers.get('User-Agent', '').lower()
        suspicious_agents = ['bot', 'crawler', 'scanner', 'curl', 'wget', 'python-requests']
        if any(agent in user_agent for agent in suspicious_agents):
            indicators.append('suspicious_user_agent')
        
        # Check for SQL injection patterns in URL
        url_str = str(request.url).lower()
        sql_patterns = ['union', 'select', 'insert', 'delete', 'drop', 'exec', '1=1', '1=2']
        if any(pattern in url_str for pattern in sql_patterns):
            indicators.append('potential_sql_injection')
        
        # Check for XSS patterns
        xss_patterns = ['<script', 'javascript:', 'onload=', 'onerror=']
        if any(pattern in url_str for pattern in xss_patterns):
            indicators.append('potential_xss')
        
        # Check request size
        content_length = request.headers.get('Content-Length')
        if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB
            indicators.append('large_request_body')
        
        # Check for directory traversal
        if '../' in url_str or '..\\' in url_str:
            indicators.append('directory_traversal')
        
        return indicators
    
    async def _get_previous_sessions(self, user_id: str) -> List[str]:
        """Get previous session IDs for user"""
        if user_id in self.request_cache:
            return self.request_cache[user_id].get('previous_sessions', [])
        return []
    
    def _should_bypass_path(self, path: str) -> bool:
        """Check if path should bypass zero trust evaluation"""
        return any(bypass_path in path for bypass_path in self.bypass_paths)
    
    async def _apply_access_decision(
        self,
        request: Request,
        call_next: Callable,
        context: SecurityContext,
        decision: AccessDecision,
        trust_score: float
    ) -> Response:
        """Apply access control decision"""
        
        if decision == AccessDecision.ALLOW:
            # Start session monitoring
            await self.engine.start_session_monitoring(context)
            return await call_next(request)
        
        elif decision == AccessDecision.DENY:
            await self._record_security_event(context, "access_denied", trust_score)
            return await self._create_security_error_response(
                "Access denied by security policy",
                status.HTTP_403_FORBIDDEN
            )
        
        elif decision == AccessDecision.CHALLENGE:
            # Check if this is a sensitive path that requires higher trust
            if any(path in request.url.path for path in self.challenge_paths):
                return await self._create_challenge_response(context)
            else:
                # Allow with monitoring
                await self.engine.start_session_monitoring(context)
                return await call_next(request)
        
        elif decision == AccessDecision.MONITOR:
            # Allow but with enhanced monitoring
            await self.engine.start_session_monitoring(context)
            response = await call_next(request)
            response.headers["X-Security-Monitor"] = "enhanced"
            return response
        
        elif decision == AccessDecision.QUARANTINE:
            await self._record_security_event(context, "quarantine", trust_score)
            return await self._create_security_error_response(
                "Request quarantined for security review",
                status.HTTP_403_FORBIDDEN
            )
        
        else:
            # Default to deny
            return await self._create_security_error_response(
                "Unknown security decision",
                status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    async def _create_challenge_response(self, context: SecurityContext) -> Response:
        """Create MFA challenge response"""
        challenge_data = {
            "error": "additional_authentication_required",
            "message": "This action requires additional authentication",
            "challenge_id": f"challenge_{uuid4().hex[:8]}",
            "required_factors": ["mfa", "device_verification"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self._record_security_event(context, "mfa_challenge", 0.5)
        
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=challenge_data,
            headers={
                "WWW-Authenticate": "Bearer",
                "X-Challenge-Required": "true"
            }
        )
    
    async def _create_security_error_response(self, message: str, status_code: int) -> Response:
        """Create security error response"""
        error_data = {
            "error": "security_violation",
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "reference": f"ref_{uuid4().hex[:8]}"
        }
        
        return JSONResponse(
            status_code=status_code,
            content=error_data,
            headers={"X-Security-Error": "true"}
        )
    
    async def _add_security_headers(self, response: Response, trust_level: TrustLevel, trust_score: float):
        """Add security headers to response"""
        # Standard security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
        }
        
        # Zero trust specific headers
        security_headers.update({
            "X-Trust-Level": trust_level.name,
            "X-Trust-Score": f"{trust_score:.3f}",
            "X-Security-Policy": "zero-trust-enforced"
        })
        
        # Apply headers
        for header, value in security_headers.items():
            response.headers[header] = value
    
    async def _update_request_tracking(self, context: SecurityContext, trust_level: TrustLevel, trust_score: float):
        """Update request tracking information"""
        user_id = context.user_id
        
        if user_id not in self.request_cache:
            self.request_cache[user_id] = {
                'recent_paths': [],
                'previous_sessions': [],
                'security_events': 0,
                'trust_history': [],
                'created_at': datetime.utcnow()
            }
        
        user_data = self.request_cache[user_id]
        
        # Update request history
        user_data['last_request_time'] = context.timestamp
        user_data['recent_paths'].append(str(context.timestamp))  # Simplified
        
        # Limit history size
        if len(user_data['recent_paths']) > 100:
            user_data['recent_paths'] = user_data['recent_paths'][-50:]
        
        # Update trust history
        user_data['trust_history'].append({
            'timestamp': context.timestamp,
            'trust_level': trust_level.name,
            'trust_score': trust_score
        })
        
        # Limit trust history
        if len(user_data['trust_history']) > 1000:
            user_data['trust_history'] = user_data['trust_history'][-500:]
        
        # Update session tracking
        if context.session_id not in user_data['previous_sessions']:
            user_data['previous_sessions'].append(context.session_id)
            
            # Limit session history
            if len(user_data['previous_sessions']) > 50:
                user_data['previous_sessions'] = user_data['previous_sessions'][-25:]
        
        # Clean up old entries
        await self._cleanup_old_cache_entries()
    
    async def _record_security_event(self, context: SecurityContext, event_type: str, trust_score: float):
        """Record security event"""
        user_id = context.user_id
        
        # Update security event counter
        if user_id in self.request_cache:
            self.request_cache[user_id]['security_events'] += 1
        
        # Rate limiting for security events
        if user_id not in self.security_event_cache:
            self.security_event_cache[user_id] = []
        
        now = datetime.utcnow()
        self.security_event_cache[user_id].append(now)
        
        # Clean old events (older than 1 hour)
        hour_ago = now - timedelta(hours=1)
        self.security_event_cache[user_id] = [
            event_time for event_time in self.security_event_cache[user_id]
            if event_time > hour_ago
        ]
        
        # Check if user exceeds security event threshold
        if len(self.security_event_cache[user_id]) > self.max_security_events_per_hour:
            logger.warning(
                "User exceeded security event threshold",
                user_id=user_id,
                event_count=len(self.security_event_cache[user_id]),
                threshold=self.max_security_events_per_hour
            )
        
        # Log the security event
        logger.warning(
            "Security event recorded",
            user_id=user_id,
            session_id=context.session_id,
            event_type=event_type,
            trust_score=trust_score,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            timestamp=context.timestamp.isoformat()
        )
    
    async def _log_security_evaluation(
        self,
        request: Request,
        context: SecurityContext,
        trust_level: TrustLevel,
        trust_score: float,
        access_decision: AccessDecision
    ):
        """Log security evaluation details"""
        logger.info(
            "Zero trust security evaluation",
            user_id=context.user_id,
            session_id=context.session_id,
            path=request.url.path,
            method=request.method,
            trust_level=trust_level.name,
            trust_score=trust_score,
            access_decision=access_decision.value,
            ip_address=context.ip_address,
            device_id=context.device_id,
            mfa_verified=context.mfa_verified,
            device_trusted=context.device_trusted,
            network_trusted=context.network_trusted,
            behavior_score=context.behavior_score,
            risk_indicators=context.risk_indicators
        )
    
    async def _cleanup_old_cache_entries(self):
        """Clean up old cache entries"""
        cutoff_time = datetime.utcnow() - self.cache_ttl
        
        # Clean request cache
        users_to_remove = []
        for user_id, user_data in self.request_cache.items():
            if user_data.get('created_at', datetime.utcnow()) < cutoff_time:
                users_to_remove.append(user_id)
        
        for user_id in users_to_remove:
            del self.request_cache[user_id]
        
        # Clean security event cache
        for user_id in list(self.security_event_cache.keys()):
            events = self.security_event_cache[user_id]
            recent_events = [event for event in events if event > cutoff_time]
            
            if recent_events:
                self.security_event_cache[user_id] = recent_events
            else:
                del self.security_event_cache[user_id]

# Middleware factory function
def create_zero_trust_middleware(config: Optional[Dict[str, Any]] = None):
    """Create zero trust middleware with configuration"""
    def middleware_factory(app: ASGIApp) -> ZeroTrustSecurityMiddleware:
        return ZeroTrustSecurityMiddleware(app, config)
    return middleware_factory