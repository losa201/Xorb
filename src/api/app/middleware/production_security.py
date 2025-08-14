#!/usr/bin/env python3
"""
Production Security Middleware
Enterprise-grade security hardening and protection mechanisms
"""

import asyncio
import logging
import time
import hashlib
import secrets
import hmac
import json
import re
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import ipaddress
import urllib.parse

from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import redis.asyncio as redis

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

try:
    from cryptography.fernet import Fernet
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SecurityAction(Enum):
    ALLOW = "allow"
    RATE_LIMIT = "rate_limit"
    CHALLENGE = "challenge"
    BLOCK = "block"
    QUARANTINE = "quarantine"

@dataclass
class SecurityThreat:
    """Security threat detection result"""
    threat_type: str
    threat_level: ThreatLevel
    confidence: float
    description: str
    indicators: List[str]
    recommended_action: SecurityAction
    metadata: Dict[str, Any]

@dataclass
class SecurityContext:
    """Security context for requests"""
    client_ip: str
    user_agent: str
    request_fingerprint: str
    threat_score: float
    reputation_score: float
    geographic_info: Optional[Dict[str, Any]]
    request_patterns: Dict[str, Any]
    security_headers: Dict[str, str]

class IPReputationService:
    """IP reputation and geolocation service"""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.known_bad_ips: Set[str] = set()
        self.known_good_ips: Set[str] = set()
        self.reputation_cache: Dict[str, Tuple[float, datetime]] = {}
        self.cache_ttl = 3600  # 1 hour

        # Load threat intelligence feeds
        self._load_threat_feeds()

    def _load_threat_feeds(self):
        """Load known malicious IP ranges and patterns"""
        # Known malicious patterns (simplified for demo)
        self.malicious_patterns = [
            # Common attack sources
            r'^192\.168\.1\.100$',  # Example internal scanner
            r'^10\.0\.0\.1$',      # Example internal gateway
            # Add real threat intelligence feeds in production
        ]

        # Known cloud provider ranges (often used for attacks)
        self.cloud_ranges = [
            ipaddress.ip_network("54.0.0.0/8"),     # AWS
            ipaddress.ip_network("104.0.0.0/8"),    # Google Cloud
            ipaddress.ip_network("13.0.0.0/8"),     # Azure
        ]

    async def get_ip_reputation(self, ip_address: str) -> float:
        """Get IP reputation score (0.0 = bad, 1.0 = good)"""
        try:
            # Check cache first
            if ip_address in self.reputation_cache:
                score, cached_at = self.reputation_cache[ip_address]
                if datetime.utcnow() - cached_at < timedelta(seconds=self.cache_ttl):
                    return score

            # Check Redis cache if available
            if self.redis_client:
                try:
                    cached_score = await self.redis_client.get(f"ip_rep:{ip_address}")
                    if cached_score:
                        return float(cached_score)
                except Exception as e:
                    logger.warning(f"Redis lookup failed: {e}")

            # Calculate reputation score
            score = await self._calculate_reputation(ip_address)

            # Cache the result
            self.reputation_cache[ip_address] = (score, datetime.utcnow())

            if self.redis_client:
                try:
                    await self.redis_client.setex(f"ip_rep:{ip_address}", self.cache_ttl, str(score))
                except Exception as e:
                    logger.warning(f"Redis cache failed: {e}")

            return score

        except Exception as e:
            logger.error(f"IP reputation lookup failed for {ip_address}: {e}")
            return 0.5  # Neutral score on error

    async def _calculate_reputation(self, ip_address: str) -> float:
        """Calculate IP reputation based on various factors"""
        score = 0.8  # Default good score

        try:
            ip_obj = ipaddress.ip_address(ip_address)

            # Check if it's a private IP (generally trusted)
            if ip_obj.is_private:
                score += 0.15

            # Check against known bad patterns
            for pattern in self.malicious_patterns:
                if re.match(pattern, ip_address):
                    score -= 0.6
                    break

            # Check if it's from a cloud provider (neutral to slightly negative)
            for cloud_range in self.cloud_ranges:
                if ip_obj in cloud_range:
                    score -= 0.1
                    break

            # Check known lists
            if ip_address in self.known_bad_ips:
                score -= 0.8
            elif ip_address in self.known_good_ips:
                score += 0.15

            # Additional checks would go here:
            # - External threat intelligence APIs
            # - Historical attack patterns
            # - Geographic risk assessment

            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.error(f"Reputation calculation failed: {e}")
            return 0.5

class RequestAnalyzer:
    """Analyzes requests for suspicious patterns and anomalies"""

    def __init__(self):
        self.suspicious_patterns = [
            # SQL Injection patterns
            (r"(?i)(union|select|insert|delete|update|drop|exec|script)", "sql_injection"),
            (r"(?i)(\-\-|\#|\/\*|\*\/)", "sql_comment"),

            # XSS patterns
            (r"(?i)(<script|javascript:|vbscript:|onload=|onerror=)", "xss"),

            # Directory traversal
            (r"(\.\.\/|\.\.\\|%2e%2e%2f|%2e%2e%5c)", "directory_traversal"),

            # Command injection
            (r"(?i)(;|\||`|\$\(|&&)", "command_injection"),

            # File inclusion
            (r"(?i)(file://|php://|data://)", "file_inclusion"),

            # LDAP injection
            (r"(\*\)|\)\(|\(\|)", "ldap_injection"),
        ]

        self.suspicious_headers = [
            "x-forwarded-for",
            "x-real-ip",
            "x-originating-ip",
            "x-cluster-client-ip"
        ]

        self.rate_limit_patterns = [
            (r"admin", 10),      # Admin endpoints: 10 req/min
            (r"auth", 20),       # Auth endpoints: 20 req/min
            (r"api", 100),       # API endpoints: 100 req/min
        ]

    async def analyze_request(self, request: Request) -> List[SecurityThreat]:
        """Analyze request for security threats"""
        threats = []

        try:
            # Analyze URL and query parameters
            url_threats = await self._analyze_url(request)
            threats.extend(url_threats)

            # Analyze headers
            header_threats = await self._analyze_headers(request)
            threats.extend(header_threats)

            # Analyze body if present
            if request.method in ["POST", "PUT", "PATCH"]:
                body_threats = await self._analyze_body(request)
                threats.extend(body_threats)

            # Analyze request patterns
            pattern_threats = await self._analyze_patterns(request)
            threats.extend(pattern_threats)

            return threats

        except Exception as e:
            logger.error(f"Request analysis failed: {e}")
            return []

    async def _analyze_url(self, request: Request) -> List[SecurityThreat]:
        """Analyze URL for suspicious patterns"""
        threats = []

        try:
            url_path = str(request.url.path)
            query_string = str(request.url.query)
            full_url = f"{url_path}?{query_string}" if query_string else url_path

            for pattern, threat_type in self.suspicious_patterns:
                if re.search(pattern, full_url):
                    threats.append(SecurityThreat(
                        threat_type=threat_type,
                        threat_level=ThreatLevel.HIGH,
                        confidence=0.8,
                        description=f"Suspicious {threat_type} pattern detected in URL",
                        indicators=[pattern],
                        recommended_action=SecurityAction.BLOCK,
                        metadata={"url": url_path, "query": query_string}
                    ))

            # Check for excessively long URLs
            if len(full_url) > 2048:
                threats.append(SecurityThreat(
                    threat_type="long_url",
                    threat_level=ThreatLevel.MEDIUM,
                    confidence=0.6,
                    description="Unusually long URL detected",
                    indicators=["url_length"],
                    recommended_action=SecurityAction.RATE_LIMIT,
                    metadata={"url_length": len(full_url)}
                ))

            return threats

        except Exception as e:
            logger.error(f"URL analysis failed: {e}")
            return []

    async def _analyze_headers(self, request: Request) -> List[SecurityThreat]:
        """Analyze request headers for threats"""
        threats = []

        try:
            # Check for suspicious user agents
            user_agent = request.headers.get("user-agent", "").lower()
            suspicious_ua_patterns = [
                r"(?i)(sqlmap|nikto|nmap|masscan|zap|burp)",
                r"(?i)(python-requests|curl|wget)[\d\.]",
                r"(?i)(bot|crawler|spider|scraper)"
            ]

            for pattern in suspicious_ua_patterns:
                if re.search(pattern, user_agent):
                    threats.append(SecurityThreat(
                        threat_type="suspicious_user_agent",
                        threat_level=ThreatLevel.MEDIUM,
                        confidence=0.7,
                        description="Suspicious user agent detected",
                        indicators=[pattern],
                        recommended_action=SecurityAction.CHALLENGE,
                        metadata={"user_agent": user_agent}
                    ))

            # Check for header injection attempts
            for header_name, header_value in request.headers.items():
                if any(char in header_value for char in ['\r', '\n', '\0']):
                    threats.append(SecurityThreat(
                        threat_type="header_injection",
                        threat_level=ThreatLevel.HIGH,
                        confidence=0.9,
                        description="Header injection attempt detected",
                        indicators=["header_injection"],
                        recommended_action=SecurityAction.BLOCK,
                        metadata={"header": header_name}
                    ))

            # Check for missing security headers (on responses)
            expected_headers = [
                "x-content-type-options",
                "x-frame-options",
                "x-xss-protection",
                "strict-transport-security"
            ]

            return threats

        except Exception as e:
            logger.error(f"Header analysis failed: {e}")
            return []

    async def _analyze_body(self, request: Request) -> List[SecurityThreat]:
        """Analyze request body for threats"""
        threats = []

        try:
            # Check content type
            content_type = request.headers.get("content-type", "")

            # Only analyze text-based content types
            if not any(ct in content_type for ct in ["json", "xml", "text", "form"]):
                return threats

            # For this example, we'll create a simple check
            # In production, you'd want to parse and analyze the actual body
            content_length = request.headers.get("content-length", "0")

            try:
                body_size = int(content_length)

                # Check for excessively large payloads
                if body_size > 10 * 1024 * 1024:  # 10MB
                    threats.append(SecurityThreat(
                        threat_type="large_payload",
                        threat_level=ThreatLevel.MEDIUM,
                        confidence=0.6,
                        description="Unusually large request payload",
                        indicators=["payload_size"],
                        recommended_action=SecurityAction.RATE_LIMIT,
                        metadata={"payload_size": body_size}
                    ))

            except ValueError:
                pass

            return threats

        except Exception as e:
            logger.error(f"Body analysis failed: {e}")
            return []

    async def _analyze_patterns(self, request: Request) -> List[SecurityThreat]:
        """Analyze request patterns and behavior"""
        threats = []

        try:
            # Check for rapid-fire requests (would need Redis for tracking)
            # Check for unusual request patterns
            # Check for automated tool signatures

            # For now, basic pattern matching
            path = request.url.path

            # Check for admin/sensitive endpoint access
            sensitive_patterns = [
                r"(?i)/(admin|config|debug|test|dev)",
                r"(?i)/\.(env|git|svn)",
                r"(?i)/(phpinfo|server-info|status)"
            ]

            for pattern in sensitive_patterns:
                if re.search(pattern, path):
                    threats.append(SecurityThreat(
                        threat_type="sensitive_endpoint_access",
                        threat_level=ThreatLevel.MEDIUM,
                        confidence=0.7,
                        description="Access to sensitive endpoint detected",
                        indicators=[pattern],
                        recommended_action=SecurityAction.CHALLENGE,
                        metadata={"endpoint": path}
                    ))

            return threats

        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return []

class SecurityResponseGenerator:
    """Generates appropriate security responses based on threat assessment"""

    def __init__(self):
        self.response_templates = {
            SecurityAction.BLOCK: {
                "status_code": 403,
                "message": "Request blocked by security policy",
                "details": "Your request has been identified as potentially malicious"
            },
            SecurityAction.RATE_LIMIT: {
                "status_code": 429,
                "message": "Rate limit exceeded",
                "details": "Too many requests. Please try again later"
            },
            SecurityAction.CHALLENGE: {
                "status_code": 403,
                "message": "Additional verification required",
                "details": "Your request requires additional verification"
            },
            SecurityAction.QUARANTINE: {
                "status_code": 403,
                "message": "Access temporarily restricted",
                "details": "Your access has been temporarily restricted"
            }
        }

    async def generate_security_response(self, threats: List[SecurityThreat],
                                       security_context: SecurityContext) -> Optional[Response]:
        """Generate appropriate security response based on threats"""

        if not threats:
            return None

        # Determine highest threat level and recommended action
        max_threat_level = max(threat.threat_level for threat in threats)

        # Get the most restrictive action
        actions = [threat.recommended_action for threat in threats]
        action_priority = {
            SecurityAction.ALLOW: 0,
            SecurityAction.RATE_LIMIT: 1,
            SecurityAction.CHALLENGE: 2,
            SecurityAction.QUARANTINE: 3,
            SecurityAction.BLOCK: 4
        }

        recommended_action = max(actions, key=lambda a: action_priority[a])

        # Apply threat score modifier
        if security_context.threat_score > 0.8:
            if recommended_action == SecurityAction.CHALLENGE:
                recommended_action = SecurityAction.BLOCK
            elif recommended_action == SecurityAction.RATE_LIMIT:
                recommended_action = SecurityAction.CHALLENGE

        # Generate response based on action
        if recommended_action == SecurityAction.ALLOW:
            return None

        response_template = self.response_templates[recommended_action]

        response_data = {
            "error": response_template["message"],
            "details": response_template["details"],
            "threat_id": hashlib.sha256(
                f"{security_context.client_ip}{time.time()}".encode()
            ).hexdigest()[:16],
            "timestamp": datetime.utcnow().isoformat()
        }

        # Add security headers
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'none'"
        }

        return JSONResponse(
            content=response_data,
            status_code=response_template["status_code"],
            headers=headers
        )

class ProductionSecurityMiddleware(BaseHTTPMiddleware):
    """Enterprise-grade production security middleware"""

    def __init__(self, app, redis_url: Optional[str] = None,
                 secret_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(app)

        self.config = config or {}
        self.secret_key = secret_key or secrets.token_urlsafe(32)

        # Initialize components
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")

        self.ip_reputation = IPReputationService(self.redis_client)
        self.request_analyzer = RequestAnalyzer()
        self.response_generator = SecurityResponseGenerator()

        # Security configuration
        self.enable_ip_filtering = self.config.get("enable_ip_filtering", True)
        self.enable_request_analysis = self.config.get("enable_request_analysis", True)
        self.enable_rate_limiting = self.config.get("enable_rate_limiting", True)
        self.enable_geographic_filtering = self.config.get("enable_geographic_filtering", False)

        # Rate limiting configuration
        self.rate_limit_window = self.config.get("rate_limit_window", 60)  # seconds
        self.default_rate_limit = self.config.get("default_rate_limit", 100)  # requests per window

        # Security thresholds
        self.reputation_threshold = self.config.get("reputation_threshold", 0.3)
        self.threat_score_threshold = self.config.get("threat_score_threshold", 0.7)

        # Whitelists and blacklists
        self.ip_whitelist = set(self.config.get("ip_whitelist", []))
        self.ip_blacklist = set(self.config.get("ip_blacklist", []))
        self.path_whitelist = self.config.get("path_whitelist", ["/health", "/metrics"])

        logger.info("Production Security Middleware initialized")

    async def dispatch(self, request: Request, call_next):
        """Main middleware dispatch method"""
        start_time = time.time()

        try:
            # Create security context
            security_context = await self._create_security_context(request)

            # Check if path is whitelisted
            if any(request.url.path.startswith(path) for path in self.path_whitelist):
                response = await call_next(request)
                return await self._add_security_headers(response, security_context)

            # IP filtering
            if self.enable_ip_filtering:
                ip_response = await self._check_ip_security(request, security_context)
                if ip_response:
                    return ip_response

            # Rate limiting
            if self.enable_rate_limiting:
                rate_limit_response = await self._check_rate_limits(request, security_context)
                if rate_limit_response:
                    return rate_limit_response

            # Request analysis
            if self.enable_request_analysis:
                analysis_response = await self._analyze_request_security(request, security_context)
                if analysis_response:
                    return analysis_response

            # Process request
            response = await call_next(request)

            # Add security headers
            response = await self._add_security_headers(response, security_context)

            # Log security event
            await self._log_security_event(request, security_context, response, start_time)

            return response

        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            # In case of middleware failure, allow request but log the error
            try:
                response = await call_next(request)
                return response
            except Exception as inner_e:
                logger.error(f"Request processing failed after middleware error: {inner_e}")
                return JSONResponse(
                    content={"error": "Internal server error"},
                    status_code=500
                )

    async def _create_security_context(self, request: Request) -> SecurityContext:
        """Create security context for the request"""

        # Get client IP (handle proxies)
        client_ip = self._get_client_ip(request)

        # Get user agent
        user_agent = request.headers.get("user-agent", "")

        # Create request fingerprint
        fingerprint_data = f"{client_ip}:{user_agent}:{request.method}:{request.url.path}"
        request_fingerprint = hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]

        # Get IP reputation
        reputation_score = await self.ip_reputation.get_ip_reputation(client_ip)

        # Calculate threat score
        threat_score = await self._calculate_threat_score(request, reputation_score)

        # Create security headers
        security_headers = self._generate_security_headers(request)

        return SecurityContext(
            client_ip=client_ip,
            user_agent=user_agent,
            request_fingerprint=request_fingerprint,
            threat_score=threat_score,
            reputation_score=reputation_score,
            geographic_info=None,  # Would integrate with geo service
            request_patterns={},
            security_headers=security_headers
        )

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address, handling proxies correctly"""

        # Check for forwarded headers (in order of preference)
        forwarded_headers = [
            "cf-connecting-ip",    # Cloudflare
            "x-real-ip",          # Nginx
            "x-forwarded-for",    # Standard proxy header
            "x-client-ip",        # Alternative
            "x-cluster-client-ip" # Kubernetes
        ]

        for header in forwarded_headers:
            value = request.headers.get(header)
            if value:
                # Take first IP if comma-separated
                ip = value.split(",")[0].strip()
                try:
                    # Validate IP address
                    ipaddress.ip_address(ip)
                    return ip
                except ValueError:
                    continue

        # Fallback to direct connection
        return request.client.host if request.client else "unknown"

    async def _calculate_threat_score(self, request: Request, reputation_score: float) -> float:
        """Calculate overall threat score for the request"""

        threat_score = 1.0 - reputation_score  # Invert reputation score

        # Adjust based on user agent
        user_agent = request.headers.get("user-agent", "").lower()
        if any(pattern in user_agent for pattern in ["bot", "crawler", "scanner"]):
            threat_score += 0.2

        # Adjust based on request method
        if request.method in ["PUT", "DELETE", "PATCH"]:
            threat_score += 0.1

        # Adjust based on path
        path = request.url.path.lower()
        if any(sensitive in path for sensitive in ["admin", "config", "debug"]):
            threat_score += 0.3

        # Adjust based on query parameters
        if len(request.query_params) > 10:
            threat_score += 0.1

        return min(1.0, threat_score)

    def _generate_security_headers(self, request: Request) -> Dict[str, str]:
        """Generate security headers for the response"""

        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            "X-Permitted-Cross-Domain-Policies": "none"
        }

    async def _check_ip_security(self, request: Request, security_context: SecurityContext) -> Optional[Response]:
        """Check IP-based security policies"""

        client_ip = security_context.client_ip

        # Check blacklist first
        if client_ip in self.ip_blacklist:
            return await self.response_generator.generate_security_response(
                [SecurityThreat(
                    threat_type="blacklisted_ip",
                    threat_level=ThreatLevel.CRITICAL,
                    confidence=1.0,
                    description="IP address is blacklisted",
                    indicators=["ip_blacklist"],
                    recommended_action=SecurityAction.BLOCK,
                    metadata={"ip": client_ip}
                )],
                security_context
            )

        # Check whitelist (skip other checks if whitelisted)
        if client_ip in self.ip_whitelist:
            return None

        # Check reputation
        if security_context.reputation_score < self.reputation_threshold:
            return await self.response_generator.generate_security_response(
                [SecurityThreat(
                    threat_type="low_reputation_ip",
                    threat_level=ThreatLevel.HIGH,
                    confidence=0.8,
                    description="IP address has low reputation score",
                    indicators=["ip_reputation"],
                    recommended_action=SecurityAction.CHALLENGE,
                    metadata={"ip": client_ip, "reputation": security_context.reputation_score}
                )],
                security_context
            )

        return None

    async def _check_rate_limits(self, request: Request, security_context: SecurityContext) -> Optional[Response]:
        """Check rate limiting policies"""

        if not self.redis_client:
            return None

        try:
            client_ip = security_context.client_ip
            path = request.url.path

            # Determine rate limit based on path
            rate_limit = self.default_rate_limit
            for pattern, limit in self.request_analyzer.rate_limit_patterns:
                if re.search(pattern, path):
                    rate_limit = limit
                    break

            # Create rate limit key
            rate_key = f"rate_limit:{client_ip}:{path}"

            # Check current count
            current_count = await self.redis_client.get(rate_key)
            if current_count is None:
                current_count = 0
            else:
                current_count = int(current_count)

            # Check if limit exceeded
            if current_count >= rate_limit:
                return await self.response_generator.generate_security_response(
                    [SecurityThreat(
                        threat_type="rate_limit_exceeded",
                        threat_level=ThreatLevel.MEDIUM,
                        confidence=1.0,
                        description="Rate limit exceeded",
                        indicators=["rate_limit"],
                        recommended_action=SecurityAction.RATE_LIMIT,
                        metadata={"ip": client_ip, "count": current_count, "limit": rate_limit}
                    )],
                    security_context
                )

            # Increment counter
            pipe = self.redis_client.pipeline()
            pipe.incr(rate_key)
            pipe.expire(rate_key, self.rate_limit_window)
            await pipe.execute()

            return None

        except Exception as e:
            logger.error(f"Rate limiting check failed: {e}")
            return None

    async def _analyze_request_security(self, request: Request, security_context: SecurityContext) -> Optional[Response]:
        """Analyze request for security threats"""

        try:
            # Perform threat analysis
            threats = await self.request_analyzer.analyze_request(request)

            # Filter threats based on threat score
            if security_context.threat_score > self.threat_score_threshold:
                # More sensitive filtering for high-threat clients
                threats = [t for t in threats if t.threat_level != ThreatLevel.LOW]

            # Generate response if threats found
            if threats:
                return await self.response_generator.generate_security_response(threats, security_context)

            return None

        except Exception as e:
            logger.error(f"Request security analysis failed: {e}")
            return None

    async def _add_security_headers(self, response: Response, security_context: SecurityContext) -> Response:
        """Add security headers to response"""

        try:
            for header, value in security_context.security_headers.items():
                response.headers[header] = value

            # Add additional security headers
            response.headers["X-Request-ID"] = security_context.request_fingerprint
            response.headers["X-Threat-Score"] = str(round(security_context.threat_score, 2))

            return response

        except Exception as e:
            logger.error(f"Adding security headers failed: {e}")
            return response

    async def _log_security_event(self, request: Request, security_context: SecurityContext,
                                 response: Response, start_time: float):
        """Log security events for monitoring and analysis"""

        try:
            processing_time = time.time() - start_time

            security_event = {
                "timestamp": datetime.utcnow().isoformat(),
                "client_ip": security_context.client_ip,
                "user_agent": security_context.user_agent,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "threat_score": security_context.threat_score,
                "reputation_score": security_context.reputation_score,
                "processing_time": processing_time,
                "request_fingerprint": security_context.request_fingerprint
            }

            # Log high-threat requests
            if security_context.threat_score > 0.7 or response.status_code in [403, 429]:
                logger.warning(f"High-threat request: {json.dumps(security_event)}")

            # Store in Redis for analysis (if available)
            if self.redis_client:
                try:
                    event_key = f"security_event:{security_context.request_fingerprint}"
                    await self.redis_client.setex(event_key, 86400, json.dumps(security_event))  # 24 hours
                except Exception as e:
                    logger.debug(f"Failed to store security event: {e}")

        except Exception as e:
            logger.error(f"Security event logging failed: {e}")

class SecurityMetricsCollector:
    """Collects and aggregates security metrics"""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.metrics = {
            "total_requests": 0,
            "blocked_requests": 0,
            "rate_limited_requests": 0,
            "challenged_requests": 0,
            "threat_detections": 0,
            "low_reputation_ips": 0
        }

    async def record_security_event(self, event_type: str, metadata: Dict[str, Any] = None):
        """Record a security event for metrics"""

        try:
            self.metrics[event_type] = self.metrics.get(event_type, 0) + 1

            if self.redis_client:
                # Store metrics in Redis
                current_hour = datetime.utcnow().strftime("%Y%m%d_%H")
                metric_key = f"security_metrics:{current_hour}:{event_type}"
                await self.redis_client.incr(metric_key)
                await self.redis_client.expire(metric_key, 86400 * 7)  # Keep for 7 days

        except Exception as e:
            logger.error(f"Failed to record security event: {e}")

    async def get_security_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get security metrics for the specified time period"""

        try:
            if not self.redis_client:
                return self.metrics

            # Aggregate metrics from Redis
            end_time = datetime.utcnow()
            metrics = {}

            for hour_offset in range(hours):
                hour_time = end_time - timedelta(hours=hour_offset)
                hour_key = hour_time.strftime("%Y%m%d_%H")

                # Get all metrics for this hour
                pattern = f"security_metrics:{hour_key}:*"
                keys = await self.redis_client.keys(pattern)

                for key in keys:
                    metric_type = key.decode().split(":")[-1]
                    value = await self.redis_client.get(key)
                    if value:
                        metrics[metric_type] = metrics.get(metric_type, 0) + int(value)

            # Add current in-memory metrics
            for metric_type, value in self.metrics.items():
                metrics[metric_type] = metrics.get(metric_type, 0) + value

            # Calculate derived metrics
            total_requests = metrics.get("total_requests", 0)
            if total_requests > 0:
                metrics["block_rate"] = metrics.get("blocked_requests", 0) / total_requests
                metrics["threat_detection_rate"] = metrics.get("threat_detections", 0) / total_requests
            else:
                metrics["block_rate"] = 0.0
                metrics["threat_detection_rate"] = 0.0

            return metrics

        except Exception as e:
            logger.error(f"Failed to get security metrics: {e}")
            return self.metrics

# Global metrics collector
_security_metrics: Optional[SecurityMetricsCollector] = None

def get_security_metrics_collector(redis_client: Optional[redis.Redis] = None) -> SecurityMetricsCollector:
    """Get global security metrics collector"""
    global _security_metrics

    if _security_metrics is None:
        _security_metrics = SecurityMetricsCollector(redis_client)

    return _security_metrics

# Example usage and configuration
def create_production_security_middleware(app, config: Dict[str, Any]):
    """Create and configure production security middleware"""

    middleware_config = {
        "enable_ip_filtering": config.get("security.ip_filtering", True),
        "enable_request_analysis": config.get("security.request_analysis", True),
        "enable_rate_limiting": config.get("security.rate_limiting", True),
        "reputation_threshold": config.get("security.reputation_threshold", 0.3),
        "threat_score_threshold": config.get("security.threat_score_threshold", 0.7),
        "rate_limit_window": config.get("security.rate_limit_window", 60),
        "default_rate_limit": config.get("security.default_rate_limit", 100),
        "ip_whitelist": config.get("security.ip_whitelist", []),
        "ip_blacklist": config.get("security.ip_blacklist", []),
        "path_whitelist": config.get("security.path_whitelist", ["/health", "/metrics"])
    }

    redis_url = config.get("redis.url")
    secret_key = config.get("security.secret_key")

    return ProductionSecurityMiddleware(
        app=app,
        redis_url=redis_url,
        secret_key=secret_key,
        config=middleware_config
    )
