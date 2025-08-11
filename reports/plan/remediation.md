# 🔧 XORB Platform Security Remediation Plan

- **Plan ID**: XORB_REMEDIATION_2025_01_11
- **Created**: January 11, 2025
- **Owner**: Principal Auditor & Lead Engineer
- **Target Completion**: February 29, 2025

## 📋 Executive Summary

This remediation plan addresses **47 security findings** across the XORB platform, prioritized by risk level and business impact. The plan follows a phased approach to ensure continuous operation while systematically hardening security posture.

- **Total Effort Estimate**: 16 person-weeks
- **Critical Issues**: 1 (24-hour SLA)
- **High Priority**: 8 (1-2 week SLA)
- **Medium Priority**: 23 (2-6 week SLA)
- **Low Priority**: 15 (6-12 week SLA)

## 🎯 Phase 1: Critical Security Fixes (Week 1)

### 🚨 P0: JWT Secret Management Vulnerability
- **Finding**: XORB-2025-001
- **Target**: 24 hours
- **Owner**: Security Team Lead

#### Current State
```python
# VULNERABLE - src/api/app/core/config.py:42
jwt_secret_key: str = Field(env="JWT_SECRET")
```

#### Secure Implementation
```python
import secrets
import hashlib
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

class SecureJWTManager:
    def __init__(self):
        self.vault_client = self._init_vault_client()
        self.secret_rotation_interval = 86400  # 24 hours
        self._current_secret = None
        self._next_secret = None

    def get_signing_key(self) -> str:
        """Get current JWT signing key with automatic rotation"""
        if self._needs_rotation():
            self._rotate_secret()
        return self._current_secret

    def _init_vault_client(self):
        """Initialize HashiCorp Vault client"""
        import hvac
        client = hvac.Client(url=os.getenv('VAULT_URL', 'http://localhost:8200'))

        # Authenticate with AppRole
        role_id = os.getenv('VAULT_ROLE_ID')
        secret_id = os.getenv('VAULT_SECRET_ID')

        if role_id and secret_id:
            client.auth.approle.login(role_id=role_id, secret_id=secret_id)

        return client

    def _generate_secret(self) -> str:
        """Generate cryptographically secure JWT secret"""
        # Generate 512-bit secret (64 bytes)
        secret = secrets.token_urlsafe(64)

        # Validate entropy
        entropy = self._calculate_entropy(secret)
        if entropy < 5.0:
            return self._generate_secret()  # Regenerate if low entropy

        return secret

    def _calculate_entropy(self, data: str) -> float:
        """Calculate Shannon entropy of string"""
        import math
        from collections import Counter

        counts = Counter(data)
        length = len(data)
        entropy = -sum((count/length) * math.log2(count/length)
                      for count in counts.values())
        return entropy

    def _rotate_secret(self):
        """Rotate JWT signing secret"""
        try:
            new_secret = self._generate_secret()

            # Store in Vault
            self.vault_client.secrets.kv.v2.create_or_update_secret(
                path='jwt-signing',
                secret={'key': new_secret, 'rotated_at': time.time()}
            )

            self._current_secret = new_secret
            logger.info("JWT secret rotated successfully")

        except Exception as e:
            logger.error(f"JWT secret rotation failed: {e}")
            raise SecurityError("Failed to rotate JWT secret")

# Updated configuration
class SecureAppSettings(BaseSettings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.jwt_manager = SecureJWTManager()

    @property
    def jwt_secret_key(self) -> str:
        return self.jwt_manager.get_signing_key()
```

#### Deployment Steps
1. **Deploy Vault integration** with AppRole authentication
2. **Generate initial secure secret** with proper entropy
3. **Update configuration loading** to use SecureJWTManager
4. **Test authentication flows** with new secret management
5. **Monitor rotation logs** for successful operations

#### Validation
```bash
# Verify secret strength
python -c "
from src.api.app.core.config import get_settings
settings = get_settings()
secret = settings.jwt_secret_key
print(f'Secret length: {len(secret)}')
print(f'Entropy: {calculate_entropy(secret):.2f}')
assert len(secret) >= 64, 'Secret too short'
assert calculate_entropy(secret) >= 5.0, 'Entropy too low'
"

# Test JWT functionality
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=testuser&password=testpass"
```

### 🔴 P1: Hardcoded Credentials Removal
- **Finding**: XORB-2025-002
- **Target**: 72 hours
- **Owner**: Development Team Lead

#### Remediation Actions
1. **Audit all hardcoded credentials** in codebase
2. **Remove test credentials** from version control history
3. **Implement secure test credential generation**
4. **Add pre-commit hooks** to prevent future credential commits

#### Implementation
```bash
# Remove credentials from git history
git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch tests/unit/test_config_security.py' \
- -prune-empty --tag-name-filter cat -- --all

# Secure test credential generation
import secrets
import pytest

@pytest.fixture
def secure_test_credentials():
    return {
        'username': f'test_user_{secrets.token_hex(8)}',
        'password': secrets.token_urlsafe(32),
        'jwt_secret': secrets.token_urlsafe(64)
    }

# Pre-commit hook for secret detection
# .pre-commit-config.yaml
repos:
- repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
    -   id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
```

### 🟠 P1: CORS Security Hardening
- **Finding**: XORB-2025-003
- **Target**: 1 week
- **Owner**: Frontend Team Lead

#### Secure CORS Implementation
```python
class SecureCORSConfig:
    def __init__(self, environment: str):
        self.environment = environment
        self.allowed_origins = self._get_secure_origins()

    def _get_secure_origins(self) -> List[str]:
        """Get validated CORS origins for environment"""
        origins_env = os.getenv('CORS_ALLOW_ORIGINS', '')

        if not origins_env:
            return self._get_default_origins()

        origins = [origin.strip() for origin in origins_env.split(',')]
        validated_origins = []

        for origin in origins:
            if self._validate_origin(origin):
                validated_origins.append(origin)
            else:
                logger.warning(f"Invalid CORS origin rejected: {origin}")

        return validated_origins or self._get_default_origins()

    def _validate_origin(self, origin: str) -> bool:
        """Validate CORS origin format and security"""
        # Never allow wildcard in production
        if origin == '*' and self.environment == 'production':
            return False

        # Allow localhost for development
        if origin.startswith('http://localhost:') and self.environment == 'development':
            return True

        # Require HTTPS for all other origins
        if not origin.startswith('https://'):
            return False

        # Validate domain format
        try:
            from urllib.parse import urlparse
            parsed = urlparse(origin)
            return bool(parsed.netloc)
        except:
            return False

    def _get_default_origins(self) -> List[str]:
        """Get secure default origins by environment"""
        defaults = {
            'production': ['https://app.xorb.enterprise'],
            'staging': ['https://staging.xorb.enterprise'],
            'development': ['http://localhost:3000', 'http://localhost:8080']
        }
        return defaults.get(self.environment, ['https://app.xorb.enterprise'])
```

## 🎯 Phase 2: High Priority Security (Weeks 2-3)

### 🔒 P2: Container Security Hardening
- **Finding**: XORB-2025-004
- **Target**: 2 weeks
- **Owner**: DevOps Team Lead

#### Secure Container Configuration
```yaml
# docker-compose.security.yml
version: '3.8'

services:
  xorb-api:
    security_opt:
      - no-new-privileges:true
      - apparmor:docker-default
      - seccomp:seccomp-profiles/default.json
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE  # Only necessary capability
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=100m
      - /var/cache:noexec,nosuid,size=50m
      - /var/log:noexec,nosuid,size=100m
    user: "1001:1001"  # Non-root user
    ulimits:
      nproc: 200
      nofile: 4096
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.5'
          pids: 200
        reservations:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8000/api/v1/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped
```

#### Security Scanning Integration
```dockerfile
# Dockerfile.secure
FROM python:3.11-slim as security-scanner

# Install security scanning tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    trivy \
    dumb-init \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r xorb && useradd -r -g xorb -u 1001 xorb

# Copy application
COPY --chown=xorb:xorb . /app
WORKDIR /app

# Install dependencies with security check
RUN pip install --no-cache-dir safety bandit && \
    safety check -r requirements.lock && \
    bandit -r src/ -f json -o security-report.json

# Switch to non-root user
USER xorb

# Use dumb-init for proper signal handling
ENTRYPOINT ["dumb-init", "--"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 🛡️ P2: Input Validation Framework
- **Finding**: XORB-2025-005
- **Target**: 2 weeks
- **Owner**: Backend Team Lead

#### Comprehensive Input Validation
```python
from pydantic import BaseModel, validator, Field
import re
from typing import List, Optional
import bleach

class SecureInputValidation:
    """Centralized input validation with security focus"""

    # Regex patterns for common inputs
    USERNAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{3,50}$')
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')

    # Dangerous patterns to reject
    INJECTION_PATTERNS = [
        re.compile(r'[\'";<>&|`$(){}[\]]'),  # SQL/Command injection chars
        re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE),  # XSS
        re.compile(r'javascript:', re.IGNORECASE),  # JavaScript injection
        re.compile(r'on\w+\s*=', re.IGNORECASE),  # Event handlers
    ]

    @classmethod
    def validate_username(cls, username: str) -> str:
        """Validate username with security checks"""
        if not username:
            raise ValueError("Username is required")

        # Check format
        if not cls.USERNAME_PATTERN.match(username):
            raise ValueError("Username contains invalid characters")

        # Check for injection patterns
        for pattern in cls.INJECTION_PATTERNS:
            if pattern.search(username):
                raise ValueError("Username contains potentially dangerous characters")

        return username.strip().lower()

    @classmethod
    def sanitize_html(cls, content: str) -> str:
        """Sanitize HTML content"""
        if not content:
            return ""

        # Allow only safe HTML tags
        allowed_tags = ['p', 'br', 'strong', 'em', 'ul', 'ol', 'li']
        allowed_attributes = {}

        return bleach.clean(
            content,
            tags=allowed_tags,
            attributes=allowed_attributes,
            strip=True
        )

    @classmethod
    def validate_file_path(cls, path: str) -> str:
        """Validate file path for directory traversal"""
        if not path:
            raise ValueError("File path is required")

        # Normalize path
        normalized = os.path.normpath(path)

        # Check for directory traversal
        if '..' in normalized or normalized.startswith('/'):
            raise ValueError("Invalid file path detected")

        return normalized

class SecurePTaaSRequest(BaseModel):
    """Secure PTaaS request validation"""

    target_host: str = Field(..., min_length=1, max_length=255)
    ports: List[int] = Field(default=[], max_items=100)
    scan_type: str = Field(..., regex=r'^(quick|comprehensive|stealth|web-focused)$')

    @validator('target_host')
    def validate_target_host(cls, v):
        """Validate target host with security checks"""
        # Basic format validation
        if not re.match(r'^[a-zA-Z0-9.-]+$', v):
            raise ValueError("Invalid host format")

        # Prevent internal network scanning
        forbidden_hosts = [
            'localhost', '127.0.0.1', '0.0.0.0',
            '10.', '172.16.', '192.168.',  # Private networks
            'metadata.google.internal',   # Cloud metadata
        ]

        for forbidden in forbidden_hosts:
            if v.startswith(forbidden):
                raise ValueError(f"Scanning {forbidden} networks is not allowed")

        return v

    @validator('ports')
    def validate_ports(cls, v):
        """Validate port numbers"""
        for port in v:
            if not (1 <= port <= 65535):
                raise ValueError(f"Invalid port number: {port}")
        return list(set(v))  # Remove duplicates
```

## 🎯 Phase 3: Medium Priority Fixes (Weeks 4-6)

### 📝 Logging Security Implementation
- **Finding**: XORB-2025-007
- **Target**: 3 weeks
- **Owner**: Backend Team Lead

#### Secure Logging Framework
```python
import re
from typing import Dict, Any

class SecureLogger:
    """Security-focused logging with PII masking"""

    # PII patterns to mask
    PII_PATTERNS = {
        'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
        'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
        'ip_address': re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
    }

    # Sensitive field names
    SENSITIVE_FIELDS = {
        'password', 'token', 'secret', 'key', 'auth', 'credential',
        'ssn', 'social_security', 'credit_card', 'card_number',
        'pin', 'cvv', 'security_code'
    }

    def mask_sensitive_data(self, data: Any) -> Any:
        """Recursively mask sensitive data in logs"""
        if isinstance(data, dict):
            return {k: self._mask_value(k, v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.mask_sensitive_data(item) for item in data]
        elif isinstance(data, str):
            return self._mask_string_patterns(data)
        else:
            return data

    def _mask_value(self, key: str, value: Any) -> Any:
        """Mask value based on key name"""
        if isinstance(key, str) and any(field in key.lower() for field in self.SENSITIVE_FIELDS):
            if isinstance(value, str) and len(value) > 0:
                return f"***MASKED:{len(value)}***"
            else:
                return "***MASKED***"

        return self.mask_sensitive_data(value)

    def _mask_string_patterns(self, text: str) -> str:
        """Mask PII patterns in text"""
        for pattern_name, pattern in self.PII_PATTERNS.items():
            text = pattern.sub(f'***{pattern_name.upper()}:MASKED***', text)
        return text

# Enhanced logging configuration
import structlog
from pythonjsonlogger import jsonlogger

def configure_secure_logging():
    """Configure secure structured logging"""
    secure_logger = SecureLogger()

    def add_security_processor(logger, name, event_dict):
        """Add security masking to log processor"""
        return secure_logger.mask_sensitive_data(event_dict)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO"),
            add_security_processor,  # Security masking processor
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
```

## 📊 Implementation Timeline

```mermaid
gantt
    title XORB Security Remediation Timeline
    dateFormat  YYYY-MM-DD
    section Critical
    JWT Secret Fix     :crit, jwt, 2025-01-11, 1d
    Credential Cleanup :crit, creds, 2025-01-12, 3d

    section High Priority
    CORS Hardening     :high, cors, 2025-01-15, 7d
    Container Security :high, container, 2025-01-15, 14d
    Input Validation   :high, input, 2025-01-22, 14d

    section Medium Priority
    Config Security    :med, config, 2025-02-05, 7d
    Logging Security   :med, logging, 2025-02-05, 21d
    Rate Limiting      :med, rate, 2025-02-12, 14d

    section Low Priority
    Dependencies       :low, deps, 2025-02-19, 21d
    TLS Hardening      :low, tls, 2025-02-26, 14d
```

## ✅ Success Criteria

### Security Metrics
- **Zero critical vulnerabilities** remaining
- **90% reduction** in high-severity findings
- **100% coverage** of security-sensitive endpoints with validation
- **Automated security scanning** integrated into CI/CD

### Compliance Targets
- **SOC 2 Type II**: 95% control compliance
- **PCI-DSS**: Level 1 merchant readiness
- **GDPR**: Full Article 32 compliance
- **ISO 27001**: 95% control implementation

### Performance Targets
- **<1% performance impact** from security enhancements
- **<100ms additional latency** for validation
- **Zero downtime** during remediation deployment

## 🔄 Monitoring & Validation

### Continuous Security Monitoring
```yaml
# security-monitoring.yml
monitors:
  - name: "JWT Secret Rotation"
    check: "vault_secret_age < 24h"
    alert: "critical"

  - name: "Authentication Anomalies"
    check: "failed_logins > 10/minute"
    alert: "high"

  - name: "Input Validation Failures"
    check: "validation_errors > 100/hour"
    alert: "medium"
```

### Security Testing Suite
```bash
# !/bin/bash
# security-test-suite.sh

echo "🔍 Running security validation suite..."

# 1. Static analysis
bandit -r src/ -f json -o reports/bandit.json

# 2. Dependency scanning
safety check -r requirements.lock

# 3. Secret detection
detect-secrets scan --baseline .secrets.baseline

# 4. Container security
trivy image xorb-platform:latest

# 5. API security testing
zap-baseline.py -t http://localhost:8000

echo "✅ Security validation complete"
```

- --
- **Plan Status**: APPROVED
- **Next Review**: January 18, 2025
- **Emergency Contact**: security@xorb.enterprise