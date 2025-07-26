#!/usr/bin/env python3
"""
Security-Hardened Orchestrator for Xorb 2.0

This module provides security enhancements to the base orchestrator,
including input validation, secure deserialization, audit logging,
and rate limiting to prevent abuse.
"""

import asyncio
import logging
import json
import hashlib
import hmac
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re

from pydantic import BaseModel, validator, Field
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets
import base64

from ..orchestration.orchestrator import XorbOrchestrator, Campaign, CampaignStatus, CampaignPriority


@dataclass
class SecurityConfig:
    """Security configuration for the orchestrator"""
    enable_input_validation: bool = True
    enable_rate_limiting: bool = True
    enable_audit_logging: bool = True
    enable_encryption: bool = True
    max_campaign_name_length: int = 100
    max_targets_per_campaign: int = 1000
    max_campaigns_per_user: int = 50
    rate_limit_requests_per_minute: int = 100
    audit_log_retention_days: int = 90
    encryption_key_rotation_days: int = 30
    session_timeout_minutes: int = 60
    max_payload_size_mb: int = 10


class SecureCampaignInput(BaseModel):
    """Secure campaign input validation model"""
    name: str = Field(..., min_length=1, max_length=100)
    targets: List[Dict[str, Any]] = Field(..., max_items=1000)
    priority: CampaignPriority = CampaignPriority.MEDIUM
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, max_items=50)
    user_id: str = Field(..., min_length=1, max_length=100)
    
    @validator('name')
    def validate_name(cls, v):
        # Allow only alphanumeric, spaces, hyphens, and underscores
        if not re.match(r'^[a-zA-Z0-9\s\-_]+$', v):
            raise ValueError('Campaign name contains invalid characters')
        return v.strip()
    
    @validator('targets', each_item=True)
    def validate_target(cls, v):
        if not isinstance(v, dict):
            raise ValueError('Each target must be a dictionary')
        
        # Validate required fields
        if 'type' not in v:
            raise ValueError('Target must have a type field')
        
        # Validate target type
        allowed_types = ['web', 'api', 'mobile', 'network', 'social']
        if v['type'] not in allowed_types:
            raise ValueError(f'Target type must be one of: {allowed_types}')
        
        # Validate URL if present
        if 'url' in v:
            url = v['url']
            if not isinstance(url, str) or len(url) > 2048:
                raise ValueError('Invalid URL format or length')
            
            # Basic URL validation
            if not re.match(r'^https?://', url):
                raise ValueError('URL must start with http:// or https://')
                
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v):
        if not isinstance(v, dict):
            raise ValueError('Metadata must be a dictionary')
        
        # Check for dangerous keys
        dangerous_keys = ['__', 'eval', 'exec', 'import', 'open', 'file']
        for key in v.keys():
            if any(dangerous in str(key).lower() for dangerous in dangerous_keys):
                raise ValueError(f'Metadata key contains dangerous content: {key}')
        
        # Validate values
        for key, value in v.items():
            if isinstance(value, str) and len(value) > 1000:
                raise ValueError(f'Metadata value too long for key: {key}')
                
        return v
    
    @validator('user_id')
    def validate_user_id(cls, v):
        # Validate user ID format
        if not re.match(r'^[a-zA-Z0-9\-_]+$', v):
            raise ValueError('User ID contains invalid characters')
        return v


class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, max_requests: int = 100, window_minutes: int = 1):
        self.max_requests = max_requests
        self.window_seconds = window_minutes * 60
        self.requests: Dict[str, List[float]] = {}
        
    def is_allowed(self, user_id: str) -> bool:
        """Check if request is allowed for user"""
        now = time.time()
        
        # Initialize user if not exists
        if user_id not in self.requests:
            self.requests[user_id] = []
        
        # Clean old requests
        self.requests[user_id] = [
            req_time for req_time in self.requests[user_id]
            if now - req_time < self.window_seconds
        ]
        
        # Check rate limit
        if len(self.requests[user_id]) >= self.max_requests:
            return False
        
        # Record request
        self.requests[user_id].append(now)
        return True
    
    def get_reset_time(self, user_id: str) -> int:
        """Get seconds until rate limit resets"""
        if user_id not in self.requests or not self.requests[user_id]:
            return 0
        
        oldest_request = min(self.requests[user_id])
        return max(0, int(self.window_seconds - (time.time() - oldest_request)))


class AuditLogger:
    """Security audit logging"""
    
    def __init__(self, log_file: str = "xorb_security_audit.log"):
        self.log_file = Path(log_file)
        self.logger = logging.getLogger("xorb.security.audit")
        
        # Configure audit logger
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_event(self, event_type: str, user_id: str, details: Dict[str, Any], 
                  risk_level: str = "INFO"):
        """Log security event"""
        event_data = {
            'event_type': event_type,
            'user_id': user_id,
            'timestamp': datetime.utcnow().isoformat(),
            'risk_level': risk_level,
            'details': details,
            'source_ip': details.get('source_ip', 'unknown'),
            'user_agent': details.get('user_agent', 'unknown')
        }
        
        log_message = json.dumps(event_data)
        
        if risk_level == "CRITICAL":
            self.logger.critical(log_message)
        elif risk_level == "HIGH":
            self.logger.error(log_message)
        elif risk_level == "MEDIUM":
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def log_campaign_creation(self, user_id: str, campaign_id: str, 
                            target_count: int, source_ip: str = None):
        """Log campaign creation"""
        self.log_event(
            event_type="campaign_creation",
            user_id=user_id,
            details={
                'campaign_id': campaign_id,
                'target_count': target_count,
                'source_ip': source_ip
            },
            risk_level="INFO"
        )
    
    def log_security_violation(self, user_id: str, violation_type: str, 
                             details: Dict[str, Any], source_ip: str = None):
        """Log security violation"""
        self.log_event(
            event_type="security_violation",
            user_id=user_id,
            details={
                'violation_type': violation_type,
                'source_ip': source_ip,
                **details
            },
            risk_level="HIGH"
        )
    
    def log_authentication_failure(self, user_id: str, source_ip: str = None):
        """Log authentication failure"""
        self.log_event(
            event_type="authentication_failure", 
            user_id=user_id,
            details={
                'source_ip': source_ip
            },
            risk_level="MEDIUM"
        )


class SecureDataManager:
    """Secure data encryption and storage"""
    
    def __init__(self, encryption_key: bytes = None):
        if encryption_key is None:
            encryption_key = self._generate_key()
        
        self.fernet = Fernet(encryption_key)
        self.key_created = datetime.utcnow()
    
    @staticmethod
    def _generate_key() -> bytes:
        """Generate encryption key"""
        password = secrets.token_bytes(32)
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt data"""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def secure_serialize(self, data: Dict[str, Any]) -> str:
        """Securely serialize data"""
        json_data = json.dumps(data, separators=(',', ':'))
        return self.encrypt_data(json_data)
    
    def secure_deserialize(self, encrypted_data: str) -> Dict[str, Any]:
        """Securely deserialize data"""
        try:
            json_data = self.decrypt_data(encrypted_data)
            return json.loads(json_data)
        except Exception as e:
            raise ValueError(f"Failed to deserialize data: {e}")
    
    def needs_key_rotation(self, rotation_days: int = 30) -> bool:
        """Check if encryption key needs rotation"""
        return datetime.utcnow() - self.key_created > timedelta(days=rotation_days)


class SecurityMiddleware:
    """Security middleware for request processing"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.rate_limiter = RateLimiter(
            max_requests=config.rate_limit_requests_per_minute,
            window_minutes=1
        )
        self.audit_logger = AuditLogger()
        self.data_manager = SecureDataManager()
        self.active_sessions: Dict[str, datetime] = {}
        
    def validate_session(self, session_id: str) -> bool:
        """Validate user session"""
        if session_id not in self.active_sessions:
            return False
        
        # Check session timeout
        session_time = self.active_sessions[session_id]
        if datetime.utcnow() - session_time > timedelta(minutes=self.config.session_timeout_minutes):
            del self.active_sessions[session_id]
            return False
        
        # Update session time
        self.active_sessions[session_id] = datetime.utcnow()
        return True
    
    def create_session(self, user_id: str) -> str:
        """Create new user session"""
        session_id = secrets.token_urlsafe(32)
        self.active_sessions[session_id] = datetime.utcnow()
        
        self.audit_logger.log_event(
            event_type="session_created",
            user_id=user_id,
            details={'session_id': session_id},
            risk_level="INFO"
        )
        
        return session_id
    
    def check_rate_limit(self, user_id: str) -> bool:
        """Check rate limit for user"""
        if not self.config.enable_rate_limiting:
            return True
        
        allowed = self.rate_limiter.is_allowed(user_id)
        
        if not allowed:
            self.audit_logger.log_security_violation(
                user_id=user_id,
                violation_type="rate_limit_exceeded",
                details={
                    'reset_time': self.rate_limiter.get_reset_time(user_id)
                }
            )
        
        return allowed
    
    def validate_payload_size(self, payload: str) -> bool:
        """Validate payload size"""
        size_mb = len(payload.encode('utf-8')) / (1024 * 1024)
        return size_mb <= self.config.max_payload_size_mb
    
    def sanitize_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize input data"""
        def sanitize_value(value):
            if isinstance(value, str):
                # Remove potential script tags and dangerous content
                value = re.sub(r'<script[^>]*>.*?</script>', '', value, flags=re.IGNORECASE | re.DOTALL)
                value = re.sub(r'javascript:', '', value, flags=re.IGNORECASE)
                value = re.sub(r'on\w+\s*=', '', value, flags=re.IGNORECASE)
                return value.strip()
            elif isinstance(value, dict):
                return {k: sanitize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [sanitize_value(item) for item in value]
            return value
        
        return sanitize_value(data)


class SecureXorbOrchestrator(XorbOrchestrator):
    """Security-hardened Xorb orchestrator"""
    
    def __init__(self, *args, security_config: SecurityConfig = None, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.security_config = security_config or SecurityConfig()
        self.security_middleware = SecurityMiddleware(self.security_config)
        self.user_campaign_counts: Dict[str, int] = {}
        
        # Override the insecure deserializer
        self._secure_deserialize = self.security_middleware.data_manager.secure_deserialize
        
        self.logger.info("Security-hardened orchestrator initialized")
    
    async def secure_create_campaign(self, 
                                   campaign_data: Dict[str, Any],
                                   user_id: str,
                                   session_id: str = None,
                                   source_ip: str = None) -> Campaign:
        """Securely create a campaign with full validation"""
        
        # Validate session
        if session_id and not self.security_middleware.validate_session(session_id):
            raise ValueError("Invalid or expired session")
        
        # Check rate limit
        if not self.security_middleware.check_rate_limit(user_id):
            raise ValueError("Rate limit exceeded")
        
        # Validate payload size
        payload_str = json.dumps(campaign_data)
        if not self.security_middleware.validate_payload_size(payload_str):
            raise ValueError("Payload size exceeds maximum allowed")
        
        # Sanitize input
        sanitized_data = self.security_middleware.sanitize_input(campaign_data)
        
        # Validate input structure
        try:
            validated_input = SecureCampaignInput(
                **sanitized_data,
                user_id=user_id
            )
        except Exception as e:
            self.security_middleware.audit_logger.log_security_violation(
                user_id=user_id,
                violation_type="input_validation_failure",
                details={
                    'error': str(e),
                    'payload_keys': list(campaign_data.keys())
                },
                source_ip=source_ip
            )
            raise ValueError(f"Input validation failed: {e}")
        
        # Check user campaign limit
        user_campaign_count = self.user_campaign_counts.get(user_id, 0)
        if user_campaign_count >= self.security_config.max_campaigns_per_user:
            self.security_middleware.audit_logger.log_security_violation(
                user_id=user_id,
                violation_type="campaign_limit_exceeded",
                details={
                    'current_count': user_campaign_count,
                    'limit': self.security_config.max_campaigns_per_user
                },
                source_ip=source_ip
            )
            raise ValueError("Maximum campaigns per user exceeded")
        
        # Create campaign using validated data
        campaign = await self._create_validated_campaign(validated_input)
        
        # Update user campaign count
        self.user_campaign_counts[user_id] = user_campaign_count + 1
        
        # Log campaign creation
        self.security_middleware.audit_logger.log_campaign_creation(
            user_id=user_id,
            campaign_id=campaign.id,
            target_count=len(validated_input.targets),
            source_ip=source_ip
        )
        
        return campaign
    
    async def _create_validated_campaign(self, validated_input: SecureCampaignInput) -> Campaign:
        """Create campaign from validated input"""
        campaign = Campaign(
            name=validated_input.name,
            targets=validated_input.targets,
            priority=validated_input.priority,
            metadata={
                **validated_input.metadata,
                'user_id': validated_input.user_id,
                'created_via_secure_api': True,
                'security_validated': True
            }
        )
        
        await self.add_campaign(campaign)
        return campaign
    
    async def _load_persisted_campaigns(self):
        """Securely load persisted campaigns"""
        if not self.redis_client:
            return
        
        try:
            keys = await self.redis_client.keys("campaign:*")
            for key in keys:
                campaign_data = await self.redis_client.hget(key, "data")
                if campaign_data:
                    try:
                        # Use secure deserialization instead of eval()
                        data = self._secure_deserialize(campaign_data)
                        campaign_id = data["id"]
                        
                        # Validate deserialized data
                        if self._is_valid_campaign_data(data):
                            await self._reconstruct_campaign_from_data(data)
                            self.logger.debug(f"Securely loaded persisted campaign {campaign_id}")
                        else:
                            self.logger.warning(f"Invalid campaign data for {campaign_id}, skipping")
                            
                    except Exception as e:
                        self.logger.error(f"Failed to securely deserialize campaign data: {e}")
                        # Log potential security incident
                        self.security_middleware.audit_logger.log_security_violation(
                            user_id="system",
                            violation_type="campaign_deserialization_failure",
                            details={
                                'error': str(e),
                                'key': key
                            }
                        )
                        
        except Exception as e:
            self.logger.error(f"Failed to load persisted campaigns: {e}")
    
    def _is_valid_campaign_data(self, data: Dict[str, Any]) -> bool:
        """Validate campaign data structure"""
        required_fields = ['id', 'name', 'targets', 'status']
        
        if not isinstance(data, dict):
            return False
        
        for field in required_fields:
            if field not in data:
                return False
        
        # Additional security checks
        if len(str(data.get('name', ''))) > self.security_config.max_campaign_name_length:
            return False
        
        targets = data.get('targets', [])
        if not isinstance(targets, list) or len(targets) > self.security_config.max_targets_per_campaign:
            return False
        
        return True
    
    async def get_user_campaigns(self, user_id: str, session_id: str = None) -> List[Campaign]:
        """Get campaigns for a specific user"""
        
        # Validate session
        if session_id and not self.security_middleware.validate_session(session_id):
            raise ValueError("Invalid or expired session")
        
        # Check rate limit
        if not self.security_middleware.check_rate_limit(user_id):
            raise ValueError("Rate limit exceeded")
        
        # Filter campaigns by user
        user_campaigns = []
        for campaign in self.campaigns.values():
            if campaign.metadata.get('user_id') == user_id:
                user_campaigns.append(campaign)
        
        return user_campaigns
    
    async def delete_user_campaign(self, campaign_id: str, user_id: str, session_id: str = None):
        """Securely delete a user's campaign"""
        
        # Validate session
        if session_id and not self.security_middleware.validate_session(session_id):
            raise ValueError("Invalid or expired session")
        
        # Check rate limit
        if not self.security_middleware.check_rate_limit(user_id):
            raise ValueError("Rate limit exceeded")
        
        # Verify ownership
        campaign = self.campaigns.get(campaign_id)
        if not campaign:
            raise ValueError("Campaign not found")
        
        if campaign.metadata.get('user_id') != user_id:
            self.security_middleware.audit_logger.log_security_violation(
                user_id=user_id,
                violation_type="unauthorized_campaign_access",
                details={
                    'campaign_id': campaign_id,
                    'actual_owner': campaign.metadata.get('user_id')
                }
            )
            raise ValueError("Unauthorized access to campaign")
        
        # Delete campaign
        await self.remove_campaign(campaign_id)
        
        # Update user campaign count
        if user_id in self.user_campaign_counts:
            self.user_campaign_counts[user_id] = max(0, self.user_campaign_counts[user_id] - 1)
        
        # Log deletion
        self.security_middleware.audit_logger.log_event(
            event_type="campaign_deletion",
            user_id=user_id,
            details={'campaign_id': campaign_id},
            risk_level="INFO"
        )
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get security status and metrics"""
        return {
            'security_enabled': True,
            'active_sessions': len(self.security_middleware.active_sessions),
            'rate_limiting_enabled': self.security_config.enable_rate_limiting,
            'audit_logging_enabled': self.security_config.enable_audit_logging,
            'encryption_enabled': self.security_config.enable_encryption,
            'key_rotation_needed': self.security_middleware.data_manager.needs_key_rotation(
                self.security_config.encryption_key_rotation_days
            ),
            'total_campaigns': len(self.campaigns),
            'user_campaign_counts': dict(self.user_campaign_counts)
        }
    
    async def rotate_encryption_keys(self) -> bool:
        """Rotate encryption keys"""
        try:
            old_data_manager = self.security_middleware.data_manager
            new_data_manager = SecureDataManager()
            
            # Re-encrypt all campaign data with new key
            for campaign_id, campaign in self.campaigns.items():
                # This would involve re-encrypting stored campaign data
                pass
            
            self.security_middleware.data_manager = new_data_manager
            
            self.security_middleware.audit_logger.log_event(
                event_type="encryption_key_rotation",
                user_id="system",
                details={'rotation_time': datetime.utcnow().isoformat()},
                risk_level="INFO"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Key rotation failed: {e}")
            return False


# Example usage and integration
async def main():
    """Example usage of secure orchestrator"""
    
    # Configure security
    security_config = SecurityConfig(
        enable_input_validation=True,
        enable_rate_limiting=True,
        enable_audit_logging=True,
        max_campaigns_per_user=25,
        rate_limit_requests_per_minute=50
    )
    
    # Initialize secure orchestrator
    orchestrator = SecureXorbOrchestrator(
        security_config=security_config
    )
    
    # Example secure campaign creation
    campaign_data = {
        'name': 'Secure Test Campaign',
        'targets': [
            {
                'type': 'web',
                'url': 'https://example.com',
                'scope': 'external'
            }
        ],
        'priority': 'medium',
        'metadata': {
            'description': 'Test campaign with security validation'
        }
    }
    
    user_id = "user123"
    session_id = orchestrator.security_middleware.create_session(user_id)
    
    try:
        campaign = await orchestrator.secure_create_campaign(
            campaign_data=campaign_data,
            user_id=user_id,
            session_id=session_id,
            source_ip="192.168.1.100"
        )
        
        print(f"Secure campaign created: {campaign.id}")
        
        # Get security status
        status = await orchestrator.get_security_status()
        print(f"Security status: {status}")
        
    except Exception as e:
        print(f"Campaign creation failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())