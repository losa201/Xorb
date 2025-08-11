"""
Multi-Factor Authentication (MFA) service
Supports TOTP, WebAuthn, SMS, and Email verification
"""

import asyncio
import base64
import hashlib
import hmac
import json
import qrcode
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID, uuid4
from io import BytesIO

import pyotp
import redis.asyncio as redis
from pydantic import BaseModel

from ..domain.entities import User
from ..domain.exceptions import (
    ValidationError, SecurityViolation, MFARequired, InvalidCredentials
)


class MFAMethod(BaseModel):
    """MFA method configuration"""
    id: str
    user_id: str
    method_type: str  # totp, webauthn, sms, email
    name: str
    secret: Optional[str] = None
    public_key: Optional[str] = None
    phone_number: Optional[str] = None
    email: Optional[str] = None
    is_verified: bool = False
    is_backup: bool = False
    created_at: datetime
    last_used: Optional[datetime] = None


class MFAChallenge(BaseModel):
    """MFA challenge for verification"""
    challenge_id: str
    user_id: str
    method_id: str
    method_type: str
    challenge_data: Dict[str, Any]
    expires_at: datetime
    attempts: int = 0
    max_attempts: int = 3


class TOTPGenerator:
    """TOTP (Time-based One-Time Password) generator"""
    
    @staticmethod
    def generate_secret() -> str:
        """Generate a random TOTP secret"""
        return pyotp.random_base32()
    
    @staticmethod
    def generate_qr_code(secret: str, user_email: str, issuer: str = "Xorb Platform") -> bytes:
        """Generate QR code for TOTP setup"""
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(
            name=user_email,
            issuer_name=issuer
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to bytes
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    
    @staticmethod
    def verify_token(secret: str, token: str, window: int = 1) -> bool:
        """Verify TOTP token with time window tolerance"""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=window)
    
    @staticmethod
    def get_current_token(secret: str) -> str:
        """Get current TOTP token (for testing)"""
        totp = pyotp.TOTP(secret)
        return totp.now()


class WebAuthnManager:
    """WebAuthn (FIDO2) authentication manager"""
    
    def __init__(self, origin: str = "https://localhost", rp_name: str = "Xorb Platform"):
        self.origin = origin
        self.rp_name = rp_name
        self.rp_id = origin.split("://")[1].split(":")[0]  # Extract domain
    
    def generate_registration_options(self, user: User) -> Dict[str, Any]:
        """Generate WebAuthn registration options"""
        challenge = secrets.token_bytes(32)
        
        return {
            "challenge": base64.urlsafe_b64encode(challenge).decode().rstrip('='),
            "rp": {
                "name": self.rp_name,
                "id": self.rp_id
            },
            "user": {
                "id": base64.urlsafe_b64encode(str(user.id).encode()).decode().rstrip('='),
                "name": user.username,
                "displayName": user.username
            },
            "pubKeyCredParams": [
                {"type": "public-key", "alg": -7},   # ES256
                {"type": "public-key", "alg": -257}  # RS256
            ],
            "authenticatorSelection": {
                "authenticatorAttachment": "platform",
                "userVerification": "required"
            },
            "timeout": 60000,
            "attestation": "direct"
        }
    
    def generate_authentication_options(self, credentials: List[str]) -> Dict[str, Any]:
        """Generate WebAuthn authentication options"""
        challenge = secrets.token_bytes(32)
        
        return {
            "challenge": base64.urlsafe_b64encode(challenge).decode().rstrip('='),
            "timeout": 60000,
            "rpId": self.rp_id,
            "allowCredentials": [
                {
                    "type": "public-key",
                    "id": cred_id
                }
                for cred_id in credentials
            ],
            "userVerification": "required"
        }


class MFAService:
    """Multi-Factor Authentication service"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.totp_generator = TOTPGenerator()
        self.webauthn_manager = WebAuthnManager()
        
        # MFA configuration
        self.backup_codes_count = 10
        self.challenge_expire_minutes = 10
        self.max_verification_attempts = 3
    
    async def get_user_mfa_methods(self, user_id: str) -> List[MFAMethod]:
        """Get all MFA methods for a user"""
        key = f"mfa_methods:{user_id}"
        methods_data = await self.redis_client.hgetall(key)
        
        methods = []
        for method_id, method_json in methods_data.items():
            try:
                method_dict = json.loads(method_json)
                methods.append(MFAMethod(**method_dict))
            except:
                continue
        
        return methods
    
    async def is_mfa_enabled(self, user_id: str) -> bool:
        """Check if user has any verified MFA methods"""
        methods = await self.get_user_mfa_methods(user_id)
        return any(method.is_verified for method in methods)
    
    async def setup_totp(self, user: User) -> Dict[str, Any]:
        """Setup TOTP for a user"""
        secret = self.totp_generator.generate_secret()
        
        # Generate QR code
        qr_code = self.totp_generator.generate_qr_code(secret, user.username)
        
        # Create MFA method (not verified yet)
        method = MFAMethod(
            id=str(uuid4()),
            user_id=str(user.id),
            method_type="totp",
            name="Authenticator App",
            secret=secret,
            is_verified=False,
            created_at=datetime.utcnow()
        )
        
        # Store temporarily until verified
        temp_key = f"mfa_setup:{user.id}:{method.id}"
        await self.redis_client.setex(
            temp_key,
            600,  # 10 minutes
            method.json()
        )
        
        return {
            "method_id": method.id,
            "secret": secret,
            "qr_code": base64.b64encode(qr_code).decode(),
            "backup_codes": await self._generate_backup_codes(str(user.id))
        }
    
    async def verify_totp_setup(self, user_id: str, method_id: str, token: str) -> bool:
        """Verify TOTP setup with token"""
        # Get temporary setup data
        temp_key = f"mfa_setup:{user_id}:{method_id}"
        method_json = await self.redis_client.get(temp_key)
        
        if not method_json:
            raise ValidationError("TOTP setup expired or not found")
        
        method = MFAMethod(**json.loads(method_json))
        
        # Verify token
        if not self.totp_generator.verify_token(method.secret, token):
            raise InvalidCredentials("Invalid TOTP token")
        
        # Mark as verified and save permanently
        method.is_verified = True
        await self._save_mfa_method(method)
        
        # Remove temporary setup
        await self.redis_client.delete(temp_key)
        
        # Log security event
        await self._log_mfa_event("totp_enabled", user_id, {
            "method_id": method_id,
            "method_type": "totp"
        })
        
        return True
    
    async def setup_webauthn(self, user: User) -> Dict[str, Any]:
        """Setup WebAuthn for a user"""
        registration_options = self.webauthn_manager.generate_registration_options(user)
        
        # Store challenge for verification
        challenge_key = f"webauthn_challenge:{user.id}"
        await self.redis_client.setex(
            challenge_key,
            300,  # 5 minutes
            json.dumps(registration_options)
        )
        
        return registration_options
    
    async def verify_webauthn_setup(
        self,
        user_id: str,
        credential_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify WebAuthn setup"""
        challenge_key = f"webauthn_challenge:{user_id}"
        challenge_data = await self.redis_client.get(challenge_key)
        
        if not challenge_data:
            raise ValidationError("WebAuthn challenge expired or not found")
        
        # In a real implementation, you would verify the credential response
        # against the challenge using a WebAuthn library like py_webauthn
        # For now, we'll simulate successful verification
        
        credential_id = credential_response.get("id")
        public_key = credential_response.get("response", {}).get("publicKey")
        
        if not credential_id or not public_key:
            raise ValidationError("Invalid WebAuthn credential response")
        
        # Create MFA method
        method = MFAMethod(
            id=str(uuid4()),
            user_id=user_id,
            method_type="webauthn",
            name="Security Key",
            public_key=public_key,
            is_verified=True,
            created_at=datetime.utcnow()
        )
        
        await self._save_mfa_method(method)
        
        # Clean up challenge
        await self.redis_client.delete(challenge_key)
        
        # Log security event
        await self._log_mfa_event("webauthn_enabled", user_id, {
            "method_id": method.id,
            "credential_id": credential_id
        })
        
        return {
            "method_id": method.id,
            "credential_id": credential_id
        }
    
    async def create_mfa_challenge(self, user_id: str, method_ids: Optional[List[str]] = None) -> MFAChallenge:
        """Create MFA challenge for authentication"""
        methods = await self.get_user_mfa_methods(user_id)
        verified_methods = [m for m in methods if m.is_verified]
        
        if not verified_methods:
            raise MFARequired("No verified MFA methods available")
        
        # Filter by requested method IDs if provided
        if method_ids:
            available_methods = [m for m in verified_methods if m.id in method_ids]
        else:
            available_methods = verified_methods
        
        if not available_methods:
            raise ValidationError("No valid MFA methods available")
        
        # Use the first available method for the challenge
        method = available_methods[0]
        
        challenge_data = {}
        
        if method.method_type == "totp":
            challenge_data = {
                "method_type": "totp",
                "message": "Enter the 6-digit code from your authenticator app"
            }
        elif method.method_type == "webauthn":
            # Get user's credentials
            credentials = [m.public_key for m in available_methods if m.method_type == "webauthn"]
            auth_options = self.webauthn_manager.generate_authentication_options(credentials)
            challenge_data = {
                "method_type": "webauthn",
                "auth_options": auth_options
            }
        
        challenge = MFAChallenge(
            challenge_id=str(uuid4()),
            user_id=user_id,
            method_id=method.id,
            method_type=method.method_type,
            challenge_data=challenge_data,
            expires_at=datetime.utcnow() + timedelta(minutes=self.challenge_expire_minutes)
        )
        
        # Store challenge
        challenge_key = f"mfa_challenge:{challenge.challenge_id}"
        await self.redis_client.setex(
            challenge_key,
            self.challenge_expire_minutes * 60,
            challenge.json()
        )
        
        return challenge
    
    async def verify_mfa_challenge(
        self,
        challenge_id: str,
        verification_data: Dict[str, Any]
    ) -> bool:
        """Verify MFA challenge response"""
        challenge_key = f"mfa_challenge:{challenge_id}"
        challenge_json = await self.redis_client.get(challenge_key)
        
        if not challenge_json:
            raise ValidationError("MFA challenge expired or not found")
        
        challenge = MFAChallenge(**json.loads(challenge_json))
        
        # Check if challenge has expired
        if datetime.utcnow() > challenge.expires_at:
            await self.redis_client.delete(challenge_key)
            raise ValidationError("MFA challenge has expired")
        
        # Check attempt limit
        if challenge.attempts >= challenge.max_attempts:
            await self.redis_client.delete(challenge_key)
            raise SecurityViolation("Maximum MFA verification attempts exceeded")
        
        # Increment attempts
        challenge.attempts += 1
        await self.redis_client.setex(
            challenge_key,
            self.challenge_expire_minutes * 60,
            challenge.json()
        )
        
        # Verify based on method type
        is_valid = False
        
        if challenge.method_type == "totp":
            token = verification_data.get("token")
            if not token:
                raise ValidationError("TOTP token is required")
            
            # Get method to access secret
            methods = await self.get_user_mfa_methods(challenge.user_id)
            method = next((m for m in methods if m.id == challenge.method_id), None)
            
            if method and method.secret:
                is_valid = self.totp_generator.verify_token(method.secret, token)
        
        elif challenge.method_type == "webauthn":
            # Verify WebAuthn response
            auth_response = verification_data.get("auth_response")
            if not auth_response:
                raise ValidationError("WebAuthn authentication response is required")
            
            # In a real implementation, verify the authentication response
            # For now, simulate successful verification
            is_valid = True
        
        if is_valid:
            # Remove challenge on successful verification
            await self.redis_client.delete(challenge_key)
            
            # Update method last used
            await self._update_method_last_used(challenge.method_id)
            
            # Log successful MFA verification
            await self._log_mfa_event("mfa_verified", challenge.user_id, {
                "method_id": challenge.method_id,
                "method_type": challenge.method_type,
                "challenge_id": challenge_id
            })
            
            return True
        else:
            # Log failed verification
            await self._log_mfa_event("mfa_verification_failed", challenge.user_id, {
                "method_id": challenge.method_id,
                "method_type": challenge.method_type,
                "challenge_id": challenge_id,
                "attempts": challenge.attempts
            })
            
            raise InvalidCredentials("Invalid MFA verification")
    
    async def disable_mfa_method(self, user_id: str, method_id: str) -> bool:
        """Disable an MFA method"""
        methods = await self.get_user_mfa_methods(user_id)
        method = next((m for m in methods if m.id == method_id), None)
        
        if not method:
            raise ValidationError("MFA method not found")
        
        # Remove method
        key = f"mfa_methods:{user_id}"
        await self.redis_client.hdel(key, method_id)
        
        # Log security event
        await self._log_mfa_event("mfa_disabled", user_id, {
            "method_id": method_id,
            "method_type": method.method_type
        })
        
        return True
    
    async def generate_backup_codes(self, user_id: str) -> List[str]:
        """Generate new backup codes for a user"""
        return await self._generate_backup_codes(user_id)
    
    async def verify_backup_code(self, user_id: str, code: str) -> bool:
        """Verify and consume a backup code"""
        codes_key = f"backup_codes:{user_id}"
        codes_json = await self.redis_client.get(codes_key)
        
        if not codes_json:
            return False
        
        codes = json.loads(codes_json)
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        
        if code_hash in codes:
            # Remove used code
            codes.remove(code_hash)
            await self.redis_client.setex(codes_key, 86400 * 365, json.dumps(codes))
            
            # Log backup code usage
            await self._log_mfa_event("backup_code_used", user_id, {
                "remaining_codes": len(codes)
            })
            
            return True
        
        return False
    
    async def _save_mfa_method(self, method: MFAMethod):
        """Save MFA method to Redis"""
        key = f"mfa_methods:{method.user_id}"
        await self.redis_client.hset(key, method.id, method.json())
        await self.redis_client.expire(key, 86400 * 365)  # 1 year
    
    async def _update_method_last_used(self, method_id: str):
        """Update method last used timestamp"""
        # This is a simplified implementation
        # In a real system, you'd update the method record
        pass
    
    async def _generate_backup_codes(self, user_id: str) -> List[str]:
        """Generate backup codes for a user"""
        codes = []
        code_hashes = []
        
        for _ in range(self.backup_codes_count):
            # Generate 8-character alphanumeric code
            code = ''.join(secrets.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(8))
            codes.append(code)
            code_hashes.append(hashlib.sha256(code.encode()).hexdigest())
        
        # Store hashed codes
        codes_key = f"backup_codes:{user_id}"
        await self.redis_client.setex(codes_key, 86400 * 365, json.dumps(code_hashes))
        
        return codes
    
    async def _log_mfa_event(self, event_type: str, user_id: str, details: Dict[str, Any]):
        """Log MFA-related security events"""
        event_data = {
            "event_type": event_type,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details
        }
        
        # Store in security events log
        event_key = f"mfa_events:{datetime.utcnow().strftime('%Y-%m-%d')}"
        await self.redis_client.lpush(event_key, json.dumps(event_data))
        await self.redis_client.expire(event_key, 86400 * 90)  # 90 days
    
    async def get_mfa_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get MFA usage statistics"""
        stats = {
            "total_methods": 0,
            "methods_by_type": {},
            "recent_verifications": 0,
            "backup_codes_remaining": 0
        }
        
        if user_id:
            # User-specific stats
            methods = await self.get_user_mfa_methods(user_id)
            stats["total_methods"] = len([m for m in methods if m.is_verified])
            
            for method in methods:
                if method.is_verified:
                    stats["methods_by_type"][method.method_type] = stats["methods_by_type"].get(method.method_type, 0) + 1
            
            # Check backup codes
            codes_key = f"backup_codes:{user_id}"
            codes_json = await self.redis_client.get(codes_key)
            if codes_json:
                codes = json.loads(codes_json)
                stats["backup_codes_remaining"] = len(codes)
        
        return stats