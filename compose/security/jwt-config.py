"""
JWT Security Configuration for Xorb PTaaS
Enhanced JWT handling with service-to-service authentication
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import jwt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

class JWTSecurityConfig:
    """Enhanced JWT security configuration for service-to-service auth"""
    
    def __init__(self):
        self.algorithm = "RS256"
        self.access_token_expire_minutes = 15
        self.refresh_token_expire_days = 7
        self.service_token_expire_hours = 24
        
        # Generate RSA key pair for JWT signing
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        self.public_key = self.private_key.public_key()
        
        # Service-specific configurations
        self.service_configs = {
            "api": {
                "issuer": "xorb-api",
                "audience": ["xorb-worker", "xorb-orchestrator"],
                "scopes": ["read", "write", "admin"]
            },
            "worker": {
                "issuer": "xorb-worker", 
                "audience": ["xorb-api", "xorb-scanner", "xorb-triage"],
                "scopes": ["execute", "report"]
            },
            "orchestrator": {
                "issuer": "xorb-orchestrator",
                "audience": ["xorb-api", "xorb-worker", "xorb-scheduler"],
                "scopes": ["orchestrate", "schedule"]
            },
            "scanner": {
                "issuer": "xorb-scanner",
                "audience": ["xorb-worker", "xorb-triage"],
                "scopes": ["scan", "report"]
            },
            "triage": {
                "issuer": "xorb-triage",
                "audience": ["xorb-api", "xorb-payments"],
                "scopes": ["analyze", "classify"]
            },
            "payments": {
                "issuer": "xorb-payments",
                "audience": ["xorb-api"],
                "scopes": ["pay", "verify"]
            }
        }
    
    def get_private_key_pem(self) -> str:
        """Get private key in PEM format"""
        return self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')
    
    def get_public_key_pem(self) -> str:
        """Get public key in PEM format"""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
    
    def create_service_token(self, service: str, target_service: str = None) -> str:
        """Create JWT token for service-to-service communication"""
        if service not in self.service_configs:
            raise ValueError(f"Unknown service: {service}")
        
        config = self.service_configs[service]
        now = datetime.utcnow()
        
        payload = {
            "iss": config["issuer"],
            "aud": config["audience"] if not target_service else [f"xorb-{target_service}"],
            "sub": f"service:{service}",
            "iat": now,
            "exp": now + timedelta(hours=self.service_token_expire_hours),
            "jti": secrets.token_urlsafe(32),
            "scope": " ".join(config["scopes"]),
            "service": service,
            "type": "service"
        }
        
        return jwt.encode(payload, self.get_private_key_pem(), algorithm=self.algorithm)
    
    def create_user_token(self, user_id: str, email: str, roles: list = None) -> Dict[str, str]:
        """Create access and refresh tokens for user authentication"""
        now = datetime.utcnow()
        roles = roles or ["researcher"]
        
        # Access token
        access_payload = {
            "iss": "xorb-api",
            "aud": ["xorb-portal", "xorb-api"],
            "sub": user_id,
            "email": email,
            "roles": roles,
            "iat": now,
            "exp": now + timedelta(minutes=self.access_token_expire_minutes),
            "jti": secrets.token_urlsafe(32),
            "type": "access"
        }
        
        # Refresh token
        refresh_payload = {
            "iss": "xorb-api",
            "aud": ["xorb-api"],
            "sub": user_id,
            "iat": now,
            "exp": now + timedelta(days=self.refresh_token_expire_days),
            "jti": secrets.token_urlsafe(32),
            "type": "refresh"
        }
        
        return {
            "access_token": jwt.encode(access_payload, self.get_private_key_pem(), algorithm=self.algorithm),
            "refresh_token": jwt.encode(refresh_payload, self.get_private_key_pem(), algorithm=self.algorithm),
            "token_type": "Bearer",
            "expires_in": self.access_token_expire_minutes * 60
        }
    
    def verify_token(self, token: str, expected_audience: str = None) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.get_public_key_pem(),
                algorithms=[self.algorithm],
                audience=expected_audience,
                options={"verify_exp": True, "verify_aud": True}
            )
            return payload
        except jwt.InvalidTokenError:
            return None
    
    def generate_secret_key(self) -> str:
        """Generate a secure secret key for fallback scenarios"""
        return secrets.token_urlsafe(64)

# Global JWT config instance
jwt_config = JWTSecurityConfig()

# Environment variables for Docker
def generate_env_vars():
    """Generate environment variables for Docker compose"""
    env_vars = {
        "JWT_PRIVATE_KEY": jwt_config.get_private_key_pem().replace('\n', '\\n'),
        "JWT_PUBLIC_KEY": jwt_config.get_public_key_pem().replace('\n', '\\n'),
        "JWT_ALGORITHM": jwt_config.algorithm,
        "JWT_ACCESS_TOKEN_EXPIRE_MINUTES": str(jwt_config.access_token_expire_minutes),
        "JWT_SECRET_KEY": jwt_config.generate_secret_key()
    }
    
    # Generate service tokens
    for service in jwt_config.service_configs.keys():
        token = jwt_config.create_service_token(service)
        env_vars[f"JWT_SERVICE_TOKEN_{service.upper()}"] = token
    
    return env_vars

if __name__ == "__main__":
    # Generate and print environment variables
    env_vars = generate_env_vars()
    
    print("# JWT Configuration Environment Variables")
    print("# Add these to your .env file")
    print()
    
    for key, value in env_vars.items():
        if key.startswith("JWT_"):
            print(f"{key}={value}")
    
    print()
    print("# Service tokens generated for inter-service communication")
    print("# These tokens are valid for 24 hours")