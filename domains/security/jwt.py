"""
XORB JWT Security Manager

Production-ready JWT token management with security best practices.
"""

try:
    import jwt
except ImportError:
    jwt = None

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from domains.core.exceptions import SecurityError


class JWTManager:
    """JWT token manager with security best practices."""
    
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY", "xorb-demo-secret-key-2024")
        self.algorithm = "HS256"
        self.default_expiry = timedelta(hours=24)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token."""
        if not jwt:
            raise SecurityError("PyJWT not available for token creation")
            
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + self.default_expiry
        
        to_encode.update({"exp": expire})
        
        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            raise SecurityError(f"Failed to create JWT token: {e}")
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token."""
        if not jwt:
            raise SecurityError("PyJWT not available for token verification")
            
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise SecurityError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise SecurityError(f"Invalid token: {e}")


# Global JWT manager instance
jwt_manager = JWTManager()