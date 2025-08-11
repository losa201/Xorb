"""
Centralized JWT Management System
Handles JWT secret retrieval, token creation, and validation across all services
"""

import os
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from functools import lru_cache

from .vault_client import get_jwt_secret


class JWTManager:
    """Centralized JWT token management"""
    
    def __init__(self, algorithm: str = "HS256"):
        self.algorithm = algorithm
        self._secret_cache: Optional[str] = None
    
    @property
    async def secret_key(self) -> str:
        """Get JWT secret key with caching"""
        if not self._secret_cache:
            self._secret_cache = await get_jwt_secret()
        return self._secret_cache
    
    def get_secret_key_sync(self) -> str:
        """Get JWT secret key synchronously (for non-async contexts)"""
        return os.getenv("JWT_SECRET", "dev-secret-change-in-production")
    
    async def create_token(
        self,
        payload: Dict[str, Any],
        expires_minutes: Optional[int] = None,
        expires_days: Optional[int] = None
    ) -> str:
        """Create JWT token with specified expiration"""
        if expires_minutes:
            exp = datetime.utcnow() + timedelta(minutes=expires_minutes)
        elif expires_days:
            exp = datetime.utcnow() + timedelta(days=expires_days)
        else:
            exp = datetime.utcnow() + timedelta(minutes=30)  # Default 30 minutes
        
        payload.update({
            "exp": exp,
            "iat": datetime.utcnow()
        })
        
        secret = await self.secret_key
        return jwt.encode(payload, secret, algorithm=self.algorithm)
    
    def create_token_sync(
        self,
        payload: Dict[str, Any],
        expires_minutes: Optional[int] = None,
        expires_days: Optional[int] = None
    ) -> str:
        """Create JWT token synchronously"""
        if expires_minutes:
            exp = datetime.utcnow() + timedelta(minutes=expires_minutes)
        elif expires_days:
            exp = datetime.utcnow() + timedelta(days=expires_days)
        else:
            exp = datetime.utcnow() + timedelta(minutes=30)
        
        payload.update({
            "exp": exp,
            "iat": datetime.utcnow()
        })
        
        secret = self.get_secret_key_sync()
        return jwt.encode(payload, secret, algorithm=self.algorithm)
    
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload"""
        try:
            secret = await self.secret_key
            payload = jwt.decode(token, secret, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.JWTError:
            return None
    
    def verify_token_sync(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token synchronously"""
        try:
            secret = self.get_secret_key_sync()
            payload = jwt.decode(token, secret, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.JWTError:
            return None
    
    async def create_access_token(
        self,
        user_id: str,
        username: str,
        roles: List[str],
        expires_minutes: int = 30
    ) -> str:
        """Create standardized access token"""
        payload = {
            "sub": user_id,
            "username": username,
            "roles": roles,
            "type": "access"
        }
        return await self.create_token(payload, expires_minutes=expires_minutes)
    
    async def create_refresh_token(
        self,
        user_id: str,
        expires_days: int = 7
    ) -> str:
        """Create standardized refresh token"""
        payload = {
            "sub": user_id,
            "type": "refresh"
        }
        return await self.create_token(payload, expires_days=expires_days)
    
    def create_access_token_sync(
        self,
        user_id: str,
        username: str,
        roles: List[str],
        expires_minutes: int = 30
    ) -> str:
        """Create standardized access token synchronously"""
        payload = {
            "sub": user_id,
            "username": username,
            "roles": roles,
            "type": "access"
        }
        return self.create_token_sync(payload, expires_minutes=expires_minutes)
    
    def create_refresh_token_sync(
        self,
        user_id: str,
        expires_days: int = 7
    ) -> str:
        """Create standardized refresh token synchronously"""
        payload = {
            "sub": user_id,
            "type": "refresh"
        }
        return self.create_token_sync(payload, expires_days=expires_days)


# Global JWT manager instance
@lru_cache()
def get_jwt_manager() -> JWTManager:
    """Get cached JWT manager instance"""
    return JWTManager()


# Convenience functions for backward compatibility
async def create_access_token(user_id: str, username: str, roles: List[str], expires_minutes: int = 30) -> str:
    """Create access token using global JWT manager"""
    manager = get_jwt_manager()
    return await manager.create_access_token(user_id, username, roles, expires_minutes)


async def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify token using global JWT manager"""
    manager = get_jwt_manager()
    return await manager.verify_token(token)


def create_access_token_sync(user_id: str, username: str, roles: List[str], expires_minutes: int = 30) -> str:
    """Create access token synchronously using global JWT manager"""
    manager = get_jwt_manager()
    return manager.create_access_token_sync(user_id, username, roles, expires_minutes)


def verify_token_sync(token: str) -> Optional[Dict[str, Any]]:
    """Verify token synchronously using global JWT manager"""
    manager = get_jwt_manager()
    return manager.verify_token_sync(token)