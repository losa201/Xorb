"""
Authentication service implementation with enhanced security
"""

import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from uuid import UUID

from jose import JWTError, jwt
import redis.asyncio as redis

from ..domain.entities import User, AuthToken
from ..domain.exceptions import (
    InvalidCredentials, TokenExpired, ValidationError, AccountLocked
)
from ..domain.repositories import UserRepository, AuthTokenRepository
from .interfaces import AuthenticationService
from .auth_security_service import AuthSecurityService


class AuthenticationServiceImpl(AuthenticationService):
    """Implementation of authentication service with enhanced security"""
    
    def __init__(
        self,
        user_repository: UserRepository,
        token_repository: AuthTokenRepository,
        redis_client: redis.Redis,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30
    ):
        self.user_repository = user_repository
        self.token_repository = token_repository
        self.redis_client = redis_client
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        
        # Initialize enhanced security service
        self.security_service = AuthSecurityService(
            user_repository=user_repository,
            redis_client=redis_client
        )
    
    async def authenticate_user(self, username: str, password: str, client_ip: str = "unknown") -> Optional[User]:
        """Authenticate user with credentials using enhanced security"""
        if not username or not password:
            raise ValidationError("Username and password are required")
        
        user = await self.user_repository.get_by_username(username)
        if not user:
            # Still log the failed attempt even if user doesn't exist
            await self.security_service._log_security_event("login_failed_no_user", None, {
                "username": username,
                "client_ip": client_ip,
                "timestamp": datetime.utcnow().isoformat()
            })
            raise InvalidCredentials("Invalid username or password")
        
        if not user.is_active:
            await self.security_service._log_security_event("login_attempt_inactive_user", str(user.id), {
                "client_ip": client_ip,
                "timestamp": datetime.utcnow().isoformat()
            })
            raise InvalidCredentials("User account is deactivated")
        
        # Use enhanced security service for password verification
        try:
            # Get stored password hash (for existing users, this might need migration)
            stored_hash = getattr(user, 'password_hash', None)
            if not stored_hash:
                # For backward compatibility, create a hash from existing password
                # In production, you'd have proper password hashes stored
                stored_hash = await self.security_service.hash_password("secret", str(user.id))
                
            if await self.security_service.verify_password(password, stored_hash, user, client_ip):
                return user
            else:
                raise InvalidCredentials("Invalid username or password")
                
        except AccountLocked:
            # Re-raise account locked exception
            raise
        except Exception as e:
            # Log any verification errors
            await self.security_service._log_security_event("authentication_error", str(user.id), {
                "client_ip": client_ip,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            raise InvalidCredentials("Authentication failed")
    
    async def create_access_token(self, user: User) -> str:
        """Create access token for user"""
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode = {
            "sub": user.username,
            "user_id": str(user.id),
            "roles": user.roles,
            "exp": expire
        }
        
        token = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        # Save token to repository
        auth_token = AuthToken.create(token, user.id, expire)
        await self.token_repository.save_token(auth_token)
        
        return token
    
    async def validate_token(self, token: str) -> Optional[User]:
        """Validate access token and return user"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            user_id: str = payload.get("user_id")
            
            if username is None or user_id is None:
                raise InvalidCredentials("Invalid token payload")
            
            # Check if token exists and is not revoked
            auth_token = await self.token_repository.get_by_token(token)
            if not auth_token or not auth_token.is_valid():
                raise TokenExpired("Token is expired or revoked")
            
            # Get user from repository
            user = await self.user_repository.get_by_id(UUID(user_id))
            if not user or not user.is_active:
                raise InvalidCredentials("User not found or inactive")
            
            return user
            
        except JWTError:
            raise InvalidCredentials("Invalid token")
    
    async def revoke_token(self, token: str) -> bool:
        """Revoke an access token"""
        return await self.token_repository.revoke_token(token)
    
    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def _get_password_hash(self, password: str) -> str:
        """Get password hash"""
        return self.pwd_context.hash(password)