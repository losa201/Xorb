"""
Enhanced Authentication Service
Handles traditional auth and SSO user management
"""

import jwt
import hashlib
import secrets
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import aioredis
import asyncio

from src.common.models.auth import User, UserRole, SSOProvider, AuthToken, Session, DEFAULT_ROLES
from src.common.config import get_settings


class AuthService:
    """Enhanced authentication service with SSO support"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.settings = get_settings()
        self.jwt_secret = self.settings.jwt_secret
        self.jwt_algorithm = "HS256"
        
    async def create_or_update_sso_user(self, provider: str, user_info: Dict[str, Any], tokens: Dict[str, Any]) -> User:
        """Create or update user from SSO provider"""
        try:
            sso_user_id = user_info.get("id")
            email = user_info.get("email")
            username = user_info.get("username") or email.split("@")[0] if email else "user"
            
            # Check if user exists by SSO provider and user ID
            existing_user = await self.get_user_by_sso_id(provider, sso_user_id)
            
            if existing_user:
                # Update existing user
                existing_user.last_login = datetime.utcnow()
                existing_user.sso_metadata.update({
                    "last_provider_sync": datetime.utcnow().isoformat(),
                    "provider_data": user_info
                })
                
                # Store updated tokens
                if tokens:
                    await self._store_sso_tokens(existing_user.id, tokens)
                    
                # Update user in storage
                await self._update_user(existing_user)
                return existing_user
            else:
                # Check if user exists by email (account linking)
                existing_user = await self.get_user_by_email(email) if email else None
                
                if existing_user:
                    # Link SSO to existing account
                    existing_user.sso_provider = SSOProvider(provider)
                    existing_user.sso_user_id = sso_user_id
                    existing_user.last_login = datetime.utcnow()
                    existing_user.sso_metadata = {
                        "linked_at": datetime.utcnow().isoformat(),
                        "provider_data": user_info
                    }
                    
                    if tokens:
                        await self._store_sso_tokens(existing_user.id, tokens)
                        
                    await self._update_user(existing_user)
                    return existing_user
                else:
                    # Create new user
                    user_id = f"sso_{provider}_{sso_user_id}"
                    
                    new_user = User(
                        id=user_id,
                        username=username,
                        email=email or f"{username}@{provider}.local",
                        role=UserRole.USER,  # Default role for SSO users
                        permissions=self._get_default_permissions(UserRole.USER),
                        is_active=True,
                        is_verified=True,  # SSO users are considered verified
                        created_at=datetime.utcnow(),
                        last_login=datetime.utcnow(),
                        sso_provider=SSOProvider(provider),
                        sso_user_id=sso_user_id,
                        sso_metadata={
                            "created_from_sso": True,
                            "provider_data": user_info
                        },
                        full_name=user_info.get("name"),
                        avatar_url=user_info.get("picture") or user_info.get("avatar_url")
                    )
                    
                    # Store user
                    await self._store_user(new_user)
                    
                    # Store SSO tokens
                    if tokens:
                        await self._store_sso_tokens(new_user.id, tokens)
                        
                    return new_user
                    
        except Exception as e:
            raise Exception(f"Failed to create/update SSO user: {str(e)}")
            
    async def generate_jwt_token(self, user: User) -> str:
        """Generate JWT token for user"""
        payload = {
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value if isinstance(user.role, UserRole) else user.role,
            "permissions": user.permissions,
            "sso_provider": user.sso_provider.value if user.sso_provider else None,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        # Store token in Redis for validation
        await self.redis.setex(
            f"jwt_token:{user.id}:{token[-8:]}",  # Use last 8 chars as identifier
            3600,  # 1 hour
            "valid"
        )
        
        return token
        
    async def validate_jwt_token(self, token: str) -> Optional[User]:
        """Validate JWT token and return user"""
        try:
            # Decode token
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Check if token is in Redis (not revoked)
            user_id = payload.get("user_id")
            token_key = f"jwt_token:{user_id}:{token[-8:]}"
            
            if not await self.redis.exists(token_key):
                return None
                
            # Get user from storage
            user = await self.get_user_by_id(user_id)
            if not user or not user.is_active:
                return None
                
            return user
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        except Exception:
            return None
            
    async def invalidate_user_session(self, user_id: str):
        """Invalidate all user sessions"""
        try:
            # Remove all JWT tokens for user
            pattern = f"jwt_token:{user_id}:*"
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
                
            # Remove SSO tokens
            await self.redis.delete(f"sso_tokens:{user_id}")
            
            # Remove user sessions
            await self.redis.delete(f"user_session:{user_id}")
            
        except Exception as e:
            print(f"Failed to invalidate session for user {user_id}: {e}")
            
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        try:
            user_data = await self.redis.get(f"user:{user_id}")
            if user_data:
                data = json.loads(user_data.decode('utf-8'))
                return self._deserialize_user(data)
            return None
        except Exception:
            return None
            
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        try:
            # In a real implementation, this would query a database
            # For now, we'll use a simple Redis lookup
            user_id = await self.redis.get(f"email_to_user:{email}")
            if user_id:
                return await self.get_user_by_id(user_id.decode('utf-8'))
            return None
        except Exception:
            return None
            
    async def get_user_by_sso_id(self, provider: str, sso_user_id: str) -> Optional[User]:
        """Get user by SSO provider and user ID"""
        try:
            user_id = await self.redis.get(f"sso_user:{provider}:{sso_user_id}")
            if user_id:
                return await self.get_user_by_id(user_id.decode('utf-8'))
            return None
        except Exception:
            return None
            
    async def _store_user(self, user: User):
        """Store user in Redis"""
        import json
        try:
            user_data = self._serialize_user(user)
            
            # Store user data
            await self.redis.set(f"user:{user.id}", json.dumps(user_data))
            
            # Create lookup indexes
            if user.email:
                await self.redis.set(f"email_to_user:{user.email}", user.id)
                
            if user.sso_provider and user.sso_user_id:
                await self.redis.set(
                    f"sso_user:{user.sso_provider.value}:{user.sso_user_id}",
                    user.id
                )
                
        except Exception as e:
            raise Exception(f"Failed to store user: {str(e)}")
            
    async def _update_user(self, user: User):
        """Update user in storage"""
        await self._store_user(user)
        
    async def _store_sso_tokens(self, user_id: str, tokens: Dict[str, Any]):
        """Store SSO tokens for user"""
        import json
        await self.redis.setex(
            f"sso_tokens:{user_id}",
            86400 * 7,  # 7 days
            json.dumps(tokens)
        )
        
    def _serialize_user(self, user: User) -> Dict[str, Any]:
        """Serialize user to dictionary"""
        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value if isinstance(user.role, UserRole) else user.role,
            "permissions": user.permissions,
            "is_active": user.is_active,
            "is_verified": user.is_verified,
            "created_at": user.created_at.isoformat(),
            "last_login": user.last_login.isoformat() if user.last_login else None,
            "sso_provider": user.sso_provider.value if user.sso_provider else None,
            "sso_user_id": user.sso_user_id,
            "sso_metadata": user.sso_metadata,
            "full_name": user.full_name,
            "avatar_url": user.avatar_url,
            "timezone": user.timezone,
            "language": user.language
        }
        
    def _deserialize_user(self, data: Dict[str, Any]) -> User:
        """Deserialize user from dictionary"""
        return User(
            id=data["id"],
            username=data["username"],
            email=data["email"],
            role=UserRole(data["role"]),
            permissions=data.get("permissions", []),
            is_active=data.get("is_active", True),
            is_verified=data.get("is_verified", False),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_login=datetime.fromisoformat(data["last_login"]) if data.get("last_login") else None,
            sso_provider=SSOProvider(data["sso_provider"]) if data.get("sso_provider") else None,
            sso_user_id=data.get("sso_user_id"),
            sso_metadata=data.get("sso_metadata", {}),
            full_name=data.get("full_name"),
            avatar_url=data.get("avatar_url"),
            timezone=data.get("timezone", "UTC"),
            language=data.get("language", "en")
        )
        
    def _get_default_permissions(self, role: UserRole) -> List[str]:
        """Get default permissions for role"""
        role_permissions = {
            UserRole.ADMIN: ["*"],
            UserRole.ANALYST: [
                "dashboard.read", "scans.*", "reports.*", 
                "compliance.read", "threats.read"
            ],
            UserRole.USER: [
                "dashboard.read", "scans.read", "reports.read"
            ],
            UserRole.READONLY: [
                "dashboard.read", "scans.read", "reports.read"
            ]
        }
        
        return role_permissions.get(role, ["dashboard.read"])