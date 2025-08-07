import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4

import aioredis
import jwt
from sqlalchemy.ext.asyncio import AsyncSession

from xorb.shared.config import PlatformConfig
from xorb.shared.models import UnifiedUser, APIKeyModel
from xorb.database.repositories import UserRepository, APIKeyRepository

# Unified Authentication and Authorization Service
class UnifiedAuthService:
    def __init__(self, db_session: AsyncSession, redis_client: aioredis.Redis):
        self.db_session = db_session
        self.redis = redis_client
        self.user_repo = UserRepository(db_session)
        self.api_key_repo = APIKeyRepository(db_session)
        self.logger = logging.getLogger(__name__)
        
    async def create_api_key(self, user_id: str, name: str, scopes: List[str]) -> tuple[str, APIKeyModel]:
        """Create a new API key for a user."""
        # Generate secure API key
        raw_key = f"xorb_{uuid4().hex}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        api_key = APIKeyModel(
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            scopes=scopes,
            rate_limit=PlatformConfig.RATE_LIMIT_REQUESTS
        )
        
        # Store in database
        await self.api_key_repo.create_api_key(api_key)
        
        return raw_key, api_key
    
    async def validate_api_key(self, api_key: str) -> Optional[APIKeyModel]:
        """Validate API key and return key model."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        key_model = await self.api_key_repo.get_api_key_by_hash(key_hash)
        if not key_model:
            return None
        
        # Update last used
        key_model.last_used = datetime.utcnow()
        await self.api_key_repo.create_api_key(key_model) # This will update if key_id exists
        
        return key_model
    
    async def create_jwt_token(self, user: UnifiedUser) -> str:
        """Create JWT token for user."""
        payload = {
            "user_id": user.id,
            "username": user.username,
            "roles": user.roles,
            "permissions": user.permissions,
            "exp": datetime.utcnow() + timedelta(hours=PlatformConfig.JWT_EXPIRY_HOURS),
            "iat": datetime.utcnow(),
            "iss": "xorb-platform"
        }
        
        return jwt.encode(payload, PlatformConfig.JWT_SECRET, algorithm=PlatformConfig.JWT_ALGORITHM)
    
    async def validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token and return payload."""
        try:
            payload = jwt.decode(
                token, 
                PlatformConfig.JWT_SECRET, 
                algorithms=[PlatformConfig.JWT_ALGORITHM]
            )
            
            # Check if token is in blacklist
            is_blacklisted = await self.redis.get(f"blacklist:{token}")
            if is_blacklisted:
                return None
                
            return payload
        except jwt.ExpiredSignatureError:
            self.logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("Invalid JWT token")
            return None
    
    async def check_permissions(self, user_perms: Dict[str, bool], required_perms: List[str]) -> bool:
        """Check if user has required permissions."""
        return all(user_perms.get(perm, False) for perm in required_perms)