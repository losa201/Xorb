#!/usr/bin/env python3
"""
XORB Advanced Security Hardening System
Enterprise-grade security with JWT authentication, RBAC, and audit logging
"""

import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import jwt
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import redis.asyncio as redis
import asyncpg
import uvicorn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
from common.security_utils import PASSWORD_CONTEXT, hash_password, verify_password

app = FastAPI(
    title="XORB Security Hardener",
    description="Enterprise Security Management for XORB Platform",
    version="1.0.0"
)

# Security configuration
SECRET_KEY = "xorb_enterprise_security_key_2024_ultra_secure"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Use centralized password context
pwd_context = PASSWORD_CONTEXT
security = HTTPBearer()

# Models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    role: str = "analyst"

class UserLogin(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class AuditLog(BaseModel):
    user_id: str
    action: str
    resource: str
    timestamp: datetime
    ip_address: str
    user_agent: str
    status: str

class SecurityHardener:
    """Advanced security management system"""

    def __init__(self):
        self.db_pool = None
        self.redis_client = None
        self.failed_attempts = {}

    async def initialize(self):
        """Initialize security system"""
        # Database connection
        database_url = "postgresql://xorb:xorb_secure_2024@localhost:5432/xorb_ptaas"
        self.db_pool = await asyncpg.create_pool(database_url)

        # Redis for session management
        self.redis_client = redis.from_url("redis://localhost:6379/3")

        # Create security tables
        await self.create_security_tables()
        await self.create_default_admin()

    async def create_security_tables(self):
        """Create security-related database tables"""
        async with self.db_pool.acquire() as conn:
            # Users table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS security_users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    role VARCHAR(20) DEFAULT 'analyst',
                    is_active BOOLEAN DEFAULT TRUE,
                    last_login TIMESTAMP,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # Sessions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS security_sessions (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES security_users(id),
                    session_token VARCHAR(255) UNIQUE NOT NULL,
                    refresh_token VARCHAR(255) UNIQUE NOT NULL,
                    ip_address INET,
                    user_agent TEXT,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)

            # Audit logs table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS security_audit_logs (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES security_users(id),
                    action VARCHAR(100) NOT NULL,
                    resource VARCHAR(200),
                    ip_address INET,
                    user_agent TEXT,
                    status VARCHAR(20),
                    details JSONB,
                    timestamp TIMESTAMP DEFAULT NOW()
                )
            """)

            # API keys table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS security_api_keys (
                    id SERIAL PRIMARY KEY,
                    key_id VARCHAR(50) UNIQUE NOT NULL,
                    key_hash VARCHAR(255) NOT NULL,
                    user_id INTEGER REFERENCES security_users(id),
                    name VARCHAR(100) NOT NULL,
                    permissions JSONB DEFAULT '[]',
                    is_active BOOLEAN DEFAULT TRUE,
                    expires_at TIMESTAMP,
                    last_used TIMESTAMP,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

    async def create_default_admin(self):
        """Create default admin user if not exists"""
        async with self.db_pool.acquire() as conn:
            existing = await conn.fetchval(
                "SELECT id FROM security_users WHERE username = 'admin'"
            )

            if not existing:
                password_hash = pwd_context.hash("XorbAdmin2024!")
                await conn.execute("""
                    INSERT INTO security_users (username, email, password_hash, role)
                    VALUES ('admin', 'admin@xorb.security', $1, 'admin')
                """, password_hash)

    def create_access_token(self, data: dict) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    def create_refresh_token(self, data: dict) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    async def verify_token(self, token: str) -> Optional[dict]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id = payload.get("sub")
            if user_id is None:
                return None

            # Check if session is still active
            session_active = await self.redis_client.get(f"session:{user_id}")
            if not session_active:
                return None

            return payload
        except jwt.PyJWTError:
            return None

    async def authenticate_user(self, username: str, password: str) -> Optional[dict]:
        """Authenticate user with username/password"""
        async with self.db_pool.acquire() as conn:
            user = await conn.fetchrow("""
                SELECT id, username, email, password_hash, role, is_active
                FROM security_users WHERE username = $1
            """, username)

            if not user or not user["is_active"]:
                return None

            if not pwd_context.verify(password, user["password_hash"]):
                return None

            # Update last login
            await conn.execute("""
                UPDATE security_users SET last_login = NOW()
                WHERE id = $1
            """, user["id"])

            return dict(user)

    async def create_user_session(self, user: dict, request: Request) -> TokenResponse:
        """Create authenticated session for user"""
        user_id = str(user["id"])

        # Create tokens
        access_token = self.create_access_token(data={"sub": user_id, "role": user["role"]})
        refresh_token = self.create_refresh_token(data={"sub": user_id})

        # Store session in Redis
        await self.redis_client.setex(
            f"session:{user_id}",
            ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            json.dumps({
                "user_id": user_id,
                "username": user["username"],
                "role": user["role"]
            })
        )

        # Store session in database
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO security_sessions
                (user_id, session_token, refresh_token, ip_address, user_agent, expires_at)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, user["id"], access_token[:50], refresh_token[:50],
                 request.client.host, request.headers.get("user-agent", ""),
                 datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )

    async def log_audit_event(self, user_id: str, action: str, resource: str,
                             request: Request, status: str = "success", details: dict = None):
        """Log audit event"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO security_audit_logs
                (user_id, action, resource, ip_address, user_agent, status, details)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, int(user_id) if user_id.isdigit() else None, action, resource,
                 request.client.host, request.headers.get("user-agent", ""),
                 status, json.dumps(details or {}))

    async def check_rate_limit(self, ip_address: str, action: str) -> bool:
        """Check rate limiting for security actions"""
        key = f"rate_limit:{action}:{ip_address}"
        current = await self.redis_client.incr(key)
        if current == 1:
            await self.redis_client.expire(key, 300)  # 5 minutes

        # Different limits for different actions
        limits = {
            "login": 5,
            "api_call": 100,
            "admin_action": 10
        }

        return current <= limits.get(action, 20)

    async def generate_api_key(self, user_id: int, name: str, permissions: List[str]) -> dict:
        """Generate API key for user"""
        import secrets

        key_id = f"xorb_{secrets.token_urlsafe(16)}"
        api_key = f"xorb_key_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO security_api_keys
                (key_id, key_hash, user_id, name, permissions)
                VALUES ($1, $2, $3, $4, $5)
            """, key_id, key_hash, user_id, name, json.dumps(permissions))

        return {
            "key_id": key_id,
            "api_key": api_key,  # Only returned once
            "name": name,
            "permissions": permissions
        }

# Initialize security system
security_hardener = SecurityHardener()

@app.on_event("startup")
async def startup_event():
    """Initialize security hardener on startup"""
    await security_hardener.initialize()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    token = credentials.credentials
    payload = await security_hardener.verify_token(token)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload

@app.post("/auth/login", response_model=TokenResponse)
async def login(user_login: UserLogin, request: Request):
    """User login endpoint"""
    # Rate limiting
    if not await security_hardener.check_rate_limit(request.client.host, "login"):
        await security_hardener.log_audit_event(
            None, "login_rate_limited", "auth", request, "failed"
        )
        raise HTTPException(status_code=429, detail="Too many login attempts")

    # Authenticate user
    user = await security_hardener.authenticate_user(user_login.username, user_login.password)
    if not user:
        await security_hardener.log_audit_event(
            None, "login_failed", "auth", request, "failed",
            {"username": user_login.username}
        )
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Create session
    tokens = await security_hardener.create_user_session(user, request)

    # Log successful login
    await security_hardener.log_audit_event(
        str(user["id"]), "login_success", "auth", request, "success"
    )

    return tokens

@app.post("/auth/register", response_model=dict)
async def register(user_create: UserCreate, request: Request):
    """User registration endpoint (admin only in production)"""
    try:
        password_hash = pwd_context.hash(user_create.password)

        async with security_hardener.db_pool.acquire() as conn:
            user_id = await conn.fetchval("""
                INSERT INTO security_users (username, email, password_hash, role)
                VALUES ($1, $2, $3, $4) RETURNING id
            """, user_create.username, user_create.email, password_hash, user_create.role)

        await security_hardener.log_audit_event(
            str(user_id), "user_created", "auth", request, "success"
        )

        return {"message": "User created successfully", "user_id": user_id}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Registration failed: {str(e)}")

@app.get("/auth/profile")
async def get_profile(current_user: dict = Depends(get_current_user), request: Request = None):
    """Get current user profile"""
    user_id = current_user["sub"]

    async with security_hardener.db_pool.acquire() as conn:
        user = await conn.fetchrow("""
            SELECT username, email, role, last_login, created_at
            FROM security_users WHERE id = $1
        """, int(user_id))

    return dict(user) if user else {}

@app.post("/auth/api-key")
async def create_api_key(
    name: str,
    permissions: List[str] = [],
    current_user: dict = Depends(get_current_user),
    request: Request = None
):
    """Create API key for current user"""
    user_id = int(current_user["sub"])

    api_key_data = await security_hardener.generate_api_key(user_id, name, permissions)

    await security_hardener.log_audit_event(
        str(user_id), "api_key_created", "auth", request, "success",
        {"key_name": name}
    )

    return api_key_data

@app.get("/security/audit-logs")
async def get_audit_logs(
    limit: int = 100,
    current_user: dict = Depends(get_current_user)
):
    """Get audit logs (admin only)"""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    async with security_hardener.db_pool.acquire() as conn:
        logs = await conn.fetch("""
            SELECT al.*, u.username
            FROM security_audit_logs al
            LEFT JOIN security_users u ON al.user_id = u.id
            ORDER BY al.timestamp DESC
            LIMIT $1
        """, limit)

    return [dict(log) for log in logs]

@app.get("/security/active-sessions")
async def get_active_sessions(current_user: dict = Depends(get_current_user)):
    """Get active user sessions"""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    async with security_hardener.db_pool.acquire() as conn:
        sessions = await conn.fetch("""
            SELECT s.*, u.username
            FROM security_sessions s
            JOIN security_users u ON s.user_id = u.id
            WHERE s.is_active = TRUE AND s.expires_at > NOW()
            ORDER BY s.created_at DESC
        """)

    return [dict(session) for session in sessions]

@app.get("/health")
async def health_check():
    """Security hardener health check"""
    return {
        "status": "healthy",
        "service": "security_hardener",
        "version": "1.0.0",
        "features": [
            "JWT Authentication",
            "RBAC Authorization",
            "Audit Logging",
            "Rate Limiting",
            "API Key Management",
            "Session Management"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
