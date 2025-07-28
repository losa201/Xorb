"""
Simple JWT implementation for demo purposes
"""

import os
from datetime import datetime, timedelta
from typing import Any

import jwt

SECRET_KEY = os.getenv("JWT_SECRET", "xorb-demo-secret-key-2024")
ALGORITHM = "HS256"

def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify(token: str) -> dict[str, Any] | None:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None

def create_demo_token() -> str:
    """Create a demo token for testing"""
    user_data = {
        "sub": "demo_user",
        "username": "demo_user",
        "role": "admin",
        "permissions": ["read", "write", "admin"]
    }
    return create_access_token(user_data)
