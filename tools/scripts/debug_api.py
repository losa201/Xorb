#!/usr/bin/env python3
"""
Debug XORB API - Check what's causing 500 errors
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'api'))

from app.main import app
from app.security.auth import authenticator, Role
import requests
import uvicorn
import threading
import time

def debug_single_endpoint():
    """Test a single endpoint to see the actual error"""

    # Start server
    print("🚀 Starting debug server...")
    server_thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": "127.0.0.1", "port": 8089, "log_level": "info"},  # info level for debugging
        daemon=True
    )
    server_thread.start()
    time.sleep(3)

    # Generate token
    token = authenticator.generate_jwt(
        user_id="debug_user",
        client_id="debug_client",
        roles=[Role.ADMIN]
    )

    headers = {"Authorization": f"Bearer {token}"}

    # Test specific endpoint
    print("\n🔍 Testing /v1/agents endpoint...")
    try:
        response = requests.get("http://127.0.0.1:8089/v1/agents", headers=headers, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Content: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

    # Test health endpoint (which works)
    print("\n✅ Testing /health endpoint...")
    try:
        response = requests.get("http://127.0.0.1:8089/health", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Content: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    debug_single_endpoint()
