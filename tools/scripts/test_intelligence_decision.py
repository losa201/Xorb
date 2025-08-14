#!/usr/bin/env python3
"""
Test Intelligence Decision Endpoint Specifically
"""
import os
# Set environment variable BEFORE importing any modules
os.environ["REQUIRE_MTLS"] = "false"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'api'))

from app.main import app
from app.security.auth import authenticator, Role
import requests
import uvicorn
import threading
import time
from datetime import datetime

def test_intelligence_decision():
    """Test the intelligence decision endpoint specifically"""

    # Start server
    print("🚀 Starting debug server...")
    server_thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": "127.0.0.1", "port": 8090, "log_level": "debug"},
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

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Test intelligence decision endpoint
    print("\n🧠 Testing /v1/intelligence/decisions endpoint...")

    decision_data = {
        "decision_type": "threat_classification",
        "context": {
            "scenario": "network_anomaly_detected",
            "available_data": {
                "indicators": 3,
                "severity_score": 0.7,
                "affected_systems": 2,
                "historical_patterns": []
            },
            "urgency_level": "high",
            "confidence_threshold": 0.8
        }
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8090/v1/intelligence/decisions",
            headers=headers,
            json=decision_data,
            timeout=10
        )
        print(f"Status: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Content: {response.text}")

        if response.status_code != 200:
            print("❌ Request failed with detailed response above")

    except Exception as e:
        print(f"Request failed with exception: {e}")

if __name__ == "__main__":
    test_intelligence_decision()
