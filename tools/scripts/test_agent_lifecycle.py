#!/usr/bin/env python3
"""
Test Agent Lifecycle - Full initialization and command execution
"""
import os
os.environ["REQUIRE_MTLS"] = "false"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'api'))

from app.main import app
from app.security.auth import authenticator, Role
import requests
import uvicorn
import threading
import time

def test_agent_full_lifecycle():
    """Test complete agent lifecycle including initialization"""
    
    # Start server
    print("🚀 Starting test server...")
    server_thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": "127.0.0.1", "port": 8091, "log_level": "error"},
        daemon=True
    )
    server_thread.start()
    time.sleep(3)
    
    # Generate token
    token = authenticator.generate_jwt(
        user_id="test_user", 
        client_id="test_client",
        roles=[Role.ADMIN]
    )
    
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    base_url = "http://127.0.0.1:8091"
    
    print("🤖 Testing Full Agent Lifecycle...")
    
    # Step 1: Create agent with auto_start
    agent_data = {
        "name": "Lifecycle Test Agent",
        "agent_type": "security_analyst",
        "capabilities": ["threat_intelligence", "log_analysis"],
        "description": "Agent for testing full lifecycle",
        "auto_start": True  # This triggers initialization
    }
    
    response = requests.post(f"{base_url}/v1/agents", headers=headers, json=agent_data)
    if response.status_code == 201:
        agent = response.json()
        agent_id = agent["id"]
        print(f"✅ Created agent: {agent['name']} ({agent_id})")
        print(f"   Initial Status: {agent['status']}")
        
        # Step 2: Wait for initialization (agent should transition to ACTIVE)
        print("⏳ Waiting 3 seconds for agent initialization...")
        time.sleep(3)
        
        # Step 3: Check agent status
        response = requests.get(f"{base_url}/v1/agents/{agent_id}/status", headers=headers)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Agent Status After Init: {status['status']}")
            
            # Step 4: Try command now that agent should be active
            if status['status'] == 'active':
                command_data = {
                    "command": "status_check",
                    "parameters": {"level": "detailed"},
                    "timeout_seconds": 30
                }
                
                response = requests.post(f"{base_url}/v1/agents/{agent_id}/commands", headers=headers, json=command_data)
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Command Success: {result['status']}")
                    if 'result' in result:
                        print(f"   Result: {result['result']}")
                    print(f"   Execution Time: {result['execution_time_ms']}ms")
                    return True
                else:
                    print(f"❌ Command Still Failed: {response.status_code} - {response.text}")
            else:
                print(f"⚠️ Agent still not active: {status['status']}")
        else:
            print(f"❌ Status check failed: {response.status_code}")
    else:
        print(f"❌ Agent creation failed: {response.status_code}")
    
    return False

if __name__ == "__main__":
    success = test_agent_full_lifecycle()
    print(f"\n{'🎉 SUCCESS!' if success else '❌ FAILED'}")