#!/usr/bin/env python3
"""
XORB Test Agent

Simple test agent for verifying autonomous operations and agent discovery.
This agent performs basic HTTP requests and validates responses.

Author: XORB Autonomous Systems
Version: 2.0.0
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AgentCapability:
    name: str
    description: str
    parameters: Dict[str, Any]


class TestAgent:
    """
    Simple test agent for autonomous operations verification.
    """
    
    def __init__(self):
        self.agent_id = "test-agent-001"
        self.agent_type = "http_scanner"
        self.status = "available"
        self.capabilities = [
            AgentCapability(
                name="http_get", 
                description="Perform HTTP GET requests",
                parameters={"url": "string", "headers": "dict"}
            ),
            AgentCapability(
                name="response_analysis",
                description="Analyze HTTP responses",
                parameters={"response": "object"}
            ),
            AgentCapability(
                name="vulnerability_scan",
                description="Basic vulnerability scanning",
                parameters={"target": "string"}
            )
        ]
        
    async def execute_mission(self, mission: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a mission and return results"""
        mission_type = mission.get('mission_type', 'unknown')
        target_url = mission.get('target_url', 'http://httpbin.org')
        
        logger.info(f"🎯 Executing mission: {mission_type} on {target_url}")
        
        try:
            if mission_type == "http_scan":
                return await self._http_scan(target_url)
            elif mission_type == "vulnerability_scan":
                return await self._vulnerability_scan(target_url)
            else:
                return await self._basic_scan(target_url)
                
        except Exception as e:
            logger.error(f"Mission failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _http_scan(self, target_url: str) -> Dict[str, Any]:
        """Perform basic HTTP scanning"""
        results = {
            "mission_type": "http_scan",
            "target": target_url,
            "timestamp": datetime.now().isoformat(),
            "findings": []
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                # Basic GET request
                async with session.get(target_url, timeout=10) as response:
                    results["findings"].append({
                        "type": "http_response",
                        "status_code": response.status,
                        "headers": dict(response.headers),
                        "content_length": len(await response.text())
                    })
                
                # Check common endpoints
                endpoints = ["/robots.txt", "/sitemap.xml", "/.well-known/security.txt"]
                for endpoint in endpoints:
                    test_url = target_url.rstrip('/') + endpoint
                    try:
                        async with session.get(test_url, timeout=5) as resp:
                            if resp.status == 200:
                                results["findings"].append({
                                    "type": "endpoint_found",
                                    "endpoint": endpoint,
                                    "status_code": resp.status
                                })
                    except:
                        pass
                        
            except Exception as e:
                results["findings"].append({
                    "type": "error",
                    "message": str(e)
                })
        
        results["status"] = "completed"
        return results
    
    async def _vulnerability_scan(self, target_url: str) -> Dict[str, Any]:
        """Perform basic vulnerability scanning"""
        results = {
            "mission_type": "vulnerability_scan", 
            "target": target_url,
            "timestamp": datetime.now().isoformat(),
            "vulnerabilities": []
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                # Check for common security headers
                async with session.get(target_url, timeout=10) as response:
                    headers = dict(response.headers)
                    
                    security_headers = [
                        "X-Frame-Options",
                        "X-Content-Type-Options", 
                        "X-XSS-Protection",
                        "Strict-Transport-Security",
                        "Content-Security-Policy"
                    ]
                    
                    for header in security_headers:
                        if header not in headers:
                            results["vulnerabilities"].append({
                                "type": "missing_security_header",
                                "header": header,
                                "severity": "medium",
                                "description": f"Missing {header} security header"
                            })
                    
                    # Check for server information disclosure
                    if "Server" in headers:
                        results["vulnerabilities"].append({
                            "type": "information_disclosure",
                            "field": "Server",
                            "value": headers["Server"],
                            "severity": "low",
                            "description": "Server header reveals software information"
                        })
                        
            except Exception as e:
                results["vulnerabilities"].append({
                    "type": "scan_error",
                    "message": str(e)
                })
        
        results["status"] = "completed"
        return results
    
    async def _basic_scan(self, target_url: str) -> Dict[str, Any]:
        """Perform basic connectivity scan"""
        results = {
            "mission_type": "basic_scan",
            "target": target_url, 
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(target_url, timeout=10) as response:
                    results["response_code"] = response.status
                    results["response_time"] = "< 10s"
                    results["accessible"] = True
                    
        except Exception as e:
            results["accessible"] = False
            results["error"] = str(e)
        
        return results
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Return agent information for registration"""
        return {
            "id": self.agent_id,
            "name": "Test HTTP Scanner",
            "agent_type": self.agent_type,
            "capabilities": [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "parameters": cap.parameters
                } for cap in self.capabilities
            ],
            "status": self.status,
            "metadata": {
                "version": "2.0.0",
                "created_at": datetime.now().isoformat(),
                "max_concurrent_missions": 3
            }
        }


async def main():
    """Test agent functionality"""
    agent = TestAgent()
    
    # Test mission
    test_mission = {
        "mission_type": "http_scan",
        "target_url": "http://httpbin.org",
        "parameters": {}
    }
    
    print("🧪 Testing XORB Agent...")
    print(f"Agent Info: {json.dumps(agent.get_agent_info(), indent=2)}")
    
    print("\n🎯 Executing test mission...")
    result = await agent.execute_mission(test_mission)
    print(f"Mission Result: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())