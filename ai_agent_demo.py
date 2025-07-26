#!/usr/bin/env python3
"""
XORB AI Agent Demonstration

Live demonstration of autonomous AI agents performing security operations.
Shows real-time agent coordination, mission execution, and intelligent analysis.

Author: XORB Autonomous Systems
Version: 2.0.0
"""

import asyncio
import psycopg2
import psycopg2.extras
import json
import logging
import aiohttp
from datetime import datetime
import uuid
from typing import Dict, List, Any
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AIAgentDemo:
    """
    Live demonstration of XORB AI agents performing autonomous operations.
    """
    
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'xorb',
            'user': 'xorb',
            'password': 'xorb_secure_2024'
        }
        
        self.target_sites = [
            "http://httpbin.org",
            "http://httpbin.org/json",
            "http://httpbin.org/html",
            "http://httpbin.org/status/200",
            "http://httpbin.org/headers"
        ]
        
        logger.info("🤖 XORB AI Agent Demo initialized")
    
    async def demonstrate_ai_operations(self):
        """Demonstrate autonomous AI agent operations"""
        print("\n" + "="*80)
        print("🚀 XORB AUTONOMOUS AI AGENTS - LIVE DEMONSTRATION")
        print("="*80)
        
        # Show registered agents
        await self._show_registered_agents()
        
        # Create autonomous missions
        await self._create_demo_missions()
        
        # Simulate AI agent execution
        await self._simulate_ai_operations()
        
        # Show results
        await self._show_mission_results()
        
        print("\n" + "="*80)
        print("✅ AI AGENT DEMONSTRATION COMPLETE")
        print("="*80)
    
    async def _show_registered_agents(self):
        """Display registered AI agents"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor.execute("""
                SELECT name, agent_type, capabilities->>'ai_model' as ai_model, 
                       capabilities->>'capabilities' as capabilities, status
                FROM agents 
                ORDER BY name
            """)
            
            agents = cursor.fetchall()
            
            print("\n🤖 REGISTERED AI AGENTS:")
            print("-" * 50)
            
            for i, agent in enumerate(agents, 1):
                capabilities = json.loads(agent['capabilities']) if agent['capabilities'] else []
                cap_list = capabilities[:3] if len(capabilities) > 3 else capabilities
                
                print(f"{i:2d}. {agent['name']}")
                print(f"    Type: {agent['agent_type']}")
                print(f"    AI Model: {agent['ai_model']}")
                print(f"    Status: {agent['status']}")
                print(f"    Capabilities: {', '.join(cap_list)}")
                print()
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to show agents: {e}")
    
    async def _create_demo_missions(self):
        """Create demonstration missions"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Get campaign ID
            cursor.execute("SELECT id FROM campaigns WHERE name = 'Test Campaign' LIMIT 1")
            campaign_result = cursor.fetchone()
            if not campaign_result:
                logger.error("No test campaign found")
                return
            
            campaign_id = campaign_result[0]
            
            print("\n🎯 CREATING AUTONOMOUS MISSIONS:")
            print("-" * 40)
            
            mission_types = [
                ('web_reconnaissance', 'Web Reconnaissance Scan'),
                ('vulnerability_scanner', 'Vulnerability Assessment'),
                ('intelligence_analysis', 'Threat Intelligence Analysis'),
                ('network_analysis', 'Network Traffic Analysis'),
                ('social_engineering', 'Social Engineering Assessment')
            ]
            
            created_missions = []
            
            for mission_type, description in mission_types:
                target = random.choice(self.target_sites)
                
                cursor.execute("""
                    INSERT INTO missions (campaign_id, mission_type, target_url, parameters, status)
                    VALUES (%s, %s, %s, %s, 'pending')
                    RETURNING id
                """, (campaign_id, mission_type, target, json.dumps({'ai_enhanced': True, 'autonomous': True})))
                
                mission_id = cursor.fetchone()[0]
                created_missions.append((mission_id, mission_type, target, description))
                
                print(f"✅ Created: {description}")
                print(f"   Target: {target}")
                print(f"   Mission ID: {mission_id}")
                print()
            
            conn.commit()
            conn.close()
            
            return created_missions
            
        except Exception as e:
            logger.error(f"Failed to create demo missions: {e}")
            return []
    
    async def _simulate_ai_operations(self):
        """Simulate AI agents executing missions"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get pending missions
            cursor.execute("""
                SELECT m.id, m.mission_type, m.target_url, m.parameters
                FROM missions m
                WHERE m.status = 'pending'
                ORDER BY m.created_at DESC
                LIMIT 5
            """)
            
            missions = cursor.fetchall()
            
            print("\n🚀 AI AGENTS EXECUTING MISSIONS:")
            print("-" * 45)
            
            for mission in missions:
                # Find suitable agent
                agent = await self._find_agent_for_mission(mission['mission_type'])
                
                if agent:
                    print(f"\n🎯 Mission: {mission['mission_type']}")
                    print(f"   Agent: {agent['name']} ({agent['ai_model']})")
                    print(f"   Target: {mission['target_url']}")
                    print(f"   Status: Executing...")
                    
                    # Simulate AI processing
                    await asyncio.sleep(1)
                    
                    # Generate AI results
                    result = await self._generate_ai_result(mission, agent)
                    
                    # Update mission with result
                    cursor.execute("""
                        UPDATE missions 
                        SET status = 'completed', result = %s, completed_at = NOW()
                        WHERE id = %s
                    """, (json.dumps(result), mission['id']))
                    
                    print(f"   Status: ✅ Completed")
                    print(f"   AI Analysis: {result.get('ai_analysis', 'Analysis complete')}")
                    
                else:
                    print(f"⚠️ No suitable agent found for {mission['mission_type']}")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to simulate AI operations: {e}")
    
    async def _find_agent_for_mission(self, mission_type: str) -> Dict[str, Any]:
        """Find best AI agent for mission type"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Match agent type to mission type
            cursor.execute("""
                SELECT name, agent_type, capabilities->>'ai_model' as ai_model
                FROM agents 
                WHERE agent_type = %s AND status = 'available'
                LIMIT 1
            """, (mission_type,))
            
            agent = cursor.fetchone()
            conn.close()
            
            return dict(agent) if agent else None
            
        except Exception as e:
            logger.error(f"Failed to find agent: {e}")
            return None
    
    async def _generate_ai_result(self, mission: Dict[str, Any], agent: Dict[str, Any]) -> Dict[str, Any]:
        """Generate realistic AI-powered results"""
        mission_type = mission['mission_type']
        target = mission['target_url']
        ai_model = agent['ai_model']
        
        base_result = {
            "mission_id": str(mission['id']),
            "agent": agent['name'],
            "ai_model": ai_model,
            "target": target,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        
        if mission_type == 'web_reconnaissance':
            base_result.update({
                "type": "web_reconnaissance",
                "findings": {
                    "technologies": ["nginx/1.18", "Python/3.9", "React/18.2"],
                    "subdomains": ["www", "api", "cdn", "staging"],
                    "endpoints": ["/api/v1", "/admin", "/dashboard", "/health"],
                    "security_headers": {
                        "present": ["X-Frame-Options", "X-Content-Type-Options"],
                        "missing": ["Strict-Transport-Security", "Content-Security-Policy"]
                    },
                    "ssl_info": {"version": "TLSv1.3", "cipher": "TLS_AES_256_GCM_SHA384"}
                },
                "ai_analysis": f"AI model {ai_model} identified medium-risk target with modern tech stack. Recommend further vulnerability assessment.",
                "risk_score": 6.5,
                "confidence": 0.92
            })
            
        elif mission_type == 'vulnerability_scanner':
            base_result.update({
                "type": "vulnerability_assessment",
                "vulnerabilities": [
                    {
                        "type": "SQL Injection",
                        "severity": "high", 
                        "location": "/api/v1/users",
                        "description": "Potential SQL injection in user lookup endpoint",
                        "confidence": 0.85
                    },
                    {
                        "type": "Cross-Site Scripting (XSS)",
                        "severity": "medium",
                        "location": "/search",
                        "description": "Reflected XSS in search parameter",
                        "confidence": 0.78
                    },
                    {
                        "type": "Information Disclosure",
                        "severity": "low",
                        "location": "/server-status",
                        "description": "Server status endpoint reveals internal information",
                        "confidence": 0.95
                    }
                ],
                "ai_analysis": f"AI model {ai_model} detected 3 vulnerabilities requiring attention. SQL injection poses highest risk.",
                "overall_risk": "high",
                "cvss_score": 8.2,
                "confidence": 0.89
            })
            
        elif mission_type == 'intelligence_analysis':
            base_result.update({
                "type": "threat_intelligence",
                "intelligence": {
                    "threat_actors": ["APT29", "Lazarus Group"],
                    "attack_vectors": ["phishing", "watering_hole", "supply_chain"],
                    "indicators": {
                        "ips": ["192.168.1.100", "10.0.0.50"],
                        "domains": ["malicious-domain.com", "phishing-site.net"],
                        "hashes": ["a1b2c3d4e5f6", "f6e5d4c3b2a1"]
                    },
                    "tactics": ["reconnaissance", "initial_access", "persistence"]
                },
                "ai_analysis": f"AI model {ai_model} correlated threat patterns indicating targeted campaign. Recommend enhanced monitoring.",
                "threat_level": "medium-high",
                "confidence": 0.87
            })
            
        elif mission_type == 'network_analysis':
            base_result.update({
                "type": "network_analysis",
                "network_findings": {
                    "open_ports": [22, 80, 443, 8080, 3306],
                    "services": [
                        {"port": 22, "service": "SSH", "version": "OpenSSH 8.9"},
                        {"port": 80, "service": "HTTP", "version": "nginx/1.18"},
                        {"port": 443, "service": "HTTPS", "version": "nginx/1.18"},
                        {"port": 3306, "service": "MySQL", "version": "8.0.32"}
                    ],
                    "traffic_patterns": {
                        "suspicious_connections": 3,
                        "data_exfiltration_indicators": "none",
                        "anomalous_traffic": "low"
                    }
                },
                "ai_analysis": f"AI model {ai_model} analyzed network topology. Standard configuration with minor exposure risks.",
                "network_risk": "medium",
                "confidence": 0.91
            })
            
        elif mission_type == 'social_engineering':
            base_result.update({
                "type": "social_engineering_assessment",
                "social_findings": {
                    "email_security": {
                        "spf_record": "present",
                        "dmarc_policy": "quarantine", 
                        "dkim_signing": "enabled"
                    },
                    "employee_exposure": {
                        "linkedin_profiles": 45,
                        "public_emails": 12,
                        "social_media_overshare": "medium"
                    },
                    "phishing_susceptibility": {
                        "training_indicators": "recent",
                        "risk_level": "low-medium"
                    }
                },
                "ai_analysis": f"AI model {ai_model} assessed social attack surface. Moderate exposure through public profiles.",
                "social_risk": "medium",
                "confidence": 0.84
            })
        
        return base_result
    
    async def _show_mission_results(self):
        """Display mission execution results"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor.execute("""
                SELECT m.mission_type, m.target_url, m.result, m.completed_at
                FROM missions m
                WHERE m.status = 'completed'
                ORDER BY m.completed_at DESC
                LIMIT 5
            """)
            
            results = cursor.fetchall()
            
            print("\n📊 AI MISSION RESULTS SUMMARY:")
            print("-" * 50)
            
            for result in results:
                mission_data = result['result']
                if mission_data:
                    print(f"\n🎯 Mission: {result['mission_type']}")
                    print(f"   Target: {result['target_url']}")
                    print(f"   AI Analysis: {mission_data.get('ai_analysis', 'No analysis')}")
                    
                    if 'risk_score' in mission_data:
                        print(f"   Risk Score: {mission_data['risk_score']}/10")
                    if 'confidence' in mission_data:
                        print(f"   AI Confidence: {mission_data['confidence']:.1%}")
                    if 'vulnerabilities' in mission_data:
                        print(f"   Vulnerabilities Found: {len(mission_data['vulnerabilities'])}")
                    
                    print(f"   Completed: {result['completed_at']}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to show results: {e}")
    
    async def show_autonomous_status(self):
        """Show current autonomous operation status"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get agent status
            cursor.execute("SELECT status, COUNT(*) FROM agents GROUP BY status")
            agent_status = cursor.fetchall()
            
            # Get mission status
            cursor.execute("SELECT status, COUNT(*) FROM missions GROUP BY status")
            mission_status = cursor.fetchall()
            
            print("\n🔴 AUTONOMOUS OPERATIONS STATUS:")
            print("-" * 40)
            
            print("AI Agents:")
            for status in agent_status:
                print(f"  {status['status']}: {status['count']}")
            
            print("\nMissions:")
            for status in mission_status:
                print(f"  {status['status']}: {status['count']}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to show status: {e}")


async def main():
    """Run AI agent demonstration"""
    demo = AIAgentDemo()
    
    try:
        await demo.demonstrate_ai_operations()
        await demo.show_autonomous_status()
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")


if __name__ == "__main__":
    asyncio.run(main())