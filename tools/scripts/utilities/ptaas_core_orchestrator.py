#!/usr/bin/env python3
"""
XORB PTaaS Core Orchestration Engine
Advanced penetration testing automation with AI-enhanced orchestration
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, asdict
import uuid
import aiohttp
import psycopg2
from psycopg2.extras import RealDictCursor
import redis.asyncio as redis

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="XORB PTaaS Core Orchestrator",
    description="Advanced penetration testing automation with AI-enhanced orchestration",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestPhase(Enum):
    RECONNAISSANCE = "reconnaissance"
    SCANNING = "scanning"
    ENUMERATION = "enumeration"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"
    EXPLOITATION = "exploitation"
    POST_EXPLOITATION = "post_exploitation"
    REPORTING = "reporting"

class TestStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentCapability(Enum):
    WEB_SCANNING = "web_scanning"
    NETWORK_SCANNING = "network_scanning"
    VULNERABILITY_SCANNING = "vulnerability_scanning"
    EXPLOITATION = "exploitation"
    RECONNAISSANCE = "reconnaissance"
    SOCIAL_ENGINEERING = "social_engineering"
    WIRELESS = "wireless"
    MOBILE = "mobile"
    API_TESTING = "api_testing"
    DATABASE_TESTING = "database_testing"

@dataclass
class TestTarget:
    target_id: str
    name: str
    type: str
    domains: List[str]
    ip_ranges: List[str]
    ports: List[int]
    excluded_targets: List[str]
    authorized_by: str
    authorization_date: datetime

@dataclass
class TestAgent:
    agent_id: str
    name: str
    capabilities: List[AgentCapability]
    status: str
    last_heartbeat: datetime
    current_task: Optional[str]
    performance_score: float
    resource_usage: Dict[str, Any]

@dataclass
class PenetrationTest:
    test_id: str
    name: str
    description: str
    target: TestTarget
    methodology: str
    status: TestStatus
    current_phase: TestPhase
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    estimated_duration: int
    actual_duration: Optional[int]
    findings: List[Dict[str, Any]]
    agents_assigned: List[str]
    progress_percentage: float
    client_id: str
    created_by: str

class TestRequest(BaseModel):
    name: str
    description: str
    target_domains: List[str]
    target_ips: List[str] = []
    target_ports: List[int] = []
    excluded_targets: List[str] = []
    methodology: str = "OWASP"
    estimated_duration: int = 48
    authorized_by: str
    client_id: str

class PTaaSOrchestrator:
    """Core PTaaS orchestration engine with AI-enhanced automation"""
    
    def __init__(self):
        self.active_tests: Dict[str, PenetrationTest] = {}
        self.available_agents: Dict[str, TestAgent] = {}
        self.test_queue: List[str] = []
        self.db_pool = None
        self.redis_client = None
        self.metrics = {
            "tests_completed": 0,
            "tests_running": 0,
            "total_findings": 0,
            "agent_utilization": 0.0,
            "average_test_duration": 0.0
        }
        
        # Initialize mock agents for demonstration
        self._initialize_mock_agents()
    
    def _initialize_mock_agents(self):
        """Initialize mock testing agents"""
        mock_agents = [
            {
                "agent_id": "AGENT-WEB-SCANNER-001",
                "name": "Advanced Web Scanner",
                "capabilities": [AgentCapability.WEB_SCANNING, AgentCapability.API_TESTING],
                "status": "available",
                "performance_score": 0.92
            },
            {
                "agent_id": "AGENT-NETWORK-RECON-002", 
                "name": "Network Reconnaissance Agent",
                "capabilities": [AgentCapability.NETWORK_SCANNING, AgentCapability.RECONNAISSANCE],
                "status": "available",
                "performance_score": 0.88
            },
            {
                "agent_id": "AGENT-VULN-ASSESSMENT-003",
                "name": "Vulnerability Assessment Engine",
                "capabilities": [AgentCapability.VULNERABILITY_SCANNING],
                "status": "available", 
                "performance_score": 0.95
            },
            {
                "agent_id": "AGENT-EXPLOITATION-004",
                "name": "Advanced Exploitation Framework",
                "capabilities": [AgentCapability.EXPLOITATION],
                "status": "available",
                "performance_score": 0.87
            },
            {
                "agent_id": "AGENT-DB-TESTER-005",
                "name": "Database Security Tester",
                "capabilities": [AgentCapability.DATABASE_TESTING],
                "status": "available",
                "performance_score": 0.90
            }
        ]
        
        for agent_data in mock_agents:
            agent = TestAgent(
                agent_id=agent_data["agent_id"],
                name=agent_data["name"],
                capabilities=agent_data["capabilities"],
                status=agent_data["status"],
                last_heartbeat=datetime.now(),
                current_task=None,
                performance_score=agent_data["performance_score"],
                resource_usage={"cpu": 0.1, "memory": 0.2, "network": 0.0}
            )
            self.available_agents[agent.agent_id] = agent
    
    async def initialize_connections(self):
        """Initialize database and Redis connections"""
        try:
            self.redis_client = redis.from_url("redis://localhost:6379")
            await self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
    
    def select_agents_for_phase(self, phase: TestPhase, target: TestTarget) -> List[TestAgent]:
        """AI-enhanced agent selection based on test phase and target characteristics"""
        phase_capability_mapping = {
            TestPhase.RECONNAISSANCE: [AgentCapability.RECONNAISSANCE, AgentCapability.NETWORK_SCANNING],
            TestPhase.SCANNING: [AgentCapability.NETWORK_SCANNING, AgentCapability.WEB_SCANNING],
            TestPhase.ENUMERATION: [AgentCapability.WEB_SCANNING, AgentCapability.API_TESTING],
            TestPhase.VULNERABILITY_ASSESSMENT: [AgentCapability.VULNERABILITY_SCANNING],
            TestPhase.EXPLOITATION: [AgentCapability.EXPLOITATION],
            TestPhase.POST_EXPLOITATION: [AgentCapability.EXPLOITATION],
            TestPhase.REPORTING: []
        }
        
        required_capabilities = phase_capability_mapping.get(phase, [])
        suitable_agents = []
        
        for agent in self.available_agents.values():
            if agent.status == "available":
                agent_capabilities = set(agent.capabilities)
                required_capabilities_set = set(required_capabilities)
                
                if required_capabilities_set.intersection(agent_capabilities):
                    suitable_agents.append(agent)
        
        # Sort by performance score (AI scoring based on historical performance)
        suitable_agents.sort(key=lambda x: x.performance_score, reverse=True)
        
        # Select top agents (limit based on phase complexity)
        max_agents = 3 if phase in [TestPhase.SCANNING, TestPhase.VULNERABILITY_ASSESSMENT] else 2
        return suitable_agents[:max_agents]
    
    async def create_penetration_test(self, request: TestRequest) -> str:
        """Create and queue a new penetration test"""
        test_id = f"PTAAS-{int(time.time())}-{str(uuid.uuid4())[:8].upper()}"
        
        target = TestTarget(
            target_id=f"TARGET-{test_id}",
            name=request.name,
            type="web_application",
            domains=request.target_domains,
            ip_ranges=request.target_ips,
            ports=request.target_ports if request.target_ports else [80, 443, 8080, 8443],
            excluded_targets=request.excluded_targets,
            authorized_by=request.authorized_by,
            authorization_date=datetime.now()
        )
        
        penetration_test = PenetrationTest(
            test_id=test_id,
            name=request.name,
            description=request.description,
            target=target,
            methodology=request.methodology,
            status=TestStatus.QUEUED,
            current_phase=TestPhase.RECONNAISSANCE,
            start_time=None,
            end_time=None,
            estimated_duration=request.estimated_duration,
            actual_duration=None,
            findings=[],
            agents_assigned=[],
            progress_percentage=0.0,
            client_id=request.client_id,
            created_by=request.authorized_by
        )
        
        self.active_tests[test_id] = penetration_test
        self.test_queue.append(test_id)
        
        # Store in Redis for persistence
        if self.redis_client:
            await self.redis_client.set(
                f"ptaas:test:{test_id}",
                json.dumps(asdict(penetration_test), default=str),
                ex=86400 * 7  # 7 days expiry
            )
        
        logger.info(f"Created penetration test: {test_id}")
        return test_id
    
    async def execute_test_phase(self, test_id: str, phase: TestPhase):
        """Execute a specific phase of the penetration test"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.active_tests[test_id]
        test.current_phase = phase
        test.status = TestStatus.RUNNING
        
        if not test.start_time:
            test.start_time = datetime.now()
        
        # Select appropriate agents for this phase
        selected_agents = self.select_agents_for_phase(phase, test.target)
        test.agents_assigned = [agent.agent_id for agent in selected_agents]
        
        logger.info(f"Executing {phase.value} phase for test {test_id} with agents: {test.agents_assigned}")
        
        # Simulate phase execution with realistic timing
        phase_duration = {
            TestPhase.RECONNAISSANCE: 30,
            TestPhase.SCANNING: 45,
            TestPhase.ENUMERATION: 60,
            TestPhase.VULNERABILITY_ASSESSMENT: 90,
            TestPhase.EXPLOITATION: 120,
            TestPhase.POST_EXPLOITATION: 60,
            TestPhase.REPORTING: 30
        }
        
        duration = phase_duration.get(phase, 60)
        
        # Update agent status
        for agent_id in test.agents_assigned:
            if agent_id in self.available_agents:
                self.available_agents[agent_id].status = "busy"
                self.available_agents[agent_id].current_task = f"{test_id}:{phase.value}"
        
        # Simulate phase execution
        await asyncio.sleep(5)  # Shortened for demo
        
        # Generate mock findings based on phase
        findings = self._generate_mock_findings(phase, test.target)
        test.findings.extend(findings)
        
        # Update progress
        phase_weights = {
            TestPhase.RECONNAISSANCE: 10,
            TestPhase.SCANNING: 15,
            TestPhase.ENUMERATION: 20,
            TestPhase.VULNERABILITY_ASSESSMENT: 25,
            TestPhase.EXPLOITATION: 20,
            TestPhase.POST_EXPLOITATION: 5,
            TestPhase.REPORTING: 5
        }
        
        current_progress = sum(phase_weights[p] for p in TestPhase if p.value <= phase.value)
        test.progress_percentage = min(current_progress, 100.0)
        
        # Release agents
        for agent_id in test.agents_assigned:
            if agent_id in self.available_agents:
                self.available_agents[agent_id].status = "available"
                self.available_agents[agent_id].current_task = None
        
        logger.info(f"Completed {phase.value} phase for test {test_id}. Found {len(findings)} issues.")
        
        # Store updated test state
        if self.redis_client:
            await self.redis_client.set(
                f"ptaas:test:{test_id}",
                json.dumps(asdict(test), default=str),
                ex=86400 * 7
            )
    
    def _generate_mock_findings(self, phase: TestPhase, target: TestTarget) -> List[Dict[str, Any]]:
        """Generate realistic mock findings based on test phase"""
        findings = []
        
        if phase == TestPhase.RECONNAISSANCE:
            findings.append({
                "finding_id": f"RECON-{int(time.time())}",
                "title": "Subdomain Discovery",
                "severity": "informational",
                "description": f"Discovered {len(target.domains) + 3} subdomains for target domain",
                "phase": phase.value,
                "timestamp": datetime.now().isoformat(),
                "details": {
                    "subdomains": [f"admin.{target.domains[0]}", f"api.{target.domains[0]}", f"staging.{target.domains[0]}"],
                    "method": "DNS enumeration and certificate transparency logs"
                }
            })
        
        elif phase == TestPhase.SCANNING:
            findings.append({
                "finding_id": f"SCAN-{int(time.time())}",
                "title": "Open Ports Identified",
                "severity": "informational",
                "description": "Multiple open ports discovered during network scanning",
                "phase": phase.value,
                "timestamp": datetime.now().isoformat(),
                "details": {
                    "open_ports": [22, 80, 443, 3306, 8080],
                    "services": ["SSH", "HTTP", "HTTPS", "MySQL", "HTTP-Alt"]
                }
            })
        
        elif phase == TestPhase.VULNERABILITY_ASSESSMENT:
            findings.extend([
                {
                    "finding_id": f"VULN-{int(time.time())}-1",
                    "title": "SQL Injection Vulnerability",
                    "severity": "high", 
                    "description": "SQL injection vulnerability detected in login form",
                    "phase": phase.value,
                    "timestamp": datetime.now().isoformat(),
                    "details": {
                        "location": f"{target.domains[0]}/login.php",
                        "parameter": "username",
                        "payload": "' OR '1'='1",
                        "cvss_score": 8.5
                    }
                },
                {
                    "finding_id": f"VULN-{int(time.time())}-2",
                    "title": "Cross-Site Scripting (XSS)",
                    "severity": "medium",
                    "description": "Reflected XSS vulnerability in search functionality",
                    "phase": phase.value,
                    "timestamp": datetime.now().isoformat(),
                    "details": {
                        "location": f"{target.domains[0]}/search",
                        "parameter": "q",
                        "payload": "<script>alert('XSS')</script>",
                        "cvss_score": 6.1
                    }
                }
            ])
        
        elif phase == TestPhase.EXPLOITATION:
            findings.append({
                "finding_id": f"EXPLOIT-{int(time.time())}",
                "title": "Successful SQL Injection Exploitation",
                "severity": "critical",
                "description": "Successfully exploited SQL injection to access database",
                "phase": phase.value,
                "timestamp": datetime.now().isoformat(),
                "details": {
                    "database_accessed": "user_database",
                    "records_extracted": 1250,
                    "sensitive_data": ["usernames", "password_hashes", "email_addresses"],
                    "impact": "Complete user database compromise"
                }
            })
        
        return findings
    
    async def run_full_penetration_test(self, test_id: str):
        """Execute a complete penetration test through all phases"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.active_tests[test_id]
        phases = list(TestPhase)
        
        try:
            for phase in phases:
                logger.info(f"Starting {phase.value} phase for test {test_id}")
                await self.execute_test_phase(test_id, phase)
                
                # Brief pause between phases
                await asyncio.sleep(2)
            
            # Mark test as completed
            test.status = TestStatus.COMPLETED
            test.end_time = datetime.now()
            test.actual_duration = int((test.end_time - test.start_time).total_seconds() / 3600)
            test.progress_percentage = 100.0
            
            # Update metrics
            self.metrics["tests_completed"] += 1
            self.metrics["total_findings"] += len(test.findings)
            
            logger.info(f"Penetration test {test_id} completed successfully with {len(test.findings)} findings")
            
        except Exception as e:
            test.status = TestStatus.FAILED
            test.end_time = datetime.now()
            logger.error(f"Penetration test {test_id} failed: {e}")
            raise
        
        finally:
            # Store final test state
            if self.redis_client:
                await self.redis_client.set(
                    f"ptaas:test:{test_id}",
                    json.dumps(asdict(test), default=str),
                    ex=86400 * 30  # 30 days for completed tests
                )

# Global orchestrator instance
orchestrator = PTaaSOrchestrator()

@app.on_event("startup")
async def startup_event():
    await orchestrator.initialize_connections()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ptaas_core_orchestrator",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "metrics": orchestrator.metrics,
        "active_tests": len(orchestrator.active_tests),
        "available_agents": len([a for a in orchestrator.available_agents.values() if a.status == "available"])
    }

@app.post("/api/v1/tests")
async def create_test(request: TestRequest, background_tasks: BackgroundTasks):
    """Create a new penetration test"""
    try:
        test_id = await orchestrator.create_penetration_test(request)
        
        # Start test execution in background
        background_tasks.add_task(orchestrator.run_full_penetration_test, test_id)
        
        return {
            "test_id": test_id,
            "status": "queued",
            "message": "Penetration test created and queued for execution",
            "estimated_completion": (datetime.now() + timedelta(hours=request.estimated_duration)).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to create test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/tests/{test_id}")
async def get_test_status(test_id: str):
    """Get status and details of a specific test"""
    if test_id not in orchestrator.active_tests:
        raise HTTPException(status_code=404, detail="Test not found")
    
    test = orchestrator.active_tests[test_id]
    return {
        "test_id": test_id,
        "name": test.name,
        "status": test.status.value,
        "current_phase": test.current_phase.value,
        "progress_percentage": test.progress_percentage,
        "start_time": test.start_time.isoformat() if test.start_time else None,
        "estimated_completion": (test.start_time + timedelta(hours=test.estimated_duration)).isoformat() if test.start_time else None,
        "findings_count": len(test.findings),
        "agents_assigned": test.agents_assigned,
        "findings": test.findings
    }

@app.get("/api/v1/tests")
async def list_tests():
    """List all penetration tests"""
    tests = []
    for test_id, test in orchestrator.active_tests.items():
        tests.append({
            "test_id": test_id,
            "name": test.name,
            "status": test.status.value,
            "current_phase": test.current_phase.value,
            "progress_percentage": test.progress_percentage,
            "findings_count": len(test.findings),
            "start_time": test.start_time.isoformat() if test.start_time else None
        })
    
    return {
        "tests": tests,
        "total_tests": len(tests),
        "metrics": orchestrator.metrics
    }

@app.get("/api/v1/agents")
async def list_agents():
    """List all available testing agents"""
    agents = []
    for agent_id, agent in orchestrator.available_agents.items():
        agents.append({
            "agent_id": agent_id,
            "name": agent.name,
            "capabilities": [cap.value for cap in agent.capabilities],
            "status": agent.status,
            "performance_score": agent.performance_score,
            "current_task": agent.current_task,
            "last_heartbeat": agent.last_heartbeat.isoformat()
        })
    
    return {
        "agents": agents,
        "total_agents": len(agents),
        "available_agents": len([a for a in agents if a["status"] == "available"]),
        "busy_agents": len([a for a in agents if a["status"] == "busy"])
    }

@app.get("/api/v1/dashboard")
async def get_dashboard_data():
    """Get dashboard overview data"""
    active_tests = [t for t in orchestrator.active_tests.values() if t.status in [TestStatus.RUNNING, TestStatus.QUEUED]]
    completed_tests = [t for t in orchestrator.active_tests.values() if t.status == TestStatus.COMPLETED]
    
    # Calculate severity distribution
    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "informational": 0}
    for test in orchestrator.active_tests.values():
        for finding in test.findings:
            severity = finding.get("severity", "informational")
            if severity in severity_counts:
                severity_counts[severity] += 1
    
    return {
        "overview": {
            "active_tests": len(active_tests),
            "completed_tests": len(completed_tests),
            "total_findings": orchestrator.metrics["total_findings"],
            "available_agents": len([a for a in orchestrator.available_agents.values() if a.status == "available"])
        },
        "recent_tests": [
            {
                "test_id": t.test_id,
                "name": t.name,
                "status": t.status.value,
                "progress": t.progress_percentage,
                "findings": len(t.findings)
            } for t in list(orchestrator.active_tests.values())[-5:]
        ],
        "severity_distribution": severity_counts,
        "agent_status": {
            "available": len([a for a in orchestrator.available_agents.values() if a.status == "available"]),
            "busy": len([a for a in orchestrator.available_agents.values() if a.status == "busy"]),
            "offline": len([a for a in orchestrator.available_agents.values() if a.status == "offline"])
        }
    }

@app.websocket("/ws/tests/{test_id}")
async def test_websocket(websocket: WebSocket, test_id: str):
    """WebSocket endpoint for real-time test updates"""
    await websocket.accept()
    
    try:
        while True:
            if test_id in orchestrator.active_tests:
                test = orchestrator.active_tests[test_id]
                await websocket.send_json({
                    "test_id": test_id,
                    "status": test.status.value,
                    "current_phase": test.current_phase.value,
                    "progress": test.progress_percentage,
                    "findings_count": len(test.findings),
                    "timestamp": datetime.now().isoformat()
                })
            
            await asyncio.sleep(5)  # Send updates every 5 seconds
            
    except Exception as e:
        logger.error(f"WebSocket error for test {test_id}: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(
        "ptaas_core_orchestrator:app",
        host="0.0.0.0",
        port=8084,
        reload=False,
        log_level="info"
    )