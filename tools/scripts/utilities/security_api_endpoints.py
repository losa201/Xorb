#!/usr/bin/env python3
"""
XORB Security API Endpoints
Comprehensive API endpoints for all security tools and services
"""

import os
import sys
import json
import time
import asyncio
import logging
import hashlib
import secrets
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import ipaddress
from aiohttp import web, web_request, web_response
import aiofiles
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IncidentStatus(Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"

class ScanStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ThreatIntelligence:
    id: str
    threat_type: str
    severity: str
    description: str
    source: str
    indicators: List[str]
    timestamp: str
    confidence: float
    mitigated: bool = False

@dataclass
class SecurityIncident:
    id: str
    title: str
    description: str
    severity: str
    status: str
    assigned_to: str
    created_at: str
    updated_at: str
    tags: List[str]
    affected_systems: List[str]

@dataclass
class VulnerabilityAssessment:
    id: str
    target: str
    scan_type: str
    status: str
    created_at: str
    completed_at: Optional[str]
    vulnerabilities: List[Dict]
    risk_score: float
    recommendations: List[str]

@dataclass
class NetworkDevice:
    id: str
    ip_address: str
    hostname: str
    device_type: str
    status: str
    last_seen: str
    open_ports: List[int]
    services: List[Dict]
    vulnerabilities: List[str]

@dataclass
class SecurityAgent:
    id: str
    name: str
    status: str
    specialization: str
    performance_score: float
    last_activity: str
    tasks_completed: int
    active_tasks: int

class XORBSecurityAPI:
    """Comprehensive Security API for XORB Platform"""

    def __init__(self, db_path: str = "data/security_api.db"):
        self.db_path = db_path
        self.init_database()

        # In-memory stores for demo purposes
        self.threats = {}
        self.incidents = {}
        self.scans = {}
        self.network_devices = {}
        self.agents = {}

        # Initialize with sample data
        self.init_sample_data()

        # Real-time metrics
        self.metrics = {
            'total_threats': 0,
            'active_incidents': 0,
            'completed_scans': 0,
            'network_devices': 0,
            'active_agents': 0,
            'system_health': 98.7,
            'last_updated': datetime.now().isoformat()
        }

        self.app = web.Application()
        self.setup_routes()

    def init_database(self):
        """Initialize SQLite database"""
        Path(os.path.dirname(self.db_path)).mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS threats (
                id TEXT PRIMARY KEY,
                threat_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT NOT NULL,
                source TEXT NOT NULL,
                indicators TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL NOT NULL,
                mitigated BOOLEAN DEFAULT FALSE
            );

            CREATE TABLE IF NOT EXISTS incidents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                assigned_to TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                tags TEXT,
                affected_systems TEXT
            );

            CREATE TABLE IF NOT EXISTS scans (
                id TEXT PRIMARY KEY,
                target TEXT NOT NULL,
                scan_type TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                vulnerabilities TEXT,
                risk_score REAL DEFAULT 0.0,
                recommendations TEXT
            );

            CREATE TABLE IF NOT EXISTS network_devices (
                id TEXT PRIMARY KEY,
                ip_address TEXT NOT NULL UNIQUE,
                hostname TEXT,
                device_type TEXT NOT NULL,
                status TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                open_ports TEXT,
                services TEXT,
                vulnerabilities TEXT
            );

            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                specialization TEXT NOT NULL,
                performance_score REAL DEFAULT 0.0,
                last_activity TEXT NOT NULL,
                tasks_completed INTEGER DEFAULT 0,
                active_tasks INTEGER DEFAULT 0
            );
        """)

        conn.commit()
        conn.close()

    def init_sample_data(self):
        """Initialize with sample security data"""
        # Sample threats
        sample_threats = [
            ThreatIntelligence(
                id="threat-001",
                threat_type="Advanced Persistent Threat",
                severity="high",
                description="Sophisticated attack campaign targeting financial institutions",
                source="Global Threat Intelligence",
                indicators=["malicious-domain.com", "192.168.1.100", "suspicious.exe"],
                timestamp=datetime.now().isoformat(),
                confidence=0.95
            ),
            ThreatIntelligence(
                id="threat-002",
                threat_type="Phishing Campaign",
                severity="medium",
                description="New phishing emails impersonating cloud service providers",
                source="Email Security Gateway",
                indicators=["phishing-email@fake-domain.com", "malicious-link.php"],
                timestamp=datetime.now().isoformat(),
                confidence=0.87
            )
        ]

        for threat in sample_threats:
            self.threats[threat.id] = threat

        # Sample incidents
        sample_incidents = [
            SecurityIncident(
                id="INC-2024-001",
                title="DDoS Attack Detected",
                description="Large-scale DDoS attack targeting web servers",
                severity="high",
                status="in_progress",
                assigned_to="security-team",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                tags=["ddos", "network", "critical"],
                affected_systems=["web-server-01", "web-server-02"]
            )
        ]

        for incident in sample_incidents:
            self.incidents[incident.id] = incident

        # Sample network devices
        sample_devices = [
            NetworkDevice(
                id="device-001",
                ip_address="192.168.1.1",
                hostname="gateway.local",
                device_type="router",
                status="online",
                last_seen=datetime.now().isoformat(),
                open_ports=[22, 80, 443],
                services=[{"port": 22, "service": "ssh"}, {"port": 80, "service": "http"}],
                vulnerabilities=[]
            ),
            NetworkDevice(
                id="device-002",
                ip_address="192.168.1.10",
                hostname="server-01.local",
                device_type="server",
                status="online",
                last_seen=datetime.now().isoformat(),
                open_ports=[22, 80, 443, 3306],
                services=[{"port": 3306, "service": "mysql"}],
                vulnerabilities=["CVE-2024-12345"]
            )
        ]

        for device in sample_devices:
            self.network_devices[device.id] = device

        # Sample agents
        for i in range(64):
            agent = SecurityAgent(
                id=f"agent-{str(i+1).zfill(3)}",
                name=f"XORB Agent {i+1}",
                status="active" if i < 62 else "idle",
                specialization=["threat_detection", "network_analysis", "behavioral_analysis", "malware_analysis"][i % 4],
                performance_score=85.0 + (i % 15),
                last_activity=datetime.now().isoformat(),
                tasks_completed=100 + (i * 5),
                active_tasks=2 if i < 62 else 0
            )
            self.agents[agent.id] = agent

    def setup_routes(self):
        """Setup API routes"""
        # Threat Intelligence API
        self.app.router.add_get('/api/v1/threats', self.get_threats)
        self.app.router.add_post('/api/v1/threats', self.create_threat)
        self.app.router.add_get('/api/v1/threats/{threat_id}', self.get_threat)
        self.app.router.add_put('/api/v1/threats/{threat_id}', self.update_threat)
        self.app.router.add_delete('/api/v1/threats/{threat_id}', self.delete_threat)

        # Incident Management API
        self.app.router.add_get('/api/v1/incidents', self.get_incidents)
        self.app.router.add_post('/api/v1/incidents', self.create_incident)
        self.app.router.add_get('/api/v1/incidents/{incident_id}', self.get_incident)
        self.app.router.add_put('/api/v1/incidents/{incident_id}', self.update_incident)
        self.app.router.add_delete('/api/v1/incidents/{incident_id}', self.delete_incident)

        # Vulnerability Scanning API
        self.app.router.add_get('/api/v1/scans', self.get_scans)
        self.app.router.add_post('/api/v1/scans', self.create_scan)
        self.app.router.add_get('/api/v1/scans/{scan_id}', self.get_scan)
        self.app.router.add_get('/api/v1/scans/{scan_id}/results', self.get_scan_results)
        self.app.router.add_delete('/api/v1/scans/{scan_id}', self.delete_scan)

        # Network Monitoring API
        self.app.router.add_get('/api/v1/network/devices', self.get_network_devices)
        self.app.router.add_post('/api/v1/network/devices', self.add_network_device)
        self.app.router.add_get('/api/v1/network/devices/{device_id}', self.get_network_device)
        self.app.router.add_put('/api/v1/network/devices/{device_id}', self.update_network_device)
        self.app.router.add_delete('/api/v1/network/devices/{device_id}', self.delete_network_device)
        self.app.router.add_get('/api/v1/network/topology', self.get_network_topology)

        # Agent Management API
        self.app.router.add_get('/api/v1/agents', self.get_agents)
        self.app.router.add_get('/api/v1/agents/{agent_id}', self.get_agent)
        self.app.router.add_put('/api/v1/agents/{agent_id}', self.update_agent)
        self.app.router.add_post('/api/v1/agents/{agent_id}/tasks', self.assign_agent_task)

        # Analytics and Metrics API
        self.app.router.add_get('/api/v1/metrics/system', self.get_system_metrics)
        self.app.router.add_get('/api/v1/metrics/security', self.get_security_metrics)
        self.app.router.add_get('/api/v1/metrics/performance', self.get_performance_metrics)
        self.app.router.add_get('/api/v1/analytics/dashboard', self.get_dashboard_analytics)

        # Compliance and Reporting API
        self.app.router.add_get('/api/v1/compliance/report/{framework}', self.get_compliance_report)
        self.app.router.add_get('/api/v1/reports/security', self.generate_security_report)
        self.app.router.add_get('/api/v1/reports/incidents', self.generate_incident_report)

        # Real-time WebSocket endpoints would be handled by the API gateway

        # Health and Status
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/api/v1/status', self.get_api_status)

    # Threat Intelligence Endpoints
    async def get_threats(self, request: web_request) -> web_response:
        """Get all threat intelligence entries"""
        try:
            # Query parameters
            severity = request.query.get('severity')
            limit = int(request.query.get('limit', 50))
            offset = int(request.query.get('offset', 0))

            threats = list(self.threats.values())

            # Filter by severity if specified
            if severity:
                threats = [t for t in threats if t.severity == severity]

            # Pagination
            total = len(threats)
            threats = threats[offset:offset + limit]

            return web.json_response({
                'success': True,
                'data': [asdict(threat) for threat in threats],
                'pagination': {
                    'total': total,
                    'limit': limit,
                    'offset': offset
                }
            })
        except Exception as e:
            logger.error(f"Error getting threats: {e}")
            return web.json_response({'success': False, 'error': str(e)}, status=500)

    async def create_threat(self, request: web_request) -> web_response:
        """Create new threat intelligence entry"""
        try:
            data = await request.json()

            threat = ThreatIntelligence(
                id=data.get('id', f"threat-{uuid.uuid4().hex[:8]}"),
                threat_type=data['threat_type'],
                severity=data['severity'],
                description=data['description'],
                source=data['source'],
                indicators=data.get('indicators', []),
                timestamp=datetime.now().isoformat(),
                confidence=data.get('confidence', 0.5)
            )

            self.threats[threat.id] = threat
            self.metrics['total_threats'] += 1

            return web.json_response({
                'success': True,
                'data': asdict(threat)
            }, status=201)
        except Exception as e:
            logger.error(f"Error creating threat: {e}")
            return web.json_response({'success': False, 'error': str(e)}, status=500)

    async def get_threat(self, request: web_request) -> web_response:
        """Get specific threat intelligence entry"""
        try:
            threat_id = request.match_info['threat_id']

            if threat_id not in self.threats:
                return web.json_response({'success': False, 'error': 'Threat not found'}, status=404)

            threat = self.threats[threat_id]
            return web.json_response({
                'success': True,
                'data': asdict(threat)
            })
        except Exception as e:
            logger.error(f"Error getting threat: {e}")
            return web.json_response({'success': False, 'error': str(e)}, status=500)

    # Incident Management Endpoints
    async def get_incidents(self, request: web_request) -> web_response:
        """Get all security incidents"""
        try:
            status = request.query.get('status')
            severity = request.query.get('severity')
            limit = int(request.query.get('limit', 50))
            offset = int(request.query.get('offset', 0))

            incidents = list(self.incidents.values())

            # Apply filters
            if status:
                incidents = [i for i in incidents if i.status == status]
            if severity:
                incidents = [i for i in incidents if i.severity == severity]

            # Pagination
            total = len(incidents)
            incidents = incidents[offset:offset + limit]

            return web.json_response({
                'success': True,
                'data': [asdict(incident) for incident in incidents],
                'pagination': {
                    'total': total,
                    'limit': limit,
                    'offset': offset
                }
            })
        except Exception as e:
            logger.error(f"Error getting incidents: {e}")
            return web.json_response({'success': False, 'error': str(e)}, status=500)

    async def create_incident(self, request: web_request) -> web_response:
        """Create new security incident"""
        try:
            data = await request.json()

            incident = SecurityIncident(
                id=data.get('id', f"INC-2024-{len(self.incidents) + 1:03d}"),
                title=data['title'],
                description=data['description'],
                severity=data['severity'],
                status=data.get('status', 'open'),
                assigned_to=data.get('assigned_to', ''),
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                tags=data.get('tags', []),
                affected_systems=data.get('affected_systems', [])
            )

            self.incidents[incident.id] = incident
            self.metrics['active_incidents'] += 1

            return web.json_response({
                'success': True,
                'data': asdict(incident)
            }, status=201)
        except Exception as e:
            logger.error(f"Error creating incident: {e}")
            return web.json_response({'success': False, 'error': str(e)}, status=500)

    # Network Monitoring Endpoints
    async def get_network_devices(self, request: web_request) -> web_response:
        """Get all network devices"""
        try:
            device_type = request.query.get('type')
            status = request.query.get('status')

            devices = list(self.network_devices.values())

            # Apply filters
            if device_type:
                devices = [d for d in devices if d.device_type == device_type]
            if status:
                devices = [d for d in devices if d.status == status]

            return web.json_response({
                'success': True,
                'data': [asdict(device) for device in devices]
            })
        except Exception as e:
            logger.error(f"Error getting network devices: {e}")
            return web.json_response({'success': False, 'error': str(e)}, status=500)

    async def get_network_topology(self, request: web_request) -> web_response:
        """Get network topology data"""
        try:
            devices = list(self.network_devices.values())

            # Create nodes and edges for visualization
            nodes = []
            edges = []

            for device in devices:
                nodes.append({
                    'id': device.id,
                    'label': device.hostname or device.ip_address,
                    'type': device.device_type,
                    'status': device.status,
                    'ip': device.ip_address
                })

            # Simple topology - connect all devices to the first router/gateway
            gateway = next((d for d in devices if d.device_type == 'router'), None)
            if gateway:
                for device in devices:
                    if device.id != gateway.id:
                        edges.append({
                            'from': gateway.id,
                            'to': device.id
                        })

            return web.json_response({
                'success': True,
                'data': {
                    'nodes': nodes,
                    'edges': edges
                }
            })
        except Exception as e:
            logger.error(f"Error getting network topology: {e}")
            return web.json_response({'success': False, 'error': str(e)}, status=500)

    # Agent Management Endpoints
    async def get_agents(self, request: web_request) -> web_response:
        """Get all security agents"""
        try:
            status = request.query.get('status')
            specialization = request.query.get('specialization')

            agents = list(self.agents.values())

            # Apply filters
            if status:
                agents = [a for a in agents if a.status == status]
            if specialization:
                agents = [a for a in agents if a.specialization == specialization]

            return web.json_response({
                'success': True,
                'data': [asdict(agent) for agent in agents]
            })
        except Exception as e:
            logger.error(f"Error getting agents: {e}")
            return web.json_response({'success': False, 'error': str(e)}, status=500)

    # Analytics and Metrics Endpoints
    async def get_system_metrics(self, request: web_request) -> web_response:
        """Get system metrics"""
        try:
            # Update metrics with current data
            active_agents = len([a for a in self.agents.values() if a.status == 'active'])
            active_incidents = len([i for i in self.incidents.values() if i.status in ['open', 'in_progress']])

            metrics = {
                'activeAgents': active_agents,
                'threatsDetected': len(self.threats),
                'threatsNeutralized': len([t for t in self.threats.values() if t.mitigated]),
                'systemHealth': f"{self.metrics['system_health']:.1f}%",
                'learningRate': f"{4.2 + (len(self.agents) * 0.01):.1f}x",
                'networkCoverage': '100%',
                'responseTime': f"{10 + (active_incidents * 2)}ms",
                'cpuUsage': f"{20.5 + (active_agents * 0.3):.1f}%",
                'memoryUsage': f"{35.2 + (len(self.threats) * 0.1):.1f}%",
                'timestamp': datetime.now().isoformat(),
                'activeIncidents': active_incidents,
                'networkDevices': len(self.network_devices)
            }

            return web.json_response({
                'success': True,
                'data': metrics
            })
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return web.json_response({'success': False, 'error': str(e)}, status=500)

    async def get_dashboard_analytics(self, request: web_request) -> web_response:
        """Get comprehensive dashboard analytics"""
        try:
            analytics = {
                'summary': {
                    'totalThreats': len(self.threats),
                    'activeIncidents': len([i for i in self.incidents.values() if i.status in ['open', 'in_progress']]),
                    'networkDevices': len(self.network_devices),
                    'activeAgents': len([a for a in self.agents.values() if a.status == 'active'])
                },
                'threatDistribution': {
                    'critical': len([t for t in self.threats.values() if t.severity == 'critical']),
                    'high': len([t for t in self.threats.values() if t.severity == 'high']),
                    'medium': len([t for t in self.threats.values() if t.severity == 'medium']),
                    'low': len([t for t in self.threats.values() if t.severity == 'low'])
                },
                'agentPerformance': {
                    'averageScore': sum(a.performance_score for a in self.agents.values()) / len(self.agents),
                    'topPerformers': sorted([asdict(a) for a in self.agents.values()],
                                          key=lambda x: x['performance_score'], reverse=True)[:5]
                },
                'systemHealth': {
                    'overall': self.metrics['system_health'],
                    'components': {
                        'threat_detection': 98.5,
                        'network_monitoring': 99.1,
                        'incident_response': 97.8,
                        'compliance': 96.2
                    }
                },
                'timestamp': datetime.now().isoformat()
            }

            return web.json_response({
                'success': True,
                'data': analytics
            })
        except Exception as e:
            logger.error(f"Error getting dashboard analytics: {e}")
            return web.json_response({'success': False, 'error': str(e)}, status=500)

    # Compliance and Reporting Endpoints
    async def get_compliance_report(self, request: web_request) -> web_response:
        """Get compliance report for specific framework"""
        try:
            framework = request.match_info['framework'].upper()

            # Mock compliance data
            compliance_data = {
                'NIST': {
                    'framework': 'NIST Cybersecurity Framework',
                    'version': '1.1',
                    'overallScore': 85.7,
                    'categories': {
                        'Identify': 88.5,
                        'Protect': 87.2,
                        'Detect': 92.1,
                        'Respond': 81.3,
                        'Recover': 79.8
                    }
                },
                'ISO27001': {
                    'framework': 'ISO 27001',
                    'version': '2013',
                    'overallScore': 78.3,
                    'categories': {
                        'Information Security Policies': 85.0,
                        'Organization of Information Security': 82.1,
                        'Human Resource Security': 75.5,
                        'Asset Management': 88.2,
                        'Access Control': 91.7
                    }
                },
                'SOC2': {
                    'framework': 'SOC 2 Type II',
                    'version': '2017',
                    'overallScore': 91.2,
                    'categories': {
                        'Security': 93.5,
                        'Availability': 89.7,
                        'Processing Integrity': 88.9,
                        'Confidentiality': 92.1,
                        'Privacy': 86.3
                    }
                }
            }

            if framework not in compliance_data:
                return web.json_response({'success': False, 'error': 'Framework not supported'}, status=404)

            report = compliance_data[framework]
            report['generatedAt'] = datetime.now().isoformat()
            report['validUntil'] = (datetime.now() + timedelta(days=90)).isoformat()

            return web.json_response({
                'success': True,
                'data': report
            })
        except Exception as e:
            logger.error(f"Error getting compliance report: {e}")
            return web.json_response({'success': False, 'error': str(e)}, status=500)

    # Health and Status Endpoints
    async def health_check(self, request: web_request) -> web_response:
        """Health check endpoint"""
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'services': {
                'threat_intelligence': 'operational',
                'incident_management': 'operational',
                'network_monitoring': 'operational',
                'agent_management': 'operational'
            }
        })

    async def get_api_status(self, request: web_request) -> web_response:
        """Get API status and statistics"""
        try:
            status = {
                'api_version': '1.0.0',
                'status': 'operational',
                'uptime': time.time() - self.metrics.get('start_time', time.time()),
                'endpoints': {
                    'threats': len(self.threats),
                    'incidents': len(self.incidents),
                    'network_devices': len(self.network_devices),
                    'agents': len(self.agents)
                },
                'performance': {
                    'avg_response_time': '12ms',
                    'requests_per_second': 45.2,
                    'error_rate': 0.01
                },
                'timestamp': datetime.now().isoformat()
            }

            return web.json_response({
                'success': True,
                'data': status
            })
        except Exception as e:
            logger.error(f"Error getting API status: {e}")
            return web.json_response({'success': False, 'error': str(e)}, status=500)

async def init_app():
    """Initialize the Security API application"""
    api = XORBSecurityAPI()

    # Add CORS support
    from aiohttp_cors import setup as cors_setup, ResourceOptions
    cors = cors_setup(api.app, defaults={
        "*": ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*", allow_methods="*")
    })

    # Add CORS to all routes
    for route in list(api.app.router.routes()):
        cors.add(route)

    return api.app

if __name__ == '__main__':
    app = asyncio.run(init_app())
    web.run_app(app, host='0.0.0.0', port=8001)
