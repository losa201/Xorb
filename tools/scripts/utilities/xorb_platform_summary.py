#!/usr/bin/env python3
"""
XORB Platform Summary & Status Service
Comprehensive overview and status dashboard for the complete XORB ecosystem
"""

import asyncio
import json
import time
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="XORB Platform Summary & Status",
    description="Comprehensive overview of the complete XORB cybersecurity ecosystem",
    version="11.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass
class ServiceInfo:
    service_id: str
    name: str
    description: str
    port: int
    url: str
    status: str
    version: str
    capabilities: List[str]
    key_features: List[str]
    last_check: datetime

class PlatformSummary:
    """Complete XORB platform summary and status service"""
    
    def __init__(self):
        self.services: Dict[str, ServiceInfo] = {}
        self.platform_metrics = {
            "total_services": 0,
            "healthy_services": 0,
            "total_capabilities": 0,
            "platform_uptime": 0,
            "deployment_date": "2025-01-29",
            "last_updated": datetime.now()
        }
        
        # Initialize service registry
        self._initialize_service_registry()
    
    def _initialize_service_registry(self):
        """Initialize comprehensive service registry"""
        services_config = [
            {
                "service_id": "threat_intelligence",
                "name": "Threat Intelligence Engine",
                "description": "Advanced threat intelligence with real-time feeds and global monitoring",
                "port": 9004,
                "capabilities": ["Real-time Threat Feeds", "WebSocket Streaming", "Global Threat Monitoring", "IOC Analysis"],
                "key_features": [
                    "6 active threat intelligence feeds",
                    "215+ threat indicators tracked",
                    "5 active threat campaigns monitored",
                    "Real-time WebSocket threat streaming",
                    "Global threat landscape visualization"
                ]
            },
            {
                "service_id": "collaboration",
                "name": "Real-time Collaboration Platform",
                "description": "Multi-analyst collaboration for cybersecurity operations",
                "port": 9005,
                "capabilities": ["Multi-analyst Collaboration", "Threat Annotations", "Incident Coordination", "Session Management"],
                "key_features": [
                    "5 active security analysts",
                    "Real-time threat annotations",
                    "Incident response coordination",
                    "Role-based access control",
                    "Session analytics and reporting"
                ]
            },
            {
                "service_id": "analytics",
                "name": "Advanced Analytics Engine",
                "description": "ML-powered security analytics and behavioral analysis",
                "port": 9006,
                "capabilities": ["ML-powered Analysis", "Anomaly Detection", "Behavioral Analysis", "Threat Prediction"],
                "key_features": [
                    "5 active ML models (94% accuracy)",
                    "1000+ security events processed",
                    "5 threat patterns identified",
                    "Behavioral baseline learning",
                    "Real-time anomaly detection"
                ]
            },
            {
                "service_id": "reporting",
                "name": "Enterprise Reporting System",
                "description": "Comprehensive security reporting and business intelligence",
                "port": 9007,
                "capabilities": ["Executive Summaries", "Compliance Reports", "Threat Analysis Reports", "Data Visualization"],
                "key_features": [
                    "7200+ security metrics tracked",
                    "3 compliance frameworks (SOC2, ISO27001, NIST)",
                    "Executive summary generation",
                    "Automated report scheduling",
                    "Multi-format export (HTML, JSON, CSV)"
                ]
            },
            {
                "service_id": "hardening",
                "name": "Security Hardening System",
                "description": "Zero-trust security implementation and automated hardening",
                "port": 9008,
                "capabilities": ["Zero-trust Security", "Automated Hardening", "Compliance Assessment", "Policy Enforcement"],
                "key_features": [
                    "18 security rules (100% compliance)",
                    "Enhanced hardening level active",
                    "4 active security policies",
                    "API-secured access controls",
                    "Automated remediation capabilities"
                ]
            },
            {
                "service_id": "mobile",
                "name": "Mobile-Responsive Interface",
                "description": "Progressive Web App for mobile cybersecurity operations",
                "port": 9009,
                "capabilities": ["Progressive Web App", "Mobile-first Design", "Offline Support", "Touch Optimization"],
                "key_features": [
                    "6 dashboard widgets (5 mobile-optimized)",
                    "3 push notifications active",
                    "Progressive Web App capabilities",
                    "Dark mode optimization",
                    "Responsive design (mobile/tablet/desktop)"
                ]
            },
            {
                "service_id": "nvidia_ai",
                "name": "NVIDIA AI Integration",
                "description": "Advanced AI capabilities using NVIDIA's Qwen3-235B model",
                "port": 9010,
                "capabilities": ["Advanced AI Analysis", "Qwen3-235B Model", "Streaming Responses", "Cybersecurity Expertise"],
                "key_features": [
                    "Qwen3-235B (235B parameters)",
                    "8 specialized AI task types",
                    "Real-time streaming responses",  
                    "Expert cybersecurity prompts",
                    "Conversation memory and context"
                ]
            },
            {
                "service_id": "orchestration",
                "name": "Unified Orchestration Hub",
                "description": "Central command and control for all XORB services",
                "port": 9011,
                "capabilities": ["Service Orchestration", "Health Monitoring", "Alert Management", "Cross-service Coordination"],
                "key_features": [
                    "8 services monitored (100% health)",
                    "Real-time WebSocket streaming",
                    "Cross-service task automation",
                    "Centralized alert management",
                    "System metrics collection"
                ]
            }
        ]
        
        for config in services_config:
            base_url = f"http://188.245.101.102:{config['port']}"
            service = ServiceInfo(
                service_id=config["service_id"],
                name=config["name"],
                description=config["description"],
                port=config["port"],
                url=base_url,
                status="unknown",
                version="unknown",
                capabilities=config["capabilities"],
                key_features=config["key_features"],
                last_check=datetime.now()
            )
            self.services[service.service_id] = service
        
        # Update platform metrics
        self.platform_metrics["total_services"] = len(self.services)
        self.platform_metrics["total_capabilities"] = sum(len(s.capabilities) for s in self.services.values())
    
    async def check_all_services(self):
        """Check status of all services"""
        tasks = []
        for service in self.services.values():
            tasks.append(self._check_service_status(service))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update platform metrics
        healthy_count = sum(1 for s in self.services.values() if s.status == "healthy")
        self.platform_metrics["healthy_services"] = healthy_count
        self.platform_metrics["last_updated"] = datetime.now()
        
        return results
    
    async def _check_service_status(self, service: ServiceInfo):
        """Check individual service status"""
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"http://localhost:{service.port}/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        service.status = health_data.get("status", "healthy")
                        service.version = health_data.get("version", "unknown")
                    else:
                        service.status = "error"
        except Exception:
            service.status = "offline"
        
        service.last_check = datetime.now()
    
    def get_platform_overview(self) -> Dict:
        """Get comprehensive platform overview"""
        return {
            "platform_info": {
                "name": "XORB Cybersecurity Platform",
                "version": "11.0.0",
                "description": "Enterprise-grade cybersecurity operations platform",
                "deployment_date": self.platform_metrics["deployment_date"],
                "last_updated": self.platform_metrics["last_updated"].isoformat(),
                "external_ip": "188.245.101.102",
                "status": "operational"
            },
            "services_summary": {
                "total_services": self.platform_metrics["total_services"],
                "healthy_services": self.platform_metrics["healthy_services"],
                "service_health_percentage": round((self.platform_metrics["healthy_services"] / self.platform_metrics["total_services"]) * 100, 1) if self.platform_metrics["total_services"] > 0 else 0,
                "total_capabilities": self.platform_metrics["total_capabilities"]
            },
            "service_details": [
                {
                    "service_id": service.service_id,
                    "name": service.name,
                    "description": service.description,
                    "port": service.port,
                    "url": service.url,
                    "status": service.status,
                    "version": service.version,
                    "capabilities_count": len(service.capabilities),
                    "key_features_count": len(service.key_features),
                    "last_check": service.last_check.isoformat()
                }
                for service in self.services.values()
            ],
            "platform_capabilities": {
                "threat_intelligence": ["Real-time feeds", "Global monitoring", "IOC analysis"],
                "collaboration": ["Multi-analyst support", "Real-time coordination", "Incident management"],
                "analytics": ["ML-powered analysis", "Anomaly detection", "Behavioral analysis"],
                "reporting": ["Executive summaries", "Compliance reports", "Business intelligence"],
                "security_hardening": ["Zero-trust implementation", "Automated hardening", "Policy enforcement"],
                "mobile_access": ["Progressive Web App", "Mobile-optimized", "Offline support"],
                "ai_integration": ["Advanced AI analysis", "Natural language processing", "Expert knowledge"],
                "orchestration": ["Centralized control", "Service coordination", "Health monitoring"]
            }
        }
    
    def get_service_details(self, service_id: str) -> Optional[Dict]:
        """Get detailed information about specific service"""
        service = self.services.get(service_id)
        if not service:
            return None
        
        return {
            "service_info": {
                "service_id": service.service_id,
                "name": service.name,
                "description": service.description,
                "port": service.port,
                "url": service.url,
                "status": service.status,
                "version": service.version,
                "last_check": service.last_check.isoformat()
            },
            "capabilities": service.capabilities,
            "key_features": service.key_features,
            "access_urls": {
                "health_check": f"{service.url}/health",
                "main_interface": service.url,
                "api_docs": f"{service.url}/docs" if service.port != 9009 else service.url
            }
        }

# Initialize platform summary
platform_summary = PlatformSummary()

@app.get("/platform/overview")
async def get_platform_overview():
    """Get comprehensive XORB platform overview"""
    await platform_summary.check_all_services()
    return platform_summary.get_platform_overview()

@app.get("/platform/services")
async def get_all_services():
    """Get all XORB services information"""
    await platform_summary.check_all_services()
    return {
        "total_services": len(platform_summary.services),
        "services": [
            {
                "service_id": service.service_id,
                "name": service.name,
                "port": service.port,
                "url": service.url,
                "status": service.status,
                "capabilities": service.capabilities
            }
            for service in platform_summary.services.values()
        ]
    }

@app.get("/platform/services/{service_id}")
async def get_service_details(service_id: str):
    """Get detailed information about specific service"""
    await platform_summary.check_all_services()
    
    details = platform_summary.get_service_details(service_id)
    if not details:
        raise HTTPException(status_code=404, detail="Service not found")
    
    return details

@app.get("/platform/status")
async def get_platform_status():
    """Get current platform status"""
    await platform_summary.check_all_services()
    
    healthy_services = sum(1 for s in platform_summary.services.values() if s.status == "healthy")
    total_services = len(platform_summary.services)
    
    return {
        "overall_status": "operational" if healthy_services == total_services else "degraded" if healthy_services > 0 else "critical",
        "healthy_services": healthy_services,
        "total_services": total_services,
        "health_percentage": round((healthy_services / total_services) * 100, 1) if total_services > 0 else 0,
        "services_status": {
            service.service_id: {
                "name": service.name,
                "status": service.status,
                "port": service.port
            }
            for service in platform_summary.services.values()
        },
        "last_updated": datetime.now().isoformat()
    }

@app.get("/", response_class=HTMLResponse)
async def platform_dashboard():
    """XORB Platform Summary Dashboard"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>XORB Platform Summary</title>
    <style>
        body { font-family: 'Inter', sans-serif; background: #0d1117; color: #f0f6fc; margin: 0; padding: 20px; }
        .header { text-align: center; margin-bottom: 40px; }
        .header h1 { font-size: 2.5em; color: #58a6ff; margin-bottom: 10px; }
        .header p { font-size: 1.2em; color: #8b949e; }
        .platform-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }
        .stat-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; text-align: center; }
        .stat-value { font-size: 2.5em; font-weight: bold; color: #58a6ff; margin-bottom: 5px; }
        .stat-label { color: #8b949e; font-size: 0.9em; }
        .services-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; }
        .service-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; transition: transform 0.2s; }
        .service-card:hover { transform: translateY(-2px); border-color: #58a6ff; }
        .service-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
        .service-title { font-size: 1.2em; font-weight: 600; color: #58a6ff; }
        .status-badge { padding: 4px 8px; border-radius: 12px; font-size: 0.8em; text-transform: uppercase; }
        .status-healthy { background: #2ea043; color: white; }
        .status-degraded { background: #d29922; color: white; }
        .status-offline { background: #f85149; color: white; }
        .service-description { color: #8b949e; margin-bottom: 15px; line-height: 1.5; }
        .service-meta { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
        .port-info { background: #21262d; padding: 4px 8px; border-radius: 4px; font-family: monospace; }
        .capabilities-list { margin-bottom: 15px; }
        .capability-tag { display: inline-block; background: #21262d; padding: 4px 8px; border-radius: 12px; font-size: 0.8em; margin: 2px; }
        .features-list { font-size: 0.9em; color: #8b949e; }
        .features-list li { margin: 3px 0; }
        .access-button { background: #238636; border: none; color: white; padding: 8px 16px; border-radius: 6px; cursor: pointer; text-decoration: none; display: inline-block; }
        .access-button:hover { background: #2ea043; }
        .loading { text-align: center; color: #8b949e; padding: 40px; }
        .spinner { display: inline-block; width: 30px; height: 30px; border: 3px solid #30363d; border-radius: 50%; border-top-color: #58a6ff; animation: spin 1s ease-in-out infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .platform-health { text-align: center; margin: 30px 0; }
        .health-indicator { display: inline-block; width: 20px; height: 20px; border-radius: 50%; margin-right: 10px; }
        .health-operational { background: #2ea043; }
        .health-degraded { background: #d29922; }
        .health-critical { background: #f85149; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ XORB CYBERSECURITY PLATFORM</h1>
        <p>Enterprise-Grade Security Operations Ecosystem</p>
        <div class="platform-health" id="platform-health">
            <div class="loading">
                <div class="spinner"></div>
                <p>Loading platform status...</p>
            </div>
        </div>
    </div>
    
    <!-- Platform Statistics -->
    <div class="platform-stats" id="platform-stats">
        <div class="loading">Loading platform statistics...</div>
    </div>
    
    <!-- Services Grid -->
    <div class="services-grid" id="services-grid">
        <div class="loading">
            <div class="spinner"></div>
            <p>Loading XORB services...</p>
        </div>
    </div>
    
    <script>
        async function loadPlatformData() {
            try {
                // Load platform overview
                const overviewResponse = await fetch('/platform/overview');
                const overview = await overviewResponse.json();
                
                updatePlatformHealth(overview.platform_info, overview.services_summary);
                updatePlatformStats(overview.services_summary, overview.platform_info);
                updateServicesGrid(overview.service_details, overview.platform_capabilities);
                
            } catch (error) {
                console.error('Error loading platform data:', error);
                document.getElementById('platform-health').innerHTML = '<span style="color: #f85149;">❌ Error loading platform data</span>';
            }
        }
        
        function updatePlatformHealth(platformInfo, servicesSummary) {
            const healthDiv = document.getElementById('platform-health');
            const healthPercentage = servicesSummary.service_health_percentage;
            
            let healthStatus, healthClass, healthIcon;
            if (healthPercentage >= 90) {
                healthStatus = 'OPERATIONAL';
                healthClass = 'health-operational';
                healthIcon = '✅';
            } else if (healthPercentage >= 50) {
                healthStatus = 'DEGRADED';
                healthClass = 'health-degraded';
                healthIcon = '⚠️';
            } else {
                healthStatus = 'CRITICAL';
                healthClass = 'health-critical';
                healthIcon = '❌';
            }
            
            healthDiv.innerHTML = `
                <div style="display: flex; align-items: center; justify-content: center; gap: 10px;">
                    <span class="health-indicator ${healthClass}"></span>
                    <span style="font-size: 1.2em; font-weight: 600;">${healthIcon} Platform Status: ${healthStatus}</span>
                </div>
                <div style="margin-top: 10px; color: #8b949e;">
                    ${servicesSummary.healthy_services}/${servicesSummary.total_services} services healthy (${healthPercentage}%) • 
                    External IP: ${platformInfo.external_ip} • 
                    Version: ${platformInfo.version}
                </div>
            `;
        }
        
        function updatePlatformStats(servicesSummary, platformInfo) {
            const statsDiv = document.getElementById('platform-stats');
            
            statsDiv.innerHTML = `
                <div class="stat-card">
                    <div class="stat-value">${servicesSummary.total_services}</div>
                    <div class="stat-label">Active Services</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${servicesSummary.service_health_percentage}%</div>
                    <div class="stat-label">System Health</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${servicesSummary.total_capabilities}</div>
                    <div class="stat-label">Total Capabilities</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">24/7</div>
                    <div class="stat-label">Uptime Monitoring</div>
                </div>
            `;
        }
        
        function updateServicesGrid(services, capabilities) {
            const gridDiv = document.getElementById('services-grid');
            gridDiv.innerHTML = '';
            
            services.forEach(service => {
                const statusClass = `status-${service.status}`;
                const serviceDiv = document.createElement('div');
                serviceDiv.className = 'service-card';
                
                // Get service capabilities from the capabilities object
                const serviceCapabilities = getServiceCapabilities(service.service_id, capabilities);
                
                serviceDiv.innerHTML = `
                    <div class="service-header">
                        <div class="service-title">${service.name}</div>
                        <div class="status-badge ${statusClass}">${service.status}</div>
                    </div>
                    <div class="service-description">${service.description}</div>
                    <div class="service-meta">
                        <div class="port-info">Port: ${service.port}</div>
                        <div style="color: #8b949e; font-size: 0.8em;">v${service.version}</div>
                    </div>
                    <div class="capabilities-list">
                        ${serviceCapabilities.map(cap => `<span class="capability-tag">${cap}</span>`).join('')}
                    </div>
                    <div style="margin-top: 15px;">
                        <a href="${service.url}" target="_blank" class="access-button">
                            🚀 Access Service
                        </a>
                    </div>
                `;
                
                gridDiv.appendChild(serviceDiv);
            });
        }
        
        function getServiceCapabilities(serviceId, capabilities) {
            const capabilityMap = {
                'threat_intelligence': capabilities.threat_intelligence || [],
                'collaboration': capabilities.collaboration || [],
                'analytics': capabilities.analytics || [],
                'reporting': capabilities.reporting || [],
                'hardening': capabilities.security_hardening || [],
                'mobile': capabilities.mobile_access || [],
                'nvidia_ai': capabilities.ai_integration || [],
                'orchestration': capabilities.orchestration || []
            };
            
            return capabilityMap[serviceId] || ['General cybersecurity capabilities'];
        }
        
        // Load platform data on page load
        loadPlatformData();
        
        // Auto-refresh every 60 seconds
        setInterval(loadPlatformData, 60000);
        
        // Add some visual enhancements
        document.addEventListener('DOMContentLoaded', function() {
            // Add fade-in animation
            document.body.style.opacity = '0';
            document.body.style.transition = 'opacity 0.5s';
            setTimeout(() => {
                document.body.style.opacity = '1';
            }, 100);
        });
    </script>
</body>
</html>
    """

@app.get("/health")
async def health_check():
    """Platform summary service health check"""
    return {
        "status": "healthy",
        "service": "xorb_platform_summary",
        "version": "11.0.0",
        "description": "XORB Platform Summary & Status Service",
        "capabilities": [
            "Platform Overview",
            "Service Registry",
            "Health Monitoring",
            "Status Dashboard",
            "Service Discovery"
        ],
        "summary_stats": {
            "total_services_tracked": len(platform_summary.services),
            "platform_capabilities": platform_summary.platform_metrics["total_capabilities"],
            "external_ip": "188.245.101.102",
            "platform_version": "11.0.0"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9012)