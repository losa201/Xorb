#!/usr/bin/env python3
"""
XORB Advanced Threat Intelligence Engine
Real-time threat intelligence feeds with global threat landscape analysis
"""

import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

import aiohttp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="XORB Threat Intelligence Engine",
    description="Advanced Threat Intelligence with Real-time Feeds",
    version="3.0.0"
)

class ThreatIndicator(BaseModel):
    indicator_type: str
    value: str
    threat_level: int
    confidence: float
    source: str
    first_seen: str
    last_seen: str
    tags: List[str]
    geo_location: Optional[str] = None

class ThreatFeed(BaseModel):
    feed_id: str
    feed_name: str
    provider: str
    indicators_count: int
    last_update: str
    feed_type: str
    reliability_score: float

class ThreatCampaign(BaseModel):
    campaign_id: str
    name: str
    threat_actors: List[str]
    techniques: List[str]
    targets: List[str]
    confidence: float
    active_since: str
    severity: int

class ConnectionManager:
    """WebSocket connection manager for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove broken connections
                self.active_connections.remove(connection)

class ThreatIntelligenceEngine:
    """Advanced threat intelligence processing engine"""
    
    def __init__(self):
        self.threat_indicators: List[ThreatIndicator] = []
        self.threat_feeds: List[ThreatFeed] = []
        self.active_campaigns: List[ThreatCampaign] = []
        self.feed_sources = {
            "global_malware_db": {"reliability": 0.95, "type": "malware"},
            "nation_state_intel": {"reliability": 0.88, "type": "apt"},
            "darkweb_monitoring": {"reliability": 0.82, "type": "underground"},
            "industry_sharing": {"reliability": 0.91, "type": "sector"},
            "honeypot_network": {"reliability": 0.85, "type": "network"},
            "sandbox_analysis": {"reliability": 0.93, "type": "behavior"}
        }
        self.connection_manager = ConnectionManager()
        self.threat_score_cache = {}
        
    async def initialize_threat_feeds(self):
        """Initialize threat intelligence feeds"""
        for feed_name, config in self.feed_sources.items():
            feed = ThreatFeed(
                feed_id=f"feed_{feed_name}_{int(time.time())}",
                feed_name=feed_name.replace("_", " ").title(),
                provider=f"XORB-{feed_name.split('_')[0].upper()}",
                indicators_count=random.randint(1000, 50000),
                last_update=datetime.now().isoformat(),
                feed_type=config["type"],
                reliability_score=config["reliability"]
            )
            self.threat_feeds.append(feed)
    
    async def generate_threat_indicators(self, count: int = 100) -> List[ThreatIndicator]:
        """Generate realistic threat indicators"""
        indicator_types = ["ip", "domain", "url", "file_hash", "email", "registry_key", "mutex"]
        threat_levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        sources = list(self.feed_sources.keys())
        
        indicators = []
        
        for _ in range(count):
            indicator_type = random.choice(indicator_types)
            
            # Generate realistic values based on type
            if indicator_type == "ip":
                value = f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}"
            elif indicator_type == "domain":
                domains = ["malicious-site.com", "phishing-portal.net", "backdoor-c2.org", "trojan-loader.biz"]
                value = random.choice(domains)
            elif indicator_type == "url":
                value = f"http://malicious-{random.randint(1000,9999)}.com/payload"
            elif indicator_type == "file_hash":
                value = f"{''.join(random.choices('abcdef0123456789', k=64))}"
            elif indicator_type == "email":
                value = f"attacker{random.randint(1,999)}@malicious-domain.com"
            else:
                value = f"{indicator_type}_indicator_{random.randint(1000,9999)}"
            
            # Generate tags based on threat intelligence
            tags = random.sample([
                "apt", "malware", "phishing", "ransomware", "banking_trojan",
                "botnet", "c2", "exploit_kit", "zero_day", "targeted_attack",
                "nation_state", "cybercrime", "insider_threat", "supply_chain"
            ], random.randint(1, 4))
            
            indicator = ThreatIndicator(
                indicator_type=indicator_type,
                value=value,
                threat_level=random.choice(threat_levels),
                confidence=round(random.uniform(0.6, 0.99), 3),
                source=random.choice(sources),
                first_seen=(datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                last_seen=datetime.now().isoformat(),
                tags=tags,
                geo_location=random.choice(["US", "CN", "RU", "IR", "KP", "UK", "DE", None])
            )
            
            indicators.append(indicator)
        
        return indicators
    
    async def generate_threat_campaigns(self) -> List[ThreatCampaign]:
        """Generate active threat campaigns"""
        campaigns = [
            {
                "name": "Operation ShadowNet",
                "threat_actors": ["APT29", "Cozy Bear"],
                "techniques": ["spear_phishing", "credential_harvesting", "lateral_movement"],
                "targets": ["government", "defense", "technology"],
                "severity": 9
            },
            {
                "name": "DarkHalo Campaign",
                "threat_actors": ["UNC2452", "SolarWinds Attackers"],
                "techniques": ["supply_chain", "backdoor_deployment", "privilege_escalation"],
                "targets": ["enterprise", "government", "critical_infrastructure"],
                "severity": 10
            },
            {
                "name": "CryptoMiner Surge",
                "threat_actors": ["8220 Gang", "TeamTNT"],
                "techniques": ["cryptojacking", "container_escape", "cloud_exploitation"],
                "targets": ["cloud_infrastructure", "containers", "kubernetes"],
                "severity": 6
            },
            {
                "name": "PhishScale Operation",
                "threat_actors": ["BulletProof Hosting", "Phishing-as-a-Service"],
                "techniques": ["credential_phishing", "business_email_compromise", "social_engineering"],
                "targets": ["financial_services", "healthcare", "retail"],
                "severity": 7
            },
            {
                "name": "RansomWave 2025",
                "threat_actors": ["LockBit", "BlackCat", "Royal"],
                "techniques": ["double_extortion", "data_encryption", "supply_chain_targeting"],
                "targets": ["healthcare", "education", "manufacturing"],
                "severity": 9
            }
        ]
        
        active_campaigns = []
        for campaign_data in campaigns:
            campaign = ThreatCampaign(
                campaign_id=f"camp_{int(time.time())}_{random.randint(1000,9999)}",
                name=campaign_data["name"],
                threat_actors=campaign_data["threat_actors"],
                techniques=campaign_data["techniques"],
                targets=campaign_data["targets"],
                confidence=round(random.uniform(0.75, 0.98), 3),
                active_since=(datetime.now() - timedelta(days=random.randint(7, 60))).isoformat(),
                severity=campaign_data["severity"]
            )
            active_campaigns.append(campaign)
        
        return active_campaigns
    
    async def analyze_threat_landscape(self) -> Dict:
        """Analyze current global threat landscape"""
        # Generate current threat indicators
        indicators = await self.generate_threat_indicators(200)
        self.threat_indicators.extend(indicators)
        
        # Keep only recent indicators (last 7 days)
        cutoff_date = datetime.now() - timedelta(days=7)
        self.threat_indicators = [
            ind for ind in self.threat_indicators 
            if datetime.fromisoformat(ind.last_seen) > cutoff_date
        ]
        
        # Analyze threat distribution
        threat_by_type = {}
        threat_by_level = {}
        threat_by_geo = {}
        
        for indicator in self.threat_indicators:
            # By type
            threat_by_type[indicator.indicator_type] = threat_by_type.get(indicator.indicator_type, 0) + 1
            
            # By threat level
            level_range = f"Level {indicator.threat_level}"
            threat_by_level[level_range] = threat_by_level.get(level_range, 0) + 1
            
            # By geography
            if indicator.geo_location:
                threat_by_geo[indicator.geo_location] = threat_by_geo.get(indicator.geo_location, 0) + 1
        
        # Calculate threat trends
        high_threat_indicators = len([ind for ind in self.threat_indicators if ind.threat_level >= 8])
        critical_indicators = len([ind for ind in self.threat_indicators if ind.threat_level >= 9])
        
        return {
            "total_indicators": len(self.threat_indicators),
            "high_threat_count": high_threat_indicators,
            "critical_threat_count": critical_indicators,
            "threat_distribution": {
                "by_type": threat_by_type,
                "by_level": threat_by_level,
                "by_geography": threat_by_geo
            },
            "threat_velocity": round(len(indicators) / 24, 2),  # Indicators per hour
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def real_time_threat_monitoring(self):
        """Real-time threat feed monitoring loop"""
        while True:
            try:
                # Generate new threat intelligence
                new_indicators = await self.generate_threat_indicators(random.randint(5, 25))
                self.threat_indicators.extend(new_indicators)
                
                # Analyze critical threats
                critical_threats = [
                    ind for ind in new_indicators 
                    if ind.threat_level >= 8
                ]
                
                if critical_threats:
                    # Broadcast critical threat alerts
                    alert_message = {
                        "type": "critical_threat_alert",
                        "count": len(critical_threats),
                        "indicators": [ind.dict() for ind in critical_threats[:5]],  # Limit to 5 for broadcast
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    await self.connection_manager.broadcast(json.dumps(alert_message))
                
                # Update threat landscape analysis
                analysis = await self.analyze_threat_landscape()
                
                # Broadcast landscape update
                landscape_message = {
                    "type": "threat_landscape_update",
                    "data": analysis,
                    "timestamp": datetime.now().isoformat()
                }
                
                await self.connection_manager.broadcast(json.dumps(landscape_message))
                
                # Wait before next update (30-60 seconds)
                await asyncio.sleep(random.randint(30, 60))
                
            except Exception as e:
                print(f"Error in threat monitoring: {e}")
                await asyncio.sleep(30)
    
    def calculate_organization_threat_score(self, organization_id: str) -> float:
        """Calculate threat score for organization"""
        # Simulate threat score calculation
        base_score = random.uniform(2.0, 8.5)
        
        # Add some persistence
        if organization_id in self.threat_score_cache:
            cached_score = self.threat_score_cache[organization_id]
            # Gradual change from cached score
            base_score = cached_score + random.uniform(-0.5, 0.5)
            base_score = max(1.0, min(10.0, base_score))
        
        self.threat_score_cache[organization_id] = base_score
        return round(base_score, 2)

# Initialize threat intelligence engine
threat_engine = ThreatIntelligenceEngine()

@app.on_event("startup")
async def startup_event():
    """Initialize threat intelligence on startup"""
    await threat_engine.initialize_threat_feeds()
    threat_engine.active_campaigns = await threat_engine.generate_threat_campaigns()
    
    # Start real-time monitoring
    asyncio.create_task(threat_engine.real_time_threat_monitoring())

@app.websocket("/ws/threat-feed")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time threat intelligence"""
    await threat_engine.connection_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            # Echo back or process client requests
            await threat_engine.connection_manager.send_personal_message(
                f"Threat Intelligence: {data}", 
                websocket
            )
    except WebSocketDisconnect:
        threat_engine.connection_manager.disconnect(websocket)

@app.get("/threat-intelligence/feeds")
async def get_threat_feeds():
    """Get available threat intelligence feeds"""
    return {
        "total_feeds": len(threat_engine.threat_feeds),
        "feeds": [feed.dict() for feed in threat_engine.threat_feeds],
        "last_update": datetime.now().isoformat()
    }

@app.get("/threat-intelligence/indicators")
async def get_threat_indicators(
    indicator_type: Optional[str] = None,
    threat_level_min: Optional[int] = None,
    limit: int = 100
):
    """Get threat indicators with filtering"""
    indicators = threat_engine.threat_indicators
    
    if indicator_type:
        indicators = [ind for ind in indicators if ind.indicator_type == indicator_type]
    
    if threat_level_min:
        indicators = [ind for ind in indicators if ind.threat_level >= threat_level_min]
    
    indicators = indicators[-limit:]  # Get most recent
    
    return {
        "total_indicators": len(threat_engine.threat_indicators),
        "filtered_count": len(indicators),
        "indicators": [ind.dict() for ind in indicators]
    }

@app.get("/threat-intelligence/campaigns")
async def get_active_campaigns():
    """Get active threat campaigns"""
    return {
        "active_campaigns": len(threat_engine.active_campaigns),
        "campaigns": [camp.dict() for camp in threat_engine.active_campaigns]
    }

@app.get("/threat-intelligence/landscape")
async def get_threat_landscape():
    """Get comprehensive threat landscape analysis"""
    analysis = await threat_engine.analyze_threat_landscape()
    return analysis

@app.get("/threat-intelligence/organization/{org_id}/score")
async def get_organization_threat_score(org_id: str):
    """Get threat score for specific organization"""
    score = threat_engine.calculate_organization_threat_score(org_id)
    
    # Determine risk level
    if score >= 8.0:
        risk_level = "Critical"
    elif score >= 6.0:
        risk_level = "High"
    elif score >= 4.0:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    
    return {
        "organization_id": org_id,
        "threat_score": score,
        "risk_level": risk_level,
        "calculated_at": datetime.now().isoformat(),
        "factors": {
            "indicator_matches": random.randint(0, 25),
            "campaign_exposure": random.randint(0, 5),
            "geographic_risk": random.choice(["Low", "Medium", "High"]),
            "industry_targeting": random.choice(["Low", "Medium", "High"])
        }
    }

@app.get("/threat-intelligence/real-time-demo", response_class=HTMLResponse)
async def real_time_demo():
    """Real-time threat intelligence demo page"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>XORB Threat Intelligence - Real-time Feed</title>
    <style>
        body { font-family: 'Courier New', monospace; background: #000; color: #00ff00; margin: 0; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .feed-container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .feed-box { border: 2px solid #00ff00; padding: 15px; height: 400px; overflow-y: auto; }
        .threat-item { margin: 10px 0; padding: 8px; border-left: 3px solid #ff6600; background: #001100; }
        .critical { border-left-color: #ff0000; background: #110000; }
        .timestamp { color: #888; font-size: 0.8em; }
        .threat-level { font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ XORB THREAT INTELLIGENCE ENGINE 🛡️</h1>
        <p>Real-time Global Threat Monitoring</p>
        <div id="connection-status">Connecting...</div>
    </div>
    
    <div class="feed-container">
        <div class="feed-box">
            <h3>🚨 Critical Threat Alerts</h3>
            <div id="critical-alerts"></div>
        </div>
        
        <div class="feed-box">
            <h3>📊 Threat Landscape Updates</h3>
            <div id="landscape-updates"></div>
        </div>
    </div>
    
    <script>
        const ws = new WebSocket('ws://188.245.101.102:9004/ws/threat-feed');
        const statusDiv = document.getElementById('connection-status');
        const alertsDiv = document.getElementById('critical-alerts');
        const landscapeDiv = document.getElementById('landscape-updates');
        
        ws.onopen = function(event) {
            statusDiv.textContent = '✅ Connected to Threat Intelligence Feed';
            statusDiv.style.color = '#00ff00';
        };
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.type === 'critical_threat_alert') {
                const alertHtml = `
                    <div class="threat-item critical">
                        <div class="threat-level">🚨 CRITICAL THREATS DETECTED: ${data.count}</div>
                        ${data.indicators.map(ind => `
                            <div>Type: ${ind.indicator_type} | Value: ${ind.value} | Level: ${ind.threat_level}</div>
                        `).join('')}
                        <div class="timestamp">${new Date(data.timestamp).toLocaleString()}</div>
                    </div>
                `;
                alertsDiv.innerHTML = alertHtml + alertsDiv.innerHTML;
                
                // Keep only last 10 alerts
                const alerts = alertsDiv.children;
                while (alerts.length > 10) {
                    alertsDiv.removeChild(alerts[alerts.length - 1]);
                }
            }
            
            if (data.type === 'threat_landscape_update') {
                const landscapeHtml = `
                    <div class="threat-item">
                        <div class="threat-level">📊 Landscape Update</div>
                        <div>Total Indicators: ${data.data.total_indicators}</div>
                        <div>High Threats: ${data.data.high_threat_count}</div>
                        <div>Critical Threats: ${data.data.critical_threat_count}</div>
                        <div>Threat Velocity: ${data.data.threat_velocity}/hour</div>
                        <div class="timestamp">${new Date(data.timestamp).toLocaleString()}</div>
                    </div>
                `;
                landscapeDiv.innerHTML = landscapeHtml + landscapeDiv.innerHTML;
                
                // Keep only last 15 updates
                const updates = landscapeDiv.children;
                while (updates.length > 15) {
                    landscapeDiv.removeChild(updates[updates.length - 1]);
                }
            }
        };
        
        ws.onclose = function(event) {
            statusDiv.textContent = '❌ Disconnected from Threat Intelligence Feed';
            statusDiv.style.color = '#ff0000';
        };
        
        // Send heartbeat every 30 seconds
        setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send('heartbeat');
            }
        }, 30000);
    </script>
</body>
</html>
    """

@app.get("/health")
async def health_check():
    """Threat intelligence engine health check"""
    return {
        "status": "healthy",
        "service": "xorb_threat_intelligence_engine",
        "version": "3.0.0",
        "capabilities": [
            "Real-time Threat Feeds",
            "Threat Indicator Analysis",
            "Campaign Tracking",
            "Landscape Monitoring",
            "WebSocket Streaming",
            "Organization Risk Scoring"
        ],
        "active_feeds": len(threat_engine.threat_feeds),
        "active_indicators": len(threat_engine.threat_indicators),
        "active_campaigns": len(threat_engine.active_campaigns),
        "websocket_connections": len(threat_engine.connection_manager.active_connections)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9004)