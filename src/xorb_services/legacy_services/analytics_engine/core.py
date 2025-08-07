#!/usr/bin/env python3
"""
XORB Advanced Analytics Engine
Machine learning-powered security analytics and behavioral analysis
"""

import asyncio
import json
import random
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import aiohttp
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="XORB Advanced Analytics Engine",
    description="ML-powered security analytics and behavioral analysis",
    version="5.0.0"
)

class AnalyticsModel(str, Enum):
    ANOMALY_DETECTION = "anomaly_detection"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    THREAT_PREDICTION = "threat_prediction"
    RISK_SCORING = "risk_scoring"
    PATTERN_RECOGNITION = "pattern_recognition"

class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityEvent:
    event_id: str
    timestamp: datetime
    source_ip: str
    destination_ip: str
    event_type: str
    protocol: str
    port: int
    payload_size: int
    user_agent: Optional[str] = None
    success: bool = True

class AnalyticsResult(BaseModel):
    analysis_id: str
    model_type: AnalyticsModel
    confidence: float
    risk_score: float
    anomaly_score: float
    behavioral_score: float
    findings: List[str]
    recommendations: List[str]
    timestamp: str
    processing_time: float

class ThreatPattern(BaseModel):
    pattern_id: str
    pattern_name: str
    description: str
    indicators: List[str]
    confidence: float
    severity: AlertSeverity
    detection_count: int
    first_seen: str
    last_seen: str

class BehavioralBaseline(BaseModel):
    entity_id: str
    entity_type: str
    baseline_metrics: Dict
    normal_patterns: List[str]
    deviation_threshold: float
    created_at: str
    updated_at: str

class MLModel:
    """Simulated machine learning model for security analytics"""
    
    def __init__(self, model_type: AnalyticsModel):
        self.model_type = model_type
        self.trained = True
        self.accuracy = random.uniform(0.85, 0.98)
        self.last_training = datetime.now()
        
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """Simulate model prediction"""
        # Simulate prediction processing time
        time.sleep(random.uniform(0.01, 0.05))
        
        # Generate simulated prediction
        confidence = random.uniform(0.7, 0.95)
        score = random.uniform(0.1, 0.9)
        
        return confidence, score
    
    def update_model(self, new_data: List):
        """Simulate model retraining"""
        self.last_training = datetime.now()
        self.accuracy = min(0.99, self.accuracy + random.uniform(0.001, 0.01))

class AdvancedAnalyticsEngine:
    """Advanced security analytics and ML processing engine"""
    
    def __init__(self):
        self.ml_models = {
            AnalyticsModel.ANOMALY_DETECTION: MLModel(AnalyticsModel.ANOMALY_DETECTION),
            AnalyticsModel.BEHAVIORAL_ANALYSIS: MLModel(AnalyticsModel.BEHAVIORAL_ANALYSIS),
            AnalyticsModel.THREAT_PREDICTION: MLModel(AnalyticsModel.THREAT_PREDICTION),
            AnalyticsModel.RISK_SCORING: MLModel(AnalyticsModel.RISK_SCORING),
            AnalyticsModel.PATTERN_RECOGNITION: MLModel(AnalyticsModel.PATTERN_RECOGNITION)
        }
        
        self.security_events: List[SecurityEvent] = []
        self.analytics_results: List[AnalyticsResult] = []
        self.threat_patterns: List[ThreatPattern] = []
        self.behavioral_baselines: Dict[str, BehavioralBaseline] = {}
        
        # Initialize with sample data
        self._initialize_sample_data()
        
    def _initialize_sample_data(self):
        """Initialize with realistic sample security data"""
        # Generate sample security events
        event_types = ["login", "file_access", "network_connection", "process_execution", "privilege_escalation"]
        protocols = ["TCP", "UDP", "HTTP", "HTTPS", "SSH", "FTP"]
        
        for i in range(1000):
            event = SecurityEvent(
                event_id=f"event_{i}_{int(time.time())}",
                timestamp=datetime.now() - timedelta(minutes=random.randint(0, 1440)),
                source_ip=f"192.168.{random.randint(1,255)}.{random.randint(1,255)}",
                destination_ip=f"10.0.{random.randint(1,255)}.{random.randint(1,255)}",
                event_type=random.choice(event_types),
                protocol=random.choice(protocols),
                port=random.choice([22, 80, 443, 3389, 21, 53, 25]),
                payload_size=random.randint(64, 65536),
                user_agent=f"Agent-{random.randint(1,10)}.0" if random.random() > 0.5 else None,
                success=random.random() > 0.1
            )
            self.security_events.append(event)
        
        # Generate threat patterns
        patterns = [
            ("Brute Force Attack", "Multiple failed login attempts from single source", ["failed_login", "repeated_attempts", "single_source"]),
            ("Data Exfiltration", "Large data transfers to external destinations", ["large_transfer", "external_destination", "sensitive_data"]),
            ("Lateral Movement", "Unusual network scanning and privilege escalation", ["network_scan", "privilege_escalation", "internal_movement"]),
            ("Malware Communication", "Suspicious C2 communication patterns", ["c2_communication", "beacon_pattern", "encrypted_payload"]),
            ("Insider Threat", "Abnormal user behavior and data access", ["unusual_access", "off_hours", "sensitive_files"])
        ]
        
        for i, (name, desc, indicators) in enumerate(patterns):
            pattern = ThreatPattern(
                pattern_id=f"pattern_{i}_{int(time.time())}",
                pattern_name=name,
                description=desc,
                indicators=indicators,
                confidence=random.uniform(0.75, 0.95),
                severity=random.choice(list(AlertSeverity)),
                detection_count=random.randint(5, 50),
                first_seen=(datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                last_seen=datetime.now().isoformat()
            )
            self.threat_patterns.append(pattern)
    
    def extract_features(self, events: List[SecurityEvent]) -> np.ndarray:
        """Extract ML features from security events"""
        features = []
        
        for event in events:
            # Extract numerical features
            feature_vector = [
                hash(event.source_ip) % 1000,  # Source IP hash
                hash(event.destination_ip) % 1000,  # Destination IP hash
                event.port,
                event.payload_size,
                int(event.success),
                event.timestamp.hour,  # Time of day
                event.timestamp.weekday(),  # Day of week
                len(event.event_type),  # Event type complexity
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    async def run_anomaly_detection(self, events: List[SecurityEvent]) -> AnalyticsResult:
        """Run anomaly detection on security events"""
        start_time = time.time()
        
        # Extract features
        features = self.extract_features(events)
        
        # Run ML model
        model = self.ml_models[AnalyticsModel.ANOMALY_DETECTION]
        confidence, anomaly_score = model.predict(features)
        
        # Generate findings
        findings = []
        recommendations = []
        
        if anomaly_score > 0.7:
            findings.append(f"High anomaly score detected: {anomaly_score:.3f}")
            findings.append(f"Analyzed {len(events)} security events")
            findings.append("Unusual patterns in network traffic detected")
            recommendations.append("Investigate source IPs with high anomaly scores")
            recommendations.append("Review authentication logs for suspicious activity")
        elif anomaly_score > 0.5:
            findings.append(f"Moderate anomaly detected: {anomaly_score:.3f}")
            recommendations.append("Monitor for escalating anomalous behavior")
        else:
            findings.append("No significant anomalies detected")
            recommendations.append("Continue routine monitoring")
        
        processing_time = time.time() - start_time
        
        result = AnalyticsResult(
            analysis_id=f"anomaly_{int(time.time())}_{random.randint(1000,9999)}",
            model_type=AnalyticsModel.ANOMALY_DETECTION,
            confidence=confidence,
            risk_score=anomaly_score * 10,  # Scale to 0-10
            anomaly_score=anomaly_score,
            behavioral_score=0.0,  # Not applicable for anomaly detection
            findings=findings,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
        
        self.analytics_results.append(result)
        return result
    
    async def run_behavioral_analysis(self, entity_id: str, events: List[SecurityEvent]) -> AnalyticsResult:
        """Run behavioral analysis for specific entity"""
        start_time = time.time()
        
        # Get or create behavioral baseline
        if entity_id not in self.behavioral_baselines:
            await self._create_behavioral_baseline(entity_id, events)
        
        baseline = self.behavioral_baselines[entity_id]
        
        # Analyze current behavior against baseline
        current_metrics = self._calculate_behavioral_metrics(events)
        behavioral_score = self._compare_to_baseline(current_metrics, baseline.baseline_metrics)
        
        model = self.ml_models[AnalyticsModel.BEHAVIORAL_ANALYSIS]
        confidence, _ = model.predict(self.extract_features(events))
        
        findings = []
        recommendations = []
        
        if behavioral_score > 0.8:
            findings.append(f"Significant behavioral deviation detected: {behavioral_score:.3f}")
            findings.append("User behavior differs significantly from established baseline")
            recommendations.append("Immediate investigation recommended")
            recommendations.append("Consider temporary access restrictions")
        elif behavioral_score > 0.6:
            findings.append(f"Moderate behavioral deviation: {behavioral_score:.3f}")
            recommendations.append("Increased monitoring recommended")
        else:
            findings.append("Behavior within normal parameters")
            recommendations.append("Continue baseline monitoring")
        
        processing_time = time.time() - start_time
        
        result = AnalyticsResult(
            analysis_id=f"behavioral_{int(time.time())}_{random.randint(1000,9999)}",
            model_type=AnalyticsModel.BEHAVIORAL_ANALYSIS,
            confidence=confidence,
            risk_score=behavioral_score * 10,
            anomaly_score=0.0,  # Not applicable
            behavioral_score=behavioral_score,
            findings=findings,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
        
        self.analytics_results.append(result)
        return result
    
    async def run_threat_prediction(self, historical_events: List[SecurityEvent]) -> AnalyticsResult:
        """Predict future threats based on historical data"""
        start_time = time.time()
        
        # Analyze historical patterns
        features = self.extract_features(historical_events)
        model = self.ml_models[AnalyticsModel.THREAT_PREDICTION]
        confidence, threat_probability = model.predict(features)
        
        # Generate threat predictions
        findings = []
        recommendations = []
        
        if threat_probability > 0.75:
            findings.append(f"High threat probability predicted: {threat_probability:.3f}")
            findings.append("Historical patterns suggest imminent security incident")
            findings.append("Multiple threat indicators converging")
            recommendations.append("Activate incident response team")
            recommendations.append("Implement enhanced monitoring")
            recommendations.append("Review and update security controls")
        elif threat_probability > 0.5:
            findings.append(f"Elevated threat level predicted: {threat_probability:.3f}")
            recommendations.append("Increase security posture")
            recommendations.append("Brief security team on potential threats")
        else:
            findings.append("Low threat probability")
            recommendations.append("Maintain current security posture")
        
        processing_time = time.time() - start_time
        
        result = AnalyticsResult(
            analysis_id=f"prediction_{int(time.time())}_{random.randint(1000,9999)}",
            model_type=AnalyticsModel.THREAT_PREDICTION,
            confidence=confidence,
            risk_score=threat_probability * 10,
            anomaly_score=0.0,
            behavioral_score=0.0,
            findings=findings,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
        
        self.analytics_results.append(result)
        return result
    
    async def run_pattern_recognition(self, events: List[SecurityEvent]) -> AnalyticsResult:
        """Identify known threat patterns in events"""
        start_time = time.time()
        
        model = self.ml_models[AnalyticsModel.PATTERN_RECOGNITION]
        features = self.extract_features(events)
        confidence, pattern_score = model.predict(features)
        
        # Simulate pattern matching
        detected_patterns = []
        for pattern in self.threat_patterns:
            if random.random() > 0.7:  # Simulate pattern detection
                detected_patterns.append(pattern)
        
        findings = []
        recommendations = []
        
        if detected_patterns:
            findings.append(f"Detected {len(detected_patterns)} known threat patterns")
            for pattern in detected_patterns[:3]:  # Show top 3
                findings.append(f"Pattern: {pattern.pattern_name} (confidence: {pattern.confidence:.2f})")
            
            recommendations.append("Investigate detected threat patterns immediately")
            recommendations.append("Cross-reference with threat intelligence feeds")
            if any(p.severity == AlertSeverity.CRITICAL for p in detected_patterns):
                recommendations.append("CRITICAL: Implement emergency response procedures")
        else:
            findings.append("No known threat patterns detected")
            recommendations.append("Continue pattern monitoring")
        
        processing_time = time.time() - start_time
        
        result = AnalyticsResult(
            analysis_id=f"pattern_{int(time.time())}_{random.randint(1000,9999)}",
            model_type=AnalyticsModel.PATTERN_RECOGNITION,
            confidence=confidence,
            risk_score=pattern_score * 10,
            anomaly_score=0.0,
            behavioral_score=0.0,
            findings=findings,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
        
        self.analytics_results.append(result)
        return result
    
    async def _create_behavioral_baseline(self, entity_id: str, events: List[SecurityEvent]):
        """Create behavioral baseline for entity"""
        metrics = self._calculate_behavioral_metrics(events)
        
        baseline = BehavioralBaseline(
            entity_id=entity_id,
            entity_type="user",  # Could be user, host, service, etc.
            baseline_metrics=metrics,
            normal_patterns=["login_hours", "file_access_patterns", "network_usage"],
            deviation_threshold=0.3,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        self.behavioral_baselines[entity_id] = baseline
    
    def _calculate_behavioral_metrics(self, events: List[SecurityEvent]) -> Dict:
        """Calculate behavioral metrics from events"""
        if not events:
            return {}
        
        # Calculate various behavioral metrics
        metrics = {
            "events_per_hour": len(events) / 24,  # Assuming 24-hour period
            "unique_destinations": len(set(e.destination_ip for e in events)),
            "avg_payload_size": np.mean([e.payload_size for e in events]),
            "success_rate": sum(1 for e in events if e.success) / len(events),
            "most_common_port": max(set(e.port for e in events), key=lambda p: sum(1 for e in events if e.port == p)),
            "time_spread": (max(e.timestamp for e in events) - min(e.timestamp for e in events)).total_seconds() / 3600
        }
        
        return metrics
    
    def _compare_to_baseline(self, current_metrics: Dict, baseline_metrics: Dict) -> float:
        """Compare current behavior to baseline"""
        if not baseline_metrics:
            return 0.0
        
        deviations = []
        for key in baseline_metrics:
            if key in current_metrics:
                baseline_val = baseline_metrics[key]
                current_val = current_metrics[key]
                
                if baseline_val > 0:
                    deviation = abs(current_val - baseline_val) / baseline_val
                    deviations.append(deviation)
        
        return np.mean(deviations) if deviations else 0.0
    
    def get_analytics_summary(self) -> Dict:
        """Get summary of analytics processing"""
        recent_results = [r for r in self.analytics_results if 
                         datetime.fromisoformat(r.timestamp) > datetime.now() - timedelta(hours=24)]
        
        model_performance = {}
        for model_type in AnalyticsModel:
            model_results = [r for r in recent_results if r.model_type == model_type]
            if model_results:
                avg_confidence = np.mean([r.confidence for r in model_results])
                avg_processing_time = np.mean([r.processing_time for r in model_results])
                model_performance[model_type] = {
                    "results_count": len(model_results),
                    "avg_confidence": round(avg_confidence, 3),
                    "avg_processing_time": round(avg_processing_time, 3)
                }
        
        return {
            "total_events_processed": len(self.security_events),
            "total_analytics_results": len(self.analytics_results),
            "results_last_24h": len(recent_results),
            "threat_patterns_detected": len(self.threat_patterns),
            "behavioral_baselines": len(self.behavioral_baselines),
            "model_performance": model_performance,
            "last_analysis": self.analytics_results[-1].timestamp if self.analytics_results else None
        }

# Initialize analytics engine
analytics_engine = AdvancedAnalyticsEngine()

@app.post("/analytics/anomaly-detection")
async def run_anomaly_detection(background_tasks: BackgroundTasks, event_limit: int = 100):
    """Run anomaly detection analysis"""
    events = analytics_engine.security_events[-event_limit:]
    result = await analytics_engine.run_anomaly_detection(events)
    return result.dict()

@app.post("/analytics/behavioral-analysis/{entity_id}")
async def run_behavioral_analysis(entity_id: str, event_limit: int = 50):
    """Run behavioral analysis for specific entity"""
    # Filter events for entity (simplified - in reality would filter by user/host)
    events = analytics_engine.security_events[-event_limit:]
    result = await analytics_engine.run_behavioral_analysis(entity_id, events)
    return result.dict()

@app.post("/analytics/threat-prediction")
async def run_threat_prediction(historical_days: int = 7):
    """Run threat prediction analysis"""
    cutoff_date = datetime.now() - timedelta(days=historical_days)
    historical_events = [e for e in analytics_engine.security_events 
                        if e.timestamp > cutoff_date]
    result = await analytics_engine.run_threat_prediction(historical_events)
    return result.dict()

@app.post("/analytics/pattern-recognition")
async def run_pattern_recognition(event_limit: int = 200):
    """Run pattern recognition analysis"""
    events = analytics_engine.security_events[-event_limit:]
    result = await analytics_engine.run_pattern_recognition(events)
    return result.dict()

@app.get("/analytics/results")
async def get_analytics_results(limit: int = 50):
    """Get recent analytics results"""
    recent_results = analytics_engine.analytics_results[-limit:]
    return {
        "total_results": len(analytics_engine.analytics_results),
        "results": [result.dict() for result in recent_results]
    }

@app.get("/analytics/patterns")
async def get_threat_patterns():
    """Get detected threat patterns"""
    return {
        "total_patterns": len(analytics_engine.threat_patterns),
        "patterns": [pattern.dict() for pattern in analytics_engine.threat_patterns]
    }

@app.get("/analytics/baselines")
async def get_behavioral_baselines():
    """Get behavioral baselines"""
    return {
        "total_baselines": len(analytics_engine.behavioral_baselines),
        "baselines": [baseline.dict() for baseline in analytics_engine.behavioral_baselines.values()]
    }

@app.get("/analytics/summary")
async def get_analytics_summary():
    """Get analytics engine summary"""
    return analytics_engine.get_analytics_summary()

@app.get("/analytics/models/performance")
async def get_model_performance():
    """Get ML model performance metrics"""
    performance = {}
    for model_type, model in analytics_engine.ml_models.items():
        performance[model_type] = {
            "accuracy": round(model.accuracy, 3),
            "last_training": model.last_training.isoformat(),
            "trained": model.trained
        }
    
    return {
        "models": performance,
        "total_models": len(analytics_engine.ml_models)
    }

@app.post("/analytics/retrain/{model_type}")
async def retrain_model(model_type: AnalyticsModel):
    """Retrain specific ML model"""
    if model_type not in analytics_engine.ml_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = analytics_engine.ml_models[model_type]
    # Simulate retraining with recent events
    training_data = analytics_engine.security_events[-500:]  # Use last 500 events
    model.update_model(training_data)
    
    return {
        "model_type": model_type,
        "retrained_at": model.last_training.isoformat(),
        "new_accuracy": round(model.accuracy, 3),
        "training_data_size": len(training_data)
    }

@app.get("/analytics/dashboard", response_class=HTMLResponse)
async def analytics_dashboard():
    """Advanced analytics dashboard"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>XORB Advanced Analytics Dashboard</title>
    <style>
        body { font-family: 'Inter', sans-serif; background: #0d1117; color: #f0f6fc; margin: 0; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; }
        .analytics-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; }
        .card-header { display: flex; justify-content: between; align-items: center; margin-bottom: 15px; }
        .card-title { font-size: 1.2em; font-weight: 600; color: #58a6ff; }
        .metric-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; }
        .metric { background: #0d1117; padding: 15px; border-radius: 6px; text-align: center; }
        .metric-value { font-size: 1.8em; font-weight: bold; color: #58a6ff; }
        .metric-label { font-size: 0.9em; color: #8b949e; margin-top: 5px; }
        .analysis-button { background: #238636; border: none; color: white; padding: 10px 15px; border-radius: 6px; cursor: pointer; margin: 5px; }
        .analysis-button:hover { background: #2ea043; }
        .results-container { max-height: 300px; overflow-y: auto; margin-top: 15px; }
        .result-item { background: #0d1117; padding: 10px; margin: 10px 0; border-radius: 6px; border-left: 4px solid #58a6ff; }
        .risk-high { border-left-color: #f85149; }
        .risk-medium { border-left-color: #d29922; }
        .risk-low { border-left-color: #3fb950; }
        .model-status { display: flex; align-items: center; gap: 10px; margin: 10px 0; }
        .status-indicator { width: 10px; height: 10px; border-radius: 50%; }
        .status-active { background: #3fb950; }
        .status-training { background: #d29922; }
        .pattern-list { max-height: 200px; overflow-y: auto; }
        .pattern-item { background: #0d1117; padding: 8px; margin: 5px 0; border-radius: 4px; }
        .loading { text-align: center; color: #8b949e; padding: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 XORB ADVANCED ANALYTICS ENGINE</h1>
        <p>Machine Learning-Powered Security Analytics</p>
        <div id="status">Loading analytics engine...</div>
    </div>
    
    <div class="dashboard-grid">
        <!-- Analytics Summary Card -->
        <div class="analytics-card">
            <div class="card-header">
                <span class="card-title">📊 Analytics Summary</span>
            </div>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-value" id="total-events">-</div>
                    <div class="metric-label">Total Events</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="total-results">-</div>
                    <div class="metric-label">Analysis Results</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="threat-patterns">-</div>
                    <div class="metric-label">Threat Patterns</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="behavioral-baselines">-</div>
                    <div class="metric-label">Behavioral Baselines</div>
                </div>
            </div>
        </div>
        
        <!-- ML Models Status Card -->
        <div class="analytics-card">
            <div class="card-header">
                <span class="card-title">🤖 ML Models Status</span>
            </div>
            <div id="models-status">
                <div class="loading">Loading model status...</div>
            </div>
        </div>
        
        <!-- Analysis Controls Card -->
        <div class="analytics-card">
            <div class="card-header">
                <span class="card-title">🔍 Run Analysis</span>
            </div>
            <div>
                <button class="analysis-button" onclick="runAnomalyDetection()">🚨 Anomaly Detection</button>
                <button class="analysis-button" onclick="runBehavioralAnalysis()">👤 Behavioral Analysis</button>
                <button class="analysis-button" onclick="runThreatPrediction()">🔮 Threat Prediction</button>
                <button class="analysis-button" onclick="runPatternRecognition()">🎯 Pattern Recognition</button>
            </div>
            <div id="analysis-status" style="margin-top: 15px; color: #8b949e;"></div>
        </div>
        
        <!-- Recent Results Card -->
        <div class="analytics-card">
            <div class="card-header">
                <span class="card-title">📋 Recent Analysis Results</span>
            </div>
            <div class="results-container" id="recent-results">
                <div class="loading">Loading recent results...</div>
            </div>
        </div>
        
        <!-- Threat Patterns Card -->
        <div class="analytics-card">
            <div class="card-header">
                <span class="card-title">⚠️ Detected Threat Patterns</span>
            </div>
            <div class="pattern-list" id="threat-patterns-list">
                <div class="loading">Loading threat patterns...</div>
            </div>
        </div>
        
        <!-- Real-time Metrics Card -->
        <div class="analytics-card">
            <div class="card-header">
                <span class="card-title">⚡ Real-time Metrics</span>
            </div>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-value" id="avg-confidence">-</div>
                    <div class="metric-label">Avg Confidence</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="processing-time">-</div>
                    <div class="metric-label">Avg Processing (ms)</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="results-24h">-</div>
                    <div class="metric-label">Results (24h)</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="high-risk-alerts">-</div>
                    <div class="metric-label">High Risk Alerts</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        async function loadDashboardData() {
            try {
                // Load analytics summary
                const summaryResponse = await fetch('/analytics/summary');
                const summary = await summaryResponse.json();
                
                document.getElementById('total-events').textContent = summary.total_events_processed.toLocaleString();
                document.getElementById('total-results').textContent = summary.total_analytics_results.toLocaleString();
                document.getElementById('threat-patterns').textContent = summary.threat_patterns_detected;
                document.getElementById('behavioral-baselines').textContent = summary.behavioral_baselines;
                document.getElementById('results-24h').textContent = summary.results_last_24h || 0;
                
                // Load model performance
                const modelsResponse = await fetch('/analytics/models/performance');
                const models = await modelsResponse.json();
                updateModelsStatus(models.models);
                
                // Load recent results
                const resultsResponse = await fetch('/analytics/results?limit=10');
                const results = await resultsResponse.json();
                updateRecentResults(results.results);
                
                // Load threat patterns
                const patternsResponse = await fetch('/analytics/patterns');
                const patterns = await patternsResponse.json();
                updateThreatPatterns(patterns.patterns);
                
                document.getElementById('status').textContent = '✅ Analytics Engine Online';
                document.getElementById('status').style.color = '#3fb950';
                
            } catch (error) {
                console.error('Error loading dashboard data:', error);
                document.getElementById('status').textContent = '❌ Error Loading Data';
                document.getElementById('status').style.color = '#f85149';
            }
        }
        
        function updateModelsStatus(models) {
            const container = document.getElementById('models-status');
            container.innerHTML = '';
            
            Object.keys(models).forEach(modelType => {
                const model = models[modelType];
                const modelDiv = document.createElement('div');
                modelDiv.className = 'model-status';
                modelDiv.innerHTML = `
                    <span class="status-indicator status-active"></span>
                    <span>${modelType.replace('_', ' ').toUpperCase()}</span>
                    <span style="margin-left: auto; color: #8b949e;">${(model.accuracy * 100).toFixed(1)}% accuracy</span>
                `;
                container.appendChild(modelDiv);
            });
        }
        
        function updateRecentResults(results) {
            const container = document.getElementById('recent-results');
            container.innerHTML = '';
            
            if (results.length === 0) {
                container.innerHTML = '<div class="loading">No recent results</div>';
                return;
            }
            
            results.forEach(result => {
                const resultDiv = document.createElement('div');
                const riskClass = result.risk_score >= 7 ? 'risk-high' : 
                                result.risk_score >= 4 ? 'risk-medium' : 'risk-low';
                resultDiv.className = `result-item ${riskClass}`;
                
                const timestamp = new Date(result.timestamp).toLocaleString();
                resultDiv.innerHTML = `
                    <div><strong>${result.model_type.replace('_', ' ').toUpperCase()}</strong></div>
                    <div>Risk Score: ${result.risk_score.toFixed(1)}/10 | Confidence: ${(result.confidence * 100).toFixed(0)}%</div>
                    <div style="font-size: 0.9em; color: #8b949e;">${timestamp}</div>
                `;
                container.appendChild(resultDiv);
            });
        }
        
        function updateThreatPatterns(patterns) {
            const container = document.getElementById('threat-patterns-list');
            container.innerHTML = '';
            
            if (patterns.length === 0) {
                container.innerHTML = '<div class="loading">No threat patterns detected</div>';
                return;
            }
            
            patterns.forEach(pattern => {
                const patternDiv = document.createElement('div');
                patternDiv.className = 'pattern-item';
                patternDiv.innerHTML = `
                    <div><strong>${pattern.pattern_name}</strong></div>
                    <div style="font-size: 0.9em;">${pattern.description}</div>
                    <div style="font-size: 0.8em; color: #8b949e;">
                        Confidence: ${(pattern.confidence * 100).toFixed(0)}% | 
                        Severity: ${pattern.severity.toUpperCase()} |
                        Detected: ${pattern.detection_count} times
                    </div>
                `;
                container.appendChild(patternDiv);
            });
        }
        
        async function runAnomalyDetection() {
            const button = event.target;
            const originalText = button.textContent;
            button.textContent = '🔄 Running...';
            button.disabled = true;
            
            document.getElementById('analysis-status').textContent = 'Running anomaly detection analysis...';
            
            try {
                const response = await fetch('/analytics/anomaly-detection', { method: 'POST' });
                const result = await response.json();
                
                document.getElementById('analysis-status').innerHTML = `
                    <strong>Anomaly Detection Complete</strong><br>
                    Risk Score: ${result.risk_score.toFixed(1)}/10 | 
                    Confidence: ${(result.confidence * 100).toFixed(0)}% |
                    Processing Time: ${(result.processing_time * 1000).toFixed(0)}ms
                `;
                
                // Refresh results
                setTimeout(loadDashboardData, 1000);
                
            } catch (error) {
                document.getElementById('analysis-status').textContent = 'Error running analysis: ' + error.message;
            } finally {
                button.textContent = originalText;
                button.disabled = false;
            }
        }
        
        async function runBehavioralAnalysis() {
            const button = event.target;
            const originalText = button.textContent;
            button.textContent = '🔄 Running...';
            button.disabled = true;
            
            document.getElementById('analysis-status').textContent = 'Running behavioral analysis...';
            
            try {
                const entityId = 'user_demo_' + Date.now();
                const response = await fetch(`/analytics/behavioral-analysis/${entityId}`, { method: 'POST' });
                const result = await response.json();
                
                document.getElementById('analysis-status').innerHTML = `
                    <strong>Behavioral Analysis Complete</strong><br>
                    Risk Score: ${result.risk_score.toFixed(1)}/10 | 
                    Behavioral Score: ${result.behavioral_score.toFixed(2)} |
                    Processing Time: ${(result.processing_time * 1000).toFixed(0)}ms
                `;
                
                setTimeout(loadDashboardData, 1000);
                
            } catch (error) {
                document.getElementById('analysis-status').textContent = 'Error running analysis: ' + error.message;
            } finally {
                button.textContent = originalText;
                button.disabled = false;
            }
        }
        
        async function runThreatPrediction() {
            const button = event.target;
            const originalText = button.textContent;
            button.textContent = '🔄 Running...';
            button.disabled = true;
            
            document.getElementById('analysis-status').textContent = 'Running threat prediction analysis...';
            
            try {
                const response = await fetch('/analytics/threat-prediction', { method: 'POST' });
                const result = await response.json();
                
                document.getElementById('analysis-status').innerHTML = `
                    <strong>Threat Prediction Complete</strong><br>
                    Risk Score: ${result.risk_score.toFixed(1)}/10 | 
                    Confidence: ${(result.confidence * 100).toFixed(0)}% |
                    Processing Time: ${(result.processing_time * 1000).toFixed(0)}ms
                `;
                
                setTimeout(loadDashboardData, 1000);
                
            } catch (error) {
                document.getElementById('analysis-status').textContent = 'Error running analysis: ' + error.message;
            } finally {
                button.textContent = originalText;
                button.disabled = false;
            }
        }
        
        async function runPatternRecognition() {
            const button = event.target;
            const originalText = button.textContent;
            button.textContent = '🔄 Running...';
            button.disabled = true;
            
            document.getElementById('analysis-status').textContent = 'Running pattern recognition analysis...';
            
            try {
                const response = await fetch('/analytics/pattern-recognition', { method: 'POST' });
                const result = await response.json();
                
                document.getElementById('analysis-status').innerHTML = `
                    <strong>Pattern Recognition Complete</strong><br>
                    Risk Score: ${result.risk_score.toFixed(1)}/10 | 
                    Confidence: ${(result.confidence * 100).toFixed(0)}% |
                    Processing Time: ${(result.processing_time * 1000).toFixed(0)}ms
                `;
                
                setTimeout(loadDashboardData, 1000);
                
            } catch (error) {
                document.getElementById('analysis-status').textContent = 'Error running analysis: ' + error.message;
            } finally {
                button.textContent = originalText;
                button.disabled = false;
            }
        }
        
        // Initialize dashboard
        loadDashboardData();
        
        // Refresh dashboard every 30 seconds
        setInterval(loadDashboardData, 30000);
    </script>
</body>
</html>
    """

@app.get("/health")
async def health_check():
    """Advanced analytics engine health check"""
    return {
        "status": "healthy",
        "service": "xorb_analytics_engine",
        "version": "5.0.0",
        "capabilities": [
            "Anomaly Detection",
            "Behavioral Analysis", 
            "Threat Prediction",
            "Pattern Recognition",
            "Risk Scoring",
            "ML Model Management",
            "Real-time Analytics"
        ],
        "engine_stats": {
            "total_security_events": len(analytics_engine.security_events),
            "total_analytics_results": len(analytics_engine.analytics_results),
            "ml_models_active": len(analytics_engine.ml_models),
            "threat_patterns": len(analytics_engine.threat_patterns),
            "behavioral_baselines": len(analytics_engine.behavioral_baselines)
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9006)