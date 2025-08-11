#!/usr/bin/env python3
"""
SIEM Integration for Cyber Range Telemetry
Comprehensive log forwarding and security event correlation
"""

import json
import time
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import socket
import ssl
import logging
import queue
import threading
from pathlib import Path

# External integrations
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Async HTTP client
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("Warning: aiohttp not available. Some async features disabled.")

# Syslog support
try:
    import syslogmp
    SYSLOG_AVAILABLE = True
except ImportError:
    SYSLOG_AVAILABLE = False
    print("Warning: syslogmp not available. Syslog integration disabled.")

class SIEMType(Enum):
    """Supported SIEM platforms"""
    SPLUNK = "splunk"
    ELASTIC = "elastic"
    QRADAR = "qradar"
    SENTINEL = "sentinel"
    CHRONICLE = "chronicle"
    SUMO_LOGIC = "sumo_logic"
    DATADOG = "datadog"
    GENERIC_REST = "generic_rest"
    SYSLOG = "syslog"

class EventSeverity(Enum):
    """Event severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EventCategory(Enum):
    """Event categories for SIEM correlation"""
    ATTACK = "attack"
    DEFENSE = "defense"
    DETECTION = "detection"
    RESPONSE = "response"
    SYSTEM = "system"
    COMPLIANCE = "compliance"
    SAFETY = "safety"
    PERFORMANCE = "performance"

@dataclass
class SIEMConfig:
    """Configuration for SIEM integration"""
    siem_type: SIEMType
    endpoint_url: str
    authentication: Dict[str, str] = field(default_factory=dict)
    
    # Connection settings
    timeout_seconds: int = 30
    max_retries: int = 3
    batch_size: int = 100
    flush_interval_seconds: int = 10
    
    # SSL/TLS settings
    verify_ssl: bool = True
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    
    # Index/destination settings
    index_name: str = "cyber-range"
    source_type: str = "xorb:cyber-range"
    
    # Filtering
    min_severity: EventSeverity = EventSeverity.INFO
    event_categories: List[EventCategory] = field(default_factory=lambda: list(EventCategory))
    
    # Rate limiting
    max_events_per_second: int = 1000
    burst_limit: int = 5000

@dataclass
class SecurityEvent:
    """Standardized security event for SIEM"""
    # Event identification
    event_id: str
    timestamp: str
    event_type: str
    category: EventCategory
    severity: EventSeverity
    
    # Source information
    source_system: str = "xorb-cyber-range"
    source_component: str = ""
    episode_id: str = ""
    round_id: int = 0
    
    # Actor information
    actor_type: str = ""  # red_agent, blue_agent, purple_agent, human
    actor_id: str = ""
    
    # Target information
    target_system: str = ""
    target_ip: Optional[str] = None
    target_port: Optional[int] = None
    target_protocol: Optional[str] = None
    
    # Event details
    action: str = ""
    technique: str = ""
    outcome: str = ""  # success, failure, partial
    
    # Threat intelligence
    mitre_technique: Optional[str] = None
    iocs: List[str] = field(default_factory=list)
    threat_score: float = 0.0
    confidence: float = 0.0
    
    # Context
    description: str = ""
    raw_data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Compliance
    compliance_frameworks: List[str] = field(default_factory=list)
    regulatory_impact: bool = False
    
    def to_cef(self) -> str:
        """Convert to Common Event Format (CEF)"""
        # CEF Header
        vendor = "XORB"
        product = "Cyber Range"
        version = "1.0"
        signature = self.event_type
        name = self.action or self.event_type
        severity = self._severity_to_cef_level()
        
        header = f"CEF:0|{vendor}|{product}|{version}|{signature}|{name}|{severity}|"
        
        # CEF Extensions
        extensions = []
        extensions.append(f"rt={int(datetime.fromisoformat(self.timestamp.replace('Z', '+00:00')).timestamp() * 1000)}")
        extensions.append(f"src={self.target_ip or 'unknown'}")
        extensions.append(f"spt={self.target_port or 0}")
        extensions.append(f"act={self.action}")
        extensions.append(f"outcome={self.outcome}")
        extensions.append(f"cs1={self.episode_id}")
        extensions.append(f"cs1Label=EpisodeID")
        extensions.append(f"cs2={self.actor_id}")
        extensions.append(f"cs2Label=ActorID")
        extensions.append(f"cs3={self.mitre_technique or 'N/A'}")
        extensions.append(f"cs3Label=MITRETechnique")
        extensions.append(f"cn1={self.threat_score}")
        extensions.append(f"cn1Label=ThreatScore")
        extensions.append(f"msg={self.description}")
        
        return header + " ".join(extensions)
    
    def to_leef(self) -> str:
        """Convert to Log Event Extended Format (LEEF)"""
        header = f"LEEF:2.0|XORB|Cyber Range|1.0|{self.event_type}|"
        
        attributes = []
        attributes.append(f"devTime={self.timestamp}")
        attributes.append(f"src={self.target_ip or 'unknown'}")
        attributes.append(f"srcPort={self.target_port or 0}")
        attributes.append(f"eventId={self.event_id}")
        attributes.append(f"cat={self.category.value}")
        attributes.append(f"sev={self.severity.value}")
        attributes.append(f"actorType={self.actor_type}")
        attributes.append(f"actorId={self.actor_id}")
        attributes.append(f"action={self.action}")
        attributes.append(f"technique={self.technique}")
        attributes.append(f"episodeId={self.episode_id}")
        attributes.append(f"mitreId={self.mitre_technique or 'N/A'}")
        attributes.append(f"threatScore={self.threat_score}")
        attributes.append(f"confidence={self.confidence}")
        
        return header + "|".join(attributes)
    
    def _severity_to_cef_level(self) -> int:
        """Convert severity to CEF numeric level"""
        severity_map = {
            EventSeverity.INFO: 1,
            EventSeverity.LOW: 3,
            EventSeverity.MEDIUM: 5,
            EventSeverity.HIGH: 7,
            EventSeverity.CRITICAL: 10
        }
        return severity_map.get(self.severity, 1)

class SIEMForwarder:
    """High-performance SIEM event forwarding"""
    
    def __init__(self, config: SIEMConfig):
        self.config = config
        self.event_queue = queue.Queue(maxsize=config.burst_limit)
        self.session = None
        self.stats = {
            "events_sent": 0,
            "events_failed": 0,
            "last_send_time": None,
            "connection_errors": 0
        }
        
        # Rate limiting
        self.rate_limiter = RateLimiter(config.max_events_per_second)
        
        # Background processing
        self._stop_event = threading.Event()
        self._worker_thread = None
        
        # Initialize session
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize HTTP session with retry strategy"""
        self.session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "PUT"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set timeout
        self.session.timeout = self.config.timeout_seconds
        
        # SSL configuration
        if not self.config.verify_ssl:
            self.session.verify = False
        elif self.config.ssl_cert_path:
            self.session.cert = (self.config.ssl_cert_path, self.config.ssl_key_path)
        
        # Authentication
        self._setup_authentication()
    
    def _setup_authentication(self):
        """Setup authentication based on SIEM type"""
        auth_config = self.config.authentication
        
        if self.config.siem_type == SIEMType.SPLUNK:
            if "token" in auth_config:
                self.session.headers.update({
                    "Authorization": f"Splunk {auth_config['token']}"
                })
            elif "username" in auth_config and "password" in auth_config:
                self.session.auth = (auth_config["username"], auth_config["password"])
        
        elif self.config.siem_type == SIEMType.ELASTIC:
            if "api_key" in auth_config:
                self.session.headers.update({
                    "Authorization": f"ApiKey {auth_config['api_key']}"
                })
            elif "username" in auth_config and "password" in auth_config:
                self.session.auth = (auth_config["username"], auth_config["password"])
        
        elif self.config.siem_type == SIEMType.SENTINEL:
            if "workspace_id" in auth_config and "shared_key" in auth_config:
                # Azure Sentinel requires custom authentication
                pass  # Will be handled in send method
        
        elif self.config.siem_type == SIEMType.DATADOG:
            if "api_key" in auth_config:
                self.session.headers.update({
                    "DD-API-KEY": auth_config["api_key"]
                })
        
        # Add common headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "XORB-CyberRange/1.0"
        })
    
    def start(self):
        """Start background event processing"""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._event_processor)
            self._worker_thread.daemon = True
            self._worker_thread.start()
    
    def stop(self):
        """Stop background processing and flush remaining events"""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=30)
        
        # Flush remaining events
        self._flush_events()
    
    def send_event(self, event: SecurityEvent) -> bool:
        """Queue an event for sending to SIEM"""
        # Check severity filter
        if event.severity.value < self.config.min_severity.value:
            return True
        
        # Check category filter
        if self.config.event_categories and event.category not in self.config.event_categories:
            return True
        
        try:
            # Apply rate limiting
            if not self.rate_limiter.acquire():
                print(f"Rate limit exceeded, dropping event: {event.event_id}")
                return False
            
            self.event_queue.put(event, block=False)
            return True
            
        except queue.Full:
            print(f"Event queue full, dropping event: {event.event_id}")
            return False
    
    def _event_processor(self):
        """Background thread for processing events"""
        batch = []
        last_flush = time.time()
        
        while not self._stop_event.is_set():
            try:
                # Get event with timeout
                try:
                    event = self.event_queue.get(timeout=1.0)
                    batch.append(event)
                except queue.Empty:
                    pass
                
                # Check if we should flush
                current_time = time.time()
                should_flush = (
                    len(batch) >= self.config.batch_size or
                    (batch and current_time - last_flush >= self.config.flush_interval_seconds)
                )
                
                if should_flush:
                    self._send_batch(batch)
                    batch.clear()
                    last_flush = current_time
                
            except Exception as e:
                print(f"Error in event processor: {e}")
                time.sleep(1)
        
        # Final flush
        if batch:
            self._send_batch(batch)
    
    def _send_batch(self, events: List[SecurityEvent]):
        """Send a batch of events to SIEM"""
        if not events:
            return
        
        try:
            if self.config.siem_type == SIEMType.SPLUNK:
                self._send_to_splunk(events)
            elif self.config.siem_type == SIEMType.ELASTIC:
                self._send_to_elastic(events)
            elif self.config.siem_type == SIEMType.SENTINEL:
                self._send_to_sentinel(events)
            elif self.config.siem_type == SIEMType.DATADOG:
                self._send_to_datadog(events)
            elif self.config.siem_type == SIEMType.SYSLOG:
                self._send_to_syslog(events)
            else:
                self._send_generic_rest(events)
            
            self.stats["events_sent"] += len(events)
            self.stats["last_send_time"] = datetime.utcnow().isoformat()
            
        except Exception as e:
            print(f"Failed to send batch to SIEM: {e}")
            self.stats["events_failed"] += len(events)
            self.stats["connection_errors"] += 1
    
    def _send_to_splunk(self, events: List[SecurityEvent]):
        """Send events to Splunk HEC"""
        payload = []
        
        for event in events:
            splunk_event = {
                "time": datetime.fromisoformat(event.timestamp.replace('Z', '+00:00')).timestamp(),
                "source": event.source_component or "cyber-range",
                "sourcetype": self.config.source_type,
                "index": self.config.index_name,
                "event": asdict(event)
            }
            payload.append(splunk_event)
        
        # Send to Splunk HEC
        response = self.session.post(
            f"{self.config.endpoint_url}/services/collector/event",
            json={"event": payload}
        )
        response.raise_for_status()
    
    def _send_to_elastic(self, events: List[SecurityEvent]):
        """Send events to Elasticsearch"""
        # Build bulk request
        bulk_data = []
        
        for event in events:
            # Index metadata
            index_doc = {
                "index": {
                    "_index": self.config.index_name,
                    "_type": "_doc",
                    "_id": event.event_id
                }
            }
            bulk_data.append(json.dumps(index_doc))
            bulk_data.append(json.dumps(asdict(event)))
        
        bulk_payload = "\n".join(bulk_data) + "\n"
        
        response = self.session.post(
            f"{self.config.endpoint_url}/_bulk",
            data=bulk_payload,
            headers={"Content-Type": "application/x-ndjson"}
        )
        response.raise_for_status()
    
    def _send_to_sentinel(self, events: List[SecurityEvent]):
        """Send events to Azure Sentinel"""
        # Azure Sentinel Log Analytics API
        workspace_id = self.config.authentication["workspace_id"]
        shared_key = self.config.authentication["shared_key"]
        
        # Build signature for authentication
        payload = json.dumps([asdict(event) for event in events])
        content_length = len(payload)
        
        date = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
        string_to_hash = f"POST\n{content_length}\napplication/json\nx-ms-date:{date}\n/api/logs"
        
        decoded_key = base64.b64decode(shared_key)
        encoded_hash = base64.b64encode(
            hmac.new(decoded_key, string_to_hash.encode('utf-8'), hashlib.sha256).digest()
        ).decode()
        
        authorization = f"SharedKey {workspace_id}:{encoded_hash}"
        
        headers = {
            "Authorization": authorization,
            "Log-Type": "CyberRange",
            "x-ms-date": date,
            "time-generated-field": "timestamp"
        }
        
        url = f"https://{workspace_id}.ods.opinsights.azure.com/api/logs?api-version=2016-04-01"
        
        response = self.session.post(url, data=payload, headers=headers)
        response.raise_for_status()
    
    def _send_to_datadog(self, events: List[SecurityEvent]):
        """Send events to Datadog Logs API"""
        logs = []
        
        for event in events:
            log_entry = {
                "ddsource": "cyber-range",
                "ddtags": ",".join(event.tags),
                "hostname": event.source_system,
                "message": event.description,
                "service": "xorb-cyber-range",
                "timestamp": datetime.fromisoformat(event.timestamp.replace('Z', '+00:00')).timestamp() * 1000,
                "level": event.severity.value.upper(),
                **asdict(event)
            }
            logs.append(log_entry)
        
        response = self.session.post(
            f"{self.config.endpoint_url}/v1/input/{self.config.authentication.get('api_key', '')}",
            json=logs
        )
        response.raise_for_status()
    
    def _send_to_syslog(self, events: List[SecurityEvent]):
        """Send events via Syslog"""
        if not SYSLOG_AVAILABLE:
            raise RuntimeError("Syslog support not available")
        
        # Implementation would use syslogmp or similar library
        # This is a placeholder for the actual implementation
        for event in events:
            # Convert to CEF format
            cef_message = event.to_cef()
            # Send via syslog (implementation specific)
            print(f"SYSLOG: {cef_message}")
    
    def _send_generic_rest(self, events: List[SecurityEvent]):
        """Send events to generic REST endpoint"""
        payload = [asdict(event) for event in events]
        
        response = self.session.post(
            self.config.endpoint_url,
            json=payload
        )
        response.raise_for_status()
    
    def _flush_events(self):
        """Flush any remaining events in the queue"""
        remaining_events = []
        
        try:
            while True:
                event = self.event_queue.get_nowait()
                remaining_events.append(event)
        except queue.Empty:
            pass
        
        if remaining_events:
            self._send_batch(remaining_events)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get forwarding statistics"""
        return {
            "queue_size": self.event_queue.qsize(),
            "events_sent": self.stats["events_sent"],
            "events_failed": self.stats["events_failed"],
            "connection_errors": self.stats["connection_errors"],
            "last_send_time": self.stats["last_send_time"],
            "success_rate": self.stats["events_sent"] / max(self.stats["events_sent"] + self.stats["events_failed"], 1)
        }

class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, rate: int, burst: int = None):
        self.rate = rate
        self.burst = burst or rate
        self.tokens = self.burst
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def acquire(self) -> bool:
        """Acquire a token, returns True if allowed"""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Add tokens based on elapsed time
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            else:
                return False

# Event factory functions
def create_attack_event(episode_id: str, round_id: int, actor_id: str,
                       technique: str, target: str, success: bool,
                       mitre_technique: str = None) -> SecurityEvent:
    """Create a standardized attack event"""
    return SecurityEvent(
        event_id=f"attack_{int(time.time() * 1000000)}",
        timestamp=datetime.utcnow().isoformat() + 'Z',
        event_type="cyber_attack",
        category=EventCategory.ATTACK,
        severity=EventSeverity.HIGH if success else EventSeverity.MEDIUM,
        episode_id=episode_id,
        round_id=round_id,
        actor_type="red_agent",
        actor_id=actor_id,
        action=technique,
        technique=technique,
        target_system=target,
        outcome="success" if success else "failure",
        mitre_technique=mitre_technique,
        description=f"Attack technique {technique} against {target}"
    )

def create_defense_event(episode_id: str, round_id: int, actor_id: str,
                        defense_action: str, target: str, effectiveness: str) -> SecurityEvent:
    """Create a standardized defense event"""
    severity_map = {
        "high": EventSeverity.HIGH,
        "medium": EventSeverity.MEDIUM,
        "low": EventSeverity.LOW
    }
    
    return SecurityEvent(
        event_id=f"defense_{int(time.time() * 1000000)}",
        timestamp=datetime.utcnow().isoformat() + 'Z',
        event_type="cyber_defense",
        category=EventCategory.DEFENSE,
        severity=severity_map.get(effectiveness, EventSeverity.MEDIUM),
        episode_id=episode_id,
        round_id=round_id,
        actor_type="blue_agent",
        actor_id=actor_id,
        action=defense_action,
        target_system=target,
        outcome=effectiveness,
        description=f"Defense action {defense_action} deployed against {target}"
    )

def create_detection_event(episode_id: str, round_id: int, detection_source: str,
                          threat_description: str, confidence: float,
                          target_ip: str = None) -> SecurityEvent:
    """Create a standardized detection event"""
    severity = EventSeverity.HIGH if confidence > 0.8 else EventSeverity.MEDIUM if confidence > 0.5 else EventSeverity.LOW
    
    return SecurityEvent(
        event_id=f"detection_{int(time.time() * 1000000)}",
        timestamp=datetime.utcnow().isoformat() + 'Z',
        event_type="threat_detection",
        category=EventCategory.DETECTION,
        severity=severity,
        episode_id=episode_id,
        round_id=round_id,
        source_component=detection_source,
        target_ip=target_ip,
        confidence=confidence,
        description=threat_description,
        outcome="detected"
    )

if __name__ == "__main__":
    # Example usage and testing
    print("Testing SIEM integration...")
    
    # Create test configuration
    config = SIEMConfig(
        siem_type=SIEMType.GENERIC_REST,
        endpoint_url="http://localhost:8080/events",
        batch_size=5,
        flush_interval_seconds=2
    )
    
    # Initialize forwarder
    forwarder = SIEMForwarder(config)
    forwarder.start()
    
    try:
        # Create and send test events
        attack_event = create_attack_event(
            episode_id="test_ep_001",
            round_id=1,
            actor_id="red_agent_001",
            technique="SQL Injection",
            target="web_application",
            success=True,
            mitre_technique="T1190"
        )
        
        defense_event = create_defense_event(
            episode_id="test_ep_001",
            round_id=1,
            actor_id="blue_agent_001",
            defense_action="WAF Rule Update",
            target="web_application",
            effectiveness="high"
        )
        
        detection_event = create_detection_event(
            episode_id="test_ep_001",
            round_id=1,
            detection_source="IDS",
            threat_description="Suspicious SQL injection attempt detected",
            confidence=0.9,
            target_ip="192.168.1.100"
        )
        
        # Send events
        print("Sending test events...")
        forwarder.send_event(attack_event)
        forwarder.send_event(defense_event)
        forwarder.send_event(detection_event)
        
        # Wait for processing
        time.sleep(5)
        
        # Get statistics
        stats = forwarder.get_statistics()
        print(f"Forwarding statistics: {stats}")
        
        # Test CEF format
        print(f"CEF format: {attack_event.to_cef()}")
        print(f"LEEF format: {detection_event.to_leef()}")
        
    finally:
        forwarder.stop()
    
    print("SIEM integration test completed!")