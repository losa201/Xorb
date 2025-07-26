"""
Real-World Threat Intelligence Integration Engine

This module provides comprehensive threat intelligence integration with external feeds,
real-time threat correlation, IoC (Indicators of Compromise) management, and
automated threat hunting capabilities for the XORB ecosystem.
"""

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from urllib.parse import urlparse
import re

import aiohttp
import structlog
from prometheus_client import Counter, Gauge, Histogram

# Metrics
THREAT_INTEL_REQUESTS = Counter('xorb_threat_intel_requests_total', 'Threat intelligence requests', ['source', 'status'])
IOC_CACHE_SIZE = Gauge('xorb_ioc_cache_size', 'Number of cached IoCs', ['type'])
THREAT_CORRELATIONS = Counter('xorb_threat_correlations_total', 'Threat correlations found', ['correlation_type'])
INTEL_PROCESSING_TIME = Histogram('xorb_intel_processing_duration_seconds', 'Intel processing time')

logger = structlog.get_logger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""
    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IoC_Type(Enum):
    """Indicator of Compromise types."""
    IP_ADDRESS = "ip_address"
    DOMAIN = "domain"
    URL = "url"
    FILE_HASH = "file_hash"
    EMAIL = "email"
    USER_AGENT = "user_agent"
    CERTIFICATE = "certificate"
    REGISTRY_KEY = "registry_key"
    MUTEX = "mutex"
    YARA_RULE = "yara_rule"


class ThreatCategory(Enum):
    """Threat categories."""
    MALWARE = "malware"
    PHISHING = "phishing"
    BOTNET = "botnet"
    APT = "apt"
    RANSOMWARE = "ransomware"
    TROJAN = "trojan"
    BACKDOOR = "backdoor"
    EXPLOIT = "exploit"
    SUSPICIOUS = "suspicious"
    RECONNAISSANCE = "reconnaissance"


@dataclass
class IoC:
    """Indicator of Compromise data structure."""
    value: str
    ioc_type: IoC_Type
    threat_level: ThreatLevel = ThreatLevel.UNKNOWN
    confidence: float = 0.0  # 0.0 to 1.0
    
    # Threat context
    threat_category: Optional[ThreatCategory] = None
    threat_actor: Optional[str] = None
    campaign: Optional[str] = None
    malware_family: Optional[str] = None
    
    # Metadata
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    source: str = "unknown"
    description: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    
    # Verification
    verified: bool = False
    false_positive: bool = False
    
    # Context data
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Normalize IoC value based on type."""
        self.value = self._normalize_value()
    
    def _normalize_value(self) -> str:
        """Normalize IoC value based on type."""
        if self.ioc_type == IoC_Type.IP_ADDRESS:
            return self.value.strip().lower()
        elif self.ioc_type == IoC_Type.DOMAIN:
            return self.value.strip().lower()
        elif self.ioc_type == IoC_Type.URL:
            return self.value.strip()
        elif self.ioc_type == IoC_Type.FILE_HASH:
            return self.value.strip().lower()
        elif self.ioc_type == IoC_Type.EMAIL:
            return self.value.strip().lower()
        else:
            return self.value.strip()
    
    def get_id(self) -> str:
        """Get unique ID for this IoC."""
        data = f"{self.ioc_type.value}:{self.value}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def is_expired(self, max_age_days: int = 30) -> bool:
        """Check if IoC has expired."""
        age = time.time() - self.last_seen
        return age > (max_age_days * 24 * 3600)
    
    def update_seen_time(self):
        """Update last seen timestamp."""
        self.last_seen = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['tags'] = list(self.tags)
        return data


@dataclass
class ThreatIntelReport:
    """Threat intelligence report structure."""
    report_id: str = field(default_factory=lambda: str(time.time()))
    timestamp: float = field(default_factory=time.time)
    source: str = "xorb"
    
    # Report content
    title: str = ""
    summary: str = ""
    threat_actors: List[str] = field(default_factory=list)
    campaigns: List[str] = field(default_factory=list)
    malware_families: List[str] = field(default_factory=list)
    
    # IoCs
    iocs: List[IoC] = field(default_factory=list)
    
    # Metadata
    confidence: float = 0.0
    severity: ThreatLevel = ThreatLevel.UNKNOWN
    tags: Set[str] = field(default_factory=set)
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)


class IThreatIntelSource(ABC):
    """Interface for threat intelligence sources."""
    
    @abstractmethod
    async def fetch_iocs(self) -> List[IoC]:
        """Fetch IoCs from the source."""
        pass
    
    @abstractmethod
    async def lookup_ioc(self, ioc_value: str, ioc_type: IoC_Type) -> Optional[IoC]:
        """Lookup a specific IoC."""
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Get source name."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if source is available."""
        pass


class VirusTotalSource(IThreatIntelSource):
    """VirusTotal threat intelligence source."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.virustotal.com/vtapi/v2"
        self.session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get HTTP session."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def fetch_iocs(self) -> List[IoC]:
        """Fetch IoCs from VirusTotal feeds."""
        # VirusTotal doesn't have a direct feed API, so this would be implemented
        # based on specific VT API endpoints
        return []
    
    async def lookup_ioc(self, ioc_value: str, ioc_type: IoC_Type) -> Optional[IoC]:
        """Lookup IoC in VirusTotal."""
        try:
            session = await self._get_session()
            
            if ioc_type == IoC_Type.IP_ADDRESS:
                url = f"{self.base_url}/ip-address/report"
                params = {"apikey": self.api_key, "ip": ioc_value}
            elif ioc_type == IoC_Type.DOMAIN:
                url = f"{self.base_url}/domain/report"
                params = {"apikey": self.api_key, "domain": ioc_value}
            elif ioc_type == IoC_Type.URL:
                url = f"{self.base_url}/url/report"
                params = {"apikey": self.api_key, "resource": ioc_value}
            elif ioc_type == IoC_Type.FILE_HASH:
                url = f"{self.base_url}/file/report"
                params = {"apikey": self.api_key, "resource": ioc_value}
            else:
                return None
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_vt_response(ioc_value, ioc_type, data)
                    
            THREAT_INTEL_REQUESTS.labels(source="virustotal", status="success").inc()
            
        except Exception as e:
            logger.error("VirusTotal lookup failed", ioc=ioc_value, error=str(e))
            THREAT_INTEL_REQUESTS.labels(source="virustotal", status="error").inc()
        
        return None
    
    def _parse_vt_response(self, ioc_value: str, ioc_type: IoC_Type, data: Dict[str, Any]) -> Optional[IoC]:
        """Parse VirusTotal response."""
        if data.get("response_code") != 1:
            return None
        
        positives = data.get("positives", 0)
        total = data.get("total", 1)
        confidence = min(positives / total, 1.0) if total > 0 else 0.0
        
        # Determine threat level based on detection ratio
        if confidence >= 0.7:
            threat_level = ThreatLevel.HIGH
        elif confidence >= 0.4:
            threat_level = ThreatLevel.MEDIUM
        elif confidence >= 0.1:
            threat_level = ThreatLevel.LOW
        else:
            threat_level = ThreatLevel.UNKNOWN
        
        return IoC(
            value=ioc_value,
            ioc_type=ioc_type,
            threat_level=threat_level,
            confidence=confidence,
            source="virustotal",
            description=f"VT detections: {positives}/{total}",
            context={"vt_data": data}
        )
    
    def get_source_name(self) -> str:
        return "virustotal"
    
    async def is_available(self) -> bool:
        """Check if VirusTotal is available."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/file/report", 
                                 params={"apikey": self.api_key, "resource": "test"}) as response:
                return response.status in [200, 204]  # 204 = not found, but API is working
        except:
            return False


class OTXSource(IThreatIntelSource):
    """AlienVault OTX threat intelligence source."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://otx.alienvault.com/api/v1"
        self.session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get HTTP session with auth headers."""
        if not self.session:
            headers = {"X-OTX-API-KEY": self.api_key}
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session
    
    async def fetch_iocs(self) -> List[IoC]:
        """Fetch IoCs from OTX feeds."""
        iocs = []
        try:
            session = await self._get_session()
            
            # Get recent pulses
            async with session.get(f"{self.base_url}/pulses/subscribed") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for pulse in data.get("results", []):
                        for indicator in pulse.get("indicators", []):
                            ioc = self._parse_otx_indicator(indicator, pulse)
                            if ioc:
                                iocs.append(ioc)
            
            THREAT_INTEL_REQUESTS.labels(source="otx", status="success").inc()
            
        except Exception as e:
            logger.error("OTX fetch failed", error=str(e))
            THREAT_INTEL_REQUESTS.labels(source="otx", status="error").inc()
        
        return iocs
    
    async def lookup_ioc(self, ioc_value: str, ioc_type: IoC_Type) -> Optional[IoC]:
        """Lookup IoC in OTX."""
        try:
            session = await self._get_session()
            
            # Map IoC types to OTX endpoints
            type_mapping = {
                IoC_Type.IP_ADDRESS: "IPv4",
                IoC_Type.DOMAIN: "domain",
                IoC_Type.URL: "url",
                IoC_Type.FILE_HASH: "file"
            }
            
            if ioc_type not in type_mapping:
                return None
            
            otx_type = type_mapping[ioc_type]
            url = f"{self.base_url}/indicators/{otx_type}/{ioc_value}/general"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_otx_general(ioc_value, ioc_type, data)
            
            THREAT_INTEL_REQUESTS.labels(source="otx", status="success").inc()
            
        except Exception as e:
            logger.error("OTX lookup failed", ioc=ioc_value, error=str(e))
            THREAT_INTEL_REQUESTS.labels(source="otx", status="error").inc()
        
        return None
    
    def _parse_otx_indicator(self, indicator: Dict[str, Any], pulse: Dict[str, Any]) -> Optional[IoC]:
        """Parse OTX indicator."""
        indicator_type = indicator.get("type", "").lower()
        indicator_value = indicator.get("indicator", "")
        
        # Map OTX types to our types
        type_mapping = {
            "ipv4": IoC_Type.IP_ADDRESS,
            "ipv6": IoC_Type.IP_ADDRESS,
            "domain": IoC_Type.DOMAIN,
            "hostname": IoC_Type.DOMAIN,
            "url": IoC_Type.URL,
            "md5": IoC_Type.FILE_HASH,
            "sha1": IoC_Type.FILE_HASH,
            "sha256": IoC_Type.FILE_HASH,
            "email": IoC_Type.EMAIL
        }
        
        if indicator_type not in type_mapping:
            return None
        
        # Determine threat level from pulse tags
        tags = pulse.get("tags", [])
        threat_level = ThreatLevel.MEDIUM  # Default
        
        if any(tag in ["apt", "targeted"] for tag in tags):
            threat_level = ThreatLevel.HIGH
        elif any(tag in ["malware", "trojan", "backdoor"] for tag in tags):
            threat_level = ThreatLevel.HIGH
        elif any(tag in ["suspicious", "phishing"] for tag in tags):
            threat_level = ThreatLevel.MEDIUM
        
        return IoC(
            value=indicator_value,
            ioc_type=type_mapping[indicator_type],
            threat_level=threat_level,
            confidence=0.8,  # OTX generally has good quality
            source="otx",
            description=pulse.get("description", ""),
            tags=set(tags),
            context={
                "pulse_id": pulse.get("id"),
                "pulse_name": pulse.get("name"),
                "author": pulse.get("author_name")
            }
        )
    
    def _parse_otx_general(self, ioc_value: str, ioc_type: IoC_Type, data: Dict[str, Any]) -> IoC:
        """Parse OTX general response."""
        pulse_count = data.get("pulse_info", {}).get("count", 0)
        
        # Higher pulse count = higher confidence
        confidence = min(pulse_count / 10.0, 1.0)
        
        threat_level = ThreatLevel.MEDIUM
        if pulse_count >= 5:
            threat_level = ThreatLevel.HIGH
        elif pulse_count >= 2:
            threat_level = ThreatLevel.MEDIUM
        elif pulse_count >= 1:
            threat_level = ThreatLevel.LOW
        
        return IoC(
            value=ioc_value,
            ioc_type=ioc_type,
            threat_level=threat_level,
            confidence=confidence,
            source="otx",
            description=f"Found in {pulse_count} OTX pulses",
            context={"otx_data": data}
        )
    
    def get_source_name(self) -> str:
        return "otx"
    
    async def is_available(self) -> bool:
        """Check if OTX is available."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/user/me") as response:
                return response.status == 200
        except:
            return False


class MISPSource(IThreatIntelSource):
    """MISP threat intelligence source."""
    
    def __init__(self, base_url: str, auth_key: str, verify_ssl: bool = True):
        self.base_url = base_url.rstrip('/')
        self.auth_key = auth_key
        self.verify_ssl = verify_ssl
        self.session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get HTTP session with auth headers."""
        if not self.session:
            headers = {
                "Authorization": self.auth_key,
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            connector = aiohttp.TCPConnector(ssl=self.verify_ssl)
            self.session = aiohttp.ClientSession(headers=headers, connector=connector)
        return self.session
    
    async def fetch_iocs(self) -> List[IoC]:
        """Fetch IoCs from MISP."""
        iocs = []
        try:
            session = await self._get_session()
            
            # Get recent events
            url = f"{self.base_url}/events/index"
            async with session.get(url) as response:
                if response.status == 200:
                    events = await response.json()
                    
                    for event in events:
                        event_iocs = await self._fetch_event_iocs(event["Event"]["id"])
                        iocs.extend(event_iocs)
            
            THREAT_INTEL_REQUESTS.labels(source="misp", status="success").inc()
            
        except Exception as e:
            logger.error("MISP fetch failed", error=str(e))
            THREAT_INTEL_REQUESTS.labels(source="misp", status="error").inc()
        
        return iocs
    
    async def _fetch_event_iocs(self, event_id: str) -> List[IoC]:
        """Fetch IoCs from a specific MISP event."""
        iocs = []
        try:
            session = await self._get_session()
            url = f"{self.base_url}/events/view/{event_id}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    event_data = await response.json()
                    event = event_data.get("Event", {})
                    
                    for attribute in event.get("Attribute", []):
                        ioc = self._parse_misp_attribute(attribute, event)
                        if ioc:
                            iocs.append(ioc)
        except Exception as e:
            logger.error("MISP event fetch failed", event_id=event_id, error=str(e))
        
        return iocs
    
    async def lookup_ioc(self, ioc_value: str, ioc_type: IoC_Type) -> Optional[IoC]:
        """Lookup IoC in MISP."""
        try:
            session = await self._get_session()
            
            # Search for attributes
            search_data = {
                "value": ioc_value,
                "returnFormat": "json"
            }
            
            url = f"{self.base_url}/attributes/restSearch"
            async with session.post(url, json=search_data) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("response", {}).get("Attribute"):
                        # Use first matching attribute
                        attribute = data["response"]["Attribute"][0]
                        return self._parse_misp_attribute(attribute)
            
            THREAT_INTEL_REQUESTS.labels(source="misp", status="success").inc()
            
        except Exception as e:
            logger.error("MISP lookup failed", ioc=ioc_value, error=str(e))
            THREAT_INTEL_REQUESTS.labels(source="misp", status="error").inc()
        
        return None
    
    def _parse_misp_attribute(self, attribute: Dict[str, Any], event: Optional[Dict[str, Any]] = None) -> Optional[IoC]:
        """Parse MISP attribute to IoC."""
        attr_type = attribute.get("type", "").lower()
        attr_value = attribute.get("value", "")
        
        # Map MISP types to our types
        type_mapping = {
            "ip-src": IoC_Type.IP_ADDRESS,
            "ip-dst": IoC_Type.IP_ADDRESS,
            "domain": IoC_Type.DOMAIN,
            "hostname": IoC_Type.DOMAIN,
            "url": IoC_Type.URL,
            "md5": IoC_Type.FILE_HASH,
            "sha1": IoC_Type.FILE_HASH,
            "sha256": IoC_Type.FILE_HASH,
            "email-src": IoC_Type.EMAIL,
            "email-dst": IoC_Type.EMAIL
        }
        
        if attr_type not in type_mapping:
            return None
        
        # Determine threat level from event info
        threat_level = ThreatLevel.MEDIUM
        if event:
            threat_level_map = {
                "1": ThreatLevel.LOW,
                "2": ThreatLevel.MEDIUM,
                "3": ThreatLevel.HIGH,
                "4": ThreatLevel.CRITICAL
            }
            event_threat_level = event.get("threat_level_id", "2")
            threat_level = threat_level_map.get(event_threat_level, ThreatLevel.MEDIUM)
        
        tags = set()
        if event:
            for tag in event.get("Tag", []):
                tags.add(tag.get("name", ""))
        
        return IoC(
            value=attr_value,
            ioc_type=type_mapping[attr_type],
            threat_level=threat_level,
            confidence=0.9,  # MISP generally has high quality
            source="misp",
            description=attribute.get("comment", ""),
            tags=tags,
            context={
                "misp_attribute_id": attribute.get("id"),
                "misp_event_id": event.get("id") if event else None,
                "category": attribute.get("category")
            }
        )
    
    def get_source_name(self) -> str:
        return "misp"
    
    async def is_available(self) -> bool:
        """Check if MISP is available."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/users/view/me") as response:
                return response.status == 200
        except:
            return False


class ThreatIntelligenceEngine:
    """Main threat intelligence engine."""
    
    def __init__(self):
        self.sources: List[IThreatIntelSource] = []
        self.ioc_cache: Dict[str, IoC] = {}
        self.cache_max_age = 24 * 3600  # 24 hours
        self.running = False
        
        # Correlation rules
        self.correlation_rules = []
        self._setup_default_correlation_rules()
    
    def add_source(self, source: IThreatIntelSource):
        """Add a threat intelligence source."""
        self.sources.append(source)
        logger.info("Added threat intelligence source", source=source.get_source_name())
    
    def _setup_default_correlation_rules(self):
        """Setup default correlation rules."""
        self.correlation_rules = [
            {
                "name": "same_network_correlation",
                "description": "Correlate IPs in same network",
                "rule": self._correlate_same_network
            },
            {
                "name": "domain_subdomain_correlation", 
                "description": "Correlate domains and subdomains",
                "rule": self._correlate_domain_hierarchy
            },
            {
                "name": "hash_family_correlation",
                "description": "Correlate file hashes by malware family",
                "rule": self._correlate_hash_families
            }
        ]
    
    async def start_intelligence_engine(self):
        """Start the threat intelligence engine."""
        self.running = True
        
        # Start background tasks
        feed_task = asyncio.create_task(self._feed_update_loop())
        correlation_task = asyncio.create_task(self._correlation_loop())
        cleanup_task = asyncio.create_task(self._cache_cleanup_loop())
        
        logger.info("Threat intelligence engine started")
        
        try:
            await asyncio.gather(feed_task, correlation_task, cleanup_task)
        except asyncio.CancelledError:
            logger.info("Threat intelligence engine stopped")
    
    async def stop_intelligence_engine(self):
        """Stop the threat intelligence engine."""
        self.running = False
    
    async def _feed_update_loop(self):
        """Update feeds from all sources."""
        while self.running:
            try:
                await self._update_feeds()
                await asyncio.sleep(3600)  # Update every hour
            except Exception as e:
                logger.error("Feed update failed", error=str(e))
                await asyncio.sleep(600)  # Retry in 10 minutes
    
    async def _correlation_loop(self):
        """Run correlation analysis."""
        while self.running:
            try:
                await self._run_correlations()
                await asyncio.sleep(1800)  # Run correlations every 30 minutes
            except Exception as e:
                logger.error("Correlation analysis failed", error=str(e))
                await asyncio.sleep(600)
    
    async def _cache_cleanup_loop(self):
        """Clean up expired cache entries."""
        while self.running:
            try:
                await self._cleanup_cache()
                await asyncio.sleep(3600)  # Cleanup every hour
            except Exception as e:
                logger.error("Cache cleanup failed", error=str(e))
                await asyncio.sleep(3600)
    
    @INTEL_PROCESSING_TIME.time()
    async def _update_feeds(self):
        """Update IoC feeds from all sources."""
        total_new_iocs = 0
        
        for source in self.sources:
            try:
                if not await source.is_available():
                    logger.warning("Source unavailable", source=source.get_source_name())
                    continue
                
                iocs = await source.fetch_iocs()
                
                for ioc in iocs:
                    ioc_id = ioc.get_id()
                    
                    if ioc_id in self.ioc_cache:
                        # Update existing IoC
                        self.ioc_cache[ioc_id].update_seen_time()
                        # Merge additional context
                        self.ioc_cache[ioc_id].context.update(ioc.context)
                    else:
                        # Add new IoC
                        self.ioc_cache[ioc_id] = ioc
                        total_new_iocs += 1
                
                logger.info("Updated feed", 
                           source=source.get_source_name(), 
                           iocs_fetched=len(iocs))
                
            except Exception as e:
                logger.error("Source update failed", 
                           source=source.get_source_name(), 
                           error=str(e))
        
        # Update metrics
        for ioc_type in IoC_Type:
            count = len([ioc for ioc in self.ioc_cache.values() if ioc.ioc_type == ioc_type])
            IOC_CACHE_SIZE.labels(type=ioc_type.value).set(count)
        
        logger.info("Feed update completed", 
                   total_iocs=len(self.ioc_cache),
                   new_iocs=total_new_iocs)
    
    async def lookup_ioc(self, ioc_value: str, ioc_type: IoC_Type) -> Optional[IoC]:
        """Lookup IoC across all sources."""
        # Check cache first
        test_ioc = IoC(value=ioc_value, ioc_type=ioc_type)
        ioc_id = test_ioc.get_id()
        
        if ioc_id in self.ioc_cache:
            cached_ioc = self.ioc_cache[ioc_id]
            if not cached_ioc.is_expired():
                cached_ioc.update_seen_time()
                return cached_ioc
        
        # Query sources
        best_ioc = None
        best_confidence = 0.0
        
        for source in self.sources:
            try:
                ioc = await source.lookup_ioc(ioc_value, ioc_type)
                if ioc and ioc.confidence > best_confidence:
                    best_ioc = ioc
                    best_confidence = ioc.confidence
            except Exception as e:
                logger.error("Source lookup failed", 
                           source=source.get_source_name(),
                           ioc=ioc_value,
                           error=str(e))
        
        if best_ioc:
            # Cache the result
            self.ioc_cache[best_ioc.get_id()] = best_ioc
        
        return best_ioc
    
    async def bulk_lookup(self, ioc_list: List[tuple]) -> Dict[str, Optional[IoC]]:
        """Bulk lookup multiple IoCs."""
        results = {}
        
        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(10)
        
        async def lookup_single(ioc_value: str, ioc_type: IoC_Type):
            async with semaphore:
                result = await self.lookup_ioc(ioc_value, ioc_type)
                results[f"{ioc_type.value}:{ioc_value}"] = result
        
        tasks = [lookup_single(value, ioc_type) for value, ioc_type in ioc_list]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def _run_correlations(self):
        """Run correlation analysis on cached IoCs."""
        if len(self.ioc_cache) < 2:
            return
        
        iocs_list = list(self.ioc_cache.values())
        
        for rule in self.correlation_rules:
            try:
                correlations = await rule["rule"](iocs_list)
                if correlations:
                    THREAT_CORRELATIONS.labels(correlation_type=rule["name"]).inc(len(correlations))
                    logger.info("Found correlations", 
                               rule=rule["name"],
                               count=len(correlations))
            except Exception as e:
                logger.error("Correlation rule failed", 
                           rule=rule["name"],
                           error=str(e))
    
    async def _correlate_same_network(self, iocs: List[IoC]) -> List[Dict[str, Any]]:
        """Correlate IPs in the same network."""
        correlations = []
        ip_iocs = [ioc for ioc in iocs if ioc.ioc_type == IoC_Type.IP_ADDRESS]
        
        # Group IPs by /24 network
        networks = {}
        for ioc in ip_iocs:
            try:
                ip_parts = ioc.value.split('.')
                if len(ip_parts) == 4:
                    network = '.'.join(ip_parts[:3]) + '.0/24'
                    if network not in networks:
                        networks[network] = []
                    networks[network].append(ioc)
            except:
                continue
        
        # Find networks with multiple IPs
        for network, network_iocs in networks.items():
            if len(network_iocs) > 1:
                correlations.append({
                    "type": "same_network",
                    "network": network,
                    "iocs": [ioc.value for ioc in network_iocs],
                    "count": len(network_iocs)
                })
        
        return correlations
    
    async def _correlate_domain_hierarchy(self, iocs: List[IoC]) -> List[Dict[str, Any]]:
        """Correlate domains and subdomains."""
        correlations = []
        domain_iocs = [ioc for ioc in iocs if ioc.ioc_type == IoC_Type.DOMAIN]
        
        # Group by parent domain
        domain_groups = {}
        for ioc in domain_iocs:
            parts = ioc.value.split('.')
            if len(parts) >= 2:
                parent = '.'.join(parts[-2:])  # Get main domain
                if parent not in domain_groups:
                    domain_groups[parent] = []
                domain_groups[parent].append(ioc)
        
        # Find groups with multiple subdomains
        for parent, group_iocs in domain_groups.items():
            if len(group_iocs) > 1:
                correlations.append({
                    "type": "domain_hierarchy",
                    "parent_domain": parent,
                    "subdomains": [ioc.value for ioc in group_iocs],
                    "count": len(group_iocs)
                })
        
        return correlations
    
    async def _correlate_hash_families(self, iocs: List[IoC]) -> List[Dict[str, Any]]:
        """Correlate file hashes by malware family."""
        correlations = []
        hash_iocs = [ioc for ioc in iocs if ioc.ioc_type == IoC_Type.FILE_HASH]
        
        # Group by malware family
        family_groups = {}
        for ioc in hash_iocs:
            if ioc.malware_family:
                family = ioc.malware_family.lower()
                if family not in family_groups:
                    family_groups[family] = []
                family_groups[family].append(ioc)
        
        # Find families with multiple hashes
        for family, group_iocs in family_groups.items():
            if len(group_iocs) > 1:
                correlations.append({
                    "type": "malware_family",
                    "family": family,
                    "hashes": [ioc.value for ioc in group_iocs],
                    "count": len(group_iocs)
                })
        
        return correlations
    
    async def _cleanup_cache(self):
        """Clean up expired cache entries."""
        before_count = len(self.ioc_cache)
        
        # Remove expired IoCs
        expired_ids = [
            ioc_id for ioc_id, ioc in self.ioc_cache.items()
            if ioc.is_expired()
        ]
        
        for ioc_id in expired_ids:
            del self.ioc_cache[ioc_id]
        
        after_count = len(self.ioc_cache)
        
        if expired_ids:
            logger.info("Cache cleanup completed",
                       removed=len(expired_ids),
                       before=before_count,
                       after=after_count)
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "total_iocs": len(self.ioc_cache),
            "sources": len(self.sources),
            "by_type": {},
            "by_threat_level": {},
            "by_source": {}
        }
        
        for ioc in self.ioc_cache.values():
            # By type
            ioc_type = ioc.ioc_type.value
            stats["by_type"][ioc_type] = stats["by_type"].get(ioc_type, 0) + 1
            
            # By threat level
            threat_level = ioc.threat_level.value
            stats["by_threat_level"][threat_level] = stats["by_threat_level"].get(threat_level, 0) + 1
            
            # By source
            source = ioc.source
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
        
        return stats
    
    async def generate_threat_report(self, timeframe_hours: int = 24) -> ThreatIntelReport:
        """Generate threat intelligence report."""
        cutoff_time = time.time() - (timeframe_hours * 3600)
        
        recent_iocs = [
            ioc for ioc in self.ioc_cache.values()
            if ioc.last_seen >= cutoff_time
        ]
        
        # Analyze threat actors and campaigns
        threat_actors = set()
        campaigns = set()
        malware_families = set()
        
        for ioc in recent_iocs:
            if ioc.threat_actor:
                threat_actors.add(ioc.threat_actor)
            if ioc.campaign:
                campaigns.add(ioc.campaign)
            if ioc.malware_family:
                malware_families.add(ioc.malware_family)
        
        # Calculate overall threat level
        high_threat_count = len([ioc for ioc in recent_iocs if ioc.threat_level == ThreatLevel.HIGH])
        critical_threat_count = len([ioc for ioc in recent_iocs if ioc.threat_level == ThreatLevel.CRITICAL])
        
        if critical_threat_count > 0:
            overall_severity = ThreatLevel.CRITICAL
        elif high_threat_count > len(recent_iocs) * 0.3:
            overall_severity = ThreatLevel.HIGH
        elif high_threat_count > 0:
            overall_severity = ThreatLevel.MEDIUM
        else:
            overall_severity = ThreatLevel.LOW
        
        return ThreatIntelReport(
            title=f"XORB Threat Intelligence Report - {timeframe_hours}h",
            summary=f"Analyzed {len(recent_iocs)} IoCs from {len(self.sources)} sources",
            threat_actors=list(threat_actors),
            campaigns=list(campaigns),
            malware_families=list(malware_families),
            iocs=recent_iocs,
            confidence=0.8,
            severity=overall_severity,
            tags={"automated", "xorb_generated"},
            context={
                "timeframe_hours": timeframe_hours,
                "total_sources": len(self.sources),
                "high_threat_iocs": high_threat_count,
                "critical_threat_iocs": critical_threat_count
            }
        )


# Global threat intelligence engine
threat_intel_engine = ThreatIntelligenceEngine()


async def initialize_threat_intelligence():
    """Initialize the threat intelligence engine."""
    await threat_intel_engine.start_intelligence_engine()


async def shutdown_threat_intelligence():
    """Shutdown the threat intelligence engine."""
    await threat_intel_engine.stop_intelligence_engine()


def get_threat_intel_engine() -> ThreatIntelligenceEngine:
    """Get the global threat intelligence engine."""
    return threat_intel_engine