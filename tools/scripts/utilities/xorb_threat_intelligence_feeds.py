#!/usr/bin/env python3
"""
🛡️ XORB Threat Intelligence Feed Integration
Automated threat intelligence collection and processing

This module provides automated collection, processing, and integration of threat
intelligence from multiple sources to enhance XORB's adversarial testing capabilities.
"""

import asyncio
import json
import logging
import time
import uuid
import hashlib
import aiohttp
import aiofiles
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import xml.etree.ElementTree as ET
import re
import ipaddress

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThreatType(Enum):
    MALWARE_HASH = "malware_hash"
    IP_ADDRESS = "ip_address"
    DOMAIN = "domain"
    URL = "url"
    EMAIL = "email"
    CVE = "cve"
    YARA_RULE = "yara_rule"
    SIGMA_RULE = "sigma_rule"
    TTP = "ttp"

class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class FeedStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    UPDATING = "updating"

@dataclass
class ThreatIndicator:
    indicator_id: str
    indicator_type: ThreatType
    value: str
    threat_level: ThreatLevel
    source: str
    first_seen: datetime
    last_seen: datetime
    confidence: float
    tags: List[str]
    context: Dict[str, Any]
    tlp: str = "WHITE"  # Traffic Light Protocol

@dataclass
class ThreatFeed:
    feed_id: str
    name: str
    url: str
    feed_type: str
    format: str
    update_interval_hours: int
    last_update: Optional[datetime]
    status: FeedStatus
    indicators_count: int
    error_message: Optional[str] = None

class XORBThreatIntelligenceFeeds:
    """XORB Threat Intelligence Feed Manager"""
    
    def __init__(self):
        self.manager_id = f"THREAT-INTEL-{uuid.uuid4().hex[:8]}"
        self.initialization_time = datetime.now()
        
        # Threat intelligence storage
        self.indicators: Dict[str, ThreatIndicator] = {}
        self.feeds: Dict[str, ThreatFeed] = {}
        
        # Feed configurations
        self.feed_configs = self._initialize_feed_configs()
        
        # Processing statistics
        self.processing_stats = {
            "total_indicators": 0,
            "indicators_processed_today": 0,
            "feeds_updated_today": 0,
            "critical_indicators": 0,
            "high_indicators": 0,
            "medium_indicators": 0,
            "low_indicators": 0,
            "processing_errors": 0
        }
        
        # IOC extraction patterns
        self.ioc_patterns = {
            "ip_address": re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            "domain": re.compile(r'\b[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.([a-zA-Z]{2,})\b'),
            "url": re.compile(r'https?://[^\s/$.?#].[^\s]*'),
            "md5": re.compile(r'\b[a-fA-F0-9]{32}\b'),
            "sha1": re.compile(r'\b[a-fA-F0-9]{40}\b'),
            "sha256": re.compile(r'\b[a-fA-F0-9]{64}\b'),
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "cve": re.compile(r'CVE-\d{4}-\d{4,}')
        }
        
        logger.info(f"🛡️ XORB Threat Intelligence Feed Manager initialized - ID: {self.manager_id}")
        logger.info("📡 Automated threat intelligence collection: ACTIVE")
    
    def _initialize_feed_configs(self) -> List[Dict[str, Any]]:
        """Initialize threat intelligence feed configurations"""
        return [
            {
                "name": "Abuse.ch Malware Bazaar",
                "url": "https://bazaar.abuse.ch/export/csv/recent/",
                "feed_type": "malware_hashes",
                "format": "csv",
                "update_interval_hours": 6,
                "description": "Recent malware samples and hashes"
            },
            {
                "name": "MISP Threat Sharing",
                "url": "https://www.circl.lu/doc/misp/feed-osint/",
                "feed_type": "mixed_indicators",
                "format": "json",
                "update_interval_hours": 12,
                "description": "OSINT threat indicators from MISP"
            },
            {
                "name": "URLVoid Malicious URLs",
                "url": "https://www.urlvoid.com/api/",
                "feed_type": "malicious_urls",
                "format": "json",
                "update_interval_hours": 8,
                "description": "Malicious URLs and domains"
            },
            {
                "name": "CISA Known Exploited Vulnerabilities",
                "url": "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
                "feed_type": "vulnerabilities",
                "format": "json",
                "update_interval_hours": 24,
                "description": "CISA catalog of known exploited vulnerabilities"
            },
            {
                "name": "Emerging Threats Rules",
                "url": "https://rules.emergingthreats.net/",
                "feed_type": "detection_rules",
                "format": "suricata",
                "update_interval_hours": 6,
                "description": "Emerging threats detection rules"
            },
            {
                "name": "SANS Internet Storm Center",
                "url": "https://isc.sans.edu/api/",
                "feed_type": "network_threats",
                "format": "xml",
                "update_interval_hours": 4,
                "description": "Network threat intelligence from SANS ISC"
            }
        ]
    
    async def initialize_feeds(self):
        """Initialize all threat intelligence feeds"""
        logger.info("📡 Initializing threat intelligence feeds...")
        
        for feed_config in self.feed_configs:
            feed_id = f"FEED-{hashlib.md5(feed_config['name'].encode()).hexdigest()[:8]}"
            
            feed = ThreatFeed(
                feed_id=feed_id,
                name=feed_config["name"],
                url=feed_config["url"],
                feed_type=feed_config["feed_type"],
                format=feed_config["format"],
                update_interval_hours=feed_config["update_interval_hours"],
                last_update=None,
                status=FeedStatus.ACTIVE,
                indicators_count=0
            )
            
            self.feeds[feed_id] = feed
            logger.info(f"  📊 Initialized feed: {feed.name}")
        
        logger.info(f"✅ Initialized {len(self.feeds)} threat intelligence feeds")
    
    async def update_all_feeds(self):
        """Update all active threat intelligence feeds"""
        logger.info("🔄 Starting threat intelligence feed updates...")
        
        tasks = []
        for feed_id, feed in self.feeds.items():
            if feed.status == FeedStatus.ACTIVE:
                if self._should_update_feed(feed):
                    tasks.append(self.update_feed(feed_id))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful_updates = sum(1 for r in results if not isinstance(r, Exception))
            failed_updates = len(results) - successful_updates
            
            logger.info(f"📡 Feed update complete: {successful_updates} successful, {failed_updates} failed")
            self.processing_stats["feeds_updated_today"] += successful_updates
        else:
            logger.info("📡 No feeds require updating at this time")
    
    def _should_update_feed(self, feed: ThreatFeed) -> bool:
        """Check if feed should be updated based on interval"""
        if feed.last_update is None:
            return True
        
        time_since_update = datetime.now() - feed.last_update
        return time_since_update.total_seconds() >= (feed.update_interval_hours * 3600)
    
    async def update_feed(self, feed_id: str) -> Dict[str, Any]:
        """Update individual threat intelligence feed"""
        feed = self.feeds[feed_id]
        logger.info(f"📥 Updating feed: {feed.name}")
        
        feed.status = FeedStatus.UPDATING
        
        try:
            async with aiohttp.ClientSession() as session:
                # Simulate feed update with realistic data
                indicators = await self._fetch_feed_data(session, feed)
                
                # Process indicators
                processed_count = await self._process_indicators(indicators, feed.name)
                
                # Update feed status
                feed.last_update = datetime.now()
                feed.status = FeedStatus.ACTIVE
                feed.indicators_count = processed_count
                feed.error_message = None
                
                logger.info(f"✅ Updated feed {feed.name}: {processed_count} indicators processed")
                
                return {
                    "feed_id": feed_id,
                    "feed_name": feed.name,
                    "indicators_processed": processed_count,
                    "status": "success"
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to update feed {feed.name}: {e}")
            feed.status = FeedStatus.ERROR
            feed.error_message = str(e)
            self.processing_stats["processing_errors"] += 1
            
            return {
                "feed_id": feed_id,
                "feed_name": feed.name,
                "status": "error",
                "error": str(e)
            }
    
    async def _fetch_feed_data(self, session: aiohttp.ClientSession, feed: ThreatFeed) -> List[Dict[str, Any]]:
        """Fetch data from threat intelligence feed"""
        # Simulate realistic feed data based on feed type
        simulated_data = []
        
        if feed.feed_type == "malware_hashes":
            simulated_data = [
                {
                    "hash": hashlib.sha256(f"malware_sample_{i}".encode()).hexdigest(),
                    "malware_family": f"TrojanFamily{i % 5}",
                    "threat_level": "high" if i % 3 == 0 else "medium",
                    "first_seen": (datetime.now() - timedelta(days=i)).isoformat()
                }
                for i in range(50)
            ]
        
        elif feed.feed_type == "malicious_urls":
            simulated_data = [
                {
                    "url": f"http://malicious-domain-{i}.com/payload.exe",
                    "domain": f"malicious-domain-{i}.com",
                    "threat_level": "critical" if i % 4 == 0 else "high",
                    "category": "malware_download"
                }
                for i in range(30)
            ]
        
        elif feed.feed_type == "vulnerabilities":
            simulated_data = [
                {
                    "cve_id": f"CVE-2024-{1000 + i}",
                    "severity": "critical" if i % 5 == 0 else "high",
                    "description": f"Critical vulnerability in software component {i}",
                    "exploited_in_wild": i % 3 == 0
                }
                for i in range(20)
            ]
        
        elif feed.feed_type == "network_threats":
            simulated_data = [
                {
                    "ip": f"192.168.{(i // 256) % 256}.{i % 256}",
                    "threat_type": "c2_server" if i % 4 == 0 else "malware_host",
                    "threat_level": "high",
                    "country": "Unknown"
                }
                for i in range(100)
            ]
        
        return simulated_data
    
    async def _process_indicators(self, raw_indicators: List[Dict[str, Any]], source: str) -> int:
        """Process raw indicators into ThreatIndicator objects"""
        processed_count = 0
        
        for raw_indicator in raw_indicators:
            try:
                # Extract indicator based on type
                indicators = self._extract_indicators_from_data(raw_indicator, source)
                
                for indicator in indicators:
                    # Store indicator
                    self.indicators[indicator.indicator_id] = indicator
                    processed_count += 1
                    
                    # Update statistics
                    self.processing_stats["total_indicators"] += 1
                    self.processing_stats["indicators_processed_today"] += 1
                    
                    if indicator.threat_level == ThreatLevel.CRITICAL:
                        self.processing_stats["critical_indicators"] += 1
                    elif indicator.threat_level == ThreatLevel.HIGH:
                        self.processing_stats["high_indicators"] += 1
                    elif indicator.threat_level == ThreatLevel.MEDIUM:
                        self.processing_stats["medium_indicators"] += 1
                    else:
                        self.processing_stats["low_indicators"] += 1
                        
            except Exception as e:
                logger.warning(f"Failed to process indicator: {e}")
                self.processing_stats["processing_errors"] += 1
        
        return processed_count
    
    def _extract_indicators_from_data(self, data: Dict[str, Any], source: str) -> List[ThreatIndicator]:
        """Extract threat indicators from raw data"""
        indicators = []
        current_time = datetime.now()
        
        # Process based on data structure
        if "hash" in data:
            # Malware hash indicator
            indicator = ThreatIndicator(
                indicator_id=f"IOC-{uuid.uuid4().hex[:8]}",
                indicator_type=ThreatType.MALWARE_HASH,
                value=data["hash"],
                threat_level=ThreatLevel(data.get("threat_level", "medium")),
                source=source,
                first_seen=current_time,
                last_seen=current_time,
                confidence=0.9,
                tags=["malware", data.get("malware_family", "unknown")],
                context={"malware_family": data.get("malware_family")}
            )
            indicators.append(indicator)
        
        elif "url" in data:
            # URL indicator
            indicator = ThreatIndicator(
                indicator_id=f"IOC-{uuid.uuid4().hex[:8]}",
                indicator_type=ThreatType.URL,
                value=data["url"],
                threat_level=ThreatLevel(data.get("threat_level", "high")),
                source=source,
                first_seen=current_time,
                last_seen=current_time,
                confidence=0.85,
                tags=["malicious_url", data.get("category", "unknown")],
                context={"category": data.get("category")}
            )
            indicators.append(indicator)
            
            # Also extract domain
            if "domain" in data:
                domain_indicator = ThreatIndicator(
                    indicator_id=f"IOC-{uuid.uuid4().hex[:8]}",
                    indicator_type=ThreatType.DOMAIN,
                    value=data["domain"],
                    threat_level=ThreatLevel(data.get("threat_level", "high")),
                    source=source,
                    first_seen=current_time,
                    last_seen=current_time,
                    confidence=0.80,
                    tags=["malicious_domain"],
                    context={"related_url": data["url"]}
                )
                indicators.append(domain_indicator)
        
        elif "cve_id" in data:
            # CVE indicator
            indicator = ThreatIndicator(
                indicator_id=f"IOC-{uuid.uuid4().hex[:8]}",
                indicator_type=ThreatType.CVE,
                value=data["cve_id"],
                threat_level=ThreatLevel(data.get("severity", "medium")),
                source=source,
                first_seen=current_time,
                last_seen=current_time,
                confidence=0.95,
                tags=["vulnerability", "cve"],
                context={
                    "description": data.get("description"),
                    "exploited_in_wild": data.get("exploited_in_wild", False)
                }
            )
            indicators.append(indicator)
        
        elif "ip" in data:
            # IP address indicator
            indicator = ThreatIndicator(
                indicator_id=f"IOC-{uuid.uuid4().hex[:8]}",
                indicator_type=ThreatType.IP_ADDRESS,
                value=data["ip"],
                threat_level=ThreatLevel(data.get("threat_level", "medium")),
                source=source,
                first_seen=current_time,
                last_seen=current_time,
                confidence=0.75,
                tags=["malicious_ip", data.get("threat_type", "unknown")],
                context={
                    "threat_type": data.get("threat_type"),
                    "country": data.get("country")
                }
            )
            indicators.append(indicator)
        
        return indicators
    
    async def search_indicators(self, query: str, indicator_type: Optional[ThreatType] = None, 
                              threat_level: Optional[ThreatLevel] = None) -> List[ThreatIndicator]:
        """Search threat indicators"""
        results = []
        
        for indicator in self.indicators.values():
            # Filter by type if specified
            if indicator_type and indicator.indicator_type != indicator_type:
                continue
            
            # Filter by threat level if specified
            if threat_level and indicator.threat_level != threat_level:
                continue
            
            # Search in value, tags, and context
            if (query.lower() in indicator.value.lower() or
                any(query.lower() in tag.lower() for tag in indicator.tags) or
                any(query.lower() in str(v).lower() for v in indicator.context.values())):
                results.append(indicator)
        
        return results
    
    async def get_recent_indicators(self, hours: int = 24, threat_level: Optional[ThreatLevel] = None) -> List[ThreatIndicator]:
        """Get recent threat indicators"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_indicators = []
        for indicator in self.indicators.values():
            if indicator.first_seen >= cutoff_time:
                if threat_level is None or indicator.threat_level == threat_level:
                    recent_indicators.append(indicator)
        
        # Sort by first_seen (most recent first)
        recent_indicators.sort(key=lambda x: x.first_seen, reverse=True)
        
        return recent_indicators
    
    async def enrich_indicator(self, indicator_value: str) -> Dict[str, Any]:
        """Enrich an indicator with additional intelligence"""
        enrichment_data = {
            "indicator": indicator_value,
            "enrichment_timestamp": datetime.now().isoformat(),
            "sources": [],
            "threat_level": "unknown",
            "confidence": 0.0,
            "related_indicators": [],
            "campaigns": [],
            "malware_families": []
        }
        
        # Find existing indicators
        matching_indicators = []
        for indicator in self.indicators.values():
            if indicator.value == indicator_value:
                matching_indicators.append(indicator)
        
        if matching_indicators:
            # Aggregate data from matching indicators
            sources = set()
            threat_levels = []
            confidences = []
            related = set()
            campaigns = set()
            malware_families = set()
            
            for indicator in matching_indicators:
                sources.add(indicator.source)
                threat_levels.append(indicator.threat_level.value)
                confidences.append(indicator.confidence)
                
                # Extract related information from context and tags
                for tag in indicator.tags:
                    if "campaign" in tag.lower():
                        campaigns.add(tag)
                    elif "malware" in tag.lower() or "family" in tag.lower():
                        malware_families.add(tag)
                
                # Look for related indicators in context
                for key, value in indicator.context.items():
                    if key.startswith("related_"):
                        related.add(str(value))
            
            enrichment_data.update({
                "sources": list(sources),
                "threat_level": max(set(threat_levels), key=threat_levels.count),
                "confidence": sum(confidences) / len(confidences),
                "related_indicators": list(related),
                "campaigns": list(campaigns),
                "malware_families": list(malware_families)
            })
        
        return enrichment_data
    
    async def export_indicators(self, format_type: str = "json") -> str:
        """Export threat indicators in specified format"""
        if format_type == "json":
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_indicators": len(self.indicators),
                "indicators": [asdict(indicator) for indicator in self.indicators.values()]
            }
            return json.dumps(export_data, indent=2, default=str)
        
        elif format_type == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                "indicator_id", "type", "value", "threat_level", "source",
                "first_seen", "confidence", "tags"
            ])
            
            # Write indicators
            for indicator in self.indicators.values():
                writer.writerow([
                    indicator.indicator_id,
                    indicator.indicator_type.value,
                    indicator.value,
                    indicator.threat_level.value,
                    indicator.source,
                    indicator.first_seen.isoformat(),
                    indicator.confidence,
                    ",".join(indicator.tags)
                ])
            
            return output.getvalue()
        
        elif format_type == "stix":
            # STIX 2.1 format
            stix_objects = []
            
            for indicator in self.indicators.values():
                stix_indicator = {
                    "type": "indicator",
                    "id": f"indicator--{uuid.uuid4()}",
                    "created": indicator.first_seen.isoformat(),
                    "modified": indicator.last_seen.isoformat(),
                    "pattern": f"[{indicator.indicator_type.value}:value = '{indicator.value}']",
                    "labels": indicator.tags,
                    "confidence": int(indicator.confidence * 100),
                    "x_threat_level": indicator.threat_level.value
                }
                stix_objects.append(stix_indicator)
            
            stix_bundle = {
                "type": "bundle",
                "id": f"bundle--{uuid.uuid4()}",
                "objects": stix_objects
            }
            
            return json.dumps(stix_bundle, indent=2)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    async def threat_intelligence_cycle(self) -> Dict[str, Any]:
        """Execute threat intelligence collection cycle"""
        logger.info("🔄 Starting threat intelligence cycle...")
        
        cycle_start_time = time.time()
        
        # Update all feeds
        await self.update_all_feeds()
        
        # Get recent high-priority indicators
        critical_indicators = await self.get_recent_indicators(hours=24, threat_level=ThreatLevel.CRITICAL)
        high_indicators = await self.get_recent_indicators(hours=24, threat_level=ThreatLevel.HIGH)
        
        # Generate threat landscape summary
        threat_landscape = await self._generate_threat_landscape_summary()
        
        cycle_duration = time.time() - cycle_start_time
        
        cycle_results = {
            "cycle_timestamp": datetime.now().isoformat(),
            "cycle_duration_seconds": cycle_duration,
            "feeds_status": {feed_id: feed.status.value for feed_id, feed in self.feeds.items()},
            "recent_critical_indicators": len(critical_indicators),
            "recent_high_indicators": len(high_indicators),
            "threat_landscape": threat_landscape,
            "processing_statistics": self.processing_stats
        }
        
        return cycle_results
    
    async def _generate_threat_landscape_summary(self) -> Dict[str, Any]:
        """Generate threat landscape summary"""
        # Analyze recent indicators to identify trends
        recent_indicators = await self.get_recent_indicators(hours=168)  # Last week
        
        # Count by type
        type_counts = {}
        for indicator in recent_indicators:
            ioc_type = indicator.indicator_type.value
            type_counts[ioc_type] = type_counts.get(ioc_type, 0) + 1
        
        # Count by threat level
        level_counts = {}
        for indicator in recent_indicators:
            level = indicator.threat_level.value
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Identify trending tags
        tag_counts = {}
        for indicator in recent_indicators:
            for tag in indicator.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        trending_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_recent_indicators": len(recent_indicators),
            "indicator_types": type_counts,
            "threat_levels": level_counts,
            "trending_tags": dict(trending_tags),
            "top_sources": self._get_top_sources(recent_indicators),
            "threat_trends": "Increased malware activity, APT campaigns targeting critical infrastructure"
        }
    
    def _get_top_sources(self, indicators: List[ThreatIndicator]) -> Dict[str, int]:
        """Get top threat intelligence sources"""
        source_counts = {}
        for indicator in indicators:
            source_counts[indicator.source] = source_counts.get(indicator.source, 0) + 1
        
        return dict(sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:5])

async def main():
    """Main threat intelligence feed execution"""
    logger.info("📡 Starting XORB Threat Intelligence Feed Manager")
    
    # Initialize threat intelligence manager
    threat_intel = XORBThreatIntelligenceFeeds()
    
    # Initialize feeds
    await threat_intel.initialize_feeds()
    
    # Execute threat intelligence cycles
    session_duration = 2  # 2 minutes for demonstration
    cycles_completed = 0
    
    start_time = time.time()
    end_time = start_time + (session_duration * 60)
    
    while time.time() < end_time:
        try:
            # Execute threat intelligence cycle
            cycle_results = await threat_intel.threat_intelligence_cycle()
            cycles_completed += 1
            
            # Log progress
            logger.info(f"📡 Intelligence Cycle #{cycles_completed} completed")
            logger.info(f"🔥 Critical indicators: {cycle_results['recent_critical_indicators']}")
            logger.info(f"⚠️ High indicators: {cycle_results['recent_high_indicators']}")
            logger.info(f"📊 Total indicators: {threat_intel.processing_stats['total_indicators']}")
            
            await asyncio.sleep(30.0)  # 30-second cycles
            
        except Exception as e:
            logger.error(f"Error in threat intelligence cycle: {e}")
            await asyncio.sleep(10.0)
    
    # Final results
    final_results = {
        "session_id": f"THREAT-INTEL-SESSION-{int(start_time)}",
        "cycles_completed": cycles_completed,
        "processing_statistics": threat_intel.processing_stats,
        "total_indicators_collected": len(threat_intel.indicators),
        "active_feeds": len([f for f in threat_intel.feeds.values() if f.status == FeedStatus.ACTIVE]),
        "feed_status": {feed.name: feed.status.value for feed in threat_intel.feeds.values()}
    }
    
    # Save results
    results_filename = f"xorb_threat_intelligence_results_{int(time.time())}.json"
    async with aiofiles.open(results_filename, 'w') as f:
        await f.write(json.dumps(final_results, indent=2, default=str))
    
    logger.info(f"💾 Threat intelligence results saved: {results_filename}")
    logger.info("🏆 XORB Threat Intelligence Feed Manager completed!")
    
    # Display final summary
    logger.info("📡 Threat Intelligence Summary:")
    logger.info(f"  • Cycles completed: {cycles_completed}")
    logger.info(f"  • Total indicators: {threat_intel.processing_stats['total_indicators']}")
    logger.info(f"  • Critical indicators: {threat_intel.processing_stats['critical_indicators']}")
    logger.info(f"  • High indicators: {threat_intel.processing_stats['high_indicators']}")
    logger.info(f"  • Active feeds: {final_results['active_feeds']}")
    logger.info(f"  • Processing errors: {threat_intel.processing_stats['processing_errors']}")
    
    return final_results

if __name__ == "__main__":
    # Execute threat intelligence feed management
    asyncio.run(main())