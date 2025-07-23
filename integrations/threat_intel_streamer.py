#!/usr/bin/env python3

import asyncio
import json
import logging
import aiohttp
import hashlib
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    from kafka import KafkaConsumer, KafkaProducer
    import redis.asyncio as redis
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    logging.warning("Streaming dependencies not available. Install with: pip install kafka-python redis")

from knowledge_fabric.core import KnowledgeFabric
from knowledge_fabric.atom import KnowledgeAtom, AtomType, Source, ConfidenceLevel


@dataclass
class ThreatIntelligence:
    """Structured threat intelligence data."""
    source: str
    intel_type: str  # 'cve', 'ioc', 'technique', 'campaign'
    title: str
    description: str
    severity: str
    confidence: float
    timestamp: datetime
    indicators: List[Dict[str, Any]]
    references: List[str]
    tags: List[str]
    raw_data: Dict[str, Any]


class CVEFeedProcessor:
    """Processes CVE feeds from multiple sources."""
    
    def __init__(self):
        self.session = None
        self.logger = logging.getLogger(__name__)
        
        self.cve_sources = {
            'nvd': 'https://services.nvd.nist.gov/rest/json/cves/2.0',
            'mitre': 'https://cve.mitre.org/data/downloads/allitems.csv',
            'github_advisory': 'https://api.github.com/advisories'
        }
    
    async def initialize(self):
        """Initialize HTTP session for CVE feeds."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'XORB-ThreatIntel/1.0'}
        )
    
    async def fetch_recent_cves(self, hours_back: int = 24) -> AsyncGenerator[ThreatIntelligence, None]:
        """Fetch recent CVEs from multiple sources."""
        if not self.session:
            await self.initialize()
        
        # Calculate time window
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=hours_back)
        
        # Fetch from NVD
        async for cve in self._fetch_nvd_cves(start_date, end_date):
            yield cve
        
        # Fetch from GitHub Security Advisories
        async for advisory in self._fetch_github_advisories(start_date, end_date):
            yield advisory
    
    async def _fetch_nvd_cves(self, start_date: datetime, end_date: datetime) -> AsyncGenerator[ThreatIntelligence, None]:
        """Fetch CVEs from NVD API."""
        try:
            url = self.cve_sources['nvd']
            params = {
                'pubStartDate': start_date.isoformat(),
                'pubEndDate': end_date.isoformat(),
                'resultsPerPage': 100
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for cve_item in data.get('vulnerabilities', []):
                        cve = cve_item.get('cve', {})
                        
                        # Extract key information
                        cve_id = cve.get('id', 'Unknown')
                        description = ''
                        if cve.get('descriptions'):
                            description = cve['descriptions'][0].get('value', '')
                        
                        # Get CVSS score
                        cvss_score = 0.0
                        severity = 'unknown'
                        if cve.get('metrics', {}).get('cvssMetricV3'):
                            cvss_data = cve['metrics']['cvssMetricV3'][0]
                            cvss_score = cvss_data.get('cvssData', {}).get('baseScore', 0.0)
                            severity = cvss_data.get('cvssData', {}).get('baseSeverity', 'unknown').lower()
                        
                        # Extract references
                        references = []
                        for ref in cve.get('references', []):
                            references.append(ref.get('url', ''))
                        
                        # Extract CPE configurations as indicators
                        indicators = []
                        for config in cve.get('configurations', []):
                            for node in config.get('nodes', []):
                                for cpe_match in node.get('cpeMatch', []):
                                    indicators.append({
                                        'type': 'cpe',
                                        'value': cpe_match.get('criteria', ''),
                                        'vulnerable': cpe_match.get('vulnerable', False)
                                    })
                        
                        yield ThreatIntelligence(
                            source='nvd',
                            intel_type='cve',
                            title=cve_id,
                            description=description,
                            severity=severity,
                            confidence=0.9,  # NVD is highly reliable
                            timestamp=datetime.utcnow(),
                            indicators=indicators,
                            references=references,
                            tags=[cve_id, 'vulnerability', severity],
                            raw_data=cve_item
                        )
                else:
                    self.logger.warning(f"Failed to fetch NVD CVEs: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Error fetching NVD CVEs: {e}")
    
    async def _fetch_github_advisories(self, start_date: datetime, end_date: datetime) -> AsyncGenerator[ThreatIntelligence, None]:
        """Fetch GitHub Security Advisories."""
        try:
            url = self.cve_sources['github_advisory']
            params = {
                'type': 'reviewed',
                'per_page': 100
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    advisories = await response.json()
                    
                    for advisory in advisories:
                        published_at = datetime.fromisoformat(advisory.get('published_at', '').replace('Z', '+00:00'))
                        
                        # Filter by date range
                        if start_date <= published_at <= end_date:
                            # Extract affected packages as indicators
                            indicators = []
                            for vuln in advisory.get('vulnerabilities', []):
                                package = vuln.get('package', {})
                                indicators.append({
                                    'type': 'package',
                                    'ecosystem': package.get('ecosystem'),
                                    'name': package.get('name'),
                                    'vulnerable_version_range': vuln.get('vulnerable_version_range')
                                })
                            
                            yield ThreatIntelligence(
                                source='github',
                                intel_type='advisory',
                                title=advisory.get('summary', ''),
                                description=advisory.get('description', ''),
                                severity=advisory.get('severity', 'unknown').lower(),
                                confidence=0.85,
                                timestamp=published_at,
                                indicators=indicators,
                                references=[advisory.get('html_url', '')],
                                tags=['github', 'advisory', advisory.get('severity', 'unknown').lower()],
                                raw_data=advisory
                            )
                            
        except Exception as e:
            self.logger.error(f"Error fetching GitHub advisories: {e}")
    
    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()


class IOCFeedProcessor:
    """Processes Indicators of Compromise (IOC) feeds."""
    
    def __init__(self):
        self.session = None
        self.logger = logging.getLogger(__name__)
        
        # Free IOC sources
        self.ioc_sources = {
            'abuse_ch': 'https://urlhaus-api.abuse.ch/v1/urls/recent/',
            'malware_bazaar': 'https://mb-api.abuse.ch/api/v1/',
            'threatfox': 'https://threatfox-api.abuse.ch/api/v1/'
        }
    
    async def initialize(self):
        """Initialize HTTP session for IOC feeds."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'XORB-ThreatIntel/1.0'}
        )
    
    async def fetch_recent_iocs(self, hours_back: int = 24) -> AsyncGenerator[ThreatIntelligence, None]:
        """Fetch recent IOCs from multiple sources."""
        if not self.session:
            await self.initialize()
        
        # Fetch from URLhaus
        async for ioc in self._fetch_urlhaus_iocs(hours_back):
            yield ioc
        
        # Fetch from ThreatFox
        async for ioc in self._fetch_threatfox_iocs(hours_back):
            yield ioc
    
    async def _fetch_urlhaus_iocs(self, hours_back: int) -> AsyncGenerator[ThreatIntelligence, None]:
        """Fetch malicious URLs from URLhaus."""
        try:
            url = self.ioc_sources['abuse_ch']
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
                    
                    for url_entry in data.get('urls', []):
                        date_added = datetime.fromisoformat(url_entry.get('date_added', '').replace('Z', '+00:00'))
                        
                        if date_added >= cutoff_time:
                            indicators = [{
                                'type': 'url',
                                'value': url_entry.get('url', ''),
                                'host': url_entry.get('host', ''),
                                'url_status': url_entry.get('url_status', '')
                            }]
                            
                            yield ThreatIntelligence(
                                source='urlhaus',
                                intel_type='ioc',
                                title=f"Malicious URL: {url_entry.get('host', 'Unknown')}",
                                description=f"Malicious URL detected: {url_entry.get('url', '')}",
                                severity='medium',
                                confidence=0.8,
                                timestamp=date_added,
                                indicators=indicators,
                                references=[url_entry.get('urlhaus_reference', '')],
                                tags=['ioc', 'malicious_url', url_entry.get('threat', 'unknown')],
                                raw_data=url_entry
                            )
                            
        except Exception as e:
            self.logger.error(f"Error fetching URLhaus IOCs: {e}")
    
    async def _fetch_threatfox_iocs(self, hours_back: int) -> AsyncGenerator[ThreatIntelligence, None]:
        """Fetch IOCs from ThreatFox."""
        try:
            url = self.ioc_sources['threatfox']
            
            # ThreatFox requires POST request
            payload = {
                'query': 'get_iocs',
                'days': max(1, hours_back // 24)
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for ioc_entry in data.get('data', []):
                        first_seen = datetime.fromisoformat(ioc_entry.get('first_seen', '').replace('Z', '+00:00'))
                        
                        indicators = [{
                            'type': ioc_entry.get('ioc_type', ''),
                            'value': ioc_entry.get('ioc_value', ''),
                            'malware_family': ioc_entry.get('malware', ''),
                            'confidence_level': ioc_entry.get('confidence_level', 0)
                        }]
                        
                        yield ThreatIntelligence(
                            source='threatfox',
                            intel_type='ioc',
                            title=f"{ioc_entry.get('ioc_type', 'IOC')}: {ioc_entry.get('malware', 'Unknown')}",
                            description=f"IOC detected for {ioc_entry.get('malware', 'unknown malware')}",
                            severity='high' if ioc_entry.get('confidence_level', 0) > 75 else 'medium',
                            confidence=ioc_entry.get('confidence_level', 50) / 100.0,
                            timestamp=first_seen,
                            indicators=indicators,
                            references=[ioc_entry.get('reference', '')],
                            tags=['ioc', ioc_entry.get('ioc_type', ''), ioc_entry.get('malware', '')],
                            raw_data=ioc_entry
                        )
                        
        except Exception as e:
            self.logger.error(f"Error fetching ThreatFox IOCs: {e}")
    
    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()


class ThreatIntelStreamer:
    """Real-time threat intelligence streaming and processing system."""
    
    def __init__(self, knowledge_fabric: KnowledgeFabric, redis_url: str = 'redis://localhost:6379'):
        self.knowledge_fabric = knowledge_fabric
        self.redis_url = redis_url
        self.redis_client = None
        
        self.cve_processor = CVEFeedProcessor()
        self.ioc_processor = IOCFeedProcessor()
        
        self.running = False
        self.logger = logging.getLogger(__name__)
        
        # Processing statistics
        self.stats = {
            'processed_count': 0,
            'error_count': 0,
            'last_update': None,
            'sources_processed': set()
        }
    
    async def initialize(self):
        """Initialize the threat intelligence streamer."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            await self.cve_processor.initialize()
            await self.ioc_processor.initialize()
            
            self.logger.info("Threat intelligence streamer initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize threat intel streamer: {e}")
            raise
    
    async def start_streaming(self, update_interval_minutes: int = 60):
        """Start continuous threat intelligence streaming."""
        self.running = True
        self.logger.info(f"Starting threat intelligence streaming (update every {update_interval_minutes} minutes)")
        
        while self.running:
            try:
                # Process recent threat intelligence
                await self._process_threat_intel_batch()
                
                # Update statistics
                self.stats['last_update'] = datetime.utcnow()
                
                # Wait for next update cycle
                await asyncio.sleep(update_interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Error in streaming loop: {e}")
                self.stats['error_count'] += 1
                await asyncio.sleep(60)  # Short retry delay
    
    async def stop_streaming(self):
        """Stop threat intelligence streaming."""
        self.running = False
        
        await self.cve_processor.close()
        await self.ioc_processor.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        self.logger.info("Threat intelligence streaming stopped")
    
    async def _process_threat_intel_batch(self, hours_back: int = 1):
        """Process a batch of threat intelligence from all sources."""
        self.logger.info(f"Processing threat intelligence from last {hours_back} hours")
        
        processed_count = 0
        
        # Process CVE feeds
        try:
            async for cve_intel in self.cve_processor.fetch_recent_cves(hours_back):
                await self._process_intel_item(cve_intel)
                processed_count += 1
                self.stats['sources_processed'].add('cve')
        except Exception as e:
            self.logger.error(f"Error processing CVE feeds: {e}")
            self.stats['error_count'] += 1
        
        # Process IOC feeds
        try:
            async for ioc_intel in self.ioc_processor.fetch_recent_iocs(hours_back):
                await self._process_intel_item(ioc_intel)
                processed_count += 1
                self.stats['sources_processed'].add('ioc')
        except Exception as e:
            self.logger.error(f"Error processing IOC feeds: {e}")
            self.stats['error_count'] += 1
        
        self.stats['processed_count'] += processed_count
        self.logger.info(f"Processed {processed_count} threat intelligence items")
    
    async def _process_intel_item(self, intel: ThreatIntelligence):
        """Process individual threat intelligence item."""
        try:
            # Create knowledge atom from threat intelligence
            atom = await self._create_knowledge_atom(intel)
            
            # Store in knowledge fabric
            await self.knowledge_fabric.store_atom(atom)
            
            # Cache in Redis for real-time access
            await self._cache_intel_item(intel)
            
            # Trigger campaign updates if relevant
            await self._check_campaign_relevance(intel)
            
        except Exception as e:
            self.logger.error(f"Error processing intel item: {e}")
            self.stats['error_count'] += 1
    
    async def _create_knowledge_atom(self, intel: ThreatIntelligence) -> KnowledgeAtom:
        """Convert threat intelligence to knowledge atom."""
        # Determine atom type
        atom_type = AtomType.INTELLIGENCE
        if intel.intel_type == 'cve':
            atom_type = AtomType.VULNERABILITY
        elif intel.intel_type == 'ioc':
            atom_type = AtomType.INTELLIGENCE
        elif intel.intel_type == 'technique':
            atom_type = AtomType.TECHNIQUE
        
        # Create source
        source = Source(
            url=intel.references[0] if intel.references else f"{intel.source}/api",
            source_type=f"{intel.source}_feed",
            reliability_score=intel.confidence,
            accessed_at=datetime.utcnow()
        )
        
        # Generate unique ID
        content_hash = hashlib.sha256(f"{intel.source}{intel.title}{intel.description}".encode()).hexdigest()[:16]
        atom_id = f"{intel.intel_type}_{intel.source}_{content_hash}"
        
        # Create atom
        atom = KnowledgeAtom(
            id=atom_id,
            atom_type=atom_type,
            content={
                'title': intel.title,
                'description': intel.description,
                'severity': intel.severity,
                'indicators': intel.indicators,
                'tags': intel.tags,
                'raw_data': intel.raw_data
            },
            confidence=intel.confidence,
            sources=[source],
            created_at=intel.timestamp,
            updated_at=datetime.utcnow(),
            metadata={
                'intel_type': intel.intel_type,
                'feed_source': intel.source,
                'severity': intel.severity,
                'indicator_count': len(intel.indicators)
            }
        )
        
        return atom
    
    async def _cache_intel_item(self, intel: ThreatIntelligence):
        """Cache intelligence item in Redis for fast access."""
        if not self.redis_client:
            return
        
        try:
            key = f"threat_intel:{intel.source}:{intel.intel_type}:{intel.timestamp.strftime('%Y%m%d')}"
            value = json.dumps(asdict(intel), default=str)
            
            # Cache for 7 days
            await self.redis_client.setex(key, 7 * 24 * 3600, value)
            
        except Exception as e:
            self.logger.error(f"Error caching intel item: {e}")
    
    async def _check_campaign_relevance(self, intel: ThreatIntelligence):
        """Check if threat intelligence is relevant to active campaigns."""
        # This would integrate with the orchestrator to check if any active campaigns
        # should be updated based on new threat intelligence
        
        # For now, just log high-confidence, high-severity intelligence
        if intel.confidence > 0.8 and intel.severity in ['high', 'critical']:
            self.logger.info(f"High-priority threat intel detected: {intel.title} (confidence: {intel.confidence:.2f})")
    
    async def get_recent_intel(self, intel_type: Optional[str] = None, 
                             source: Optional[str] = None, 
                             hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get recent threat intelligence from cache."""
        if not self.redis_client:
            return []
        
        try:
            pattern = "threat_intel:"
            if source:
                pattern += f"{source}:"
            else:
                pattern += "*:"
            
            if intel_type:
                pattern += f"{intel_type}:"
            else:
                pattern += "*:"
            
            pattern += "*"
            
            keys = await self.redis_client.keys(pattern)
            intel_items = []
            
            for key in keys:
                data = await self.redis_client.get(key)
                if data:
                    intel_items.append(json.loads(data))
            
            # Filter by time range
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
            filtered_items = []
            
            for item in intel_items:
                timestamp = datetime.fromisoformat(item['timestamp'])
                if timestamp >= cutoff_time:
                    filtered_items.append(item)
            
            # Sort by timestamp (newest first)
            filtered_items.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return filtered_items
            
        except Exception as e:
            self.logger.error(f"Error retrieving recent intel: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get threat intelligence processing statistics."""
        return {
            **self.stats,
            'sources_processed': list(self.stats['sources_processed']),
            'is_running': self.running
        }


class ThreatIntelDashboard:
    """Dashboard for monitoring threat intelligence feeds."""
    
    def __init__(self, streamer: ThreatIntelStreamer):
        self.streamer = streamer
        self.logger = logging.getLogger(__name__)
    
    async def display_dashboard(self):
        """Display real-time threat intelligence dashboard."""
        stats = self.streamer.get_stats()
        recent_intel = await self.streamer.get_recent_intel(hours_back=24)
        
        print("\n" + "="*60)
        print("XORB THREAT INTELLIGENCE DASHBOARD")
        print("="*60)
        
        # Statistics
        print(f"\nSTATISTICS:")
        print(f"  Total Processed: {stats['processed_count']}")
        print(f"  Errors: {stats['error_count']}")
        print(f"  Last Update: {stats['last_update']}")
        print(f"  Sources Active: {', '.join(stats['sources_processed'])}")
        print(f"  Status: {'RUNNING' if stats['is_running'] else 'STOPPED'}")
        
        # Recent intelligence summary
        print(f"\nRECENT INTELLIGENCE (24h):")
        intel_by_type = {}
        intel_by_severity = {}
        
        for intel in recent_intel:
            intel_type = intel.get('intel_type', 'unknown')
            severity = intel.get('severity', 'unknown')
            
            intel_by_type[intel_type] = intel_by_type.get(intel_type, 0) + 1
            intel_by_severity[severity] = intel_by_severity.get(severity, 0) + 1
        
        print(f"  By Type: {dict(intel_by_type)}")
        print(f"  By Severity: {dict(intel_by_severity)}")
        
        # Top recent items
        print(f"\nTOP RECENT ITEMS:")
        for i, intel in enumerate(recent_intel[:5]):
            print(f"  {i+1}. [{intel.get('severity', 'unknown').upper()}] {intel.get('title', 'Unknown')}")
            print(f"     Source: {intel.get('source', 'unknown')} | Confidence: {intel.get('confidence', 0):.2f}")
        
        print("\n" + "="*60)