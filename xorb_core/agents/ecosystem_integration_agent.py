#!/usr/bin/env python3
"""
🔗 EcosystemIntegrationAgent - Phase 12.3 Implementation
Integrates with global security ecosystems and standards bodies for threat intelligence federation.

Part of the XORB Ecosystem - Phase 12: Autonomous Defense & Planetary Scale Operations
"""

import asyncio
import logging
import time
import uuid
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import aioredis
import asyncpg
from prometheus_client import Counter, Histogram, Gauge, Summary
import structlog
import hashlib
import base64
from urllib.parse import urljoin
import ssl

# Metrics
integration_operations_total = Counter('xorb_integration_operations_total', 'Total integration operations', ['partner', 'operation_type'])
threat_intel_shared_total = Counter('xorb_threat_intel_shared_total', 'Total threat intelligence shared', ['partner', 'indicator_type'])
threat_intel_received_total = Counter('xorb_threat_intel_received_total', 'Total threat intelligence received', ['partner', 'indicator_type'])
integration_latency_seconds = Histogram('xorb_integration_latency_seconds', 'Integration operation latency', ['partner'])
partner_availability = Gauge('xorb_partner_availability', 'Partner system availability', ['partner'])
compliance_score = Gauge('xorb_compliance_score', 'Current compliance score', ['framework'])

logger = structlog.get_logger("ecosystem_integration_agent")

class IntegrationType(Enum):
    """Types of ecosystem integrations"""
    CERT = "cert"
    ISAC = "isac"
    VENDOR = "vendor"
    GOVERNMENT = "government"
    ACADEMIC = "academic"
    COMMERCIAL = "commercial"
    COMMUNITY = "community"

class StandardProtocol(Enum):
    """Supported standard protocols"""
    STIX_TAXII_21 = "stix_taxii_21"
    STIX_TAXII_20 = "stix_taxii_20"
    MISP = "misp"
    OPENCTI = "opencti"
    CVE_API = "cve_api"
    CAPEC_API = "capec_api"
    ATT_CK_API = "attck_api"
    REST_API = "rest_api"
    WEBHOOK = "webhook"

class IndicatorType(Enum):
    """Types of threat indicators"""
    IP_ADDRESS = "ip_address"
    DOMAIN = "domain"
    URL = "url"
    FILE_HASH = "file_hash"
    EMAIL = "email"
    CVE = "cve"
    MALWARE_FAMILY = "malware_family"
    ATTACK_PATTERN = "attack_pattern"
    CAMPAIGN = "campaign"
    THREAT_ACTOR = "threat_actor"

class ComplianceFramework(Enum):
    """Compliance frameworks"""
    MITRE_ATTCK = "mitre_attck"
    NIST_CSF = "nist_csf"
    ISO27001 = "iso27001"
    SOC2 = "soc2"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"

@dataclass
class IntegrationPartner:
    """Integration partner configuration"""
    partner_id: str
    name: str
    partner_type: IntegrationType
    protocols: List[StandardProtocol]
    endpoints: Dict[str, str]
    authentication: Dict[str, Any]
    capabilities: List[str]
    trust_level: float
    data_sharing_agreement: Dict[str, Any]
    rate_limits: Dict[str, int]
    last_contact: Optional[datetime]
    status: str  # active, inactive, error
    metadata: Dict[str, Any]

@dataclass
class ThreatIndicator:
    """Threat intelligence indicator"""
    indicator_id: str
    indicator_type: IndicatorType
    value: str
    confidence: float
    severity: str
    first_seen: datetime
    last_seen: datetime
    source: str
    tlp_marking: str  # Traffic Light Protocol
    ioc_data: Dict[str, Any]
    context: Dict[str, Any]
    tags: List[str]
    relationships: List[Dict[str, Any]]

@dataclass
class STIXBundle:
    """STIX 2.1 bundle representation"""
    bundle_id: str
    spec_version: str
    objects: List[Dict[str, Any]]
    created: datetime
    modified: datetime
    source_partner: str
    confidence_level: float

@dataclass
class ComplianceMapping:
    """Compliance framework mapping"""
    framework: ComplianceFramework
    control_id: str
    control_name: str
    xorb_capability: str
    implementation_status: str
    evidence: List[str]
    last_assessed: datetime
    compliance_score: float

class EcosystemIntegrationAgent:
    """
    🔗 Ecosystem Integration Agent
    
    Integrates with global security ecosystems:
    - STIX/TAXII 2.1 protocol implementation
    - Integration with national CERTs
    - Compliance with international standards
    - Automated threat intelligence federation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.agent_id = f"ecosystem-integration-{uuid.uuid4().hex[:8]}"
        self.is_running = False
        
        # Configuration parameters
        self.max_concurrent_operations = self.config.get('max_concurrent_operations', 20)
        self.sync_interval = self.config.get('sync_interval', 3600)  # 1 hour
        self.batch_size = self.config.get('batch_size', 100)
        self.retry_attempts = self.config.get('retry_attempts', 3)
        self.timeout = self.config.get('timeout', 30)
        
        # TLP (Traffic Light Protocol) settings
        self.default_tlp = self.config.get('default_tlp', 'amber')
        self.sharing_policy = self.config.get('sharing_policy', {
            'white': ['all'],
            'green': ['partners', 'community'],
            'amber': ['partners'],
            'red': []
        })
        
        # Storage and communication
        self.redis_pool = None
        self.db_pool = None
        self.http_session = None
        
        # Integration state
        self.partners: Dict[str, IntegrationPartner] = {}
        self.active_connections: Dict[str, aiohttp.ClientSession] = {}
        self.threat_indicators: Dict[str, ThreatIndicator] = {}
        self.compliance_mappings: Dict[ComplianceFramework, List[ComplianceMapping]] = {}
        
        # STIX/TAXII state
        self.taxii_collections: Dict[str, Dict[str, Any]] = {}
        self.stix_objects_cache: Dict[str, Dict[str, Any]] = {}
        
        # Rate limiting
        self.rate_limiters: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.sharing_stats: Dict[str, Dict[str, int]] = {}
        self.compliance_scores: Dict[ComplianceFramework, float] = {}
        
        logger.info("EcosystemIntegrationAgent initialized", agent_id=self.agent_id)

    async def initialize(self):
        """Initialize the ecosystem integration agent"""
        try:
            # Initialize Redis connection
            self.redis_pool = aioredis.ConnectionPool.from_url(
                self.config.get('redis_url', 'redis://localhost:6379'),
                max_connections=20
            )
            
            # Initialize PostgreSQL connection
            self.db_pool = await asyncpg.create_pool(
                self.config.get('postgres_url', 'postgresql://localhost:5432/xorb'),
                min_size=5,
                max_size=20
            )
            
            # Initialize HTTP session
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(
                limit=100,
                ssl=ssl.create_default_context()
            )
            self.http_session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
            
            # Initialize database schema
            await self._initialize_database()
            
            # Load integration partners
            await self._load_integration_partners()
            
            # Initialize compliance mappings
            await self._initialize_compliance_mappings()
            
            # Load TAXII collections
            await self._load_taxii_collections()
            
            logger.info("EcosystemIntegrationAgent initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize EcosystemIntegrationAgent", error=str(e))
            raise

    async def start_integration(self):
        """Start the ecosystem integration process"""
        if self.is_running:
            logger.warning("Integration already running")
            return
            
        self.is_running = True
        logger.info("Starting ecosystem integration process")
        
        try:
            # Start integration loops
            sync_task = asyncio.create_task(self._threat_intel_sync_loop())
            sharing_task = asyncio.create_task(self._threat_intel_sharing_loop())
            compliance_task = asyncio.create_task(self._compliance_monitoring_loop())
            health_task = asyncio.create_task(self._partner_health_monitoring_loop())
            collection_task = asyncio.create_task(self._taxii_collection_loop())
            
            await asyncio.gather(
                sync_task, sharing_task, compliance_task, 
                health_task, collection_task
            )
            
        except Exception as e:
            logger.error("Integration process failed", error=str(e))
            raise
        finally:
            self.is_running = False

    async def stop_integration(self):
        """Stop the ecosystem integration process"""
        logger.info("Stopping ecosystem integration process")
        self.is_running = False

    async def _threat_intel_sync_loop(self):
        """Synchronize threat intelligence with partners"""
        while self.is_running:
            try:
                logger.debug("Starting threat intelligence synchronization")
                
                # Sync with each active partner
                sync_tasks = []
                for partner in self.partners.values():
                    if partner.status == 'active':
                        task = asyncio.create_task(self._sync_with_partner(partner))
                        sync_tasks.append(task)
                
                if sync_tasks:
                    await asyncio.gather(*sync_tasks, return_exceptions=True)
                
                await asyncio.sleep(self.sync_interval)
                
            except Exception as e:
                logger.error("Threat intel sync loop failed", error=str(e))
                await asyncio.sleep(300)  # 5 minute error backoff

    async def _threat_intel_sharing_loop(self):
        """Share threat intelligence with partners"""
        while self.is_running:
            try:
                # Process outbound threat intelligence queue
                await self._process_outbound_intelligence()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error("Threat intel sharing loop failed", error=str(e))
                await asyncio.sleep(60)

    async def _compliance_monitoring_loop(self):
        """Monitor compliance with standards frameworks"""
        while self.is_running:
            try:
                # Update compliance scores
                await self._update_compliance_scores()
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error("Compliance monitoring loop failed", error=str(e))
                await asyncio.sleep(600)

    async def _partner_health_monitoring_loop(self):
        """Monitor partner system health and availability"""
        while self.is_running:
            try:
                # Check partner health
                health_tasks = []
                for partner in self.partners.values():
                    task = asyncio.create_task(self._check_partner_health(partner))
                    health_tasks.append(task)
                
                if health_tasks:
                    await asyncio.gather(*health_tasks, return_exceptions=True)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error("Partner health monitoring failed", error=str(e))
                await asyncio.sleep(300)

    async def _taxii_collection_loop(self):
        """Manage TAXII collection operations"""
        while self.is_running:
            try:
                # Update TAXII collections
                for partner in self.partners.values():
                    if StandardProtocol.STIX_TAXII_21 in partner.protocols:
                        await self._update_taxii_collections(partner)
                
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except Exception as e:
                logger.error("TAXII collection loop failed", error=str(e))
                await asyncio.sleep(600)

    async def _load_integration_partners(self):
        """Load integration partner configurations"""
        try:
            # Load from database
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM integration_partners WHERE active = true")
                
                for row in rows:
                    partner_data = json.loads(row['partner_data'])
                    partner = IntegrationPartner(**partner_data)
                    self.partners[partner.partner_id] = partner
                    
                    # Initialize rate limiter
                    self.rate_limiters[partner.partner_id] = {
                        'requests': 0,
                        'last_reset': time.time(),
                        'limit': partner.rate_limits.get('requests_per_hour', 1000)
                    }
            
            # Add default partners if none exist
            if not self.partners:
                await self._create_default_partners()
            
            logger.info("Integration partners loaded", count=len(self.partners))
            
        except Exception as e:
            logger.error("Failed to load integration partners", error=str(e))
            await self._create_default_partners()

    async def _create_default_partners(self):
        """Create default integration partners"""
        default_partners = [
            {
                'partner_id': 'us-cert',
                'name': 'US-CERT',
                'partner_type': IntegrationType.CERT,
                'protocols': [StandardProtocol.STIX_TAXII_21],
                'endpoints': {
                    'taxii2_root': 'https://cisa.gov/taxii2/',
                    'discovery': 'https://cisa.gov/taxii2/discovery/'
                },
                'authentication': {'type': 'api_key', 'key_location': 'header'},
                'capabilities': ['indicators', 'campaigns', 'malware'],
                'trust_level': 0.95,
                'data_sharing_agreement': {'tlp_levels': ['white', 'green', 'amber']},
                'rate_limits': {'requests_per_hour': 1000},
                'status': 'active',
                'metadata': {'region': 'US', 'sector': 'government'}
            },
            {
                'partner_id': 'mitre-attck',
                'name': 'MITRE ATT&CK',
                'partner_type': IntegrationType.COMMUNITY,
                'protocols': [StandardProtocol.ATT_CK_API, StandardProtocol.STIX_TAXII_21],
                'endpoints': {
                    'api_root': 'https://attack.mitre.org/api/',
                    'taxii2_root': 'https://cti-taxii.mitre.org/'
                },
                'authentication': {'type': 'none'},
                'capabilities': ['techniques', 'tactics', 'groups', 'software'],
                'trust_level': 1.0,
                'data_sharing_agreement': {'tlp_levels': ['white']},
                'rate_limits': {'requests_per_hour': 500},
                'status': 'active',
                'metadata': {'type': 'knowledge_base'}
            },
            {
                'partner_id': 'cve-api',
                'name': 'CVE API',
                'partner_type': IntegrationType.COMMUNITY,
                'protocols': [StandardProtocol.CVE_API],
                'endpoints': {
                    'api_root': 'https://services.nvd.nist.gov/rest/json/'
                },
                'authentication': {'type': 'api_key', 'optional': True},
                'capabilities': ['vulnerabilities', 'cpe'],
                'trust_level': 1.0,
                'data_sharing_agreement': {'tlp_levels': ['white']},
                'rate_limits': {'requests_per_hour': 2000},
                'status': 'active',
                'metadata': {'type': 'vulnerability_database'}
            }
        ]
        
        for partner_config in default_partners:
            partner = IntegrationPartner(
                last_contact=None,
                **partner_config
            )
            self.partners[partner.partner_id] = partner
            
            # Initialize rate limiter
            self.rate_limiters[partner.partner_id] = {
                'requests': 0,
                'last_reset': time.time(),
                'limit': partner.rate_limits.get('requests_per_hour', 1000)
            }
            
            # Persist to database
            await self._persist_partner(partner)
        
        logger.info("Default integration partners created", count=len(default_partners))

    async def _sync_with_partner(self, partner: IntegrationPartner):
        """Synchronize threat intelligence with a specific partner"""
        logger.debug("Syncing with partner", partner_id=partner.partner_id)
        
        try:
            sync_start = time.time()
            
            # Check rate limits
            if not await self._check_rate_limit(partner.partner_id):
                logger.warning("Rate limit exceeded for partner", partner_id=partner.partner_id)
                return
            
            with integration_latency_seconds.labels(partner=partner.partner_id).time():
                if StandardProtocol.STIX_TAXII_21 in partner.protocols:
                    await self._sync_taxii21(partner)
                elif StandardProtocol.CVE_API in partner.protocols:
                    await self._sync_cve_api(partner)
                elif StandardProtocol.ATT_CK_API in partner.protocols:
                    await self._sync_attack_api(partner)
                elif StandardProtocol.MISP in partner.protocols:
                    await self._sync_misp(partner)
            
            # Update partner status
            partner.last_contact = datetime.utcnow()
            partner.status = 'active'
            
            integration_operations_total.labels(
                partner=partner.partner_id,
                operation_type='sync'
            ).inc()
            
            sync_duration = time.time() - sync_start
            logger.debug("Partner sync completed", 
                        partner_id=partner.partner_id,
                        duration=sync_duration)
        
        except Exception as e:
            logger.error("Partner sync failed", 
                        partner_id=partner.partner_id,
                        error=str(e))
            partner.status = 'error'

    async def _sync_taxii21(self, partner: IntegrationPartner):
        """Synchronize using TAXII 2.1 protocol"""
        base_url = partner.endpoints.get('taxii2_root')
        if not base_url:
            raise ValueError(f"No TAXII 2.1 endpoint for partner {partner.partner_id}")
        
        # Discover available collections
        discovery_url = urljoin(base_url, 'discovery/')
        headers = await self._get_auth_headers(partner)
        headers['Accept'] = 'application/taxii+json;version=2.1'
        
        async with self.http_session.get(discovery_url, headers=headers) as response:
            if response.status == 200:
                discovery_data = await response.json()
                
                # Get available API roots
                for api_root_url in discovery_data.get('api_roots', []):
                    await self._sync_taxii_api_root(partner, api_root_url)
            else:
                logger.warning("TAXII discovery failed", 
                              partner_id=partner.partner_id,
                              status=response.status)

    async def _sync_taxii_api_root(self, partner: IntegrationPartner, api_root_url: str):
        """Sync a specific TAXII API root"""
        headers = await self._get_auth_headers(partner)
        headers['Accept'] = 'application/taxii+json;version=2.1'
        
        # Get collections
        collections_url = urljoin(api_root_url, 'collections/')
        
        async with self.http_session.get(collections_url, headers=headers) as response:
            if response.status == 200:
                collections_data = await response.json()
                
                for collection in collections_data.get('collections', []):
                    collection_id = collection['id']
                    collection_url = urljoin(collections_url, f"{collection_id}/objects/")
                    
                    # Get objects from collection
                    await self._fetch_taxii_objects(partner, collection_url, collection)
            else:
                logger.warning("TAXII collections fetch failed", 
                              partner_id=partner.partner_id,
                              status=response.status)

    async def _fetch_taxii_objects(self, partner: IntegrationPartner, objects_url: str, collection_info: Dict[str, Any]):
        """Fetch STIX objects from TAXII collection"""
        headers = await self._get_auth_headers(partner)
        headers['Accept'] = 'application/stix+json;version=2.1'
        
        # Add time filter for incremental sync
        params = {}
        if partner.last_contact:
            params['added_after'] = partner.last_contact.isoformat()
        
        params['limit'] = self.batch_size
        
        async with self.http_session.get(objects_url, headers=headers, params=params) as response:
            if response.status == 200:
                envelope = await response.json()
                objects = envelope.get('objects', [])
                
                logger.debug("Fetched STIX objects", 
                            partner_id=partner.partner_id,
                            collection=collection_info['id'],
                            count=len(objects))
                
                # Process STIX objects
                for stix_object in objects:
                    await self._process_stix_object(stix_object, partner.partner_id)
                
                threat_intel_received_total.labels(
                    partner=partner.partner_id,
                    indicator_type='stix_objects'
                ).inc(len(objects))
            else:
                logger.warning("TAXII objects fetch failed", 
                              partner_id=partner.partner_id,
                              status=response.status)

    async def _sync_cve_api(self, partner: IntegrationPartner):
        """Synchronize with CVE API"""
        base_url = partner.endpoints.get('api_root')
        if not base_url:
            return
        
        # Get recent CVEs
        cves_url = urljoin(base_url, 'cves/2.0/')
        
        headers = await self._get_auth_headers(partner)
        
        params = {
            'resultsPerPage': self.batch_size,
            'startIndex': 0
        }
        
        # Add time filter for incremental sync
        if partner.last_contact:
            params['lastModStartDate'] = partner.last_contact.isoformat()
        
        async with self.http_session.get(cves_url, headers=headers, params=params) as response:
            if response.status == 200:
                cve_data = await response.json()
                vulnerabilities = cve_data.get('vulnerabilities', [])
                
                logger.debug("Fetched CVE data", 
                            partner_id=partner.partner_id,
                            count=len(vulnerabilities))
                
                # Process CVE data
                for vuln in vulnerabilities:
                    await self._process_cve_data(vuln, partner.partner_id)
                
                threat_intel_received_total.labels(
                    partner=partner.partner_id,
                    indicator_type='cve'
                ).inc(len(vulnerabilities))
            else:
                logger.warning("CVE API fetch failed", 
                              partner_id=partner.partner_id,
                              status=response.status)

    async def _sync_attack_api(self, partner: IntegrationPartner):
        """Synchronize with MITRE ATT&CK API"""
        # This would typically use the TAXII endpoint
        # For now, we'll simulate processing ATT&CK data
        logger.debug("Syncing ATT&CK data", partner_id=partner.partner_id)
        
        # In a real implementation, this would fetch:
        # - Techniques
        # - Tactics  
        # - Groups
        # - Software
        # - Mitigations
        
        # Simulate processing
        await asyncio.sleep(1)
        
        threat_intel_received_total.labels(
            partner=partner.partner_id,
            indicator_type='attack_pattern'
        ).inc(10)  # Simulated count

    async def _sync_misp(self, partner: IntegrationPartner):
        """Synchronize with MISP instance"""
        base_url = partner.endpoints.get('api_root')
        if not base_url:
            return
        
        # Get recent events
        events_url = urljoin(base_url, 'events/restSearch')
        
        headers = await self._get_auth_headers(partner)
        headers['Content-Type'] = 'application/json'
        
        # Search payload
        search_payload = {
            'returnFormat': 'json',
            'limit': self.batch_size,
            'enforceWarninglist': True
        }
        
        # Add time filter
        if partner.last_contact:
            search_payload['timestamp'] = partner.last_contact.timestamp()
        
        async with self.http_session.post(events_url, headers=headers, json=search_payload) as response:
            if response.status == 200:
                events_data = await response.json()
                events = events_data.get('response', [])
                
                logger.debug("Fetched MISP events", 
                            partner_id=partner.partner_id,
                            count=len(events))
                
                # Process MISP events
                for event in events:
                    await self._process_misp_event(event, partner.partner_id)
                
                threat_intel_received_total.labels(
                    partner=partner.partner_id,
                    indicator_type='misp_event'
                ).inc(len(events))
            else:
                logger.warning("MISP sync failed", 
                              partner_id=partner.partner_id,
                              status=response.status)

    async def _process_stix_object(self, stix_object: Dict[str, Any], source_partner: str):
        """Process a STIX 2.1 object"""
        object_type = stix_object.get('type')
        object_id = stix_object.get('id')
        
        # Extract threat indicators
        if object_type == 'indicator':
            await self._extract_indicator_from_stix(stix_object, source_partner)
        elif object_type == 'malware':
            await self._process_malware_stix(stix_object, source_partner)
        elif object_type == 'campaign':
            await self._process_campaign_stix(stix_object, source_partner)
        elif object_type == 'threat-actor':
            await self._process_threat_actor_stix(stix_object, source_partner)
        
        # Cache STIX object
        self.stix_objects_cache[object_id] = {
            'object': stix_object,
            'source': source_partner,
            'cached_at': datetime.utcnow()
        }

    async def _extract_indicator_from_stix(self, stix_indicator: Dict[str, Any], source_partner: str):
        """Extract threat indicator from STIX indicator object"""
        pattern = stix_indicator.get('pattern', '')
        labels = stix_indicator.get('labels', [])
        
        # Parse STIX pattern to extract IoC
        ioc_value = await self._parse_stix_pattern(pattern)
        if not ioc_value:
            return
        
        # Determine indicator type
        indicator_type = await self._determine_indicator_type(ioc_value, pattern)
        
        # Create threat indicator
        indicator = ThreatIndicator(
            indicator_id=stix_indicator['id'],
            indicator_type=indicator_type,
            value=ioc_value,
            confidence=self._extract_confidence(stix_indicator),
            severity=self._extract_severity(labels),
            first_seen=datetime.fromisoformat(stix_indicator.get('created', datetime.utcnow().isoformat()).replace('Z', '+00:00')),
            last_seen=datetime.fromisoformat(stix_indicator.get('modified', datetime.utcnow().isoformat()).replace('Z', '+00:00')),
            source=source_partner,
            tlp_marking=self._extract_tlp_marking(stix_indicator),
            ioc_data={'stix_pattern': pattern, 'labels': labels},
            context=stix_indicator,
            tags=labels,
            relationships=[]
        )
        
        self.threat_indicators[indicator.indicator_id] = indicator
        
        # Persist to database
        await self._persist_threat_indicator(indicator)

    async def _process_cve_data(self, cve_vuln: Dict[str, Any], source_partner: str):
        """Process CVE vulnerability data"""
        cve_data = cve_vuln.get('cve', {})
        cve_id = cve_data.get('id', '')
        
        if not cve_id:
            return
        
        # Extract CVE metrics
        metrics = cve_vuln.get('metrics', {})
        cvss_data = {}
        
        # Get CVSS scores
        for version in ['cvssMetricV31', 'cvssMetricV30', 'cvssMetricV2']:
            if version in metrics:
                cvss_data[version] = metrics[version]
        
        # Create CVE indicator
        indicator = ThreatIndicator(
            indicator_id=f"cve-{cve_id}",
            indicator_type=IndicatorType.CVE,
            value=cve_id,
            confidence=1.0,  # CVE data is authoritative
            severity=self._cvss_to_severity(cvss_data),
            first_seen=datetime.fromisoformat(cve_data.get('published', datetime.utcnow().isoformat()).replace('Z', '+00:00')),
            last_seen=datetime.fromisoformat(cve_data.get('lastModified', datetime.utcnow().isoformat()).replace('Z', '+00:00')),
            source=source_partner,
            tlp_marking='white',  # CVE data is public
            ioc_data={'cvss': cvss_data, 'descriptions': cve_data.get('descriptions', [])},
            context=cve_vuln,
            tags=['vulnerability', 'cve'],
            relationships=[]
        )
        
        self.threat_indicators[indicator.indicator_id] = indicator
        await self._persist_threat_indicator(indicator)

    async def _process_misp_event(self, misp_event: Dict[str, Any], source_partner: str):
        """Process MISP event"""
        event = misp_event.get('Event', {})
        event_id = event.get('id', '')
        
        # Process attributes
        for attribute in event.get('Attribute', []):
            await self._process_misp_attribute(attribute, event_id, source_partner)

    async def _process_misp_attribute(self, attribute: Dict[str, Any], event_id: str, source_partner: str):
        """Process MISP attribute"""
        attr_type = attribute.get('type', '')
        value = attribute.get('value', '')
        
        if not value:
            return
        
        # Map MISP type to indicator type
        indicator_type = self._map_misp_type_to_indicator_type(attr_type)
        if not indicator_type:
            return
        
        # Create indicator
        indicator = ThreatIndicator(
            indicator_id=f"misp-{event_id}-{attribute.get('id', '')}",
            indicator_type=indicator_type,
            value=value,
            confidence=float(attribute.get('confidence', 50)) / 100,
            severity=self._misp_threat_level_to_severity(attribute.get('threat_level_id', '3')),
            first_seen=datetime.fromisoformat(attribute.get('timestamp', datetime.utcnow().isoformat())),
            last_seen=datetime.utcnow(),
            source=source_partner,
            tlp_marking=self._misp_distribution_to_tlp(attribute.get('distribution', '0')),
            ioc_data={'misp_type': attr_type, 'category': attribute.get('category', '')},
            context=attribute,
            tags=[attribute.get('category', '')],
            relationships=[]
        )
        
        self.threat_indicators[indicator.indicator_id] = indicator
        await self._persist_threat_indicator(indicator)

    async def _process_outbound_intelligence(self):
        """Process outbound threat intelligence sharing"""
        redis = aioredis.Redis(connection_pool=self.redis_pool)
        
        # Get pending outbound intelligence
        intel_data = await redis.blpop('xorb:outbound_intel', timeout=5)
        
        if intel_data:
            _, intel_json = intel_data
            intel_item = json.loads(intel_json)
            
            # Determine which partners to share with based on TLP
            tlp_marking = intel_item.get('tlp_marking', self.default_tlp)
            eligible_partners = self._get_eligible_partners_for_tlp(tlp_marking)
            
            # Share with each eligible partner
            for partner_id in eligible_partners:
                if partner_id in self.partners:
                    partner = self.partners[partner_id]
                    await self._share_intelligence_with_partner(intel_item, partner)

    async def _share_intelligence_with_partner(self, intel_item: Dict[str, Any], partner: IntegrationPartner):
        """Share intelligence with a specific partner"""
        try:
            if StandardProtocol.STIX_TAXII_21 in partner.protocols:
                await self._share_via_taxii21(intel_item, partner)
            elif StandardProtocol.MISP in partner.protocols:
                await self._share_via_misp(intel_item, partner)
            elif StandardProtocol.REST_API in partner.protocols:
                await self._share_via_rest_api(intel_item, partner)
            
            threat_intel_shared_total.labels(
                partner=partner.partner_id,
                indicator_type=intel_item.get('indicator_type', 'unknown')
            ).inc()
            
            logger.debug("Intelligence shared with partner", 
                        partner_id=partner.partner_id,
                        indicator_type=intel_item.get('indicator_type'))
        
        except Exception as e:
            logger.error("Failed to share intelligence with partner", 
                        partner_id=partner.partner_id,
                        error=str(e))

    async def _share_via_taxii21(self, intel_item: Dict[str, Any], partner: IntegrationPartner):
        """Share intelligence via TAXII 2.1"""
        # Convert to STIX bundle
        stix_bundle = await self._create_stix_bundle([intel_item])
        
        # Find appropriate collection
        collection_url = await self._get_taxii_collection_url(partner, 'indicators')
        if not collection_url:
            logger.warning("No TAXII collection found for partner", partner_id=partner.partner_id)
            return
        
        headers = await self._get_auth_headers(partner)
        headers['Content-Type'] = 'application/stix+json;version=2.1'
        
        async with self.http_session.post(collection_url, headers=headers, json=stix_bundle) as response:
            if response.status in [200, 201, 202]:
                logger.debug("STIX bundle shared successfully", partner_id=partner.partner_id)
            else:
                logger.warning("STIX bundle sharing failed", 
                              partner_id=partner.partner_id,
                              status=response.status)

    async def _check_partner_health(self, partner: IntegrationPartner):
        """Check partner system health"""
        try:
            # Perform health check based on partner type
            if StandardProtocol.STIX_TAXII_21 in partner.protocols:
                health_url = partner.endpoints.get('discovery', partner.endpoints.get('taxii2_root'))
            else:
                health_url = partner.endpoints.get('health', partner.endpoints.get('api_root'))
            
            if not health_url:
                return
            
            headers = await self._get_auth_headers(partner)
            
            start_time = time.time()
            async with self.http_session.head(health_url, headers=headers) as response:
                response_time = time.time() - start_time
                
                if response.status < 400:
                    partner_availability.labels(partner=partner.partner_id).set(1)
                    partner.status = 'active'
                    logger.debug("Partner health check passed", 
                                partner_id=partner.partner_id,
                                response_time=response_time)
                else:
                    partner_availability.labels(partner=partner.partner_id).set(0)
                    partner.status = 'error'
                    logger.warning("Partner health check failed", 
                                  partner_id=partner.partner_id,
                                  status=response.status)
        
        except Exception as e:
            partner_availability.labels(partner=partner.partner_id).set(0)
            partner.status = 'error'
            logger.error("Partner health check error", 
                        partner_id=partner.partner_id,
                        error=str(e))

    async def _update_compliance_scores(self):
        """Update compliance framework scores"""
        for framework in ComplianceFramework:
            score = await self._calculate_compliance_score(framework)
            self.compliance_scores[framework] = score
            compliance_score.labels(framework=framework.value).set(score)
        
        logger.debug("Compliance scores updated", scores=self.compliance_scores)

    async def _calculate_compliance_score(self, framework: ComplianceFramework) -> float:
        """Calculate compliance score for a framework"""
        mappings = self.compliance_mappings.get(framework, [])
        
        if not mappings:
            return 0.0
        
        total_score = sum(mapping.compliance_score for mapping in mappings)
        return total_score / len(mappings)

    async def _initialize_compliance_mappings(self):
        """Initialize compliance framework mappings"""
        # MITRE ATT&CK mappings
        mitre_mappings = [
            ComplianceMapping(
                framework=ComplianceFramework.MITRE_ATTCK,
                control_id='T1566',
                control_name='Phishing',
                xorb_capability='email_analysis',
                implementation_status='implemented',
                evidence=['email_scanning', 'url_analysis'],
                last_assessed=datetime.utcnow(),
                compliance_score=0.9
            ),
            ComplianceMapping(
                framework=ComplianceFramework.MITRE_ATTCK,
                control_id='T1190',
                control_name='Exploit Public-Facing Application',
                xorb_capability='vulnerability_scanning',
                implementation_status='implemented',
                evidence=['web_app_scanning', 'patch_management'],
                last_assessed=datetime.utcnow(),
                compliance_score=0.85
            )
        ]
        
        # NIST CSF mappings
        nist_mappings = [
            ComplianceMapping(
                framework=ComplianceFramework.NIST_CSF,
                control_id='ID.AM-1',
                control_name='Physical devices and systems within the organization are inventoried',
                xorb_capability='asset_discovery',
                implementation_status='implemented',
                evidence=['network_scanning', 'asset_inventory'],
                last_assessed=datetime.utcnow(),
                compliance_score=0.95
            ),
            ComplianceMapping(
                framework=ComplianceFramework.NIST_CSF,
                control_id='DE.CM-1',
                control_name='The network is monitored to detect potential cybersecurity events',
                xorb_capability='network_monitoring',
                implementation_status='implemented',
                evidence=['traffic_analysis', 'anomaly_detection'],
                last_assessed=datetime.utcnow(),
                compliance_score=0.9
            )
        ]
        
        self.compliance_mappings[ComplianceFramework.MITRE_ATTCK] = mitre_mappings
        self.compliance_mappings[ComplianceFramework.NIST_CSF] = nist_mappings
        
        logger.info("Compliance mappings initialized", 
                   frameworks=len(self.compliance_mappings))

    # Utility methods
    
    async def _get_auth_headers(self, partner: IntegrationPartner) -> Dict[str, str]:
        """Get authentication headers for partner"""
        headers = {}
        auth_config = partner.authentication
        
        if auth_config.get('type') == 'api_key':
            key_location = auth_config.get('key_location', 'header')
            api_key = auth_config.get('api_key', 'dummy_key')  # Would be from secure storage
            
            if key_location == 'header':
                headers['Authorization'] = f"Bearer {api_key}"
            elif key_location == 'query':
                # Would be handled in URL params
                pass
        
        elif auth_config.get('type') == 'basic':
            username = auth_config.get('username', '')
            password = auth_config.get('password', '')  # Would be from secure storage
            
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers['Authorization'] = f"Basic {credentials}"
        
        return headers

    async def _check_rate_limit(self, partner_id: str) -> bool:
        """Check if partner rate limit allows request"""
        if partner_id not in self.rate_limiters:
            return True
        
        limiter = self.rate_limiters[partner_id]
        current_time = time.time()
        
        # Reset counter if hour has passed
        if current_time - limiter['last_reset'] >= 3600:
            limiter['requests'] = 0
            limiter['last_reset'] = current_time
        
        # Check limit
        if limiter['requests'] >= limiter['limit']:
            return False
        
        limiter['requests'] += 1
        return True

    async def _parse_stix_pattern(self, pattern: str) -> Optional[str]:
        """Parse STIX pattern to extract IoC value"""
        # Simplified STIX pattern parsing
        if "file:hashes.MD5" in pattern:
            # Extract MD5 hash
            import re
            match = re.search(r"file:hashes\.MD5\s*=\s*'([^']+)'", pattern)
            return match.group(1) if match else None
        
        elif "domain-name:value" in pattern:
            # Extract domain
            import re
            match = re.search(r"domain-name:value\s*=\s*'([^']+)'", pattern)
            return match.group(1) if match else None
        
        elif "ipv4-addr:value" in pattern:
            # Extract IP address
            import re
            match = re.search(r"ipv4-addr:value\s*=\s*'([^']+)'", pattern)
            return match.group(1) if match else None
        
        elif "url:value" in pattern:
            # Extract URL
            import re
            match = re.search(r"url:value\s*=\s*'([^']+)'", pattern)
            return match.group(1) if match else None
        
        return None

    async def _determine_indicator_type(self, value: str, pattern: str) -> IndicatorType:
        """Determine indicator type from value and pattern"""
        if "file:hashes" in pattern:
            return IndicatorType.FILE_HASH
        elif "domain-name:value" in pattern:
            return IndicatorType.DOMAIN
        elif "ipv4-addr:value" in pattern or "ipv6-addr:value" in pattern:
            return IndicatorType.IP_ADDRESS
        elif "url:value" in pattern:
            return IndicatorType.URL
        elif "email-addr:value" in pattern:
            return IndicatorType.EMAIL
        else:
            # Try to infer from value
            import re
            if re.match(r'^[0-9a-fA-F]{32}$', value):  # MD5
                return IndicatorType.FILE_HASH
            elif re.match(r'^[0-9a-fA-F]{40}$', value):  # SHA1
                return IndicatorType.FILE_HASH
            elif re.match(r'^[0-9a-fA-F]{64}$', value):  # SHA256
                return IndicatorType.FILE_HASH
            elif re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', value):  # IPv4
                return IndicatorType.IP_ADDRESS
            elif '.' in value and not value.startswith('http'):  # Domain
                return IndicatorType.DOMAIN
            elif value.startswith('http'):  # URL
                return IndicatorType.URL
            else:
                return IndicatorType.DOMAIN  # Default

    def _extract_confidence(self, stix_object: Dict[str, Any]) -> float:
        """Extract confidence score from STIX object"""
        # STIX 2.1 confidence is 0-100
        confidence = stix_object.get('confidence', 50)
        return confidence / 100.0

    def _extract_severity(self, labels: List[str]) -> str:
        """Extract severity from STIX labels"""
        if any(label in ['malicious-activity', 'attribution'] for label in labels):
            return 'high'
        elif any(label in ['anomalous-activity'] for label in labels):
            return 'medium'
        else:
            return 'low'

    def _extract_tlp_marking(self, stix_object: Dict[str, Any]) -> str:
        """Extract TLP marking from STIX object"""
        markings = stix_object.get('object_marking_refs', [])
        
        # Common TLP marking identifiers
        tlp_mappings = {
            'marking-definition--f88d31f6-486f-44da-b317-01333bde0b82': 'white',
            'marking-definition--34098fce-860f-48ae-8e50-ebd3cc5e41da': 'green',
            'marking-definition--f88d31f6-486f-44da-b317-01333bde0b82': 'amber',
            'marking-definition--5e57c739-391a-4eb3-b6be-7d15ca92d5ed': 'red'
        }
        
        for marking in markings:
            if marking in tlp_mappings:
                return tlp_mappings[marking]
        
        return self.default_tlp

    def _cvss_to_severity(self, cvss_data: Dict[str, Any]) -> str:
        """Convert CVSS score to severity"""
        # Try CVSS v3.1 first, then v3.0, then v2
        for version in ['cvssMetricV31', 'cvssMetricV30', 'cvssMetricV2']:
            if version in cvss_data and cvss_data[version]:
                score = cvss_data[version][0].get('cvssData', {}).get('baseScore', 0)
                
                if score >= 9.0:
                    return 'critical'
                elif score >= 7.0:
                    return 'high'
                elif score >= 4.0:
                    return 'medium'
                else:
                    return 'low'
        
        return 'medium'  # Default

    def _map_misp_type_to_indicator_type(self, misp_type: str) -> Optional[IndicatorType]:
        """Map MISP attribute type to indicator type"""
        mapping = {
            'ip-src': IndicatorType.IP_ADDRESS,
            'ip-dst': IndicatorType.IP_ADDRESS,
            'domain': IndicatorType.DOMAIN,
            'hostname': IndicatorType.DOMAIN,
            'url': IndicatorType.URL,
            'md5': IndicatorType.FILE_HASH,
            'sha1': IndicatorType.FILE_HASH,
            'sha256': IndicatorType.FILE_HASH,
            'email-src': IndicatorType.EMAIL,
            'email-dst': IndicatorType.EMAIL,
            'malware-type': IndicatorType.MALWARE_FAMILY
        }
        
        return mapping.get(misp_type)

    def _misp_threat_level_to_severity(self, threat_level_id: str) -> str:
        """Convert MISP threat level to severity"""
        levels = {
            '1': 'high',
            '2': 'medium',
            '3': 'low',
            '4': 'low'
        }
        return levels.get(threat_level_id, 'medium')

    def _misp_distribution_to_tlp(self, distribution: str) -> str:
        """Convert MISP distribution to TLP"""
        mapping = {
            '0': 'red',      # Your organization only
            '1': 'amber',    # This community only
            '2': 'amber',    # Connected communities
            '3': 'green',    # All communities
            '4': 'white',    # Sharing group
            '5': 'white'     # Inherit event
        }
        return mapping.get(distribution, 'amber')

    def _get_eligible_partners_for_tlp(self, tlp_marking: str) -> List[str]:
        """Get partners eligible for sharing based on TLP marking"""
        eligible = []
        sharing_policy = self.sharing_policy.get(tlp_marking, [])
        
        for partner_id, partner in self.partners.items():
            if partner.status != 'active':
                continue
            
            agreement = partner.data_sharing_agreement
            allowed_tlp = agreement.get('tlp_levels', [])
            
            if tlp_marking in allowed_tlp:
                # Check sharing policy
                if 'all' in sharing_policy:
                    eligible.append(partner_id)
                elif 'partners' in sharing_policy and partner.partner_type in [IntegrationType.CERT, IntegrationType.ISAC]:
                    eligible.append(partner_id)
                elif 'community' in sharing_policy and partner.partner_type == IntegrationType.COMMUNITY:
                    eligible.append(partner_id)
        
        return eligible

    async def _create_stix_bundle(self, intel_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create STIX 2.1 bundle from intelligence items"""
        bundle_id = f"bundle--{uuid.uuid4()}"
        
        objects = []
        
        for item in intel_items:
            # Create STIX indicator object
            indicator_id = f"indicator--{uuid.uuid4()}"
            
            # Create STIX pattern
            pattern = await self._create_stix_pattern(item)
            
            stix_indicator = {
                "type": "indicator",
                "spec_version": "2.1",
                "id": indicator_id,
                "created": datetime.utcnow().isoformat() + "Z",
                "modified": datetime.utcnow().isoformat() + "Z",
                "pattern": pattern,
                "labels": [item.get('indicator_type', 'unknown')],
                "confidence": int(item.get('confidence', 0.5) * 100)
            }
            
            objects.append(stix_indicator)
        
        bundle = {
            "type": "bundle",
            "id": bundle_id,
            "spec_version": "2.1",
            "objects": objects
        }
        
        return bundle

    async def _create_stix_pattern(self, intel_item: Dict[str, Any]) -> str:
        """Create STIX pattern from intelligence item"""
        indicator_type = intel_item.get('indicator_type')
        value = intel_item.get('value', '')
        
        if indicator_type == 'ip_address':
            return f"[ipv4-addr:value = '{value}']"
        elif indicator_type == 'domain':
            return f"[domain-name:value = '{value}']"
        elif indicator_type == 'url':
            return f"[url:value = '{value}']"
        elif indicator_type == 'file_hash':
            # Determine hash type by length
            if len(value) == 32:
                return f"[file:hashes.MD5 = '{value}']"
            elif len(value) == 40:
                return f"[file:hashes.SHA-1 = '{value}']"
            elif len(value) == 64:
                return f"[file:hashes.SHA-256 = '{value}']"
        elif indicator_type == 'email':
            return f"[email-addr:value = '{value}']"
        
        return f"[x-unknown:value = '{value}']"

    async def _load_taxii_collections(self):
        """Load TAXII collection information"""
        for partner in self.partners.values():
            if StandardProtocol.STIX_TAXII_21 in partner.protocols:
                await self._discover_taxii_collections(partner)

    async def _discover_taxii_collections(self, partner: IntegrationPartner):
        """Discover TAXII collections for a partner"""
        try:
            base_url = partner.endpoints.get('taxii2_root')
            if not base_url:
                return
            
            discovery_url = urljoin(base_url, 'discovery/')
            headers = await self._get_auth_headers(partner)
            headers['Accept'] = 'application/taxii+json;version=2.1'
            
            async with self.http_session.get(discovery_url, headers=headers) as response:
                if response.status == 200:
                    discovery_data = await response.json()
                    
                    for api_root_url in discovery_data.get('api_roots', []):
                        collections_url = urljoin(api_root_url, 'collections/')
                        
                        async with self.http_session.get(collections_url, headers=headers) as coll_response:
                            if coll_response.status == 200:
                                collections_data = await coll_response.json()
                                
                                partner_collections = {}
                                for collection in collections_data.get('collections', []):
                                    partner_collections[collection['id']] = {
                                        'title': collection.get('title', ''),
                                        'description': collection.get('description', ''),
                                        'can_read': collection.get('can_read', False),
                                        'can_write': collection.get('can_write', False),
                                        'media_types': collection.get('media_types', []),
                                        'url': urljoin(collections_url, f"{collection['id']}/")
                                    }
                                
                                self.taxii_collections[partner.partner_id] = partner_collections
                                
                                logger.debug("TAXII collections discovered", 
                                            partner_id=partner.partner_id,
                                            collections=len(partner_collections))
        
        except Exception as e:
            logger.error("TAXII collection discovery failed", 
                        partner_id=partner.partner_id,
                        error=str(e))

    async def _update_taxii_collections(self, partner: IntegrationPartner):
        """Update TAXII collection information"""
        await self._discover_taxii_collections(partner)

    async def _get_taxii_collection_url(self, partner: IntegrationPartner, collection_type: str) -> Optional[str]:
        """Get TAXII collection URL for a specific type"""
        collections = self.taxii_collections.get(partner.partner_id, {})
        
        # Look for appropriate collection
        for collection_id, collection_info in collections.items():
            title = collection_info.get('title', '').lower()
            description = collection_info.get('description', '').lower()
            
            if collection_type in title or collection_type in description:
                if collection_info.get('can_write', False):
                    return urljoin(collection_info['url'], 'objects/')
        
        # Return first writable collection as fallback
        for collection_info in collections.values():
            if collection_info.get('can_write', False):
                return urljoin(collection_info['url'], 'objects/')
        
        return None

    async def _persist_partner(self, partner: IntegrationPartner):
        """Persist partner configuration to database"""
        try:
            async with self.db_pool.acquire() as conn:
                partner_data = json.dumps(asdict(partner), default=str)
                await conn.execute("""
                    INSERT INTO integration_partners 
                    (partner_id, name, partner_type, partner_data, active, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (partner_id) DO UPDATE SET
                    partner_data = $4, updated_at = CURRENT_TIMESTAMP
                """, partner.partner_id, partner.name, partner.partner_type.value,
                    partner_data, True, datetime.utcnow())
        
        except Exception as e:
            logger.error("Failed to persist partner", 
                        partner_id=partner.partner_id, error=str(e))

    async def _persist_threat_indicator(self, indicator: ThreatIndicator):
        """Persist threat indicator to database"""
        try:
            async with self.db_pool.acquire() as conn:
                indicator_data = json.dumps(asdict(indicator), default=str)
                await conn.execute("""
                    INSERT INTO threat_indicators 
                    (indicator_id, indicator_type, value, confidence, severity, 
                     source, tlp_marking, indicator_data, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (indicator_id) DO UPDATE SET
                    confidence = $4, severity = $5, indicator_data = $8,
                    updated_at = CURRENT_TIMESTAMP
                """, indicator.indicator_id, indicator.indicator_type.value,
                    indicator.value, indicator.confidence, indicator.severity,
                    indicator.source, indicator.tlp_marking, indicator_data,
                    indicator.first_seen)
        
        except Exception as e:
            logger.error("Failed to persist threat indicator", 
                        indicator_id=indicator.indicator_id, error=str(e))

    async def _initialize_database(self):
        """Initialize database schema"""
        async with self.db_pool.acquire() as conn:
            # Integration partners table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS integration_partners (
                    partner_id VARCHAR PRIMARY KEY,
                    name VARCHAR NOT NULL,
                    partner_type VARCHAR NOT NULL,
                    partner_data JSONB NOT NULL,
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Threat indicators table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS threat_indicators (
                    indicator_id VARCHAR PRIMARY KEY,
                    indicator_type VARCHAR NOT NULL,
                    value VARCHAR NOT NULL,
                    confidence REAL NOT NULL,
                    severity VARCHAR NOT NULL,
                    source VARCHAR NOT NULL,
                    tlp_marking VARCHAR NOT NULL,
                    indicator_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Compliance mappings table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_mappings (
                    id SERIAL PRIMARY KEY,
                    framework VARCHAR NOT NULL,
                    control_id VARCHAR NOT NULL,
                    control_name VARCHAR NOT NULL,
                    xorb_capability VARCHAR NOT NULL,
                    implementation_status VARCHAR NOT NULL,
                    compliance_score REAL NOT NULL,
                    mapping_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(framework, control_id)
                )
            """)
            
            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_partners_type ON integration_partners(partner_type)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_indicators_type ON threat_indicators(indicator_type)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_indicators_source ON threat_indicators(source)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_indicators_tlp ON threat_indicators(tlp_marking)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_compliance_framework ON compliance_mappings(framework)")

    # Public API methods
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status"""
        active_partners = len([p for p in self.partners.values() if p.status == 'active'])
        total_indicators = len(self.threat_indicators)
        
        return {
            "status": "running" if self.is_running else "stopped",
            "partners": {
                "total": len(self.partners),
                "active": active_partners,
                "by_type": {
                    partner_type.value: len([p for p in self.partners.values() 
                                           if p.partner_type == partner_type])
                    for partner_type in IntegrationType
                }
            },
            "threat_intelligence": {
                "total_indicators": total_indicators,
                "by_type": {
                    indicator_type.value: len([i for i in self.threat_indicators.values() 
                                             if i.indicator_type == indicator_type])
                    for indicator_type in IndicatorType
                }
            },
            "compliance_scores": self.compliance_scores,
            "sharing_stats": self.sharing_stats,
            "agent_id": self.agent_id
        }

    async def add_integration_partner(self, partner_config: Dict[str, Any]) -> str:
        """Add a new integration partner"""
        partner = IntegrationPartner(**partner_config)
        self.partners[partner.partner_id] = partner
        
        # Initialize rate limiter
        self.rate_limiters[partner.partner_id] = {
            'requests': 0,
            'last_reset': time.time(),
            'limit': partner.rate_limits.get('requests_per_hour', 1000)
        }
        
        # Persist to database
        await self._persist_partner(partner)
        
        logger.info("Integration partner added", partner_id=partner.partner_id)
        return partner.partner_id

    async def share_threat_intelligence(self, indicators: List[Dict[str, Any]], tlp_marking: str = None):
        """Share threat intelligence with appropriate partners"""
        redis = aioredis.Redis(connection_pool=self.redis_pool)
        
        tlp = tlp_marking or self.default_tlp
        
        for indicator in indicators:
            intel_item = {
                **indicator,
                'tlp_marking': tlp,
                'shared_at': datetime.utcnow().isoformat(),
                'source': 'xorb'
            }
            
            await redis.lpush('xorb:outbound_intel', json.dumps(intel_item))
        
        logger.info("Threat intelligence queued for sharing", 
                   count=len(indicators), tlp=tlp)

    async def shutdown(self):
        """Shutdown the ecosystem integration agent"""
        logger.info("Shutting down EcosystemIntegrationAgent")
        
        self.is_running = False
        
        # Close HTTP sessions
        for session in self.active_connections.values():
            await session.close()
        
        if self.http_session:
            await self.http_session.close()
        
        # Close connections
        if self.redis_pool:
            await self.redis_pool.disconnect()
        
        if self.db_pool:
            await self.db_pool.close()
        
        logger.info("EcosystemIntegrationAgent shutdown complete")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="XORB Ecosystem Integration Agent")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--sync-interval", type=int, default=3600, help="Sync interval in seconds")
    
    args = parser.parse_args()
    
    config = {
        'sync_interval': args.sync_interval,
        'redis_url': 'redis://localhost:6379',
        'postgres_url': 'postgresql://localhost:5432/xorb'
    }
    
    async def main():
        agent = EcosystemIntegrationAgent(config)
        await agent.initialize()
        await agent.start_integration()
    
    asyncio.run(main())