"""Threat Intelligence Service - Real-time threat feeds and ML classification."""
import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

import httpx
from sqlalchemy import select, update, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth.models import UserClaims
from ..domain.tenant_entities import Finding
from ..infrastructure.database import get_async_session
from ..infrastructure.vector_store import get_vector_store
from ..infrastructure.observability import get_metrics_collector, add_trace_context
from ..jobs.models import JobType, JobPriority, JobScheduleRequest
from ..jobs.service import JobService
import structlog

logger = structlog.get_logger("threat_intelligence")


class ThreatIndicator:
    """Threat indicator with enrichment data."""
    
    def __init__(
        self,
        ioc_type: str,
        value: str,
        confidence: float,
        severity: str,
        sources: List[str],
        tags: List[str],
        first_seen: datetime,
        last_seen: datetime,
        metadata: Optional[Dict] = None
    ):
        self.id = uuid4()
        self.ioc_type = ioc_type  # ip, domain, hash, url, etc.
        self.value = value
        self.confidence = confidence  # 0.0 - 1.0
        self.severity = severity  # low, medium, high, critical
        self.sources = sources
        self.tags = tags
        self.first_seen = first_seen
        self.last_seen = last_seen
        self.metadata = metadata or {}
        
        # Generate hash for deduplication
        self.hash = hashlib.sha256(f"{ioc_type}:{value}".encode()).hexdigest()


class ThreatFeed:
    """Threat intelligence feed configuration."""
    
    def __init__(
        self,
        name: str,
        url: str,
        feed_type: str,
        format_type: str = "json",
        api_key: Optional[str] = None,
        update_interval: int = 3600,  # seconds
        enabled: bool = True
    ):
        self.name = name
        self.url = url
        self.feed_type = feed_type  # commercial, open_source, internal
        self.format_type = format_type  # json, xml, csv, stix
        self.api_key = api_key
        self.update_interval = update_interval
        self.enabled = enabled
        self.last_updated: Optional[datetime] = None


class ThreatIntelligenceEngine:
    """Main threat intelligence processing engine."""
    
    def __init__(self, job_service: JobService):
        self.job_service = job_service
        self.vector_store = get_vector_store()
        self.metrics = get_metrics_collector()
        
        # Default threat feeds
        self.feeds = [
            ThreatFeed(
                name="MISP Threat Sharing",
                url="https://www.circl.lu/doc/misp/feed-osint/",
                feed_type="open_source",
                format_type="json"
            ),
            ThreatFeed(
                name="Abuse.ch Malware Hashes",
                url="https://bazaar.abuse.ch/export/txt/sha256/recent/",
                feed_type="open_source", 
                format_type="csv"
            ),
            ThreatFeed(
                name="URLhaus Malicious URLs",
                url="https://urlhaus.abuse.ch/downloads/csv_recent/",
                feed_type="open_source",
                format_type="csv"
            )
        ]
        
        # In-memory cache for recent indicators
        self.indicator_cache: Dict[str, ThreatIndicator] = {}
        self.cache_ttl = 3600  # 1 hour
    
    async def process_threat_feeds(self, tenant_id: UUID) -> Dict[str, int]:
        """Process all enabled threat feeds for a tenant."""
        results = {
            "feeds_processed": 0,
            "indicators_added": 0,
            "indicators_updated": 0,
            "errors": 0
        }
        
        start_time = datetime.utcnow()
        
        for feed in self.feeds:
            if not feed.enabled:
                continue
                
            try:
                indicators = await self._fetch_feed_data(feed)
                
                for indicator in indicators:
                    try:
                        # Store indicator with tenant isolation
                        await self._store_threat_indicator(indicator, tenant_id)
                        
                        # Generate vector embedding for similarity search
                        await self._generate_indicator_embedding(indicator, tenant_id)
                        
                        results["indicators_added"] += 1
                        
                    except Exception as e:
                        logger.error(
                            "Failed to process indicator",
                            indicator_value=indicator.value,
                            error=str(e)
                        )
                        results["errors"] += 1
                
                feed.last_updated = datetime.utcnow()
                results["feeds_processed"] += 1
                
                logger.info(
                    "Processed threat feed",
                    feed_name=feed.name,
                    indicators_count=len(indicators),
                    tenant_id=str(tenant_id)
                )
                
            except Exception as e:
                logger.error(
                    "Failed to process threat feed",
                    feed_name=feed.name,
                    error=str(e)
                )
                results["errors"] += 1
        
        # Record metrics
        duration = (datetime.utcnow() - start_time).total_seconds()
        self.metrics.record_job_execution("threat_feed_processing", duration, results["errors"] == 0)
        
        add_trace_context(
            operation="threat_feed_processing",
            tenant_id=str(tenant_id),
            feeds_processed=results["feeds_processed"],
            indicators_added=results["indicators_added"]
        )
        
        return results
    
    async def _fetch_feed_data(self, feed: ThreatFeed) -> List[ThreatIndicator]:
        """Fetch and parse threat feed data."""
        indicators = []
        
        headers = {"User-Agent": "XORB-ThreatIntel/1.0"}
        if feed.api_key:
            headers["Authorization"] = f"Bearer {feed.api_key}"
        
        try:
            async with httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            ) as client:
                response = await client.get(feed.url, headers=headers)
                response.raise_for_status()
                
                # Handle different content types
                content_type = response.headers.get("content-type", "").lower()
                
                if feed.format_type == "json" or "json" in content_type:
                    try:
                        data = response.json()
                        indicators = self._parse_json_feed(data, feed)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in feed {feed.name}: {e}")
                        
                elif feed.format_type == "csv" or "csv" in content_type:
                    text_data = response.text
                    indicators = self._parse_csv_feed(text_data, feed)
                    
                elif feed.format_type == "stix":
                    try:
                        data = response.json()
                        indicators = self._parse_stix_feed(data, feed)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid STIX JSON in feed {feed.name}: {e}")
                        
                elif feed.format_type == "xml":
                    # Basic XML parsing for feeds like RSS/Atom
                    indicators = self._parse_xml_feed(response.text, feed)
                    
                else:
                    # Try to auto-detect format
                    text_data = response.text.strip()
                    if text_data.startswith('{') or text_data.startswith('['):
                        try:
                            data = response.json()
                            indicators = self._parse_json_feed(data, feed)
                        except:
                            indicators = self._parse_text_feed(text_data, feed)
                    elif ',' in text_data or '\t' in text_data:
                        indicators = self._parse_csv_feed(text_data, feed)
                    else:
                        indicators = self._parse_text_feed(text_data, feed)
                
                logger.info(f"Fetched {len(indicators)} indicators from {feed.name}")
                
        except httpx.TimeoutException:
            logger.error(f"Timeout fetching feed {feed.name}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code} fetching feed {feed.name}")
        except httpx.RequestError as e:
            logger.error(f"Request error fetching feed {feed.name}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching feed {feed.name}: {e}")
        
        return indicators
    
    def _parse_json_feed(self, data: Dict, feed: ThreatFeed) -> List[ThreatIndicator]:
        """Parse JSON threat feed data."""
        indicators = []
        
        # Generic JSON parser - adapt based on feed structure
        if isinstance(data, list):
            items = data
        elif "indicators" in data:
            items = data["indicators"]
        elif "objects" in data:
            items = data["objects"]
        else:
            items = [data]
        
        for item in items[:1000]:  # Limit to prevent memory issues
            try:
                indicator = ThreatIndicator(
                    ioc_type=item.get("type", "unknown"),
                    value=item.get("value", item.get("pattern", "")),
                    confidence=float(item.get("confidence", 0.5)),
                    severity=item.get("severity", "medium"),
                    sources=[feed.name],
                    tags=item.get("tags", []),
                    first_seen=self._parse_timestamp(item.get("first_seen")),
                    last_seen=self._parse_timestamp(item.get("last_seen")),
                    metadata=item
                )
                
                if indicator.value:  # Only add if we have a value
                    indicators.append(indicator)
                    
            except Exception as e:
                logger.debug(f"Failed to parse indicator: {e}")
        
        return indicators
    
    def _parse_csv_feed(self, text_data: str, feed: ThreatFeed) -> List[ThreatIndicator]:
        """Parse CSV threat feed data."""
        indicators = []
        lines = text_data.strip().split('\n')
        
        # Skip header if present
        data_lines = lines[1:] if lines and not lines[0].startswith('#') else lines
        
        for line in data_lines[:1000]:  # Limit to prevent memory issues
            if line.startswith('#') or not line.strip():
                continue
                
            try:
                parts = line.split(',')
                if len(parts) >= 1:
                    value = parts[0].strip().strip('"')
                    
                    # Determine IOC type based on value
                    ioc_type = self._detect_ioc_type(value)
                    
                    indicator = ThreatIndicator(
                        ioc_type=ioc_type,
                        value=value,
                        confidence=0.7,  # Default confidence for CSV feeds
                        severity="medium",
                        sources=[feed.name],
                        tags=[],
                        first_seen=datetime.utcnow(),
                        last_seen=datetime.utcnow(),
                        metadata={"raw_line": line}
                    )
                    
                    indicators.append(indicator)
                    
            except Exception as e:
                logger.debug(f"Failed to parse CSV line: {e}")
        
        return indicators
    
    def _parse_stix_feed(self, data: Dict, feed: ThreatFeed) -> List[ThreatIndicator]:
        """Parse STIX format threat feed data."""
        indicators = []
        
        # STIX 2.0/2.1 format
        if "objects" in data:
            for obj in data["objects"]:
                if obj.get("type") == "indicator":
                    try:
                        pattern = obj.get("pattern", "")
                        value = self._extract_value_from_stix_pattern(pattern)
                        
                        indicator = ThreatIndicator(
                            ioc_type=self._detect_ioc_type(value),
                            value=value,
                            confidence=float(obj.get("confidence", 50)) / 100.0,
                            severity=self._map_stix_severity(obj.get("labels", [])),
                            sources=[feed.name],
                            tags=obj.get("labels", []),
                            first_seen=self._parse_timestamp(obj.get("created")),
                            last_seen=self._parse_timestamp(obj.get("modified")),
                            metadata=obj
                        )
                        
                        indicators.append(indicator)
                        
                    except Exception as e:
                        logger.debug(f"Failed to parse STIX indicator: {e}")
        
        return indicators
    
    def _parse_xml_feed(self, xml_data: str, feed: ThreatFeed) -> List[ThreatIndicator]:
        """Parse XML threat feed data (RSS/Atom style)."""
        indicators = []
        
        try:
            # Basic XML parsing without external dependencies
            import re
            
            # Extract items/entries from XML
            item_pattern = r'<(?:item|entry)>(.*?)</(?:item|entry)>'
            items = re.findall(item_pattern, xml_data, re.DOTALL | re.IGNORECASE)
            
            for item in items[:1000]:  # Limit items
                try:
                    # Extract common fields
                    title_match = re.search(r'<title[^>]*>(.*?)</title>', item, re.DOTALL | re.IGNORECASE)
                    desc_match = re.search(r'<(?:description|summary)[^>]*>(.*?)</(?:description|summary)>', item, re.DOTALL | re.IGNORECASE)
                    link_match = re.search(r'<link[^>]*>([^<]*)</link>', item, re.IGNORECASE)
                    
                    title = title_match.group(1).strip() if title_match else ""
                    description = desc_match.group(1).strip() if desc_match else ""
                    link = link_match.group(1).strip() if link_match else ""
                    
                    # Try to extract IOCs from title and description
                    content = f"{title} {description}".lower()
                    
                    # Look for IP addresses
                    ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
                    ips = re.findall(ip_pattern, content)
                    
                    # Look for domains
                    domain_pattern = r'\b[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}\b'
                    domains = re.findall(domain_pattern, content)
                    
                    # Look for hashes
                    hash_pattern = r'\b[a-fA-F0-9]{32,64}\b'
                    hashes = re.findall(hash_pattern, content)
                    
                    # Create indicators for found IOCs
                    for ip in ips:
                        indicator = ThreatIndicator(
                            ioc_type="ip",
                            value=ip,
                            confidence=0.6,
                            severity="medium",
                            sources=[feed.name],
                            tags=["xml_feed"],
                            first_seen=datetime.utcnow(),
                            last_seen=datetime.utcnow(),
                            metadata={"title": title, "description": description, "link": link}
                        )
                        indicators.append(indicator)
                    
                    for domain in domains:
                        if not any(domain.endswith(tld) for tld in ['.com', '.org', '.net', '.gov', '.edu', '.io', '.co']):
                            continue  # Skip non-standard TLDs that might be false positives
                        
                        indicator = ThreatIndicator(
                            ioc_type="domain",
                            value=domain,
                            confidence=0.6,
                            severity="medium", 
                            sources=[feed.name],
                            tags=["xml_feed"],
                            first_seen=datetime.utcnow(),
                            last_seen=datetime.utcnow(),
                            metadata={"title": title, "description": description, "link": link}
                        )
                        indicators.append(indicator)
                    
                    for hash_val in hashes:
                        hash_type = "md5" if len(hash_val) == 32 else "sha1" if len(hash_val) == 40 else "sha256"
                        
                        indicator = ThreatIndicator(
                            ioc_type=hash_type,
                            value=hash_val.lower(),
                            confidence=0.8,
                            severity="high",
                            sources=[feed.name],
                            tags=["xml_feed", "malware"],
                            first_seen=datetime.utcnow(),
                            last_seen=datetime.utcnow(),
                            metadata={"title": title, "description": description, "link": link}
                        )
                        indicators.append(indicator)
                        
                except Exception as e:
                    logger.debug(f"Failed to parse XML item: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to parse XML feed: {e}")
        
        return indicators
    
    def _parse_text_feed(self, text_data: str, feed: ThreatFeed) -> List[ThreatIndicator]:
        """Parse plain text threat feed data."""
        indicators = []
        
        try:
            lines = text_data.strip().split('\n')
            
            for line in lines[:1000]:  # Limit lines
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('//'):
                    continue
                
                # Extract the first token as potential IOC
                ioc_value = line.split()[0] if line.split() else line
                
                # Detect IOC type and create indicator
                ioc_type = self._detect_ioc_type(ioc_value)
                
                if ioc_type != "unknown":
                    indicator = ThreatIndicator(
                        ioc_type=ioc_type,
                        value=ioc_value,
                        confidence=0.7,
                        severity="medium",
                        sources=[feed.name],
                        tags=["text_feed"],
                        first_seen=datetime.utcnow(),
                        last_seen=datetime.utcnow(),
                        metadata={"raw_line": line}
                    )
                    indicators.append(indicator)
                    
        except Exception as e:
            logger.error(f"Failed to parse text feed: {e}")
        
        return indicators
    
    def _detect_ioc_type(self, value: str) -> str:
        """Detect IOC type from value."""
        import re
        
        # IP address
        if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', value):
            return "ip"
        
        # Domain
        if re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value):
            return "domain"
        
        # URL
        if value.startswith(('http://', 'https://', 'ftp://')):
            return "url"
        
        # Hash (MD5, SHA1, SHA256)
        if re.match(r'^[a-fA-F0-9]{32}$', value):
            return "md5"
        elif re.match(r'^[a-fA-F0-9]{40}$', value):
            return "sha1"
        elif re.match(r'^[a-fA-F0-9]{64}$', value):
            return "sha256"
        
        # Email
        if '@' in value and '.' in value:
            return "email"
        
        return "unknown"
    
    def _extract_value_from_stix_pattern(self, pattern: str) -> str:
        """Extract IOC value from STIX pattern."""
        import re
        
        # Simple regex to extract values from STIX patterns
        # e.g., "[file:hashes.MD5 = 'd41d8cd98f00b204e9800998ecf8427e']"
        match = re.search(r"'([^']+)'", pattern)
        if match:
            return match.group(1)
        
        match = re.search(r'"([^"]+)"', pattern)
        if match:
            return match.group(1)
        
        return pattern
    
    def _map_stix_severity(self, labels: List[str]) -> str:
        """Map STIX labels to severity levels."""
        severity_map = {
            "malicious-activity": "high",
            "attribution": "medium",
            "anomalous-activity": "medium",
            "benign": "low"
        }
        
        for label in labels:
            if label in severity_map:
                return severity_map[label]
        
        return "medium"
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> datetime:
        """Parse timestamp string to datetime."""
        if not timestamp_str:
            return datetime.utcnow()
        
        try:
            # Try common timestamp formats
            for fmt in [
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ", 
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d"
            ]:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue
        except Exception:
            pass
        
        return datetime.utcnow()
    
    async def _store_threat_indicator(self, indicator: ThreatIndicator, tenant_id: UUID) -> None:
        """Store threat indicator with tenant isolation."""
        # Store in cache
        self.indicator_cache[indicator.hash] = indicator
        
        # Store in database as a finding
        async with get_async_session() as session:
            # Set tenant context
            await session.execute(
                "SELECT set_config('app.tenant_id', :tenant_id, false)",
                {"tenant_id": str(tenant_id)}
            )
            
            # Check if indicator already exists
            existing = await session.execute(
                select(Finding).where(
                    and_(
                        Finding.tenant_id == tenant_id,
                        Finding.custom_metadata["ioc_hash"].astext == indicator.hash
                    )
                )
            )
            
            if existing.scalar_one_or_none():
                # Update existing
                await session.execute(
                    update(Finding)
                    .where(Finding.custom_metadata["ioc_hash"].astext == indicator.hash)
                    .values(
                        updated_at=datetime.utcnow(),
                        custom_metadata={
                            "ioc_hash": indicator.hash,
                            "ioc_type": indicator.ioc_type,
                            "ioc_value": indicator.value,
                            "confidence": indicator.confidence,
                            "sources": indicator.sources,
                            "last_seen": indicator.last_seen.isoformat()
                        }
                    )
                )
            else:
                # Create new finding
                finding = Finding(
                    tenant_id=tenant_id,
                    title=f"Threat Indicator: {indicator.ioc_type.upper()}",
                    description=f"Malicious {indicator.ioc_type}: {indicator.value}",
                    severity=indicator.severity,
                    category="threat_intelligence",
                    tags=["threat_intel", indicator.ioc_type] + indicator.tags,
                    custom_metadata={
                        "ioc_hash": indicator.hash,
                        "ioc_type": indicator.ioc_type,
                        "ioc_value": indicator.value,
                        "confidence": indicator.confidence,
                        "sources": indicator.sources,
                        "first_seen": indicator.first_seen.isoformat(),
                        "last_seen": indicator.last_seen.isoformat(),
                        "metadata": indicator.metadata
                    },
                    created_by="threat_intelligence_engine"
                )
                
                session.add(finding)
            
            await session.commit()
    
    async def _generate_indicator_embedding(self, indicator: ThreatIndicator, tenant_id: UUID) -> None:
        """Generate vector embedding for threat indicator."""
        try:
            # Create text representation for embedding
            text_content = f"""
            Type: {indicator.ioc_type}
            Value: {indicator.value}
            Severity: {indicator.severity}
            Tags: {', '.join(indicator.tags)}
            Sources: {', '.join(indicator.sources)}
            """
            
            # Generate embedding (would use actual embedding service)
            from ..infrastructure.vector_store import openai_embedding_function
            embedding = await openai_embedding_function(text_content)
            
            # Store in vector database
            await self.vector_store.add_vector(
                vector=embedding,
                tenant_id=tenant_id,
                source_type="threat_intelligence",
                source_id=indicator.id,
                content_hash=indicator.hash,
                embedding_model="threat_intel_model",
                metadata={
                    "ioc_type": indicator.ioc_type,
                    "severity": indicator.severity,
                    "confidence": indicator.confidence,
                    "sources": indicator.sources
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for indicator: {e}")
    
    async def enrich_evidence(self, evidence_hash: str, tenant_id: UUID) -> Dict:
        """Enrich evidence with threat intelligence."""
        enrichment_data = {
            "threat_matches": [],
            "risk_score": 0.0,
            "recommendations": []
        }
        
        # Check against cached indicators
        for indicator in self.indicator_cache.values():
            if evidence_hash.lower() == indicator.value.lower():
                enrichment_data["threat_matches"].append({
                    "indicator_type": indicator.ioc_type,
                    "severity": indicator.severity,
                    "confidence": indicator.confidence,
                    "sources": indicator.sources,
                    "tags": indicator.tags
                })
                
                # Calculate risk score
                severity_scores = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
                risk_score = severity_scores.get(indicator.severity, 0.5) * indicator.confidence
                enrichment_data["risk_score"] = max(enrichment_data["risk_score"], risk_score)
        
        # Vector similarity search for related threats
        try:
            # Generate embedding for evidence
            from ..infrastructure.vector_store import openai_embedding_function
            evidence_embedding = await openai_embedding_function(evidence_hash)
            
            # Search for similar threat indicators
            similar_threats = await self.vector_store.search_similar(
                query_vector=evidence_embedding,
                tenant_id=tenant_id,
                source_type="threat_intelligence",
                limit=5,
                similarity_threshold=0.8
            )
            
            for threat in similar_threats:
                enrichment_data["threat_matches"].append({
                    "similarity": threat["similarity"],
                    "metadata": threat["metadata"]
                })
        
        except Exception as e:
            logger.error(f"Failed to perform vector search: {e}")
        
        # Generate recommendations
        if enrichment_data["risk_score"] > 0.7:
            enrichment_data["recommendations"].append("Immediate isolation recommended")
            enrichment_data["recommendations"].append("Escalate to security team")
        elif enrichment_data["risk_score"] > 0.4:
            enrichment_data["recommendations"].append("Enhanced monitoring recommended")
            enrichment_data["recommendations"].append("Additional analysis required")
        
        return enrichment_data
    
    async def schedule_feed_updates(self, tenant_id: UUID) -> List[str]:
        """Schedule threat feed update jobs."""
        job_ids = []
        
        for feed in self.feeds:
            if not feed.enabled:
                continue
            
            # Schedule feed update job
            job_request = JobScheduleRequest(
                job_type=JobType.CUSTOM,
                payload={
                    "operation": "threat_feed_update",
                    "feed_name": feed.name,
                    "tenant_id": str(tenant_id)
                },
                priority=JobPriority.NORMAL,
                tenant_id=tenant_id,
                user_id="system",
                tags=["threat_intelligence", "feed_update"],
                delay_seconds=0
            )
            
            result = await self.job_service.schedule_job(job_request)
            job_ids.append(result["job_id"])
        
        return job_ids
    
    async def get_threat_statistics(self, tenant_id: UUID) -> Dict:
        """Get threat intelligence statistics for tenant."""
        async with get_async_session() as session:
            # Set tenant context
            await session.execute(
                "SELECT set_config('app.tenant_id', :tenant_id, false)",
                {"tenant_id": str(tenant_id)}
            )
            
            # Get threat intelligence findings
            result = await session.execute(
                select(Finding).where(
                    and_(
                        Finding.tenant_id == tenant_id,
                        Finding.category == "threat_intelligence"
                    )
                )
            )
            
            findings = result.scalars().all()
            
            stats = {
                "total_indicators": len(findings),
                "by_severity": {"low": 0, "medium": 0, "high": 0, "critical": 0},
                "by_type": {},
                "recent_indicators": 0,
                "feeds_active": len([f for f in self.feeds if f.enabled])
            }
            
            # Calculate statistics
            week_ago = datetime.utcnow() - timedelta(days=7)
            
            for finding in findings:
                # Severity breakdown
                stats["by_severity"][finding.severity] += 1
                
                # Type breakdown
                ioc_type = finding.custom_metadata.get("ioc_type", "unknown")
                stats["by_type"][ioc_type] = stats["by_type"].get(ioc_type, 0) + 1
                
                # Recent indicators
                if finding.created_at > week_ago:
                    stats["recent_indicators"] += 1
            
            return stats


# Global threat intelligence engine
_threat_engine: Optional[ThreatIntelligenceEngine] = None


def get_threat_intelligence_engine(job_service: JobService) -> ThreatIntelligenceEngine:
    """Get global threat intelligence engine."""
    global _threat_engine
    if _threat_engine is None:
        _threat_engine = ThreatIntelligenceEngine(job_service)
    return _threat_engine