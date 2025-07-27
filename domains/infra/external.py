"""
External Service Adapters - Third-party API Integrations

Implementations of application ports for external services like
NVIDIA API, Redis, NATS, Slack, etc.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
import nats
import redis.asyncio as redis
import structlog
from cachetools import TTLCache
from openai import AsyncOpenAI

from ..application.ports import (
    CacheService,
    EmbeddingService,
    EventPublisher,
    NotificationService,
    SecurityScanner
)
from ..domain import (
    CampaignId,
    DomainEvent,
    Embedding,
    Target
)

__all__ = [
    "NvidiaEmbeddingService",
    "RedisCache",
    "NatsEventPublisher",
    "SlackNotificationService",
    "NucleiSecurityScanner",
    "CompositeSecurityScanner"
]

log = structlog.get_logger(__name__)


class NvidiaEmbeddingService(EmbeddingService):
    """NVIDIA API implementation of EmbeddingService"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        cache_ttl_seconds: int = 86400,  # 24 hours
        max_cache_size: int = 10000
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._local_cache = TTLCache(maxsize=max_cache_size, ttl=cache_ttl_seconds)
        
        log.info("NVIDIA Embedding Service initialized", base_url=base_url)
    
    async def generate_embedding(
        self,
        text: str,
        model: str = "nvidia/embed-qa-4",
        input_type: str = "query"
    ) -> Embedding:
        """Generate embedding for text with local caching"""
        
        # Check local cache first
        cache_key = self._make_cache_key(text, model, input_type)
        if cache_key in self._local_cache:
            log.debug("Local cache hit for embedding", cache_key=cache_key)
            return self._local_cache[cache_key]
        
        try:
            log.debug("Generating embedding via NVIDIA API", 
                     model=model, input_type=input_type, text_length=len(text))
            
            response = await self._client.embeddings.create(
                input=[text],
                model=model,
                encoding_format="float",
                extra_body={
                    "input_type": input_type,
                    "truncate": "NONE"
                }
            )
            
            embedding = Embedding.from_list(
                vector=response.data[0].embedding,
                model=model
            )
            
            # Cache the result
            self._local_cache[cache_key] = embedding
            
            log.debug("Embedding generated successfully", 
                     dimension=embedding.dimension, model=model)
            
            return embedding
            
        except Exception as e:
            log.error("Failed to generate embedding", error=str(e), model=model)
            raise
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model: str = "nvidia/embed-qa-4",
        input_type: str = "query"
    ) -> List[Embedding]:
        """Generate embeddings for multiple texts with batching"""
        
        if not texts:
            return []
        
        # Check cache for each text
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._make_cache_key(text, model, input_type)
            if cache_key in self._local_cache:
                embeddings.append(self._local_cache[cache_key])
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            log.debug("Generating batch embeddings", 
                     batch_size=len(uncached_texts), model=model)
            
            try:
                response = await self._client.embeddings.create(
                    input=uncached_texts,
                    model=model,
                    encoding_format="float",
                    extra_body={
                        "input_type": input_type,
                        "truncate": "NONE"
                    }
                )
                
                # Process results
                for j, embedding_data in enumerate(response.data):
                    original_index = uncached_indices[j]
                    text = uncached_texts[j]
                    
                    embedding = Embedding.from_list(
                        vector=embedding_data.embedding,
                        model=model
                    )
                    
                    # Update results
                    embeddings[original_index] = embedding
                    
                    # Cache the result
                    cache_key = self._make_cache_key(text, model, input_type)
                    self._local_cache[cache_key] = embedding
                
                log.debug("Batch embeddings generated", count=len(uncached_texts))
                
            except Exception as e:
                log.error("Failed to generate batch embeddings", error=str(e))
                raise
        
        return embeddings
    
    async def compute_similarity(
        self,
        embedding1: Embedding,
        embedding2: Embedding,
        metric: str = "cosine"
    ) -> float:
        """Compute similarity between embeddings"""
        
        if embedding1.dimension != embedding2.dimension:
            raise ValueError("Embeddings must have the same dimension")
        
        if metric == "cosine":
            return embedding1.similarity_cosine(embedding2)
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
    
    def _make_cache_key(self, text: str, model: str, input_type: str) -> str:
        """Generate cache key using SHA-1 hash"""
        content = f"{model}:{input_type}:{text}"
        return hashlib.sha1(content.encode('utf-8')).hexdigest()


class RedisCache(CacheService):
    """Redis implementation of CacheService"""
    
    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._redis: Optional[redis.Redis] = None
        
        log.info("Redis cache service initialized", url=redis_url)
    
    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection"""
        if self._redis is None:
            self._redis = redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=False
            )
        return self._redis
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        
        try:
            r = await self._get_redis()
            value = await r.get(key)
            
            if value:
                return json.loads(value.decode('utf-8'))
            
            return None
            
        except Exception as e:
            log.warning("Cache get failed", key=key, error=str(e))
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl_seconds: Optional[int] = None
    ) -> None:
        """Set value in cache"""
        
        try:
            r = await self._get_redis()
            serialized_value = json.dumps(value, default=str)
            
            if ttl_seconds:
                await r.setex(key, ttl_seconds, serialized_value)
            else:
                await r.set(key, serialized_value)
            
            log.debug("Cache set successful", key=key, ttl=ttl_seconds)
            
        except Exception as e:
            log.warning("Cache set failed", key=key, error=str(e))
    
    async def delete(self, key: str) -> None:
        """Delete value from cache"""
        
        try:
            r = await self._get_redis()
            await r.delete(key)
            
            log.debug("Cache delete successful", key=key)
            
        except Exception as e:
            log.warning("Cache delete failed", key=key, error=str(e))
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        
        try:
            r = await self._get_redis()
            result = await r.exists(key)
            return bool(result)
            
        except Exception as e:
            log.warning("Cache exists check failed", key=key, error=str(e))
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern"""
        
        try:
            r = await self._get_redis()
            keys = await r.keys(pattern)
            
            if keys:
                deleted_count = await r.delete(*keys)
                log.info("Cache pattern cleared", pattern=pattern, count=deleted_count)
                return deleted_count
            
            return 0
            
        except Exception as e:
            log.warning("Cache pattern clear failed", pattern=pattern, error=str(e))
            return 0


class NatsEventPublisher(EventPublisher):
    """NATS JetStream implementation of EventPublisher"""
    
    def __init__(self, nats_url: str, stream_name: str = "XORB_EVENTS") -> None:
        self._nats_url = nats_url
        self._stream_name = stream_name
        self._nc: Optional[nats.NATS] = None
        self._js: Optional[nats.js.JetStreamContext] = None
        
        log.info("NATS event publisher initialized", url=nats_url, stream=stream_name)
    
    async def _get_jetstream(self) -> nats.js.JetStreamContext:
        """Get or create NATS JetStream connection"""
        
        if self._nc is None:
            self._nc = await nats.connect(self._nats_url)
            self._js = self._nc.jetstream()
            
            # Ensure stream exists
            try:
                await self._js.stream_info(self._stream_name)
            except:
                # Create stream if it doesn't exist
                from nats.js.api import StreamConfig
                
                stream_config = StreamConfig(
                    name=self._stream_name,
                    subjects=[f"{self._stream_name.lower()}.*"],
                    max_age=7 * 24 * 3600,  # 7 days
                    max_bytes=10 * 1024 * 1024 * 1024,  # 10GB
                    storage="file"
                )
                
                await self._js.add_stream(stream_config)
                log.info("Created NATS stream", stream=self._stream_name)
        
        return self._js
    
    async def publish(self, event: DomainEvent) -> None:
        """Publish a domain event"""
        
        try:
            js = await self._get_jetstream()
            
            # Create CloudEvents-compliant message
            message = {
                "specversion": "1.0",
                "type": event.event_type,
                "source": "xorb-system",
                "id": str(event.event_id),
                "time": event.occurred_at.isoformat(),
                "datacontenttype": "application/json",
                "data": self._serialize_event_data(event)
            }
            
            subject = f"{self._stream_name.lower()}.{event.event_type.lower()}"
            
            await js.publish(
                subject=subject,
                payload=json.dumps(message).encode('utf-8')
            )
            
            log.debug("Event published", 
                     event_type=event.event_type, 
                     event_id=str(event.event_id),
                     subject=subject)
            
        except Exception as e:
            log.error("Failed to publish event", 
                     event_type=event.event_type,
                     error=str(e))
            raise
    
    async def publish_batch(self, events: List[DomainEvent]) -> None:
        """Publish multiple domain events"""
        
        for event in events:
            await self.publish(event)
    
    def _serialize_event_data(self, event: DomainEvent) -> Dict[str, Any]:
        """Serialize event data for JSON transport"""
        
        # Convert event attributes to dict, handling special types
        data = {}
        
        for key, value in event.__dict__.items():
            if key.startswith('_'):
                continue
            
            if hasattr(value, 'value'):  # Handle enums
                data[key] = value.value
            elif hasattr(value, '__dict__'):  # Handle objects
                data[key] = str(value)
            else:
                data[key] = value
        
        return data


class SlackNotificationService(NotificationService):
    """Slack implementation of NotificationService"""
    
    def __init__(self, webhook_url: str, default_channel: str = "#security") -> None:
        self._webhook_url = webhook_url
        self._default_channel = default_channel
        
        log.info("Slack notification service initialized", channel=default_channel)
    
    async def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "info",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send an alert notification to Slack"""
        
        # Map severity to Slack colors
        color_map = {
            "critical": "#FF0000",
            "high": "#FF6600", 
            "warning": "#FFCC00",
            "info": "#0099CC",
            "success": "#00CC00"
        }
        
        color = color_map.get(severity, "#999999")
        
        # Create Slack message
        slack_message = {
            "channel": self._default_channel,
            "username": "Xorb Security Bot",
            "icon_emoji": ":shield:",
            "attachments": [
                {
                    "color": color,
                    "title": f"{severity.upper()}: {title}",
                    "text": message,
                    "timestamp": int(time.time()),
                    "fields": [
                        {
                            "title": field,
                            "value": str(value),
                            "short": True
                        }
                        for field, value in (metadata or {}).items()
                    ]
                }
            ]
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._webhook_url,
                    json=slack_message,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    log.debug("Slack notification sent", title=title, severity=severity)
                else:
                    log.warning("Slack notification failed", 
                               status_code=response.status_code,
                               response=response.text)
                    
        except Exception as e:
            log.error("Failed to send Slack notification", error=str(e), title=title)
    
    async def send_campaign_update(
        self,
        campaign_id: CampaignId,
        status: str,
        details: str
    ) -> None:
        """Send campaign status update"""
        
        await self.send_alert(
            title=f"Campaign {status.title()}",
            message=f"Campaign {campaign_id}: {details}",
            severity="info",
            metadata={"campaign_id": str(campaign_id), "status": status}
        )


class NucleiSecurityScanner(SecurityScanner):
    """Nuclei-based security scanner implementation"""
    
    def __init__(self, nuclei_binary_path: str = "nuclei") -> None:
        self._nuclei_path = nuclei_binary_path
        
        log.info("Nuclei security scanner initialized", binary=nuclei_binary_path)
    
    async def scan_target(
        self,
        target: Target,
        scan_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform security scan using Nuclei"""
        
        # Prepare Nuclei command
        cmd_args = [
            self._nuclei_path,
            "-json",  # JSON output
            "-silent",  # Reduce noise
        ]
        
        # Add target domains
        for domain in target.scope.domains:
            cmd_args.extend(["-target", domain])
        
        # Add scan configuration
        if scan_config.get("templates"):
            cmd_args.extend(["-t", scan_config["templates"]])
        
        if scan_config.get("severity"):
            cmd_args.extend(["-severity", scan_config["severity"]])
        
        try:
            log.info("Starting Nuclei scan", target=target.name, config=scan_config)
            
            # Execute Nuclei
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                log.warning("Nuclei scan failed", 
                           return_code=process.returncode,
                           error=stderr.decode())
                return {"findings": [], "errors": [stderr.decode()]}
            
            # Parse JSON output
            findings = []
            for line in stdout.decode().strip().split('\n'):
                if line:
                    try:
                        finding_data = json.loads(line)
                        findings.append(self._parse_nuclei_finding(finding_data))
                    except json.JSONDecodeError:
                        continue
            
            log.info("Nuclei scan completed", 
                    target=target.name, 
                    findings_count=len(findings))
            
            return {
                "findings": findings,
                "scan_config": scan_config,
                "target": target.name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            log.error("Nuclei scan error", error=str(e), target=target.name)
            return {"findings": [], "errors": [str(e)]}
    
    async def validate_scope(self, target: Target, url: str) -> bool:
        """Validate if URL is within target scope"""
        
        from urllib.parse import urlparse
        
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        return target.scope.is_domain_in_scope(domain)
    
    async def get_scan_capabilities(self) -> List[str]:
        """Get available scanning capabilities"""
        
        return [
            "web_vulnerability_scanning",
            "subdomain_enumeration", 
            "port_scanning",
            "dns_enumeration",
            "ssl_analysis"
        ]
    
    def _parse_nuclei_finding(self, nuclei_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Nuclei finding data into standard format"""
        
        # Map Nuclei severity to our severity levels
        severity_map = {
            "critical": "critical",
            "high": "high",
            "medium": "medium",
            "low": "low",
            "info": "info",
            "unknown": "info"
        }
        
        return {
            "title": nuclei_data.get("info", {}).get("name", "Unknown Finding"),
            "description": nuclei_data.get("info", {}).get("description", ""),
            "severity": severity_map.get(
                nuclei_data.get("info", {}).get("severity", "info"), "info"
            ),
            "evidence": {
                "url": nuclei_data.get("matched-at", ""),
                "template": nuclei_data.get("template-id", ""),
                "matcher_name": nuclei_data.get("matcher-name", ""),
                "extracted_results": nuclei_data.get("extracted-results", []),
                "curl_command": nuclei_data.get("curl-command", ""),
                "raw_data": nuclei_data
            }
        }


class CompositeSecurityScanner(SecurityScanner):
    """Composite scanner that orchestrates multiple scanning tools"""
    
    def __init__(self, scanners: List[SecurityScanner]) -> None:
        self._scanners = scanners
        
        log.info("Composite security scanner initialized", 
                scanner_count=len(scanners))
    
    async def scan_target(
        self,
        target: Target,
        scan_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform security scan using all available scanners"""
        
        all_findings = []
        all_errors = []
        
        # Run all scanners concurrently
        scanner_tasks = [
            scanner.scan_target(target, scan_config)
            for scanner in self._scanners
        ]
        
        results = await asyncio.gather(*scanner_tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                all_errors.append(f"Scanner {i} failed: {str(result)}")
            else:
                all_findings.extend(result.get("findings", []))
                all_errors.extend(result.get("errors", []))
        
        return {
            "findings": all_findings,
            "errors": all_errors,
            "scanner_count": len(self._scanners),
            "target": target.name,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def validate_scope(self, target: Target, url: str) -> bool:
        """Validate if URL is within target scope using first scanner"""
        
        if self._scanners:
            return await self._scanners[0].validate_scope(target, url)
        return False
    
    async def get_scan_capabilities(self) -> List[str]:
        """Get combined scanning capabilities from all scanners"""
        
        all_capabilities = set()
        
        for scanner in self._scanners:
            capabilities = await scanner.get_scan_capabilities()
            all_capabilities.update(capabilities)
        
        return list(all_capabilities)