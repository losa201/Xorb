"""
Production metrics collection and monitoring service
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json

from ..infrastructure.redis_compatibility import get_redis_client
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest

from .base_service import XORBService

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Metric value with metadata"""
    name: str
    value: Union[int, float]
    labels: Dict[str, str]
    timestamp: datetime
    metric_type: str  # "counter", "gauge", "histogram"


class ProductionMetricsService(XORBService):
    """
    Production-grade metrics collection service with Prometheus integration
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/1"):
        super().__init__()
        self.redis_url = redis_url
        self.redis_client = None
        
        # Prometheus registry
        self.registry = CollectorRegistry()
        
        # Core metrics
        self._setup_core_metrics()
        
        # In-memory metric storage for fast access
        self._metric_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        
        # Performance tracking
        self._request_durations: Dict[str, List[float]] = defaultdict(list)
        self._error_counts: Dict[str, int] = defaultdict(int)
        
        logger.info("Production metrics service initialized")
    
    def _setup_core_metrics(self):
        """Setup core Prometheus metrics"""
        
        # API Metrics
        self.api_requests_total = Counter(
            'xorb_api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.api_request_duration = Histogram(
            'xorb_api_request_duration_seconds',
            'API request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.api_errors_total = Counter(
            'xorb_api_errors_total',
            'Total API errors',
            ['endpoint', 'error_type'],
            registry=self.registry
        )
        
        # PTaaS Metrics
        self.ptaas_scans_total = Counter(
            'xorb_ptaas_scans_total',
            'Total PTaaS scans',
            ['scan_type', 'status'],
            registry=self.registry
        )
        
        self.ptaas_scan_duration = Histogram(
            'xorb_ptaas_scan_duration_seconds',
            'PTaaS scan duration',
            ['scan_type'],
            registry=self.registry
        )
        
        self.ptaas_vulnerabilities_found = Counter(
            'xorb_ptaas_vulnerabilities_found_total',
            'Total vulnerabilities found',
            ['severity', 'scan_type'],
            registry=self.registry
        )
        
        # Database Metrics
        self.database_connections_active = Gauge(
            'xorb_database_connections_active',
            'Active database connections',
            registry=self.registry
        )
        
        self.database_query_duration = Histogram(
            'xorb_database_query_duration_seconds',
            'Database query duration',
            ['operation'],
            registry=self.registry
        )
        
        # Security Metrics
        self.auth_attempts_total = Counter(
            'xorb_auth_attempts_total',
            'Total authentication attempts',
            ['status', 'provider'],
            registry=self.registry
        )
        
        self.security_events_total = Counter(
            'xorb_security_events_total',
            'Total security events',
            ['event_type', 'severity'],
            registry=self.registry
        )
        
        # System Metrics
        self.system_cpu_usage = Gauge(
            'xorb_system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'xorb_system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'xorb_system_disk_usage_percent',
            'System disk usage percentage',
            registry=self.registry
        )
        
        # Business Metrics
        self.active_users = Gauge(
            'xorb_active_users',
            'Number of active users',
            ['timeframe'],
            registry=self.registry
        )
        
        self.revenue_metrics = Gauge(
            'xorb_revenue_usd',
            'Revenue metrics in USD',
            ['metric_type'],
            registry=self.registry
        )
    
    async def initialize(self):
        """Initialize metrics service"""
        await super().initialize()
        
        try:
            # Temporarily disabled due to aioredis version compatibility with Python 3.12
            # self.redis_client = await aioredis.from_url(self.redis_url)
            # await self.redis_client.ping()
            # logger.info("Connected to Redis for metrics storage")
            self.redis_client = None
            logger.warning("Redis connection disabled due to aioredis compatibility issues with Python 3.12")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.redis_client = None
        
        # Start background tasks
        asyncio.create_task(self._background_metrics_collector())
        asyncio.create_task(self._metrics_cleanup_task())
        
        logger.info("Metrics service initialized successfully")
    
    # ================================================================
    # CORE METRIC RECORDING
    # ================================================================
    
    async def record_api_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_seconds: float,
        error_type: str = None
    ):
        """Record API request metrics"""
        # Prometheus metrics
        self.api_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        self.api_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration_seconds)
        
        if error_type:
            self.api_errors_total.labels(
                endpoint=endpoint,
                error_type=error_type
            ).inc()
        
        # Cache for fast queries
        metric_key = f"api_request:{endpoint}:{method}"
        self._request_durations[metric_key].append(duration_seconds)
        
        if status_code >= 400:
            self._error_counts[metric_key] += 1
        
        # Store in Redis for persistence
        if self.redis_client:
            await self._store_metric_in_redis(
                "api_request",
                {
                    "method": method,
                    "endpoint": endpoint,
                    "status_code": status_code,
                    "duration": duration_seconds,
                    "error_type": error_type,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    async def record_ptaas_scan(
        self,
        scan_type: str,
        status: str,
        duration_seconds: float,
        vulnerabilities_found: Dict[str, int] = None
    ):
        """Record PTaaS scan metrics"""
        # Prometheus metrics
        self.ptaas_scans_total.labels(
            scan_type=scan_type,
            status=status
        ).inc()
        
        if status == "completed":
            self.ptaas_scan_duration.labels(
                scan_type=scan_type
            ).observe(duration_seconds)
        
        # Record vulnerabilities by severity
        if vulnerabilities_found:
            for severity, count in vulnerabilities_found.items():
                self.ptaas_vulnerabilities_found.labels(
                    severity=severity,
                    scan_type=scan_type
                ).inc(count)
        
        # Store in Redis
        if self.redis_client:
            await self._store_metric_in_redis(
                "ptaas_scan",
                {
                    "scan_type": scan_type,
                    "status": status,
                    "duration": duration_seconds,
                    "vulnerabilities": vulnerabilities_found,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    async def record_security_event(
        self,
        event_type: str,
        severity: str,
        details: Dict[str, Any] = None
    ):
        """Record security event metrics"""
        self.security_events_total.labels(
            event_type=event_type,
            severity=severity
        ).inc()
        
        # Store detailed event in Redis
        if self.redis_client:
            await self._store_metric_in_redis(
                "security_event",
                {
                    "event_type": event_type,
                    "severity": severity,
                    "details": details or {},
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    async def record_authentication_attempt(
        self,
        status: str,
        provider: str = "local",
        username: str = None,
        ip_address: str = None
    ):
        """Record authentication attempt"""
        self.auth_attempts_total.labels(
            status=status,
            provider=provider
        ).inc()
        
        # Store in Redis for security analysis
        if self.redis_client:
            await self._store_metric_in_redis(
                "auth_attempt",
                {
                    "status": status,
                    "provider": provider,
                    "username": username,
                    "ip_address": ip_address,
                    "timestamp": datetime.utcnow().isoformat()
                },
                ttl=86400 * 30  # Keep auth logs for 30 days
            )
    
    async def update_system_metrics(
        self,
        cpu_percent: float,
        memory_percent: float,
        disk_percent: float
    ):
        """Update system resource metrics"""
        self.system_cpu_usage.set(cpu_percent)
        self.system_memory_usage.set(memory_percent)
        self.system_disk_usage.set(disk_percent)
    
    async def update_business_metrics(
        self,
        active_users_1h: int,
        active_users_24h: int,
        revenue_mrr: float,
        revenue_arr: float
    ):
        """Update business KPI metrics"""
        self.active_users.labels(timeframe="1h").set(active_users_1h)
        self.active_users.labels(timeframe="24h").set(active_users_24h)
        
        self.revenue_metrics.labels(metric_type="mrr").set(revenue_mrr)
        self.revenue_metrics.labels(metric_type="arr").set(revenue_arr)
    
    # ================================================================
    # METRIC RETRIEVAL
    # ================================================================
    
    async def get_api_metrics(self, endpoint: str = None, hours: int = 24) -> Dict[str, Any]:
        """Get API performance metrics"""
        if not self.redis_client:
            return self._get_cached_api_metrics(endpoint)
        
        # Get metrics from Redis
        start_time = datetime.utcnow() - timedelta(hours=hours)
        metrics = await self._get_metrics_from_redis("api_request", start_time)
        
        # Filter by endpoint if specified
        if endpoint:
            metrics = [m for m in metrics if m.get("endpoint") == endpoint]
        
        if not metrics:
            return {"total_requests": 0, "average_response_time": 0, "error_rate": 0}
        
        # Calculate aggregated metrics
        total_requests = len(metrics)
        error_requests = len([m for m in metrics if m.get("status_code", 200) >= 400])
        durations = [m.get("duration", 0) for m in metrics]
        
        return {
            "total_requests": total_requests,
            "error_requests": error_requests,
            "error_rate": (error_requests / total_requests) * 100 if total_requests > 0 else 0,
            "average_response_time": sum(durations) / len(durations) if durations else 0,
            "min_response_time": min(durations) if durations else 0,
            "max_response_time": max(durations) if durations else 0,
            "p95_response_time": self._calculate_percentile(durations, 95) if durations else 0,
            "requests_per_minute": total_requests / (hours * 60) if hours > 0 else 0
        }
    
    async def get_ptaas_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get PTaaS performance metrics"""
        if not self.redis_client:
            return {"total_scans": 0, "completed_scans": 0, "average_duration": 0}
        
        start_time = datetime.utcnow() - timedelta(hours=hours)
        metrics = await self._get_metrics_from_redis("ptaas_scan", start_time)
        
        if not metrics:
            return {"total_scans": 0, "completed_scans": 0, "average_duration": 0}
        
        total_scans = len(metrics)
        completed_scans = len([m for m in metrics if m.get("status") == "completed"])
        durations = [m.get("duration", 0) for m in metrics if m.get("status") == "completed"]
        
        # Aggregate vulnerabilities
        vulnerabilities = defaultdict(int)
        for metric in metrics:
            if metric.get("vulnerabilities"):
                for severity, count in metric["vulnerabilities"].items():
                    vulnerabilities[severity] += count
        
        return {
            "total_scans": total_scans,
            "completed_scans": completed_scans,
            "failed_scans": total_scans - completed_scans,
            "success_rate": (completed_scans / total_scans) * 100 if total_scans > 0 else 0,
            "average_duration": sum(durations) / len(durations) if durations else 0,
            "vulnerabilities_by_severity": dict(vulnerabilities),
            "scans_per_hour": total_scans / hours if hours > 0 else 0
        }
    
    async def get_security_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get security event metrics"""
        if not self.redis_client:
            return {"total_events": 0, "events_by_type": {}, "events_by_severity": {}}
        
        start_time = datetime.utcnow() - timedelta(hours=hours)
        security_events = await self._get_metrics_from_redis("security_event", start_time)
        auth_events = await self._get_metrics_from_redis("auth_attempt", start_time)
        
        # Aggregate security events
        events_by_type = defaultdict(int)
        events_by_severity = defaultdict(int)
        
        for event in security_events:
            events_by_type[event.get("event_type", "unknown")] += 1
            events_by_severity[event.get("severity", "unknown")] += 1
        
        # Authentication metrics
        auth_by_status = defaultdict(int)
        failed_ips = defaultdict(int)
        
        for auth in auth_events:
            auth_by_status[auth.get("status", "unknown")] += 1
            if auth.get("status") == "failed" and auth.get("ip_address"):
                failed_ips[auth["ip_address"]] += 1
        
        return {
            "total_security_events": len(security_events),
            "events_by_type": dict(events_by_type),
            "events_by_severity": dict(events_by_severity),
            "authentication": {
                "total_attempts": len(auth_events),
                "by_status": dict(auth_by_status),
                "failed_login_rate": (auth_by_status["failed"] / len(auth_events)) * 100 if auth_events else 0,
                "top_failed_ips": dict(sorted(failed_ips.items(), key=lambda x: x[1], reverse=True)[:10])
            }
        }
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus formatted metrics"""
        return generate_latest(self.registry).decode('utf-8')
    
    # ================================================================
    # HELPER METHODS
    # ================================================================
    
    def _get_cached_api_metrics(self, endpoint: str = None) -> Dict[str, Any]:
        """Get API metrics from in-memory cache"""
        if endpoint:
            key = f"api_request:{endpoint}"
            durations = []
            errors = 0
            
            for method_key in self._request_durations:
                if endpoint in method_key:
                    durations.extend(self._request_durations[method_key])
                    errors += self._error_counts[method_key]
        else:
            durations = []
            errors = 0
            for key in self._request_durations:
                durations.extend(self._request_durations[key])
                errors += self._error_counts[key]
        
        total_requests = len(durations)
        
        return {
            "total_requests": total_requests,
            "error_requests": errors,
            "error_rate": (errors / total_requests) * 100 if total_requests > 0 else 0,
            "average_response_time": sum(durations) / len(durations) if durations else 0,
            "p95_response_time": self._calculate_percentile(durations, 95) if durations else 0
        }
    
    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value"""
        if not values:
            return 0
        
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    async def _store_metric_in_redis(
        self,
        metric_type: str,
        data: Dict[str, Any],
        ttl: int = 86400 * 7  # 7 days default
    ):
        """Store metric in Redis with TTL"""
        if not self.redis_client:
            return
        
        try:
            key = f"metrics:{metric_type}:{int(time.time() * 1000)}"
            await self.redis_client.setex(key, ttl, json.dumps(data))
        except Exception as e:
            logger.error(f"Failed to store metric in Redis: {e}")
    
    async def _get_metrics_from_redis(
        self,
        metric_type: str,
        start_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get metrics from Redis"""
        if not self.redis_client:
            return []
        
        try:
            pattern = f"metrics:{metric_type}:*"
            keys = await self.redis_client.keys(pattern)
            
            metrics = []
            for key in keys:
                try:
                    # Extract timestamp from key
                    timestamp_ms = int(key.decode().split(":")[-1])
                    timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
                    
                    if timestamp >= start_time:
                        data = await self.redis_client.get(key)
                        if data:
                            metric = json.loads(data)
                            metric["_timestamp"] = timestamp
                            metrics.append(metric)
                except (ValueError, json.JSONDecodeError) as e:
                    logger.warning(f"Failed to parse metric key {key}: {e}")
                    continue
            
            return sorted(metrics, key=lambda x: x.get("_timestamp", datetime.min))
            
        except Exception as e:
            logger.error(f"Failed to get metrics from Redis: {e}")
            return []
    
    async def _background_metrics_collector(self):
        """Background task for collecting system metrics"""
        while True:
            try:
                await asyncio.sleep(30)  # Collect every 30 seconds
                
                # Collect system metrics
                import psutil
                
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                await self.update_system_metrics(
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    disk_percent=(disk.used / disk.total) * 100
                )
                
                # Update database connection count (if available)
                try:
                    from ..infrastructure.database import get_database_manager
                    db_manager = get_database_manager()
                    if hasattr(db_manager, 'engine') and db_manager.engine:
                        pool = db_manager.engine.pool
                        self.database_connections_active.set(pool.checkedout())
                except Exception:
                    pass
                
            except Exception as e:
                logger.error(f"Background metrics collection error: {e}")
                await asyncio.sleep(5)
    
    async def _metrics_cleanup_task(self):
        """Background task for cleaning up old metrics"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up in-memory caches
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
                
                for metric_key in list(self._metric_cache.keys()):
                    # Keep only recent metrics in memory
                    cache = self._metric_cache[metric_key]
                    while cache and len(cache) > 100:  # Keep max 100 recent items
                        cache.popleft()
                
                # Clean up request duration tracking
                for endpoint_key in list(self._request_durations.keys()):
                    durations = self._request_durations[endpoint_key]
                    if len(durations) > 1000:  # Keep max 1000 recent requests
                        self._request_durations[endpoint_key] = durations[-500:]
                
                logger.debug("Metrics cleanup completed")
                
            except Exception as e:
                logger.error(f"Metrics cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def export_metrics_to_file(self, filepath: str, format: str = "json") -> bool:
        """Export metrics to file"""
        try:
            if format == "json":
                metrics_data = {
                    "export_timestamp": datetime.utcnow().isoformat(),
                    "api_metrics": await self.get_api_metrics(),
                    "ptaas_metrics": await self.get_ptaas_metrics(),
                    "security_metrics": await self.get_security_metrics(),
                }
                
                with open(filepath, 'w') as f:
                    json.dump(metrics_data, f, indent=2)
                    
            elif format == "prometheus":
                with open(filepath, 'w') as f:
                    f.write(self.get_prometheus_metrics())
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Metrics exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False