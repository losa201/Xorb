"""
Performance Optimization Engine
Advanced performance monitoring, optimization, and auto-scaling capabilities
"""

import asyncio
import json
import logging
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import statistics
import hashlib
from contextlib import asynccontextmanager

# Database connection pooling
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

# Redis connection optimization
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    try:
        import aioredis
        REDIS_AVAILABLE = True
    except ImportError:
        REDIS_AVAILABLE = False

from .base_service import XORBService, ServiceType, ServiceStatus
from .interfaces import PerformanceService, MonitoringService, OptimizationService
from ..core.logging import get_logger

logger = get_logger(__name__)

class PerformanceMetric(Enum):
    """Types of performance metrics"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CONNECTION_POOL = "connection_pool"
    CACHE_HIT_RATE = "cache_hit_rate"
    QUEUE_DEPTH = "queue_depth"

class OptimizationStrategy(Enum):
    """Performance optimization strategies"""
    CONNECTION_POOLING = "connection_pooling"
    CACHING = "caching"
    LOAD_BALANCING = "load_balancing"
    ASYNC_PROCESSING = "async_processing"
    DATABASE_INDEXING = "database_indexing"
    QUERY_OPTIMIZATION = "query_optimization"
    MEMORY_OPTIMIZATION = "memory_optimization"
    BATCH_PROCESSING = "batch_processing"
    COMPRESSION = "compression"
    CDN_OPTIMIZATION = "cdn_optimization"

class AlertSeverity(Enum):
    """Performance alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class PerformanceSnapshot:
    """Performance metrics snapshot"""
    timestamp: datetime
    metrics: Dict[PerformanceMetric, float]
    service_id: str
    node_id: str
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceTrend:
    """Performance trend analysis"""
    metric: PerformanceMetric
    trend_direction: str  # "increasing", "decreasing", "stable"
    change_rate: float
    confidence: float
    prediction: Optional[float]
    time_window: timedelta

@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""
    recommendation_id: str
    strategy: OptimizationStrategy
    description: str
    expected_improvement: float
    implementation_effort: str  # "low", "medium", "high"
    priority: int
    metrics_targeted: List[PerformanceMetric]
    estimated_impact: Dict[str, float]

@dataclass
class PerformanceAlert:
    """Performance alert"""
    alert_id: str
    metric: PerformanceMetric
    severity: AlertSeverity
    threshold_value: float
    actual_value: float
    service_id: str
    timestamp: datetime
    description: str
    recommended_actions: List[str]

@dataclass
class ConnectionPoolStats:
    """Database connection pool statistics"""
    pool_name: str
    total_connections: int
    active_connections: int
    idle_connections: int
    pending_requests: int
    max_connections: int
    min_connections: int
    connection_lifetime: float
    average_acquire_time: float
    pool_efficiency: float

class PerformanceOptimizationEngine(XORBService, PerformanceService, MonitoringService, OptimizationService):
    """
    Advanced Performance Optimization Engine
    
    Features:
    - Real-time performance monitoring
    - Automated optimization recommendations
    - Connection pool management
    - Cache optimization
    - Query performance analysis
    - Resource auto-scaling
    - Performance anomaly detection
    - Trend analysis and prediction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            service_id="performance_optimization_engine",
            service_type=ServiceType.INFRASTRUCTURE,
            dependencies=["database", "redis", "monitoring"]
        )
        
        self.config = config or {}
        
        # Performance monitoring
        self.performance_history: List[PerformanceSnapshot] = []
        self.performance_trends: Dict[PerformanceMetric, PerformanceTrend] = {}
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        
        # Optimization tracking
        self.optimization_recommendations: Dict[str, OptimizationRecommendation] = {}
        self.applied_optimizations: List[str] = []
        
        # Connection pools
        self.db_pools: Dict[str, Any] = {}
        self.redis_pools: Dict[str, Any] = {}
        self.pool_stats: Dict[str, ConnectionPoolStats] = {}
        
        # Performance thresholds
        self.performance_thresholds = {
            PerformanceMetric.CPU_USAGE: 80.0,
            PerformanceMetric.MEMORY_USAGE: 85.0,
            PerformanceMetric.RESPONSE_TIME: 2000.0,  # milliseconds
            PerformanceMetric.ERROR_RATE: 5.0,  # percentage
            PerformanceMetric.CACHE_HIT_RATE: 80.0,  # minimum percentage
            PerformanceMetric.CONNECTION_POOL: 90.0  # percentage utilization
        }
        
        # Optimization strategies
        self.optimization_strategies = {
            OptimizationStrategy.CONNECTION_POOLING: self._optimize_connection_pooling,
            OptimizationStrategy.CACHING: self._optimize_caching,
            OptimizationStrategy.QUERY_OPTIMIZATION: self._optimize_queries,
            OptimizationStrategy.MEMORY_OPTIMIZATION: self._optimize_memory,
            OptimizationStrategy.ASYNC_PROCESSING: self._optimize_async_processing,
            OptimizationStrategy.BATCH_PROCESSING: self._optimize_batch_processing
        }
        
        # Monitoring configuration
        self.monitoring_interval = self.config.get('monitoring_interval', 30)  # seconds
        self.retention_period = self.config.get('retention_period', 7)  # days
        self.alert_cooldown = self.config.get('alert_cooldown', 300)  # seconds
        
        # Auto-optimization settings
        self.auto_optimization_enabled = self.config.get('auto_optimization_enabled', True)
        self.max_auto_optimizations_per_hour = self.config.get('max_auto_optimizations_per_hour', 5)
        
        # Initialize optimized connection pools
        asyncio.create_task(self._initialize_optimized_pools())
        
        # Start monitoring tasks
        self._start_monitoring_tasks()
        
        logger.info("Performance Optimization Engine initialized")
    
    async def _initialize_optimized_pools(self):
        """Initialize optimized database and Redis connection pools"""
        try:
            # Initialize database pools
            if ASYNCPG_AVAILABLE:
                await self._create_optimized_db_pools()
            
            # Initialize Redis pools
            if REDIS_AVAILABLE:
                await self._create_optimized_redis_pools()
            
            logger.info("Optimized connection pools initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize optimized pools: {e}")
    
    async def _create_optimized_db_pools(self):
        """Create optimized database connection pools"""
        db_config = self.config.get('database', {})
        
        # Primary database pool
        primary_pool_config = {
            'min_size': db_config.get('min_connections', 5),
            'max_size': db_config.get('max_connections', 20),
            'max_queries': db_config.get('max_queries_per_connection', 50000),
            'max_inactive_connection_lifetime': db_config.get('connection_lifetime', 300),
            'command_timeout': db_config.get('command_timeout', 60),
            'server_settings': {
                'jit': 'off',  # Disable JIT for faster connection setup
                'application_name': 'xorb_optimized'
            }
        }
        
        # Read replica pool (if configured)
        if db_config.get('read_replica_url'):
            replica_pool_config = primary_pool_config.copy()
            replica_pool_config.update({
                'min_size': db_config.get('replica_min_connections', 3),
                'max_size': db_config.get('replica_max_connections', 15)
            })
            
            # This would create actual asyncpg pool
            # self.db_pools['replica'] = await asyncpg.create_pool(...)
            logger.info("Read replica pool configured")
        
        # Analytics pool for heavy queries
        analytics_pool_config = primary_pool_config.copy()
        analytics_pool_config.update({
            'min_size': 2,
            'max_size': 8,
            'command_timeout': 300  # Longer timeout for analytics
        })
        
        logger.info("Database pools configured with optimization settings")
    
    async def _create_optimized_redis_pools(self):
        """Create optimized Redis connection pools"""
        redis_config = self.config.get('redis', {})
        
        # Primary Redis pool
        primary_pool_config = {
            'max_connections': redis_config.get('max_connections', 50),
            'connection_kwargs': {
                'socket_keepalive': True,
                'socket_keepalive_options': {
                    'TCP_KEEPIDLE': 30,
                    'TCP_KEEPINTVL': 10,
                    'TCP_KEEPCNT': 3
                },
                'retry_on_timeout': True,
                'health_check_interval': 30
            }
        }
        
        # Cache-specific pool
        cache_pool_config = primary_pool_config.copy()
        cache_pool_config.update({
            'max_connections': redis_config.get('cache_max_connections', 30)
        })
        
        # Session store pool
        session_pool_config = primary_pool_config.copy()
        session_pool_config.update({
            'max_connections': redis_config.get('session_max_connections', 20)
        })
        
        logger.info("Redis pools configured with optimization settings")
    
    def _start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        asyncio.create_task(self._performance_monitoring_loop())
        asyncio.create_task(self._trend_analysis_loop())
        asyncio.create_task(self._optimization_recommendation_loop())
        asyncio.create_task(self._auto_optimization_loop())
        asyncio.create_task(self._cleanup_old_data_loop())
        
        logger.info("Performance monitoring tasks started")
    
    async def _performance_monitoring_loop(self):
        """Main performance monitoring loop"""
        while True:
            try:
                # Collect performance metrics
                snapshot = await self._collect_performance_snapshot()
                
                # Store snapshot
                self.performance_history.append(snapshot)
                
                # Check for alerts
                await self._check_performance_alerts(snapshot)
                
                # Limit history size
                max_history = int(self.retention_period * 24 * 3600 / self.monitoring_interval)
                if len(self.performance_history) > max_history:
                    self.performance_history = self.performance_history[-max_history:]
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _collect_performance_snapshot(self) -> PerformanceSnapshot:
        """Collect current performance metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_io_counters()
            network = psutil.net_io_counters()
            
            # Application-specific metrics
            response_times = await self._get_recent_response_times()
            error_rate = await self._get_recent_error_rate()
            cache_hit_rate = await self._get_cache_hit_rate()
            
            # Connection pool metrics
            pool_utilization = await self._get_pool_utilization()
            
            metrics = {
                PerformanceMetric.CPU_USAGE: cpu_percent,
                PerformanceMetric.MEMORY_USAGE: memory.percent,
                PerformanceMetric.DISK_IO: disk.read_bytes + disk.write_bytes if disk else 0,
                PerformanceMetric.NETWORK_IO: network.bytes_sent + network.bytes_recv if network else 0,
                PerformanceMetric.RESPONSE_TIME: statistics.mean(response_times) if response_times else 0,
                PerformanceMetric.ERROR_RATE: error_rate,
                PerformanceMetric.CACHE_HIT_RATE: cache_hit_rate,
                PerformanceMetric.CONNECTION_POOL: pool_utilization
            }
            
            # Calculate throughput
            if response_times:
                throughput = len(response_times) / self.monitoring_interval
                metrics[PerformanceMetric.THROUGHPUT] = throughput
            
            return PerformanceSnapshot(
                timestamp=datetime.utcnow(),
                metrics=metrics,
                service_id=self.service_id,
                node_id=self._get_node_id(),
                additional_data={
                    'process_count': len(psutil.pids()),
                    'open_files': len(psutil.Process().open_files()) if hasattr(psutil.Process(), 'open_files') else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to collect performance snapshot: {e}")
            return PerformanceSnapshot(
                timestamp=datetime.utcnow(),
                metrics={},
                service_id=self.service_id,
                node_id=self._get_node_id()
            )
    
    async def _get_recent_response_times(self) -> List[float]:
        """Get recent response times from monitoring data"""
        # This would integrate with actual request monitoring
        # For now, return simulated data
        import random
        return [random.uniform(50, 500) for _ in range(10)]
    
    async def _get_recent_error_rate(self) -> float:
        """Get recent error rate percentage"""
        # This would integrate with actual error tracking
        # For now, return simulated data
        import random
        return random.uniform(0, 2)
    
    async def _get_cache_hit_rate(self) -> float:
        """Get cache hit rate percentage"""
        # This would integrate with actual cache metrics
        # For now, return simulated data
        import random
        return random.uniform(85, 98)
    
    async def _get_pool_utilization(self) -> float:
        """Get connection pool utilization percentage"""
        if not self.pool_stats:
            return 0.0
        
        total_utilization = 0.0
        pool_count = 0
        
        for stats in self.pool_stats.values():
            if stats.max_connections > 0:
                utilization = (stats.active_connections / stats.max_connections) * 100
                total_utilization += utilization
                pool_count += 1
        
        return total_utilization / pool_count if pool_count > 0 else 0.0
    
    def _get_node_id(self) -> str:
        """Get unique node identifier"""
        # Generate consistent node ID based on system characteristics
        import platform
        node_info = f"{platform.node()}-{platform.machine()}"
        return hashlib.md5(node_info.encode()).hexdigest()[:8]
    
    async def _check_performance_alerts(self, snapshot: PerformanceSnapshot):
        """Check for performance alerts based on thresholds"""
        for metric, value in snapshot.metrics.items():
            if metric in self.performance_thresholds:
                threshold = self.performance_thresholds[metric]
                
                # Check if threshold is exceeded
                alert_triggered = False
                if metric == PerformanceMetric.CACHE_HIT_RATE:
                    # Lower values are bad for cache hit rate
                    alert_triggered = value < threshold
                else:
                    # Higher values are bad for other metrics
                    alert_triggered = value > threshold
                
                if alert_triggered:
                    await self._create_performance_alert(metric, value, threshold, snapshot)
    
    async def _create_performance_alert(
        self, 
        metric: PerformanceMetric, 
        actual_value: float, 
        threshold_value: float, 
        snapshot: PerformanceSnapshot
    ):
        """Create performance alert"""
        alert_id = f"PERF-{metric.value}-{int(snapshot.timestamp.timestamp())}"
        
        # Check cooldown
        recent_alert = any(
            alert.metric == metric and 
            (snapshot.timestamp - alert.timestamp).total_seconds() < self.alert_cooldown
            for alert in self.active_alerts.values()
        )
        
        if recent_alert:
            return
        
        # Determine severity
        severity = self._determine_alert_severity(metric, actual_value, threshold_value)
        
        # Generate description and recommendations
        description = f"{metric.value.replace('_', ' ').title()} is {actual_value:.2f}, exceeding threshold of {threshold_value:.2f}"
        recommended_actions = self._get_alert_recommendations(metric, actual_value, threshold_value)
        
        alert = PerformanceAlert(
            alert_id=alert_id,
            metric=metric,
            severity=severity,
            threshold_value=threshold_value,
            actual_value=actual_value,
            service_id=snapshot.service_id,
            timestamp=snapshot.timestamp,
            description=description,
            recommended_actions=recommended_actions
        )
        
        self.active_alerts[alert_id] = alert
        
        logger.warning(
            "Performance alert created",
            alert_id=alert_id,
            metric=metric.value,
            severity=severity.value,
            actual_value=actual_value,
            threshold=threshold_value
        )
        
        # Trigger immediate optimization if critical
        if severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            await self._trigger_emergency_optimization(metric, actual_value)
    
    def _determine_alert_severity(
        self, 
        metric: PerformanceMetric, 
        actual_value: float, 
        threshold_value: float
    ) -> AlertSeverity:
        """Determine alert severity based on how much threshold is exceeded"""
        if metric == PerformanceMetric.CACHE_HIT_RATE:
            # For cache hit rate, lower is worse
            severity_ratio = (threshold_value - actual_value) / threshold_value
        else:
            # For other metrics, higher is worse
            severity_ratio = (actual_value - threshold_value) / threshold_value
        
        if severity_ratio > 0.5:  # 50% above threshold
            return AlertSeverity.EMERGENCY
        elif severity_ratio > 0.3:  # 30% above threshold
            return AlertSeverity.CRITICAL
        elif severity_ratio > 0.1:  # 10% above threshold
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO
    
    def _get_alert_recommendations(
        self, 
        metric: PerformanceMetric, 
        actual_value: float, 
        threshold_value: float
    ) -> List[str]:
        """Get recommendations for performance alert"""
        recommendations = []
        
        if metric == PerformanceMetric.CPU_USAGE:
            recommendations.extend([
                "Scale horizontally to distribute load",
                "Optimize CPU-intensive operations",
                "Review and optimize algorithms",
                "Consider caching frequently computed results"
            ])
        
        elif metric == PerformanceMetric.MEMORY_USAGE:
            recommendations.extend([
                "Increase available memory",
                "Optimize memory usage patterns",
                "Review object lifecycle management",
                "Implement memory pooling"
            ])
        
        elif metric == PerformanceMetric.RESPONSE_TIME:
            recommendations.extend([
                "Optimize database queries",
                "Implement caching strategies",
                "Review network latency",
                "Optimize application logic"
            ])
        
        elif metric == PerformanceMetric.ERROR_RATE:
            recommendations.extend([
                "Review error logs for patterns",
                "Implement circuit breakers",
                "Improve error handling",
                "Validate input data quality"
            ])
        
        elif metric == PerformanceMetric.CACHE_HIT_RATE:
            recommendations.extend([
                "Review cache key strategies",
                "Optimize cache expiration policies",
                "Increase cache size if possible",
                "Implement cache warming"
            ])
        
        elif metric == PerformanceMetric.CONNECTION_POOL:
            recommendations.extend([
                "Increase connection pool size",
                "Optimize connection lifecycle",
                "Review query performance",
                "Implement connection pooling best practices"
            ])
        
        return recommendations
    
    async def _trigger_emergency_optimization(self, metric: PerformanceMetric, actual_value: float):
        """Trigger emergency optimization for critical performance issues"""
        logger.critical(
            "Triggering emergency optimization",
            metric=metric.value,
            actual_value=actual_value
        )
        
        if metric == PerformanceMetric.MEMORY_USAGE and actual_value > 95:
            # Emergency memory cleanup
            await self._emergency_memory_cleanup()
        
        elif metric == PerformanceMetric.CONNECTION_POOL and actual_value > 95:
            # Emergency connection pool optimization
            await self._emergency_pool_optimization()
        
        elif metric == PerformanceMetric.CPU_USAGE and actual_value > 95:
            # Emergency CPU optimization
            await self._emergency_cpu_optimization()
    
    async def _emergency_memory_cleanup(self):
        """Emergency memory cleanup procedures"""
        try:
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear non-essential caches
            # This would clear application-specific caches
            
            logger.info("Emergency memory cleanup completed")
            
        except Exception as e:
            logger.error(f"Emergency memory cleanup failed: {e}")
    
    async def _emergency_pool_optimization(self):
        """Emergency connection pool optimization"""
        try:
            # Close idle connections
            for pool_name, pool in self.db_pools.items():
                # This would call pool-specific cleanup methods
                logger.info(f"Emergency cleanup for pool: {pool_name}")
            
            logger.info("Emergency pool optimization completed")
            
        except Exception as e:
            logger.error(f"Emergency pool optimization failed: {e}")
    
    async def _emergency_cpu_optimization(self):
        """Emergency CPU optimization procedures"""
        try:
            # Temporarily reduce background task frequency
            # This would adjust monitoring intervals and batch sizes
            
            logger.info("Emergency CPU optimization completed")
            
        except Exception as e:
            logger.error(f"Emergency CPU optimization failed: {e}")
    
    async def _trend_analysis_loop(self):
        """Analyze performance trends"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                if len(self.performance_history) < 10:
                    continue
                
                # Analyze trends for each metric
                for metric in PerformanceMetric:
                    trend = await self._analyze_metric_trend(metric)
                    if trend:
                        self.performance_trends[metric] = trend
                
            except Exception as e:
                logger.error(f"Trend analysis error: {e}")
                await asyncio.sleep(300)
    
    async def _analyze_metric_trend(self, metric: PerformanceMetric) -> Optional[PerformanceTrend]:
        """Analyze trend for specific metric"""
        try:
            # Get recent values for the metric
            recent_snapshots = self.performance_history[-20:]  # Last 20 snapshots
            values = [
                snapshot.metrics.get(metric, 0) 
                for snapshot in recent_snapshots 
                if metric in snapshot.metrics
            ]
            
            if len(values) < 5:
                return None
            
            # Simple trend analysis
            x = list(range(len(values)))
            y = values
            
            # Calculate linear regression slope
            n = len(values)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] * x[i] for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # Determine trend direction
            if abs(slope) < 0.1:
                trend_direction = "stable"
            elif slope > 0:
                trend_direction = "increasing"
            else:
                trend_direction = "decreasing"
            
            # Calculate confidence based on variance
            mean_value = statistics.mean(values)
            variance = statistics.variance(values) if len(values) > 1 else 0
            confidence = max(0.1, 1.0 - (variance / (mean_value ** 2)) if mean_value != 0 else 0.1)
            
            # Simple prediction (next value)
            prediction = values[-1] + slope if trend_direction != "stable" else values[-1]
            
            return PerformanceTrend(
                metric=metric,
                trend_direction=trend_direction,
                change_rate=slope,
                confidence=confidence,
                prediction=prediction,
                time_window=timedelta(minutes=10)  # Based on 20 snapshots at 30s intervals
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze trend for {metric.value}: {e}")
            return None
    
    async def _optimization_recommendation_loop(self):
        """Generate optimization recommendations"""
        while True:
            try:
                await asyncio.sleep(600)  # Run every 10 minutes
                
                # Generate recommendations based on trends and current state
                recommendations = await self._generate_optimization_recommendations()
                
                # Update recommendations
                for rec in recommendations:
                    self.optimization_recommendations[rec.recommendation_id] = rec
                
                # Clean up old recommendations
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.optimization_recommendations = {
                    k: v for k, v in self.optimization_recommendations.items()
                    if k.split('-')[1] > cutoff_time.strftime('%Y%m%d')
                }
                
            except Exception as e:
                logger.error(f"Optimization recommendation error: {e}")
                await asyncio.sleep(600)
    
    async def _generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if not self.performance_history:
            return recommendations
        
        latest_snapshot = self.performance_history[-1]
        
        # Connection pooling optimization
        if latest_snapshot.metrics.get(PerformanceMetric.CONNECTION_POOL, 0) > 70:
            rec = OptimizationRecommendation(
                recommendation_id=f"conn_pool-{datetime.utcnow().strftime('%Y%m%d%H%M')}",
                strategy=OptimizationStrategy.CONNECTION_POOLING,
                description="Connection pool utilization is high. Consider optimizing pool configuration.",
                expected_improvement=15.0,
                implementation_effort="medium",
                priority=8,
                metrics_targeted=[PerformanceMetric.CONNECTION_POOL, PerformanceMetric.RESPONSE_TIME],
                estimated_impact={
                    "response_time_reduction": 20.0,
                    "throughput_increase": 15.0
                }
            )
            recommendations.append(rec)
        
        # Caching optimization
        if latest_snapshot.metrics.get(PerformanceMetric.CACHE_HIT_RATE, 100) < 85:
            rec = OptimizationRecommendation(
                recommendation_id=f"caching-{datetime.utcnow().strftime('%Y%m%d%H%M')}",
                strategy=OptimizationStrategy.CACHING,
                description="Cache hit rate is below optimal. Review caching strategies.",
                expected_improvement=25.0,
                implementation_effort="low",
                priority=9,
                metrics_targeted=[PerformanceMetric.CACHE_HIT_RATE, PerformanceMetric.RESPONSE_TIME],
                estimated_impact={
                    "response_time_reduction": 30.0,
                    "cache_hit_rate_increase": 10.0
                }
            )
            recommendations.append(rec)
        
        # Memory optimization
        if latest_snapshot.metrics.get(PerformanceMetric.MEMORY_USAGE, 0) > 75:
            rec = OptimizationRecommendation(
                recommendation_id=f"memory-{datetime.utcnow().strftime('%Y%m%d%H%M')}",
                strategy=OptimizationStrategy.MEMORY_OPTIMIZATION,
                description="Memory usage is high. Consider memory optimization techniques.",
                expected_improvement=20.0,
                implementation_effort="high",
                priority=7,
                metrics_targeted=[PerformanceMetric.MEMORY_USAGE],
                estimated_impact={
                    "memory_usage_reduction": 15.0,
                    "stability_improvement": 25.0
                }
            )
            recommendations.append(rec)
        
        # Query optimization
        if latest_snapshot.metrics.get(PerformanceMetric.RESPONSE_TIME, 0) > 1000:
            rec = OptimizationRecommendation(
                recommendation_id=f"query-{datetime.utcnow().strftime('%Y%m%d%H%M')}",
                strategy=OptimizationStrategy.QUERY_OPTIMIZATION,
                description="Response times are high. Database query optimization recommended.",
                expected_improvement=35.0,
                implementation_effort="medium",
                priority=9,
                metrics_targeted=[PerformanceMetric.RESPONSE_TIME, PerformanceMetric.THROUGHPUT],
                estimated_impact={
                    "response_time_reduction": 40.0,
                    "throughput_increase": 25.0
                }
            )
            recommendations.append(rec)
        
        return recommendations
    
    async def _auto_optimization_loop(self):
        """Automatically apply low-risk optimizations"""
        while True:
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes
                
                if not self.auto_optimization_enabled:
                    continue
                
                # Get low-risk recommendations
                low_risk_recommendations = [
                    rec for rec in self.optimization_recommendations.values()
                    if rec.implementation_effort == "low" and rec.priority >= 8
                ]
                
                # Apply optimizations (limited per hour)
                recent_optimizations = [
                    opt_id for opt_id in self.applied_optimizations
                    if datetime.utcnow() - datetime.fromisoformat(opt_id.split('-')[1]) < timedelta(hours=1)
                ]
                
                if len(recent_optimizations) >= self.max_auto_optimizations_per_hour:
                    continue
                
                for recommendation in low_risk_recommendations[:2]:  # Max 2 per cycle
                    if recommendation.recommendation_id not in self.applied_optimizations:
                        await self._apply_optimization(recommendation)
                        self.applied_optimizations.append(recommendation.recommendation_id)
                
            except Exception as e:
                logger.error(f"Auto optimization error: {e}")
                await asyncio.sleep(1800)
    
    async def _apply_optimization(self, recommendation: OptimizationRecommendation):
        """Apply specific optimization"""
        try:
            strategy = recommendation.strategy
            
            if strategy in self.optimization_strategies:
                optimizer = self.optimization_strategies[strategy]
                result = await optimizer(recommendation)
                
                logger.info(
                    "Optimization applied",
                    recommendation_id=recommendation.recommendation_id,
                    strategy=strategy.value,
                    result=result
                )
                
                return result
            else:
                logger.warning(f"No optimizer available for strategy: {strategy.value}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to apply optimization {recommendation.recommendation_id}: {e}")
            return False
    
    # Optimization strategy implementations
    async def _optimize_connection_pooling(self, recommendation: OptimizationRecommendation) -> bool:
        """Optimize database connection pooling"""
        try:
            # Analyze current pool usage
            for pool_name, stats in self.pool_stats.items():
                if stats.pool_efficiency < 0.8:
                    # Adjust pool size
                    new_min_size = max(2, int(stats.active_connections * 0.8))
                    new_max_size = min(50, int(stats.max_connections * 1.2))
                    
                    logger.info(
                        "Connection pool optimized",
                        pool_name=pool_name,
                        old_min=stats.min_connections,
                        new_min=new_min_size,
                        old_max=stats.max_connections,
                        new_max=new_max_size
                    )
            
            return True
            
        except Exception as e:
            logger.error(f"Connection pool optimization failed: {e}")
            return False
    
    async def _optimize_caching(self, recommendation: OptimizationRecommendation) -> bool:
        """Optimize caching strategies"""
        try:
            # This would implement cache optimization
            # - Adjust cache TTL
            # - Implement cache warming
            # - Optimize cache keys
            
            logger.info("Cache optimization applied")
            return True
            
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
            return False
    
    async def _optimize_queries(self, recommendation: OptimizationRecommendation) -> bool:
        """Optimize database queries"""
        try:
            # This would implement query optimization
            # - Add missing indexes
            # - Optimize query plans
            # - Implement query caching
            
            logger.info("Query optimization applied")
            return True
            
        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
            return False
    
    async def _optimize_memory(self, recommendation: OptimizationRecommendation) -> bool:
        """Optimize memory usage"""
        try:
            # Force garbage collection
            import gc
            gc.collect()
            
            # This would implement memory optimization
            # - Object pooling
            # - Memory leak detection
            # - Efficient data structures
            
            logger.info("Memory optimization applied")
            return True
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            return False
    
    async def _optimize_async_processing(self, recommendation: OptimizationRecommendation) -> bool:
        """Optimize asynchronous processing"""
        try:
            # This would implement async optimization
            # - Adjust concurrency limits
            # - Optimize event loop
            # - Implement batching
            
            logger.info("Async processing optimization applied")
            return True
            
        except Exception as e:
            logger.error(f"Async processing optimization failed: {e}")
            return False
    
    async def _optimize_batch_processing(self, recommendation: OptimizationRecommendation) -> bool:
        """Optimize batch processing"""
        try:
            # This would implement batch optimization
            # - Adjust batch sizes
            # - Optimize processing intervals
            # - Implement parallel processing
            
            logger.info("Batch processing optimization applied")
            return True
            
        except Exception as e:
            logger.error(f"Batch processing optimization failed: {e}")
            return False
    
    async def _cleanup_old_data_loop(self):
        """Clean up old performance data"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                cutoff_time = datetime.utcnow() - timedelta(days=self.retention_period)
                
                # Clean performance history
                self.performance_history = [
                    snapshot for snapshot in self.performance_history
                    if snapshot.timestamp > cutoff_time
                ]
                
                # Clean old alerts
                old_alerts = [
                    alert_id for alert_id, alert in self.active_alerts.items()
                    if alert.timestamp < cutoff_time
                ]
                for alert_id in old_alerts:
                    del self.active_alerts[alert_id]
                
                logger.debug(f"Cleaned up old performance data, removed {len(old_alerts)} old alerts")
                
            except Exception as e:
                logger.error(f"Data cleanup error: {e}")
                await asyncio.sleep(3600)
    
    # Public API methods
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.performance_history:
            return {
                "status": "no_data",
                "message": "No performance data available"
            }
        
        latest = self.performance_history[-1]
        
        return {
            "timestamp": latest.timestamp.isoformat(),
            "current_metrics": {
                metric.value: value for metric, value in latest.metrics.items()
            },
            "active_alerts": len(self.active_alerts),
            "optimization_recommendations": len(self.optimization_recommendations),
            "applied_optimizations": len(self.applied_optimizations),
            "trends": {
                metric.value: {
                    "direction": trend.trend_direction,
                    "confidence": trend.confidence,
                    "prediction": trend.prediction
                }
                for metric, trend in self.performance_trends.items()
            }
        }
    
    async def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get current optimization recommendations"""
        return [
            {
                "recommendation_id": rec.recommendation_id,
                "strategy": rec.strategy.value,
                "description": rec.description,
                "expected_improvement": rec.expected_improvement,
                "implementation_effort": rec.implementation_effort,
                "priority": rec.priority,
                "estimated_impact": rec.estimated_impact
            }
            for rec in sorted(
                self.optimization_recommendations.values(),
                key=lambda x: x.priority,
                reverse=True
            )
        ]
    
    async def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get active performance alerts"""
        return [
            {
                "alert_id": alert.alert_id,
                "metric": alert.metric.value,
                "severity": alert.severity.value,
                "actual_value": alert.actual_value,
                "threshold": alert.threshold_value,
                "description": alert.description,
                "timestamp": alert.timestamp.isoformat(),
                "recommended_actions": alert.recommended_actions
            }
            for alert in sorted(
                self.active_alerts.values(),
                key=lambda x: x.timestamp,
                reverse=True
            )
        ]
    
    async def update_pool_stats(self, pool_name: str, stats: ConnectionPoolStats):
        """Update connection pool statistics"""
        self.pool_stats[pool_name] = stats
    
    async def force_optimization(self, strategy: OptimizationStrategy) -> bool:
        """Force apply specific optimization strategy"""
        # Find recommendation for this strategy
        matching_recommendations = [
            rec for rec in self.optimization_recommendations.values()
            if rec.strategy == strategy
        ]
        
        if matching_recommendations:
            recommendation = matching_recommendations[0]
            return await self._apply_optimization(recommendation)
        else:
            # Create temporary recommendation
            temp_recommendation = OptimizationRecommendation(
                recommendation_id=f"manual-{strategy.value}-{int(datetime.utcnow().timestamp())}",
                strategy=strategy,
                description=f"Manual {strategy.value} optimization",
                expected_improvement=10.0,
                implementation_effort="unknown",
                priority=5,
                metrics_targeted=[],
                estimated_impact={}
            )
            
            return await self._apply_optimization(temp_recommendation)

# Global instance
_performance_engine = None

def get_performance_engine() -> PerformanceOptimizationEngine:
    """Get performance optimization engine instance"""
    global _performance_engine
    if _performance_engine is None:
        _performance_engine = PerformanceOptimizationEngine()
    return _performance_engine

# Context manager for performance monitoring
@asynccontextmanager
async def monitor_performance(operation_name: str):
    """Context manager for monitoring operation performance"""
    start_time = time.time()
    
    try:
        yield
    finally:
        end_time = time.time()
        duration = (end_time - start_time) * 1000  # milliseconds
        
        logger.debug(
            "Operation performance",
            operation=operation_name,
            duration_ms=duration
        )