"""
Advanced Performance Monitor - Principal Auditor Implementation
Production-ready performance monitoring and optimization for cybersecurity systems
"""

import asyncio
import json
import logging
import time
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

# Advanced monitoring imports with graceful fallbacks
try:
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available, using fallback implementations")

try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available, using fallback metrics")

from .base_service import XORBService, ServiceHealth, ServiceStatus

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of performance metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str]
    metric_type: MetricType

@dataclass
class PerformanceAlert:
    """Performance alert data structure"""
    alert_id: str
    metric_name: str
    current_value: float
    threshold: float
    severity: AlertSeverity
    description: str
    timestamp: datetime
    resolution_suggestions: List[str]

@dataclass
class OptimizationRecommendation:
    """System optimization recommendation"""
    recommendation_id: str
    component: str
    issue_description: str
    recommendation: str
    expected_improvement: str
    implementation_complexity: str
    priority: int
    estimated_impact: float

class PerformanceProfiler:
    """Advanced performance profiler for detailed analysis"""
    
    def __init__(self):
        self.active_profiles = {}
        self.profile_results = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def start_profiling(self, session_id: str, component: str):
        """Start performance profiling session"""
        profile_data = {
            'session_id': session_id,
            'component': component,
            'start_time': time.time(),
            'cpu_usage_history': deque(maxlen=1000),
            'memory_usage_history': deque(maxlen=1000),
            'io_stats': {'read_bytes': 0, 'write_bytes': 0},
            'function_calls': defaultdict(int),
            'execution_times': defaultdict(list)
        }
        self.active_profiles[session_id] = profile_data
        return profile_data
    
    def record_function_call(self, session_id: str, function_name: str, execution_time: float):
        """Record function call performance"""
        if session_id in self.active_profiles:
            profile = self.active_profiles[session_id]
            profile['function_calls'][function_name] += 1
            profile['execution_times'][function_name].append(execution_time)
    
    def stop_profiling(self, session_id: str) -> Dict[str, Any]:
        """Stop profiling and return results"""
        if session_id not in self.active_profiles:
            return {}
        
        profile = self.active_profiles.pop(session_id)
        end_time = time.time()
        
        # Calculate performance statistics
        results = {
            'session_id': session_id,
            'component': profile['component'],
            'duration': end_time - profile['start_time'],
            'function_performance': {},
            'resource_usage': {
                'avg_cpu': np.mean(list(profile['cpu_usage_history'])) if profile['cpu_usage_history'] else 0,
                'max_cpu': max(profile['cpu_usage_history']) if profile['cpu_usage_history'] else 0,
                'avg_memory': np.mean(list(profile['memory_usage_history'])) if profile['memory_usage_history'] else 0,
                'max_memory': max(profile['memory_usage_history']) if profile['memory_usage_history'] else 0,
            },
            'io_stats': profile['io_stats']
        }
        
        # Analyze function performance
        for func_name, call_count in profile['function_calls'].items():
            execution_times = profile['execution_times'][func_name]
            results['function_performance'][func_name] = {
                'call_count': call_count,
                'total_time': sum(execution_times),
                'avg_time': np.mean(execution_times) if execution_times else 0,
                'max_time': max(execution_times) if execution_times else 0,
                'min_time': min(execution_times) if execution_times else 0
            }
        
        self.profile_results[session_id] = results
        return results

class AdvancedPerformanceMonitor(XORBService):
    """Production-ready advanced performance monitoring and optimization service"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        
        # Performance data storage
        self.metrics_history = defaultdict(lambda: deque(maxlen=10000))
        self.alerts_history = deque(maxlen=1000)
        self.optimization_recommendations = []
        
        # Monitoring configuration
        self.monitoring_interval = config.get('monitoring_interval', 30)  # seconds
        self.alert_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'response_time': 5.0,  # seconds
            'error_rate': 5.0,  # percentage
            'throughput_degradation': 20.0  # percentage
        }
        
        # Performance baselines
        self.baselines = {}
        self.baseline_window = timedelta(days=7)
        
        # Advanced monitoring components
        self.profiler = PerformanceProfiler()
        self.anomaly_detector = None
        self.metric_scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
        # Prometheus metrics (if available)
        if PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_task = None
        
        logger.info("Advanced Performance Monitor initialized")
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics collectors"""
        try:
            self.prometheus_metrics = {
                'api_request_duration': Histogram(
                    'xorb_api_request_duration_seconds',
                    'API request duration in seconds',
                    ['method', 'endpoint', 'status']
                ),
                'security_scan_duration': Histogram(
                    'xorb_security_scan_duration_seconds',
                    'Security scan duration in seconds',
                    ['scan_type', 'target_type']
                ),
                'threat_detection_accuracy': Gauge(
                    'xorb_threat_detection_accuracy',
                    'Threat detection accuracy percentage'
                ),
                'system_cpu_usage': Gauge(
                    'xorb_system_cpu_usage_percent',
                    'System CPU usage percentage'
                ),
                'system_memory_usage': Gauge(
                    'xorb_system_memory_usage_percent',
                    'System memory usage percentage'
                ),
                'active_security_sessions': Gauge(
                    'xorb_active_security_sessions',
                    'Number of active security sessions'
                ),
                'error_rate': Counter(
                    'xorb_errors_total',
                    'Total number of errors',
                    ['component', 'error_type']
                ),
                'ml_model_inference_time': Histogram(
                    'xorb_ml_model_inference_seconds',
                    'ML model inference time in seconds',
                    ['model_name', 'model_type']
                )
            }
            logger.info("Prometheus metrics setup completed")
        except Exception as e:
            logger.error(f"Failed to setup Prometheus metrics: {e}")
    
    async def initialize(self) -> bool:
        """Initialize the performance monitor"""
        try:
            logger.info("Initializing Advanced Performance Monitor...")
            
            # Initialize anomaly detection
            if SKLEARN_AVAILABLE:
                await self._initialize_anomaly_detection()
            
            # Calculate initial baselines
            await self._calculate_baselines()
            
            # Start monitoring
            await self.start_monitoring()
            
            self.status = ServiceStatus.HEALTHY
            logger.info("Advanced Performance Monitor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize performance monitor: {e}")
            self.status = ServiceStatus.UNHEALTHY
            return False
    
    async def _initialize_anomaly_detection(self):
        """Initialize ML-based anomaly detection"""
        try:
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            logger.info("Anomaly detection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize anomaly detection: {e}")
    
    async def start_monitoring(self):
        """Start real-time performance monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop real-time performance monitoring"""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metrics = await self._collect_system_metrics()
                
                # Collect application metrics
                app_metrics = await self._collect_application_metrics()
                
                # Combine all metrics
                all_metrics = {**system_metrics, **app_metrics}
                
                # Store metrics
                await self._store_metrics(all_metrics)
                
                # Check for alerts
                alerts = await self._check_alert_conditions(all_metrics)
                if alerts:
                    await self._process_alerts(alerts)
                
                # Detect anomalies
                if self.anomaly_detector:
                    anomalies = await self._detect_performance_anomalies(all_metrics)
                    if anomalies:
                        await self._process_anomalies(anomalies)
                
                # Update Prometheus metrics
                if PROMETHEUS_AVAILABLE:
                    await self._update_prometheus_metrics(all_metrics)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            network_io = psutil.net_io_counters()
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                'system_cpu_percent': cpu_percent,
                'system_cpu_count': cpu_count,
                'system_memory_percent': memory.percent,
                'system_memory_available': memory.available,
                'system_memory_used': memory.used,
                'system_swap_percent': swap.percent,
                'system_disk_percent': (disk_usage.used / disk_usage.total) * 100,
                'system_disk_free': disk_usage.free,
                'system_disk_read_bytes': disk_io.read_bytes if disk_io else 0,
                'system_disk_write_bytes': disk_io.write_bytes if disk_io else 0,
                'system_network_bytes_sent': network_io.bytes_sent if network_io else 0,
                'system_network_bytes_recv': network_io.bytes_recv if network_io else 0,
                'process_memory_rss': process_memory.rss,
                'process_memory_vms': process_memory.vms,
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {}
    
    async def _collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-specific performance metrics"""
        try:
            metrics = {}
            
            # Python-specific metrics
            metrics['python_gc_collections'] = sum(gc.get_count())
            metrics['python_gc_objects'] = len(gc.get_objects())
            
            # Asyncio metrics
            loop = asyncio.get_event_loop()
            tasks = asyncio.all_tasks(loop)
            metrics['asyncio_task_count'] = len(tasks)
            metrics['asyncio_running_tasks'] = len([t for t in tasks if not t.done()])
            
            # Custom application metrics (these would be injected by other services)
            metrics.update(await self._get_custom_application_metrics())
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")
            return {}
    
    async def _get_custom_application_metrics(self) -> Dict[str, Any]:
        """Get custom application metrics from other services"""
        # This would integrate with other services to collect their metrics
        return {
            'ptaas_active_scans': 0,  # Would be injected by PTaaS service
            'threat_predictions_per_minute': 0,  # Would be injected by threat prediction
            'behavioral_profiles_analyzed': 0,  # Would be injected by behavioral analytics
            'quantum_crypto_operations': 0,  # Would be injected by quantum security
            'autonomous_plans_active': 0  # Would be injected by orchestrator
        }
    
    async def _store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics in history"""
        timestamp = datetime.utcnow()
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                metric = PerformanceMetric(
                    name=metric_name,
                    value=float(value),
                    timestamp=timestamp,
                    labels={},
                    metric_type=MetricType.GAUGE
                )
                self.metrics_history[metric_name].append(metric)
    
    async def _check_alert_conditions(self, metrics: Dict[str, Any]) -> List[PerformanceAlert]:
        """Check if any metrics exceed alert thresholds"""
        alerts = []
        
        for metric_name, threshold in self.alert_thresholds.items():
            if metric_name in metrics:
                current_value = metrics[metric_name]
                
                if isinstance(current_value, (int, float)) and current_value > threshold:
                    alert = PerformanceAlert(
                        alert_id=f"alert_{int(time.time())}_{metric_name}",
                        metric_name=metric_name,
                        current_value=current_value,
                        threshold=threshold,
                        severity=self._determine_alert_severity(metric_name, current_value, threshold),
                        description=f"{metric_name} exceeded threshold: {current_value:.2f} > {threshold:.2f}",
                        timestamp=datetime.utcnow(),
                        resolution_suggestions=self._get_resolution_suggestions(metric_name)
                    )
                    alerts.append(alert)
        
        return alerts
    
    def _determine_alert_severity(self, metric_name: str, current_value: float, threshold: float) -> AlertSeverity:
        """Determine alert severity based on how much threshold is exceeded"""
        excess_percentage = ((current_value - threshold) / threshold) * 100
        
        if excess_percentage > 50:
            return AlertSeverity.CRITICAL
        elif excess_percentage > 25:
            return AlertSeverity.HIGH
        elif excess_percentage > 10:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def _get_resolution_suggestions(self, metric_name: str) -> List[str]:
        """Get resolution suggestions for specific metrics"""
        suggestions = {
            'cpu_usage': [
                "Scale up compute resources",
                "Optimize CPU-intensive algorithms",
                "Implement task queuing",
                "Profile and optimize hot code paths"
            ],
            'memory_usage': [
                "Increase available memory",
                "Implement memory caching strategies",
                "Optimize data structures",
                "Fix memory leaks",
                "Implement garbage collection tuning"
            ],
            'disk_usage': [
                "Clean up temporary files",
                "Archive old data",
                "Implement data compression",
                "Add storage capacity"
            ],
            'response_time': [
                "Optimize database queries",
                "Implement caching",
                "Scale horizontal infrastructure",
                "Optimize network configuration"
            ],
            'error_rate': [
                "Review recent code changes",
                "Check service dependencies",
                "Implement circuit breakers",
                "Improve error handling"
            ]
        }
        
        return suggestions.get(metric_name, ["Investigate root cause", "Contact system administrator"])
    
    async def _process_alerts(self, alerts: List[PerformanceAlert]):
        """Process and handle performance alerts"""
        for alert in alerts:
            self.alerts_history.append(alert)
            
            logger.warning(
                f"Performance Alert [{alert.severity.value.upper()}]: {alert.description}"
            )
            
            # Auto-remediation for certain conditions
            if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
                await self._attempt_auto_remediation(alert)
    
    async def _attempt_auto_remediation(self, alert: PerformanceAlert):
        """Attempt automatic remediation for critical alerts"""
        try:
            metric_name = alert.metric_name
            
            if metric_name == 'system_memory_percent' and alert.current_value > 90:
                # Force garbage collection
                gc.collect()
                logger.info("Performed garbage collection for high memory usage")
            
            elif metric_name == 'asyncio_task_count' and alert.current_value > 1000:
                # Cancel completed tasks
                loop = asyncio.get_event_loop()
                tasks = [t for t in asyncio.all_tasks(loop) if t.done()]
                for task in tasks[:100]:  # Cancel up to 100 completed tasks
                    task.cancel()
                logger.info(f"Cleaned up {len(tasks)} completed asyncio tasks")
            
            # Log auto-remediation attempt
            logger.info(f"Attempted auto-remediation for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed auto-remediation for alert {alert.alert_id}: {e}")
    
    async def generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate system optimization recommendations based on performance data"""
        try:
            recommendations = []
            
            # Analyze recent performance trends
            performance_analysis = await self._analyze_performance_trends()
            
            # CPU optimization recommendations
            if performance_analysis['cpu']['avg_usage'] > 70:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"opt_cpu_{int(time.time())}",
                    component="CPU",
                    issue_description="High CPU usage detected consistently",
                    recommendation="Implement CPU optimization strategies: code profiling, algorithm optimization, task distribution",
                    expected_improvement="20-40% CPU usage reduction",
                    implementation_complexity="Medium",
                    priority=8,
                    estimated_impact=0.3
                ))
            
            # Memory optimization recommendations
            if performance_analysis['memory']['avg_usage'] > 80:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"opt_memory_{int(time.time())}",
                    component="Memory",
                    issue_description="High memory usage patterns observed",
                    recommendation="Implement memory optimization: object pooling, lazy loading, memory-efficient data structures",
                    expected_improvement="15-30% memory usage reduction",
                    implementation_complexity="Medium",
                    priority=7,
                    estimated_impact=0.25
                ))
            
            # I/O optimization recommendations
            if performance_analysis.get('io', {}).get('bottleneck_detected', False):
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"opt_io_{int(time.time())}",
                    component="I/O",
                    issue_description="I/O bottlenecks affecting performance",
                    recommendation="Implement I/O optimization: async I/O, connection pooling, caching strategies",
                    expected_improvement="30-50% I/O performance improvement",
                    implementation_complexity="High",
                    priority=9,
                    estimated_impact=0.4
                ))
            
            # Sort by priority and store
            recommendations.sort(key=lambda x: x.priority, reverse=True)
            self.optimization_recommendations.extend(recommendations)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate optimization recommendations: {e}")
            return []
    
    async def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        try:
            analysis = {
                'cpu': {'avg_usage': 0, 'trend': 'stable'},
                'memory': {'avg_usage': 0, 'trend': 'stable'},
                'io': {'bottleneck_detected': False}
            }
            
            # Analyze CPU trends
            if 'system_cpu_percent' in self.metrics_history:
                cpu_metrics = list(self.metrics_history['system_cpu_percent'])[-100:]  # Last 100 samples
                if cpu_metrics:
                    cpu_values = [m.value for m in cpu_metrics]
                    analysis['cpu']['avg_usage'] = np.mean(cpu_values) if cpu_values else 0
                    
                    # Simple trend detection
                    if len(cpu_values) >= 10:
                        recent_avg = np.mean(cpu_values[-10:])
                        older_avg = np.mean(cpu_values[-20:-10]) if len(cpu_values) >= 20 else recent_avg
                        if recent_avg > older_avg * 1.1:
                            analysis['cpu']['trend'] = 'increasing'
                        elif recent_avg < older_avg * 0.9:
                            analysis['cpu']['trend'] = 'decreasing'
            
            # Analyze memory trends
            if 'system_memory_percent' in self.metrics_history:
                memory_metrics = list(self.metrics_history['system_memory_percent'])[-100:]
                if memory_metrics:
                    memory_values = [m.value for m in memory_metrics]
                    analysis['memory']['avg_usage'] = np.mean(memory_values) if memory_values else 0
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze performance trends: {e}")
            return {}
    
    async def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data"""
        try:
            # Current system status
            current_metrics = await self._collect_system_metrics()
            
            # Recent alerts
            recent_alerts = list(self.alerts_history)[-10:]
            
            # Performance trends
            trends = await self._analyze_performance_trends()
            
            # Optimization recommendations
            recommendations = await self.generate_optimization_recommendations()
            
            return {
                'current_metrics': current_metrics,
                'recent_alerts': [asdict(alert) for alert in recent_alerts],
                'performance_trends': trends,
                'optimization_recommendations': [asdict(rec) for rec in recommendations[-5:]],
                'monitoring_status': {
                    'active': self.monitoring_active,
                    'interval': self.monitoring_interval,
                    'metrics_collected': sum(len(history) for history in self.metrics_history.values()),
                    'alerts_generated': len(self.alerts_history)
                },
                'system_health_score': await self._calculate_system_health_score()
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance dashboard: {e}")
            return {}
    
    async def _calculate_system_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""
        try:
            score = 100.0
            
            # Get recent metrics
            current_metrics = await self._collect_system_metrics()
            
            # Deduct points for high resource usage
            if current_metrics.get('system_cpu_percent', 0) > 80:
                score -= 20
            elif current_metrics.get('system_cpu_percent', 0) > 60:
                score -= 10
            
            if current_metrics.get('system_memory_percent', 0) > 85:
                score -= 20
            elif current_metrics.get('system_memory_percent', 0) > 70:
                score -= 10
            
            if current_metrics.get('system_disk_percent', 0) > 90:
                score -= 15
            elif current_metrics.get('system_disk_percent', 0) > 80:
                score -= 8
            
            # Deduct points for recent alerts
            recent_alerts = [alert for alert in self.alerts_history 
                           if (datetime.utcnow() - alert.timestamp).total_seconds() < 3600]
            
            critical_alerts = [a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]
            high_alerts = [a for a in recent_alerts if a.severity == AlertSeverity.HIGH]
            
            score -= len(critical_alerts) * 15
            score -= len(high_alerts) * 8
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"Failed to calculate system health score: {e}")
            return 50.0  # Default neutral score
    
    def get_health(self) -> ServiceHealth:
        """Get service health status"""
        try:
            is_healthy = (
                self.status == ServiceStatus.HEALTHY and
                self.monitoring_active
            )
            
            return ServiceHealth(
                service_name="AdvancedPerformanceMonitor",
                is_healthy=is_healthy,
                status=self.status.value,
                details={
                    "monitoring_active": self.monitoring_active,
                    "metrics_history_size": sum(len(history) for history in self.metrics_history.values()),
                    "alerts_count": len(self.alerts_history),
                    "recommendations_count": len(self.optimization_recommendations)
                }
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return ServiceHealth(
                service_name="AdvancedPerformanceMonitor",
                is_healthy=False,
                status="unhealthy",
                details={"error": str(e)}
            )