#!/usr/bin/env python3
"""
Advanced Metrics Collector for Xorb 2.0 RL Performance Tracking

This module provides comprehensive metrics collection specifically designed for
reinforcement learning performance monitoring, EPYC optimization tracking,
and knowledge fabric analytics. Integrates with Prometheus, OpenTelemetry,
and custom ML performance metrics.
"""

import asyncio
import logging
import time
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from pathlib import Path
import threading

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, CollectorRegistry, 
        generate_latest, CONTENT_TYPE_LATEST, Info
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available. Install with: pip install prometheus-client")

try:
    from opentelemetry import trace, metrics as otel_metrics
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logging.warning("OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk")

import psutil


@dataclass
class RLPerformanceMetrics:
    """Reinforcement Learning performance metrics"""
    agent_selection_accuracy: float = 0.0
    campaign_success_rate: float = 0.0
    reward_convergence_rate: float = 0.0
    exploration_vs_exploitation_ratio: float = 0.0
    q_value_stability: float = 0.0
    learning_rate_effectiveness: float = 0.0
    episode_length_avg: float = 0.0
    cumulative_reward: float = 0.0
    model_loss: float = 0.0
    epsilon_decay_rate: float = 0.0


@dataclass
class EPYCPerformanceMetrics:
    """EPYC-specific performance metrics"""
    core_utilization_efficiency: float = 0.0
    numa_memory_locality_ratio: float = 0.0
    l3_cache_hit_ratio: float = 0.0
    ccx_load_balance_score: float = 0.0
    memory_bandwidth_utilization: float = 0.0
    context_switches_per_second: float = 0.0
    cpu_frequency_scaling_efficiency: float = 0.0
    thermal_throttling_incidents: int = 0
    power_efficiency_score: float = 0.0
    hyperthreading_effectiveness: float = 0.0


@dataclass
class KnowledgeFabricMetrics:
    """Knowledge fabric performance metrics"""
    knowledge_graph_coverage_ratio: float = 0.0
    prediction_confidence_avg: float = 0.0
    atom_relationship_density: float = 0.0
    hot_cache_hit_ratio: float = 0.0
    warm_cache_hit_ratio: float = 0.0
    query_response_time_p95: float = 0.0
    knowledge_freshness_score: float = 0.0
    validation_success_rate: float = 0.0
    cross_reference_accuracy: float = 0.0
    semantic_similarity_score: float = 0.0


@dataclass
class CampaignMetrics:
    """Campaign execution metrics"""
    campaign_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    target_count: int = 0
    agent_count: int = 0
    findings_total: int = 0
    findings_high_severity: int = 0
    resource_efficiency_score: float = 0.0
    execution_time_seconds: float = 0.0
    cost_per_finding: float = 0.0
    success_rate: float = 0.0
    agent_utilization_avg: float = 0.0
    network_latency_avg: float = 0.0
    error_rate: float = 0.0
    retry_count: int = 0


@dataclass 
class SystemHealthMetrics:
    """Overall system health metrics"""
    overall_health_score: float = 0.0
    uptime_percentage: float = 0.0
    error_rate: float = 0.0
    response_time_p99: float = 0.0
    resource_saturation_level: float = 0.0
    concurrent_campaigns: int = 0
    active_agents: int = 0
    memory_pressure: float = 0.0
    disk_io_pressure: float = 0.0
    network_throughput: float = 0.0


class AdvancedMetricsCollector:
    """
    Advanced metrics collector for comprehensive RL and system performance monitoring
    """
    
    def __init__(self, 
                 collection_interval: int = 15,
                 retention_hours: int = 24,
                 epyc_cores: int = 64):
        
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours
        self.epyc_cores = epyc_cores
        self.logger = logging.getLogger(__name__)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Metrics storage
        self.rl_metrics_history = deque(maxlen=retention_hours * (3600 // collection_interval))
        self.epyc_metrics_history = deque(maxlen=retention_hours * (3600 // collection_interval))
        self.knowledge_metrics_history = deque(maxlen=retention_hours * (3600 // collection_interval))
        self.campaign_metrics = {}  # campaign_id -> CampaignMetrics
        self.system_health_history = deque(maxlen=retention_hours * (3600 // collection_interval))
        
        # Real-time tracking
        self.active_campaigns = set()
        self.active_agents = defaultdict(dict)
        self.performance_baselines = {}
        
        # Prometheus metrics
        self.prometheus_registry = None
        self.prometheus_metrics = {}
        
        # OpenTelemetry
        self.otel_meter = None
        self.otel_metrics = {}
        
        # Collection task
        self._collection_task = None
        self._running = False
        
        # Initialize metrics systems
        self._initialize_prometheus()
        self._initialize_opentelemetry()
        
        self.logger.info(f"Advanced metrics collector initialized with {collection_interval}s interval")
    
    def _initialize_prometheus(self):
        """Initialize Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.prometheus_registry = CollectorRegistry()
        
        # RL Performance Metrics
        self.prometheus_metrics['rl_agent_selection_accuracy'] = Histogram(
            'xorb_rl_agent_selection_accuracy',
            'Accuracy of RL-based agent selection',
            buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics['rl_campaign_success_rate'] = Histogram(
            'xorb_rl_campaign_success_rate',
            'Success rate of RL-optimized campaigns',
            buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics['rl_reward_convergence'] = Gauge(
            'xorb_rl_reward_convergence_rate',
            'Rate of reward convergence in RL training',
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics['rl_q_value_stability'] = Gauge(
            'xorb_rl_q_value_stability',
            'Stability of Q-values in DQN',
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics['rl_exploration_ratio'] = Gauge(
            'xorb_rl_exploration_exploitation_ratio',
            'Ratio of exploration vs exploitation in RL',
            registry=self.prometheus_registry
        )
        
        # EPYC Performance Metrics
        self.prometheus_metrics['epyc_core_efficiency'] = Gauge(
            'xorb_epyc_core_utilization_efficiency',
            'EPYC core utilization efficiency',
            ['numa_node', 'ccx'],
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics['epyc_numa_locality'] = Gauge(
            'xorb_epyc_numa_memory_locality_ratio',
            'NUMA memory locality ratio for EPYC',
            ['numa_node'],
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics['epyc_l3_cache_hits'] = Gauge(
            'xorb_epyc_l3_cache_hit_ratio',
            'L3 cache hit ratio per CCX',
            ['ccx'],
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics['epyc_memory_bandwidth'] = Gauge(
            'xorb_epyc_memory_bandwidth_utilization',
            'Memory bandwidth utilization',
            ['numa_node'],
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics['epyc_context_switches'] = Counter(
            'xorb_epyc_context_switches_total',
            'Total context switches',
            registry=self.prometheus_registry
        )
        
        # Knowledge Fabric Metrics
        self.prometheus_metrics['kf_coverage_ratio'] = Gauge(
            'xorb_knowledge_fabric_coverage_ratio',
            'Knowledge graph coverage ratio',
            ['atom_type'],
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics['kf_prediction_confidence'] = Histogram(
            'xorb_knowledge_fabric_prediction_confidence',
            'Knowledge fabric prediction confidence',
            buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics['kf_hot_cache_hits'] = Counter(
            'xorb_knowledge_fabric_hot_cache_hits_total',
            'Knowledge fabric hot cache hits',
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics['kf_warm_cache_hits'] = Counter(
            'xorb_knowledge_fabric_warm_cache_hits_total',
            'Knowledge fabric warm cache hits',
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics['kf_query_response_time'] = Histogram(
            'xorb_knowledge_fabric_query_response_seconds',
            'Knowledge fabric query response time',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.prometheus_registry
        )
        
        # Campaign Metrics
        self.prometheus_metrics['campaign_duration'] = Histogram(
            'xorb_campaign_duration_seconds',
            'Campaign execution duration',
            ['campaign_type', 'priority'],
            buckets=[60, 300, 600, 1800, 3600, 7200, 14400, 28800],
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics['campaign_findings'] = Histogram(
            'xorb_campaign_findings_total',
            'Total findings per campaign',
            ['severity'],
            buckets=[0, 1, 5, 10, 25, 50, 100, 200],
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics['campaign_resource_efficiency'] = Histogram(
            'xorb_campaign_resource_efficiency_score',
            'Campaign resource efficiency score',
            buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
            registry=self.prometheus_registry
        )
        
        # System Health Metrics
        self.prometheus_metrics['system_health_score'] = Gauge(
            'xorb_system_health_score',
            'Overall system health score',
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics['system_concurrent_campaigns'] = Gauge(
            'xorb_system_concurrent_campaigns',
            'Number of concurrent campaigns',
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics['system_active_agents'] = Gauge(
            'xorb_system_active_agents',
            'Number of active agents',
            ['agent_type'],
            registry=self.prometheus_registry
        )
        
        self.prometheus_metrics['system_memory_pressure'] = Gauge(
            'xorb_system_memory_pressure',
            'System memory pressure',
            registry=self.prometheus_registry
        )
        
        self.logger.info("Prometheus metrics initialized")
    
    def _initialize_opentelemetry(self):
        """Initialize OpenTelemetry metrics"""
        if not OTEL_AVAILABLE:
            return
        
        try:
            # Create meter provider with Prometheus exporter
            metric_reader = PrometheusMetricReader()
            provider = MeterProvider(metric_readers=[metric_reader])
            otel_metrics.set_meter_provider(provider)
            
            # Get meter
            self.otel_meter = otel_metrics.get_meter("xorb.advanced_metrics", "2.0.0")
            
            # Create OpenTelemetry metrics
            self.otel_metrics['rl_performance_counter'] = self.otel_meter.create_counter(
                "xorb_rl_performance_events_total",
                description="Total RL performance events"
            )
            
            self.otel_metrics['epyc_performance_gauge'] = self.otel_meter.create_gauge(
                "xorb_epyc_performance_current",
                description="Current EPYC performance metrics"
            )
            
            self.otel_metrics['knowledge_fabric_histogram'] = self.otel_meter.create_histogram(
                "xorb_knowledge_fabric_operations_duration",
                description="Knowledge fabric operation durations"
            )
            
            self.logger.info("OpenTelemetry metrics initialized")
            
        except Exception as e:
            self.logger.warning(f"OpenTelemetry initialization failed: {e}")
    
    async def start_collection(self):
        """Start metrics collection"""
        if self._running:
            return
        
        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        self.logger.info("Started metrics collection")
    
    async def stop_collection(self):
        """Stop metrics collection"""
        if not self._running:
            return
        
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped metrics collection")
    
    async def _collection_loop(self):
        """Main collection loop"""
        while self._running:
            try:
                await self._collect_all_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_all_metrics(self):
        """Collect all metrics"""
        timestamp = datetime.utcnow()
        
        # Collect different metric types concurrently
        tasks = [
            self._collect_rl_metrics(),
            self._collect_epyc_metrics(),
            self._collect_knowledge_fabric_metrics(),
            self._collect_system_health_metrics()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Store collected metrics
        with self._lock:
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.warning(f"Metrics collection task {i} failed: {result}")
                    continue
                
                if i == 0:  # RL metrics
                    self.rl_metrics_history.append((timestamp, result))
                elif i == 1:  # EPYC metrics
                    self.epyc_metrics_history.append((timestamp, result))
                elif i == 2:  # Knowledge fabric metrics
                    self.knowledge_metrics_history.append((timestamp, result))
                elif i == 3:  # System health metrics
                    self.system_health_history.append((timestamp, result))
        
        # Update Prometheus metrics
        await self._update_prometheus_metrics(results)
    
    async def _collect_rl_metrics(self) -> RLPerformanceMetrics:
        """Collect reinforcement learning performance metrics"""
        metrics = RLPerformanceMetrics()
        
        try:
            # Mock RL metrics collection - in production, these would come from actual RL systems
            # This would integrate with the DQN agent selector and other RL components
            
            # Agent selection accuracy (from recent campaigns)
            recent_campaigns = [m for m in self.campaign_metrics.values() 
                             if m.end_time and m.end_time > datetime.utcnow() - timedelta(hours=1)]
            
            if recent_campaigns:
                success_rates = [c.success_rate for c in recent_campaigns]
                metrics.campaign_success_rate = statistics.mean(success_rates)
                metrics.agent_selection_accuracy = min(1.0, metrics.campaign_success_rate * 1.2)
            
            # Simulated RL-specific metrics (in production, from actual RL training)
            metrics.reward_convergence_rate = 0.85  # Mock convergence
            metrics.exploration_vs_exploitation_ratio = 0.15  # 15% exploration
            metrics.q_value_stability = 0.92  # High stability
            metrics.learning_rate_effectiveness = 0.78
            metrics.episode_length_avg = 150.0
            metrics.cumulative_reward = sum([c.findings_total * 10 for c in recent_campaigns])
            metrics.model_loss = 0.05  # Low loss indicates good training
            metrics.epsilon_decay_rate = 0.995
            
        except Exception as e:
            self.logger.debug(f"RL metrics collection error: {e}")
        
        return metrics
    
    async def _collect_epyc_metrics(self) -> EPYCPerformanceMetrics:
        """Collect EPYC-specific performance metrics"""
        metrics = EPYCPerformanceMetrics()
        
        try:
            # CPU utilization and efficiency
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # EPYC-specific calculations
            expected_max_efficiency = 0.85  # EPYC typically caps at 85% for thermal reasons
            metrics.core_utilization_efficiency = min(1.0, cpu_percent / (expected_max_efficiency * 100))
            
            # NUMA memory locality (mock - in production from /proc/vmstat or perf tools)
            metrics.numa_memory_locality_ratio = 0.85  # Typical for well-optimized EPYC workloads
            
            # L3 cache hit ratio (mock - in production from perf counters)
            metrics.l3_cache_hit_ratio = 0.78  # Typical L3 hit ratio
            
            # CCX load balance (mock - in production from per-CCX monitoring)
            metrics.ccx_load_balance_score = 0.92  # Well balanced across CCXs
            
            # Memory bandwidth utilization
            memory_stats = psutil.virtual_memory()
            memory_utilization = memory_stats.percent / 100.0
            # EPYC has high memory bandwidth, so utilization can be higher
            metrics.memory_bandwidth_utilization = min(1.0, memory_utilization * 1.2)
            
            # Context switches
            cpu_stats = psutil.cpu_stats()
            if hasattr(self, '_last_ctx_switches'):
                ctx_diff = cpu_stats.ctx_switches - self._last_ctx_switches
                time_diff = self.collection_interval
                metrics.context_switches_per_second = ctx_diff / time_diff
            self._last_ctx_switches = cpu_stats.ctx_switches
            
            # CPU frequency scaling efficiency
            try:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    frequency_efficiency = cpu_freq.current / cpu_freq.max if cpu_freq.max > 0 else 1.0
                    metrics.cpu_frequency_scaling_efficiency = frequency_efficiency
            except:
                metrics.cpu_frequency_scaling_efficiency = 1.0
            
            # Thermal throttling (mock - in production from thermal sensors)
            metrics.thermal_throttling_incidents = 0  # No throttling in optimal conditions
            
            # Power efficiency (mock - in production from power sensors)
            metrics.power_efficiency_score = 0.88  # EPYC is power-efficient
            
            # Hyperthreading effectiveness
            # EPYC SMT effectiveness varies by workload
            metrics.hyperthreading_effectiveness = 0.75  # 75% effectiveness is typical
            
        except Exception as e:
            self.logger.debug(f"EPYC metrics collection error: {e}")
        
        return metrics
    
    async def _collect_knowledge_fabric_metrics(self) -> KnowledgeFabricMetrics:
        """Collect knowledge fabric performance metrics"""
        metrics = KnowledgeFabricMetrics()
        
        try:
            # Mock knowledge fabric metrics - in production, from actual knowledge fabric
            
            # Knowledge graph coverage
            metrics.knowledge_graph_coverage_ratio = 0.72  # 72% coverage
            
            # Prediction confidence
            metrics.prediction_confidence_avg = 0.81  # High confidence
            
            # Atom relationship density
            metrics.atom_relationship_density = 0.35  # Good interconnectedness
            
            # Cache performance
            metrics.hot_cache_hit_ratio = 0.92  # Excellent hot cache performance
            metrics.warm_cache_hit_ratio = 0.68  # Good warm cache performance
            
            # Query performance
            metrics.query_response_time_p95 = 0.025  # 25ms P95 response time
            
            # Knowledge freshness
            metrics.knowledge_freshness_score = 0.88  # Fresh knowledge
            
            # Validation metrics
            metrics.validation_success_rate = 0.94  # High validation success
            
            # Cross-reference accuracy
            metrics.cross_reference_accuracy = 0.87  # Good cross-referencing
            
            # Semantic similarity
            metrics.semantic_similarity_score = 0.79  # Good semantic understanding
            
        except Exception as e:
            self.logger.debug(f"Knowledge fabric metrics collection error: {e}")
        
        return metrics
    
    async def _collect_system_health_metrics(self) -> SystemHealthMetrics:
        """Collect overall system health metrics"""
        metrics = SystemHealthMetrics()
        
        try:
            # Overall health score calculation
            cpu_health = 1.0 - (psutil.cpu_percent() / 100.0) if psutil.cpu_percent() < 90 else 0.5
            memory_health = 1.0 - (psutil.virtual_memory().percent / 100.0) if psutil.virtual_memory().percent < 85 else 0.3
            disk_health = 1.0 - (psutil.disk_usage('/').percent / 100.0) if psutil.disk_usage('/').percent < 80 else 0.4
            
            metrics.overall_health_score = (cpu_health + memory_health + disk_health) / 3.0
            
            # Uptime
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            uptime_hours = uptime_seconds / 3600
            # Assume 99.9% uptime target
            metrics.uptime_percentage = min(100.0, (uptime_hours / (uptime_hours + 0.1)) * 100)
            
            # Error rate (mock - in production from application logs)
            metrics.error_rate = 0.02  # 2% error rate
            
            # Response time P99 (mock)
            metrics.response_time_p99 = 0.15  # 150ms P99
            
            # Resource saturation
            cpu_sat = psutil.cpu_percent() / 100.0
            mem_sat = psutil.virtual_memory().percent / 100.0
            disk_sat = psutil.disk_usage('/').percent / 100.0
            metrics.resource_saturation_level = max(cpu_sat, mem_sat, disk_sat)
            
            # Concurrent campaigns and agents
            metrics.concurrent_campaigns = len(self.active_campaigns)
            metrics.active_agents = sum(len(agents) for agents in self.active_agents.values())
            
            # Memory pressure
            memory_info = psutil.virtual_memory()
            metrics.memory_pressure = memory_info.percent / 100.0
            
            # Disk I/O pressure
            try:
                disk_io = psutil.disk_io_counters()
                if hasattr(self, '_last_disk_io'):
                    read_bytes = disk_io.read_bytes - self._last_disk_io.read_bytes
                    write_bytes = disk_io.write_bytes - self._last_disk_io.write_bytes
                    total_bytes = read_bytes + write_bytes
                    # Normalize to MB/s and calculate pressure (mock threshold: 100 MB/s)
                    io_mbps = (total_bytes / (1024 * 1024)) / self.collection_interval
                    metrics.disk_io_pressure = min(1.0, io_mbps / 100.0)
                self._last_disk_io = disk_io
            except:
                metrics.disk_io_pressure = 0.1
            
            # Network throughput
            try:
                net_io = psutil.net_io_counters()
                if hasattr(self, '_last_net_io'):
                    bytes_sent = net_io.bytes_sent - self._last_net_io.bytes_sent
                    bytes_recv = net_io.bytes_recv - self._last_net_io.bytes_recv
                    total_bytes = bytes_sent + bytes_recv
                    # Convert to Mbps
                    throughput_mbps = (total_bytes * 8) / (1024 * 1024) / self.collection_interval
                    metrics.network_throughput = throughput_mbps
                self._last_net_io = net_io
            except:
                metrics.network_throughput = 10.0  # Default 10 Mbps
            
        except Exception as e:
            self.logger.debug(f"System health metrics collection error: {e}")
        
        return metrics
    
    async def _update_prometheus_metrics(self, collected_metrics: List[Any]):
        """Update Prometheus metrics with collected data"""
        if not PROMETHEUS_AVAILABLE or not self.prometheus_metrics:
            return
        
        try:
            rl_metrics, epyc_metrics, kf_metrics, health_metrics = collected_metrics
            
            # Update RL metrics
            if isinstance(rl_metrics, RLPerformanceMetrics):
                self.prometheus_metrics['rl_agent_selection_accuracy'].observe(rl_metrics.agent_selection_accuracy)
                self.prometheus_metrics['rl_campaign_success_rate'].observe(rl_metrics.campaign_success_rate)
                self.prometheus_metrics['rl_reward_convergence'].set(rl_metrics.reward_convergence_rate)
                self.prometheus_metrics['rl_q_value_stability'].set(rl_metrics.q_value_stability)
                self.prometheus_metrics['rl_exploration_ratio'].set(rl_metrics.exploration_vs_exploitation_ratio)
            
            # Update EPYC metrics
            if isinstance(epyc_metrics, EPYCPerformanceMetrics):
                # For NUMA nodes (assuming 2 nodes)
                for numa_node in range(2):
                    self.prometheus_metrics['epyc_core_efficiency'].labels(
                        numa_node=str(numa_node), ccx='all'
                    ).set(epyc_metrics.core_utilization_efficiency)
                    
                    self.prometheus_metrics['epyc_numa_locality'].labels(
                        numa_node=str(numa_node)
                    ).set(epyc_metrics.numa_memory_locality_ratio)
                    
                    self.prometheus_metrics['epyc_memory_bandwidth'].labels(
                        numa_node=str(numa_node)
                    ).set(epyc_metrics.memory_bandwidth_utilization)
                
                # For CCX metrics (assuming 8 CCX total)
                for ccx in range(8):
                    self.prometheus_metrics['epyc_l3_cache_hits'].labels(
                        ccx=str(ccx)
                    ).set(epyc_metrics.l3_cache_hit_ratio)
                
                self.prometheus_metrics['epyc_context_switches'].inc(
                    epyc_metrics.context_switches_per_second * self.collection_interval
                )
            
            # Update Knowledge Fabric metrics
            if isinstance(kf_metrics, KnowledgeFabricMetrics):
                atom_types = ['vulnerability', 'exploit', 'intelligence', 'target', 'tool']
                for atom_type in atom_types:
                    self.prometheus_metrics['kf_coverage_ratio'].labels(
                        atom_type=atom_type
                    ).set(kf_metrics.knowledge_graph_coverage_ratio)
                
                self.prometheus_metrics['kf_prediction_confidence'].observe(kf_metrics.prediction_confidence_avg)
                self.prometheus_metrics['kf_hot_cache_hits'].inc(100)  # Mock cache hits
                self.prometheus_metrics['kf_warm_cache_hits'].inc(50)
                self.prometheus_metrics['kf_query_response_time'].observe(kf_metrics.query_response_time_p95)
            
            # Update System Health metrics
            if isinstance(health_metrics, SystemHealthMetrics):
                self.prometheus_metrics['system_health_score'].set(health_metrics.overall_health_score)
                self.prometheus_metrics['system_concurrent_campaigns'].set(health_metrics.concurrent_campaigns)
                self.prometheus_metrics['system_memory_pressure'].set(health_metrics.memory_pressure)
                
                # Active agents by type
                agent_types = ['recon', 'web_crawler', 'vulnerability_scanner', 'nuclei']
                for agent_type in agent_types:
                    agent_count = len(self.active_agents.get(agent_type, {}))
                    self.prometheus_metrics['system_active_agents'].labels(
                        agent_type=agent_type
                    ).set(agent_count)
        
        except Exception as e:
            self.logger.debug(f"Prometheus metrics update error: {e}")
    
    async def record_campaign_start(self, campaign_id: str, campaign_data: Dict[str, Any]):
        """Record campaign start metrics"""
        with self._lock:
            self.active_campaigns.add(campaign_id)
            
            self.campaign_metrics[campaign_id] = CampaignMetrics(
                campaign_id=campaign_id,
                start_time=datetime.utcnow(),
                target_count=len(campaign_data.get('targets', [])),
                agent_count=len(campaign_data.get('agents', []))
            )
        
        self.logger.debug(f"Recorded campaign start: {campaign_id}")
    
    async def record_campaign_completion(self, campaign_id: str, results: Dict[str, Any]):
        """Record campaign completion metrics"""
        with self._lock:
            self.active_campaigns.discard(campaign_id)
            
            if campaign_id in self.campaign_metrics:
                campaign = self.campaign_metrics[campaign_id]
                campaign.end_time = datetime.utcnow()
                campaign.execution_time_seconds = (campaign.end_time - campaign.start_time).total_seconds()
                campaign.findings_total = results.get('findings_count', 0)
                campaign.findings_high_severity = results.get('high_severity_findings', 0)
                campaign.success_rate = results.get('success_rate', 0.0)
                campaign.resource_efficiency_score = results.get('resource_efficiency', 0.0)
                campaign.error_rate = results.get('error_rate', 0.0)
                campaign.retry_count = results.get('retry_count', 0)
                
                # Calculate cost per finding
                if campaign.findings_total > 0:
                    estimated_cost = campaign.execution_time_seconds * campaign.agent_count * 0.001  # Mock cost
                    campaign.cost_per_finding = estimated_cost / campaign.findings_total
                
                # Update Prometheus campaign metrics
                if PROMETHEUS_AVAILABLE and self.prometheus_metrics:
                    campaign_type = results.get('campaign_type', 'standard')
                    priority = results.get('priority', 'medium')
                    
                    self.prometheus_metrics['campaign_duration'].labels(
                        campaign_type=campaign_type, priority=priority
                    ).observe(campaign.execution_time_seconds)
                    
                    self.prometheus_metrics['campaign_findings'].labels(
                        severity='high'
                    ).observe(campaign.findings_high_severity)
                    
                    self.prometheus_metrics['campaign_findings'].labels(
                        severity='all'
                    ).observe(campaign.findings_total)
                    
                    self.prometheus_metrics['campaign_resource_efficiency'].observe(
                        campaign.resource_efficiency_score
                    )
        
        self.logger.debug(f"Recorded campaign completion: {campaign_id}")
    
    async def record_agent_activity(self, agent_id: str, agent_type: str, activity_data: Dict[str, Any]):
        """Record agent activity metrics"""
        with self._lock:
            if agent_type not in self.active_agents:
                self.active_agents[agent_type] = {}
            
            self.active_agents[agent_type][agent_id] = {
                'last_activity': datetime.utcnow(),
                'activity_count': self.active_agents[agent_type].get(agent_id, {}).get('activity_count', 0) + 1,
                'status': activity_data.get('status', 'active'),
                'performance_score': activity_data.get('performance_score', 0.5)
            }
    
    async def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        if not PROMETHEUS_AVAILABLE or not self.prometheus_registry:
            return "# Prometheus metrics not available\n"
        
        return generate_latest(self.prometheus_registry).decode('utf-8')
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        with self._lock:
            # Get latest metrics
            latest_rl = self.rl_metrics_history[-1][1] if self.rl_metrics_history else RLPerformanceMetrics()
            latest_epyc = self.epyc_metrics_history[-1][1] if self.epyc_metrics_history else EPYCPerformanceMetrics()
            latest_kf = self.knowledge_metrics_history[-1][1] if self.knowledge_metrics_history else KnowledgeFabricMetrics()
            latest_health = self.system_health_history[-1][1] if self.system_health_history else SystemHealthMetrics()
            
            # Recent campaign statistics
            recent_campaigns = [c for c in self.campaign_metrics.values() 
                             if c.end_time and c.end_time > datetime.utcnow() - timedelta(hours=24)]
            
            campaign_stats = {
                'total_campaigns_24h': len(recent_campaigns),
                'avg_success_rate': statistics.mean([c.success_rate for c in recent_campaigns]) if recent_campaigns else 0.0,
                'avg_execution_time': statistics.mean([c.execution_time_seconds for c in recent_campaigns]) if recent_campaigns else 0.0,
                'total_findings': sum([c.findings_total for c in recent_campaigns]),
                'avg_resource_efficiency': statistics.mean([c.resource_efficiency_score for c in recent_campaigns]) if recent_campaigns else 0.0
            }
            
            return {
                'collection_timestamp': datetime.utcnow().isoformat(),
                'collection_interval_seconds': self.collection_interval,
                'metrics_retention_hours': self.retention_hours,
                'rl_performance': asdict(latest_rl),
                'epyc_performance': asdict(latest_epyc),
                'knowledge_fabric': asdict(latest_kf),
                'system_health': asdict(latest_health),
                'campaign_statistics': campaign_stats,
                'active_campaigns': len(self.active_campaigns),
                'active_agents_by_type': {
                    agent_type: len(agents) 
                    for agent_type, agents in self.active_agents.items()
                },
                'data_points': {
                    'rl_metrics': len(self.rl_metrics_history),
                    'epyc_metrics': len(self.epyc_metrics_history),
                    'knowledge_metrics': len(self.knowledge_metrics_history),
                    'health_metrics': len(self.system_health_history)
                }
            }
    
    async def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends over time"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self._lock:
            # Filter metrics by time
            rl_trends = [(ts, metrics) for ts, metrics in self.rl_metrics_history if ts > cutoff_time]
            epyc_trends = [(ts, metrics) for ts, metrics in self.epyc_metrics_history if ts > cutoff_time]
            health_trends = [(ts, metrics) for ts, metrics in self.system_health_history if ts > cutoff_time]
            
            trends = {
                'time_range_hours': hours,
                'data_points': len(rl_trends),
                'rl_performance_trend': {
                    'agent_selection_accuracy': [m.agent_selection_accuracy for _, m in rl_trends],
                    'campaign_success_rate': [m.campaign_success_rate for _, m in rl_trends],
                    'reward_convergence': [m.reward_convergence_rate for _, m in rl_trends]
                },
                'epyc_performance_trend': {
                    'core_efficiency': [m.core_utilization_efficiency for _, m in epyc_trends],
                    'numa_locality': [m.numa_memory_locality_ratio for _, m in epyc_trends],
                    'memory_bandwidth': [m.memory_bandwidth_utilization for _, m in epyc_trends]
                },
                'system_health_trend': {
                    'overall_score': [m.overall_health_score for _, m in health_trends],
                    'memory_pressure': [m.memory_pressure for _, m in health_trends],
                    'concurrent_campaigns': [m.concurrent_campaigns for _, m in health_trends]
                }
            }
            
            # Calculate trend statistics
            for category, metrics in trends.items():
                if category.endswith('_trend') and metrics:
                    for metric_name, values in metrics.items():
                        if values:
                            trends[f'{category}_{metric_name}_avg'] = statistics.mean(values)
                            trends[f'{category}_{metric_name}_trend'] = 'increasing' if values[-1] > values[0] else 'decreasing'
            
            return trends


if __name__ == "__main__":
    async def main():
        # Example usage
        collector = AdvancedMetricsCollector(collection_interval=5, epyc_cores=64)
        
        # Start collection
        await collector.start_collection()
        
        # Simulate some campaign activity
        await collector.record_campaign_start("test_campaign_1", {
            'targets': [{'hostname': 'example.com'}],
            'agents': ['recon', 'web_crawler']
        })
        
        await collector.record_agent_activity("agent_1", "recon", {
            'status': 'active',
            'performance_score': 0.85
        })
        
        # Wait for some metrics collection
        await asyncio.sleep(10)
        
        # Get metrics summary
        summary = await collector.get_metrics_summary()
        print(f"Metrics Summary: {json.dumps(summary, indent=2, default=str)}")
        
        # Get Prometheus metrics
        prom_metrics = await collector.get_prometheus_metrics()
        print(f"Prometheus metrics length: {len(prom_metrics)} characters")
        
        # Complete campaign
        await collector.record_campaign_completion("test_campaign_1", {
            'findings_count': 5,
            'high_severity_findings': 2,
            'success_rate': 0.8,
            'resource_efficiency': 0.75,
            'campaign_type': 'intelligence',
            'priority': 'high'
        })
        
        # Get performance trends
        trends = await collector.get_performance_trends(1)  # 1 hour
        print(f"Performance trends: {json.dumps(trends, indent=2, default=str)}")
        
        # Stop collection
        await collector.stop_collection()
    
    asyncio.run(main())