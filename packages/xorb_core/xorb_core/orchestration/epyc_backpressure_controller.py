#!/usr/bin/env python3
"""
EPYC-Specific Backpressure Controller for Xorb 2.0

This module implements intelligent backpressure control specifically optimized
for AMD EPYC processors. It monitors system resources, NUMA topology, and
thermal characteristics to prevent resource exhaustion and maintain optimal
performance under high load conditions.
"""

import asyncio
import logging
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import threading

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("NumPy not available for statistical calculations")


class BackpressureLevel(Enum):
    """Backpressure severity levels"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ResourceType(Enum):
    """Resource types for monitoring"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    NUMA_LOCALITY = "numa_locality"
    THERMAL = "thermal"
    CACHE_PRESSURE = "cache_pressure"
    CONTEXT_SWITCHES = "context_switches"


@dataclass
class ResourceThresholds:
    """Resource thresholds for backpressure triggering"""
    # CPU thresholds (percentage)
    cpu_low: float = 70.0
    cpu_medium: float = 80.0
    cpu_high: float = 90.0
    cpu_critical: float = 95.0
    
    # Memory thresholds (percentage)
    memory_low: float = 75.0
    memory_medium: float = 85.0
    memory_high: float = 92.0
    memory_critical: float = 97.0
    
    # NUMA locality thresholds (ratio, lower is worse)
    numa_locality_low: float = 0.7
    numa_locality_medium: float = 0.6
    numa_locality_high: float = 0.5
    numa_locality_critical: float = 0.4
    
    # Context switch thresholds (per second)
    context_switches_low: float = 50000
    context_switches_medium: float = 100000
    context_switches_high: float = 200000
    context_switches_critical: float = 400000
    
    # Cache pressure thresholds (miss ratio)
    cache_pressure_low: float = 0.15
    cache_pressure_medium: float = 0.25
    cache_pressure_high: float = 0.35
    cache_pressure_critical: float = 0.50
    
    # Thermal thresholds (°C) - EPYC specific
    thermal_low: float = 65.0
    thermal_medium: float = 72.0
    thermal_high: float = 80.0
    thermal_critical: float = 85.0


@dataclass
class ResourceSnapshot:
    """Snapshot of system resource utilization"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    numa_locality_ratio: float
    context_switches_per_sec: float
    cache_miss_ratio: float
    thermal_avg_temp: float
    thermal_max_temp: float
    disk_io_mbps: float
    network_io_mbps: float
    load_average_1m: float
    load_average_5m: float
    load_average_15m: float
    active_processes: int
    zombie_processes: int


@dataclass
class BackpressureAction:
    """Action to take when backpressure is needed"""
    action_type: str  # 'throttle', 'pause', 'scale_down', 'priority_adjust'
    severity: BackpressureLevel
    resource_type: ResourceType
    target_reduction: float  # Percentage reduction needed
    duration_seconds: int
    affected_components: List[str]
    reason: str
    recovery_condition: str


@dataclass
class BackpressureState:
    """Current backpressure state"""
    level: BackpressureLevel = BackpressureLevel.NONE
    active_actions: List[BackpressureAction] = field(default_factory=list)
    triggered_at: Optional[datetime] = None
    resource_snapshots: deque = field(default_factory=lambda: deque(maxlen=100))
    performance_impact: float = 0.0  # 0.0 = no impact, 1.0 = severe impact
    recovery_progress: float = 0.0  # 0.0 = no recovery, 1.0 = fully recovered


class EPYCBackpressureController:
    """
    EPYC-optimized backpressure controller for intelligent resource management
    """
    
    def __init__(self, 
                 epyc_cores: int = 64,
                 numa_nodes: int = 2,
                 monitoring_interval: float = 2.0,
                 history_retention_minutes: int = 60):
        
        self.epyc_cores = epyc_cores
        self.numa_nodes = numa_nodes
        self.monitoring_interval = monitoring_interval
        self.history_retention_minutes = history_retention_minutes
        
        self.logger = logging.getLogger(__name__)
        
        # Resource thresholds (EPYC-optimized)
        self.thresholds = ResourceThresholds()
        
        # Current state
        self.state = BackpressureState()
        self._lock = threading.RLock()
        
        # Monitoring and prediction
        self.resource_history = deque(maxlen=int(history_retention_minutes * 60 / monitoring_interval))
        self.prediction_window = 30  # seconds
        
        # Callbacks for backpressure actions
        self.action_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Performance tracking
        self.performance_metrics = {
            'backpressure_events_count': 0,
            'total_throttle_time_seconds': 0.0,
            'prevented_overload_events': 0,
            'false_positive_rate': 0.0,
            'recovery_time_avg_seconds': 0.0,
            'epyc_specific_optimizations': 0
        }
        
        # EPYC-specific configuration
        self.epyc_config = {
            'ccx_count': 8,  # 8 CCX (Core Complex) for EPYC 7702
            'cores_per_ccx': 4,
            'l3_cache_per_ccx_mb': 32,
            'memory_channels_per_numa': 8,
            'typical_thermal_limit': 90.0,  # EPYC thermal limit
            'boost_clock_mhz': 3350,  # Max boost for EPYC 7702
            'base_clock_mhz': 2000   # Base clock for EPYC 7702
        }
        
        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Initialize baseline metrics
        self._baseline_metrics: Optional[ResourceSnapshot] = None
        self._establishing_baseline = True
        self._baseline_samples = deque(maxlen=30)  # 1 minute of samples
        
        self.logger.info(f"EPYC Backpressure Controller initialized for {epyc_cores} cores, {numa_nodes} NUMA nodes")
    
    async def start_monitoring(self):
        """Start resource monitoring and backpressure control"""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Started EPYC backpressure monitoring")
    
    async def stop_monitoring(self):
        """Stop resource monitoring"""
        if not self._running:
            return
        
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped EPYC backpressure monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                # Collect resource metrics
                snapshot = await self._collect_resource_snapshot()
                
                with self._lock:
                    self.resource_history.append(snapshot)
                    self.state.resource_snapshots.append(snapshot)
                
                # Establish baseline if needed
                if self._establishing_baseline:
                    await self._update_baseline(snapshot)
                else:
                    # Analyze for backpressure needs
                    await self._analyze_backpressure_needs(snapshot)
                
                # Update recovery progress if under backpressure
                if self.state.level != BackpressureLevel.NONE:
                    await self._update_recovery_progress(snapshot)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in backpressure monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval * 2)  # Back off on error
    
    async def _collect_resource_snapshot(self) -> ResourceSnapshot:
        """Collect comprehensive resource metrics"""
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)
        
        # Load average
        try:
            load_avg = psutil.getloadavg()
            load_1m, load_5m, load_15m = load_avg
        except AttributeError:
            # Windows doesn't have getloadavg
            load_1m = load_5m = load_15m = cpu_percent / 100.0 * self.epyc_cores
        
        # Process metrics
        process_count = len(psutil.pids())
        zombie_count = 0
        try:
            for proc in psutil.process_iter(['status']):
                if proc.info['status'] == psutil.STATUS_ZOMBIE:
                    zombie_count += 1
        except:
            pass
        
        # Context switches
        cpu_stats = psutil.cpu_stats()
        context_switches_per_sec = 0.0
        if hasattr(self, '_last_ctx_switches'):
            ctx_diff = cpu_stats.ctx_switches - self._last_ctx_switches
            context_switches_per_sec = ctx_diff / self.monitoring_interval
        self._last_ctx_switches = cpu_stats.ctx_switches
        
        # Disk I/O
        disk_io_mbps = 0.0
        try:
            disk_io = psutil.disk_io_counters()
            if hasattr(self, '_last_disk_io'):
                read_bytes = disk_io.read_bytes - self._last_disk_io.read_bytes
                write_bytes = disk_io.write_bytes - self._last_disk_io.write_bytes
                total_bytes = read_bytes + write_bytes
                disk_io_mbps = (total_bytes / (1024**2)) / self.monitoring_interval
            self._last_disk_io = disk_io
        except:
            pass
        
        # Network I/O
        network_io_mbps = 0.0
        try:
            net_io = psutil.net_io_counters()
            if hasattr(self, '_last_net_io'):
                sent_bytes = net_io.bytes_sent - self._last_net_io.bytes_sent
                recv_bytes = net_io.bytes_recv - self._last_net_io.bytes_recv
                total_bytes = sent_bytes + recv_bytes
                network_io_mbps = (total_bytes / (1024**2)) / self.monitoring_interval
            self._last_net_io = net_io
        except:
            pass
        
        # NUMA locality estimation (mock - in production would use perf tools)
        numa_locality_ratio = await self._estimate_numa_locality(cpu_percent, memory_percent)
        
        # Cache miss ratio estimation (mock - in production would use perf counters)
        cache_miss_ratio = await self._estimate_cache_miss_ratio(load_1m, context_switches_per_sec)
        
        # Thermal metrics (mock - in production would read from sensors)
        thermal_avg, thermal_max = await self._get_thermal_metrics(cpu_percent)
        
        return ResourceSnapshot(
            timestamp=datetime.utcnow(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            numa_locality_ratio=numa_locality_ratio,
            context_switches_per_sec=context_switches_per_sec,
            cache_miss_ratio=cache_miss_ratio,
            thermal_avg_temp=thermal_avg,
            thermal_max_temp=thermal_max,
            disk_io_mbps=disk_io_mbps,
            network_io_mbps=network_io_mbps,
            load_average_1m=load_1m,
            load_average_5m=load_5m,
            load_average_15m=load_15m,
            active_processes=process_count,
            zombie_processes=zombie_count
        )
    
    async def _estimate_numa_locality(self, cpu_percent: float, memory_percent: float) -> float:
        """Estimate NUMA memory locality ratio"""
        # Mock estimation based on load patterns
        # In production, this would use perf tools or /proc/vmstat
        
        # High CPU + high memory usually indicates good locality
        if cpu_percent > 70 and memory_percent > 60:
            base_locality = 0.85
        elif cpu_percent > 50 and memory_percent > 40:
            base_locality = 0.75
        else:
            base_locality = 0.65
        
        # Add some realistic variation
        import random
        variation = random.uniform(-0.1, 0.1)
        return max(0.3, min(0.95, base_locality + variation))
    
    async def _estimate_cache_miss_ratio(self, load_avg: float, context_switches: float) -> float:
        """Estimate cache miss ratio"""
        # Mock estimation based on load and context switches
        # High context switches usually indicate cache thrashing
        
        load_factor = min(1.0, load_avg / self.epyc_cores)
        ctx_switch_factor = min(1.0, context_switches / 100000)  # Normalize to 100k switches/sec
        
        # Base miss ratio increases with load and context switches
        base_miss_ratio = 0.05  # 5% baseline
        load_penalty = load_factor * 0.15
        ctx_penalty = ctx_switch_factor * 0.25
        
        return min(0.6, base_miss_ratio + load_penalty + ctx_penalty)
    
    async def _get_thermal_metrics(self, cpu_percent: float) -> Tuple[float, float]:
        """Get thermal metrics (mock implementation)"""
        # Mock thermal data based on CPU usage
        # In production, would read from /sys/class/hwmon/ or lm-sensors
        
        base_temp = 45.0  # Base temperature
        load_temp_increase = (cpu_percent / 100.0) * 35.0  # Up to 35°C increase under load
        
        avg_temp = base_temp + load_temp_increase * 0.8
        max_temp = base_temp + load_temp_increase
        
        # Add some realistic variation
        import random
        avg_temp += random.uniform(-2.0, 2.0)
        max_temp += random.uniform(-1.0, 3.0)
        
        return avg_temp, max_temp
    
    async def _update_baseline(self, snapshot: ResourceSnapshot):
        """Update baseline metrics during initialization"""
        self._baseline_samples.append(snapshot)
        
        if len(self._baseline_samples) >= 30:  # 1 minute of samples
            # Calculate baseline from samples
            if NUMPY_AVAILABLE:
                cpu_values = [s.cpu_percent for s in self._baseline_samples]
                memory_values = [s.memory_percent for s in self._baseline_samples]
                
                baseline_cpu = np.percentile(cpu_values, 50)  # Median
                baseline_memory = np.percentile(memory_values, 50)
                
                self._baseline_metrics = ResourceSnapshot(
                    timestamp=datetime.utcnow(),
                    cpu_percent=baseline_cpu,
                    memory_percent=baseline_memory,
                    memory_available_gb=np.mean([s.memory_available_gb for s in self._baseline_samples]),
                    numa_locality_ratio=np.mean([s.numa_locality_ratio for s in self._baseline_samples]),
                    context_switches_per_sec=np.mean([s.context_switches_per_sec for s in self._baseline_samples]),
                    cache_miss_ratio=np.mean([s.cache_miss_ratio for s in self._baseline_samples]),
                    thermal_avg_temp=np.mean([s.thermal_avg_temp for s in self._baseline_samples]),
                    thermal_max_temp=np.mean([s.thermal_max_temp for s in self._baseline_samples]),
                    disk_io_mbps=np.mean([s.disk_io_mbps for s in self._baseline_samples]),
                    network_io_mbps=np.mean([s.network_io_mbps for s in self._baseline_samples]),
                    load_average_1m=np.mean([s.load_average_1m for s in self._baseline_samples]),
                    load_average_5m=np.mean([s.load_average_5m for s in self._baseline_samples]),
                    load_average_15m=np.mean([s.load_average_15m for s in self._baseline_samples]),
                    active_processes=int(np.mean([s.active_processes for s in self._baseline_samples])),
                    zombie_processes=int(np.mean([s.zombie_processes for s in self._baseline_samples]))
                )
            else:
                # Fallback without NumPy
                self._baseline_metrics = self._baseline_samples[-1]
            
            self._establishing_baseline = False
            self.logger.info(f"Established baseline: CPU {self._baseline_metrics.cpu_percent:.1f}%, "
                           f"Memory {self._baseline_metrics.memory_percent:.1f}%")
    
    async def _analyze_backpressure_needs(self, snapshot: ResourceSnapshot):
        """Analyze current metrics and determine if backpressure is needed"""
        
        current_level = self._calculate_backpressure_level(snapshot)
        
        with self._lock:
            previous_level = self.state.level
            
            # Check if backpressure level changed
            if current_level != previous_level:
                await self._handle_backpressure_change(previous_level, current_level, snapshot)
            
            # Update state
            self.state.level = current_level
            
            # Trigger actions if needed
            if current_level != BackpressureLevel.NONE:
                await self._trigger_backpressure_actions(current_level, snapshot)
    
    def _calculate_backpressure_level(self, snapshot: ResourceSnapshot) -> BackpressureLevel:
        """Calculate appropriate backpressure level based on metrics"""
        
        # Collect violation scores for each resource type
        violations = {}
        
        # CPU violations
        if snapshot.cpu_percent >= self.thresholds.cpu_critical:
            violations[ResourceType.CPU] = BackpressureLevel.CRITICAL
        elif snapshot.cpu_percent >= self.thresholds.cpu_high:
            violations[ResourceType.CPU] = BackpressureLevel.HIGH
        elif snapshot.cpu_percent >= self.thresholds.cpu_medium:
            violations[ResourceType.CPU] = BackpressureLevel.MEDIUM
        elif snapshot.cpu_percent >= self.thresholds.cpu_low:
            violations[ResourceType.CPU] = BackpressureLevel.LOW
        
        # Memory violations
        if snapshot.memory_percent >= self.thresholds.memory_critical:
            violations[ResourceType.MEMORY] = BackpressureLevel.CRITICAL
        elif snapshot.memory_percent >= self.thresholds.memory_high:
            violations[ResourceType.MEMORY] = BackpressureLevel.HIGH
        elif snapshot.memory_percent >= self.thresholds.memory_medium:
            violations[ResourceType.MEMORY] = BackpressureLevel.MEDIUM
        elif snapshot.memory_percent >= self.thresholds.memory_low:
            violations[ResourceType.MEMORY] = BackpressureLevel.LOW
        
        # NUMA locality violations (inverted - lower is worse)
        if snapshot.numa_locality_ratio <= self.thresholds.numa_locality_critical:
            violations[ResourceType.NUMA_LOCALITY] = BackpressureLevel.CRITICAL
        elif snapshot.numa_locality_ratio <= self.thresholds.numa_locality_high:
            violations[ResourceType.NUMA_LOCALITY] = BackpressureLevel.HIGH
        elif snapshot.numa_locality_ratio <= self.thresholds.numa_locality_medium:
            violations[ResourceType.NUMA_LOCALITY] = BackpressureLevel.MEDIUM
        elif snapshot.numa_locality_ratio <= self.thresholds.numa_locality_low:
            violations[ResourceType.NUMA_LOCALITY] = BackpressureLevel.LOW
        
        # Context switch violations
        if snapshot.context_switches_per_sec >= self.thresholds.context_switches_critical:
            violations[ResourceType.CONTEXT_SWITCHES] = BackpressureLevel.CRITICAL
        elif snapshot.context_switches_per_sec >= self.thresholds.context_switches_high:
            violations[ResourceType.CONTEXT_SWITCHES] = BackpressureLevel.HIGH
        elif snapshot.context_switches_per_sec >= self.thresholds.context_switches_medium:
            violations[ResourceType.CONTEXT_SWITCHES] = BackpressureLevel.MEDIUM
        elif snapshot.context_switches_per_sec >= self.thresholds.context_switches_low:
            violations[ResourceType.CONTEXT_SWITCHES] = BackpressureLevel.LOW
        
        # Cache pressure violations
        if snapshot.cache_miss_ratio >= self.thresholds.cache_pressure_critical:
            violations[ResourceType.CACHE_PRESSURE] = BackpressureLevel.CRITICAL
        elif snapshot.cache_miss_ratio >= self.thresholds.cache_pressure_high:
            violations[ResourceType.CACHE_PRESSURE] = BackpressureLevel.HIGH
        elif snapshot.cache_miss_ratio >= self.thresholds.cache_pressure_medium:
            violations[ResourceType.CACHE_PRESSURE] = BackpressureLevel.MEDIUM
        elif snapshot.cache_miss_ratio >= self.thresholds.cache_pressure_low:
            violations[ResourceType.CACHE_PRESSURE] = BackpressureLevel.LOW
        
        # Thermal violations (EPYC-specific)
        if snapshot.thermal_max_temp >= self.thresholds.thermal_critical:
            violations[ResourceType.THERMAL] = BackpressureLevel.CRITICAL
        elif snapshot.thermal_max_temp >= self.thresholds.thermal_high:
            violations[ResourceType.THERMAL] = BackpressureLevel.HIGH
        elif snapshot.thermal_max_temp >= self.thresholds.thermal_medium:
            violations[ResourceType.THERMAL] = BackpressureLevel.MEDIUM
        elif snapshot.thermal_max_temp >= self.thresholds.thermal_low:
            violations[ResourceType.THERMAL] = BackpressureLevel.LOW
        
        # Return the highest severity violation
        if not violations:
            return BackpressureLevel.NONE
        
        severity_order = [
            BackpressureLevel.CRITICAL,
            BackpressureLevel.HIGH,
            BackpressureLevel.MEDIUM,
            BackpressureLevel.LOW
        ]
        
        for level in severity_order:
            if level in violations.values():
                return level
        
        return BackpressureLevel.NONE
    
    async def _handle_backpressure_change(self, 
                                        previous_level: BackpressureLevel,
                                        current_level: BackpressureLevel,
                                        snapshot: ResourceSnapshot):
        """Handle backpressure level changes"""
        
        if current_level == BackpressureLevel.NONE and previous_level != BackpressureLevel.NONE:
            # Recovery from backpressure
            self.state.triggered_at = None
            self.state.recovery_progress = 1.0
            self.performance_metrics['recovery_time_avg_seconds'] = (
                (datetime.utcnow() - self.state.triggered_at).total_seconds()
                if self.state.triggered_at else 0.0
            )
            
            self.logger.info(f"Recovered from backpressure level {previous_level.value}")
            await self._notify_callbacks('recovery', {
                'previous_level': previous_level,
                'recovery_metrics': snapshot
            })
            
        elif current_level != BackpressureLevel.NONE and previous_level == BackpressureLevel.NONE:
            # New backpressure detected
            self.state.triggered_at = datetime.utcnow()
            self.state.recovery_progress = 0.0
            self.performance_metrics['backpressure_events_count'] += 1
            
            self.logger.warning(f"Backpressure triggered at level {current_level.value}")
            await self._notify_callbacks('backpressure_start', {
                'level': current_level,
                'triggering_metrics': snapshot
            })
            
        elif current_level != previous_level:
            # Backpressure level changed
            self.logger.info(f"Backpressure level changed from {previous_level.value} to {current_level.value}")
            await self._notify_callbacks('level_change', {
                'previous_level': previous_level,
                'current_level': current_level,
                'metrics': snapshot
            })
    
    async def _trigger_backpressure_actions(self, level: BackpressureLevel, snapshot: ResourceSnapshot):
        """Trigger appropriate backpressure actions"""
        
        actions = []
        
        # Generate actions based on backpressure level and resource constraints
        if level == BackpressureLevel.LOW:
            actions.extend(await self._generate_low_pressure_actions(snapshot))
        elif level == BackpressureLevel.MEDIUM:
            actions.extend(await self._generate_medium_pressure_actions(snapshot))
        elif level == BackpressureLevel.HIGH:
            actions.extend(await self._generate_high_pressure_actions(snapshot))
        elif level == BackpressureLevel.CRITICAL:
            actions.extend(await self._generate_critical_pressure_actions(snapshot))
        
        # Execute actions
        for action in actions:
            await self._execute_backpressure_action(action)
        
        with self._lock:
            self.state.active_actions = actions
    
    async def _generate_low_pressure_actions(self, snapshot: ResourceSnapshot) -> List[BackpressureAction]:
        """Generate actions for low backpressure"""
        actions = []
        
        # Gentle throttling for high CPU
        if snapshot.cpu_percent >= self.thresholds.cpu_low:
            actions.append(BackpressureAction(
                action_type='throttle',
                severity=BackpressureLevel.LOW,
                resource_type=ResourceType.CPU,
                target_reduction=0.1,  # 10% reduction
                duration_seconds=30,
                affected_components=['new_campaigns'],
                reason=f'CPU utilization at {snapshot.cpu_percent:.1f}%',
                recovery_condition='CPU < 65%'
            ))
        
        # Memory pressure warning
        if snapshot.memory_percent >= self.thresholds.memory_low:
            actions.append(BackpressureAction(
                action_type='priority_adjust',
                severity=BackpressureLevel.LOW,
                resource_type=ResourceType.MEMORY,
                target_reduction=0.05,  # 5% reduction
                duration_seconds=60,
                affected_components=['low_priority_agents'],
                reason=f'Memory utilization at {snapshot.memory_percent:.1f}%',
                recovery_condition='Memory < 70%'
            ))
        
        return actions
    
    async def _generate_medium_pressure_actions(self, snapshot: ResourceSnapshot) -> List[BackpressureAction]:
        """Generate actions for medium backpressure"""
        actions = []
        
        # More aggressive throttling
        if snapshot.cpu_percent >= self.thresholds.cpu_medium:
            actions.append(BackpressureAction(
                action_type='throttle',
                severity=BackpressureLevel.MEDIUM,
                resource_type=ResourceType.CPU,
                target_reduction=0.2,  # 20% reduction
                duration_seconds=60,
                affected_components=['new_campaigns', 'non_critical_agents'],
                reason=f'CPU utilization at {snapshot.cpu_percent:.1f}%',
                recovery_condition='CPU < 75%'
            ))
        
        # Memory conservation
        if snapshot.memory_percent >= self.thresholds.memory_medium:
            actions.append(BackpressureAction(
                action_type='scale_down',
                severity=BackpressureLevel.MEDIUM,
                resource_type=ResourceType.MEMORY,
                target_reduction=0.15,  # 15% reduction
                duration_seconds=120,
                affected_components=['memory_intensive_agents'],
                reason=f'Memory utilization at {snapshot.memory_percent:.1f}%',
                recovery_condition='Memory < 80%'
            ))
        
        # NUMA locality improvement
        if snapshot.numa_locality_ratio <= self.thresholds.numa_locality_medium:
            actions.append(BackpressureAction(
                action_type='numa_rebalance',
                severity=BackpressureLevel.MEDIUM,
                resource_type=ResourceType.NUMA_LOCALITY,
                target_reduction=0.1,
                duration_seconds=90,
                affected_components=['distributed_agents'],
                reason=f'NUMA locality at {snapshot.numa_locality_ratio:.2f}',
                recovery_condition='NUMA locality > 0.65'
            ))
        
        return actions
    
    async def _generate_high_pressure_actions(self, snapshot: ResourceSnapshot) -> List[BackpressureAction]:
        """Generate actions for high backpressure"""
        actions = []
        
        # Significant throttling
        actions.append(BackpressureAction(
            action_type='throttle',
            severity=BackpressureLevel.HIGH,
            resource_type=ResourceType.CPU,
            target_reduction=0.4,  # 40% reduction
            duration_seconds=120,
            affected_components=['all_new_campaigns', 'background_tasks'],
            reason=f'High resource pressure detected',
            recovery_condition='System load normalized'
        ))
        
        # Pause non-essential operations
        actions.append(BackpressureAction(
            action_type='pause',
            severity=BackpressureLevel.HIGH,
            resource_type=ResourceType.MEMORY,
            target_reduction=0.3,  # 30% reduction
            duration_seconds=180,
            affected_components=['analytics_agents', 'reporting_tasks'],
            reason='High memory pressure',
            recovery_condition='Memory < 85%'
        ))
        
        return actions
    
    async def _generate_critical_pressure_actions(self, snapshot: ResourceSnapshot) -> List[BackpressureAction]:
        """Generate actions for critical backpressure"""
        actions = []
        
        # Emergency throttling
        actions.append(BackpressureAction(
            action_type='emergency_throttle',
            severity=BackpressureLevel.CRITICAL,
            resource_type=ResourceType.CPU,
            target_reduction=0.7,  # 70% reduction
            duration_seconds=300,
            affected_components=['all_campaigns', 'all_agents'],
            reason='Critical system overload',
            recovery_condition='All metrics below high thresholds'
        ))
        
        # Force garbage collection and cleanup
        actions.append(BackpressureAction(
            action_type='cleanup',
            severity=BackpressureLevel.CRITICAL,
            resource_type=ResourceType.MEMORY,
            target_reduction=0.5,  # 50% reduction
            duration_seconds=60,
            affected_components=['memory_caches', 'temporary_data'],
            reason='Critical memory pressure',
            recovery_condition='Memory < 90%'
        ))
        
        return actions
    
    async def _execute_backpressure_action(self, action: BackpressureAction):
        """Execute a backpressure action"""
        
        self.logger.info(f"Executing backpressure action: {action.action_type} "
                        f"(severity: {action.severity.value}, "
                        f"target reduction: {action.target_reduction:.1%})")
        
        # Track throttling time
        if 'throttle' in action.action_type:
            self.performance_metrics['total_throttle_time_seconds'] += action.duration_seconds
        
        # Notify callbacks
        await self._notify_callbacks(action.action_type, {
            'action': action,
            'timestamp': datetime.utcnow()
        })
    
    async def _update_recovery_progress(self, snapshot: ResourceSnapshot):
        """Update recovery progress during backpressure"""
        
        if not self.state.triggered_at:
            return
        
        # Calculate progress based on metric improvements
        progress_factors = []
        
        # CPU recovery
        if snapshot.cpu_percent < self.thresholds.cpu_medium:
            cpu_progress = 1.0 - (snapshot.cpu_percent / self.thresholds.cpu_medium)
            progress_factors.append(cpu_progress)
        
        # Memory recovery
        if snapshot.memory_percent < self.thresholds.memory_medium:
            memory_progress = 1.0 - (snapshot.memory_percent / self.thresholds.memory_medium)
            progress_factors.append(memory_progress)
        
        # NUMA locality recovery
        if snapshot.numa_locality_ratio > self.thresholds.numa_locality_medium:
            numa_progress = (snapshot.numa_locality_ratio - self.thresholds.numa_locality_medium) / (1.0 - self.thresholds.numa_locality_medium)
            progress_factors.append(numa_progress)
        
        # Calculate overall progress
        if progress_factors:
            self.state.recovery_progress = sum(progress_factors) / len(progress_factors)
        else:
            self.state.recovery_progress = 0.0
        
        self.state.recovery_progress = max(0.0, min(1.0, self.state.recovery_progress))
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for backpressure events"""
        self.action_callbacks[event_type].append(callback)
        self.logger.debug(f"Registered callback for event type: {event_type}")
    
    async def _notify_callbacks(self, event_type: str, data: Dict[str, Any]):
        """Notify registered callbacks"""
        callbacks = self.action_callbacks.get(event_type, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                self.logger.error(f"Callback error for {event_type}: {e}")
    
    async def get_current_state(self) -> Dict[str, Any]:
        """Get current backpressure state"""
        with self._lock:
            latest_snapshot = (
                self.resource_history[-1] if self.resource_history else None
            )
            
            return {
                'backpressure_level': self.state.level.value,
                'recovery_progress': self.state.recovery_progress,
                'active_actions_count': len(self.state.active_actions),
                'active_actions': [
                    {
                        'type': action.action_type,
                        'severity': action.severity.value,
                        'resource': action.resource_type.value,
                        'target_reduction': action.target_reduction,
                        'reason': action.reason
                    }
                    for action in self.state.active_actions
                ],
                'latest_metrics': {
                    'timestamp': latest_snapshot.timestamp.isoformat() if latest_snapshot else None,
                    'cpu_percent': latest_snapshot.cpu_percent if latest_snapshot else 0.0,
                    'memory_percent': latest_snapshot.memory_percent if latest_snapshot else 0.0,
                    'numa_locality_ratio': latest_snapshot.numa_locality_ratio if latest_snapshot else 0.0,
                    'thermal_max_temp': latest_snapshot.thermal_max_temp if latest_snapshot else 0.0,
                    'context_switches_per_sec': latest_snapshot.context_switches_per_sec if latest_snapshot else 0.0
                } if latest_snapshot else {},
                'performance_metrics': self.performance_metrics.copy(),
                'epyc_config': self.epyc_config.copy(),
                'baseline_established': not self._establishing_baseline,
                'monitoring_active': self._running
            }
    
    async def get_resource_trends(self, minutes: int = 10) -> Dict[str, Any]:
        """Get resource utilization trends"""
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        with self._lock:
            recent_snapshots = [
                snapshot for snapshot in self.resource_history
                if snapshot.timestamp > cutoff_time
            ]
        
        if len(recent_snapshots) < 2:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Calculate trends
        cpu_values = [s.cpu_percent for s in recent_snapshots]
        memory_values = [s.memory_percent for s in recent_snapshots]
        numa_values = [s.numa_locality_ratio for s in recent_snapshots]
        thermal_values = [s.thermal_max_temp for s in recent_snapshots]
        
        trends = {
            'time_window_minutes': minutes,
            'data_points': len(recent_snapshots),
            'cpu_trend': {
                'current': cpu_values[-1],
                'average': sum(cpu_values) / len(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values),
                'direction': 'increasing' if cpu_values[-1] > cpu_values[0] else 'decreasing'
            },
            'memory_trend': {
                'current': memory_values[-1],
                'average': sum(memory_values) / len(memory_values),
                'min': min(memory_values),
                'max': max(memory_values),
                'direction': 'increasing' if memory_values[-1] > memory_values[0] else 'decreasing'
            },
            'numa_trend': {
                'current': numa_values[-1],
                'average': sum(numa_values) / len(numa_values),
                'min': min(numa_values),
                'max': max(numa_values),
                'direction': 'improving' if numa_values[-1] > numa_values[0] else 'degrading'
            },
            'thermal_trend': {
                'current': thermal_values[-1],
                'average': sum(thermal_values) / len(thermal_values),
                'min': min(thermal_values),
                'max': max(thermal_values),
                'direction': 'increasing' if thermal_values[-1] > thermal_values[0] else 'decreasing'
            }
        }
        
        return trends
    
    async def simulate_load_test(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Simulate load conditions for testing backpressure response"""
        
        self.logger.info(f"Starting load simulation for {duration_seconds} seconds")
        
        start_time = datetime.utcnow()
        initial_state = await self.get_current_state()
        
        # Mock high load conditions
        original_thresholds = self.thresholds
        test_thresholds = ResourceThresholds(
            cpu_low=30.0,  # Lower thresholds to trigger backpressure easily
            cpu_medium=40.0,
            cpu_high=50.0,
            cpu_critical=60.0,
            memory_low=40.0,
            memory_medium=50.0,
            memory_high=60.0,
            memory_critical=70.0
        )
        
        self.thresholds = test_thresholds
        
        # Wait for test duration
        await asyncio.sleep(duration_seconds)
        
        # Restore original thresholds
        self.thresholds = original_thresholds
        
        end_time = datetime.utcnow()
        final_state = await self.get_current_state()
        
        test_results = {
            'test_duration_seconds': duration_seconds,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'initial_state': initial_state,
            'final_state': final_state,
            'backpressure_triggered': final_state['backpressure_level'] != 'none',
            'actions_taken': len(final_state['active_actions']),
            'performance_impact': final_state.get('recovery_progress', 0.0)
        }
        
        self.logger.info(f"Load simulation completed. Backpressure triggered: {test_results['backpressure_triggered']}")
        return test_results


# Integration example
class BackpressureIntegratedOrchestrator:
    """Example integration with orchestrator"""
    
    def __init__(self, base_orchestrator):
        self.base_orchestrator = base_orchestrator
        self.backpressure_controller = EPYCBackpressureController()
        
        # Register callbacks
        self.backpressure_controller.register_callback('throttle', self._throttle_campaigns)
        self.backpressure_controller.register_callback('pause', self._pause_operations)
        self.backpressure_controller.register_callback('emergency_throttle', self._emergency_stop)
    
    async def _throttle_campaigns(self, data: Dict[str, Any]):
        """Handle campaign throttling"""
        action = data['action']
        reduction = action.target_reduction
        
        # Reduce campaign creation rate
        # Implementation would adjust orchestrator parameters
        print(f"Throttling campaigns by {reduction:.1%}")
    
    async def _pause_operations(self, data: Dict[str, Any]):
        """Handle operation pausing"""
        action = data['action']
        components = action.affected_components
        
        # Pause specified components
        print(f"Pausing operations: {', '.join(components)}")
    
    async def _emergency_stop(self, data: Dict[str, Any]):
        """Handle emergency throttling"""
        action = data['action']
        
        # Emergency stop all non-critical operations
        print(f"EMERGENCY: Stopping {action.target_reduction:.1%} of operations")


if __name__ == "__main__":
    async def main():
        # Example usage
        controller = EPYCBackpressureController(epyc_cores=64, numa_nodes=2)
        
        # Start monitoring
        await controller.start_monitoring()
        
        # Wait for baseline establishment
        await asyncio.sleep(65)  # Just over 1 minute for baseline
        
        # Get current state
        state = await controller.get_current_state()
        print(f"Current state: {state}")
        
        # Run load test
        test_results = await controller.simulate_load_test(30)
        print(f"Load test results: {test_results}")
        
        # Get trends
        trends = await controller.get_resource_trends(5)
        print(f"Resource trends: {trends}")
        
        # Stop monitoring
        await controller.stop_monitoring()
        
        print("Backpressure controller demo completed")
    
    asyncio.run(main())