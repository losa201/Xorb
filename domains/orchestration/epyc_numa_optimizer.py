#!/usr/bin/env python3
"""
EPYC NUMA-Aware Memory Management and Process Optimization

This module provides NUMA-aware optimization for AMD EPYC processors,
specifically designed for the EPYC 7702 (64 cores, 128 threads) architecture.
Optimizes memory locality, CPU affinity, and resource allocation.
"""

import asyncio
import logging
import os
import psutil
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import json

try:
    import numa
    NUMA_AVAILABLE = True
except ImportError:
    NUMA_AVAILABLE = False
    logging.warning("python-numa not available. Install with: pip install python-numa")


@dataclass
class EPYCTopology:
    """EPYC processor topology information"""
    total_cores: int = 64
    total_threads: int = 128
    numa_nodes: int = 2
    ccx_per_numa: int = 4  # Core Complex (CCX) per NUMA node
    cores_per_ccx: int = 8  # 4 cores per CCX, with SMT = 8 threads
    l3_cache_per_ccx: int = 32 * 1024 * 1024  # 32MB L3 cache per CCX
    memory_channels_per_numa: int = 8  # 8 memory channels per NUMA node
    
    @property
    def threads_per_numa(self) -> int:
        return self.total_threads // self.numa_nodes
    
    @property
    def cores_per_numa(self) -> int:
        return self.total_cores // self.numa_nodes


@dataclass
class NUMAAffinityMapping:
    """NUMA affinity mapping for processes"""
    numa_node: int
    cpu_list: List[int]
    memory_policy: str  # 'bind', 'interleave', 'preferred'
    expected_locality_ratio: float  # Expected memory locality (0.0-1.0)


@dataclass
class ProcessResourceAllocation:
    """Resource allocation for a specific process"""
    process_id: str
    numa_affinity: NUMAAffinityMapping
    memory_limit: int  # Bytes
    cpu_shares: int
    io_priority: int
    process_type: str  # 'orchestrator', 'agent', 'worker', 'ml_training'


class EPYCNUMAOptimizer:
    """
    NUMA-aware optimizer for EPYC processors
    Manages CPU affinity, memory allocation, and process placement
    """
    
    def __init__(self, epyc_topology: Optional[EPYCTopology] = None):
        self.topology = epyc_topology or EPYCTopology()
        self.logger = logging.getLogger(__name__)
        
        # NUMA availability check
        self.numa_available = NUMA_AVAILABLE and self._check_numa_support()
        
        # Process tracking
        self.process_allocations: Dict[str, ProcessResourceAllocation] = {}
        self.numa_utilization: Dict[int, float] = {i: 0.0 for i in range(self.topology.numa_nodes)}
        
        # Performance monitoring
        self.performance_metrics = {
            'memory_locality_ratio': 0.0,
            'cross_numa_traffic': 0.0,
            'cache_hit_ratio': 0.0,
            'context_switches_per_second': 0.0
        }
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        if self.numa_available:
            self._initialize_numa_topology()
        else:
            self.logger.warning("NUMA optimization disabled - running in compatibility mode")
    
    def _check_numa_support(self) -> bool:
        """Check if NUMA is supported and available"""
        try:
            if not NUMA_AVAILABLE:
                return False
            
            # Check if NUMA nodes exist
            numa_nodes = numa.get_max_node() + 1
            if numa_nodes < 2:
                self.logger.info("Single NUMA node detected, NUMA optimization not needed")
                return False
            
            # Verify NUMA functionality
            node_list = list(range(numa_nodes))
            cpu_list = numa.node_to_cpus(0)
            
            return len(node_list) >= 2 and len(cpu_list) > 0
            
        except Exception as e:
            self.logger.warning(f"NUMA support check failed: {e}")
            return False
    
    def _initialize_numa_topology(self):
        """Initialize NUMA topology mapping"""
        try:
            # Detect actual NUMA topology
            detected_nodes = numa.get_max_node() + 1
            self.topology.numa_nodes = min(detected_nodes, self.topology.numa_nodes)
            
            # Map CPU cores to NUMA nodes
            self.numa_cpu_mapping = {}
            for node in range(self.topology.numa_nodes):
                try:
                    cpus = numa.node_to_cpus(node)
                    self.numa_cpu_mapping[node] = list(cpus)
                    self.logger.info(f"NUMA node {node}: CPUs {cpus}")
                except Exception as e:
                    self.logger.warning(f"Failed to get CPUs for NUMA node {node}: {e}")
                    # Fallback: distribute CPUs evenly
                    start_cpu = node * (self.topology.total_threads // self.topology.numa_nodes)
                    end_cpu = start_cpu + (self.topology.total_threads // self.topology.numa_nodes)
                    self.numa_cpu_mapping[node] = list(range(start_cpu, end_cpu))
            
            self.logger.info(f"NUMA topology initialized: {self.topology.numa_nodes} nodes")
            
        except Exception as e:
            self.logger.error(f"NUMA topology initialization failed: {e}")
            self.numa_available = False
    
    async def allocate_process_resources(self, 
                                       process_id: str,
                                       process_type: str,
                                       cpu_requirement: float,
                                       memory_requirement: int,
                                       locality_preference: str = 'auto') -> ProcessResourceAllocation:
        """
        Allocate NUMA-aware resources for a process
        
        Args:
            process_id: Unique identifier for the process
            process_type: Type of process ('orchestrator', 'agent', 'worker', 'ml_training')
            cpu_requirement: CPU requirement (0.0-1.0 fraction of total)
            memory_requirement: Memory requirement in bytes
            locality_preference: 'auto', 'node_0', 'node_1', 'distributed'
        
        Returns:
            ProcessResourceAllocation with NUMA affinity mapping
        """
        with self._lock:
            # Determine optimal NUMA placement
            numa_affinity = await self._calculate_optimal_numa_placement(
                process_type, cpu_requirement, memory_requirement, locality_preference
            )
            
            # Calculate resource limits
            cpu_shares = max(1, int(cpu_requirement * 1024))  # CPU shares for cgroups
            io_priority = self._get_io_priority(process_type)
            
            # Create allocation
            allocation = ProcessResourceAllocation(
                process_id=process_id,
                numa_affinity=numa_affinity,
                memory_limit=memory_requirement,
                cpu_shares=cpu_shares,
                io_priority=io_priority,
                process_type=process_type
            )
            
            # Track allocation
            self.process_allocations[process_id] = allocation
            
            # Update NUMA utilization tracking
            self.numa_utilization[numa_affinity.numa_node] += cpu_requirement
            
            self.logger.info(f"Allocated NUMA resources for {process_id}: "
                           f"Node {numa_affinity.numa_node}, CPUs {numa_affinity.cpu_list}")
            
            return allocation
    
    async def _calculate_optimal_numa_placement(self,
                                              process_type: str,
                                              cpu_requirement: float,
                                              memory_requirement: int,
                                              locality_preference: str) -> NUMAAffinityMapping:
        """Calculate optimal NUMA node placement for a process"""
        
        if not self.numa_available:
            # Fallback: use all CPUs
            return NUMAAffinityMapping(
                numa_node=0,
                cpu_list=list(range(self.topology.total_threads)),
                memory_policy='interleave',
                expected_locality_ratio=0.5
            )
        
        # Handle explicit preferences
        if locality_preference == 'node_0':
            target_node = 0
        elif locality_preference == 'node_1':
            target_node = min(1, self.topology.numa_nodes - 1)
        elif locality_preference == 'distributed':
            # For distributed workloads, use interleaved memory
            return self._create_distributed_affinity(cpu_requirement)
        else:
            # Auto-select based on current utilization and process type
            target_node = await self._select_optimal_numa_node(process_type, cpu_requirement)
        
        # Calculate CPU allocation within the NUMA node
        cpus_needed = max(1, int(cpu_requirement * len(self.numa_cpu_mapping[target_node])))
        available_cpus = self.numa_cpu_mapping[target_node]
        
        # For ML training and orchestration, prefer physical cores (even numbered)
        if process_type in ['ml_training', 'orchestrator']:
            preferred_cpus = [cpu for cpu in available_cpus if cpu % 2 == 0]
            if len(preferred_cpus) >= cpus_needed:
                selected_cpus = preferred_cpus[:cpus_needed]
            else:
                selected_cpus = available_cpus[:cpus_needed]
        else:
            selected_cpus = available_cpus[:cpus_needed]
        
        # Determine memory policy
        memory_policy = self._get_memory_policy(process_type, memory_requirement)
        
        # Calculate expected locality ratio
        expected_locality = 0.95 if memory_policy == 'bind' else 0.7
        
        return NUMAAffinityMapping(
            numa_node=target_node,
            cpu_list=selected_cpus,
            memory_policy=memory_policy,
            expected_locality_ratio=expected_locality
        )
    
    async def _select_optimal_numa_node(self, process_type: str, cpu_requirement: float) -> int:
        """Select optimal NUMA node based on current utilization and process type"""
        
        # Get current utilization for each NUMA node
        node_scores = {}
        
        for node in range(self.topology.numa_nodes):
            current_util = self.numa_utilization[node]
            
            # Base score: prefer less utilized nodes
            score = 1.0 - current_util
            
            # Process type preferences
            if process_type == 'orchestrator':
                # Orchestrator prefers node 0 (typically where system processes run)
                if node == 0:
                    score += 0.2
            elif process_type == 'ml_training':
                # ML training benefits from dedicated resources
                if current_util < 0.5:  # Prefer nodes with more available capacity
                    score += 0.3
            elif process_type == 'agent':
                # Agents can be distributed more freely
                score += 0.1  # Slight preference for any available node
            
            # Memory bandwidth consideration
            # Node 0 typically has better memory bandwidth for system tasks
            if node == 0 and process_type in ['orchestrator', 'api']:
                score += 0.1
            
            node_scores[node] = score
        
        # Select node with highest score
        optimal_node = max(node_scores.keys(), key=lambda k: node_scores[k])
        
        self.logger.debug(f"NUMA node scores: {node_scores}, selected: {optimal_node}")
        return optimal_node
    
    def _create_distributed_affinity(self, cpu_requirement: float) -> NUMAAffinityMapping:
        """Create distributed CPU affinity across all NUMA nodes"""
        all_cpus = []
        for node in range(self.topology.numa_nodes):
            all_cpus.extend(self.numa_cpu_mapping[node])
        
        # For distributed workloads, use interleaved memory
        return NUMAAffinityMapping(
            numa_node=-1,  # Special value indicating all nodes
            cpu_list=all_cpus,
            memory_policy='interleave',
            expected_locality_ratio=0.6
        )
    
    def _get_memory_policy(self, process_type: str, memory_requirement: int) -> str:
        """Determine optimal memory policy based on process characteristics"""
        
        # Large memory consumers benefit from binding to avoid remote access
        if memory_requirement > 8 * 1024 * 1024 * 1024:  # > 8GB
            return 'bind'
        
        # Process type specific policies
        if process_type == 'ml_training':
            return 'bind'  # ML training benefits from local memory
        elif process_type == 'orchestrator':
            return 'bind'  # Orchestrator needs predictable performance
        elif process_type == 'worker':
            return 'preferred'  # Workers can tolerate some remote access
        else:
            return 'interleave'  # Default: interleave for balanced access
    
    def _get_io_priority(self, process_type: str) -> int:
        """Get I/O priority based on process type"""
        priorities = {
            'orchestrator': 1,  # Highest priority
            'api': 2,
            'ml_training': 3,   # Lower priority for batch workloads
            'worker': 4,
            'agent': 5
        }
        return priorities.get(process_type, 4)
    
    async def apply_process_affinity(self, allocation: ProcessResourceAllocation, pid: int) -> bool:
        """Apply NUMA affinity to a running process"""
        if not self.numa_available:
            self.logger.debug(f"NUMA not available, skipping affinity for PID {pid}")
            return True
        
        try:
            # Set CPU affinity
            cpu_mask = allocation.numa_affinity.cpu_list
            os.sched_setaffinity(pid, cpu_mask)
            
            # Set memory policy if NUMA library is available
            if NUMA_AVAILABLE:
                numa_node = allocation.numa_affinity.numa_node
                
                if numa_node >= 0:  # Specific NUMA node
                    if allocation.numa_affinity.memory_policy == 'bind':
                        numa.set_membind_for_task(pid, [numa_node])
                    elif allocation.numa_affinity.memory_policy == 'preferred':
                        numa.set_preferred_for_task(pid, numa_node)
                else:  # Distributed across all nodes
                    node_list = list(range(self.topology.numa_nodes))
                    numa.set_interleave_for_task(pid, node_list)
            
            self.logger.info(f"Applied NUMA affinity to PID {pid}: "
                           f"CPUs {cpu_mask}, NUMA node {allocation.numa_affinity.numa_node}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply NUMA affinity to PID {pid}: {e}")
            return False
    
    async def optimize_current_process(self, process_type: str, cpu_requirement: float = 0.5) -> bool:
        """Optimize the current process with NUMA awareness"""
        current_pid = os.getpid()
        process_id = f"current_{current_pid}"
        
        # Get current process memory usage for requirement estimation
        try:
            process = psutil.Process(current_pid)
            memory_requirement = process.memory_info().vms  # Virtual memory size
        except Exception:
            memory_requirement = 2 * 1024 * 1024 * 1024  # Default 2GB
        
        # Allocate resources
        allocation = await self.allocate_process_resources(
            process_id, process_type, cpu_requirement, memory_requirement
        )
        
        # Apply to current process
        return await self.apply_process_affinity(allocation, current_pid)
    
    async def get_numa_utilization_stats(self) -> Dict[str, Any]:
        """Get current NUMA utilization statistics"""
        stats = {
            'numa_available': self.numa_available,
            'topology': {
                'numa_nodes': self.topology.numa_nodes,
                'total_cores': self.topology.total_cores,
                'total_threads': self.topology.total_threads,
                'cores_per_numa': self.topology.cores_per_numa,
                'threads_per_numa': self.topology.threads_per_numa
            },
            'utilization': self.numa_utilization.copy(),
            'active_processes': len(self.process_allocations),
            'performance_metrics': self.performance_metrics.copy()
        }
        
        if self.numa_available and NUMA_AVAILABLE:
            # Add real-time NUMA statistics
            try:
                for node in range(self.topology.numa_nodes):
                    node_stats = numa.node_meminfo(node)
                    stats[f'numa_node_{node}_memory'] = node_stats
            except Exception as e:
                self.logger.debug(f"Failed to get NUMA memory stats: {e}")
        
        return stats
    
    async def monitor_performance_metrics(self):
        """Monitor NUMA performance metrics continuously"""
        while True:
            try:
                await self._update_performance_metrics()
                await asyncio.sleep(30)  # Update every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error monitoring NUMA performance: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Memory locality ratio (simplified calculation)
            total_memory_accesses = 0
            local_memory_accesses = 0
            
            for allocation in self.process_allocations.values():
                # Estimate based on expected locality ratio
                expected_local = allocation.numa_affinity.expected_locality_ratio
                total_memory_accesses += 1000  # Mock access count
                local_memory_accesses += 1000 * expected_local
            
            if total_memory_accesses > 0:
                self.performance_metrics['memory_locality_ratio'] = (
                    local_memory_accesses / total_memory_accesses
                )
            
            # Context switches (system-wide)
            try:
                ctx_switches = psutil.cpu_stats().ctx_switches
                if hasattr(self, '_last_ctx_switches'):
                    time_diff = 30  # seconds between measurements
                    ctx_per_sec = (ctx_switches - self._last_ctx_switches) / time_diff
                    self.performance_metrics['context_switches_per_second'] = ctx_per_sec
                self._last_ctx_switches = ctx_switches
            except Exception:
                pass
            
            # Cross-NUMA traffic estimation
            cross_numa_ratio = 0.0
            for allocation in self.process_allocations.values():
                if allocation.numa_affinity.numa_node == -1:  # Distributed
                    cross_numa_ratio += 0.4  # Distributed processes generate cross-NUMA traffic
                else:
                    cross_numa_ratio += (1.0 - allocation.numa_affinity.expected_locality_ratio) * 0.2
            
            if self.process_allocations:
                self.performance_metrics['cross_numa_traffic'] = (
                    cross_numa_ratio / len(self.process_allocations)
                )
            
        except Exception as e:
            self.logger.debug(f"Error updating performance metrics: {e}")
    
    async def optimize_for_workload(self, workload_type: str) -> Dict[str, Any]:
        """Optimize NUMA configuration for specific workload types"""
        optimizations = {}
        
        if workload_type == 'ml_training':
            # ML training optimization
            optimizations = {
                'memory_policy': 'bind',
                'cpu_isolation': True,
                'huge_pages': True,
                'recommended_affinity': 'single_numa',
                'thread_placement': 'physical_cores_first'
            }
            
        elif workload_type == 'high_throughput_api':
            # High-throughput API optimization
            optimizations = {
                'memory_policy': 'interleave',
                'cpu_isolation': False,
                'huge_pages': False,
                'recommended_affinity': 'distributed',
                'thread_placement': 'all_threads'
            }
            
        elif workload_type == 'batch_processing':
            # Batch processing optimization
            optimizations = {
                'memory_policy': 'preferred',
                'cpu_isolation': True,
                'huge_pages': True,
                'recommended_affinity': 'numa_aware',
                'thread_placement': 'ccx_aligned'
            }
            
        elif workload_type == 'real_time_orchestration':
            # Real-time orchestration optimization
            optimizations = {
                'memory_policy': 'bind',
                'cpu_isolation': True,
                'huge_pages': False,
                'recommended_affinity': 'node_0',
                'thread_placement': 'dedicated_cores'
            }
        
        # Apply optimizations
        await self._apply_workload_optimizations(optimizations)
        
        return optimizations
    
    async def _apply_workload_optimizations(self, optimizations: Dict[str, Any]):
        """Apply workload-specific optimizations"""
        try:
            # Enable huge pages if recommended
            if optimizations.get('huge_pages', False):
                await self._configure_huge_pages()
            
            # Configure CPU isolation if recommended
            if optimizations.get('cpu_isolation', False):
                await self._configure_cpu_isolation()
            
            self.logger.info(f"Applied workload optimizations: {optimizations}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply workload optimizations: {e}")
    
    async def _configure_huge_pages(self):
        """Configure huge pages for better memory performance"""
        try:
            # This would typically require root privileges
            # In production, this might be handled by init containers or privileged DaemonSets
            hugepages_path = Path("/proc/sys/vm/nr_hugepages")
            if hugepages_path.exists():
                # Recommend allocation based on available memory
                total_memory_gb = psutil.virtual_memory().total // (1024**3)
                recommended_hugepages = min(1024, total_memory_gb * 50)  # ~100MB worth
                
                self.logger.info(f"Recommended huge pages: {recommended_hugepages}")
                # Note: Actual write would require privileges
                # with open(hugepages_path, 'w') as f:
                #     f.write(str(recommended_hugepages))
                
        except Exception as e:
            self.logger.debug(f"Huge pages configuration failed (expected without privileges): {e}")
    
    async def _configure_cpu_isolation(self):
        """Configure CPU isolation for critical processes"""
        try:
            # CPU isolation typically requires kernel parameters
            # This would be configured via kernel command line: isolcpus=2-31,34-63
            isolated_cpus = []
            
            # Reserve some CPUs for system tasks, isolate others for applications
            for node in range(self.topology.numa_nodes):
                node_cpus = self.numa_cpu_mapping[node]
                # Keep first 2 CPUs per NUMA node for system tasks
                system_cpus = node_cpus[:2]
                app_cpus = node_cpus[2:]
                isolated_cpus.extend(app_cpus)
            
            self.logger.info(f"Recommended isolated CPUs: {isolated_cpus}")
            
        except Exception as e:
            self.logger.debug(f"CPU isolation configuration: {e}")
    
    async def deallocate_process_resources(self, process_id: str):
        """Deallocate resources for a process"""
        with self._lock:
            if process_id in self.process_allocations:
                allocation = self.process_allocations[process_id]
                
                # Update utilization tracking
                numa_node = allocation.numa_affinity.numa_node
                if numa_node >= 0:
                    cpu_fraction = len(allocation.numa_affinity.cpu_list) / self.topology.total_threads
                    self.numa_utilization[numa_node] = max(0.0, 
                        self.numa_utilization[numa_node] - cpu_fraction)
                
                # Remove allocation
                del self.process_allocations[process_id]
                
                self.logger.info(f"Deallocated resources for process {process_id}")
    
    async def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get optimization recommendations based on current state"""
        recommendations = {
            'memory_optimization': [],
            'cpu_optimization': [],
            'numa_optimization': [],
            'performance_warnings': []
        }
        
        # Analyze current utilization
        max_util = max(self.numa_utilization.values())
        min_util = min(self.numa_utilization.values())
        util_imbalance = max_util - min_util
        
        if util_imbalance > 0.3:
            recommendations['numa_optimization'].append({
                'type': 'load_balancing',
                'priority': 'high',
                'description': f'NUMA utilization imbalance detected: {util_imbalance:.2f}',
                'action': 'Consider redistributing processes across NUMA nodes'
            })
        
        # Memory locality analysis
        locality_ratio = self.performance_metrics.get('memory_locality_ratio', 0.0)
        if locality_ratio < 0.8:
            recommendations['memory_optimization'].append({
                'type': 'memory_locality',
                'priority': 'medium',
                'description': f'Low memory locality ratio: {locality_ratio:.2f}',
                'action': 'Consider using bind memory policy for large consumers'
            })
        
        # Context switch analysis
        ctx_switches = self.performance_metrics.get('context_switches_per_second', 0)
        if ctx_switches > 50000:  # High context switch rate
            recommendations['cpu_optimization'].append({
                'type': 'context_switches',
                'priority': 'medium',
                'description': f'High context switch rate: {ctx_switches:.0f}/sec',
                'action': 'Consider CPU affinity and process consolidation'
            })
        
        return recommendations


# Integration helpers for Kubernetes environments
class KubernetesNUMAIntegration:
    """Integration helpers for Kubernetes NUMA-aware deployments"""
    
    @staticmethod
    def generate_numa_aware_pod_spec(allocation: ProcessResourceAllocation) -> Dict[str, Any]:
        """Generate Kubernetes pod specification with NUMA awareness"""
        
        numa_node = allocation.numa_affinity.numa_node
        cpu_list = allocation.numa_affinity.cpu_list
        
        pod_spec = {
            'apiVersion': 'v1',
            'kind': 'Pod',
            'metadata': {
                'annotations': {
                    'numa.kubernetes.io/numa-node': str(numa_node) if numa_node >= 0 else 'any',
                    'numa.kubernetes.io/cpu-list': ','.join(map(str, cpu_list)),
                    'numa.kubernetes.io/memory-policy': allocation.numa_affinity.memory_policy,
                    'xorb.ai/epyc-optimized': 'true'
                }
            },
            'spec': {
                'containers': [{
                    'name': 'app',
                    'resources': {
                        'requests': {
                            'cpu': f'{len(cpu_list)}',
                            'memory': f'{allocation.memory_limit // (1024*1024)}Mi'
                        },
                        'limits': {
                            'cpu': f'{len(cpu_list)}',
                            'memory': f'{allocation.memory_limit // (1024*1024)}Mi',
                            'hugepages-2Mi': '1Gi' if allocation.memory_limit > 4*1024*1024*1024 else '0'
                        }
                    },
                    'env': [
                        {'name': 'NUMA_NODE', 'value': str(numa_node)},
                        {'name': 'CPU_LIST', 'value': ','.join(map(str, cpu_list))},
                        {'name': 'OMP_NUM_THREADS', 'value': str(len(cpu_list))},
                        {'name': 'OPENBLAS_NUM_THREADS', 'value': str(len(cpu_list))},
                        {'name': 'MKL_NUM_THREADS', 'value': str(len(cpu_list))}
                    ]
                }],
                'nodeSelector': {
                    'kubernetes.io/arch': 'amd64',
                    'node.kubernetes.io/cpu-family': 'EPYC',
                    'xorb.ai/numa-topology': 'available'
                },
                'affinity': {
                    'nodeAffinity': {
                        'requiredDuringSchedulingIgnoredDuringExecution': {
                            'nodeSelectorTerms': [{
                                'matchExpressions': [{
                                    'key': 'node.kubernetes.io/cpu-family',
                                    'operator': 'In',
                                    'values': ['EPYC']
                                }]
                            }]
                        }
                    }
                }
            }
        }
        
        return pod_spec


if __name__ == "__main__":
    async def main():
        # Example usage and testing
        optimizer = EPYCNUMAOptimizer()
        
        # Test process allocation
        allocation = await optimizer.allocate_process_resources(
            process_id="test_orchestrator",
            process_type="orchestrator",
            cpu_requirement=0.25,  # 25% of total CPU
            memory_requirement=8 * 1024 * 1024 * 1024  # 8GB
        )
        
        print(f"Allocated resources: {allocation}")
        
        # Get utilization stats
        stats = await optimizer.get_numa_utilization_stats()
        print(f"NUMA stats: {json.dumps(stats, indent=2)}")
        
        # Get optimization recommendations
        recommendations = await optimizer.get_optimization_recommendations()
        print(f"Recommendations: {json.dumps(recommendations, indent=2)}")
        
        # Optimize current process
        await optimizer.optimize_current_process("orchestrator", 0.5)
        
        # Generate Kubernetes pod spec
        k8s_integration = KubernetesNUMAIntegration()
        pod_spec = k8s_integration.generate_numa_aware_pod_spec(allocation)
        print(f"Kubernetes pod spec: {json.dumps(pod_spec, indent=2)}")
        
        # Cleanup
        await optimizer.deallocate_process_resources("test_orchestrator")
    
    asyncio.run(main())