#!/usr/bin/env python3
"""
XORB Runtime System Analysis and Optimization Engine
"""

import json
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Any

class XorbSystemAnalyzer:
    def __init__(self):
        self.hardware_profile = self._load_hardware_profile()
        self.container_stats = {}
        self.optimization_recommendations = []
        
    def _load_hardware_profile(self) -> Dict[str, Any]:
        """Load the detected hardware profile."""
        try:
            with open('hardware_profile.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "cpu_cores": 16,
                "memory_total_gb": 30,
                "architecture": "x86_64",
                "optimization_profile": "server"
            }
    
    def collect_container_metrics(self) -> Dict[str, Any]:
        """Collect detailed container metrics."""
        try:
            result = subprocess.run([
                'docker', 'stats', '--no-stream', '--format',
                'table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}\t{{.PIDs}}'
            ], capture_output=True, text=True, timeout=10)
            
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            
            metrics = {}
            for line in lines:
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 7:
                        container = parts[0]
                        cpu_perc = parts[1].replace('%', '')
                        mem_usage = parts[2]
                        mem_perc = parts[3].replace('%', '')
                        net_io = parts[4]
                        block_io = parts[5]
                        pids = parts[6]
                        
                        metrics[container] = {
                            'cpu_percent': float(cpu_perc) if cpu_perc != '--' else 0.0,
                            'memory_usage': mem_usage,
                            'memory_percent': float(mem_perc) if mem_perc != '--' else 0.0,
                            'network_io': net_io,
                            'block_io': block_io,
                            'processes': int(pids) if pids.isdigit() else 0
                        }
            
            return metrics
        except Exception as e:
            print(f"Error collecting container metrics: {e}")
            return {}
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics."""
        try:
            # CPU load
            uptime_result = subprocess.run(['uptime'], capture_output=True, text=True)
            load_avg = uptime_result.stdout.split('load average:')[1].strip().split(',')
            load_1min = float(load_avg[0].strip())
            
            # Memory
            free_result = subprocess.run(['free', '-b'], capture_output=True, text=True)
            mem_lines = free_result.stdout.split('\n')
            mem_line = [line for line in mem_lines if line.startswith('Mem:')][0]
            mem_parts = mem_line.split()
            total_mem = int(mem_parts[1])
            used_mem = int(mem_parts[2])
            free_mem = int(mem_parts[3])
            
            # Disk
            df_result = subprocess.run(['df', '/'], capture_output=True, text=True)
            df_lines = df_result.stdout.split('\n')
            disk_line = df_lines[1]
            disk_parts = disk_line.split()
            disk_used_percent = int(disk_parts[4].replace('%', ''))
            
            return {
                'load_average_1min': load_1min,
                'memory_total_bytes': total_mem,
                'memory_used_bytes': used_mem,
                'memory_free_bytes': free_mem,
                'memory_used_percent': (used_mem / total_mem) * 100,
                'disk_used_percent': disk_used_percent
            }
        except Exception as e:
            print(f"Error collecting system metrics: {e}")
            return {}
    
    def analyze_performance_bottlenecks(self) -> List[Dict[str, Any]]:
        """Analyze system for performance bottlenecks."""
        bottlenecks = []
        
        system_metrics = self.get_system_metrics()
        container_metrics = self.collect_container_metrics()
        
        # CPU Analysis
        cpu_cores = self.hardware_profile.get('cpu_cores', 16)
        load_avg = system_metrics.get('load_average_1min', 0)
        
        if load_avg > cpu_cores * 0.8:
            bottlenecks.append({
                'type': 'cpu_overload',
                'severity': 'high',
                'description': f'System load ({load_avg:.2f}) exceeds 80% of available cores ({cpu_cores})',
                'recommendation': 'Reduce concurrent agent count or increase CPU resources'
            })
        
        # Memory Analysis
        mem_used_percent = system_metrics.get('memory_used_percent', 0)
        if mem_used_percent > 85:
            bottlenecks.append({
                'type': 'memory_pressure',
                'severity': 'high',
                'description': f'Memory usage at {mem_used_percent:.1f}%',
                'recommendation': 'Reduce cache sizes or increase memory allocation'
            })
        
        # Container-specific analysis
        for container, metrics in container_metrics.items():
            if metrics['cpu_percent'] > 80:
                bottlenecks.append({
                    'type': 'container_cpu_high',
                    'severity': 'medium',
                    'container': container,
                    'description': f'{container} using {metrics["cpu_percent"]:.1f}% CPU',
                    'recommendation': f'Consider increasing CPU limit for {container}'
                })
            
            if metrics['memory_percent'] > 90:
                bottlenecks.append({
                    'type': 'container_memory_high',
                    'severity': 'high',
                    'container': container,
                    'description': f'{container} using {metrics["memory_percent"]:.1f}% of allocated memory',
                    'recommendation': f'Increase memory limit for {container}'
                })
        
        return bottlenecks
    
    def generate_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate specific optimization recommendations."""
        recommendations = []
        
        hardware = self.hardware_profile
        system_metrics = self.get_system_metrics()
        container_metrics = self.collect_container_metrics()
        
        # CPU Optimizations
        cpu_cores = hardware.get('cpu_cores', 16)
        load_avg = system_metrics.get('load_average_1min', 0)
        
        if load_avg < cpu_cores * 0.3:
            recommendations.append({
                'category': 'performance',
                'priority': 'medium',
                'title': 'Increase Agent Concurrency',
                'description': f'System load ({load_avg:.2f}) is low for {cpu_cores} cores',
                'action': f'Increase MAX_CONCURRENT_AGENTS from 12 to {min(20, cpu_cores + 4)}',
                'expected_impact': 'Improved throughput and resource utilization'
            })
        
        # Memory Optimizations
        total_mem_gb = hardware.get('memory_total_gb', 30)
        if total_mem_gb > 20:
            recommendations.append({
                'category': 'memory',
                'priority': 'high',
                'title': 'Optimize Cache Allocation',
                'description': f'Abundant memory ({total_mem_gb}GB) available',
                'action': f'Increase CACHE_SIZE_MB from 2048 to {min(8192, int(total_mem_gb * 1024 * 0.1))}MB',
                'expected_impact': 'Faster data access and reduced disk I/O'
            })
        
        # NUMA Optimizations (EPYC specific)
        if 'EPYC' in hardware.get('cpu_model', ''):
            recommendations.append({
                'category': 'cpu_affinity',
                'priority': 'high',
                'title': 'Enable NUMA Optimization',
                'description': 'AMD EPYC processor detected',
                'action': 'Configure CPU affinity and NUMA node pinning for containers',
                'expected_impact': 'Reduced memory latency and improved cache performance'
            })
        
        # Container-specific optimizations
        for container, metrics in container_metrics.items():
            if 'orchestrator' in container and metrics['memory_percent'] < 50:
                recommendations.append({
                    'category': 'container_tuning',
                    'priority': 'medium',
                    'title': f'Increase {container} Memory Allocation',
                    'description': f'{container} using only {metrics["memory_percent"]:.1f}% of allocated memory',
                    'action': f'Increase memory limit to improve performance buffer',
                    'expected_impact': 'Better handling of peak workloads'
                })
        
        # Network optimizations
        recommendations.append({
            'category': 'network',
            'priority': 'medium',
            'title': 'Optimize Docker Network Mode',
            'description': 'Default bridge network may have latency overhead',
            'action': 'Consider host networking for latency-critical components',
            'expected_impact': 'Reduced network latency between services'
        })
        
        return recommendations
    
    def calculate_optimization_score(self) -> int:
        """Calculate overall system optimization score (0-100)."""
        score = 100
        
        bottlenecks = self.analyze_performance_bottlenecks()
        
        # Deduct points for bottlenecks
        for bottleneck in bottlenecks:
            if bottleneck['severity'] == 'high':
                score -= 20
            elif bottleneck['severity'] == 'medium':
                score -= 10
            elif bottleneck['severity'] == 'low':
                score -= 5
        
        # Bonus points for good utilization
        system_metrics = self.get_system_metrics()
        load_avg = system_metrics.get('load_average_1min', 0)
        cpu_cores = self.hardware_profile.get('cpu_cores', 16)
        
        utilization = min(load_avg / cpu_cores, 1.0)
        if 0.3 <= utilization <= 0.7:  # Sweet spot
            score += 10
        
        return max(0, min(100, score))
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive system analysis report."""
        system_metrics = self.get_system_metrics()
        container_metrics = self.collect_container_metrics()
        bottlenecks = self.analyze_performance_bottlenecks()
        recommendations = self.generate_optimization_recommendations()
        optimization_score = self.calculate_optimization_score()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'hardware_profile': self.hardware_profile,
            'system_metrics': system_metrics,
            'container_metrics': container_metrics,
            'bottlenecks': bottlenecks,
            'optimization_recommendations': recommendations,
            'optimization_score': optimization_score,
            'summary': {
                'total_containers': len(container_metrics),
                'healthy_containers': len([c for c in container_metrics.values() if c['cpu_percent'] < 80]),
                'bottleneck_count': len(bottlenecks),
                'high_priority_recommendations': len([r for r in recommendations if r['priority'] == 'high']),
                'system_status': 'optimal' if optimization_score >= 80 else 'good' if optimization_score >= 60 else 'needs_attention'
            }
        }
        
        return report

if __name__ == '__main__':
    analyzer = XorbSystemAnalyzer()
    report = analyzer.generate_report()
    
    print(json.dumps(report, indent=2))