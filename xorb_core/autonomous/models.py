from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

@dataclass
class AutonomousDecision:
    """Record of autonomous decision made by the orchestrator"""
    decision_id: str
    decision_type: str
    context: Dict[str, Any]
    rationale: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    outcome: Optional[str] = None
    feedback_score: Optional[str] = None
    learning_applied: bool = False

@dataclass
class WorkloadProfile:
    """Profile of current workload characteristics"""
    total_active_tasks: int = 0
    task_type_distribution: Dict[str, int] = field(default_factory=dict)
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    success_rates_by_type: Dict[str, float] = field(default_factory=dict)
    average_execution_times: Dict[str, float] = field(default_factory=dict)
    failure_patterns: List[Dict[str, Any]] = field(default_factory=list)

class WorkloadAnalyzer:
    """Analyze workload patterns for optimization"""
    
    def __init__(self):
        self.logger = structlog.get_logger("WorkloadAnalyzer")
        
    async def analyze_current_workload(self, 
                                     active_executions: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current workload characteristics"""
        
        analysis = {
            'total_tasks': len(active_executions),
            'agent_distribution': defaultdict(int),
            'priority_distribution': defaultdict(int),
            'runtime_analysis': {
                'short_running': 0,  # < 30s
                'medium_running': 0,  # 30s - 5min
                'long_running': 0    # > 5min
            }
        }
        
        now = datetime.utcnow()
        
        for context in active_executions.values():
            # Agent distribution
            analysis['agent_distribution'][context.agent_name] += 1
            
            # Priority distribution
            analysis['priority_distribution'][context.priority] += 1
            
            # Runtime analysis
            if context.started_at:
                runtime = (now - context.started_at).total_seconds()
                if runtime < 30:
                    analysis['runtime_analysis']['short_running'] += 1
                elif runtime < 300:
                    analysis['runtime_analysis']['medium_running'] += 1
                else:
                    analysis['runtime_analysis']['long_running'] += 1
        
        return analysis


class PerformanceOptimizer:
    """Optimize performance based on historical data"""
    
    def __init__(self):
        self.logger = structlog.get_logger("PerformanceOptimizer")
        
    async def optimize_agent_allocation(self, 
                                      performance_data: Dict[str, Any],
                                      current_allocation: Dict[str, int]) -> Dict[str, int]:
        """Optimize agent allocation based on performance data"""
        
        optimized_allocation = current_allocation.copy()
        
        # Increase allocation for high-performing agents
        for agent_type, metrics in performance_data.items():
            success_rate = metrics.get('success_rate', 0.5)
            avg_time = metrics.get('avg_execution_time', 30.0)
            
            if success_rate > 0.8 and avg_time < 60.0:
                # High performance - increase allocation
                optimized_allocation[agent_type] = min(
                    optimized_allocation.get(agent_type, 1) + 1, 8
                )
            elif success_rate < 0.5 or avg_time > 120.0:
                # Poor performance - decrease allocation
                optimized_allocation[agent_type] = max(
                    optimized_allocation.get(agent_type, 1) - 1, 1
                )
        
        return optimized_allocation