"""
Advanced AI reasoning evaluation framework for Xorb 2.0.

This module provides comprehensive evaluation metrics, benchmarking capabilities,
and continuous assessment tools for all AI reasoning components.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for AI components."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    throughput: float = 0.0
    resource_efficiency: float = 0.0
    stability_score: float = 0.0
    
    # Component-specific metrics
    confidence_correlation: float = 0.0
    temporal_consistency: float = 0.0
    causal_accuracy: float = 0.0
    uncertainty_calibration: float = 0.0
    simulation_realism: float = 0.0
    
    # Meta-metrics
    overall_score: float = 0.0
    evaluation_timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class BenchmarkResult:
    """Results from benchmark execution."""
    component_name: str
    benchmark_name: str
    metrics: EvaluationMetrics
    execution_time: float
    resource_usage: Dict[str, float]
    error_log: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseEvaluator(ABC):
    """Base class for component evaluators."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.results_history: List[BenchmarkResult] = []
    
    @abstractmethod
    async def evaluate(self, component: Any, test_data: Any) -> EvaluationMetrics:
        """Evaluate component performance."""
        pass
    
    @abstractmethod
    def generate_report(self) -> Dict[str, Any]:
        """Generate evaluation report."""
        pass

class IntrospectiveReasoningEvaluator(BaseEvaluator):
    """Evaluator for introspective reasoning component."""
    
    def __init__(self):
        super().__init__("introspective_reasoning")
    
    async def evaluate(self, reasoning_system, test_scenarios: List[Dict]) -> EvaluationMetrics:
        """Evaluate introspective reasoning performance."""
        start_time = datetime.utcnow()
        
        reflection_accuracies = []
        confidence_scores = []
        latencies = []
        
        for scenario in test_scenarios:
            scenario_start = datetime.utcnow()
            
            try:
                # Generate reflection for scenario
                reflection = await reasoning_system.generate_reflection(
                    scenario['action'],
                    scenario['context']
                )
                
                # Calculate latency
                latency = (datetime.utcnow() - scenario_start).total_seconds()
                latencies.append(latency)
                
                # Evaluate reflection quality
                accuracy = self._evaluate_reflection_accuracy(
                    reflection, scenario['expected_insights']
                )
                reflection_accuracies.append(accuracy)
                confidence_scores.append(reflection.confidence)
                
            except Exception as e:
                logger.error(f"Evaluation error for scenario {scenario['id']}: {e}")
                reflection_accuracies.append(0.0)
                confidence_scores.append(0.0)
        
        # Calculate metrics
        metrics = EvaluationMetrics(
            accuracy=np.mean(reflection_accuracies),
            latency_p50=np.percentile(latencies, 50),
            latency_p95=np.percentile(latencies, 95),
            latency_p99=np.percentile(latencies, 99),
            confidence_correlation=spearmanr(reflection_accuracies, confidence_scores)[0],
            throughput=len(test_scenarios) / (datetime.utcnow() - start_time).total_seconds()
        )
        
        # Calculate overall score
        metrics.overall_score = (
            metrics.accuracy * 0.4 +
            (1.0 - min(metrics.latency_p95 / 5.0, 1.0)) * 0.3 +  # Latency penalty
            abs(metrics.confidence_correlation) * 0.3
        )
        
        return metrics
    
    def _evaluate_reflection_accuracy(self, reflection, expected_insights: List[str]) -> float:
        """Evaluate how well reflection matches expected insights."""
        if not reflection.insights:
            return 0.0
        
        insight_matches = 0
        for expected in expected_insights:
            for actual in reflection.insights:
                if self._semantic_similarity(expected, actual) > 0.7:
                    insight_matches += 1
                    break
        
        return insight_matches / len(expected_insights)
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts (simplified)."""
        # In practice, use embeddings and cosine similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0

class TemporalMemoryEvaluator(BaseEvaluator):
    """Evaluator for temporal memory system."""
    
    def __init__(self):
        super().__init__("temporal_memory")
    
    async def evaluate(self, memory_system, test_sequences: List[Dict]) -> EvaluationMetrics:
        """Evaluate temporal memory performance."""
        start_time = datetime.utcnow()
        
        recall_accuracies = []
        consolidation_scores = []
        temporal_consistencies = []
        
        for sequence in test_sequences:
            # Store sequence of events
            for event in sequence['events']:
                await memory_system.store_memory(event)
            
            # Test recall accuracy
            for query in sequence['recall_queries']:
                recalled = await memory_system.recall_memories(
                    query['context'],
                    query['time_range']
                )
                
                accuracy = self._evaluate_recall_accuracy(
                    recalled, query['expected_memories']
                )
                recall_accuracies.append(accuracy)
            
            # Test temporal consistency
            consistency = await self._evaluate_temporal_consistency(
                memory_system, sequence['events']
            )
            temporal_consistencies.append(consistency)
        
        # Test consolidation effectiveness
        consolidation_score = await self._evaluate_consolidation(memory_system)
        
        metrics = EvaluationMetrics(
            accuracy=np.mean(recall_accuracies),
            temporal_consistency=np.mean(temporal_consistencies),
            stability_score=consolidation_score,
            throughput=len(test_sequences) / (datetime.utcnow() - start_time).total_seconds()
        )
        
        metrics.overall_score = (
            metrics.accuracy * 0.4 +
            metrics.temporal_consistency * 0.3 +
            metrics.stability_score * 0.3
        )
        
        return metrics
    
    def _evaluate_recall_accuracy(self, recalled: List, expected: List) -> float:
        """Evaluate recall accuracy."""
        if not expected:
            return 1.0 if not recalled else 0.0
        
        correct_recalls = 0
        for exp_memory in expected:
            for recalled_memory in recalled:
                if self._memory_match(exp_memory, recalled_memory):
                    correct_recalls += 1
                    break
        
        return correct_recalls / len(expected)
    
    def _memory_match(self, memory1: Dict, memory2: Dict) -> bool:
        """Check if two memories match."""
        return (
            memory1.get('event_id') == memory2.get('event_id') or
            self._semantic_similarity(
                memory1.get('content', ''),
                memory2.get('content', '')
            ) > 0.8
        )
    
    async def _evaluate_temporal_consistency(self, memory_system, events: List) -> float:
        """Evaluate temporal consistency of stored memories."""
        # Check if temporal ordering is preserved
        stored_memories = await memory_system.get_all_memories()
        
        if len(stored_memories) < 2:
            return 1.0
        
        consistency_violations = 0
        for i in range(len(stored_memories) - 1):
            if stored_memories[i].timestamp > stored_memories[i + 1].timestamp:
                consistency_violations += 1
        
        return 1.0 - (consistency_violations / max(len(stored_memories) - 1, 1))
    
    async def _evaluate_consolidation(self, memory_system) -> float:
        """Evaluate memory consolidation effectiveness."""
        pre_consolidation_count = len(await memory_system.get_all_memories())
        await memory_system.consolidate_memories()
        post_consolidation_count = len(await memory_system.get_all_memories())
        
        # Good consolidation should reduce memory count while preserving important information
        consolidation_ratio = post_consolidation_count / max(pre_consolidation_count, 1)
        return min(1.0, 2.0 - consolidation_ratio)  # Optimal around 50% reduction

class HierarchicalPolicyEvaluator(BaseEvaluator):
    """Evaluator for hierarchical policy switching."""
    
    def __init__(self):
        super().__init__("hierarchical_policy")
    
    async def evaluate(self, policy_system, test_scenarios: List[Dict]) -> EvaluationMetrics:
        """Evaluate hierarchical policy performance."""
        start_time = datetime.utcnow()
        
        switching_accuracies = []
        adaptation_speeds = []
        stability_scores = []
        
        for scenario in test_scenarios:
            # Simulate scenario conditions
            await policy_system.update_context(scenario['context'])
            
            # Track policy switches
            switches_before = policy_system.get_switch_count()
            
            # Execute scenario
            result = await policy_system.execute_scenario(scenario)
            
            switches_after = policy_system.get_switch_count()
            switch_count = switches_after - switches_before
            
            # Evaluate switching appropriateness
            expected_switches = scenario.get('expected_switches', 0)
            switch_accuracy = 1.0 - abs(switch_count - expected_switches) / max(expected_switches, 1)
            switching_accuracies.append(switch_accuracy)
            
            # Evaluate adaptation speed
            adaptation_time = scenario.get('adaptation_time', 0)
            adaptation_speed = 1.0 / (1.0 + adaptation_time)
            adaptation_speeds.append(adaptation_speed)
            
            # Evaluate stability (fewer unnecessary switches)
            stability = 1.0 / (1.0 + max(0, switch_count - expected_switches))
            stability_scores.append(stability)
        
        metrics = EvaluationMetrics(
            accuracy=np.mean(switching_accuracies),
            stability_score=np.mean(stability_scores),
            resource_efficiency=np.mean(adaptation_speeds),
            throughput=len(test_scenarios) / (datetime.utcnow() - start_time).total_seconds()
        )
        
        metrics.overall_score = (
            metrics.accuracy * 0.4 +
            metrics.stability_score * 0.4 +
            metrics.resource_efficiency * 0.2
        )
        
        return metrics

class ComprehensiveEvaluationFramework:
    """Main evaluation framework coordinating all component evaluators."""
    
    def __init__(self):
        self.evaluators = {
            'introspective_reasoning': IntrospectiveReasoningEvaluator(),
            'temporal_memory': TemporalMemoryEvaluator(),
            'hierarchical_policy': HierarchicalPolicyEvaluator(),
            # Add other evaluators as needed
        }
        
        self.benchmark_suite = BenchmarkSuite()
        self.report_generator = EvaluationReportGenerator()
    
    async def run_comprehensive_evaluation(self, components: Dict[str, Any]) -> Dict[str, BenchmarkResult]:
        """Run comprehensive evaluation across all components."""
        results = {}
        
        for component_name, component in components.items():
            if component_name in self.evaluators:
                evaluator = self.evaluators[component_name]
                
                # Get test data for component
                test_data = await self.benchmark_suite.get_test_data(component_name)
                
                # Run evaluation
                metrics = await evaluator.evaluate(component, test_data)
                
                # Create benchmark result
                result = BenchmarkResult(
                    component_name=component_name,
                    benchmark_name="comprehensive_evaluation",
                    metrics=metrics,
                    execution_time=(datetime.utcnow() - metrics.evaluation_timestamp).total_seconds(),
                    resource_usage=await self._get_resource_usage(component_name)
                )
                
                results[component_name] = result
                logger.info(f"Completed evaluation for {component_name}: score {metrics.overall_score:.3f}")
        
        return results
    
    async def continuous_monitoring(self, components: Dict[str, Any], interval: int = 300):
        """Run continuous monitoring and evaluation."""
        while True:
            try:
                results = await self.run_comprehensive_evaluation(components)
                await self._update_monitoring_dashboard(results)
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(60)  # Shorter retry interval
    
    async def _get_resource_usage(self, component_name: str) -> Dict[str, float]:
        """Get resource usage metrics for component."""
        # This would integrate with actual monitoring systems
        return {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'gpu_usage': 0.0,
            'network_io': 0.0,
            'disk_io': 0.0
        }
    
    async def _update_monitoring_dashboard(self, results: Dict[str, BenchmarkResult]):
        """Update monitoring dashboard with latest results."""
        # Integration with Grafana/Prometheus would go here
        for component_name, result in results.items():
            logger.info(f"Dashboard update - {component_name}: {result.metrics.overall_score:.3f}")

class BenchmarkSuite:
    """Benchmark test suite for AI components."""
    
    def __init__(self):
        self.test_data_cache = {}
    
    async def get_test_data(self, component_name: str) -> Any:
        """Get test data for specific component."""
        if component_name not in self.test_data_cache:
            self.test_data_cache[component_name] = await self._load_test_data(component_name)
        
        return self.test_data_cache[component_name]
    
    async def _load_test_data(self, component_name: str) -> Any:
        """Load test data for component."""
        test_data_generators = {
            'introspective_reasoning': self._generate_reasoning_test_data,
            'temporal_memory': self._generate_memory_test_data,
            'hierarchical_policy': self._generate_policy_test_data,
        }
        
        generator = test_data_generators.get(component_name)
        if generator:
            return await generator()
        
        return []
    
    async def _generate_reasoning_test_data(self) -> List[Dict]:
        """Generate test scenarios for introspective reasoning."""
        return [
            {
                'id': 'reasoning_001',
                'action': 'agent_selection',
                'context': {'target': 'web_app', 'complexity': 'high'},
                'expected_insights': [
                    'High complexity requires specialized agents',
                    'Web application suggests specific attack vectors'
                ]
            },
            # Add more test scenarios
        ]
    
    async def _generate_memory_test_data(self) -> List[Dict]:
        """Generate test sequences for temporal memory."""
        return [
            {
                'id': 'memory_001',
                'events': [
                    {'event_id': 'e1', 'content': 'Initial reconnaissance', 'timestamp': datetime.utcnow()},
                    {'event_id': 'e2', 'content': 'Vulnerability discovery', 'timestamp': datetime.utcnow()},
                ],
                'recall_queries': [
                    {
                        'context': 'vulnerability',
                        'time_range': timedelta(hours=1),
                        'expected_memories': [{'event_id': 'e2'}]
                    }
                ]
            }
        ]
    
    async def _generate_policy_test_data(self) -> List[Dict]:
        """Generate test scenarios for hierarchical policy."""
        return [
            {
                'id': 'policy_001',
                'context': {'threat_level': 'high', 'resources': 'limited'},
                'expected_switches': 2,
                'adaptation_time': 1.5
            }
        ]

class EvaluationReportGenerator:
    """Generate comprehensive evaluation reports."""
    
    def __init__(self, output_dir: str = "/tmp/xorb_evaluation_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_comprehensive_report(self, results: Dict[str, BenchmarkResult]) -> str:
        """Generate comprehensive evaluation report."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"xorb_evaluation_report_{timestamp}.html"
        
        # Generate HTML report
        html_content = self._generate_html_report(results)
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        # Generate visualizations
        self._generate_visualizations(results, timestamp)
        
        logger.info(f"Comprehensive evaluation report generated: {report_path}")
        return str(report_path)
    
    def _generate_html_report(self, results: Dict[str, BenchmarkResult]) -> str:
        """Generate HTML content for evaluation report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Xorb AI Reasoning Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ccc; }
                .good { background-color: #d4edda; }
                .warning { background-color: #fff3cd; }
                .critical { background-color: #f8d7da; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Xorb AI Reasoning Evaluation Report</h1>
            <p>Generated: {timestamp}</p>
        """.format(timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
        
        # Summary section
        html += "<h2>Summary</h2>\n<table>\n"
        html += "<tr><th>Component</th><th>Overall Score</th><th>Accuracy</th><th>Latency P95</th><th>Status</th></tr>\n"
        
        for component_name, result in results.items():
            metrics = result.metrics
            status = self._get_status_class(metrics.overall_score)
            html += f"<tr class='{status}'>"
            html += f"<td>{component_name}</td>"
            html += f"<td>{metrics.overall_score:.3f}</td>"
            html += f"<td>{metrics.accuracy:.3f}</td>"
            html += f"<td>{metrics.latency_p95:.3f}s</td>"
            html += f"<td>{status.upper()}</td>"
            html += "</tr>\n"
        
        html += "</table>\n"
        
        # Detailed metrics for each component
        for component_name, result in results.items():
            html += f"<h2>{component_name.replace('_', ' ').title()}</h2>\n"
            html += self._generate_component_section(result)
        
        html += "</body></html>"
        return html
    
    def _get_status_class(self, score: float) -> str:
        """Get CSS class based on score."""
        if score >= 0.8:
            return "good"
        elif score >= 0.6:
            return "warning"
        else:
            return "critical"
    
    def _generate_component_section(self, result: BenchmarkResult) -> str:
        """Generate HTML section for component results."""
        metrics = result.metrics
        html = "<table>\n"
        
        metric_items = [
            ("Overall Score", f"{metrics.overall_score:.3f}"),
            ("Accuracy", f"{metrics.accuracy:.3f}"),
            ("Precision", f"{metrics.precision:.3f}"),
            ("Recall", f"{metrics.recall:.3f}"),
            ("F1 Score", f"{metrics.f1_score:.3f}"),
            ("Latency P50", f"{metrics.latency_p50:.3f}s"),
            ("Latency P95", f"{metrics.latency_p95:.3f}s"),
            ("Latency P99", f"{metrics.latency_p99:.3f}s"),
            ("Throughput", f"{metrics.throughput:.2f} ops/sec"),
            ("Stability Score", f"{metrics.stability_score:.3f}"),
        ]
        
        for name, value in metric_items:
            html += f"<tr><td>{name}</td><td>{value}</td></tr>\n"
        
        html += "</table>\n"
        return html
    
    def _generate_visualizations(self, results: Dict[str, BenchmarkResult], timestamp: str):
        """Generate visualization charts for results."""
        # Overall scores comparison
        component_names = list(results.keys())
        overall_scores = [result.metrics.overall_score for result in results.values()]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(component_names, overall_scores)
        
        # Color bars based on score
        for bar, score in zip(bars, overall_scores):
            if score >= 0.8:
                bar.set_color('green')
            elif score >= 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.title('AI Component Overall Scores')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"overall_scores_{timestamp}.png")
        plt.close()
        
        # Latency comparison
        plt.figure(figsize=(12, 6))
        latencies_p95 = [result.metrics.latency_p95 for result in results.values()]
        plt.bar(component_names, latencies_p95, color='skyblue')
        plt.title('AI Component Latency (95th Percentile)')
        plt.ylabel('Latency (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"latency_comparison_{timestamp}.png")
        plt.close()
        
        logger.info(f"Evaluation visualizations saved to {self.output_dir}")