#!/usr/bin/env python3
"""
Replay Store Metrics Schema for Parquet Storage
High-performance time-series metrics for cyber range analytics
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import numpy as np
import uuid

class MetricType(Enum):
    """Types of metrics collected"""
    PERFORMANCE = "performance"
    SECURITY = "security"
    LEARNING = "learning"
    BUSINESS = "business"
    SYSTEM = "system"
    COMPLIANCE = "compliance"

class AggregationType(Enum):
    """Metric aggregation types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"

@dataclass
class MetricDefinition:
    """Definition of a metric for consistent collection"""
    name: str
    metric_type: MetricType
    aggregation_type: AggregationType
    description: str
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)  # warning, critical

@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    episode_id: str
    round_id: int
    metric_name: str
    metric_type: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class CyberRangeMetrics:
    """Comprehensive metrics collection for cyber range episodes"""
    
    def __init__(self):
        self.metrics_definitions = self._initialize_metrics()
        self.current_episode_id = ""
        self.current_round_id = 0
        self._metrics_buffer = []
    
    def _initialize_metrics(self) -> Dict[str, MetricDefinition]:
        """Initialize all metric definitions"""
        metrics = {}
        
        # Learning Metrics
        metrics["episodes_per_day"] = MetricDefinition(
            name="episodes_per_day",
            metric_type=MetricType.LEARNING,
            aggregation_type=AggregationType.GAUGE,
            description="Number of episodes completed per day",
            unit="episodes/day",
            thresholds={"warning": 50, "critical": 100}
        )
        
        metrics["promotion_cadence"] = MetricDefinition(
            name="promotion_cadence",
            metric_type=MetricType.LEARNING,
            aggregation_type=AggregationType.COUNTER,
            description="Policy promotions per week",
            unit="promotions/week",
            thresholds={"warning": 1, "critical": 2}
        )
        
        metrics["novelty_index"] = MetricDefinition(
            name="novelty_index",
            metric_type=MetricType.LEARNING,
            aggregation_type=AggregationType.GAUGE,
            description="Novelty score of episode tactics",
            unit="score",
            thresholds={"warning": 0.6, "critical": 0.8}
        )
        
        metrics["archive_coverage"] = MetricDefinition(
            name="archive_coverage",
            metric_type=MetricType.LEARNING,
            aggregation_type=AggregationType.GAUGE,
            description="MAP-Elites archive fill ratio",
            unit="percentage",
            thresholds={"warning": 0.7, "critical": 0.8}
        )
        
        # Defense Metrics
        metrics["detection_latency_p50"] = MetricDefinition(
            name="detection_latency_p50",
            metric_type=MetricType.SECURITY,
            aggregation_type=AggregationType.TIMER,
            description="50th percentile detection latency",
            unit="milliseconds",
            thresholds={"warning": 30000, "critical": 15000}
        )
        
        metrics["detection_latency_p95"] = MetricDefinition(
            name="detection_latency_p95",
            metric_type=MetricType.SECURITY,
            aggregation_type=AggregationType.TIMER,
            description="95th percentile detection latency",
            unit="milliseconds",
            thresholds={"warning": 60000, "critical": 30000}
        )
        
        metrics["false_positive_rate"] = MetricDefinition(
            name="false_positive_rate",
            metric_type=MetricType.SECURITY,
            aggregation_type=AggregationType.RATE,
            description="False positive detection rate",
            unit="percentage",
            thresholds={"warning": 0.1, "critical": 0.05}
        )
        
        metrics["residual_risk_score"] = MetricDefinition(
            name="residual_risk_score",
            metric_type=MetricType.SECURITY,
            aggregation_type=AggregationType.GAUGE,
            description="Remaining security risk after defenses",
            unit="score",
            thresholds={"warning": 0.4, "critical": 0.6}
        )
        
        metrics["replay_pass_rate"] = MetricDefinition(
            name="replay_pass_rate",
            metric_type=MetricType.SECURITY,
            aggregation_type=AggregationType.RATE,
            description="Percentage of replays passing validation",
            unit="percentage",
            thresholds={"warning": 0.9, "critical": 0.95}
        )
        
        # Business Metrics
        metrics["mttr"] = MetricDefinition(
            name="mttr",
            metric_type=MetricType.BUSINESS,
            aggregation_type=AggregationType.TIMER,
            description="Mean Time To Recovery",
            unit="minutes",
            thresholds={"warning": 60, "critical": 30}
        )
        
        metrics["mttc"] = MetricDefinition(
            name="mttc",
            metric_type=MetricType.BUSINESS,
            aggregation_type=AggregationType.TIMER,
            description="Mean Time To Containment",
            unit="minutes",
            thresholds={"warning": 30, "critical": 15}
        )
        
        metrics["time_to_patch"] = MetricDefinition(
            name="time_to_patch",
            metric_type=MetricType.BUSINESS,
            aggregation_type=AggregationType.TIMER,
            description="Time from PR creation to deployment",
            unit="minutes",
            thresholds={"warning": 120, "critical": 60}
        )
        
        metrics["sla_adherence"] = MetricDefinition(
            name="sla_adherence",
            metric_type=MetricType.BUSINESS,
            aggregation_type=AggregationType.GAUGE,
            description="SLA adherence during attack simulations",
            unit="percentage",
            thresholds={"warning": 0.95, "critical": 0.99}
        )
        
        metrics["zero_day_findings"] = MetricDefinition(
            name="zero_day_findings",
            metric_type=MetricType.SECURITY,
            aggregation_type=AggregationType.COUNTER,
            description="Zero-day class findings per month",
            unit="findings/month",
            thresholds={"warning": 2, "critical": 3}
        )
        
        # Performance Metrics
        metrics["episode_duration"] = MetricDefinition(
            name="episode_duration",
            metric_type=MetricType.PERFORMANCE,
            aggregation_type=AggregationType.TIMER,
            description="Total episode execution time",
            unit="seconds",
            thresholds={"warning": 300, "critical": 600}
        )
        
        metrics["resource_utilization"] = MetricDefinition(
            name="resource_utilization",
            metric_type=MetricType.PERFORMANCE,
            aggregation_type=AggregationType.GAUGE,
            description="Compute resource utilization",
            unit="percentage",
            thresholds={"warning": 0.8, "critical": 0.9}
        )
        
        metrics["memory_usage"] = MetricDefinition(
            name="memory_usage",
            metric_type=MetricType.PERFORMANCE,
            aggregation_type=AggregationType.GAUGE,
            description="Memory usage during episode",
            unit="gigabytes",
            thresholds={"warning": 8, "critical": 12}
        )
        
        # Compliance Metrics
        metrics["compliance_violations"] = MetricDefinition(
            name="compliance_violations",
            metric_type=MetricType.COMPLIANCE,
            aggregation_type=AggregationType.COUNTER,
            description="Compliance framework violations",
            unit="violations",
            thresholds={"warning": 1, "critical": 0}
        )
        
        metrics["audit_trail_completeness"] = MetricDefinition(
            name="audit_trail_completeness",
            metric_type=MetricType.COMPLIANCE,
            aggregation_type=AggregationType.GAUGE,
            description="Completeness of audit trail",
            unit="percentage",
            thresholds={"warning": 0.95, "critical": 0.99}
        )
        
        return metrics
    
    def record_metric(self, metric_name: str, value: float, 
                     tags: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """Record a metric point"""
        if metric_name not in self.metrics_definitions:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        metric_def = self.metrics_definitions[metric_name]
        
        metric_point = MetricPoint(
            timestamp=datetime.utcnow(),
            episode_id=self.current_episode_id,
            round_id=self.current_round_id,
            metric_name=metric_name,
            metric_type=metric_def.metric_type.value,
            value=value,
            tags=tags or {},
            metadata=metadata or {}
        )
        
        self._metrics_buffer.append(metric_point)
    
    def record_episode_metrics(self, episode_summary: Dict[str, Any]):
        """Record comprehensive episode metrics"""
        episode_id = episode_summary.get("episode_id", "")
        self.current_episode_id = episode_id
        
        # Learning metrics
        if "learning_metrics" in episode_summary:
            learning = episode_summary["learning_metrics"]
            
            if "novelty_score" in learning:
                self.record_metric("novelty_index", learning["novelty_score"])
            
            if "archive_fill_ratio" in learning:
                self.record_metric("archive_coverage", learning["archive_fill_ratio"])
        
        # Security metrics
        if "security_metrics" in episode_summary:
            security = episode_summary["security_metrics"]
            
            if "detection_times" in security:
                detection_times = security["detection_times"]
                if detection_times:
                    self.record_metric("detection_latency_p50", np.percentile(detection_times, 50))
                    self.record_metric("detection_latency_p95", np.percentile(detection_times, 95))
            
            if "false_positive_rate" in security:
                self.record_metric("false_positive_rate", security["false_positive_rate"])
            
            if "residual_risk" in security:
                self.record_metric("residual_risk_score", security["residual_risk"])
        
        # Performance metrics
        if "performance_metrics" in episode_summary:
            performance = episode_summary["performance_metrics"]
            
            if "duration_seconds" in performance:
                self.record_metric("episode_duration", performance["duration_seconds"])
            
            if "resource_utilization" in performance:
                self.record_metric("resource_utilization", performance["resource_utilization"])
            
            if "memory_usage_gb" in performance:
                self.record_metric("memory_usage", performance["memory_usage_gb"])
        
        # Business metrics
        if "business_metrics" in episode_summary:
            business = episode_summary["business_metrics"]
            
            if "mttr_minutes" in business:
                self.record_metric("mttr", business["mttr_minutes"])
            
            if "mttc_minutes" in business:
                self.record_metric("mttc", business["mttc_minutes"])
        
        # Compliance metrics
        if "compliance_metrics" in episode_summary:
            compliance = episode_summary["compliance_metrics"]
            
            if "violations_count" in compliance:
                self.record_metric("compliance_violations", compliance["violations_count"])
            
            if "audit_completeness" in compliance:
                self.record_metric("audit_trail_completeness", compliance["audit_completeness"])
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Convert metrics buffer to pandas DataFrame"""
        if not self._metrics_buffer:
            return pd.DataFrame()
        
        data = []
        for metric in self._metrics_buffer:
            row = {
                "timestamp": metric.timestamp,
                "episode_id": metric.episode_id,
                "round_id": metric.round_id,
                "metric_name": metric.metric_name,
                "metric_type": metric.metric_type,
                "value": metric.value,
            }
            # Flatten tags
            for key, value in metric.tags.items():
                row[f"tag_{key}"] = value
            
            # Add metadata
            row["metadata"] = str(metric.metadata) if metric.metadata else ""
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_to_parquet(self, file_path: str):
        """Save metrics to Parquet file with compression"""
        df = self.get_metrics_dataframe()
        if df.empty:
            return
        
        # Optimize data types
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["episode_id"] = df["episode_id"].astype("category")
        df["metric_name"] = df["metric_name"].astype("category")
        df["metric_type"] = df["metric_type"].astype("category")
        df["value"] = pd.to_numeric(df["value"], downcast="float")
        df["round_id"] = pd.to_numeric(df["round_id"], downcast="integer")
        
        # Save with compression and metadata
        table = pa.Table.from_pandas(df)
        
        # Add metadata
        metadata = {
            "schema_version": "1.0.0",
            "created_timestamp": datetime.utcnow().isoformat(),
            "metric_definitions": str(len(self.metrics_definitions)),
            "total_metrics": str(len(self._metrics_buffer))
        }
        
        table = table.replace_schema_metadata(metadata)
        
        # Write with compression
        pq.write_table(
            table,
            file_path,
            compression="snappy",
            use_dictionary=["episode_id", "metric_name", "metric_type"],
            row_group_size=10000
        )
    
    def load_from_parquet(self, file_path: str) -> pd.DataFrame:
        """Load metrics from Parquet file"""
        return pd.read_parquet(file_path)
    
    def calculate_kpis(self, timeframe_hours: int = 24) -> Dict[str, Any]:
        """Calculate Key Performance Indicators for dashboard"""
        df = self.get_metrics_dataframe()
        if df.empty:
            return {}
        
        # Filter to timeframe
        cutoff_time = datetime.utcnow() - timedelta(hours=timeframe_hours)
        df = df[df["timestamp"] >= cutoff_time]
        
        kpis = {}
        
        # Learning KPIs
        learning_metrics = df[df["metric_type"] == "learning"]
        if not learning_metrics.empty:
            kpis["learning"] = {
                "avg_novelty_index": learning_metrics[learning_metrics["metric_name"] == "novelty_index"]["value"].mean(),
                "latest_archive_coverage": learning_metrics[learning_metrics["metric_name"] == "archive_coverage"]["value"].iloc[-1] if len(learning_metrics) > 0 else 0,
                "episodes_completed": len(learning_metrics["episode_id"].unique())
            }
        
        # Security KPIs
        security_metrics = df[df["metric_type"] == "security"]
        if not security_metrics.empty:
            detection_p95 = security_metrics[security_metrics["metric_name"] == "detection_latency_p95"]["value"]
            false_positive = security_metrics[security_metrics["metric_name"] == "false_positive_rate"]["value"]
            
            kpis["security"] = {
                "avg_detection_latency_p95": detection_p95.mean() if not detection_p95.empty else 0,
                "avg_false_positive_rate": false_positive.mean() if not false_positive.empty else 0,
                "latest_residual_risk": security_metrics[security_metrics["metric_name"] == "residual_risk_score"]["value"].iloc[-1] if len(security_metrics) > 0 else 0
            }
        
        # Business KPIs
        business_metrics = df[df["metric_type"] == "business"]
        if not business_metrics.empty:
            mttr = business_metrics[business_metrics["metric_name"] == "mttr"]["value"]
            mttc = business_metrics[business_metrics["metric_name"] == "mttc"]["value"]
            
            kpis["business"] = {
                "avg_mttr": mttr.mean() if not mttr.empty else 0,
                "avg_mttc": mttc.mean() if not mttc.empty else 0,
                "zero_day_findings": business_metrics[business_metrics["metric_name"] == "zero_day_findings"]["value"].sum()
            }
        
        return kpis
    
    def check_thresholds(self) -> List[Dict[str, Any]]:
        """Check metrics against defined thresholds"""
        alerts = []
        df = self.get_metrics_dataframe()
        
        for metric_point in self._metrics_buffer[-100:]:  # Check last 100 metrics
            metric_def = self.metrics_definitions.get(metric_point.metric_name)
            if not metric_def or not metric_def.thresholds:
                continue
            
            value = metric_point.value
            
            if "critical" in metric_def.thresholds:
                threshold = metric_def.thresholds["critical"]
                if (metric_def.name in ["detection_latency_p95", "false_positive_rate", "mttr", "mttc"] and value > threshold) or \
                   (metric_def.name in ["archive_coverage", "replay_pass_rate", "sla_adherence"] and value < threshold):
                    alerts.append({
                        "severity": "critical",
                        "metric": metric_def.name,
                        "value": value,
                        "threshold": threshold,
                        "timestamp": metric_point.timestamp,
                        "episode_id": metric_point.episode_id
                    })
            
            elif "warning" in metric_def.thresholds:
                threshold = metric_def.thresholds["warning"]
                if (metric_def.name in ["detection_latency_p95", "false_positive_rate", "mttr", "mttc"] and value > threshold) or \
                   (metric_def.name in ["archive_coverage", "replay_pass_rate", "sla_adherence"] and value < threshold):
                    alerts.append({
                        "severity": "warning",
                        "metric": metric_def.name,
                        "value": value,
                        "threshold": threshold,
                        "timestamp": metric_point.timestamp,
                        "episode_id": metric_point.episode_id
                    })
        
        return alerts
    
    def clear_buffer(self):
        """Clear the metrics buffer"""
        self._metrics_buffer.clear()

if __name__ == "__main__":
    # Example usage and testing
    print("Testing metrics schema and Parquet storage...")
    
    # Initialize metrics collector
    metrics = CyberRangeMetrics()
    metrics.current_episode_id = "test_episode_001"
    metrics.current_round_id = 1
    
    # Record some sample metrics
    metrics.record_metric("novelty_index", 0.75, tags={"agent": "red_team"})
    metrics.record_metric("detection_latency_p95", 25000, tags={"system": "ids"})
    metrics.record_metric("false_positive_rate", 0.08, tags={"detector": "ml_model"})
    metrics.record_metric("archive_coverage", 0.82, tags={"algorithm": "map_elites"})
    
    # Record episode metrics
    episode_summary = {
        "episode_id": "test_episode_001",
        "learning_metrics": {
            "novelty_score": 0.78,
            "archive_fill_ratio": 0.85
        },
        "security_metrics": {
            "detection_times": [15000, 22000, 18000, 30000],
            "false_positive_rate": 0.06,
            "residual_risk": 0.35
        },
        "performance_metrics": {
            "duration_seconds": 180,
            "resource_utilization": 0.7,
            "memory_usage_gb": 4.2
        }
    }
    
    metrics.record_episode_metrics(episode_summary)
    
    # Save to Parquet
    metrics.save_to_parquet("/tmp/test_metrics.parquet")
    
    # Load and verify
    loaded_df = metrics.load_from_parquet("/tmp/test_metrics.parquet")
    print(f"Loaded {len(loaded_df)} metric points from Parquet")
    print(f"Metrics: {loaded_df['metric_name'].unique()}")
    
    # Calculate KPIs
    kpis = metrics.calculate_kpis()
    print(f"KPIs calculated: {list(kpis.keys())}")
    
    # Check thresholds
    alerts = metrics.check_thresholds()
    print(f"Threshold alerts: {len(alerts)}")
    
    print("Metrics schema test completed successfully!")