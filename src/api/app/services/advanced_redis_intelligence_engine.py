"""
XORB Advanced Redis Intelligence Engine - AI-Powered Redis Operations
Implements machine learning for Redis optimization, predictive caching, and intelligent data management
"""

import asyncio
import logging
import json
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import pickle
import hashlib
from collections import defaultdict, deque
import statistics

# ML and AI imports with graceful fallbacks
try:
    import sklearn
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, silhouette_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from .interfaces import IntelligenceService
from .base_service import XORBService, ServiceType
from ..infrastructure.advanced_redis_orchestrator import AdvancedRedisOrchestrator, get_redis_orchestrator

logger = logging.getLogger(__name__)


class CachePredictionModel(Enum):
    """Cache prediction models"""
    NEURAL_NETWORK = "neural_network"
    RANDOM_FOREST = "random_forest"
    TIME_SERIES = "time_series"
    PATTERN_MATCHING = "pattern_matching"
    ENSEMBLE = "ensemble"


class OptimizationStrategy(Enum):
    """Redis optimization strategies"""
    PERFORMANCE = "performance"
    MEMORY_EFFICIENCY = "memory_efficiency"
    COST_OPTIMIZATION = "cost_optimization"
    LATENCY_MINIMIZATION = "latency_minimization"
    THROUGHPUT_MAXIMIZATION = "throughput_maximization"


@dataclass
class CacheAccessPattern:
    """Cache access pattern analysis"""
    key: str
    access_count: int
    hit_rate: float
    miss_rate: float
    average_interval: float
    variance: float
    trending_direction: str  # up, down, stable
    access_times: List[float]
    size_bytes: int
    ttl_effectiveness: float


@dataclass
class RedisPerformanceMetrics:
    """Comprehensive Redis performance metrics"""
    timestamp: float
    operations_per_second: float
    memory_usage_mb: float
    memory_fragmentation_ratio: float
    cache_hit_rate: float
    network_throughput_mbps: float
    cpu_usage_percent: float
    connection_count: int
    key_count: int
    expired_keys_per_second: float
    evicted_keys_per_second: float
    replication_lag_ms: float
    slow_queries_count: int
    blocked_clients: int


@dataclass
class CacheOptimizationRecommendation:
    """Cache optimization recommendation"""
    key_pattern: str
    current_performance: float
    predicted_improvement: float
    recommended_action: str
    confidence_score: float
    estimated_impact: Dict[str, float]
    implementation_complexity: str  # low, medium, high
    resource_requirements: Dict[str, Any]


@dataclass
class PredictiveCacheEntry:
    """Predictive cache entry"""
    key: str
    predicted_access_time: float
    confidence: float
    data_size: int
    computation_cost: float
    business_value: float
    prefetch_priority: int


class AdvancedRedisIntelligenceEngine(IntelligenceService, XORBService):
    """AI-powered Redis intelligence engine for optimization and prediction"""

    def __init__(self, orchestrator: AdvancedRedisOrchestrator):
        super().__init__(service_type=ServiceType.INTELLIGENCE)
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(__name__)

        # ML Models (with fallbacks if sklearn not available)
        self.ml_models = {}
        self.model_last_trained = {}
        self.training_data_cache = defaultdict(list)

        # Access pattern tracking
        self.access_patterns: Dict[str, CacheAccessPattern] = {}
        self.performance_history: deque = deque(maxlen=1000)
        self.optimization_history: List[CacheOptimizationRecommendation] = []

        # Predictive caching
        self.prediction_cache: Dict[str, PredictiveCacheEntry] = {}
        self.prefetch_queue: deque = deque()
        self.cache_warming_strategies: Dict[str, Any] = {}

        # Intelligence configuration
        self.config = {
            "enable_ml_optimization": ML_AVAILABLE,
            "prediction_horizon_hours": 24,
            "optimization_frequency_minutes": 30,
            "min_samples_for_training": 100,
            "model_retrain_frequency_hours": 6,
            "prediction_confidence_threshold": 0.7,
            "cache_warming_threshold": 0.8,
            "anomaly_detection_sensitivity": 0.95
        }

        # Performance thresholds
        self.performance_thresholds = {
            "hit_rate_warning": 0.7,
            "hit_rate_critical": 0.5,
            "memory_usage_warning": 0.8,
            "memory_usage_critical": 0.95,
            "latency_warning_ms": 100,
            "latency_critical_ms": 500,
            "throughput_degradation": 0.2
        }

        # Active monitoring and optimization
        self.is_monitoring_active = False
        self.monitoring_tasks: List[asyncio.Task] = []

    async def initialize(self) -> bool:
        """Initialize Redis intelligence engine"""
        try:
            self.logger.info("Initializing Redis Intelligence Engine")

            # Initialize ML models if available
            if ML_AVAILABLE:
                await self._initialize_ml_models()

            # Load historical data
            await self._load_historical_data()

            # Start monitoring tasks
            await self._start_monitoring()

            self.logger.info("Redis Intelligence Engine initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Redis Intelligence Engine: {e}")
            return False

    async def _initialize_ml_models(self):
        """Initialize machine learning models"""
        if not ML_AVAILABLE:
            self.logger.warning("ML libraries not available, using fallback algorithms")
            return

        try:
            # Cache hit rate prediction model
            self.ml_models["hit_rate_predictor"] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

            # Memory usage prediction model
            self.ml_models["memory_predictor"] = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42
            )

            # Anomaly detection model
            self.ml_models["anomaly_detector"] = IsolationForest(
                contamination=0.1,
                random_state=42
            )

            # Access pattern clustering
            self.ml_models["pattern_clusterer"] = KMeans(
                n_clusters=5,
                random_state=42
            )

            # Performance trend analysis
            self.ml_models["trend_analyzer"] = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )

            self.logger.info("ML models initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize ML models: {e}")

    async def _load_historical_data(self):
        """Load historical performance and access data"""
        try:
            client = await self.orchestrator.get_optimal_client("read")

            # Load performance metrics history
            metrics_key = "intelligence:performance_history"
            historical_metrics = await client.get(metrics_key)

            if historical_metrics:
                metrics_data = pickle.loads(historical_metrics)
                self.performance_history.extend(metrics_data[-500:])  # Load last 500 points

            # Load access patterns
            patterns_key = "intelligence:access_patterns"
            patterns_data = await client.get(patterns_key)

            if patterns_data:
                self.access_patterns = pickle.loads(patterns_data)

            self.logger.info(f"Loaded {len(self.performance_history)} historical metrics and {len(self.access_patterns)} access patterns")

        except Exception as e:
            self.logger.error(f"Failed to load historical data: {e}")

    async def _start_monitoring(self):
        """Start continuous monitoring and optimization tasks"""
        self.is_monitoring_active = True

        # Performance monitoring task
        monitor_task = asyncio.create_task(self._performance_monitoring_loop())
        self.monitoring_tasks.append(monitor_task)

        # Optimization task
        optimize_task = asyncio.create_task(self._optimization_loop())
        self.monitoring_tasks.append(optimize_task)

        # Predictive caching task
        predict_task = asyncio.create_task(self._predictive_caching_loop())
        self.monitoring_tasks.append(predict_task)

        # Anomaly detection task
        anomaly_task = asyncio.create_task(self._anomaly_detection_loop())
        self.monitoring_tasks.append(anomaly_task)

        self.logger.info("Started monitoring and optimization tasks")

    async def _performance_monitoring_loop(self):
        """Continuous performance monitoring loop"""
        while self.is_monitoring_active:
            try:
                # Collect current performance metrics
                metrics = await self._collect_performance_metrics()

                if metrics:
                    self.performance_history.append(metrics)

                    # Analyze performance trends
                    if len(self.performance_history) >= 10:
                        await self._analyze_performance_trends()

                    # Check for performance alerts
                    await self._check_performance_alerts(metrics)

                await asyncio.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)

    async def _optimization_loop(self):
        """Continuous optimization loop"""
        while self.is_monitoring_active:
            try:
                # Run optimization analysis
                recommendations = await self.analyze_cache_optimization()

                if recommendations:
                    # Apply high-confidence, low-risk optimizations automatically
                    await self._apply_automatic_optimizations(recommendations)

                # Retrain ML models periodically
                if ML_AVAILABLE and len(self.performance_history) >= self.config["min_samples_for_training"]:
                    await self._retrain_ml_models()

                await asyncio.sleep(self.config["optimization_frequency_minutes"] * 60)

            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(300)

    async def _predictive_caching_loop(self):
        """Predictive caching loop"""
        while self.is_monitoring_active:
            try:
                # Generate cache predictions
                predictions = await self._generate_cache_predictions()

                # Execute high-priority prefetching
                await self._execute_predictive_prefetching(predictions)

                # Update cache warming strategies
                await self._update_cache_warming_strategies()

                await asyncio.sleep(300)  # Run every 5 minutes

            except Exception as e:
                self.logger.error(f"Predictive caching error: {e}")
                await asyncio.sleep(600)

    async def _anomaly_detection_loop(self):
        """Anomaly detection loop"""
        while self.is_monitoring_active:
            try:
                if len(self.performance_history) >= 50:
                    anomalies = await self._detect_performance_anomalies()

                    if anomalies:
                        await self._handle_performance_anomalies(anomalies)

                await asyncio.sleep(120)  # Check every 2 minutes

            except Exception as e:
                self.logger.error(f"Anomaly detection error: {e}")
                await asyncio.sleep(300)

    async def _collect_performance_metrics(self) -> Optional[RedisPerformanceMetrics]:
        """Collect comprehensive Redis performance metrics"""
        try:
            client = await self.orchestrator.get_optimal_client("read")
            info = await client.info()

            # Extract metrics from Redis INFO
            metrics = RedisPerformanceMetrics(
                timestamp=time.time(),
                operations_per_second=info.get('instantaneous_ops_per_sec', 0),
                memory_usage_mb=info.get('used_memory', 0) / (1024 * 1024),
                memory_fragmentation_ratio=info.get('mem_fragmentation_ratio', 1.0),
                cache_hit_rate=self._calculate_hit_rate(info),
                network_throughput_mbps=info.get('instantaneous_input_kbps', 0) / 1024,
                cpu_usage_percent=0,  # Would need system monitoring integration
                connection_count=info.get('connected_clients', 0),
                key_count=info.get('db0', {}).get('keys', 0),
                expired_keys_per_second=info.get('instantaneous_expired_per_sec', 0),
                evicted_keys_per_second=info.get('instantaneous_evicted_per_sec', 0),
                replication_lag_ms=0,  # Would need replication monitoring
                slow_queries_count=len(await client.slowlog_get(10)),
                blocked_clients=info.get('blocked_clients', 0)
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to collect performance metrics: {e}")
            return None

    def _calculate_hit_rate(self, info: Dict[str, Any]) -> float:
        """Calculate cache hit rate from Redis info"""
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 0)
        total = hits + misses
        return hits / total if total > 0 else 0.0

    async def _analyze_performance_trends(self):
        """Analyze performance trends using ML or statistical methods"""
        if not self.performance_history:
            return

        try:
            # Extract time series data
            timestamps = [m.timestamp for m in self.performance_history]
            hit_rates = [m.cache_hit_rate for m in self.performance_history]
            memory_usage = [m.memory_usage_mb for m in self.performance_history]
            ops_per_sec = [m.operations_per_second for m in self.performance_history]

            # Calculate trends
            hit_rate_trend = self._calculate_trend(hit_rates)
            memory_trend = self._calculate_trend(memory_usage)
            performance_trend = self._calculate_trend(ops_per_sec)

            # Store trend analysis
            trend_analysis = {
                "timestamp": time.time(),
                "hit_rate_trend": hit_rate_trend,
                "memory_trend": memory_trend,
                "performance_trend": performance_trend,
                "volatility": {
                    "hit_rate": np.std(hit_rates) if hit_rates else 0,
                    "memory": np.std(memory_usage) if memory_usage else 0,
                    "performance": np.std(ops_per_sec) if ops_per_sec else 0
                }
            }

            # Save analysis
            client = await self.orchestrator.get_optimal_client("write")
            await client.setex(
                "intelligence:trend_analysis",
                3600,
                pickle.dumps(trend_analysis)
            )

        except Exception as e:
            self.logger.error(f"Trend analysis error: {e}")

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from time series values"""
        if len(values) < 3:
            return "stable"

        # Simple linear regression slope
        x = list(range(len(values)))
        n = len(values)

        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x_sq = sum(xi * xi for xi in x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_sq - sum_x * sum_x)

        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"

    async def _check_performance_alerts(self, metrics: RedisPerformanceMetrics):
        """Check for performance alerts and notifications"""
        alerts = []

        # Hit rate alerts
        if metrics.cache_hit_rate < self.performance_thresholds["hit_rate_critical"]:
            alerts.append({
                "type": "critical",
                "metric": "hit_rate",
                "value": metrics.cache_hit_rate,
                "threshold": self.performance_thresholds["hit_rate_critical"],
                "message": f"Critical: Cache hit rate is {metrics.cache_hit_rate:.2%}"
            })
        elif metrics.cache_hit_rate < self.performance_thresholds["hit_rate_warning"]:
            alerts.append({
                "type": "warning",
                "metric": "hit_rate",
                "value": metrics.cache_hit_rate,
                "threshold": self.performance_thresholds["hit_rate_warning"],
                "message": f"Warning: Cache hit rate is {metrics.cache_hit_rate:.2%}"
            })

        # Memory usage alerts
        memory_usage_ratio = metrics.memory_usage_mb / 1024  # Assuming 1GB limit
        if memory_usage_ratio > self.performance_thresholds["memory_usage_critical"]:
            alerts.append({
                "type": "critical",
                "metric": "memory_usage",
                "value": memory_usage_ratio,
                "threshold": self.performance_thresholds["memory_usage_critical"],
                "message": f"Critical: Memory usage is {memory_usage_ratio:.1%}"
            })

        # Send alerts if any
        if alerts:
            await self._send_performance_alerts(alerts)

    async def _send_performance_alerts(self, alerts: List[Dict[str, Any]]):
        """Send performance alerts via pub/sub"""
        try:
            for alert in alerts:
                await self.orchestrator.pubsub_coordinator.publish(
                    "performance_alerts",
                    alert
                )

            self.logger.warning(f"Sent {len(alerts)} performance alerts")

        except Exception as e:
            self.logger.error(f"Failed to send alerts: {e}")

    async def analyze_cache_optimization(self) -> List[CacheOptimizationRecommendation]:
        """Analyze cache usage and generate optimization recommendations"""
        recommendations = []

        try:
            # Analyze access patterns
            pattern_analysis = await self._analyze_access_patterns()

            # Generate recommendations based on patterns
            for pattern_key, analysis in pattern_analysis.items():
                if analysis["optimization_potential"] > 0.3:
                    recommendation = await self._generate_optimization_recommendation(
                        pattern_key,
                        analysis
                    )
                    if recommendation:
                        recommendations.append(recommendation)

            # ML-based recommendations if available
            if ML_AVAILABLE and len(self.performance_history) >= self.config["min_samples_for_training"]:
                ml_recommendations = await self._generate_ml_recommendations()
                recommendations.extend(ml_recommendations)

            # Sort by potential impact
            recommendations.sort(
                key=lambda x: x.predicted_improvement * x.confidence_score,
                reverse=True
            )

            return recommendations[:10]  # Return top 10 recommendations

        except Exception as e:
            self.logger.error(f"Cache optimization analysis error: {e}")
            return []

    async def _analyze_access_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Analyze Redis access patterns"""
        client = await self.orchestrator.get_optimal_client("read")
        pattern_analysis = {}

        try:
            # Get recent access data from Redis
            access_keys = await client.keys("access_pattern:*")

            for key in access_keys[:100]:  # Limit analysis to prevent performance issues
                pattern_data = await client.get(key)
                if pattern_data:
                    data = pickle.loads(pattern_data)

                    # Calculate optimization metrics
                    analysis = {
                        "hit_rate": data.get("hits", 0) / max(data.get("total_accesses", 1), 1),
                        "access_frequency": len(data.get("access_times", [])),
                        "data_size": data.get("size_bytes", 0),
                        "ttl_effectiveness": self._calculate_ttl_effectiveness(data),
                        "optimization_potential": 0.0
                    }

                    # Calculate optimization potential
                    if analysis["hit_rate"] < 0.5:
                        analysis["optimization_potential"] += 0.4

                    if analysis["access_frequency"] > 100:
                        analysis["optimization_potential"] += 0.3

                    if analysis["ttl_effectiveness"] < 0.7:
                        analysis["optimization_potential"] += 0.3

                    pattern_analysis[key] = analysis

            return pattern_analysis

        except Exception as e:
            self.logger.error(f"Access pattern analysis error: {e}")
            return {}

    def _calculate_ttl_effectiveness(self, data: Dict[str, Any]) -> float:
        """Calculate TTL effectiveness score"""
        try:
            access_times = data.get("access_times", [])
            if len(access_times) < 2:
                return 1.0

            # Calculate intervals between accesses
            intervals = [access_times[i+1] - access_times[i] for i in range(len(access_times)-1)]
            avg_interval = statistics.mean(intervals)

            # Compare with TTL
            ttl = data.get("ttl", 3600)

            if avg_interval > ttl:
                return 0.3  # TTL too short
            elif avg_interval < ttl * 0.1:
                return 0.5  # TTL too long
            else:
                return 1.0  # TTL is appropriate

        except Exception:
            return 1.0

    async def _generate_optimization_recommendation(
        self,
        pattern_key: str,
        analysis: Dict[str, Any]
    ) -> Optional[CacheOptimizationRecommendation]:
        """Generate specific optimization recommendation"""
        try:
            if analysis["hit_rate"] < 0.5:
                # Low hit rate - recommend TTL adjustment or removal
                return CacheOptimizationRecommendation(
                    key_pattern=pattern_key,
                    current_performance=analysis["hit_rate"],
                    predicted_improvement=0.3,
                    recommended_action="Adjust TTL or consider cache removal",
                    confidence_score=0.8,
                    estimated_impact={
                        "memory_savings": analysis["data_size"] * 0.5,
                        "performance_improvement": 0.15
                    },
                    implementation_complexity="low",
                    resource_requirements={"cpu": "minimal", "memory": "reduced"}
                )

            elif analysis["access_frequency"] > 100 and analysis["hit_rate"] > 0.8:
                # High frequency, high hit rate - recommend memory tier promotion
                return CacheOptimizationRecommendation(
                    key_pattern=pattern_key,
                    current_performance=analysis["hit_rate"],
                    predicted_improvement=0.2,
                    recommended_action="Promote to memory tier",
                    confidence_score=0.9,
                    estimated_impact={
                        "latency_improvement": 0.5,
                        "memory_cost": analysis["data_size"]
                    },
                    implementation_complexity="medium",
                    resource_requirements={"cpu": "minimal", "memory": "increased"}
                )

            return None

        except Exception as e:
            self.logger.error(f"Recommendation generation error: {e}")
            return None

    async def _generate_ml_recommendations(self) -> List[CacheOptimizationRecommendation]:
        """Generate ML-based optimization recommendations"""
        if not ML_AVAILABLE:
            return []

        recommendations = []

        try:
            # Prepare training data
            features, targets = self._prepare_ml_training_data()

            if len(features) < self.config["min_samples_for_training"]:
                return []

            # Train predictive model
            model = self.ml_models["hit_rate_predictor"]
            model.fit(features, targets)

            # Generate predictions for potential optimizations
            # This is a simplified example - in practice, you'd evaluate many scenarios
            optimization_scenarios = self._generate_optimization_scenarios()

            for scenario in optimization_scenarios:
                prediction = model.predict([scenario["features"]])[0]

                if prediction > scenario["current_performance"] * 1.1:  # 10% improvement threshold
                    recommendation = CacheOptimizationRecommendation(
                        key_pattern=scenario["pattern"],
                        current_performance=scenario["current_performance"],
                        predicted_improvement=prediction - scenario["current_performance"],
                        recommended_action=scenario["action"],
                        confidence_score=min(0.9, prediction),
                        estimated_impact=scenario["impact"],
                        implementation_complexity=scenario["complexity"],
                        resource_requirements=scenario["resources"]
                    )
                    recommendations.append(recommendation)

            return recommendations[:5]  # Return top 5 ML recommendations

        except Exception as e:
            self.logger.error(f"ML recommendations error: {e}")
            return []

    def _prepare_ml_training_data(self) -> Tuple[List[List[float]], List[float]]:
        """Prepare training data for ML models"""
        features = []
        targets = []

        for metrics in self.performance_history:
            # Feature vector: [ops_per_sec, memory_usage, connection_count, key_count]
            feature_vector = [
                metrics.operations_per_second,
                metrics.memory_usage_mb,
                metrics.connection_count,
                metrics.key_count
            ]

            # Target: cache hit rate
            target = metrics.cache_hit_rate

            features.append(feature_vector)
            targets.append(target)

        return features, targets

    def _generate_optimization_scenarios(self) -> List[Dict[str, Any]]:
        """Generate optimization scenarios for ML evaluation"""
        # This is a simplified example - real implementation would be more sophisticated
        return [
            {
                "pattern": "user_sessions:*",
                "features": [100, 500, 50, 1000],
                "current_performance": 0.7,
                "action": "Increase TTL",
                "impact": {"memory": 0.1, "performance": 0.15},
                "complexity": "low",
                "resources": {"memory": "increased"}
            },
            {
                "pattern": "api_cache:*",
                "features": [200, 300, 30, 500],
                "current_performance": 0.6,
                "action": "Implement compression",
                "impact": {"memory": -0.3, "performance": 0.1},
                "complexity": "medium",
                "resources": {"cpu": "increased", "memory": "reduced"}
            }
        ]

    async def _apply_automatic_optimizations(self, recommendations: List[CacheOptimizationRecommendation]):
        """Apply low-risk optimizations automatically"""
        for rec in recommendations:
            if (rec.confidence_score > 0.8 and
                rec.implementation_complexity == "low" and
                rec.predicted_improvement > 0.1):

                try:
                    success = await self._execute_optimization(rec)
                    if success:
                        self.logger.info(f"Applied automatic optimization: {rec.recommended_action}")

                except Exception as e:
                    self.logger.error(f"Failed to apply optimization: {e}")

    async def _execute_optimization(self, recommendation: CacheOptimizationRecommendation) -> bool:
        """Execute specific optimization recommendation"""
        try:
            client = await self.orchestrator.get_optimal_client("write")

            if "TTL" in recommendation.recommended_action:
                # Adjust TTL for matching keys
                pattern = recommendation.key_pattern.replace("*", "")
                keys = await client.keys(f"{pattern}*")

                for key in keys[:10]:  # Limit to prevent performance impact
                    await client.expire(key, 7200)  # Example: set to 2 hours

                return True

            # Add more optimization implementations here
            return False

        except Exception as e:
            self.logger.error(f"Optimization execution error: {e}")
            return False

    async def predict_cache_performance(
        self,
        time_horizon_hours: int = 24,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Predict cache performance for specified time horizon"""
        try:
            if not ML_AVAILABLE or len(self.performance_history) < 50:
                return self._fallback_performance_prediction(time_horizon_hours)

            # Prepare time series data
            features, targets = self._prepare_time_series_data()

            # Train prediction model
            model = self.ml_models["trend_analyzer"]
            model.fit(features, targets)

            # Generate predictions
            future_points = time_horizon_hours * 2  # Every 30 minutes
            predictions = []

            last_features = features[-1]

            for i in range(future_points):
                prediction = model.predict([last_features])[0]
                predictions.append({
                    "timestamp": time.time() + (i * 1800),  # 30-minute intervals
                    "predicted_hit_rate": max(0, min(1, prediction)),
                    "confidence": confidence_threshold
                })

                # Update features for next prediction (simplified)
                last_features = last_features[1:] + [prediction]

            return {
                "predictions": predictions,
                "model_accuracy": self._calculate_model_accuracy(),
                "confidence_interval": 0.1,
                "recommendation": self._generate_performance_recommendation(predictions)
            }

        except Exception as e:
            self.logger.error(f"Performance prediction error: {e}")
            return self._fallback_performance_prediction(time_horizon_hours)

    def _prepare_time_series_data(self) -> Tuple[List[List[float]], List[float]]:
        """Prepare time series data for prediction"""
        features = []
        targets = []

        # Use sliding window approach
        window_size = 5

        for i in range(window_size, len(self.performance_history)):
            # Features: hit rates from previous windows
            feature_window = [
                self.performance_history[i-j].cache_hit_rate
                for j in range(window_size, 0, -1)
            ]

            # Target: next hit rate
            target = self.performance_history[i].cache_hit_rate

            features.append(feature_window)
            targets.append(target)

        return features, targets

    def _fallback_performance_prediction(self, time_horizon_hours: int) -> Dict[str, Any]:
        """Fallback prediction using statistical methods"""
        if not self.performance_history:
            return {"error": "No historical data available"}

        # Simple moving average prediction
        recent_hit_rates = [m.cache_hit_rate for m in list(self.performance_history)[-10:]]
        avg_hit_rate = statistics.mean(recent_hit_rates) if recent_hit_rates else 0.5

        predictions = []
        for i in range(time_horizon_hours * 2):
            predictions.append({
                "timestamp": time.time() + (i * 1800),
                "predicted_hit_rate": avg_hit_rate,
                "confidence": 0.6
            })

        return {
            "predictions": predictions,
            "model_accuracy": 0.6,
            "confidence_interval": 0.2,
            "recommendation": "Consider enabling ML for better predictions"
        }

    def _calculate_model_accuracy(self) -> float:
        """Calculate model accuracy based on recent predictions vs actual"""
        # Simplified accuracy calculation
        return 0.85  # In practice, this would compare predictions to actual values

    def _generate_performance_recommendation(self, predictions: List[Dict[str, Any]]) -> str:
        """Generate recommendation based on performance predictions"""
        avg_predicted_rate = statistics.mean([p["predicted_hit_rate"] for p in predictions])

        if avg_predicted_rate < 0.6:
            return "Cache performance is predicted to degrade. Consider optimization."
        elif avg_predicted_rate > 0.85:
            return "Cache performance looks optimal."
        else:
            return "Cache performance is stable but could be improved."

    async def _generate_cache_predictions(self) -> List[PredictiveCacheEntry]:
        """Generate predictive cache entries for prefetching"""
        predictions = []

        try:
            # Analyze access patterns to predict future cache needs
            for key, pattern in self.access_patterns.items():
                if pattern.trending_direction == "up" and pattern.hit_rate > 0.7:
                    # Predict when this key will be accessed next
                    predicted_time = time.time() + pattern.average_interval

                    prediction = PredictiveCacheEntry(
                        key=key,
                        predicted_access_time=predicted_time,
                        confidence=min(0.9, pattern.hit_rate),
                        data_size=pattern.size_bytes,
                        computation_cost=self._estimate_computation_cost(key),
                        business_value=self._estimate_business_value(key),
                        prefetch_priority=self._calculate_prefetch_priority(pattern)
                    )

                    predictions.append(prediction)

            # Sort by priority
            predictions.sort(key=lambda x: x.prefetch_priority, reverse=True)

            return predictions[:50]  # Return top 50 predictions

        except Exception as e:
            self.logger.error(f"Cache prediction generation error: {e}")
            return []

    def _estimate_computation_cost(self, key: str) -> float:
        """Estimate computational cost of regenerating cache entry"""
        # Simplified cost estimation based on key patterns
        if "scan_result" in key:
            return 0.9  # High cost
        elif "user_session" in key:
            return 0.3  # Low cost
        elif "threat_intel" in key:
            return 0.7  # Medium cost
        else:
            return 0.5  # Default medium cost

    def _estimate_business_value(self, key: str) -> float:
        """Estimate business value of cache entry"""
        # Simplified business value estimation
        if "critical" in key or "security" in key:
            return 0.9  # High value
        elif "user" in key:
            return 0.7  # Medium-high value
        elif "analytics" in key:
            return 0.5  # Medium value
        else:
            return 0.3  # Low value

    def _calculate_prefetch_priority(self, pattern: CacheAccessPattern) -> int:
        """Calculate prefetch priority score"""
        priority = 0

        # High hit rate = higher priority
        priority += int(pattern.hit_rate * 50)

        # High access frequency = higher priority
        priority += min(50, pattern.access_count // 10)

        # Trending up = higher priority
        if pattern.trending_direction == "up":
            priority += 20

        # Large data size = lower priority (to avoid memory pressure)
        priority -= min(30, pattern.size_bytes // 1024)

        return max(0, priority)

    async def _execute_predictive_prefetching(self, predictions: List[PredictiveCacheEntry]):
        """Execute predictive prefetching based on predictions"""
        high_priority_predictions = [
            p for p in predictions
            if p.prefetch_priority > 70 and p.confidence > self.config["prediction_confidence_threshold"]
        ]

        for prediction in high_priority_predictions[:10]:  # Limit to top 10
            try:
                # Check if key needs prefetching
                client = await self.orchestrator.get_optimal_client("read")
                exists = await client.exists(prediction.key)

                if not exists:
                    # Add to prefetch queue
                    self.prefetch_queue.append(prediction)

                    # Execute prefetch if conditions are met
                    if len(self.prefetch_queue) >= 5:
                        await self._execute_prefetch_batch()

            except Exception as e:
                self.logger.error(f"Predictive prefetching error for {prediction.key}: {e}")

    async def _execute_prefetch_batch(self):
        """Execute batch prefetching"""
        batch = []
        for _ in range(min(5, len(self.prefetch_queue))):
            if self.prefetch_queue:
                batch.append(self.prefetch_queue.popleft())

        for prediction in batch:
            try:
                # Simulate cache warming (in practice, this would regenerate the data)
                client = await self.orchestrator.get_optimal_client("write")
                await client.setex(
                    prediction.key,
                    3600,  # 1 hour TTL
                    f"prefetched_data_{time.time()}"
                )

                self.logger.info(f"Prefetched cache entry: {prediction.key}")

            except Exception as e:
                self.logger.error(f"Prefetch execution error: {e}")

    async def _update_cache_warming_strategies(self):
        """Update cache warming strategies based on performance"""
        try:
            # Analyze which cache warming strategies are most effective
            strategies = {}

            for key, pattern in self.access_patterns.items():
                namespace = key.split(":")[0] if ":" in key else "default"

                if namespace not in strategies:
                    strategies[namespace] = {
                        "total_accesses": 0,
                        "hit_rate": 0,
                        "warming_effectiveness": 0
                    }

                strategies[namespace]["total_accesses"] += pattern.access_count
                strategies[namespace]["hit_rate"] += pattern.hit_rate

            # Calculate average hit rates per namespace
            for namespace, stats in strategies.items():
                if stats["total_accesses"] > 0:
                    stats["hit_rate"] /= stats["total_accesses"]
                    stats["warming_effectiveness"] = stats["hit_rate"] * 0.8 + (stats["total_accesses"] / 1000) * 0.2

            self.cache_warming_strategies = strategies

        except Exception as e:
            self.logger.error(f"Cache warming strategy update error: {e}")

    async def _detect_performance_anomalies(self) -> List[Dict[str, Any]]:
        """Detect performance anomalies using ML or statistical methods"""
        anomalies = []

        try:
            if ML_AVAILABLE and len(self.performance_history) >= 50:
                # ML-based anomaly detection
                features = [[
                    m.cache_hit_rate,
                    m.memory_usage_mb,
                    m.operations_per_second,
                    m.connection_count
                ] for m in self.performance_history]

                detector = self.ml_models["anomaly_detector"]
                detector.fit(features)

                # Check recent metrics
                recent_features = features[-10:]
                anomaly_scores = detector.decision_function(recent_features)

                for i, score in enumerate(anomaly_scores):
                    if score < -0.5:  # Anomaly threshold
                        metrics = list(self.performance_history)[-10 + i]
                        anomalies.append({
                            "timestamp": metrics.timestamp,
                            "type": "ml_detected",
                            "score": score,
                            "metrics": asdict(metrics),
                            "severity": "high" if score < -0.7 else "medium"
                        })

            else:
                # Statistical anomaly detection
                anomalies.extend(await self._statistical_anomaly_detection())

            return anomalies

        except Exception as e:
            self.logger.error(f"Anomaly detection error: {e}")
            return []

    async def _statistical_anomaly_detection(self) -> List[Dict[str, Any]]:
        """Statistical anomaly detection using z-scores"""
        anomalies = []

        if len(self.performance_history) < 20:
            return anomalies

        # Calculate z-scores for key metrics
        hit_rates = [m.cache_hit_rate for m in self.performance_history]
        memory_usage = [m.memory_usage_mb for m in self.performance_history]

        hit_rate_mean = statistics.mean(hit_rates)
        hit_rate_std = statistics.stdev(hit_rates) if len(hit_rates) > 1 else 0

        memory_mean = statistics.mean(memory_usage)
        memory_std = statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0

        # Check recent metrics for anomalies
        recent_metrics = list(self.performance_history)[-5:]

        for metrics in recent_metrics:
            hit_rate_z = abs(metrics.cache_hit_rate - hit_rate_mean) / max(hit_rate_std, 0.01)
            memory_z = abs(metrics.memory_usage_mb - memory_mean) / max(memory_std, 0.01)

            if hit_rate_z > 3 or memory_z > 3:  # 3-sigma rule
                anomalies.append({
                    "timestamp": metrics.timestamp,
                    "type": "statistical",
                    "hit_rate_z_score": hit_rate_z,
                    "memory_z_score": memory_z,
                    "metrics": asdict(metrics),
                    "severity": "high" if max(hit_rate_z, memory_z) > 4 else "medium"
                })

        return anomalies

    async def _handle_performance_anomalies(self, anomalies: List[Dict[str, Any]]):
        """Handle detected performance anomalies"""
        for anomaly in anomalies:
            try:
                # Log anomaly
                self.logger.warning(f"Performance anomaly detected: {anomaly}")

                # Send alert
                await self.orchestrator.pubsub_coordinator.publish(
                    "performance_anomalies",
                    anomaly
                )

                # Auto-remediation for specific anomaly types
                if anomaly.get("severity") == "high":
                    await self._attempt_auto_remediation(anomaly)

            except Exception as e:
                self.logger.error(f"Anomaly handling error: {e}")

    async def _attempt_auto_remediation(self, anomaly: Dict[str, Any]):
        """Attempt automatic remediation for severe anomalies"""
        try:
            metrics = anomaly.get("metrics", {})

            # High memory usage - trigger cleanup
            if metrics.get("memory_usage_mb", 0) > 800:  # > 800MB
                await self._trigger_memory_cleanup()

            # Low hit rate - clear problematic cache patterns
            if metrics.get("cache_hit_rate", 1) < 0.3:
                await self._clear_low_performing_cache()

            self.logger.info(f"Attempted auto-remediation for anomaly: {anomaly['type']}")

        except Exception as e:
            self.logger.error(f"Auto-remediation error: {e}")

    async def _trigger_memory_cleanup(self):
        """Trigger memory cleanup operations"""
        try:
            client = await self.orchestrator.get_optimal_client("write")

            # Remove expired keys
            await client.execute_command("MEMORY", "PURGE")

            # Clear low-value cache entries
            low_value_patterns = ["temp:*", "cache:low_priority:*"]
            for pattern in low_value_patterns:
                keys = await client.keys(pattern)
                if keys:
                    await client.delete(*keys[:100])  # Limit deletion

            self.logger.info("Memory cleanup triggered")

        except Exception as e:
            self.logger.error(f"Memory cleanup error: {e}")

    async def _clear_low_performing_cache(self):
        """Clear cache patterns with poor performance"""
        try:
            # Identify and clear cache entries with hit rate < 0.2
            client = await self.orchestrator.get_optimal_client("write")

            for key, pattern in self.access_patterns.items():
                if pattern.hit_rate < 0.2 and pattern.access_count > 10:
                    await client.delete(key)
                    self.logger.info(f"Cleared low-performing cache: {key}")

        except Exception as e:
            self.logger.error(f"Low-performing cache cleanup error: {e}")

    async def _retrain_ml_models(self):
        """Retrain ML models with latest data"""
        if not ML_AVAILABLE or len(self.performance_history) < self.config["min_samples_for_training"]:
            return

        try:
            # Retrain hit rate predictor
            features, targets = self._prepare_ml_training_data()

            if len(features) >= self.config["min_samples_for_training"]:
                # Split data for validation
                X_train, X_test, y_train, y_test = train_test_split(
                    features, targets, test_size=0.2, random_state=42
                )

                # Train model
                model = self.ml_models["hit_rate_predictor"]
                model.fit(X_train, y_train)

                # Evaluate model
                predictions = model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)

                self.logger.info(f"Retrained hit rate predictor. MSE: {mse:.4f}")

                # Update last trained timestamp
                self.model_last_trained["hit_rate_predictor"] = time.time()

            # Retrain other models...

        except Exception as e:
            self.logger.error(f"Model retraining error: {e}")

    async def get_intelligence_report(self) -> Dict[str, Any]:
        """Get comprehensive intelligence report"""
        try:
            # Current performance metrics
            current_metrics = list(self.performance_history)[-1] if self.performance_history else None

            # Performance trends
            trend_analysis = await self._analyze_performance_trends()

            # Optimization recommendations
            recommendations = await self.analyze_cache_optimization()

            # Cache predictions
            predictions = await self.predict_cache_performance()

            # Anomaly status
            anomalies = await self._detect_performance_anomalies()

            return {
                "timestamp": time.time(),
                "current_performance": asdict(current_metrics) if current_metrics else None,
                "performance_trends": trend_analysis,
                "optimization_recommendations": [asdict(r) for r in recommendations[:5]],
                "performance_predictions": predictions,
                "anomalies": anomalies,
                "ml_status": {
                    "enabled": ML_AVAILABLE,
                    "models_trained": len(self.model_last_trained),
                    "last_training": max(self.model_last_trained.values()) if self.model_last_trained else None
                },
                "cache_statistics": {
                    "tracked_patterns": len(self.access_patterns),
                    "performance_history_points": len(self.performance_history),
                    "active_predictions": len(self.prediction_cache),
                    "prefetch_queue_size": len(self.prefetch_queue)
                },
                "system_health": {
                    "monitoring_active": self.is_monitoring_active,
                    "active_tasks": len(self.monitoring_tasks),
                    "orchestrator_status": await self.orchestrator.get_cluster_status()
                }
            }

        except Exception as e:
            self.logger.error(f"Intelligence report generation error: {e}")
            return {"error": str(e), "timestamp": time.time()}

    async def shutdown(self):
        """Shutdown Redis intelligence engine"""
        self.logger.info("Shutting down Redis Intelligence Engine")

        self.is_monitoring_active = False

        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()

        # Save state before shutdown
        try:
            await self._save_state()
        except Exception as e:
            self.logger.error(f"Error saving state during shutdown: {e}")

        self.logger.info("Redis Intelligence Engine shutdown complete")

    async def _save_state(self):
        """Save current state to Redis"""
        try:
            client = await self.orchestrator.get_optimal_client("write")

            # Save performance history
            history_data = list(self.performance_history)[-500:]  # Keep last 500 points
            await client.setex(
                "intelligence:performance_history",
                86400 * 7,  # 7 days
                pickle.dumps([asdict(m) for m in history_data])
            )

            # Save access patterns
            await client.setex(
                "intelligence:access_patterns",
                86400 * 7,  # 7 days
                pickle.dumps(self.access_patterns)
            )

            self.logger.info("State saved successfully")

        except Exception as e:
            self.logger.error(f"State saving error: {e}")


# Factory function for dependency injection
async def create_redis_intelligence_engine() -> AdvancedRedisIntelligenceEngine:
    """Create Redis intelligence engine instance"""
    orchestrator = await get_redis_orchestrator()
    engine = AdvancedRedisIntelligenceEngine(orchestrator)
    await engine.initialize()
    return engine
