#!/usr/bin/env python3
"""
Xorb Intelligent Scheduler with Leaky Bucket and Predictive Pacing
Optimizes scan scheduling based on system resources, historical patterns, and predictive analysis
"""

import asyncio
import json
import logging
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import asyncpg
import aioredis
import numpy as np
from nats.aio.client import Client as NATS
from prometheus_client import Counter, Histogram, Gauge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
scheduled_scans = Counter('xorb_scheduler_scans_scheduled_total', 'Total scheduled scans', ['priority'])
queue_size = Gauge('xorb_scheduler_queue_size', 'Current scheduler queue size')
prediction_accuracy = Histogram('xorb_scheduler_prediction_accuracy', 'Prediction accuracy score')
resource_utilization = Gauge('xorb_scheduler_resource_utilization', 'Predicted resource utilization')

class ScanPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class ScanRequest:
    """Scan request with scheduling metadata"""
    id: str
    organization_id: str
    targets: List[str]
    priority: ScanPriority
    estimated_duration: int  # seconds
    resource_requirements: Dict[str, float]
    submitted_at: datetime
    scheduled_for: Optional[datetime] = None
    attempts: int = 0
    max_attempts: int = 3

@dataclass
class SystemMetrics:
    """Current system metrics for scheduling decisions"""
    cpu_usage: float
    memory_usage: float
    active_scans: int
    queue_length: int
    avg_scan_duration: float
    timestamp: datetime

@dataclass
class SchedulingPrediction:
    """Predictive scheduling analysis"""
    optimal_start_time: datetime
    expected_completion: datetime
    confidence_score: float
    resource_impact: Dict[str, float]
    reasoning: str

class LeakyBucketRateLimiter:
    """Leaky bucket implementation for scan rate limiting"""
    
    def __init__(self, capacity: int, leak_rate: float):
        self.capacity = capacity
        self.leak_rate = leak_rate  # tokens per second
        self.current_tokens = capacity
        self.last_update = time.time()
    
    def can_consume(self, tokens: int = 1) -> Tuple[bool, float]:
        """Check if tokens can be consumed, return (allowed, wait_time)"""
        self._leak()
        
        if self.current_tokens >= tokens:
            self.current_tokens -= tokens
            return True, 0.0
        else:
            # Calculate wait time until enough tokens are available
            needed_tokens = tokens - self.current_tokens
            wait_time = needed_tokens / self.leak_rate
            return False, wait_time
    
    def _leak(self):
        """Leak tokens based on time elapsed"""
        now = time.time()
        elapsed = now - self.last_update
        leaked = elapsed * self.leak_rate
        
        self.current_tokens = min(self.capacity, self.current_tokens + leaked)
        self.last_update = now
    
    def get_status(self) -> Dict[str, float]:
        """Get current bucket status"""
        self._leak()
        return {
            "current_tokens": self.current_tokens,
            "capacity": self.capacity,
            "utilization": (self.capacity - self.current_tokens) / self.capacity,
            "estimated_refill_time": (self.capacity - self.current_tokens) / self.leak_rate
        }

class PredictiveAnalyzer:
    """Analyzes historical data to predict optimal scheduling"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.cpu_model = LinearRegression()
        self.duration_model = LinearRegression()
        self.trained = False
        self.historical_data = []
    
    def add_historical_data(self, scan_data: Dict):
        """Add historical scan data for training"""
        self.historical_data.append(scan_data)
        
        # Retrain if we have enough data
        if len(self.historical_data) >= 50:
            self.train_models()
    
    def train_models(self):
        """Train predictive models on historical data"""
        if len(self.historical_data) < 10:
            return
        
        try:
            # Prepare training data
            features = []
            cpu_targets = []
            duration_targets = []
            
            for data in self.historical_data[-100:]:  # Use last 100 data points
                feature_vector = [
                    data.get('target_count', 1),
                    data.get('template_count', 10),
                    data.get('hour_of_day', 12),
                    data.get('day_of_week', 3),
                    data.get('concurrent_scans', 1),
                    data.get('queue_size', 0)
                ]
                features.append(feature_vector)
                cpu_targets.append(data.get('peak_cpu_usage', 50))
                duration_targets.append(data.get('actual_duration', 600))
            
            features = np.array(features)
            cpu_targets = np.array(cpu_targets)
            duration_targets = np.array(duration_targets)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train models
            self.cpu_model.fit(features_scaled, cpu_targets)
            self.duration_model.fit(features_scaled, duration_targets)
            
            self.trained = True
            logger.info("Predictive models trained successfully")
            
        except Exception as e:
            logger.error(f"Failed to train models: {e}")
    
    def predict_scan_impact(self, scan_request: ScanRequest, current_metrics: SystemMetrics) -> SchedulingPrediction:
        """Predict the impact and optimal timing for a scan"""
        
        if not self.trained:
            # Fallback to heuristic prediction
            return self._heuristic_prediction(scan_request, current_metrics)
        
        try:
            # Prepare feature vector
            now = datetime.now()
            feature_vector = np.array([[
                len(scan_request.targets),
                scan_request.resource_requirements.get('templates', 10),
                now.hour,
                now.weekday(),
                current_metrics.active_scans,
                current_metrics.queue_length
            ]])
            
            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Make predictions
            predicted_cpu = self.cpu_model.predict(feature_vector_scaled)[0]
            predicted_duration = self.duration_model.predict(feature_vector_scaled)[0]
            
            # Calculate optimal start time based on current load and predictions
            optimal_delay = self._calculate_optimal_delay(
                current_metrics, predicted_cpu, predicted_duration
            )
            
            optimal_start = now + timedelta(seconds=optimal_delay)
            expected_completion = optimal_start + timedelta(seconds=predicted_duration)
            
            # Calculate confidence based on model accuracy
            confidence = self._calculate_confidence(scan_request, current_metrics)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                predicted_cpu, predicted_duration, optimal_delay, current_metrics
            )
            
            return SchedulingPrediction(
                optimal_start_time=optimal_start,
                expected_completion=expected_completion,
                confidence_score=confidence,
                resource_impact={
                    "cpu": predicted_cpu,
                    "duration": predicted_duration,
                    "memory": scan_request.resource_requirements.get('memory', 1.0)
                },
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Prediction failed, using heuristic: {e}")
            return self._heuristic_prediction(scan_request, current_metrics)
    
    def _heuristic_prediction(self, scan_request: ScanRequest, current_metrics: SystemMetrics) -> SchedulingPrediction:
        """Fallback heuristic prediction when ML models aren't available"""
        
        # Simple heuristic based on current load and scan size
        target_count = len(scan_request.targets)
        estimated_duration = min(1800, max(300, target_count * 30))  # 5min to 30min
        estimated_cpu = min(90, 20 + target_count * 2)
        
        # Delay if system is overloaded
        delay = 0
        if current_metrics.cpu_usage > 80:
            delay += 300  # 5 minutes
        if current_metrics.active_scans > 10:
            delay += current_metrics.active_scans * 30
        
        optimal_start = datetime.now() + timedelta(seconds=delay)
        expected_completion = optimal_start + timedelta(seconds=estimated_duration)
        
        return SchedulingPrediction(
            optimal_start_time=optimal_start,
            expected_completion=expected_completion,
            confidence_score=0.6,  # Lower confidence for heuristic
            resource_impact={"cpu": estimated_cpu, "duration": estimated_duration},
            reasoning=f"Heuristic: {target_count} targets, {delay}s delay due to load"
        )
    
    def _calculate_optimal_delay(self, metrics: SystemMetrics, predicted_cpu: float, predicted_duration: float) -> int:
        """Calculate optimal delay based on current system state"""
        
        delay = 0
        
        # CPU-based delay
        if metrics.cpu_usage + predicted_cpu > 85:
            delay += max(300, (metrics.cpu_usage + predicted_cpu - 85) * 10)
        
        # Queue-based delay
        if metrics.queue_length > 20:
            delay += (metrics.queue_length - 20) * 60
        
        # Active scans delay
        if metrics.active_scans > 15:
            delay += (metrics.active_scans - 15) * 120
        
        return min(delay, 3600)  # Max 1 hour delay
    
    def _calculate_confidence(self, scan_request: ScanRequest, metrics: SystemMetrics) -> float:
        """Calculate confidence score for the prediction"""
        
        base_confidence = 0.8 if self.trained else 0.6
        
        # Reduce confidence for unusual conditions
        if metrics.cpu_usage > 90:
            base_confidence *= 0.8
        if metrics.active_scans > 20:
            base_confidence *= 0.9
        if len(scan_request.targets) > 100:
            base_confidence *= 0.85
        
        return max(0.3, min(1.0, base_confidence))
    
    def _generate_reasoning(self, cpu: float, duration: float, delay: int, metrics: SystemMetrics) -> str:
        """Generate human-readable reasoning for the scheduling decision"""
        
        reasons = []
        
        if delay > 0:
            reasons.append(f"Delayed {delay}s due to system load")
        
        if cpu > 70:
            reasons.append(f"High CPU impact predicted ({cpu:.1f}%)")
        
        if metrics.active_scans > 10:
            reasons.append(f"Queue consideration ({metrics.active_scans} active scans)")
        
        if duration > 1200:
            reasons.append(f"Long duration scan ({duration/60:.1f} min)")
        
        return "; ".join(reasons) if reasons else "Optimal conditions for immediate scheduling"

class IntelligentScheduler:
    """Main intelligent scheduler with leaky bucket and predictive capabilities"""
    
    def __init__(self):
        self.db_pool = None
        self.redis = None
        self.nats = None
        
        # Scheduling components
        self.rate_limiter = LeakyBucketRateLimiter(capacity=50, leak_rate=0.5)  # 50 scans max, 0.5/sec leak
        self.predictor = PredictiveAnalyzer()
        
        # In-memory queues by priority
        self.priority_queues = {
            ScanPriority.CRITICAL: [],
            ScanPriority.HIGH: [],
            ScanPriority.MEDIUM: [],
            ScanPriority.LOW: []
        }
        
        # Scheduling state
        self.running = False
        self.current_metrics = SystemMetrics(
            cpu_usage=0, memory_usage=0, active_scans=0,
            queue_length=0, avg_scan_duration=600, timestamp=datetime.now()
        )
    
    async def initialize(self):
        """Initialize database and messaging connections"""
        
        # Database connection
        database_url = "postgresql://xorb:xorb_secure_2024@postgres:5432/xorb_ptaas"
        self.db_pool = await asyncpg.create_pool(database_url, min_size=5, max_size=10)
        
        # Redis connection
        self.redis = await aioredis.create_redis_pool("redis://redis:6379/0")
        
        # NATS connection
        self.nats = NATS()
        await self.nats.connect("nats://nats:4222")
        
        # Subscribe to scan requests
        await self.nats.subscribe("scans.request", cb=self.handle_scan_request)
        
        # Load historical data for prediction training
        await self.load_historical_data()
        
        logger.info("Intelligent scheduler initialized")
    
    async def load_historical_data(self):
        """Load historical scan data for predictive training"""
        try:
            async with self.db_pool.acquire() as conn:
                # Load recent scan history
                rows = await conn.fetch("""
                    SELECT 
                        target_count,
                        template_count,
                        EXTRACT(hour FROM created_at) as hour_of_day,
                        EXTRACT(dow FROM created_at) as day_of_week,
                        concurrent_scans,
                        queue_size,
                        peak_cpu_usage,
                        actual_duration
                    FROM scan_history 
                    WHERE created_at > NOW() - INTERVAL '30 days'
                    ORDER BY created_at DESC
                    LIMIT 500
                """)
                
                for row in rows:
                    self.predictor.add_historical_data(dict(row))
                
                logger.info(f"Loaded {len(rows)} historical scan records")
                
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
    
    async def handle_scan_request(self, msg):
        """Handle incoming scan requests"""
        try:
            data = json.loads(msg.data.decode())
            
            scan_request = ScanRequest(
                id=data['id'],
                organization_id=data['organization_id'],
                targets=data['targets'],
                priority=ScanPriority(data.get('priority', 3)),
                estimated_duration=data.get('estimated_duration', 600),
                resource_requirements=data.get('resource_requirements', {}),
                submitted_at=datetime.fromisoformat(data['submitted_at'])
            )
            
            await self.schedule_scan(scan_request)
            
        except Exception as e:
            logger.error(f"Failed to handle scan request: {e}")
    
    async def schedule_scan(self, scan_request: ScanRequest):
        """Intelligently schedule a scan request"""
        
        # Get prediction for optimal scheduling
        prediction = self.predictor.predict_scan_impact(scan_request, self.current_metrics)
        
        # Update scan request with predicted timing
        scan_request.scheduled_for = prediction.optimal_start_time
        
        # Add to appropriate priority queue
        self.priority_queues[scan_request.priority].append(scan_request)
        
        # Update metrics
        scheduled_scans.labels(priority=scan_request.priority.name.lower()).inc()
        self._update_queue_metrics()
        
        # Store scheduling decision
        await self.store_scheduling_decision(scan_request, prediction)
        
        logger.info(f"Scheduled scan {scan_request.id} for {prediction.optimal_start_time} "
                   f"(confidence: {prediction.confidence_score:.2f})")
    
    async def store_scheduling_decision(self, scan_request: ScanRequest, prediction: SchedulingPrediction):
        """Store scheduling decision for analysis"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO scheduling_decisions (
                        scan_id, organization_id, priority, submitted_at, scheduled_for,
                        prediction_confidence, expected_duration, reasoning
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, 
                scan_request.id, scan_request.organization_id, scan_request.priority.value,
                scan_request.submitted_at, scan_request.scheduled_for,
                prediction.confidence_score, prediction.resource_impact.get('duration', 600),
                prediction.reasoning)
        except Exception as e:
            logger.error(f"Failed to store scheduling decision: {e}")
    
    async def process_queue(self):
        """Main queue processing loop"""
        
        while self.running:
            try:
                # Update current system metrics
                await self.update_system_metrics()
                
                # Process queues by priority
                for priority in ScanPriority:
                    await self.process_priority_queue(priority)
                
                # Update queue metrics
                self._update_queue_metrics()
                
                # Sleep before next cycle
                await asyncio.sleep(10)  # 10-second processing cycle
                
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(30)
    
    async def process_priority_queue(self, priority: ScanPriority):
        """Process scans from a specific priority queue"""
        
        queue = self.priority_queues[priority]
        if not queue:
            return
        
        # Sort by scheduled time
        queue.sort(key=lambda x: x.scheduled_for or datetime.now())
        
        # Process ready scans
        now = datetime.now()
        ready_scans = [scan for scan in queue if scan.scheduled_for and scan.scheduled_for <= now]
        
        for scan in ready_scans[:5]:  # Process up to 5 scans per cycle
            # Check rate limiting
            allowed, wait_time = self.rate_limiter.can_consume(1)
            
            if not allowed:
                logger.debug(f"Rate limited, waiting {wait_time:.1f}s")
                break
            
            # Check system capacity
            if not await self.check_system_capacity(scan):
                continue
            
            # Execute scan
            await self.execute_scan(scan)
            
            # Remove from queue
            queue.remove(scan)
    
    async def check_system_capacity(self, scan_request: ScanRequest) -> bool:
        """Check if system has capacity for the scan"""
        
        # CPU check
        if self.current_metrics.cpu_usage > 85:
            return False
        
        # Memory check
        required_memory = scan_request.resource_requirements.get('memory', 1.0)
        if self.current_metrics.memory_usage + required_memory > 90:
            return False
        
        # Concurrent scans check
        if self.current_metrics.active_scans >= 20:
            return False
        
        return True
    
    async def execute_scan(self, scan_request: ScanRequest):
        """Execute a scheduled scan"""
        
        try:
            # Prepare scan message
            scan_message = {
                'id': scan_request.id,
                'organization_id': scan_request.organization_id,
                'targets': scan_request.targets,
                'priority': scan_request.priority.value,
                'scheduled_at': scan_request.scheduled_for.isoformat(),
                'executed_at': datetime.now().isoformat()
            }
            
            # Send to scanner
            await self.nats.publish("scans.execute", json.dumps(scan_message).encode())
            
            # Update metrics
            self.current_metrics.active_scans += 1
            
            logger.info(f"Executed scan {scan_request.id}")
            
        except Exception as e:
            logger.error(f"Failed to execute scan {scan_request.id}: {e}")
            
            # Retry logic
            scan_request.attempts += 1
            if scan_request.attempts < scan_request.max_attempts:
                # Reschedule with delay
                scan_request.scheduled_for = datetime.now() + timedelta(minutes=5)
                self.priority_queues[scan_request.priority].append(scan_request)
    
    async def update_system_metrics(self):
        """Update current system metrics from monitoring"""
        
        try:
            # Get metrics from Redis cache (updated by monitoring system)
            metrics_data = await self.redis.get("system_metrics:current")
            
            if metrics_data:
                data = json.loads(metrics_data)
                self.current_metrics = SystemMetrics(
                    cpu_usage=data.get('cpu_usage', 0),
                    memory_usage=data.get('memory_usage', 0),
                    active_scans=data.get('active_scans', 0),
                    queue_length=sum(len(q) for q in self.priority_queues.values()),
                    avg_scan_duration=data.get('avg_scan_duration', 600),
                    timestamp=datetime.now()
                )
            
            # Update Prometheus metrics
            resource_utilization.set(max(self.current_metrics.cpu_usage, self.current_metrics.memory_usage))
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def _update_queue_metrics(self):
        """Update queue size metrics"""
        total_queue_size = sum(len(q) for q in self.priority_queues.values())
        queue_size.set(total_queue_size)
    
    async def get_queue_status(self) -> Dict:
        """Get current queue status"""
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': asdict(self.current_metrics),
            'rate_limiter': self.rate_limiter.get_status(),
            'queue_sizes': {
                priority.name.lower(): len(queue) 
                for priority, queue in self.priority_queues.items()
            },
            'predictor_trained': self.predictor.trained,
            'total_queued': sum(len(q) for q in self.priority_queues.values())
        }
        
        return status
    
    async def start(self):
        """Start the intelligent scheduler"""
        self.running = True
        
        # Start queue processing
        processing_task = asyncio.create_task(self.process_queue())
        
        try:
            await processing_task
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        finally:
            self.running = False

async def main():
    """Main function for the intelligent scheduler"""
    scheduler = IntelligentScheduler()
    
    try:
        await scheduler.initialize()
        logger.info("🚀 Starting Xorb Intelligent Scheduler with Predictive Pacing")
        await scheduler.start()
    except Exception as e:
        logger.error(f"Scheduler failed: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())