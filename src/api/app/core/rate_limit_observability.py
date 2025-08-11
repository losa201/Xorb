"""
Comprehensive observability for adaptive rate limiting system.

This module provides enterprise-grade monitoring with:
- Detailed metrics for allows/blocks, tokens, latency, FP/FN estimates
- Structured logging with correlation IDs and tenant isolation
- Distributed tracing for rate limiting decisions
- Real-time dashboards and alerting integration
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4
import structlog

from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry
import opentelemetry.trace as trace
from opentelemetry.trace import Status, StatusCode

from .adaptive_rate_limiter import RateLimitResult, PolicyScope, LimitAlgorithm, ReputationLevel
from .rate_limit_policies import RateLimitContext, PolicyType

logger = structlog.get_logger("rate_limit_observability")


class DecisionOutcome(Enum):
    """Rate limiting decision outcomes"""
    ALLOWED = "allowed"
    BLOCKED_RATE_LIMIT = "blocked_rate_limit"
    BLOCKED_CIRCUIT_BREAKER = "blocked_circuit_breaker"
    BLOCKED_BACKOFF = "blocked_backoff"
    BLOCKED_EMERGENCY = "blocked_emergency"
    ERROR_FAIL_OPEN = "error_fail_open"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class RateLimitEvent:
    """Structured rate limiting event for logging and analysis"""
    timestamp: float
    correlation_id: str
    decision_outcome: DecisionOutcome
    
    # Request context
    scope: PolicyScope
    identifier_hash: str  # Hashed for privacy
    endpoint: Optional[str]
    user_id_hash: Optional[str]
    tenant_id: Optional[str]
    ip_address_hash: Optional[str]
    user_agent_hash: Optional[str]
    
    # Rate limiting details
    policy_matched: Optional[str]
    algorithm_used: LimitAlgorithm
    tokens_remaining: int
    retry_after_seconds: Optional[int]
    reputation_level: ReputationLevel
    backoff_level: int
    
    # Performance metrics
    computation_time_ms: float
    redis_hits: int
    cache_hits: int
    
    # Policy hierarchy
    overrides_applied: List[str] = field(default_factory=list)
    hard_caps_enforced: bool = False
    
    # Metadata
    shadow_mode: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertCondition:
    """Alert condition configuration"""
    name: str
    description: str
    severity: AlertSeverity
    
    # Condition thresholds
    metric_name: str
    threshold_value: float
    comparison_operator: str  # >, <, >=, <=, ==
    time_window_seconds: int
    
    # Grouping and filtering
    label_filters: Dict[str, str] = field(default_factory=dict)
    group_by_labels: List[str] = field(default_factory=list)
    
    # Alert behavior
    cooldown_seconds: int = 300
    max_alerts_per_hour: int = 10
    
    # Notification channels
    notify_channels: List[str] = field(default_factory=list)


class RateLimitMetrics:
    """Comprehensive Prometheus metrics for rate limiting"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None, prefix: str = "rate_limiter"):
        self.registry = registry
        self.prefix = prefix
        
        # Core decision metrics
        self.decisions_total = Counter(
            f'{prefix}_decisions_total',
            'Total rate limiting decisions',
            ['scope', 'algorithm', 'outcome', 'reputation', 'tenant_id'],
            registry=registry
        )
        
        self.tokens_remaining = Gauge(
            f'{prefix}_tokens_remaining',
            'Current tokens remaining in buckets',
            ['scope', 'key_hash', 'tenant_id'],
            registry=registry
        )
        
        self.computation_time = Histogram(
            f'{prefix}_computation_time_seconds',
            'Rate limit computation time',
            ['algorithm', 'cache_hit'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            registry=registry
        )
        
        # Policy hierarchy metrics
        self.policy_overrides = Counter(
            f'{prefix}_policy_overrides_total',
            'Policy overrides applied',
            ['override_type', 'scope', 'tenant_id'],
            registry=registry
        )
        
        self.hard_caps_enforced = Counter(
            f'{prefix}_hard_caps_enforced_total',
            'Hard caps enforced',
            ['scope', 'cap_type', 'tenant_id'],
            registry=registry
        )
        
        # Redis operations
        self.redis_operations = Counter(
            f'{prefix}_redis_operations_total',
            'Redis operations',
            ['operation', 'result'],
            registry=registry
        )
        
        self.redis_latency = Histogram(
            f'{prefix}_redis_latency_seconds',
            'Redis operation latency',
            ['operation'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            registry=registry
        )
        
        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            f'{prefix}_circuit_breaker_open',
            'Circuit breaker state (1=open, 0=closed)',
            ['scope', 'policy_id'],
            registry=registry
        )
        
        self.circuit_breaker_failures = Counter(
            f'{prefix}_circuit_breaker_failures_total',
            'Circuit breaker failures',
            ['scope', 'policy_id'],
            registry=registry
        )
        
        # Reputation scoring
        self.reputation_scores = Histogram(
            f'{prefix}_reputation_scores',
            'Reputation score distribution',
            ['scope'],
            buckets=[0, 20, 40, 60, 80, 100],
            registry=registry
        )
        
        self.reputation_changes = Counter(
            f'{prefix}_reputation_changes_total',
            'Reputation level changes',
            ['from_level', 'to_level', 'scope'],
            registry=registry
        )
        
        # Backoff tracking
        self.backoff_activations = Counter(
            f'{prefix}_backoff_activations_total',
            'Progressive backoff activations',
            ['level', 'scope'],
            registry=registry
        )
        
        self.backoff_durations = Histogram(
            f'{prefix}_backoff_duration_seconds',
            'Backoff duration distribution',
            ['level'],
            buckets=[60, 300, 900, 1800, 3600, 7200],
            registry=registry
        )
        
        # Emergency and shadow mode
        self.emergency_mode_activations = Counter(
            f'{prefix}_emergency_mode_activations_total',
            'Emergency mode activations',
            ['trigger_reason'],
            registry=registry
        )
        
        self.shadow_mode_decisions = Counter(
            f'{prefix}_shadow_mode_decisions_total',
            'Shadow mode decisions (would have blocked)',
            ['scope', 'algorithm', 'reason'],
            registry=registry
        )
        
        # False positive/negative estimation
        self.estimated_false_positives = Gauge(
            f'{prefix}_estimated_false_positives',
            'Estimated false positive rate',
            ['scope', 'time_window'],
            registry=registry
        )
        
        self.estimated_false_negatives = Gauge(
            f'{prefix}_estimated_false_negatives',
            'Estimated false negative rate',
            ['scope', 'time_window'],
            registry=registry
        )
        
        # System health
        self.system_health_score = Gauge(
            f'{prefix}_system_health_score',
            'Overall rate limiter health score (0-100)',
            registry=registry
        )
        
        self.policy_cache_hit_ratio = Gauge(
            f'{prefix}_policy_cache_hit_ratio',
            'Policy cache hit ratio',
            registry=registry
        )
        
        # Tenant isolation metrics
        self.tenant_request_volumes = Counter(
            f'{prefix}_tenant_request_volumes_total',
            'Request volumes per tenant',
            ['tenant_id', 'outcome'],
            registry=registry
        )
        
        self.tenant_isolation_violations = Counter(
            f'{prefix}_tenant_isolation_violations_total',
            'Tenant isolation violations',
            ['violation_type', 'tenant_id'],
            registry=registry
        )


class RateLimitTracer:
    """Distributed tracing for rate limiting decisions"""
    
    def __init__(self, tracer_name: str = "rate_limiter"):
        self.tracer = trace.get_tracer(tracer_name)
    
    def trace_rate_limit_check(
        self,
        context: RateLimitContext,
        correlation_id: str
    ):
        """Create a span for rate limit checking"""
        return self.tracer.start_span(
            "rate_limit_check",
            attributes={
                "rate_limit.scope": context.scope.value,
                "rate_limit.endpoint": context.endpoint or "",
                "rate_limit.tenant_id": str(context.tenant_id) if context.tenant_id else "",
                "rate_limit.is_authenticated": context.is_authenticated,
                "rate_limit.correlation_id": correlation_id,
                "rate_limit.business_hours": context.business_hours
            }
        )
    
    def add_policy_resolution_span(
        self,
        parent_span,
        policies_checked: int,
        overrides_applied: List[str],
        final_policy: str
    ):
        """Add child span for policy resolution"""
        with self.tracer.start_span(
            "policy_resolution",
            parent=parent_span
        ) as span:
            span.set_attributes({
                "policy.policies_checked": policies_checked,
                "policy.overrides_applied": ",".join(overrides_applied),
                "policy.final_policy": final_policy
            })
    
    def add_algorithm_execution_span(
        self,
        parent_span,
        algorithm: LimitAlgorithm,
        redis_operations: int,
        computation_time_ms: float
    ):
        """Add child span for algorithm execution"""
        with self.tracer.start_span(
            "algorithm_execution",
            parent=parent_span
        ) as span:
            span.set_attributes({
                "algorithm.type": algorithm.value,
                "algorithm.redis_operations": redis_operations,
                "algorithm.computation_time_ms": computation_time_ms
            })
    
    def record_decision(
        self,
        span,
        result: RateLimitResult,
        event: RateLimitEvent
    ):
        """Record the final decision in the span"""
        span.set_attributes({
            "decision.allowed": result.allowed,
            "decision.tokens_remaining": result.tokens_remaining,
            "decision.retry_after_seconds": result.retry_after_seconds or 0,
            "decision.reputation_level": result.reputation_level.value,
            "decision.backoff_level": result.backoff_level,
            "decision.circuit_breaker_open": result.circuit_breaker_open,
            "performance.computation_time_ms": result.computation_time_ms,
            "performance.redis_hits": result.redis_hits,
            "performance.cache_hits": result.cache_hits
        })
        
        # Set span status
        if result.allowed:
            span.set_status(Status(StatusCode.OK))
        else:
            span.set_status(Status(StatusCode.ERROR, f"Rate limited: {event.decision_outcome.value}"))


class RateLimitObservability:
    """Comprehensive observability manager for rate limiting"""
    
    def __init__(
        self,
        metrics_registry: Optional[CollectorRegistry] = None,
        enable_tracing: bool = True,
        enable_detailed_logging: bool = True
    ):
        self.metrics = RateLimitMetrics(registry=metrics_registry)
        self.tracer = RateLimitTracer() if enable_tracing else None
        self.enable_detailed_logging = enable_detailed_logging
        
        # Event storage for analysis
        self.recent_events: List[RateLimitEvent] = []
        self.max_events = 10000
        
        # Alert conditions
        self.alert_conditions: List[AlertCondition] = []
        self.active_alerts: Dict[str, float] = {}  # alert_name -> last_fired_time
        
        # False positive/negative tracking
        self.fp_fn_tracker = FalsePositiveNegativeTracker()
        
        self._setup_default_alerts()
        
        logger.info("Rate limit observability initialized",
                   enable_tracing=enable_tracing,
                   enable_detailed_logging=enable_detailed_logging)
    
    def _setup_default_alerts(self):
        """Setup default alert conditions"""
        self.alert_conditions = [
            AlertCondition(
                name="high_block_rate",
                description="High rate of blocked requests",
                severity=AlertSeverity.WARNING,
                metric_name="rate_limiter_decisions_total",
                threshold_value=100.0,
                comparison_operator=">=",
                time_window_seconds=300,
                label_filters={"outcome": "blocked_rate_limit"},
                group_by_labels=["scope", "tenant_id"]
            ),
            AlertCondition(
                name="circuit_breaker_open",
                description="Circuit breaker opened",
                severity=AlertSeverity.CRITICAL,
                metric_name="rate_limiter_circuit_breaker_open",
                threshold_value=1.0,
                comparison_operator=">=",
                time_window_seconds=60,
                group_by_labels=["scope"]
            ),
            AlertCondition(
                name="high_computation_latency",
                description="High rate limiting computation latency",
                severity=AlertSeverity.WARNING,
                metric_name="rate_limiter_computation_time_seconds",
                threshold_value=0.1,
                comparison_operator=">=",
                time_window_seconds=300
            ),
            AlertCondition(
                name="redis_errors",
                description="High Redis error rate",
                severity=AlertSeverity.CRITICAL,
                metric_name="rate_limiter_redis_operations_total",
                threshold_value=10.0,
                comparison_operator=">=",
                time_window_seconds=120,
                label_filters={"result": "error"}
            ),
            AlertCondition(
                name="emergency_mode_active",
                description="Emergency rate limiting activated",
                severity=AlertSeverity.EMERGENCY,
                metric_name="rate_limiter_emergency_mode_activations_total",
                threshold_value=1.0,
                comparison_operator=">=",
                time_window_seconds=60
            )
        ]
    
    def record_decision(
        self,
        context: RateLimitContext,
        result: RateLimitResult,
        correlation_id: Optional[str] = None
    ) -> RateLimitEvent:
        """Record a rate limiting decision with full observability"""
        correlation_id = correlation_id or str(uuid4())[:12]
        
        # Determine decision outcome
        outcome = self._determine_outcome(result)
        
        # Create structured event
        event = RateLimitEvent(
            timestamp=time.time(),
            correlation_id=correlation_id,
            decision_outcome=outcome,
            scope=context.scope,
            identifier_hash=self._hash_identifier(context),
            endpoint=context.endpoint,
            user_id_hash=self._hash_value(context.user_id) if context.user_id else None,
            tenant_id=str(context.tenant_id) if context.tenant_id else None,
            ip_address_hash=self._hash_value(context.ip_address) if context.ip_address else None,
            user_agent_hash=self._hash_value(context.user_agent) if context.user_agent else None,
            policy_matched=result.policy_matched.scope.value if result.policy_matched else None,
            algorithm_used=result.algorithm_used,
            tokens_remaining=result.tokens_remaining,
            retry_after_seconds=result.retry_after_seconds,
            reputation_level=result.reputation_level,
            backoff_level=result.backoff_level,
            computation_time_ms=result.computation_time_ms,
            redis_hits=result.redis_hits,
            cache_hits=result.cache_hits,
            overrides_applied=result.decision_metadata.get('overrides_applied', []),
            hard_caps_enforced=result.decision_metadata.get('hard_caps_enforced', False),
            shadow_mode=result.decision_metadata.get('shadow_mode', False),
            metadata=result.decision_metadata
        )
        
        # Record metrics
        self._record_metrics(event, context)
        
        # Structured logging
        if self.enable_detailed_logging:
            self._log_decision(event, context)
        
        # Store event for analysis
        self._store_event(event)
        
        # Update false positive/negative tracking
        self.fp_fn_tracker.record_decision(event, context)
        
        # Check alert conditions
        self._check_alerts(event)
        
        return event
    
    def start_trace(self, context: RateLimitContext, correlation_id: str):
        """Start distributed trace for rate limiting"""
        if self.tracer:
            return self.tracer.trace_rate_limit_check(context, correlation_id)
        return None
    
    def _determine_outcome(self, result: RateLimitResult) -> DecisionOutcome:
        """Determine the decision outcome from result"""
        if result.allowed:
            if "limiter_error" in result.decision_metadata.get('reason', ''):
                return DecisionOutcome.ERROR_FAIL_OPEN
            return DecisionOutcome.ALLOWED
        
        if result.circuit_breaker_open:
            return DecisionOutcome.BLOCKED_CIRCUIT_BREAKER
        elif result.backoff_level > 0:
            return DecisionOutcome.BLOCKED_BACKOFF
        elif "emergency" in result.decision_metadata.get('reason', ''):
            return DecisionOutcome.BLOCKED_EMERGENCY
        else:
            return DecisionOutcome.BLOCKED_RATE_LIMIT
    
    def _hash_identifier(self, context: RateLimitContext) -> str:
        """Create hash of identifier for privacy"""
        import hashlib
        
        identifier_parts = [
            context.scope.value,
            context.endpoint or "",
            context.user_id or "",
            str(context.tenant_id) if context.tenant_id else "",
            context.ip_address or ""
        ]
        identifier = ":".join(identifier_parts)
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]
    
    def _hash_value(self, value: str) -> str:
        """Hash a single value for privacy"""
        import hashlib
        return hashlib.sha256(value.encode()).hexdigest()[:12]
    
    def _record_metrics(self, event: RateLimitEvent, context: RateLimitContext):
        """Record comprehensive metrics"""
        tenant_id = event.tenant_id or "no_tenant"
        
        # Core decision metrics
        self.metrics.decisions_total.labels(
            scope=event.scope.value,
            algorithm=event.algorithm_used.value,
            outcome=event.decision_outcome.value,
            reputation=event.reputation_level.value,
            tenant_id=tenant_id
        ).inc()
        
        # Computation time
        cache_hit = "yes" if event.cache_hits > 0 else "no"
        self.metrics.computation_time.labels(
            algorithm=event.algorithm_used.value,
            cache_hit=cache_hit
        ).observe(event.computation_time_ms / 1000)
        
        # Redis operations
        self.metrics.redis_operations.labels(
            operation="rate_check",
            result="success" if event.redis_hits > 0 else "cache"
        ).inc()
        
        # Policy overrides
        for override_type in event.overrides_applied:
            self.metrics.policy_overrides.labels(
                override_type=override_type,
                scope=event.scope.value,
                tenant_id=tenant_id
            ).inc()
        
        # Hard caps
        if event.hard_caps_enforced:
            self.metrics.hard_caps_enforced.labels(
                scope=event.scope.value,
                cap_type="rate_limit",
                tenant_id=tenant_id
            ).inc()
        
        # Reputation scoring
        reputation_score = {
            ReputationLevel.EXCELLENT: 90,
            ReputationLevel.GOOD: 70,
            ReputationLevel.NEUTRAL: 50,
            ReputationLevel.POOR: 30,
            ReputationLevel.BLOCKED: 10
        }[event.reputation_level]
        
        self.metrics.reputation_scores.labels(
            scope=event.scope.value
        ).observe(reputation_score)
        
        # Backoff tracking
        if event.backoff_level > 0:
            self.metrics.backoff_activations.labels(
                level=str(event.backoff_level),
                scope=event.scope.value
            ).inc()
        
        # Shadow mode
        if event.shadow_mode and event.decision_outcome != DecisionOutcome.ALLOWED:
            self.metrics.shadow_mode_decisions.labels(
                scope=event.scope.value,
                algorithm=event.algorithm_used.value,
                reason=event.decision_outcome.value
            ).inc()
        
        # Tenant volumes
        self.metrics.tenant_request_volumes.labels(
            tenant_id=tenant_id,
            outcome="allowed" if event.decision_outcome == DecisionOutcome.ALLOWED else "blocked"
        ).inc()
    
    def _log_decision(self, event: RateLimitEvent, context: RateLimitContext):
        """Log structured decision with correlation ID"""
        log_data = {
            "correlation_id": event.correlation_id,
            "decision_outcome": event.decision_outcome.value,
            "scope": event.scope.value,
            "endpoint": event.endpoint,
            "tenant_id": event.tenant_id,
            "algorithm": event.algorithm_used.value,
            "tokens_remaining": event.tokens_remaining,
            "computation_time_ms": event.computation_time_ms,
            "reputation_level": event.reputation_level.value,
            "overrides_applied": event.overrides_applied,
            "shadow_mode": event.shadow_mode
        }
        
        if event.decision_outcome == DecisionOutcome.ALLOWED:
            logger.debug("Rate limit decision: ALLOWED", **log_data)
        else:
            logger.warning(
                "Rate limit decision: BLOCKED",
                retry_after_seconds=event.retry_after_seconds,
                backoff_level=event.backoff_level,
                **log_data
            )
    
    def _store_event(self, event: RateLimitEvent):
        """Store event for analysis"""
        self.recent_events.append(event)
        
        # Trim to max size
        if len(self.recent_events) > self.max_events:
            self.recent_events = self.recent_events[-self.max_events:]
    
    def _check_alerts(self, event: RateLimitEvent):
        """Check if any alert conditions are met"""
        # Simple implementation - in production, integrate with alerting system
        current_time = time.time()
        
        for condition in self.alert_conditions:
            last_fired = self.active_alerts.get(condition.name, 0)
            
            # Check cooldown
            if current_time - last_fired < condition.cooldown_seconds:
                continue
            
            # Simple threshold check (in production, use proper metric queries)
            if self._evaluate_condition(condition, event):
                self._fire_alert(condition, event)
                self.active_alerts[condition.name] = current_time
    
    def _evaluate_condition(self, condition: AlertCondition, event: RateLimitEvent) -> bool:
        """Evaluate if an alert condition is met"""
        # Simplified evaluation - in production, query metrics backend
        if condition.metric_name == "rate_limiter_decisions_total":
            if condition.label_filters.get("outcome") == event.decision_outcome.value:
                return True
        
        return False
    
    def _fire_alert(self, condition: AlertCondition, event: RateLimitEvent):
        """Fire an alert"""
        logger.error(
            f"ALERT: {condition.name}",
            severity=condition.severity.value,
            description=condition.description,
            correlation_id=event.correlation_id,
            tenant_id=event.tenant_id,
            scope=event.scope.value
        )
    
    def get_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""
        if not self.recent_events:
            return 100.0
        
        recent_events = [
            e for e in self.recent_events
            if time.time() - e.timestamp < 300  # Last 5 minutes
        ]
        
        if not recent_events:
            return 100.0
        
        # Calculate health factors
        allowed_ratio = len([e for e in recent_events if e.decision_outcome == DecisionOutcome.ALLOWED]) / len(recent_events)
        avg_computation_time = sum(e.computation_time_ms for e in recent_events) / len(recent_events)
        circuit_breaker_events = len([e for e in recent_events if e.decision_outcome == DecisionOutcome.BLOCKED_CIRCUIT_BREAKER])
        
        # Health score calculation
        health_score = 100.0
        health_score *= allowed_ratio  # Penalize high block rates
        health_score *= max(0.5, 1.0 - (avg_computation_time / 100))  # Penalize high latency
        health_score *= max(0.1, 1.0 - (circuit_breaker_events / len(recent_events)))  # Penalize circuit breaker trips
        
        return max(0.0, min(100.0, health_score))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive observability statistics"""
        current_time = time.time()
        recent_events = [
            e for e in self.recent_events
            if current_time - e.timestamp < 300
        ]
        
        if not recent_events:
            return {"status": "no_recent_activity"}
        
        # Calculate statistics
        total_events = len(recent_events)
        allowed_events = len([e for e in recent_events if e.decision_outcome == DecisionOutcome.ALLOWED])
        blocked_events = total_events - allowed_events
        
        avg_computation_time = sum(e.computation_time_ms for e in recent_events) / total_events
        avg_tokens_remaining = sum(e.tokens_remaining for e in recent_events) / total_events
        
        # Group by outcome
        outcome_stats = {}
        for outcome in DecisionOutcome:
            count = len([e for e in recent_events if e.decision_outcome == outcome])
            outcome_stats[outcome.value] = count
        
        # Group by scope
        scope_stats = {}
        for scope in PolicyScope:
            count = len([e for e in recent_events if e.scope == scope])
            scope_stats[scope.value] = count
        
        return {
            "time_window_seconds": 300,
            "total_events": total_events,
            "allowed_events": allowed_events,
            "blocked_events": blocked_events,
            "allow_rate": allowed_events / total_events if total_events > 0 else 0,
            "avg_computation_time_ms": avg_computation_time,
            "avg_tokens_remaining": avg_tokens_remaining,
            "health_score": self.get_health_score(),
            "outcome_distribution": outcome_stats,
            "scope_distribution": scope_stats,
            "active_alerts": len(self.active_alerts),
            "fp_fn_estimates": self.fp_fn_tracker.get_estimates()
        }


class FalsePositiveNegativeTracker:
    """Track and estimate false positives and false negatives"""
    
    def __init__(self):
        self.legitimate_blocks = 0  # Correctly blocked malicious requests
        self.false_positives = 0   # Incorrectly blocked legitimate requests
        self.legitimate_allows = 0  # Correctly allowed legitimate requests
        self.false_negatives = 0   # Incorrectly allowed malicious requests
        
        self.window_start = time.time()
        self.window_duration = 3600  # 1 hour window
    
    def record_decision(self, event: RateLimitEvent, context: RateLimitContext):
        """Record a decision for FP/FN analysis"""
        current_time = time.time()
        
        # Reset window if needed
        if current_time - self.window_start > self.window_duration:
            self._reset_window()
        
        # Simple heuristics for FP/FN detection
        # In production, this would use ML models and manual feedback
        
        is_likely_legitimate = self._is_likely_legitimate_request(event, context)
        
        if event.decision_outcome == DecisionOutcome.ALLOWED:
            if is_likely_legitimate:
                self.legitimate_allows += 1
            else:
                self.false_negatives += 1  # Should have been blocked
        else:
            if is_likely_legitimate:
                self.false_positives += 1  # Incorrectly blocked
            else:
                self.legitimate_blocks += 1
    
    def _is_likely_legitimate_request(self, event: RateLimitEvent, context: RateLimitContext) -> bool:
        """Heuristic to determine if request is likely legitimate"""
        # Consider legitimate if:
        # - Authenticated user with good reputation
        # - Business hours request
        # - Normal endpoint access pattern
        # - Low request rate
        
        if (context.is_authenticated and 
            event.reputation_level in [ReputationLevel.EXCELLENT, ReputationLevel.GOOD] and
            context.business_hours and
            event.tokens_remaining > 10):
            return True
        
        return False
    
    def _reset_window(self):
        """Reset the tracking window"""
        self.legitimate_blocks = 0
        self.false_positives = 0
        self.legitimate_allows = 0
        self.false_negatives = 0
        self.window_start = time.time()
    
    def get_estimates(self) -> Dict[str, float]:
        """Get false positive and false negative rate estimates"""
        total_blocks = self.legitimate_blocks + self.false_positives
        total_allows = self.legitimate_allows + self.false_negatives
        
        fp_rate = self.false_positives / total_blocks if total_blocks > 0 else 0.0
        fn_rate = self.false_negatives / total_allows if total_allows > 0 else 0.0
        
        accuracy = (self.legitimate_blocks + self.legitimate_allows) / (
            total_blocks + total_allows
        ) if (total_blocks + total_allows) > 0 else 1.0
        
        return {
            "false_positive_rate": fp_rate,
            "false_negative_rate": fn_rate,
            "accuracy": accuracy,
            "sample_size": total_blocks + total_allows,
            "window_hours": self.window_duration / 3600
        }