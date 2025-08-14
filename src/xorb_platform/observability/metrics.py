#!/usr/bin/env python3
"""
XORB Core SLIs (Service Level Indicators) Metrics Module

This module implements the 5 core SLIs for XORB platform monitoring:
1. bus_publish_to_deliver_p95_ms - Message bus delivery latency
2. consumer_redelivery_rate - Message redelivery rate
3. evidence_ingest_p95_ms - Evidence ingestion latency
4. auth_error_rate - Authentication error rate
5. mtls_handshake_fail_rate - mTLS handshake failure rate

Metrics are exposed in Prometheus format for monitoring and alerting.
"""

import time
import logging
from typing import Dict, Any, Optional
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY
)
from flask import Flask, Response
import threading
from datetime import datetime, timedelta
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics definitions
# SLI 1: Message bus publish-to-deliver latency
bus_publish_deliver_histogram = Histogram(
    'xorb_bus_publish_to_deliver_seconds',
    'Time from message publish to successful delivery',
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

# SLI 2: Consumer redelivery rate
consumer_redelivery_counter = Counter(
    'xorb_consumer_redeliveries_total',
    'Total number of message redeliveries',
    labelnames=['consumer_id', 'topic', 'reason']
)

consumer_delivery_counter = Counter(
    'xorb_consumer_deliveries_total',
    'Total number of successful message deliveries',
    labelnames=['consumer_id', 'topic']
)

# SLI 3: Evidence ingestion latency
evidence_ingest_histogram = Histogram(
    'xorb_evidence_ingest_seconds',
    'Time to ingest evidence into the system',
    labelnames=['evidence_type', 'source'],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
)

# SLI 4: Authentication error rate
auth_success_counter = Counter(
    'xorb_auth_attempts_total',
    'Total authentication attempts',
    labelnames=['method', 'result']
)

# SLI 5: mTLS handshake failure rate
mtls_handshake_counter = Counter(
    'xorb_mtls_handshakes_total',
    'Total mTLS handshake attempts',
    labelnames=['result', 'peer_type']
)

# Additional operational metrics
system_health_gauge = Gauge(
    'xorb_system_health_score',
    'Overall system health score (0-1)',
    labelnames=['component']
)

# SLI calculation gauges (for dashboard convenience)
bus_deliver_p95_gauge = Gauge(
    'xorb_sli_bus_publish_to_deliver_p95_ms',
    'P95 message bus delivery latency in milliseconds'
)

consumer_redelivery_rate_gauge = Gauge(
    'xorb_sli_consumer_redelivery_rate',
    'Consumer redelivery rate (redeliveries/total_deliveries)'
)

evidence_ingest_p95_gauge = Gauge(
    'xorb_sli_evidence_ingest_p95_ms',
    'P95 evidence ingestion latency in milliseconds'
)

auth_error_rate_gauge = Gauge(
    'xorb_sli_auth_error_rate',
    'Authentication error rate (errors/total_attempts)'
)

mtls_handshake_fail_rate_gauge = Gauge(
    'xorb_sli_mtls_handshake_fail_rate',
    'mTLS handshake failure rate (failures/total_attempts)'
)

class XORBMetricsCollector:
    """Core metrics collector for XORB SLIs"""

    def __init__(self):
        self.start_time = time.time()
        self.calculation_lock = threading.Lock()
        self.last_calculation = time.time()
        self.calculation_interval = 60  # Calculate SLIs every 60 seconds

        # Start background SLI calculation thread
        self.calculation_thread = threading.Thread(target=self._calculate_slis_loop, daemon=True)
        self.calculation_thread.start()

        logger.info("XORB Metrics Collector initialized")

    def record_bus_delivery(self, duration_seconds: float):
        """Record message bus publish-to-deliver duration"""
        bus_publish_deliver_histogram.observe(duration_seconds)

    def record_redelivery(self, consumer_id: str, topic: str, reason: str = "timeout"):
        """Record a message redelivery"""
        consumer_redelivery_counter.labels(
            consumer_id=consumer_id,
            topic=topic,
            reason=reason
        ).inc()

    def record_successful_delivery(self, consumer_id: str, topic: str):
        """Record successful message delivery"""
        consumer_delivery_counter.labels(
            consumer_id=consumer_id,
            topic=topic
        ).inc()

    def record_evidence_ingest(self, duration_seconds: float, evidence_type: str, source: str):
        """Record evidence ingestion duration"""
        evidence_ingest_histogram.labels(
            evidence_type=evidence_type,
            source=source
        ).observe(duration_seconds)

    def record_auth_attempt(self, method: str, success: bool):
        """Record authentication attempt"""
        result = "success" if success else "error"
        auth_success_counter.labels(method=method, result=result).inc()

    def record_mtls_handshake(self, success: bool, peer_type: str = "unknown"):
        """Record mTLS handshake attempt"""
        result = "success" if success else "failure"
        mtls_handshake_counter.labels(result=result, peer_type=peer_type).inc()

    def set_system_health(self, component: str, score: float):
        """Set system health score for component (0.0-1.0)"""
        system_health_gauge.labels(component=component).set(max(0.0, min(1.0, score)))

    def _calculate_slis_loop(self):
        """Background thread to calculate SLI values"""
        while True:
            try:
                time.sleep(self.calculation_interval)
                with self.calculation_lock:
                    self._calculate_slis()
                    self.last_calculation = time.time()
            except Exception as e:
                logger.error(f"Error calculating SLIs: {e}")

    def _calculate_slis(self):
        """Calculate and update SLI gauge values"""
        try:
            # SLI 1: Bus delivery P95 latency
            bus_samples = bus_publish_deliver_histogram._value._sum
            if hasattr(bus_publish_deliver_histogram._value, '_buckets'):
                # Calculate P95 from histogram buckets
                p95_seconds = self._calculate_percentile(bus_publish_deliver_histogram, 0.95)
                bus_deliver_p95_gauge.set(p95_seconds * 1000)  # Convert to milliseconds

            # SLI 2: Consumer redelivery rate
            total_redeliveries = sum(
                metric.samples[0].value
                for metric in consumer_redelivery_counter.collect()[0].samples
            )
            total_deliveries = sum(
                metric.samples[0].value
                for metric in consumer_delivery_counter.collect()[0].samples
            )

            if total_deliveries > 0:
                redelivery_rate = total_redeliveries / (total_deliveries + total_redeliveries)
                consumer_redelivery_rate_gauge.set(redelivery_rate)

            # SLI 3: Evidence ingest P95 latency
            p95_evidence = self._calculate_percentile(evidence_ingest_histogram, 0.95)
            evidence_ingest_p95_gauge.set(p95_evidence * 1000)  # Convert to milliseconds

            # SLI 4: Authentication error rate
            auth_metrics = list(auth_success_counter.collect()[0].samples)
            total_auth = sum(sample.value for sample in auth_metrics)
            error_auth = sum(
                sample.value for sample in auth_metrics
                if sample.labels.get('result') == 'error'
            )

            if total_auth > 0:
                auth_error_rate = error_auth / total_auth
                auth_error_rate_gauge.set(auth_error_rate)

            # SLI 5: mTLS handshake failure rate
            mtls_metrics = list(mtls_handshake_counter.collect()[0].samples)
            total_handshakes = sum(sample.value for sample in mtls_metrics)
            failed_handshakes = sum(
                sample.value for sample in mtls_metrics
                if sample.labels.get('result') == 'failure'
            )

            if total_handshakes > 0:
                mtls_fail_rate = failed_handshakes / total_handshakes
                mtls_handshake_fail_rate_gauge.set(mtls_fail_rate)

            logger.debug("SLI calculations updated successfully")

        except Exception as e:
            logger.error(f"Failed to calculate SLIs: {e}")

    def _calculate_percentile(self, histogram, percentile: float) -> float:
        """Calculate percentile from Prometheus histogram"""
        # Simplified percentile calculation
        # In production, use proper quantile calculation
        buckets = getattr(histogram._value, '_buckets', {})
        if not buckets:
            return 0.0

        total_count = sum(buckets.values())
        if total_count == 0:
            return 0.0

        target_count = total_count * percentile
        cumulative = 0

        for bucket_bound, count in sorted(buckets.items()):
            cumulative += count
            if cumulative >= target_count:
                return bucket_bound

        return max(buckets.keys()) if buckets else 0.0

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current SLI values for monitoring"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "slis": {
                "bus_publish_to_deliver_p95_ms": bus_deliver_p95_gauge._value._value,
                "consumer_redelivery_rate": consumer_redelivery_rate_gauge._value._value,
                "evidence_ingest_p95_ms": evidence_ingest_p95_gauge._value._value,
                "auth_error_rate": auth_error_rate_gauge._value._value,
                "mtls_handshake_fail_rate": mtls_handshake_fail_rate_gauge._value._value,
            },
            "system_uptime_seconds": time.time() - self.start_time,
            "last_calculation": self.last_calculation
        }

# Global collector instance
metrics_collector = XORBMetricsCollector()

# Flask app for metrics endpoint
app = Flask(__name__)

@app.route('/metrics')
def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(REGISTRY), mimetype=CONTENT_TYPE_LATEST)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.route('/slis')
def slis_endpoint():
    """SLI summary endpoint"""
    return metrics_collector.get_metrics_summary()

def simulate_metrics():
    """Generate sample metrics for testing"""
    import random

    logger.info("Generating sample SLI metrics for testing...")

    # Simulate bus deliveries
    for _ in range(100):
        duration = random.uniform(0.001, 0.5)  # 1ms to 500ms
        metrics_collector.record_bus_delivery(duration)

    # Simulate redeliveries
    for _ in range(10):
        metrics_collector.record_redelivery(
            consumer_id=f"consumer_{random.randint(1,5)}",
            topic=f"topic_{random.randint(1,3)}",
            reason=random.choice(["timeout", "error", "retry"])
        )

    # Simulate successful deliveries
    for _ in range(500):
        metrics_collector.record_successful_delivery(
            consumer_id=f"consumer_{random.randint(1,5)}",
            topic=f"topic_{random.randint(1,3)}"
        )

    # Simulate evidence ingestion
    for _ in range(50):
        duration = random.uniform(0.1, 10.0)  # 100ms to 10s
        metrics_collector.record_evidence_ingest(
            duration,
            evidence_type=random.choice(["scan_result", "log_entry", "artifact"]),
            source=random.choice(["nmap", "nuclei", "nikto", "manual"])
        )

    # Simulate authentication
    for _ in range(200):
        success = random.random() > 0.05  # 95% success rate
        method = random.choice(["jwt", "oauth", "apikey"])
        metrics_collector.record_auth_attempt(method, success)

    # Simulate mTLS handshakes
    for _ in range(150):
        success = random.random() > 0.02  # 98% success rate
        peer_type = random.choice(["client", "server", "peer"])
        metrics_collector.record_mtls_handshake(success, peer_type)

    # Set system health
    for component in ["api", "orchestrator", "database", "redis"]:
        health_score = random.uniform(0.85, 1.0)  # 85-100% healthy
        metrics_collector.set_system_health(component, health_score)

    logger.info("Sample metrics generation completed")

def start_metrics_server(host: str = "0.0.0.0", port: int = 9090, debug: bool = False):
    """Start the metrics server"""
    logger.info(f"Starting XORB SLI metrics server on {host}:{port}")

    if os.getenv('XORB_SIMULATE_METRICS', 'false').lower() == 'true':
        # Generate sample data in a background thread
        simulation_thread = threading.Thread(target=lambda: simulate_metrics(), daemon=True)
        simulation_thread.start()

    try:
        app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        logger.info("Metrics server shutting down...")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XORB SLI Metrics Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9090, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--simulate", action="store_true", help="Generate sample metrics")

    args = parser.parse_args()

    if args.simulate:
        os.environ['XORB_SIMULATE_METRICS'] = 'true'

    start_metrics_server(host=args.host, port=args.port, debug=args.debug)
