"""
XORB Phase G5 Service Level Indicator (SLI) Metrics
Core reliability metrics for customer-facing SLOs
"""

import time
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from opentelemetry import metrics
from .instrumentation import get_meter


@dataclass
class SLITarget:
    """SLI target definition with error budget."""
    name: str
    target_percentile: float  # e.g., 0.95 for P95
    target_value_ms: float    # Target latency in milliseconds
    error_budget_percent: float = 1.0  # 1% error budget (99% SLO)
    measurement_window_hours: int = 24  # Rolling window for SLO calculation


class SLIMetrics:
    """
    Comprehensive SLI metrics collection for XORB platform reliability.

    Tracks the core customer-facing reliability signals:
    - bus_publish_to_deliver_p95_ms: Message bus latency
    - evidence_ingest_p95_ms: Evidence processing latency
    - auth_error_rate: Authentication failure rate
    - mtls_handshake_fail_rate: mTLS connection failure rate
    - replay_backlog_depth: Replay stream backlog (read-only)
    """

    def __init__(self):
        self.meter = get_meter()
        self._setup_core_metrics()
        self._sli_targets = self._define_sli_targets()

        # State tracking for derived metrics
        self._bus_publish_times: Dict[str, float] = {}
        self._evidence_ingest_times: Dict[str, float] = {}

    def _setup_core_metrics(self) -> None:
        """Initialize all core SLI metrics."""

        # Bus publish-to-deliver latency
        self.bus_publish_to_deliver_duration = self.meter.create_histogram(
            "bus_publish_to_deliver_duration_ms",
            description="Time from message publish to consumer delivery (milliseconds)",
            unit="ms"
        )

        # Evidence ingest latency
        self.evidence_ingest_duration = self.meter.create_histogram(
            "evidence_ingest_duration_ms",
            description="Evidence processing end-to-end latency (milliseconds)",
            unit="ms"
        )

        # Authentication error rate
        self.auth_requests_total = self.meter.create_counter(
            "auth_requests_total",
            description="Total authentication requests"
        )

        self.auth_errors_total = self.meter.create_counter(
            "auth_errors_total",
            description="Total authentication errors"
        )

        # mTLS handshake failures
        self.mtls_handshakes_total = self.meter.create_counter(
            "mtls_handshakes_total",
            description="Total mTLS handshake attempts"
        )

        self.mtls_handshake_failures_total = self.meter.create_counter(
            "mtls_handshake_failures_total",
            description="Total mTLS handshake failures"
        )

        # Replay backlog depth (read-only from existing replay system)
        self.replay_backlog_depth = self.meter.create_up_down_counter(
            "replay_backlog_depth_messages",
            description="Current replay stream backlog depth in messages"
        )

        # Error budget tracking
        self.slo_error_budget_remaining = self.meter.create_gauge(
            "slo_error_budget_remaining_percent",
            description="Remaining SLO error budget percentage"
        )

        self.slo_budget_burn_rate = self.meter.create_gauge(
            "slo_budget_burn_rate_hourly",
            description="Current SLO error budget burn rate per hour"
        )

        print("✅ SLI metrics initialized")

    def _define_sli_targets(self) -> List[SLITarget]:
        """Define SLI targets and SLOs for the platform."""
        return [
            SLITarget(
                name="bus_publish_to_deliver_p95_ms",
                target_percentile=0.95,
                target_value_ms=100.0,  # P95 < 100ms
                error_budget_percent=1.0  # 99% SLO
            ),
            SLITarget(
                name="evidence_ingest_p95_ms",
                target_percentile=0.95,
                target_value_ms=500.0,  # P95 < 500ms for evidence processing
                error_budget_percent=0.5  # 99.5% SLO (higher bar for evidence)
            ),
            SLITarget(
                name="auth_error_rate",
                target_percentile=1.0,  # Error rate (not percentile)
                target_value_ms=1.0,   # < 1% error rate
                error_budget_percent=0.1  # 99.9% SLO for auth
            ),
            SLITarget(
                name="mtls_handshake_fail_rate",
                target_percentile=1.0,  # Error rate (not percentile)
                target_value_ms=0.5,   # < 0.5% failure rate
                error_budget_percent=0.1  # 99.9% SLO for mTLS
            )
        ]

    # Bus Publish-to-Deliver Latency Tracking
    def record_bus_publish_start(self, message_id: str, tenant_id: str, subject: str) -> None:
        """Record the start of a bus message publish operation."""
        self._bus_publish_times[message_id] = time.time()

        # Record publish attempt
        labels = {
            "tenant_id": tenant_id,
            "subject_prefix": subject.split('.')[0] if '.' in subject else subject,
            "operation": "publish"
        }

        bus_operations = self.meter.create_counter(
            "bus_operations_total",
            description="Total bus operations"
        )
        bus_operations.add(1, labels)

    def record_bus_deliver_complete(
        self,
        message_id: str,
        tenant_id: str,
        subject: str,
        consumer_name: str,
        success: bool = True
    ) -> None:
        """Record completion of message delivery to consumer."""
        if message_id not in self._bus_publish_times:
            return  # No publish start time recorded

        # Calculate publish-to-deliver latency
        publish_time = self._bus_publish_times.pop(message_id)
        latency_ms = (time.time() - publish_time) * 1000

        labels = {
            "tenant_id": tenant_id,
            "subject_prefix": subject.split('.')[0] if '.' in subject else subject,
            "consumer": consumer_name,
            "success": str(success).lower()
        }

        # Record the core SLI metric
        self.bus_publish_to_deliver_duration.record(latency_ms, labels)

        # Record delivery result
        if success:
            bus_deliveries_success = self.meter.create_counter(
                "bus_deliveries_success_total",
                description="Successful message deliveries"
            )
            bus_deliveries_success.add(1, labels)
        else:
            bus_deliveries_failed = self.meter.create_counter(
                "bus_deliveries_failed_total",
                description="Failed message deliveries"
            )
            bus_deliveries_failed.add(1, labels)

    # Evidence Ingest Latency Tracking
    def record_evidence_ingest_start(
        self,
        evidence_id: str,
        tenant_id: str,
        evidence_type: str
    ) -> None:
        """Record the start of evidence ingestion."""
        self._evidence_ingest_times[evidence_id] = time.time()

        labels = {
            "tenant_id": tenant_id,
            "evidence_type": evidence_type,
            "operation": "ingest_start"
        }

        evidence_operations = self.meter.create_counter(
            "evidence_operations_total",
            description="Total evidence operations"
        )
        evidence_operations.add(1, labels)

    def record_evidence_ingest_complete(
        self,
        evidence_id: str,
        tenant_id: str,
        evidence_type: str,
        size_bytes: int,
        success: bool = True,
        error_type: Optional[str] = None
    ) -> None:
        """Record completion of evidence ingestion."""
        if evidence_id not in self._evidence_ingest_times:
            return

        # Calculate end-to-end ingest latency
        ingest_start = self._evidence_ingest_times.pop(evidence_id)
        latency_ms = (time.time() - ingest_start) * 1000

        labels = {
            "tenant_id": tenant_id,
            "evidence_type": evidence_type,
            "success": str(success).lower()
        }

        if error_type:
            labels["error_type"] = error_type

        # Record core SLI metric
        self.evidence_ingest_duration.record(latency_ms, labels)

        # Record evidence size
        evidence_size = self.meter.create_histogram(
            "evidence_size_bytes",
            description="Evidence artifact size in bytes"
        )
        evidence_size.record(size_bytes, labels)

    # Authentication Error Rate Tracking
    def record_auth_request(
        self,
        tenant_id: str,
        auth_method: str,
        success: bool,
        error_type: Optional[str] = None
    ) -> None:
        """Record authentication request and result."""
        labels = {
            "tenant_id": tenant_id,
            "auth_method": auth_method,
            "success": str(success).lower()
        }

        if error_type:
            labels["error_type"] = error_type

        # Record total auth requests
        self.auth_requests_total.add(1, labels)

        # Record errors if failed
        if not success:
            self.auth_errors_total.add(1, labels)

    # mTLS Handshake Failure Rate Tracking
    def record_mtls_handshake(
        self,
        client_cert_subject: str,
        success: bool,
        failure_reason: Optional[str] = None,
        handshake_duration_ms: Optional[float] = None
    ) -> None:
        """Record mTLS handshake attempt and result."""
        labels = {
            "cert_subject": client_cert_subject,
            "success": str(success).lower()
        }

        if failure_reason:
            labels["failure_reason"] = failure_reason

        # Record total handshake attempts
        self.mtls_handshakes_total.add(1, labels)

        # Record failures
        if not success:
            self.mtls_handshake_failures_total.add(1, labels)

        # Record handshake duration if available
        if handshake_duration_ms is not None:
            handshake_duration = self.meter.create_histogram(
                "mtls_handshake_duration_ms",
                description="mTLS handshake duration in milliseconds"
            )
            handshake_duration.record(handshake_duration_ms, labels)

    # Replay Backlog Depth (Read-only from existing G4 implementation)
    def update_replay_backlog_depth(
        self,
        tenant_id: str,
        stream_name: str,
        backlog_messages: int
    ) -> None:
        """Update current replay stream backlog depth."""
        labels = {
            "tenant_id": tenant_id,
            "stream_name": stream_name,
            "stream_type": "replay"
        }

        self.replay_backlog_depth.add(backlog_messages, labels)

    # Error Budget Tracking
    def calculate_error_budget_burn(
        self,
        sli_name: str,
        actual_error_rate: float,
        time_window_hours: float = 1.0
    ) -> float:
        """
        Calculate error budget burn rate for an SLI.

        Args:
            sli_name: Name of the SLI being measured
            actual_error_rate: Current measured error rate (0.0-1.0)
            time_window_hours: Time window for burn rate calculation

        Returns:
            Error budget burn rate as percentage per hour
        """
        target = next((t for t in self._sli_targets if t.name == sli_name), None)
        if not target:
            return 0.0

        target_error_rate = target.error_budget_percent / 100.0
        excess_error_rate = max(0.0, actual_error_rate - target_error_rate)

        # Calculate burn rate as percentage of total budget per hour
        burn_rate = (excess_error_rate / target_error_rate) * 100.0 / time_window_hours

        labels = {
            "sli_name": sli_name,
            "target_error_budget": str(target.error_budget_percent)
        }

        self.slo_budget_burn_rate.set(burn_rate, labels)

        return burn_rate

    def get_sli_targets(self) -> List[SLITarget]:
        """Get all defined SLI targets."""
        return self._sli_targets

    @asynccontextmanager
    async def track_bus_operation(self, message_id: str, tenant_id: str, subject: str):
        """Context manager for tracking complete bus publish-to-deliver operations."""
        self.record_bus_publish_start(message_id, tenant_id, subject)
        try:
            yield
            # Success case handled by explicit record_bus_deliver_complete call
        except Exception as e:
            # Record failed delivery
            self.record_bus_deliver_complete(
                message_id, tenant_id, subject, "unknown", success=False
            )
            raise

    @asynccontextmanager
    async def track_evidence_ingest(
        self,
        evidence_id: str,
        tenant_id: str,
        evidence_type: str
    ):
        """Context manager for tracking complete evidence ingest operations."""
        self.record_evidence_ingest_start(evidence_id, tenant_id, evidence_type)
        try:
            yield
            # Success case handled by explicit record_evidence_ingest_complete call
        except Exception as e:
            # Record failed ingest
            self.record_evidence_ingest_complete(
                evidence_id, tenant_id, evidence_type, 0,
                success=False, error_type=type(e).__name__
            )
            raise


# Global SLI metrics instance
_sli_metrics: Optional[SLIMetrics] = None


def get_sli_metrics() -> SLIMetrics:
    """Get the global SLI metrics instance."""
    global _sli_metrics
    if _sli_metrics is None:
        _sli_metrics = SLIMetrics()
    return _sli_metrics


# Convenience functions for common operations
def record_bus_latency(message_id: str, tenant_id: str, subject: str, latency_ms: float) -> None:
    """Record bus publish-to-deliver latency directly."""
    sli = get_sli_metrics()
    labels = {
        "tenant_id": tenant_id,
        "subject_prefix": subject.split('.')[0] if '.' in subject else subject
    }
    sli.bus_publish_to_deliver_duration.record(latency_ms, labels)


def record_auth_failure(tenant_id: str, auth_method: str, error_type: str) -> None:
    """Record authentication failure."""
    sli = get_sli_metrics()
    sli.record_auth_request(tenant_id, auth_method, False, error_type)


def record_mtls_failure(cert_subject: str, failure_reason: str) -> None:
    """Record mTLS handshake failure."""
    sli = get_sli_metrics()
    sli.record_mtls_handshake(cert_subject, False, failure_reason)
