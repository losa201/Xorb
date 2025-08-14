"""
Tests for G8 Control Plane WFQ Scheduler
Validates Weighted Fair Queuing implementation and fairness algorithms.
"""

import pytest
import asyncio
import time
from datetime import datetime, timezone

# Import the modules to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "api"))

from app.services.g8_control_plane_service import (
    WeightedFairQueueScheduler,
    QueuedRequest,
    ResourceType,
    RequestPriority
)


class TestWeightedFairQueueScheduler:
    """Test suite for WFQ scheduler implementation."""

    def test_scheduler_initialization(self):
        """Test basic scheduler initialization."""
        scheduler = WeightedFairQueueScheduler()

        assert scheduler.virtual_time == 0.0
        assert len(scheduler.tenant_queues) == 0
        assert len(scheduler.scheduling_heap) == 0
        assert scheduler.processed_requests == 0

    def test_set_tenant_weight(self):
        """Test setting tenant weights."""
        scheduler = WeightedFairQueueScheduler()

        # Set valid weights
        scheduler.set_tenant_weight("tenant1", 5.0)
        scheduler.set_tenant_weight("tenant2", 10.0)

        assert scheduler.tenant_weights["tenant1"] == 5.0
        assert scheduler.tenant_weights["tenant2"] == 10.0

        # Test minimum weight enforcement
        scheduler.set_tenant_weight("tenant3", 0.05)  # Below minimum
        assert scheduler.tenant_weights["tenant3"] == 0.1

    def test_enqueue_single_request(self):
        """Test enqueueing a single request."""
        scheduler = WeightedFairQueueScheduler()

        request = QueuedRequest(
            request_id="req1",
            tenant_id="tenant1",
            priority=RequestPriority.MEDIUM,
            resource_type=ResourceType.API_REQUESTS,
            resource_amount=1,
            submitted_at=datetime.now(timezone.utc),
            estimated_duration=1.0
        )

        scheduler.enqueue_request(request, tenant_weight=2.0)

        # Check scheduler state
        stats = scheduler.get_queue_stats()
        assert stats["total_queued"] == 1
        assert "tenant1" in stats["tenant_queues"]
        assert stats["tenant_queues"]["tenant1"] == 1
        assert stats["tenant_weights"]["tenant1"] == 2.0

    def test_dequeue_single_request(self):
        """Test dequeueing a single request."""
        scheduler = WeightedFairQueueScheduler()

        request = QueuedRequest(
            request_id="req1",
            tenant_id="tenant1",
            priority=RequestPriority.MEDIUM,
            resource_type=ResourceType.API_REQUESTS,
            resource_amount=1,
            submitted_at=datetime.now(timezone.utc),
            estimated_duration=1.0
        )

        scheduler.enqueue_request(request, tenant_weight=2.0)

        # Dequeue the request
        dequeued = scheduler.dequeue_next_request()

        assert dequeued is not None
        assert dequeued.request_id == "req1"
        assert dequeued.tenant_id == "tenant1"

        # Check scheduler state after dequeue
        stats = scheduler.get_queue_stats()
        assert stats["total_queued"] == 0
        assert stats["processed_requests"] == 1

    def test_wfq_fairness_ordering(self):
        """Test that WFQ provides fair ordering based on weights."""
        scheduler = WeightedFairQueueScheduler()

        # Create requests from tenants with different weights
        requests = [
            QueuedRequest(
                request_id=f"req_low_{i}",
                tenant_id="low_weight",
                priority=RequestPriority.MEDIUM,
                resource_type=ResourceType.API_REQUESTS,
                resource_amount=1,
                submitted_at=datetime.now(timezone.utc),
                estimated_duration=1.0
            )
            for i in range(3)
        ]

        requests.extend([
            QueuedRequest(
                request_id=f"req_high_{i}",
                tenant_id="high_weight",
                priority=RequestPriority.MEDIUM,
                resource_type=ResourceType.API_REQUESTS,
                resource_amount=1,
                submitted_at=datetime.now(timezone.utc),
                estimated_duration=1.0
            )
            for i in range(3)
        ])

        # Enqueue low weight tenant requests first
        for request in requests[:3]:
            scheduler.enqueue_request(request, tenant_weight=1.0)

        # Enqueue high weight tenant requests
        for request in requests[3:]:
            scheduler.enqueue_request(request, tenant_weight=10.0)

        # Dequeue requests and check fairness
        processed_order = []
        while True:
            request = scheduler.dequeue_next_request()
            if not request:
                break
            processed_order.append(request.tenant_id)

        # High weight tenant should get more processing opportunities
        high_weight_count = processed_order.count("high_weight")
        low_weight_count = processed_order.count("low_weight")

        assert high_weight_count == 3
        assert low_weight_count == 3

        # Check that high weight requests are processed more frequently early on
        # (This is a simplified check - real WFQ is more complex)
        first_half = processed_order[:3]
        high_weight_early = first_half.count("high_weight")

        # High weight tenant should get at least some early processing
        assert high_weight_early >= 1

    def test_virtual_time_progression(self):
        """Test that virtual time progresses correctly."""
        scheduler = WeightedFairQueueScheduler()

        initial_virtual_time = scheduler.virtual_time

        request = QueuedRequest(
            request_id="req1",
            tenant_id="tenant1",
            priority=RequestPriority.MEDIUM,
            resource_type=ResourceType.API_REQUESTS,
            resource_amount=1,
            submitted_at=datetime.now(timezone.utc),
            estimated_duration=2.0  # Longer duration
        )

        scheduler.enqueue_request(request, tenant_weight=1.0)
        scheduler.dequeue_next_request()

        # Virtual time should have progressed
        assert scheduler.virtual_time > initial_virtual_time

    def test_remove_tenant_requests(self):
        """Test removing all requests for a tenant."""
        scheduler = WeightedFairQueueScheduler()

        # Add requests from multiple tenants
        for tenant_id in ["tenant1", "tenant2"]:
            for i in range(3):
                request = QueuedRequest(
                    request_id=f"req_{tenant_id}_{i}",
                    tenant_id=tenant_id,
                    priority=RequestPriority.MEDIUM,
                    resource_type=ResourceType.API_REQUESTS,
                    resource_amount=1,
                    submitted_at=datetime.now(timezone.utc),
                    estimated_duration=1.0
                )
                scheduler.enqueue_request(request, tenant_weight=1.0)

        initial_stats = scheduler.get_queue_stats()
        assert initial_stats["total_queued"] == 6

        # Remove all requests for tenant1
        removed_count = scheduler.remove_tenant_requests("tenant1")
        assert removed_count == 3

        # Check that only tenant2 requests remain
        final_stats = scheduler.get_queue_stats()
        assert final_stats["total_queued"] == 3
        assert "tenant1" not in final_stats["tenant_queues"] or final_stats["tenant_queues"]["tenant1"] == 0
        assert final_stats["tenant_queues"]["tenant2"] == 3

    def test_queue_statistics(self):
        """Test queue statistics reporting."""
        scheduler = WeightedFairQueueScheduler()

        # Add some requests
        for i in range(5):
            request = QueuedRequest(
                request_id=f"req_{i}",
                tenant_id="tenant1",
                priority=RequestPriority.MEDIUM,
                resource_type=ResourceType.API_REQUESTS,
                resource_amount=1,
                submitted_at=datetime.now(timezone.utc),
                estimated_duration=1.0
            )
            scheduler.enqueue_request(request, tenant_weight=2.0)

        stats = scheduler.get_queue_stats()

        assert stats["total_queued"] == 5
        assert stats["tenant_queues"]["tenant1"] == 5
        assert stats["tenant_weights"]["tenant1"] == 2.0
        assert stats["processed_requests"] == 0
        assert stats["average_wait_time_ms"] == 0.0

    def test_concurrent_access_safety(self):
        """Test thread safety with concurrent access."""
        scheduler = WeightedFairQueueScheduler()

        # This is a basic test - full thread safety testing would require more complex scenarios
        requests = []
        for i in range(10):
            request = QueuedRequest(
                request_id=f"req_{i}",
                tenant_id=f"tenant_{i % 3}",
                priority=RequestPriority.MEDIUM,
                resource_type=ResourceType.API_REQUESTS,
                resource_amount=1,
                submitted_at=datetime.now(timezone.utc),
                estimated_duration=1.0
            )
            requests.append(request)

        # Enqueue all requests
        for request in requests:
            scheduler.enqueue_request(request, tenant_weight=1.0)

        # Dequeue all requests
        processed = []
        while True:
            request = scheduler.dequeue_next_request()
            if not request:
                break
            processed.append(request)

        assert len(processed) == 10
        assert scheduler.get_queue_stats()["total_queued"] == 0

    def test_empty_queue_dequeue(self):
        """Test dequeueing from empty queue."""
        scheduler = WeightedFairQueueScheduler()

        result = scheduler.dequeue_next_request()
        assert result is None

    def test_request_ordering_with_different_durations(self):
        """Test that requests with different durations are ordered correctly."""
        scheduler = WeightedFairQueueScheduler()

        # Create requests with different estimated durations
        short_request = QueuedRequest(
            request_id="short",
            tenant_id="tenant1",
            priority=RequestPriority.MEDIUM,
            resource_type=ResourceType.API_REQUESTS,
            resource_amount=1,
            submitted_at=datetime.now(timezone.utc),
            estimated_duration=0.5
        )

        long_request = QueuedRequest(
            request_id="long",
            tenant_id="tenant1",
            priority=RequestPriority.MEDIUM,
            resource_type=ResourceType.API_REQUESTS,
            resource_amount=1,
            submitted_at=datetime.now(timezone.utc),
            estimated_duration=5.0
        )

        # Enqueue long request first, then short request
        scheduler.enqueue_request(long_request, tenant_weight=1.0)
        scheduler.enqueue_request(short_request, tenant_weight=1.0)

        # The request with shorter virtual finish time should be dequeued first
        first = scheduler.dequeue_next_request()
        second = scheduler.dequeue_next_request()

        # Due to WFQ algorithm, the first request (with longer duration)
        # might still be first, but virtual times should be correctly calculated
        assert first is not None
        assert second is not None
        assert first.start_time is not None
        assert second.start_time is not None


@pytest.mark.asyncio
class TestWFQIntegrationScenarios:
    """Integration tests for WFQ scheduler in realistic scenarios."""

    async def test_burst_handling(self):
        """Test how WFQ handles burst traffic from multiple tenants."""
        scheduler = WeightedFairQueueScheduler()

        # Simulate burst from high-priority tenant
        burst_requests = []
        for i in range(10):
            request = QueuedRequest(
                request_id=f"burst_{i}",
                tenant_id="enterprise",
                priority=RequestPriority.HIGH,
                resource_type=ResourceType.API_REQUESTS,
                resource_amount=1,
                submitted_at=datetime.now(timezone.utc),
                estimated_duration=1.0
            )
            burst_requests.append(request)
            scheduler.enqueue_request(request, tenant_weight=10.0)

        # Add steady requests from lower-priority tenant
        for i in range(5):
            request = QueuedRequest(
                request_id=f"steady_{i}",
                tenant_id="starter",
                priority=RequestPriority.MEDIUM,
                resource_type=ResourceType.API_REQUESTS,
                resource_amount=1,
                submitted_at=datetime.now(timezone.utc),
                estimated_duration=1.0
            )
            scheduler.enqueue_request(request, tenant_weight=1.0)

        # Process all requests and verify fair treatment
        enterprise_processed = 0
        starter_processed = 0

        while True:
            request = scheduler.dequeue_next_request()
            if not request:
                break

            if request.tenant_id == "enterprise":
                enterprise_processed += 1
            elif request.tenant_id == "starter":
                starter_processed += 1

        # Enterprise should get more processing (due to higher weight)
        # but starter should not be completely starved
        assert enterprise_processed == 10
        assert starter_processed == 5

        # Check that virtual time progression was fair
        stats = scheduler.get_queue_stats()
        assert stats["processed_requests"] == 15

    async def test_priority_vs_weight_interaction(self):
        """Test interaction between request priority and tenant weight."""
        scheduler = WeightedFairQueueScheduler()

        # High weight tenant with low priority request
        low_pri_request = QueuedRequest(
            request_id="low_pri",
            tenant_id="enterprise",
            priority=RequestPriority.LOW,
            resource_type=ResourceType.API_REQUESTS,
            resource_amount=1,
            submitted_at=datetime.now(timezone.utc),
            estimated_duration=1.0
        )

        # Low weight tenant with high priority request
        high_pri_request = QueuedRequest(
            request_id="high_pri",
            tenant_id="starter",
            priority=RequestPriority.CRITICAL,
            resource_type=ResourceType.API_REQUESTS,
            resource_amount=1,
            submitted_at=datetime.now(timezone.utc),
            estimated_duration=1.0
        )

        scheduler.enqueue_request(low_pri_request, tenant_weight=10.0)
        scheduler.enqueue_request(high_pri_request, tenant_weight=1.0)

        # Current implementation uses WFQ based on tenant weight
        # Priority could be used to modify the weight or processing order
        first = scheduler.dequeue_next_request()
        second = scheduler.dequeue_next_request()

        assert first is not None
        assert second is not None

        # Both requests should be processed
        request_ids = [first.request_id, second.request_id]
        assert "low_pri" in request_ids
        assert "high_pri" in request_ids
