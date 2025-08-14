"""
XORB Phase G8 Control Plane Service
Implements Weighted Fair Queuing (WFQ) scheduler and per-tenant quotas for fair resource allocation.

Provides advanced scheduling algorithms, tenant isolation, and resource management
with fairness guarantees and real-time quota enforcement.
"""

import asyncio
import time
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import heapq
import threading
from pathlib import Path

# Priority and request classification
class RequestPriority(Enum):
    """Request priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class ResourceType(Enum):
    """Types of resources that can be quota-limited."""
    API_REQUESTS = "api_requests"
    SCAN_JOBS = "scan_jobs"
    STORAGE_GB = "storage_gb"
    COMPUTE_HOURS = "compute_hours"
    BANDWIDTH_MBPS = "bandwidth_mbps"
    CONCURRENT_SCANS = "concurrent_scans"


@dataclass
class TenantProfile:
    """Tenant configuration and limits."""
    tenant_id: str
    tier: str  # "enterprise", "professional", "starter"
    weight: float  # WFQ weight (higher = more priority)
    quotas: Dict[ResourceType, int]
    burst_allowance: Dict[ResourceType, int]
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "tier": self.tier,
            "weight": self.weight,
            "quotas": {k.value: v for k, v in self.quotas.items()},
            "burst_allowance": {k.value: v for k, v in self.burst_allowance.items()},
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class QueuedRequest:
    """Request waiting in WFQ scheduler."""
    request_id: str
    tenant_id: str
    priority: RequestPriority
    resource_type: ResourceType
    resource_amount: int
    submitted_at: datetime
    start_time: Optional[float] = None  # Virtual start time for WFQ
    estimated_duration: float = 1.0  # seconds
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ResourceUsage:
    """Current resource usage for a tenant."""
    tenant_id: str
    resource_type: ResourceType
    current_usage: int
    peak_usage: int
    total_requests: int
    rejected_requests: int
    last_request_time: datetime
    window_start: datetime

    def reset_window(self):
        """Reset usage window (for rate limiting)."""
        self.current_usage = 0
        self.total_requests = 0
        self.rejected_requests = 0
        self.window_start = datetime.now(timezone.utc)


@dataclass
class FairnessMetrics:
    """Fairness tracking metrics."""
    tenant_id: str
    allocated_bandwidth: float
    actual_throughput: float
    fairness_score: float  # 0-1, where 1 is perfectly fair
    queue_wait_time_ms: float
    resource_starvation_count: int
    last_updated: datetime


class WeightedFairQueueScheduler:
    """
    Weighted Fair Queuing scheduler implementation.

    Provides fair scheduling across tenants based on assigned weights,
    preventing any single tenant from monopolizing resources.
    """

    def __init__(self):
        # Per-tenant queues
        self.tenant_queues: Dict[str, deque[QueuedRequest]] = defaultdict(deque)

        # WFQ state
        self.virtual_time: float = 0.0
        self.tenant_virtual_finish_time: Dict[str, float] = defaultdict(float)
        self.tenant_weights: Dict[str, float] = {}

        # Priority queue for scheduling (min-heap by virtual finish time)
        self.scheduling_heap: List[Tuple[float, str, QueuedRequest]] = []

        # Statistics
        self.processed_requests: int = 0
        self.total_wait_time: float = 0.0

        # Thread safety
        self.lock = threading.RLock()

    def set_tenant_weight(self, tenant_id: str, weight: float):
        """Set WFQ weight for tenant."""
        with self.lock:
            self.tenant_weights[tenant_id] = max(0.1, weight)  # Minimum weight

    def enqueue_request(self, request: QueuedRequest, tenant_weight: float):
        """Add request to tenant queue and WFQ scheduler."""
        with self.lock:
            # Set tenant weight
            self.set_tenant_weight(request.tenant_id, tenant_weight)

            # Calculate virtual start time
            current_virtual_time = max(
                self.virtual_time,
                self.tenant_virtual_finish_time[request.tenant_id]
            )

            # Calculate virtual finish time (start + service_time/weight)
            service_time = request.estimated_duration
            weight = self.tenant_weights[request.tenant_id]
            virtual_finish_time = current_virtual_time + (service_time / weight)

            # Update tenant's virtual finish time
            self.tenant_virtual_finish_time[request.tenant_id] = virtual_finish_time
            request.start_time = virtual_finish_time

            # Add to tenant queue
            self.tenant_queues[request.tenant_id].append(request)

            # Add to scheduling heap
            heapq.heappush(
                self.scheduling_heap,
                (virtual_finish_time, request.tenant_id, request)
            )

    def dequeue_next_request(self) -> Optional[QueuedRequest]:
        """Get next request to process based on WFQ algorithm."""
        with self.lock:
            while self.scheduling_heap:
                virtual_finish_time, tenant_id, request = heapq.heappop(self.scheduling_heap)

                # Check if request is still in tenant queue (not cancelled)
                if (self.tenant_queues[tenant_id] and
                    self.tenant_queues[tenant_id][0].request_id == request.request_id):

                    # Remove from tenant queue
                    self.tenant_queues[tenant_id].popleft()

                    # Update virtual time to this request's start time
                    self.virtual_time = max(self.virtual_time, request.start_time)

                    # Update statistics
                    self.processed_requests += 1
                    wait_time = time.time() - request.submitted_at.timestamp()
                    self.total_wait_time += wait_time

                    return request

            return None

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get current queue statistics."""
        with self.lock:
            tenant_queue_lengths = {
                tenant_id: len(queue)
                for tenant_id, queue in self.tenant_queues.items()
            }

            avg_wait_time = (
                self.total_wait_time / self.processed_requests
                if self.processed_requests > 0 else 0.0
            )

            return {
                "total_queued": sum(tenant_queue_lengths.values()),
                "tenant_queues": tenant_queue_lengths,
                "virtual_time": self.virtual_time,
                "processed_requests": self.processed_requests,
                "average_wait_time_ms": avg_wait_time * 1000,
                "tenant_weights": dict(self.tenant_weights)
            }

    def remove_tenant_requests(self, tenant_id: str) -> int:
        """Remove all requests for a tenant (e.g., for quota violations)."""
        with self.lock:
            removed_count = len(self.tenant_queues[tenant_id])
            self.tenant_queues[tenant_id].clear()

            # Rebuild scheduling heap without this tenant's requests
            new_heap = []
            for vft, tid, req in self.scheduling_heap:
                if tid != tenant_id:
                    new_heap.append((vft, tid, req))

            self.scheduling_heap = new_heap
            heapq.heapify(self.scheduling_heap)

            return removed_count


class QuotaManager:
    """Manages per-tenant resource quotas and usage tracking."""

    def __init__(self, storage_path: str = "quota_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

        # In-memory state
        self.tenant_profiles: Dict[str, TenantProfile] = {}
        self.usage_tracking: Dict[str, Dict[ResourceType, ResourceUsage]] = defaultdict(dict)

        # Rate limiting windows (1 minute, 1 hour, 1 day)
        self.rate_windows = [60, 3600, 86400]  # seconds

        # Lock for thread safety
        self.lock = threading.RLock()

        # Load existing profiles
        self._load_tenant_profiles()

    def create_tenant_profile(
        self,
        tenant_id: str,
        tier: str = "starter",
        custom_quotas: Optional[Dict[ResourceType, int]] = None
    ) -> TenantProfile:
        """Create new tenant profile with default or custom quotas."""

        # Default quotas by tier
        tier_quotas = {
            "enterprise": {
                ResourceType.API_REQUESTS: 10000,  # per hour
                ResourceType.SCAN_JOBS: 100,       # per day
                ResourceType.STORAGE_GB: 1000,
                ResourceType.COMPUTE_HOURS: 500,   # per month
                ResourceType.BANDWIDTH_MBPS: 1000,
                ResourceType.CONCURRENT_SCANS: 50
            },
            "professional": {
                ResourceType.API_REQUESTS: 2000,
                ResourceType.SCAN_JOBS: 25,
                ResourceType.STORAGE_GB: 100,
                ResourceType.COMPUTE_HOURS: 100,
                ResourceType.BANDWIDTH_MBPS: 200,
                ResourceType.CONCURRENT_SCANS: 10
            },
            "starter": {
                ResourceType.API_REQUESTS: 500,
                ResourceType.SCAN_JOBS: 5,
                ResourceType.STORAGE_GB: 10,
                ResourceType.COMPUTE_HOURS: 20,
                ResourceType.BANDWIDTH_MBPS: 50,
                ResourceType.CONCURRENT_SCANS: 2
            }
        }

        # Tier-based weights for WFQ
        tier_weights = {
            "enterprise": 10.0,
            "professional": 3.0,
            "starter": 1.0
        }

        quotas = custom_quotas or tier_quotas.get(tier, tier_quotas["starter"])
        weight = tier_weights.get(tier, 1.0)

        # Burst allowance (20% above quota for short periods)
        burst_allowance = {rt: int(quota * 0.2) for rt, quota in quotas.items()}

        profile = TenantProfile(
            tenant_id=tenant_id,
            tier=tier,
            weight=weight,
            quotas=quotas,
            burst_allowance=burst_allowance,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )

        with self.lock:
            self.tenant_profiles[tenant_id] = profile
            self._save_tenant_profile(profile)

            # Initialize usage tracking
            for resource_type in ResourceType:
                self.usage_tracking[tenant_id][resource_type] = ResourceUsage(
                    tenant_id=tenant_id,
                    resource_type=resource_type,
                    current_usage=0,
                    peak_usage=0,
                    total_requests=0,
                    rejected_requests=0,
                    last_request_time=datetime.now(timezone.utc),
                    window_start=datetime.now(timezone.utc)
                )

        print(f"✅ Created {tier} tenant profile for {tenant_id} (weight: {weight})")
        return profile

    def check_quota_availability(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        amount: int = 1
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Check if tenant has quota available for resource request."""

        with self.lock:
            if tenant_id not in self.tenant_profiles:
                return False, f"Tenant {tenant_id} not found", {}

            profile = self.tenant_profiles[tenant_id]
            usage = self.usage_tracking[tenant_id][resource_type]

            # Get quota limit
            quota_limit = profile.quotas.get(resource_type, 0)
            burst_limit = quota_limit + profile.burst_allowance.get(resource_type, 0)

            # Check current usage
            would_exceed_quota = usage.current_usage + amount > quota_limit
            would_exceed_burst = usage.current_usage + amount > burst_limit

            # Calculate utilization
            utilization = (usage.current_usage / quota_limit * 100) if quota_limit > 0 else 0

            quota_info = {
                "tenant_id": tenant_id,
                "resource_type": resource_type.value,
                "current_usage": usage.current_usage,
                "quota_limit": quota_limit,
                "burst_limit": burst_limit,
                "requested_amount": amount,
                "would_exceed_quota": would_exceed_quota,
                "would_exceed_burst": would_exceed_burst,
                "utilization_percent": utilization,
                "window_start": usage.window_start.isoformat()
            }

            if would_exceed_burst:
                return False, f"Request would exceed burst limit ({burst_limit})", quota_info
            elif would_exceed_quota:
                # Allow burst usage but warn
                return True, f"Using burst allowance ({quota_limit} → {burst_limit})", quota_info
            else:
                return True, "Within quota limits", quota_info

    def consume_quota(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        amount: int = 1
    ) -> bool:
        """Consume quota for a tenant resource."""

        available, message, info = self.check_quota_availability(tenant_id, resource_type, amount)

        if available:
            with self.lock:
                usage = self.usage_tracking[tenant_id][resource_type]
                usage.current_usage += amount
                usage.total_requests += 1
                usage.peak_usage = max(usage.peak_usage, usage.current_usage)
                usage.last_request_time = datetime.now(timezone.utc)
                return True
        else:
            with self.lock:
                usage = self.usage_tracking[tenant_id][resource_type]
                usage.rejected_requests += 1
                return False

    def release_quota(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        amount: int = 1
    ):
        """Release consumed quota (for resources like concurrent_scans)."""

        with self.lock:
            if tenant_id in self.usage_tracking:
                usage = self.usage_tracking[tenant_id][resource_type]
                usage.current_usage = max(0, usage.current_usage - amount)

    def reset_usage_windows(self):
        """Reset usage tracking windows (called by background task)."""

        current_time = datetime.now(timezone.utc)

        with self.lock:
            for tenant_id, resource_usage in self.usage_tracking.items():
                for resource_type, usage in resource_usage.items():
                    # Reset if window has expired (1 hour window)
                    if (current_time - usage.window_start).total_seconds() > 3600:
                        usage.reset_window()

    def get_tenant_usage_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Get comprehensive usage statistics for tenant."""

        with self.lock:
            if tenant_id not in self.tenant_profiles:
                return {"error": "Tenant not found"}

            profile = self.tenant_profiles[tenant_id]
            usage_stats = {}

            for resource_type in ResourceType:
                usage = self.usage_tracking[tenant_id][resource_type]
                quota_limit = profile.quotas.get(resource_type, 0)
                burst_limit = quota_limit + profile.burst_allowance.get(resource_type, 0)

                usage_stats[resource_type.value] = {
                    "current_usage": usage.current_usage,
                    "quota_limit": quota_limit,
                    "burst_limit": burst_limit,
                    "utilization_percent": (usage.current_usage / quota_limit * 100) if quota_limit > 0 else 0,
                    "total_requests": usage.total_requests,
                    "rejected_requests": usage.rejected_requests,
                    "peak_usage": usage.peak_usage,
                    "last_request": usage.last_request_time.isoformat(),
                    "window_start": usage.window_start.isoformat()
                }

            return {
                "tenant_profile": profile.to_dict(),
                "usage_statistics": usage_stats,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }

    def _load_tenant_profiles(self):
        """Load tenant profiles from storage."""
        try:
            profiles_file = self.storage_path / "tenant_profiles.json"
            if profiles_file.exists():
                with open(profiles_file, 'r') as f:
                    profiles_data = json.load(f)

                for tenant_data in profiles_data:
                    profile = TenantProfile(
                        tenant_id=tenant_data["tenant_id"],
                        tier=tenant_data["tier"],
                        weight=tenant_data["weight"],
                        quotas={ResourceType(k): v for k, v in tenant_data["quotas"].items()},
                        burst_allowance={ResourceType(k): v for k, v in tenant_data["burst_allowance"].items()},
                        created_at=datetime.fromisoformat(tenant_data["created_at"]),
                        updated_at=datetime.fromisoformat(tenant_data["updated_at"])
                    )
                    self.tenant_profiles[profile.tenant_id] = profile

                print(f"✅ Loaded {len(self.tenant_profiles)} tenant profiles")
        except Exception as e:
            print(f"⚠️ Failed to load tenant profiles: {e}")

    def _save_tenant_profile(self, profile: TenantProfile):
        """Save single tenant profile to storage."""
        try:
            profiles_file = self.storage_path / "tenant_profiles.json"

            # Load existing profiles
            profiles_data = []
            if profiles_file.exists():
                with open(profiles_file, 'r') as f:
                    profiles_data = json.load(f)

            # Update or add this profile
            updated = False
            for i, existing in enumerate(profiles_data):
                if existing["tenant_id"] == profile.tenant_id:
                    profiles_data[i] = profile.to_dict()
                    updated = True
                    break

            if not updated:
                profiles_data.append(profile.to_dict())

            # Save back to file
            with open(profiles_file, 'w') as f:
                json.dump(profiles_data, f, indent=2, default=str)

        except Exception as e:
            print(f"❌ Failed to save tenant profile: {e}")


class FairnessEngine:
    """Monitors and enforces fairness across tenants."""

    def __init__(self, quota_manager: QuotaManager):
        self.quota_manager = quota_manager
        self.fairness_metrics: Dict[str, FairnessMetrics] = {}
        self.fairness_history: List[Dict[str, Any]] = []

        # Fairness thresholds
        self.min_fairness_score = 0.7  # Below this triggers rebalancing
        self.starvation_threshold = 10  # Max rejected requests before intervention

    def calculate_fairness_scores(self) -> Dict[str, FairnessMetrics]:
        """Calculate fairness scores for all tenants."""

        current_time = datetime.now(timezone.utc)
        tenant_metrics = {}

        # Get all tenant usage
        for tenant_id in self.quota_manager.tenant_profiles.keys():
            usage_stats = self.quota_manager.get_tenant_usage_stats(tenant_id)
            profile = self.quota_manager.tenant_profiles[tenant_id]

            # Calculate fairness score based on:
            # 1. Actual throughput vs entitled throughput (based on weight)
            # 2. Queue wait times
            # 3. Resource starvation incidents

            api_usage = usage_stats["usage_statistics"][ResourceType.API_REQUESTS.value]
            entitled_share = profile.weight  # Simplified - would be more complex in reality
            actual_throughput = api_usage["total_requests"]

            # Fairness score (simplified calculation)
            if entitled_share > 0:
                fairness_score = min(1.0, actual_throughput / (entitled_share * 100))
            else:
                fairness_score = 0.0

            metrics = FairnessMetrics(
                tenant_id=tenant_id,
                allocated_bandwidth=entitled_share,
                actual_throughput=actual_throughput,
                fairness_score=fairness_score,
                queue_wait_time_ms=0.0,  # Would come from WFQ scheduler
                resource_starvation_count=api_usage["rejected_requests"],
                last_updated=current_time
            )

            tenant_metrics[tenant_id] = metrics

        self.fairness_metrics = tenant_metrics
        return tenant_metrics

    def identify_fairness_violations(self) -> List[Dict[str, Any]]:
        """Identify tenants experiencing unfair treatment."""

        violations = []
        fairness_scores = self.calculate_fairness_scores()

        for tenant_id, metrics in fairness_scores.items():
            if metrics.fairness_score < self.min_fairness_score:
                violations.append({
                    "tenant_id": tenant_id,
                    "violation_type": "low_fairness_score",
                    "fairness_score": metrics.fairness_score,
                    "threshold": self.min_fairness_score,
                    "severity": "high" if metrics.fairness_score < 0.5 else "medium"
                })

            if metrics.resource_starvation_count >= self.starvation_threshold:
                violations.append({
                    "tenant_id": tenant_id,
                    "violation_type": "resource_starvation",
                    "rejected_requests": metrics.resource_starvation_count,
                    "threshold": self.starvation_threshold,
                    "severity": "critical"
                })

        return violations

    def generate_fairness_report(self) -> Dict[str, Any]:
        """Generate comprehensive fairness report."""

        fairness_scores = self.calculate_fairness_scores()
        violations = self.identify_fairness_violations()

        # Overall system fairness (Jain's Fairness Index)
        if fairness_scores:
            scores = [m.fairness_score for m in fairness_scores.values()]
            sum_scores = sum(scores)
            sum_squares = sum(score ** 2 for score in scores)
            n = len(scores)

            # Jain's Fairness Index
            system_fairness = (sum_scores ** 2) / (n * sum_squares) if sum_squares > 0 else 0
        else:
            system_fairness = 1.0

        report = {
            "report_generated_at": datetime.now(timezone.utc).isoformat(),
            "system_fairness_index": system_fairness,
            "tenant_count": len(fairness_scores),
            "violations_count": len(violations),
            "tenant_metrics": {
                tid: {
                    "fairness_score": metrics.fairness_score,
                    "allocated_bandwidth": metrics.allocated_bandwidth,
                    "actual_throughput": metrics.actual_throughput,
                    "starvation_count": metrics.resource_starvation_count
                }
                for tid, metrics in fairness_scores.items()
            },
            "fairness_violations": violations,
            "recommendations": self._generate_fairness_recommendations(violations)
        }

        # Store in history
        self.fairness_history.append(report)

        # Keep only last 24 hours of history
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        self.fairness_history = [
            r for r in self.fairness_history
            if datetime.fromisoformat(r["report_generated_at"]) > cutoff_time
        ]

        return report

    def _generate_fairness_recommendations(self, violations: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations to improve fairness."""

        recommendations = []

        for violation in violations:
            tenant_id = violation["tenant_id"]

            if violation["violation_type"] == "low_fairness_score":
                recommendations.append(
                    f"Consider increasing WFQ weight for tenant {tenant_id} or investigating resource bottlenecks"
                )
            elif violation["violation_type"] == "resource_starvation":
                recommendations.append(
                    f"Tenant {tenant_id} experiencing resource starvation - review quota allocation or system capacity"
                )

        if not recommendations:
            recommendations.append("System fairness is within acceptable parameters")

        return recommendations


class G8ControlPlaneService:
    """
    Main Control Plane service implementing WFQ scheduling and quota management.

    Orchestrates fair resource allocation across tenants using advanced scheduling
    algorithms and real-time quota enforcement.
    """

    def __init__(self, storage_path: str = "control_plane_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

        # Core components
        self.wfq_scheduler = WeightedFairQueueScheduler()
        self.quota_manager = QuotaManager(str(self.storage_path / "quotas"))
        self.fairness_engine = FairnessEngine(self.quota_manager)

        # Processing state
        self.active_requests: Dict[str, QueuedRequest] = {}
        self.processing_stats = {
            "total_processed": 0,
            "total_rejected": 0,
            "average_wait_time": 0.0,
            "fairness_violations": 0
        }

        # Background task control
        self._running = False
        self._background_task = None

    async def start(self):
        """Start the control plane background processing."""
        if self._running:
            return

        self._running = True
        self._background_task = asyncio.create_task(self._background_processor())
        print("🎛️ G8 Control Plane started with WFQ scheduler and quota management")

    async def stop(self):
        """Stop the control plane."""
        self._running = False
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        print("🛑 G8 Control Plane stopped")

    async def submit_request(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        priority: RequestPriority = RequestPriority.MEDIUM,
        resource_amount: int = 1,
        estimated_duration: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Submit a resource request for scheduling and quota check.

        Returns:
            (accepted, message, request_id)
        """

        # Check quota availability
        available, quota_message, quota_info = self.quota_manager.check_quota_availability(
            tenant_id, resource_type, resource_amount
        )

        if not available:
            self.processing_stats["total_rejected"] += 1
            return False, f"Quota exceeded: {quota_message}", None

        # Create request
        request_id = f"req_{tenant_id}_{int(time.time() * 1000000)}"
        request = QueuedRequest(
            request_id=request_id,
            tenant_id=tenant_id,
            priority=priority,
            resource_type=resource_type,
            resource_amount=resource_amount,
            submitted_at=datetime.now(timezone.utc),
            estimated_duration=estimated_duration,
            metadata=metadata or {}
        )

        # Get tenant profile for weight
        if tenant_id not in self.quota_manager.tenant_profiles:
            # Create default profile
            self.quota_manager.create_tenant_profile(tenant_id)

        tenant_profile = self.quota_manager.tenant_profiles[tenant_id]

        # Add to WFQ scheduler
        self.wfq_scheduler.enqueue_request(request, tenant_profile.weight)

        print(f"📥 Queued request {request_id} for tenant {tenant_id} (type: {resource_type.value})")

        return True, f"Request queued successfully: {quota_message}", request_id

    async def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """Get status of a submitted request."""

        if request_id in self.active_requests:
            request = self.active_requests[request_id]
            return {
                "request_id": request_id,
                "status": "processing",
                "tenant_id": request.tenant_id,
                "submitted_at": request.submitted_at.isoformat(),
                "processing_time": (datetime.now(timezone.utc) - request.submitted_at).total_seconds()
            }

        # Check if still in queue (simplified check)
        queue_stats = self.wfq_scheduler.get_queue_stats()
        in_queue = any(
            request_id in str(queue_stats["tenant_queues"])
            for tenant_id in queue_stats["tenant_queues"]
        )

        if in_queue:
            return {
                "request_id": request_id,
                "status": "queued",
                "estimated_wait_time": queue_stats["average_wait_time_ms"] / 1000
            }

        return {
            "request_id": request_id,
            "status": "completed_or_not_found"
        }

    async def get_tenant_status(self, tenant_id: str) -> Dict[str, Any]:
        """Get comprehensive status for a tenant."""

        # Usage statistics
        usage_stats = self.quota_manager.get_tenant_usage_stats(tenant_id)

        # Queue statistics
        queue_stats = self.wfq_scheduler.get_queue_stats()
        tenant_queue_length = queue_stats["tenant_queues"].get(tenant_id, 0)

        # Fairness metrics
        fairness_metrics = self.fairness_engine.calculate_fairness_scores()
        tenant_fairness = fairness_metrics.get(tenant_id)

        return {
            "tenant_id": tenant_id,
            "queue_length": tenant_queue_length,
            "usage_statistics": usage_stats,
            "fairness_metrics": {
                "fairness_score": tenant_fairness.fairness_score if tenant_fairness else 0.0,
                "resource_starvation_count": tenant_fairness.resource_starvation_count if tenant_fairness else 0
            },
            "status_generated_at": datetime.now(timezone.utc).isoformat()
        }

    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall control plane system status."""

        # Queue statistics
        queue_stats = self.wfq_scheduler.get_queue_stats()

        # Fairness report
        fairness_report = self.fairness_engine.generate_fairness_report()

        # System health
        system_health = {
            "control_plane_running": self._running,
            "total_tenants": len(self.quota_manager.tenant_profiles),
            "total_queued_requests": queue_stats["total_queued"],
            "processing_statistics": self.processing_stats,
            "fairness_index": fairness_report["system_fairness_index"],
            "fairness_violations": fairness_report["violations_count"]
        }

        return {
            "system_health": system_health,
            "queue_statistics": queue_stats,
            "fairness_report": fairness_report,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

    async def _background_processor(self):
        """Background task for processing queued requests."""

        while self._running:
            try:
                # Process next request from WFQ scheduler
                request = self.wfq_scheduler.dequeue_next_request()

                if request:
                    # Consume quota
                    quota_consumed = self.quota_manager.consume_quota(
                        request.tenant_id,
                        request.resource_type,
                        request.resource_amount
                    )

                    if quota_consumed:
                        # Start processing
                        self.active_requests[request.request_id] = request
                        print(f"🔄 Processing request {request.request_id} for tenant {request.tenant_id}")

                        # Simulate processing (in real system, this would trigger actual work)
                        asyncio.create_task(self._simulate_request_processing(request))

                        self.processing_stats["total_processed"] += 1
                    else:
                        print(f"❌ Quota consumption failed for request {request.request_id}")
                        self.processing_stats["total_rejected"] += 1
                else:
                    # No requests to process, sleep briefly
                    await asyncio.sleep(0.1)

                # Periodic maintenance tasks
                if int(time.time()) % 60 == 0:  # Every minute
                    self.quota_manager.reset_usage_windows()
                    await asyncio.sleep(1)  # Avoid duplicate maintenance

                # Brief sleep to prevent CPU spinning
                await asyncio.sleep(0.01)

            except Exception as e:
                print(f"❌ Error in background processor: {e}")
                await asyncio.sleep(1)

    async def _simulate_request_processing(self, request: QueuedRequest):
        """Simulate processing a request (replace with actual processing logic)."""

        try:
            # Simulate processing time
            await asyncio.sleep(request.estimated_duration)

            # Release quota for concurrent resources
            if request.resource_type == ResourceType.CONCURRENT_SCANS:
                self.quota_manager.release_quota(
                    request.tenant_id,
                    request.resource_type,
                    request.resource_amount
                )

            # Remove from active requests
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]

            print(f"✅ Completed request {request.request_id} for tenant {request.tenant_id}")

        except Exception as e:
            print(f"❌ Error processing request {request.request_id}: {e}")


# Global service instance
_g8_control_plane_service: Optional[G8ControlPlaneService] = None


async def get_g8_control_plane_service() -> G8ControlPlaneService:
    """Get the global G8 control plane service instance."""
    global _g8_control_plane_service
    if _g8_control_plane_service is None:
        _g8_control_plane_service = G8ControlPlaneService()
        await _g8_control_plane_service.start()
    return _g8_control_plane_service


async def shutdown_g8_control_plane_service():
    """Shutdown the global control plane service."""
    global _g8_control_plane_service
    if _g8_control_plane_service:
        await _g8_control_plane_service.stop()
        _g8_control_plane_service = None
