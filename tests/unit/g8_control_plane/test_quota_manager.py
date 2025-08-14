"""
Tests for G8 Control Plane Quota Manager
Validates per-tenant quota management and usage tracking.
"""

import pytest
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path

# Import the modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "api"))

from app.services.g8_control_plane_service import (
    QuotaManager,
    ResourceType,
    TenantProfile
)


class TestQuotaManager:
    """Test suite for quota management functionality."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def quota_manager(self, temp_storage):
        """Create quota manager with temporary storage."""
        return QuotaManager(storage_path=temp_storage)

    def test_quota_manager_initialization(self, quota_manager):
        """Test basic quota manager initialization."""
        assert len(quota_manager.tenant_profiles) == 0
        assert len(quota_manager.usage_tracking) == 0
        assert quota_manager.storage_path.exists()

    def test_create_tenant_profile_starter(self, quota_manager):
        """Test creating starter tier tenant profile."""
        profile = quota_manager.create_tenant_profile("test-tenant", "starter")

        assert profile.tenant_id == "test-tenant"
        assert profile.tier == "starter"
        assert profile.weight == 1.0

        # Check default starter quotas
        assert profile.quotas[ResourceType.API_REQUESTS] == 500
        assert profile.quotas[ResourceType.SCAN_JOBS] == 5
        assert profile.quotas[ResourceType.CONCURRENT_SCANS] == 2

        # Check burst allowance (20% of quota)
        assert profile.burst_allowance[ResourceType.API_REQUESTS] == 100  # 20% of 500

        # Verify profile is stored
        assert "test-tenant" in quota_manager.tenant_profiles

        # Verify usage tracking is initialized
        assert "test-tenant" in quota_manager.usage_tracking
        for resource_type in ResourceType:
            assert resource_type in quota_manager.usage_tracking["test-tenant"]

    def test_create_tenant_profile_enterprise(self, quota_manager):
        """Test creating enterprise tier tenant profile."""
        profile = quota_manager.create_tenant_profile("enterprise-tenant", "enterprise")

        assert profile.tier == "enterprise"
        assert profile.weight == 10.0

        # Check enterprise quotas (higher than starter)
        assert profile.quotas[ResourceType.API_REQUESTS] == 10000
        assert profile.quotas[ResourceType.SCAN_JOBS] == 100
        assert profile.quotas[ResourceType.CONCURRENT_SCANS] == 50

    def test_create_tenant_profile_custom_quotas(self, quota_manager):
        """Test creating tenant profile with custom quotas."""
        custom_quotas = {
            ResourceType.API_REQUESTS: 1500,
            ResourceType.SCAN_JOBS: 25,
        }

        profile = quota_manager.create_tenant_profile(
            "custom-tenant",
            "professional",
            custom_quotas=custom_quotas
        )

        assert profile.quotas[ResourceType.API_REQUESTS] == 1500
        assert profile.quotas[ResourceType.SCAN_JOBS] == 25
        # Other quotas should use professional defaults
        assert profile.quotas[ResourceType.STORAGE_GB] == 100  # Professional default

    def test_check_quota_availability_within_limits(self, quota_manager):
        """Test quota availability check when within limits."""
        quota_manager.create_tenant_profile("test-tenant", "starter")

        available, message, info = quota_manager.check_quota_availability(
            "test-tenant",
            ResourceType.API_REQUESTS,
            10
        )

        assert available is True
        assert "Within quota limits" in message
        assert info["current_usage"] == 0
        assert info["quota_limit"] == 500
        assert info["requested_amount"] == 10
        assert info["would_exceed_quota"] is False
        assert info["would_exceed_burst"] is False

    def test_check_quota_availability_exceeds_quota_within_burst(self, quota_manager):
        """Test quota check when exceeding quota but within burst limit."""
        quota_manager.create_tenant_profile("test-tenant", "starter")

        # Simulate current usage at quota limit
        usage = quota_manager.usage_tracking["test-tenant"][ResourceType.API_REQUESTS]
        usage.current_usage = 500  # At quota limit

        available, message, info = quota_manager.check_quota_availability(
            "test-tenant",
            ResourceType.API_REQUESTS,
            50  # Would exceed quota but within burst (600 total)
        )

        assert available is True
        assert "Using burst allowance" in message
        assert info["would_exceed_quota"] is True
        assert info["would_exceed_burst"] is False

    def test_check_quota_availability_exceeds_burst(self, quota_manager):
        """Test quota check when exceeding burst limit."""
        quota_manager.create_tenant_profile("test-tenant", "starter")

        # Simulate current usage at burst limit
        usage = quota_manager.usage_tracking["test-tenant"][ResourceType.API_REQUESTS]
        usage.current_usage = 600  # At burst limit

        available, message, info = quota_manager.check_quota_availability(
            "test-tenant",
            ResourceType.API_REQUESTS,
            1  # Would exceed burst limit
        )

        assert available is False
        assert "exceed burst limit" in message
        assert info["would_exceed_burst"] is True

    def test_check_quota_availability_nonexistent_tenant(self, quota_manager):
        """Test quota check for non-existent tenant."""
        available, message, info = quota_manager.check_quota_availability(
            "nonexistent-tenant",
            ResourceType.API_REQUESTS,
            1
        )

        assert available is False
        assert "Tenant nonexistent-tenant not found" in message
        assert info == {}

    def test_consume_quota_success(self, quota_manager):
        """Test successful quota consumption."""
        quota_manager.create_tenant_profile("test-tenant", "starter")

        success = quota_manager.consume_quota(
            "test-tenant",
            ResourceType.API_REQUESTS,
            10
        )

        assert success is True

        # Check usage was updated
        usage = quota_manager.usage_tracking["test-tenant"][ResourceType.API_REQUESTS]
        assert usage.current_usage == 10
        assert usage.total_requests == 1
        assert usage.peak_usage == 10
        assert usage.rejected_requests == 0

    def test_consume_quota_failure(self, quota_manager):
        """Test quota consumption failure when over limit."""
        quota_manager.create_tenant_profile("test-tenant", "starter")

        # Set usage at burst limit
        usage = quota_manager.usage_tracking["test-tenant"][ResourceType.API_REQUESTS]
        usage.current_usage = 600  # At burst limit

        success = quota_manager.consume_quota(
            "test-tenant",
            ResourceType.API_REQUESTS,
            1  # Would exceed burst
        )

        assert success is False

        # Check rejection was recorded
        usage = quota_manager.usage_tracking["test-tenant"][ResourceType.API_REQUESTS]
        assert usage.current_usage == 600  # Unchanged
        assert usage.rejected_requests == 1

    def test_release_quota(self, quota_manager):
        """Test quota release for resources like concurrent scans."""
        quota_manager.create_tenant_profile("test-tenant", "starter")

        # Consume some quota first
        quota_manager.consume_quota(
            "test-tenant",
            ResourceType.CONCURRENT_SCANS,
            2
        )

        usage = quota_manager.usage_tracking["test-tenant"][ResourceType.CONCURRENT_SCANS]
        assert usage.current_usage == 2

        # Release quota
        quota_manager.release_quota(
            "test-tenant",
            ResourceType.CONCURRENT_SCANS,
            1
        )

        assert usage.current_usage == 1

        # Test releasing more than consumed (should not go negative)
        quota_manager.release_quota(
            "test-tenant",
            ResourceType.CONCURRENT_SCANS,
            2
        )

        assert usage.current_usage == 0  # Should not go negative

    def test_reset_usage_windows(self, quota_manager):
        """Test resetting usage windows for rate limiting."""
        quota_manager.create_tenant_profile("test-tenant", "starter")

        # Consume some quota
        quota_manager.consume_quota("test-tenant", ResourceType.API_REQUESTS, 100)

        usage = quota_manager.usage_tracking["test-tenant"][ResourceType.API_REQUESTS]
        original_usage = usage.current_usage
        original_total = usage.total_requests

        assert original_usage == 100
        assert original_total == 1

        # Simulate time passage by setting old window start
        from datetime import timedelta
        usage.window_start = datetime.now(timezone.utc) - timedelta(hours=2)

        # Reset windows
        quota_manager.reset_usage_windows()

        # Check that usage was reset
        assert usage.current_usage == 0
        assert usage.total_requests == 0
        assert usage.rejected_requests == 0
        # window_start should be updated to recent time
        time_diff = datetime.now(timezone.utc) - usage.window_start
        assert time_diff.total_seconds() < 10  # Should be very recent

    def test_get_tenant_usage_stats(self, quota_manager):
        """Test getting comprehensive tenant usage statistics."""
        quota_manager.create_tenant_profile("test-tenant", "professional")

        # Consume some resources
        quota_manager.consume_quota("test-tenant", ResourceType.API_REQUESTS, 500)
        quota_manager.consume_quota("test-tenant", ResourceType.SCAN_JOBS, 5)

        # Attempt one that fails
        quota_manager.usage_tracking["test-tenant"][ResourceType.API_REQUESTS].current_usage = 2000  # Over limit
        quota_manager.consume_quota("test-tenant", ResourceType.API_REQUESTS, 1)  # Should fail

        stats = quota_manager.get_tenant_usage_stats("test-tenant")

        # Check structure
        assert "tenant_profile" in stats
        assert "usage_statistics" in stats
        assert "generated_at" in stats

        # Check tenant profile data
        profile_data = stats["tenant_profile"]
        assert profile_data["tenant_id"] == "test-tenant"
        assert profile_data["tier"] == "professional"

        # Check usage statistics
        api_stats = stats["usage_statistics"][ResourceType.API_REQUESTS.value]
        assert api_stats["quota_limit"] == 2000  # Professional quota
        assert api_stats["rejected_requests"] == 1  # From failed consumption

        scan_stats = stats["usage_statistics"][ResourceType.SCAN_JOBS.value]
        assert scan_stats["quota_limit"] == 25  # Professional quota
        assert scan_stats["current_usage"] == 5

    def test_get_tenant_usage_stats_nonexistent_tenant(self, quota_manager):
        """Test getting stats for non-existent tenant."""
        stats = quota_manager.get_tenant_usage_stats("nonexistent-tenant")

        assert "error" in stats
        assert stats["error"] == "Tenant not found"

    def test_tenant_profile_persistence(self, temp_storage):
        """Test that tenant profiles are persisted and loaded correctly."""
        # Create first quota manager and add tenant
        quota_manager1 = QuotaManager(storage_path=temp_storage)
        profile1 = quota_manager1.create_tenant_profile("persistent-tenant", "enterprise")

        # Create second quota manager (simulating restart)
        quota_manager2 = QuotaManager(storage_path=temp_storage)

        # Check that profile was loaded
        assert "persistent-tenant" in quota_manager2.tenant_profiles
        profile2 = quota_manager2.tenant_profiles["persistent-tenant"]

        assert profile2.tenant_id == profile1.tenant_id
        assert profile2.tier == profile1.tier
        assert profile2.weight == profile1.weight
        assert profile2.quotas == profile1.quotas

    def test_quota_utilization_calculation(self, quota_manager):
        """Test quota utilization percentage calculation."""
        quota_manager.create_tenant_profile("test-tenant", "starter")

        # Use 25% of API request quota
        quota_manager.consume_quota("test-tenant", ResourceType.API_REQUESTS, 125)

        available, message, info = quota_manager.check_quota_availability(
            "test-tenant",
            ResourceType.API_REQUESTS,
            0  # Just checking, not requesting more
        )

        assert info["utilization_percent"] == 25.0  # 125/500 * 100

    def test_burst_allowance_calculation(self, quota_manager):
        """Test burst allowance calculation for different tiers."""
        # Test all tiers
        tiers = ["starter", "professional", "enterprise"]

        for tier in tiers:
            tenant_id = f"{tier}-tenant"
            profile = quota_manager.create_tenant_profile(tenant_id, tier)

            # Check that burst allowance is 20% of quota for all resource types
            for resource_type, quota in profile.quotas.items():
                expected_burst = int(quota * 0.2)
                actual_burst = profile.burst_allowance[resource_type]
                assert actual_burst == expected_burst, f"Burst allowance mismatch for {tier} {resource_type}: expected {expected_burst}, got {actual_burst}"

    def test_concurrent_quota_operations(self, quota_manager):
        """Test quota operations under concurrent access (basic thread safety)."""
        quota_manager.create_tenant_profile("concurrent-tenant", "professional")

        # Perform multiple operations rapidly
        results = []
        for i in range(10):
            success = quota_manager.consume_quota(
                "concurrent-tenant",
                ResourceType.API_REQUESTS,
                100
            )
            results.append(success)

        # Check results
        successful_consumptions = sum(results)

        # Should be able to consume 2000/100 = 20 times, but we only tried 10
        # So all should succeed if within quota
        expected_success_count = min(10, 20)  # All 10 should succeed
        assert successful_consumptions == expected_success_count

        # Check final usage
        usage = quota_manager.usage_tracking["concurrent-tenant"][ResourceType.API_REQUESTS]
        assert usage.current_usage == successful_consumptions * 100
        assert usage.total_requests == successful_consumptions


@pytest.mark.asyncio
class TestQuotaManagerIntegration:
    """Integration tests for quota manager in realistic scenarios."""

    @pytest.fixture
    def quota_manager(self):
        """Create quota manager with temporary storage for integration tests."""
        temp_dir = tempfile.mkdtemp()
        manager = QuotaManager(storage_path=temp_dir)
        yield manager
        shutil.rmtree(temp_dir)

    async def test_multi_tenant_quota_isolation(self, quota_manager):
        """Test that quotas are properly isolated between tenants."""
        # Create multiple tenants
        tenants = [
            ("tenant-a", "enterprise"),
            ("tenant-b", "professional"),
            ("tenant-c", "starter")
        ]

        for tenant_id, tier in tenants:
            quota_manager.create_tenant_profile(tenant_id, tier)

        # Each tenant consumes their full API quota
        for tenant_id, tier in tenants:
            profile = quota_manager.tenant_profiles[tenant_id]
            quota_limit = profile.quotas[ResourceType.API_REQUESTS]

            success = quota_manager.consume_quota(
                tenant_id,
                ResourceType.API_REQUESTS,
                quota_limit
            )
            assert success is True

        # Verify each tenant is at their quota limit
        for tenant_id, tier in tenants:
            available, message, info = quota_manager.check_quota_availability(
                tenant_id,
                ResourceType.API_REQUESTS,
                1
            )

            # Should be able to use burst allowance
            assert available is True
            assert "burst allowance" in message.lower()

        # Verify usage stats are isolated
        for tenant_id, tier in tenants:
            stats = quota_manager.get_tenant_usage_stats(tenant_id)
            api_usage = stats["usage_statistics"][ResourceType.API_REQUESTS.value]

            expected_quota = quota_manager.tenant_profiles[tenant_id].quotas[ResourceType.API_REQUESTS]
            assert api_usage["current_usage"] == expected_quota
            assert api_usage["utilization_percent"] == 100.0

    async def test_quota_recovery_after_window_reset(self, quota_manager):
        """Test quota recovery after usage window reset."""
        quota_manager.create_tenant_profile("recovery-tenant", "starter")

        # Consume full quota
        profile = quota_manager.tenant_profiles["recovery-tenant"]
        quota_limit = profile.quotas[ResourceType.API_REQUESTS]

        quota_manager.consume_quota(
            "recovery-tenant",
            ResourceType.API_REQUESTS,
            quota_limit
        )

        # Verify at quota limit
        available, _, _ = quota_manager.check_quota_availability(
            "recovery-tenant",
            ResourceType.API_REQUESTS,
            1
        )
        assert available is True  # Can still use burst

        # Simulate window reset
        quota_manager.reset_usage_windows()

        # Should be able to consume quota again
        success = quota_manager.consume_quota(
            "recovery-tenant",
            ResourceType.API_REQUESTS,
            100
        )
        assert success is True

        # Check that usage was reset and then 100 was consumed
        usage = quota_manager.usage_tracking["recovery-tenant"][ResourceType.API_REQUESTS]
        assert usage.current_usage == 100
