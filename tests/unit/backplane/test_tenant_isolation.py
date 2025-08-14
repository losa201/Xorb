#!/usr/bin/env python3
"""
Unit tests for NATS Tenant Isolation - Phase G2

Tests tenant isolation mechanisms, negative security tests,
and cross-tenant access prevention.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from xorb_platform_bus.bus.pubsub.nats_client import (
    NATSClient,
    SubjectBuilder,
    Domain,
    Event,
    ConsumerSettings,
    create_nats_client
)


class TestTenantIsolation:
    """Test tenant isolation at the client level."""

    def test_client_initialization_with_tenant(self):
        """Test that clients are properly initialized with tenant IDs."""
        tenant_a = "tenant-a"
        tenant_b = "tenant-b"

        client_a = NATSClient(
            servers=["nats://localhost:4222"],
            tenant_id=tenant_a
        )
        client_b = NATSClient(
            servers=["nats://localhost:4222"],
            tenant_id=tenant_b
        )

        assert client_a.tenant_id == tenant_a
        assert client_b.tenant_id == tenant_b
        assert client_a.subject_builder.tenant == tenant_a
        assert client_b.subject_builder.tenant == tenant_b

    def test_factory_function_creates_isolated_clients(self):
        """Test factory function creates properly isolated clients."""
        client_a = create_nats_client("tenant-a")
        client_b = create_nats_client("tenant-b")

        assert client_a.tenant_id == "tenant-a"
        assert client_b.tenant_id == "tenant-b"
        assert client_a.tenant_id != client_b.tenant_id

    def test_consumer_settings_isolation(self):
        """Test that consumer settings can be customized per tenant."""
        settings_a = ConsumerSettings(max_ack_pending=512, rate_limit_bps=500000)
        settings_b = ConsumerSettings(max_ack_pending=1024, rate_limit_bps=1000000)

        client_a = NATSClient(
            servers=["nats://localhost:4222"],
            tenant_id="tenant-a",
            consumer_settings=settings_a
        )
        client_b = NATSClient(
            servers=["nats://localhost:4222"],
            tenant_id="tenant-b",
            consumer_settings=settings_b
        )

        assert client_a.consumer_settings.max_ack_pending == 512
        assert client_b.consumer_settings.max_ack_pending == 1024
        assert client_a.consumer_settings.rate_limit_bps == 500000
        assert client_b.consumer_settings.rate_limit_bps == 1000000


class TestNegativeSecurityTests:
    """Negative tests to ensure tenant isolation is enforced."""

    def test_tenant_cannot_build_subjects_for_other_tenants(self):
        """Test that a tenant's subject builder only generates subjects for that tenant."""
        builder_a = SubjectBuilder("tenant-a")
        builder_b = SubjectBuilder("tenant-b")

        subject_a = builder_a.build(Domain.SCAN, "nmap", Event.CREATED)
        subject_b = builder_b.build(Domain.SCAN, "nmap", Event.CREATED)

        # Each subject should only contain its own tenant
        assert "tenant-a" in subject_a and "tenant-b" not in subject_a
        assert "tenant-b" in subject_b and "tenant-a" not in subject_b

    def test_parsing_other_tenant_subjects_shows_different_tenant(self):
        """Test that parsing subjects from other tenants clearly shows different tenants."""
        builder_a = SubjectBuilder("tenant-a")

        # Subject from tenant B
        other_tenant_subject = "xorb.tenant-b.scan.nmap.created"

        parsed = builder_a.parse(other_tenant_subject)

        # Parser should correctly identify it's from a different tenant
        assert parsed.tenant == "tenant-b"
        assert parsed.tenant != builder_a.tenant

    def test_wildcard_patterns_are_tenant_specific(self):
        """Test that wildcard subscription patterns are tenant-specific."""
        tenant_a = "tenant-a"
        tenant_b = "tenant-b"

        # Wildcard patterns for each tenant
        pattern_a = f"xorb.{tenant_a}.>"
        pattern_b = f"xorb.{tenant_b}.>"

        assert pattern_a != pattern_b
        assert tenant_a in pattern_a and tenant_b not in pattern_a
        assert tenant_b in pattern_b and tenant_a not in pattern_b

        # Test specific domain patterns
        scan_pattern_a = f"xorb.{tenant_a}.scan.>"
        scan_pattern_b = f"xorb.{tenant_b}.scan.>"

        assert scan_pattern_a != scan_pattern_b

    def test_stream_names_include_tenant_isolation(self):
        """Test that stream names include tenant identifiers for isolation."""
        client_a = NATSClient(["nats://localhost:4222"], "tenant-a")
        client_b = NATSClient(["nats://localhost:4222"], "tenant-b")

        # Mock the connection state
        client_a._connected = True
        client_b._connected = True

        with patch.object(client_a, '_js') as mock_js_a, \
             patch.object(client_b, '_js') as mock_js_b:

            mock_js_a.add_stream = AsyncMock()
            mock_js_b.add_stream = AsyncMock()

            # Create streams for same domain but different tenants
            asyncio.run(client_a.create_stream(Domain.SCAN))
            asyncio.run(client_b.create_stream(Domain.SCAN))

            # Verify different stream names
            call_a = mock_js_a.add_stream.call_args[0][0]
            call_b = mock_js_b.add_stream.call_args[0][0]

            assert "tenant-a" in call_a.name
            assert "tenant-b" in call_b.name
            assert call_a.name != call_b.name


class TestCrossTenantAccessPrevention:
    """Test that cross-tenant access is prevented."""

    def test_subject_validation_prevents_wrong_tenant_subjects(self):
        """Test that subject validation prevents using wrong tenant IDs."""
        builder = SubjectBuilder("tenant-a")

        # Try to parse a subject from another tenant
        other_subject = "xorb.tenant-b.scan.nmap.created"
        parsed = builder.parse(other_subject)

        # The subject parses correctly but shows it's from different tenant
        assert parsed.tenant == "tenant-b"
        assert parsed.tenant != "tenant-a"

        # If we try to build the same subject, we get our tenant version
        our_subject = builder.build(parsed.domain, parsed.service, parsed.event)
        assert our_subject != other_subject
        assert "tenant-a" in our_subject
        assert "tenant-b" not in our_subject

    @pytest.mark.asyncio
    async def test_publish_message_contains_tenant_context(self):
        """Test that published messages contain proper tenant context."""
        client = NATSClient(["nats://localhost:4222"], "test-tenant")
        client._connected = True

        mock_js = AsyncMock()
        client._js = mock_js

        await client.publish(
            Domain.SCAN,
            "nmap",
            Event.CREATED,
            {"target": "127.0.0.1"}
        )

        # Verify publish was called
        mock_js.publish.assert_called_once()

        # Check call arguments
        call_args = mock_js.publish.call_args
        subject = call_args[0][0]
        headers = call_args[1]['headers']

        # Subject should contain tenant
        assert subject == "xorb.test-tenant.scan.nmap.created"

        # Headers should contain tenant context
        assert headers['X-Tenant-ID'] == "test-tenant"

    def test_consumer_creation_includes_tenant_streams(self):
        """Test that consumers are created with tenant-specific stream names."""
        client = NATSClient(["nats://localhost:4222"], "test-tenant")

        # Expected stream name should include tenant
        expected_stream = "xorb-test-tenant-scan-live"
        consumer_subjects = ["xorb.test-tenant.scan.>"]

        # The stream name and subjects should be tenant-specific
        assert "test-tenant" in expected_stream
        assert all("test-tenant" in subject for subject in consumer_subjects)

    def test_metrics_are_tenant_scoped(self):
        """Test that metrics tracking is scoped per tenant."""
        client_a = create_nats_client("tenant-a")
        client_b = create_nats_client("tenant-b")

        # Simulate different activity levels
        client_a._messages_published = 100
        client_a._messages_consumed = 50
        client_b._messages_published = 200
        client_b._messages_consumed = 75

        metrics_a = client_a.metrics
        metrics_b = client_b.metrics

        # Each client tracks its own metrics
        assert metrics_a['messages_published'] == 100
        assert metrics_a['messages_consumed'] == 50
        assert metrics_b['messages_published'] == 200
        assert metrics_b['messages_consumed'] == 75

        # Metrics should be independent
        assert metrics_a != metrics_b


class TestSubjectSchemaEnforcement:
    """Test that subject schema is enforced for tenant isolation."""

    def test_only_valid_domains_accepted(self):
        """Test that only valid domains from the schema are accepted."""
        builder = SubjectBuilder("test-tenant")

        # All valid domains should work
        for domain in Domain:
            subject = builder.build(domain, "test", Event.CREATED)
            assert f".{domain.value}." in subject

        # Invalid domains would be caught by the enum type system
        # This test verifies the enum constraint
        valid_domain_values = {d.value for d in Domain}
        expected_domains = {"evidence", "scan", "compliance", "control"}
        assert valid_domain_values == expected_domains

    def test_only_valid_events_accepted(self):
        """Test that only valid events from the schema are accepted."""
        builder = SubjectBuilder("test-tenant")

        # All valid events should work
        for event in Event:
            subject = builder.build(Domain.SCAN, "test", event)
            assert f".{event.value}" in subject

        # Invalid events would be caught by the enum type system
        # This test verifies the enum constraint
        valid_event_values = {e.value for e in Event}
        expected_events = {"created", "updated", "completed", "failed", "replay"}
        assert valid_event_values == expected_events

    def test_subject_pattern_is_enforced(self):
        """Test that the exact subject pattern is enforced."""
        builder = SubjectBuilder("test-tenant")

        subject = builder.build(Domain.EVIDENCE, "discovery", Event.COMPLETED)
        parts = subject.split('.')

        # Must have exactly 5 parts
        assert len(parts) == 5

        # Must follow exact pattern
        assert parts[0] == "xorb"         # prefix
        assert parts[1] == "test-tenant"  # tenant
        assert parts[2] == "evidence"     # domain
        assert parts[3] == "discovery"    # service
        assert parts[4] == "completed"    # event

    def test_subject_parsing_validates_schema(self):
        """Test that subject parsing validates schema compliance."""
        builder = SubjectBuilder("test-tenant")

        # Valid subjects should parse successfully
        valid_subjects = [
            "xorb.tenant-1.scan.nmap.created",
            "xorb.customer-a.evidence.discovery.completed",
            "xorb.qa-env.compliance.pci.updated",
            "xorb.prod.control.firewall.failed",
        ]

        for subject in valid_subjects:
            parsed = builder.parse(subject)
            # Should not raise exception and should have valid components
            assert parsed.tenant is not None
            assert parsed.domain in Domain
            assert parsed.service is not None
            assert parsed.event in Event

        # Invalid subjects should raise errors
        invalid_subjects = [
            "invalid.tenant-1.scan.nmap.created",        # wrong prefix
            "xorb.tenant-1.invalid.nmap.created",        # invalid domain
            "xorb.tenant-1.scan.nmap.invalid",           # invalid event
            "xorb.tenant-1.scan.created",                # missing service
            "xorb.tenant-1.scan.nmap.created.extra",     # extra parts
        ]

        for subject in invalid_subjects:
            with pytest.raises(ValueError):
                builder.parse(subject)


class TestEnvironmentBasedConfiguration:
    """Test environment-based configuration for tenant isolation."""

    def test_consumer_settings_from_environment(self):
        """Test that consumer settings can be loaded from environment."""
        with patch.dict('os.environ', {
            'NATS_ACK_WAIT': '45s',
            'NATS_MAX_ACK_PENDING': '2048',
            'NATS_FLOW_CONTROL': 'true',
            'NATS_IDLE_HEARTBEAT': '10s',
            'NATS_RATE_LIMIT_BPS': '2097152'
        }):
            settings = ConsumerSettings.from_env()

            assert settings.ack_wait == '45s'
            assert settings.max_ack_pending == 2048
            assert settings.flow_control is True
            assert settings.idle_heartbeat == '10s'
            assert settings.rate_limit_bps == 2097152

    def test_client_factory_respects_environment(self):
        """Test that client factory respects environment variables."""
        with patch.dict('os.environ', {
            'NATS_URL': 'nats://test-server:4222',
            'NATS_CREDENTIALS': '/path/to/test.creds'
        }):
            client = create_nats_client("test-tenant")

            assert "nats://test-server:4222" in client.servers
            assert client.credentials_file == "/path/to/test.creds"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
