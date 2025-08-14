#!/usr/bin/env python3
"""
Unit tests for NATS Subject Builder and Schema Validation - Phase G2

Tests the v1 immutable subject schema validation and tenant isolation.
"""

import pytest
from xorb_platform_bus.bus.pubsub.nats_client import (
    SubjectBuilder,
    SubjectComponents,
    Domain,
    Event
)


class TestSubjectComponents:
    """Test subject component validation."""

    def test_valid_components(self):
        """Test valid subject components."""
        components = SubjectComponents(
            tenant="t-qa",
            domain=Domain.SCAN,
            service="nmap",
            event=Event.CREATED
        )
        assert components.validate() is True
        assert components.to_subject() == "xorb.t-qa.scan.nmap.created"

    def test_tenant_validation(self):
        """Test tenant name validation rules."""
        # Valid tenant names
        valid_tenants = [
            "abc",           # minimum length
            "tenant-1",      # with hyphens
            "a" * 63,        # maximum length
            "test123",       # alphanumeric
            "t-qa-env",      # multiple hyphens
        ]

        for tenant in valid_tenants:
            components = SubjectComponents(
                tenant=tenant,
                domain=Domain.SCAN,
                service="test",
                event=Event.CREATED
            )
            assert components.validate() is True, f"Tenant '{tenant}' should be valid"

    def test_invalid_tenants(self):
        """Test invalid tenant names."""
        invalid_tenants = [
            "ab",            # too short
            "a" * 64,        # too long
            "-test",         # starts with hyphen
            "test-",         # ends with hyphen
            "test.invalid",  # contains dot
            "test@invalid",  # contains special char
            "",              # empty
            "test_with_underscores",  # underscores not allowed in current impl
        ]

        for tenant in invalid_tenants:
            components = SubjectComponents(
                tenant=tenant,
                domain=Domain.SCAN,
                service="test",
                event=Event.CREATED
            )
            assert components.validate() is False, f"Tenant '{tenant}' should be invalid"

    def test_service_validation(self):
        """Test service name validation rules."""
        # Valid service names
        valid_services = [
            "n",             # minimum length
            "nmap",          # standard name
            "ssl-scan",      # with hyphens
            "a" * 32,        # maximum length
            "test123",       # alphanumeric
        ]

        for service in valid_services:
            components = SubjectComponents(
                tenant="test-tenant",
                domain=Domain.SCAN,
                service=service,
                event=Event.CREATED
            )
            assert components.validate() is True, f"Service '{service}' should be valid"

    def test_invalid_services(self):
        """Test invalid service names."""
        invalid_services = [
            "",              # empty
            "a" * 33,        # too long
            "-test",         # starts with hyphen
            "test-",         # ends with hyphen
            "test.invalid",  # contains dot
            "test@invalid",  # contains special char
        ]

        for service in invalid_services:
            components = SubjectComponents(
                tenant="test-tenant",
                domain=Domain.SCAN,
                service=service,
                event=Event.CREATED
            )
            assert components.validate() is False, f"Service '{service}' should be invalid"

    def test_domain_enumeration(self):
        """Test that only valid domains are accepted."""
        # All valid domains should work
        for domain in Domain:
            components = SubjectComponents(
                tenant="test-tenant",
                domain=domain,
                service="test",
                event=Event.CREATED
            )
            assert components.validate() is True
            subject = components.to_subject()
            assert domain.value in subject

    def test_event_enumeration(self):
        """Test that only valid events are accepted."""
        # All valid events should work
        for event in Event:
            components = SubjectComponents(
                tenant="test-tenant",
                domain=Domain.SCAN,
                service="test",
                event=event
            )
            assert components.validate() is True
            subject = components.to_subject()
            assert event.value in subject

    def test_subject_generation(self):
        """Test subject string generation."""
        components = SubjectComponents(
            tenant="tenant-1",
            domain=Domain.EVIDENCE,
            service="discovery",
            event=Event.COMPLETED
        )

        subject = components.to_subject()
        assert subject == "xorb.tenant-1.evidence.discovery.completed"

    def test_invalid_components_raise_error(self):
        """Test that invalid components raise ValueError on to_subject()."""
        components = SubjectComponents(
            tenant="x",  # too short
            domain=Domain.SCAN,
            service="test",
            event=Event.CREATED
        )

        with pytest.raises(ValueError, match="Invalid subject components"):
            components.to_subject()


class TestSubjectBuilder:
    """Test the subject builder functionality."""

    def test_builder_initialization(self):
        """Test builder initialization with tenant."""
        builder = SubjectBuilder("test-tenant")
        assert builder.tenant == "test-tenant"

    def test_build_valid_subject(self):
        """Test building valid subjects."""
        builder = SubjectBuilder("t-qa")

        subject = builder.build(Domain.SCAN, "nmap", Event.CREATED)
        assert subject == "xorb.t-qa.scan.nmap.created"

        subject = builder.build(Domain.EVIDENCE, "discovery", Event.COMPLETED)
        assert subject == "xorb.t-qa.evidence.discovery.completed"

    def test_build_invalid_subject_raises_error(self):
        """Test that invalid components raise errors."""
        builder = SubjectBuilder("t")  # Invalid tenant (too short)

        with pytest.raises(ValueError):
            builder.build(Domain.SCAN, "test", Event.CREATED)

    def test_parse_valid_subject(self):
        """Test parsing valid subjects."""
        builder = SubjectBuilder("t-qa")

        subject = "xorb.t-qa.scan.nmap.created"
        components = builder.parse(subject)

        assert components.tenant == "t-qa"
        assert components.domain == Domain.SCAN
        assert components.service == "nmap"
        assert components.event == Event.CREATED

    def test_parse_invalid_subject_format(self):
        """Test parsing invalid subject formats."""
        builder = SubjectBuilder("t-qa")

        invalid_subjects = [
            "invalid",                           # wrong format
            "nats.t-qa.scan.nmap.created",      # wrong prefix
            "xorb.t-qa.scan.created",           # missing service
            "xorb.t-qa.scan.nmap.created.extra", # too many parts
            "",                                  # empty
        ]

        for subject in invalid_subjects:
            with pytest.raises(ValueError, match="Invalid subject format"):
                builder.parse(subject)

    def test_parse_invalid_domain(self):
        """Test parsing subjects with invalid domains."""
        builder = SubjectBuilder("t-qa")

        subject = "xorb.t-qa.invalid-domain.nmap.created"
        with pytest.raises(ValueError, match="Invalid subject components"):
            builder.parse(subject)

    def test_parse_invalid_event(self):
        """Test parsing subjects with invalid events."""
        builder = SubjectBuilder("t-qa")

        subject = "xorb.t-qa.scan.nmap.invalid-event"
        with pytest.raises(ValueError, match="Invalid subject components"):
            builder.parse(subject)

    def test_round_trip_consistency(self):
        """Test that build -> parse -> build is consistent."""
        builder = SubjectBuilder("tenant-test")

        # Build subject
        original_subject = builder.build(Domain.COMPLIANCE, "pci-dss", Event.UPDATED)

        # Parse it back
        components = builder.parse(original_subject)

        # Build again from parsed components
        rebuilt_subject = components.to_subject()

        assert original_subject == rebuilt_subject

    def test_tenant_isolation_in_subjects(self):
        """Test that different tenants generate different subjects."""
        builder_a = SubjectBuilder("tenant-a")
        builder_b = SubjectBuilder("tenant-b")

        subject_a = builder_a.build(Domain.SCAN, "nmap", Event.CREATED)
        subject_b = builder_b.build(Domain.SCAN, "nmap", Event.CREATED)

        assert subject_a != subject_b
        assert "tenant-a" in subject_a
        assert "tenant-b" in subject_b
        assert subject_a == "xorb.tenant-a.scan.nmap.created"
        assert subject_b == "xorb.tenant-b.scan.nmap.created"


class TestSchemaImmutability:
    """Test that the v1 schema is immutable and complete."""

    def test_domain_values_immutable(self):
        """Test that Domain enum has expected values (v1 schema)."""
        expected_domains = {"evidence", "scan", "compliance", "control"}
        actual_domains = {domain.value for domain in Domain}

        assert actual_domains == expected_domains, "Domain enum has changed - schema is supposed to be immutable!"

    def test_event_values_immutable(self):
        """Test that Event enum has expected values (v1 schema)."""
        expected_events = {"created", "updated", "completed", "failed", "replay"}
        actual_events = {event.value for event in Event}

        assert actual_events == expected_events, "Event enum has changed - schema is supposed to be immutable!"

    def test_subject_pattern_immutable(self):
        """Test that subject pattern is exactly as specified."""
        builder = SubjectBuilder("test")
        subject = builder.build(Domain.SCAN, "test", Event.CREATED)

        # Must follow exact pattern: xorb.<tenant>.<domain>.<service>.<event>
        parts = subject.split('.')
        assert len(parts) == 5
        assert parts[0] == "xorb"
        assert parts[1] == "test"  # tenant
        assert parts[2] == "scan"  # domain
        assert parts[3] == "test"  # service
        assert parts[4] == "created"  # event


class TestTenantIsolationScenarios:
    """Test various tenant isolation scenarios."""

    def test_different_tenants_different_subjects(self):
        """Test that different tenants have completely isolated subjects."""
        scenarios = [
            ("tenant-a", "tenant-b"),
            ("qa-env", "prod-env"),
            ("customer-1", "customer-2"),
            ("dev-test", "staging-test"),
        ]

        for tenant_a, tenant_b in scenarios:
            builder_a = SubjectBuilder(tenant_a)
            builder_b = SubjectBuilder(tenant_b)

            # Same operation, different tenants
            subject_a = builder_a.build(Domain.SCAN, "nmap", Event.CREATED)
            subject_b = builder_b.build(Domain.SCAN, "nmap", Event.CREATED)

            # Must be different
            assert subject_a != subject_b

            # Each should contain their own tenant
            assert tenant_a in subject_a
            assert tenant_b in subject_b

            # Neither should contain the other tenant
            assert tenant_a not in subject_b
            assert tenant_b not in subject_a

    def test_same_tenant_same_subjects(self):
        """Test that the same tenant generates consistent subjects."""
        tenant_id = "consistent-tenant"

        builder_1 = SubjectBuilder(tenant_id)
        builder_2 = SubjectBuilder(tenant_id)

        subject_1 = builder_1.build(Domain.EVIDENCE, "discovery", Event.COMPLETED)
        subject_2 = builder_2.build(Domain.EVIDENCE, "discovery", Event.COMPLETED)

        # Same tenant, same parameters = same subject
        assert subject_1 == subject_2
        assert subject_1 == "xorb.consistent-tenant.evidence.discovery.completed"

    def test_wildcard_patterns_for_tenant_filtering(self):
        """Test that wildcard patterns can be constructed for tenant filtering."""
        tenant_id = "test-tenant"
        builder = SubjectBuilder(tenant_id)

        # Tenant can subscribe to all their domains
        tenant_wildcard = f"xorb.{tenant_id}.>"
        assert tenant_wildcard == "xorb.test-tenant.>"

        # Tenant can subscribe to specific domain
        domain_wildcard = f"xorb.{tenant_id}.{Domain.SCAN.value}.>"
        assert domain_wildcard == "xorb.test-tenant.scan.>"

        # These patterns should match subjects from the same tenant
        test_subject = builder.build(Domain.SCAN, "nmap", Event.CREATED)
        assert test_subject.startswith("xorb.test-tenant.")

    @pytest.mark.parametrize("tenant_id,domain,service,event", [
        ("tenant-1", Domain.SCAN, "nmap", Event.CREATED),
        ("tenant-2", Domain.EVIDENCE, "discovery", Event.COMPLETED),
        ("qa-env", Domain.COMPLIANCE, "sox", Event.UPDATED),
        ("prod-env", Domain.CONTROL, "firewall", Event.FAILED),
        ("customer-a", Domain.EVIDENCE, "forensics", Event.REPLAY),
    ])
    def test_parameterized_subject_generation(self, tenant_id, domain, service, event):
        """Test subject generation with various parameter combinations."""
        builder = SubjectBuilder(tenant_id)
        subject = builder.build(domain, service, event)

        # Verify format
        expected = f"xorb.{tenant_id}.{domain.value}.{service}.{event.value}"
        assert subject == expected

        # Verify parsing round-trip
        parsed = builder.parse(subject)
        assert parsed.tenant == tenant_id
        assert parsed.domain == domain
        assert parsed.service == service
        assert parsed.event == event


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
