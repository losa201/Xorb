#!/usr/bin/env python3
"""
Unit tests for Subject Linter - Phase G2

Tests the subject linter tool for NATS schema validation.
"""

import json
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open

# Import the subject linter functions directly
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "tools" / "backplane"))

from subject_lint import (
    validate_subject,
    find_subjects_in_file,
    scan_files,
    format_violations_table,
    get_schema_info,
    VALID_DOMAINS,
    VALID_EVENTS
)


class TestSubjectValidation:
    """Test individual subject validation."""

    def test_valid_subjects(self):
        """Test validation of valid subjects."""
        valid_subjects = [
            "xorb.t-qa.scan.nmap.created",
            "xorb.tenant-1.evidence.discovery.completed",
            "xorb.customer-a.compliance.pci-dss.updated",
            "xorb.prod-env.control.firewall.failed",
            "xorb.demo-tenant.scan.nuclei.replay",
            "xorb.abc.evidence.forensics.created",  # minimum tenant length
            "xorb." + "a" * 63 + ".scan.test.created",  # maximum tenant length
        ]

        for subject in valid_subjects:
            is_valid, error = validate_subject(subject)
            assert is_valid, f"Subject '{subject}' should be valid, got error: {error}"

    def test_invalid_subjects(self):
        """Test validation of invalid subjects."""
        invalid_subjects = [
            # Format violations
            ("", "Subject is empty"),
            ("invalid", "does not match pattern"),
            ("nats.t-qa.scan.nmap.created", "does not match pattern"),
            ("xorb.t-qa.scan.created", "does not match pattern"),  # missing service
            ("xorb.t-qa.scan.nmap.created.extra", "does not match pattern"),  # extra part

            # Tenant violations
            ("xorb.ab.scan.nmap.created", "Must be 3-63 characters"),  # too short
            ("xorb." + "a" * 64 + ".scan.nmap.created", "Must be 3-63 characters"),  # too long
            ("xorb.-invalid.scan.nmap.created", "no dots/hyphens at start/end"),  # starts with hyphen
            ("xorb.invalid-.scan.nmap.created", "no dots/hyphens at start/end"),  # ends with hyphen
            ("xorb.invalid@tenant.scan.nmap.created", "Must be alphanumeric"),  # special char

            # Domain violations
            ("xorb.t-qa.invalid-domain.nmap.created", "Invalid domain"),
            ("xorb.t-qa.SCAN.nmap.created", "Invalid domain"),  # case sensitive

            # Service violations
            ("xorb.t-qa.scan..created", "Must be 1-32 characters"),  # empty service
            ("xorb.t-qa.scan." + "a" * 33 + ".created", "Must be 1-32 characters"),  # too long
            ("xorb.t-qa.scan.-invalid.created", "no dots/hyphens at start/end"),  # starts with hyphen
            ("xorb.t-qa.scan.invalid-.created", "no dots/hyphens at start/end"),  # ends with hyphen

            # Event violations
            ("xorb.t-qa.scan.nmap.invalid-event", "Invalid event"),
            ("xorb.t-qa.scan.nmap.CREATED", "Invalid event"),  # case sensitive
        ]

        for subject, expected_error_fragment in invalid_subjects:
            is_valid, error = validate_subject(subject)
            assert not is_valid, f"Subject '{subject}' should be invalid"
            assert expected_error_fragment in error, f"Error message should contain '{expected_error_fragment}', got: {error}"

    def test_schema_immutability(self):
        """Test that the schema constants are as expected (immutable)."""
        assert VALID_DOMAINS == {"evidence", "scan", "compliance", "control"}
        assert VALID_EVENTS == {"created", "updated", "completed", "failed", "replay"}

    def test_get_schema_info(self):
        """Test schema information function."""
        schema = get_schema_info()

        assert schema["version"] == "v1"
        assert schema["immutable"] is True
        assert schema["pattern"] == "xorb.<tenant>.<domain>.<service>.<event>"
        assert set(schema["domains"]) == VALID_DOMAINS
        assert set(schema["events"]) == VALID_EVENTS
        assert "alphanumeric" in schema["tenant_rules"]
        assert "3-63 chars" in schema["tenant_rules"]


class TestFileScanning:
    """Test file scanning functionality."""

    def test_find_subjects_in_file(self):
        """Test finding subjects in a file."""
        file_content = '''
        # Example NATS subjects in code
        subject1 = "xorb.t-qa.scan.nmap.created"
        subject2 = "xorb.tenant-1.evidence.discovery.completed"
        not_a_subject = "some other string"
        invalid_subject = "xorb.invalid.scan.nmap.started"
        '''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(file_content)
            temp_path = Path(f.name)

        try:
            subjects = find_subjects_in_file(temp_path)

            # Should find both subjects
            assert len(subjects) == 3  # 2 valid + 1 invalid

            subject_strings = [s[0] for s in subjects]
            assert "xorb.t-qa.scan.nmap.created" in subject_strings
            assert "xorb.tenant-1.evidence.discovery.completed" in subject_strings
            assert "xorb.invalid.scan.nmap.started" in subject_strings

        finally:
            temp_path.unlink()

    def test_scan_files_with_violations(self):
        """Test scanning files for violations."""
        # Create test files
        valid_file_content = '''
        # Valid subjects
        subject = "xorb.t-qa.scan.nmap.created"
        '''

        invalid_file_content = '''
        # Invalid subjects
        subject = "xorb.invalid.scan.nmap.started"  # invalid domain and event
        '''

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create valid file
            valid_file = temp_path / "valid.py"
            valid_file.write_text(valid_file_content)

            # Create invalid file
            invalid_file = temp_path / "invalid.py"
            invalid_file.write_text(invalid_file_content)

            # Scan files
            violations = scan_files([valid_file, invalid_file])

            # Should find violations only in invalid file
            assert len(violations) > 0

            violation_files = [v[2] for v in violations]
            assert str(invalid_file) in violation_files

    def test_scan_files_with_allowlist(self):
        """Test scanning files with allowlist pattern."""
        file_content = '''
        # Mixed subjects
        real_subject = "xorb.t-qa.scan.nmap.created"
        test_subject = "test.example.scan.nmap.created"  # should be ignored
        example_subject = "example.tenant.scan.nmap.created"  # should be ignored
        '''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(file_content)
            temp_path = Path(f.name)

        try:
            # Scan with allowlist
            violations = scan_files([temp_path], allowlist_pattern="test.*|example.*")

            # Should not find violations for allowlisted subjects
            violation_subjects = [v[0] for v in violations]
            assert "test.example.scan.nmap.created" not in violation_subjects
            assert "example.tenant.scan.nmap.created" not in violation_subjects

        finally:
            temp_path.unlink()

    def test_format_violations_table(self):
        """Test formatting violations as a table."""
        violations = [
            ("xorb.invalid.scan.nmap.started", 10, "/path/to/file.py", "Invalid domain 'invalid'"),
            ("xorb.t-qa.scan.nmap.invalid", 20, "/path/to/other.py", "Invalid event 'invalid'"),
        ]

        table = format_violations_table(violations)

        assert "Subject" in table
        assert "Line" in table
        assert "File" in table
        assert "Error" in table
        assert "xorb.invalid.scan.nmap.started" in table
        assert "Invalid domain" in table
        assert "/path/to/file.py" in table
        assert "10" in table


class TestLinterIntegration:
    """Test linter integration scenarios."""

    def test_linter_with_real_subject_patterns(self):
        """Test linter with realistic subject patterns."""
        test_subjects = [
            # Valid production-like subjects
            "xorb.customer-1.scan.nmap.created",
            "xorb.tenant-prod.evidence.vulnerability-scan.completed",
            "xorb.qa-env.compliance.pci-dss.updated",
            "xorb.staging.control.access-control.failed",

            # Common mistake patterns
            "xorb.test.scan.nmap.started",  # invalid event
            "xorb.t.evidence.discovery.created",  # tenant too short
            "xorb.customer-1.scanning.nmap.created",  # invalid domain
        ]

        for subject in test_subjects:
            is_valid, error = validate_subject(subject)

            if "started" in subject or subject == "xorb.t.evidence.discovery.created" or "scanning" in subject:
                assert not is_valid, f"Subject '{subject}' should be invalid"
            else:
                assert is_valid, f"Subject '{subject}' should be valid, got error: {error}"

    def test_linter_handles_edge_cases(self):
        """Test linter handles edge cases properly."""
        edge_cases = [
            # Empty and whitespace
            ("", False),
            ("   ", False),
            ("\n", False),

            # Unicode and special characters
            ("xorb.tenant-ñ.scan.nmap.created", False),  # non-ASCII
            ("xorb.tenant@1.scan.nmap.created", False),  # special char

            # Case sensitivity
            ("XORB.tenant-1.scan.nmap.created", False),
            ("xorb.TENANT-1.scan.nmap.created", False),
            ("xorb.tenant-1.SCAN.nmap.created", False),
            ("xorb.tenant-1.scan.nmap.CREATED", False),

            # Boundary conditions
            ("xorb.abc.scan.n.created", True),  # minimum service length
            ("xorb.abc.scan." + "a" * 32 + ".created", True),  # maximum service length
        ]

        for subject, should_be_valid in edge_cases:
            is_valid, _ = validate_subject(subject)
            if should_be_valid:
                assert is_valid, f"Subject '{subject}' should be valid"
            else:
                assert not is_valid, f"Subject '{subject}' should be invalid"

    def test_linter_performance_with_many_subjects(self):
        """Test linter performance with many subjects."""
        # Generate many subjects for performance testing
        subjects = []
        for i in range(1000):
            if i % 2 == 0:
                # Valid subjects
                subjects.append(f"xorb.tenant-{i}.scan.tool-{i}.created")
            else:
                # Invalid subjects
                subjects.append(f"xorb.t{i}.invalid-domain.tool-{i}.invalid-event")

        valid_count = 0
        invalid_count = 0

        for subject in subjects:
            is_valid, _ = validate_subject(subject)
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1

        # Should be roughly half valid, half invalid
        assert valid_count == 500
        assert invalid_count == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
