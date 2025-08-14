#!/usr/bin/env python3
"""
NATS Subject Linter for Xorb Backplane.

Validates NATS subject strings against the immutable v1 schema:
xorb.<tenant>.<domain>.<service>.<event>

Where:
- domain ∈ {evidence, scan, compliance, control}
- event  ∈ {created, updated, completed, failed, replay}
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional


# Schema constants
VALID_DOMAINS = {"evidence", "scan", "compliance", "control"}
VALID_EVENTS = {"created", "updated", "completed", "failed", "replay"}
SUBJECT_PATTERN = re.compile(r"^xorb\.([^.]+)\.([^.]+)\.([^.]+)\.([^.]+)$")


def validate_subject(subject: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a single NATS subject against the schema.

    Args:
        subject: The subject string to validate

    Returns:
        Tuple of (is_valid, error_message_or_None)
    """
    if not subject:
        return False, "Subject is empty"

    match = SUBJECT_PATTERN.match(subject)
    if not match:
        return False, f"Subject '{subject}' does not match pattern xorb.<tenant>.<domain>.<service>.<event>"

    _, domain, _, event = match.groups()

    if domain not in VALID_DOMAINS:
        return False, f"Invalid domain '{domain}' in subject '{subject}'. Valid domains: {', '.join(VALID_DOMAINS)}"

    if event not in VALID_EVENTS:
        return False, f"Invalid event '{event}' in subject '{subject}'. Valid events: {', '.join(VALID_EVENTS)}"

    return True, None


def find_subjects_in_file(file_path: Path) -> List[Tuple[str, int, str]]:
    """
    Find all NATS subjects in a file.

    Args:
        file_path: Path to the file to scan

    Returns:
        List of tuples (subject, line_number, line_content)
    """
    subjects = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                # Find all potential subjects in the line
                # This is a simple approach - in a real implementation you might want more sophisticated parsing
                for match in re.finditer(r'xorb\.[^.]+\.[^.]+\.[^.]+\.[^.]+', line):
                    subject = match.group()
                    subjects.append((subject, line_num, line.strip()))
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)

    return subjects


def scan_files(file_paths: List[Path], allowlist_pattern: Optional[str] = None) -> List[Tuple[str, int, str, str]]:
    """
    Scan files for NATS subjects.

    Args:
        file_paths: List of file paths to scan
        allowlist_pattern: Optional regex pattern to allowlist subjects

    Returns:
        List of tuples (subject, line_number, file_path, error_message)
    """
    violations = []
    allowlist_regex = re.compile(allowlist_pattern) if allowlist_pattern else None

    for file_path in file_paths:
        if file_path.is_file():
            subjects = find_subjects_in_file(file_path)
            for subject, line_num, line_content in subjects:
                # Check if subject is in allowlist
                if allowlist_regex and allowlist_regex.search(subject):
                    continue

                is_valid, error = validate_subject(subject)
                if not is_valid:
                    violations.append((subject, line_num, str(file_path), error))
        else:
            print(f"Warning: {file_path} is not a file", file=sys.stderr)

    return violations


def format_violations_table(violations: List[Tuple[str, int, str, str]]) -> str:
    """
    Format violations as a table.

    Args:
        violations: List of violations

    Returns:
        Formatted table string
    """
    if not violations:
        return ""

    # Calculate column widths
    subject_width = max(len("Subject"), max(len(v[0]) for v in violations))
    line_width = max(len("Line"), max(len(str(v[1])) for v in violations))
    file_width = max(len("File"), max(len(v[2]) for v in violations))
    error_width = max(len("Error"), max(len(v[3]) for v in violations))

    # Create format string
    format_str = f"{{:<{subject_width}}} | {{:<{line_width}}} | {{:<{file_width}}} | {{:<{error_width}}}"

    # Create table
    lines = [format_str.format("Subject", "Line", "File", "Error")]
    lines.append("-" * len(lines[0]))

    for subject, line_num, file_path, error in violations:
        lines.append(format_str.format(subject, line_num, file_path, error))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Lint NATS subjects in files against Xorb schema")
    parser.add_argument(
        "--paths",
        nargs="+",
        help="Paths or files to scan for subject strings"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on first violation"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    parser.add_argument(
        "--allowlist",
        help="Regex pattern to allowlist subjects"
    )

    args = parser.parse_args()

    # Determine input source
    if args.paths:
        # Expand paths
        file_paths = []
        for path_str in args.paths:
            path = Path(path_str)
            if path.is_file():
                file_paths.append(path)
            elif path.is_dir():
                # Recursively find files
                for ext in ["*.py", "*.go", "*.ts", "*.md"]:
                    file_paths.extend(path.rglob(ext))
            else:
                print(f"Warning: {path} is not a file or directory", file=sys.stderr)
    else:
        # Read from stdin
        subjects = []
        for line in sys.stdin:
            subject = line.strip()
            if subject:
                subjects.append((subject, 0, "stdin"))

        violations = []
        for subject, _, _ in subjects:
            is_valid, error = validate_subject(subject)
            if not is_valid:
                violations.append((subject, 0, "stdin", error))
                if args.strict:
                    break

        if args.json:
            result = {
                "valid": len(violations) == 0,
                "violations": [
                    {
                        "subject": v[0],
                        "line": v[1],
                        "file": v[2],
                        "error": v[3]
                    }
                    for v in violations
                ]
            }
            print(json.dumps(result, indent=2))
        else:
            if violations:
                print(format_violations_table(violations))

        sys.exit(1 if violations else 0)

    # Scan files
    violations = scan_files(file_paths, args.allowlist)

    if args.strict and violations:
        # Only report first violation in strict mode
        violations = violations[:1]

    if args.json:
        result = {
            "valid": len(violations) == 0,
            "violations": [
                {
                    "subject": v[0],
                    "line": v[1],
                    "file": v[2],
                    "error": v[3]
                }
                for v in violations
            ]
        }
        print(json.dumps(result, indent=2))
    else:
        if violations:
            print(format_violations_table(violations))

    sys.exit(1 if violations else 0)


if __name__ == "__main__":
    main()
