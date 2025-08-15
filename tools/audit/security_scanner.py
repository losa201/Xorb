#!/usr/bin/env python3
"""Security and ADR compliance scanner for XORB monorepo audit."""

import json
import re
import subprocess
from pathlib import Path


def check_adr_002_compliance(root_path):
    """Check ADR-002 compliance: No Redis pub/sub bus usage."""
    root = Path(root_path)
    violations = []
    compliant_usage = []

    for py_file in root.rglob("*.py"):
        try:
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Redis pub/sub patterns (violations)
            pubsub_patterns = [
                r"redis.*\.publish\s*\(",
                r"redis.*\.subscribe\s*\(",
                r"redis.*\.psubscribe\s*\(",
                r"\.pubsub\s*\(",
                r"pubsub.*channel",
                r"message.*bus.*redis",
            ]

            # Compliant Redis patterns (cache usage)
            cache_patterns = [
                r"redis.*\.get\s*\(",
                r"redis.*\.set\s*\(",
                r"redis.*\.delete\s*\(",
                r"redis.*\.expire\s*\(",
                r"cache.*redis",
            ]

            for pattern in pubsub_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    violations.append(
                        {
                            "file": str(py_file.relative_to(root)),
                            "pattern": pattern,
                            "line": content[: match.start()].count("\n") + 1,
                            "context": content[
                                max(0, match.start() - 50) : match.end() + 50
                            ].strip(),
                            "severity": "HIGH"
                            if "publish" in pattern or "subscribe" in pattern
                            else "MEDIUM",
                        }
                    )

            for pattern in cache_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    compliant_usage.append(
                        {
                            "file": str(py_file.relative_to(root)),
                            "pattern": pattern,
                            "line": content[: match.start()].count("\n") + 1,
                        }
                    )

        except (IOError, OSError, UnicodeDecodeError):
            continue

    return {
        "violations": violations,
        "compliant_usage": compliant_usage,
        "compliance_status": "FAIL" if violations else "PASS",
    }


def check_adr_003_logging_compliance(root_path):
    """Check ADR-003 compliance: Proper logging and secret handling."""
    root = Path(root_path)
    violations = []

    for py_file in root.rglob("*.py"):
        try:
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Dangerous logging patterns
            dangerous_patterns = [
                r'log.*\..*["\'].*password.*["\']',
                r'log.*\..*["\'].*token.*["\']',
                r'log.*\..*["\'].*key.*["\']',
                r'log.*\..*["\'].*secret.*["\']',
                r"print\s*\(.*password",
                r"print\s*\(.*token",
                r"print\s*\(.*key",
                r"print\s*\(.*secret",
                r"log.*\..*\{.*password.*\}",
                r'f".*{.*password.*}.*"',
                r"logger.*debug.*password",
                r"logger.*info.*token",
            ]

            for pattern in dangerous_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    violations.append(
                        {
                            "file": str(py_file.relative_to(root)),
                            "pattern": pattern,
                            "line": content[: match.start()].count("\n") + 1,
                            "context": content[
                                max(0, match.start() - 50) : match.end() + 50
                            ].strip(),
                            "severity": "HIGH",
                        }
                    )

        except (IOError, OSError, UnicodeDecodeError):
            continue

    return {
        "violations": violations,
        "compliance_status": "FAIL" if violations else "PASS",
    }


def run_bandit_scan(root_path):
    """Run Bandit security scanner."""
    try:
        result = subprocess.run(
            ["bandit", "-r", root_path, "-f", "json", "--skip", "B101,B601"],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0 or result.returncode == 1:  # 1 = issues found
            return json.loads(result.stdout)
        else:
            return {"error": result.stderr, "results": []}
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
        FileNotFoundError,
    ):
        return {"error": "Bandit scan failed or not available", "results": []}


def run_ruff_scan(root_path):
    """Run Ruff linter for security and quality issues."""
    try:
        result = subprocess.run(
            ["ruff", "check", root_path, "--select", "S,E,F,W", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=300,
        )

        output_lines = [
            line.strip() for line in result.stdout.split("\n") if line.strip()
        ]
        findings = []
        for line in output_lines:
            try:
                findings.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        return findings
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return []


def scan_dockerfile_security(root_path):
    """Scan Dockerfiles for security issues."""
    root = Path(root_path)
    issues = []

    dockerfiles = list(root.rglob("Dockerfile*"))

    for dockerfile in dockerfiles:
        try:
            with open(dockerfile, "r", encoding="utf-8") as f:
                content = f.read()

            # Docker security patterns
            security_checks = [
                {
                    "pattern": r"FROM.*:latest",
                    "issue": "Using :latest tag - should pin specific versions",
                    "severity": "MEDIUM",
                },
                {
                    "pattern": r"RUN.*sudo",
                    "issue": "Using sudo in Docker - potential privilege escalation",
                    "severity": "HIGH",
                },
                {
                    "pattern": r"USER\s+root",
                    "issue": "Running as root user - security risk",
                    "severity": "HIGH",
                },
                {
                    "pattern": r"COPY\s+\.\s+/",
                    "issue": "Copying entire context - may include secrets",
                    "severity": "MEDIUM",
                },
                {
                    "pattern": r"ENV.*PASSWORD",
                    "issue": "Hardcoded password in environment - security risk",
                    "severity": "HIGH",
                },
                {
                    "pattern": r"ENV.*SECRET",
                    "issue": "Hardcoded secret in environment - security risk",
                    "severity": "HIGH",
                },
            ]

            for check in security_checks:
                matches = re.finditer(check["pattern"], content, re.IGNORECASE)
                for match in matches:
                    issues.append(
                        {
                            "file": str(dockerfile.relative_to(root)),
                            "line": content[: match.start()].count("\n") + 1,
                            "issue": check["issue"],
                            "severity": check["severity"],
                            "pattern": check["pattern"],
                            "context": content[
                                max(0, match.start() - 30) : match.end() + 30
                            ].strip(),
                        }
                    )

        except (IOError, OSError, UnicodeDecodeError):
            continue

    return issues


def check_supply_chain_security(root_path):
    """Check supply chain security indicators."""
    root = Path(root_path)
    findings = []

    # Check for requirement files
    req_files = (
        list(root.rglob("requirements*.txt"))
        + list(root.rglob("pyproject.toml"))
        + list(root.rglob("package.json"))
    )

    for req_file in req_files:
        try:
            with open(req_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Supply chain security patterns
            security_indicators = []

            # Check for pinned versions
            if req_file.suffix == ".txt":
                # Look for unpinned requirements
                unpinned_pattern = (
                    r"^([a-zA-Z0-9\-_]+)(?:\s*>=?\s*[\d.]+)?(?:\s*<\s*[\d.]+)?\s*$"
                )
                for line_num, line in enumerate(content.split("\n"), 1):
                    line = line.strip()
                    if (
                        line
                        and not line.startswith("#")
                        and "==" not in line
                        and "~=" not in line
                    ):
                        match = re.match(unpinned_pattern, line)
                        if match:
                            security_indicators.append(
                                {
                                    "file": str(req_file.relative_to(root)),
                                    "line": line_num,
                                    "issue": f"Unpinned dependency: {match.group(1)}",
                                    "severity": "MEDIUM",
                                    "type": "supply_chain",
                                }
                            )

            # Check for development packages in production
            dev_patterns = [r"pytest", r"debug", r"dev.*server", r"test.*framework"]

            for pattern in dev_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    security_indicators.append(
                        {
                            "file": str(req_file.relative_to(root)),
                            "line": content[: match.start()].count("\n") + 1,
                            "issue": f"Development dependency in production: {match.group()}",
                            "severity": "LOW",
                            "type": "supply_chain",
                        }
                    )

            findings.extend(security_indicators)

        except (IOError, OSError, UnicodeDecodeError):
            continue

    return findings


def scan_github_actions_security(root_path):
    """Scan GitHub Actions workflows for security issues."""
    root = Path(root_path)
    issues = []

    workflow_dir = root / ".github" / "workflows"
    if not workflow_dir.exists():
        return issues

    for workflow_file in workflow_dir.rglob("*.yml"):
        try:
            with open(workflow_file, "r", encoding="utf-8") as f:
                content = f.read()

            # GitHub Actions security patterns
            security_checks = [
                {
                    "pattern": r"uses:.*@master",
                    "issue": "Using @master branch - should pin to specific SHA or tag",
                    "severity": "MEDIUM",
                },
                {
                    "pattern": r"run:.*\$\{\{.*\}\}",
                    "issue": "Potential code injection via expression",
                    "severity": "HIGH",
                },
                {
                    "pattern": r"secrets\.\w+",
                    "issue": "Direct secret usage - ensure proper handling",
                    "severity": "LOW",
                },
                {
                    "pattern": r"pull_request_target",
                    "issue": "pull_request_target can be dangerous - review carefully",
                    "severity": "HIGH",
                },
            ]

            for check in security_checks:
                matches = re.finditer(check["pattern"], content, re.IGNORECASE)
                for match in matches:
                    issues.append(
                        {
                            "file": str(workflow_file.relative_to(root)),
                            "line": content[: match.start()].count("\n") + 1,
                            "issue": check["issue"],
                            "severity": check["severity"],
                            "context": content[
                                max(0, match.start() - 50) : match.end() + 50
                            ].strip(),
                        }
                    )

        except (IOError, OSError, UnicodeDecodeError):
            continue

    return issues


def find_hardcoded_secrets(root_path):
    """Find potential hardcoded secrets."""
    root = Path(root_path)
    secrets = []

    # Common secret patterns
    secret_patterns = [
        {
            "name": "API Key",
            "pattern": r'["\']?[Aa][Pp][Ii]_?[Kk][Ee][Yy]["\']?\s*[:=]\s*["\']([A-Za-z0-9]{20,})["\']',
            "severity": "HIGH",
        },
        {
            "name": "Password",
            "pattern": r'["\']?[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd]["\']?\s*[:=]\s*["\']([^"\']{8,})["\']',
            "severity": "HIGH",
        },
        {
            "name": "JWT Token",
            "pattern": r"eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*",
            "severity": "HIGH",
        },
        {
            "name": "Private Key",
            "pattern": r"-----BEGIN [A-Z ]+ PRIVATE KEY-----",
            "severity": "CRITICAL",
        },
        {
            "name": "AWS Access Key",
            "pattern": r"AKIA[0-9A-Z]{16}",
            "severity": "CRITICAL",
        },
    ]

    for file_path in root.rglob("*"):
        if file_path.is_file() and file_path.suffix in [
            ".py",
            ".js",
            ".ts",
            ".json",
            ".yml",
            ".yaml",
            ".env",
        ]:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                for secret_check in secret_patterns:
                    matches = re.finditer(secret_check["pattern"], content)
                    for match in matches:
                        # Skip obvious test/example values
                        matched_text = match.group()
                        if any(
                            skip in matched_text.lower()
                            for skip in ["test", "example", "dummy", "fake", "xxx"]
                        ):
                            continue

                        secrets.append(
                            {
                                "file": str(file_path.relative_to(root)),
                                "line": content[: match.start()].count("\n") + 1,
                                "type": secret_check["name"],
                                "severity": secret_check["severity"],
                                "preview": matched_text[:50] + "..."
                                if len(matched_text) > 50
                                else matched_text,
                            }
                        )

            except (IOError, OSError, UnicodeDecodeError):
                continue

    return secrets


if __name__ == "__main__":
    import sys

    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    print(f"Running security and compliance scan on: {root_dir}")

    # Run all security scans
    adr_002_results = check_adr_002_compliance(root_dir)
    adr_003_results = check_adr_003_logging_compliance(root_dir)
    bandit_results = run_bandit_scan(root_dir)
    ruff_results = run_ruff_scan(root_dir)
    dockerfile_issues = scan_dockerfile_security(root_dir)
    supply_chain_findings = check_supply_chain_security(root_dir)
    github_actions_issues = scan_github_actions_security(root_dir)
    hardcoded_secrets = find_hardcoded_secrets(root_dir)

    # Compile security findings
    security_data = {
        "adr_compliance": {"adr_002": adr_002_results, "adr_003": adr_003_results},
        "static_analysis": {"bandit": bandit_results, "ruff": ruff_results},
        "infrastructure_security": {
            "dockerfile_issues": dockerfile_issues,
            "github_actions": github_actions_issues,
        },
        "supply_chain": supply_chain_findings,
        "secrets_scan": hardcoded_secrets,
        "summary": {
            "adr_002_violations": len(adr_002_results.get("violations", [])),
            "adr_003_violations": len(adr_003_results.get("violations", [])),
            "bandit_issues": len(bandit_results.get("results", [])),
            "ruff_issues": len(ruff_results),
            "dockerfile_issues": len(dockerfile_issues),
            "supply_chain_issues": len(supply_chain_findings),
            "github_actions_issues": len(github_actions_issues),
            "potential_secrets": len(hardcoded_secrets),
        },
    }

    # Save security findings
    with open("docs/audit/catalog/security_findings.json", "w") as f:
        json.dump(security_data, f, indent=2)

    print("Security scan completed:")
    print(f"  ADR-002 violations: {security_data['summary']['adr_002_violations']}")
    print(f"  ADR-003 violations: {security_data['summary']['adr_003_violations']}")
    print(f"  Bandit issues: {security_data['summary']['bandit_issues']}")
    print(f"  Ruff issues: {security_data['summary']['ruff_issues']}")
    print(f"  Dockerfile issues: {security_data['summary']['dockerfile_issues']}")
    print(f"  Supply chain issues: {security_data['summary']['supply_chain_issues']}")
    print(
        f"  GitHub Actions issues: {security_data['summary']['github_actions_issues']}"
    )
    print(f"  Potential secrets: {security_data['summary']['potential_secrets']}")
