#!/usr/bin/env python3
"""Orchestration and messaging scanner for XORB monorepo audit."""

import json
import re
from pathlib import Path


def find_temporal_workflows(root_path):
    """Find Temporal workflow definitions and activities."""
    root = Path(root_path)
    workflows = []
    activities = []

    python_files = list(root.rglob("*.py"))

    for py_file in python_files:
        try:
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Temporal workflow patterns
            workflow_patterns = [
                r"@workflow\.defn\s*(?:\([^)]*\))?\s*class\s+(\w+)",
                r"@temporalio\.workflow\.run\s*def\s+(\w+)",
                r"class\s+(\w+).*Workflow",
                r'workflow\.execute_activity\s*\(\s*["\']?(\w+)["\']?',
                r"WorkflowMethod.*def\s+(\w+)",
            ]

            # Activity patterns
            activity_patterns = [
                r"@activity\.defn\s*(?:\([^)]*\))?\s*def\s+(\w+)",
                r"@temporalio\.activity\.activity\s*def\s+(\w+)",
                r"ActivityMethod.*def\s+(\w+)",
            ]

            for pattern in workflow_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    workflows.append(
                        {
                            "file": str(py_file.relative_to(root)),
                            "name": match.group(1),
                            "type": "workflow",
                            "line": content[: match.start()].count("\n") + 1,
                        }
                    )

            for pattern in activity_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    activities.append(
                        {
                            "file": str(py_file.relative_to(root)),
                            "name": match.group(1),
                            "type": "activity",
                            "line": content[: match.start()].count("\n") + 1,
                        }
                    )

        except (IOError, OSError, UnicodeDecodeError):
            continue

    return workflows, activities


def find_ptaas_domain_models(root_path):
    """Find PTaaS domain models and state machines."""
    root = Path(root_path)
    models = []
    state_machines = []

    for py_file in root.rglob("*.py"):
        try:
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Domain model patterns
            model_patterns = [
                r"class\s+(PTaaS\w+).*:",
                r"class\s+(Job\w+).*:",
                r"class\s+(\w+Job).*:",
                r"class\s+(\w*Status).*:",
                r"class\s+(\w*State).*:",
                r"class\s+(\w*Execution).*:",
            ]

            # State machine patterns
            state_patterns = [
                r"(QUEUED|RUNNING|COMPLETED|FAILED|PAUSED|CANCELLED)",
                r"JobStatus\.",
                r"\.transition\s*\(",
                r"state.*machine",
                r"finite.*state",
            ]

            for pattern in model_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    models.append(
                        {
                            "file": str(py_file.relative_to(root)),
                            "name": match.group(1),
                            "type": "domain_model",
                            "line": content[: match.start()].count("\n") + 1,
                        }
                    )

            for pattern in state_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    state_machines.append(
                        {
                            "file": str(py_file.relative_to(root)),
                            "pattern": pattern,
                            "match": match.group(),
                            "line": content[: match.start()].count("\n") + 1,
                        }
                    )

        except (IOError, OSError, UnicodeDecodeError):
            continue

    return models, state_machines


def find_nats_jetstream_usage(root_path):
    """Find NATS JetStream configuration and usage patterns."""
    root = Path(root_path)
    nats_usage = []

    for py_file in root.rglob("*.py"):
        try:
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # NATS JetStream patterns
            nats_patterns = [
                r"jetstream",
                r"nats.*connect",
                r'\.publish\s*\(\s*["\']([^"\']*xorb[^"\']*)["\']',
                r'\.subscribe\s*\(\s*["\']([^"\']*xorb[^"\']*)["\']',
                r"consumer.*config",
                r"stream.*config",
                r"durability.*name",
                r"at.*least.*once",
                r"ack.*wait",
            ]

            for pattern in nats_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    subject = (
                        match.group(1) if len(match.groups()) > 0 else match.group()
                    )
                    nats_usage.append(
                        {
                            "file": str(py_file.relative_to(root)),
                            "pattern": pattern,
                            "subject": subject,
                            "line": content[: match.start()].count("\n") + 1,
                            "context": content[
                                max(0, match.start() - 50) : match.end() + 50
                            ].strip(),
                        }
                    )

        except (IOError, OSError, UnicodeDecodeError):
            continue

    return nats_usage


def find_redis_usage(root_path):
    """Find Redis usage patterns to check ADR-002 compliance."""
    root = Path(root_path)
    redis_usage = []

    for py_file in root.rglob("*.py"):
        try:
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Redis patterns - check for pub/sub vs cache usage
            redis_patterns = {
                "cache_operations": [
                    r"redis.*get\s*\(",
                    r"redis.*set\s*\(",
                    r"redis.*delete\s*\(",
                    r"redis.*expire\s*\(",
                    r"cache.*get",
                    r"cache.*set",
                ],
                "pubsub_operations": [
                    r"redis.*publish\s*\(",
                    r"redis.*subscribe\s*\(",
                    r"pubsub",
                    r"\.psubscribe",
                    r"\.punsubscribe",
                ],
            }

            for category, patterns in redis_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        redis_usage.append(
                            {
                                "file": str(py_file.relative_to(root)),
                                "category": category,
                                "pattern": pattern,
                                "line": content[: match.start()].count("\n") + 1,
                                "context": content[
                                    max(0, match.start() - 50) : match.end() + 50
                                ].strip(),
                            }
                        )

        except (IOError, OSError, UnicodeDecodeError):
            continue

    return redis_usage


def analyze_orchestrator_architecture(root_path):
    """Analyze the orchestrator service architecture."""
    root = Path(root_path)
    orchestrator_info = {
        "main_files": [],
        "config_files": [],
        "workflow_files": [],
        "activity_files": [],
        "error_handling": [],
        "retry_policies": [],
        "circuit_breakers": [],
    }

    # Look specifically in orchestrator directory
    orchestrator_dir = root / "src" / "orchestrator"
    if orchestrator_dir.exists():
        for py_file in orchestrator_dir.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                rel_path = str(py_file.relative_to(root))

                # Categorize files
                if py_file.name == "main.py":
                    orchestrator_info["main_files"].append(rel_path)
                elif "workflow" in py_file.name.lower():
                    orchestrator_info["workflow_files"].append(rel_path)
                elif "activity" in py_file.name.lower():
                    orchestrator_info["activity_files"].append(rel_path)
                elif "config" in py_file.name.lower():
                    orchestrator_info["config_files"].append(rel_path)

                # Look for error handling patterns
                error_patterns = [
                    r"try.*except",
                    r"retry.*policy",
                    r"exponential.*backoff",
                    r"circuit.*breaker",
                    r"max.*attempts",
                    r"timeout",
                ]

                for pattern in error_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        if "retry" in pattern or "backoff" in pattern:
                            orchestrator_info["retry_policies"].append(
                                {
                                    "file": rel_path,
                                    "pattern": pattern,
                                    "line": content[: match.start()].count("\n") + 1,
                                }
                            )
                        elif "circuit" in pattern:
                            orchestrator_info["circuit_breakers"].append(
                                {
                                    "file": rel_path,
                                    "pattern": pattern,
                                    "line": content[: match.start()].count("\n") + 1,
                                }
                            )
                        else:
                            orchestrator_info["error_handling"].append(
                                {
                                    "file": rel_path,
                                    "pattern": pattern,
                                    "line": content[: match.start()].count("\n") + 1,
                                }
                            )

            except (IOError, OSError, UnicodeDecodeError):
                continue

    return orchestrator_info


def find_evidence_handling(root_path):
    """Find evidence handling and G7 compliance patterns."""
    root = Path(root_path)
    evidence_patterns = []

    for py_file in root.rglob("*.py"):
        try:
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Evidence and audit patterns
            patterns = [
                r"evidence",
                r"audit.*trail",
                r"chain.*of.*custody",
                r"integrity.*check",
                r"hash.*verification",
                r"digital.*signature",
                r"forensic",
                r"G7.*compliance",
                r"legal.*grade",
            ]

            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    evidence_patterns.append(
                        {
                            "file": str(py_file.relative_to(root)),
                            "pattern": pattern,
                            "line": content[: match.start()].count("\n") + 1,
                            "context": content[
                                max(0, match.start() - 30) : match.end() + 30
                            ].strip(),
                        }
                    )

        except (IOError, OSError, UnicodeDecodeError):
            continue

    return evidence_patterns


def find_g8_quota_enforcement(root_path):
    """Find G8 quota and fairness enforcement patterns."""
    root = Path(root_path)
    quota_patterns = []

    for py_file in root.rglob("*.py"):
        try:
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Quota and fairness patterns
            patterns = [
                r"quota",
                r"rate.*limit",
                r"fairness",
                r"weighted.*fair.*queue",
                r"WFQ",
                r"throttle",
                r"tenant.*isolation",
                r"resource.*limit",
                r"G8.*compliance",
            ]

            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    quota_patterns.append(
                        {
                            "file": str(py_file.relative_to(root)),
                            "pattern": pattern,
                            "line": content[: match.start()].count("\n") + 1,
                            "context": content[
                                max(0, match.start() - 30) : match.end() + 30
                            ].strip(),
                        }
                    )

        except (IOError, OSError, UnicodeDecodeError):
            continue

    return quota_patterns


if __name__ == "__main__":
    import sys

    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    print(f"Scanning orchestration and messaging in: {root_dir}")

    # Scan orchestration components
    workflows, activities = find_temporal_workflows(root_dir)
    models, state_machines = find_ptaas_domain_models(root_dir)
    nats_usage = find_nats_jetstream_usage(root_dir)
    redis_usage = find_redis_usage(root_dir)
    orchestrator_info = analyze_orchestrator_architecture(root_dir)
    evidence_patterns = find_evidence_handling(root_dir)
    quota_patterns = find_g8_quota_enforcement(root_dir)

    # Compile orchestrator analysis
    orchestrator_data = {
        "temporal_workflows": workflows,
        "temporal_activities": activities,
        "domain_models": models,
        "state_machines": state_machines,
        "nats_jetstream": nats_usage,
        "redis_usage": redis_usage,
        "orchestrator_architecture": orchestrator_info,
        "evidence_handling": evidence_patterns,
        "quota_enforcement": quota_patterns,
        "summary": {
            "workflows": len(workflows),
            "activities": len(activities),
            "domain_models": len(models),
            "state_patterns": len(state_machines),
            "nats_patterns": len(nats_usage),
            "redis_patterns": len(redis_usage),
            "evidence_patterns": len(evidence_patterns),
            "quota_patterns": len(quota_patterns),
        },
    }

    # Save orchestrator analysis
    with open("docs/audit/catalog/orchestrator_map.json", "w") as f:
        json.dump(orchestrator_data, f, indent=2)

    print("Orchestration analysis completed:")
    print(f"  Temporal workflows: {len(workflows)}")
    print(f"  Temporal activities: {len(activities)}")
    print(f"  Domain models: {len(models)}")
    print(f"  State machine patterns: {len(state_machines)}")
    print(f"  NATS usage patterns: {len(nats_usage)}")
    print(f"  Redis usage patterns: {len(redis_usage)}")
    print(f"  Evidence handling: {len(evidence_patterns)}")
    print(f"  Quota enforcement: {len(quota_patterns)}")
