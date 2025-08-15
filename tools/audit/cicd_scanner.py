#!/usr/bin/env python3
"""CI/CD and operations scanner for XORB monorepo audit."""

import json
import re
import yaml
from pathlib import Path


def scan_makefile_targets(root_path):
    """Scan Makefile for available targets."""
    root = Path(root_path)
    makefiles = (
        list(root.glob("Makefile"))
        + list(root.glob("makefile"))
        + list(root.rglob("Makefile"))
    )

    targets = []

    for makefile in makefiles:
        try:
            with open(makefile, "r", encoding="utf-8") as f:
                content = f.read()

            # Find make targets
            target_pattern = r"^([a-zA-Z0-9_-]+):\s*([^#\n]*)"
            for match in re.finditer(target_pattern, content, re.MULTILINE):
                target_name = match.group(1)
                dependencies = match.group(2).strip()

                # Skip internal targets starting with underscore or dot
                if not target_name.startswith("_") and not target_name.startswith("."):
                    targets.append(
                        {
                            "file": str(makefile.relative_to(root)),
                            "target": target_name,
                            "dependencies": dependencies.split()
                            if dependencies
                            else [],
                            "line": content[: match.start()].count("\n") + 1,
                        }
                    )

        except (IOError, OSError, UnicodeDecodeError):
            continue

    return targets


def analyze_github_actions(root_path):
    """Analyze GitHub Actions workflows."""
    root = Path(root_path)
    workflows_dir = root / ".github" / "workflows"

    workflows = []

    if workflows_dir.exists():
        for workflow_file in workflows_dir.glob("*.yml"):
            try:
                with open(workflow_file, "r", encoding="utf-8") as f:
                    workflow_data = yaml.safe_load(f)

                workflow_info = {
                    "file": str(workflow_file.relative_to(root)),
                    "name": workflow_data.get("name", workflow_file.stem),
                    "triggers": list(workflow_data.get("on", {}).keys())
                    if isinstance(workflow_data.get("on"), dict)
                    else [str(workflow_data.get("on", ""))],
                    "jobs": list(workflow_data.get("jobs", {}).keys()),
                    "job_count": len(workflow_data.get("jobs", {})),
                    "uses_secrets": "secrets." in open(workflow_file, "r").read(),
                    "has_security_scans": any(
                        "security" in job_name.lower() or "scan" in job_name.lower()
                        for job_name in workflow_data.get("jobs", {})
                    ),
                    "has_artifacts": any(
                        "upload-artifact" in str(job)
                        for job in workflow_data.get("jobs", {}).values()
                    ),
                    "environment_usage": [],
                }

                # Check for environment usage
                for job_name, job_config in workflow_data.get("jobs", {}).items():
                    if "environment" in job_config:
                        workflow_info["environment_usage"].append(
                            {"job": job_name, "environment": job_config["environment"]}
                        )

                workflows.append(workflow_info)

            except (IOError, yaml.YAMLError):
                continue

    return workflows


def scan_monitoring_config(root_path):
    """Scan for monitoring and observability configuration."""
    root = Path(root_path)
    monitoring_configs = []

    # Look for monitoring configuration files
    monitoring_patterns = [
        "**/prometheus*.yml",
        "**/prometheus*.yaml",
        "**/grafana/**/*.json",
        "**/grafana/**/*.yml",
        "**/alertmanager*.yml",
        "**/alertmanager*.yaml",
        "**/jaeger*.yml",
        "**/jaeger*.yaml",
        "**/otel*.yml",
        "**/otel*.yaml",
    ]

    for pattern in monitoring_patterns:
        for config_file in root.glob(pattern):
            if config_file.is_file():
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    config_info = {
                        "file": str(config_file.relative_to(root)),
                        "type": "unknown",
                        "size": len(content),
                        "contains_secrets": bool(
                            re.search(
                                r"password|token|key|secret", content, re.IGNORECASE
                            )
                        ),
                    }

                    # Determine config type
                    if "prometheus" in config_file.name.lower():
                        config_info["type"] = "prometheus"
                    elif "grafana" in str(config_file).lower():
                        config_info["type"] = "grafana"
                    elif "alertmanager" in config_file.name.lower():
                        config_info["type"] = "alertmanager"
                    elif "jaeger" in config_file.name.lower():
                        config_info["type"] = "jaeger"
                    elif "otel" in config_file.name.lower():
                        config_info["type"] = "opentelemetry"

                    monitoring_configs.append(config_info)

                except (IOError, OSError, UnicodeDecodeError):
                    continue

    return monitoring_configs


def find_runbooks_and_docs(root_path):
    """Find operational runbooks and documentation."""
    root = Path(root_path)
    runbooks = []

    # Look for runbook patterns
    runbook_patterns = [
        "**/RUNBOOK*.md",
        "**/runbook*.md",
        "**/INCIDENT*.md",
        "**/incident*.md",
        "**/ROLLBACK*.md",
        "**/rollback*.md",
        "**/CHAOS*.md",
        "**/chaos*.md",
        "**/DISASTER*.md",
        "**/disaster*.md",
        "**/RELEASE*.md",
        "**/release*.md",
    ]

    for pattern in runbook_patterns:
        for doc_file in root.glob(pattern):
            if doc_file.is_file():
                try:
                    with open(doc_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    runbook_info = {
                        "file": str(doc_file.relative_to(root)),
                        "type": "runbook",
                        "size": len(content),
                        "word_count": len(content.split()),
                        "has_procedures": bool(
                            re.search(
                                r"step\s+\d+|procedure|process", content, re.IGNORECASE
                            )
                        ),
                        "has_contacts": bool(
                            re.search(
                                r"contact|email|phone|escalat", content, re.IGNORECASE
                            )
                        ),
                        "last_updated": "unknown",  # Could parse git info
                    }

                    # Determine runbook type
                    filename_lower = doc_file.name.lower()
                    if "incident" in filename_lower:
                        runbook_info["type"] = "incident_response"
                    elif "rollback" in filename_lower:
                        runbook_info["type"] = "rollback"
                    elif "chaos" in filename_lower:
                        runbook_info["type"] = "chaos_engineering"
                    elif "disaster" in filename_lower:
                        runbook_info["type"] = "disaster_recovery"
                    elif "release" in filename_lower:
                        runbook_info["type"] = "release_management"

                    runbooks.append(runbook_info)

                except (IOError, OSError, UnicodeDecodeError):
                    continue

    return runbooks


def validate_prometheus_rules(root_path):
    """Validate Prometheus rules files."""
    root = Path(root_path)
    rules_validation = []

    # Find Prometheus rules files
    rules_files = list(root.rglob("*rules*.yml")) + list(root.rglob("*rules*.yaml"))

    for rules_file in rules_files:
        try:
            with open(rules_file, "r", encoding="utf-8") as f:
                content = f.read()

            validation_result = {
                "file": str(rules_file.relative_to(root)),
                "valid_yaml": True,
                "has_groups": False,
                "rule_count": 0,
                "alert_count": 0,
                "recording_count": 0,
                "errors": [],
            }

            try:
                rules_data = yaml.safe_load(content)

                if isinstance(rules_data, dict) and "groups" in rules_data:
                    validation_result["has_groups"] = True

                    for group in rules_data["groups"]:
                        if "rules" in group:
                            for rule in group["rules"]:
                                validation_result["rule_count"] += 1
                                if "alert" in rule:
                                    validation_result["alert_count"] += 1
                                elif "record" in rule:
                                    validation_result["recording_count"] += 1

            except yaml.YAMLError as e:
                validation_result["valid_yaml"] = False
                validation_result["errors"].append(f"YAML parsing error: {str(e)}")

            rules_validation.append(validation_result)

        except (IOError, OSError, UnicodeDecodeError):
            continue

    return rules_validation


def analyze_deployment_configs(root_path):
    """Analyze deployment configurations."""
    root = Path(root_path)
    deployments = []

    # Look for deployment configurations
    deployment_patterns = [
        "**/docker-compose*.yml",
        "**/docker-compose*.yaml",
        "**/kubernetes/*.yml",
        "**/kubernetes/*.yaml",
        "**/k8s/*.yml",
        "**/k8s/*.yaml",
        "**/helm/**/*.yml",
        "**/helm/**/*.yaml",
        "**/*deploy*.yml",
        "**/*deploy*.yaml",
    ]

    for pattern in deployment_patterns:
        for deploy_file in root.glob(pattern):
            if deploy_file.is_file():
                try:
                    with open(deploy_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    deployment_info = {
                        "file": str(deploy_file.relative_to(root)),
                        "type": "unknown",
                        "size": len(content),
                        "has_health_checks": bool(
                            re.search(
                                r"health.*check|liveness|readiness",
                                content,
                                re.IGNORECASE,
                            )
                        ),
                        "has_resource_limits": bool(
                            re.search(
                                r"limits|requests|memory|cpu", content, re.IGNORECASE
                            )
                        ),
                        "has_secrets": bool(
                            re.search(r"secret|password|token", content, re.IGNORECASE)
                        ),
                        "environment_count": len(
                            re.findall(r"environment|env:", content, re.IGNORECASE)
                        ),
                    }

                    # Determine deployment type
                    if "docker-compose" in deploy_file.name:
                        deployment_info["type"] = "docker-compose"
                        # Parse docker-compose specific info
                        try:
                            compose_data = yaml.safe_load(content)
                            if "services" in compose_data:
                                deployment_info["service_count"] = len(
                                    compose_data["services"]
                                )
                                deployment_info["services"] = list(
                                    compose_data["services"].keys()
                                )
                        except yaml.YAMLError:
                            pass
                    elif any(
                        kw in str(deploy_file).lower() for kw in ["kubernetes", "k8s"]
                    ):
                        deployment_info["type"] = "kubernetes"
                    elif "helm" in str(deploy_file).lower():
                        deployment_info["type"] = "helm"

                    deployments.append(deployment_info)

                except (IOError, OSError, UnicodeDecodeError):
                    continue

    return deployments


def scan_release_artifacts(root_path):
    """Scan for release and artifact management."""
    root = Path(root_path)
    artifacts = []

    # Look for release-related files
    release_patterns = [
        "**/CHANGELOG*.md",
        "**/CHANGES*.md",
        "**/HISTORY*.md",
        "**/VERSION*",
        "**/version*",
        "**/*.rpm",
        "**/*.deb",
        "**/*.tar.gz",
        "**/*.zip",
        "**/release*.yml",
        "**/release*.yaml",
    ]

    for pattern in release_patterns:
        for artifact_file in root.glob(pattern):
            if artifact_file.is_file():
                try:
                    stat_info = artifact_file.stat()

                    artifact_info = {
                        "file": str(artifact_file.relative_to(root)),
                        "type": "unknown",
                        "size": stat_info.st_size,
                        "is_binary": artifact_file.suffix
                        in [".rpm", ".deb", ".tar.gz", ".zip", ".bundle"],
                    }

                    # Determine artifact type
                    filename_lower = artifact_file.name.lower()
                    if "changelog" in filename_lower or "changes" in filename_lower:
                        artifact_info["type"] = "changelog"
                    elif "version" in filename_lower:
                        artifact_info["type"] = "version"
                    elif "release" in filename_lower:
                        artifact_info["type"] = "release_config"
                    elif artifact_file.suffix in [".rpm", ".deb"]:
                        artifact_info["type"] = "package"
                    elif artifact_file.suffix in [".tar.gz", ".zip"]:
                        artifact_info["type"] = "archive"

                    artifacts.append(artifact_info)

                except (IOError, OSError):
                    continue

    return artifacts


if __name__ == "__main__":
    import sys

    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    print(f"Scanning CI/CD and operations in: {root_dir}")

    # Run all CI/CD scans
    makefile_targets = scan_makefile_targets(root_dir)
    github_workflows = analyze_github_actions(root_dir)
    monitoring_configs = scan_monitoring_config(root_dir)
    runbooks = find_runbooks_and_docs(root_dir)
    prometheus_rules = validate_prometheus_rules(root_dir)
    deployment_configs = analyze_deployment_configs(root_dir)
    release_artifacts = scan_release_artifacts(root_dir)

    # Compile CI/CD analysis
    cicd_data = {
        "build_system": {
            "makefile_targets": makefile_targets,
            "target_count": len(makefile_targets),
        },
        "ci_cd": {
            "github_workflows": github_workflows,
            "workflow_count": len(github_workflows),
        },
        "monitoring": {
            "configs": monitoring_configs,
            "prometheus_rules": prometheus_rules,
            "config_count": len(monitoring_configs),
        },
        "operations": {"runbooks": runbooks, "runbook_count": len(runbooks)},
        "deployment": {
            "configs": deployment_configs,
            "deployment_count": len(deployment_configs),
        },
        "release_management": {
            "artifacts": release_artifacts,
            "artifact_count": len(release_artifacts),
        },
        "summary": {
            "total_makefile_targets": len(makefile_targets),
            "total_workflows": len(github_workflows),
            "total_monitoring_configs": len(monitoring_configs),
            "total_runbooks": len(runbooks),
            "total_deployments": len(deployment_configs),
            "total_artifacts": len(release_artifacts),
        },
    }

    # Save CI/CD analysis
    with open("docs/audit/catalog/ci_jobs.json", "w") as f:
        json.dump(cicd_data, f, indent=2)

    # Save make targets separately for easier reference
    with open("docs/audit/catalog/make_targets.json", "w") as f:
        json.dump(makefile_targets, f, indent=2)

    print("CI/CD analysis completed:")
    print(f"  Makefile targets: {len(makefile_targets)}")
    print(f"  GitHub workflows: {len(github_workflows)}")
    print(f"  Monitoring configs: {len(monitoring_configs)}")
    print(f"  Runbooks: {len(runbooks)}")
    print(f"  Deployment configs: {len(deployment_configs)}")
    print(f"  Release artifacts: {len(release_artifacts)}")
