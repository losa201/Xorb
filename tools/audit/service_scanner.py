#!/usr/bin/env python3
"""Service and endpoint scanner for XORB monorepo audit."""

import json
import re
import yaml
from pathlib import Path


def find_fastapi_routes(root_path):
    """Find FastAPI routes and endpoints."""
    root = Path(root_path)
    routes = []

    # Look for FastAPI router patterns
    python_files = list(root.rglob("*.py"))

    for py_file in python_files:
        try:
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Look for FastAPI patterns
            fastapi_patterns = [
                r'@(router|app)\.(get|post|put|delete|patch|options|head)\s*\(\s*["\']([^"\']+)["\']',
                r'\.add_api_route\s*\(\s*["\']([^"\']+)["\']',
                r"APIRouter\s*\(",
                r"FastAPI\s*\(",
            ]

            file_routes = []
            for pattern in fastapi_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    if len(match.groups()) >= 3:
                        method = match.group(2).upper()
                        path = match.group(3)
                        file_routes.append(
                            {
                                "file": str(py_file.relative_to(root)),
                                "method": method,
                                "path": path,
                                "line": content[: match.start()].count("\n") + 1,
                            }
                        )

            # Also look for router includes
            router_includes = re.finditer(r"app\.include_router\s*\(\s*(\w+)", content)
            for match in router_includes:
                file_routes.append(
                    {
                        "file": str(py_file.relative_to(root)),
                        "type": "router_include",
                        "router": match.group(1),
                        "line": content[: match.start()].count("\n") + 1,
                    }
                )

            routes.extend(file_routes)

        except (IOError, OSError, UnicodeDecodeError):
            continue

    return routes


def find_express_routes(root_path):
    """Find Express.js/Node.js routes."""
    root = Path(root_path)
    routes = []

    js_files = list(root.rglob("*.js")) + list(root.rglob("*.ts"))

    for js_file in js_files:
        try:
            with open(js_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Express patterns
            express_patterns = [
                r'(app|router)\.(get|post|put|delete|patch|use)\s*\(\s*["\']([^"\']+)["\']',
                r'\.route\s*\(\s*["\']([^"\']+)["\']',
            ]

            for pattern in express_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    routes.append(
                        {
                            "file": str(js_file.relative_to(root)),
                            "framework": "express",
                            "method": match.group(2).upper()
                            if len(match.groups()) >= 2
                            else "UNKNOWN",
                            "path": match.group(3)
                            if len(match.groups()) >= 3
                            else match.group(1),
                            "line": content[: match.start()].count("\n") + 1,
                        }
                    )

        except (IOError, OSError, UnicodeDecodeError):
            continue

    return routes


def parse_docker_compose_files(root_path):
    """Parse docker-compose files to extract service definitions."""
    root = Path(root_path)
    services = {}

    compose_files = list(root.glob("docker-compose*.yml")) + list(
        root.glob("docker-compose*.yaml")
    )
    compose_files.extend(list(root.rglob("docker-compose*.yml")))
    compose_files.extend(list(root.rglob("docker-compose*.yaml")))

    for compose_file in compose_files:
        try:
            with open(compose_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if "services" in data:
                for service_name, service_config in data["services"].items():
                    service_info = {
                        "name": service_name,
                        "file": str(compose_file.relative_to(root)),
                        "image": service_config.get("image", ""),
                        "build": service_config.get("build", ""),
                        "ports": service_config.get("ports", []),
                        "environment": service_config.get("environment", {}),
                        "volumes": service_config.get("volumes", []),
                        "depends_on": service_config.get("depends_on", []),
                        "networks": service_config.get("networks", []),
                        "healthcheck": service_config.get("healthcheck", {}),
                    }

                    # Create unique key for service
                    service_key = f"{compose_file.name}:{service_name}"
                    services[service_key] = service_info

        except (IOError, yaml.YAMLError):
            continue

    return services


def find_grpc_services(root_path):
    """Find gRPC service definitions in .proto files."""
    root = Path(root_path)
    grpc_services = []

    proto_files = list(root.rglob("*.proto"))

    for proto_file in proto_files:
        try:
            with open(proto_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Find service definitions
            service_pattern = r"service\s+(\w+)\s*\{([^}]+)\}"
            rpc_pattern = r"rpc\s+(\w+)\s*\(\s*(\w+)\s*\)\s*returns\s*\(\s*(\w+)\s*\)"

            for service_match in re.finditer(service_pattern, content, re.DOTALL):
                service_name = service_match.group(1)
                service_body = service_match.group(2)

                rpcs = []
                for rpc_match in re.finditer(rpc_pattern, service_body):
                    rpcs.append(
                        {
                            "method": rpc_match.group(1),
                            "request": rpc_match.group(2),
                            "response": rpc_match.group(3),
                        }
                    )

                grpc_services.append(
                    {
                        "file": str(proto_file.relative_to(root)),
                        "service": service_name,
                        "rpcs": rpcs,
                    }
                )

        except (IOError, OSError, UnicodeDecodeError):
            continue

    return grpc_services


def extract_openapi_specs(root_path):
    """Extract OpenAPI/Swagger specifications."""
    root = Path(root_path)
    openapi_specs = []

    # Look for OpenAPI files
    openapi_patterns = [
        "*openapi*.json",
        "*openapi*.yml",
        "*openapi*.yaml",
        "*swagger*.json",
        "*swagger*.yml",
        "*swagger*.yaml",
    ]

    for pattern in openapi_patterns:
        for spec_file in root.rglob(pattern):
            try:
                with open(spec_file, "r", encoding="utf-8") as f:
                    if spec_file.suffix.lower() == ".json":
                        spec_data = json.load(f)
                    else:
                        spec_data = yaml.safe_load(f)

                openapi_specs.append(
                    {
                        "file": str(spec_file.relative_to(root)),
                        "version": spec_data.get(
                            "openapi", spec_data.get("swagger", "unknown")
                        ),
                        "title": spec_data.get("info", {}).get("title", "Unknown"),
                        "version_info": spec_data.get("info", {}).get(
                            "version", "unknown"
                        ),
                        "paths": list(spec_data.get("paths", {}).keys()),
                        "servers": spec_data.get("servers", []),
                    }
                )

            except (IOError, json.JSONDecodeError, yaml.YAMLError):
                continue

    return openapi_specs


def find_message_bus_usage(root_path):
    """Find NATS JetStream and other message bus usage."""
    root = Path(root_path)
    bus_usage = []

    # Search for NATS patterns
    patterns = {
        "nats_jetstream": [
            r"jetstream",
            r"\.publish\s*\(",
            r"\.subscribe\s*\(",
            r"xorb\..*\.ptaas\.job\.",
            r"consumer.*group",
            r"stream.*config",
        ],
        "redis_pubsub": [
            r"redis.*pub",
            r"redis.*sub",
            r"\.publish\s*\(",
            r"\.subscribe\s*\(",
            r"pubsub",
        ],
        "rabbitmq": [r"rabbitmq", r"amqp", r"queue\.declare", r"exchange\.declare"],
    }

    for file_path in root.rglob("*.py"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            for bus_type, bus_patterns in patterns.items():
                for pattern in bus_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        bus_usage.append(
                            {
                                "file": str(file_path.relative_to(root)),
                                "bus_type": bus_type,
                                "pattern": pattern,
                                "line": content[: match.start()].count("\n") + 1,
                                "context": content[
                                    max(0, match.start() - 50) : match.end() + 50
                                ].strip(),
                            }
                        )

        except (IOError, OSError, UnicodeDecodeError):
            continue

    return bus_usage


def analyze_service_dependencies(services, routes):
    """Analyze service dependencies and communication patterns."""
    dependencies = {}

    for service_key, service in services.items():
        deps = {
            "declared_deps": service.get("depends_on", []),
            "env_deps": [],
            "network_deps": service.get("networks", []),
            "volume_deps": service.get("volumes", []),
        }

        # Analyze environment variables for service dependencies
        env = service.get("environment", {})
        if isinstance(env, dict):
            for key, value in env.items():
                if any(
                    keyword in key.lower()
                    for keyword in ["url", "host", "endpoint", "service"]
                ):
                    deps["env_deps"].append(f"{key}={value}")
        elif isinstance(env, list):
            for item in env:
                if "=" in str(item) and any(
                    keyword in str(item).lower()
                    for keyword in ["url", "host", "endpoint", "service"]
                ):
                    deps["env_deps"].append(str(item))

        dependencies[service_key] = deps

    return dependencies


if __name__ == "__main__":
    import sys

    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    print(f"Scanning services and endpoints in: {root_dir}")

    # Scan all service types
    fastapi_routes = find_fastapi_routes(root_dir)
    express_routes = find_express_routes(root_dir)
    docker_services = parse_docker_compose_files(root_dir)
    grpc_services = find_grpc_services(root_dir)
    openapi_specs = extract_openapi_specs(root_dir)
    bus_usage = find_message_bus_usage(root_dir)

    # Analyze dependencies
    dependencies = analyze_service_dependencies(
        docker_services, fastapi_routes + express_routes
    )

    # Save endpoints catalog
    endpoints_data = {
        "fastapi_routes": fastapi_routes,
        "express_routes": express_routes,
        "grpc_services": grpc_services,
        "openapi_specs": openapi_specs,
        "message_bus_usage": bus_usage,
    }

    with open("docs/audit/catalog/endpoints.json", "w") as f:
        json.dump(endpoints_data, f, indent=2)

    # Save services catalog
    services_data = {
        "docker_services": docker_services,
        "dependencies": dependencies,
        "service_summary": {
            "total_docker_services": len(docker_services),
            "total_fastapi_routes": len(fastapi_routes),
            "total_express_routes": len(express_routes),
            "total_grpc_services": len(grpc_services),
            "total_openapi_specs": len(openapi_specs),
            "message_bus_patterns": len(bus_usage),
        },
    }

    with open("docs/audit/catalog/services.json", "w") as f:
        json.dump(services_data, f, indent=2)

    print("Services catalog generated:")
    print(f"  Docker services: {len(docker_services)}")
    print(f"  FastAPI routes: {len(fastapi_routes)}")
    print(f"  Express routes: {len(express_routes)}")
    print(f"  gRPC services: {len(grpc_services)}")
    print(f"  OpenAPI specs: {len(openapi_specs)}")
    print(f"  Message bus usage: {len(bus_usage)}")
