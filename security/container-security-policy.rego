# Open Policy Agent (OPA) policy for container security
# This policy enforces security standards for container images

package container.security

import rego.v1

# Default deny - containers must pass all security checks
default allow := false

# Allow containers that pass all security requirements
allow if {
    not has_critical_vulnerabilities
    not has_dangerous_capabilities
    not runs_as_root
    has_health_check
    has_resource_limits
    not has_secrets_in_env
    uses_secure_base_image
}

# Check for critical vulnerabilities
has_critical_vulnerabilities if {
    input.vulnerabilities[_].severity == "CRITICAL"
}

# Check for dangerous capabilities
has_dangerous_capabilities if {
    dangerous_caps := {"SYS_ADMIN", "NET_ADMIN", "SYS_TIME", "SYS_MODULE"}
    input.security_context.capabilities.add[_] in dangerous_caps
}

# Check if running as root
runs_as_root if {
    input.security_context.run_as_user == 0
}

runs_as_root if {
    not input.security_context.run_as_user
    not input.security_context.run_as_non_root
}

# Check for health check
has_health_check if {
    input.config.health_check
}

# Check for resource limits
has_resource_limits if {
    input.config.resources.limits.memory
    input.config.resources.limits.cpu
}

# Check for secrets in environment variables
has_secrets_in_env if {
    env_var := input.config.env[_]
    contains(lower(env_var.name), "secret")
}

has_secrets_in_env if {
    env_var := input.config.env[_]
    contains(lower(env_var.name), "password")
}

has_secrets_in_env if {
    env_var := input.config.env[_]
    contains(lower(env_var.name), "token")
}

# Check base image security
uses_secure_base_image if {
    # Allow specific secure base images
    secure_images := {
        "python:3.12-slim-bullseye",
        "python:3.12-alpine",
        "gcr.io/distroless/python3",
        "chainguard.dev/python"
    }

    some image in secure_images
    startswith(input.config.image, image)
}

# Security violations with details
violations contains msg if {
    has_critical_vulnerabilities
    msg := "Container has critical vulnerabilities"
}

violations contains msg if {
    has_dangerous_capabilities
    msg := "Container has dangerous capabilities"
}

violations contains msg if {
    runs_as_root
    msg := "Container runs as root user"
}

violations contains msg if {
    not has_health_check
    msg := "Container missing health check"
}

violations contains msg if {
    not has_resource_limits
    msg := "Container missing resource limits"
}

violations contains msg if {
    has_secrets_in_env
    msg := "Container has secrets in environment variables"
}

violations contains msg if {
    not uses_secure_base_image
    msg := "Container uses insecure base image"
}

# Helper functions
lower(str) := output if {
    output := strings.lower(str)
}

contains(str, substr) if {
    strings.contains(str, substr)
}

startswith(str, prefix) if {
    strings.starts_with(str, prefix)
}
