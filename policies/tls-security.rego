# OPA/Conftest Security Policies for XORB Platform TLS Configuration
# Ensures secure TLS/mTLS configuration across all services

package xorb.security.tls

import future.keywords.contains
import future.keywords.if
import future.keywords.in

# Default deny policy
default allow = false

# Allowed TLS versions
allowed_tls_versions := {
    "TLSv1.2",
    "TLSv1.3"
}

# Forbidden weak TLS versions
forbidden_tls_versions := {
    "SSLv2",
    "SSLv3", 
    "TLSv1.0",
    "TLSv1.1"
}

# Secure cipher suites
secure_ciphers := {
    "TLS_AES_256_GCM_SHA384",
    "TLS_CHACHA20_POLY1305_SHA256",
    "TLS_AES_128_GCM_SHA256",
    "ECDHE-ECDSA-AES256-GCM-SHA384",
    "ECDHE-ECDSA-CHACHA20-POLY1305",
    "ECDHE-ECDSA-AES128-GCM-SHA256",
    "ECDHE-RSA-AES256-GCM-SHA384",
    "ECDHE-RSA-CHACHA20-POLY1305",
    "ECDHE-RSA-AES128-GCM-SHA256"
}

# Weak cipher suites to deny
weak_ciphers := {
    "RC4",
    "DES",
    "3DES",
    "MD5",
    "SHA1",
    "NULL",
    "EXPORT",
    "ADH",
    "aNULL"
}

# Services that must require client certificates
mtls_required_services := {
    "api",
    "agent", 
    "orchestrator",
    "redis"
}

# Docker Compose TLS Configuration Validation
deny[msg] {
    input.services[service_name]
    service := input.services[service_name]
    
    # Check for plaintext ports exposure
    service.ports[_] == "6379:6379"  # Redis plaintext
    msg := sprintf("Service '%s' exposes plaintext Redis port 6379", [service_name])
}

deny[msg] {
    input.services[service_name]
    service := input.services[service_name]
    
    # Check for plaintext Docker daemon
    service.ports[_] == "2375:2375"  # Docker plaintext
    msg := sprintf("Service '%s' exposes plaintext Docker port 2375", [service_name])
}

deny[msg] {
    input.services[service_name]
    service := input.services[service_name]
    
    # Check for missing TLS environment variables
    service_name in mtls_required_services
    not service.environment.TLS_ENABLED
    msg := sprintf("Service '%s' missing TLS_ENABLED environment variable", [service_name])
}

deny[msg] {
    input.services[service_name]
    service := input.services[service_name]
    
    # Check for TLS disabled
    service.environment.TLS_ENABLED == "false"
    msg := sprintf("Service '%s' has TLS explicitly disabled", [service_name])
}

# Envoy Configuration Validation
deny[msg] {
    input.static_resources.listeners[_].filter_chains[_].transport_socket.typed_config.require_client_certificate == false
    msg := "Envoy listener does not require client certificates - mTLS not enforced"
}

deny[msg] {
    input.static_resources.listeners[_].filter_chains[_].transport_socket.typed_config.tls_params.tls_minimum_protocol_version == version
    version in forbidden_tls_versions
    msg := sprintf("Envoy configured with forbidden TLS version: %s", [version])
}

deny[msg] {
    input.static_resources.listeners[_].filter_chains[_].transport_socket.typed_config.tls_params.cipher_suites[_] == cipher
    cipher_lower := lower(cipher)
    weak_cipher := weak_ciphers[_]
    contains(cipher_lower, lower(weak_cipher))
    msg := sprintf("Envoy configured with weak cipher suite: %s", [cipher])
}

# Redis Configuration Validation
deny[msg] {
    input.port != 0
    msg := "Redis plaintext port is enabled - must be disabled (port 0)"
}

deny[msg] {
    not input["tls-port"]
    msg := "Redis TLS port not configured"
}

deny[msg] {
    not input["tls-auth-clients"]
    msg := "Redis client certificate authentication not enabled"
}

deny[msg] {
    input["tls-auth-clients"] != "yes"
    msg := "Redis client certificate authentication not properly enabled"
}

# Certificate File Validation
deny[msg] {
    input.kind == "Certificate"
    input.spec.duration
    duration_hours := time.parse_duration_ns(input.spec.duration) / 1000000000 / 3600
    duration_hours > 720  # 30 days
    msg := sprintf("Certificate duration too long: %d hours (max 720 hours/30 days)", [duration_hours])
}

deny[msg] {
    input.kind == "Certificate"
    input.spec.renewBefore
    renew_hours := time.parse_duration_ns(input.spec.renewBefore) / 1000000000 / 3600
    duration_hours := time.parse_duration_ns(input.spec.duration) / 1000000000 / 3600
    renew_hours < (duration_hours / 3)  # Renew before 1/3 of lifetime
    msg := sprintf("Certificate renewal threshold too low: %d hours (should be at least %d hours)", [renew_hours, duration_hours / 3])
}

# Kubernetes Security Policies
deny[msg] {
    input.kind == "PeerAuthentication"
    input.spec.mtls.mode != "STRICT"
    msg := "PeerAuthentication must use STRICT mTLS mode"
}

deny[msg] {
    input.kind == "DestinationRule"
    input.spec.trafficPolicy.tls.mode != "ISTIO_MUTUAL"
    msg := "DestinationRule must use ISTIO_MUTUAL TLS mode"
}

deny[msg] {
    input.kind == "DestinationRule"
    input.spec.trafficPolicy.tls.minProtocolVersion
    input.spec.trafficPolicy.tls.minProtocolVersion in forbidden_tls_versions
    msg := sprintf("DestinationRule uses forbidden minimum TLS version: %s", [input.spec.trafficPolicy.tls.minProtocolVersion])
}

# Container Security Validation
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    
    # Check for TLS certificate volumes
    volume_mounts := {vm.mountPath | vm := container.volumeMounts[_]}
    not "/run/tls" in volume_mounts
    not "/certs" in volume_mounts
    msg := sprintf("Container '%s' missing TLS certificate volume mounts", [container.name])
}

deny[msg] {
    input.kind == "Deployment" 
    container := input.spec.template.spec.containers[_]
    
    # Check for running as root
    container.securityContext.runAsUser == 0
    msg := sprintf("Container '%s' running as root user", [container.name])
}

# Network Policy Validation
deny[msg] {
    input.kind == "NetworkPolicy"
    not input.spec.policyTypes
    msg := "NetworkPolicy missing policyTypes specification"
}

deny[msg] {
    input.kind == "NetworkPolicy"
    not "Ingress" in input.spec.policyTypes
    not "Egress" in input.spec.policyTypes
    msg := "NetworkPolicy must specify both Ingress and Egress policies"
}

# Service Security Validation
deny[msg] {
    input.kind == "Service"
    port := input.spec.ports[_]
    
    # Check for insecure ports
    insecure_ports := {6379, 2375, 5432, 3306, 27017}  # Common plaintext ports
    port.port in insecure_ports
    port.name != "tls"
    not contains(port.name, "secure")
    msg := sprintf("Service exposes potentially insecure port %d without TLS indication", [port.port])
}

# Certificate Authority Security
deny[msg] {
    input.kind == "Secret"
    input.type == "Opaque"
    contains(input.metadata.name, "ca")
    
    # Check for proper CA secret handling
    not input.metadata.annotations["security.xorb.io/ca-secret"]
    msg := "CA secret missing security annotation"
}

# File Permission Validation (for local files)
deny[msg] {
    input.type == "file"
    contains(input.path, "key.pem")
    input.permissions != "400"
    msg := sprintf("Private key file '%s' has insecure permissions: %s (should be 400)", [input.path, input.permissions])
}

deny[msg] {
    input.type == "file"
    contains(input.path, "cert.pem")
    input.permissions != "444"
    input.permissions != "400"  # Also allow 400 for certs
    msg := sprintf("Certificate file '%s' has insecure permissions: %s (should be 444 or 400)", [input.path, input.permissions])
}

# Container Image Security
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    
    # Check for latest tag
    contains(container.image, ":latest")
    msg := sprintf("Container '%s' uses 'latest' tag which is not allowed", [container.name])
}

# Secret Management Validation
deny[msg] {
    input.kind in {"Deployment", "StatefulSet", "DaemonSet"}
    container := input.spec.template.spec.containers[_]
    env := container.env[_]
    
    # Check for hardcoded secrets
    contains(lower(env.name), "password")
    not env.valueFrom
    msg := sprintf("Container '%s' has hardcoded password in environment variable '%s'", [container.name, env.name])
}

deny[msg] {
    input.kind in {"Deployment", "StatefulSet", "DaemonSet"}
    container := input.spec.template.spec.containers[_]
    env := container.env[_]
    
    # Check for hardcoded TLS keys
    contains(lower(env.name), "key")
    contains(lower(env.name), "tls")
    not env.valueFrom
    msg := sprintf("Container '%s' has hardcoded TLS key in environment variable '%s'", [container.name, env.name])
}

# Allow rules for compliant configurations
allow {
    # All deny rules passed
    count(deny) == 0
}

# Warnings for best practices
warn[msg] {
    input.kind == "Certificate"
    input.spec.privateKey.algorithm != "RSA"
    input.spec.privateKey.size < 2048
    msg := "Certificate uses key smaller than 2048 bits - consider increasing key size"
}

warn[msg] {
    input.services[service_name]
    service := input.services[service_name]
    
    # Check for missing security options
    not service.security_opt
    msg := sprintf("Service '%s' missing security options - consider adding no-new-privileges", [service_name])
}

warn[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    
    # Check for missing resource limits
    not container.resources.limits
    msg := sprintf("Container '%s' missing resource limits", [container.name])
}