# XORB Federated Node Deployment - Enterprise Terraform Configuration
# Hardened, modular, and composable infrastructure for sovereign XORB nodes

terraform {
  required_version = ">= 1.5"
  required_providers {
    hcloud = {
      source  = "hetznercloud/hcloud"
      version = "~> 1.42"
    }
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.20"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
}

# Variables for node configuration
variable "node_region" {
  description = "XORB node deployment region"
  type        = string
  default     = "eu-central"
  validation {
    condition = contains([
      "eu-central", "us-east", "us-west",
      "asia-pacific", "eu-west", "ca-central"
    ], var.node_region)
    error_message = "Node region must be a valid deployment region."
  }
}

variable "node_tier" {
  description = "XORB node performance tier"
  type        = string
  default     = "enterprise"
  validation {
    condition = contains([
      "development", "staging", "production", "enterprise", "government"
    ], var.node_tier)
    error_message = "Node tier must be development, staging, production, enterprise, or government."
  }
}

variable "compliance_frameworks" {
  description = "Required compliance frameworks for this node"
  type        = list(string)
  default     = ["GDPR", "ISO27001", "SOC2"]
}

variable "federation_enabled" {
  description = "Enable federated learning capabilities"
  type        = bool
  default     = true
}

variable "quantum_crypto_enabled" {
  description = "Enable post-quantum cryptography"
  type        = bool
  default     = true
}

# Local configuration mapping based on node tier
locals {
  node_configs = {
    development = {
      server_type    = "cx21"
      node_count     = 1
      storage_size   = 20
      backup_enabled = false
      monitoring     = "basic"
    }
    staging = {
      server_type    = "cx31"
      node_count     = 2
      storage_size   = 40
      backup_enabled = true
      monitoring     = "standard"
    }
    production = {
      server_type    = "cx41"
      node_count     = 3
      storage_size   = 80
      backup_enabled = true
      monitoring     = "enhanced"
    }
    enterprise = {
      server_type    = "cx51"
      node_count     = 5
      storage_size   = 160
      backup_enabled = true
      monitoring     = "comprehensive"
    }
    government = {
      server_type    = "cx51"
      node_count     = 7
      storage_size   = 320
      backup_enabled = true
      monitoring     = "sovereign"
    }
  }

  current_config = local.node_configs[var.node_tier]

  # Compliance-specific configurations
  compliance_config = {
    GDPR = {
      data_encryption     = true
      audit_logging      = true
      data_residency     = "EU"
      retention_policies = true
    }
    ISO27001 = {
      access_controls    = true
      asset_management   = true
      incident_response  = true
      risk_assessment    = true
    }
    SOC2 = {
      continuous_monitoring = true
      change_management    = true
      disaster_recovery    = true
      vendor_management    = true
    }
    NIS2 = {
      threat_sharing      = true
      risk_reporting      = true
      incident_notification = true
      supply_chain_security = true
    }
  }
}

# Generate SSH key pair for secure access
resource "tls_private_key" "xorb_node_key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "hcloud_ssh_key" "xorb_node_ssh" {
  name       = "xorb-node-${var.node_region}-${random_id.node_id.hex}"
  public_key = tls_private_key.xorb_node_key.public_key_openssh

  labels = {
    node_region = var.node_region
    node_tier   = var.node_tier
    managed_by  = "terraform"
    project     = "xorb-federated"
  }
}

# Random node identifier
resource "random_id" "node_id" {
  byte_length = 8
}

# Dedicated network for XORB node
resource "hcloud_network" "xorb_network" {
  name     = "xorb-network-${var.node_region}-${random_id.node_id.hex}"
  ip_range = "10.0.0.0/16"

  labels = {
    node_region = var.node_region
    node_tier   = var.node_tier
    purpose     = "xorb-federated-network"
  }
}

resource "hcloud_network_subnet" "xorb_subnet" {
  network_id   = hcloud_network.xorb_network.id
  type         = "cloud"
  network_zone = "eu-central"
  ip_range     = "10.0.1.0/24"
}

# Security group with zero-trust principles
resource "hcloud_firewall" "xorb_firewall" {
  name = "xorb-firewall-${var.node_region}-${random_id.node_id.hex}"

  labels = {
    node_region = var.node_region
    node_tier   = var.node_tier
    security    = "zero-trust"
  }

  # SSH access (restricted to management network)
  rule {
    direction = "in"
    port      = "22"
    protocol  = "tcp"
    source_ips = [
      "10.0.0.0/16"  # Internal network only
    ]
  }

  # HTTPS/TLS (public access through load balancer)
  rule {
    direction = "in"
    port      = "443"
    protocol  = "tcp"
    source_ips = [
      "0.0.0.0/0",
      "::/0"
    ]
  }

  # HTTP (redirect to HTTPS)
  rule {
    direction = "in"
    port      = "80"
    protocol  = "tcp"
    source_ips = [
      "0.0.0.0/0",
      "::/0"
    ]
  }

  # XORB federated communication (encrypted)
  rule {
    direction = "in"
    port      = "9000-9010"
    protocol  = "tcp"
    source_ips = [
      "10.0.0.0/16"
    ]
  }

  # Health check endpoints
  rule {
    direction = "in"
    port      = "8080"
    protocol  = "tcp"
    source_ips = [
      "10.0.0.0/16"
    ]
  }

  # Deny all other inbound traffic
  rule {
    direction = "in"
    port      = "any"
    protocol  = "tcp"
    source_ips = [
      "0.0.0.0/0"
    ]
    action = "drop"
  }
}

# Cloud-init configuration for hardened XORB node
data "template_file" "cloud_init" {
  template = file("${path.module}/cloud-init-xorb-node.yml")

  vars = {
    node_id               = random_id.node_id.hex
    node_region          = var.node_region
    node_tier            = var.node_tier
    compliance_frameworks = jsonencode(var.compliance_frameworks)
    federation_enabled    = var.federation_enabled
    quantum_crypto_enabled = var.quantum_crypto_enabled
    ssh_public_key       = tls_private_key.xorb_node_key.public_key_openssh
  }
}

# XORB node servers with high availability
resource "hcloud_server" "xorb_nodes" {
  count       = local.current_config.node_count
  name        = "xorb-node-${var.node_region}-${count.index + 1}-${random_id.node_id.hex}"
  image       = "ubuntu-22.04"
  server_type = local.current_config.server_type
  location    = "fsn1"  # Hetzner Falkenstein (EU)

  ssh_keys  = [hcloud_ssh_key.xorb_node_ssh.id]
  firewall_ids = [hcloud_firewall.xorb_firewall.id]

  user_data = base64gzip(data.template_file.cloud_init.rendered)

  network {
    network_id = hcloud_network.xorb_network.id
    ip         = "10.0.1.${count.index + 10}"
  }

  labels = {
    node_region   = var.node_region
    node_tier     = var.node_tier
    node_index    = count.index + 1
    xorb_role     = count.index == 0 ? "primary" : "secondary"
    managed_by    = "terraform"
    project       = "xorb-federated"
    compliance    = join(",", var.compliance_frameworks)
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Dedicated storage volumes for persistent data
resource "hcloud_volume" "xorb_storage" {
  count    = local.current_config.node_count
  name     = "xorb-storage-${var.node_region}-${count.index + 1}-${random_id.node_id.hex}"
  size     = local.current_config.storage_size
  location = "fsn1"

  labels = {
    node_region = var.node_region
    node_tier   = var.node_tier
    node_index  = count.index + 1
    purpose     = "xorb-persistent-data"
  }
}

# Attach storage volumes to nodes
resource "hcloud_volume_attachment" "xorb_storage_attachment" {
  count     = local.current_config.node_count
  volume_id = hcloud_volume.xorb_storage[count.index].id
  server_id = hcloud_server.xorb_nodes[count.index].id
  automount = false  # Manual mount for encryption
}

# Load balancer for high availability
resource "hcloud_load_balancer" "xorb_lb" {
  name               = "xorb-lb-${var.node_region}-${random_id.node_id.hex}"
  load_balancer_type = var.node_tier == "government" ? "lb21" : "lb11"
  location           = "fsn1"

  labels = {
    node_region = var.node_region
    node_tier   = var.node_tier
    purpose     = "xorb-load-balancing"
  }
}

resource "hcloud_load_balancer_network" "xorb_lb_network" {
  load_balancer_id = hcloud_load_balancer.xorb_lb.id
  network_id       = hcloud_network.xorb_network.id
  ip              = "10.0.1.5"
}

# Load balancer service configuration
resource "hcloud_load_balancer_service" "xorb_https" {
  load_balancer_id = hcloud_load_balancer.xorb_lb.id
  protocol         = "https"
  listen_port      = 443
  destination_port = 8080

  health_check {
    protocol = "http"
    port     = 8080
    interval = 15
    timeout  = 10
    retries  = 3
    http {
      path         = "/health"
      status_codes = ["2??", "3??"]
    }
  }

  http {
    sticky_sessions = true
    redirect_http   = true
    cookie_name     = "XORB_SESSION"
    cookie_lifetime = 3600
  }
}

# Target groups for load balancer
resource "hcloud_load_balancer_target" "xorb_targets" {
  count            = local.current_config.node_count
  type             = "server"
  load_balancer_id = hcloud_load_balancer.xorb_lb.id
  server_id        = hcloud_server.xorb_nodes[count.index].id
  use_private_ip   = true
}

# DNS configuration (if Cloudflare is configured)
resource "cloudflare_record" "xorb_node_dns" {
  count   = var.node_region == "eu-central" ? 1 : 0
  zone_id = var.cloudflare_zone_id
  name    = var.node_region == "eu-central" ? "de" : var.node_region
  value   = hcloud_load_balancer.xorb_lb.ipv4
  type    = "A"
  proxied = true

  comment = "XORB federated node - ${var.node_region}"
}

# IPv6 DNS record
resource "cloudflare_record" "xorb_node_dns_ipv6" {
  count   = var.node_region == "eu-central" ? 1 : 0
  zone_id = var.cloudflare_zone_id
  name    = var.node_region == "eu-central" ? "de" : var.node_region
  value   = hcloud_load_balancer.xorb_lb.ipv6
  type    = "AAAA"
  proxied = true

  comment = "XORB federated node IPv6 - ${var.node_region}"
}

# Backup configuration (if enabled)
resource "hcloud_server_backup" "xorb_backup" {
  count     = local.current_config.backup_enabled ? local.current_config.node_count : 0
  server_id = hcloud_server.xorb_nodes[count.index].id

  labels = {
    node_region = var.node_region
    node_tier   = var.node_tier
    backup_type = "automated"
  }
}

# Output values for other modules
output "node_ips" {
  description = "IP addresses of XORB nodes"
  value = {
    public_ipv4  = hcloud_server.xorb_nodes[*].ipv4_address
    public_ipv6  = hcloud_server.xorb_nodes[*].ipv6_address
    private_ips  = hcloud_server.xorb_nodes[*].network[0].ip
  }
}

output "load_balancer" {
  description = "Load balancer information"
  value = {
    ipv4 = hcloud_load_balancer.xorb_lb.ipv4
    ipv6 = hcloud_load_balancer.xorb_lb.ipv6
    name = hcloud_load_balancer.xorb_lb.name
  }
}

output "network_info" {
  description = "Network configuration"
  value = {
    network_id   = hcloud_network.xorb_network.id
    network_name = hcloud_network.xorb_network.name
    ip_range     = hcloud_network.xorb_network.ip_range
  }
}

output "ssh_private_key" {
  description = "SSH private key for node access"
  value       = tls_private_key.xorb_node_key.private_key_pem
  sensitive   = true
}

output "node_configuration" {
  description = "Node configuration summary"
  value = {
    node_id               = random_id.node_id.hex
    node_region          = var.node_region
    node_tier            = var.node_tier
    node_count           = local.current_config.node_count
    compliance_frameworks = var.compliance_frameworks
    federation_enabled    = var.federation_enabled
    quantum_crypto_enabled = var.quantum_crypto_enabled
  }
}

# Compliance reporting
output "compliance_status" {
  description = "Compliance framework status"
  value = {
    for framework in var.compliance_frameworks :
    framework => local.compliance_config[framework]
  }
}

# Variables that need to be provided
variable "cloudflare_zone_id" {
  description = "Cloudflare Zone ID for DNS management"
  type        = string
  default     = ""
}

variable "hcloud_token" {
  description = "Hetzner Cloud API token"
  type        = string
  sensitive   = true
}

variable "cloudflare_api_token" {
  description = "Cloudflare API token"
  type        = string
  sensitive   = true
  default     = ""
}
