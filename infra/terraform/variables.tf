# XORB Terraform Variables
# Comprehensive variable definitions for infrastructure configuration

# Project and environment
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "xorb"
}

variable "environment" {
  description = "Environment name (development, staging, production)"
  type        = string
  default     = "development"

  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be one of: development, staging, production."
  }
}

variable "project_owner" {
  description = "Project owner/team"
  type        = string
  default     = "xorb-security-team"
}

# AWS Configuration
variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-west-2"
}

variable "terraform_state_bucket" {
  description = "S3 bucket for Terraform state"
  type        = string
}

variable "terraform_lock_table" {
  description = "DynamoDB table for Terraform state locking"
  type        = string
}

# Network Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "az_count" {
  description = "Number of availability zones to use"
  type        = number
  default     = 3

  validation {
    condition     = var.az_count >= 2 && var.az_count <= 6
    error_message = "AZ count must be between 2 and 6."
  }
}

variable "enable_vpn_gateway" {
  description = "Enable VPN gateway for VPC"
  type        = bool
  default     = false
}

# Kubernetes Configuration
variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "cluster_endpoint_public_access" {
  description = "Enable public access to cluster endpoint"
  type        = bool
  default     = true
}

# Node Group Configuration - General
variable "general_node_instance_types" {
  description = "Instance types for general node group"
  type        = list(string)
  default     = ["t3.large", "t3.xlarge"]
}

variable "general_node_min_size" {
  description = "Minimum size for general node group"
  type        = number
  default     = 1
}

variable "general_node_max_size" {
  description = "Maximum size for general node group"
  type        = number
  default     = 10
}

variable "general_node_desired_size" {
  description = "Desired size for general node group"
  type        = number
  default     = 3
}

# Node Group Configuration - ML Compute
variable "ml_node_instance_types" {
  description = "Instance types for ML compute node group"
  type        = list(string)
  default     = ["c5.2xlarge", "c5.4xlarge", "m5.2xlarge"]
}

variable "ml_node_capacity_type" {
  description = "Capacity type for ML nodes (ON_DEMAND or SPOT)"
  type        = string
  default     = "SPOT"

  validation {
    condition     = contains(["ON_DEMAND", "SPOT"], var.ml_node_capacity_type)
    error_message = "ML node capacity type must be ON_DEMAND or SPOT."
  }
}

variable "ml_node_min_size" {
  description = "Minimum size for ML compute node group"
  type        = number
  default     = 0
}

variable "ml_node_max_size" {
  description = "Maximum size for ML compute node group"
  type        = number
  default     = 5
}

variable "ml_node_desired_size" {
  description = "Desired size for ML compute node group"
  type        = number
  default     = 1
}

# Node Group Configuration - Monitoring
variable "monitoring_node_instance_types" {
  description = "Instance types for monitoring node group"
  type        = list(string)
  default     = ["m5.large", "m5.xlarge"]
}

variable "monitoring_node_min_size" {
  description = "Minimum size for monitoring node group"
  type        = number
  default     = 1
}

variable "monitoring_node_max_size" {
  description = "Maximum size for monitoring node group"
  type        = number
  default     = 3
}

variable "monitoring_node_desired_size" {
  description = "Desired size for monitoring node group"
  type        = number
  default     = 2
}

# Database Configuration - PostgreSQL
variable "postgres_version" {
  description = "PostgreSQL version"
  type        = string
  default     = "15.4"
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "rds_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 100
}

variable "rds_max_allocated_storage" {
  description = "RDS maximum allocated storage in GB"
  type        = number
  default     = 1000
}

variable "database_name" {
  description = "Name of the database"
  type        = string
  default     = "xorb"
}

variable "database_username" {
  description = "Database username"
  type        = string
  default     = "xorb"
}

variable "rds_maintenance_window" {
  description = "RDS maintenance window"
  type        = string
  default     = "sun:03:00-sun:04:00"
}

variable "rds_backup_window" {
  description = "RDS backup window"
  type        = string
  default     = "02:00-03:00"
}

variable "rds_backup_retention_period" {
  description = "RDS backup retention period in days"
  type        = number
  default     = 7
}

variable "rds_performance_insights_enabled" {
  description = "Enable RDS Performance Insights"
  type        = bool
  default     = true
}

# Database Configuration - Redis
variable "redis_version" {
  description = "Redis engine version"
  type        = string
  default     = "7.0"
}

variable "redis_node_type" {
  description = "Redis node type"
  type        = string
  default     = "cache.t3.medium"
}

variable "redis_num_nodes" {
  description = "Number of Redis nodes"
  type        = number
  default     = 1
}

variable "redis_maintenance_window" {
  description = "Redis maintenance window"
  type        = string
  default     = "sun:03:00-sun:04:00"
}

variable "redis_snapshot_window" {
  description = "Redis snapshot window"
  type        = string
  default     = "02:00-03:00"
}

variable "redis_snapshot_retention_limit" {
  description = "Redis snapshot retention limit in days"
  type        = number
  default     = 7
}

# Monitoring and Logging
variable "cloudwatch_log_retention_days" {
  description = "CloudWatch log retention period in days"
  type        = number
  default     = 30
}

# Security Configuration
variable "enable_security_scanning" {
  description = "Enable security scanning features"
  type        = bool
  default     = true
}

variable "enable_compliance_monitoring" {
  description = "Enable compliance monitoring"
  type        = bool
  default     = true
}

# Cost Optimization
variable "enable_cost_optimization" {
  description = "Enable cost optimization features"
  type        = bool
  default     = true
}

variable "spot_instance_percentage" {
  description = "Percentage of spot instances to use"
  type        = number
  default     = 50

  validation {
    condition     = var.spot_instance_percentage >= 0 && var.spot_instance_percentage <= 100
    error_message = "Spot instance percentage must be between 0 and 100."
  }
}

# Backup and Disaster Recovery
variable "enable_cross_region_backup" {
  description = "Enable cross-region backup"
  type        = bool
  default     = false
}

variable "backup_retention_days" {
  description = "Backup retention period in days"
  type        = number
  default     = 30
}

# Feature Flags
variable "enable_adversarial_features" {
  description = "Enable adversarial ML features"
  type        = bool
  default     = true
}

variable "enable_gpu_nodes" {
  description = "Enable GPU nodes for ML workloads"
  type        = bool
  default     = false
}

variable "enable_monitoring_stack" {
  description = "Enable comprehensive monitoring stack"
  type        = bool
  default     = true
}

variable "enable_service_mesh" {
  description = "Enable service mesh (Istio)"
  type        = bool
  default     = false
}

# Domain and SSL
variable "domain_name" {
  description = "Domain name for XORB platform"
  type        = string
  default     = ""
}

variable "ssl_certificate_arn" {
  description = "ARN of SSL certificate in ACM"
  type        = string
  default     = ""
}

# Additional Tags
variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# Resource Limits
variable "max_pods_per_node" {
  description = "Maximum number of pods per node"
  type        = number
  default     = 110
}

variable "node_volume_size" {
  description = "EBS volume size for worker nodes in GB"
  type        = number
  default     = 100
}

# Autoscaling Configuration
variable "enable_cluster_autoscaler" {
  description = "Enable cluster autoscaler"
  type        = bool
  default     = true
}

variable "enable_horizontal_pod_autoscaler" {
  description = "Enable horizontal pod autoscaler"
  type        = bool
  default     = true
}

variable "enable_vertical_pod_autoscaler" {
  description = "Enable vertical pod autoscaler"
  type        = bool
  default     = false
}

# Development and Testing
variable "enable_debug_mode" {
  description = "Enable debug mode for development"
  type        = bool
  default     = false
}

variable "enable_test_resources" {
  description = "Create additional resources for testing"
  type        = bool
  default     = false
}

# Compliance and Security Standards
variable "compliance_standards" {
  description = "List of compliance standards to adhere to"
  type        = list(string)
  default     = ["SOC2", "ISO27001", "NIST"]
}

variable "security_scan_schedule" {
  description = "Schedule for security scans (cron format)"
  type        = string
  default     = "0 2 * * *"  # Daily at 2 AM
}
