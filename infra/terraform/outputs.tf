# XORB Terraform Outputs
# Export important infrastructure information

# Cluster Information
output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_version" {
  description = "The Kubernetes version for the EKS cluster"
  value       = module.eks.cluster_version
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "cluster_iam_role_arn" {
  description = "IAM role ARN associated with EKS cluster"
  value       = module.eks.cluster_iam_role_arn
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

# Network Information
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = module.vpc.private_subnets
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = module.vpc.public_subnets
}

output "database_subnet_ids" {
  description = "IDs of the database subnets"
  value       = module.vpc.database_subnets
}

output "nat_gateway_ids" {
  description = "IDs of the NAT gateways"
  value       = module.vpc.natgw_ids
}

# Database Information
output "rds_endpoint" {
  description = "PostgreSQL RDS instance endpoint"
  value       = module.rds.db_instance_endpoint
}

output "rds_port" {
  description = "PostgreSQL RDS instance port"
  value       = module.rds.db_instance_port
}

output "rds_database_name" {
  description = "PostgreSQL RDS database name"
  value       = module.rds.db_instance_name
}

output "rds_username" {
  description = "PostgreSQL RDS master username"
  value       = module.rds.db_instance_username
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis ElastiCache endpoint"
  value       = module.redis.cluster_address
}

output "redis_port" {
  description = "Redis ElastiCache port"
  value       = module.redis.port
}

# Security Information
output "kms_key_id" {
  description = "KMS key ID for encryption"
  value       = aws_kms_key.xorb.id
}

output "kms_key_arn" {
  description = "KMS key ARN for encryption"
  value       = aws_kms_key.xorb.arn
}

output "secrets_manager_secret_arn" {
  description = "ARN of the Secrets Manager secret"
  value       = aws_secretsmanager_secret.xorb_secrets.arn
}

# Load Balancer Information
output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = module.alb.lb_dns_name
}

output "alb_zone_id" {
  description = "Zone ID of the Application Load Balancer"
  value       = module.alb.lb_zone_id
}

output "alb_arn" {
  description = "ARN of the Application Load Balancer"
  value       = module.alb.lb_arn
}

# Storage Information
output "s3_data_bucket" {
  description = "Name of the S3 bucket for XORB data"
  value       = aws_s3_bucket.xorb_data.id
}

output "s3_data_bucket_arn" {
  description = "ARN of the S3 bucket for XORB data"
  value       = aws_s3_bucket.xorb_data.arn
}

output "s3_logs_bucket" {
  description = "Name of the S3 bucket for ALB logs"
  value       = aws_s3_bucket.alb_logs.id
}

# Node Group Information
output "node_groups" {
  description = "EKS node groups information"
  value = {
    general = {
      arn           = module.eks.eks_managed_node_groups.general.node_group_arn
      status        = module.eks.eks_managed_node_groups.general.node_group_status
      capacity_type = module.eks.eks_managed_node_groups.general.capacity_type
    }
    ml_compute = {
      arn           = module.eks.eks_managed_node_groups.ml_compute.node_group_arn
      status        = module.eks.eks_managed_node_groups.ml_compute.node_group_status
      capacity_type = module.eks.eks_managed_node_groups.ml_compute.capacity_type
    }
    monitoring = {
      arn           = module.eks.eks_managed_node_groups.monitoring.node_group_arn
      status        = module.eks.eks_managed_node_groups.monitoring.node_group_status
      capacity_type = module.eks.eks_managed_node_groups.monitoring.capacity_type
    }
  }
}

# CloudWatch Information
output "cloudwatch_log_groups" {
  description = "CloudWatch log groups"
  value = {
    for service, log_group in aws_cloudwatch_log_group.xorb_logs : 
    service => {
      name = log_group.name
      arn  = log_group.arn
    }
  }
}

# IAM Information
output "service_role_arn" {
  description = "ARN of the XORB service IAM role"
  value       = aws_iam_role.xorb_service_role.arn
}

# Environment Information
output "environment" {
  description = "Environment name"
  value       = var.environment
}

output "region" {
  description = "AWS region"
  value       = var.aws_region
}

output "availability_zones" {
  description = "Availability zones used"
  value       = local.azs
}

# Configuration for kubectl
output "configure_kubectl" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}

# Connection strings (for application configuration)
output "database_connection_string" {
  description = "Database connection string template"
  value       = "postgresql://${module.rds.db_instance_username}:PASSWORD@${module.rds.db_instance_endpoint}:${module.rds.db_instance_port}/${module.rds.db_instance_name}"
  sensitive   = true
}

output "redis_connection_string" {
  description = "Redis connection string template"
  value       = "redis://:PASSWORD@${module.redis.cluster_address}:${module.redis.port}"
  sensitive   = true
}

# Monitoring endpoints
output "monitoring_endpoints" {
  description = "Monitoring service endpoints"
  value = {
    prometheus = "http://prometheus.${var.domain_name}"
    grafana    = "http://grafana.${var.domain_name}"
    jaeger     = "http://jaeger.${var.domain_name}"
  }
}

# Cost optimization information
output "cost_optimization" {
  description = "Cost optimization recommendations"
  value = {
    spot_instance_savings = "Estimated 60-70% savings on ML compute nodes using Spot instances"
    storage_optimization  = "Use S3 Intelligent Tiering for long-term data storage"
    monitoring_costs     = "Monitor CloudWatch costs and adjust log retention as needed"
  }
}

# Security recommendations
output "security_recommendations" {
  description = "Security configuration recommendations"
  value = {
    secrets_rotation = "Rotate database passwords every 90 days using Secrets Manager"
    encryption      = "All data encrypted at rest using customer-managed KMS keys"
    network_security = "Private subnets for all workloads, public access only through ALB"
    compliance      = "Configure according to SOC2 and ISO27001 standards"
  }
}

# Next steps
output "deployment_next_steps" {
  description = "Next steps for deployment"
  value = [
    "1. Configure kubectl: ${local.cluster_name}",
    "2. Install AWS Load Balancer Controller",
    "3. Deploy XORB applications using Helm charts",
    "4. Configure monitoring dashboards",
    "5. Set up alerting rules",
    "6. Configure backup procedures"
  ]
}