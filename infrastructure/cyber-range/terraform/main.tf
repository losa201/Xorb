# XORB PTaaS Red vs Blue Cyber Range - Terraform Infrastructure
# Provisions cloud infrastructure for the cyber range environment

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
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

# Variables
variable "region" {
  description = "AWS region for the cyber range"
  type        = string
  default     = "us-west-2"
}

variable "availability_zones" {
  description = "Availability zones for the cyber range"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b", "us-west-2c"]
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "xorb-cyber-range"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "cyber-range"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "enable_flow_logs" {
  description = "Enable VPC Flow Logs for monitoring"
  type        = bool
  default     = true
}

variable "enable_kill_switch" {
  description = "Enable emergency kill switch capabilities"
  type        = bool
  default     = true
}

variable "max_exercise_duration" {
  description = "Maximum exercise duration in hours"
  type        = number
  default     = 8
}

variable "node_instance_types" {
  description = "Instance types for EKS worker nodes"
  type = object({
    control_plane = string
    red_team      = string
    blue_team     = string
    targets       = string
  })
  default = {
    control_plane = "m5.xlarge"
    red_team      = "m5.2xlarge"
    blue_team     = "m5.2xlarge"
    targets       = "m5.4xlarge"
  }
}

# Provider Configuration
provider "aws" {
  region = var.region
  
  default_tags {
    tags = {
      Environment = var.environment
      Project     = "XORB-PTaaS-CyberRange"
      ManagedBy   = "Terraform"
      Purpose     = "RedBlueTeamExercises"
    }
  }
}

# Random password for various services
resource "random_password" "cyber_range_secrets" {
  for_each = toset([
    "grafana_admin",
    "elasticsearch_password",
    "kibana_encryption_key",
    "jupyter_token",
    "misp_admin",
    "wazuh_api",
    "gophish_admin",
    "metasploit_db"
  ])
  
  length  = 32
  special = true
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# VPC and Networking
module "vpc" {
  source = "./modules/vpc"
  
  name = "${var.cluster_name}-vpc"
  cidr = var.vpc_cidr
  
  azs              = var.availability_zones
  private_subnets  = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets   = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  database_subnets = ["10.0.201.0/24", "10.0.202.0/24", "10.0.203.0/24"]
  
  # Cyber range specific subnets
  intra_subnets = [
    "10.0.10.0/24",  # Control plane
    "10.0.20.0/24",  # Red team
    "10.0.30.0/24",  # Blue team
    "10.0.100.0/24", # Target web
    "10.0.110.0/24", # Target internal
    "10.0.120.0/24", # Target OT/IoT
    "10.0.200.0/24"  # Simulation
  ]
  
  enable_nat_gateway   = true
  enable_vpn_gateway   = false
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  # Enable flow logs for security monitoring
  enable_flow_log                      = var.enable_flow_logs
  create_flow_log_cloudwatch_iam_role  = true
  create_flow_log_cloudwatch_log_group = true
  
  tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "CyberRange"                                 = "true"
    "SecurityMonitoring"                         = "enabled"
  }
  
  public_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/elb"                     = "1"
    "CyberRangeZone"                             = "public"
  }
  
  private_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/internal-elb"            = "1"
    "CyberRangeZone"                             = "private"
  }
  
  intra_subnet_tags = {
    "CyberRangeZone" = "isolated"
    "SecurityLevel"  = "high"
  }
}

# Security Groups
resource "aws_security_group" "cyber_range_control" {
  name_prefix = "${var.cluster_name}-control-"
  vpc_id      = module.vpc.vpc_id
  description = "Security group for cyber range control plane"
  
  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    description = "XORB Orchestrator"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  ingress {
    description = "Kill Switch Emergency"
    from_port   = 8081
    to_port     = 8081
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "${var.cluster_name}-control-sg"
    Zone = "control-plane"
  }
}

resource "aws_security_group" "cyber_range_red_team" {
  name_prefix = "${var.cluster_name}-red-team-"
  vpc_id      = module.vpc.vpc_id
  description = "Security group for red team infrastructure"
  
  # Restrict red team access to only target networks
  ingress {
    description = "Red team tools"
    from_port   = 1024
    to_port     = 65535
    protocol    = "tcp"
    cidr_blocks = ["10.0.100.0/24", "10.0.110.0/24", "10.0.120.0/24"]
  }
  
  # Block access to control and blue team networks
  egress {
    description = "Block control plane"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["10.0.10.0/24"]
  }
  
  egress {
    description = "Block blue team"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["10.0.30.0/24"]
  }
  
  # Allow access to targets
  egress {
    description = "Target networks"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["10.0.100.0/24", "10.0.110.0/24", "10.0.120.0/24"]
  }
  
  tags = {
    Name = "${var.cluster_name}-red-team-sg"
    Zone = "red-team"
    IsolationLevel = "strict"
  }
}

resource "aws_security_group" "cyber_range_blue_team" {
  name_prefix = "${var.cluster_name}-blue-team-"
  vpc_id      = module.vpc.vpc_id
  description = "Security group for blue team SOC"
  
  ingress {
    description = "Kibana"
    from_port   = 5601
    to_port     = 5601
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  ingress {
    description = "Elasticsearch"
    from_port   = 9200
    to_port     = 9200
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  ingress {
    description = "Logstash"
    from_port   = 5044
    to_port     = 5046
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  ingress {
    description = "Wazuh"
    from_port   = 1514
    to_port     = 1516
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  ingress {
    description = "Jupyter"
    from_port   = 8888
    to_port     = 8888
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "${var.cluster_name}-blue-team-sg"
    Zone = "blue-team"
  }
}

# EKS Cluster
module "eks" {
  source = "./modules/eks"
  
  cluster_name                   = var.cluster_name
  cluster_version               = "1.28"
  cluster_endpoint_public_access = true
  
  # Enhanced security for cyber range
  cluster_endpoint_public_access_cidrs = ["0.0.0.0/0"]  # Restrict in production
  cluster_endpoint_private_access      = true
  
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
    aws-efs-csi-driver = {
      most_recent = true
    }
  }
  
  vpc_id                   = module.vpc.vpc_id
  subnet_ids               = module.vpc.private_subnets
  control_plane_subnet_ids = module.vpc.intra_subnets
  
  # Enhanced monitoring and logging
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  # Node groups for different cyber range components
  eks_managed_node_groups = {
    control_plane = {
      name            = "control-plane"
      use_name_prefix = true
      
      subnet_ids = [module.vpc.private_subnets[0]]
      
      min_size     = 1
      max_size     = 3
      desired_size = 2
      
      ami_type       = "AL2_x86_64"
      instance_types = [var.node_instance_types.control_plane]
      
      k8s_labels = {
        "cyber-range.xorb.io/zone" = "control-plane"
        "cyber-range.xorb.io/role" = "management"
      }
      
      taints = [
        {
          key    = "cyber-range.xorb.io/control-plane"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
      
      tags = {
        Zone = "control-plane"
        Purpose = "management"
      }
    }
    
    red_team = {
      name            = "red-team"
      use_name_prefix = true
      
      subnet_ids = [module.vpc.private_subnets[1]]
      
      min_size     = 1
      max_size     = 5
      desired_size = 2
      
      ami_type       = "AL2_x86_64"
      instance_types = [var.node_instance_types.red_team]
      
      k8s_labels = {
        "cyber-range.xorb.io/zone" = "red-team"
        "cyber-range.xorb.io/role" = "attack"
      }
      
      taints = [
        {
          key    = "cyber-range.xorb.io/red-team"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
      
      tags = {
        Zone = "red-team"
        Purpose = "attack-simulation"
        IsolationLevel = "strict"
      }
    }
    
    blue_team = {
      name            = "blue-team"
      use_name_prefix = true
      
      subnet_ids = [module.vpc.private_subnets[2]]
      
      min_size     = 1
      max_size     = 5
      desired_size = 3
      
      ami_type       = "AL2_x86_64"
      instance_types = [var.node_instance_types.blue_team]
      
      k8s_labels = {
        "cyber-range.xorb.io/zone" = "blue-team"
        "cyber-range.xorb.io/role" = "defense"
      }
      
      taints = [
        {
          key    = "cyber-range.xorb.io/blue-team"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
      
      tags = {
        Zone = "blue-team"
        Purpose = "defense-monitoring"
      }
    }
    
    targets = {
      name            = "targets"
      use_name_prefix = true
      
      subnet_ids = module.vpc.private_subnets
      
      min_size     = 2
      max_size     = 10
      desired_size = 5
      
      ami_type       = "AL2_x86_64"
      instance_types = [var.node_instance_types.targets]
      
      k8s_labels = {
        "cyber-range.xorb.io/zone" = "targets"
        "cyber-range.xorb.io/role" = "target-environment"
      }
      
      taints = [
        {
          key    = "cyber-range.xorb.io/targets"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
      
      tags = {
        Zone = "targets"
        Purpose = "vulnerable-applications"
      }
    }
  }
  
  # Enable IRSA for service accounts
  enable_irsa = true
  
  # Cluster security group additional rules
  cluster_security_group_additional_rules = {
    ingress_nodes_ephemeral_ports_tcp = {
      description                = "Node groups to cluster API"
      protocol                   = "tcp"
      from_port                  = 1025
      to_port                    = 65535
      type                       = "ingress"
      source_node_security_group = true
    }
  }
  
  # Node security group additional rules
  node_security_group_additional_rules = {
    ingress_self_all = {
      description = "Node to node all ports/protocols"
      protocol    = "-1"
      from_port   = 0
      to_port     = 0
      type        = "ingress"
      self        = true
    }
    
    ingress_cluster_api_ephemeral = {
      description                   = "Cluster API to node ephemeral ports"
      protocol                      = "tcp"
      from_port                     = 1025
      to_port                       = 65535
      type                          = "ingress"
      source_cluster_security_group = true
    }
    
    # Cyber range specific rules
    ingress_blue_team_monitoring = {
      description = "Blue team monitoring access"
      protocol    = "tcp"
      from_port   = 9100
      to_port     = 9500
      type        = "ingress"
      cidr_blocks = ["10.0.30.0/24"]
    }
  }
  
  tags = {
    Environment = var.environment
    CyberRange = "true"
    KillSwitchEnabled = var.enable_kill_switch
  }
}

# S3 Bucket for cyber range artifacts and logs
resource "aws_s3_bucket" "cyber_range_artifacts" {
  bucket = "${var.cluster_name}-artifacts-${random_id.bucket_suffix.hex}"
  
  tags = {
    Name        = "${var.cluster_name}-artifacts"
    Environment = var.environment
    Purpose     = "cyber-range-logs-and-artifacts"
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket_versioning" "cyber_range_artifacts" {
  bucket = aws_s3_bucket.cyber_range_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "cyber_range_artifacts" {
  bucket = aws_s3_bucket.cyber_range_artifacts.id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "cyber_range_artifacts" {
  bucket = aws_s3_bucket.cyber_range_artifacts.id
  
  rule {
    id     = "cyber_range_retention"
    status = "Enabled"
    
    expiration {
      days = 90
    }
    
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# RDS for persistent data storage
resource "aws_db_subnet_group" "cyber_range" {
  name       = "${var.cluster_name}-db-subnet-group"
  subnet_ids = module.vpc.database_subnets
  
  tags = {
    Name = "${var.cluster_name}-db-subnet-group"
  }
}

resource "aws_security_group" "rds" {
  name_prefix = "${var.cluster_name}-rds-"
  vpc_id      = module.vpc.vpc_id
  description = "Security group for RDS databases"
  
  ingress {
    description = "PostgreSQL"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  ingress {
    description = "MySQL"
    from_port   = 3306
    to_port     = 3306
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "${var.cluster_name}-rds-sg"
  }
}

# PostgreSQL for XORB and Metasploit
resource "aws_db_instance" "cyber_range_postgres" {
  identifier = "${var.cluster_name}-postgres"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.medium"
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type          = "gp3"
  storage_encrypted     = true
  
  db_name  = "cyberrange"
  username = "cyberrange_admin"
  password = random_password.cyber_range_secrets["metasploit_db"].result
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.cyber_range.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = true
  deletion_protection = false
  
  tags = {
    Name        = "${var.cluster_name}-postgres"
    Environment = var.environment
    Purpose     = "cyber-range-data"
  }
}

# ElastiCache Redis for caching and session management
resource "aws_elasticache_subnet_group" "cyber_range" {
  name       = "${var.cluster_name}-cache-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "redis" {
  name_prefix = "${var.cluster_name}-redis-"
  vpc_id      = module.vpc.vpc_id
  description = "Security group for Redis cluster"
  
  ingress {
    description = "Redis"
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  tags = {
    Name = "${var.cluster_name}-redis-sg"
  }
}

resource "aws_elasticache_replication_group" "cyber_range" {
  replication_group_id       = "${var.cluster_name}-redis"
  description                = "Redis cluster for cyber range"
  
  engine               = "redis"
  engine_version       = "7.0"
  node_type            = "cache.t3.medium"
  port                 = 6379
  parameter_group_name = "default.redis7"
  
  num_cache_clusters = 2
  
  subnet_group_name  = aws_elasticache_subnet_group.cyber_range.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = {
    Name        = "${var.cluster_name}-redis"
    Environment = var.environment
  }
}

# IAM Role for Kill Switch Lambda
resource "aws_iam_role" "kill_switch_lambda" {
  count = var.enable_kill_switch ? 1 : 0
  
  name = "${var.cluster_name}-kill-switch-lambda"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "kill_switch_lambda" {
  count = var.enable_kill_switch ? 1 : 0
  
  name = "${var.cluster_name}-kill-switch-policy"
  role = aws_iam_role.kill_switch_lambda[0].id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "eks:UpdateClusterConfig",
          "eks:DescribeCluster",
          "ec2:DescribeSecurityGroups",
          "ec2:AuthorizeSecurityGroupIngress",
          "ec2:RevokeSecurityGroupIngress",
          "ec2:AuthorizeSecurityGroupEgress",
          "ec2:RevokeSecurityGroupEgress"
        ]
        Resource = "*"
      }
    ]
  })
}

# CloudWatch Log Groups for monitoring
resource "aws_cloudwatch_log_group" "cyber_range_logs" {
  for_each = toset([
    "control-plane",
    "red-team", 
    "blue-team",
    "targets",
    "kill-switch"
  ])
  
  name              = "/aws/cyber-range/${var.cluster_name}/${each.key}"
  retention_in_days = 30
  
  tags = {
    Environment = var.environment
    Component   = each.key
  }
}

# Outputs
output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ids attached to the cluster control plane"
  value       = module.eks.cluster_security_group_id
}

output "cluster_iam_role_name" {
  description = "IAM role name associated with EKS cluster"
  value       = module.eks.cluster_iam_role_name
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
}

output "cluster_name" {
  description = "The name of the EKS cluster"
  value       = module.eks.cluster_name
}

output "vpc_id" {
  description = "ID of the VPC where the cluster is deployed"
  value       = module.vpc.vpc_id
}

output "private_subnets" {
  description = "List of IDs of private subnets"
  value       = module.vpc.private_subnets
}

output "postgres_endpoint" {
  description = "RDS PostgreSQL endpoint"
  value       = aws_db_instance.cyber_range_postgres.endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = aws_elasticache_replication_group.cyber_range.configuration_endpoint_address
  sensitive   = true
}

output "s3_artifacts_bucket" {
  description = "S3 bucket for cyber range artifacts"
  value       = aws_s3_bucket.cyber_range_artifacts.bucket
}

output "cyber_range_passwords" {
  description = "Generated passwords for cyber range services"
  value = {
    for service, password in random_password.cyber_range_secrets :
    service => password.result
  }
  sensitive = true
}

# Local file outputs for kubectl configuration
resource "local_file" "kubeconfig" {
  content = templatefile("${path.module}/templates/kubeconfig.tpl", {
    cluster_name                         = module.eks.cluster_name
    cluster_endpoint                     = module.eks.cluster_endpoint
    cluster_certificate_authority_data  = module.eks.cluster_certificate_authority_data
    region                              = var.region
  })
  filename = "${path.module}/kubeconfig_${var.cluster_name}"
  
  depends_on = [module.eks]
}

resource "local_file" "cyber_range_config" {
  content = templatefile("${path.module}/templates/cyber-range-config.yaml.tpl", {
    cluster_name    = var.cluster_name
    postgres_host   = aws_db_instance.cyber_range_postgres.address
    postgres_db     = aws_db_instance.cyber_range_postgres.db_name
    postgres_user   = aws_db_instance.cyber_range_postgres.username
    redis_endpoint  = aws_elasticache_replication_group.cyber_range.configuration_endpoint_address
    s3_bucket       = aws_s3_bucket.cyber_range_artifacts.bucket
    vpc_id          = module.vpc.vpc_id
    region          = var.region
    passwords       = random_password.cyber_range_secrets
  })
  filename = "${path.module}/cyber-range-config.yaml"
  
  depends_on = [
    aws_db_instance.cyber_range_postgres,
    aws_elasticache_replication_group.cyber_range
  ]
}