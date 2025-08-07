# XORB Infrastructure as Code - Main Configuration
# Multi-cloud Terraform configuration for XORB platform deployment

terraform {
  required_version = ">= 1.5"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
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

  # Backend configuration for state management
  backend "s3" {
    bucket         = var.terraform_state_bucket
    key            = "xorb/terraform.tfstate"
    region         = var.aws_region
    encrypt        = true
    dynamodb_table = var.terraform_lock_table
  }
}

# Provider configurations
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "XORB"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Owner       = var.project_owner
    }
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  token                  = data.aws_eks_cluster_auth.cluster.token
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    token                  = data.aws_eks_cluster_auth.cluster.token
  }
}

# Data sources
data "aws_eks_cluster_auth" "cluster" {
  name = module.eks.cluster_name
}

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Local values
locals {
  cluster_name = "${var.project_name}-${var.environment}"
  
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
    Owner       = var.project_owner
  }

  # Network configuration
  vpc_cidr = var.vpc_cidr
  azs      = slice(data.aws_availability_zones.available.names, 0, var.az_count)
  
  # Subnets
  private_subnet_cidrs = [for i in range(var.az_count) : cidrsubnet(local.vpc_cidr, 8, i)]
  public_subnet_cidrs  = [for i in range(var.az_count) : cidrsubnet(local.vpc_cidr, 8, i + 100)]
  database_subnet_cidrs = [for i in range(var.az_count) : cidrsubnet(local.vpc_cidr, 8, i + 200)]
}

# Random password generation
resource "random_password" "db_passwords" {
  for_each = toset(["postgres", "redis"])
  
  length  = 32
  special = true
}

# KMS key for encryption
resource "aws_kms_key" "xorb" {
  description             = "XORB encryption key"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-kms-key"
  })
}

resource "aws_kms_alias" "xorb" {
  name          = "alias/${local.cluster_name}"
  target_key_id = aws_kms_key.xorb.key_id
}

# VPC and Networking
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${local.cluster_name}-vpc"
  cidr = local.vpc_cidr

  azs              = local.azs
  private_subnets  = local.private_subnet_cidrs
  public_subnets   = local.public_subnet_cidrs
  database_subnets = local.database_subnet_cidrs

  enable_nat_gateway   = true
  enable_vpn_gateway   = var.enable_vpn_gateway
  enable_dns_hostnames = true
  enable_dns_support   = true

  # EKS specific configurations
  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
  }

  tags = local.common_tags
}

# Security Groups
resource "aws_security_group" "additional_sg" {
  name        = "${local.cluster_name}-additional-sg"
  description = "Additional security group for XORB services"
  vpc_id      = module.vpc.vpc_id

  # Adversarial engine ports
  ingress {
    from_port   = 8001
    to_port     = 8002
    protocol    = "tcp"
    cidr_blocks = [local.vpc_cidr]
    description = "Adversarial engine ports"
  }

  # ML defense ports
  ingress {
    from_port   = 8003
    to_port     = 8004
    protocol    = "tcp"
    cidr_blocks = [local.vpc_cidr]
    description = "ML defense engine ports"
  }

  # Monitoring ports
  ingress {
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = [local.vpc_cidr]
    description = "Prometheus"
  }

  ingress {
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = [local.vpc_cidr]
    description = "Grafana"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-additional-sg"
  })
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = local.cluster_name
  cluster_version = var.kubernetes_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Cluster endpoint configuration
  cluster_endpoint_public_access  = var.cluster_endpoint_public_access
  cluster_endpoint_private_access = true

  # Encryption
  cluster_encryption_config = {
    provider_key_arn = aws_kms_key.xorb.arn
    resources        = ["secrets"]
  }

  # Cluster addons
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
  }

  # Managed node groups
  eks_managed_node_groups = {
    # General purpose nodes
    general = {
      name = "${local.cluster_name}-general"
      
      instance_types = var.general_node_instance_types
      capacity_type  = "ON_DEMAND"
      
      min_size     = var.general_node_min_size
      max_size     = var.general_node_max_size
      desired_size = var.general_node_desired_size

      labels = {
        role = "general"
      }

      taints = []

      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "general"
      }

      tags = {
        Name = "${local.cluster_name}-general-node"
      }
    }

    # High-compute nodes for ML workloads
    ml_compute = {
      name = "${local.cluster_name}-ml-compute"
      
      instance_types = var.ml_node_instance_types
      capacity_type  = var.ml_node_capacity_type
      
      min_size     = var.ml_node_min_size
      max_size     = var.ml_node_max_size
      desired_size = var.ml_node_desired_size

      labels = {
        role = "ml-compute"
        "node.kubernetes.io/instance-type" = "cpu"
      }

      taints = [
        {
          key    = "ml-workload"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]

      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "ml-compute"
        Workload    = "ml"
      }

      tags = {
        Name = "${local.cluster_name}-ml-compute-node"
      }
    }

    # Monitoring and observability nodes
    monitoring = {
      name = "${local.cluster_name}-monitoring"
      
      instance_types = var.monitoring_node_instance_types
      capacity_type  = "ON_DEMAND"
      
      min_size     = var.monitoring_node_min_size
      max_size     = var.monitoring_node_max_size
      desired_size = var.monitoring_node_desired_size

      labels = {
        role = "monitoring"
      }

      taints = [
        {
          key    = "monitoring"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]

      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "monitoring"
        Workload    = "monitoring"
      }

      tags = {
        Name = "${local.cluster_name}-monitoring-node"
      }
    }
  }

  # Additional security groups
  node_security_group_additional_rules = {
    ingress_cluster_api = {
      description = "Cluster API to node groups"
      protocol    = "tcp"
      from_port   = 443
      to_port     = 443
      type        = "ingress"
      source_cluster_security_group = true
    }
  }

  tags = local.common_tags
}

# RDS PostgreSQL for persistent data
module "rds" {
  source = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"

  identifier = "${local.cluster_name}-postgres"

  engine               = "postgres"
  engine_version       = var.postgres_version
  family               = "postgres15"
  major_engine_version = "15"
  instance_class       = var.rds_instance_class

  allocated_storage     = var.rds_allocated_storage
  max_allocated_storage = var.rds_max_allocated_storage
  storage_encrypted     = true
  kms_key_id           = aws_kms_key.xorb.arn

  db_name  = var.database_name
  username = var.database_username
  password = random_password.db_passwords["postgres"].result
  port     = 5432

  vpc_security_group_ids = [module.vpc.database_security_group_id]
  db_subnet_group_name   = module.vpc.database_subnet_group

  maintenance_window      = var.rds_maintenance_window
  backup_window          = var.rds_backup_window
  backup_retention_period = var.rds_backup_retention_period

  # Enhanced monitoring
  monitoring_interval    = 60
  monitoring_role_name   = "${local.cluster_name}-rds-monitoring-role"
  create_monitoring_role = true

  # Performance insights
  performance_insights_enabled = var.rds_performance_insights_enabled
  performance_insights_retention_period = 7

  # Deletion protection
  deletion_protection = var.environment == "production"

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-postgres"
  })
}

# ElastiCache Redis for caching and sessions
module "redis" {
  source = "terraform-aws-modules/elasticache/aws"
  version = "~> 1.0"

  cluster_id         = "${local.cluster_name}-redis"
  description        = "XORB Redis cluster"

  node_type          = var.redis_node_type
  num_cache_nodes    = var.redis_num_nodes
  parameter_group_name = "default.redis7"
  engine_version     = var.redis_version
  port               = 6379

  subnet_ids         = module.vpc.database_subnets
  security_group_ids = [module.vpc.database_security_group_id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = random_password.db_passwords["redis"].result

  maintenance_window = var.redis_maintenance_window
  snapshot_window    = var.redis_snapshot_window
  snapshot_retention_limit = var.redis_snapshot_retention_limit

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-redis"
  })
}

# Application Load Balancer
module "alb" {
  source = "terraform-aws-modules/alb/aws"
  version = "~> 8.0"

  name = "${local.cluster_name}-alb"

  load_balancer_type = "application"
  vpc_id             = module.vpc.vpc_id
  subnets            = module.vpc.public_subnets
  security_groups    = [module.vpc.default_security_group_id, aws_security_group.additional_sg.id]

  # Access logs
  access_logs = {
    bucket  = aws_s3_bucket.alb_logs.id
    enabled = true
  }

  # Target groups and listeners will be managed by Kubernetes ingress controller

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-alb"
  })
}

# S3 buckets
resource "aws_s3_bucket" "alb_logs" {
  bucket = "${local.cluster_name}-alb-logs-${random_id.bucket_suffix.hex}"

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-alb-logs"
  })
}

resource "aws_s3_bucket" "xorb_data" {
  bucket = "${local.cluster_name}-data-${random_id.bucket_suffix.hex}"

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-data"
  })
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# S3 bucket configurations
resource "aws_s3_bucket_versioning" "xorb_data" {
  bucket = aws_s3_bucket.xorb_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "xorb_data" {
  bucket = aws_s3_bucket.xorb_data.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        kms_master_key_id = aws_kms_key.xorb.arn
        sse_algorithm     = "aws:kms"
      }
    }
  }
}

resource "aws_s3_bucket_public_access_block" "xorb_data" {
  bucket = aws_s3_bucket.xorb_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# IAM roles and policies
resource "aws_iam_role" "xorb_service_role" {
  name = "${local.cluster_name}-service-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      },
    ]
  })

  tags = local.common_tags
}

# Secrets Manager for sensitive configuration
resource "aws_secretsmanager_secret" "xorb_secrets" {
  name        = "${local.cluster_name}-secrets"
  description = "XORB application secrets"
  kms_key_id  = aws_kms_key.xorb.arn

  tags = local.common_tags
}

resource "aws_secretsmanager_secret_version" "xorb_secrets" {
  secret_id = aws_secretsmanager_secret.xorb_secrets.id
  secret_string = jsonencode({
    postgres_password = random_password.db_passwords["postgres"].result
    redis_password    = random_password.db_passwords["redis"].result
    jwt_secret        = random_password.jwt_secret.result
  })
}

resource "random_password" "jwt_secret" {
  length  = 64
  special = true
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "xorb_logs" {
  for_each = toset([
    "adversarial-engine",
    "ml-defense",
    "api-gateway",
    "orchestrator"
  ])

  name              = "/aws/eks/${local.cluster_name}/${each.key}"
  retention_in_days = var.cloudwatch_log_retention_days
  kms_key_id       = aws_kms_key.xorb.arn

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-${each.key}-logs"
  })
}