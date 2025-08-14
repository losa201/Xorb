# NATS Accounts Variables

variable "tenant_id" {
  description = "The tenant identifier"
  type        = string
  default     = "demo"
}

variable "scanner_password" {
  description = "Password for the scanner user"
  type        = string
  sensitive   = true
}

variable "max_connections" {
  description = "Maximum connections for the account"
  type        = number
  default     = 100
}

variable "max_subscriptions" {
  description = "Maximum subscriptions for the account"
  type        = number
  default     = 1000
}

variable "max_payload" {
  description = "Maximum payload size in bytes"
  type        = number
  default     = 1048576  # 1MB
}

variable "max_memory" {
  description = "Maximum memory for JetStream in bytes"
  type        = number
  default     = 1073741824  # 1GB
}

variable "max_storage" {
  description = "Maximum storage for JetStream in bytes"
  type        = number
  default     = 10737418240  # 10GB
}

variable "rate_limit" {
  description = "Rate limit in messages per second"
  type        = number
  default     = 1000
}

# Starter tier variables
variable "starter_max_connections" {
  description = "Maximum connections for Starter tier"
  type        = number
  default     = 10
}

variable "starter_max_memory" {
  description = "Maximum memory for Starter tier"
  type        = number
  default     = 104857600  # 100MB
}

variable "starter_max_storage" {
  description = "Maximum storage for Starter tier"
  type        = number
  default     = 1073741824  # 1GB
}

# Pro tier variables
variable "pro_max_connections" {
  description = "Maximum connections for Pro tier"
  type        = number
  default     = 100
}

variable "pro_max_memory" {
  description = "Maximum memory for Pro tier"
  type        = number
  default     = 1073741824  # 1GB
}

variable "pro_max_storage" {
  description = "Maximum storage for Pro tier"
  type        = number
  default     = 10737418240  # 10GB
}

# Enterprise tier variables
variable "enterprise_max_connections" {
  description = "Maximum connections for Enterprise tier"
  type        = number
  default     = 1000
}

variable "enterprise_max_memory" {
  description = "Maximum memory for Enterprise tier"
  type        = number
  default     = 10737418240  # 10GB
}

variable "enterprise_max_storage" {
  description = "Maximum storage for Enterprise tier"
  type        = number
  default     = 107374182400  # 100GB
}