# Example Tenant Quotas per Tier

# Starter Tier
starter_max_connections = 10
starter_max_memory      = 104857600    # 100MB
starter_max_storage     = 1073741824   # 1GB

# Pro Tier
pro_max_connections     = 100
pro_max_memory          = 1073741824   # 1GB
pro_max_storage         = 10737418240  # 10GB

# Enterprise Tier
enterprise_max_connections = 1000
enterprise_max_memory      = 10737418240   # 10GB
enterprise_max_storage     = 107374182400  # 100GB

# Default values for a demo tenant
tenant_id               = "demo-tenant"
scanner_password        = "secure-password-123"
max_connections         = 50
max_subscriptions       = 1000
max_payload             = 1048576       # 1MB
max_memory              = 536870912     # 512MB
max_storage             = 5368709120    # 5GB
rate_limit              = 1000          # messages per second
