#  Configuration Reference

This comprehensive guide covers all configuration options for the XORB Red/Blue Agent Framework, from basic setup to advanced enterprise deployments.

##  📋 Configuration Overview

The framework uses a layered configuration approach:

1. **Environment Variables** - Runtime configuration and secrets
2. **Configuration Files** - Structured configuration data
3. **Capability Manifests** - Technique definitions and policies
4. **Docker Compose** - Service orchestration and networking
5. **Kubernetes Manifests** - Production deployment configuration

##  🌐 Environment Variables

###  Core Application Settings

```bash
#  Application Environment
XORB_ENV=production                    # Environment: development, staging, production
DEBUG=false                           # Enable debug logging and features
LOG_LEVEL=INFO                        # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json                       # Log format: json, text
SERVICE_NAME=xorb-red-blue-agents     # Service identifier for logging and metrics

#  API Configuration
API_HOST=0.0.0.0                      # API server bind address
API_PORT=8000                         # API server port
API_WORKERS=4                         # Number of worker processes
API_TIMEOUT=30                        # Request timeout in seconds
CORS_ALLOW_ORIGINS=*                  # CORS allowed origins (comma-separated)
CORS_ALLOW_METHODS=GET,POST,PUT,DELETE,OPTIONS  # CORS allowed methods
CORS_ALLOW_HEADERS=*                  # CORS allowed headers
```

###  Database Configuration

```bash
#  PostgreSQL Database
DATABASE_URL=postgresql://user:pass@host:5432/dbname  # Primary database connection
DATABASE_POOL_SIZE=20                 # Connection pool size
DATABASE_MAX_OVERFLOW=30              # Maximum pool overflow
DATABASE_POOL_TIMEOUT=30              # Pool connection timeout
DATABASE_POOL_RECYCLE=3600           # Connection recycle time
DATABASE_SSL_MODE=prefer             # SSL mode: disable, allow, prefer, require
DATABASE_SCHEMA=public               # Default schema

#  Database Migrations
ALEMBIC_CONFIG=alembic.ini           # Alembic configuration file
AUTO_MIGRATE=true                    # Auto-run migrations on startup
MIGRATION_TIMEOUT=300                # Migration timeout in seconds

#  Read Replicas (Optional)
DATABASE_READ_URL=postgresql://readonly:pass@read-host:5432/dbname
READ_REPLICA_RATIO=0.3               # Percentage of reads to route to replica
```

###  Redis Configuration

```bash
#  Redis Cache and Session Storage
REDIS_URL=redis://redis:6379/0       # Primary Redis connection
REDIS_PASSWORD=secure_password       # Redis password (if required)
REDIS_SSL=false                      # Use SSL for Redis connection
REDIS_TIMEOUT=5                      # Connection timeout
REDIS_RETRY_ATTEMPTS=3               # Retry attempts for failed operations

#  Redis Cluster (Optional)
REDIS_CLUSTER_ENABLED=false          # Enable Redis cluster mode
REDIS_CLUSTER_NODES=redis1:6379,redis2:6379,redis3:6379  # Cluster node list
REDIS_CLUSTER_SKIP_FULL_COVERAGE=false  # Skip full coverage check

#  Cache Configuration
CACHE_TTL_DEFAULT=3600               # Default cache TTL in seconds
CACHE_TTL_TECHNIQUES=7200            # Technique cache TTL
CACHE_TTL_SESSIONS=1800              # Session cache TTL
CACHE_PREFIX=xorb_agents             # Cache key prefix
```

###  Temporal Workflow Engine

```bash
#  Temporal Configuration
TEMPORAL_HOST=temporal:7233           # Temporal server address
TEMPORAL_NAMESPACE=xorb-agents        # Temporal namespace
TEMPORAL_TASK_QUEUE=agent-tasks      # Default task queue name
TEMPORAL_CLIENT_TIMEOUT=30           # Client connection timeout
TEMPORAL_WORKFLOW_TIMEOUT=86400      # Default workflow timeout (24 hours)
TEMPORAL_ACTIVITY_TIMEOUT=3600       # Default activity timeout (1 hour)

#  Temporal TLS (Optional)
TEMPORAL_TLS_ENABLED=false           # Enable TLS for Temporal connection
TEMPORAL_TLS_CERT_PATH=/certs/client.crt    # Client certificate path
TEMPORAL_TLS_KEY_PATH=/certs/client.key     # Client private key path
TEMPORAL_TLS_CA_PATH=/certs/ca.crt          # CA certificate path
```

###  Security Configuration

```bash
#  Authentication and Authorization
JWT_SECRET=your-super-secure-jwt-secret-key-here  # JWT signing secret
JWT_ALGORITHM=HS256                   # JWT algorithm
JWT_EXPIRATION=3600                   # JWT expiration time in seconds
JWT_REFRESH_EXPIRATION=86400          # Refresh token expiration
API_KEY_HEADER=X-API-Key             # API key header name

#  Encryption
ENCRYPTION_KEY=your-32-character-encryption-key!!  # Data encryption key
ENCRYPTION_ALGORITHM=AES-256-GCM      # Encryption algorithm
HASH_ALGORITHM=bcrypt                # Password hashing algorithm
BCRYPT_ROUNDS=12                     # Bcrypt rounds for password hashing

#  Rate Limiting
RATE_LIMIT_ENABLED=true              # Enable rate limiting
RATE_LIMIT_PER_MINUTE=60             # Requests per minute per user
RATE_LIMIT_PER_HOUR=1000             # Requests per hour per user
RATE_LIMIT_PER_DAY=10000             # Requests per day per user
RATE_LIMIT_REDIS_KEY_PREFIX=rl:      # Redis key prefix for rate limiting

#  Session Security
SESSION_SECURE=true                  # Secure session cookies
SESSION_HTTP_ONLY=true               # HTTP-only session cookies
SESSION_SAME_SITE=strict             # SameSite cookie attribute
SESSION_TIMEOUT=1800                 # Session timeout in seconds
```

###  Sandbox Configuration

```bash
#  Docker Configuration
DOCKER_HOST=unix:///var/run/docker.sock  # Docker daemon socket
DOCKER_API_VERSION=auto              # Docker API version
DOCKER_TIMEOUT=60                    # Docker operation timeout
DOCKER_TLS_VERIFY=false              # Verify Docker daemon TLS certificate

#  Kata Containers (Optional)
KATA_RUNTIME_ENABLED=false           # Enable Kata containers
KATA_RUNTIME_PATH=/usr/bin/kata-runtime  # Kata runtime binary path
KATA_CONFIG_PATH=/etc/kata-containers/config.toml  # Kata configuration

#  Sandbox Limits
MAX_SANDBOXES_GLOBAL=100             # Maximum sandboxes globally
MAX_SANDBOXES_PER_MISSION=10         # Maximum sandboxes per mission
MAX_SANDBOXES_PER_USER=5             # Maximum sandboxes per user
DEFAULT_SANDBOX_TTL=3600             # Default sandbox TTL in seconds
MAX_SANDBOX_TTL=86400                # Maximum allowed sandbox TTL
SANDBOX_CLEANUP_INTERVAL=300         # Cleanup check interval in seconds

#  Resource Defaults
DEFAULT_CPU_CORES=1.0                # Default CPU allocation
DEFAULT_MEMORY_MB=512                # Default memory allocation
DEFAULT_DISK_MB=1024                 # Default disk allocation
DEFAULT_NETWORK_MB=100               # Default network bandwidth

#  Network Configuration
SANDBOX_NETWORK_DRIVER=bridge        # Default network driver
SANDBOX_NETWORK_SUBNET=172.20.0.0/16  # Sandbox network subnet
SANDBOX_DNS_SERVERS=8.8.8.8,8.8.4.4  # DNS servers for sandboxes
SANDBOX_ISOLATION_ENABLED=true       # Enable network isolation
```

###  Machine Learning and Analytics

```bash
#  Learning Engine
ML_ENABLED=true                      # Enable machine learning features
ML_MODEL_PATH=/app/models            # Path to ML models
ML_UPDATE_INTERVAL=3600              # Model update interval in seconds
ML_TRAINING_ENABLED=true             # Enable model training
ML_BATCH_SIZE=32                     # Training batch size
ML_LEARNING_RATE=0.001               # Learning rate for training

#  Feature Engineering
FEATURE_EXTRACTION_ENABLED=true     # Enable feature extraction
FEATURE_WINDOW_SIZE=100              # Feature window size
FEATURE_UPDATE_INTERVAL=300          # Feature update interval

#  Model Storage
MODEL_STORAGE_BACKEND=filesystem     # Storage backend: filesystem, s3, gcs
MODEL_VERSIONING_ENABLED=true       # Enable model versioning
MODEL_RETENTION_DAYS=30              # Model retention period

#  Analytics
ANALYTICS_ENABLED=true               # Enable analytics collection
ANALYTICS_BATCH_SIZE=1000            # Analytics batch size
ANALYTICS_FLUSH_INTERVAL=60          # Analytics flush interval
TELEMETRY_SAMPLING_RATE=1.0          # Telemetry sampling rate (0.0-1.0)
```

###  Monitoring and Observability

```bash
#  Prometheus Metrics
PROMETHEUS_ENABLED=true              # Enable Prometheus metrics
PROMETHEUS_PORT=9090                 # Prometheus metrics port
PROMETHEUS_METRICS_PATH=/metrics     # Metrics endpoint path
PROMETHEUS_NAMESPACE=xorb_agents     # Metrics namespace
PROMETHEUS_PUSH_GATEWAY=http://pushgateway:9091  # Push gateway URL

#  Health Checks
HEALTH_CHECK_ENABLED=true           # Enable health check endpoints
HEALTH_CHECK_INTERVAL=30            # Health check interval
HEALTH_CHECK_TIMEOUT=10             # Health check timeout
HEALTH_CHECK_RETRIES=3              # Health check retry attempts

#  Logging
LOG_TO_FILE=false                   # Log to file in addition to stdout
LOG_FILE_PATH=/var/log/xorb/agents.log  # Log file path
LOG_FILE_MAX_SIZE=100MB             # Maximum log file size
LOG_FILE_BACKUP_COUNT=5             # Number of backup log files
LOG_STRUCTURED=true                 # Use structured logging

#  Distributed Tracing
TRACING_ENABLED=false               # Enable distributed tracing
TRACING_BACKEND=jaeger              # Tracing backend: jaeger, zipkin
TRACING_ENDPOINT=http://jaeger:14268/api/traces  # Tracing endpoint
TRACING_SAMPLE_RATE=0.1             # Trace sampling rate
```

###  External Integrations

```bash
#  Notification Services
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
SLACK_CHANNEL=#security-ops          # Default Slack channel
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR/WEBHOOK
TEAMS_WEBHOOK_URL=https://outlook.office.com/webhook/YOUR/TEAMS/WEBHOOK

#  Email Configuration
EMAIL_ENABLED=true                  # Enable email notifications
EMAIL_SMTP_HOST=smtp.your-domain.com  # SMTP server host
EMAIL_SMTP_PORT=587                 # SMTP server port
EMAIL_SMTP_USER=alerts@your-domain.com  # SMTP username
EMAIL_SMTP_PASSWORD=your-smtp-password  # SMTP password
EMAIL_SMTP_TLS=true                 # Enable SMTP TLS
EMAIL_FROM_ADDRESS=noreply@your-domain.com  # From email address
EMAIL_FROM_NAME=XORB Security Platform     # From name

#  Webhook Configuration
WEBHOOK_TIMEOUT=30                  # Webhook timeout in seconds
WEBHOOK_RETRY_ATTEMPTS=3            # Webhook retry attempts
WEBHOOK_RETRY_DELAY=5               # Webhook retry delay in seconds

#  External APIs
VIRUSTOTAL_API_KEY=your-vt-api-key  # VirusTotal API key
SHODAN_API_KEY=your-shodan-api-key  # Shodan API key
THREAT_INTEL_API_KEY=your-ti-api-key  # Threat intelligence API key
THREAT_INTEL_API_URL=https://api.threatintel.com  # Threat intelligence API URL
```

###  Compliance and Audit

```bash
#  Audit Logging
AUDIT_LOG_ENABLED=true              # Enable audit logging
AUDIT_LOG_LEVEL=INFO                # Audit log level
AUDIT_LOG_FILE=/var/log/xorb/audit.log  # Audit log file path
AUDIT_LOG_RETENTION_DAYS=365        # Audit log retention period
AUDIT_LOG_ENCRYPTION=true           # Encrypt audit logs

#  Compliance
SOC2_COMPLIANCE=true                # Enable SOC2 compliance features
HIPAA_COMPLIANCE=false              # Enable HIPAA compliance features
GDPR_COMPLIANCE=true                # Enable GDPR compliance features
PCI_COMPLIANCE=false                # Enable PCI compliance features

#  Data Retention
DATA_RETENTION_ENABLED=true         # Enable data retention policies
MISSION_DATA_RETENTION_DAYS=90      # Mission data retention
TELEMETRY_DATA_RETENTION_DAYS=30    # Telemetry data retention
LOG_DATA_RETENTION_DAYS=365         # Log data retention
BACKUP_RETENTION_DAYS=90            # Backup retention period

#  Privacy
ANONYMIZE_PII=true                  # Anonymize personally identifiable information
DATA_MINIMIZATION=true              # Enable data minimization practices
CONSENT_REQUIRED=false              # Require explicit consent for data collection
```

##  📁 Configuration Files

###  Main Configuration (`config/app.yaml`)

```yaml
#  Main application configuration
application:
  name: "XORB Red/Blue Agent Framework"
  version: "2.1.0"
  environment: "${XORB_ENV:-development}"
  debug: "${DEBUG:-false}"

#  Database configuration
database:
  url: "${DATABASE_URL}"
  pool_size: "${DATABASE_POOL_SIZE:-20}"
  max_overflow: "${DATABASE_MAX_OVERFLOW:-30}"
  echo: "${DATABASE_ECHO:-false}"

#  Redis configuration
redis:
  url: "${REDIS_URL}"
  timeout: "${REDIS_TIMEOUT:-5}"
  retry_attempts: "${REDIS_RETRY_ATTEMPTS:-3}"

#  Temporal configuration
temporal:
  host: "${TEMPORAL_HOST:-temporal:7233}"
  namespace: "${TEMPORAL_NAMESPACE:-xorb-agents}"
  task_queue: "${TEMPORAL_TASK_QUEUE:-agent-tasks}"

#  Security configuration
security:
  jwt:
    secret: "${JWT_SECRET}"
    algorithm: "${JWT_ALGORITHM:-HS256}"
    expiration: "${JWT_EXPIRATION:-3600}"
  encryption:
    key: "${ENCRYPTION_KEY}"
    algorithm: "${ENCRYPTION_ALGORITHM:-AES-256-GCM}"
  rate_limiting:
    enabled: "${RATE_LIMIT_ENABLED:-true}"
    per_minute: "${RATE_LIMIT_PER_MINUTE:-60}"
    per_hour: "${RATE_LIMIT_PER_HOUR:-1000}"

#  Sandbox configuration
sandbox:
  docker:
    host: "${DOCKER_HOST:-unix:///var/run/docker.sock}"
    timeout: "${DOCKER_TIMEOUT:-60}"
  limits:
    max_global: "${MAX_SANDBOXES_GLOBAL:-100}"
    max_per_mission: "${MAX_SANDBOXES_PER_MISSION:-10}"
    default_ttl: "${DEFAULT_SANDBOX_TTL:-3600}"
  resources:
    default_cpu: "${DEFAULT_CPU_CORES:-1.0}"
    default_memory: "${DEFAULT_MEMORY_MB:-512}"
    default_disk: "${DEFAULT_DISK_MB:-1024}"

#  Machine learning configuration
machine_learning:
  enabled: "${ML_ENABLED:-true}"
  model_path: "${ML_MODEL_PATH:-/app/models}"
  update_interval: "${ML_UPDATE_INTERVAL:-3600}"
  training_enabled: "${ML_TRAINING_ENABLED:-true}"

#  Monitoring configuration
monitoring:
  prometheus:
    enabled: "${PROMETHEUS_ENABLED:-true}"
    port: "${PROMETHEUS_PORT:-9090}"
    namespace: "${PROMETHEUS_NAMESPACE:-xorb_agents}"
  health_checks:
    enabled: "${HEALTH_CHECK_ENABLED:-true}"
    interval: "${HEALTH_CHECK_INTERVAL:-30}"
    timeout: "${HEALTH_CHECK_TIMEOUT:-10}"
  logging:
    level: "${LOG_LEVEL:-INFO}"
    format: "${LOG_FORMAT:-json}"
    structured: "${LOG_STRUCTURED:-true}"

#  External integrations
integrations:
  notifications:
    slack:
      webhook_url: "${SLACK_WEBHOOK_URL}"
      channel: "${SLACK_CHANNEL:-#security-ops}"
    email:
      enabled: "${EMAIL_ENABLED:-true}"
      smtp_host: "${EMAIL_SMTP_HOST}"
      smtp_port: "${EMAIL_SMTP_PORT:-587}"
      from_address: "${EMAIL_FROM_ADDRESS}"
  threat_intelligence:
    api_key: "${THREAT_INTEL_API_KEY}"
    api_url: "${THREAT_INTEL_API_URL}"

#  Compliance and audit
compliance:
  audit_logging:
    enabled: "${AUDIT_LOG_ENABLED:-true}"
    file_path: "${AUDIT_LOG_FILE:-/var/log/xorb/audit.log}"
    retention_days: "${AUDIT_LOG_RETENTION_DAYS:-365}"
  data_retention:
    enabled: "${DATA_RETENTION_ENABLED:-true}"
    mission_data_days: "${MISSION_DATA_RETENTION_DAYS:-90}"
    telemetry_data_days: "${TELEMETRY_DATA_RETENTION_DAYS:-30}"
  privacy:
    anonymize_pii: "${ANONYMIZE_PII:-true}"
    data_minimization: "${DATA_MINIMIZATION:-true}"
```

###  Logging Configuration (`config/logging.yaml`)

```yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt: "%Y-%m-%d %H:%M:%S"

  json:
    class: "pythonjsonlogger.jsonlogger.JsonFormatter"
    format: "%(asctime)s %(name)s %(levelname)s %(message)s"

  detailed:
    format: "%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d] %(funcName)s(): %(message)s"

handlers:
  console:
    class: logging.StreamHandler
    level: "${LOG_LEVEL:-INFO}"
    formatter: "${LOG_FORMAT:-json}"
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: "${LOG_LEVEL:-INFO}"
    formatter: json
    filename: "${LOG_FILE_PATH:-/var/log/xorb/agents.log}"
    maxBytes: 104857600  # 100MB
    backupCount: 5
    encoding: utf8

  audit:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: json
    filename: "${AUDIT_LOG_FILE:-/var/log/xorb/audit.log}"
    maxBytes: 104857600  # 100MB
    backupCount: 10
    encoding: utf8

loggers:
  xorb:
    level: "${LOG_LEVEL:-INFO}"
    handlers: [console, file]
    propagate: false

  xorb.audit:
    level: INFO
    handlers: [audit]
    propagate: false

  xorb.security:
    level: WARNING
    handlers: [console, file, audit]
    propagate: false

  uvicorn:
    level: INFO
    handlers: [console]
    propagate: false

  sqlalchemy:
    level: WARNING
    handlers: [console]
    propagate: false

root:
  level: WARNING
  handlers: [console]
```

###  Agent Configuration (`config/agents.yaml`)

```yaml
#  Agent type definitions and configurations
agents:
  red_recon:
    name: "Reconnaissance Agent"
    description: "Information gathering and target discovery"
    image: "xorb/red-recon:latest"
    categories:
      - reconnaissance
      - discovery
    resource_requirements:
      cpu_cores: 1.0
      memory_mb: 512
      disk_mb: 1024
      network_mb: 50
    max_concurrent_missions: 3
    default_timeout: 1800
    techniques:
      - recon.port_scan
      - recon.service_enum
      - recon.web_crawl
      - recon.dns_enum
      - recon.subdomain_enum
      - recon.network_discovery
      - recon.os_fingerprint
      - recon.vulnerability_scan

  red_exploit:
    name: "Exploitation Agent"
    description: "Initial access and privilege escalation"
    image: "xorb/red-exploit:latest"
    categories:
      - initial_access
      - privilege_escalation
      - credential_access
    resource_requirements:
      cpu_cores: 2.0
      memory_mb: 1024
      disk_mb: 2048
      network_mb: 100
    max_concurrent_missions: 2
    default_timeout: 3600
    techniques:
      - exploit.web_sqli
      - exploit.brute_force
      - exploit.credential_stuffing
      - exploit.buffer_overflow
      - exploit.privilege_escalation

  red_persistence:
    name: "Persistence Agent"
    description: "Maintaining access and establishing persistence"
    image: "xorb/red-persistence:latest"
    categories:
      - persistence
    resource_requirements:
      cpu_cores: 1.0
      memory_mb: 512
      disk_mb: 1024
      network_mb: 25
    max_concurrent_missions: 5
    default_timeout: 1800
    techniques:
      - persist.web_shell
      - persist.scheduled_task
      - persist.service_creation
      - persist.user_account
      - persist.startup_script
      - persist.registry_autorun
      - persist.ssh_keys

  red_evasion:
    name: "Evasion Agent"
    description: "Defense evasion and stealth techniques"
    image: "xorb/red-evasion:latest"
    categories:
      - defense_evasion
    resource_requirements:
      cpu_cores: 1.5
      memory_mb: 768
      disk_mb: 1536
      network_mb: 75
    max_concurrent_missions: 3
    default_timeout: 2400
    techniques:
      - evasion.process_hollowing
      - evasion.domain_fronting
      - evasion.log_clearing
      - evasion.timestomp
      - evasion.file_masquerading
      - evasion.traffic_obfuscation
      - evasion.av_bypass

  red_collection:
    name: "Collection Agent"
    description: "Data collection and credential harvesting"
    image: "xorb/red-collection:latest"
    categories:
      - collection
      - credential_access
    resource_requirements:
      cpu_cores: 1.0
      memory_mb: 512
      disk_mb: 2048
      network_mb: 50
    max_concurrent_missions: 4
    default_timeout: 2400
    techniques:
      - collection.keylogging
      - collection.screen_capture
      - collection.credential_dumping
      - collection.file_search
      - collection.browser_data
      - collection.network_shares
      - collection.clipboard

  blue_detect:
    name: "Detection Agent"
    description: "Threat detection and monitoring"
    image: "xorb/blue-detect:latest"
    categories:
      - detection
    resource_requirements:
      cpu_cores: 2.0
      memory_mb: 2048
      disk_mb: 1024
      network_mb: 100
    max_concurrent_missions: 10
    default_timeout: -1  # Always running
    techniques:
      - detect.network_anomaly
      - detect.process_monitoring
      - detect.log_analysis
      - detect.behavioral_analysis
      - detect.file_integrity
      - detect.memory_analysis

  blue_hunt:
    name: "Threat Hunting Agent"
    description: "Proactive threat hunting"
    image: "xorb/blue-hunt:latest"
    categories:
      - threat_hunting
    resource_requirements:
      cpu_cores: 2.0
      memory_mb: 2048
      disk_mb: 2048
      network_mb: 100
    max_concurrent_missions: 5
    default_timeout: 7200
    techniques:
      - hunt.lateral_movement
      - hunt.persistence
      - hunt.command_control
      - hunt.data_exfiltration
      - hunt.privilege_escalation

  blue_analyze:
    name: "Analysis Agent"
    description: "Security event analysis and correlation"
    image: "xorb/blue-analyze:latest"
    categories:
      - analysis
    resource_requirements:
      cpu_cores: 4.0
      memory_mb: 4096
      disk_mb: 2048
      network_mb: 200
    max_concurrent_missions: 3
    default_timeout: 3600
    techniques:
      - analyze.log_correlation
      - analyze.threat_attribution
      - analyze.attack_reconstruction
      - analyze.impact_assessment
      - analyze.forensic_analysis

  blue_respond:
    name: "Incident Response Agent"
    description: "Automated incident response and mitigation"
    image: "xorb/blue-respond:latest"
    categories:
      - mitigation
      - recovery
    resource_requirements:
      cpu_cores: 1.0
      memory_mb: 1024
      disk_mb: 1024
      network_mb: 50
    max_concurrent_missions: 8
    default_timeout: 1800
    techniques:
      - respond.isolate_host
      - respond.block_iocs
      - respond.kill_process
      - respond.quarantine_file
      - respond.reset_credentials
      - respond.patch_vulnerability
      - recover.forensic_collection
      - recover.system_restore

#  Global agent settings
global_settings:
  max_agents_per_mission: 10
  agent_startup_timeout: 300
  agent_shutdown_timeout: 60
  health_check_interval: 30
  performance_monitoring: true
  auto_scaling: true
  resource_monitoring: true
```

##  🎯 Capability Manifests

###  Environment Policies (`config/environment_policies.json`)

```json
{
  "production": {
    "description": "Production environment with strict security controls",
    "allowed_categories": [
      "detection",
      "analysis",
      "mitigation",
      "recovery",
      "threat_hunting"
    ],
    "denied_techniques": ["*"],
    "allowed_techniques": [
      "detect.network_anomaly",
      "detect.process_monitoring",
      "detect.log_analysis",
      "hunt.lateral_movement",
      "hunt.persistence",
      "analyze.log_correlation",
      "respond.isolate_host",
      "respond.block_iocs"
    ],
    "max_risk_level": "medium",
    "max_concurrent_agents": 5,
    "sandbox_constraints": {
      "network_isolation": true,
      "read_only_filesystem": true,
      "no_privilege_escalation": true,
      "resource_limits": {
        "cpu_cores": 2,
        "memory_mb": 1024,
        "disk_mb": 2048,
        "network_bandwidth_mb": 50
      },
      "time_limits": {
        "max_execution_time": 3600,
        "idle_timeout": 300
      },
      "allowed_outbound_connections": [
        "security.company.com",
        "threat-intel.company.com",
        "logging.company.com"
      ],
      "blocked_outbound_connections": [
        "*"
      ],
      "security_policies": [
        "no_internet_access",
        "audit_all_actions",
        "encrypt_all_data"
      ]
    },
    "compliance_requirements": [
      "SOC2",
      "ISO27001",
      "GDPR"
    ],
    "audit_level": "comprehensive"
  },

  "staging": {
    "description": "Staging environment for testing with moderate restrictions",
    "allowed_categories": [
      "reconnaissance",
      "initial_access",
      "execution",
      "persistence",
      "privilege_escalation",
      "defense_evasion",
      "credential_access",
      "discovery",
      "lateral_movement",
      "collection",
      "command_control",
      "detection",
      "analysis",
      "mitigation",
      "recovery",
      "threat_hunting"
    ],
    "denied_techniques": [
      "exploit.web_sqli",
      "persist.web_shell",
      "evasion.process_hollowing",
      "collection.keylogging",
      "collection.credential_dumping"
    ],
    "allowed_techniques": [],
    "max_risk_level": "high",
    "max_concurrent_agents": 15,
    "sandbox_constraints": {
      "network_isolation": true,
      "read_only_filesystem": false,
      "no_privilege_escalation": false,
      "resource_limits": {
        "cpu_cores": 4,
        "memory_mb": 4096,
        "disk_mb": 8192,
        "network_bandwidth_mb": 100
      },
      "time_limits": {
        "max_execution_time": 7200,
        "idle_timeout": 600
      },
      "allowed_outbound_connections": [
        "*.staging.company.com",
        "security-tools.company.com",
        "threat-intel.company.com"
      ],
      "blocked_outbound_connections": [
        "*.production.company.com",
        "*.prod.company.com"
      ],
      "security_policies": [
        "monitor_all_actions",
        "encrypt_sensitive_data"
      ]
    },
    "compliance_requirements": [
      "SOC2"
    ],
    "audit_level": "standard"
  },

  "development": {
    "description": "Development environment with minimal restrictions",
    "allowed_categories": [
      "reconnaissance",
      "initial_access",
      "execution",
      "persistence",
      "privilege_escalation",
      "defense_evasion",
      "credential_access",
      "discovery",
      "lateral_movement",
      "collection",
      "command_control",
      "exfiltration",
      "detection",
      "analysis",
      "mitigation",
      "recovery",
      "threat_hunting"
    ],
    "denied_techniques": [],
    "allowed_techniques": [],
    "max_risk_level": "critical",
    "max_concurrent_agents": 20,
    "sandbox_constraints": {
      "network_isolation": false,
      "read_only_filesystem": false,
      "no_privilege_escalation": false,
      "resource_limits": {
        "cpu_cores": 8,
        "memory_mb": 8192,
        "disk_mb": 16384,
        "network_bandwidth_mb": 1000
      },
      "time_limits": {
        "max_execution_time": 14400,
        "idle_timeout": 1200
      },
      "allowed_outbound_connections": ["*"],
      "blocked_outbound_connections": [],
      "security_policies": [
        "log_all_actions"
      ]
    },
    "compliance_requirements": [],
    "audit_level": "minimal"
  },

  "cyber_range": {
    "description": "Cyber range environment with full capabilities",
    "allowed_categories": [
      "reconnaissance",
      "initial_access",
      "execution",
      "persistence",
      "privilege_escalation",
      "defense_evasion",
      "credential_access",
      "discovery",
      "lateral_movement",
      "collection",
      "command_control",
      "exfiltration",
      "impact",
      "detection",
      "analysis",
      "mitigation",
      "recovery",
      "threat_hunting"
    ],
    "denied_techniques": [],
    "allowed_techniques": [],
    "max_risk_level": "critical",
    "max_concurrent_agents": 50,
    "sandbox_constraints": {
      "network_isolation": true,
      "read_only_filesystem": false,
      "no_privilege_escalation": false,
      "resource_limits": {
        "cpu_cores": 16,
        "memory_mb": 16384,
        "disk_mb": 32768,
        "network_bandwidth_mb": 1000
      },
      "time_limits": {
        "max_execution_time": 28800,
        "idle_timeout": 1800
      },
      "allowed_outbound_connections": [
        "*.cyber-range.internal"
      ],
      "blocked_outbound_connections": [
        "*.company.com",
        "*.prod.*",
        "*.production.*"
      ],
      "security_policies": [
        "comprehensive_logging",
        "real_time_monitoring"
      ]
    },
    "compliance_requirements": [],
    "audit_level": "comprehensive"
  }
}
```

###  Technique Definitions

Technique definitions are stored in separate JSON files under `config/techniques/`:

- `red_team_techniques.json` - Red team attack techniques
- `blue_team_techniques.json` - Blue team defense techniques
- `custom_techniques.json` - Organization-specific techniques

Example technique definition:

```json
{
  "id": "recon.advanced_port_scan",
  "name": "Advanced Port Scanning",
  "category": "reconnaissance",
  "description": "Advanced port scanning with evasion techniques",
  "mitre_id": "T1046",
  "platforms": ["linux", "windows", "macos"],
  "risk_level": "medium",
  "stealth_level": "high",
  "detection_difficulty": "high",
  "dependencies": [],
  "parameters": [
    {
      "name": "target",
      "type": "string",
      "required": true,
      "description": "Target IP address or hostname",
      "constraints": {
        "pattern": "^(?:[0-9]{1,3}\\.){3}[0-9]{1,3}$|^[a-zA-Z0-9.-]+$"
      }
    },
    {
      "name": "ports",
      "type": "string",
      "required": false,
      "default": "1-65535",
      "description": "Port range to scan",
      "constraints": {
        "pattern": "^\\d+(-\\d+)?(,\\d+(-\\d+)?)*$"
      }
    },
    {
      "name": "scan_type",
      "type": "string",
      "required": false,
      "default": "stealth",
      "description": "Type of scan to perform",
      "constraints": {
        "choices": ["tcp", "udp", "syn", "stealth", "fragmented"]
      }
    },
    {
      "name": "timing",
      "type": "string",
      "required": false,
      "default": "adaptive",
      "description": "Scan timing template",
      "constraints": {
        "choices": ["paranoid", "sneaky", "polite", "normal", "aggressive", "insane", "adaptive"]
      }
    },
    {
      "name": "evasion_techniques",
      "type": "list",
      "required": false,
      "default": ["decoy_scans", "source_port_randomization"],
      "description": "Evasion techniques to employ",
      "constraints": {
        "choices": ["decoy_scans", "source_port_randomization", "packet_fragmentation", "timing_randomization"]
      }
    }
  ],
  "outputs": {
    "open_ports": {
      "type": "list",
      "description": "List of open ports discovered"
    },
    "service_fingerprints": {
      "type": "dict",
      "description": "Service fingerprinting results"
    },
    "os_fingerprint": {
      "type": "dict",
      "description": "Operating system fingerprinting results"
    },
    "evasion_success": {
      "type": "dict",
      "description": "Success rate of evasion techniques"
    }
  },
  "metadata": {
    "author": "XORB Security Team",
    "version": "2.1.0",
    "created": "2024-01-01T00:00:00Z",
    "updated": "2024-01-15T10:30:00Z",
    "tags": ["scanning", "reconnaissance", "evasion"],
    "references": [
      "https://nmap.org/book/man-port-scanning-techniques.html",
      "https://attack.mitre.org/techniques/T1046/"
    ]
  }
}
```

##  🐳 Docker Compose Configuration

###  Development Environment (`docker-compose.development.yml`)

```yaml
version: '3.8'

services:
  # Core Infrastructure
  postgres-dev:
    image: ankane/pgvector:v0.5.1
    container_name: xorb-postgres-dev
    environment:
      POSTGRES_DB: xorb_agents_dev
      POSTGRES_USER: xorb_dev
      POSTGRES_PASSWORD: dev_password_123
    ports:
      - "5432:5432"
    volumes:
      - postgres_dev_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U xorb_dev -d xorb_agents_dev"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis-dev:
    image: redis:7-alpine
    container_name: xorb-redis-dev
    ports:
      - "6379:6379"
    volumes:
      - redis_dev_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 5

  temporal-dev:
    image: temporalio/auto-setup:1.22.4
    container_name: xorb-temporal-dev
    environment:
      - DB=postgresql
      - DB_PORT=5432
      - POSTGRES_USER=xorb_dev
      - POSTGRES_PWD=dev_password_123
      - POSTGRES_SEEDS=postgres-dev
      - POSTGRES_DB=temporal_dev
    ports:
      - "7233:7233"
      - "8233:8233"
    depends_on:
      postgres-dev:
        condition: service_healthy

  # Agent Framework Services
  agent-scheduler:
    build:
      context: .
      dockerfile: docker/Dockerfile.scheduler
      target: development
    container_name: xorb-agent-scheduler-dev
    environment:
      - XORB_ENV=development
      - DEBUG=true
      - LOG_LEVEL=DEBUG
      - DATABASE_URL=postgresql://xorb_dev:dev_password_123@postgres-dev:5432/xorb_agents_dev
      - REDIS_URL=redis://redis-dev:6379/0
      - TEMPORAL_HOST=temporal-dev:7233
      - JWT_SECRET=dev_jwt_secret_key_very_long
      - ENCRYPTION_KEY=dev_encryption_key_32_chars_long!!
    ports:
      - "8000:8000"
    volumes:
      - ./src:/app/src:ro
      - ./config:/app/config:ro
      - ./logs:/app/logs
    depends_on:
      postgres-dev:
        condition: service_healthy
      redis-dev:
        condition: service_healthy
      temporal-dev:
        condition: service_started
    restart: unless-stopped

  sandbox-orchestrator:
    build:
      context: .
      dockerfile: docker/Dockerfile.sandbox-orchestrator
      target: development
    container_name: xorb-sandbox-orchestrator-dev
    environment:
      - XORB_ENV=development
      - DEBUG=true
      - LOG_LEVEL=DEBUG
      - REDIS_URL=redis://redis-dev:6379/1
      - DOCKER_HOST=unix:///var/run/docker.sock
      - MAX_SANDBOXES_GLOBAL=50
    ports:
      - "8001:8001"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./src:/app/src:ro
      - ./config:/app/config:ro
      - ./logs:/app/logs
    depends_on:
      redis-dev:
        condition: service_healthy
    restart: unless-stopped
    privileged: true

  telemetry-collector:
    build:
      context: .
      dockerfile: docker/Dockerfile.telemetry
      target: development
    container_name: xorb-telemetry-collector-dev
    environment:
      - XORB_ENV=development
      - DEBUG=true
      - LOG_LEVEL=DEBUG
      - DATABASE_URL=postgresql://xorb_dev:dev_password_123@postgres-dev:5432/xorb_agents_dev
      - REDIS_URL=redis://redis-dev:6379/2
      - ML_ENABLED=true
      - ANALYTICS_ENABLED=true
    ports:
      - "8002:8002"
    volumes:
      - ./src:/app/src:ro
      - ./config:/app/config:ro
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      postgres-dev:
        condition: service_healthy
      redis-dev:
        condition: service_healthy
    restart: unless-stopped

  # Monitoring Stack
  prometheus:
    image: prom/prometheus:latest
    container_name: xorb-prometheus-dev
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./monitoring/prometheus/rules.yml:/etc/prometheus/rules.yml:ro
      - prometheus_dev_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'

  grafana:
    image: grafana/grafana:latest
    container_name: xorb-grafana-dev
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=dev_admin_123
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro
      - grafana_dev_data:/var/lib/grafana

networks:
  default:
    name: xorb-agents-dev
    driver: bridge

volumes:
  postgres_dev_data:
  redis_dev_data:
  prometheus_dev_data:
  grafana_dev_data:
```

###  Production Environment (`docker-compose.production.yml`)

```yaml
version: '3.8'

services:
  # Production services with security hardening
  postgres:
    image: ankane/pgvector:v0.5.1
    container_name: xorb-postgres-prod
    environment:
      POSTGRES_DB: xorb_agents
      POSTGRES_USER: ${DATABASE_USER}
      POSTGRES_PASSWORD: ${DATABASE_PASSWORD}
      POSTGRES_INITDB_ARGS: "--encoding=UTF-8 --lc-collate=C --lc-ctype=C"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./config/postgres/postgresql.conf:/etc/postgresql/postgresql.conf:ro
      - ./config/postgres/pg_hba.conf:/etc/postgresql/pg_hba.conf:ro
    command: >
      postgres
      -c config_file=/etc/postgresql/postgresql.conf
      -c hba_file=/etc/postgresql/pg_hba.conf
    networks:
      - internal
    restart: always
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
      - /var/run/postgresql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DATABASE_USER} -d xorb_agents"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 30s

  redis:
    image: redis:7-alpine
    container_name: xorb-redis-prod
    command: >
      redis-server
      --requirepass ${REDIS_PASSWORD}
      --maxmemory 2gb
      --maxmemory-policy allkeys-lru
      --save 900 1 300 10 60 10000
      --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - internal
    restart: always
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
    healthcheck:
      test: ["CMD", "redis-cli", "--no-auth-warning", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 30s
      timeout: 10s
      retries: 5

  agent-scheduler:
    image: xorb/agent-scheduler:${VERSION:-latest}
    container_name: xorb-agent-scheduler-prod
    environment:
      - XORB_ENV=production
      - DEBUG=false
      - LOG_LEVEL=INFO
      - DATABASE_URL=postgresql://${DATABASE_USER}:${DATABASE_PASSWORD}@postgres:5432/xorb_agents
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - TEMPORAL_HOST=temporal:7233
      - JWT_SECRET=${JWT_SECRET}
      - ENCRYPTION_KEY=${ENCRYPTION_KEY}
      - PROMETHEUS_ENABLED=true
      - AUDIT_LOG_ENABLED=true
    ports:
      - "8000:8000"
    volumes:
      - ./config:/app/config:ro
      - logs:/app/logs
      - ./certs:/app/certs:ro
    networks:
      - internal
      - external
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: always
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
      - /app/temp
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  sandbox-orchestrator:
    image: xorb/sandbox-orchestrator:${VERSION:-latest}
    container_name: xorb-sandbox-orchestrator-prod
    environment:
      - XORB_ENV=production
      - DEBUG=false
      - LOG_LEVEL=INFO
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/1
      - DOCKER_HOST=unix:///var/run/docker.sock
      - MAX_SANDBOXES_GLOBAL=100
      - KATA_RUNTIME_ENABLED=${KATA_ENABLED:-false}
    ports:
      - "8001:8001"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./config:/app/config:ro
      - logs:/app/logs
    networks:
      - internal
    depends_on:
      redis:
        condition: service_healthy
    restart: always
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 4G
        reservations:
          cpus: '2.0'
          memory: 2G

networks:
  internal:
    driver: bridge
    internal: true
  external:
    driver: bridge

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  logs:
    driver: local

secrets:
  database_password:
    external: true
  redis_password:
    external: true
  jwt_secret:
    external: true
  encryption_key:
    external: true
```

##  ☸️ Kubernetes Configuration

###  Helm Values (`values.yaml`)

```yaml
#  Global configuration
global:
  environment: production
  version: "2.1.0"
  registry: "xorb"
  imagePullPolicy: IfNotPresent

  # Security context
  securityContext:
    runAsNonRoot: true
    runAsUser: 1001
    fsGroup: 1001

  # Resource defaults
  resources:
    requests:
      cpu: 100m
      memory: 128Mi
    limits:
      cpu: 500m
      memory: 512Mi

#  Agent Scheduler
agentScheduler:
  enabled: true
  replicaCount: 3

  image:
    repository: xorb/agent-scheduler
    tag: ""  # Uses global.version if empty

  service:
    type: ClusterIP
    port: 8000

  ingress:
    enabled: true
    className: nginx
    annotations:
      nginx.ingress.kubernetes.io/ssl-redirect: "true"
      nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    hosts:
      - host: agents.company.com
        paths:
          - path: /
            pathType: Prefix
    tls:
      - secretName: agents-tls
        hosts:
          - agents.company.com

  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 2000m
      memory: 4Gi

  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: 80

  env:
    XORB_ENV: production
    LOG_LEVEL: INFO
    PROMETHEUS_ENABLED: "true"
    AUDIT_LOG_ENABLED: "true"

#  Sandbox Orchestrator
sandboxOrchestrator:
  enabled: true
  replicaCount: 2

  image:
    repository: xorb/sandbox-orchestrator

  nodeSelector:
    node-type: sandbox-worker

  resources:
    requests:
      cpu: 1000m
      memory: 2Gi
    limits:
      cpu: 4000m
      memory: 8Gi

  env:
    MAX_SANDBOXES_GLOBAL: "200"
    KATA_RUNTIME_ENABLED: "true"

#  Telemetry Collector
telemetryCollector:
  enabled: true
  replicaCount: 2

  image:
    repository: xorb/telemetry-collector

  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 2000m
      memory: 4Gi

  env:
    ML_ENABLED: "true"
    ANALYTICS_ENABLED: "true"

#  PostgreSQL
postgresql:
  enabled: true
  image:
    repository: ankane/pgvector
    tag: v0.5.1

  auth:
    postgresPassword: ""  # Set via secret
    username: xorb
    password: ""  # Set via secret
    database: xorb_agents

  primary:
    persistence:
      enabled: true
      size: 100Gi
      storageClass: fast-ssd

    resources:
      requests:
        cpu: 1000m
        memory: 2Gi
      limits:
        cpu: 4000m
        memory: 8Gi

    configuration: |
      shared_preload_libraries = 'vector'
      max_connections = 200
      shared_buffers = 256MB
      effective_cache_size = 1GB
      maintenance_work_mem = 64MB
      checkpoint_completion_target = 0.9
      wal_buffers = 16MB
      default_statistics_target = 100
      random_page_cost = 1.1
      effective_io_concurrency = 200

  readReplicas:
    replicaCount: 2
    resources:
      requests:
        cpu: 500m
        memory: 1Gi
      limits:
        cpu: 2000m
        memory: 4Gi

#  Redis
redis:
  enabled: true
  architecture: replication

  auth:
    enabled: true
    password: ""  # Set via secret

  master:
    persistence:
      enabled: true
      size: 20Gi
      storageClass: fast-ssd

    resources:
      requests:
        cpu: 500m
        memory: 1Gi
      limits:
        cpu: 2000m
        memory: 4Gi

  replica:
    replicaCount: 2
    resources:
      requests:
        cpu: 250m
        memory: 512Mi
      limits:
        cpu: 1000m
        memory: 2Gi

#  Temporal
temporal:
  enabled: true

  server:
    replicaCount: 3
    resources:
      requests:
        cpu: 1000m
        memory: 2Gi
      limits:
        cpu: 4000m
        memory: 8Gi

  persistence:
    enabled: true
    size: 50Gi
    storageClass: fast-ssd

  postgresql:
    enabled: false  # Use external PostgreSQL

#  Monitoring
monitoring:
  prometheus:
    enabled: true

    server:
      retention: 30d
      resources:
        requests:
          cpu: 500m
          memory: 2Gi
        limits:
          cpu: 2000m
          memory: 8Gi

      storage:
        size: 100Gi
        storageClass: fast-ssd

  grafana:
    enabled: true

    ingress:
      enabled: true
      hosts:
        - grafana.company.com

    resources:
      requests:
        cpu: 250m
        memory: 512Mi
      limits:
        cpu: 1000m
        memory: 2Gi

  alertmanager:
    enabled: true

    config:
      global:
        smtp_smarthost: 'mail.company.com:587'
        smtp_from: 'alerts@company.com'

      route:
        group_by: ['alertname']
        group_wait: 10s
        group_interval: 10s
        repeat_interval: 1h
        receiver: 'web.hook'

      receivers:
        - name: 'web.hook'
          slack_configs:
            - api_url: '${SLACK_WEBHOOK_URL}'
              channel: '#security-alerts'

#  Service Mesh (Optional)
istio:
  enabled: false

#  Network Policies
networkPolicies:
  enabled: true

#  Pod Security Policies
podSecurityPolicy:
  enabled: true

#  RBAC
rbac:
  create: true

#  Service Account
serviceAccount:
  create: true
  annotations: {}

#  Secrets
secrets:
  create: true

#  ConfigMaps
configMaps:
  create: true
```

##  🔍 Configuration Validation

###  Validation Script (`scripts/validate-config.py`)

```python
# !/usr/bin/env python3
"""
Configuration validation script for XORB Red/Blue Agent Framework
"""

import os
import sys
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

#  Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ConfigValidator:
    """Validates XORB agent framework configuration"""

    def __init__(self, config_dir: Path = Path("config")):
        self.config_dir = config_dir
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_all(self) -> bool:
        """Validate all configuration components"""
        logger.info("Starting configuration validation...")

        # Validate environment variables
        self.validate_environment_variables()

        # Validate configuration files
        self.validate_config_files()

        # Validate capability manifests
        self.validate_capability_manifests()

        # Validate Docker configuration
        self.validate_docker_config()

        # Report results
        self.report_results()

        return len(self.errors) == 0

    def validate_environment_variables(self):
        """Validate required environment variables"""
        logger.info("Validating environment variables...")

        required_vars = [
            'DATABASE_URL',
            'REDIS_URL',
            'JWT_SECRET',
            'ENCRYPTION_KEY'
        ]

        for var in required_vars:
            if not os.getenv(var):
                self.errors.append(f"Required environment variable {var} is not set")

        # Validate JWT secret length
        jwt_secret = os.getenv('JWT_SECRET', '')
        if len(jwt_secret) < 32:
            self.errors.append("JWT_SECRET must be at least 32 characters long")

        # Validate encryption key length
        encryption_key = os.getenv('ENCRYPTION_KEY', '')
        if len(encryption_key) != 32:
            self.errors.append("ENCRYPTION_KEY must be exactly 32 characters long")

    def validate_config_files(self):
        """Validate YAML configuration files"""
        logger.info("Validating configuration files...")

        config_files = [
            'app.yaml',
            'logging.yaml',
            'agents.yaml'
        ]

        for filename in config_files:
            filepath = self.config_dir / filename
            if not filepath.exists():
                self.errors.append(f"Configuration file {filename} not found")
                continue

            try:
                with open(filepath) as f:
                    yaml.safe_load(f)
                logger.info(f"✓ {filename} is valid YAML")
            except yaml.YAMLError as e:
                self.errors.append(f"Invalid YAML in {filename}: {e}")

    def validate_capability_manifests(self):
        """Validate capability manifest files"""
        logger.info("Validating capability manifests...")

        techniques_dir = self.config_dir / "techniques"
        if not techniques_dir.exists():
            self.errors.append("Techniques directory not found")
            return

        manifest_files = list(techniques_dir.glob("*.json"))
        if not manifest_files:
            self.warnings.append("No technique manifest files found")
            return

        for filepath in manifest_files:
            try:
                with open(filepath) as f:
                    manifest = json.load(f)

                self.validate_technique_manifest(manifest, filepath.name)
                logger.info(f"✓ {filepath.name} is valid")
            except json.JSONDecodeError as e:
                self.errors.append(f"Invalid JSON in {filepath.name}: {e}")

    def validate_technique_manifest(self, manifest: Dict[str, Any], filename: str):
        """Validate individual technique manifest"""
        required_fields = ['manifest_version', 'techniques']

        for field in required_fields:
            if field not in manifest:
                self.errors.append(f"Missing required field '{field}' in {filename}")

        if 'techniques' in manifest:
            for i, technique in enumerate(manifest['techniques']):
                self.validate_technique_definition(technique, f"{filename}[{i}]")

    def validate_technique_definition(self, technique: Dict[str, Any], location: str):
        """Validate individual technique definition"""
        required_fields = ['id', 'name', 'category', 'description']

        for field in required_fields:
            if field not in technique:
                self.errors.append(f"Missing required field '{field}' in technique at {location}")

        # Validate parameters if present
        if 'parameters' in technique:
            for i, param in enumerate(technique['parameters']):
                self.validate_parameter_definition(param, f"{location}.parameters[{i}]")

    def validate_parameter_definition(self, param: Dict[str, Any], location: str):
        """Validate technique parameter definition"""
        required_fields = ['name', 'type', 'required']

        for field in required_fields:
            if field not in param:
                self.errors.append(f"Missing required field '{field}' in parameter at {location}")

        # Validate parameter type
        valid_types = ['string', 'int', 'float', 'bool', 'list']
        if param.get('type') not in valid_types:
            self.errors.append(f"Invalid parameter type '{param.get('type')}' at {location}")

    def validate_docker_config(self):
        """Validate Docker Compose configuration"""
        logger.info("Validating Docker configuration...")

        compose_files = [
            'docker-compose.yml',
            'docker-compose.development.yml',
            'docker-compose.production.yml'
        ]

        for filename in compose_files:
            filepath = Path(filename)
            if filepath.exists():
                try:
                    with open(filepath) as f:
                        yaml.safe_load(f)
                    logger.info(f"✓ {filename} is valid YAML")
                except yaml.YAMLError as e:
                    self.errors.append(f"Invalid YAML in {filename}: {e}")
            else:
                self.warnings.append(f"Docker Compose file {filename} not found")

    def report_results(self):
        """Report validation results"""
        logger.info("Configuration validation complete")

        if self.warnings:
            logger.warning(f"Found {len(self.warnings)} warnings:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")

        if self.errors:
            logger.error(f"Found {len(self.errors)} errors:")
            for error in self.errors:
                logger.error(f"  - {error}")
        else:
            logger.info("✓ All configuration is valid")

def main():
    """Main function"""
    validator = ConfigValidator()

    if validator.validate_all():
        logger.info("Configuration validation passed")
        sys.exit(0)
    else:
        logger.error("Configuration validation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

##  🛠️ Configuration Tools

###  Environment Generator (`scripts/generate-env.py`)

```python
# !/usr/bin/env python3
"""
Generate environment configuration files with secure defaults
"""

import secrets
import string
from pathlib import Path

def generate_secret(length: int = 32) -> str:
    """Generate a cryptographically secure random string"""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def generate_env_file(environment: str = "development"):
    """Generate .env file with secure defaults"""

    config = {
        # Core settings
        'XORB_ENV': environment,
        'DEBUG': 'false' if environment == 'production' else 'true',
        'LOG_LEVEL': 'INFO' if environment == 'production' else 'DEBUG',

        # Database
        'DATABASE_URL': f'postgresql://xorb:secure_password@postgres:5432/xorb_agents_{environment}',
        'DATABASE_POOL_SIZE': '20',
        'DATABASE_MAX_OVERFLOW': '30',

        # Redis
        'REDIS_URL': 'redis://redis:6379/0',
        'REDIS_PASSWORD': generate_secret(16),

        # Security
        'JWT_SECRET': generate_secret(64),
        'ENCRYPTION_KEY': generate_secret(32),

        # Sandbox
        'MAX_SANDBOXES_GLOBAL': '50' if environment == 'development' else '100',
        'DEFAULT_SANDBOX_TTL': '3600',

        # Monitoring
        'PROMETHEUS_ENABLED': 'true',
        'HEALTH_CHECK_ENABLED': 'true',

        # External services
        'SLACK_WEBHOOK_URL': '',
        'EMAIL_SMTP_HOST': '',
        'EMAIL_SMTP_USER': '',
        'EMAIL_SMTP_PASSWORD': '',
    }

    # Write to file
    env_file = Path(f'.env.{environment}')
    with open(env_file, 'w') as f:
        f.write(f"# XORB Red/Blue Agent Framework - {environment.title()} Environment\n")
        f.write(f"# Generated configuration file\n\n")

        for key, value in config.items():
            f.write(f"{key}={value}\n")

    print(f"Generated {env_file}")
    print(f"Please review and update the configuration as needed")

if __name__ == "__main__":
    import sys
    environment = sys.argv[1] if len(sys.argv) > 1 else "development"
    generate_env_file(environment)
```

---

This comprehensive configuration reference covers all aspects of the XORB Red/Blue Agent Framework configuration. For specific deployment scenarios, refer to the [Installation Guide](./installation.md) and [Operations Documentation](./operations/).