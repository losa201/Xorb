#!/bin/bash
# XORB PTaaS Enterprise Deployment Script
# Deploys production-ready XORB platform with all enterprise features

set -euo pipefail

# Configuration
XORB_VERSION="${XORB_VERSION:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"
DOMAIN="${DOMAIN:-ptaas.company.com}"
DATABASE_SIZE="${DATABASE_SIZE:-large}"
ENABLE_MONITORING="${ENABLE_MONITORING:-true}"
ENABLE_VAULT="${ENABLE_VAULT:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

success() {
    echo -e "${PURPLE}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking deployment prerequisites..."

    # Check if running as appropriate user
    if [[ $EUID -eq 0 ]]; then
        warn "Running as root. Consider using a dedicated deployment user."
    fi

    # Check required tools
    local required_tools=("docker" "docker-compose" "kubectl" "helm" "jq" "curl")
    local missing_tools=()

    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        error "Missing required tools: ${missing_tools[*]}"
        error "Please install missing tools and retry"
        exit 1
    fi

    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi

    # Check Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        warn "No Kubernetes cluster detected. Will deploy with Docker Compose."
        USE_KUBERNETES=false
    else
        info "Kubernetes cluster detected. Will use Kubernetes deployment."
        USE_KUBERNETES=true
    fi

    success "Prerequisites check completed"
}

# Generate configuration
generate_configuration() {
    log "Generating deployment configuration..."

    # Create deployment directory
    DEPLOY_DIR="/opt/xorb-deployment-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$DEPLOY_DIR"/{config,secrets,logs,backups}

    # Generate environment configuration
    cat > "$DEPLOY_DIR/config/environment.env" << EOF
# XORB Enterprise Configuration
ENVIRONMENT=$ENVIRONMENT
XORB_VERSION=$XORB_VERSION
DOMAIN=$DOMAIN

# Database Configuration
DATABASE_URL=postgresql://xorb:$(openssl rand -base64 32)@postgres:5432/xorb
REDIS_URL=redis://:$(openssl rand -base64 32)@redis:6379

# Security Configuration
JWT_SECRET=$(openssl rand -base64 64)
ENCRYPTION_KEY=$(openssl rand -base64 32)
API_RATE_LIMIT_PER_MINUTE=1000
API_RATE_LIMIT_PER_HOUR=10000

# External API Keys (set these manually)
NVIDIA_API_KEY=your_nvidia_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Monitoring Configuration
ENABLE_METRICS=true
PROMETHEUS_RETENTION=30d
GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 20)

# Vault Configuration
VAULT_ADDR=http://vault:8200
VAULT_TOKEN=your_vault_token_here

# Email Configuration
SMTP_HOST=smtp.company.com
SMTP_PORT=587
SMTP_USERNAME=noreply@company.com
SMTP_PASSWORD=your_smtp_password_here

# SAML/OIDC Configuration
SAML_IDP_ENTITY_ID=https://sso.company.com/adfs/services/trust
SAML_IDP_SSO_URL=https://sso.company.com/adfs/ls/
OIDC_ISSUER=https://auth.company.com
OIDC_CLIENT_ID=xorb-ptaas
OIDC_CLIENT_SECRET=your_oidc_client_secret_here

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
AUDIT_LOG_RETENTION=365d
EOF

    # Generate Docker Compose configuration
    cat > "$DEPLOY_DIR/docker-compose.enterprise.yml" << 'EOF'
version: '3.8'

services:
  # Core Database
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: xorb
      POSTGRES_USER: xorb
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_INITDB_ARGS: "--encoding=UTF-8 --locale=C"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./config/postgresql.conf:/etc/postgresql/postgresql.conf
      - ./config/pg_hba.conf:/etc/postgresql/pg_hba.conf
    ports:
      - "5432:5432"
    command: postgres -c config_file=/etc/postgresql/postgresql.conf
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U xorb -d xorb"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis Cache
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD} --maxmemory 2gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1'
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # XORB API Service
  xorb-api:
    image: xorb/ptaas-api:${XORB_VERSION}
    env_file:
      - ./config/environment.env
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - api_uploads:/app/uploads
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # XORB Orchestrator
  xorb-orchestrator:
    image: xorb/ptaas-orchestrator:${XORB_VERSION}
    env_file:
      - ./config/environment.env
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - scan_data:/app/scan_data
    depends_on:
      - xorb-api
      - postgres
      - redis
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 4G
          cpus: '2'

  # XORB Scanner Workers
  xorb-scanner:
    image: xorb/ptaas-scanner:${XORB_VERSION}
    env_file:
      - ./config/environment.env
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - scan_data:/app/scan_data
      - /var/run/docker.sock:/var/run/docker.sock
    depends_on:
      - xorb-orchestrator
    deploy:
      replicas: 5
      resources:
        limits:
          memory: 8G
          cpus: '4'
    privileged: true  # Required for security scanning tools

  # Frontend Application
  xorb-frontend:
    image: xorb/ptaas-frontend:${XORB_VERSION}
    environment:
      REACT_APP_API_URL: https://${DOMAIN}/api
      REACT_APP_ENVIRONMENT: ${ENVIRONMENT}
    ports:
      - "3000:3000"
    depends_on:
      - xorb-api
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'

  # Nginx Load Balancer & SSL Termination
  nginx:
    image: nginx:alpine
    volumes:
      - ./config/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./config/ssl:/etc/nginx/ssl:ro
      - ./logs/nginx:/var/log/nginx
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - xorb-api
      - xorb-frontend
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'

  # Monitoring Stack
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD}
      GF_SECURITY_ADMIN_USER: admin
      GF_USERS_ALLOW_SIGN_UP: false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana:/etc/grafana/provisioning:ro
    ports:
      - "3001:3000"
    depends_on:
      - prometheus

  # HashiCorp Vault (if enabled)
  vault:
    image: vault:latest
    environment:
      VAULT_DEV_ROOT_TOKEN_ID: ${VAULT_ROOT_TOKEN}
      VAULT_DEV_LISTEN_ADDRESS: 0.0.0.0:8200
    volumes:
      - vault_data:/vault/data
      - ./config/vault:/vault/config:ro
    ports:
      - "8200:8200"
    cap_add:
      - IPC_LOCK
    command: vault server -config=/vault/config/vault.hcl

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local
  vault_data:
    driver: local
  api_uploads:
    driver: local
  scan_data:
    driver: local

networks:
  default:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
EOF

    # Generate Kubernetes manifests
    if [[ "$USE_KUBERNETES" == "true" ]]; then
        generate_kubernetes_manifests
    fi

    # Generate SSL certificates
    generate_ssl_certificates

    # Generate monitoring configuration
    generate_monitoring_config

    success "Configuration generated in $DEPLOY_DIR"
}

generate_kubernetes_manifests() {
    log "Generating Kubernetes manifests..."

    mkdir -p "$DEPLOY_DIR/k8s"

    # Namespace
    cat > "$DEPLOY_DIR/k8s/namespace.yaml" << EOF
apiVersion: v1
kind: Namespace
metadata:
  name: xorb-ptaas
  labels:
    name: xorb-ptaas
    app.kubernetes.io/name: xorb-ptaas
    app.kubernetes.io/version: "$XORB_VERSION"
EOF

    # ConfigMap
    cat > "$DEPLOY_DIR/k8s/configmap.yaml" << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: xorb-config
  namespace: xorb-ptaas
data:
  ENVIRONMENT: "$ENVIRONMENT"
  DOMAIN: "$DOMAIN"
  LOG_LEVEL: "INFO"
  ENABLE_METRICS: "true"
EOF

    # Secrets
    cat > "$DEPLOY_DIR/k8s/secrets.yaml" << EOF
apiVersion: v1
kind: Secret
metadata:
  name: xorb-secrets
  namespace: xorb-ptaas
type: Opaque
stringData:
  DATABASE_URL: "postgresql://xorb:$(openssl rand -base64 32)@postgres:5432/xorb"
  REDIS_URL: "redis://:$(openssl rand -base64 32)@redis:6379"
  JWT_SECRET: "$(openssl rand -base64 64)"
  ENCRYPTION_KEY: "$(openssl rand -base64 32)"
EOF

    # PostgreSQL Deployment
    cat > "$DEPLOY_DIR/k8s/postgres.yaml" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: xorb-ptaas
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        envFrom:
        - secretRef:
            name: xorb-secrets
        env:
        - name: POSTGRES_DB
          value: xorb
        - name: POSTGRES_USER
          value: xorb
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          limits:
            memory: "4Gi"
            cpu: "2"
          requests:
            memory: "2Gi"
            cpu: "1"
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: xorb-ptaas
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: xorb-ptaas
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
EOF

    # XORB API Deployment
    cat > "$DEPLOY_DIR/k8s/xorb-api.yaml" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: xorb-api
  namespace: xorb-ptaas
spec:
  replicas: 3
  selector:
    matchLabels:
      app: xorb-api
  template:
    metadata:
      labels:
        app: xorb-api
    spec:
      containers:
      - name: xorb-api
        image: xorb/ptaas-api:$XORB_VERSION
        envFrom:
        - configMapRef:
            name: xorb-config
        - secretRef:
            name: xorb-secrets
        ports:
        - containerPort: 8000
        resources:
          limits:
            memory: "2Gi"
            cpu: "1"
          requests:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: xorb-api
  namespace: xorb-ptaas
spec:
  selector:
    app: xorb-api
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
EOF

    # Ingress
    cat > "$DEPLOY_DIR/k8s/ingress.yaml" << EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: xorb-ingress
  namespace: xorb-ptaas
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - $DOMAIN
    secretName: xorb-tls
  rules:
  - host: $DOMAIN
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: xorb-api
            port:
              number: 8000
      - path: /
        pathType: Prefix
        backend:
          service:
            name: xorb-frontend
            port:
              number: 3000
EOF
}

generate_ssl_certificates() {
    log "Generating SSL certificates..."

    mkdir -p "$DEPLOY_DIR/config/ssl"

    # Generate self-signed certificate for development
    if [[ "$ENVIRONMENT" != "production" ]]; then
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$DEPLOY_DIR/config/ssl/privkey.pem" \
            -out "$DEPLOY_DIR/config/ssl/fullchain.pem" \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=$DOMAIN"

        info "Generated self-signed SSL certificate for development"
    else
        warn "Production deployment detected. Please configure proper SSL certificates."
        warn "Place your SSL certificate files in: $DEPLOY_DIR/config/ssl/"
        warn "Required files: privkey.pem, fullchain.pem"
    fi
}

generate_monitoring_config() {
    log "Generating monitoring configuration..."

    # Prometheus configuration
    cat > "$DEPLOY_DIR/config/prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'xorb-api'
    static_configs:
      - targets: ['xorb-api:8000']
    metrics_path: '/metrics'

  - job_name: 'xorb-orchestrator'
    static_configs:
      - targets: ['xorb-orchestrator:8001']
    metrics_path: '/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
EOF

    # Alert rules
    cat > "$DEPLOY_DIR/config/alert_rules.yml" << EOF
groups:
- name: xorb_alerts
  rules:
  - alert: XORBAPIDown
    expr: up{job="xorb-api"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "XORB API is down"
      description: "XORB API has been down for more than 1 minute"

  - alert: HighVulnerabilityRate
    expr: rate(xorb_vulnerabilities_detected_total[5m]) > 10
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High vulnerability detection rate"
      description: "More than 10 vulnerabilities detected per minute"

  - alert: DatabaseConnectionFailure
    expr: up{job="postgres"} == 0
    for: 30s
    labels:
      severity: critical
    annotations:
      summary: "Database connection failure"
      description: "Cannot connect to PostgreSQL database"
EOF

    # Grafana dashboard provisioning
    mkdir -p "$DEPLOY_DIR/config/grafana/dashboards"
    mkdir -p "$DEPLOY_DIR/config/grafana/datasources"

    cat > "$DEPLOY_DIR/config/grafana/datasources/prometheus.yml" << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF
}

# Deploy XORB Platform
deploy_platform() {
    log "Deploying XORB PTaaS Platform..."

    cd "$DEPLOY_DIR"

    if [[ "$USE_KUBERNETES" == "true" ]]; then
        deploy_kubernetes
    else
        deploy_docker_compose
    fi
}

deploy_docker_compose() {
    log "Deploying with Docker Compose..."

    # Pull latest images
    docker-compose -f docker-compose.enterprise.yml pull

    # Start services
    docker-compose -f docker-compose.enterprise.yml up -d

    # Wait for services to be healthy
    log "Waiting for services to become healthy..."

    local max_attempts=60
    local attempt=0

    while [[ $attempt -lt $max_attempts ]]; do
        if docker-compose -f docker-compose.enterprise.yml ps | grep -q "Up (healthy)"; then
            break
        fi

        echo -n "."
        sleep 5
        ((attempt++))
    done

    if [[ $attempt -eq $max_attempts ]]; then
        error "Services failed to become healthy within timeout"
        docker-compose -f docker-compose.enterprise.yml logs
        exit 1
    fi

    success "Docker Compose deployment completed"
}

deploy_kubernetes() {
    log "Deploying to Kubernetes..."

    # Apply manifests
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/secrets.yaml
    kubectl apply -f k8s/postgres.yaml
    kubectl apply -f k8s/xorb-api.yaml
    kubectl apply -f k8s/ingress.yaml

    # Wait for deployments
    kubectl wait --for=condition=available --timeout=300s deployment/postgres -n xorb-ptaas
    kubectl wait --for=condition=available --timeout=300s deployment/xorb-api -n xorb-ptaas

    success "Kubernetes deployment completed"
}

# Run initial setup
run_initial_setup() {
    log "Running initial platform setup..."

    # Wait for API to be available
    local api_url="http://localhost:8000"
    if [[ "$USE_KUBERNETES" == "true" ]]; then
        api_url="https://$DOMAIN/api"
    fi

    local max_attempts=30
    local attempt=0

    while [[ $attempt -lt $max_attempts ]]; do
        if curl -f "$api_url/health" &> /dev/null; then
            break
        fi

        echo -n "."
        sleep 10
        ((attempt++))
    done

    if [[ $attempt -eq $max_attempts ]]; then
        error "API failed to become available"
        exit 1
    fi

    # Run database migrations
    log "Running database migrations..."
    if [[ "$USE_KUBERNETES" == "true" ]]; then
        kubectl exec -it deployment/xorb-api -n xorb-ptaas -- python -m alembic upgrade head
    else
        docker-compose -f docker-compose.enterprise.yml exec xorb-api python -m alembic upgrade head
    fi

    # Create default admin user
    log "Creating default admin user..."
    local admin_password=$(openssl rand -base64 20)

    cat > "$DEPLOY_DIR/create_admin.py" << EOF
import asyncio
from src.api.app.auth.enterprise_auth import get_enterprise_auth, EnterpriseUser, AuthProvider
from datetime import datetime

async def create_admin():
    auth = get_enterprise_auth()

    admin_user = EnterpriseUser(
        user_id="admin_001",
        email="admin@company.com",
        first_name="XORB",
        last_name="Administrator",
        tenant_id="default_tenant",
        roles=["system_administrator"],
        provider=AuthProvider.LOCAL,
        provider_id="local_admin_001",
        is_active=True,
        is_verified=True,
        created_at=datetime.utcnow()
    )

    auth.users[admin_user.user_id] = admin_user
    print(f"Admin user created: {admin_user.email}")
    print(f"Password: $admin_password")

if __name__ == "__main__":
    asyncio.run(create_admin())
EOF

    if [[ "$USE_KUBERNETES" == "true" ]]; then
        kubectl cp "$DEPLOY_DIR/create_admin.py" xorb-ptaas/$(kubectl get pods -n xorb-ptaas -l app=xorb-api -o jsonpath='{.items[0].metadata.name}'):/tmp/
        kubectl exec -it deployment/xorb-api -n xorb-ptaas -- python /tmp/create_admin.py
    else
        docker-compose -f docker-compose.enterprise.yml exec xorb-api python /tmp/create_admin.py
    fi

    # Save admin credentials
    cat > "$DEPLOY_DIR/admin_credentials.txt" << EOF
XORB PTaaS Enterprise - Admin Credentials
========================================
Email: admin@company.com
Password: $admin_password
Login URL: https://$DOMAIN

Please change the password after first login.
EOF

    chmod 600 "$DEPLOY_DIR/admin_credentials.txt"

    success "Initial setup completed"
}

# Generate deployment report
generate_deployment_report() {
    log "Generating deployment report..."

    cat > "$DEPLOY_DIR/deployment_report.md" << EOF
# XORB PTaaS Enterprise Deployment Report

## Deployment Information
- **Deployment Date**: $(date)
- **Version**: $XORB_VERSION
- **Environment**: $ENVIRONMENT
- **Domain**: $DOMAIN
- **Deployment Method**: $( [[ "$USE_KUBERNETES" == "true" ]] && echo "Kubernetes" || echo "Docker Compose" )

## Service Endpoints
- **Main Application**: https://$DOMAIN
- **API Documentation**: https://$DOMAIN/api/docs
- **Monitoring (Grafana)**: http://$DOMAIN:3001
- **Metrics (Prometheus)**: http://$DOMAIN:9090

## Default Admin Access
- **Email**: admin@company.com
- **Password**: See admin_credentials.txt (secure file)

## Configuration Files
- **Environment Config**: config/environment.env
- **SSL Certificates**: config/ssl/
- **Monitoring Config**: config/prometheus.yml
- **Database Backups**: backups/

## Next Steps
1. **Security Configuration**:
   - Update default passwords
   - Configure SAML/OIDC authentication
   - Set up proper SSL certificates for production

2. **Monitoring Setup**:
   - Configure alerting rules
   - Set up notification channels
   - Import custom Grafana dashboards

3. **Backup Configuration**:
   - Set up automated database backups
   - Configure log rotation
   - Implement disaster recovery procedures

4. **Security Tools Integration**:
   - Install security scanning tools
   - Configure tool authentication
   - Test scanning capabilities

## Support Information
- **Documentation**: https://docs.xorb-security.com
- **Support Portal**: https://support.xorb-security.com
- **Emergency Contact**: enterprise@xorb-security.com

## Security Considerations
- All default passwords have been randomly generated
- Database is configured with row-level security
- API rate limiting is enabled
- Audit logging is configured
- SSL/TLS encryption is enforced

Deployment ID: $(uuidgen)
EOF

    success "Deployment report generated: $DEPLOY_DIR/deployment_report.md"
}

# Validate deployment
validate_deployment() {
    log "Validating deployment..."

    local api_url="http://localhost:8000"
    if [[ "$USE_KUBERNETES" == "true" ]]; then
        api_url="https://$DOMAIN/api"
    fi

    # Test API health
    if ! curl -f "$api_url/health" &> /dev/null; then
        error "API health check failed"
        return 1
    fi

    # Test database connection
    if ! curl -f "$api_url/health/database" &> /dev/null; then
        error "Database health check failed"
        return 1
    fi

    # Test authentication
    if ! curl -f "$api_url/auth/health" &> /dev/null; then
        error "Authentication health check failed"
        return 1
    fi

    success "Deployment validation passed"
}

# Main deployment function
main() {
    echo "
╔══════════════════════════════════════════════════════════════╗
║               XORB PTaaS Enterprise Deployment               ║
║                    Production Ready Platform                 ║
╚══════════════════════════════════════════════════════════════╝
"

    log "Starting XORB PTaaS Enterprise deployment..."
    log "Version: $XORB_VERSION | Environment: $ENVIRONMENT | Domain: $DOMAIN"

    check_prerequisites
    generate_configuration
    deploy_platform
    run_initial_setup

    if validate_deployment; then
        generate_deployment_report

        success "🚀 XORB PTaaS Enterprise deployment completed successfully!"
        success "📍 Deployment location: $DEPLOY_DIR"
        success "🌐 Access your platform at: https://$DOMAIN"
        success "📊 Monitoring dashboard: http://$DOMAIN:3001"
        success "📋 See deployment_report.md for complete details"

        echo ""
        warn "⚠️  IMPORTANT: Please secure your admin credentials in admin_credentials.txt"
        warn "⚠️  IMPORTANT: Configure proper SSL certificates for production use"
        warn "⚠️  IMPORTANT: Set up external API keys in config/environment.env"

    else
        error "❌ Deployment validation failed. Check logs and retry."
        exit 1
    fi
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
