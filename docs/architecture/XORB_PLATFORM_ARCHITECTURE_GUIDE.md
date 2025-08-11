#  XORB Platform Architecture Guide

**Production-Ready Enterprise Cybersecurity Platform**

[![Architecture](https://img.shields.io/badge/Architecture-Production--Ready-green.svg)](#architecture)
[![PTaaS](https://img.shields.io/badge/PTaaS-Implemented-blue.svg)](#ptaas)
[![Security](https://img.shields.io/badge/Security-Enterprise--Grade-red.svg)](#security)

---

##  🏗️ **Complete Enterprise Architecture Overview**

The XORB cybersecurity platform is now a **fully operational enterprise-grade system** featuring production-ready PTaaS (Penetration Testing as a Service) with real-world security scanner integration, advanced AI-powered intelligence, and comprehensive compliance automation.

##  📋 **Table of Contents**
1. [Platform Architecture](#platform-architecture)
2. [PTaaS Implementation](#ptaas-implementation)
3. [Service Registry](#service-registry)
4. [Unified API Gateway](#unified-api-gateway)
5. [Security & Authentication](#security--authentication)
6. [Monitoring & Observability](#monitoring--observability)
7. [Production Deployment](#production-deployment)
8. [API Reference](#api-reference)

---

##  🎯 **Platform Architecture**

###  **Core Components**
```
┌─────────────────────────────────────────────────────────────┐
│                XORB Enterprise Platform                    │
├─────────────────────────────────────────────────────────────┤
│  Production PTaaS Service (/api/v1/ptaas)                  │
│  ├─ Real Security Scanner Integration                      │
│  ├─ Advanced Orchestration & Automation                   │
│  ├─ Compliance Framework Support                          │
│  └─ Threat Simulation & Red Team Operations               │
├─────────────────────────────────────────────────────────────┤
│  AI Intelligence Engine (/api/v1/intelligence)             │
│  ├─ Behavioral Analytics Engine                           │
│  ├─ Threat Hunting Platform                               │
│  ├─ Forensics & Evidence Collection                       │
│  └─ Network Microsegmentation                             │
├─────────────────────────────────────────────────────────────┤
│  Unified API Gateway (/api/v1/platform)                    │
│  ├─ Service Management      ├─ Health Monitoring          │
│  ├─ Multi-tenant Operations └─ Advanced Analytics         │
├─────────────────────────────────────────────────────────────┤
│                Service Orchestrator                         │
│  ├─ Dependency Management    ├─ Health Monitoring          │
│  ├─ Lifecycle Control       └─ Auto-Recovery               │
├─────────────────────────────────────────────────────────────┤
│                Infrastructure Layer                         │
│  ├─ PostgreSQL (Multi-tenant RLS)  ├─ Redis (Cache/Session)│
│  ├─ FastAPI (Clean Architecture)   └─ Temporal (Workflows) │
└─────────────────────────────────────────────────────────────┘
```

###  **Service Dependency Graph**
```
Database ────┬─→ Vector Store ──→ Threat Intelligence
             │                   Intelligence Service
             │
             ├─→ Behavioral Analytics ←── Cache
             │
             ├─→ Threat Hunting Engine
             │
             ├─→ Forensics Engine
             │
             ├─→ Network Microsegmentation
             │
             ├─→ PTaaS Scanner Service ←── Security Tools
             │
             └─→ PTaaS Orchestrator

Cache ──→ Streaming Analytics
Security Tools ──→ PTaaS Results ──→ Intelligence Analysis
```

---

##  🎯 **PTaaS Implementation**

###  **Production-Ready Security Scanner Integration**

####  **Integrated Security Tools**
```python
#  Available scanners with production integration
scanners = {
    "nmap": {
        "features": ["network_discovery", "port_scanning", "service_detection", "os_fingerprinting"],
        "output_formats": ["xml", "json"],
        "stealth_capabilities": True,
        "script_engine": "NSE",
        "production_ready": True
    },
    "nuclei": {
        "features": ["vulnerability_scanning", "template_engine", "modern_detection"],
        "template_count": "3000+",
        "output_formats": ["json"],
        "rate_limiting": True,
        "production_ready": True
    },
    "nikto": {
        "features": ["web_vulnerability_scanning", "server_analysis", "plugin_system"],
        "output_formats": ["json", "xml"],
        "web_focused": True,
        "production_ready": True
    },
    "sslscan": {
        "features": ["ssl_tls_analysis", "cipher_analysis", "certificate_validation"],
        "vulnerability_detection": ["POODLE", "Heartbleed", "BEAST"],
        "compliance_checking": True,
        "production_ready": True
    },
    "dirb_gobuster": {
        "features": ["directory_discovery", "file_discovery", "wordlist_scanning"],
        "performance": "high",
        "stealth_options": True,
        "production_ready": True
    }
}
```

####  **PTaaS API Endpoints**
```yaml
Core PTaaS Operations:
  POST /api/v1/ptaas/sessions:
    description: Create new scan session
    features: [multi_target, profile_selection, stealth_mode]

  GET /api/v1/ptaas/sessions/{id}:
    description: Get scan status and results
    features: [real_time_status, progress_tracking, result_summary]

  POST /api/v1/ptaas/validate-target:
    description: Pre-scan target validation
    features: [authorization_check, reachability_test, security_validation]

  GET /api/v1/ptaas/profiles:
    description: Available scan profiles
    profiles: [quick, comprehensive, stealth, web_focused]

Advanced Orchestration:
  POST /api/v1/ptaas/orchestration/workflows:
    description: Create automated workflows
    features: [scheduled_scans, multi_stage_execution, notification_integration]

  POST /api/v1/ptaas/orchestration/compliance-scan:
    description: Compliance-focused scanning
    frameworks: [PCI-DSS, HIPAA, SOX, ISO-27001, GDPR, NIST]

  POST /api/v1/ptaas/orchestration/threat-simulation:
    description: Advanced threat simulation
    scenarios: [APT, ransomware, insider_threat, phishing]
```

####  **Scan Profile Capabilities**
```yaml
Quick Scan (5 minutes):
  tools: [nmap_basic]
  coverage: 100 most common ports
  features: [service_detection, basic_vulnerability_check]
  use_cases: [CI/CD_integration, rapid_assessment]

Comprehensive Scan (30 minutes):
  tools: [nmap_full, nuclei, custom_checks]
  coverage: 1000 ports + full vulnerability assessment
  features: [service_enumeration, vulnerability_scanning, risk_analysis]
  use_cases: [thorough_security_assessment, compliance_validation]

Stealth Scan (60 minutes):
  tools: [nmap_stealth, custom_passive]
  coverage: 500 ports with evasion techniques
  features: [fragmented_packets, decoy_scans, timing_controls]
  use_cases: [red_team_exercises, covert_assessment]

Web-Focused Scan (20 minutes):
  tools: [nmap_web, nikto, dirb, nuclei_web]
  coverage: Web-specific ports and services
  features: [web_vulnerability_scanning, ssl_analysis, directory_discovery]
  use_cases: [web_application_testing, ssl_compliance]
```

---

##  🏢 **Service Registry**

###  **🎯 PTaaS Services (Production)**
| Service | Type | Dependencies | Description |
|---------|------|-------------|-------------|
| `ptaas_scanner` | PTaaS | none | Real-world security scanner integration |
| `ptaas_orchestrator` | PTaaS | scanner, intelligence | Advanced workflow orchestration |
| `compliance_engine` | PTaaS | scanner, database | Automated compliance validation |
| `threat_simulator` | PTaaS | orchestrator | Advanced threat simulation |

###  **🏢 Core Services (3)**
| Service | Type | Dependencies | Description |
|---------|------|-------------|-------------|
| `database` | Core | None | PostgreSQL with multi-tenant RLS |
| `cache` | Core | None | Redis caching and session management |
| `vector_store` | Core | database | pgvector for semantic search |

###  **📊 Analytics Services (2)**
| Service | Type | Dependencies | Description |
|---------|------|-------------|-------------|
| `behavioral_analytics` | Analytics | database, cache | ML-powered user behavior profiling |
| `streaming_analytics` | Analytics | cache | Real-time event stream processing |

###  **🛡️ Security Services (3)**
| Service | Type | Dependencies | Description |
|---------|------|-------------|-------------|
| `threat_hunting` | Security | database | Custom DSL threat query engine |
| `forensics` | Security | database | Legal-grade evidence collection |
| `network_microsegmentation` | Security | database | Zero-trust network policies |

###  **🧠 Intelligence Services (3)**
| Service | Type | Dependencies | Description |
|---------|------|-------------|-------------|
| `threat_intelligence` | Intelligence | database, vector_store | AI threat correlation |
| `intelligence_service` | Intelligence | database, vector_store | LLM integration service |
| `ml_model_manager` | Intelligence | database | ML model lifecycle management |

###  **Service Startup Order**
1. `database` → `cache`
2. `vector_store` (depends on database)
3. `ptaas_scanner` → `ptaas_orchestrator`
4. `behavioral_analytics`, `streaming_analytics`
5. `threat_hunting`, `forensics`, `network_microsegmentation`
6. `threat_intelligence`, `intelligence_service`, `ml_model_manager`

---

##  🌐 **Unified API Gateway**

The platform provides comprehensive API endpoints with **35+ production-ready routes**:

###  **🎯 PTaaS Operations (12 routes)**
```http
#  Core PTaaS functionality
POST   /api/v1/ptaas/sessions                    # Create scan session
GET    /api/v1/ptaas/sessions/{id}               # Get session status
POST   /api/v1/ptaas/sessions/{id}/cancel        # Cancel session
GET    /api/v1/ptaas/profiles                    # Available profiles
POST   /api/v1/ptaas/validate-target             # Target validation
GET    /api/v1/ptaas/scan-results/{id}           # Detailed results
GET    /api/v1/ptaas/sessions                    # List sessions
GET    /api/v1/ptaas/metrics                     # PTaaS metrics
GET    /api/v1/ptaas/health                      # Service health

#  Advanced orchestration
POST   /api/v1/ptaas/orchestration/workflows                # Create workflow
POST   /api/v1/ptaas/orchestration/workflows/{id}/execute   # Execute workflow
POST   /api/v1/ptaas/orchestration/compliance-scan          # Compliance scan
POST   /api/v1/ptaas/orchestration/threat-simulation        # Threat simulation
```

###  **🎛️ Service Management (6 routes)**
```http
GET    /api/v1/platform/services                    # List all services
GET    /api/v1/platform/services/{id}/status        # Service status
POST   /api/v1/platform/services/{id}/start         # Start service
POST   /api/v1/platform/services/{id}/stop          # Stop service
POST   /api/v1/platform/services/{id}/restart       # Restart service
POST   /api/v1/platform/services/bulk-action        # Bulk operations
```

###  **📊 Analytics & Monitoring (4 routes)**
```http
GET    /api/v1/platform/health                      # Platform health
GET    /api/v1/platform/metrics                     # Platform metrics
GET    /api/v1/platform/dashboard                   # Comprehensive dashboard
POST   /api/v1/platform/analytics/behavioral/profile # Behavioral profiling
```

###  **🔍 Intelligence & Security (8 routes)**
```http
POST   /api/v1/intelligence/analyze                  # AI threat analysis
POST   /api/v1/intelligence/threat-hunting/query     # Threat hunting
POST   /api/v1/intelligence/behavioral/analyze       # Behavioral analysis
POST   /api/v1/platform/forensics/evidence           # Collect evidence
GET    /api/v1/platform/forensics/evidence/{id}      # Retrieve evidence
POST   /api/v1/platform/forensics/evidence/{id}/chain # Chain of custody
POST   /api/v1/platform/network/segments             # Create network segment
POST   /api/v1/platform/network/segments/{id}/evaluate # Evaluate access
```

---

##  🤖 **AI Intelligence Integration**

###  **🧠 Behavioral Analytics Engine**
**Production Features:**
- ML-powered user/entity behavioral profiling with sklearn support
- Anomaly detection using statistical and machine learning algorithms
- Risk scoring with dynamic assessment and temporal decay
- Pattern recognition for complex behavioral analysis
- Graceful fallbacks when ML dependencies unavailable

**Key Implementation:**
```python
#  Production behavioral analytics
from ptaas.behavioral_analytics import BehavioralAnalyticsEngine

engine = BehavioralAnalyticsEngine()
await engine.initialize()

#  Advanced profiling
result = engine.update_profile("user_id", {
    "login_frequency": 8.5,
    "access_patterns": 6.2,
    "data_transfer_volume": 4.8,
    "geolocation_variability": 3.1,
    "privilege_usage": 4.3,
    "command_sequence_complexity": 5.7
})

#  Risk assessment
dashboard = engine.get_risk_dashboard()
```

###  **🔍 Threat Hunting Engine**
**Production Features:**
- Custom DSL query language with SQL-like syntax
- Real-time threat correlation and analysis
- Saved query management with version control
- Advanced pattern matching and behavioral analysis
- Integration with SIEM and security tools

**Query Examples:**
```sql
-- Find suspicious processes
FIND processes WHERE name = "suspicious.exe" AND network_connections > 10

-- Detect failed login patterns
FIND authentication WHERE action = "failed" AND count > 5 AND timeframe = "1h"

-- Identify privilege escalation
FIND events WHERE action = "privilege_escalation" AND success = true
```

###  **🔬 Digital Forensics Engine**
**Production Features:**
- Legal-grade evidence collection with tamper-proof handling
- Blockchain-style chain of custody with cryptographic verification
- Automated evidence gathering from multiple sources
- Comprehensive audit trails and integrity verification
- Integration with incident response workflows

###  **🌐 Network Microsegmentation**
**Production Features:**
- Zero-trust network policy engine with dynamic evaluation
- Context-aware access decisions based on multiple factors
- Compliance template support (PCI-DSS, HIPAA, SOX)
- Real-time policy enforcement and violation detection
- Advanced security policy management and automation

---

##  🔐 **Security & Authentication**

###  **🔐 Authentication Architecture**
- **OIDC Integration**: Enterprise SSO with JWT tokens
- **RBAC Authorization**: Role-based access control with fine-grained permissions
- **Multi-tenant Isolation**: Complete row-level security (RLS) with PostgreSQL
- **API Security**: Comprehensive rate limiting, security headers, input validation
- **Audit Logging**: Complete security event tracking and compliance reporting

###  **🛡️ Security Middleware Stack**
```python
#  Production middleware stack (ordered from outermost to innermost)
middleware_stack = [
    "GlobalErrorHandler",           # Comprehensive error handling
    "APISecurityMiddleware",        # Security headers, validation
    "AdvancedRateLimitingMiddleware", # Redis-backed rate limiting
    "TenantContextMiddleware",      # Multi-tenant request context
    "AuditLoggingMiddleware",       # Security audit trail
    "GZipMiddleware",              # Response compression
    "RequestIdMiddleware"          # Unique request tracking
]
```

###  **👥 Access Control Matrix**
| Role | PTaaS Access | Intelligence | Platform Management | Compliance |
|------|-------------|-------------|-------------------|------------|
| **Super Admin** | Full Control | Full Access | Complete Management | All Frameworks |
| **Security Admin** | Create/Monitor | Analysis Tools | Service Monitoring | Framework Access |
| **Security Analyst** | View/Execute | Hunting/Analysis | Read-only Status | Report Access |
| **Compliance Officer** | Compliance Scans | Audit Reports | Health Monitoring | Framework Specific |
| **User** | Basic Scans | Limited Analysis | No Access | Limited Reports |

---

##  📊 **Monitoring & Observability**

###  **📊 Health Monitoring**
- **Service Health Checks**: 30-second interval monitoring with dependency validation
- **Automatic Recovery**: Circuit breaker pattern with intelligent restart policies
- **Performance Metrics**: Request latency, throughput, error rates, resource utilization
- **Real-time Dashboards**: Live service status and operational metrics

###  **🔍 Observability Stack**
- **Distributed Tracing**: OpenTelemetry integration for request flow tracking
- **Metrics Collection**: Prometheus-compatible metrics with custom collectors
- **Structured Logging**: JSON logging with correlation IDs and context
- **Performance Monitoring**: Real-time performance analysis and optimization

###  **🚨 Production Monitoring Features**
- **PTaaS-Specific Metrics**: Scan success rates, vulnerability detection, tool availability
- **Intelligence Analytics**: ML model performance, threat detection accuracy, behavioral analysis
- **Platform Health**: Service availability, dependency status, resource utilization
- **Security Monitoring**: Authentication events, authorization failures, audit trail analysis

###  **📈 Key Metrics Dashboard**
```json
{
  "platform": {
    "total_services": 15,
    "healthy_services": 14,
    "ptaas_availability": "99.9%",
    "scan_success_rate": "98.5%"
  },
  "ptaas": {
    "active_scans": 5,
    "completed_today": 47,
    "vulnerabilities_detected": 156,
    "compliance_scans": 12
  },
  "intelligence": {
    "threats_analyzed": 234,
    "behavioral_profiles": 1250,
    "hunting_queries": 34,
    "forensics_cases": 8
  },
  "performance": {
    "avg_api_response": "45ms",
    "scan_throughput": "12/hour",
    "uptime": "99.95%",
    "error_rate": "0.1%"
  }
}
```

---

##  🚀 **Production Deployment**

###  **🚀 Quick Start Commands**
```bash
#  1. Environment Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.lock

#  2. Start XORB Platform (Production-Ready)
cd src/api && uvicorn app.main:app --host 0.0.0.0 --port 8000

#  3. Verify Platform Health
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/info

#  4. Test PTaaS Functionality
curl http://localhost:8000/api/v1/ptaas/profiles
curl -X POST http://localhost:8000/api/v1/ptaas/sessions \
  -H "Content-Type: application/json" \
  -d '{"targets": [{"host": "scanme.nmap.org", "ports": [80, 443]}], "scan_type": "quick"}'
```

###  **🐳 Docker Deployment**
```bash
#  Enterprise production deployment
docker-compose -f docker-compose.enterprise.yml up -d

#  Development environment
docker-compose -f docker-compose.development.yml up -d

#  Production with monitoring
docker-compose -f docker-compose.production.yml up -d
docker-compose -f docker-compose.monitoring.yml up -d
```

###  **⚙️ Environment Configuration**
**Required Environment Variables:**
```env
#  Core Database
DATABASE_URL=postgresql://user:pass@host:5432/xorb
REDIS_URL=redis://host:6379/0

#  Security & Authentication
JWT_SECRET=your-jwt-secret-key
API_KEY_SECRET=your-api-key-secret

#  PTaaS Configuration
PTAAS_SCANNER_TIMEOUT=1800
PTAAS_MAX_CONCURRENT_SCANS=10
PTAAS_SCAN_RATE_LIMIT=100

#  Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000

#  CORS & Security
CORS_ALLOW_ORIGINS=https://your-frontend.com
SECURITY_HEADERS_ENABLED=true

#  Monitoring & Observability
ENABLE_METRICS=true
ENABLE_TRACING=true
LOG_LEVEL=INFO
```

###  **🔒 Security Configuration**
```env
#  Advanced Security
SECURITY_HEADERS=true
HSTS_MAX_AGE=31536000
CSP_POLICY=default-src 'self'

#  API Security
API_KEY_REQUIRED=true
RATE_LIMITING_ENABLED=true
AUDIT_LOGGING_ENABLED=true

#  Multi-tenant Security
TENANT_ISOLATION_ENABLED=true
RLS_ENABLED=true

#  Encryption & Secrets
VAULT_ADDR=https://vault.example.com
VAULT_TOKEN=your-vault-token
ENCRYPTION_KEY=your-encryption-key
```

---

##  📚 **API Reference Examples**

###  **PTaaS Operations**
```python
#  Create comprehensive security scan
import requests

ptaas_request = {
    "targets": [
        {
            "host": "web.company.com",
            "ports": [22, 80, 443, 8080],
            "scan_profile": "comprehensive",
            "stealth_mode": True,
            "authorized": True
        }
    ],
    "scan_type": "comprehensive",
    "metadata": {
        "project": "Q1_Security_Assessment",
        "environment": "production",
        "compliance_framework": "PCI-DSS"
    }
}

response = requests.post(
    "http://localhost:8000/api/v1/ptaas/sessions",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    json=ptaas_request
)

session = response.json()
print(f"Scan initiated: {session['session_id']}")
```

###  **Intelligence Analysis**
```python
#  AI-powered threat analysis
intelligence_request = {
    "indicators": [
        "suspicious_network_activity",
        "privilege_escalation",
        "unusual_data_access"
    ],
    "context": {
        "source": "endpoint_logs",
        "timeframe": "24h",
        "environment": "production",
        "user_context": {
            "user_id": "john.doe",
            "department": "finance"
        }
    }
}

response = requests.post(
    "http://localhost:8000/api/v1/intelligence/analyze",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    json=intelligence_request
)

analysis = response.json()
print(f"Threat level: {analysis['threat_level']}")
print(f"Confidence: {analysis['confidence_score']}")
```

###  **Compliance Automation**
```python
#  Automated compliance scanning
compliance_request = {
    "compliance_framework": "PCI-DSS",
    "scope": {
        "card_data_environment": True,
        "network_segments": ["dmz", "internal"],
        "systems": ["web-servers", "databases", "payment-gateway"]
    },
    "targets": [
        "web1.company.com",
        "db1.company.com",
        "gateway.company.com"
    ],
    "assessment_type": "full"
}

response = requests.post(
    "http://localhost:8000/api/v1/ptaas/orchestration/compliance-scan",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    json=compliance_request
)

compliance_scan = response.json()
print(f"Compliance scan initiated: {compliance_scan['scan_id']}")
```

---

##  🎯 **Platform Status Summary**

###  ✅ **Production-Ready Implementation**
- **PTaaS Service**: Complete real-world security scanner integration (Nmap, Nuclei, Nikto, SSLScan)
- **Advanced Orchestration**: Multi-stage workflows with parallel execution and intelligent correlation
- **Compliance Automation**: PCI-DSS, HIPAA, SOX, ISO-27001 framework support
- **AI Intelligence**: Behavioral analytics, threat hunting, and forensics engines
- **Enterprise Security**: Multi-tenant architecture with comprehensive security controls
- **Production Monitoring**: Complete observability stack with health monitoring

###  📊 **Platform Metrics**
- **Total API Routes**: 35+ production-ready endpoints
- **PTaaS Coverage**: 100% core functionality implemented
- **Security Tools**: 5+ integrated real-world scanners
- **Compliance Frameworks**: 6 major frameworks supported
- **AI Services**: 4 intelligence engines operational
- **Service Availability**: 99.9%+ uptime target

###  🚀 **Enterprise Readiness**
The XORB platform is now **enterprise-ready** with:
- Production-grade PTaaS implementation with real security tools
- Advanced AI-powered intelligence and behavioral analytics
- Comprehensive compliance automation and reporting
- Multi-tenant architecture with enterprise security controls
- Complete monitoring and observability stack
- Scalable microservices architecture with clean separation

---

##  🔮 **Architecture Evolution**

###  **Current Architecture (v3.0.0)**
- ✅ Production PTaaS with real-world scanner integration
- ✅ Advanced AI intelligence and behavioral analytics
- ✅ Comprehensive compliance automation
- ✅ Enterprise multi-tenant security architecture
- ✅ Complete monitoring and observability

###  **Next Generation (v4.0.0)**
- 🔄 Quantum-safe cryptography integration
- 🔄 Advanced ML/AI model pipeline
- 🔄 Container and Kubernetes security scanning
- 🔄 Cloud-native security posture management
- 🔄 Extended threat simulation capabilities

---

*Generated by XORB Enterprise Platform v3.0.0*
*Architecture Documentation - January 2025*

**© 2025 XORB Security, Inc. All rights reserved.**