# XORB Data Flows & Architecture

**Audit Date**: 2025-08-15
**Architecture**: Microservices with NATS JetStream messaging
**Data Stores**: PostgreSQL, SQLite, Redis (cache-only)

## System Context Diagram

```mermaid
graph TB
    User[Security Analyst] --> WebUI[PTaaS Web UI]
    API_Client[API Client] --> LB[Load Balancer]

    WebUI --> LB
    LB --> API[XORB API Service]

    API --> NATS[NATS JetStream]
    API --> DB[(PostgreSQL)]
    API --> Cache[(Redis Cache)]

    NATS --> Orchestrator[PTaaS Orchestrator]
    NATS --> Scanner[Scanner Services]
    NATS --> Intelligence[Intelligence Engine]

    Orchestrator --> Temporal[Temporal Workflows]
    Orchestrator --> JobDB[(SQLite Jobs)]

    Scanner --> Tools[Security Tools]
    Tools --> Nmap[Nmap]
    Tools --> Nuclei[Nuclei]
    Tools --> Nikto[Nikto]

    Intelligence --> External[External APIs]
    External --> NVIDIA[NVIDIA API]
    External --> OpenRouter[OpenRouter]

    API --> Evidence[Evidence Service G7]
    API --> ControlPlane[Control Plane G8]

    Evidence --> AuditDB[(Audit Storage)]
    ControlPlane --> Quotas[(Quota Management)]

    Prometheus[Prometheus] --> API
    Prometheus --> Orchestrator
    Prometheus --> Scanner

    Grafana[Grafana] --> Prometheus
```

## PTaaS End-to-End Workflow

```mermaid
sequenceDiagram
    participant User
    participant API
    participant NATS
    participant Orchestrator
    participant Scanner
    participant Evidence
    participant DB

    User->>API: POST /ptaas/sessions
    Note over API: Validate request, check quotas
    API->>DB: Store session metadata
    API->>Evidence: Emit G7 evidence (session_created)
    API->>NATS: Publish xorb.tenant123.ptaas.job.queued
    API-->>User: 201 Created {session_id}

    NATS->>Orchestrator: Consume job message
    Note over Orchestrator: WFQ scheduler assigns priority
    Orchestrator->>Evidence: Emit G7 evidence (job_started)
    Orchestrator->>NATS: Publish xorb.tenant123.ptaas.job.running

    Orchestrator->>Scanner: Dispatch scan tasks
    Note over Scanner: Execute nmap, nuclei, nikto
    Scanner->>Evidence: Emit G7 evidence (scan_executed)
    Scanner->>NATS: Publish xorb.tenant123.ptaas.findings

    Scanner->>Orchestrator: Task completion
    Orchestrator->>Evidence: Emit G7 evidence (job_completed)
    Orchestrator->>NATS: Publish xorb.tenant123.ptaas.job.completed

    API->>NATS: Consume completion message
    API->>DB: Update session status
    Note over API: Session marked as completed

    User->>API: GET /ptaas/sessions/{id}
    API->>DB: Query session data
    API-->>User: Session results + findings
```

## NATS Subject Topology

```mermaid
graph LR
    subgraph "NATS JetStream Subjects"
        subgraph "PTaaS Domain"
            JobQueued[xorb.*.ptaas.job.queued]
            JobRunning[xorb.*.ptaas.job.running]
            JobCompleted[xorb.*.ptaas.job.completed]
            JobFailed[xorb.*.ptaas.job.failed]
            JobAudit[xorb.*.ptaas.job.audit]
        end

        subgraph "Discovery Domain"
            DiscStarted[xorb.*.discovery.started]
            DiscCompleted[xorb.*.discovery.completed]
            DiscFailed[xorb.*.discovery.failed]
        end

        subgraph "Evidence Domain"
            EvidenceCollected[xorb.*.evidence.collected]
            EvidenceVerified[xorb.*.evidence.verified]
            EvidenceStored[xorb.*.evidence.stored]
        end

        subgraph "Alert Domain"
            SecurityAlerts[xorb.*.alerts.security]
            PerfAlerts[xorb.*.alerts.performance]
            ComplianceAlerts[xorb.*.alerts.compliance]
        end
    end

    API --> JobQueued
    Orchestrator --> JobRunning
    Orchestrator --> JobCompleted
    Scanner --> EvidenceCollected
    Intelligence --> SecurityAlerts
```

## Service Dependencies Flow

```mermaid
graph TD
    subgraph "External Dependencies"
        GitHub[GitHub API]
        NVIDIA[NVIDIA API]
        OpenRouter[OpenRouter API]
        Slack[Slack API]
    end

    subgraph "Infrastructure Layer"
        PostgreSQL[(PostgreSQL)]
        Redis[(Redis Cache)]
        NATS[NATS JetStream]
        Temporal[Temporal Server]
        Vault[HashiCorp Vault]
    end

    subgraph "Core Services"
        API[XORB API]
        Orchestrator[PTaaS Orchestrator]
        Scanner[Scanner Core]
        Intelligence[Intelligence Engine]
        Evidence[Evidence Service]
        ControlPlane[Control Plane]
    end

    subgraph "Observability"
        Prometheus[Prometheus]
        Grafana[Grafana]
        Jaeger[Jaeger Tracing]
    end

    API --> PostgreSQL
    API --> Redis
    API --> NATS
    API --> Vault

    Orchestrator --> NATS
    Orchestrator --> Temporal

    Scanner --> NATS
    Intelligence --> NATS
    Intelligence --> NVIDIA
    Intelligence --> OpenRouter

    Evidence --> PostgreSQL
    ControlPlane --> Redis

    API --> GitHub
    API --> Slack

    Prometheus --> API
    Prometheus --> Orchestrator
    Prometheus --> Scanner

    Grafana --> Prometheus
    Jaeger --> API
```

## PTaaS Job State Machine

```mermaid
stateDiagram-v2
    [*] --> Queued: Job submitted

    Queued --> Scheduled: WFQ scheduler selects
    Queued --> Failed: Validation error

    Scheduled --> Running: Orchestrator starts
    Scheduled --> Cancelled: User cancellation

    Running --> Paused: User pause request
    Running --> Completed: All tasks done
    Running --> Failed: Execution error
    Running --> Cancelled: User cancellation

    Paused --> Running: User resume
    Paused --> Cancelled: User cancellation

    Completed --> [*]: Final state
    Failed --> [*]: Final state
    Cancelled --> [*]: Final state

    note right of Running
        Emits G7 evidence at each transition
        Updates fairness metrics
        Publishes NATS events
    end note
```

## Evidence Chain Flow (G7)

```mermaid
sequenceDiagram
    participant Service
    participant Evidence
    participant Vault
    participant AuditDB
    participant Compliance

    Service->>Evidence: emit_evidence(event_type, payload)
    Note over Evidence: Generate chain link
    Evidence->>Vault: Sign with evidence key
    Evidence->>AuditDB: Store signed evidence

    Note over Evidence: Chain integrity validation
    Evidence->>Evidence: Validate previous hash
    Evidence->>Evidence: Compute new chain hash

    Evidence->>Compliance: Publish audit event
    Note over Compliance: SOC2, PCI-DSS compliance

    Service->>Evidence: verify_chain(session_id)
    Evidence->>AuditDB: Query chain links
    Evidence->>Vault: Verify signatures
    Evidence-->>Service: Chain validation result
```

## Weighted Fair Queueing (G8) Flow

```mermaid
graph TB
    subgraph "Tenant Queues"
        T1Q[Tenant 1 Queue]
        T2Q[Tenant 2 Queue]
        T3Q[Tenant 3 Queue]
    end

    subgraph "Priority Queues per Tenant"
        T1H[High Priority]
        T1M[Medium Priority]
        T1L[Low Priority]

        T2H[High Priority]
        T2M[Medium Priority]
        T2L[Low Priority]
    end

    subgraph "WFQ Scheduler"
        Scheduler[G8 WFQ Scheduler]
        VirtualTime[Virtual Time Calculator]
        Fairness[Fairness Index Monitor]
    end

    T1Q --> T1H
    T1Q --> T1M
    T1Q --> T1L

    T2Q --> T2H
    T2Q --> T2M
    T2Q --> T2L

    Scheduler --> VirtualTime
    Scheduler --> Fairness

    T1H --> Scheduler
    T1M --> Scheduler
    T1L --> Scheduler
    T2H --> Scheduler
    T2M --> Scheduler
    T2L --> Scheduler

    Scheduler --> WorkerPool[PTaaS Worker Pool]
    Fairness --> Metrics[Prometheus Metrics]
```

## Data Persistence Patterns

### PostgreSQL Schema
```mermaid
erDiagram
    USERS ||--o{ SESSIONS : owns
    TENANTS ||--o{ USERS : contains
    SESSIONS ||--o{ FINDINGS : produces
    SESSIONS ||--o{ EVIDENCE_CHAINS : generates

    USERS {
        uuid id PK
        string username
        string email
        jsonb roles
        timestamp created_at
        timestamp last_login
    }

    TENANTS {
        uuid id PK
        string name
        jsonb quotas
        jsonb settings
        timestamp created_at
    }

    SESSIONS {
        uuid id PK
        uuid user_id FK
        uuid tenant_id FK
        string status
        jsonb targets
        jsonb metadata
        timestamp created_at
        timestamp completed_at
    }

    FINDINGS {
        uuid id PK
        uuid session_id FK
        string severity
        string title
        text description
        jsonb technical_details
        timestamp discovered_at
    }

    EVIDENCE_CHAINS {
        uuid id PK
        uuid session_id FK
        string event_type
        text payload_hash
        text signature
        text previous_hash
        timestamp created_at
    }
```

### NATS JetStream Configuration
```yaml
streams:
  xorb-ptaas:
    subjects: ["xorb.*.ptaas.>"]
    retention: workqueue
    max_age: 7d
    storage: file
    replicas: 3

  xorb-evidence:
    subjects: ["xorb.*.evidence.>"]
    retention: limits
    max_age: 7y  # 7 year retention for compliance
    storage: file
    replicas: 3

  xorb-alerts:
    subjects: ["xorb.*.alerts.>"]
    retention: limits
    max_age: 30d
    storage: memory
```

## Performance Data Flows

### Metrics Collection
```mermaid
graph LR
    subgraph "Services"
        API[API Service]
        Orchestrator[Orchestrator]
        Scanner[Scanner]
    end

    subgraph "Metrics Pipeline"
        Prometheus[Prometheus]
        Grafana[Grafana]
        AlertManager[Alert Manager]
    end

    API --> |/metrics| Prometheus
    Orchestrator --> |/metrics| Prometheus
    Scanner --> |/metrics| Prometheus

    Prometheus --> Grafana
    Prometheus --> AlertManager

    AlertManager --> Slack[Slack Notifications]
    AlertManager --> Email[Email Alerts]
```

### Tracing Flow
```mermaid
graph TB
    Request[HTTP Request] --> API
    API --> |trace-id| NATS
    NATS --> |trace-id| Orchestrator
    Orchestrator --> |trace-id| Scanner

    API --> Jaeger[Jaeger Collector]
    Orchestrator --> Jaeger
    Scanner --> Jaeger

    Jaeger --> JaegerQuery[Jaeger Query UI]
    Jaeger --> Grafana[Grafana Tracing]
```

## Security Data Flows

### mTLS Certificate Flow
```mermaid
sequenceDiagram
    participant Service
    participant Vault
    participant CA
    participant Peer

    Service->>Vault: Request certificate
    Vault->>CA: Sign certificate request
    CA->>Vault: Return signed certificate
    Vault->>Service: Deliver certificate + key

    Service->>Peer: Initiate mTLS connection
    Service->>Peer: Present certificate
    Peer->>Service: Verify + present certificate
    Note over Service,Peer: Mutual authentication successful

    Service->>Vault: Certificate near expiry
    Vault->>CA: Generate new certificate
    Vault->>Service: Hot reload new certificate
```

### Secret Management Flow
```mermaid
graph TB
    subgraph "Secret Sources"
        EnvVars[Environment Variables]
        VaultSecrets[Vault Secrets]
        K8sSecrets[Kubernetes Secrets]
    end

    subgraph "Services"
        API[API Service]
        Orchestrator[Orchestrator]
        Scanner[Scanner]
    end

    VaultSecrets --> API
    VaultSecrets --> Orchestrator
    K8sSecrets --> Scanner

    EnvVars --> |Development| API

    subgraph "Secret Rotation"
        CronJob[Rotation Cron]
        VaultAgent[Vault Agent]
    end

    CronJob --> VaultAgent
    VaultAgent --> VaultSecrets
```

## Disaster Recovery Flows

### Backup Strategy
```mermaid
graph TB
    subgraph "Primary Data"
        PostgresMain[PostgreSQL Primary]
        NATSMain[NATS Primary]
        VaultMain[Vault Primary]
    end

    subgraph "Backup Systems"
        PostgresBackup[PostgreSQL Backup]
        NATSBackup[NATS Backup]
        VaultBackup[Vault Backup]
        S3[S3 Archive Storage]
    end

    subgraph "Recovery Process"
        RPO[RPO: 15 minutes]
        RTO[RTO: 1 hour]
    end

    PostgresMain --> |Streaming replication| PostgresBackup
    NATSMain --> |Cluster replication| NATSBackup
    VaultMain --> |Raft snapshots| VaultBackup

    PostgresBackup --> |Daily dumps| S3
    NATSBackup --> |Stream backups| S3
    VaultBackup --> |Encrypted snapshots| S3
```

---

*This data flow documentation provides comprehensive visibility into how data moves through the XORB platform, including messaging patterns, state transitions, evidence chains, and operational flows.*
