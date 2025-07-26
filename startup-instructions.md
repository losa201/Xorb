# 🚀 XORB Phase 12 Deployment Instructions

## Quick Start

### 1. Prerequisites
```bash
# Ensure Docker and Docker Compose are installed
docker --version
docker-compose --version

# Create logs directory
mkdir -p logs
chmod 755 logs
```

### 2. Launch XORB Phase 12 System
```bash
# Build and start all services
docker-compose -f docker-compose.xorb-phase12.yml up --build -d

# Check all services are running
docker-compose -f docker-compose.xorb-phase12.yml ps

# View logs from all agents
docker-compose -f docker-compose.xorb-phase12.yml logs -f
```

### 3. Verify Services

#### Check Service Health
```bash
# Check Redis
docker exec xorb-redis redis-cli ping

# Check PostgreSQL
docker exec xorb-postgres pg_isready -U xorb_user -d xorb

# Check Prometheus metrics for each agent
curl http://localhost:8000/metrics  # Orchestrator
curl http://localhost:8001/metrics  # Evolutionary Defense
curl http://localhost:8002/metrics  # Threat Propagation
curl http://localhost:8003/metrics  # Autonomous Response
curl http://localhost:8004/metrics  # Ecosystem Integration
```

#### Access Monitoring Dashboards
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/xorb_admin_2023)
- **Redis Commander**: http://localhost:8081

### 4. Test Redis Pub/Sub Communication

#### Test High Priority Threat Signal
```bash
# Connect to Redis CLI
docker exec -it xorb-redis redis-cli

# Send a critical threat signal to trigger autonomous response
LPUSH high_priority_threats '{"signal_id":"test_001","threat_type":"malware","priority":"critical","confidence":0.85,"source_ip":"192.168.1.100","target_assets":["web-server-01","db-server-02"],"indicators":["malicious.com","abc123def456"],"timestamp":"2025-01-26T12:00:00Z","context":{"detection_method":"signature","attack_vector":"network"}}'

# Monitor the coordination channel
SUBSCRIBE xorb:coordination
```

#### Test Agent Coordination
```bash
# In Redis CLI, publish coordination message
PUBLISH xorb:coordination '{"type":"test_message","from_agent":"test","to_agent":"autonomous-response-001","timestamp":"2025-01-26T12:00:00Z"}'

# Check agent status updates
SUBSCRIBE xorb:agent_status
```

### 5. Monitor Agent Activity

#### View Real-time Logs
```bash
# Autonomous Response Agent
docker logs -f xorb-autonomous-response

# Orchestrator Agent
docker logs -f xorb-orchestrator

# All agents
docker-compose -f docker-compose.xorb-phase12.yml logs -f orchestrator-agent evolutionary-defense-agent threat-propagation-agent autonomous-response-agent
```

#### Check Database Activity
```bash
# Connect to PostgreSQL
docker exec -it xorb-postgres psql -U xorb_user -d xorb

# Check audit logs
SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT 10;

# Check agent registry
SELECT agent_id, agent_type, status, last_heartbeat FROM agent_registry;

# Check active response executions
SELECT execution_id, signal_id, status, created_at FROM response_executions WHERE status = 'executing';
```

### 6. Performance Testing

#### Generate Simulated Threats
```bash
# The xorb-simulator container automatically generates threat signals
# Check simulator logs
docker logs -f xorb-simulator

# Manually trigger critical threats
docker exec xorb-simulator python -c "
import asyncio
import json
import aioredis
import time
from datetime import datetime

async def send_critical_threat():
    redis = aioredis.Redis.from_url('redis://redis:6379')
    signal = {
        'signal_id': f'critical_{int(time.time())}',
        'threat_type': 'ransomware',
        'priority': 'critical',
        'confidence': 0.95,
        'source_ip': '203.0.113.50',
        'target_assets': ['file-server-01', 'backup-server-02'],
        'indicators': ['ransomware.exe', 'malicious-domain.net'],
        'timestamp': datetime.utcnow().isoformat(),
        'context': {'attack_vector': 'email', 'campaign': 'test'}
    }
    await redis.lpush('high_priority_threats', json.dumps(signal))
    print('Critical threat signal sent')

asyncio.run(send_critical_threat())
"
```

#### Monitor Metrics
```bash
# Query Prometheus for key metrics
curl -s 'http://localhost:9090/api/v1/query?query=xorb_response_actions_total' | jq
curl -s 'http://localhost:9090/api/v1/query?query=xorb_threat_signals_processed_total' | jq
curl -s 'http://localhost:9090/api/v1/query?query=xorb_active_responses' | jq

# Get response stage durations
curl -s 'http://localhost:9090/api/v1/query?query=xorb_response_stage_duration_seconds' | jq
```

### 7. Troubleshooting

#### Common Issues

**Services not starting:**
```bash
# Check Docker resource limits
docker system df
docker system prune -f

# Rebuild images
docker-compose -f docker-compose.xorb-phase12.yml build --no-cache
```

**Database connection issues:**
```bash
# Check PostgreSQL logs
docker logs xorb-postgres

# Verify database initialization
docker exec xorb-postgres psql -U xorb_user -d xorb -c "SELECT COUNT(*) FROM agent_registry;"
```

**Redis connection issues:**
```bash
# Check Redis status
docker exec xorb-redis redis-cli info server

# Test Redis commands
docker exec xorb-redis redis-cli set test_key test_value
docker exec xorb-redis redis-cli get test_key
```

**Agent health check failures:**
```bash
# Check agent container status
docker-compose -f docker-compose.xorb-phase12.yml ps

# View agent startup logs
docker logs xorb-autonomous-response

# Test metrics endpoint manually
docker exec xorb-autonomous-response curl -f http://localhost:8003/metrics
```

### 8. Scaling and Configuration

#### Environment Variables
```bash
# Create .env file for custom configuration
cat > .env << EOF
REDIS_URL=redis://redis:6379
POSTGRES_URL=postgresql://xorb_user:xorb_secure_pass_2023@postgres:5432/xorb
LOG_LEVEL=INFO
MIN_CONFIDENCE_THRESHOLD=0.72
MAX_CONCURRENT_RESPONSES=10
SIMULATION_INTERVAL=30
THREAT_GENERATION_RATE=0.1
EOF
```

#### Scale Services
```bash
# Scale specific agents
docker-compose -f docker-compose.xorb-phase12.yml up --scale autonomous-response-agent=3 -d

# Update resource limits
docker-compose -f docker-compose.xorb-phase12.yml config
```

### 9. Production Considerations

#### Security Hardening
```bash
# Use production-grade secrets
openssl rand -base64 32  # Generate secure passwords

# Enable TLS for Redis
# Enable SSL for PostgreSQL
# Use proper firewall rules
```

#### Monitoring Setup
```bash
# Configure Grafana dashboards
# Set up alerting rules in Prometheus
# Configure log aggregation
```

### 10. Shutdown

#### Graceful Shutdown
```bash
# Stop all services gracefully
docker-compose -f docker-compose.xorb-phase12.yml down

# Remove volumes (WARNING: This deletes all data)
docker-compose -f docker-compose.xorb-phase12.yml down -v

# Clean up Docker resources
docker system prune -f
```

## Key Endpoints

| Service | Port | Endpoint | Purpose |
|---------|------|----------|---------|
| Orchestrator | 8000 | /metrics | Prometheus metrics |
| Evolutionary Defense | 8001 | /metrics | Prometheus metrics |
| Threat Propagation | 8002 | /metrics | Prometheus metrics |
| Autonomous Response | 8003 | /metrics | Prometheus metrics |
| Ecosystem Integration | 8004 | /metrics | Prometheus metrics |
| Prometheus | 9090 | / | Metrics dashboard |
| Grafana | 3000 | / | Visualization dashboard |
| Redis Commander | 8081 | / | Redis management |

## Success Criteria

✅ **All services start and pass health checks**  
✅ **Redis pub/sub communication active between agents**  
✅ **Prometheus metrics being emitted from all agents**  
✅ **Critical threat signals trigger autonomous responses**  
✅ **Multi-stage response execution (isolation → collaboration → patching)**  
✅ **Audit trail logging to PostgreSQL**  
✅ **Real-time monitoring dashboards functional**

The XORB Phase 12 autonomous security system is now operational and ready for production use!