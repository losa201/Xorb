# XORB Supreme Enhanced Edition

🚀 **AI-Augmented Red Team & Bug Bounty Orchestration System**

A cutting-edge cybersecurity platform combining machine learning, advanced automation, and enterprise-grade security for professional red team operations and bug bounty hunting.

## 🚀 Enhanced Features

### 🧠 ML-Powered Intelligence
- **XGBoost Target Prioritization**: Machine learning models predict target value and success probability  
- **Adaptive Campaign Management**: Real-time strategy adjustment based on performance metrics
- **ROI Optimization**: Intelligent bounty program analysis and submission automation
- **Vector-Enhanced Knowledge Base**: Semantic search with Qdrant and sentence transformers
- **Multi-LLM Integration**: OpenRouter support with Claude, GPT-4, Gemini, and Kimi models
- **Pydantic AI Validation**: Structured, validated outputs for security research

### 🏗️ Event-Driven Architecture
- **Redis Streams Integration**: Real-time event processing for system coordination
- **Intelligent Load Balancing**: Dynamic agent allocation based on performance and resource availability
- **Multi-Engine Agent System**: ZAP, Nuclei, and stealth Playwright agents working in concert
- **Campaign Management**: Automated scheduling and resource allocation with ML optimization

### 🛡️ Production Security Hardening
- **Multi-Layer Authentication**: Session management, IP access control, and MFA support
- **Rate Limiting & DDoS Protection**: Token bucket algorithms with Redis backend
- **Comprehensive Audit Logging**: Security event tracking and compliance reporting
- **Data Encryption**: AES-256 encryption for sensitive data at rest and in transit
- **Rules of Engagement**: Advanced compliance and safety controls

### 🔗 Hybrid LLM Architecture
- **Local Model Fallback**: CPU-optimized local LLM for offline operations
- **Context-Aware Prompt Optimization**: Specialized prompts for different security contexts
- **Intelligent Model Selection**: Automatic failover between remote and local models
- **Cost Optimization**: Smart model routing and request batching

### 📊 Resource Optimization
- **Real-Time System Monitoring**: CPU, memory, disk, and network usage tracking
- **Intelligent Process Management**: Component health monitoring and auto-restart
- **Deployment Mode Optimization**: Environment-specific resource allocation
- **Auto-Scaling**: Dynamic resource allocation based on workload

### 💰 Advanced Monetization
- **HackerOne Integration**: Automated bug bounty submission with ROI analysis
- **CVSS Calculation**: Professional vulnerability scoring with ML enhancement
- **Earnings Tracking**: Advanced revenue and performance analytics
- **Bounty Intelligence**: ML-powered program analysis and prioritization

## 🏗 Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  ML Orchestrator │────│  Event Bus      │────│  Security Mgr   │
│  - XGBoost       │    │  - Redis Stream │    │  - Auth/AuthZ   │
│  - Adaptive Mgmt │    │  - Real-time    │    │  - Rate Limiting│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Vector Knowledge │    │ Multi-Engine    │    │ Hybrid LLM      │
│ - Qdrant Vector │    │ Agents          │    │ - Remote/Local  │
│ - Semantic Search│    │ - ZAP/Nuclei    │    │ - Prompt Opt    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Bounty Intel    │    │ Report Gen      │    │ Deploy Optimizer│
│ - ROI Analysis  │    │ - CVSS Scoring  │    │ - Resource Mgmt │
│ - Auto Submit   │    │ - Prof Reports  │    │ - Health Monitor│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- **OS**: Ubuntu 24.04 LTS (recommended) 
- **CPU**: 4-8 cores (minimum 2 cores)
- **RAM**: 8-16GB (minimum 4GB)
- **Storage**: 50GB SSD (minimum 10GB)
- **Python**: 3.12+
- **Network**: Broadband internet connection

### Installation

1. **System Dependencies**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.12 and dependencies
sudo apt install python3.12 python3.12-venv python3-pip redis-server postgresql-client -y

# Install Docker (optional, for containerized deployment)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

2. **Clone and Setup**
```bash
git clone <repository-url>
cd xorb

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate
```

3. **Install Core Dependencies**
```bash
# Install basic requirements
pip install -r requirements.txt

# Install ML dependencies (recommended for enhanced features)
pip install xgboost scikit-learn pandas numpy

# Install vector search dependencies (optional)
pip install qdrant-client sentence-transformers torch

# Install security dependencies (required for production)
pip install bcrypt PyJWT cryptography
```

4. **Configure Services**
```bash
# Start Redis server
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Start Qdrant vector database (optional, for semantic search)
docker run -p 6333:6333 qdrant/qdrant

# Install Playwright browsers
playwright install chromium
```

5. **Initialize System**
```bash
# Copy example configuration
cp config.example.json config.json

# Initialize knowledge base
python -m knowledge_fabric.core --init
```

### Configuration

Edit `config.json` with your settings:

```json
{
    "redis_url": "redis://localhost:6379/0",
    "database_url": "sqlite+aiosqlite:///./xorb_enhanced.db",
    "openrouter_api_key": "your_openrouter_api_key",
    "hackerone_api_key": "your_hackerone_api_key",
    "security_level": "production",
    "deployment_mode": "production",
    "enable_monitoring": true,
    "enable_ml": true,
    "enable_vector_search": true,
    "enable_bounty_intelligence": true,
    "log_level": "INFO",
    "components": {
        "orchestrator": {"enabled": true, "ml_enabled": true},
        "knowledge_fabric": {"enabled": true, "vector_enabled": true},
        "agent_manager": {"enabled": true, "multi_engine": true},
        "llm_client": {"enabled": true, "hybrid_mode": true},
        "bounty_intelligence": {"enabled": true},
        "security_manager": {"enabled": true},
        "deployment_optimizer": {"enabled": true},
        "event_bus": {"enabled": true},
        "dashboard": {"enabled": true}
    }
}
```

## 🎮 Usage

### Running XORB Supreme

```bash
# Production mode (full security and ML features)
python main.py --mode production --security production

# Development mode with demo campaign
python main.py --mode development --security development --demo

# Staging mode without monitoring dashboard
python main.py --mode staging --no-dashboard --config custom_config.json

# View all available options
python main.py --help
```

### Command Line Options

- `--config`: Specify custom configuration file
- `--mode`: Deployment mode (development/staging/production)
- `--security`: Security level (development/staging/production/high_security)  
- `--demo`: Create and run a demonstration campaign
- `--no-dashboard`: Disable the monitoring dashboard

### ML-Powered Campaign Management
```python
from orchestration.ml_orchestrator import IntelligentOrchestrator, CampaignPriority

# Initialize ML-enhanced orchestrator
orchestrator = IntelligentOrchestrator()
await orchestrator.start()

# Create intelligent campaign with ML target prioritization
campaign_id = await orchestrator.create_intelligent_campaign(
    name="Enterprise Security Assessment",
    targets=[
        {"hostname": "app.example.com", "ports": [80, 443, 8080]},
        {"hostname": "api.example.com", "ports": [443, 3000]}
    ],
    priority=CampaignPriority.HIGH,
    metadata={"ml_enabled": True, "adaptive": True}
)

# Start campaign with real-time adaptation
await orchestrator.start_campaign(campaign_id)

# Get ML-enhanced statistics
stats = await orchestrator.get_ml_orchestrator_stats()
print(f"Prediction accuracy: {stats['avg_prediction_accuracy']:.2%}")
```

### Vector-Enhanced Knowledge Management
```python
from knowledge_fabric.vector_fabric import VectorKnowledgeFabric
from knowledge_fabric.atom import KnowledgeAtom, AtomType

# Initialize vector-enhanced knowledge fabric
fabric = VectorKnowledgeFabric()
await fabric.initialize()

# Semantic search for vulnerabilities
results = await fabric.semantic_search(
    query="SQL injection authentication bypass techniques",
    atom_type=AtomType.VULNERABILITY,
    limit=10,
    score_threshold=0.8
)

# Find related attack patterns
related = await fabric.find_related_atoms(results[0], limit=5)
print(f"Found {len(related)} related techniques")

# Cluster similar vulnerabilities
clusters = await fabric.cluster_atoms(
    atom_type=AtomType.VULNERABILITY,
    min_cluster_size=3
)
```

### Bounty Intelligence
```python
from integrations.bounty_intelligence import BountyIntelligenceEngine

# Initialize bounty intelligence
bounty_intel = BountyIntelligenceEngine()
await bounty_intel.initialize()

# Analyze program ROI
analysis = await bounty_intel.analyze_program_value(
    program_handle="example-corp"
)

print(f"Expected ROI: ${analysis['expected_roi']:.2f}")
print(f"Risk score: {analysis['risk_assessment']['overall_risk']}")

# Get prioritized programs
programs = await bounty_intel.get_prioritized_programs(limit=10)
for program in programs:
    print(f"{program['name']}: ${program['roi_score']:.2f} expected value")
```

### Multi-Engine Agent System
```python
from agents.multi_engine_agents import MultiEngineAgentManager

# Initialize agent manager with multiple engines
agent_manager = MultiEngineAgentManager()
await agent_manager.initialize()

# Create coordinated scan
scan_results = await agent_manager.execute_coordinated_scan(
    target="https://example.com",
    engines=["zap", "nuclei", "playwright"],
    coordination_strategy="parallel"
)

# Get performance metrics
performance = await agent_manager.get_performance_metrics()
print(f"ZAP findings: {performance['zap']['findings_count']}")
print(f"Nuclei coverage: {performance['nuclei']['template_coverage']}")
```

### Report Generation
```python
from reports.report_generator import ReportGenerator, SecurityReport, Finding

# Create report
metadata = ReportMetadata(
    title="Security Assessment Report",
    campaign_id="campaign-123",
    targets=["example.com"],
    executive_summary="Comprehensive security assessment findings"
)

report = SecurityReport(metadata)

# Add findings
finding = Finding(
    title="SQL Injection Vulnerability",
    description="Authentication bypass via SQL injection",
    severity="high",
    cvss_score=8.1,
    affected_targets=["example.com"],
    remediation="Implement parameterized queries"
)

report.add_finding(finding)

# Generate reports
generator = ReportGenerator()
files = generator.generate_report(report, [ReportFormat.MARKDOWN, ReportFormat.JSON])
```

## 🔧 Components

### Orchestrator
- **Campaign Scheduling**: Intelligent workload distribution
- **Agent Management**: Dynamic agent assignment and monitoring  
- **Compliance**: Rules of engagement enforcement
- **Audit Logging**: Comprehensive security logging

### Knowledge Fabric
- **Multi-tier Storage**: Hot (Redis), Warm (SQLite), Cold (S3) storage
- **ML Prediction**: Confidence scoring and success prediction
- **Relationship Mapping**: Connected security intelligence
- **Auto-validation**: LLM-powered knowledge verification

### Agents
- **Playwright Agent**: Advanced web application testing
- **Reconnaissance Agent**: Network and DNS discovery
- **Vulnerability Scanner**: Automated security testing
- **Post-Exploitation**: Persistence and privilege escalation

### LLM Integration  
- **Structured Outputs**: Pydantic AI validation
- **Multi-Provider**: OpenRouter with multiple models
- **Knowledge Enhancement**: Continuous intelligence gathering
- **Cost Tracking**: Request and spending monitoring

### Reporting
- **CVSS Calculation**: Professional vulnerability scoring
- **Multiple Formats**: Markdown, JSON, HTML, PDF
- **Template System**: Customizable report templates
- **Evidence Management**: Automated proof collection

### Monitoring
- **Real-time Dashboard**: Terminal-based monitoring
- **Prometheus Metrics**: Industry-standard observability
- **Health Checks**: Component status monitoring
- **Alerting**: Configurable notification system

## 🛡️ Security & Compliance

### Rules of Engagement
- **Scope Validation**: Automatic target verification
- **Rate Limiting**: Respectful testing practices
- **robots.txt Compliance**: Automated compliance checking
- **Approval Workflows**: Manual review for sensitive operations

### Audit & Logging
- **Signed Logs**: Cryptographic integrity verification
- **Structured Events**: JSON-formatted audit trails
- **Compliance Reporting**: SOX/PCI/ISO 27001 ready
- **Evidence Chain**: Forensic-grade evidence handling

### Data Protection
- **Encryption**: At-rest and in-transit encryption
- **Access Control**: Role-based permission system
- **Data Retention**: Configurable retention policies
- **Anonymization**: PII protection capabilities

## 📊 Monitoring & Observability

### Metrics Available
- Campaign success rates and timing
- Agent performance and utilization
- Knowledge base growth and quality
- API usage and cost tracking
- System resource utilization

### Dashboard Features
- Real-time system status
- Campaign progress tracking
- Agent activity monitoring
- Knowledge base statistics
- Cost and earnings tracking

### Health Checks
```bash
# System health check
poetry run python monitoring/dashboard.py --health

# Prometheus metrics
poetry run python monitoring/dashboard.py --prometheus
```

## 🔌 Integrations

### HackerOne
- **Program Discovery**: Automatic scope identification
- **Report Submission**: Automated vulnerability reporting
- **Status Tracking**: Report lifecycle management
- **Earnings Integration**: Revenue tracking and analytics

### OpenRouter LLM
- **Multi-Model Support**: Claude, GPT-4, Gemini, Kimi
- **Structured Outputs**: Pydantic validation
- **Cost Optimization**: Model selection and request batching
- **Rate Limiting**: Respectful API usage

### GitHub Actions
- **CI/CD Pipeline**: Automated testing and deployment
- **Security Scanning**: Integrated vulnerability detection
- **Report Generation**: Automated documentation
- **Artifact Management**: Build and deployment artifacts

## 🧪 Testing

### Run Test Suite
```bash
# Full test suite
poetry run pytest

# With coverage
poetry run pytest --cov=xorb --cov-report=html

# Specific component tests
poetry run pytest tests/test_orchestrator.py -v
```

### Integration Tests
```bash
# Test knowledge fabric
poetry run python -m knowledge_fabric.core

# Test agent functionality  
poetry run python -m agents.playwright_agent

# Test LLM integration
poetry run python -m integrations.openrouter_client
```

## 📈 Performance

### Scalability
- **Horizontal Scaling**: Multi-instance orchestration
- **Resource Management**: Intelligent resource allocation
- **Cloud Bursting**: Automatic cloud resource utilization
- **Load Balancing**: Distributed campaign processing

### Optimization
- **Knowledge Caching**: Multi-tier caching strategy
- **Request Batching**: Efficient API utilization
- **Lazy Loading**: On-demand resource initialization
- **Connection Pooling**: Optimized database connections

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
poetry install --with dev

# Install pre-commit hooks
pre-commit install

# Run linting
poetry run black . && poetry run flake8
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [https://docs.xorb.ai](https://docs.xorb.ai)
- **Issues**: [GitHub Issues](https://github.com/your-org/xorb/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/xorb/discussions)
- **Security**: security@xorb.ai

## 🏆 Acknowledgments

- **OpenRouter**: Multi-LLM API platform
- **HackerOne**: Bug bounty platform integration
- **Playwright**: Browser automation framework
- **Pydantic AI**: Structured LLM outputs
- **FastAPI**: High-performance API framework

---

**⚡ Built for security professionals, by security professionals**

*XORB combines the power of AI with proven security testing methodologies to create the most advanced red team orchestration platform available today.*