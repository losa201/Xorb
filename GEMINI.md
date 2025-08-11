# XORB Cybersecurity Platform - GEMINI Context

## 📌 Project Overview
XORB is an enterprise-grade cybersecurity operations platform designed for penetration testing as a service (PTaaS), compliance validation, and advanced attack simulation. The system integrates security orchestration, compliance frameworks (NIST, CIS, ISO27001, SOC2), and real-time attack simulation capabilities with enhanced AI-powered intelligence.

**Key Components:**
- **Compliance Engine**: Implements security standards validation with real-time SIEM integration
- **Attack Simulation**: Real-world attack scenario generation with ML-powered threat analysis
- **SIEM Integration**: Security event monitoring and response with behavioral analytics
- **Orchestration**: Automated security testing workflows with AI optimization
- **Security Middleware**: Enhanced security layer with OWASP-compliant HTTP headers

## 🛠️ Building and Running
### Dependencies
- Python 3.10+
- Docker (for containerized services)
- SIEM integration components
- Environment configuration (.env files)

### Deployment
```bash
# Initial setup
chmod +x deploy.sh
./deploy.sh
```

### Service Activation
```bash
# Start core services
python3 activate_xorb_services.py

# Run compliance validation
python3 compliance_validation.py

# Launch attack simulation
python3 activate_attack_simulation.py

# Start security monitoring
python3 activate_siem_engine.py
```

## 🧪 Testing & Validation
- **Compliance Tests**: Run `compliance_validation.py` with specific framework parameters
- **Attack Simulation**: Execute `demonstrate_enhanced_capabilities.py` for scenario testing
- **Security Validation**: Use `security_validation_report.md` for audit findings
- **Audit Reports**: Review output in `AUDIT_REPORT.md` and `COMPREHENSIVE_SYSTEM_AUDIT.md`

## 📂 Directory Structure
```
/root/Xorb/
├── Security Frameworks
│   ├── compliance_template.py  # Base compliance implementation
│   ├── compliance_validation.py  # Framework validation logic
│   └── compliance_monitoring.py  # Real-time compliance monitoring
│
├── Attack Simulation
│   ├── activate_attack_simulation.py  # Core simulation engine
│   ├── attack_simulation_results.json  # Simulation output
│   └── container_exploitation.py  # Container security testing
│
├── Orchestration
│   ├── activate_xorb_services.py  # Service orchestrator
│   ├── demonstrate_implementation.py  # Workflow demonstrator
│   └── security_middleware.py  # Enhanced security layer
│
├── SIEM Integration
│   ├── activate_siem_engine.py  # SIEM interface
│   └── siem_integration.md  # Integration documentation
│
├── Documentation
│   ├── AUDIT_REPORT.md  # Security audit findings
│   ├── COMPREHENSIVE_SYSTEM_AUDIT.md  # Detailed system audit
│   └── security_policy.md  # Security requirements
│
└── Deployment
    ├── deploy.sh  # Deployment automation script
    └── .env.template  # Environment configuration
```

## 🧱 Development Conventions
- **Python Style**: PEP8-compliant with type hints
- **Compliance Implementation**: Class-based framework inheritance
- **Attack Simulation**: Scenario-driven with JSON output
- **Configuration**: Environment variables via `.env` files
- **Logging**: Structured JSON logging across components
- **Security**: OWASP Top 10 compliant security headers
- **AI Integration**: Machine learning threat analysis with PyTorch

## 📚 Key Documentation
1. `AUDIT_REPORT.md` - Security audit findings
2. `COMPREHENSIVE_SYSTEM_AUDIT.md` - Detailed system architecture review
3. `security_validation_report.md` - Security testing results
4. `CLAUDE.md` - Security implementation guidelines
5. `docs/api/` - Complete API documentation
6. `docs/architecture/` - Technical architecture details

## ⚠️ Security Considerations
- All components require strict access controls
- Attack simulation should only be run in isolated environments
- Compliance validation requires up-to-date framework definitions
- SIEM integration needs secure authentication configuration
- Security middleware enforces OWASP-recommended HTTP headers
- Multi-tenant architecture with complete data isolation

## 🔄 Workflow Integration
1. Deploy infrastructure with `deploy.sh`
2. Start core services via `activate_xorb_services.py`
3. Run compliance checks against target systems
4. Execute attack simulations for validation
5. Analyze results in JSON output and audit reports
6. Monitor security events through SIEM integration

## 🧠 AI-Powered Capabilities
- Machine learning threat intelligence with 87%+ accuracy
- Behavioral analytics with multi-algorithm approach
- Threat hunting with natural language query translation
- Predictive analytics for risk forecasting
- ML-powered vulnerability correlation

## 🏢 Enterprise Features
- Multi-tenant architecture with row-level security
- Advanced authentication (JWT, RBAC, MFA)
- Rate limiting and audit logging
- Industry-specific solutions (PCI-DSS, HIPAA, ISO27001)
- Advanced reporting with interactive dashboards

This context document should be used as the foundation for all future development, testing, and operational activities within the XORB platform.