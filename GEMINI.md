# XORB Cybersecurity Platform - GEMINI Context

## 📌 Project Overview
XORB is an enterprise-grade cybersecurity operations platform designed for penetration testing as a service (PTaaS), compliance validation, and advanced attack simulation. The system integrates security orchestration, compliance frameworks (NIST, CIS, ISO27001, SOC2), and real-time attack simulation capabilities.

**Key Components:**
- **Compliance Engine**: Implements security standards validation
- **Attack Simulation**: Real-world attack scenario generation
- **SIEM Integration**: Security event monitoring and response
- **Orchestration**: Automated security testing workflows

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
```

## 🧪 Testing & Validation
- **Compliance Tests**: Run `compliance_validation.py` with specific framework parameters
- **Attack Simulation**: Execute `demonstrate_enhanced_capabilities.py` for scenario testing
- **Audit Reports**: Review output in `AUDIT_REPORT.md` and `COMPREHENSIVE_SYSTEM_AUDIT.md`

## 📂 Directory Structure
```
/root/Xorb/
├── Security Frameworks
│   ├── compliance_template.py  # Base compliance implementation
│   └── compliance_validation.py  # Framework validation logic
│
├── Attack Simulation
│   ├── activate_attack_simulation.py  # Core simulation engine
│   └── attack_simulation_results.json  # Simulation output
│
├── Orchestration
│   ├── activate_xorb_services.py  # Service orchestrator
│   └── demonstrate_implementation.py  # Workflow demonstrator
│
├── SIEM Integration
│   └── activate_siem_engine.py  # SIEM interface
│
├── Documentation
│   ├── AUDIT_REPORT.md  # Security audit findings
│   └── COMPREHENSIVE_SYSTEM_AUDIT.md  # Detailed system audit
│
└── Deployment
    └── deploy.sh  # Deployment automation script
```

## 🧱 Development Conventions
- **Python Style**: PEP8-compliant with type hints
- **Compliance Implementation**: Class-based framework inheritance
- **Attack Simulation**: Scenario-driven with JSON output
- **Configuration**: Environment variables via `.env` files
- **Logging**: Structured JSON logging across components

## 📚 Key Documentation
1. `AUDIT_REPORT.md` - Security audit findings
2. `COMPREHENSIVE_SYSTEM_AUDIT.md` - Detailed system architecture review
3. `CLAUDE.md` - Security implementation guidelines

## ⚠️ Security Considerations
- All components require strict access controls
- Attack simulation should only be run in isolated environments
- Compliance validation requires up-to-date framework definitions
- SIEM integration needs secure authentication configuration

## 🔄 Workflow Integration
1. Deploy infrastructure with `deploy.sh`
2. Start core services via `activate_xorb_services.py`
3. Run compliance checks against target systems
4. Execute attack simulations for validation
5. Analyze results in JSON output and audit reports

This context document should be used as the foundation for all future development, testing, and operational activities within the XORB platform.