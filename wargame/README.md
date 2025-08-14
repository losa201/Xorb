# 🚀 XORB Red vs Blue Wargame

## Overview

The XORB Red vs Blue Wargame is a sophisticated continuous cybersecurity simulation where AI-powered Red and Blue Teams engage in realistic cyber warfare within a synthetic enterprise environment (Purple Team).

## 🎯 Wargame Architecture

### Teams

#### 🔴 Red Team (AI Adversary)
- **Role**: Sophisticated attacker with real-world TTPs
- **Capabilities**: 
  - Reconnaissance (OSINT, network scanning, service enumeration)
  - Initial Access (exploit public applications, cloud misconfigurations)
  - Credential Access (default accounts, memory dumping)
  - Lateral Movement (network scanning, privilege escalation)
  - Persistence (web shells, backdoor accounts)
  - Data Exfiltration (database dumps, cloud storage access)
- **Adaptation**: Learns from Blue Team responses and adjusts tactics

#### 🔵 Blue Team (AI Defender)
- **Role**: Advanced defense stack with detection and response
- **Capabilities**:
  - Prevention (vulnerability patching, access control, attack surface reduction)
  - Detection (network monitoring, WAF, database activity monitoring)
  - Deception (honeypots, canary tokens, decoy services)
  - Response (network isolation, enhanced monitoring, threat hunting)
  - Intelligence (IOC extraction, signature updates)
- **Resources**: Limited budget requiring strategic deployment decisions

#### 🟣 Purple Team (Synthetic Environment)
- **Role**: Realistic customer organization simulation
- **Organization**: Meridian Dynamics Corp (50 employees, 5 departments)
- **Infrastructure**:
  - Network topology (DMZ, internal VLANs, cloud services)
  - Applications (WordPress site, HR portal, customer portal, APIs)
  - Vulnerabilities (weak credentials, exposed endpoints, misconfigurations)
  - Monitoring and logging systems

## 🏗️ Environment Architecture

### Meridian Dynamics Corp Profile
- **Industry**: Technology Consulting
- **Size**: 50 employees across 5 departments
- **Infrastructure**: Hybrid cloud (AWS + on-premises)
- **Domains**: meridiandynamics.com, md-consulting.net

### Network Topology
```
External (203.0.113.50)
├── DMZ (192.168.1.0/24)
│   ├── Web Server (nginx 1.18.0)
│   └── Mail Server (postfix 3.4.13)
├── Internal LAN (10.0.0.0/16)
│   ├── Executive VLAN (10.0.10.0/24)
│   ├── HR VLAN (10.0.20.0/24)
│   ├── Engineering VLAN (10.0.30.0/24)
│   ├── Marketing VLAN (10.0.40.0/24)
│   └── Finance VLAN (10.0.50.0/24)
└── Cloud (AWS us-east-1)
    ├── S3 Bucket (meridian-docs-2024)
    └── RDS PostgreSQL (13.7)
```

### Initial Vulnerabilities
- **VULN-001**: Default admin credentials in HR Portal (admin/password123)
- **VULN-002**: Exposed debug endpoint in File Storage API
- **VULN-003**: Publicly readable S3 bucket with sensitive documents
- **VULN-004**: Outdated WordPress plugin with RCE vulnerability (CVE-2023-6000)

## 🎮 Wargame Execution

### Round Structure
1. **Setup Phase**: Purple environment initialization with seeded vulnerabilities
2. **Attack Phase**: Red Team reconnaissance, exploitation, and data exfiltration
3. **Defense Phase**: Blue Team detection, prevention, and response
4. **Assessment Phase**: Purple Team verification and impact assessment
5. **Evolution Phase**: Environment changes and adaptations

### Round 1 Results Summary

#### 🔴 Red Team Performance
- **Actions**: 18 total, 17 successful (94.4% success rate)
- **Compromised Assets**: 3 critical systems
  - AWS S3 Bucket (data exfiltration)
  - HR Portal (credential access)
  - Corporate Website (initial access)
- **Persistence**: ✅ Established (web shells, backdoor accounts)
- **Data Exfiltration**: ✅ Successful (847 documents, employee records)

#### 🔵 Blue Team Performance
- **Actions**: 21 defensive measures deployed
- **Detections**: 11 threats identified (61.1% detection rate)
- **Countermeasures**: 3 critical vulnerabilities patched
- **Resource Cost**: 67 units (high spending)
- **Prevention Score**: 42.9% effectiveness

#### 🟣 Environment Evolution
- **Risk Level**: HIGH
- **Defense Maturity**: INTERMEDIATE
- **Attack Surface**: Reduced after Blue Team interventions
- **Vulnerabilities Patched**: 3 of 4 initial vulnerabilities addressed

## 📊 Key Metrics

### Security Posture Assessment
- **Overall Risk**: HIGH (due to 94% Red Team success rate)
- **Attack Surface**: MEDIUM (reduced from large after patches)
- **Defense Maturity**: INTERMEDIATE (good detection, needs prevention improvement)

### Lessons Learned
- High attack success rate indicates need for improved preventive controls
- Persistence establishment requires enhanced endpoint detection and response
- Blue Team detection capabilities are effective but resource-intensive

### Recommendations
- Implement multi-factor authentication for administrative accounts
- Review and audit all cloud storage configurations
- Optimize defense spending for better cost-effectiveness

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- XORB platform environment

### Quick Start
```bash
# Navigate to wargame directory
cd /root/Xorb/wargame

# Run single demonstration round
python3 demo_round.py

# Run continuous wargame (3 rounds)
python3 wargame_orchestrator.py
```

### Directory Structure
```
wargame/
├── purple/
│   ├── environment_state.json    # Current environment configuration
│   └── threat_model.json        # Threat model and risk assessment
├── red/
│   └── red_team_agent.py        # AI red team adversary
├── blue/
│   └── blue_team_agent.py       # AI blue team defender
├── reports/
│   ├── red/                     # Red team attack reports
│   ├── blue/                    # Blue team defense reports
│   └── purple/                  # Environment summaries
├── wargame_orchestrator.py      # Main orchestration engine
└── demo_round.py               # Single round demonstration
```

## 📋 Report Structure

### Red Team Reports (`/reports/red/attacks_round_X.json`)
- Detailed attack timeline with MITRE ATT&CK techniques
- Success/failure status for each action
- Compromised assets and persistence mechanisms
- Impact assessment (confidentiality, integrity, availability)

### Blue Team Reports (`/reports/blue/defenses_round_X.json`)
- Detection events with confidence scores
- Deployed countermeasures and effectiveness
- Resource costs and optimization metrics
- Threat intelligence gathering

### Purple Team Summaries (`/reports/purple/round_X_summary.json`)
- Comprehensive round analysis
- Performance metrics for both teams
- Environment evolution tracking
- Lessons learned and recommendations

## 🔧 Customization

### Adding New Attack Techniques
Edit `red/red_team_agent.py` to add new phases or techniques:
```python
def new_attack_phase(self, env_state: Dict) -> List[AttackAction]:
    # Implement new attack techniques
    pass
```

### Enhancing Blue Team Capabilities
Modify `blue/blue_team_agent.py` to add new detection or response capabilities:
```python
def deploy_new_defense(self, env_state: Dict) -> List[DefenseAction]:
    # Implement new defense mechanisms
    pass
```

### Environment Modifications
Update `purple/environment_state.json` to change:
- Organization structure and size
- Network topology and services
- Application stack and versions
- Vulnerability landscape

## 🎯 Future Enhancements

- **Machine Learning Integration**: Adaptive AI that learns from previous rounds
- **Real-time Visualization**: Web-based dashboard for live wargame monitoring
- **Compliance Frameworks**: Integrate regulatory compliance requirements
- **Multi-tenant Support**: Support for multiple simultaneous organizations
- **Advanced Deception**: Dynamic honeypot and deception technology deployment
- **Integration with XORB PTaaS**: Connect with real security scanning tools

## 📈 Educational Value

This wargame provides hands-on experience with:
- **Cyber Attack Lifecycle**: Complete kill chain simulation
- **Defense in Depth**: Layered security approach
- **Threat Intelligence**: IOC extraction and sharing
- **Incident Response**: Detection and containment procedures
- **Risk Assessment**: Quantitative security metrics
- **Security Economics**: Resource allocation and cost-effectiveness

---

The XORB Red vs Blue Wargame represents a cutting-edge approach to cybersecurity training and assessment, providing realistic, continuous simulation of advanced persistent threat scenarios within a safe, controlled environment.