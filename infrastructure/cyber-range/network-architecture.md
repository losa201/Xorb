# XORB PTaaS Red vs Blue Cyber Range - Network Architecture

##  Network Topology Diagram

```mermaid
graph TB
    subgraph "Control Plane - DMZ"
        XORB["🎯 XORB Orchestrator<br/>Control Center"]
        ADMIN["👤 Admin Console<br/>Management Interface"]
        MONITOR["📊 Monitoring Stack<br/>Grafana + Prometheus"]
        KILLSWITCH["🔴 Kill Switch<br/>Emergency Stop"]
    end

    subgraph "Red Team Infrastructure"
        REDCMD["🔴 Red Team C2<br/>Command & Control"]
        REDTOOLS["⚔️ Attack Tools<br/>Metasploit + Custom"]
        PHISHING["📧 Phishing Infra<br/>SET + GoPhish"]
        EXFIL["📤 Exfiltration<br/>Data Staging"]
    end

    subgraph "Blue Team SOC"
        BLUESIEM["🔵 Blue SIEM<br/>ELK Stack + Wazuh"]
        BLUEIDS["🛡️ IDS/IPS<br/>Suricata + Zeek"]
        BLUEIR["🚨 Incident Response<br/>DFIR Workstation"]
        BLUEHUNT["🔍 Threat Hunting<br/>Jupyter + MISP"]
    end

    subgraph "Target Environment - Production Simulation"
        subgraph "Web Tier"
            WEBAPP["🌐 Web Application<br/>DVWA + Custom Apps"]
            WEBDB["🗄️ Web Database<br/>MySQL + PostgreSQL"]
        end

        subgraph "Internal Network"
            FILESERVER["📁 File Server<br/>SMB + FTP"]
            MAILSERVER["📧 Mail Server<br/>Postfix + Dovecot"]
            ADSERVER["🏢 Active Directory<br/>Domain Controller"]
        end

        subgraph "IoT/OT Simulation"
            IOTDEVICES["📱 IoT Devices<br/>Modbus + MQTT"]
            SCADA["⚙️ SCADA System<br/>HMI Simulation"]
        end
    end

    subgraph "Network Segmentation & Security"
        FIREWALL["🔥 Cyber Range Firewall<br/>pfSense + iptables"]
        ROUTER["🔀 Core Router<br/>Inter-VLAN Routing"]
        SWITCH["🔌 Managed Switch<br/>VLAN Segmentation"]
    end

    subgraph "Simulation & Scenarios"
        SCENARIOS["📋 Scenario Engine<br/>Automated Deployments"]
        TRAFFIC["🌊 Traffic Generator<br/>Realistic Simulation"]
        VULNGEN["🎯 Vuln Generator<br/>Dynamic Injection"]
    end

    %% Network Connections
    XORB --> FIREWALL
    ADMIN --> XORB
    MONITOR --> FIREWALL
    KILLSWITCH --> FIREWALL

    FIREWALL --> ROUTER
    ROUTER --> SWITCH

    REDCMD --> SWITCH
    REDTOOLS --> SWITCH
    PHISHING --> SWITCH
    EXFIL --> SWITCH

    BLUESIEM --> SWITCH
    BLUEIDS --> SWITCH
    BLUEIR --> SWITCH
    BLUEHUNT --> SWITCH

    SWITCH --> WEBAPP
    SWITCH --> WEBDB
    SWITCH --> FILESERVER
    SWITCH --> MAILSERVER
    SWITCH --> ADSERVER
    SWITCH --> IOTDEVICES
    SWITCH --> SCADA

    SCENARIOS --> FIREWALL
    TRAFFIC --> SWITCH
    VULNGEN --> SWITCH

    %% Styling
    classDef redTeam fill:#ff6b6b,stroke:#c92a2a,stroke-width:2px,color:#fff
    classDef blueTeam fill:#4dabf7,stroke:#1971c2,stroke-width:2px,color:#fff
    classDef control fill:#51cf66,stroke:#37b24d,stroke-width:2px,color:#fff
    classDef target fill:#ffd43b,stroke:#fab005,stroke-width:2px,color:#000
    classDef infra fill:#868e96,stroke:#495057,stroke-width:2px,color:#fff

    class REDCMD,REDTOOLS,PHISHING,EXFIL redTeam
    class BLUESIEM,BLUEIDS,BLUEIR,BLUEHUNT blueTeam
    class XORB,ADMIN,MONITOR,KILLSWITCH control
    class WEBAPP,WEBDB,FILESERVER,MAILSERVER,ADSERVER,IOTDEVICES,SCADA target
    class FIREWALL,ROUTER,SWITCH,SCENARIOS,TRAFFIC,VULNGEN infra
```text

##  Network Segmentation Strategy

###  VLAN Architecture

| VLAN ID | Name | Purpose | IP Range | Security Zone |
|---------|------|---------|----------|---------------|
| 10 | CONTROL | XORB Control Plane | 10.10.10.0/24 | Management |
| 20 | RED_TEAM | Red Team Infrastructure | 10.20.0.0/16 | Isolated Attack |
| 30 | BLUE_TEAM | Blue Team SOC | 10.30.0.0/24 | Monitoring |
| 100 | TARGET_WEB | Web Applications | 10.100.0.0/24 | DMZ Target |
| 110 | TARGET_INTERNAL | Internal Services | 10.110.0.0/24 | Internal Target |
| 120 | TARGET_OT | OT/IoT Devices | 10.120.0.0/24 | OT Target |
| 200 | SIMULATION | Traffic/Vuln Gen | 10.200.0.0/24 | Simulation |

###  Firewall Rules Matrix

####  Staging Mode (Safe Training)
```text
CONTROL → ALL_VLANS: ALLOW (Management)
BLUE_TEAM → TARGET_*: ALLOW (Monitoring)
RED_TEAM → TARGET_*: DENY (Blocked attacks)
RED_TEAM → BLUE_TEAM: DENY
TARGET_* → INTERNET: DENY
SIMULATION → TARGET_*: ALLOW (Benign traffic)
```text

####  Live Exercise Mode (Active Red Team)
```text
CONTROL → ALL_VLANS: ALLOW (Management)
BLUE_TEAM → TARGET_*: ALLOW (Monitoring)
RED_TEAM → TARGET_*: ALLOW (Active attacks)
RED_TEAM → BLUE_TEAM: DENY
RED_TEAM → CONTROL: DENY
TARGET_* → INTERNET: CONTROLLED (Limited egress)
SIMULATION → TARGET_*: ALLOW (Background traffic)
```text

##  Security Isolation Features

###  Container Isolation
- Each team operates in isolated container namespaces
- Network policies prevent cross-contamination
- Resource limits prevent DoS between teams

###  Data Isolation
- Separate data volumes for Red/Blue teams
- Encrypted storage for sensitive artifacts
- Automated data sanitization between exercises

###  Network Isolation
- Software-defined networking (SDN) controls
- Microsegmentation with Calico NetworkPolicies
- Dynamic VLAN assignment based on exercise mode

###  Monitoring & Logging
- All network traffic captured and analyzed
- Real-time attack/defense correlation
- Comprehensive audit trail for post-exercise analysis

##  Emergency Controls

###  Kill Switch Mechanisms
1. **Network Kill Switch**: Immediate isolation of all attack traffic
2. **Container Kill Switch**: Emergency shutdown of Red Team containers
3. **Data Protection**: Automatic backup and isolation of target systems
4. **Communication Kill**: Emergency blue team notification system

###  Safety Guardrails
- Automated malware detection and quarantine
- Real-time traffic analysis for dangerous payloads
- Geographic IP restrictions
- Time-based exercise windows with automatic shutdown

##  Physical Network Requirements

###  Hardware Specifications
- **Minimum**: 3 physical hosts (Control, Red/Blue, Targets)
- **Recommended**: 5+ hosts with dedicated networking equipment
- **Network**: Gigabit Ethernet minimum, 10G preferred
- **Storage**: SSD for container storage, NAS for persistent data

###  Resource Allocation
- **Control Plane**: 8 vCPU, 16GB RAM, 100GB SSD
- **Red Team**: 16 vCPU, 32GB RAM, 200GB SSD
- **Blue Team**: 12 vCPU, 24GB RAM, 150GB SSD
- **Target Environment**: 20 vCPU, 48GB RAM, 500GB SSD
- **Total Minimum**: 56 vCPU, 120GB RAM, 950GB Storage