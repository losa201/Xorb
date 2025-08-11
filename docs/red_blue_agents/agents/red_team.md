#  Red Team Agents

The XORB Red/Blue Agent Framework includes specialized red team agents that simulate real-world attack scenarios. These agents follow the MITRE ATT&CK framework and provide comprehensive offensive security testing capabilities.

##  🎯 Overview

Red team agents are designed to emulate sophisticated adversaries and test defensive capabilities across the entire attack lifecycle. Each agent specializes in specific phases of an attack, from initial reconnaissance to data collection and persistence.

###  Agent Types

1. **[Reconnaissance Agent](#reconnaissance-agent)** - Information gathering and target discovery
2. **[Exploitation Agent](#exploitation-agent)** - Initial access and vulnerability exploitation
3. **[Persistence Agent](#persistence-agent)** - Maintaining access and establishing persistence
4. **[Evasion Agent](#evasion-agent)** - Defense evasion and stealth techniques
5. **[Collection Agent](#collection-agent)** - Data collection and credential harvesting

###  Common Features

- **MITRE ATT&CK Integration**: All techniques mapped to MITRE ATT&CK framework
- **Environment-Aware**: Respects environment policies and constraints
- **Safety Controls**: Built-in safety mechanisms for production environments
- **Telemetry Collection**: Comprehensive logging and metrics collection
- **Autonomous Operation**: Self-directed execution based on mission objectives
- **Learning Integration**: Techniques adapt based on success/failure patterns

##  🔍 Reconnaissance Agent

The Reconnaissance Agent specializes in information gathering and target discovery, implementing techniques from the MITRE ATT&CK Reconnaissance and Discovery tactics.

###  Supported Techniques

####  Network Reconnaissance

**`recon.port_scan` (T1046 - Network Service Scanning)**
- Discover open ports and services on target systems
- Supports TCP, UDP, and stealth scanning modes
- Configurable timing and port ranges
- Intelligent service identification

```json
{
  "technique_id": "recon.port_scan",
  "parameters": {
    "target": "scanme.nmap.org",
    "ports": "1-1000",
    "scan_type": "tcp",
    "timing": "normal"
  }
}
```

**`recon.service_enum` (T1046 - Network Service Scanning)**
- Enumerate services and gather version information
- Banner grabbing and service fingerprinting
- Dependent on port scanning results
- Aggressive and passive enumeration modes

```json
{
  "technique_id": "recon.service_enum",
  "parameters": {
    "target": "example.com",
    "ports": [22, 80, 443],
    "aggressive": false
  }
}
```

####  Web Application Reconnaissance

**`recon.web_crawl` (T1592.002 - Gather Victim Host Information)**
- Crawl web applications to discover endpoints and technologies
- Extract forms, links, and technology stack information
- Configurable crawl depth and user agents
- Technology detection and fingerprinting

```json
{
  "technique_id": "recon.web_crawl",
  "parameters": {
    "url": "https://example.com",
    "depth": 3,
    "user_agent": "Mozilla/5.0 (compatible; XORBBot/1.0)"
  }
}
```

####  DNS and Infrastructure

**`recon.dns_enum` (T1590.002 - DNS)**
- Comprehensive DNS enumeration
- Multiple record type queries (A, AAAA, MX, NS, TXT)
- DNS zone transfer attempts
- Subdomain discovery through DNS

```json
{
  "technique_id": "recon.dns_enum",
  "parameters": {
    "domain": "example.com",
    "record_types": ["A", "AAAA", "MX", "NS", "TXT"]
  }
}
```

**`recon.subdomain_enum` (T1590.002 - DNS)**
- Subdomain enumeration using wordlists
- DNS brute forcing with common subdomains
- Integration with public subdomain databases
- Certificate transparency log analysis

```json
{
  "technique_id": "recon.subdomain_enum",
  "parameters": {
    "domain": "example.com",
    "wordlist": ["www", "mail", "ftp", "admin", "dev"]
  }
}
```

####  Network Discovery

**`recon.network_discovery` (T1018 - Remote System Discovery)**
- Discover live hosts on network segments
- ARP scanning and ping sweeps
- Network topology mapping
- Asset discovery and classification

```json
{
  "technique_id": "recon.network_discovery",
  "parameters": {
    "network": "192.168.1.0/24"
  }
}
```

**`recon.os_fingerprint` (T1082 - System Information Discovery)**
- Operating system fingerprinting
- Network stack analysis
- Service behavior analysis
- Passive and active fingerprinting

```json
{
  "technique_id": "recon.os_fingerprint",
  "parameters": {
    "target": "example.com"
  }
}
```

###  Execution Flow

The Reconnaissance Agent follows a systematic approach:

1. **Target Validation**: Verify target accessibility and authorization
2. **Network Mapping**: Discover network topology and live hosts
3. **Service Discovery**: Identify running services and open ports
4. **Technology Stack**: Determine technology stack and versions
5. **Vulnerability Assessment**: Identify potential attack vectors
6. **Intelligence Gathering**: Collect metadata and context information

###  Sample Output

```json
{
  "reconnaissance_results": {
    "target": "example.com",
    "ip_addresses": ["192.168.1.100"],
    "open_ports": [
      {
        "port": 22,
        "protocol": "tcp",
        "service": "ssh",
        "version": "OpenSSH 8.3",
        "banner": "SSH-2.0-OpenSSH_8.3"
      },
      {
        "port": 80,
        "protocol": "tcp",
        "service": "http",
        "version": "Apache 2.4.41",
        "technologies": ["PHP 7.4", "WordPress 5.8"]
      }
    ],
    "subdomains": [
      "www.example.com",
      "mail.example.com",
      "admin.example.com"
    ],
    "web_technologies": [
      "Apache",
      "PHP",
      "WordPress",
      "jQuery"
    ],
    "potential_vulnerabilities": [
      {
        "type": "Outdated Software",
        "description": "WordPress version 5.8 has known vulnerabilities",
        "severity": "medium"
      }
    ]
  }
}
```

##  ⚔️ Exploitation Agent

The Exploitation Agent specializes in initial access and privilege escalation, implementing techniques from the MITRE ATT&CK Initial Access and Privilege Escalation tactics.

###  Supported Techniques

####  Web Application Exploitation

**`exploit.web_sqli` (T1190 - Exploit Public-Facing Application)**
- SQL injection attack automation
- Multiple injection techniques (boolean, union, time-based, error-based)
- Database fingerprinting and enumeration
- Payload customization for different database types

```json
{
  "technique_id": "exploit.web_sqli",
  "parameters": {
    "url": "https://vulnerable-app.com/search.php",
    "parameter": "id",
    "technique": "boolean",
    "database": "mysql"
  }
}
```

**`exploit.web_shell` (T1505.003 - Web Shell)**
- Web shell upload and deployment
- Multiple shell types (PHP, ASP, JSP, Python)
- Bypass techniques for upload restrictions
- Shell obfuscation and encoding

```json
{
  "technique_id": "exploit.web_shell",
  "parameters": {
    "upload_url": "https://vulnerable-app.com/upload.php",
    "shell_type": "php",
    "filename": "shell.php"
  }
}
```

####  Credential Attacks

**`exploit.brute_force` (T1110 - Brute Force)**
- Automated brute force attacks
- Support for multiple protocols (SSH, RDP, HTTP, FTP)
- Smart wordlist management
- Rate limiting and evasion techniques

```json
{
  "technique_id": "exploit.brute_force",
  "parameters": {
    "target": "ssh://example.com:22",
    "usernames": ["admin", "user", "test"],
    "passwords": ["password", "123456", "admin"],
    "threads": 5,
    "delay": 1.0
  }
}
```

**`exploit.credential_stuffing` (T1110.004 - Credential Stuffing)**
- Use previously breached credentials
- Large-scale credential testing
- Success rate optimization
- Credential validation and verification

```json
{
  "technique_id": "exploit.credential_stuffing",
  "parameters": {
    "target": "https://app.example.com/login",
    "credentials": [
      {"username": "user1", "password": "pass1"},
      {"username": "user2", "password": "pass2"}
    ]
  }
}
```

####  System Exploitation

**`exploit.buffer_overflow` (T1068 - Exploitation for Privilege Escalation)**
- Buffer overflow exploitation
- Payload generation and delivery
- Exploit development and testing
- Memory corruption techniques

**`exploit.privilege_escalation` (T1068 - Exploitation for Privilege Escalation)**
- Local privilege escalation
- Kernel exploit detection and execution
- SUID/SGID binary exploitation
- Service misconfiguration abuse

```json
{
  "technique_id": "exploit.privilege_escalation",
  "parameters": {
    "method": "auto",
    "target_user": "root"
  }
}
```

###  Exploitation Workflow

1. **Target Assessment**: Analyze reconnaissance data for exploitable services
2. **Vulnerability Prioritization**: Rank vulnerabilities by exploitability and impact
3. **Exploit Selection**: Choose appropriate exploitation techniques
4. **Payload Generation**: Create custom payloads for target environment
5. **Exploitation Execution**: Execute exploits with safety controls
6. **Access Verification**: Confirm successful exploitation
7. **Post-Exploitation**: Prepare for persistence and lateral movement

###  Safety Controls

The Exploitation Agent includes multiple safety mechanisms:

- **Sandbox Isolation**: All exploits run in isolated containers
- **Target Validation**: Verify authorization before exploitation
- **Damage Prevention**: Avoid destructive actions in production
- **Rate Limiting**: Prevent DoS conditions during exploitation
- **Logging**: Comprehensive audit trail of all activities

##  🔒 Persistence Agent

The Persistence Agent specializes in maintaining access to compromised systems, implementing techniques from the MITRE ATT&CK Persistence tactic.

###  Supported Techniques

####  Web-Based Persistence

**`persist.web_shell` (T1505.003 - Server Software Component)**
- Deploy and maintain web shells
- Multiple backup shell locations
- Shell health monitoring and restoration
- Stealth techniques to avoid detection

```json
{
  "technique_id": "persist.web_shell",
  "parameters": {
    "shell_url": "https://compromised-site.com/shell.php",
    "backup_locations": [
      "https://compromised-site.com/images/logo.php",
      "https://compromised-site.com/cache/data.php"
    ]
  }
}
```

####  System-Level Persistence

**`persist.scheduled_task` (T1053.005 - Scheduled Task)**
- Create scheduled tasks for persistence
- Windows Task Scheduler integration
- Linux cron job manipulation
- Task obfuscation and hiding

```json
{
  "technique_id": "persist.scheduled_task",
  "parameters": {
    "task_name": "SystemUpdate",
    "command": "powershell.exe -WindowStyle Hidden -File C:\\temp\\update.ps1",
    "trigger": "daily"
  }
}
```

**`persist.service_creation` (T1543.003 - Windows Service)**
- Create system services for persistence
- Service configuration and management
- Privilege escalation through services
- Service hiding and obfuscation

```json
{
  "technique_id": "persist.service_creation",
  "parameters": {
    "service_name": "UpdateService",
    "service_path": "C:\\Windows\\System32\\update.exe",
    "start_type": "auto"
  }
}
```

####  User Account Manipulation

**`persist.user_account` (T1136.001 - Local Account)**
- Create hidden user accounts
- Modify existing account permissions
- Password policy bypass
- Account privilege escalation

```json
{
  "technique_id": "persist.user_account",
  "parameters": {
    "username": "support",
    "password": "ComplexPass123!",
    "groups": ["Administrators"]
  }
}
```

####  File System Persistence

**`persist.startup_script` (T1037 - Boot or Logon Initialization Scripts)**
- Modify startup scripts and configurations
- Registry autorun entries (Windows)
- Systemd service files (Linux)
- Login hook manipulation (macOS)

```json
{
  "technique_id": "persist.startup_script",
  "parameters": {
    "script_path": "/etc/init.d/network-monitor",
    "script_content": "#!/bin/bash\n# Network monitoring service\n/opt/monitor/agent &"
  }
}
```

**`persist.registry_autorun` (T1547.001 - Registry Run Keys)**
- Windows registry autorun entries
- Multiple registry locations
- Registry key obfuscation
- Fileless persistence techniques

```json
{
  "technique_id": "persist.registry_autorun",
  "parameters": {
    "key_path": "HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run",
    "value_name": "SecurityUpdate",
    "value_data": "C:\\Windows\\System32\\security.exe"
  }
}
```

####  Network-Based Persistence

**`persist.ssh_keys` (T1098.004 - SSH Authorized Keys)**
- SSH key-based persistence
- Authorized_keys file manipulation
- Key generation and installation
- Stealth key hiding techniques

```json
{
  "technique_id": "persist.ssh_keys",
  "parameters": {
    "target_user": "admin",
    "public_key": "ssh-rsa AAAAB3NzaC1yc2EAAAADAQAB..."
  }
}
```

###  Persistence Strategy

The Persistence Agent employs a multi-layered approach:

1. **Primary Persistence**: Establish main access mechanism
2. **Backup Methods**: Deploy secondary persistence techniques
3. **Health Monitoring**: Regularly verify persistence mechanisms
4. **Restoration**: Automatically restore lost access
5. **Stealth Maintenance**: Keep persistence mechanisms hidden
6. **Evolution**: Adapt techniques based on defensive responses

##  👻 Evasion Agent

The Evasion Agent specializes in defense evasion and stealth techniques, implementing techniques from the MITRE ATT&CK Defense Evasion tactic.

###  Supported Techniques

####  Process Manipulation

**`evasion.process_hollowing` (T1055.012 - Process Hollowing)**
- Inject malicious code into legitimate processes
- Memory manipulation and code injection
- Process replacement techniques
- Anti-debugging and anti-analysis

```json
{
  "technique_id": "evasion.process_hollowing",
  "parameters": {
    "target_process": "notepad.exe",
    "payload": "base64_encoded_shellcode"
  }
}
```

####  Network Evasion

**`evasion.domain_fronting` (T1090.004 - Domain Fronting)**
- Hide C2 communications using domain fronting
- CDN-based traffic obfuscation
- SSL/TLS certificate manipulation
- Traffic pattern randomization

```json
{
  "technique_id": "evasion.domain_fronting",
  "parameters": {
    "front_domain": "legitimate-cdn.cloudflare.com",
    "real_c2": "malicious-c2.com",
    "cdn_provider": "cloudflare"
  }
}
```

**`evasion.traffic_obfuscation` (T1001 - Data Obfuscation)**
- Encrypt and obfuscate network traffic
- Protocol mimicry and tunneling
- Steganographic data hiding
- Traffic timing randomization

```json
{
  "technique_id": "evasion.traffic_obfuscation",
  "parameters": {
    "obfuscation_method": "encryption",
    "port": 443
  }
}
```

####  File System Evasion

**`evasion.timestomp` (T1070.006 - Timestomp)**
- Manipulate file timestamps
- Hide file creation and modification times
- Timestamp normalization
- File attribute manipulation

```json
{
  "technique_id": "evasion.timestomp",
  "parameters": {
    "target_files": ["/var/log/suspicious.log"],
    "timestamp": "2020-01-01 00:00:00"
  }
}
```

**`evasion.file_masquerading` (T1036 - Masquerading)**
- Disguise malicious files as legitimate ones
- File extension manipulation
- Icon and metadata spoofing
- Legitimate binary replacement

```json
{
  "technique_id": "evasion.file_masquerading",
  "parameters": {
    "source_file": "/tmp/malware.exe",
    "target_name": "notepad.exe",
    "legitimate_path": "C:\\Windows\\System32"
  }
}
```

####  Anti-Forensics

**`evasion.log_clearing` (T1070.002 - Clear Linux or Mac System Logs)**
- Clear system and application logs
- Selective log entry deletion
- Log rotation manipulation
- Audit trail obfuscation

```json
{
  "technique_id": "evasion.log_clearing",
  "parameters": {
    "log_types": ["system", "auth", "application"]
  }
}
```

####  Antivirus Evasion

**`evasion.av_bypass` (T1027 - Obfuscated Files or Information)**
- Bypass antivirus and EDR solutions
- Payload obfuscation and packing
- Signature evasion techniques
- Behavioral analysis evasion

```json
{
  "technique_id": "evasion.av_bypass",
  "parameters": {
    "payload_type": "executable",
    "techniques": ["packing", "obfuscation", "encryption"]
  }
}
```

###  Evasion Strategy

The Evasion Agent implements a comprehensive stealth strategy:

1. **Threat Assessment**: Identify defensive capabilities
2. **Evasion Planning**: Select appropriate evasion techniques
3. **Stealth Execution**: Implement evasion measures
4. **Monitoring**: Continuously assess detection risk
5. **Adaptation**: Adjust techniques based on defensive responses
6. **Recovery**: Re-establish stealth if detected

##  📊 Collection Agent

The Collection Agent specializes in data collection and credential harvesting, implementing techniques from the MITRE ATT&CK Collection and Credential Access tactics.

###  Supported Techniques

####  Input Capture

**`collection.keylogging` (T1056.001 - Keylogging)**
- Capture keystrokes from target systems
- Hardware and software keyloggers
- Credential extraction from keystrokes
- Stealth keylogging techniques

```json
{
  "technique_id": "collection.keylogging",
  "parameters": {
    "duration": 3600,
    "output_file": "keylog.txt"
  }
}
```

**`collection.screen_capture` (T1113 - Screen Capture)**
- Capture screenshots and screen recordings
- Periodic screenshot collection
- Screen content analysis
- Visual data extraction

```json
{
  "technique_id": "collection.screen_capture",
  "parameters": {
    "interval": 60,
    "quality": "medium"
  }
}
```

**`collection.clipboard` (T1115 - Clipboard Data)**
- Monitor and capture clipboard content
- Real-time clipboard surveillance
- Sensitive data detection
- Clipboard history analysis

```json
{
  "technique_id": "collection.clipboard",
  "parameters": {
    "duration": 300
  }
}
```

####  Credential Harvesting

**`collection.credential_dumping` (T1003 - OS Credential Dumping)**
- Extract credentials from system memory
- Password hash dumping
- Cached credential extraction
- Token and ticket harvesting

```json
{
  "technique_id": "collection.credential_dumping",
  "parameters": {
    "dump_method": "auto",
    "target_users": ["admin", "user"]
  }
}
```

**`collection.browser_data` (T1555.003 - Credentials from Web Browsers)**
- Extract saved passwords from browsers
- Cookie and session token harvesting
- Browser history and bookmark analysis
- Stored form data extraction

```json
{
  "technique_id": "collection.browser_data",
  "parameters": {
    "browsers": ["chrome", "firefox", "edge"],
    "data_types": ["passwords", "cookies", "history"]
  }
}
```

####  File and Data Collection

**`collection.file_search` (T1005 - Data from Local System)**
- Search for sensitive files and documents
- Pattern-based file discovery
- Content analysis and filtering
- Metadata extraction

```json
{
  "technique_id": "collection.file_search",
  "parameters": {
    "search_paths": ["/home", "/Documents"],
    "file_patterns": ["*.doc", "*.pdf", "*.xlsx"],
    "keywords": ["password", "confidential", "secret"]
  }
}
```

**`collection.network_shares` (T1039 - Data from Network Shared Drive)**
- Enumerate and access network shares
- SMB/CIFS share discovery
- Network file system access
- Remote data collection

```json
{
  "technique_id": "collection.network_shares",
  "parameters": {
    "target_networks": ["192.168.1.0/24"]
  }
}
```

####  Forensic Collection

**`collection.memory_dump` (T1005 - Data from Local System)**
- Create memory dumps for analysis
- Live memory acquisition
- Process memory extraction
- Artifact preservation

**`collection.forensic_artifacts` (T1005 - Data from Local System)**
- Collect forensic artifacts
- System registry extraction
- Log file collection
- Temporal analysis data

###  Collection Strategy

The Collection Agent follows a systematic data collection approach:

1. **Target Profiling**: Identify valuable data sources
2. **Access Planning**: Determine collection methods
3. **Stealth Collection**: Gather data without detection
4. **Data Processing**: Filter and analyze collected data
5. **Secure Storage**: Encrypt and store collected data
6. **Exfiltration Preparation**: Prepare data for extraction

##  🔧 Agent Configuration

###  Environment Policies

Red team agents respect environment-specific policies:

```json
{
  "production": {
    "allowed_techniques": [],
    "denied_techniques": ["*"],
    "max_risk_level": "none"
  },
  "staging": {
    "allowed_techniques": ["recon.*"],
    "denied_techniques": ["exploit.*", "persist.*"],
    "max_risk_level": "medium"
  },
  "development": {
    "allowed_techniques": ["*"],
    "denied_techniques": [],
    "max_risk_level": "critical"
  },
  "cyber_range": {
    "allowed_techniques": ["*"],
    "denied_techniques": [],
    "max_risk_level": "critical"
  }
}
```

###  Resource Constraints

Each agent type has specific resource requirements:

```json
{
  "red_recon": {
    "cpu_cores": 1.0,
    "memory_mb": 512,
    "disk_mb": 1024,
    "network_bandwidth_mb": 50
  },
  "red_exploit": {
    "cpu_cores": 2.0,
    "memory_mb": 1024,
    "disk_mb": 2048,
    "network_bandwidth_mb": 100
  },
  "red_persistence": {
    "cpu_cores": 1.0,
    "memory_mb": 512,
    "disk_mb": 1024,
    "network_bandwidth_mb": 25
  }
}
```

##  📈 Performance Metrics

Red team agents collect comprehensive performance metrics:

###  Execution Metrics
- **Success Rate**: Percentage of successful technique executions
- **Execution Time**: Average time per technique
- **Resource Usage**: CPU, memory, and network utilization
- **Error Rate**: Frequency of execution errors

###  Effectiveness Metrics
- **Detection Rate**: How often agents are detected
- **Evasion Success**: Effectiveness of evasion techniques
- **Persistence Duration**: How long persistence mechanisms survive
- **Data Collection Volume**: Amount of data successfully collected

###  Learning Metrics
- **Technique Adaptation**: How techniques improve over time
- **Target Success**: Success rates against different target types
- **Environment Performance**: Effectiveness in different environments

##  🚨 Safety and Ethics

###  Built-in Safety Controls

1. **Environment Validation**: Verify environment before execution
2. **Target Authorization**: Confirm target authorization
3. **Damage Prevention**: Avoid destructive actions
4. **Data Protection**: Protect sensitive data during collection
5. **Access Limitation**: Respect access boundaries
6. **Audit Logging**: Comprehensive activity logging

###  Ethical Guidelines

1. **Authorized Testing Only**: Only operate against authorized targets
2. **Responsible Disclosure**: Report vulnerabilities responsibly
3. **Data Minimization**: Collect only necessary data
4. **Privacy Respect**: Protect personal and sensitive information
5. **Legal Compliance**: Comply with all applicable laws
6. **Professional Standards**: Follow industry best practices

###  Usage Restrictions

- **Production Environments**: Most techniques disabled by default
- **Unauthorized Targets**: Strict validation prevents unauthorized access
- **Destructive Actions**: Harmful activities are blocked
- **Data Exfiltration**: Sensitive data handling controls
- **Compliance**: Automatic compliance with regulations

##  🔗 Integration Points

###  With Blue Team Agents
- **Detection Testing**: Test blue team detection capabilities
- **Response Validation**: Validate incident response procedures
- **Hunt Exercise**: Provide indicators for threat hunting
- **Training Data**: Generate training data for ML models

###  With XORB Platform
- **PTaaS Integration**: Full integration with PTaaS workflows
- **Threat Intelligence**: Feed findings into threat intelligence
- **Compliance Reporting**: Generate compliance reports
- **Risk Assessment**: Contribute to risk assessment processes

###  With External Tools
- **SIEM Integration**: Send events to SIEM systems
- **Vulnerability Scanners**: Complement automated scanning
- **Forensic Tools**: Integrate with forensic workflows
- **Ticketing Systems**: Create tickets for findings

---

For more detailed information, see:
- [Blue Team Agents](./blue_team.md)
- [Custom Agent Development](./custom_agents.md)
- [Capability Registry](../components/capability_registry.md)
- [Security Model](../security/security_model.md)