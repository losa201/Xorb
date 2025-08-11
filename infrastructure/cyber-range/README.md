# XORB PTaaS Red vs Blue Cyber Range

A comprehensive, production-ready cyber range environment for conducting realistic Red vs Blue team exercises with advanced security controls, monitoring, and automation.

##  🏗️ Architecture Overview

The XORB PTaaS Cyber Range is a self-contained, isolated environment designed to simulate real-world attack and defense scenarios. It provides:

- **Isolated Network Segments**: Separate VLANs for control, red team, blue team, targets, and simulation
- **Mode-Based Operation**: Safe staging mode and live exercise mode with real attacks
- **Emergency Controls**: Kill switch for immediate exercise termination
- **Comprehensive Monitoring**: Real-time security event correlation and performance metrics
- **Automated Orchestration**: Campaign management with scenario-based target deployment

##  🌐 Network Topology

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Control       │    │   Red Team      │    │   Blue Team     │
│   Plane         │    │   Infrastructure│    │   SOC           │
│   10.10.10.0/24 │    │   10.20.0.0/16  │    │   10.30.0.0/24  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Core Router   │
                    │   & Firewall    │
                    └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Targets   │    │ Internal Targets│    │   OT/IoT        │
│   10.100.0.0/24 │    │   10.110.0.0/24 │    │   10.120.0.0/24 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```text

##  🚀 Quick Start

###  Prerequisites

- Kubernetes cluster (minimum 56 vCPU, 120GB RAM, 950GB storage)
- kubectl configured and connected
- Docker for building custom images
- Terraform (optional, for cloud deployment)
- Root access for firewall configuration

###  1. Infrastructure Deployment

####  Option A: Cloud Deployment (AWS)
```bash
# Deploy cloud infrastructure
cd terraform/
terraform init
terraform plan -var="cluster_name=xorb-cyber-range"
terraform apply

# Configure kubectl
export KUBECONFIG=./kubeconfig_xorb-cyber-range
```text

####  Option B: Local Kubernetes
```bash
# Verify cluster resources
kubectl top nodes
kubectl get namespaces
```text

###  2. Deploy Cyber Range Components

```bash
# Create namespaces and network policies
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/network-policies.yaml

# Deploy control plane
kubectl apply -f k8s/control-plane.yaml

# Deploy team infrastructure
kubectl apply -f k8s/red-team.yaml
kubectl apply -f k8s/blue-team.yaml

# Deploy target environments
kubectl apply -f k8s/targets.yaml

# Deploy monitoring stack
kubectl apply -f monitoring/prometheus-config.yaml
kubectl apply -f monitoring/grafana-config.yaml
kubectl apply -f monitoring/alertmanager-config.yaml
```text

###  3. Configure Firewall Rules

```bash
# Make scripts executable
chmod +x scripts/*.sh firewall/*.sh

# Start in staging mode (safe training)
sudo ./firewall/iptables-staging.sh

# Verify configuration
sudo ./firewall/iptables-staging.sh status
```text

###  4. Verify Deployment

```bash
# Check all pods are running
kubectl get pods --all-namespaces

# Access control interfaces
kubectl port-forward -n cyber-range-control svc/xorb-admin-service 3000:3000
kubectl port-forward -n cyber-range-control svc/grafana-service 3001:3000

# Open in browser
# - Admin Console: http://localhost:3000
# - Grafana: http://localhost:3001 (admin/SecureAdminPass123!)
```text

##  🎯 Operation Modes

###  Staging Mode (Default)
- **Purpose**: Safe training and preparation
- **Red Team**: Attacks are BLOCKED at firewall level
- **Blue Team**: Full monitoring capabilities active
- **Targets**: Available for blue team analysis
- **Safety**: Maximum - no real attacks possible

```bash
# Switch to staging mode
sudo ./scripts/mode-switch.sh staging

# Verify staging mode
sudo ./scripts/mode-switch.sh status
```text

###  Live Exercise Mode
- **Purpose**: Active red vs blue exercises
- **Red Team**: Attacks are ENABLED with rate limiting
- **Blue Team**: Enhanced monitoring and alerting
- **Targets**: Fully exposed to attacks
- **Safety**: Controlled - time limits and kill switch active

```bash
# Switch to live mode (requires confirmation)
sudo ./scripts/mode-switch.sh live

# Monitor exercise
watch kubectl get pods --all-namespaces
tail -f /var/log/cyber-range/mode-switch.log
```text

##  🚨 Emergency Controls

###  Kill Switch Activation
Immediately terminates all attack activities and isolates the environment:

```bash
# Manual activation
sudo ./scripts/kill-switch.sh activate

# Specific reason codes
sudo ./scripts/kill-switch.sh activate security_breach
sudo ./scripts/kill-switch.sh activate malware_detected

# Check status
sudo ./scripts/kill-switch.sh status

# Restore from backup
sudo ./scripts/kill-switch.sh restore /path/to/backup
```text

###  Auto-Termination
- Exercises auto-terminate after configured duration (default: 8 hours)
- Time-based warnings: 1 hour, 30 minutes, 10 minutes before termination
- Graceful shutdown with data preservation

##  📊 Monitoring & Observability

###  Access Points
- **Grafana**: http://localhost:3001 (admin/SecureAdminPass123!)
- **Prometheus**: http://localhost:9092
- **AlertManager**: http://localhost:9093
- **Kibana**: http://localhost:5601 (blue team SIEM)

###  Key Dashboards
- **Cyber Range Overview**: Exercise status, team performance, infrastructure health
- **Red Team Operations**: Attack success rate, active tools, target compromise status
- **Blue Team SOC**: Detection rate, alert volume, SIEM health, response times
- **Network Security**: Traffic flows, policy violations, blocked connections

###  Alerting
- **Kill Switch**: Immediate notifications to all teams
- **Security Incidents**: Real-time attack detection and response
- **Infrastructure**: Pod restarts, resource exhaustion, system health
- **Performance**: Team scoring, exercise metrics, SLA monitoring

##  🎮 Exercise Scenarios

###  Available Scenarios
1. **Web Application Penetration Testing**: DVWA, SQL injection, XSS, CSRF
2. **Network Lateral Movement**: Internal server compromise, privilege escalation
3. **Advanced Persistent Threat (APT)**: Spear phishing, domain admin compromise
4. **Insider Threat**: Privilege abuse, data exfiltration simulation
5. **Ransomware Defense**: Encryption simulation, backup recovery testing

###  Campaign Management

```bash
# Create new campaign via API
curl -X POST http://localhost:8080/api/v1/campaigns \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Advanced Web Pentest",
    "scenario": "web_app_pentest",
    "duration_hours": 4,
    "mode": "staging"
  }'

# Start campaign
curl -X POST http://localhost:8080/api/v1/campaigns/{id}/start

# Monitor campaign status
curl http://localhost:8080/api/v1/campaigns/{id}/status
```text

##  🛡️ Security Features

###  Network Isolation
- **Kubernetes Network Policies**: Microsegmentation between teams and targets
- **iptables Rules**: Layer 3/4 filtering with rate limiting
- **VLAN Segmentation**: Physical network separation
- **DNS Filtering**: Suspicious domain blocking

###  Attack Detection
- **Real-time IDS/IPS**: Suricata with custom rules
- **Network Monitoring**: Zeek for protocol analysis
- **SIEM Integration**: ELK stack with automated correlation
- **Behavioral Analysis**: ML-powered anomaly detection

###  Data Protection
- **Encryption at Rest**: All persistent volumes encrypted
- **Encryption in Transit**: TLS for all communications
- **Access Controls**: RBAC with principle of least privilege
- **Audit Logging**: Comprehensive security event logging

##  🔧 Configuration

###  Environment Variables
```bash
# Core configuration
export CYBER_RANGE_MODE="staging"          # staging, live
export KILL_SWITCH_ENABLED="true"
export MAX_EXERCISE_DURATION="8h"
export AUTO_RESET="true"

# Network configuration
export RED_TEAM_NETWORK="10.20.0.0/16"
export BLUE_TEAM_NETWORK="10.30.0.0/24"
export TARGET_NETWORKS="10.100.0.0/24,10.110.0.0/24,10.120.0.0/24"

# Security settings
export RATE_LIMITING="true"
export MALWARE_DETECTION="true"
export GEOGRAPHIC_RESTRICTIONS="true"
```text

###  Customization
- **Target Configuration**: Add custom vulnerable applications
- **Tool Integration**: Deploy additional red/blue team tools
- **Scenario Scripting**: Create custom attack scenarios
- **Alert Rules**: Configure custom monitoring alerts

##  📚 Team Guides

###  Red Team Access
```bash
# Access red team tools
kubectl exec -it -n cyber-range-red deployment/metasploit -- /bin/bash
kubectl exec -it -n cyber-range-red deployment/attack-tools -- /bin/bash

# Check allowed targets
curl http://red-team-c2-service:8080/api/targets

# View attack logs
kubectl logs -n cyber-range-red deployment/red-team-c2 -f
```text

###  Blue Team Access
```bash
# Access SIEM interfaces
kubectl port-forward -n cyber-range-blue svc/kibana 5601:5601
kubectl port-forward -n cyber-range-blue svc/jupyter-hunting 8888:8888

# Check detection alerts
kubectl logs -n cyber-range-blue deployment/wazuh-manager -f

# Access threat hunting platform
# Jupyter: http://localhost:8888 (token in logs)
```text

###  White Team Controls
```bash
# Exercise management
kubectl port-forward -n cyber-range-control svc/xorb-admin-service 3000:3000

# Real-time monitoring
kubectl port-forward -n cyber-range-control svc/grafana-service 3001:3000

# Emergency controls
sudo ./scripts/kill-switch.sh status
sudo ./scripts/mode-switch.sh status
```text

##  🚨 Troubleshooting

###  Common Issues

####  Pods Not Starting
```bash
# Check resource quotas
kubectl describe quota -n cyber-range-red

# Check node resources
kubectl top nodes

# Check pod events
kubectl describe pod -n cyber-range-red <pod-name>
```text

####  Network Connectivity Issues
```bash
# Verify network policies
kubectl get networkpolicy --all-namespaces

# Check iptables rules
sudo iptables -L -n | grep CYBER-RANGE

# Test connectivity
kubectl exec -it -n cyber-range-red <pod> -- ping 10.100.0.10
```text

####  Kill Switch Not Working
```bash
# Check kill switch logs
tail -f /var/log/cyber-range/kill-switch.log

# Verify permissions
ls -la /opt/cyber-range/scripts/kill-switch.sh

# Manual network isolation
sudo iptables -P FORWARD DROP
```text

###  Log Locations
- **Kill Switch**: `/var/log/cyber-range/kill-switch.log`
- **Mode Switch**: `/var/log/cyber-range/mode-switch.log`
- **Firewall**: `/var/log/cyber-range/firewall-*.log`
- **Campaigns**: `/var/log/cyber-range/campaigns/`
- **Backups**: `/var/log/cyber-range/backups/`

###  Support Commands
```bash
# Health check all components
kubectl get pods --all-namespaces | grep -v Running
kubectl get services --all-namespaces

# Validate environment
./scripts/validate-environment.sh

# Generate debug report
./scripts/debug-report.sh
```text

##  🔄 Maintenance

###  Regular Tasks
- **Daily**: Review exercise logs and performance metrics
- **Weekly**: Update threat intelligence feeds and detection rules
- **Monthly**: Patch and update all container images
- **Quarterly**: Review and update attack scenarios

###  Backup Procedures
```bash
# Backup configurations
kubectl get all --all-namespaces -o yaml > cyber-range-backup.yaml

# Backup persistent data
./scripts/backup-data.sh

# Test restore procedures
./scripts/test-restore.sh
```text

###  Updates
```bash
# Update container images
kubectl set image deployment/xorb-orchestrator xorb-orchestrator=xorb/orchestrator:latest -n cyber-range-control

# Rolling restart services
kubectl rollout restart deployment --all -n cyber-range-control
```text

##  📄 License

This cyber range implementation is part of the XORB PTaaS platform and is licensed under the MIT License. See the LICENSE file for details.

##  🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test thoroughly in an isolated environment
4. Submit a pull request with detailed description

##  📞 Support

For technical support or questions:
- **Documentation**: https://docs.xorb-security.com/cyber-range
- **Issues**: https://github.com/xorb-security/cyber-range/issues
- **Email**: cyber-range-support@xorb-security.com
- **Slack**: #cyber-range-support

- --

- **⚠️ Security Notice**: This cyber range contains intentionally vulnerable systems and should only be deployed in isolated, controlled environments. Never expose cyber range components to production networks or the public internet.