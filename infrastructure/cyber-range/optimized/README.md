# XORB PTaaS Cyber Range - Resource Optimized Edition
##  Optimized for 16 vCPU AMD RYZEN EPYC 7002 / 32GB RAM

This optimized version reduces resource requirements by **75%** while maintaining essential Red vs Blue cyber range functionality.

##  📊 Resource Optimization Summary

| Metric | Original | Optimized | Reduction |
|--------|----------|-----------|-----------|
| **CPU Requirements** | 56 vCPU | 14 vCPU | **75%** |
| **Memory Requirements** | 120GB | 28GB | **77%** |
| **Storage Requirements** | 950GB | 50GB | **95%** |
| **Node Count** | 5+ nodes | 1 node | **80%** |
| **Component Count** | 25+ services | 12 services | **52%** |

##  🎯 Optimized Architecture

###  Single-Node Deployment
```text
┌─────────────────────────────────────────────────────────────┐
│              AMD RYZEN EPYC 7002 (16 vCPU / 32GB)          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Control     │ │ Red Team    │ │ Blue Team   │          │
│  │ Plane       │ │ Essential   │ │ SIEM Lite   │          │
│  │ 3v/6G       │ │ 4v/8G       │ │ 4v/8G       │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
│                   ┌─────────────┐                          │
│                   │ Targets     │                          │
│                   │ Minimal     │                          │
│                   │ 3v/6G       │                          │
│                   └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```text

###  Resource Distribution
- **System Reserve**: 2 vCPU / 4GB RAM
- **Control Plane**: 3 vCPU / 6GB RAM (XORB Orchestrator, Grafana, Prometheus)
- **Red Team**: 4 vCPU / 8GB RAM (Essential tools, lightweight C2)
- **Blue Team**: 4 vCPU / 8GB RAM (Elasticsearch, Kibana, Suricata)
- **Targets**: 3 vCPU / 6GB RAM (3 vulnerable applications)

##  🚀 Quick Start Guide

###  Prerequisites
```bash
# Verify system resources
free -h
nproc
lscpu | grep -i epyc

# Ensure Kubernetes is running
kubectl cluster-info
kubectl get nodes
```text

###  1. Deploy Optimized Infrastructure
```bash
# Clone and navigate to optimized deployment
git clone https://github.com/xorb-security/cyber-range.git
cd cyber-range/infrastructure/cyber-range/optimized

# Deploy all components in order
kubectl apply -f resource-optimized-namespace.yaml
kubectl apply -f lightweight-control-plane.yaml
kubectl apply -f essential-red-team.yaml
kubectl apply -f lightweight-blue-team.yaml
kubectl apply -f minimal-targets.yaml

# Wait for all pods to be ready
kubectl wait --for=condition=ready pod --all --all-namespaces --timeout=600s
```text

###  2. Configure Lightweight Firewall
```bash
# Setup firewall for staging mode (safe training)
sudo ./lightweight-firewall.sh staging

# Verify firewall status
sudo ./lightweight-firewall.sh status
```text

###  3. Access Services
```bash
# Get node IP for NodePort services
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')

echo "🎮 Cyber Range Access Points:"
echo "Admin Console:    http://$NODE_IP:30080"
echo "Grafana:          http://$NODE_IP:30300 (admin/admin)"
echo "Prometheus:       http://$NODE_IP:30090"
echo "Kibana (SIEM):    http://$NODE_IP:30601"
echo "Blue Team SOC:    http://$NODE_IP:30800"
```text

##  🔴 Red Team Operations (Essential Tools)

###  Available Tools
- **Network Scanning**: Nmap (lightweight configuration)
- **Web Testing**: curl, wget, basic web vulnerability testing
- **Command & Control**: Lightweight nginx-based C2
- **Basic Payloads**: Educational reverse shells and web shells

###  Red Team Access
```bash
# Access essential red team tools
kubectl exec -it -n cyber-range-red deployment/red-team-essential-tools -- /bin/sh

# Quick network discovery
nmap -sn 10.244.0.0/16

# Web vulnerability testing
curl "http://web-target-service.cyber-range-targets.svc.cluster.local/?id=1' OR '1'='1"

# SSH brute force simulation
nmap -p 22 network-target-service.cyber-range-targets.svc.cluster.local
```text

###  Simplified Attack Scenarios
1. **Web App Exploitation**: SQL injection, XSS on DVWA-like target
2. **SSH Brute Force**: Weak credentials on network target
3. **File Server Access**: Anonymous FTP and weak authentication

##  🔵 Blue Team Operations (SIEM Lite)

###  Available Capabilities
- **Log Collection**: Filebeat for container and system logs
- **Search & Analysis**: Single-node Elasticsearch with 512MB heap
- **Visualization**: Kibana with essential dashboards
- **Intrusion Detection**: Lightweight Suricata IDS
- **Monitoring Dashboard**: Simple web interface for SOC status

###  Blue Team Access
```bash
# Access Kibana for log analysis
echo "Kibana: http://$NODE_IP:30601"

# Check Suricata alerts
kubectl logs -n cyber-range-blue daemonset/suricata-lite | grep ALERT

# Monitor container logs
kubectl logs -f -n cyber-range-targets deployment/web-target-lite
```text

###  Lightweight SIEM Queries
```json
# In Kibana Discover (index: cyber-range-logs-*)

# Red team activity
kubernetes.namespace:cyber-range-red AND message:*attack*

# Failed login attempts
message:*failed* OR message:*invalid*

# Web requests to targets
kubernetes.namespace:cyber-range-targets AND message:*GET*
```text

##  🎯 Target Environment (Minimal)

###  Available Targets
1. **Web Application**: PHP-based vulnerable app with SQL injection, XSS
2. **SSH Server**: Alpine Linux with weak credentials
3. **File Server**: FTP server with anonymous access

###  Target Information
```bash
# Get target service information
kubectl get svc -n cyber-range-targets

# Access target discovery info
curl http://$NODE_IP:30800/targets.json
```text

##  🚨 Emergency Controls

###  Kill Switch (Lightweight)
```bash
# Immediate attack blocking
sudo ./lightweight-firewall.sh kill

# Check kill switch status
sudo ./lightweight-firewall.sh status

# Restore to staging mode
sudo ./lightweight-firewall.sh staging
```text

###  Mode Switching
```bash
# Safe training mode (default)
sudo ./lightweight-firewall.sh staging

# Live exercise mode (allows real attacks)
sudo ./lightweight-firewall.sh live

# Clear all cyber range rules
sudo ./lightweight-firewall.sh clear
```text

##  📊 Resource Monitoring

###  Real-time Monitoring
```bash
# Monitor cluster resources
watch kubectl top nodes

# Monitor pod resources
watch kubectl top pods --all-namespaces

# Check resource quotas
kubectl describe quota --all-namespaces
```text

###  Performance Optimization
```bash
# AMD EPYC specific optimizations (automatic in firewall script)
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Memory optimization
echo 1 | sudo tee /proc/sys/vm/swappiness

# Network optimization for iptables
echo 65536 | sudo tee /proc/sys/net/netfilter/nf_conntrack_max
```text

##  🔧 Troubleshooting

###  Common Issues

####  Pods Stuck in Pending (Insufficient Resources)
```bash
# Check available resources
kubectl describe nodes | grep -A 5 Allocatable

# Reduce resource requests if needed
kubectl patch deployment web-target-lite -n cyber-range-targets -p '{"spec":{"template":{"spec":{"containers":[{"name":"apache-php","resources":{"requests":{"memory":"64Mi","cpu":"25m"}}}]}}}}'
```text

####  Elasticsearch Won't Start (Memory Issues)
```bash
# Check Elasticsearch logs
kubectl logs -n cyber-range-blue deployment/elasticsearch-lite

# Reduce heap size further
kubectl patch deployment elasticsearch-lite -n cyber-range-blue -p '{"spec":{"template":{"spec":{"containers":[{"name":"elasticsearch","env":[{"name":"ES_JAVA_OPTS","value":"-Xms128m -Xmx256m"}]}]}}}}'
```text

####  High Memory Usage
```bash
# Find memory-intensive pods
kubectl top pods --all-namespaces --sort-by=memory

# Restart high-memory components
kubectl rollout restart deployment elasticsearch-lite -n cyber-range-blue
```text

###  Resource Tuning
```bash
# Increase limits if you have spare resources
kubectl patch deployment <deployment> -n <namespace> -p '{"spec":{"template":{"spec":{"containers":[{"name":"<container>","resources":{"limits":{"memory":"1Gi","cpu":"500m"}}}]}}}}'

# Decrease limits to save resources
kubectl patch deployment <deployment> -n <namespace> -p '{"spec":{"template":{"spec":{"containers":[{"name":"<container>","resources":{"requests":{"memory":"32Mi","cpu":"10m"}}}]}}}}'
```text

##  🎮 Exercise Workflow

###  1. Pre-Exercise (5 min)
```bash
# Verify all systems operational
kubectl get pods --all-namespaces | grep -v Running

# Check baseline resource usage
kubectl top nodes && kubectl top pods --all-namespaces

# Ensure staging mode active
sudo ./lightweight-firewall.sh status
```text

###  2. Exercise Briefing (10 min)
- **Red Team**: Access essential tools, identify 3 targets
- **Blue Team**: Monitor Kibana dashboard, set up alert monitoring
- **White Team**: Monitor resource usage, prepare kill switch

###  3. Exercise Execution (60-90 min)
```bash
# Red Team starts with reconnaissance
kubectl exec -it -n cyber-range-red deployment/red-team-essential-tools -- nmap -sF target-discovery-service.cyber-range-targets.svc.cluster.local

# Blue Team monitors for activity
# Access Kibana at http://NODE_IP:30601

# Switch to live mode when ready
sudo ./lightweight-firewall.sh live
```text

###  4. Post-Exercise Analysis (15 min)
```bash
# Export logs for analysis
kubectl logs -n cyber-range-blue deployment/elasticsearch-lite > exercise-logs.json

# Generate resource usage report
kubectl top pods --all-namespaces > resource-report.txt

# Return to staging mode
sudo ./lightweight-firewall.sh staging
```text

##  💡 Optimization Features

###  AMD EPYC Optimizations
- **CPU Governor**: Performance mode for maximum throughput
- **NUMA Awareness**: Memory allocation optimization
- **Thread Affinity**: Optimized for EPYC architecture
- **Cache Optimization**: L3 cache utilization improvements

###  Container Optimizations
- **Alpine Images**: 5MB base vs 72MB Ubuntu
- **Single Process**: Minimal init systems
- **Resource Limits**: Strict memory and CPU boundaries
- **Efficient Logging**: Reduced log verbosity

###  Network Optimizations
- **Simplified Rules**: <50 iptables rules total
- **String Matching**: Efficient namespace-based filtering
- **Reduced Logging**: Only critical events logged
- **Connection Limits**: Optimized for single-node deployment

##  🎯 Expected Performance

###  Capacity Metrics
- **Concurrent Users**: 4-6 (2-3 red team, 2-3 blue team)
- **Exercise Duration**: 1-2 hours optimal
- **Attack Success Rate**: 80-90% (simplified targets)
- **Detection Rate**: 60-70% (lightweight monitoring)

###  Resource Utilization
- **CPU Usage**: 60-80% during active exercises
- **Memory Usage**: 70-85% of available RAM
- **Network**: <5% of available bandwidth
- **Storage**: <2GB persistent data

##  📈 Scaling Options

###  If You Have Spare Resources
```bash
# Increase Elasticsearch heap
kubectl patch deployment elasticsearch-lite -n cyber-range-blue -p '{"spec":{"template":{"spec":{"containers":[{"name":"elasticsearch","env":[{"name":"ES_JAVA_OPTS","value":"-Xms512m -Xmx1g"}]}]}}}}'

# Add more red team tools
kubectl scale deployment red-team-essential-tools --replicas=2 -n cyber-range-red

# Enable additional monitoring
kubectl patch deployment grafana-lite -n cyber-range-control -p '{"spec":{"template":{"spec":{"containers":[{"name":"grafana","env":[{"name":"GF_ALERTING_ENABLED","value":"true"}]}]}}}}'
```text

###  If You Need to Reduce Further
```bash
# Reduce to minimal configuration
kubectl scale deployment grafana-lite --replicas=0 -n cyber-range-control
kubectl scale deployment target-monitor --replicas=0 -n cyber-range-targets

# Use only essential targets
kubectl delete deployment file-target-lite -n cyber-range-targets
```text

- --

##  🎉 Success!

Your XORB PTaaS Cyber Range is now running efficiently on your 16 vCPU AMD RYZEN EPYC 7002 system with 32GB RAM!

- *Key Benefits:**
✅ **75% Resource Reduction** - Fits on single powerful server
✅ **Essential Functionality** - Core red vs blue capabilities maintained
✅ **AMD EPYC Optimized** - Maximum performance from your hardware
✅ **Production Ready** - Real attacks and monitoring in lightweight package
✅ **Emergency Controls** - Kill switch and safety measures included

- *Next Steps:**
1. Run your first exercise in staging mode
2. Monitor resource usage and tune as needed
3. Graduate to live mode for real attack scenarios
4. Scale individual components based on usage patterns

- *Support:** Check the main repository documentation for detailed guides and troubleshooting assistance.