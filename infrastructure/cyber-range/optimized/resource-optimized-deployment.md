#  XORB PTaaS Cyber Range - Resource Optimized Deployment
##  For 16 vCPU AMD RYZEN EPYC 7002 / 32GB RAM

This optimized configuration reduces the cyber range resource requirements by 75% while maintaining core functionality for effective Red vs Blue exercises.

##  📊 Resource Allocation Summary

###  Original vs Optimized Requirements

| Component | Original | Optimized | Savings |
|-----------|----------|-----------|---------|
| **Total CPU** | 56 vCPU | 14 vCPU | 75% |
| **Total Memory** | 120GB | 28GB | 77% |
| **Storage** | 950GB | 50GB | 95% |
| **Node Count** | 5+ nodes | 1 node | 80% |

###  Optimized Resource Distribution

| Namespace | CPU Request | CPU Limit | Memory Request | Memory Limit |
|-----------|-------------|-----------|----------------|--------------|
| **Control Plane** | 1.5 vCPU | 3 vCPU | 3GB | 6GB |
| **Red Team** | 2 vCPU | 4 vCPU | 4GB | 8GB |
| **Blue Team** | 2 vCPU | 4 vCPU | 4GB | 8GB |
| **Targets** | 1.5 vCPU | 3 vCPU | 3GB | 6GB |
| **System Reserve** | 2 vCPU | 2 vCPU | 4GB | 4GB |
| **Total** | **9 vCPU** | **16 vCPU** | **18GB** | **32GB** |

##  🎯 Optimization Strategies

###  1. Component Consolidation
- **Single Node Deployment**: All components on one AMD EPYC server
- **Shared Services**: Combined monitoring stack
- **Essential Tools Only**: Core red/blue team capabilities
- **Lightweight Alternatives**: Alpine images, single-node databases

###  2. Memory Optimization
- **Reduced JVM Heaps**: Elasticsearch 512MB (was 2GB)
- **Minimal Buffers**: Limited index buffers and caches
- **No ML/AI**: Disabled machine learning features
- **Simplified Logging**: Essential logs only

###  3. CPU Optimization
- **Single-threaded Modes**: Suricata, Elasticsearch single node
- **Reduced Concurrency**: Limited parallel operations
- **AMD EPYC Affinity**: CPU topology awareness
- **Performance Governor**: High-performance CPU scaling

###  4. Storage Optimization
- **EmptyDir Volumes**: Temporary storage for logs
- **Size Limits**: Strict storage quotas
- **Reduced Retention**: 3-day log retention (was 30 days)
- **Minimal Images**: Alpine Linux base images

##  🚀 Quick Deployment Guide

###  Prerequisites Verification
```bash
#  Check system resources
free -h
nproc
lscpu | grep "Model name"

#  Verify AMD EPYC optimizations
cat /proc/cpuinfo | grep -i epyc
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

###  Step 1: Deploy Optimized Namespaces
```bash
#  Deploy resource-optimized namespaces
kubectl apply -f optimized/resource-optimized-namespace.yaml

#  Verify resource quotas
kubectl describe quota --all-namespaces
```

###  Step 2: Deploy Control Plane (Lightweight)
```bash
#  Deploy minimal control plane
kubectl apply -f optimized/lightweight-control-plane.yaml

#  Wait for control plane
kubectl wait --for=condition=available --timeout=300s deployment --all -n cyber-range-control

#  Check resource usage
kubectl top pods -n cyber-range-control
```

###  Step 3: Deploy Essential Red Team
```bash
#  Deploy lightweight red team tools
kubectl apply -f optimized/essential-red-team.yaml

#  Verify red team deployment
kubectl get pods -n cyber-range-red
kubectl top pods -n cyber-range-red
```

###  Step 4: Deploy Blue Team SIEM Lite
```bash
#  Deploy minimal SIEM stack
kubectl apply -f optimized/lightweight-blue-team.yaml

#  Wait for Elasticsearch to start (may take 2-3 minutes)
kubectl wait --for=condition=ready --timeout=600s pod -l app=elasticsearch-lite -n cyber-range-blue

#  Check SIEM status
kubectl get pods -n cyber-range-blue
```

###  Step 5: Deploy Minimal Targets
```bash
#  Deploy 3 essential targets
kubectl apply -f optimized/minimal-targets.yaml

#  Verify targets are running
kubectl get pods -n cyber-range-targets
kubectl get svc -n cyber-range-targets
```

###  Step 6: Configure AMD EPYC Optimizations
```bash
#  Set CPU governor to performance
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

#  Disable swap for better performance
sudo swapoff -a

#  Configure transparent huge pages
echo 'madvise' | sudo tee /sys/kernel/mm/transparent_hugepage/enabled

#  Set vm.swappiness for memory optimization
echo 'vm.swappiness = 1' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

##  🔧 Access Points (NodePort Services)

Since we're using a single node, services are exposed via NodePort:

```bash
#  Get node IP
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')

echo "Access Points:"
echo "Admin Console:    http://$NODE_IP:30080"
echo "Grafana:          http://$NODE_IP:30300 (admin/admin)"
echo "Prometheus:       http://$NODE_IP:30090"
echo "Kibana (SIEM):    http://$NODE_IP:30601"
echo "Elasticsearch:    http://$NODE_IP:30920"
echo "Blue Team SOC:    http://$NODE_IP:30800"
```

##  🎮 Optimized Exercise Scenarios

###  Available Scenarios (Resource Conscious)
1. **Web Application Basics**: Single DVWA target with SQL injection, XSS
2. **Network Fundamentals**: SSH brute force and privilege escalation
3. **File Server Compromise**: FTP anonymous access and credential theft

###  Simplified Red Team Operations
```bash
#  Access red team tools
kubectl exec -it -n cyber-range-red deployment/red-team-essential-tools -- /bin/sh

#  Basic network scan
nmap -sS -F web-target-service.cyber-range-targets.svc.cluster.local

#  Web vulnerability testing
curl "http://web-target-service.cyber-range-targets.svc.cluster.local/?id=1' OR '1'='1"

#  SSH brute force attempt
nmap -p 22 --script ssh-brute network-target-service.cyber-range-targets.svc.cluster.local
```

###  Blue Team Monitoring (Lightweight)
```bash
#  Access Kibana for log analysis
#  URL: http://NODE_IP:30601

#  Basic queries in Kibana:
#  - index: cyber-range-logs-*
#  - Search: log_type:container AND kubernetes.namespace:cyber-range-red
#  - Time range: Last 1 hour

#  Check Suricata alerts
kubectl logs -n cyber-range-blue daemonset/suricata-lite | grep "ALERT"
```

##  🚨 Resource Monitoring

###  Real-time Resource Usage
```bash
#  Monitor overall cluster resources
watch kubectl top nodes

#  Monitor per-namespace usage
watch kubectl top pods --all-namespaces

#  Check resource quotas
kubectl describe quota --all-namespaces
```

###  Performance Tuning Commands
```bash
#  Check memory pressure
kubectl describe nodes | grep -A 5 "Conditions"

#  Monitor disk usage
df -h

#  Check for OOM kills
dmesg | grep -i "killed process"

#  Monitor CPU throttling
cat /sys/fs/cgroup/cpu/cpu.stat
```

##  🔧 Troubleshooting Optimized Deployment

###  Common Resource Issues

####  Issue: Pods Stuck in Pending (Insufficient Resources)
```bash
#  Check resource availability
kubectl describe nodes

#  Check resource quotas
kubectl describe quota -n cyber-range-targets

#  Solution: Reduce resource requests
kubectl patch deployment web-target-lite -n cyber-range-targets -p '{"spec":{"template":{"spec":{"containers":[{"name":"apache-php","resources":{"requests":{"memory":"64Mi","cpu":"25m"}}}]}}}}'
```

####  Issue: Elasticsearch Won't Start (Memory Issues)
```bash
#  Check Elasticsearch logs
kubectl logs -n cyber-range-blue deployment/elasticsearch-lite

#  Reduce memory further if needed
kubectl patch deployment elasticsearch-lite -n cyber-range-blue -p '{"spec":{"template":{"spec":{"containers":[{"name":"elasticsearch","env":[{"name":"ES_JAVA_OPTS","value":"-Xms128m -Xmx256m"}]}]}}}}'
```

####  Issue: High Memory Usage
```bash
#  Identify memory hogs
kubectl top pods --all-namespaces --sort-by=memory

#  Restart high-memory pods
kubectl rollout restart deployment elasticsearch-lite -n cyber-range-blue

#  Enable memory limits enforcement
kubectl patch deployment <deployment-name> -n <namespace> -p '{"spec":{"template":{"spec":{"containers":[{"name":"<container>","resources":{"limits":{"memory":"256Mi"}}}]}}}}'
```

##  📊 Performance Expectations

###  Expected Performance Metrics
- **Exercise Duration**: 1-2 hours (optimized scenarios)
- **Concurrent Users**: 2-4 red team members, 2-3 blue team members
- **Attack Success Rate**: 80-90% (simplified targets)
- **Detection Rate**: 60-70% (lightweight monitoring)
- **System Response Time**: <5 seconds for most operations

###  Resource Utilization Targets
- **CPU Usage**: 60-80% during active exercises
- **Memory Usage**: 70-85% of available RAM
- **Disk I/O**: Minimal due to ephemeral storage
- **Network**: <10% of available bandwidth

##  🎯 Exercise Workflow (Optimized)

###  1. Pre-Exercise Setup (5 minutes)
```bash
#  Verify all components running
kubectl get pods --all-namespaces | grep -v Running

#  Check resource usage baseline
kubectl top nodes && kubectl top pods --all-namespaces
```

###  2. Exercise Execution (60-90 minutes)
```bash
#  Red Team: Start with reconnaissance
kubectl exec -it -n cyber-range-red deployment/red-team-essential-tools -- nmap -sn 10.100.0.0/24

#  Blue Team: Monitor in Kibana
#  Access: http://NODE_IP:30601
```

###  3. Post-Exercise Analysis (15 minutes)
```bash
#  Export logs for analysis
kubectl logs -n cyber-range-blue deployment/elasticsearch-lite > exercise-logs.txt

#  Generate resource usage report
kubectl top pods --all-namespaces > resource-usage-report.txt
```

##  💡 Optimization Tips

###  AMD EPYC Specific Optimizations
```bash
#  Enable NUMA balancing
echo 1 | sudo tee /proc/sys/kernel/numa_balancing

#  Set CPU affinity for Kubernetes
systemctl edit kubelet
#  Add: Environment="KUBELET_EXTRA_ARGS=--cpu-manager-policy=static"

#  Optimize network stack
echo 'net.core.rmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
```

###  Container Optimizations
```bash
#  Use memory-efficient base images
#  alpine:latest (5MB) vs ubuntu:latest (72MB)

#  Disable debug logging in production
kubectl set env deployment --all LOG_LEVEL=WARN --all-namespaces

#  Enable resource limits enforcement
kubectl patch deployment --all --type='merge' -p='{"spec":{"template":{"spec":{"containers":[{"resources":{"limits":{"memory":"512Mi","cpu":"500m"}}}]}}}}' --all-namespaces
```

---

##  🎉 Deployment Complete!

Your optimized XORB PTaaS Cyber Range is now running efficiently on your 16 vCPU AMD RYZEN EPYC 7002 system with 32GB RAM. The system provides:

✅ **75% Resource Reduction** while maintaining core functionality
✅ **Essential Red vs Blue Capabilities** for effective training
✅ **Real-time Monitoring** with lightweight SIEM
✅ **Emergency Controls** with kill switch functionality
✅ **AMD EPYC Optimizations** for maximum performance

**Next Steps**:
1. Access the admin console at `http://NODE_IP:30080`
2. Start your first exercise with staging mode
3. Monitor resource usage and adjust as needed
4. Scale up individual components if you have spare resources