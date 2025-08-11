# XORB PTaaS Cyber Range - Deployment Guide

This comprehensive guide walks you through deploying the XORB PTaaS Red vs Blue cyber range from initial setup to first exercise execution.

##  📋 Prerequisites Checklist

###  Infrastructure Requirements
- [ ] Kubernetes cluster (minimum v1.28)
- [ ] Minimum resources: 56 vCPU, 120GB RAM, 950GB storage
- [ ] Fast SSD storage class available
- [ ] Load balancer support (MetalLB, cloud provider, etc.)
- [ ] Container registry access (Docker Hub or private registry)

###  Software Requirements
- [ ] kubectl (v1.28+) configured and connected
- [ ] Docker (v20.0+) for building custom images
- [ ] Terraform (v1.5+) if using cloud deployment
- [ ] Git for cloning repositories
- [ ] jq for JSON processing
- [ ] curl for API testing

###  Network Requirements
- [ ] Cluster nodes can communicate on all required ports
- [ ] External load balancer IPs available
- [ ] DNS resolution working (optional but recommended)
- [ ] Firewall rules allowing required traffic

###  Access Requirements
- [ ] Cluster admin permissions
- [ ] Root/sudo access on nodes (for iptables configuration)
- [ ] Container registry push permissions (if building custom images)

##  🏗️ Phase 1: Infrastructure Preparation

###  Step 1.1: Validate Cluster Resources
```bash
# Check cluster info
kubectl cluster-info

# Verify node resources
kubectl top nodes

# Check available storage classes
kubectl get storageclass

# Verify load balancer capability
kubectl get svc -A | grep LoadBalancer
```

Expected output should show sufficient resources and working load balancer.

###  Step 1.2: Create Storage Classes (if needed)
```bash
# Example fast SSD storage class for critical components
cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs  # or your provider
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
EOF
```

###  Step 1.3: Verify Required Commands
```bash
# Test required tools
for cmd in kubectl docker terraform jq curl; do
    if command -v $cmd >/dev/null 2>&1; then
        echo "✅ $cmd is available"
    else
        echo "❌ $cmd is NOT available - please install"
    fi
done
```

##  🚀 Phase 2: Core Infrastructure Deployment

###  Step 2.1: Deploy Namespaces and Network Policies
```bash
# Clone the repository (if not already done)
git clone https://github.com/xorb-security/cyber-range.git
cd cyber-range/infrastructure/cyber-range

# Create all namespaces
kubectl apply -f k8s/namespace.yaml

# Verify namespaces created
kubectl get namespaces | grep cyber-range

# Apply network policies
kubectl apply -f k8s/network-policies.yaml

# Verify network policies
kubectl get networkpolicy --all-namespaces
```

Expected output:
```
cyber-range            Active   <age>
cyber-range-blue       Active   <age>
cyber-range-control    Active   <age>
cyber-range-red        Active   <age>
cyber-range-simulation Active   <age>
cyber-range-targets    Active   <age>
```

###  Step 2.2: Deploy Control Plane
```bash
# Deploy control plane components
kubectl apply -f k8s/control-plane.yaml

# Wait for control plane to be ready
kubectl wait --for=condition=available --timeout=300s deployment --all -n cyber-range-control

# Check control plane status
kubectl get pods -n cyber-range-control
```

Monitor the pods until all are running:
```bash
# Watch pods come online
watch kubectl get pods -n cyber-range-control
```

###  Step 2.3: Configure Secrets
```bash
# Create required secrets
kubectl create secret generic xorb-db-secret \
  --from-literal=url="postgresql://username:password@host:5432/database" \
  -n cyber-range-control

kubectl create secret generic xorb-redis-secret \
  --from-literal=url="redis://redis-host:6379" \
  -n cyber-range-control

kubectl create secret generic grafana-secret \
  --from-literal=admin-password="SecureAdminPass123!" \
  -n cyber-range-control

# Verify secrets created
kubectl get secrets -n cyber-range-control
```

##  🔴 Phase 3: Red Team Infrastructure

###  Step 3.1: Deploy Red Team Components
```bash
# Deploy red team infrastructure
kubectl apply -f k8s/red-team.yaml

# Wait for red team components
kubectl wait --for=condition=available --timeout=600s deployment --all -n cyber-range-red

# Check red team status
kubectl get pods -n cyber-range-red
```

###  Step 3.2: Configure Red Team Tools
```bash
# Create red team secrets
kubectl create secret generic red-team-db-secret \
  --from-literal=url="postgresql://red_user:red_pass@postgres:5432/red_db" \
  -n cyber-range-red

kubectl create secret generic phishing-secret \
  --from-literal=admin-password="RedTeamPhish123!" \
  -n cyber-range-red

# Configure Metasploit scripts
kubectl create configmap metasploit-scripts \
  --from-literal=init.rc="
workspace -a cyber_range
setg RHOSTS 10.100.0.0/24
setg THREADS 10
setg VERBOSE true
" \
  -n cyber-range-red

# Verify red team deployment
kubectl get all -n cyber-range-red
```

##  🔵 Phase 4: Blue Team Infrastructure

###  Step 4.1: Deploy Blue Team Components
```bash
# Deploy blue team infrastructure
kubectl apply -f k8s/blue-team.yaml

# Wait for blue team components (this may take longer due to ELK stack)
kubectl wait --for=condition=available --timeout=900s deployment --all -n cyber-range-blue

# Check blue team status
kubectl get pods -n cyber-range-blue
```

###  Step 4.2: Configure Blue Team SIEM
```bash
# Create blue team secrets
kubectl create secret generic jupyter-secret \
  --from-literal=token="BlueTeamJupyter123!" \
  -n cyber-range-blue

kubectl create secret generic misp-secret \
  --from-literal=admin-passphrase="BlueTeamMISP123!" \
  -n cyber-range-blue

# Configure Logstash pipelines
kubectl create configmap logstash-config \
  --from-literal=logstash.conf="
input {
  beats {
    port => 5044
  }
}
filter {
  if [fields][log_type] == \"cyber_range\" {
    grok {
      match => { \"message\" => \"%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} %{GREEDYDATA:message}\" }
    }
  }
}
output {
  elasticsearch {
    hosts => [\"elasticsearch:9200\"]
    index => \"cyber-range-%{+YYYY.MM.dd}\"
  }
}
" \
  -n cyber-range-blue

# Configure Suricata rules
kubectl create configmap suricata-rules \
  --from-literal=cyber-range.rules="
alert tcp any any -> 10.100.0.0/24 any (msg:\"Red Team Attack Detected\"; sid:1000001;)
alert tcp any any -> 10.110.0.0/24 any (msg:\"Internal Network Attack\"; sid:1000002;)
alert tcp any any -> any 4444 (msg:\"Reverse Shell Detected\"; sid:1000003;)
" \
  -n cyber-range-blue

# Verify blue team deployment
kubectl get all -n cyber-range-blue
```

##  🎯 Phase 5: Target Environment Deployment

###  Step 5.1: Deploy Target Systems
```bash
# Create target environment configmap
kubectl create configmap target-scenarios \
  --from-literal=web-scenario="dvwa,webgoat,mutillidae" \
  --from-literal=network-scenario="ubuntu,centos,windows" \
  --from-literal=iot-scenario="modbus,mqtt,ics" \
  -n cyber-range-targets

# Deploy target systems (this creates a basic set)
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dvwa-target
  namespace: cyber-range-targets
  labels:
    app: dvwa-target
    cyber-range.xorb.io/target-type: web
spec:
  replicas: 1
  selector:
    matchLabels:
      app: dvwa-target
  template:
    metadata:
      labels:
        app: dvwa-target
        cyber-range.xorb.io/target-type: web
    spec:
      containers:
      - name: dvwa
        image: vulnerables/web-dvwa:latest
        ports:
        - containerPort: 80
        env:
        - name: MYSQL_DATABASE
          value: dvwa
        - name: MYSQL_USER
          value: dvwa
        - name: MYSQL_PASSWORD
          value: p@ssw0rd
- --
apiVersion: v1
kind: Service
metadata:
  name: dvwa-target-service
  namespace: cyber-range-targets
spec:
  selector:
    app: dvwa-target
  ports:
  - port: 80
    targetPort: 80
  type: ClusterIP
EOF

# Verify target deployment
kubectl get pods -n cyber-range-targets
```

##  📊 Phase 6: Monitoring Stack Deployment

###  Step 6.1: Deploy Prometheus
```bash
# Apply Prometheus configuration
kubectl apply -f monitoring/prometheus-config.yaml

# Deploy Prometheus
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: cyber-range-control
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:v2.45.0
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: prometheus-config
          mountPath: /etc/prometheus
        - name: prometheus-storage
          mountPath: /prometheus
      volumes:
      - name: prometheus-config
        configMap:
          name: prometheus-config
      - name: prometheus-storage
        emptyDir: {}
- --
apiVersion: v1
kind: Service
metadata:
  name: prometheus-service
  namespace: cyber-range-control
spec:
  selector:
    app: prometheus
  ports:
  - port: 9090
    targetPort: 9090
  type: LoadBalancer
EOF
```

###  Step 6.2: Deploy Grafana
```bash
# Apply Grafana configuration
kubectl apply -f monitoring/grafana-config.yaml

# Deploy Grafana
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: cyber-range-control
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:10.0.0
        ports:
        - containerPort: 3000
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          valueFrom:
            secretKeyRef:
              name: grafana-secret
              key: admin-password
        volumeMounts:
        - name: grafana-config
          mountPath: /etc/grafana/provisioning
      volumes:
      - name: grafana-config
        configMap:
          name: grafana-config
- --
apiVersion: v1
kind: Service
metadata:
  name: grafana-service
  namespace: cyber-range-control
spec:
  selector:
    app: grafana
  ports:
  - port: 3000
    targetPort: 3000
  type: LoadBalancer
EOF
```

###  Step 6.3: Deploy AlertManager
```bash
# Apply AlertManager configuration
kubectl apply -f monitoring/alertmanager-config.yaml

# Wait for monitoring stack
kubectl wait --for=condition=available --timeout=300s deployment prometheus grafana -n cyber-range-control
```

##  🔧 Phase 7: Firewall and Security Configuration

###  Step 7.1: Setup Firewall Scripts
```bash
# Make scripts executable
chmod +x scripts/*.sh firewall/*.sh

# Install iptables if not already installed
sudo apt-get update && sudo apt-get install -y iptables

# Verify iptables functionality
sudo iptables -L
```

###  Step 7.2: Configure Initial Firewall Rules (Staging Mode)
```bash
# Apply staging mode firewall rules
sudo ./firewall/iptables-staging.sh

# Verify firewall configuration
sudo ./firewall/iptables-staging.sh verify

# Check current mode
sudo ./scripts/mode-switch.sh status
```

Expected output should show "STAGING" mode with red team attacks blocked.

##  🧪 Phase 8: Validation and Testing

###  Step 8.1: Verify All Components
```bash
# Check all pods are running
kubectl get pods --all-namespaces | grep -v Running

# Check services
kubectl get services --all-namespaces

# Check network policies
kubectl get networkpolicy --all-namespaces

# Check storage
kubectl get pvc --all-namespaces
```

###  Step 8.2: Test Connectivity
```bash
# Get service IPs
kubectl get svc --all-namespaces

# Test admin console access
kubectl port-forward -n cyber-range-control svc/xorb-admin-service 3000:3000 &

# Test Grafana access
kubectl port-forward -n cyber-range-control svc/grafana-service 3001:3000 &

# Test in browser
curl -I http://localhost:3000
curl -I http://localhost:3001
```

###  Step 8.3: Test Kill Switch
```bash
# Test kill switch functionality (dry run)
sudo ./scripts/kill-switch.sh test

# Expected output should show all components available
```

###  Step 8.4: Test Mode Switching
```bash
# Test mode switch (dry run)
sudo ./scripts/mode-switch.sh dry-run live

# Verify current mode
sudo ./scripts/mode-switch.sh status
```

##  🎮 Phase 9: First Exercise Setup

###  Step 9.1: Access Control Interfaces
```bash
# Access admin console
kubectl port-forward -n cyber-range-control svc/xorb-admin-service 3000:3000 &
echo "Admin Console: http://localhost:3000"

# Access Grafana
kubectl port-forward -n cyber-range-control svc/grafana-service 3001:3000 &
echo "Grafana: http://localhost:3001 (admin/SecureAdminPass123!)"

# Access Blue Team Kibana
kubectl port-forward -n cyber-range-blue svc/kibana 5601:5601 &
echo "Kibana: http://localhost:5601"
```

###  Step 9.2: Create First Campaign
```bash
# Test orchestrator API
kubectl port-forward -n cyber-range-control svc/xorb-orchestrator-service 8080:8080 &

# Create test campaign
curl -X POST http://localhost:8080/api/v1/campaigns \
  -H "Content-Type: application/json" \
  -d '{
    "name": "First Test Exercise",
    "description": "Initial validation exercise",
    "scenario": "web_app_pentest",
    "duration_hours": 2,
    "mode": "staging"
  }'

# Check campaign status
curl http://localhost:8080/api/v1/campaigns
```

###  Step 9.3: Verify Team Access
```bash
# Red team tool access (should be blocked in staging)
kubectl exec -it -n cyber-range-red deployment/attack-tools -- nmap -p 80 10.100.0.10

# Blue team monitoring access
kubectl exec -it -n cyber-range-blue deployment/kibana -- curl -I http://elasticsearch:9200

# Target accessibility
kubectl exec -it -n cyber-range-targets deployment/dvwa-target -- curl -I http://localhost
```

##  🚨 Phase 10: Emergency Procedures Testing

###  Step 10.1: Test Kill Switch
```bash
# Test kill switch activation
sudo ./scripts/kill-switch.sh activate manual_test

# Verify all attack traffic is blocked
kubectl exec -it -n cyber-range-red deployment/attack-tools -- ping -c 3 10.100.0.10

# Check kill switch status
sudo ./scripts/kill-switch.sh status

# Deactivate kill switch
sudo ./scripts/kill-switch.sh deactivate
```

###  Step 10.2: Test Mode Switching to Live
```bash
# Switch to live mode (with confirmation)
sudo ./scripts/mode-switch.sh live

# Verify live mode active
sudo ./scripts/mode-switch.sh status

# Test red team can now reach targets
kubectl exec -it -n cyber-range-red deployment/attack-tools -- ping -c 3 10.100.0.10

# Switch back to staging
sudo ./scripts/mode-switch.sh staging
```

##  ✅ Phase 11: Deployment Validation Checklist

###  Infrastructure Validation
- [ ] All namespaces created and active
- [ ] All pods running and healthy
- [ ] Services accessible via port-forwarding
- [ ] Persistent volumes mounted correctly
- [ ] Network policies applied and effective

###  Security Validation
- [ ] Staging mode blocks red team attacks
- [ ] Live mode allows controlled attacks
- [ ] Kill switch immediately blocks all attacks
- [ ] Blue team monitoring captures all activity
- [ ] Network segmentation working correctly

###  Monitoring Validation
- [ ] Prometheus collecting metrics from all components
- [ ] Grafana dashboards loading with data
- [ ] AlertManager configured and responsive
- [ ] Log aggregation working in ELK stack
- [ ] Real-time monitoring shows exercise activity

###  Operational Validation
- [ ] Mode switching works reliably
- [ ] Campaign creation and management functional
- [ ] Emergency procedures tested and working
- [ ] Team access controls properly configured
- [ ] Exercise scenarios deploy successfully

##  🔧 Troubleshooting Common Issues

###  Issue: Pods Stuck in Pending State
```bash
# Check node resources
kubectl top nodes

# Check resource quotas
kubectl describe quota --all-namespaces

# Check storage class
kubectl get storageclass
kubectl describe pvc -n <namespace>

# Solution: Scale cluster or adjust resource requests
```

###  Issue: Network Connectivity Problems
```bash
# Check network policies
kubectl get networkpolicy --all-namespaces
kubectl describe networkpolicy <policy-name> -n <namespace>

# Check iptables rules
sudo iptables -L -n | grep CYBER-RANGE

# Solution: Verify firewall rules and network policies
sudo ./firewall/iptables-staging.sh verify
```

###  Issue: Monitoring Not Working
```bash
# Check Prometheus targets
kubectl port-forward -n cyber-range-control svc/prometheus-service 9090:9090
# Visit http://localhost:9090/targets

# Check service discovery
kubectl get pods -n cyber-range-control -l app=prometheus
kubectl logs -n cyber-range-control deployment/prometheus

# Solution: Verify service annotations and network policies
```

###  Issue: Kill Switch Not Responsive
```bash
# Check script permissions
ls -la scripts/kill-switch.sh

# Check logs
tail -f /var/log/cyber-range/kill-switch.log

# Manual emergency isolation
sudo iptables -P FORWARD DROP
```

##  📞 Support and Next Steps

###  Documentation
- **Architecture Guide**: See `infrastructure/cyber-range/README.md`
- **API Documentation**: Available at http://localhost:8080/docs
- **Monitoring Guide**: Check Grafana dashboards for operational metrics

###  Support Channels
- **Issues**: Report problems via GitHub issues
- **Documentation**: Check the comprehensive README.md
- **Community**: Join the discussion forums

###  Next Steps
1. **Team Training**: Familiarize red and blue teams with their tools
2. **Scenario Customization**: Add organization-specific targets and scenarios
3. **Integration**: Connect with existing security tools and workflows
4. **Automation**: Set up scheduled exercises and automated reporting

- --

- *🎉 Congratulations!** Your XORB PTaaS Cyber Range is now deployed and ready for Red vs Blue exercises. Start with staging mode for training, then progress to live exercises with real attack scenarios.