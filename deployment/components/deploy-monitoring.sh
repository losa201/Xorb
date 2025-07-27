#!/bin/bash

echo "📊 Deploying XORB Monitoring Components"

ENVIRONMENT="$1"
REGION="$2"

# Deploy Prometheus
kubectl apply -f - <<YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: xorb-prometheus
  namespace: xorb-$ENVIRONMENT
spec:
  replicas: 1
  selector:
    matchLabels:
      app: xorb-prometheus
  template:
    metadata:
      labels:
        app: xorb-prometheus
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
          name: xorb-prometheus-config
      - name: prometheus-storage
        persistentVolumeClaim:
          claimName: prometheus-storage
---
apiVersion: v1
kind: Service
metadata:
  name: xorb-prometheus
  namespace: xorb-$ENVIRONMENT
spec:
  selector:
    app: xorb-prometheus
  ports:
  - port: 9090
    targetPort: 9090
YAML

# Deploy Grafana
kubectl apply -f - <<YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: xorb-grafana
  namespace: xorb-$ENVIRONMENT
spec:
  replicas: 1
  selector:
    matchLabels:
      app: xorb-grafana
  template:
    metadata:
      labels:
        app: xorb-grafana
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
              name: xorb-monitoring-secrets
              key: grafana-admin-password
        volumeMounts:
        - name: grafana-storage
          mountPath: /var/lib/grafana
      volumes:
      - name: grafana-storage
        persistentVolumeClaim:
          claimName: grafana-storage
---
apiVersion: v1
kind: Service
metadata:
  name: xorb-grafana
  namespace: xorb-$ENVIRONMENT
spec:
  selector:
    app: xorb-grafana
  ports:
  - port: 3000
    targetPort: 3000
  type: LoadBalancer
YAML

echo "✅ Monitoring components deployed successfully"
