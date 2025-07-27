#!/bin/bash

echo "📡 Deploying XORB API Components"

ENVIRONMENT="$1"
REGION="$2"

kubectl apply -f - <<YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: xorb-api
  namespace: xorb-$ENVIRONMENT
spec:
  replicas: $([ "$ENVIRONMENT" = "production" ] && echo "3" || echo "2")
  selector:
    matchLabels:
      app: xorb-api
  template:
    metadata:
      labels:
        app: xorb-api
    spec:
      containers:
      - name: api
        image: xorb/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "$ENVIRONMENT"
        - name: REGION
          value: "$REGION"
        - name: DATABASE_URL
          value: "postgresql://xorb:$(kubectl get secret xorb-database-secrets -o jsonpath='{.data.postgres-password}' | base64 -d)@xorb-postgres:5432/xorb"
        - name: REDIS_URL
          value: "redis://:$(kubectl get secret xorb-database-secrets -o jsonpath='{.data.redis-password}' | base64 -d)@xorb-redis:6379"
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: xorb-auth-secrets
              key: jwt-secret
        resources:
          requests:
            cpu: $([ "$ENVIRONMENT" = "production" ] && echo "500m" || echo "200m")
            memory: $([ "$ENVIRONMENT" = "production" ] && echo "512Mi" || echo "256Mi")
          limits:
            cpu: $([ "$ENVIRONMENT" = "production" ] && echo "2000m" || echo "1000m")
            memory: $([ "$ENVIRONMENT" = "production" ] && echo "2Gi" || echo "1Gi")
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: xorb-api
  namespace: xorb-$ENVIRONMENT
spec:
  selector:
    app: xorb-api
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
YAML

echo "✅ API components deployed successfully"
