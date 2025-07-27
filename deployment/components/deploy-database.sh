#!/bin/bash

echo "🗄️  Deploying XORB Database Components"

ENVIRONMENT="$1"
REGION="$2"

# Load secrets
source /root/Xorb/secrets/docker-secrets.sh

# Deploy PostgreSQL
kubectl apply -f - <<YAML
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: xorb-postgres
  namespace: xorb-$ENVIRONMENT
spec:
  serviceName: xorb-postgres
  replicas: $([ "$ENVIRONMENT" = "production" ] && echo "3" || echo "1")
  selector:
    matchLabels:
      app: xorb-postgres
  template:
    metadata:
      labels:
        app: xorb-postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: xorb
        - name: POSTGRES_USER
          value: xorb
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: xorb-database-secrets
              key: postgres-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 20Gi
YAML

# Deploy Redis
kubectl apply -f - <<YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: xorb-redis
  namespace: xorb-$ENVIRONMENT
spec:
  replicas: $([ "$ENVIRONMENT" = "production" ] && echo "3" || echo "1")
  selector:
    matchLabels:
      app: xorb-redis
  template:
    metadata:
      labels:
        app: xorb-redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        command: ["redis-server"]
        args: ["--requirepass", "$(kubectl get secret xorb-database-secrets -o jsonpath='{.data.redis-password}' | base64 -d)"]
YAML

echo "✅ Database components deployed successfully"
