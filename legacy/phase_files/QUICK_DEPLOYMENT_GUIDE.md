# 🚀 XORB Quick Deployment Guide

## ⚡ **Instant Production Deployment**

### **Prerequisites (5 minutes)**
```bash
# Required tools
- Docker & Docker Compose
- Git
- 8GB+ RAM (16GB+ recommended)
- NVIDIA API key
```

### **1. Clone & Configure (2 minutes)**
```bash
# Clone repository
git clone https://github.com/losa201/Xorb.git
cd Xorb

# Setup environment
cp .env.example .env

# Edit .env with your credentials:
# NVIDIA_API_KEY=your-actual-nvidia-api-key
# POSTGRES_PASSWORD=your-secure-password
# REDIS_PASSWORD=your-secure-password
```

### **2. Deploy Core Services (3 minutes)**
```bash
# Start essential services
docker-compose up -d postgres redis temporal

# Verify health
docker-compose ps
```

### **3. Deploy XORB Platform (5 minutes)**
```bash
# Full platform deployment
docker-compose up -d

# Verify all services
docker-compose ps
curl http://localhost:8000/health  # API health check
```

### **4. Validation & Testing (2 minutes)**
```bash
# Run deployment readiness check
python3 scripts/deployment_readiness_check.py

# Expected result: 100% success rate, READY status
```

---

## 🎯 **Service Endpoints**

| Service | URL | Purpose |
|---------|-----|---------|
| **XORB API** | http://localhost:8000 | Main REST API |
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |
| **Temporal UI** | http://localhost:8233 | Workflow monitoring |
| **PostgreSQL** | localhost:5432 | Primary database |
| **Redis** | localhost:6380 | Cache & sessions |

---

## 🔧 **Quick Commands**

```bash
# View logs
docker-compose logs -f api

# Restart service
docker-compose restart api

# Scale workers
docker-compose up -d --scale worker=3

# Stop all services
docker-compose down

# Clean restart
docker-compose down && docker-compose up -d
```

---

## 🛡️ **Security Quick Check**

```bash
# Verify no hardcoded secrets
grep -r "nvapi-" . --exclude-dir=.git || echo "✅ Secure"

# Check environment variables
docker-compose config | grep NVIDIA_API_KEY
```

---

## 📊 **Monitoring Dashboard**

Access real-time metrics at:
- **Health**: http://localhost:8000/health
- **Metrics**: http://localhost:9001/metrics
- **Status**: `docker-compose ps`

---

## 🚨 **Troubleshooting**

### **Common Issues & Solutions**

**Service won't start:**
```bash
docker-compose logs [service-name]
docker-compose restart [service-name]
```

**Port conflicts:**
```bash
# Check running processes
sudo lsof -i :8000
# Kill conflicting process or change port in docker-compose.yml
```

**Permission issues:**
```bash
sudo chown -R $USER:$USER .
chmod +x scripts/*.py
```

**Environment variables not loading:**
```bash
# Verify .env file exists and has correct format
cat .env
# Restart with explicit env file
docker-compose --env-file .env up -d
```

---

## 🏆 **Deployment Verification**

✅ **Success Indicators:**
- All containers show "Up" status
- API responds with 200 at /health
- No error logs in container output
- Readiness check shows 100% success

🚫 **Failure Indicators:**
- Containers in "Restarting" state
- API returns 500 errors
- Database connection failures
- Missing environment variables

---

## 🌟 **Next Steps**

After successful deployment:

1. **Configure Authentication** - Set up user accounts and API keys
2. **Load Test** - Verify performance under expected load
3. **Backup Setup** - Configure automated database backups
4. **Monitoring** - Deploy Prometheus/Grafana for production monitoring
5. **SSL/TLS** - Add reverse proxy with SSL certificates

---

## 📞 **Support**

- **Documentation**: [Full Deployment Report](./FINAL_DEPLOYMENT_REPORT.md)
- **Issues**: [GitHub Issues](https://github.com/losa201/Xorb/issues)
- **Security**: [Security Guide](./SECURITY_DEPLOYMENT_COMPLETE.md)

**Total Deployment Time: ~15 minutes**  
**Expertise Level: Beginner to Advanced**  
**Production Ready: ✅ YES**