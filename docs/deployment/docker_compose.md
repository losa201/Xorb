# XORB Docker Compose Configuration Guide

This document explains the Docker Compose configuration files used in the XORB architecture.

## Overview

XORB provides multiple Docker Compose configurations to support different deployment scenarios:

1. **docker-compose.yml** - Main production configuration
2. **docker-compose.dev.yml** - Development environment with debug tools
3. **docker-compose.fixed.yml** - Fixed configuration for specific deployment requirements

## Main Configuration (docker-compose.yml)

The main configuration defines the core services that make up the XORB architecture:

```yaml
version: '3.8'
services:
  # Core XORB Services
  fusion-orchestrator:
    image: xorb/orchestrator:latest
    container_name: xorb-orchestrator
    environment:
      - CONFIG_PATH=/config/orchestrator.yaml
    volumes:
      - ./config:/config
    depends_on:
      - prometheus

  # Monitoring Stack
  prometheus:
    image: prometheus:latest
    container_name: xorb-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yaml:/etc/prometheus/prometheus.yaml

  grafana:
    image: grafana/grafana:latest
    container_name: xorb-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./grafana:/var/lib/grafana

  # Agent Network
  agent-network:
    image: xorb/agent-network:latest
    container_name: xorb-agent-network
    ports:
      - "8080:8080"
    environment:
      - NETWORK_MODE=swarm

  # Security Components
  security-gateway:
    image: xorb/security-gateway:latest
    container_name: xorb-security-gateway
    ports:
      - "443:443"
    environment:
      - TLS_ENABLED=true

volumes:
  prometheus_data:
  grafana_data:
```

## Development Configuration (docker-compose.dev.yml)

The development configuration extends the main configuration with additional tools and debug capabilities:

```yaml
version: '3.8'

services:
  # Development Tools
  dev-tools:
    image: xorb/dev-tools:latest
    container_name: xorb-dev-tools
    volumes:
      - .:/workspace
    environment:
      - DEBUG_MODE=true
    ports:
      - "5678:5678"  # Debugger port

  # Test Database
  test-db:
    image: postgres:13
    container_name: xorb-test-db
    environment:
      - POSTGRES_USER=xorb
      - POSTGRES_PASSWORD=xorbtest
    ports:
      - "5432:5432"
    volumes:
      - testdb_data:/var/lib/postgresql/data

  # Code Server
  code-server:
    image: codercom/code-server:latest
    container_name: xorb-code-server
    volumes:
      - .:/home/coder/project
    environment:
      - PASSWORD=development
    ports:
      - "8081:8080"

volumes:
  testdb_data:
```

## Fixed Configuration (docker-compose.fixed.yml)

The fixed configuration contains immutable settings for specific deployment requirements:

```yaml
version: '3.8'

services:
  # Fixed Infrastructure
  fixed-infrastructure:
    image: xorb/fixed-infra:latest
    container_name: xorb-fixed-infra
    environment:
      - INFRA_MODE=production
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./certs:/certs

  # Backup Service
  backup-service:
    image: xorb/backup:latest
    container_name: xorb-backup-service
    environment:
      - BACKUP_SCHEDULE=daily
    volumes:
      - ./backups:/backups

  # Logging Aggregator
  logging-aggregator:
    image: xorb/logging:latest
    container_name: xorb-logging-aggregator
    ports:
      - "514:514"
      - "514:514/udp"
    volumes:
      - ./logs:/logs

volumes:
  backup_data:
  log_data:
```

## Usage Instructions

### Starting the System

To start the system in production mode:
```bash
docker-compose up -d
```

For development mode with debug tools:
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

For fixed infrastructure deployment:
```bash
docker-compose -f docker-compose.yml -f docker-compose.fixed.yml up -d
```

### Building Images

To build all images:
```bash
docker-compose build
```

To build specific services:
```bash
docker-compose build fusion-orchestrator security-gateway
```

## Configuration Management

The configuration uses environment variables from the `.env` file for customization. Common configuration options include:

- `CONFIG_PATH`: Path to configuration files
- `NETWORK_MODE`: Network configuration mode
- `TLS_ENABLED`: TLS configuration flag
- `DEBUG_MODE`: Debugging flag for development

## Monitoring and Maintenance

The monitoring stack provides insights into the system through:

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (credentials: admin/admin)

To view logs for a specific service:
```bash
docker logs xorb-orchestrator
```

To execute commands in a running container:
```bash
docker exec -it xorb-orchestrator bash
```

## Best Practices

1. **Environment Separation**: Use different compose files for development, testing, and production
2. **Security**: Regularly update security components and rotate credentials
3. **Monitoring**: Set up alerts in Grafana for critical metrics
4. **Backups**: Regularly test backup and restore procedures
5. **Versioning**: Use image tags to track versions of deployed services

## Troubleshooting

### Common Issues

- **Port Conflicts**: Check for conflicting services on ports 80, 443, 9090, and 3000
- **Configuration Errors**: Verify environment variables in .env file
- **Network Issues**: Ensure Docker network settings allow inter-container communication
- **Permission Problems**: Check file permissions for mounted volumes

### Debugging Tips

- Use `docker-compose config` to validate compose files
- Check container logs with `docker logs <container-name>`
- Use `docker inspect` to examine container configuration
- For network issues, use `docker network inspect` to examine network settings

## Next Steps

1. Review the [Agent Documentation](../agents/) to understand individual components
2. Check the [Execution Guide](../execution/) for running the strategic fusion process
3. Explore the [Monitoring Guide](../monitoring/) for setting up alerts and dashboards
4. Review the [Security Guide](../security/) for securing the deployment

This documentation provides a comprehensive overview of the Docker Compose configuration for XORB. For more detailed information about specific services, refer to their individual documentation.