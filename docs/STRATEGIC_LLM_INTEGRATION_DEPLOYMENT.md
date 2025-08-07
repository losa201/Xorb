# Strategic LLM Integration Deployment Guide

## Overview

This document provides comprehensive deployment instructions for the XORB Strategic LLM Integration, featuring OpenRouter and NVIDIA API capabilities with supreme cognitive precision.

## Architecture Overview

The strategic LLM integration provides:
- **Multi-Provider Support**: NVIDIA and OpenRouter APIs
- **Task-Specific Optimization**: Security-focused AI capabilities
- **EPYC Architecture Optimization**: AMD EPYC 16-core deployment
- **Cost Management**: Intelligent budget controls
- **Strategic Intelligence**: Advanced reasoning and analysis

## Prerequisites

### Required API Keys

1. **NVIDIA API Key**
   - Sign up at: https://build.nvidia.com/
   - Generate API key for production use
   - Required for advanced reasoning and payload generation

2. **OpenRouter API Key**
   - Sign up at: https://openrouter.ai/
   - Generate API key with sufficient credits
   - Required for diverse model access

### Environment Setup

Create a `.env` file in the project root:

```bash
# Strategic LLM Configuration (Free Tier)
NVIDIA_API_KEY=your_nvidia_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Request Limits (Free Tier - No Cost!)
LLM_FREE_TIER=true
LLM_DAILY_REQUEST_LIMIT=100
LLM_MONTHLY_REQUEST_LIMIT=2000
LLM_HOURLY_REQUEST_LIMIT=20
LLM_EMERGENCY_REQUEST_LIMIT=150

# EPYC Optimization
EPYC_OPTIMIZATION=true
AI_ENHANCED=true
```

## Deployment Steps

### 1. Verify Configuration

```bash
# Check configuration
cat config/llm_strategic_config.json

# Verify environment variables
source .env
echo "NVIDIA API: ${NVIDIA_API_KEY:0:10}..."
echo "OpenRouter API: ${OPENROUTER_API_KEY:0:10}..."
```

### 2. Deploy with Docker Compose

```bash
# Deploy the strategic platform
cd /root/Xorb
docker-compose -f infra/docker-compose.yml up -d

# Verify services
docker-compose ps
```

### 3. Health Check

```bash
# Check intelligence engine health
curl http://localhost:8001/health

# Check AI integration status
curl http://localhost:8001/status
```

## Strategic Capabilities

### 1. AI-Enhanced Vulnerability Analysis

**Endpoint**: `POST /campaigns/{campaign_id}/ai-analysis`

**Example Request**:
```bash
curl -X POST http://localhost:8001/campaigns/test-campaign/ai-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "scan_results": [
      {
        "id": "scan_123",
        "findings": [
          {
            "type": "open_port",
            "port": 22,
            "service": "ssh",
            "severity": "medium"
          }
        ]
      }
    ]
  }'
```

### 2. Strategic Payload Generation

**Endpoint**: `POST /payloads/generate`

**Example Request**:
```bash
curl -X POST http://localhost:8001/payloads/generate \
  -H "Content-Type: application/json" \
  -d '{
    "vulnerabilities": {
      "type": "web_application",
      "details": "SQL injection vulnerability"
    },
    "target_context": {
      "technology": "PHP/MySQL",
      "environment": "production"
    },
    "complexity": "enhanced"
  }'
```

### 3. Intelligence Fusion

**Endpoint**: `POST /intelligence/fusion`

**Example Request**:
```bash
curl -X POST http://localhost:8001/intelligence/fusion \
  -H "Content-Type: application/json" \
  -d '{
    "sources": {
      "network_scans": [...],
      "web_analysis": [...],
      "threat_intel": [...]
    },
    "objectives": ["correlation", "threat_assessment", "risk_scoring"]
  }'
```

## Model Selection Strategy

### NVIDIA Models (Preferred for Advanced Tasks)

1. **nvidia/llama-3.1-nemotron-70b-instruct**
   - Best for: Payload generation, vulnerability analysis
   - Cost: $0.004 per 1K tokens
   - Optimization: EPYC-optimized

2. **nvidia/llama-3.1-405b-instruct**
   - Best for: Threat assessment, intelligence fusion
   - Cost: $0.008 per 1K tokens
   - Features: Supreme reasoning capabilities

### OpenRouter Models (Cost-Effective Options)

1. **anthropic/claude-3.5-sonnet**
   - Best for: Report generation, risk scoring
   - Cost: $0.003 per 1K tokens
   - Features: Long context, structured output

2. **openai/gpt-4o**
   - Best for: Anomaly detection, attack simulation
   - Cost: $0.005 per 1K tokens
   - Features: Function calling, creativity

3. **google/gemini-pro-1.5**
   - Best for: Large-scale analysis
   - Cost: $0.002 per 1K tokens
   - Features: 1M token context window

## Free Tier Management

### Request Controls

- **Daily Requests**: 100 (configurable)
- **Monthly Requests**: 2000 (configurable) 
- **Hourly Rate Limit**: 20 requests/hour (configurable)
- **Emergency Allowance**: 150 requests (break-glass scenarios)
- **Cost**: $0.00 (100% Free!)

### Monitoring

```bash
# Check current usage
curl http://localhost:8001/intelligence/statistics

# Response includes:
{
  "ai_integration": {
    "free_tier": true,
    "daily_requests": 45,
    "monthly_requests": 567,
    "hourly_requests": 8,
    "request_limits": {
      "daily": 100,
      "monthly": 2000,
      "per_hour": 20
    },
    "cache_hit_rate": 0.35,
    "provider_distribution": {
      "nvidia": 15,
      "openrouter": 8
    },
    "cost_savings": "100% (Free Tier)"
  }
}
```

## Security Considerations

### API Key Security

1. **Environment Variables**: Never commit API keys to version control
2. **Secure Storage**: Use container secrets or vault systems
3. **Rotation**: Regularly rotate API keys
4. **Monitoring**: Monitor for unusual API usage

### Authorization Controls

```python
# All AI requests include authorization context
{
  "authorized_testing_only": True,
  "compliance_requirements": "SOC2_TYPE2",
  "user_context": {
    "role": "penetration_tester",
    "clearance": "authorized"
  }
}
```

## Performance Optimization

### EPYC Architecture Optimization

- **CPU Affinity**: Optimized for 16-core EPYC processors
- **Memory Management**: 32GB RAM optimization
- **Concurrent Processing**: Parallel AI request handling
- **Request Batching**: Efficient token usage

### Caching Strategy

- **Request Caching**: 1-hour TTL for analysis results
- **Model Response Caching**: Intelligent deduplication
- **Context Caching**: Preserve conversation context

## Troubleshooting

### Common Issues

1. **API Key Invalid**
   ```bash
   # Verify API key
   curl -H "Authorization: Bearer $NVIDIA_API_KEY" \
        https://integrate.api.nvidia.com/v1/models
   ```

2. **Budget Exceeded**
   ```bash
   # Check budget status
   curl http://localhost:8001/health
   # Look for "ai_budget_status": "critical"
   ```

3. **Model Unavailable**
   ```bash
   # Check model availability
   curl http://localhost:8001/status
   # Verify "models_available": true
   ```

### Logs and Monitoring

```bash
# View intelligence engine logs
docker-compose logs xorb-intelligence-engine

# Monitor real-time activity
docker-compose logs -f xorb-intelligence-engine | grep "Strategic"
```

## Advanced Configuration

### Custom Prompt Templates

Edit `config/llm_strategic_config.json`:

```json
{
  "prompt_templates": {
    "custom_analysis": "Your custom security analysis template",
    "payload_generation": "Enhanced payload generation template"
  }
}
```

### Model Selection Preferences

```json
{
  "security_tasks": {
    "vulnerability_analysis": {
      "preferred_providers": ["nvidia"],
      "fallback_providers": ["openrouter"],
      "complexity_threshold": "advanced"
    }
  }
}
```

## Integration Examples

### Python Integration

```python
import aiohttp
import asyncio

async def analyze_vulnerabilities(scan_results):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'http://localhost:8001/campaigns/test/ai-analysis',
            json={'scan_results': scan_results}
        ) as response:
            return await response.json()

# Usage
results = asyncio.run(analyze_vulnerabilities(scan_data))
```

### REST API Integration

```bash
#!/bin/bash
# AI-enhanced security workflow

# 1. Generate payloads
PAYLOADS=$(curl -s -X POST http://localhost:8001/payloads/generate \
  -H "Content-Type: application/json" \
  -d '{"vulnerabilities": {...}}')

# 2. Analyze results
ANALYSIS=$(curl -s -X POST http://localhost:8001/campaigns/test/ai-analysis \
  -H "Content-Type: application/json" \
  -d '{"scan_results": [...]}')

# 3. Generate report
echo "Strategic analysis complete with AI enhancement"
```

## Monitoring and Alerting

### Key Metrics

- **Request Volume**: AI requests per hour
- **Cost Tracking**: Real-time budget consumption
- **Model Performance**: Response times and accuracy
- **Cache Efficiency**: Hit rates and optimization

### Grafana Dashboard

Import the strategic AI dashboard:
```bash
# Located at: grafana/xorb-strategic-ai-dashboard.json
# Metrics: Cost, performance, model usage, cache hits
```

## Support and Maintenance

### Regular Maintenance Tasks

1. **API Key Rotation**: Monthly rotation recommended
2. **Budget Review**: Weekly budget utilization analysis
3. **Model Performance**: Monitor and optimize model selection
4. **Cache Cleanup**: Regular cache maintenance

### Performance Tuning

1. **Request Batching**: Optimize for cost efficiency
2. **Model Selection**: Adjust based on performance metrics
3. **Cache Tuning**: Optimize TTL and hit rates
4. **EPYC Optimization**: Monitor CPU utilization

## Conclusion

The Strategic LLM Integration provides supreme cognitive capabilities for cybersecurity operations. With proper deployment and configuration, it enables:

- **Advanced Threat Analysis**: AI-powered vulnerability assessment
- **Strategic Intelligence**: Multi-source intelligence fusion
- **Cost-Effective Operations**: Intelligent budget management
- **EPYC Optimization**: Hardware-specific performance tuning

For additional support, refer to the comprehensive logs and monitoring capabilities built into the platform.

---

**Deployment Status**: ✅ Complete
**AI Enhancement**: ✅ Active
**EPYC Optimization**: ✅ Enabled
**Strategic Capabilities**: ✅ Fully Operational