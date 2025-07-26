FROM python:3.11-slim

# Build argument for agent name
ARG AGENT_NAME
ENV AGENT_NAME=${AGENT_NAME}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for agents
RUN pip install --no-cache-dir \
    prometheus-client \
    structlog \
    asyncpg \
    aioredis \
    aiohttp \
    networkx \
    numpy \
    uvicorn \
    fastapi

# Copy source code
COPY ecosystem/ /app/ecosystem/
COPY xorb_core/ /app/xorb_core/
COPY xorb_common/ /app/xorb_common/

# Create logs directory
RUN mkdir -p /app/logs

# Set Python path
ENV PYTHONPATH=/app:/app/ecosystem:/app/xorb_core:/app/xorb_common

# Health check script
COPY <<EOF /app/healthcheck.py
#!/usr/bin/env python3
import asyncio
import aiohttp
import sys
import os

async def check_health():
    port = os.environ.get('PROMETHEUS_PORT', '8000')
    url = f"http://localhost:{port}/metrics"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    print("Health check passed")
                    return True
    except Exception as e:
        print(f"Health check failed: {e}")
    return False

if __name__ == "__main__":
    result = asyncio.run(check_health())
    sys.exit(0 if result else 1)
EOF

RUN chmod +x /app/healthcheck.py

# Startup script
COPY <<EOF /app/start_agent.py
#!/usr/bin/env python3
import os
import sys
import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    agent_name = os.environ.get('AGENT_NAME')
    if not agent_name:
        print("ERROR: AGENT_NAME environment variable not set")
        sys.exit(1)
    
    print(f"Starting XORB agent: {agent_name}")
    
    # Import and run the specific agent
    try:
        if agent_name == 'orchestrator_agent':
            from ecosystem.agents.orchestrator_agent import OrchestratorAgent
            config = {
                'redis_url': os.environ.get('REDIS_URL', 'redis://localhost:6379'),
                'postgres_url': os.environ.get('POSTGRES_URL', 'postgresql://localhost:5432/xorb'),
                'prometheus_port': int(os.environ.get('PROMETHEUS_PORT', '8000')),
                'agent_id': os.environ.get('AGENT_ID', 'orchestrator-001'),
                'orchestration_cycle_ms': int(os.environ.get('ORCHESTRATION_CYCLE_MS', '500')),
                'max_concurrent_agents': int(os.environ.get('MAX_CONCURRENT_AGENTS', '32'))
            }
            agent = OrchestratorAgent(config)
            await agent.initialize()
            await agent.start_orchestration()
            
        elif agent_name == 'evolutionary_defense_agent':
            from ecosystem.agents.evolutionary_defense_agent import EvolutionaryDefenseAgent
            config = {
                'redis_url': os.environ.get('REDIS_URL', 'redis://localhost:6379'),
                'postgres_url': os.environ.get('POSTGRES_URL', 'postgresql://localhost:5432/xorb'),
                'prometheus_port': int(os.environ.get('PROMETHEUS_PORT', '8001')),
                'agent_id': os.environ.get('AGENT_ID', 'evolutionary-defense-001'),
                'population_size': int(os.environ.get('POPULATION_SIZE', '50')),
                'mutation_rate': float(os.environ.get('MUTATION_RATE', '0.1')),
                'evolution_interval': int(os.environ.get('EVOLUTION_INTERVAL', '3600'))
            }
            agent = EvolutionaryDefenseAgent(config)
            await agent.initialize()
            await agent.start_evolution()
            
        elif agent_name == 'threat_propagation_modeling_agent':
            from ecosystem.agents.threat_propagation_modeling_agent import ThreatPropagationModelingAgent
            config = {
                'redis_url': os.environ.get('REDIS_URL', 'redis://localhost:6379'),
                'postgres_url': os.environ.get('POSTGRES_URL', 'postgresql://localhost:5432/xorb'),
                'prometheus_port': int(os.environ.get('PROMETHEUS_PORT', '8002')),
                'agent_id': os.environ.get('AGENT_ID', 'threat-propagation-001'),
                'simulation_steps': int(os.environ.get('SIMULATION_STEPS', '1000')),
                'monte_carlo_runs': int(os.environ.get('MONTE_CARLO_RUNS', '100')),
                'max_network_size': int(os.environ.get('MAX_NETWORK_SIZE', '10000'))
            }
            agent = ThreatPropagationModelingAgent(config)
            await agent.initialize()
            await agent.start_modeling()
            
        elif agent_name == 'autonomous_response_agent':
            from ecosystem.agents.autonomous_response_agent import AutonomousResponseAgent
            config = {
                'redis_url': os.environ.get('REDIS_URL', 'redis://localhost:6379'),
                'postgres_url': os.environ.get('POSTGRES_URL', 'postgresql://localhost:5432/xorb'),
                'prometheus_port': int(os.environ.get('PROMETHEUS_PORT', '8003')),
                'agent_id': os.environ.get('AGENT_ID', 'autonomous-response-001'),
                'min_confidence_threshold': float(os.environ.get('MIN_CONFIDENCE_THRESHOLD', '0.72')),
                'max_concurrent_responses': int(os.environ.get('MAX_CONCURRENT_RESPONSES', '10')),
                'default_response_timeout': int(os.environ.get('DEFAULT_RESPONSE_TIMEOUT', '300'))
            }
            agent = AutonomousResponseAgent(config)
            await agent.initialize()
            await agent.start_response_processing()
            
        elif agent_name == 'ecosystem_integration_agent':
            from ecosystem.agents.ecosystem_integration_agent import EcosystemIntegrationAgent
            config = {
                'redis_url': os.environ.get('REDIS_URL', 'redis://localhost:6379'),
                'postgres_url': os.environ.get('POSTGRES_URL', 'postgresql://localhost:5432/xorb'),
                'prometheus_port': int(os.environ.get('PROMETHEUS_PORT', '8004')),
                'agent_id': os.environ.get('AGENT_ID', 'ecosystem-integration-001'),
                'sync_interval': int(os.environ.get('SYNC_INTERVAL', '3600')),
                'max_concurrent_operations': int(os.environ.get('MAX_CONCURRENT_OPERATIONS', '20'))
            }
            agent = EcosystemIntegrationAgent(config)
            await agent.initialize()
            await agent.start_integration()
            
        else:
            print(f"ERROR: Unknown agent name: {agent_name}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"Shutting down {agent_name}")
        if 'agent' in locals():
            await agent.shutdown()
    except Exception as e:
        print(f"ERROR: Agent {agent_name} failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
EOF

RUN chmod +x /app/start_agent.py

# Expose metrics port (will be overridden by env var)
EXPOSE 8000

# Start the agent
CMD ["python", "/app/start_agent.py"]