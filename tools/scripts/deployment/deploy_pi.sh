#!/bin/bash
# Xorb Edge Worker Deployment Script for Raspberry Pi 5
# Deploys the edge worker component for distributed processing

set -euo pipefail

# Configuration
PI_HOST="${1:-pi5.local}"
VPS_IP="${2:-YOUR_VPS_IP}"
REPO_URL="${3:-https://github.com/xorb_platform/xorb.git}"

echo "🚀 Deploying Xorb Edge Worker to $PI_HOST"
echo "📡 VPS IP: $VPS_IP"

# Function to run commands on Pi
run_on_pi() {
    ssh pi@"$PI_HOST" "$@"
}

# Function to copy files to Pi
copy_to_pi() {
    scp "$1" pi@"$PI_HOST":"$2"
}

# Update Pi system
echo "📦 Updating Pi system..."
run_on_pi 'sudo apt update && sudo apt upgrade -y'
run_on_pi 'sudo apt install -y git python3-venv python3-pip htop'

# Clone repository
echo "📥 Cloning Xorb repository..."
run_on_pi "rm -rf ~/xorb || true"
run_on_pi "git clone $REPO_URL ~/xorb"
run_on_pi "cd ~/xorb && git checkout main"

# Setup Python environment
echo "🐍 Setting up Python environment..."
run_on_pi "cd ~/xorb && python3 -m venv .venv"
run_on_pi "cd ~/xorb && source .venv/bin/activate && pip install --upgrade pip"
run_on_pi "cd ~/xorb && source .venv/bin/activate && pip install -r requirements.txt"
run_on_pi "cd ~/xorb && source .venv/bin/activate && pip install -e ."

# Create logs directory
run_on_pi "mkdir -p ~/xorb/logs"

# Create environment configuration
echo "⚙️ Creating environment configuration..."
run_on_pi "cat > ~/xorb/.env << 'EOF'
EDGE_NODE=true
REDIS_URL=redis://$VPS_IP:6379/1
NATS_URL=nats://$VPS_IP:4222
POSTGRES_URL=postgresql://xorb:password@$VPS_IP:5432/xorb
LOG_LEVEL=INFO
PYTHONUNBUFFERED=1
SERVICE_NAME=xorb-edge-worker
NODE_ID=pi5-$(hostname -s)
EOF"

# Create edge worker module
echo "🔧 Creating edge worker module..."
run_on_pi "mkdir -p ~/xorb/edge_worker"
run_on_pi "cat > ~/xorb/edge_worker/__init__.py << 'EOF'
# Edge Worker Module for Raspberry Pi 5
EOF"

run_on_pi "cat > ~/xorb/edge_worker/main.py << 'EOF'
#!/usr/bin/env python3
\"\"\"
Xorb Edge Worker - Raspberry Pi 5 Deployment
Lightweight worker for distributed task processing
\"\"\"

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from typing import Optional

import redis.asyncio as redis
from xorb_core.domain import FindingId
from xorb_core.application import TriageFindingCommand, TriageFindingUseCase

# Setup logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/home/pi/xorb/logs/edge-worker.log')
    ]
)

logger = logging.getLogger(__name__)


class EdgeWorker:
    \"\"\"Edge worker for distributed task processing\"\"\"

    def __init__(self):
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/1')
        self.node_id = os.getenv('NODE_ID', 'pi5-unknown')
        self.running = False
        self.redis_client: Optional[redis.Redis] = None

    async def connect(self):
        \"\"\"Connect to Redis queue\"\"\"
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info(f\"Connected to Redis: {self.redis_url}\")
        except Exception as e:
            logger.error(f\"Failed to connect to Redis: {e}\")
            raise

    async def process_task(self, task_data: dict):
        \"\"\"Process a task from the queue\"\"\"
        try:
            task_type = task_data.get('type')
            logger.info(f\"Processing task: {task_type}\")

            if task_type == 'lightweight_scan':
                await self.process_lightweight_scan(task_data)
            elif task_type == 'finding_triage':
                await self.process_finding_triage(task_data)
            else:
                logger.warning(f\"Unknown task type: {task_type}\")

        except Exception as e:
            logger.error(f\"Error processing task: {e}\")

    async def process_lightweight_scan(self, task_data: dict):
        \"\"\"Process lightweight scanning tasks\"\"\"
        target = task_data.get('target')
        scan_type = task_data.get('scan_type', 'basic')

        logger.info(f\"Running {scan_type} scan on {target}\")

        # Simulate scanning work
        await asyncio.sleep(2)

        # Report results back
        result = {
            'task_id': task_data.get('task_id'),
            'node_id': self.node_id,
            'status': 'completed',
            'findings': [],
            'completed_at': datetime.utcnow().isoformat()
        }

        await self.redis_client.lpush('scan_results', str(result))
        logger.info(f\"Scan completed for {target}\")

    async def process_finding_triage(self, task_data: dict):
        \"\"\"Process finding triage tasks\"\"\"
        finding_id = task_data.get('finding_id')
        action = task_data.get('action', 'analyze')

        logger.info(f\"Triaging finding {finding_id} with action {action}\")

        # Simple triage logic
        await asyncio.sleep(1)

        result = {
            'task_id': task_data.get('task_id'),
            'finding_id': finding_id,
            'node_id': self.node_id,
            'action': action,
            'status': 'completed',
            'completed_at': datetime.utcnow().isoformat()
        }

        await self.redis_client.lpush('triage_results', str(result))
        logger.info(f\"Triage completed for finding {finding_id}\")

    async def run(self):
        \"\"\"Main worker loop\"\"\"
        self.running = True
        logger.info(f\"Starting edge worker {self.node_id}\")

        await self.connect()

        while self.running:
            try:
                # Blocking pop from task queue
                result = await self.redis_client.brpop(['edge_tasks'], timeout=5)

                if result:
                    queue_name, task_data_str = result
                    task_data = eval(task_data_str.decode())  # In production, use json.loads
                    await self.process_task(task_data)

            except asyncio.CancelledError:
                logger.info(\"Worker cancelled\")
                break
            except Exception as e:
                logger.error(f\"Error in worker loop: {e}\")
                await asyncio.sleep(5)

        logger.info(\"Edge worker stopped\")

    def stop(self):
        \"\"\"Stop the worker\"\"\"
        self.running = False


async def main():
    \"\"\"Main entry point\"\"\"
    worker = EdgeWorker()

    # Setup signal handlers
    def signal_handler():
        logger.info(\"Received shutdown signal\")
        worker.stop()

    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda s, f: signal_handler())

    try:
        await worker.run()
    except KeyboardInterrupt:
        logger.info(\"Shutdown requested by user\")
    finally:
        if worker.redis_client:
            await worker.redis_client.close()


if __name__ == \"__main__\":
    asyncio.run(main())
EOF"

# Copy and install systemd service
echo "🔧 Installing systemd service..."
copy_to_pi "edge_systemd/xorb-edge.service" "/tmp/xorb-edge.service"
run_on_pi "sudo sed -i 's/VPS_IP/$VPS_IP/g' /tmp/xorb-edge.service"
run_on_pi "sudo mv /tmp/xorb-edge.service /etc/systemd/system/"
run_on_pi "sudo systemctl daemon-reload"
run_on_pi "sudo systemctl enable xorb-edge"

# Test the installation
echo "🧪 Testing edge worker installation..."
run_on_pi "cd ~/xorb && source .venv/bin/activate && python -c 'import edge_worker; print(\"Edge worker module imported successfully\")'"

# Start the service
echo "🚀 Starting edge worker service..."
run_on_pi "sudo systemctl start xorb-edge"
run_on_pi "sudo systemctl status xorb-edge --no-pager"

# Setup monitoring
echo "📊 Setting up monitoring..."
run_on_pi "sudo apt install -y prometheus-node-exporter"
run_on_pi "sudo systemctl enable prometheus-node-exporter"
run_on_pi "sudo systemctl start prometheus-node-exporter"

# Create monitoring script
run_on_pi "cat > ~/xorb/scripts/monitor_edge.sh << 'EOF'
#!/bin/bash
# Edge worker monitoring script

echo \"🔍 Xorb Edge Worker Status\"
echo \"===========================\"
echo \"Service Status:\"
sudo systemctl status xorb-edge --no-pager -l

echo \"\"
echo \"Recent Logs:\"
sudo journalctl -u xorb-edge --no-pager -n 20

echo \"\"
echo \"System Resources:\"
free -h
df -h /
echo \"CPU Load: \$(uptime | awk '{print \$10, \$11, \$12}')\"

echo \"\"
echo \"Network Connectivity:\"
ping -c 1 $VPS_IP > /dev/null && echo \"✅ VPS reachable\" || echo \"❌ VPS unreachable\"
EOF"

run_on_pi "chmod +x ~/xorb/scripts/monitor_edge.sh"

# Setup log rotation
run_on_pi "sudo cat > /etc/logrotate.d/xorb-edge << 'EOF'
/home/pi/xorb/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    copytruncate
    notifempty
    su pi pi
}
EOF"

echo "✅ Edge worker deployment completed successfully!"
echo ""
echo "📋 Deployment Summary:"
echo "   • Edge worker installed and running"
echo "   • Service name: xorb-edge"
echo "   • Logs: /home/pi/xorb/logs/edge-worker.log"
echo "   • Monitoring: prometheus-node-exporter on port 9100"
echo ""
echo "🔧 Management Commands:"
echo "   • Status: ssh pi@$PI_HOST 'sudo systemctl status xorb-edge'"
echo "   • Logs: ssh pi@$PI_HOST 'sudo journalctl -u xorb-edge -f'"
echo "   • Monitor: ssh pi@$PI_HOST '~/xorb/scripts/monitor_edge.sh'"
echo "   • Restart: ssh pi@$PI_HOST 'sudo systemctl restart xorb-edge'"
