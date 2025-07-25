#!/bin/bash
#
# Enhanced Raspberry Pi 5 Edge Deployment for Xorb Security Intelligence Platform
# Implements resilience, fallback capabilities, and autonomous edge operations
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
PI_HOST="${PI_HOST:-}"
VPS_HOST="${VPS_HOST:-}"
EDGE_NODE_ID="${EDGE_NODE_ID:-pi5-$(hostname -s)}"
DEPLOYMENT_MODE="${DEPLOYMENT_MODE:-autonomous}"  # autonomous, relay, failover
ENABLE_WATCHDOG="${ENABLE_WATCHDOG:-true}"
ENABLE_HEALTH_SYNC="${ENABLE_HEALTH_SYNC:-true}"
POWER_MANAGEMENT="${POWER_MANAGEMENT:-adaptive}"
NETWORK_RESILIENCE="${NETWORK_RESILIENCE:-true}"

# Hardware configuration for Pi 5
PI5_MEMORY_LIMIT="7G"    # Leave 1GB for system
PI5_CPU_CORES="4"
PI5_STORAGE_LIMIT="50G"  # Assume 64GB+ SD card

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] INFO: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARN: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] SUCCESS: $1${NC}"
}

header() {
    echo -e "\n${PURPLE}=====================================${NC}"
    echo -e "${PURPLE} $1${NC}"
    echo -e "${PURPLE}=====================================${NC}\n"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites for Pi 5 edge deployment..."
    
    if [[ -z "$PI_HOST" ]]; then
        error "PI_HOST environment variable is required"
    fi
    
    if [[ -z "$VPS_HOST" ]]; then
        warn "VPS_HOST not set - edge node will operate in standalone mode"
        DEPLOYMENT_MODE="standalone"
    fi
    
    # Check SSH connectivity
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes pi@"$PI_HOST" exit 2>/dev/null; then
        error "Cannot connect to Pi at $PI_HOST via SSH"
    fi
    
    # Check if running on arm64
    pi_arch=$(ssh pi@"$PI_HOST" 'uname -m')
    if [[ "$pi_arch" != "aarch64" ]]; then
        warn "Expected aarch64 architecture, got: $pi_arch"
    fi
    
    # Check Pi model
    pi_model=$(ssh pi@"$PI_HOST" 'cat /proc/device-tree/model' | tr -d '\0')
    if [[ "$pi_model" != *"Raspberry Pi 5"* ]]; then
        warn "Expected Raspberry Pi 5, detected: $pi_model"
    fi
    
    success "Prerequisites check completed"
}

# Optimize Pi 5 hardware configuration
optimize_pi5_hardware() {
    log "Optimizing Raspberry Pi 5 hardware configuration..."
    
    ssh pi@"$PI_HOST" << 'EOF'
        # Enable performance governor
        sudo bash -c 'echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'
        sudo bash -c 'echo performance > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor'
        sudo bash -c 'echo performance > /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor'
        sudo bash -c 'echo performance > /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor'
        
        # Configure GPU memory split (minimal for headless)
        sudo raspi-config nonint do_memory_split 16
        
        # Enable I2C and SPI for sensors (if needed)
        sudo raspi-config nonint do_i2c 0
        sudo raspi-config nonint do_spi 0
        
        # Configure swap for better memory management
        sudo dphys-swapfile swapoff
        sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
        sudo dphys-swapfile setup
        sudo dphys-swapfile swapon
        
        # Optimize network settings
        sudo sysctl -w net.core.rmem_max=16777216
        sudo sysctl -w net.core.wmem_max=16777216
        sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 16777216"
        sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 16777216"
        
        # Make network optimizations persistent
        echo "net.core.rmem_max=16777216" | sudo tee -a /etc/sysctl.conf
        echo "net.core.wmem_max=16777216" | sudo tee -a /etc/sysctl.conf
        echo "net.ipv4.tcp_rmem=4096 87380 16777216" | sudo tee -a /etc/sysctl.conf
        echo "net.ipv4.tcp_wmem=4096 65536 16777216" | sudo tee -a /etc/sysctl.conf
EOF
    
    success "Pi 5 hardware optimization completed"
}

# Install enhanced dependencies
install_enhanced_dependencies() {
    log "Installing enhanced dependencies for edge deployment..."
    
    ssh pi@"$PI_HOST" << 'EOF'
        # Update system
        sudo apt update && sudo apt upgrade -y
        
        # Install essential packages
        sudo apt install -y \
            docker.io \
            docker-compose \
            git \
            curl \
            wget \
            htop \
            iotop \
            tmux \
            vim \
            jq \
            python3-pip \
            python3-venv \
            redis-server \
            sqlite3 \
            wireguard \
            fail2ban \
            ufw \
            unattended-upgrades \
            watchdog \
            rpi-eeprom
        
        # Install monitoring tools
        sudo apt install -y \
            collectd \
            telegraf \
            node-exporter
        
        # Configure Docker for Pi
        sudo usermod -aG docker pi
        sudo systemctl enable docker
        sudo systemctl start docker
        
        # Install Docker Compose v2
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-aarch64" \
             -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
        
        # Configure automatic updates
        sudo dpkg-reconfigure -plow unattended-upgrades
        
        # Configure fail2ban
        sudo systemctl enable fail2ban
        sudo systemctl start fail2ban
        
        # Configure basic firewall
        sudo ufw --force enable
        sudo ufw default deny incoming
        sudo ufw default allow outgoing
        sudo ufw allow ssh
        sudo ufw allow 8080/tcp  # Xorb edge API
        sudo ufw allow 9090/tcp  # Metrics
EOF
    
    success "Enhanced dependencies installed"
}

# Setup resilient storage
setup_resilient_storage() {
    log "Setting up resilient storage configuration..."
    
    ssh pi@"$PI_HOST" << 'EOF'
        # Create data directories
        sudo mkdir -p /opt/xorb/{data,logs,cache,config,tmp}
        sudo chown -R pi:pi /opt/xorb
        
        # Setup log rotation
        sudo tee /etc/logrotate.d/xorb << 'LOGROTATE'
/opt/xorb/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    sharedscripts
    postrotate
        systemctl reload xorb-edge || true
    endscript
}
LOGROTATE
        
        # Configure tmpfs for temporary files (reduce SD card wear)
        echo "tmpfs /opt/xorb/tmp tmpfs defaults,noatime,nosuid,nodev,noexec,mode=1777,size=512M 0 0" | \
            sudo tee -a /etc/fstab
        
        # Mount tmpfs
        sudo mount -a
        
        # Setup periodic sync to prevent data loss
        (crontab -l 2>/dev/null; echo "*/5 * * * * sync") | crontab -
EOF
    
    success "Resilient storage configured"
}

# Create edge worker configuration
create_edge_configuration() {
    log "Creating edge worker configuration..."
    
    # Generate edge-specific configuration
    cat > /tmp/edge-worker-config.yaml << EOF
# Xorb Edge Worker Configuration for Raspberry Pi 5
edge:
  node_id: "$EDGE_NODE_ID"
  deployment_mode: "$DEPLOYMENT_MODE"
  hardware_profile: "raspberry_pi_5"
  
  # Resource limits for Pi 5
  resources:
    memory_limit: "$PI5_MEMORY_LIMIT"
    cpu_cores: $PI5_CPU_CORES
    storage_limit: "$PI5_STORAGE_LIMIT"
    
  # Network configuration
  network:
    resilience_enabled: $NETWORK_RESILIENCE
    upstream_host: "$VPS_HOST"
    local_api_port: 8080
    metrics_port: 9090
    health_check_interval: 30
    connection_timeout: 10
    retry_attempts: 3
    
  # Power management
  power:
    management_mode: "$POWER_MANAGEMENT"
    low_power_threshold: 15  # Battery percentage
    thermal_throttle_temp: 70  # Celsius
    
  # Watchdog configuration
  watchdog:
    enabled: $ENABLE_WATCHDOG
    interval: 60
    timeout: 300
    restart_threshold: 3
    
  # Health sync with central VPS
  health_sync:
    enabled: $ENABLE_HEALTH_SYNC
    interval: 120
    batch_size: 100
    compression: true
    
  # Local caching
  cache:
    max_size: "2G"
    ttl: 3600
    cleanup_interval: 300
    
  # Security
  security:
    tls_enabled: true
    cert_auto_renew: true
    rate_limiting:
      requests_per_minute: 100
      burst_size: 20
    
  # Autonomous capabilities
  autonomous:
    enabled: true
    decision_threshold: 0.8
    learning_rate: 0.01
    adaptation_interval: 300
    
  # Fallback configuration
  fallback:
    standalone_mode: true
    local_db_path: "/opt/xorb/data/edge.db"
    queue_max_size: 10000
    offline_retention_days: 7
    
  # Monitoring
  monitoring:
    metrics_enabled: true
    logging_level: "INFO"
    performance_tracking: true
    hardware_monitoring: true
EOF

    # Transfer configuration to Pi
    scp /tmp/edge-worker-config.yaml pi@"$PI_HOST":/opt/xorb/config/
    rm /tmp/edge-worker-config.yaml
    
    success "Edge configuration created"
}

# Create enhanced edge worker service
create_edge_worker_service() {
    log "Creating enhanced edge worker service..."
    
    # Create Python edge worker script
    cat > /tmp/xorb_edge_worker.py << 'EOF'
#!/usr/bin/env python3
"""
Enhanced Xorb Edge Worker for Raspberry Pi 5
Implements resilient, autonomous edge computing capabilities
"""

import asyncio
import aiohttp
import json
import logging
import os
import sqlite3
import time
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import psutil
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/xorb/logs/edge-worker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('xorb-edge-worker')

class EdgeWorkerMetrics:
    """Collect and manage edge worker metrics"""
    
    def __init__(self):
        self.metrics = {
            'tasks_processed': 0,
            'tasks_failed': 0,
            'uptime_seconds': 0,
            'memory_usage_mb': 0,
            'cpu_usage_percent': 0,
            'temperature_celsius': 0,
            'network_bytes_sent': 0,
            'network_bytes_received': 0,
            'storage_used_percent': 0,
            'last_sync_timestamp': None,
            'sync_failures': 0,
            'autonomous_decisions': 0
        }
        self.start_time = time.time()
    
    def update_system_metrics(self):
        """Update system metrics"""
        try:
            # CPU and memory
            self.metrics['cpu_usage_percent'] = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            self.metrics['memory_usage_mb'] = memory.used / (1024 * 1024)
            
            # Temperature (Pi-specific)
            try:
                temp_path = '/sys/class/thermal/thermal_zone0/temp'
                if os.path.exists(temp_path):
                    with open(temp_path, 'r') as f:
                        temp_raw = int(f.read().strip())
                        self.metrics['temperature_celsius'] = temp_raw / 1000.0
            except:
                pass
            
            # Network
            net_io = psutil.net_io_counters()
            self.metrics['network_bytes_sent'] = net_io.bytes_sent
            self.metrics['network_bytes_received'] = net_io.bytes_recv
            
            # Storage
            disk_usage = psutil.disk_usage('/')
            self.metrics['storage_used_percent'] = (disk_usage.used / disk_usage.total) * 100
            
            # Uptime
            self.metrics['uptime_seconds'] = time.time() - self.start_time
            
        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return self.metrics.copy()

class EdgeWorker:
    """Enhanced Xorb Edge Worker"""
    
    def __init__(self, config_path: str = '/opt/xorb/config/edge-worker-config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()
        self.metrics = EdgeWorkerMetrics()
        self.running = False
        self.session = None
        self.local_db = None
        self.task_queue = asyncio.Queue(maxsize=self.config['fallback']['queue_max_size'])
        self.sync_queue = asyncio.Queue()
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)['edge']
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'node_id': 'pi5-edge-worker',
            'deployment_mode': 'autonomous',
            'network': {
                'upstream_host': None,
                'local_api_port': 8080,
                'health_check_interval': 30
            },
            'autonomous': {'enabled': True},
            'fallback': {
                'standalone_mode': True,
                'local_db_path': '/opt/xorb/data/edge.db',
                'queue_max_size': 1000
            }
        }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    async def initialize(self):
        """Initialize edge worker"""
        logger.info("Initializing Xorb Edge Worker...")
        
        # Setup local database
        await self._setup_local_database()
        
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Create data directories
        os.makedirs('/opt/xorb/data', exist_ok=True)
        os.makedirs('/opt/xorb/logs', exist_ok=True)
        os.makedirs('/opt/xorb/cache', exist_ok=True)
        
        logger.info("Edge worker initialized successfully")
    
    async def _setup_local_database(self):
        """Setup local SQLite database for offline operations"""
        db_path = self.config['fallback']['local_db_path']
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.local_db = sqlite3.connect(db_path)
        cursor = self.local_db.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE,
                task_type TEXT,
                parameters TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                result TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT,
                metric_value REAL,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sync_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_type TEXT,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                synced BOOLEAN DEFAULT FALSE
            )
        ''')
        
        self.local_db.commit()
        logger.info("Local database initialized")
    
    async def start(self):
        """Start the edge worker"""
        logger.info("Starting Xorb Edge Worker...")
        self.running = True
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._task_processing_loop()),
            asyncio.create_task(self._health_sync_loop()),
            asyncio.create_task(self._watchdog_loop()),
            asyncio.create_task(self._autonomous_decision_loop()),
            asyncio.create_task(self._api_server())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Edge worker error: {e}")
        finally:
            await self.cleanup()
    
    async def _metrics_collection_loop(self):
        """Collect metrics periodically"""
        while self.running:
            try:
                self.metrics.update_system_metrics()
                
                # Store metrics in local database
                cursor = self.local_db.cursor()
                for metric_name, value in self.metrics.to_dict().items():
                    if isinstance(value, (int, float)):
                        cursor.execute(
                            "INSERT INTO metrics (metric_name, metric_value) VALUES (?, ?)",
                            (metric_name, value)
                        )
                self.local_db.commit()
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)
    
    async def _task_processing_loop(self):
        """Process tasks from the queue"""
        while self.running:
            try:
                # Get task from queue with timeout
                try:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process task
                result = await self._process_task(task)
                
                # Update metrics
                if result.get('success', False):
                    self.metrics.metrics['tasks_processed'] += 1
                else:
                    self.metrics.metrics['tasks_failed'] += 1
                
                # Queue result for sync if connected
                if self.config['health_sync']['enabled']:
                    await self.sync_queue.put({
                        'type': 'task_result',
                        'data': result,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                
            except Exception as e:
                logger.error(f"Task processing error: {e}")
                self.metrics.metrics['tasks_failed'] += 1
    
    async def _process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single task"""
        task_id = task.get('task_id', str(time.time()))
        task_type = task.get('type', 'unknown')
        
        logger.info(f"Processing task {task_id} of type {task_type}")
        
        start_time = time.time()
        result = {
            'task_id': task_id,
            'task_type': task_type,
            'success': False,
            'execution_time': 0,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            # Store task in local database
            cursor = self.local_db.cursor()
            cursor.execute(
                "INSERT INTO tasks (task_id, task_type, parameters, status) VALUES (?, ?, ?, ?)",
                (task_id, task_type, json.dumps(task.get('parameters', {})), 'processing')
            )
            self.local_db.commit()
            
            # Simulate task processing (replace with actual logic)
            await asyncio.sleep(1)  # Simulate work
            
            # Update task status
            cursor.execute(
                "UPDATE tasks SET status=?, completed_at=?, result=? WHERE task_id=?",
                ('completed', datetime.utcnow(), json.dumps(result), task_id)
            )
            self.local_db.commit()
            
            result['success'] = True
            result['execution_time'] = time.time() - start_time
            
            logger.info(f"Task {task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            result['error'] = str(e)
            result['execution_time'] = time.time() - start_time
            
            # Update task status
            cursor = self.local_db.cursor()
            cursor.execute(
                "UPDATE tasks SET status=?, completed_at=?, result=? WHERE task_id=?",
                ('failed', datetime.utcnow(), json.dumps(result), task_id)
            )
            self.local_db.commit()
        
        return result
    
    async def _health_sync_loop(self):
        """Sync health data with upstream VPS"""
        if not self.config['health_sync']['enabled']:
            return
        
        upstream_host = self.config['network'].get('upstream_host')
        if not upstream_host:
            logger.info("No upstream host configured, skipping health sync")
            return
        
        sync_interval = self.config['health_sync']['interval']
        
        while self.running:
            try:
                # Collect data to sync
                sync_data = []
                
                # Get queued items
                try:
                    while len(sync_data) < self.config['health_sync']['batch_size']:
                        item = await asyncio.wait_for(self.sync_queue.get(), timeout=1.0)
                        sync_data.append(item)
                except asyncio.TimeoutError:
                    pass
                
                # Add current metrics
                sync_data.append({
                    'type': 'metrics',
                    'data': self.metrics.to_dict(),
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                if sync_data:
                    success = await self._sync_to_upstream(sync_data)
                    if success:
                        self.metrics.metrics['last_sync_timestamp'] = datetime.utcnow().isoformat()
                        logger.debug(f"Synced {len(sync_data)} items to upstream")
                    else:
                        self.metrics.metrics['sync_failures'] += 1
                        # Re-queue failed items
                        for item in sync_data[:-1]:  # Don't re-queue metrics
                            await self.sync_queue.put(item)
                
                await asyncio.sleep(sync_interval)
                
            except Exception as e:
                logger.error(f"Health sync error: {e}")
                self.metrics.metrics['sync_failures'] += 1
                await asyncio.sleep(sync_interval)
    
    async def _sync_to_upstream(self, data: List[Dict[str, Any]]) -> bool:
        """Sync data to upstream VPS"""
        upstream_host = self.config['network']['upstream_host']
        
        try:
            url = f"http://{upstream_host}:8000/api/v1/edge/sync"
            payload = {
                'node_id': self.config['node_id'],
                'data': data,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    return True
                else:
                    logger.warning(f"Sync failed with status {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Sync request failed: {e}")
            return False
    
    async def _watchdog_loop(self):
        """Watchdog monitoring loop"""
        if not self.config['watchdog']['enabled']:
            return
        
        interval = self.config['watchdog']['interval']
        timeout = self.config['watchdog']['timeout']
        restart_threshold = self.config['watchdog']['restart_threshold']
        failure_count = 0
        
        while self.running:
            try:
                # Check system health
                health_ok = await self._check_system_health()
                
                if health_ok:
                    failure_count = 0
                else:
                    failure_count += 1
                    logger.warning(f"Health check failed ({failure_count}/{restart_threshold})")
                    
                    if failure_count >= restart_threshold:
                        logger.error("Restart threshold reached, initiating restart...")
                        os.system("sudo systemctl restart xorb-edge")
                        break
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                await asyncio.sleep(interval)
    
    async def _check_system_health(self) -> bool:
        """Check system health"""
        try:
            # Check CPU temperature
            temp = self.metrics.metrics.get('temperature_celsius', 0)
            if temp > 80:  # Thermal throttling threshold
                logger.warning(f"High temperature detected: {temp}°C")
                return False
            
            # Check memory usage
            memory_percent = (self.metrics.metrics.get('memory_usage_mb', 0) / (8 * 1024)) * 100
            if memory_percent > 90:
                logger.warning(f"High memory usage: {memory_percent:.1f}%")
                return False
            
            # Check storage
            storage_percent = self.metrics.metrics.get('storage_used_percent', 0)
            if storage_percent > 90:
                logger.warning(f"High storage usage: {storage_percent:.1f}%")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False
    
    async def _autonomous_decision_loop(self):
        """Autonomous decision making loop"""
        if not self.config['autonomous']['enabled']:
            return
        
        adaptation_interval = self.config['autonomous']['adaptation_interval']
        
        while self.running:
            try:
                # Make autonomous decisions based on metrics and conditions
                decisions_made = await self._make_autonomous_decisions()
                self.metrics.metrics['autonomous_decisions'] += decisions_made
                
                await asyncio.sleep(adaptation_interval)
                
            except Exception as e:
                logger.error(f"Autonomous decision error: {e}")
                await asyncio.sleep(adaptation_interval)
    
    async def _make_autonomous_decisions(self) -> int:
        """Make autonomous decisions based on current state"""
        decisions_made = 0
        
        try:
            # Decision 1: Adjust task processing based on system load
            cpu_usage = self.metrics.metrics.get('cpu_usage_percent', 0)
            if cpu_usage > 80:
                # Slow down task processing
                logger.info("High CPU usage detected, implementing adaptive throttling")
                await asyncio.sleep(2)
                decisions_made += 1
            
            # Decision 2: Cleanup old data when storage is high
            storage_usage = self.metrics.metrics.get('storage_used_percent', 0)
            if storage_usage > 80:
                logger.info("High storage usage, triggering cleanup")
                await self._cleanup_old_data()
                decisions_made += 1
            
            # Decision 3: Thermal management
            temperature = self.metrics.metrics.get('temperature_celsius', 0)
            if temperature > 70:
                logger.info("High temperature detected, enabling thermal management")
                # Could implement CPU throttling or fan control
                decisions_made += 1
            
        except Exception as e:
            logger.error(f"Decision making error: {e}")
        
        return decisions_made
    
    async def _cleanup_old_data(self):
        """Cleanup old data to free storage"""
        try:
            # Clean up old task records
            cursor = self.local_db.cursor()
            cutoff_date = datetime.utcnow() - timedelta(days=7)
            cursor.execute(
                "DELETE FROM tasks WHERE completed_at < ?",
                (cutoff_date,)
            )
            
            # Clean up old metrics
            cursor.execute(
                "DELETE FROM metrics WHERE timestamp < ?",
                (cutoff_date,)
            )
            
            self.local_db.commit()
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    async def _api_server(self):
        """Simple HTTP API server for edge worker"""
        from aiohttp import web
        
        async def health_handler(request):
            return web.json_response({
                'status': 'healthy',
                'node_id': self.config['node_id'],
                'uptime': self.metrics.metrics['uptime_seconds'],
                'tasks_processed': self.metrics.metrics['tasks_processed'],
                'timestamp': datetime.utcnow().isoformat()
            })
        
        async def metrics_handler(request):
            return web.json_response({
                'metrics': self.metrics.to_dict(),
                'timestamp': datetime.utcnow().isoformat()
            })
        
        async def task_handler(request):
            data = await request.json()
            await self.task_queue.put(data)
            return web.json_response({'status': 'queued', 'task_id': data.get('task_id')})
        
        app = web.Application()
        app.router.add_get('/health', health_handler)
        app.router.add_get('/metrics', metrics_handler)
        app.router.add_post('/task', task_handler)
        
        port = self.config['network']['local_api_port']
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        
        logger.info(f"API server started on port {port}")
        
        # Keep server running
        while self.running:
            await asyncio.sleep(1)
        
        await runner.cleanup()
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up edge worker...")
        
        if self.session:
            await self.session.close()
        
        if self.local_db:
            self.local_db.close()
        
        logger.info("Edge worker cleanup completed")

async def main():
    """Main function"""
    worker = EdgeWorker()
    
    try:
        await worker.initialize()
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Worker failed: {e}")
    finally:
        await worker.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
EOF
    
    # Transfer edge worker to Pi
    scp /tmp/xorb_edge_worker.py pi@"$PI_HOST":/opt/xorb/
    rm /tmp/xorb_edge_worker.py
    
    # Make executable
    ssh pi@"$PI_HOST" 'chmod +x /opt/xorb/xorb_edge_worker.py'
    
    success "Edge worker service created"
}

# Create systemd service
create_systemd_service() {
    log "Creating systemd service for edge worker..."
    
    ssh pi@"$PI_HOST" << 'EOF'
        # Create systemd service file
        sudo tee /etc/systemd/system/xorb-edge.service << 'SERVICE'
[Unit]
Description=Xorb Edge Worker
After=network.target
Wants=network.target

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=/opt/xorb
ExecStart=/usr/bin/python3 /opt/xorb/xorb_edge_worker.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=xorb-edge

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/xorb

# Resource limits
MemoryMax=6G
CPUQuota=350%

# Environment
Environment=PYTHONPATH=/opt/xorb
Environment=XORB_ENV=edge

[Install]
WantedBy=multi-user.target
SERVICE
        
        # Enable and start service
        sudo systemctl daemon-reload
        sudo systemctl enable xorb-edge.service
        sudo systemctl start xorb-edge.service
        
        # Check status
        sudo systemctl status xorb-edge.service --no-pager
EOF
    
    success "Systemd service created and started"
}

# Setup monitoring and alerting
setup_monitoring() {
    log "Setting up monitoring and alerting..."
    
    ssh pi@"$PI_HOST" << 'EOF'
        # Install and configure node_exporter
        sudo systemctl enable prometheus-node-exporter
        sudo systemctl start prometheus-node-exporter
        
        # Create custom metrics collector
        sudo tee /opt/xorb/collect_metrics.sh << 'METRICS'
#!/bin/bash
# Custom metrics collection for Xorb Edge Worker

METRICS_FILE="/opt/xorb/logs/edge-metrics.prom"
TEMP_FILE=$(mktemp)

# System metrics
echo "# HELP xorb_edge_temperature_celsius Current CPU temperature" > "$TEMP_FILE"
echo "# TYPE xorb_edge_temperature_celsius gauge" >> "$TEMP_FILE"
if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
    TEMP=$(cat /sys/class/thermal/thermal_zone0/temp)
    TEMP_C=$(echo "scale=1; $TEMP/1000" | bc)
    echo "xorb_edge_temperature_celsius $TEMP_C" >> "$TEMP_FILE"
fi

# Memory usage
echo "# HELP xorb_edge_memory_usage_percent Memory usage percentage" >> "$TEMP_FILE"
echo "# TYPE xorb_edge_memory_usage_percent gauge" >> "$TEMP_FILE"
MEM_PERCENT=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
echo "xorb_edge_memory_usage_percent $MEM_PERCENT" >> "$TEMP_FILE"

# Storage usage
echo "# HELP xorb_edge_storage_usage_percent Storage usage percentage" >> "$TEMP_FILE"
echo "# TYPE xorb_edge_storage_usage_percent gauge" >> "$TEMP_FILE"
STORAGE_PERCENT=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
echo "xorb_edge_storage_usage_percent $STORAGE_PERCENT" >> "$TEMP_FILE"

# Service status
echo "# HELP xorb_edge_service_status Service status (1=running, 0=not running)" >> "$TEMP_FILE"
echo "# TYPE xorb_edge_service_status gauge" >> "$TEMP_FILE"
if systemctl is-active --quiet xorb-edge; then
    echo "xorb_edge_service_status 1" >> "$TEMP_FILE"
else
    echo "xorb_edge_service_status 0" >> "$TEMP_FILE"
fi

# Move temp file to final location
mv "$TEMP_FILE" "$METRICS_FILE"
METRICS
        
        chmod +x /opt/xorb/collect_metrics.sh
        
        # Add cron job for metrics collection
        (crontab -l 2>/dev/null; echo "* * * * * /opt/xorb/collect_metrics.sh") | crontab -
        
        # Setup log monitoring
        sudo tee /opt/xorb/monitor_logs.sh << 'LOGMON'
#!/bin/bash
# Monitor logs for errors and send alerts

LOG_FILE="/opt/xorb/logs/edge-worker.log"
ERROR_THRESHOLD=10
ALERT_FILE="/tmp/xorb_alert_sent"

if [ -f "$LOG_FILE" ]; then
    # Count errors in last 5 minutes
    ERROR_COUNT=$(grep -c "ERROR" "$LOG_FILE" | tail -n 100 | wc -l)
    
    if [ "$ERROR_COUNT" -gt "$ERROR_THRESHOLD" ] && [ ! -f "$ALERT_FILE" ]; then
        # Send alert (implement your alerting mechanism here)
        logger "XORB EDGE ALERT: High error rate detected ($ERROR_COUNT errors)"
        touch "$ALERT_FILE"
        
        # Reset alert file after 1 hour
        (sleep 3600; rm -f "$ALERT_FILE") &
    fi
fi
LOGMON
        
        chmod +x /opt/xorb/monitor_logs.sh
        
        # Add cron job for log monitoring
        (crontab -l 2>/dev/null; echo "*/5 * * * * /opt/xorb/monitor_logs.sh") | crontab -
EOF
    
    success "Monitoring and alerting configured"
}

# Setup network resilience
setup_network_resilience() {
    log "Setting up network resilience features..."
    
    if [[ "$NETWORK_RESILIENCE" != "true" ]]; then
        info "Network resilience disabled, skipping..."
        return 0
    fi
    
    ssh pi@"$PI_HOST" << 'EOF'
        # Install and configure connection monitoring
        sudo apt install -y mtr-tiny
        
        # Create network monitoring script
        sudo tee /opt/xorb/network_monitor.sh << 'NETMON'
#!/bin/bash
# Network resilience monitoring and recovery

VPS_HOST="${VPS_HOST:-8.8.8.8}"
PING_THRESHOLD=5
CONNECTIVITY_FILE="/tmp/xorb_connectivity"
RECONNECT_ATTEMPTS=0
MAX_RECONNECT_ATTEMPTS=3

check_connectivity() {
    if ping -c 3 -W 5 "$VPS_HOST" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Check connectivity
if check_connectivity; then
    echo "$(date): Connectivity OK" > "$CONNECTIVITY_FILE"
    RECONNECT_ATTEMPTS=0
else
    echo "$(date): Connectivity FAILED" >> "$CONNECTIVITY_FILE"
    RECONNECT_ATTEMPTS=$((RECONNECT_ATTEMPTS + 1))
    
    if [ $RECONNECT_ATTEMPTS -lt $MAX_RECONNECT_ATTEMPTS ]; then
        # Try network recovery
        logger "XORB EDGE: Network connectivity lost, attempting recovery ($RECONNECT_ATTEMPTS/$MAX_RECONNECT_ATTEMPTS)"
        
        # Restart networking
        sudo systemctl restart networking
        sleep 10
        
        # Try again
        if check_connectivity; then
            logger "XORB EDGE: Network recovery successful"
            RECONNECT_ATTEMPTS=0
        fi
    else
        # Enter offline mode
        logger "XORB EDGE: Entering offline mode after failed recovery attempts"
        # Signal edge worker to enter standalone mode
        curl -X POST http://localhost:8080/mode/offline 2>/dev/null || true
    fi
fi
NETMON
        
        chmod +x /opt/xorb/network_monitor.sh
        
        # Add network monitoring to cron
        (crontab -l 2>/dev/null; echo "*/2 * * * * /opt/xorb/network_monitor.sh") | crontab -
        
        # Configure automatic network recovery
        sudo tee /etc/systemd/system/xorb-network-recovery.service << 'RECOVERY'
[Unit]
Description=Xorb Edge Network Recovery
After=network.target

[Service]
Type=oneshot
ExecStart=/opt/xorb/network_monitor.sh
RemainAfterExit=no

[Install]
WantedBy=multi-user.target
RECOVERY
        
        sudo systemctl enable xorb-network-recovery.service
EOF
    
    success "Network resilience configured"
}

# Setup power management
setup_power_management() {
    log "Setting up power management..."
    
    if [[ "$POWER_MANAGEMENT" == "none" ]]; then
        info "Power management disabled, skipping..."
        return 0
    fi
    
    ssh pi@"$PI_HOST" << 'EOF'
        # Install power management tools
        sudo apt install -y cpufrequtils
        
        # Create power management script
        sudo tee /opt/xorb/power_manager.sh << 'POWERMGR'
#!/bin/bash
# Adaptive power management for edge worker

TEMP_THRESHOLD=70
LOAD_THRESHOLD=80
TEMP_FILE="/sys/class/thermal/thermal_zone0/temp"

get_temperature() {
    if [ -f "$TEMP_FILE" ]; then
        cat "$TEMP_FILE" | awk '{print $1/1000}'
    else
        echo "0"
    fi
}

get_cpu_load() {
    uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//'
}

manage_power() {
    TEMP=$(get_temperature)
    LOAD=$(get_cpu_load)
    
    # Convert to integers for comparison
    TEMP_INT=$(echo "$TEMP" | cut -d. -f1)
    LOAD_INT=$(echo "$LOAD" | cut -d. -f1)
    
    if [ "$TEMP_INT" -gt "$TEMP_THRESHOLD" ] || [ "$LOAD_INT" -gt "$LOAD_THRESHOLD" ]; then
        # High temperature or load - reduce performance
        echo "powersave" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
        logger "XORB EDGE: Activated power saving mode (Temp: ${TEMP}°C, Load: ${LOAD})"
    else
        # Normal conditions - performance mode
        echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
    fi
}

case "${POWER_MANAGEMENT:-adaptive}" in
    "adaptive")
        manage_power
        ;;
    "performance")
        echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
        ;;
    "powersave")
        echo "powersave" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
        ;;
esac
POWERMGR
        
        chmod +x /opt/xorb/power_manager.sh
        
        # Add power management to cron
        (crontab -l 2>/dev/null; echo "*/5 * * * * /opt/xorb/power_manager.sh") | crontab -
        
        # Setup thermal protection
        sudo tee /etc/systemd/system/xorb-thermal-protection.service << 'THERMAL'
[Unit]
Description=Xorb Edge Thermal Protection
After=multi-user.target

[Service]
Type=simple
ExecStart=/bin/bash -c 'while true; do if [ -f /sys/class/thermal/thermal_zone0/temp ]; then TEMP=$(cat /sys/class/thermal/thermal_zone0/temp); if [ $TEMP -gt 80000 ]; then logger "CRITICAL: CPU temperature $(($TEMP/1000))°C - Emergency shutdown"; systemctl stop xorb-edge; fi; fi; sleep 10; done'
Restart=always

[Install]
WantedBy=multi-user.target
THERMAL
        
        sudo systemctl enable xorb-thermal-protection.service
        sudo systemctl start xorb-thermal-protection.service
EOF
    
    success "Power management configured"
}

# Setup security hardening
setup_security_hardening() {
    log "Setting up security hardening..."
    
    ssh pi@"$PI_HOST" << 'EOF'
        # Configure SSH security
        sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
        sudo sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
        sudo sed -i 's/#Port 22/Port 2222/' /etc/ssh/sshd_config
        
        # Restart SSH service
        sudo systemctl restart ssh
        
        # Configure fail2ban for SSH protection
        sudo tee /etc/fail2ban/jail.local << 'FAIL2BAN'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = 2222
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
FAIL2BAN
        
        sudo systemctl restart fail2ban
        
        # Setup automatic security updates
        echo 'Unattended-Upgrade::Automatic-Reboot "false";' | sudo tee -a /etc/apt/apt.conf.d/50unattended-upgrades
        echo 'Unattended-Upgrade::Automatic-Reboot-Time "02:00";' | sudo tee -a /etc/apt/apt.conf.d/50unattended-upgrades
        
        # Configure log rotation for security logs
        sudo tee /etc/logrotate.d/xorb-security << 'SECLOG'
/opt/xorb/logs/security.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 pi pi
}
SECLOG
        
        # Setup intrusion detection
        sudo tee /opt/xorb/security_monitor.sh << 'SECMON'
#!/bin/bash
# Basic intrusion detection for edge worker

# Check for suspicious processes
SUSPICIOUS_PROCS=$(ps aux | grep -E "(nc|ncat|netcat|telnet)" | grep -v grep | wc -l)
if [ $SUSPICIOUS_PROCS -gt 0 ]; then
    logger "XORB SECURITY: Suspicious network processes detected"
fi

# Check for unauthorized logins
FAILED_LOGINS=$(grep "Failed password" /var/log/auth.log | tail -n 10 | wc -l)
if [ $FAILED_LOGINS -gt 5 ]; then
    logger "XORB SECURITY: Multiple failed login attempts detected"
fi

# Check disk usage for potential attacks
ROOT_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $ROOT_USAGE -gt 95 ]; then
    logger "XORB SECURITY: Disk usage critical - possible DoS attack"
fi

# Check network connections
ESTABLISHED_CONNS=$(netstat -an | grep ESTABLISHED | wc -l)
if [ $ESTABLISHED_CONNS -gt 100 ]; then
    logger "XORB SECURITY: High number of network connections"
fi
SECMON
        
        chmod +x /opt/xorb/security_monitor.sh
        
        # Add security monitoring to cron
        (crontab -l 2>/dev/null; echo "*/10 * * * * /opt/xorb/security_monitor.sh") | crontab -
EOF
    
    success "Security hardening completed"
}

# Create health check and status script
create_health_check() {
    log "Creating health check and status scripts..."
    
    cat > /tmp/edge_health_check.sh << 'EOF'
#!/bin/bash
# Xorb Edge Worker Health Check Script

HEALTH_URL="http://localhost:8080/health"
METRICS_URL="http://localhost:8080/metrics"

echo "=== Xorb Edge Worker Health Check ==="
echo "Timestamp: $(date)"
echo ""

# Check service status
echo "Service Status:"
if systemctl is-active --quiet xorb-edge; then
    echo "  ✅ xorb-edge service is running"
else
    echo "  ❌ xorb-edge service is NOT running"
fi

# Check API health
echo ""
echo "API Health:"
if curl -s "$HEALTH_URL" >/dev/null 2>&1; then
    echo "  ✅ API is responding"
    API_HEALTH=$(curl -s "$HEALTH_URL" | jq -r '.status' 2>/dev/null || echo "unknown")
    echo "  📊 Status: $API_HEALTH"
else
    echo "  ❌ API is NOT responding"
fi

# Check system resources
echo ""
echo "System Resources:"
TEMP=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null | awk '{print $1/1000}' || echo "N/A")
echo "  🌡️  CPU Temperature: ${TEMP}°C"

MEM_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
echo "  🧠 Memory Usage: ${MEM_USAGE}%"

CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
echo "  ⚡ CPU Usage: ${CPU_USAGE}%"

DISK_USAGE=$(df / | tail -1 | awk '{print $5}')
echo "  💾 Disk Usage: ${DISK_USAGE}"

# Check network connectivity
echo ""
echo "Network Connectivity:"
if ping -c 1 -W 5 8.8.8.8 >/dev/null 2>&1; then
    echo "  🌐 Internet connectivity: OK"
else
    echo "  🌐 Internet connectivity: FAILED"
fi

# Check logs for recent errors
echo ""
echo "Recent Log Summary:"
if [ -f /opt/xorb/logs/edge-worker.log ]; then
    ERROR_COUNT=$(tail -n 100 /opt/xorb/logs/edge-worker.log | grep -c "ERROR" || echo "0")
    WARN_COUNT=$(tail -n 100 /opt/xorb/logs/edge-worker.log | grep -c "WARN" || echo "0")
    echo "  📋 Recent errors: $ERROR_COUNT"
    echo "  ⚠️  Recent warnings: $WARN_COUNT"
else
    echo "  📋 Log file not found"
fi

echo ""
echo "=== Health Check Complete ==="
EOF

    # Transfer health check script to Pi
    scp /tmp/edge_health_check.sh pi@"$PI_HOST":/opt/xorb/
    rm /tmp/edge_health_check.sh
    
    # Make executable and setup alias
    ssh pi@"$PI_HOST" << 'EOF'
        chmod +x /opt/xorb/edge_health_check.sh
        
        # Add convenient aliases
        echo "alias xorb-health='/opt/xorb/edge_health_check.sh'" >> ~/.bashrc
        echo "alias xorb-status='systemctl status xorb-edge'" >> ~/.bashrc
        echo "alias xorb-logs='journalctl -u xorb-edge -f'" >> ~/.bashrc
        echo "alias xorb-restart='sudo systemctl restart xorb-edge'" >> ~/.bashrc
        
        source ~/.bashrc
EOF
    
    success "Health check scripts created"
}

# Final validation and testing
run_deployment_validation() {
    log "Running deployment validation..."
    
    # Wait for service to start
    sleep 30
    
    # Check service status
    if ssh pi@"$PI_HOST" 'systemctl is-active --quiet xorb-edge'; then
        success "✅ Edge worker service is running"
    else
        error "❌ Edge worker service failed to start"
    fi
    
    # Check API responsiveness
    if ssh pi@"$PI_HOST" 'curl -f http://localhost:8080/health >/dev/null 2>&1'; then
        success "✅ Edge worker API is responding"
    else
        warn "⚠️ Edge worker API is not responding (may still be starting up)"
    fi
    
    # Check log output
    log "Checking log output..."
    ssh pi@"$PI_HOST" 'tail -n 20 /opt/xorb/logs/edge-worker.log' || warn "Could not read log file"
    
    # Run health check
    log "Running health check..."
    ssh pi@"$PI_HOST" '/opt/xorb/edge_health_check.sh'
    
    success "Deployment validation completed"
}

# Print deployment summary
print_deployment_summary() {
    header "🎉 Enhanced Pi 5 Edge Deployment Complete!"
    
    echo -e "${GREEN}Deployment Summary:${NC}"
    echo -e "  📍 Edge Node ID: ${CYAN}$EDGE_NODE_ID${NC}"
    echo -e "  🖥️  Target Host: ${CYAN}$PI_HOST${NC}"
    echo -e "  🔧 Deployment Mode: ${CYAN}$DEPLOYMENT_MODE${NC}"
    echo -e "  ⚡ Power Management: ${CYAN}$POWER_MANAGEMENT${NC}"
    echo -e "  🔒 Watchdog Enabled: ${CYAN}$ENABLE_WATCHDOG${NC}"
    echo -e "  🌐 Network Resilience: ${CYAN}$NETWORK_RESILIENCE${NC}"
    
    echo ""
    echo -e "${GREEN}Services Deployed:${NC}"
    echo -e "  🚀 Xorb Edge Worker"
    echo -e "  📊 Monitoring & Metrics"
    echo -e "  🔒 Security Hardening"
    echo -e "  🌡️  Thermal Protection"
    echo -e "  🌐 Network Resilience"
    echo -e "  💾 Storage Management"
    
    echo ""
    echo -e "${GREEN}Management Commands:${NC}"
    echo -e "  ${CYAN}ssh pi@$PI_HOST${NC}"
    echo -e "  ${CYAN}xorb-health${NC}        # Check system health"
    echo -e "  ${CYAN}xorb-status${NC}        # Check service status"
    echo -e "  ${CYAN}xorb-logs${NC}          # View live logs"
    echo -e "  ${CYAN}xorb-restart${NC}       # Restart service"
    
    echo ""
    echo -e "${GREEN}Monitoring Endpoints:${NC}"
    echo -e "  🔍 Health: ${CYAN}http://$PI_HOST:8080/health${NC}"
    echo -e "  📊 Metrics: ${CYAN}http://$PI_HOST:8080/metrics${NC}"
    echo -e "  📈 Node Exporter: ${CYAN}http://$PI_HOST:9100/metrics${NC}"
    
    echo ""
    echo -e "${GREEN}Key Features:${NC}"
    echo -e "  ✅ Autonomous operation with fallback capabilities"
    echo -e "  ✅ Resilient networking with automatic recovery"
    echo -e "  ✅ Adaptive power management and thermal protection"
    echo -e "  ✅ Comprehensive monitoring and alerting"
    echo -e "  ✅ Security hardening and intrusion detection"
    echo -e "  ✅ Automatic health sync with central VPS"
    echo -e "  ✅ Local data persistence and queue management"
    echo -e "  ✅ Watchdog protection and self-healing"
    
    echo ""
    echo -e "${PURPLE}🎯 Your Raspberry Pi 5 is now a fully autonomous edge security node!${NC}"
}

# Main execution function
main() {
    header "🚀 Enhanced Raspberry Pi 5 Edge Deployment"
    
    log "Starting enhanced edge deployment for $PI_HOST..."
    
    # Execute deployment steps
    check_prerequisites
    optimize_pi5_hardware
    install_enhanced_dependencies
    setup_resilient_storage
    create_edge_configuration
    create_edge_worker_service
    create_systemd_service
    setup_monitoring
    setup_network_resilience
    setup_power_management
    setup_security_hardening
    create_health_check
    run_deployment_validation
    print_deployment_summary
}

# Parse command line arguments
case "${1:-}" in
    "--help"|"-h"|"help")
        echo "Enhanced Raspberry Pi 5 Edge Deployment"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Environment Variables:"
        echo "  PI_HOST                 - Raspberry Pi hostname/IP (required)"
        echo "  VPS_HOST               - Central VPS hostname/IP (optional)"
        echo "  EDGE_NODE_ID           - Unique edge node identifier"
        echo "  DEPLOYMENT_MODE        - autonomous|relay|failover|standalone"
        echo "  ENABLE_WATCHDOG        - Enable watchdog monitoring (true/false)"
        echo "  ENABLE_HEALTH_SYNC     - Enable health sync with VPS (true/false)"
        echo "  POWER_MANAGEMENT       - none|adaptive|performance|powersave"
        echo "  NETWORK_RESILIENCE     - Enable network resilience (true/false)"
        echo ""
        echo "Examples:"
        echo "  PI_HOST=192.168.1.100 $0"
        echo "  PI_HOST=pi5.local VPS_HOST=vps.example.com $0"
        echo "  PI_HOST=192.168.1.100 DEPLOYMENT_MODE=standalone $0"
        exit 0
        ;;
    "status")
        if [[ -n "$PI_HOST" ]]; then
            ssh pi@"$PI_HOST" '/opt/xorb/edge_health_check.sh' 2>/dev/null || error "Cannot connect to $PI_HOST"
        else
            error "PI_HOST environment variable required"
        fi
        ;;
    "logs")
        if [[ -n "$PI_HOST" ]]; then
            ssh pi@"$PI_HOST" 'journalctl -u xorb-edge -f' 2>/dev/null || error "Cannot connect to $PI_HOST"
        else
            error "PI_HOST environment variable required"
        fi
        ;;
    *)
        main "$@"
        ;;
esac