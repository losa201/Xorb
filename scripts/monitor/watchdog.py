#!/usr/bin/env python3
"""
XORB System Watchdog - Self-healing and health monitoring
"""

import logging
import subprocess
import time

import docker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/xorb-watchdog.log'),
        logging.StreamHandler()
    ]
)

class XorbWatchdog:
    def __init__(self):
        self.client = docker.from_env()
        self.required_containers = [
            'xorb-postgres', 'xorb-redis', 'xorb-api',
            'xorb-orchestrator', 'xorb-worker', 'xorb-prometheus'
        ]
        self.restart_attempts = {}
        self.max_restart_attempts = 3
        self.restart_backoff = [30, 60, 120]  # Exponential backoff

    def check_container_health(self, container_name):
        try:
            container = self.client.containers.get(container_name)
            return container.status == 'running'
        except docker.errors.NotFound:
            logging.error(f"Container {container_name} not found")
            return False
        except Exception as e:
            logging.error(f"Error checking {container_name}: {e}")
            return False

    def restart_container(self, container_name):
        attempts = self.restart_attempts.get(container_name, 0)

        if attempts >= self.max_restart_attempts:
            logging.error(f"Max restart attempts reached for {container_name}")
            return False

        try:
            container = self.client.containers.get(container_name)
            logging.info(f"Restarting container {container_name} (attempt {attempts + 1})")
            container.restart()

            # Wait for container to be healthy
            time.sleep(10)

            if self.check_container_health(container_name):
                logging.info(f"Successfully restarted {container_name}")
                self.restart_attempts[container_name] = 0
                return True
            else:
                self.restart_attempts[container_name] = attempts + 1
                logging.error(f"Failed to restart {container_name}")
                return False

        except Exception as e:
            logging.error(f"Error restarting {container_name}: {e}")
            self.restart_attempts[container_name] = attempts + 1
            return False

    def monitor_system_resources(self):
        try:
            # Check disk space
            result = subprocess.run(['df', '/'], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                usage = lines[1].split()
                usage_percent = int(usage[4].rstrip('%'))
                if usage_percent > 90:
                    logging.warning(f"High disk usage: {usage_percent}%")

            # Check memory usage
            with open('/proc/meminfo') as f:
                meminfo = f.read()
                total = int([line for line in meminfo.split('\n') if 'MemTotal' in line][0].split()[1])
                available = int([line for line in meminfo.split('\n') if 'MemAvailable' in line][0].split()[1])
                usage_percent = (1 - available / total) * 100

                if usage_percent > 90:
                    logging.warning(f"High memory usage: {usage_percent:.1f}%")

        except Exception as e:
            logging.error(f"Error monitoring system resources: {e}")

    def run(self):
        logging.info("XORB Watchdog started")

        while True:
            try:
                # Check all required containers
                for container_name in self.required_containers:
                    if not self.check_container_health(container_name):
                        logging.warning(f"Container {container_name} is not healthy")

                        # Wait before restart attempt
                        attempts = self.restart_attempts.get(container_name, 0)
                        if attempts < len(self.restart_backoff):
                            time.sleep(self.restart_backoff[attempts])

                        self.restart_container(container_name)

                # Monitor system resources
                self.monitor_system_resources()

                # Sleep before next check
                time.sleep(60)  # Check every minute

            except KeyboardInterrupt:
                logging.info("Watchdog stopped by user")
                break
            except Exception as e:
                logging.error(f"Watchdog error: {e}")
                time.sleep(30)

if __name__ == "__main__":
    watchdog = XorbWatchdog()
    watchdog.run()
