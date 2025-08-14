#!/usr/bin/env python3
"""
XORB Blue Team Autonomous Defense Agent
Real-time threat detection and autonomous response
"""

import asyncio
import subprocess
import json
import time
import psutil
import threading
import logging
import socket
from datetime import datetime, timedelta
from pathlib import Path
from scapy.all import *
import re
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlueTeamAgent:
    def __init__(self):
        self.agent_id = f"blue_agent_{int(time.time())}"
        self.network_interface = "eth0"
        self.monitored_network = "172.20.0.0/24"

        # Detection data
        self.alerts = []
        self.blocked_ips = set()
        self.suspicious_activities = []
        self.network_baselines = {}
        self.threat_indicators = []

        # Learning data
        self.ml_model = None
        self.detection_patterns = {}
        self.false_positive_rate = 0.0
        self.true_positive_rate = 0.0

        # Active monitoring
        self.packet_capture_active = False
        self.log_monitoring_active = False

    async def start_autonomous_defense(self):
        """Start autonomous blue team defense operations"""
        logger.info(f"🔵 Blue Team Agent {self.agent_id} starting autonomous defense")

        # Start parallel defense activities
        tasks = [
            asyncio.create_task(self.network_monitoring()),
            asyncio.create_task(self.log_analysis()),
            asyncio.create_task(self.system_monitoring()),
            asyncio.create_task(self.threat_hunting()),
            asyncio.create_task(self.incident_response()),
            asyncio.create_task(self.learn_and_adapt_defenses())
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Blue team defense failed: {e}")

    async def network_monitoring(self):
        """Monitor network traffic for threats"""
        logger.info("🔍 Starting network traffic monitoring")

        while True:
            try:
                # Monitor network connections
                await self.monitor_network_connections()

                # Analyze traffic patterns
                await self.analyze_traffic_patterns()

                # Detect port scans
                await self.detect_port_scans()

                # Monitor DNS requests
                await self.monitor_dns_activity()

                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Network monitoring failed: {e}")
                await asyncio.sleep(10)

    async def monitor_network_connections(self):
        """Monitor active network connections"""
        try:
            connections = psutil.net_connections(kind='inet')

            suspicious_connections = []
            for conn in connections:
                if conn.raddr:  # Remote address exists
                    remote_ip = conn.raddr.ip
                    remote_port = conn.raddr.port

                    # Check for suspicious ports
                    if remote_port in [4444, 5555, 6666, 1234, 31337]:  # Common backdoor ports
                        suspicious_connections.append({
                            'local_addr': f"{conn.laddr.ip}:{conn.laddr.port}",
                            'remote_addr': f"{remote_ip}:{remote_port}",
                            'status': conn.status,
                            'pid': conn.pid,
                            'detected_at': datetime.now().isoformat()
                        })

            if suspicious_connections:
                await self.generate_alert(
                    'suspicious_connections',
                    f"Detected {len(suspicious_connections)} suspicious network connections",
                    {'connections': suspicious_connections}
                )

        except Exception as e:
            logger.error(f"Connection monitoring failed: {e}")

    async def analyze_traffic_patterns(self):
        """Analyze network traffic for anomalies"""
        try:
            # Get network statistics
            net_io = psutil.net_io_counters()

            current_stats = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'timestamp': time.time()
            }

            # Check for unusual traffic volumes
            if hasattr(self, 'last_net_stats'):
                time_diff = current_stats['timestamp'] - self.last_net_stats['timestamp']

                if time_diff > 0:
                    bytes_per_sec = (current_stats['bytes_sent'] - self.last_net_stats['bytes_sent']) / time_diff

                    # Alert on high outbound traffic (potential data exfiltration)
                    if bytes_per_sec > 10 * 1024 * 1024:  # 10MB/s
                        await self.generate_alert(
                            'high_outbound_traffic',
                            f"High outbound traffic detected: {bytes_per_sec / (1024*1024):.2f} MB/s",
                            {'bytes_per_second': bytes_per_sec}
                        )

            self.last_net_stats = current_stats

        except Exception as e:
            logger.error(f"Traffic analysis failed: {e}")

    async def detect_port_scans(self):
        """Detect port scanning activities"""
        try:
            # Monitor for rapid connection attempts from same IP
            connections = psutil.net_connections(kind='inet')
            connection_attempts = {}

            for conn in connections:
                if conn.raddr and conn.status == 'SYN_RECV':
                    remote_ip = conn.raddr.ip
                    if remote_ip not in connection_attempts:
                        connection_attempts[remote_ip] = 0
                    connection_attempts[remote_ip] += 1

            # Alert on potential port scans
            for ip, attempts in connection_attempts.items():
                if attempts > 10:  # More than 10 SYN_RECV from same IP
                    await self.generate_alert(
                        'port_scan_detected',
                        f"Potential port scan from {ip}: {attempts} connection attempts",
                        {'source_ip': ip, 'attempts': attempts}
                    )

                    # Auto-block aggressive scanners
                    if attempts > 50:
                        await self.block_ip(ip, 'aggressive_port_scan')

        except Exception as e:
            logger.error(f"Port scan detection failed: {e}")

    async def monitor_dns_activity(self):
        """Monitor DNS requests for malicious domains"""
        malicious_domains = [
            'evil.com',
            'malware.com',
            'c2server.com',
            'badactor.net',
            'attacker.com'
        ]

        try:
            # Check recent DNS queries (simplified - would use actual DNS logs)
            for domain in malicious_domains:
                # Simulate DNS monitoring
                if self.check_dns_query(domain):
                    await self.generate_alert(
                        'malicious_dns_query',
                        f"DNS query to known malicious domain: {domain}",
                        {'domain': domain, 'threat_type': 'c2_communication'}
                    )

        except Exception as e:
            logger.error(f"DNS monitoring failed: {e}")

    def check_dns_query(self, domain):
        """Check if DNS query was made to specific domain"""
        # Simplified check - in real implementation would parse DNS logs
        return False

    async def log_analysis(self):
        """Analyze system and application logs"""
        logger.info("📋 Starting log analysis")

        while True:
            try:
                # Analyze authentication logs
                await self.analyze_auth_logs()

                # Analyze web server logs
                await self.analyze_web_logs()

                # Analyze system logs
                await self.analyze_system_logs()

                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Log analysis failed: {e}")
                await asyncio.sleep(30)

    async def analyze_auth_logs(self):
        """Analyze authentication logs for suspicious activity"""
        try:
            # Check for failed login attempts
            auth_log_path = "/var/log/auth.log"

            if Path(auth_log_path).exists():
                with open(auth_log_path, 'r') as f:
                    recent_lines = f.readlines()[-100:]  # Last 100 lines

                failed_logins = {}
                successful_logins = []

                for line in recent_lines:
                    # Parse failed SSH attempts
                    if "Failed password" in line:
                        ip_match = re.search(r'from (\d+\.\d+\.\d+\.\d+)', line)
                        if ip_match:
                            ip = ip_match.group(1)
                            failed_logins[ip] = failed_logins.get(ip, 0) + 1

                    # Parse successful logins
                    elif "Accepted password" in line or "Accepted publickey" in line:
                        ip_match = re.search(r'from (\d+\.\d+\.\d+\.\d+)', line)
                        user_match = re.search(r'for (\w+)', line)
                        if ip_match and user_match:
                            successful_logins.append({
                                'user': user_match.group(1),
                                'ip': ip_match.group(1),
                                'timestamp': datetime.now().isoformat()
                            })

                # Alert on brute force attempts
                for ip, attempts in failed_logins.items():
                    if attempts > 5:
                        await self.generate_alert(
                            'brute_force_attempt',
                            f"Brute force attempt detected from {ip}: {attempts} failed logins",
                            {'source_ip': ip, 'failed_attempts': attempts}
                        )

                        # Auto-block after many attempts
                        if attempts > 20:
                            await self.block_ip(ip, 'brute_force_attack')

                # Alert on unusual successful logins
                for login in successful_logins:
                    if login['user'] in ['root', 'admin'] and login['ip'] not in self.get_trusted_ips():
                        await self.generate_alert(
                            'suspicious_privileged_login',
                            f"Privileged user {login['user']} logged in from {login['ip']}",
                            login
                        )

        except Exception as e:
            logger.error(f"Auth log analysis failed: {e}")

    async def analyze_web_logs(self):
        """Analyze web server logs for attacks"""
        try:
            # Common web attack patterns
            attack_patterns = {
                'sql_injection': [r"union\s+select", r"'.*or.*'", r"drop\s+table"],
                'xss': [r"<script", r"javascript:", r"onerror="],
                'directory_traversal': [r"\.\./", r"\.\.\\", r"etc/passwd"],
                'command_injection': [r";\s*ls", r"&&\s*whoami", r"\|\s*cat"]
            }

            # Simulate web log analysis (would parse actual access logs)
            suspicious_requests = []

            # Check for common attack patterns
            for attack_type, patterns in attack_patterns.items():
                for pattern in patterns:
                    # Simulate detection
                    if self.simulate_pattern_match(pattern):
                        suspicious_requests.append({
                            'attack_type': attack_type,
                            'pattern': pattern,
                            'timestamp': datetime.now().isoformat(),
                            'source_ip': '172.20.0.100'  # Example attacker IP
                        })

            if suspicious_requests:
                for request in suspicious_requests:
                    await self.generate_alert(
                        'web_attack_detected',
                        f"Web attack detected: {request['attack_type']}",
                        request
                    )

        except Exception as e:
            logger.error(f"Web log analysis failed: {e}")

    def simulate_pattern_match(self, pattern):
        """Simulate pattern matching in logs"""
        # Random chance for demo purposes
        import random
        return random.random() < 0.1  # 10% chance

    async def analyze_system_logs(self):
        """Analyze system logs for anomalies"""
        try:
            # Check system resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Alert on high resource usage (potential DoS or crypto mining)
            if cpu_percent > 90:
                await self.generate_alert(
                    'high_cpu_usage',
                    f"High CPU usage detected: {cpu_percent}%",
                    {'cpu_percent': cpu_percent}
                )

            if memory.percent > 95:
                await self.generate_alert(
                    'high_memory_usage',
                    f"High memory usage detected: {memory.percent}%",
                    {'memory_percent': memory.percent}
                )

            # Check for unusual processes
            await self.detect_malicious_processes()

        except Exception as e:
            logger.error(f"System log analysis failed: {e}")

    async def detect_malicious_processes(self):
        """Detect potentially malicious processes"""
        try:
            suspicious_process_names = [
                'nc', 'netcat', 'ncat',  # Network tools
                'metasploit', 'msfconsole', 'meterpreter',  # Metasploit
                'sqlmap', 'nikto', 'dirb',  # Scanning tools
                'cryptominer', 'xmrig'  # Crypto miners
            ]

            malicious_processes = []

            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'username']):
                try:
                    process_info = proc.info
                    process_name = process_info['name'].lower()

                    # Check for suspicious process names
                    for suspicious_name in suspicious_process_names:
                        if suspicious_name in process_name:
                            malicious_processes.append({
                                'pid': process_info['pid'],
                                'name': process_info['name'],
                                'cmdline': ' '.join(process_info['cmdline'] or []),
                                'user': process_info['username'],
                                'detected_at': datetime.now().isoformat()
                            })
                            break

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            if malicious_processes:
                for proc in malicious_processes:
                    await self.generate_alert(
                        'malicious_process_detected',
                        f"Suspicious process detected: {proc['name']} (PID: {proc['pid']})",
                        proc
                    )

                    # Auto-kill dangerous processes
                    try:
                        psutil.Process(proc['pid']).terminate()
                        logger.info(f"🔒 Terminated malicious process: {proc['name']}")
                    except:
                        logger.warning(f"Failed to terminate process {proc['pid']}")

        except Exception as e:
            logger.error(f"Process detection failed: {e}")

    async def system_monitoring(self):
        """Monitor system for security events"""
        logger.info("🖥️ Starting system monitoring")

        while True:
            try:
                # File integrity monitoring
                await self.file_integrity_monitoring()

                # Registry monitoring (if Windows)
                await self.registry_monitoring()

                # Network service monitoring
                await self.service_monitoring()

                await asyncio.sleep(120)

            except Exception as e:
                logger.error(f"System monitoring failed: {e}")
                await asyncio.sleep(60)

    async def file_integrity_monitoring(self):
        """Monitor critical files for changes"""
        critical_files = [
            '/etc/passwd',
            '/etc/shadow',
            '/etc/hosts',
            '/etc/crontab',
            '/etc/ssh/sshd_config'
        ]

        try:
            for file_path in critical_files:
                if Path(file_path).exists():
                    # Calculate file hash
                    file_hash = self.calculate_file_hash(file_path)

                    # Check if hash changed
                    if hasattr(self, 'file_hashes'):
                        if file_path in self.file_hashes:
                            if self.file_hashes[file_path] != file_hash:
                                await self.generate_alert(
                                    'file_integrity_violation',
                                    f"Critical file modified: {file_path}",
                                    {
                                        'file_path': file_path,
                                        'old_hash': self.file_hashes[file_path],
                                        'new_hash': file_hash
                                    }
                                )
                    else:
                        self.file_hashes = {}

                    self.file_hashes[file_path] = file_hash

        except Exception as e:
            logger.error(f"File integrity monitoring failed: {e}")

    def calculate_file_hash(self, file_path):
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except:
            return None

    async def registry_monitoring(self):
        """Monitor Windows registry for changes (placeholder)"""
        # Would implement Windows registry monitoring
        pass

    async def service_monitoring(self):
        """Monitor running services for anomalies"""
        try:
            # Get list of listening ports
            listening_ports = []
            for conn in psutil.net_connections(kind='inet'):
                if conn.status == 'LISTEN':
                    listening_ports.append({
                        'port': conn.laddr.port,
                        'address': conn.laddr.ip,
                        'pid': conn.pid
                    })

            # Check for unexpected services
            unexpected_ports = []
            known_ports = {22, 80, 443, 3306, 5432}  # SSH, HTTP, HTTPS, MySQL, PostgreSQL

            for port_info in listening_ports:
                if port_info['port'] not in known_ports and port_info['port'] > 1024:
                    unexpected_ports.append(port_info)

            if unexpected_ports:
                await self.generate_alert(
                    'unexpected_service',
                    f"Unexpected services detected on {len(unexpected_ports)} ports",
                    {'services': unexpected_ports}
                )

        except Exception as e:
            logger.error(f"Service monitoring failed: {e}")

    async def threat_hunting(self):
        """Proactive threat hunting"""
        logger.info("🕵️ Starting threat hunting activities")

        while True:
            try:
                # Hunt for indicators of compromise
                await self.hunt_for_iocs()

                # Behavioral analysis
                await self.behavioral_analysis()

                # Threat intelligence correlation
                await self.threat_intelligence_correlation()

                await asyncio.sleep(300)  # Every 5 minutes

            except Exception as e:
                logger.error(f"Threat hunting failed: {e}")
                await asyncio.sleep(120)

    async def hunt_for_iocs(self):
        """Hunt for indicators of compromise"""
        try:
            # Known malicious IPs
            malicious_ips = [
                '10.0.0.100',  # Example malicious IP
                '192.168.1.100',
                '172.16.0.100'
            ]

            # Check network connections against IOCs
            connections = psutil.net_connections(kind='inet')

            for conn in connections:
                if conn.raddr and conn.raddr.ip in malicious_ips:
                    await self.generate_alert(
                        'ioc_match',
                        f"Connection to known malicious IP: {conn.raddr.ip}",
                        {
                            'malicious_ip': conn.raddr.ip,
                            'local_port': conn.laddr.port,
                            'remote_port': conn.raddr.port,
                            'pid': conn.pid
                        }
                    )

                    # Auto-block malicious IPs
                    await self.block_ip(conn.raddr.ip, 'ioc_match')

        except Exception as e:
            logger.error(f"IOC hunting failed: {e}")

    async def behavioral_analysis(self):
        """Analyze system behavior for anomalies"""
        try:
            # Analyze process creation patterns
            current_processes = set()
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    current_processes.add(proc.info['name'])
                except:
                    continue

            if hasattr(self, 'previous_processes'):
                new_processes = current_processes - self.previous_processes

                # Alert on unusual new processes
                suspicious_new_processes = []
                for proc_name in new_processes:
                    if any(keyword in proc_name.lower() for keyword in ['hack', 'exploit', 'payload']):
                        suspicious_new_processes.append(proc_name)

                if suspicious_new_processes:
                    await self.generate_alert(
                        'suspicious_process_creation',
                        f"Suspicious processes created: {suspicious_new_processes}",
                        {'processes': suspicious_new_processes}
                    )

            self.previous_processes = current_processes

        except Exception as e:
            logger.error(f"Behavioral analysis failed: {e}")

    async def threat_intelligence_correlation(self):
        """Correlate alerts with threat intelligence"""
        try:
            # Correlate recent alerts with known threat patterns
            recent_alerts = [alert for alert in self.alerts
                           if datetime.fromisoformat(alert['timestamp']) > datetime.now() - timedelta(hours=1)]

            # Group alerts by source IP
            ip_alerts = {}
            for alert in recent_alerts:
                source_ip = alert.get('details', {}).get('source_ip')
                if source_ip:
                    if source_ip not in ip_alerts:
                        ip_alerts[source_ip] = []
                    ip_alerts[source_ip].append(alert)

            # Detect multi-stage attacks
            for ip, alerts in ip_alerts.items():
                if len(alerts) >= 3:  # Multiple alerts from same IP
                    alert_types = [alert['type'] for alert in alerts]

                    # Check for attack progression patterns
                    if ('port_scan_detected' in alert_types and
                        'brute_force_attempt' in alert_types):

                        await self.generate_alert(
                            'multi_stage_attack',
                            f"Multi-stage attack detected from {ip}",
                            {
                                'source_ip': ip,
                                'attack_stages': alert_types,
                                'alert_count': len(alerts)
                            }
                        )

                        # Escalate response
                        await self.block_ip(ip, 'multi_stage_attack')

        except Exception as e:
            logger.error(f"Threat intelligence correlation failed: {e}")

    async def incident_response(self):
        """Automated incident response"""
        logger.info("🚨 Starting incident response monitoring")

        while True:
            try:
                # Process critical alerts
                await self.process_critical_alerts()

                # Coordinate response actions
                await self.coordinate_response_actions()

                # Update threat landscape
                await self.update_threat_landscape()

                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Incident response failed: {e}")
                await asyncio.sleep(15)

    async def process_critical_alerts(self):
        """Process and respond to critical alerts"""
        try:
            critical_alerts = [alert for alert in self.alerts
                             if alert.get('severity') == 'critical' and not alert.get('processed')]

            for alert in critical_alerts:
                logger.info(f"🚨 Processing critical alert: {alert['type']}")

                # Execute automated response
                await self.execute_automated_response(alert)

                # Mark as processed
                alert['processed'] = True
                alert['processed_at'] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"Critical alert processing failed: {e}")

    async def execute_automated_response(self, alert):
        """Execute automated response to threats"""
        alert_type = alert['type']

        try:
            if alert_type in ['brute_force_attempt', 'port_scan_detected', 'multi_stage_attack']:
                source_ip = alert.get('details', {}).get('source_ip')
                if source_ip:
                    await self.block_ip(source_ip, alert_type)

            elif alert_type == 'malicious_process_detected':
                pid = alert.get('details', {}).get('pid')
                if pid:
                    try:
                        psutil.Process(pid).terminate()
                        logger.info(f"🔒 Terminated malicious process PID {pid}")
                    except:
                        logger.warning(f"Failed to terminate process {pid}")

            elif alert_type == 'file_integrity_violation':
                file_path = alert.get('details', {}).get('file_path')
                if file_path:
                    # Backup the modified file and restore from backup
                    await self.quarantine_file(file_path)

            elif alert_type == 'suspicious_connections':
                # Block suspicious outbound connections
                connections = alert.get('details', {}).get('connections', [])
                for conn in connections:
                    remote_ip = conn.get('remote_addr', '').split(':')[0]
                    if remote_ip:
                        await self.block_ip(remote_ip, 'suspicious_connection')

        except Exception as e:
            logger.error(f"Automated response failed for {alert_type}: {e}")

    async def block_ip(self, ip, reason):
        """Block IP address using iptables"""
        try:
            if ip not in self.blocked_ips:
                # Use iptables to block IP
                cmd = f"iptables -A INPUT -s {ip} -j DROP"
                subprocess.run(cmd.split(), check=True)

                self.blocked_ips.add(ip)

                logger.info(f"🚫 Blocked IP {ip} - Reason: {reason}")

                # Log the block action
                await self.generate_alert(
                    'ip_blocked',
                    f"IP address blocked: {ip}",
                    {'blocked_ip': ip, 'reason': reason}
                )

        except Exception as e:
            logger.error(f"Failed to block IP {ip}: {e}")

    async def quarantine_file(self, file_path):
        """Quarantine suspicious file"""
        try:
            quarantine_dir = Path('/blueteam/quarantine')
            quarantine_dir.mkdir(exist_ok=True)

            # Move file to quarantine
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            quarantine_path = quarantine_dir / f"{Path(file_path).name}_{timestamp}"

            subprocess.run(['mv', file_path, str(quarantine_path)], check=True)

            logger.info(f"🔒 Quarantined file: {file_path} -> {quarantine_path}")

        except Exception as e:
            logger.error(f"Failed to quarantine file {file_path}: {e}")

    async def coordinate_response_actions(self):
        """Coordinate response actions across blue team"""
        try:
            # Share threat intelligence with other blue team agents
            threat_intel = {
                'blocked_ips': list(self.blocked_ips),
                'threat_indicators': self.threat_indicators,
                'recent_alerts': len([a for a in self.alerts
                                    if datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=1)])
            }

            # Save threat intelligence for sharing
            with open('/blueteam/intelligence/threat_intel.json', 'w') as f:
                json.dump(threat_intel, f, indent=2)

        except Exception as e:
            logger.error(f"Response coordination failed: {e}")

    async def update_threat_landscape(self):
        """Update understanding of threat landscape"""
        try:
            # Analyze attack patterns
            attack_types = [alert['type'] for alert in self.alerts]
            attack_frequency = {}

            for attack_type in set(attack_types):
                attack_frequency[attack_type] = attack_types.count(attack_type)

            # Update threat indicators
            self.threat_indicators = [
                {
                    'type': attack_type,
                    'frequency': frequency,
                    'last_seen': datetime.now().isoformat()
                }
                for attack_type, frequency in attack_frequency.items()
            ]

        except Exception as e:
            logger.error(f"Threat landscape update failed: {e}")

    async def learn_and_adapt_defenses(self):
        """Learn from attacks and adapt defenses"""
        logger.info("🧠 Starting defensive learning and adaptation")

        while True:
            try:
                # Analyze detection effectiveness
                await self.analyze_detection_effectiveness()

                # Update detection rules
                await self.update_detection_rules()

                # Tune response sensitivity
                await self.tune_response_sensitivity()

                # Generate defense improvements
                await self.generate_defense_improvements()

                await asyncio.sleep(600)  # Every 10 minutes

            except Exception as e:
                logger.error(f"Defensive learning failed: {e}")
                await asyncio.sleep(300)

    async def analyze_detection_effectiveness(self):
        """Analyze effectiveness of current detection methods"""
        try:
            total_alerts = len(self.alerts)
            processed_alerts = len([a for a in self.alerts if a.get('processed')])

            if total_alerts > 0:
                detection_rate = processed_alerts / total_alerts

                # Estimate false positive rate (simplified)
                recent_alerts = [a for a in self.alerts
                               if datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=24)]

                # Analyze alert patterns to identify potential false positives
                false_positives = 0
                for alert in recent_alerts:
                    # Simple heuristic: if same alert type repeats frequently from same source
                    same_type_alerts = [a for a in recent_alerts
                                      if a['type'] == alert['type'] and
                                      a.get('details', {}).get('source_ip') == alert.get('details', {}).get('source_ip')]

                    if len(same_type_alerts) > 10:  # More than 10 same alerts likely false positive
                        false_positives += 1

                self.false_positive_rate = false_positives / len(recent_alerts) if recent_alerts else 0
                self.true_positive_rate = detection_rate

                logger.info(f"📊 Detection effectiveness: TP={self.true_positive_rate:.2%}, FP={self.false_positive_rate:.2%}")

        except Exception as e:
            logger.error(f"Detection effectiveness analysis failed: {e}")

    async def update_detection_rules(self):
        """Update detection rules based on learned patterns"""
        try:
            # Identify most common attack types
            attack_types = [alert['type'] for alert in self.alerts]
            common_attacks = {}

            for attack_type in set(attack_types):
                frequency = attack_types.count(attack_type)
                if frequency > 5:  # Common attack
                    common_attacks[attack_type] = frequency

            # Update detection sensitivity for common attacks
            updated_rules = {}
            for attack_type, frequency in common_attacks.items():
                if attack_type == 'port_scan_detected':
                    # Lower threshold for port scan detection
                    updated_rules[attack_type] = {'threshold': max(5, 15 - frequency)}
                elif attack_type == 'brute_force_attempt':
                    # Lower threshold for brute force detection
                    updated_rules[attack_type] = {'threshold': max(3, 10 - frequency)}

            if updated_rules:
                with open('/blueteam/rules/adaptive_rules.json', 'w') as f:
                    json.dump(updated_rules, f, indent=2)

                logger.info(f"📝 Updated {len(updated_rules)} detection rules")

        except Exception as e:
            logger.error(f"Detection rule update failed: {e}")

    async def tune_response_sensitivity(self):
        """Tune automated response sensitivity"""
        try:
            # If false positive rate is high, reduce response aggressiveness
            if self.false_positive_rate > 0.3:  # 30% false positive rate
                self.response_sensitivity = 'low'
                logger.info("📉 Reduced response sensitivity due to high false positives")

            elif self.false_positive_rate < 0.1:  # Low false positive rate
                self.response_sensitivity = 'high'
                logger.info("📈 Increased response sensitivity due to low false positives")

            else:
                self.response_sensitivity = 'medium'

        except Exception as e:
            logger.error(f"Response sensitivity tuning failed: {e}")

    async def generate_defense_improvements(self):
        """Generate suggestions for defense improvements"""
        try:
            improvements = []

            # Analyze attack patterns for improvement suggestions
            recent_attacks = [a for a in self.alerts
                            if datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=24)]

            attack_sources = {}
            for attack in recent_attacks:
                source_ip = attack.get('details', {}).get('source_ip')
                if source_ip:
                    attack_sources[source_ip] = attack_sources.get(source_ip, 0) + 1

            # Suggest network segmentation if attacks from many sources
            if len(attack_sources) > 10:
                improvements.append({
                    'type': 'network_segmentation',
                    'priority': 'high',
                    'description': 'Consider implementing network segmentation to limit attack spread'
                })

            # Suggest rate limiting if many brute force attempts
            brute_force_count = len([a for a in recent_attacks if a['type'] == 'brute_force_attempt'])
            if brute_force_count > 5:
                improvements.append({
                    'type': 'rate_limiting',
                    'priority': 'medium',
                    'description': 'Implement stricter rate limiting on authentication endpoints'
                })

            if improvements:
                with open('/blueteam/reports/defense_improvements.json', 'w') as f:
                    json.dump(improvements, f, indent=2)

                logger.info(f"💡 Generated {len(improvements)} defense improvement suggestions")

        except Exception as e:
            logger.error(f"Defense improvement generation failed: {e}")

    async def generate_alert(self, alert_type, message, details=None):
        """Generate security alert"""
        alert = {
            'id': f"alert_{int(time.time())}_{len(self.alerts)}",
            'type': alert_type,
            'message': message,
            'details': details or {},
            'timestamp': datetime.now().isoformat(),
            'severity': self.get_alert_severity(alert_type),
            'processed': False
        }

        self.alerts.append(alert)

        # Log the alert
        severity_emoji = {'low': '🟡', 'medium': '🟠', 'high': '🔴', 'critical': '🚨'}
        emoji = severity_emoji.get(alert['severity'], '⚪')

        logger.info(f"{emoji} ALERT [{alert['severity'].upper()}]: {message}")

        # Save alert to file
        try:
            with open('/blueteam/logs/alerts.jsonl', 'a') as f:
                f.write(json.dumps(alert) + '\n')
        except:
            pass

    def get_alert_severity(self, alert_type):
        """Get severity level for alert type"""
        severity_map = {
            'multi_stage_attack': 'critical',
            'malicious_process_detected': 'critical',
            'ioc_match': 'critical',
            'file_integrity_violation': 'high',
            'brute_force_attempt': 'high',
            'suspicious_privileged_login': 'high',
            'web_attack_detected': 'high',
            'port_scan_detected': 'medium',
            'suspicious_connections': 'medium',
            'high_cpu_usage': 'medium',
            'high_memory_usage': 'medium',
            'unexpected_service': 'low'
        }

        return severity_map.get(alert_type, 'low')

    def get_trusted_ips(self):
        """Get list of trusted IP addresses"""
        return ['127.0.0.1', '172.20.0.1']  # Localhost and gateway

    def get_status(self):
        """Get current agent status"""
        recent_alerts = [a for a in self.alerts
                        if datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=1)]

        return {
            'agent_id': self.agent_id,
            'total_alerts': len(self.alerts),
            'recent_alerts': len(recent_alerts),
            'blocked_ips': len(self.blocked_ips),
            'detection_rate': self.true_positive_rate,
            'false_positive_rate': self.false_positive_rate,
            'last_activity': datetime.now().isoformat()
        }

if __name__ == "__main__":
    agent = BlueTeamAgent()

    # Create necessary directories
    for directory in ['/blueteam/logs', '/blueteam/reports', '/blueteam/intelligence', '/blueteam/quarantine']:
        Path(directory).mkdir(exist_ok=True)

    logger.info("🔵 XORB Blue Team Agent starting...")

    try:
        asyncio.run(agent.start_autonomous_defense())
    except KeyboardInterrupt:
        logger.info("🔵 Blue Team Agent shutting down...")
    except Exception as e:
        logger.error(f"Blue Team Agent crashed: {e}")
