"""
Xorb Extended Detection & Response (XDR) Layer
Module: Endpoint Detection (eBPF-based)
"""

import bcc
import json
import datetime
from typing import Dict, List, Any
import logging
import os
import sys
import socket
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/var/log/xorb/xdr_endpoint.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("XorbXDR.EndpointDetection")

class EndpointDetector:
    """
    eBPF-based endpoint detection system for monitoring system calls
    and identifying suspicious behavior patterns.
    """

    def __init__(self):
        self.bpf_program = self._load_bpf_program()
        self.system_info = self._collect_system_info()
        self.suspicious_patterns = self._load_suspicious_patterns()
        self.alerts = []

    def _load_bpf_program(self) -> bcc.BPF:
        """
        Load eBPF program for monitoring system calls
        """
        try:
            bpf_code = """
            #include <uapi/linux/ptrace.h>
            #include <uapi/linux/limits.h>

            struct event_t {
                u64 pid;
                u64 tgid;
                char comm[TASK_COMM_LEN];
                int syscall_id;
                u64 timestamp;
            };

            BPF_PERF_OUTPUT(events);

            int syscall__execve(struct pt_regs *ctx, const char __user *filenamep)
            {
                struct event_t event = {};
                event.pid = bpf_get_current_pid_tgid();
                event.tgid = event.pid >> 32;
                event.syscall_id = 59; // execve syscall number
                event.timestamp = bpf_ktime_get_ns();
                bpf_get_current_comm(&event.comm, sizeof(event.comm));
                events.perf_submit(ctx, &event, sizeof(event));
                return 0;
            }
            """

            bpf = bcc.BPF(text=bpf_code)
            bpf.attach_kprobe(event="sys_execve", fn_name="syscall__execve")
            return bpf

        except Exception as e:
            logger.error(f"Failed to load eBPF program: {str(e)}")
            raise

    def _collect_system_info(self) -> Dict[str, Any]:
        """
        Collect system information for context-aware detection
        """
        try:
            return {
                "hostname": socket.gethostname(),
                "kernel_version": os.uname().release,
                "os": os.uname().sysname,
                "os_version": os.uname().version,
                "cpu_count": os.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "boot_time": psutil.boot_time(),
                "processes": len(psutil.pids())
            }
        except Exception as e:
            logger.error(f"Failed to collect system info: {str(e)}")
            return {}

    def _load_suspicious_patterns(self) -> Dict[str, Any]:
        """
        Load suspicious behavior patterns from configuration
        """
        try:
            with open("/etc/xorb/xdr/suspicious_patterns.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Suspicious patterns file not found, using defaults")
            return {
                "suspicious_commands": ["/bin/sh", "/bin/bash", "/usr/bin/perl", "/usr/bin/python", "/usr/bin/ruby", "/usr/bin/nc", "/usr/bin/netcat", "/usr/bin/wget", "/usr/bin/curl"],
                "suspicious_directories": ["/tmp", "/dev/shm", "/var/tmp", "/run/shm"],
                "suspicious_users": ["root", "www-data", "nobody"],
                "high_risk_processes": ["sshd", "sudo", "su", "named", "httpd", "mysqld"]
            }
        except Exception as e:
            logger.error(f"Failed to load suspicious patterns: {str(e)}")
            return {}

    def _process_event(self, cpu, data, size):
        """
        Process eBPF events and detect suspicious behavior
        """
        try:
            event = self.bpf_program["events"].event(data)

            # Convert event data to dictionary
            event_data = {
                "pid": event.pid,
                "tgid": event.tgid,
                "comm": event.comm.decode('utf-8', 'replace'),
                "syscall_id": event.syscall_id,
                "timestamp": event.timestamp,
                "timestamp_human": datetime.datetime.fromtimestamp(event.timestamp / 1e9).isoformat()
            }

            # Check for suspicious patterns
            alert = self._check_suspicious_patterns(event_data)

            if alert:
                self.alerts.append(alert)
                logger.info(f"Suspicious activity detected: {json.dumps(alert)}")
                self._generate_alert(alert)

            return event_data

        except Exception as e:
            logger.error(f"Failed to process event: {str(e)}")
            return None

    def _check_suspicious_patterns(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check event against suspicious patterns
        """
        try:
            # Check for suspicious commands
            if event["comm"] in self.suspicious_patterns.get("suspicious_commands", []):
                return {
                    "type": "suspicious_command_execution",
                    "description": f"Suspicious command executed: {event['comm']}",
                    "severity": 8,
                    "event": event
                }

            # Check for suspicious process creation in temporary directories
            if any(event["comm"].startswith(dir) for dir in self.suspicious_patterns.get("suspicious_directories", [])):
                return {
                    "type": "suspicious_process_in_temp_dir",
                    "description": f"Process created in suspicious directory: {event['comm']}",
                    "severity": 7,
                    "event": event
                }

            # Check for suspicious process names
            if event["comm"] in self.suspicious_patterns.get("high_risk_processes", []):
                return {
                    "type": "high_risk_process_creation",
                    "description": f"High risk process created: {event['comm']}",
                    "severity": 6,
                    "event": event
                }

            return None

        except Exception as e:
            logger.error(f"Failed to check suspicious patterns: {str(e)}")
            return None

    def _generate_alert(self, alert: Dict[str, Any]) -> None:
        """
        Generate and send alert for suspicious activity
        """
        try:
            # Add system info to alert
            alert["system_info"] = self.system_info

            # Add timestamp
            alert["alert_timestamp"] = datetime.datetime.now().isoformat()

            # Send alert to SIEM
            self._send_to_siem(alert)

            # Execute response playbook
            self._execute_response_playbook(alert)

        except Exception as e:
            logger.error(f"Failed to generate alert: {str(e)}")

    def _send_to_siem(self, alert: Dict[str, Any]) -> None:
        """
        Send alert to SIEM system
        """
        # Implementation would integrate with SIEM system
        logger.debug(f"Sending alert to SIEM: {json.dumps(alert)}")

    def _execute_response_playbook(self, alert: Dict[str, Any]) -> None:
        """
        Execute response playbook based on alert type
        """
        try:
            playbook_path = f"/etc/xorb/xdr/playbooks/{alert['type']}.py"
            if os.path.exists(playbook_path):
                logger.info(f"Executing response playbook: {playbook_path}")
                # Implementation would execute the playbook
            else:
                logger.warning(f"No response playbook found for alert type: {alert['type']}")
        except Exception as e:
            logger.error(f"Failed to execute response playbook: {str(e)}")

    def start_monitoring(self) -> None:
        """
        Start monitoring system calls
        """
        try:
            logger.info("Starting endpoint monitoring...")
            self.bpf_program["events"].open_perf_buffer(self._process_event)

            while True:
                try:
                    self.bpf_program.perf_buffer_poll()
                except KeyboardInterrupt:
                    break

        except Exception as e:
            logger.error(f"Monitoring failed: {str(e)}")
            raise

    def get_alerts(self) -> List[Dict[str, Any]]:
        """
        Get list of detected alerts
        """
        return self.alerts

if __name__ == "__main__":
    detector = EndpointDetector()
    detector.start_monitoring()
