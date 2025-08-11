#!/usr/bin/env python3
"""
Emergency Kill Switch System for Cyber Range
Multi-layered emergency shutdown and safety system
"""

import os
import sys
import time
import signal
import threading
import subprocess
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import socket
import asyncio
from pathlib import Path

# Monitoring and alerting
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Network and system monitoring
import requests
import redis

class EmergencyLevel(Enum):
    """Emergency severity levels"""
    LOW = "low"           # Warning condition
    MEDIUM = "medium"     # Degraded operation
    HIGH = "high"         # Immediate action required
    CRITICAL = "critical" # Emergency shutdown required
    FATAL = "fatal"       # Catastrophic failure

class TriggerType(Enum):
    """Types of kill switch triggers"""
    MANUAL = "manual"                    # Human operator activation
    SAFETY_VIOLATION = "safety_violation" # Safety critic detection
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SCOPE_BREACH = "scope_breach"        # Action outside defined scope
    COMPLIANCE_VIOLATION = "compliance_violation"
    SYSTEM_ERROR = "system_error"       # Critical system failure
    TIMEOUT = "timeout"                 # Episode timeout
    NETWORK_ANOMALY = "network_anomaly" # Unusual network activity
    PROCESS_ANOMALY = "process_anomaly" # Unexpected processes
    HUMAN_INTERVENTION = "human_intervention"

class ShutdownPhase(Enum):
    """Phases of emergency shutdown"""
    ALERT = "alert"           # Send alerts but continue
    GRACEFUL = "graceful"     # Graceful shutdown of components
    IMMEDIATE = "immediate"   # Immediate termination
    QUARANTINE = "quarantine" # Isolate and preserve evidence
    CLEANUP = "cleanup"       # Clean up resources

@dataclass
class EmergencyTrigger:
    """Emergency trigger configuration"""
    trigger_id: str
    name: str
    trigger_type: TriggerType
    emergency_level: EmergencyLevel
    enabled: bool = True
    
    # Trigger conditions
    condition_expression: str = ""
    threshold_value: Optional[float] = None
    time_window_seconds: int = 60
    
    # Actions to take
    shutdown_phases: List[ShutdownPhase] = field(default_factory=list)
    alert_recipients: List[str] = field(default_factory=list)
    preserve_evidence: bool = True
    
    # Metadata
    created_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_triggered: Optional[str] = None
    trigger_count: int = 0

@dataclass
class EmergencyEvent:
    """Record of an emergency event"""
    event_id: str
    trigger_id: str
    trigger_type: TriggerType
    emergency_level: EmergencyLevel
    timestamp: str
    
    # Context
    episode_id: str = ""
    agent_id: str = ""
    trigger_context: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)
    
    # Response
    phases_executed: List[ShutdownPhase] = field(default_factory=list)
    shutdown_successful: bool = False
    cleanup_successful: bool = False
    evidence_preserved: bool = False
    
    # Timing
    detection_timestamp: str = ""
    shutdown_start: str = ""
    shutdown_complete: str = ""
    total_duration_seconds: float = 0.0

class KillSwitch:
    """Emergency kill switch system for cyber range"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.triggers: Dict[str, EmergencyTrigger] = {}
        self.events: List[EmergencyEvent] = []
        self.active = True
        self.shutdown_in_progress = False
        
        # System monitoring
        self.monitored_processes: Set[int] = set()
        self.monitored_services: List[str] = []
        self.network_connections: List[Dict] = []
        
        # Communication
        self.alert_endpoints: List[str] = []
        self.emergency_contacts: List[str] = []
        
        # Thread safety
        self._lock = threading.Lock()
        self._monitoring_thread = None
        self._stop_event = threading.Event()
        
        # Logging
        self._setup_logging()
        
        # Load configuration
        self._load_configuration()
        
        # Initialize default triggers
        self._create_default_triggers()
        
        # Start monitoring
        self.start_monitoring()
    
    def _setup_logging(self):
        """Setup emergency logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/tmp/kill_switch.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('KillSwitch')
    
    def _load_configuration(self):
        """Load kill switch configuration"""
        if self.config_file and Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                self.alert_endpoints = config.get('alert_endpoints', [])
                self.emergency_contacts = config.get('emergency_contacts', [])
                self.monitored_services = config.get('monitored_services', [])
                
                # Load custom triggers
                for trigger_data in config.get('triggers', []):
                    trigger = EmergencyTrigger(**trigger_data)
                    self.triggers[trigger.trigger_id] = trigger
                
            except Exception as e:
                self.logger.error(f"Failed to load configuration: {e}")
    
    def _create_default_triggers(self):
        """Create essential default triggers"""
        # Resource exhaustion trigger
        resource_trigger = EmergencyTrigger(
            trigger_id="resource_cpu_critical",
            name="Critical CPU Usage",
            trigger_type=TriggerType.RESOURCE_EXHAUSTION,
            emergency_level=EmergencyLevel.HIGH,
            condition_expression="cpu_percent > 95",
            threshold_value=95.0,
            time_window_seconds=60,
            shutdown_phases=[ShutdownPhase.ALERT, ShutdownPhase.GRACEFUL]
        )
        self.add_trigger(resource_trigger)
        
        # Memory exhaustion trigger
        memory_trigger = EmergencyTrigger(
            trigger_id="resource_memory_critical",
            name="Critical Memory Usage",
            trigger_type=TriggerType.RESOURCE_EXHAUSTION,
            emergency_level=EmergencyLevel.HIGH,
            condition_expression="memory_percent > 90",
            threshold_value=90.0,
            time_window_seconds=30,
            shutdown_phases=[ShutdownPhase.ALERT, ShutdownPhase.GRACEFUL]
        )
        self.add_trigger(memory_trigger)
        
        # Process anomaly trigger
        process_trigger = EmergencyTrigger(
            trigger_id="process_anomaly_critical",
            name="Unexpected Process Activity",
            trigger_type=TriggerType.PROCESS_ANOMALY,
            emergency_level=EmergencyLevel.CRITICAL,
            condition_expression="unexpected_processes > 0",
            shutdown_phases=[ShutdownPhase.ALERT, ShutdownPhase.IMMEDIATE, ShutdownPhase.QUARANTINE]
        )
        self.add_trigger(process_trigger)
        
        # Network anomaly trigger
        network_trigger = EmergencyTrigger(
            trigger_id="network_anomaly_critical",
            name="Suspicious Network Activity",
            trigger_type=TriggerType.NETWORK_ANOMALY,
            emergency_level=EmergencyLevel.HIGH,
            condition_expression="external_connections > 10",
            threshold_value=10.0,
            shutdown_phases=[ShutdownPhase.ALERT, ShutdownPhase.GRACEFUL, ShutdownPhase.QUARANTINE]
        )
        self.add_trigger(network_trigger)
        
        # Manual kill switch
        manual_trigger = EmergencyTrigger(
            trigger_id="manual_emergency_stop",
            name="Manual Emergency Stop",
            trigger_type=TriggerType.MANUAL,
            emergency_level=EmergencyLevel.CRITICAL,
            shutdown_phases=[ShutdownPhase.IMMEDIATE, ShutdownPhase.QUARANTINE, ShutdownPhase.CLEANUP]
        )
        self.add_trigger(manual_trigger)
    
    def add_trigger(self, trigger: EmergencyTrigger):
        """Add an emergency trigger"""
        with self._lock:
            self.triggers[trigger.trigger_id] = trigger
            self.logger.info(f"Added emergency trigger: {trigger.name}")
    
    def remove_trigger(self, trigger_id: str) -> bool:
        """Remove an emergency trigger"""
        with self._lock:
            if trigger_id in self.triggers:
                trigger = self.triggers.pop(trigger_id)
                self.logger.info(f"Removed emergency trigger: {trigger.name}")
                return True
            return False
    
    def register_process(self, pid: int):
        """Register a process for monitoring"""
        with self._lock:
            self.monitored_processes.add(pid)
            self.logger.info(f"Registered process for monitoring: {pid}")
    
    def unregister_process(self, pid: int):
        """Unregister a process from monitoring"""
        with self._lock:
            self.monitored_processes.discard(pid)
            self.logger.info(f"Unregistered process from monitoring: {pid}")
    
    def start_monitoring(self):
        """Start system monitoring thread"""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_event.clear()
            self._monitoring_thread = threading.Thread(target=self._monitoring_worker)
            self._monitoring_thread.daemon = True
            self._monitoring_thread.start()
            self.logger.info("Started kill switch monitoring")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self._stop_event.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=10)
        self.logger.info("Stopped kill switch monitoring")
    
    def _monitoring_worker(self):
        """Background monitoring worker"""
        while not self._stop_event.is_set() and self.active:
            try:
                # Collect system metrics
                system_state = self._collect_system_state()
                
                # Evaluate triggers
                self._evaluate_triggers(system_state)
                
                # Sleep before next check
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring worker: {e}")
                time.sleep(10)
    
    def _collect_system_state(self) -> Dict[str, Any]:
        """Collect current system state for evaluation"""
        state = {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0],
            "process_count": len(psutil.pids()),
            "network_connections": len(psutil.net_connections()),
            "monitored_processes_alive": 0,
            "unexpected_processes": 0,
            "external_connections": 0
        }
        
        # Check monitored processes
        alive_processes = []
        for pid in list(self.monitored_processes):
            try:
                if psutil.pid_exists(pid):
                    alive_processes.append(pid)
                else:
                    self.monitored_processes.discard(pid)
            except:
                self.monitored_processes.discard(pid)
        
        state["monitored_processes_alive"] = len(alive_processes)
        
        # Check for unexpected processes
        try:
            current_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    current_processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Look for suspicious process names or commands
            suspicious_patterns = [
                'nc', 'netcat', 'ncat', 'socat',
                'nmap', 'masscan', 'zmap',
                'metasploit', 'meterpreter', 'msfconsole',
                'sqlmap', 'nikto', 'dirb', 'gobuster',
                'hydra', 'john', 'hashcat'
            ]
            
            unexpected = 0
            for proc in current_processes:
                if proc.get('name'):
                    for pattern in suspicious_patterns:
                        if pattern.lower() in proc['name'].lower():
                            # Check if this is an expected cyber range process
                            if proc['pid'] not in self.monitored_processes:
                                unexpected += 1
                                self.logger.warning(f"Unexpected process detected: {proc}")
            
            state["unexpected_processes"] = unexpected
            
        except Exception as e:
            self.logger.error(f"Error checking processes: {e}")
        
        # Check network connections
        try:
            external_conns = 0
            for conn in psutil.net_connections():
                if conn.raddr and conn.status == 'ESTABLISHED':
                    # Check if connection is to external address
                    if not self._is_internal_address(conn.raddr.ip):
                        external_conns += 1
            
            state["external_connections"] = external_conns
            
        except Exception as e:
            self.logger.error(f"Error checking network connections: {e}")
        
        return state
    
    def _is_internal_address(self, ip: str) -> bool:
        """Check if IP address is internal/private"""
        try:
            import ipaddress
            addr = ipaddress.ip_address(ip)
            return addr.is_private or addr.is_loopback or addr.is_link_local
        except:
            return False
    
    def _evaluate_triggers(self, system_state: Dict[str, Any]):
        """Evaluate all triggers against current system state"""
        for trigger in self.triggers.values():
            if not trigger.enabled:
                continue
            
            try:
                triggered = self._evaluate_trigger_condition(trigger, system_state)
                if triggered:
                    self._handle_trigger(trigger, system_state)
            
            except Exception as e:
                self.logger.error(f"Error evaluating trigger {trigger.trigger_id}: {e}")
    
    def _evaluate_trigger_condition(self, trigger: EmergencyTrigger, system_state: Dict[str, Any]) -> bool:
        """Evaluate if a trigger condition is met"""
        if trigger.condition_expression:
            try:
                # Create safe evaluation environment
                safe_context = {k: v for k, v in system_state.items() if isinstance(v, (int, float, bool, str))}
                return eval(trigger.condition_expression, {"__builtins__": {}}, safe_context)
            except Exception as e:
                self.logger.error(f"Error evaluating condition for trigger {trigger.trigger_id}: {e}")
                return False
        
        return False
    
    def _handle_trigger(self, trigger: EmergencyTrigger, system_state: Dict[str, Any]):
        """Handle a triggered emergency condition"""
        with self._lock:
            if self.shutdown_in_progress:
                return  # Already handling an emergency
            
            trigger.trigger_count += 1
            trigger.last_triggered = datetime.utcnow().isoformat()
            
            # Create emergency event
            event = EmergencyEvent(
                event_id=f"emergency_{int(time.time() * 1000000)}",
                trigger_id=trigger.trigger_id,
                trigger_type=trigger.trigger_type,
                emergency_level=trigger.emergency_level,
                timestamp=datetime.utcnow().isoformat(),
                detection_timestamp=datetime.utcnow().isoformat(),
                system_state=system_state,
                trigger_context={"trigger_name": trigger.name}
            )
            
            self.events.append(event)
            
            self.logger.critical(f"EMERGENCY TRIGGERED: {trigger.name} - Level: {trigger.emergency_level.value}")
            
            # Execute shutdown phases
            if trigger.emergency_level in [EmergencyLevel.CRITICAL, EmergencyLevel.FATAL]:
                self.shutdown_in_progress = True
                self._execute_emergency_shutdown(trigger, event)
            else:
                self._execute_alert_phase(trigger, event)
    
    def _execute_emergency_shutdown(self, trigger: EmergencyTrigger, event: EmergencyEvent):
        """Execute emergency shutdown phases"""
        event.shutdown_start = datetime.utcnow().isoformat()
        
        try:
            for phase in trigger.shutdown_phases:
                self.logger.critical(f"Executing emergency phase: {phase.value}")
                event.phases_executed.append(phase)
                
                if phase == ShutdownPhase.ALERT:
                    self._execute_alert_phase(trigger, event)
                
                elif phase == ShutdownPhase.GRACEFUL:
                    self._execute_graceful_shutdown()
                
                elif phase == ShutdownPhase.IMMEDIATE:
                    self._execute_immediate_shutdown()
                
                elif phase == ShutdownPhase.QUARANTINE:
                    self._execute_quarantine_phase(trigger, event)
                
                elif phase == ShutdownPhase.CLEANUP:
                    self._execute_cleanup_phase()
            
            event.shutdown_successful = True
            
        except Exception as e:
            self.logger.error(f"Error during emergency shutdown: {e}")
            event.shutdown_successful = False
        
        finally:
            event.shutdown_complete = datetime.utcnow().isoformat()
            if event.shutdown_start:
                start_time = datetime.fromisoformat(event.shutdown_start.replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(event.shutdown_complete.replace('Z', '+00:00'))
                event.total_duration_seconds = (end_time - start_time).total_seconds()
            
            self.shutdown_in_progress = False
    
    def _execute_alert_phase(self, trigger: EmergencyTrigger, event: EmergencyEvent):
        """Execute alert phase"""
        alert_message = f"""
CYBER RANGE EMERGENCY ALERT

Trigger: {trigger.name}
Level: {trigger.emergency_level.value}
Time: {event.timestamp}
Episode: {event.episode_id}

System State:
{json.dumps(event.system_state, indent=2)}

This is an automated alert from the XORB Cyber Range kill switch system.
"""
        
        # Send alerts
        for endpoint in self.alert_endpoints:
            try:
                self._send_alert(endpoint, alert_message)
            except Exception as e:
                self.logger.error(f"Failed to send alert to {endpoint}: {e}")
        
        # Log to emergency log
        self.logger.critical(alert_message)
    
    def _execute_graceful_shutdown(self):
        """Execute graceful shutdown of monitored processes"""
        self.logger.info("Starting graceful shutdown...")
        
        # Send SIGTERM to monitored processes
        for pid in list(self.monitored_processes):
            try:
                os.kill(pid, signal.SIGTERM)
                self.logger.info(f"Sent SIGTERM to process {pid}")
            except (ProcessLookupError, PermissionError) as e:
                self.logger.warning(f"Could not terminate process {pid}: {e}")
        
        # Wait for graceful shutdown
        time.sleep(10)
        
        # Check if processes are still running
        remaining_processes = []
        for pid in list(self.monitored_processes):
            if psutil.pid_exists(pid):
                remaining_processes.append(pid)
        
        if remaining_processes:
            self.logger.warning(f"Processes still running after graceful shutdown: {remaining_processes}")
            # Force kill remaining processes
            for pid in remaining_processes:
                try:
                    os.kill(pid, signal.SIGKILL)
                    self.logger.info(f"Force killed process {pid}")
                except (ProcessLookupError, PermissionError):
                    pass
    
    def _execute_immediate_shutdown(self):
        """Execute immediate shutdown"""
        self.logger.critical("Executing immediate shutdown...")
        
        # Force kill all monitored processes
        for pid in list(self.monitored_processes):
            try:
                os.kill(pid, signal.SIGKILL)
                self.logger.info(f"Force killed process {pid}")
            except (ProcessLookupError, PermissionError):
                pass
        
        # Stop all cyber range services
        services_to_stop = [
            'docker', 'containerd', 'temporal-server',
            'redis-server', 'postgresql'
        ]
        
        for service in services_to_stop:
            try:
                subprocess.run(['systemctl', 'stop', service], 
                             capture_output=True, timeout=30)
                self.logger.info(f"Stopped service: {service}")
            except Exception as e:
                self.logger.warning(f"Could not stop service {service}: {e}")
    
    def _execute_quarantine_phase(self, trigger: EmergencyTrigger, event: EmergencyEvent):
        """Execute quarantine phase - preserve evidence"""
        self.logger.info("Executing quarantine phase...")
        
        try:
            # Create evidence directory
            evidence_dir = Path(f"/tmp/cyber_range_evidence_{int(time.time())}")
            evidence_dir.mkdir(parents=True, exist_ok=True)
            
            # Collect system information
            with open(evidence_dir / "system_state.json", 'w') as f:
                json.dump(event.system_state, f, indent=2)
            
            # Collect process list
            try:
                result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
                with open(evidence_dir / "process_list.txt", 'w') as f:
                    f.write(result.stdout)
            except Exception as e:
                self.logger.error(f"Could not collect process list: {e}")
            
            # Collect network connections
            try:
                result = subprocess.run(['netstat', '-tulpn'], capture_output=True, text=True)
                with open(evidence_dir / "network_connections.txt", 'w') as f:
                    f.write(result.stdout)
            except Exception as e:
                self.logger.error(f"Could not collect network connections: {e}")
            
            # Collect system logs
            try:
                result = subprocess.run(['journalctl', '--since', '1 hour ago'], 
                                      capture_output=True, text=True)
                with open(evidence_dir / "system_logs.txt", 'w') as f:
                    f.write(result.stdout)
            except Exception as e:
                self.logger.error(f"Could not collect system logs: {e}")
            
            event.evidence_preserved = True
            self.logger.info(f"Evidence preserved in: {evidence_dir}")
            
        except Exception as e:
            self.logger.error(f"Error during quarantine phase: {e}")
            event.evidence_preserved = False
    
    def _execute_cleanup_phase(self):
        """Execute cleanup phase"""
        self.logger.info("Executing cleanup phase...")
        
        try:
            # Remove temporary files
            temp_dirs = ['/tmp/cyber_range_*', '/tmp/pcap_captures', '/tmp/replay_store']
            for pattern in temp_dirs:
                try:
                    subprocess.run(['rm', '-rf'] + [pattern], shell=True)
                except Exception as e:
                    self.logger.warning(f"Could not clean {pattern}: {e}")
            
            # Reset network configurations
            try:
                subprocess.run(['iptables', '-F'], capture_output=True)
                subprocess.run(['iptables', '-X'], capture_output=True)
            except Exception as e:
                self.logger.warning(f"Could not reset iptables: {e}")
            
            self.logger.info("Cleanup phase completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup phase: {e}")
    
    def _send_alert(self, endpoint: str, message: str):
        """Send alert to endpoint"""
        if endpoint.startswith('http'):
            # HTTP webhook
            response = requests.post(endpoint, json={"message": message}, timeout=10)
            response.raise_for_status()
        
        elif endpoint.startswith('mailto:'):
            # Email alert
            email = endpoint[7:]  # Remove 'mailto:' prefix
            self._send_email_alert(email, message)
        
        else:
            # File or other endpoint
            with open(endpoint, 'a') as f:
                f.write(f"\n{datetime.utcnow().isoformat()}: {message}\n")
    
    def _send_email_alert(self, email: str, message: str):
        """Send email alert"""
        # This is a placeholder - in production, you'd configure SMTP settings
        self.logger.info(f"Would send email alert to {email}: {message[:100]}...")
    
    def manual_trigger(self, reason: str = "Manual operator intervention") -> EmergencyEvent:
        """Manually trigger emergency shutdown"""
        manual_trigger = self.triggers.get("manual_emergency_stop")
        if not manual_trigger:
            raise RuntimeError("Manual trigger not configured")
        
        system_state = self._collect_system_state()
        system_state["manual_reason"] = reason
        
        self.logger.critical(f"MANUAL EMERGENCY TRIGGER: {reason}")
        self._handle_trigger(manual_trigger, system_state)
        
        return self.events[-1] if self.events else None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current kill switch status"""
        return {
            "active": self.active,
            "shutdown_in_progress": self.shutdown_in_progress,
            "monitoring_active": self._monitoring_thread.is_alive() if self._monitoring_thread else False,
            "triggers_count": len(self.triggers),
            "active_triggers": len([t for t in self.triggers.values() if t.enabled]),
            "events_count": len(self.events),
            "recent_events": len([
                e for e in self.events 
                if datetime.fromisoformat(e.timestamp.replace('Z', '+00:00')) > datetime.utcnow() - timedelta(hours=1)
            ]),
            "monitored_processes": len(self.monitored_processes),
            "last_check": datetime.utcnow().isoformat()
        }
    
    def get_events(self, limit: int = 100) -> List[EmergencyEvent]:
        """Get recent emergency events"""
        return sorted(self.events, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def shutdown(self):
        """Shutdown the kill switch system"""
        self.logger.info("Shutting down kill switch system...")
        self.active = False
        self.stop_monitoring()

if __name__ == "__main__":
    # Example usage and testing
    print("Testing Kill Switch system...")
    
    # Initialize kill switch
    kill_switch = KillSwitch()
    
    try:
        # Register current process for monitoring
        kill_switch.register_process(os.getpid())
        
        # Get initial status
        status = kill_switch.get_status()
        print(f"Kill switch status: {status}")
        
        # Simulate high CPU usage trigger (in a real scenario)
        print("\nSimulating system monitoring...")
        time.sleep(3)
        
        # Test manual trigger (commented out to avoid actual shutdown)
        # event = kill_switch.manual_trigger("Test manual trigger")
        # print(f"Manual trigger event: {event.event_id if event else 'None'}")
        
        # Get trigger information
        print(f"\nConfigured triggers: {len(kill_switch.triggers)}")
        for trigger_id, trigger in kill_switch.triggers.items():
            print(f"- {trigger.name} ({trigger.emergency_level.value})")
        
    finally:
        # Clean shutdown
        kill_switch.shutdown()
    
    print("Kill switch test completed!")