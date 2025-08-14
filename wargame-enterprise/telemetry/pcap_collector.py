#!/usr/bin/env python3
"""
Network Traffic Collection and PCAP Management
High-performance packet capture and analysis for cyber range episodes
"""

import os
import time
import signal
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import hashlib
import asyncio
import psutil

# Network analysis imports
try:
    import scapy.all as scapy
    from scapy.layers.inet import IP, TCP, UDP, ICMP
    from scapy.layers.http import HTTP, HTTPRequest, HTTPResponse
    from scapy.layers.dns import DNS, DNSQR, DNSRR
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("Warning: Scapy not available. Some packet analysis features disabled.")

try:
    import pyshark
    PYSHARK_AVAILABLE = True
except ImportError:
    PYSHARK_AVAILABLE = False
    print("Warning: PyShark not available. Some packet analysis features disabled.")

@dataclass
class NetworkInterface:
    """Network interface configuration"""
    name: str
    ip_address: str
    subnet: str
    capture_enabled: bool = True
    filter_expression: str = ""
    max_packet_size: int = 65535

@dataclass
class CaptureConfig:
    """PCAP capture configuration"""
    # Interface settings
    interfaces: List[NetworkInterface] = field(default_factory=list)
    promiscuous_mode: bool = True
    
    # Capture limits
    max_file_size_mb: int = 100
    max_capture_duration_minutes: int = 60
    rotation_interval_minutes: int = 15
    
    # Filtering
    bpf_filter: str = ""  # Berkeley Packet Filter
    exclude_internal_traffic: bool = False
    capture_only_threats: bool = False
    
    # Storage
    output_directory: str = "/tmp/pcap_captures"
    compression_enabled: bool = True
    encryption_enabled: bool = False
    
    # Analysis
    real_time_analysis: bool = True
    threat_detection_enabled: bool = True
    flow_analysis_enabled: bool = True

@dataclass
class PacketMetrics:
    """Packet-level metrics and statistics"""
    timestamp: float
    size_bytes: int
    protocol: str
    src_ip: str
    dst_ip: str
    src_port: Optional[int] = None
    dst_port: Optional[int] = None
    tcp_flags: Optional[str] = None
    payload_size: int = 0
    
    # Analysis results
    threat_score: float = 0.0
    anomaly_score: float = 0.0
    signature_matches: List[str] = field(default_factory=list)
    geoip_info: Dict[str, str] = field(default_factory=dict)

@dataclass
class NetworkFlow:
    """Network flow tracking"""
    flow_id: str
    start_timestamp: float
    end_timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    
    # Flow statistics
    packets_forward: int = 0
    packets_backward: int = 0
    bytes_forward: int = 0
    bytes_backward: int = 0
    
    # Flow characteristics
    duration_seconds: float = 0.0
    avg_packet_size: float = 0.0
    packets_per_second: float = 0.0
    
    # Security analysis
    suspicious_patterns: List[str] = field(default_factory=list)
    threat_indicators: List[str] = field(default_factory=list)
    flow_classification: str = "normal"  # normal, suspicious, malicious

class PCAPCollector:
    """High-performance PCAP collection and management"""
    
    def __init__(self, config: CaptureConfig):
        self.config = config
        self.capture_processes = {}
        self.active_captures = {}
        self.capture_stats = {}
        self.flows = {}
        self.threat_signatures = []
        self._stop_event = threading.Event()
        self._analysis_thread = None
        
        # Initialize output directory
        Path(self.config.output_directory).mkdir(parents=True, exist_ok=True)
        
        # Load threat signatures
        self._load_threat_signatures()
    
    def _load_threat_signatures(self):
        """Load threat detection signatures"""
        # Basic threat patterns for demonstration
        self.threat_signatures = [
            {
                "name": "port_scan",
                "pattern": "tcp_syn_flood",
                "description": "Port scanning activity detected",
                "severity": "medium"
            },
            {
                "name": "dns_tunneling",
                "pattern": "large_dns_queries",
                "description": "Potential DNS tunneling detected",
                "severity": "high"
            },
            {
                "name": "sql_injection",
                "pattern": "http_sql_injection",
                "description": "SQL injection attempt detected",
                "severity": "high"
            },
            {
                "name": "credential_stuffing",
                "pattern": "repeated_login_failures",
                "description": "Credential stuffing attack detected",
                "severity": "high"
            }
        ]
    
    def start_capture(self, episode_id: str, interfaces: Optional[List[str]] = None) -> Dict[str, str]:
        """Start packet capture for an episode"""
        capture_info = {}
        
        # Use configured interfaces or provided list
        target_interfaces = interfaces or [iface.name for iface in self.config.interfaces]
        
        for interface in target_interfaces:
            capture_file = self._generate_capture_filename(episode_id, interface)
            
            # Start tcpdump/dumpcap for high-performance capture
            if self._is_dumpcap_available():
                capture_cmd = self._build_dumpcap_command(interface, capture_file)
            else:
                capture_cmd = self._build_tcpdump_command(interface, capture_file)
            
            try:
                process = subprocess.Popen(
                    capture_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid  # Create new process group
                )
                
                self.capture_processes[interface] = process
                self.active_captures[interface] = {
                    "episode_id": episode_id,
                    "capture_file": capture_file,
                    "start_time": time.time(),
                    "process": process
                }
                
                capture_info[interface] = capture_file
                print(f"Started capture on interface {interface}: {capture_file}")
                
            except Exception as e:
                print(f"Failed to start capture on interface {interface}: {e}")
        
        # Start real-time analysis if enabled
        if self.config.real_time_analysis and not self._analysis_thread:
            self._analysis_thread = threading.Thread(target=self._real_time_analysis_worker)
            self._analysis_thread.daemon = True
            self._analysis_thread.start()
        
        return capture_info
    
    def stop_capture(self, episode_id: str) -> Dict[str, Dict[str, Any]]:
        """Stop packet capture and return statistics"""
        capture_results = {}
        
        for interface, capture_info in list(self.active_captures.items()):
            if capture_info["episode_id"] == episode_id:
                try:
                    # Terminate capture process
                    process = self.capture_processes.get(interface)
                    if process:
                        # Send SIGTERM to process group
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                        process.wait(timeout=10)
                    
                    # Calculate statistics
                    end_time = time.time()
                    duration = end_time - capture_info["start_time"]
                    
                    # Get file information
                    capture_file = capture_info["capture_file"]
                    file_stats = self._get_file_statistics(capture_file)
                    
                    capture_results[interface] = {
                        "capture_file": capture_file,
                        "duration_seconds": duration,
                        "file_size_bytes": file_stats["size"],
                        "packet_count": file_stats["packet_count"],
                        "checksum": file_stats["checksum"]
                    }
                    
                    # Compress if enabled
                    if self.config.compression_enabled:
                        compressed_file = self._compress_pcap_file(capture_file)
                        capture_results[interface]["compressed_file"] = compressed_file
                    
                    # Clean up
                    del self.capture_processes[interface]
                    del self.active_captures[interface]
                    
                    print(f"Stopped capture on interface {interface}")
                    
                except Exception as e:
                    print(f"Error stopping capture on interface {interface}: {e}")
        
        return capture_results
    
    def stop_all_captures(self):
        """Stop all active captures"""
        for episode_id in set(info["episode_id"] for info in self.active_captures.values()):
            self.stop_capture(episode_id)
        
        # Stop analysis thread
        self._stop_event.set()
        if self._analysis_thread:
            self._analysis_thread.join(timeout=5)
    
    def _generate_capture_filename(self, episode_id: str, interface: str) -> str:
        """Generate unique filename for capture"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{episode_id}_{interface}_{timestamp}.pcap"
        return os.path.join(self.config.output_directory, filename)
    
    def _is_dumpcap_available(self) -> bool:
        """Check if dumpcap is available (part of Wireshark)"""
        try:
            subprocess.run(["dumpcap", "--version"], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _build_dumpcap_command(self, interface: str, output_file: str) -> List[str]:
        """Build dumpcap command for high-performance capture"""
        cmd = [
            "dumpcap",
            "-i", interface,
            "-w", output_file,
            "-q",  # Quiet mode
        ]
        
        # Add promiscuous mode
        if self.config.promiscuous_mode:
            cmd.extend(["-p"])
        
        # Add capture filter
        if self.config.bpf_filter:
            cmd.extend(["-f", self.config.bpf_filter])
        
        # Add file size limit
        if self.config.max_file_size_mb > 0:
            cmd.extend(["-b", f"filesize:{self.config.max_file_size_mb * 1024}"])
        
        # Add duration limit
        if self.config.max_capture_duration_minutes > 0:
            cmd.extend(["-a", f"duration:{self.config.max_capture_duration_minutes * 60}"])
        
        return cmd
    
    def _build_tcpdump_command(self, interface: str, output_file: str) -> List[str]:
        """Build tcpdump command as fallback"""
        cmd = [
            "tcpdump",
            "-i", interface,
            "-w", output_file,
            "-q",  # Quiet mode
        ]
        
        # Add promiscuous mode
        if not self.config.promiscuous_mode:
            cmd.append("-p")
        
        # Add capture filter
        if self.config.bpf_filter:
            cmd.append(self.config.bpf_filter)
        
        return cmd
    
    def _get_file_statistics(self, file_path: str) -> Dict[str, Any]:
        """Get statistics for captured PCAP file"""
        stats = {
            "size": 0,
            "packet_count": 0,
            "checksum": ""
        }
        
        try:
            # File size
            stats["size"] = os.path.getsize(file_path)
            
            # Calculate checksum
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            stats["checksum"] = sha256_hash.hexdigest()
            
            # Packet count using capinfos if available
            try:
                result = subprocess.run(
                    ["capinfos", "-c", file_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
                # Parse output for packet count
                for line in result.stdout.split('\n'):
                    if "Number of packets" in line:
                        stats["packet_count"] = int(line.split(':')[1].strip())
                        break
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback: estimate based on file size
                stats["packet_count"] = stats["size"] // 64  # Rough estimate
        
        except Exception as e:
            print(f"Error getting file statistics: {e}")
        
        return stats
    
    def _compress_pcap_file(self, file_path: str) -> str:
        """Compress PCAP file using gzip"""
        compressed_path = f"{file_path}.gz"
        
        try:
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # Remove original file
            os.remove(file_path)
            return compressed_path
            
        except Exception as e:
            print(f"Error compressing PCAP file: {e}")
            return file_path
    
    def _real_time_analysis_worker(self):
        """Worker thread for real-time packet analysis"""
        if not SCAPY_AVAILABLE:
            print("Real-time analysis disabled: Scapy not available")
            return
        
        print("Starting real-time packet analysis...")
        
        while not self._stop_event.is_set():
            try:
                # Monitor active captures for real-time analysis
                for interface, capture_info in self.active_captures.items():
                    if self.config.threat_detection_enabled:
                        self._analyze_interface_traffic(interface)
                
                time.sleep(1)  # Analysis interval
                
            except Exception as e:
                print(f"Error in real-time analysis: {e}")
                time.sleep(5)
    
    def _analyze_interface_traffic(self, interface: str):
        """Analyze traffic on a specific interface"""
        try:
            # Capture packets for analysis (small window)
            packets = scapy.sniff(iface=interface, count=10, timeout=1)
            
            for packet in packets:
                metrics = self._extract_packet_metrics(packet)
                if metrics:
                    self._detect_threats(metrics)
                    self._update_flow_tracking(metrics)
            
        except Exception as e:
            print(f"Error analyzing interface {interface}: {e}")
    
    def _extract_packet_metrics(self, packet) -> Optional[PacketMetrics]:
        """Extract metrics from a packet"""
        try:
            if not packet.haslayer(IP):
                return None
            
            ip_layer = packet[IP]
            
            metrics = PacketMetrics(
                timestamp=time.time(),
                size_bytes=len(packet),
                protocol=ip_layer.proto,
                src_ip=ip_layer.src,
                dst_ip=ip_layer.dst,
                payload_size=len(packet.payload) if packet.payload else 0
            )
            
            # TCP specific fields
            if packet.haslayer(TCP):
                tcp_layer = packet[TCP]
                metrics.src_port = tcp_layer.sport
                metrics.dst_port = tcp_layer.dport
                metrics.tcp_flags = str(tcp_layer.flags)
                metrics.protocol = "TCP"
            
            # UDP specific fields
            elif packet.haslayer(UDP):
                udp_layer = packet[UDP]
                metrics.src_port = udp_layer.sport
                metrics.dst_port = udp_layer.dport
                metrics.protocol = "UDP"
            
            # ICMP
            elif packet.haslayer(ICMP):
                metrics.protocol = "ICMP"
            
            return metrics
            
        except Exception as e:
            print(f"Error extracting packet metrics: {e}")
            return None
    
    def _detect_threats(self, metrics: PacketMetrics):
        """Detect threats based on packet metrics"""
        threats_detected = []
        
        # Port scanning detection
        if metrics.tcp_flags == "S" and metrics.dst_port:
            # Track SYN packets for port scanning
            scan_key = f"{metrics.src_ip}_scan"
            if scan_key not in self.capture_stats:
                self.capture_stats[scan_key] = {"ports": set(), "timestamp": time.time()}
            
            self.capture_stats[scan_key]["ports"].add(metrics.dst_port)
            
            # Detect if scanning multiple ports
            if len(self.capture_stats[scan_key]["ports"]) > 10:
                threats_detected.append("port_scan")
        
        # DNS tunneling detection
        if metrics.protocol == "UDP" and metrics.dst_port == 53:
            if metrics.payload_size > 512:  # Large DNS query
                threats_detected.append("dns_tunneling")
        
        # Update metrics with threat information
        if threats_detected:
            metrics.threat_score = len(threats_detected) * 0.3
            metrics.signature_matches = threats_detected
            print(f"Threats detected: {threats_detected} from {metrics.src_ip}")
    
    def _update_flow_tracking(self, metrics: PacketMetrics):
        """Update network flow tracking"""
        if not metrics.src_port or not metrics.dst_port:
            return
        
        # Create flow ID
        flow_id = f"{metrics.src_ip}:{metrics.src_port}-{metrics.dst_ip}:{metrics.dst_port}-{metrics.protocol}"
        
        if flow_id not in self.flows:
            self.flows[flow_id] = NetworkFlow(
                flow_id=flow_id,
                start_timestamp=metrics.timestamp,
                end_timestamp=metrics.timestamp,
                src_ip=metrics.src_ip,
                dst_ip=metrics.dst_ip,
                src_port=metrics.src_port,
                dst_port=metrics.dst_port,
                protocol=metrics.protocol
            )
        
        # Update flow statistics
        flow = self.flows[flow_id]
        flow.end_timestamp = metrics.timestamp
        flow.packets_forward += 1
        flow.bytes_forward += metrics.size_bytes
        flow.duration_seconds = flow.end_timestamp - flow.start_timestamp
        
        # Calculate derived metrics
        if flow.duration_seconds > 0:
            total_packets = flow.packets_forward + flow.packets_backward
            total_bytes = flow.bytes_forward + flow.bytes_backward
            flow.packets_per_second = total_packets / flow.duration_seconds
            flow.avg_packet_size = total_bytes / total_packets if total_packets > 0 else 0
    
    def get_capture_statistics(self) -> Dict[str, Any]:
        """Get comprehensive capture statistics"""
        stats = {
            "active_captures": len(self.active_captures),
            "total_flows": len(self.flows),
            "capture_processes": list(self.capture_processes.keys()),
            "interfaces": []
        }
        
        # Interface statistics
        for interface, capture_info in self.active_captures.items():
            interface_stats = {
                "interface": interface,
                "episode_id": capture_info["episode_id"],
                "duration": time.time() - capture_info["start_time"],
                "capture_file": capture_info["capture_file"]
            }
            
            # Add file size if available
            try:
                file_size = os.path.getsize(capture_info["capture_file"])
                interface_stats["file_size_bytes"] = file_size
            except:
                interface_stats["file_size_bytes"] = 0
            
            stats["interfaces"].append(interface_stats)
        
        return stats
    
    def export_flow_analysis(self, output_file: str):
        """Export flow analysis to JSON"""
        flow_data = []
        
        for flow in self.flows.values():
            flow_dict = {
                "flow_id": flow.flow_id,
                "start_timestamp": flow.start_timestamp,
                "end_timestamp": flow.end_timestamp,
                "src_ip": flow.src_ip,
                "dst_ip": flow.dst_ip,
                "src_port": flow.src_port,
                "dst_port": flow.dst_port,
                "protocol": flow.protocol,
                "packets_forward": flow.packets_forward,
                "packets_backward": flow.packets_backward,
                "bytes_forward": flow.bytes_forward,
                "bytes_backward": flow.bytes_backward,
                "duration_seconds": flow.duration_seconds,
                "avg_packet_size": flow.avg_packet_size,
                "packets_per_second": flow.packets_per_second,
                "suspicious_patterns": flow.suspicious_patterns,
                "threat_indicators": flow.threat_indicators,
                "flow_classification": flow.flow_classification
            }
            flow_data.append(flow_dict)
        
        with open(output_file, 'w') as f:
            json.dump(flow_data, f, indent=2)

if __name__ == "__main__":
    # Example usage and testing
    print("Testing PCAP collector...")
    
    # Create test configuration
    config = CaptureConfig(
        interfaces=[
            NetworkInterface(
                name="lo",  # Loopback interface for testing
                ip_address="127.0.0.1",
                subnet="127.0.0.0/8"
            )
        ],
        max_capture_duration_minutes=1,
        output_directory="/tmp/test_pcap",
        real_time_analysis=True,
        threat_detection_enabled=True
    )
    
    # Initialize collector
    collector = PCAPCollector(config)
    
    try:
        # Start capture
        print("Starting test capture...")
        capture_info = collector.start_capture("test_episode_001", ["lo"])
        print(f"Capture started: {capture_info}")
        
        # Let it run for a short time
        time.sleep(5)
        
        # Generate some traffic for testing
        try:
            subprocess.run(["ping", "-c", "3", "127.0.0.1"], 
                         capture_output=True, timeout=10)
        except:
            pass
        
        # Get statistics
        stats = collector.get_capture_statistics()
        print(f"Capture statistics: {stats}")
        
        # Stop capture
        results = collector.stop_capture("test_episode_001")
        print(f"Capture results: {results}")
        
    finally:
        # Clean up
        collector.stop_all_captures()
    
    print("PCAP collector test completed!")