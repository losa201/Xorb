#!/usr/bin/env python3
"""
Xorb Multi-Modal AI Analysis Engine
Phase 6.4 - Code + Network + Logs + Infrastructure Correlation
"""

import asyncio
import json
import logging
import os
import re
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import tempfile
import subprocess

import asyncpg
import aioredis
import aiofiles
from elasticsearch import AsyncElasticsearch
import networkx as nx
from scapy.all import rdpcap, IP, TCP, UDP
import numpy as np
from openai import AsyncOpenAI
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog

# Configure logging
logger = structlog.get_logger("xorb.multimodal_ai")

# Phase 6.4 Metrics
multimodal_analysis_total = Counter(
    'multimodal_analysis_total',
    'Total multi-modal analyses performed',
    ['analysis_type', 'data_sources']
)

analysis_correlation_score = Histogram(
    'analysis_correlation_score',
    'Correlation scores between different data sources',
    ['source_a', 'source_b']
)

data_processing_duration = Histogram(
    'data_processing_duration_seconds',
    'Time to process different data types',
    ['data_type', 'processing_stage']
)

attack_path_discoveries = Counter(
    'attack_path_discoveries_total',
    'Attack paths discovered through correlation',
    ['path_complexity', 'confidence_level']
)

anomaly_detection_accuracy = Gauge(
    'anomaly_detection_accuracy',
    'Accuracy of anomaly detection models',
    ['detection_type', 'data_source']
)

cross_correlation_insights = Counter(
    'cross_correlation_insights_total',
    'Insights discovered through cross-modal correlation',
    ['insight_type', 'confidence']
)

class DataSourceType(Enum):
    CODE_REPOSITORY = "code_repository"
    NETWORK_TRAFFIC = "network_traffic"
    APPLICATION_LOGS = "application_logs"
    SYSTEM_LOGS = "system_logs"
    INFRASTRUCTURE_CONFIG = "infrastructure_config"
    RUNTIME_BEHAVIOR = "runtime_behavior"

class AnalysisType(Enum):
    VULNERABILITY_CORRELATION = "vulnerability_correlation"
    ATTACK_PATH_ANALYSIS = "attack_path_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    THREAT_HUNTING = "threat_hunting"

@dataclass
class CodeAnalysisResult:
    """Code analysis results"""
    repository_url: str
    commit_hash: str
    files_analyzed: List[str]
    
    # Static analysis results
    vulnerabilities: List[Dict]
    dependencies: List[Dict]
    security_patterns: List[Dict]
    code_quality_metrics: Dict
    
    # Dynamic analysis
    entry_points: List[str]
    data_flows: List[Dict]
    api_endpoints: List[Dict]
    
    # ML insights
    complexity_score: float
    maintainability_score: float
    security_score: float

@dataclass
class NetworkAnalysisResult:
    """Network traffic analysis results"""
    capture_file: str
    analysis_timeframe: Tuple[datetime, datetime]
    
    # Traffic patterns
    flow_statistics: Dict
    protocol_distribution: Dict
    connection_patterns: List[Dict]
    
    # Anomaly detection
    suspicious_flows: List[Dict]
    port_scan_indicators: List[Dict]
    data_exfiltration_indicators: List[Dict]
    
    # ML insights
    baseline_deviation_score: float
    anomaly_confidence: float
    threat_indicators: List[Dict]

@dataclass
class LogAnalysisResult:
    """Log analysis results"""
    log_sources: List[str]
    time_range: Tuple[datetime, datetime]
    
    # Pattern analysis
    error_patterns: List[Dict]
    security_events: List[Dict]
    performance_metrics: Dict
    
    # Correlation findings
    temporal_correlations: List[Dict]
    cross_service_patterns: List[Dict]
    
    # ML insights
    anomaly_score: float
    threat_indicators: List[Dict]
    behavioral_changes: List[Dict]

@dataclass
class InfrastructureAnalysisResult:
    """Infrastructure configuration analysis"""
    infrastructure_type: str  # k8s, terraform, docker, etc.
    config_files: List[str]
    
    # Configuration analysis
    misconfigurations: List[Dict]
    security_gaps: List[Dict]
    compliance_issues: List[Dict]
    
    # Attack surface
    exposed_services: List[Dict]
    privilege_escalation_paths: List[Dict]
    network_topology: Dict
    
    # Risk assessment
    risk_score: float
    critical_paths: List[Dict]

@dataclass
class MultiModalCorrelation:
    """Cross-modal correlation result"""
    correlation_id: str
    data_sources: List[DataSourceType]
    analysis_types: List[AnalysisType]
    
    # Correlation findings
    correlation_score: float
    confidence_level: str
    key_insights: List[str]
    
    # Attack chain reconstruction
    attack_timeline: List[Dict]
    attack_techniques: List[str]
    affected_assets: List[str]
    
    # Risk assessment
    impact_score: float
    likelihood_score: float
    business_risk: str
    
    # Recommendations
    immediate_actions: List[str]
    strategic_improvements: List[str]
    
    # Metadata
    analyzed_at: datetime
    model_version: str

class CodeAnalyzer:
    """Analyzes code repositories for security patterns"""
    
    def __init__(self):
        self.git_analyzers = ["semgrep", "bandit", "eslint", "gosec"]
        self.dependency_analyzers = ["safety", "audit", "snyk"]
        
    async def analyze_repository(self, repo_url: str, branch: str = "main") -> CodeAnalysisResult:
        """Comprehensive code repository analysis"""
        
        start_time = datetime.now()
        
        try:
            with data_processing_duration.labels(
                data_type="code_repository",
                processing_stage="full_analysis"
            ).time():
                
                # Clone repository
                repo_path = await self._clone_repository(repo_url, branch)
                
                # Static analysis
                vulnerabilities = await self._run_static_analysis(repo_path)
                dependencies = await self._analyze_dependencies(repo_path)
                security_patterns = await self._identify_security_patterns(repo_path)
                quality_metrics = await self._calculate_quality_metrics(repo_path)
                
                # Dynamic analysis (entry points, data flows)
                entry_points = await self._identify_entry_points(repo_path)
                data_flows = await self._analyze_data_flows(repo_path)
                api_endpoints = await self._discover_api_endpoints(repo_path)
                
                # ML-based scoring
                complexity_score = await self._calculate_complexity_score(repo_path)
                maintainability_score = await self._calculate_maintainability_score(repo_path)
                security_score = await self._calculate_security_score(vulnerabilities, security_patterns)
                
                result = CodeAnalysisResult(
                    repository_url=repo_url,
                    commit_hash=await self._get_current_commit(repo_path),
                    files_analyzed=await self._get_analyzed_files(repo_path),
                    vulnerabilities=vulnerabilities,
                    dependencies=dependencies,
                    security_patterns=security_patterns,
                    code_quality_metrics=quality_metrics,
                    entry_points=entry_points,
                    data_flows=data_flows,
                    api_endpoints=api_endpoints,
                    complexity_score=complexity_score,
                    maintainability_score=maintainability_score,
                    security_score=security_score
                )
                
                # Cleanup
                await self._cleanup_repository(repo_path)
                
                duration = (datetime.now() - start_time).total_seconds()
                logger.info("Code analysis completed",
                           repository=repo_url,
                           vulnerabilities_found=len(vulnerabilities),
                           security_score=security_score,
                           duration=duration)
                
                return result
                
        except Exception as e:
            logger.error("Code analysis failed", repository=repo_url, error=str(e))
            raise
    
    async def _clone_repository(self, repo_url: str, branch: str) -> str:
        """Clone repository to temporary directory"""
        
        temp_dir = tempfile.mkdtemp(prefix="xorb_repo_")
        
        clone_cmd = ["git", "clone", "--depth", "1", "--branch", branch, repo_url, temp_dir]
        
        process = await asyncio.create_subprocess_exec(
            *clone_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Failed to clone repository: {repo_url}")
        
        return temp_dir
    
    async def _run_static_analysis(self, repo_path: str) -> List[Dict]:
        """Run multiple static analysis tools"""
        
        vulnerabilities = []
        
        # Run Semgrep for multi-language analysis
        try:
            semgrep_cmd = ["semgrep", "--config=auto", "--json", repo_path]
            process = await asyncio.create_subprocess_exec(
                *semgrep_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                semgrep_results = json.loads(stdout.decode())
                
                for result in semgrep_results.get("results", []):
                    vulnerabilities.append({
                        "tool": "semgrep",
                        "rule_id": result.get("check_id"),
                        "severity": result.get("extra", {}).get("severity", "medium"),
                        "message": result.get("extra", {}).get("message", ""),
                        "file": result.get("path"),
                        "line": result.get("start", {}).get("line"),
                        "confidence": "high"
                    })
        except Exception as e:
            logger.warning("Semgrep analysis failed", error=str(e))
        
        # Add other analyzers (bandit for Python, etc.)
        
        return vulnerabilities
    
    async def _analyze_dependencies(self, repo_path: str) -> List[Dict]:
        """Analyze project dependencies for vulnerabilities"""
        
        dependencies = []
        
        # Check for package files
        package_files = {
            "requirements.txt": "python",
            "package.json": "javascript",
            "Gemfile": "ruby",
            "pom.xml": "java",
            "go.mod": "go"
        }
        
        for package_file, language in package_files.items():
            file_path = os.path.join(repo_path, package_file)
            if os.path.exists(file_path):
                deps = await self._parse_dependencies(file_path, language)
                dependencies.extend(deps)
        
        return dependencies
    
    async def _parse_dependencies(self, file_path: str, language: str) -> List[Dict]:
        """Parse dependencies from specific file types"""
        
        dependencies = []
        
        try:
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
            
            if language == "python" and "requirements.txt" in file_path:
                # Parse Python requirements
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Simple parsing - would use proper parser in production
                        dep_name = line.split('==')[0].split('>=')[0].split('<=')[0]
                        dependencies.append({
                            "name": dep_name.strip(),
                            "language": language,
                            "file": file_path,
                            "vulnerabilities": []  # Would check vulnerability DB
                        })
            
            elif language == "javascript" and "package.json" in file_path:
                # Parse Node.js package.json
                package_data = json.loads(content)
                
                for dep_type in ["dependencies", "devDependencies"]:
                    for dep_name, version in package_data.get(dep_type, {}).items():
                        dependencies.append({
                            "name": dep_name,
                            "version": version,
                            "language": language,
                            "type": dep_type,
                            "file": file_path,
                            "vulnerabilities": []
                        })
        
        except Exception as e:
            logger.warning("Failed to parse dependencies", file=file_path, error=str(e))
        
        return dependencies
    
    async def _identify_security_patterns(self, repo_path: str) -> List[Dict]:
        """Identify security-relevant code patterns"""
        
        patterns = []
        
        # Security patterns to look for
        security_checks = [
            {
                "pattern": r"password\s*=\s*['\"][^'\"]+['\"]",
                "type": "hardcoded_password",
                "severity": "high"
            },
            {
                "pattern": r"api_key\s*=\s*['\"][^'\"]+['\"]",
                "type": "hardcoded_api_key",
                "severity": "high"
            },
            {
                "pattern": r"eval\s*\(",
                "type": "code_injection_risk",
                "severity": "high"
            },
            {
                "pattern": r"innerHTML\s*=",
                "type": "xss_risk",
                "severity": "medium"
            }
        ]
        
        # Search through all code files
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith(('.py', '.js', '.java', '.cs', '.go', '.php')):
                    file_path = os.path.join(root, file)
                    
                    try:
                        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                            content = await f.read()
                        
                        for check in security_checks:
                            matches = re.finditer(check["pattern"], content, re.IGNORECASE)
                            
                            for match in matches:
                                line_num = content[:match.start()].count('\n') + 1
                                
                                patterns.append({
                                    "type": check["type"],
                                    "severity": check["severity"],
                                    "file": file_path,
                                    "line": line_num,
                                    "match": match.group(0)[:100],  # Truncate for safety
                                    "pattern": check["pattern"]
                                })
                    
                    except Exception as e:
                        continue  # Skip files that can't be read
        
        return patterns
    
    async def _calculate_quality_metrics(self, repo_path: str) -> Dict:
        """Calculate code quality metrics"""
        
        metrics = {
            "total_files": 0,
            "total_lines": 0,
            "comment_ratio": 0.0,
            "complexity_score": 0.0,
            "test_coverage": 0.0
        }
        
        total_lines = 0
        comment_lines = 0
        
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith(('.py', '.js', '.java', '.cs', '.go')):
                    metrics["total_files"] += 1
                    file_path = os.path.join(root, file)
                    
                    try:
                        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                            lines = await f.readlines()
                        
                        file_lines = len(lines)
                        file_comments = sum(1 for line in lines if line.strip().startswith(('#', '//', '/*')))
                        
                        total_lines += file_lines
                        comment_lines += file_comments
                    
                    except Exception as e:
                        continue
        
        metrics["total_lines"] = total_lines
        if total_lines > 0:
            metrics["comment_ratio"] = comment_lines / total_lines
        
        return metrics
    
    async def _identify_entry_points(self, repo_path: str) -> List[str]:
        """Identify application entry points"""
        
        entry_points = []
        
        # Common entry point patterns
        patterns = [
            r"def main\(",  # Python main function
            r"public static void main",  # Java main
            r"func main\(",  # Go main
            r"app\.listen\(",  # Express.js server
            r"@app\.route\(",  # Flask routes
            r"@RestController",  # Spring Boot controllers
        ]
        
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith(('.py', '.js', '.java', '.go')):
                    file_path = os.path.join(root, file)
                    
                    try:
                        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                            content = await f.read()
                        
                        for pattern in patterns:
                            if re.search(pattern, content):
                                entry_points.append(file_path)
                                break
                    
                    except Exception as e:
                        continue
        
        return entry_points
    
    async def _analyze_data_flows(self, repo_path: str) -> List[Dict]:
        """Analyze data flow patterns"""
        
        # Simplified data flow analysis
        # In production, would use proper AST analysis
        
        flows = []
        
        # Look for common data flow patterns
        flow_patterns = [
            {
                "pattern": r"request\.get\(['\"](\w+)['\"]",
                "type": "user_input",
                "source": "HTTP parameter"
            },
            {
                "pattern": r"cursor\.execute\(",
                "type": "database_query",
                "sink": "database"
            },
            {
                "pattern": r"subprocess\.call\(",
                "type": "command_execution",
                "sink": "system_command"
            }
        ]
        
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith(('.py', '.js', '.java')):
                    file_path = os.path.join(root, file)
                    
                    try:
                        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                            content = await f.read()
                        
                        for pattern_info in flow_patterns:
                            matches = re.finditer(pattern_info["pattern"], content)
                            
                            for match in matches:
                                line_num = content[:match.start()].count('\n') + 1
                                
                                flows.append({
                                    "type": pattern_info["type"],
                                    "file": file_path,
                                    "line": line_num,
                                    "source": pattern_info.get("source", "unknown"),
                                    "sink": pattern_info.get("sink", "unknown"),
                                    "data": match.group(0)
                                })
                    
                    except Exception as e:
                        continue
        
        return flows
    
    async def _discover_api_endpoints(self, repo_path: str) -> List[Dict]:
        """Discover API endpoints"""
        
        endpoints = []
        
        # API endpoint patterns
        endpoint_patterns = [
            {
                "pattern": r"@app\.route\(['\"]([^'\"]+)['\"].*methods=\[([^\]]+)\]",
                "framework": "flask"
            },
            {
                "pattern": r"@RestController.*@RequestMapping\(['\"]([^'\"]+)['\"]",
                "framework": "spring"
            },
            {
                "pattern": r"app\.(get|post|put|delete)\(['\"]([^'\"]+)['\"]",
                "framework": "express"
            }
        ]
        
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith(('.py', '.js', '.java')):
                    file_path = os.path.join(root, file)
                    
                    try:
                        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                            content = await f.read()
                        
                        for pattern_info in endpoint_patterns:
                            matches = re.finditer(pattern_info["pattern"], content, re.MULTILINE | re.DOTALL)
                            
                            for match in matches:
                                endpoints.append({
                                    "path": match.group(1) if len(match.groups()) >= 1 else "unknown",
                                    "methods": match.group(2).split(',') if len(match.groups()) >= 2 else ["GET"],
                                    "framework": pattern_info["framework"],
                                    "file": file_path,
                                    "line": content[:match.start()].count('\n') + 1
                                })
                    
                    except Exception as e:
                        continue
        
        return endpoints
    
    async def _calculate_complexity_score(self, repo_path: str) -> float:
        """Calculate code complexity score"""
        
        # Simplified complexity calculation
        # In production, would use proper cyclomatic complexity tools
        
        total_complexity = 0
        total_functions = 0
        
        complexity_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'case']
        
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.py'):  # Focus on Python for now
                    file_path = os.path.join(root, file)
                    
                    try:
                        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                            content = await f.read()
                        
                        # Count functions
                        function_count = len(re.findall(r'def \w+\(', content))
                        total_functions += function_count
                        
                        # Count complexity indicators
                        for keyword in complexity_keywords:
                            total_complexity += len(re.findall(rf'\b{keyword}\b', content))
                    
                    except Exception as e:
                        continue
        
        return total_complexity / max(1, total_functions)
    
    async def _calculate_maintainability_score(self, repo_path: str) -> float:
        """Calculate maintainability score"""
        
        # Simple maintainability heuristics
        score = 1.0
        
        # Check for documentation
        has_readme = os.path.exists(os.path.join(repo_path, "README.md"))
        if has_readme:
            score += 0.2
        
        # Check for tests
        test_dirs = ['test', 'tests', 'spec']
        has_tests = any(os.path.exists(os.path.join(repo_path, test_dir)) for test_dir in test_dirs)
        if has_tests:
            score += 0.3
        
        # Check for configuration management
        config_files = ['requirements.txt', 'package.json', 'Dockerfile', '.gitignore']
        config_count = sum(1 for config in config_files if os.path.exists(os.path.join(repo_path, config)))
        score += (config_count / len(config_files)) * 0.2
        
        return min(1.0, score)
    
    async def _calculate_security_score(self, vulnerabilities: List[Dict], patterns: List[Dict]) -> float:
        """Calculate security score based on findings"""
        
        score = 1.0
        
        # Penalize based on vulnerability severity
        for vuln in vulnerabilities:
            severity = vuln.get("severity", "medium").lower()
            if severity == "critical":
                score -= 0.2
            elif severity == "high":
                score -= 0.1
            elif severity == "medium":
                score -= 0.05
        
        # Penalize based on security patterns
        for pattern in patterns:
            severity = pattern.get("severity", "medium").lower()
            if severity == "high":
                score -= 0.1
            elif severity == "medium":
                score -= 0.05
        
        return max(0.0, score)
    
    async def _get_current_commit(self, repo_path: str) -> str:
        """Get current commit hash"""
        
        try:
            process = await asyncio.create_subprocess_exec(
                "git", "rev-parse", "HEAD",
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE
            )
            
            stdout, _ = await process.communicate()
            return stdout.decode().strip()
        
        except Exception as e:
            return "unknown"
    
    async def _get_analyzed_files(self, repo_path: str) -> List[str]:
        """Get list of analyzed files"""
        
        files = []
        
        for root, dirs, filenames in os.walk(repo_path):
            for filename in filenames:
                if filename.endswith(('.py', '.js', '.java', '.cs', '.go', '.php')):
                    files.append(os.path.join(root, filename))
        
        return files
    
    async def _cleanup_repository(self, repo_path: str):
        """Clean up cloned repository"""
        
        import shutil
        try:
            shutil.rmtree(repo_path)
        except Exception as e:
            logger.warning("Failed to cleanup repository", path=repo_path, error=str(e))

class NetworkAnalyzer:
    """Analyzes network traffic for security patterns"""
    
    def __init__(self):
        self.anomaly_models = {}
        
    async def analyze_traffic_capture(self, pcap_file: str) -> NetworkAnalysisResult:
        """Analyze network traffic from PCAP file"""
        
        start_time = datetime.now()
        
        try:
            with data_processing_duration.labels(
                data_type="network_traffic",
                processing_stage="pcap_analysis"
            ).time():
                
                # Parse PCAP file
                packets = rdpcap(pcap_file)
                
                # Extract flow statistics
                flow_stats = await self._extract_flow_statistics(packets)
                protocol_dist = await self._analyze_protocol_distribution(packets)
                connections = await self._analyze_connection_patterns(packets)
                
                # Anomaly detection
                suspicious_flows = await self._detect_suspicious_flows(packets)
                port_scans = await self._detect_port_scans(packets)
                exfiltration = await self._detect_data_exfiltration(packets)
                
                # ML-based analysis
                baseline_deviation = await self._calculate_baseline_deviation(flow_stats)
                anomaly_confidence = await self._calculate_anomaly_confidence(suspicious_flows)
                threat_indicators = await self._identify_threat_indicators(packets)
                
                # Determine time range
                timestamps = [float(pkt.time) for pkt in packets if hasattr(pkt, 'time')]
                time_range = (
                    datetime.fromtimestamp(min(timestamps)) if timestamps else datetime.now(),
                    datetime.fromtimestamp(max(timestamps)) if timestamps else datetime.now()
                )
                
                result = NetworkAnalysisResult(
                    capture_file=pcap_file,
                    analysis_timeframe=time_range,
                    flow_statistics=flow_stats,
                    protocol_distribution=protocol_dist,
                    connection_patterns=connections,
                    suspicious_flows=suspicious_flows,
                    port_scan_indicators=port_scans,
                    data_exfiltration_indicators=exfiltration,
                    baseline_deviation_score=baseline_deviation,
                    anomaly_confidence=anomaly_confidence,
                    threat_indicators=threat_indicators
                )
                
                duration = (datetime.now() - start_time).total_seconds()
                logger.info("Network analysis completed",
                           pcap_file=pcap_file,
                           packets_analyzed=len(packets),
                           suspicious_flows=len(suspicious_flows),
                           duration=duration)
                
                return result
                
        except Exception as e:
            logger.error("Network analysis failed", pcap_file=pcap_file, error=str(e))
            raise
    
    async def _extract_flow_statistics(self, packets) -> Dict:
        """Extract network flow statistics"""
        
        flows = {}
        stats = {
            "total_packets": len(packets),
            "total_bytes": 0,
            "unique_flows": 0,
            "avg_packet_size": 0,
            "flow_durations": []
        }
        
        for pkt in packets:
            if IP in pkt:
                # Create flow identifier
                if TCP in pkt:
                    flow_id = f"{pkt[IP].src}:{pkt[TCP].sport}-{pkt[IP].dst}:{pkt[TCP].dport}"
                elif UDP in pkt:
                    flow_id = f"{pkt[IP].src}:{pkt[UDP].sport}-{pkt[IP].dst}:{pkt[UDP].dport}"
                else:
                    flow_id = f"{pkt[IP].src}-{pkt[IP].dst}"
                
                if flow_id not in flows:
                    flows[flow_id] = {
                        "start_time": float(pkt.time) if hasattr(pkt, 'time') else 0,
                        "end_time": float(pkt.time) if hasattr(pkt, 'time') else 0,
                        "packet_count": 0,
                        "byte_count": 0
                    }
                
                flows[flow_id]["packet_count"] += 1
                flows[flow_id]["byte_count"] += len(pkt)
                flows[flow_id]["end_time"] = float(pkt.time) if hasattr(pkt, 'time') else 0
                
                stats["total_bytes"] += len(pkt)
        
        stats["unique_flows"] = len(flows)
        stats["avg_packet_size"] = stats["total_bytes"] / max(1, stats["total_packets"])
        stats["flow_durations"] = [flow["end_time"] - flow["start_time"] for flow in flows.values()]
        
        return stats
    
    async def _analyze_protocol_distribution(self, packets) -> Dict:
        """Analyze protocol distribution"""
        
        protocols = {}
        
        for pkt in packets:
            if IP in pkt:
                proto = pkt[IP].proto
                proto_name = {1: "ICMP", 6: "TCP", 17: "UDP"}.get(proto, f"Proto_{proto}")
                
                if proto_name not in protocols:
                    protocols[proto_name] = {"count": 0, "bytes": 0}
                
                protocols[proto_name]["count"] += 1
                protocols[proto_name]["bytes"] += len(pkt)
        
        return protocols
    
    async def _analyze_connection_patterns(self, packets) -> List[Dict]:
        """Analyze connection patterns"""
        
        connections = []
        connection_map = {}
        
        for pkt in packets:
            if IP in pkt and TCP in pkt:
                src_ip = pkt[IP].src
                dst_ip = pkt[IP].dst
                dst_port = pkt[TCP].dport
                
                conn_key = f"{src_ip}-{dst_ip}:{dst_port}"
                
                if conn_key not in connection_map:
                    connection_map[conn_key] = {
                        "src_ip": src_ip,
                        "dst_ip": dst_ip,
                        "dst_port": dst_port,
                        "packet_count": 0,
                        "first_seen": float(pkt.time) if hasattr(pkt, 'time') else 0,
                        "last_seen": float(pkt.time) if hasattr(pkt, 'time') else 0,
                        "flags": set()
                    }
                
                connection_map[conn_key]["packet_count"] += 1
                connection_map[conn_key]["last_seen"] = float(pkt.time) if hasattr(pkt, 'time') else 0
                connection_map[conn_key]["flags"].add(pkt[TCP].flags)
        
        # Convert to list format
        for conn in connection_map.values():
            conn["flags"] = list(conn["flags"])
            connections.append(conn)
        
        return connections
    
    async def _detect_suspicious_flows(self, packets) -> List[Dict]:
        """Detect suspicious network flows"""
        
        suspicious = []
        
        # Port analysis for common attack patterns
        port_counts = {}
        ip_port_combinations = {}
        
        for pkt in packets:
            if IP in pkt:
                src_ip = pkt[IP].src
                
                if TCP in pkt:
                    dst_port = pkt[TCP].dport
                    
                    # Track port scanning behavior
                    if src_ip not in ip_port_combinations:
                        ip_port_combinations[src_ip] = set()
                    ip_port_combinations[src_ip].add(dst_port)
                    
                    # Track port frequency
                    if dst_port not in port_counts:
                        port_counts[dst_port] = 0
                    port_counts[dst_port] += 1
        
        # Identify port scanning
        for src_ip, ports in ip_port_combinations.items():
            if len(ports) > 20:  # Scanned more than 20 ports
                suspicious.append({
                    "type": "port_scan",
                    "src_ip": src_ip,
                    "ports_scanned": len(ports),
                    "severity": "high" if len(ports) > 100 else "medium",
                    "confidence": min(1.0, len(ports) / 100.0)
                })
        
        # Identify unusual port usage
        for port, count in port_counts.items():
            if port > 49152 and count > 100:  # High traffic on ephemeral ports
                suspicious.append({
                    "type": "unusual_port_activity",
                    "port": port,
                    "packet_count": count,
                    "severity": "medium",
                    "confidence": 0.6
                })
        
        return suspicious
    
    async def _detect_port_scans(self, packets) -> List[Dict]:
        """Detect port scanning indicators"""
        
        port_scans = []
        
        # Track SYN packets without corresponding ACK
        syn_packets = {}
        
        for pkt in packets:
            if IP in pkt and TCP in pkt:
                src_ip = pkt[IP].src
                dst_ip = pkt[IP].dst
                dst_port = pkt[TCP].dport
                flags = pkt[TCP].flags
                
                # SYN scan detection
                if flags == 2:  # SYN flag only
                    key = f"{src_ip}-{dst_ip}"
                    
                    if key not in syn_packets:
                        syn_packets[key] = {"ports": set(), "timestamp": float(pkt.time) if hasattr(pkt, 'time') else 0}
                    
                    syn_packets[key]["ports"].add(dst_port)
        
        # Identify scan patterns
        for key, data in syn_packets.items():
            src_ip, dst_ip = key.split('-')
            
            if len(data["ports"]) >= 10:  # Scanned 10+ ports
                port_scans.append({
                    "scan_type": "syn_scan",
                    "source_ip": src_ip,
                    "target_ip": dst_ip,
                    "ports_scanned": list(data["ports"]),
                    "port_count": len(data["ports"]),
                    "timestamp": data["timestamp"],
                    "severity": "high" if len(data["ports"]) > 50 else "medium"
                })
        
        return port_scans
    
    async def _detect_data_exfiltration(self, packets) -> List[Dict]:
        """Detect potential data exfiltration"""
        
        exfiltration_indicators = []
        
        # Track outbound data volumes
        outbound_data = {}
        
        for pkt in packets:
            if IP in pkt:
                src_ip = pkt[IP].src
                dst_ip = pkt[IP].dst
                
                # Assume internal IPs start with 10., 172.16-31., or 192.168.
                is_internal_src = (src_ip.startswith('10.') or 
                                  src_ip.startswith('192.168.') or
                                  any(src_ip.startswith(f'172.{i}.') for i in range(16, 32)))
                
                is_internal_dst = (dst_ip.startswith('10.') or 
                                  dst_ip.startswith('192.168.') or
                                  any(dst_ip.startswith(f'172.{i}.') for i in range(16, 32)))
                
                # Track outbound traffic (internal to external)
                if is_internal_src and not is_internal_dst:
                    if src_ip not in outbound_data:
                        outbound_data[src_ip] = {"bytes": 0, "destinations": set()}
                    
                    outbound_data[src_ip]["bytes"] += len(pkt)
                    outbound_data[src_ip]["destinations"].add(dst_ip)
        
        # Identify potential exfiltration
        for src_ip, data in outbound_data.items():
            # Large outbound transfers or many destinations
            if data["bytes"] > 100 * 1024 * 1024:  # > 100MB
                exfiltration_indicators.append({
                    "type": "large_outbound_transfer",
                    "source_ip": src_ip,
                    "bytes_transferred": data["bytes"],
                    "destination_count": len(data["destinations"]),
                    "severity": "high",
                    "confidence": 0.7
                })
            elif len(data["destinations"]) > 20:  # Many different destinations
                exfiltration_indicators.append({
                    "type": "multiple_destination_transfer",
                    "source_ip": src_ip,
                    "destination_count": len(data["destinations"]),
                    "bytes_transferred": data["bytes"],
                    "severity": "medium",
                    "confidence": 0.6
                })
        
        return exfiltration_indicators
    
    async def _calculate_baseline_deviation(self, flow_stats: Dict) -> float:
        """Calculate deviation from baseline traffic patterns"""
        
        # Simplified baseline comparison
        # In production, would compare against historical baselines
        
        # Assume baseline values (would be learned from historical data)
        baseline = {
            "avg_packet_size": 500,
            "flows_per_minute": 100,
            "protocol_ratio_tcp": 0.7
        }
        
        current_avg_size = flow_stats.get("avg_packet_size", 500)
        deviation_score = abs(current_avg_size - baseline["avg_packet_size"]) / baseline["avg_packet_size"]
        
        return min(1.0, deviation_score)
    
    async def _calculate_anomaly_confidence(self, suspicious_flows: List[Dict]) -> float:
        """Calculate confidence in anomaly detection"""
        
        if not suspicious_flows:
            return 0.0
        
        # Weight by severity and count
        total_score = 0
        for flow in suspicious_flows:
            severity_weight = {"high": 1.0, "medium": 0.6, "low": 0.3}.get(flow.get("severity", "low"), 0.3)
            confidence = flow.get("confidence", 0.5)
            total_score += severity_weight * confidence
        
        return min(1.0, total_score / len(suspicious_flows))
    
    async def _identify_threat_indicators(self, packets) -> List[Dict]:
        """Identify known threat indicators"""
        
        indicators = []
        
        # Known malicious IP patterns (simplified)
        suspicious_ips = ["suspicious.example.com", "malware.domain.com"]
        
        # DNS analysis for suspicious domains
        dns_queries = []
        
        for pkt in packets:
            if IP in pkt and UDP in pkt:
                if pkt[UDP].dport == 53:  # DNS query
                    # Simple DNS analysis (would use proper DNS parsing)
                    try:
                        # Extract domain from DNS query (simplified)
                        payload = bytes(pkt[UDP].payload)
                        # Basic pattern matching for suspicious domains
                        for suspicious_ip in suspicious_ips:
                            if suspicious_ip.encode() in payload:
                                indicators.append({
                                    "type": "suspicious_dns_query",
                                    "domain": suspicious_ip,
                                    "source_ip": pkt[IP].src,
                                    "severity": "high",
                                    "timestamp": float(pkt.time) if hasattr(pkt, 'time') else 0
                                })
                    except:
                        pass
        
        return indicators

class LogAnalyzer:
    """Analyzes application and system logs"""
    
    def __init__(self):
        self.es_client = None
        
    async def initialize(self, elasticsearch_url: str):
        """Initialize Elasticsearch client"""
        self.es_client = AsyncElasticsearch([elasticsearch_url])
    
    async def analyze_logs(self, 
                          log_sources: List[str], 
                          time_range: Tuple[datetime, datetime]) -> LogAnalysisResult:
        """Analyze logs from multiple sources"""
        
        start_time = datetime.now()
        
        try:
            with data_processing_duration.labels(
                data_type="application_logs",
                processing_stage="log_analysis"
            ).time():
                
                # Query logs from Elasticsearch
                logs = await self._query_logs(log_sources, time_range)
                
                # Pattern analysis
                error_patterns = await self._analyze_error_patterns(logs)
                security_events = await self._identify_security_events(logs)
                performance_metrics = await self._extract_performance_metrics(logs)
                
                # Correlation analysis
                temporal_correlations = await self._find_temporal_correlations(logs)
                cross_service_patterns = await self._analyze_cross_service_patterns(logs)
                
                # ML-based analysis
                anomaly_score = await self._calculate_log_anomaly_score(logs)
                threat_indicators = await self._identify_log_threat_indicators(logs)
                behavioral_changes = await self._detect_behavioral_changes(logs)
                
                result = LogAnalysisResult(
                    log_sources=log_sources,
                    time_range=time_range,
                    error_patterns=error_patterns,
                    security_events=security_events,
                    performance_metrics=performance_metrics,
                    temporal_correlations=temporal_correlations,
                    cross_service_patterns=cross_service_patterns,
                    anomaly_score=anomaly_score,
                    threat_indicators=threat_indicators,
                    behavioral_changes=behavioral_changes
                )
                
                duration = (datetime.now() - start_time).total_seconds()
                logger.info("Log analysis completed",
                           log_sources_count=len(log_sources),
                           log_entries_analyzed=len(logs),
                           anomaly_score=anomaly_score,
                           duration=duration)
                
                return result
                
        except Exception as e:
            logger.error("Log analysis failed", error=str(e))
            raise
    
    async def _query_logs(self, log_sources: List[str], time_range: Tuple[datetime, datetime]) -> List[Dict]:
        """Query logs from Elasticsearch"""
        
        if not self.es_client:
            # Return mock data if ES not available
            return self._generate_mock_logs(log_sources, time_range)
        
        try:
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "terms": {
                                    "source": log_sources
                                }
                            },
                            {
                                "range": {
                                    "@timestamp": {
                                        "gte": time_range[0].isoformat(),
                                        "lte": time_range[1].isoformat()
                                    }
                                }
                            }
                        ]
                    }
                },
                "size": 10000,
                "sort": [
                    {
                        "@timestamp": {
                            "order": "asc"
                        }
                    }
                ]
            }
            
            response = await self.es_client.search(
                index="logs-*",
                body=query
            )
            
            return [hit["_source"] for hit in response["hits"]["hits"]]
            
        except Exception as e:
            logger.warning("Failed to query Elasticsearch, using mock data", error=str(e))
            return self._generate_mock_logs(log_sources, time_range)
    
    def _generate_mock_logs(self, log_sources: List[str], time_range: Tuple[datetime, datetime]) -> List[Dict]:
        """Generate mock log data for testing"""
        
        logs = []
        
        # Generate sample log entries
        log_patterns = [
            {"level": "INFO", "message": "User login successful", "user": "john.doe"},
            {"level": "ERROR", "message": "Database connection failed", "service": "api"},
            {"level": "WARN", "message": "High memory usage detected", "memory_percent": 85},
            {"level": "INFO", "message": "Request processed successfully", "response_time": 150},
            {"level": "ERROR", "message": "Authentication failed", "user": "admin", "ip": "192.168.1.100"},
            {"level": "INFO", "message": "File uploaded", "file_size": 1024000},
            {"level": "WARN", "message": "Unusual access pattern detected", "ip": "10.0.0.50"}
        ]
        
        # Generate logs across time range
        import random
        start_timestamp = time_range[0].timestamp()
        end_timestamp = time_range[1].timestamp()
        
        for _ in range(100):  # Generate 100 sample logs
            timestamp = start_timestamp + random.random() * (end_timestamp - start_timestamp)
            pattern = random.choice(log_patterns)
            
            log_entry = {
                "@timestamp": datetime.fromtimestamp(timestamp).isoformat(),
                "source": random.choice(log_sources),
                **pattern
            }
            
            logs.append(log_entry)
        
        return logs
    
    async def _analyze_error_patterns(self, logs: List[Dict]) -> List[Dict]:
        """Analyze error patterns in logs"""
        
        error_patterns = []
        error_counts = {}
        
        for log in logs:
            if log.get("level") in ["ERROR", "CRITICAL"]:
                message = log.get("message", "")
                
                # Simple pattern extraction (would use more sophisticated NLP)
                pattern_key = re.sub(r'\d+', 'N', message)  # Replace numbers with N
                pattern_key = re.sub(r'[a-f0-9]{8,}', 'HASH', pattern_key)  # Replace hashes
                
                if pattern_key not in error_counts:
                    error_counts[pattern_key] = {
                        "count": 0,
                        "first_seen": log.get("@timestamp"),
                        "last_seen": log.get("@timestamp"),
                        "sources": set(),
                        "example": message
                    }
                
                error_counts[pattern_key]["count"] += 1
                error_counts[pattern_key]["last_seen"] = log.get("@timestamp")
                error_counts[pattern_key]["sources"].add(log.get("source", "unknown"))
        
        # Convert to list format
        for pattern, data in error_counts.items():
            if data["count"] >= 3:  # Only include patterns with multiple occurrences
                error_patterns.append({
                    "pattern": pattern,
                    "count": data["count"],
                    "first_seen": data["first_seen"],
                    "last_seen": data["last_seen"],
                    "sources": list(data["sources"]),
                    "example_message": data["example"],
                    "severity": "high" if data["count"] > 10 else "medium"
                })
        
        return error_patterns
    
    async def _identify_security_events(self, logs: List[Dict]) -> List[Dict]:
        """Identify security-related events"""
        
        security_events = []
        
        # Security event patterns
        security_patterns = [
            {
                "pattern": r"authentication.*(failed|failure)",
                "type": "auth_failure",
                "severity": "medium"
            },
            {
                "pattern": r"(login|access).*(denied|blocked)",
                "type": "access_denied",
                "severity": "medium"
            },
            {
                "pattern": r"(sql injection|xss|csrf)",
                "type": "attack_attempt",
                "severity": "high"
            },
            {
                "pattern": r"unusual.*pattern",
                "type": "anomalous_behavior",
                "severity": "medium"
            },
            {
                "pattern": r"privilege.*escalation",
                "type": "privilege_escalation",
                "severity": "high"
            }
        ]
        
        for log in logs:
            message = log.get("message", "").lower()
            
            for pattern_info in security_patterns:
                if re.search(pattern_info["pattern"], message):
                    security_events.append({
                        "type": pattern_info["type"],
                        "severity": pattern_info["severity"],
                        "message": log.get("message"),
                        "timestamp": log.get("@timestamp"),
                        "source": log.get("source"),
                        "user": log.get("user"),
                        "ip": log.get("ip"),
                        "pattern_matched": pattern_info["pattern"]
                    })
        
        return security_events
    
    async def _extract_performance_metrics(self, logs: List[Dict]) -> Dict:
        """Extract performance metrics from logs"""
        
        metrics = {
            "response_times": [],
            "error_rate": 0.0,
            "throughput": 0.0,
            "memory_usage": [],
            "cpu_usage": []
        }
        
        total_requests = 0
        error_count = 0
        
        for log in logs:
            # Extract response times
            if "response_time" in log:
                metrics["response_times"].append(float(log["response_time"]))
            
            # Count requests and errors
            if log.get("level") in ["INFO", "ERROR", "WARN"]:
                total_requests += 1
                if log.get("level") == "ERROR":
                    error_count += 1
            
            # Extract resource usage
            if "memory_percent" in log:
                metrics["memory_usage"].append(float(log["memory_percent"]))
            
            if "cpu_percent" in log:
                metrics["cpu_usage"].append(float(log["cpu_percent"]))
        
        # Calculate derived metrics
        if total_requests > 0:
            metrics["error_rate"] = error_count / total_requests
        
        if metrics["response_times"]:
            metrics["avg_response_time"] = np.mean(metrics["response_times"])
            metrics["p95_response_time"] = np.percentile(metrics["response_times"], 95)
        
        return metrics
    
    async def _find_temporal_correlations(self, logs: List[Dict]) -> List[Dict]:
        """Find temporal correlations between events"""
        
        correlations = []
        
        # Group logs by time windows (5-minute windows)
        time_windows = {}
        
        for log in logs:
            try:
                timestamp = datetime.fromisoformat(log.get("@timestamp", "").replace('Z', '+00:00'))
                window_key = timestamp.replace(minute=timestamp.minute // 5 * 5, second=0, microsecond=0)
                
                if window_key not in time_windows:
                    time_windows[window_key] = []
                
                time_windows[window_key].append(log)
            except:
                continue
        
        # Look for correlations within time windows
        for window_time, window_logs in time_windows.items():
            if len(window_logs) < 2:
                continue
            
            # Group by event types
            event_types = {}
            for log in window_logs:
                event_type = f"{log.get('level', 'INFO')}_{log.get('source', 'unknown')}"
                if event_type not in event_types:
                    event_types[event_type] = 0
                event_types[event_type] += 1
            
            # Find co-occurring events
            if len(event_types) >= 2:
                event_list = list(event_types.keys())
                for i in range(len(event_list)):
                    for j in range(i + 1, len(event_list)):
                        correlations.append({
                            "time_window": window_time.isoformat(),
                            "event_a": event_list[i],
                            "event_b": event_list[j],
                            "count_a": event_types[event_list[i]],
                            "count_b": event_types[event_list[j]],
                            "correlation_strength": min(event_types[event_list[i]], event_types[event_list[j]]) / max(event_types[event_list[i]], event_types[event_list[j]])
                        })
        
        # Filter for strong correlations
        strong_correlations = [c for c in correlations if c["correlation_strength"] > 0.5]
        
        return strong_correlations[:10]  # Return top 10
    
    async def _analyze_cross_service_patterns(self, logs: List[Dict]) -> List[Dict]:
        """Analyze patterns across different services"""
        
        patterns = []
        
        # Group logs by service
        service_logs = {}
        for log in logs:
            service = log.get("source", "unknown")
            if service not in service_logs:
                service_logs[service] = []
            service_logs[service].append(log)
        
        # Look for similar patterns across services
        if len(service_logs) >= 2:
            services = list(service_logs.keys())
            
            for i in range(len(services)):
                for j in range(i + 1, len(services)):
                    service_a = services[i]
                    service_b = services[j]
                    
                    # Count error patterns in each service
                    errors_a = len([log for log in service_logs[service_a] if log.get("level") == "ERROR"])
                    errors_b = len([log for log in service_logs[service_b] if log.get("level") == "ERROR"])
                    
                    if errors_a > 0 and errors_b > 0:
                        patterns.append({
                            "pattern_type": "cross_service_errors",
                            "service_a": service_a,
                            "service_b": service_b,
                            "error_count_a": errors_a,
                            "error_count_b": errors_b,
                            "correlation_score": min(errors_a, errors_b) / max(errors_a, errors_b),
                            "description": f"Correlated errors between {service_a} and {service_b}"
                        })
        
        return patterns
    
    async def _calculate_log_anomaly_score(self, logs: List[Dict]) -> float:
        """Calculate anomaly score for log patterns"""
        
        # Simple anomaly detection based on error rates and patterns
        total_logs = len(logs)
        error_logs = len([log for log in logs if log.get("level") == "ERROR"])
        
        if total_logs == 0:
            return 0.0
        
        error_rate = error_logs / total_logs
        
        # Anomaly score based on error rate (higher error rate = higher anomaly)
        anomaly_score = min(1.0, error_rate * 5)  # Scale error rate
        
        # Increase score for unusual patterns
        unique_sources = len(set(log.get("source", "unknown") for log in logs))
        if unique_sources > 10:  # Many different sources
            anomaly_score += 0.2
        
        return min(1.0, anomaly_score)
    
    async def _identify_log_threat_indicators(self, logs: List[Dict]) -> List[Dict]:
        """Identify threat indicators in logs"""
        
        indicators = []
        
        # Known threat indicators
        threat_patterns = [
            {
                "pattern": r"(\b(?:\d{1,3}\.){3}\d{1,3}\b)",
                "type": "suspicious_ip",
                "description": "IP address in logs"
            },
            {
                "pattern": r"(admin|root|administrator)",
                "type": "privileged_account_activity",
                "description": "Activity from privileged accounts"
            },
            {
                "pattern": r"(brute.?force|dictionary.?attack)",
                "type": "brute_force_attempt",
                "description": "Brute force attack indicators"
            }
        ]
        
        for log in logs:
            message = log.get("message", "").lower()
            
            for pattern_info in threat_patterns:
                matches = re.findall(pattern_info["pattern"], message, re.IGNORECASE)
                
                if matches:
                    indicators.append({
                        "type": pattern_info["type"],
                        "description": pattern_info["description"],
                        "matches": matches,
                        "log_message": log.get("message"),
                        "timestamp": log.get("@timestamp"),
                        "source": log.get("source"),
                        "severity": "medium"
                    })
        
        return indicators
    
    async def _detect_behavioral_changes(self, logs: List[Dict]) -> List[Dict]:
        """Detect behavioral changes in log patterns"""
        
        changes = []
        
        # Simple behavioral change detection based on log volumes
        # Group logs by hour
        hourly_logs = {}
        
        for log in logs:
            try:
                timestamp = datetime.fromisoformat(log.get("@timestamp", "").replace('Z', '+00:00'))
                hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
                
                if hour_key not in hourly_logs:
                    hourly_logs[hour_key] = 0
                
                hourly_logs[hour_key] += 1
            except:
                continue
        
        # Calculate baseline and detect deviations
        if len(hourly_logs) >= 2:
            log_counts = list(hourly_logs.values())
            mean_count = np.mean(log_counts)
            std_count = np.std(log_counts)
            
            for hour, count in hourly_logs.items():
                if std_count > 0:
                    z_score = abs(count - mean_count) / std_count
                    
                    if z_score > 2:  # More than 2 standard deviations
                        changes.append({
                            "type": "log_volume_anomaly",
                            "timestamp": hour.isoformat(),
                            "actual_count": count,
                            "expected_count": mean_count,
                            "z_score": z_score,
                            "severity": "high" if z_score > 3 else "medium"
                        })
        
        return changes

class MultiModalCorrelationEngine:
    """Correlates insights across all data sources"""
    
    def __init__(self):
        self.code_analyzer = CodeAnalyzer()
        self.network_analyzer = NetworkAnalyzer()
        self.log_analyzer = LogAnalyzer()
        self.ai_client = None
        self.db_pool = None
        
    async def initialize(self, config: Dict):
        """Initialize correlation engine"""
        
        logger.info("Initializing Multi-Modal Correlation Engine...")
        
        # Initialize AI client
        self.ai_client = AsyncOpenAI(api_key=config.get("openai_api_key"))
        
        # Initialize database
        database_url = config.get("database_url")
        self.db_pool = await asyncpg.create_pool(database_url, min_size=2, max_size=5)
        
        # Initialize log analyzer
        await self.log_analyzer.initialize(config.get("elasticsearch_url", "http://localhost:9200"))
        
        # Create correlation tables
        await self._create_correlation_tables()
        
        logger.info("Multi-Modal Correlation Engine initialized successfully")
    
    async def _create_correlation_tables(self):
        """Create database tables for correlation results"""
        
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS multimodal_correlations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    correlation_id VARCHAR(255) UNIQUE NOT NULL,
                    data_sources JSONB NOT NULL,
                    analysis_types JSONB NOT NULL,
                    correlation_score FLOAT NOT NULL,
                    confidence_level VARCHAR(20) NOT NULL,
                    key_insights JSONB NOT NULL,
                    attack_timeline JSONB NOT NULL,
                    attack_techniques JSONB NOT NULL,
                    affected_assets JSONB NOT NULL,
                    impact_score FLOAT NOT NULL,
                    likelihood_score FLOAT NOT NULL,
                    business_risk VARCHAR(20) NOT NULL,
                    immediate_actions JSONB NOT NULL,
                    strategic_improvements JSONB NOT NULL,
                    analyzed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    model_version VARCHAR(20) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_correlations_score 
                ON multimodal_correlations(correlation_score DESC);
                
                CREATE INDEX IF NOT EXISTS idx_correlations_analyzed 
                ON multimodal_correlations(analyzed_at);
            """)
    
    async def perform_multimodal_analysis(
        self,
        vulnerability_id: str,
        analysis_config: Dict
    ) -> MultiModalCorrelation:
        """Perform comprehensive multi-modal analysis"""
        
        start_time = datetime.now()
        correlation_id = f"multimodal_{vulnerability_id}_{int(start_time.timestamp())}"
        
        try:
            # Gather data from all sources
            analysis_results = {}
            data_sources = []
            analysis_types = []
            
            # Code analysis
            if analysis_config.get("repository_url"):
                logger.info("Starting code analysis", vulnerability_id=vulnerability_id)
                code_result = await self.code_analyzer.analyze_repository(
                    analysis_config["repository_url"],
                    analysis_config.get("branch", "main")
                )
                analysis_results["code"] = code_result
                data_sources.append(DataSourceType.CODE_REPOSITORY)
                analysis_types.append(AnalysisType.VULNERABILITY_CORRELATION)
            
            # Network analysis
            if analysis_config.get("pcap_file"):
                logger.info("Starting network analysis", vulnerability_id=vulnerability_id)
                network_result = await self.network_analyzer.analyze_traffic_capture(
                    analysis_config["pcap_file"]
                )
                analysis_results["network"] = network_result
                data_sources.append(DataSourceType.NETWORK_TRAFFIC)
                analysis_types.append(AnalysisType.ANOMALY_DETECTION)
            
            # Log analysis
            if analysis_config.get("log_sources"):
                logger.info("Starting log analysis", vulnerability_id=vulnerability_id)
                time_range = (
                    analysis_config.get("start_time", datetime.now() - timedelta(hours=24)),
                    analysis_config.get("end_time", datetime.now())
                )
                log_result = await self.log_analyzer.analyze_logs(
                    analysis_config["log_sources"],
                    time_range
                )
                analysis_results["logs"] = log_result
                data_sources.append(DataSourceType.APPLICATION_LOGS)
                analysis_types.append(AnalysisType.BEHAVIORAL_ANALYSIS)
            
            # Cross-modal correlation
            logger.info("Performing cross-modal correlation", vulnerability_id=vulnerability_id)
            correlation_result = await self._perform_cross_modal_correlation(
                analysis_results, vulnerability_id
            )
            
            # AI-enhanced analysis
            ai_insights = await self._generate_ai_insights(analysis_results, correlation_result)
            
            # Build final correlation result
            multimodal_correlation = MultiModalCorrelation(
                correlation_id=correlation_id,
                data_sources=data_sources,
                analysis_types=analysis_types,
                correlation_score=correlation_result["correlation_score"],
                confidence_level=correlation_result["confidence_level"],
                key_insights=ai_insights["key_insights"],
                attack_timeline=correlation_result["attack_timeline"],
                attack_techniques=correlation_result["attack_techniques"],
                affected_assets=correlation_result["affected_assets"],
                impact_score=correlation_result["impact_score"],
                likelihood_score=correlation_result["likelihood_score"],
                business_risk=correlation_result["business_risk"],
                immediate_actions=ai_insights["immediate_actions"],
                strategic_improvements=ai_insights["strategic_improvements"],
                analyzed_at=start_time,
                model_version="6.4.0"
            )
            
            # Store correlation result
            await self._store_correlation_result(multimodal_correlation)
            
            # Update metrics
            multimodal_analysis_total.labels(
                analysis_type="comprehensive",
                data_sources=len(data_sources)
            ).inc()
            
            if len(data_sources) >= 2:
                analysis_correlation_score.labels(
                    source_a=data_sources[0].value,
                    source_b=data_sources[1].value
                ).observe(correlation_result["correlation_score"])
            
            attack_path_discoveries.labels(
                path_complexity="high" if len(correlation_result["attack_techniques"]) > 3 else "medium",
                confidence_level=correlation_result["confidence_level"]
            ).inc()
            
            cross_correlation_insights.labels(
                insight_type="multimodal_correlation",
                confidence=correlation_result["confidence_level"]
            ).inc()
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info("Multi-modal analysis completed",
                       vulnerability_id=vulnerability_id,
                       correlation_id=correlation_id,
                       data_sources=len(data_sources),
                       correlation_score=correlation_result["correlation_score"],
                       duration=duration)
            
            return multimodal_correlation
            
        except Exception as e:
            logger.error("Multi-modal analysis failed",
                        vulnerability_id=vulnerability_id,
                        error=str(e))
            raise
    
    async def _perform_cross_modal_correlation(
        self, 
        analysis_results: Dict, 
        vulnerability_id: str
    ) -> Dict:
        """Perform cross-modal correlation analysis"""
        
        correlation_score = 0.0
        attack_timeline = []
        attack_techniques = []
        affected_assets = []
        
        # Code-Network correlation
        if "code" in analysis_results and "network" in analysis_results:
            code_result = analysis_results["code"]
            network_result = analysis_results["network"]
            
            # Correlate API endpoints with network traffic
            api_endpoints = code_result.api_endpoints
            suspicious_flows = network_result.suspicious_flows
            
            for endpoint in api_endpoints:
                for flow in suspicious_flows:
                    if str(endpoint.get("path", "")).strip("/") in str(flow.get("type", "")):
                        correlation_score += 0.2
                        attack_techniques.append("API_exploitation")
                        affected_assets.append(f"API_endpoint_{endpoint.get('path')}")
            
            # Correlate vulnerabilities with network anomalies
            if code_result.vulnerabilities and network_result.threat_indicators:
                correlation_score += 0.3
                attack_techniques.append("network_reconnaissance")
        
        # Code-Log correlation
        if "code" in analysis_results and "logs" in analysis_results:
            code_result = analysis_results["code"]
            log_result = analysis_results["logs"]
            
            # Correlate code vulnerabilities with security events
            vuln_types = [v.get("rule_id", "") for v in code_result.vulnerabilities]
            security_events = log_result.security_events
            
            for vuln_type in vuln_types:
                for event in security_events:
                    if any(keyword in event.get("message", "").lower() 
                          for keyword in ["injection", "xss", "auth"]):
                        correlation_score += 0.25
                        attack_techniques.append("exploitation_attempt")
                        
                        # Add to timeline
                        attack_timeline.append({
                            "timestamp": event.get("timestamp"),
                            "event": "Security event correlates with code vulnerability",
                            "source": "log_analysis",
                            "details": event.get("message")
                        })
        
        # Network-Log correlation
        if "network" in analysis_results and "logs" in analysis_results:
            network_result = analysis_results["network"]
            log_result = analysis_results["logs"]
            
            # Correlate network anomalies with log events
            if network_result.suspicious_flows and log_result.security_events:
                correlation_score += 0.2
                attack_techniques.append("lateral_movement")
                
                # Time-based correlation
                for flow in network_result.suspicious_flows:
                    for event in log_result.security_events:
                        # Simple time correlation (would be more sophisticated in production)
                        attack_timeline.append({
                            "timestamp": event.get("timestamp"),
                            "event": f"Network anomaly correlated with {event.get('type')}",
                            "source": "network_log_correlation",
                            "confidence": 0.7
                        })
        
        # Determine confidence level
        confidence_level = "high" if correlation_score > 0.7 else "medium" if correlation_score > 0.4 else "low"
        
        # Calculate impact and likelihood scores
        impact_score = min(1.0, correlation_score * 1.2)  # Slightly higher than correlation
        likelihood_score = correlation_score * 0.8  # Slightly lower than correlation
        
        # Determine business risk
        if impact_score > 0.8:
            business_risk = "critical"
        elif impact_score > 0.6:
            business_risk = "high"
        elif impact_score > 0.3:
            business_risk = "medium"
        else:
            business_risk = "low"
        
        # Sort timeline by timestamp
        attack_timeline.sort(key=lambda x: x.get("timestamp", ""))
        
        return {
            "correlation_score": min(1.0, correlation_score),
            "confidence_level": confidence_level,
            "attack_timeline": attack_timeline,
            "attack_techniques": list(set(attack_techniques)),  # Remove duplicates
            "affected_assets": list(set(affected_assets)),
            "impact_score": impact_score,
            "likelihood_score": likelihood_score,
            "business_risk": business_risk
        }
    
    async def _generate_ai_insights(self, analysis_results: Dict, correlation_result: Dict) -> Dict:
        """Generate AI-enhanced insights from correlation"""
        
        # Build context for AI analysis
        context = {
            "correlation_score": correlation_result["correlation_score"],
            "attack_techniques": correlation_result["attack_techniques"],
            "data_sources": list(analysis_results.keys()),
            "findings_summary": {}
        }
        
        # Summarize findings from each source
        if "code" in analysis_results:
            code_result = analysis_results["code"]
            context["findings_summary"]["code"] = {
                "vulnerabilities_count": len(code_result.vulnerabilities),
                "security_score": code_result.security_score,
                "high_severity_vulns": len([v for v in code_result.vulnerabilities if v.get("severity") == "high"])
            }
        
        if "network" in analysis_results:
            network_result = analysis_results["network"]
            context["findings_summary"]["network"] = {
                "suspicious_flows": len(network_result.suspicious_flows),
                "anomaly_confidence": network_result.anomaly_confidence,
                "threat_indicators": len(network_result.threat_indicators)
            }
        
        if "logs" in analysis_results:
            log_result = analysis_results["logs"]
            context["findings_summary"]["logs"] = {
                "security_events": len(log_result.security_events),
                "anomaly_score": log_result.anomaly_score,
                "error_patterns": len(log_result.error_patterns)
            }
        
        # Generate AI insights
        prompt = self._build_ai_insights_prompt(context)
        
        try:
            response = await self.ai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cybersecurity expert analyzing correlated findings from multiple data sources. Provide actionable insights and recommendations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            ai_insights = json.loads(response.choices[0].message.content)
            
            return {
                "key_insights": ai_insights.get("key_insights", []),
                "immediate_actions": ai_insights.get("immediate_actions", []),
                "strategic_improvements": ai_insights.get("strategic_improvements", [])
            }
            
        except Exception as e:
            logger.error("Failed to generate AI insights", error=str(e))
            
            # Return default insights
            return {
                "key_insights": [
                    f"Multi-modal correlation score: {correlation_result['correlation_score']:.2f}",
                    f"Attack techniques identified: {', '.join(correlation_result['attack_techniques'])}",
                    f"Business risk level: {correlation_result['business_risk']}"
                ],
                "immediate_actions": [
                    "Review high-correlation findings immediately",
                    "Implement monitoring for identified attack techniques",
                    "Validate findings through manual investigation"
                ],
                "strategic_improvements": [
                    "Enhance cross-domain monitoring capabilities",
                    "Improve correlation algorithms based on findings",
                    "Develop automated response playbooks"
                ]
            }
    
    def _build_ai_insights_prompt(self, context: Dict) -> str:
        """Build prompt for AI insights generation"""
        
        return f"""
Analyze this multi-modal security correlation and provide insights:

CORRELATION ANALYSIS:
- Correlation Score: {context['correlation_score']:.2f}
- Attack Techniques: {', '.join(context['attack_techniques'])}
- Data Sources: {', '.join(context['data_sources'])}

FINDINGS SUMMARY:
{json.dumps(context['findings_summary'], indent=2)}

Please provide analysis in the following JSON format:
{{
  "key_insights": [
    "Most important insight from correlation",
    "Second key finding",
    "Third critical observation"
  ],
  "immediate_actions": [
    "Urgent action 1",
    "Urgent action 2", 
    "Urgent action 3"
  ],
  "strategic_improvements": [
    "Long-term improvement 1",
    "Strategic enhancement 2",
    "Process improvement 3"
  ]
}}

Focus on:
1. Cross-domain attack patterns
2. Risk prioritization
3. Actionable recommendations
4. Prevention strategies
"""
    
    async def _store_correlation_result(self, correlation: MultiModalCorrelation):
        """Store correlation result in database"""
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO multimodal_correlations
                    (correlation_id, data_sources, analysis_types, correlation_score,
                     confidence_level, key_insights, attack_timeline, attack_techniques,
                     affected_assets, impact_score, likelihood_score, business_risk,
                     immediate_actions, strategic_improvements, analyzed_at, model_version)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                """,
                correlation.correlation_id,
                json.dumps([ds.value for ds in correlation.data_sources]),
                json.dumps([at.value for at in correlation.analysis_types]),
                correlation.correlation_score,
                correlation.confidence_level,
                json.dumps(correlation.key_insights),
                json.dumps(correlation.attack_timeline),
                json.dumps(correlation.attack_techniques),
                json.dumps(correlation.affected_assets),
                correlation.impact_score,
                correlation.likelihood_score,
                correlation.business_risk,
                json.dumps(correlation.immediate_actions),
                json.dumps(correlation.strategic_improvements),
                correlation.analyzed_at,
                correlation.model_version)
                
        except Exception as e:
            logger.error("Failed to store correlation result", error=str(e))
    
    async def get_correlation_statistics(self) -> Dict:
        """Get comprehensive correlation statistics"""
        
        try:
            async with self.db_pool.acquire() as conn:
                stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_correlations,
                        AVG(correlation_score) as avg_correlation_score,
                        COUNT(*) FILTER (WHERE confidence_level = 'high') as high_confidence_count,
                        COUNT(*) FILTER (WHERE business_risk = 'critical') as critical_risk_count,
                        AVG(impact_score) as avg_impact_score,
                        AVG(likelihood_score) as avg_likelihood_score
                    FROM multimodal_correlations
                    WHERE analyzed_at >= NOW() - INTERVAL '30 days'
                """)
                
                return {
                    "total_correlations": stats['total_correlations'],
                    "average_correlation_score": float(stats['avg_correlation_score'] or 0),
                    "high_confidence_correlations": stats['high_confidence_count'],
                    "critical_risk_correlations": stats['critical_risk_count'],
                    "average_impact_score": float(stats['avg_impact_score'] or 0),
                    "average_likelihood_score": float(stats['avg_likelihood_score'] or 0),
                    "analysis_capabilities": [
                        "code_repository_analysis",
                        "network_traffic_analysis", 
                        "log_correlation_analysis",
                        "ai_enhanced_insights"
                    ],
                    "generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error("Failed to get correlation statistics", error=str(e))
            return {"error": str(e)}

async def main():
    """Main multi-modal analysis service"""
    
    # Start Prometheus metrics server
    start_http_server(8013)
    
    # Initialize correlation engine
    config = {
        "database_url": os.getenv("DATABASE_URL", 
                                 "postgresql://xorb:xorb_secure_2024@postgres:5432/xorb_ptaas"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "elasticsearch_url": os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    }
    
    engine = MultiModalCorrelationEngine()
    await engine.initialize(config)
    
    logger.info("🔗 Xorb Multi-Modal AI Analysis Engine started",
               service_version="6.4.0",
               features=["code_analysis", "network_analysis", "log_correlation", 
                        "cross_modal_correlation", "ai_insights"])
    
    try:
        # Keep service running
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutting down multi-modal analysis engine")

if __name__ == "__main__":
    asyncio.run(main())