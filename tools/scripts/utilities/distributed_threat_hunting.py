#!/usr/bin/env python3
"""
XORB Distributed Threat Hunting System

Advanced threat hunting platform with:
- Distributed agent coordination across federated nodes
- Real-time threat intelligence correlation
- Behavioral anomaly detection with ML
- Automated threat response and containment
- Cross-node threat pattern recognition
- Quantum-safe communication between hunters

Author: XORB Platform Team
Version: 2.1.0
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import numpy as np
from pathlib import Path
import aioredis
import asyncpg
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import httpx
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ThreatSeverity(Enum):
    """Threat severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ThreatCategory(Enum):
    """Threat categories for classification"""
    MALWARE = "malware"
    APT = "apt"
    RANSOMWARE = "ransomware"
    DATA_EXFILTRATION = "data_exfiltration"
    LATERAL_MOVEMENT = "lateral_movement"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    RECONNAISSANCE = "reconnaissance"
    COMMAND_CONTROL = "command_control"
    PERSISTENCE = "persistence"
    DEFENSE_EVASION = "defense_evasion"

class HuntingTaskStatus(Enum):
    """Status of hunting tasks"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ThreatIndicator:
    """Represents a threat indicator"""
    id: str
    type: str  # ip, domain, hash, yara_rule, etc.
    value: str
    confidence: float  # 0.0 to 1.0
    severity: ThreatSeverity
    category: ThreatCategory
    source: str
    first_seen: datetime
    last_seen: datetime
    metadata: Dict[str, Any]
    ttl: Optional[int] = None  # Time to live in seconds

@dataclass
class ThreatEvent:
    """Represents a detected threat event"""
    id: str
    node_id: str
    timestamp: datetime
    severity: ThreatSeverity
    category: ThreatCategory
    indicators: List[ThreatIndicator]
    raw_data: Dict[str, Any]
    confidence: float
    false_positive_probability: float
    metadata: Dict[str, Any]
    remediation_actions: List[str]

@dataclass
class HuntingTask:
    """Represents a distributed hunting task"""
    id: str
    name: str
    description: str
    query: str  # Hunting query (KQL, Sigma, custom)
    target_nodes: List[str]
    status: HuntingTaskStatus
    created_by: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    results: Dict[str, Any]
    metadata: Dict[str, Any]
    priority: int = 5  # 1-10, 10 being highest

@dataclass
class HuntingAgent:
    """Represents a threat hunting agent"""
    id: str
    node_id: str
    capabilities: List[str]
    status: str
    last_heartbeat: datetime
    load_average: float
    active_tasks: int
    max_concurrent_tasks: int
    version: str

class DistributedThreatHunter:
    """Main distributed threat hunting coordinator"""
    
    def __init__(
        self,
        node_id: str,
        redis_url: str = "redis://localhost:6379/7",
        postgres_dsn: str = "postgresql://xorb:password@localhost:5432/xorb",
        threat_intel_feeds: Optional[List[str]] = None
    ):
        self.node_id = node_id
        self.redis_url = redis_url
        self.postgres_dsn = postgres_dsn
        self.threat_intel_feeds = threat_intel_feeds or []
        
        # Connection pools
        self.redis_pool = None
        self.postgres_pool = None
        
        # Agent management
        self.agents: Dict[str, HuntingAgent] = {}
        self.active_tasks: Dict[str, HuntingTask] = {}
        
        # Threat intelligence cache
        self.threat_indicators: Dict[str, ThreatIndicator] = {}
        self.threat_patterns: Dict[str, Any] = {}
        
        # ML models for anomaly detection
        self.anomaly_detector = None
        self.false_positive_classifier = None
        
        # Configuration
        self.config = {
            "max_concurrent_hunts": 50,
            "heartbeat_interval": 30,
            "task_timeout": 3600,  # 1 hour
            "threat_intel_refresh": 300,  # 5 minutes
            "anomaly_threshold": 0.8,
            "false_positive_threshold": 0.3
        }
    
    async def initialize(self):
        """Initialize the threat hunting system"""
        try:
            # Initialize Redis connection
            self.redis_pool = aioredis.from_url(self.redis_url)
            await self.redis_pool.ping()
            
            # Initialize PostgreSQL connection
            self.postgres_pool = await asyncpg.create_pool(self.postgres_dsn)
            
            # Create database tables
            await self._create_tables()
            
            # Initialize ML models
            await self._initialize_ml_models()
            
            # Load threat intelligence
            await self._load_threat_intelligence()
            
            # Start background tasks
            asyncio.create_task(self._heartbeat_monitor())
            asyncio.create_task(self._threat_intel_updater())
            asyncio.create_task(self._task_scheduler())
            
            logger.info(f"Distributed threat hunter initialized for node {self.node_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize threat hunter: {e}")
            raise
    
    async def _create_tables(self):
        """Create database tables for threat hunting"""
        create_tables_sql = """
        CREATE TABLE IF NOT EXISTS threat_indicators (
            id VARCHAR(64) PRIMARY KEY,
            type VARCHAR(50) NOT NULL,
            value TEXT NOT NULL,
            confidence FLOAT NOT NULL,
            severity VARCHAR(20) NOT NULL,
            category VARCHAR(50) NOT NULL,
            source VARCHAR(100) NOT NULL,
            first_seen TIMESTAMP DEFAULT NOW(),
            last_seen TIMESTAMP DEFAULT NOW(),
            metadata JSONB,
            ttl INTEGER,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE TABLE IF NOT EXISTS threat_events (
            id VARCHAR(64) PRIMARY KEY,
            node_id VARCHAR(100) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            severity VARCHAR(20) NOT NULL,
            category VARCHAR(50) NOT NULL,
            indicators JSONB NOT NULL,
            raw_data JSONB NOT NULL,
            confidence FLOAT NOT NULL,
            false_positive_probability FLOAT NOT NULL,
            metadata JSONB,
            remediation_actions JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE TABLE IF NOT EXISTS hunting_tasks (
            id VARCHAR(64) PRIMARY KEY,
            name VARCHAR(200) NOT NULL,
            description TEXT,
            query TEXT NOT NULL,
            target_nodes JSONB NOT NULL,
            status VARCHAR(20) NOT NULL,
            created_by VARCHAR(100) NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            results JSONB,
            metadata JSONB,
            priority INTEGER DEFAULT 5
        );
        
        CREATE TABLE IF NOT EXISTS hunting_agents (
            id VARCHAR(64) PRIMARY KEY,
            node_id VARCHAR(100) NOT NULL,
            capabilities JSONB NOT NULL,
            status VARCHAR(20) NOT NULL,
            last_heartbeat TIMESTAMP DEFAULT NOW(),
            load_average FLOAT DEFAULT 0,
            active_tasks INTEGER DEFAULT 0,
            max_concurrent_tasks INTEGER DEFAULT 10,
            version VARCHAR(50),
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE TABLE IF NOT EXISTS threat_patterns (
            id VARCHAR(64) PRIMARY KEY,
            name VARCHAR(200) NOT NULL,
            pattern_type VARCHAR(50) NOT NULL,
            pattern_data JSONB NOT NULL,
            confidence FLOAT NOT NULL,
            created_by VARCHAR(100),
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_threat_events_timestamp ON threat_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_threat_events_severity ON threat_events(severity);
        CREATE INDEX IF NOT EXISTS idx_threat_events_node_id ON threat_events(node_id);
        CREATE INDEX IF NOT EXISTS idx_hunting_tasks_status ON hunting_tasks(status);
        CREATE INDEX IF NOT EXISTS idx_hunting_agents_node_id ON hunting_agents(node_id);
        """
        
        async with self.postgres_pool.acquire() as conn:
            await conn.execute(create_tables_sql)
    
    async def _initialize_ml_models(self):
        """Initialize ML models for anomaly detection"""
        try:
            # In a real implementation, these would be proper ML models
            # For now, we'll create placeholder implementations
            self.anomaly_detector = SimpleAnomalyDetector()
            self.false_positive_classifier = FalsePositiveClassifier()
            
            logger.info("ML models initialized for threat hunting")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
            raise
    
    async def _load_threat_intelligence(self):
        """Load threat intelligence from various feeds"""
        try:
            # Load from database first
            async with self.postgres_pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM threat_indicators WHERE ttl IS NULL OR ttl > $1", int(time.time()))
                
                for row in rows:
                    indicator = ThreatIndicator(
                        id=row['id'],
                        type=row['type'],
                        value=row['value'],
                        confidence=row['confidence'],
                        severity=ThreatSeverity(row['severity']),
                        category=ThreatCategory(row['category']),
                        source=row['source'],
                        first_seen=row['first_seen'],
                        last_seen=row['last_seen'],
                        metadata=row['metadata'] or {},
                        ttl=row['ttl']
                    )
                    self.threat_indicators[indicator.id] = indicator
            
            # Load from external feeds
            for feed_url in self.threat_intel_feeds:
                await self._fetch_threat_feed(feed_url)
            
            logger.info(f"Loaded {len(self.threat_indicators)} threat indicators")
            
        except Exception as e:
            logger.error(f"Failed to load threat intelligence: {e}")
    
    async def _fetch_threat_feed(self, feed_url: str):
        """Fetch threat intelligence from external feed"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(feed_url, timeout=30)
                response.raise_for_status()
                
                # Parse feed (assuming JSON format)
                feed_data = response.json()
                
                for item in feed_data.get('indicators', []):
                    indicator = self._parse_threat_indicator(item)
                    if indicator:
                        self.threat_indicators[indicator.id] = indicator
                        await self._store_threat_indicator(indicator)
                        
        except Exception as e:
            logger.error(f"Failed to fetch threat feed {feed_url}: {e}")
    
    def _parse_threat_indicator(self, data: Dict[str, Any]) -> Optional[ThreatIndicator]:
        """Parse threat indicator from feed data"""
        try:
            indicator_id = hashlib.sha256(f"{data['type']}:{data['value']}".encode()).hexdigest()
            
            return ThreatIndicator(
                id=indicator_id,
                type=data['type'],
                value=data['value'],
                confidence=data.get('confidence', 0.5),
                severity=ThreatSeverity(data.get('severity', 'medium')),
                category=ThreatCategory(data.get('category', 'malware')),
                source=data.get('source', 'unknown'),
                first_seen=datetime.fromisoformat(data.get('first_seen', datetime.utcnow().isoformat())),
                last_seen=datetime.fromisoformat(data.get('last_seen', datetime.utcnow().isoformat())),
                metadata=data.get('metadata', {}),
                ttl=data.get('ttl')
            )
            
        except Exception as e:
            logger.error(f"Failed to parse threat indicator: {e}")
            return None
    
    async def _store_threat_indicator(self, indicator: ThreatIndicator):
        """Store threat indicator in database"""
        try:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO threat_indicators 
                    (id, type, value, confidence, severity, category, source, first_seen, last_seen, metadata, ttl)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (id) DO UPDATE SET
                        confidence = $4,
                        last_seen = $9,
                        metadata = $10
                """, 
                    indicator.id, indicator.type, indicator.value, indicator.confidence,
                    indicator.severity.value, indicator.category.value, indicator.source,
                    indicator.first_seen, indicator.last_seen, json.dumps(indicator.metadata),
                    indicator.ttl
                )
                
        except Exception as e:
            logger.error(f"Failed to store threat indicator: {e}")
    
    async def register_agent(self, agent: HuntingAgent) -> bool:
        """Register a new hunting agent"""
        try:
            self.agents[agent.id] = agent
            
            # Store in database
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO hunting_agents 
                    (id, node_id, capabilities, status, last_heartbeat, load_average, active_tasks, max_concurrent_tasks, version)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (id) DO UPDATE SET
                        status = $4,
                        last_heartbeat = $5,
                        load_average = $6,
                        active_tasks = $7
                """,
                    agent.id, agent.node_id, json.dumps(agent.capabilities), agent.status,
                    agent.last_heartbeat, agent.load_average, agent.active_tasks,
                    agent.max_concurrent_tasks, agent.version
                )
            
            # Announce agent registration
            await self.redis_pool.publish(
                "threat_hunting:agent_events",
                json.dumps({
                    "type": "agent_registered",
                    "agent_id": agent.id,
                    "node_id": agent.node_id,
                    "timestamp": datetime.utcnow().isoformat()
                })
            )
            
            logger.info(f"Registered hunting agent {agent.id} from node {agent.node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent.id}: {e}")
            return False
    
    async def create_hunting_task(
        self,
        name: str,
        description: str,
        query: str,
        target_nodes: Optional[List[str]] = None,
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new distributed hunting task"""
        
        task_id = hashlib.sha256(f"{name}:{query}:{time.time()}".encode()).hexdigest()
        
        task = HuntingTask(
            id=task_id,
            name=name,
            description=description,
            query=query,
            target_nodes=target_nodes or list(self.agents.keys()),
            status=HuntingTaskStatus.PENDING,
            created_by=self.node_id,
            created_at=datetime.utcnow(),
            started_at=None,
            completed_at=None,
            results={},
            metadata=metadata or {},
            priority=priority
        )
        
        # Store task
        self.active_tasks[task_id] = task
        
        try:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO hunting_tasks 
                    (id, name, description, query, target_nodes, status, created_by, created_at, priority, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                    task.id, task.name, task.description, task.query,
                    json.dumps(task.target_nodes), task.status.value, task.created_by,
                    task.created_at, task.priority, json.dumps(task.metadata)
                )
            
            # Notify task scheduler
            await self.redis_pool.publish(
                "threat_hunting:task_events",
                json.dumps({
                    "type": "task_created",
                    "task_id": task_id,
                    "priority": priority,
                    "timestamp": datetime.utcnow().isoformat()
                })
            )
            
            logger.info(f"Created hunting task {task_id}: {name}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to create hunting task: {e}")
            raise
    
    async def execute_hunting_task(self, task_id: str) -> Dict[str, Any]:
        """Execute a hunting task across distributed agents"""
        
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.active_tasks[task_id]
        task.status = HuntingTaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        
        logger.info(f"Executing hunting task {task_id}: {task.name}")
        
        try:
            # Select capable agents
            selected_agents = await self._select_agents_for_task(task)
            
            if not selected_agents:
                raise Exception("No suitable agents available for task")
            
            # Distribute task to agents
            agent_results = await self._distribute_task_to_agents(task, selected_agents)
            
            # Correlate results from all agents
            correlated_results = await self._correlate_hunting_results(agent_results)
            
            # Apply ML analysis for anomaly detection
            analyzed_results = await self._analyze_hunting_results(correlated_results)
            
            # Generate threat events from significant findings
            threat_events = await self._generate_threat_events(analyzed_results, task)
            
            # Update task with results
            task.results = {
                "agent_results": agent_results,
                "correlated_results": correlated_results,
                "analyzed_results": analyzed_results,
                "threat_events": [asdict(event) for event in threat_events],
                "summary": {
                    "agents_used": len(selected_agents),
                    "total_findings": len(analyzed_results.get("findings", [])),
                    "threat_events_generated": len(threat_events),
                    "execution_time": (datetime.utcnow() - task.started_at).total_seconds()
                }
            }
            
            task.status = HuntingTaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            
            # Update database
            await self._update_task_status(task)
            
            # Store threat events
            for event in threat_events:
                await self._store_threat_event(event)
            
            logger.info(f"Completed hunting task {task_id} with {len(threat_events)} threat events")
            
            return task.results
            
        except Exception as e:
            task.status = HuntingTaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            task.results = {"error": str(e)}
            
            await self._update_task_status(task)
            
            logger.error(f"Failed to execute hunting task {task_id}: {e}")
            raise
    
    async def _select_agents_for_task(self, task: HuntingTask) -> List[HuntingAgent]:
        """Select the best agents for a hunting task"""
        suitable_agents = []
        
        for agent_id in task.target_nodes:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                
                # Check if agent is healthy and available
                if (agent.status == "active" and 
                    agent.active_tasks < agent.max_concurrent_tasks and
                    (datetime.utcnow() - agent.last_heartbeat).total_seconds() < 60):
                    
                    suitable_agents.append(agent)
        
        # Sort by load average (prefer less loaded agents)
        suitable_agents.sort(key=lambda a: a.load_average)
        
        return suitable_agents
    
    async def _distribute_task_to_agents(
        self, 
        task: HuntingTask, 
        agents: List[HuntingAgent]
    ) -> Dict[str, Any]:
        """Distribute hunting task to selected agents"""
        
        agent_results = {}
        
        # Create agent tasks
        agent_tasks = []
        for agent in agents:
            agent_tasks.append(self._execute_on_agent(agent, task))
        
        # Execute tasks concurrently with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*agent_tasks, return_exceptions=True),
                timeout=self.config["task_timeout"]
            )
            
            for i, result in enumerate(results):
                agent_id = agents[i].id
                if isinstance(result, Exception):
                    agent_results[agent_id] = {"error": str(result)}
                else:
                    agent_results[agent_id] = result
                    
        except asyncio.TimeoutError:
            logger.warning(f"Task {task.id} timed out")
            for i, agent in enumerate(agents):
                if agent.id not in agent_results:
                    agent_results[agent.id] = {"error": "timeout"}
        
        return agent_results
    
    async def _execute_on_agent(self, agent: HuntingAgent, task: HuntingTask) -> Dict[str, Any]:
        """Execute hunting task on a specific agent"""
        
        # Simulate agent execution - in reality would send to agent via secure channel
        try:
            # Simulate hunting query execution
            await asyncio.sleep(np.random.uniform(1, 5))  # Simulate processing time
            
            # Generate mock results based on task type
            results = await self._simulate_agent_hunting(agent, task)
            
            return {
                "agent_id": agent.id,
                "node_id": agent.node_id,
                "status": "completed",
                "findings": results,
                "execution_time": np.random.uniform(1, 10),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "agent_id": agent.id,
                "node_id": agent.node_id,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _simulate_agent_hunting(self, agent: HuntingAgent, task: HuntingTask) -> List[Dict[str, Any]]:
        """Simulate hunting results from an agent"""
        
        findings = []
        
        # Generate realistic findings based on task query
        if "malware" in task.query.lower():
            if np.random.random() < 0.3:  # 30% chance of finding something
                findings.append({
                    "type": "suspicious_process",
                    "details": {
                        "process_name": "suspicious.exe",
                        "pid": np.random.randint(1000, 9999),
                        "parent_process": "explorer.exe",
                        "command_line": "suspicious.exe --connect 192.168.1.100",
                        "hash": hashlib.sha256(f"malware{np.random.randint(1, 1000)}".encode()).hexdigest()
                    },
                    "confidence": np.random.uniform(0.7, 0.95),
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": f"agent_{agent.id}"
                })
        
        elif "network" in task.query.lower():
            if np.random.random() < 0.4:  # 40% chance of network findings
                findings.append({
                    "type": "suspicious_network_connection",
                    "details": {
                        "source_ip": f"192.168.1.{np.random.randint(1, 254)}",
                        "destination_ip": f"203.0.113.{np.random.randint(1, 254)}",
                        "port": np.random.choice([4444, 8080, 443, 53]),
                        "protocol": "TCP",
                        "bytes_transferred": np.random.randint(1024, 10485760)
                    },
                    "confidence": np.random.uniform(0.6, 0.9),
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": f"agent_{agent.id}"
                })
        
        elif "user" in task.query.lower():
            if np.random.random() < 0.2:  # 20% chance of user behavior findings
                findings.append({
                    "type": "anomalous_user_behavior",
                    "details": {
                        "username": f"user{np.random.randint(1, 100)}",
                        "login_time": datetime.utcnow().isoformat(),
                        "source_ip": f"10.0.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                        "anomaly_score": np.random.uniform(0.7, 0.95),
                        "unusual_patterns": ["off-hours login", "multiple failed attempts"]
                    },
                    "confidence": np.random.uniform(0.5, 0.8),
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": f"agent_{agent.id}"
                })
        
        return findings
    
    async def _correlate_hunting_results(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Correlate results from multiple agents to identify patterns"""
        
        all_findings = []
        correlation_groups = {}
        
        # Collect all findings
        for agent_id, result in agent_results.items():
            if result.get("status") == "completed":
                for finding in result.get("findings", []):
                    finding["agent_id"] = agent_id
                    all_findings.append(finding)
        
        # Group similar findings
        for finding in all_findings:
            correlation_key = self._generate_correlation_key(finding)
            if correlation_key not in correlation_groups:
                correlation_groups[correlation_key] = []
            correlation_groups[correlation_key].append(finding)
        
        # Identify significant correlations
        significant_correlations = []
        for key, findings in correlation_groups.items():
            if len(findings) > 1:  # Multiple agents found similar things
                correlation_confidence = min(1.0, len(findings) * 0.3)
                
                significant_correlations.append({
                    "correlation_key": key,
                    "finding_count": len(findings),
                    "agents_involved": list(set(f["agent_id"] for f in findings)),
                    "confidence": correlation_confidence,
                    "findings": findings
                })
        
        return {
            "total_findings": len(all_findings),
            "correlation_groups": len(correlation_groups),
            "significant_correlations": significant_correlations,
            "unique_agents": len(set(f["agent_id"] for f in all_findings))
        }
    
    def _generate_correlation_key(self, finding: Dict[str, Any]) -> str:
        """Generate a correlation key for similar findings"""
        
        # Extract key attributes for correlation
        finding_type = finding.get("type", "unknown")
        
        if finding_type == "suspicious_process":
            process_name = finding.get("details", {}).get("process_name", "")
            return f"process:{process_name}"
        
        elif finding_type == "suspicious_network_connection":
            dest_ip = finding.get("details", {}).get("destination_ip", "")
            port = finding.get("details", {}).get("port", "")
            return f"network:{dest_ip}:{port}"
        
        elif finding_type == "anomalous_user_behavior":
            username = finding.get("details", {}).get("username", "")
            return f"user:{username}"
        
        return f"other:{finding_type}"
    
    async def _analyze_hunting_results(self, correlated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply ML analysis to hunting results"""
        
        analyzed_findings = []
        
        for correlation in correlated_results.get("significant_correlations", []):
            # Apply anomaly detection
            anomaly_score = await self.anomaly_detector.analyze(correlation)
            
            # Apply false positive classification
            fp_probability = await self.false_positive_classifier.classify(correlation)
            
            # Determine final confidence
            final_confidence = correlation["confidence"] * (1 - fp_probability) * anomaly_score
            
            if final_confidence > self.config["anomaly_threshold"]:
                analyzed_findings.append({
                    "correlation": correlation,
                    "anomaly_score": anomaly_score,
                    "false_positive_probability": fp_probability,
                    "final_confidence": final_confidence,
                    "recommended_actions": self._generate_recommendations(correlation, final_confidence)
                })
        
        return {
            "analyzed_findings": analyzed_findings,
            "high_confidence_findings": [f for f in analyzed_findings if f["final_confidence"] > 0.9],
            "analysis_summary": {
                "total_analyzed": len(analyzed_findings),
                "high_confidence": len([f for f in analyzed_findings if f["final_confidence"] > 0.9]),
                "medium_confidence": len([f for f in analyzed_findings if 0.7 <= f["final_confidence"] <= 0.9]),
                "low_confidence": len([f for f in analyzed_findings if f["final_confidence"] < 0.7])
            }
        }
    
    def _generate_recommendations(self, correlation: Dict[str, Any], confidence: float) -> List[str]:
        """Generate remediation recommendations based on findings"""
        
        recommendations = []
        correlation_key = correlation["correlation_key"]
        
        if "process:" in correlation_key:
            if confidence > 0.9:
                recommendations.extend([
                    "Immediately quarantine affected systems",
                    "Block process execution via endpoint protection",
                    "Collect memory dump for forensic analysis",
                    "Check for lateral movement indicators"
                ])
            else:
                recommendations.extend([
                    "Monitor process behavior closely",
                    "Collect additional forensic artifacts",
                    "Review process execution context"
                ])
        
        elif "network:" in correlation_key:
            if confidence > 0.9:
                recommendations.extend([
                    "Block suspicious IP addresses at firewall",
                    "Analyze network traffic patterns",
                    "Check for data exfiltration indicators",
                    "Review DNS queries and responses"
                ])
            else:
                recommendations.extend([
                    "Monitor network connections",
                    "Analyze traffic content if possible",
                    "Check reputation of destination IP"
                ])
        
        elif "user:" in correlation_key:
            if confidence > 0.9:
                recommendations.extend([
                    "Temporarily disable affected user accounts",
                    "Force password reset",
                    "Review user access permissions",
                    "Analyze user activity timeline"
                ])
            else:
                recommendations.extend([
                    "Monitor user activity closely",
                    "Review recent login patterns",
                    "Check for privilege escalation attempts"
                ])
        
        # Always add general recommendations
        recommendations.extend([
            "Update threat intelligence databases",
            "Review and update detection rules",
            "Document findings for future reference"
        ])
        
        return recommendations
    
    async def _generate_threat_events(
        self, 
        analyzed_results: Dict[str, Any], 
        task: HuntingTask
    ) -> List[ThreatEvent]:
        """Generate threat events from significant findings"""
        
        threat_events = []
        
        for finding in analyzed_results.get("high_confidence_findings", []):
            correlation = finding["correlation"]
            
            # Determine threat category and severity
            category = self._determine_threat_category(correlation)
            severity = self._determine_threat_severity(finding["final_confidence"])
            
            # Create threat indicators
            indicators = []
            for f in correlation["findings"]:
                indicator_id = hashlib.sha256(json.dumps(f, sort_keys=True).encode()).hexdigest()
                
                indicator = ThreatIndicator(
                    id=indicator_id,
                    type="hunting_finding",
                    value=json.dumps(f["details"]),
                    confidence=f["confidence"],
                    severity=severity,
                    category=category,
                    source=f"hunting_task_{task.id}",
                    first_seen=datetime.fromisoformat(f["timestamp"]),
                    last_seen=datetime.fromisoformat(f["timestamp"]),
                    metadata={"agent_id": f["agent_id"], "finding_type": f["type"]}
                )
                indicators.append(indicator)
            
            # Create threat event
            event_id = hashlib.sha256(f"{task.id}:{correlation['correlation_key']}:{time.time()}".encode()).hexdigest()
            
            threat_event = ThreatEvent(
                id=event_id,
                node_id=self.node_id,
                timestamp=datetime.utcnow(),
                severity=severity,
                category=category,
                indicators=indicators,
                raw_data=correlation,
                confidence=finding["final_confidence"],
                false_positive_probability=finding["false_positive_probability"],
                metadata={
                    "hunting_task_id": task.id,
                    "correlation_key": correlation["correlation_key"],
                    "agents_involved": correlation["agents_involved"]
                },
                remediation_actions=finding["recommended_actions"]
            )
            
            threat_events.append(threat_event)
        
        return threat_events
    
    def _determine_threat_category(self, correlation: Dict[str, Any]) -> ThreatCategory:
        """Determine threat category based on correlation data"""
        
        correlation_key = correlation["correlation_key"]
        
        if "process:" in correlation_key:
            return ThreatCategory.MALWARE
        elif "network:" in correlation_key:
            return ThreatCategory.COMMAND_CONTROL
        elif "user:" in correlation_key:
            return ThreatCategory.PRIVILEGE_ESCALATION
        else:
            return ThreatCategory.RECONNAISSANCE
    
    def _determine_threat_severity(self, confidence: float) -> ThreatSeverity:
        """Determine threat severity based on confidence score"""
        
        if confidence >= 0.95:
            return ThreatSeverity.CRITICAL
        elif confidence >= 0.85:
            return ThreatSeverity.HIGH
        elif confidence >= 0.7:
            return ThreatSeverity.MEDIUM
        else:
            return ThreatSeverity.LOW
    
    async def _store_threat_event(self, event: ThreatEvent):
        """Store threat event in database"""
        try:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO threat_events 
                    (id, node_id, timestamp, severity, category, indicators, raw_data, 
                     confidence, false_positive_probability, metadata, remediation_actions)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                    event.id, event.node_id, event.timestamp,
                    event.severity.value, event.category.value,
                    json.dumps([asdict(i) for i in event.indicators]),
                    json.dumps(event.raw_data),
                    event.confidence, event.false_positive_probability,
                    json.dumps(event.metadata), json.dumps(event.remediation_actions)
                )
                
        except Exception as e:
            logger.error(f"Failed to store threat event: {e}")
    
    async def _update_task_status(self, task: HuntingTask):
        """Update task status in database"""
        try:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE hunting_tasks 
                    SET status = $1, started_at = $2, completed_at = $3, results = $4
                    WHERE id = $5
                """,
                    task.status.value, task.started_at, task.completed_at,
                    json.dumps(task.results), task.id
                )
                
        except Exception as e:
            logger.error(f"Failed to update task status: {e}")
    
    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats"""
        while True:
            try:
                current_time = datetime.utcnow()
                stale_agents = []
                
                for agent_id, agent in self.agents.items():
                    if (current_time - agent.last_heartbeat).total_seconds() > 120:  # 2 minutes
                        stale_agents.append(agent_id)
                
                # Remove stale agents
                for agent_id in stale_agents:
                    logger.warning(f"Removing stale agent {agent_id}")
                    del self.agents[agent_id]
                
                await asyncio.sleep(self.config["heartbeat_interval"])
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(30)
    
    async def _threat_intel_updater(self):
        """Periodically update threat intelligence"""
        while True:
            try:
                await self._load_threat_intelligence()
                await asyncio.sleep(self.config["threat_intel_refresh"])
                
            except Exception as e:
                logger.error(f"Error updating threat intelligence: {e}")
                await asyncio.sleep(60)
    
    async def _task_scheduler(self):
        """Schedule and prioritize hunting tasks"""
        while True:
            try:
                # Get pending tasks ordered by priority
                pending_tasks = [
                    task for task in self.active_tasks.values()
                    if task.status == HuntingTaskStatus.PENDING
                ]
                
                # Sort by priority (higher number = higher priority)
                pending_tasks.sort(key=lambda t: t.priority, reverse=True)
                
                # Execute high priority tasks
                for task in pending_tasks[:5]:  # Limit concurrent executions
                    if len([t for t in self.active_tasks.values() if t.status == HuntingTaskStatus.RUNNING]) < self.config["max_concurrent_hunts"]:
                        asyncio.create_task(self.execute_hunting_task(task.id))
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in task scheduler: {e}")
                await asyncio.sleep(30)
    
    async def get_hunting_status(self) -> Dict[str, Any]:
        """Get current hunting system status"""
        return {
            "node_id": self.node_id,
            "agents": {
                "total": len(self.agents),
                "active": len([a for a in self.agents.values() if a.status == "active"]),
                "agents": [asdict(agent) for agent in self.agents.values()]
            },
            "tasks": {
                "total": len(self.active_tasks),
                "pending": len([t for t in self.active_tasks.values() if t.status == HuntingTaskStatus.PENDING]),
                "running": len([t for t in self.active_tasks.values() if t.status == HuntingTaskStatus.RUNNING]),
                "completed": len([t for t in self.active_tasks.values() if t.status == HuntingTaskStatus.COMPLETED]),
                "failed": len([t for t in self.active_tasks.values() if t.status == HuntingTaskStatus.FAILED])
            },
            "threat_intelligence": {
                "indicators": len(self.threat_indicators),
                "categories": len(set(i.category for i in self.threat_indicators.values()))
            },
            "system_health": {
                "uptime": time.time(),  # Would track actual uptime
                "memory_usage": "unknown",  # Would get actual metrics
                "cpu_usage": "unknown"
            }
        }
    
    async def cleanup(self):
        """Clean up resources"""
        if self.redis_pool:
            await self.redis_pool.close()
        if self.postgres_pool:
            await self.postgres_pool.close()
        
        logger.info("Distributed threat hunter cleaned up")

# Simple ML model implementations for demonstration
class SimpleAnomalyDetector:
    """Simple anomaly detector for threat hunting"""
    
    async def analyze(self, correlation: Dict[str, Any]) -> float:
        """Analyze correlation for anomalies"""
        # Simple heuristic-based scoring
        base_score = 0.5
        
        # More findings = higher anomaly score
        finding_count = correlation.get("finding_count", 1)
        score_boost = min(0.4, finding_count * 0.1)
        
        # Multiple agents = higher confidence
        agent_count = len(correlation.get("agents_involved", []))
        agent_boost = min(0.3, agent_count * 0.1)
        
        return min(1.0, base_score + score_boost + agent_boost)

class FalsePositiveClassifier:
    """Simple false positive classifier"""
    
    async def classify(self, correlation: Dict[str, Any]) -> float:
        """Classify probability of false positive"""
        # Simple heuristic - fewer agents reporting = higher FP probability
        agent_count = len(correlation.get("agents_involved", []))
        
        if agent_count >= 3:
            return 0.1  # Low FP probability
        elif agent_count == 2:
            return 0.3  # Medium FP probability
        else:
            return 0.6  # High FP probability

# Example usage
async def main():
    """Example usage of distributed threat hunting"""
    
    hunter = DistributedThreatHunter(
        node_id="xorb-node-eu-central-1",
        threat_intel_feeds=[
            "https://api.threatintel.example.com/indicators",
            "https://feeds.example.com/malware-hashes"
        ]
    )
    
    try:
        await hunter.initialize()
        
        # Register some mock agents
        agent1 = HuntingAgent(
            id="agent-1",
            node_id="xorb-node-eu-central-1",
            capabilities=["process_monitoring", "network_analysis"],
            status="active",
            last_heartbeat=datetime.utcnow(),
            load_average=0.3,
            active_tasks=0,
            max_concurrent_tasks=10,
            version="2.1.0"
        )
        
        agent2 = HuntingAgent(
            id="agent-2",
            node_id="xorb-node-eu-west-1",
            capabilities=["file_analysis", "registry_monitoring"],
            status="active",
            last_heartbeat=datetime.utcnow(),
            load_average=0.5,
            active_tasks=1,
            max_concurrent_tasks=8,
            version="2.1.0"
        )
        
        await hunter.register_agent(agent1)
        await hunter.register_agent(agent2)
        
        # Create hunting tasks
        task_id = await hunter.create_hunting_task(
            name="Hunt for Suspicious Processes",
            description="Search for potentially malicious processes across all nodes",
            query="SELECT * FROM processes WHERE name LIKE '%suspicious%' OR command_line CONTAINS 'malware'",
            priority=8
        )
        
        # Execute task
        results = await hunter.execute_hunting_task(task_id)
        print(f"Hunting results: {json.dumps(results, indent=2, default=str)}")
        
        # Get system status
        status = await hunter.get_hunting_status()
        print(f"System status: {json.dumps(status, indent=2, default=str)}")
        
    finally:
        await hunter.cleanup()

if __name__ == "__main__":
    asyncio.run(main())