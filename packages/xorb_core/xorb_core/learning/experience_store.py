"""
Experience Store for Learning Systems
Manages storage and retrieval of execution experiences for ML training
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import hashlib

import clickhouse_driver
import pandas as pd
from clickhouse_driver import Client

from ..monitoring.learning_metrics import LearningMetricsCollector

logger = logging.getLogger(__name__)

@dataclass
class CampaignExecution:
    """Record of campaign execution for learning"""
    execution_id: str
    campaign_id: str
    agent_id: str
    start_time: datetime
    end_time: Optional[datetime]
    success: bool
    resource_usage: Dict[str, float]
    context: Dict[str, Any]
    outcome_metrics: Dict[str, float]
    error_details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data['start_time'] = self.start_time.isoformat()
        data['end_time'] = self.end_time.isoformat() if self.end_time else None
        return data

@dataclass 
class WorkflowExecution:
    """Workflow execution record for evolution learning"""
    template_id: str
    execution_id: str
    context: Dict[str, Any]
    start_time: float
    end_time: Optional[float]
    success: bool
    metrics: Dict[str, float]
    resource_usage: Dict[str, float]
    error_details: Optional[str] = None

class ExperienceStore:
    """ClickHouse-based storage for learning experiences"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.host = self.config.get('host', 'clickhouse.xorb-prod.svc.cluster.local')
        self.port = self.config.get('port', 9000)
        self.database = self.config.get('database', 'xorb_learning')
        self.username = self.config.get('username', 'xorb')
        self.password = self.config.get('password', 'xorb-experience-store')
        
        self.client = None
        self.metrics_collector = LearningMetricsCollector()
        
        # Connection pool for async operations
        self._connection_pool = []
        self._pool_size = self.config.get('pool_size', 5)
        
    async def initialize(self):
        """Initialize ClickHouse connection and create tables"""
        try:
            self.client = Client(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password,
                settings={'use_numpy': True}
            )
            
            await self._create_tables()
            logger.info("Experience store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize experience store: {e}")
            raise
    
    async def _create_tables(self):
        """Create necessary tables if they don't exist"""
        
        # Campaign executions table
        campaign_executions_ddl = """
        CREATE TABLE IF NOT EXISTS campaign_executions (
            execution_id String,
            campaign_id String,
            agent_id String,
            start_time DateTime64(3),
            end_time Nullable(DateTime64(3)),
            success Boolean,
            resource_usage Map(String, Float64),
            context Map(String, String),
            outcome_metrics Map(String, Float64),
            error_details Nullable(String),
            created_at DateTime64(3) DEFAULT now64()
        ) ENGINE = MergeTree()
        ORDER BY (campaign_id, start_time)
        PARTITION BY toYYYYMM(start_time)
        """
        
        # Agent performance summary table
        agent_performance_ddl = """
        CREATE TABLE IF NOT EXISTS agent_performance (
            agent_id String,
            timestamp DateTime64(3),
            success_rate Float64,
            avg_execution_time Float64,
            resource_efficiency Float64,
            error_count UInt64,
            total_executions UInt64,
            created_at DateTime64(3) DEFAULT now64()
        ) ENGINE = MergeTree()
        ORDER BY (agent_id, timestamp)
        PARTITION BY toYYYYMM(timestamp)
        """
        
        # Bandit updates table
        bandit_updates_ddl = """
        CREATE TABLE IF NOT EXISTS bandit_updates (
            agent_id String,
            timestamp DateTime64(3),
            reward Float64,
            context Map(String, String),
            algorithm String,
            created_at DateTime64(3) DEFAULT now64()
        ) ENGINE = MergeTree()
        ORDER BY (agent_id, timestamp)
        PARTITION BY toYYYYMM(timestamp)
        """
        
        # Workflow executions table
        workflow_executions_ddl = """
        CREATE TABLE IF NOT EXISTS workflow_executions (
            template_id String,
            execution_id String,
            context Map(String, String),
            start_time DateTime64(3),
            end_time Nullable(DateTime64(3)),
            success Boolean,
            metrics Map(String, Float64),
            resource_usage Map(String, Float64),
            error_details Nullable(String),
            created_at DateTime64(3) DEFAULT now64()
        ) ENGINE = MergeTree()
        ORDER BY (template_id, start_time)
        PARTITION BY toYYYYMM(start_time)
        """
        
        tables = [
            campaign_executions_ddl,
            agent_performance_ddl,
            bandit_updates_ddl,
            workflow_executions_ddl
        ]
        
        for ddl in tables:
            try:
                self.client.execute(ddl)
                logger.debug(f"Created/verified table: {ddl.split()[5]}")
            except Exception as e:
                logger.error(f"Failed to create table: {e}")
                raise
    
    async def store_campaign_execution(self, execution: CampaignExecution):
        """Store campaign execution record"""
        try:
            # Convert context and metrics to strings for ClickHouse Map type
            context_str = {k: str(v) for k, v in execution.context.items()}
            
            data = {
                'execution_id': execution.execution_id,
                'campaign_id': execution.campaign_id,
                'agent_id': execution.agent_id,
                'start_time': execution.start_time,
                'end_time': execution.end_time,
                'success': execution.success,
                'resource_usage': execution.resource_usage,
                'context': context_str,
                'outcome_metrics': execution.outcome_metrics,
                'error_details': execution.error_details
            }
            
            self.client.execute(
                "INSERT INTO campaign_executions VALUES",
                [data]
            )
            
            # Record metrics
            await self.metrics_collector.record_experience_stored(
                'campaign_execution', execution.agent_id
            )
            
            logger.debug(f"Stored campaign execution: {execution.execution_id}")
            
        except Exception as e:
            logger.error(f"Failed to store campaign execution: {e}")
            raise
    
    async def store_bandit_update(self, agent_id: str, context: Dict[str, Any], 
                                reward: float, algorithm: str = "unknown"):
        """Store bandit algorithm update"""
        try:
            # Convert context to string map
            context_str = {k: str(v) for k, v in context.items()}
            
            data = {
                'agent_id': agent_id,
                'timestamp': datetime.now(),
                'reward': reward,
                'context': context_str,
                'algorithm': algorithm
            }
            
            self.client.execute(
                "INSERT INTO bandit_updates VALUES",
                [data]
            )
            
            await self.metrics_collector.record_experience_stored(
                'bandit_update', agent_id
            )
            
            logger.debug(f"Stored bandit update for agent: {agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to store bandit update: {e}")
            raise
    
    async def store_workflow_execution(self, execution: WorkflowExecution):
        """Store workflow execution for evolution learning"""
        try:
            # Convert context to string map
            context_str = {k: str(v) for k, v in execution.context.items()}
            
            data = {
                'template_id': execution.template_id,
                'execution_id': execution.execution_id,
                'context': context_str,
                'start_time': datetime.fromtimestamp(execution.start_time),
                'end_time': datetime.fromtimestamp(execution.end_time) if execution.end_time else None,
                'success': execution.success,
                'metrics': execution.metrics,
                'resource_usage': execution.resource_usage,
                'error_details': execution.error_details
            }
            
            self.client.execute(
                "INSERT INTO workflow_executions VALUES",
                [data]
            )
            
            await self.metrics_collector.record_experience_stored(
                'workflow_execution', execution.template_id
            )
            
            logger.debug(f"Stored workflow execution: {execution.execution_id}")
            
        except Exception as e:
            logger.error(f"Failed to store workflow execution: {e}")
            raise
    
    async def get_agent_performance_history(self, agent_id: str, 
                                          days: int = 7) -> List[Dict[str, Any]]:
        """Get agent performance history for specified period"""
        try:
            query = """
            SELECT 
                agent_id,
                toStartOfHour(start_time) as hour,
                countIf(success = 1) / count() as success_rate,
                avg(toUnixTimestamp(end_time) - toUnixTimestamp(start_time)) as avg_duration,
                avg(resource_usage['cpu_utilization']) as avg_cpu_usage,
                avg(resource_usage['memory_utilization']) as avg_memory_usage,
                count() as total_executions
            FROM campaign_executions 
            WHERE agent_id = %(agent_id)s 
                AND start_time >= subtractDays(now(), %(days)s)
                AND end_time IS NOT NULL
            GROUP BY agent_id, hour
            ORDER BY hour
            """
            
            result = self.client.execute(
                query,
                {'agent_id': agent_id, 'days': days}
            )
            
            columns = ['agent_id', 'hour', 'success_rate', 'avg_duration', 
                      'avg_cpu_usage', 'avg_memory_usage', 'total_executions']
            
            return [dict(zip(columns, row)) for row in result]
            
        except Exception as e:
            logger.error(f"Failed to get agent performance history: {e}")
            return []
    
    async def get_recent_campaign_executions(self, limit: int = 100) -> List[CampaignExecution]:
        """Get recent campaign executions for training"""
        try:
            query = """
            SELECT *
            FROM campaign_executions
            WHERE start_time >= subtractDays(now(), 7)
            ORDER BY start_time DESC
            LIMIT %(limit)s
            """
            
            result = self.client.execute(query, {'limit': limit})
            
            executions = []
            for row in result:
                execution = CampaignExecution(
                    execution_id=row[0],
                    campaign_id=row[1],
                    agent_id=row[2],
                    start_time=row[3],
                    end_time=row[4],
                    success=row[5],
                    resource_usage=row[6],
                    context={k: v for k, v in row[7].items()},  # Convert back from string map
                    outcome_metrics=row[8],
                    error_details=row[9]
                )
                executions.append(execution)
            
            return executions
            
        except Exception as e:
            logger.error(f"Failed to get recent executions: {e}")
            return []
    
    async def get_recent_workflow_executions(self, limit: int = 100) -> List[WorkflowExecution]:
        """Get recent workflow executions for policy learning"""
        try:
            query = """
            SELECT *
            FROM workflow_executions
            WHERE start_time >= subtractDays(now(), 7)
            ORDER BY start_time DESC
            LIMIT %(limit)s
            """
            
            result = self.client.execute(query, {'limit': limit})
            
            executions = []
            for row in result:
                execution = WorkflowExecution(
                    template_id=row[0],
                    execution_id=row[1],
                    context={k: v for k, v in row[2].items()},
                    start_time=row[3].timestamp(),
                    end_time=row[4].timestamp() if row[4] else None,
                    success=row[5],
                    metrics=row[6],
                    resource_usage=row[7],
                    error_details=row[8]
                )
                executions.append(execution)
            
            return executions
            
        except Exception as e:
            logger.error(f"Failed to get recent workflow executions: {e}")
            return []
    
    async def get_training_features(self, days: int = 30) -> pd.DataFrame:
        """Extract features for ML training"""
        try:
            query = """
            SELECT 
                agent_id,
                campaign_id,
                toHour(start_time) as hour_of_day,
                toDayOfWeek(start_time) as day_of_week,
                success,
                toUnixTimestamp(end_time) - toUnixTimestamp(start_time) as duration,
                resource_usage['cpu_utilization'] as cpu_usage,
                resource_usage['memory_utilization'] as memory_usage,
                resource_usage['network_utilization'] as network_usage,
                outcome_metrics['efficiency_score'] as efficiency_score,
                outcome_metrics['quality_score'] as quality_score,
                context['priority'] as priority,
                context['complexity'] as complexity
            FROM campaign_executions
            WHERE start_time >= subtractDays(now(), %(days)s)
                AND end_time IS NOT NULL
            ORDER BY start_time
            """
            
            result = self.client.execute(query, {'days': days})
            
            columns = [
                'agent_id', 'campaign_id', 'hour_of_day', 'day_of_week',
                'success', 'duration', 'cpu_usage', 'memory_usage', 'network_usage',
                'efficiency_score', 'quality_score', 'priority', 'complexity'
            ]
            
            df = pd.DataFrame(result, columns=columns)
            
            # Convert string columns to appropriate types
            numeric_columns = ['duration', 'cpu_usage', 'memory_usage', 'network_usage',
                             'efficiency_score', 'quality_score', 'priority', 'complexity']
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill missing values
            df = df.fillna(0)
            
            logger.info(f"Extracted {len(df)} training samples")
            return df
            
        except Exception as e:
            logger.error(f"Failed to extract training features: {e}")
            return pd.DataFrame()
    
    async def get_drift_detection_data(self, window_hours: int = 24) -> Dict[str, Any]:
        """Get data for concept drift detection"""
        try:
            query = """
            SELECT 
                toStartOfHour(start_time) as hour,
                avg(if(success, 1, 0)) as success_rate,
                avg(toUnixTimestamp(end_time) - toUnixTimestamp(start_time)) as avg_duration,
                avg(resource_usage['cpu_utilization']) as avg_cpu_usage,
                stddevPop(resource_usage['cpu_utilization']) as std_cpu_usage,
                count() as sample_count
            FROM campaign_executions
            WHERE start_time >= subtractHours(now(), %(window_hours)s)
                AND end_time IS NOT NULL
            GROUP BY hour
            ORDER BY hour
            """
            
            result = self.client.execute(query, {'window_hours': window_hours})
            
            if not result:
                return {}
            
            hours, success_rates, durations, cpu_usages, cpu_stds, counts = zip(*result)
            
            return {
                'timestamps': [h.timestamp() for h in hours],
                'success_rates': list(success_rates),
                'avg_durations': list(durations),
                'avg_cpu_usage': list(cpu_usages),
                'std_cpu_usage': list(cpu_stds),
                'sample_counts': list(counts)
            }
            
        except Exception as e:
            logger.error(f"Failed to get drift detection data: {e}")
            return {}
    
    async def cleanup_old_data(self, retention_days: int = 90):
        """Clean up old experience data"""
        try:
            tables = ['campaign_executions', 'agent_performance', 'bandit_updates', 'workflow_executions']
            
            for table in tables:
                query = f"""
                ALTER TABLE {table} 
                DELETE WHERE created_at < subtractDays(now(), %(retention_days)s)
                """
                
                self.client.execute(query, {'retention_days': retention_days})
                logger.info(f"Cleaned up old data from {table}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get experience store statistics"""
        try:
            stats = {}
            
            # Campaign executions stats
            query = """
            SELECT 
                count() as total_executions,
                countIf(success = 1) / count() as overall_success_rate,
                uniq(agent_id) as unique_agents,
                uniq(campaign_id) as unique_campaigns
            FROM campaign_executions
            WHERE start_time >= subtractDays(now(), 30)
            """
            
            result = self.client.execute(query)
            if result:
                stats['campaign_executions'] = {
                    'total_executions': result[0][0],
                    'success_rate': result[0][1],
                    'unique_agents': result[0][2],
                    'unique_campaigns': result[0][3]
                }
            
            # Workflow executions stats
            query = """
            SELECT 
                count() as total_workflows,
                countIf(success = 1) / count() as workflow_success_rate,
                uniq(template_id) as unique_templates
            FROM workflow_executions
            WHERE start_time >= subtractDays(now(), 30)
            """
            
            result = self.client.execute(query)
            if result:
                stats['workflow_executions'] = {
                    'total_workflows': result[0][0],
                    'success_rate': result[0][1],
                    'unique_templates': result[0][2]
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    async def close(self):
        """Close database connections"""
        try:
            if self.client:
                self.client.disconnect()
            logger.info("Experience store connections closed")
        except Exception as e:
            logger.error(f"Error closing experience store: {e}")