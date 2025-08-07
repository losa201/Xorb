#!/usr/bin/env python3
"""
🧠 XORB COGNITIVE CORTEX - GLOBAL LLM CONFIGURATION (OMEGA)
Self-optimizing, fully auditable, multi-model LLM core within XORB

This module establishes XORB's shared intelligence layer across all modules, agents,
and PRKMT phases with adaptive learning, reinforcement feedback, and encrypted telemetry.
"""

import asyncio
import json
import logging
import time
import hashlib
import secrets
import yaml
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import threading
import queue
from collections import defaultdict, deque
import base64
import sqlite3
from cryptography.fernet import Fernet
import faiss

try:
    from openai import OpenAI
    import aiohttp
except ImportError:
    OpenAI = None
    aiohttp = None
    logging.warning("Required libraries not available. Install openai and aiohttp.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskType(Enum):
    NARRATIVE = "narrative"
    SUMMARY = "summary"
    THREAT_REPORT = "threat_report"
    EXPLOIT = "exploit"
    PARAM_FUZZ = "param_fuzz"
    INJECTION = "injection"
    CODE_ANALYSIS = "code_analysis"
    SOURCE_MAPPING = "source_mapping"
    LOGIC_FLOW = "logic_flow"
    MULTI_HOP_THREAT_CHAINS = "multi_hop_threat_chains"
    LONG_FORM_REASONING = "long_form_reasoning"

class LLMProvider(Enum):
    OPENROUTER = "openrouter"
    NVIDIA = "nvidia"

class FallbackReason(Enum):
    LATENCY_THRESHOLD = "latency_threshold"
    OUTPUT_ENTROPY = "output_entropy"
    HALLUCINATION_SCORE = "hallucination_score"
    MODEL_FAILURE = "model_failure"
    API_ERROR = "api_error"

@dataclass
class LLMModel:
    provider: LLMProvider
    model_name: str
    profile: str
    tasks: List[TaskType]
    api_endpoint: str
    performance_score: float = 1.0
    usage_count: int = 0
    success_rate: float = 1.0
    avg_latency: float = 0.0

@dataclass
class LLMRequest:
    request_id: str
    task_type: TaskType
    prompt: str
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    selected_model: Optional[str] = None
    fallback_used: bool = False
    confidence_score: Optional[float] = None
    latency: Optional[float] = None
    output_hash: Optional[str] = None

@dataclass
class LLMResponse:
    request_id: str
    model_used: str
    content: str
    confidence_score: float
    latency: float
    token_count: int
    fallback_reason: Optional[FallbackReason] = None
    entropy_score: Optional[float] = None
    hallucination_score: Optional[float] = None

@dataclass
class FeedbackSignal:
    signal_id: str
    request_id: str
    signal_type: str
    success_score: float
    exploit_chain_success: Optional[bool] = None
    valid_payload_detection: Optional[bool] = None
    anomaly_resilience: Optional[float] = None
    hallucination_drift: Optional[float] = None
    llm_exploit_coherence: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AuditLogEntry:
    timestamp: str
    agent_id: str
    model: str
    task_type: str
    latency: float
    confidence_score: float
    output_hash: str
    signal_trace: str
    fallback_used: bool
    learning_tag: str

class XORBEnsembleRouter:
    """Adaptive multi-model routing engine with reinforcement learning"""
    
    def __init__(self):
        self.router_id = f"ENSEMBLE-ROUTER-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Model registry
        self.models = self._initialize_models()
        self.routing_rules = self._initialize_routing_rules()
        
        # Performance tracking
        self.performance_history = defaultdict(deque)
        self.model_weights = {model.model_name: 1.0 for model in self.models.values()}
        
        # Adaptive learning
        self.feedback_signals = deque(maxlen=10000)
        self.learning_cache = {}
        
        # Vector store for similarity matching
        self.vector_dimension = 768
        self.vector_index = faiss.IndexFlatIP(self.vector_dimension)
        self.request_vectors = []
        self.request_mappings = {}
        
        # Configuration
        self.latency_threshold = 1500  # ms
        self.entropy_threshold = 1.6
        self.hallucination_threshold = 0.8
        self.cache_ttl = 12 * 3600  # 12 hours
        
        logger.info(f"🧠 XORB Ensemble Router initialized - ID: {self.router_id}")
    
    def _initialize_models(self) -> Dict[str, LLMModel]:
        """Initialize LLM model registry"""
        models = {}
        
        # OpenRouter models
        openrouter_key = os.getenv('OPENROUTER_API_KEY', 'demo_key')
        
        models['openrouter/horizon-beta'] = LLMModel(
            provider=LLMProvider.OPENROUTER,
            model_name='openrouter/horizon-beta',
            profile='lightweight logic, summaries, narratives',
            tasks=[TaskType.NARRATIVE, TaskType.SUMMARY, TaskType.THREAT_REPORT],
            api_endpoint='https://openrouter.ai/api/v1/chat/completions'
        )
        
        models['z-ai/glm-4.5-air:free'] = LLMModel(
            provider=LLMProvider.OPENROUTER,
            model_name='z-ai/glm-4.5-air:free',
            profile='rapid exploit synthesis, multilingual input, token fuzz',
            tasks=[TaskType.EXPLOIT, TaskType.PARAM_FUZZ, TaskType.INJECTION],
            api_endpoint='https://openrouter.ai/api/v1/chat/completions'
        )
        
        models['qwen/qwen3-coder:free'] = LLMModel(
            provider=LLMProvider.OPENROUTER,
            model_name='qwen/qwen3-coder:free',
            profile='code logic, param abuse, logic inference',
            tasks=[TaskType.CODE_ANALYSIS, TaskType.SOURCE_MAPPING, TaskType.LOGIC_FLOW],
            api_endpoint='https://openrouter.ai/api/v1/chat/completions'
        )
        
        # NVIDIA model
        nvidia_key = os.getenv('NVIDIA_API_KEY', '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')
        
        models['nvidia/qwen3-235b-a22b'] = LLMModel(
            provider=LLMProvider.NVIDIA,
            model_name='nvidia/qwen3-235b-a22b',
            profile='deep reasoning, multi-hop threats, strategy simulation',
            tasks=[TaskType.MULTI_HOP_THREAT_CHAINS, TaskType.LONG_FORM_REASONING],
            api_endpoint='https://integrate.api.nvidia.com/v1'
        )
        
        return models
    
    def _initialize_routing_rules(self) -> Dict[TaskType, List[str]]:
        """Initialize task-based routing rules"""
        return {
            TaskType.NARRATIVE: ['openrouter/horizon-beta'],
            TaskType.SUMMARY: ['openrouter/horizon-beta'],
            TaskType.THREAT_REPORT: ['openrouter/horizon-beta'],
            TaskType.EXPLOIT: ['z-ai/glm-4.5-air:free'],
            TaskType.PARAM_FUZZ: ['z-ai/glm-4.5-air:free'],
            TaskType.INJECTION: ['z-ai/glm-4.5-air:free'],
            TaskType.CODE_ANALYSIS: ['qwen/qwen3-coder:free'],
            TaskType.SOURCE_MAPPING: ['qwen/qwen3-coder:free'],
            TaskType.LOGIC_FLOW: ['qwen/qwen3-coder:free'],
            TaskType.MULTI_HOP_THREAT_CHAINS: ['nvidia/qwen3-235b-a22b'],
            TaskType.LONG_FORM_REASONING: ['nvidia/qwen3-235b-a22b']
        }
    
    async def route_request(self, request: LLMRequest) -> str:
        """Route request to optimal model using adaptive strategy"""
        try:
            # Get candidate models for task type
            candidates = self.routing_rules.get(request.task_type, ['openrouter/horizon-beta'])
            
            # Apply memory-weighted selection
            selected_model = self._select_weighted_model(candidates, request)
            
            # Check cache for similar requests
            cached_response = self._check_cache(request)
            if cached_response:
                logger.info(f"🧠 Cache hit for request {request.request_id}")
                return cached_response
            
            request.selected_model = selected_model
            return selected_model
            
        except Exception as e:
            logger.error(f"❌ Routing error: {e}")
            return 'openrouter/horizon-beta'  # Fallback
    
    def _select_weighted_model(self, candidates: List[str], request: LLMRequest) -> str:
        """Select model using performance weights and adaptive learning"""
        if len(candidates) == 1:
            return candidates[0]
        
        # Calculate weighted scores
        scores = {}
        for model_name in candidates:
            if model_name in self.models:
                model = self.models[model_name]
                
                # Base performance score
                base_score = model.performance_score
                
                # Weight by historical success rate
                success_weight = model.success_rate
                
                # Weight by average latency (inverse)
                latency_weight = 1.0 / max(model.avg_latency, 100)  # min 100ms baseline
                
                # Weight by recent performance
                recent_weight = self.model_weights.get(model_name, 1.0)
                
                # Combined score
                scores[model_name] = base_score * success_weight * latency_weight * recent_weight
        
        # Select highest scoring model
        if scores:
            return max(scores.keys(), key=lambda k: scores[k])
        
        return candidates[0]  # Fallback to first candidate
    
    def _check_cache(self, request: LLMRequest) -> Optional[str]:
        """Check vector cache for similar requests"""
        try:
            # Simple hash-based cache for demonstration
            request_hash = hashlib.sha256(f"{request.task_type.value}_{request.prompt}".encode()).hexdigest()
            
            if request_hash in self.learning_cache:
                cache_entry = self.learning_cache[request_hash]
                if (datetime.now() - cache_entry['timestamp']).total_seconds() < self.cache_ttl:
                    return cache_entry['model']
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Cache check error: {e}")
            return None
    
    def update_performance(self, request: LLMRequest, response: LLMResponse):
        """Update model performance metrics"""
        try:
            model_name = response.model_used
            if model_name in self.models:
                model = self.models[model_name]
                
                # Update usage statistics
                model.usage_count += 1
                
                # Update success rate
                success = response.confidence_score > 0.7 and not response.fallback_reason
                model.success_rate = (model.success_rate * (model.usage_count - 1) + (1.0 if success else 0.0)) / model.usage_count
                
                # Update average latency
                model.avg_latency = (model.avg_latency * (model.usage_count - 1) + response.latency) / model.usage_count
                
                # Update performance history
                self.performance_history[model_name].append({
                    'timestamp': datetime.now(),
                    'latency': response.latency,
                    'confidence': response.confidence_score,
                    'success': success
                })
                
                # Limit history size
                if len(self.performance_history[model_name]) > 1000:
                    self.performance_history[model_name].popleft()
            
            # Update cache
            request_hash = hashlib.sha256(f"{request.task_type.value}_{request.prompt}".encode()).hexdigest()
            self.learning_cache[request_hash] = {
                'model': model_name,
                'timestamp': datetime.now(),
                'performance': response.confidence_score
            }
            
        except Exception as e:
            logger.error(f"❌ Performance update error: {e}")
    
    def process_feedback_signal(self, signal: FeedbackSignal):
        """Process reinforcement learning feedback signal"""
        try:
            self.feedback_signals.append(signal)
            
            # Find related request
            request_id = signal.request_id
            
            # Update model weights based on feedback
            if signal.success_score > 0.8:
                # Positive feedback - increase weight
                if hasattr(signal, 'model_used'):
                    current_weight = self.model_weights.get(signal.model_used, 1.0)
                    self.model_weights[signal.model_used] = min(2.0, current_weight * 1.1)
            elif signal.success_score < 0.3:
                # Negative feedback - decrease weight
                if hasattr(signal, 'model_used'):
                    current_weight = self.model_weights.get(signal.model_used, 1.0)
                    self.model_weights[signal.model_used] = max(0.1, current_weight * 0.9)
            
            logger.info(f"🔄 Processed feedback signal: {signal.signal_type} = {signal.success_score}")
            
        except Exception as e:
            logger.error(f"❌ Feedback processing error: {e}")
    
    def get_router_status(self) -> Dict[str, Any]:
        """Get comprehensive router status"""
        return {
            'router_id': self.router_id,
            'models_registered': len(self.models),
            'model_weights': self.model_weights,
            'total_requests_cached': len(self.learning_cache),
            'feedback_signals_received': len(self.feedback_signals),
            'routing_rules': {task.value: models for task, models in self.routing_rules.items()},
            'performance_summary': {
                model_name: {
                    'usage_count': model.usage_count,
                    'success_rate': model.success_rate,
                    'avg_latency': model.avg_latency,
                    'performance_score': model.performance_score
                }
                for model_name, model in self.models.items()
            }
        }

class XORBSecureTelemetryLake:
    """Encrypted audit logging and telemetry storage"""
    
    def __init__(self):
        self.lake_id = f"TELEMETRY-LAKE-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize encryption
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
        # Initialize SQLite database for audit logs
        self.db_path = "data/xorb_audit_logs.db"
        Path("data").mkdir(exist_ok=True)
        self._initialize_database()
        
        # Configuration
        self.retention_days = 90
        self.log_format = "JSONL"
        
        logger.info(f"🔒 XORB Secure Telemetry Lake initialized - ID: {self.lake_id}")
    
    def _initialize_database(self):
        """Initialize audit log database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    latency REAL NOT NULL,
                    confidence_score REAL NOT NULL,
                    output_hash TEXT NOT NULL,
                    signal_trace TEXT,
                    fallback_used BOOLEAN NOT NULL,
                    learning_tag TEXT,
                    encrypted_data TEXT NOT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_logs(timestamp)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_agent_id ON audit_logs(agent_id)
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Database initialization error: {e}")
    
    def log_audit_entry(self, entry: AuditLogEntry):
        """Log encrypted audit entry"""
        try:
            # Encrypt sensitive data
            sensitive_data = {
                'agent_id': entry.agent_id,
                'signal_trace': entry.signal_trace,
                'learning_tag': entry.learning_tag
            }
            
            encrypted_data = self.cipher.encrypt(json.dumps(sensitive_data).encode())
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO audit_logs 
                (timestamp, agent_id, model, task_type, latency, confidence_score, 
                 output_hash, signal_trace, fallback_used, learning_tag, encrypted_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.timestamp,
                entry.agent_id,
                entry.model,
                entry.task_type,
                entry.latency,
                entry.confidence_score,
                entry.output_hash,
                entry.signal_trace,
                entry.fallback_used,
                entry.learning_tag,
                encrypted_data.decode()
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"🔒 Audit entry logged: {entry.agent_id} -> {entry.model}")
            
        except Exception as e:
            logger.error(f"❌ Audit logging error: {e}")
    
    def query_audit_logs(self, start_time: datetime, end_time: datetime, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query audit logs with decryption"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if agent_id:
                cursor.execute('''
                    SELECT * FROM audit_logs 
                    WHERE timestamp BETWEEN ? AND ? AND agent_id = ?
                    ORDER BY timestamp DESC
                ''', (start_time.isoformat(), end_time.isoformat(), agent_id))
            else:
                cursor.execute('''
                    SELECT * FROM audit_logs 
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY timestamp DESC
                ''', (start_time.isoformat(), end_time.isoformat()))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Decrypt and format results
            results = []
            for row in rows:
                try:
                    encrypted_data = row[11].encode()
                    decrypted_data = json.loads(self.cipher.decrypt(encrypted_data).decode())
                    
                    result = {
                        'id': row[0],
                        'timestamp': row[1],
                        'agent_id': row[2],
                        'model': row[3],
                        'task_type': row[4],
                        'latency': row[5],
                        'confidence_score': row[6],
                        'output_hash': row[7],
                        'signal_trace': decrypted_data.get('signal_trace'),
                        'fallback_used': row[9],
                        'learning_tag': decrypted_data.get('learning_tag')
                    }
                    results.append(result)
                    
                except Exception as decrypt_error:
                    logger.warning(f"⚠️ Failed to decrypt audit entry {row[0]}: {decrypt_error}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Audit query error: {e}")
            return []
    
    def cleanup_old_logs(self):
        """Clean up logs older than retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM audit_logs WHERE timestamp < ?
            ''', (cutoff_date.isoformat(),))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logger.info(f"🧹 Cleaned up {deleted_count} old audit log entries")
            
        except Exception as e:
            logger.error(f"❌ Log cleanup error: {e}")
    
    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Get telemetry lake summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total logs
            cursor.execute('SELECT COUNT(*) FROM audit_logs')
            total_logs = cursor.fetchone()[0]
            
            # Logs by model
            cursor.execute('SELECT model, COUNT(*) FROM audit_logs GROUP BY model')
            logs_by_model = dict(cursor.fetchall())
            
            # Recent activity (last 24 hours)
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            cursor.execute('SELECT COUNT(*) FROM audit_logs WHERE timestamp > ?', (yesterday,))
            recent_activity = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'lake_id': self.lake_id,
                'total_audit_logs': total_logs,
                'logs_by_model': logs_by_model,
                'recent_activity_24h': recent_activity,
                'retention_days': self.retention_days,
                'encryption_enabled': True,
                'database_path': self.db_path
            }
            
        except Exception as e:
            logger.error(f"❌ Telemetry summary error: {e}")
            return {}

class XORBCognitiveCortex:
    """Main Cognitive Cortex orchestrating all LLM operations"""
    
    def __init__(self):
        self.cortex_id = f"COGNITIVE-CORTEX-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize core components
        self.ensemble_router = XORBEnsembleRouter()
        self.telemetry_lake = XORBSecureTelemetryLake()
        
        # Request tracking
        self.active_requests = {}
        self.request_history = deque(maxlen=10000)
        
        # Drift detection
        self.drift_detector = self._initialize_drift_detector()
        
        # API clients
        self.openai_clients = self._initialize_openai_clients()
        
        # Background tasks
        self.running = True
        self.maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self.maintenance_thread.start()
        
        logger.info(f"🧠 XORB Cognitive Cortex initialized - ID: {self.cortex_id}")
    
    def _initialize_drift_detector(self) -> Dict[str, Any]:
        """Initialize drift detection system"""
        return {
            'entropy_history': deque(maxlen=1000),
            'hallucination_history': deque(maxlen=1000),
            'baseline_entropy': 1.0,
            'baseline_hallucination': 0.2
        }
    
    def _initialize_openai_clients(self) -> Dict[str, OpenAI]:
        """Initialize OpenAI clients for different providers"""
        clients = {}
        
        if OpenAI:
            # OpenRouter client
            openrouter_key = os.getenv('OPENROUTER_API_KEY', 'demo_key')
            clients['openrouter'] = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key
            )
            
            # NVIDIA client
            nvidia_key = os.getenv('NVIDIA_API_KEY', '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')
            clients['nvidia'] = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=nvidia_key
            )
        
        return clients
    
    async def process_llm_request(self, task_type: TaskType, prompt: str, context: Dict[str, Any] = None, agent_id: str = "unknown") -> LLMResponse:
        """Process LLM request through the cognitive cortex"""
        try:
            # Create request
            request = LLMRequest(
                request_id=f"REQ-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{secrets.token_hex(4)}",
                task_type=task_type,
                prompt=prompt,
                context=context or {}
            )
            
            self.active_requests[request.request_id] = request
            
            # Route to optimal model
            selected_model = await self.ensemble_router.route_request(request)
            
            # Execute request
            start_time = time.time()
            response = await self._execute_llm_request(selected_model, request)
            
            # Calculate metrics
            response.latency = (time.time() - start_time) * 1000  # ms
            response.output_hash = hashlib.sha256(response.content.encode()).hexdigest()[:16]
            
            # Detect drift
            await self._detect_drift(response)
            
            # Update performance
            self.ensemble_router.update_performance(request, response)
            
            # Log audit entry
            audit_entry = AuditLogEntry(
                timestamp=datetime.now().isoformat(),
                agent_id=agent_id,
                model=response.model_used,
                task_type=task_type.value,
                latency=response.latency,
                confidence_score=response.confidence_score,
                output_hash=response.output_hash,
                signal_trace=f"task:{task_type.value}|model:{selected_model}",
                fallback_used=response.fallback_reason is not None,
                learning_tag=f"cortex:{self.cortex_id}"
            )
            
            self.telemetry_lake.log_audit_entry(audit_entry)
            
            # Clean up
            del self.active_requests[request.request_id]
            self.request_history.append(request)
            
            logger.info(f"🧠 Processed {task_type.value} request in {response.latency:.1f}ms")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ LLM request processing error: {e}")
            # Return fallback response
            return LLMResponse(
                request_id=request.request_id,
                model_used="fallback",
                content=f"Error processing request: {str(e)}",
                confidence_score=0.0,
                latency=0.0,
                token_count=0,
                fallback_reason=FallbackReason.MODEL_FAILURE
            )
    
    async def _execute_llm_request(self, model_name: str, request: LLMRequest) -> LLMResponse:
        """Execute actual LLM request"""
        try:
            model = self.ensemble_router.models.get(model_name)
            if not model:
                raise Exception(f"Model {model_name} not found")
            
            # Get appropriate client
            if model.provider == LLMProvider.OPENROUTER:
                client = self.openai_clients.get('openrouter')
            elif model.provider == LLMProvider.NVIDIA:
                client = self.openai_clients.get('nvidia')
            else:
                raise Exception(f"No client available for provider {model.provider}")
            
            if not client:
                raise Exception("OpenAI client not available")
            
            # Execute request
            completion = client.chat.completions.create(
                model=model.model_name,
                messages=[
                    {"role": "system", "content": "You are XORB's cognitive security analysis engine. Provide accurate, helpful security analysis."},
                    {"role": "user", "content": request.prompt}
                ],
                max_tokens=4096,
                temperature=0.1
            )
            
            content = completion.choices[0].message.content
            token_count = completion.usage.total_tokens if completion.usage else 0
            
            # Calculate confidence score (simplified)
            confidence_score = min(1.0, len(content) / 1000.0)  # Length-based confidence
            
            return LLMResponse(
                request_id=request.request_id,
                model_used=model_name,
                content=content,
                confidence_score=confidence_score,
                latency=0.0,  # Will be set by caller
                token_count=token_count
            )
            
        except Exception as e:
            logger.error(f"❌ LLM execution error: {e}")
            
            # Fallback to horizon-beta if available
            if model_name != 'openrouter/horizon-beta':
                logger.warning(f"🔄 Falling back to horizon-beta for request {request.request_id}")
                return await self._execute_llm_request('openrouter/horizon-beta', request)
            
            # Ultimate fallback
            return LLMResponse(
                request_id=request.request_id,
                model_used=model_name,
                content="Fallback response due to model unavailability.",
                confidence_score=0.5,
                latency=0.0,
                token_count=0,
                fallback_reason=FallbackReason.API_ERROR
            )
    
    async def _detect_drift(self, response: LLMResponse):
        """Detect model drift in response quality"""
        try:
            # Calculate entropy (simplified)
            content = response.content
            if content:
                # Character frequency entropy
                char_freq = defaultdict(int)
                for char in content:
                    char_freq[char] += 1
                
                total_chars = len(content)
                entropy = -sum((freq/total_chars) * np.log2(freq/total_chars) for freq in char_freq.values())
                response.entropy_score = entropy
                
                # Update drift detector
                self.drift_detector['entropy_history'].append(entropy)
                
                # Check for drift
                if entropy > self.drift_detector['baseline_entropy'] * 2:
                    logger.warning(f"⚠️ High entropy detected: {entropy:.2f}")
                    response.fallback_reason = FallbackReason.OUTPUT_ENTROPY
            
            # Hallucination detection (simplified - check for repetitive patterns)
            if content and len(content) > 100:
                # Look for repeated phrases
                words = content.lower().split()
                word_freq = defaultdict(int)
                for word in words:
                    word_freq[word] += 1
                
                max_freq = max(word_freq.values()) if word_freq else 0
                hallucination_score = max_freq / len(words) if words else 0
                response.hallucination_score = hallucination_score
                
                self.drift_detector['hallucination_history'].append(hallucination_score)
                
                if hallucination_score > self.drift_detector['baseline_hallucination'] * 3:
                    logger.warning(f"⚠️ Potential hallucination detected: {hallucination_score:.2f}")
                    response.fallback_reason = FallbackReason.HALLUCINATION_SCORE
            
        except Exception as e:
            logger.error(f"❌ Drift detection error: {e}")
    
    def submit_feedback(self, request_id: str, feedback_type: str, success_score: float, **kwargs) -> str:
        """Submit feedback signal for reinforcement learning"""
        try:
            signal = FeedbackSignal(
                signal_id=f"SIG-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{secrets.token_hex(4)}",
                request_id=request_id,
                signal_type=feedback_type,
                success_score=success_score,
                **kwargs
            )
            
            self.ensemble_router.process_feedback_signal(signal)
            
            logger.info(f"📡 Feedback submitted: {feedback_type} = {success_score}")
            return signal.signal_id
            
        except Exception as e:
            logger.error(f"❌ Feedback submission error: {e}")
            return ""
    
    def _maintenance_loop(self):
        """Background maintenance tasks"""
        while self.running:
            try:
                # Clean up old logs
                self.telemetry_lake.cleanup_old_logs()
                
                # Update baseline drift metrics
                if len(self.drift_detector['entropy_history']) > 100:
                    recent_entropy = list(self.drift_detector['entropy_history'])[-100:]
                    self.drift_detector['baseline_entropy'] = np.mean(recent_entropy)
                
                if len(self.drift_detector['hallucination_history']) > 100:
                    recent_hallucination = list(self.drift_detector['hallucination_history'])[-100:]
                    self.drift_detector['baseline_hallucination'] = np.mean(recent_hallucination)
                
                # Sleep for maintenance interval
                time.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"❌ Maintenance loop error: {e}")
                time.sleep(300)  # 5 minutes on error
    
    def get_cortex_status(self) -> Dict[str, Any]:
        """Get comprehensive cortex status"""
        try:
            router_status = self.ensemble_router.get_router_status()
            telemetry_status = self.telemetry_lake.get_telemetry_summary()
            
            return {
                'cortex_id': self.cortex_id,
                'timestamp': datetime.now().isoformat(),
                'active_requests': len(self.active_requests),
                'total_requests_processed': len(self.request_history),
                'router_status': router_status,
                'telemetry_status': telemetry_status,
                'drift_detector': {
                    'entropy_baseline': self.drift_detector['baseline_entropy'],
                    'hallucination_baseline': self.drift_detector['baseline_hallucination'],
                    'samples_collected': len(self.drift_detector['entropy_history'])
                },
                'maintenance_thread_active': self.maintenance_thread.is_alive(),
                'openai_clients_available': list(self.openai_clients.keys())
            }
            
        except Exception as e:
            logger.error(f"❌ Status retrieval error: {e}")
            return {}
    
    def shutdown(self):
        """Gracefully shutdown cognitive cortex"""
        logger.info("🛑 Shutting down XORB Cognitive Cortex")
        self.running = False

# Global cortex instance
global_cortex = None

def get_cortex() -> XORBCognitiveCortex:
    """Get global cortex instance"""
    global global_cortex
    if global_cortex is None:
        global_cortex = XORBCognitiveCortex()
    return global_cortex

async def llm_request(task_type: str, prompt: str, context: Dict[str, Any] = None, agent_id: str = "unknown") -> str:
    """Simplified API for LLM requests"""
    cortex = get_cortex()
    task_enum = TaskType(task_type.lower())
    response = await cortex.process_llm_request(task_enum, prompt, context, agent_id)
    return response.content

async def main():
    """Demonstrate XORB Cognitive Cortex"""
    logger.info("🧠 Starting XORB Cognitive Cortex demonstration")
    
    cortex = XORBCognitiveCortex()
    
    try:
        # Test different task types
        tasks = [
            (TaskType.CODE_ANALYSIS, "Analyze this JavaScript function for security vulnerabilities: function login(user, pass) { eval(user); }"),
            (TaskType.EXPLOIT, "Generate a safe demonstration payload for SQL injection testing"),
            (TaskType.THREAT_REPORT, "Create an executive summary of API security risks"),
            (TaskType.MULTI_HOP_THREAT_CHAINS, "Model attack paths from initial access to data exfiltration")
        ]
        
        for task_type, prompt in tasks:
            response = await cortex.process_llm_request(task_type, prompt, agent_id="demo-agent")
            logger.info(f"✅ {task_type.value}: {len(response.content)} chars, confidence: {response.confidence_score:.2f}")
        
        # Submit feedback
        cortex.submit_feedback("demo-request", "exploit_success", 0.9, exploit_chain_success=True)
        
        # Get status
        status = cortex.get_cortex_status()
        logger.info(f"📊 Cortex Status: {status['total_requests_processed']} requests processed")
        
        return status
        
    finally:
        cortex.shutdown()

if __name__ == "__main__":
    asyncio.run(main())