import os
import json
import logging
import signal
import sys
import redis
import importlib
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
import jwt
from cryptography.fernet import Fernet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("XORBWorker")

# Configuration from environment
REDIS_HOST = os.getenv("REDIS_HOST", os.getenv("REDIS_URL", "redis://redis:6379/0").replace("redis://", "").split(":")[0])
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
WORKER_ID = os.getenv("WORKER_ID", "worker-001")
# JWT_SECRET moved to centralized JWT manager
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", Fernet.generate_key().decode())

# Task handler registry
task_handlers: Dict[str, Callable] = {}

# Security utilities
class WorkerSecurity:
    @staticmethod
    def decrypt_data(encrypted_data: str) -> str:
        """Decrypt encrypted task data"""
        try:
            fernet = Fernet(ENCRYPTION_KEY.encode())
            return fernet.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            logger.error(f"Decryption error: {str(e)}")
            return encrypted_data

    @staticmethod
    def verify_token(token: str) -> Optional[Dict]:
        """Verify JWT token from task using centralized JWT manager"""
        try:
            # Import here to avoid circular dependencies
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
            from common.jwt_manager import verify_token_sync
            return verify_token_sync(token)
        except ValueError as e:
            logger.warning(f"Token verification failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"Token verification error: {e}")
            return None

# Task data models
@dataclass
class TaskMetadata:
    task_id: str
    module: str
    priority: int
    created_at: str
    expires_at: Optional[str] = None
    retries: int = 0
    max_retries: int = 3

@dataclass
class Task:
    metadata: TaskMetadata
    data: Dict[str, Any]
    token: str

# Task handlers
class TaskHandler:
    @staticmethod
    def register_handler(module: str):
        """Decorator to register task handlers"""
        def decorator(func):
            task_handlers[module] = func
            return func
        return decorator

    @staticmethod
    def handle_threat_intel(task: Task) -> Dict[str, Any]:
        """Handle threat intelligence tasks"""
        logger.info(f"Processing threat intel: {task.metadata.task_id}")
        try:
            # In real implementation, this would interface with threat intel platforms
            intel_data = task.data.get("intel", {})
            
            # Simulate processing
            result = {
                "task_id": task.metadata.task_id,
                "status": "completed",
                "findings": {
                    "related_incidents": [f"INC-2025-001-{intel_data.get('ioc_type')}"]
                },
                "processed_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Threat intel processed: {task.metadata.task_id}")
            return result
        except Exception as e:
            logger.error(f"Error processing threat intel: {str(e)}")
            return {"status": "failed", "error": str(e)}

    @staticmethod
    def handle_deception_grid(task: Task) -> Dict[str, Any]:
        """Handle deception grid tasks"""
        logger.info(f"Processing deception grid: {task.metadata.task_id}")
        try:
            # In real implementation, this would manage deception grid nodes
            decoy_data = task.data.get("decoy", {})
            
            # Simulate processing
            result = {
                "task_id": task.metadata.task_id,
                "status": "completed",
                "decoy_id": f"DCY-{datetime.utcnow().timestamp()}",
                "ip": f"192.168.1.{task.metadata.task_id[-3:]}",
                "mac": f"00:1A:2B:{task.metadata.task_id[4:6]}:{task.metadata.task_id[6:8]}:{task.metadata.task_id[8:10]}",
                "services": decoy_data.get("services", ["ssh", "http"]),
                "status": "active"
            }
            
            logger.info(f"Deception grid processed: {task.metadata.task_id}")
            return result
        except Exception as e:
            logger.error(f"Error processing deception grid: {str(e)}")
            return {"status": "failed", "error": str(e)}

    @staticmethod
    def handle_quantum_crypto(task: Task) -> Dict[str, Any]:
        """Handle quantum crypto tasks"""
        logger.info(f"Processing quantum crypto: {task.metadata.task_id}")
        try:
            # In real implementation, this would interface with quantum-safe crypto libraries
            crypto_data = task.data.get("crypto", {})
            
            # Simulate processing
            result = {
                "task_id": task.metadata.task_id,
                "status": "completed",
                "session_id": f"KEX-{datetime.utcnow().timestamp()}",
                "shared_secret": f"SECRET-{task.metadata.task_id}",
                "algorithm": crypto_data.get("algorithm", "kyber512"),
                "key_size": 32,
                "expires_in": "3600s"
            }
            
            logger.info(f"Quantum crypto processed: {task.metadata.task_id}")
            return result
        except Exception as e:
            logger.error(f"Error processing quantum crypto: {str(e)}")
            return {"status": "failed", "error": str(e)}

    @staticmethod
    def handle_compliance(task: Task) -> Dict[str, Any]:
        """Handle compliance tasks"""
        logger.info(f"Processing compliance: {task.metadata.task_id}")
        try:
            # In real implementation, this would interface with compliance frameworks
            compliance_data = task.data.get("compliance", {})
            
            # Simulate processing
            result = {
                "task_id": task.metadata.task_id,
                "status": "completed",
                "framework": compliance_data.get("framework", "CIS"),
                "findings": [
                    {
                        "control": "CIS-1.1",
                        "status": "passed",
                        "description": "Firewall configured correctly"
                    },
                    {
                        "control": "CIS-2.3",
                        "status": "failed",
                        "description": "Weak password policy"
                    }
                ],
                "score": 85.5,
                "recommendations": ["update_password_policy", "enable_mfa"]
            }
            
            logger.info(f"Compliance processed: {task.metadata.task_id}")
            return result
        except Exception as e:
            logger.error(f"Error processing compliance: {str(e)}")
            return {"status": "failed", "error": str(e)}

# Register task handlers
TaskHandler.register_handler("threat_intel")(TaskHandler.handle_threat_intel)
TaskHandler.register_handler("deception_grid")(TaskHandler.handle_deception_grid)
TaskHandler.register_handler("quantum_crypto")(TaskHandler.handle_quantum_crypto)
TaskHandler.register_handler("compliance")(TaskHandler.handle_compliance)

class WorkerService:
    def __init__(self):
        self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        self.running = True
        self.setup_signal_handlers()
        
    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
    def shutdown(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Shutting down worker {WORKER_ID}...")
        self.running = False
        
    def process_task(self, task_data: str) -> None:
        """Process a single task"""
        try:
            # Parse task data
            task_json = json.loads(task_data)
            metadata = TaskMetadata(**task_json["metadata"])
            token = task_json["token"]
            
            # Verify token
            token_data = WorkerSecurity.verify_token(token)
            if not token_data:
                logger.error("Invalid token")
                return
            
            # Decrypt data
            encrypted_data = task_json["data"]
            decrypted_data = WorkerSecurity.decrypt_data(encrypted_data)
            data = json.loads(decrypted_data)
            
            # Create task object
            task = Task(metadata=metadata, data=data, token=token)
            
            # Get handler
            handler = task_handlers.get(metadata.module)
            if not handler:
                logger.error(f"No handler found for module: {metadata.module}")
                return
            
            # Process task
            result = handler(task)
            
            # Store result
            self.redis_client.setex(
                f"result:{metadata.task_id}", 
                3600,  # 1 hour expiration
                json.dumps(result)
            )
            
            logger.info(f"Task completed: {metadata.task_id}")
            
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            # Increment retry counter
            metadata.retries += 1
            if metadata.retries < metadata.max_retries:
                # Requeue task
                self.redis_client.rpush("tasks", task_data)
                logger.info(f"Task requeued: {metadata.task_id} (retry {metadata.retries})")
            else:
                logger.error(f"Task failed after {metadata.max_retries} retries: {metadata.task_id}")
    
    def run(self):
        """Run the worker service"""
        logger.info(f"Starting XORB worker {WORKER_ID}")
        try:
            while self.running:
                # Wait for task
                task_data = self.redis_client.brpop("tasks", timeout=1)
                if task_data:
                    self.process_task(task_data[1])
        except Exception as e:
            logger.error(f"Worker error: {str(e)}")
        finally:
            logger.info("Worker stopped")

if __name__ == "__main__":
    worker = WorkerService()
    worker.run()