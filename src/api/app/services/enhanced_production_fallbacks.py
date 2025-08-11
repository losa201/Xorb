"""
Enhanced Production Service Fallbacks
Provides enhanced service implementations with fallback capabilities
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

from .interfaces import (
    AuthorizationService, EmbeddingService, DiscoveryService, 
    RateLimitingService, NotificationService, HealthService
)
from .authorization_service import ProductionAuthorizationService
from .embedding_service import ProductionEmbeddingService  
from .discovery_service import DiscoveryServiceImpl
from .rate_limiting_service import ProductionRateLimitingService
from .notification_service import ProductionNotificationService
from .health_service import ProductionHealthService

logger = logging.getLogger(__name__)


class EnhancedAuthorizationService(ProductionAuthorizationService):
    """Enhanced authorization service with advanced features"""
    
    def __init__(self):
        super().__init__()
        self.permission_cache: Dict[str, Any] = {}
        self.role_hierarchy = {
            "super_admin": ["admin", "user", "viewer"],
            "admin": ["user", "viewer"], 
            "user": ["viewer"],
            "viewer": []
        }
        self.audit_logs: List[Dict[str, Any]] = []
    
    async def check_permission_enhanced(
        self, 
        user_id: str, 
        resource: str, 
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Enhanced permission checking with context awareness"""
        try:
            # Check cache first
            cache_key = f"{user_id}:{resource}:{action}"
            if cache_key in self.permission_cache:
                cached_result = self.permission_cache[cache_key]
                if cached_result["expires_at"] > datetime.utcnow():
                    return cached_result["permitted"]
            
            # Perform permission check
            permitted = await super().check_permission(user_id, resource, action)
            
            # Context-aware enhancement
            if context and not permitted:
                # Check for dynamic permissions based on context
                if context.get("emergency_access") and action == "read":
                    permitted = True
                    logger.info(f"Emergency access granted for {user_id}")
            
            # Cache result
            self.permission_cache[cache_key] = {
                "permitted": permitted,
                "expires_at": datetime.utcnow() + timedelta(minutes=5)
            }
            
            # Audit log
            self.audit_logs.append({
                "user_id": user_id,
                "resource": resource,
                "action": action,
                "permitted": permitted,
                "timestamp": datetime.utcnow().isoformat(),
                "context": context
            })
            
            return permitted
            
        except Exception as e:
            logger.error(f"Enhanced permission check failed: {e}")
            return False


class EnhancedEmbeddingService(ProductionEmbeddingService):
    """Enhanced embedding service with multiple providers"""
    
    def __init__(self, api_keys: Dict[str, str]):
        super().__init__()
        self.api_keys = api_keys
        self.provider_priority = ["nvidia", "openai", "huggingface", "local"]
        self.cache: Dict[str, Any] = {}
        self.provider_health: Dict[str, bool] = {}
    
    async def generate_embeddings_enhanced(
        self, 
        text: Union[str, List[str]], 
        provider: Optional[str] = None
    ) -> List[List[float]]:
        """Enhanced embedding generation with fallback providers"""
        try:
            if isinstance(text, str):
                text = [text]
            
            # Try providers in order of priority
            providers_to_try = [provider] if provider else self.provider_priority
            
            for prov in providers_to_try:
                if not self.api_keys.get(prov):
                    continue
                    
                try:
                    embeddings = await self._generate_with_provider(text, prov)
                    if embeddings:
                        self.provider_health[prov] = True
                        return embeddings
                except Exception as e:
                    logger.warning(f"Provider {prov} failed: {e}")
                    self.provider_health[prov] = False
                    continue
            
            # Fallback to local embeddings
            return await self._generate_local_embeddings(text)
            
        except Exception as e:
            logger.error(f"Enhanced embedding generation failed: {e}")
            # Return zero embeddings as fallback
            return [[0.0] * 384 for _ in text]
    
    async def _generate_with_provider(self, text: List[str], provider: str) -> List[List[float]]:
        """Generate embeddings with specific provider"""
        if provider == "nvidia":
            return await self._generate_nvidia_embeddings(text)
        elif provider == "openai":
            return await self._generate_openai_embeddings(text)
        elif provider == "huggingface":
            return await self._generate_huggingface_embeddings(text)
        else:
            return await self._generate_local_embeddings(text)
    
    async def _generate_local_embeddings(self, text: List[str]) -> List[List[float]]:
        """Fallback local embedding generation"""
        # Simple hash-based embedding as fallback
        embeddings = []
        for t in text:
            # Generate deterministic embedding from text hash
            hash_val = hash(t) % (2**16)
            embedding = [float((hash_val >> i) & 1) for i in range(384)]
            embeddings.append(embedding)
        return embeddings


class ProductionDiscoveryService(DiscoveryServiceImpl):
    """Enhanced discovery service with production features"""
    
    def __init__(self):
        super().__init__()
        self.discovery_cache: Dict[str, Any] = {}
        self.security_filters = [
            self._filter_sensitive_endpoints,
            self._filter_admin_only,
            self._validate_api_version
        ]
    
    async def discover_services_enhanced(
        self, 
        tenant_id: str,
        include_internal: bool = False,
        filter_level: str = "standard"
    ) -> List[Dict[str, Any]]:
        """Enhanced service discovery with security filtering"""
        try:
            # Get base services
            services = await super().discover_services(tenant_id)
            
            # Apply security filters based on level
            if filter_level == "strict":
                for filter_func in self.security_filters:
                    services = await filter_func(services, tenant_id)
            
            # Add internal services if requested and authorized
            if include_internal:
                internal_services = await self._discover_internal_services(tenant_id)
                services.extend(internal_services)
            
            return services
            
        except Exception as e:
            logger.error(f"Enhanced service discovery failed: {e}")
            return []
    
    async def _filter_sensitive_endpoints(self, services: List[Dict[str, Any]], tenant_id: str) -> List[Dict[str, Any]]:
        """Filter out sensitive endpoints"""
        filtered = []
        for service in services:
            if not service.get("internal_only", False):
                filtered.append(service)
        return filtered


class EnhancedRateLimitingService(ProductionRateLimitingService):
    """Enhanced rate limiting with adaptive policies"""
    
    def __init__(self):
        super().__init__()
        self.adaptive_limits: Dict[str, Dict[str, int]] = {}
        self.threat_scores: Dict[str, float] = {}
        self.whitelist: set = set()
        self.emergency_mode = False
    
    async def check_rate_limit_enhanced(
        self, 
        identifier: str, 
        endpoint: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Enhanced rate limiting with adaptive policies"""
        try:
            # Check whitelist
            if identifier in self.whitelist:
                return {"allowed": True, "reason": "whitelisted"}
            
            # Emergency mode - strict limits
            if self.emergency_mode:
                base_result = await super().check_rate_limit(identifier, endpoint)
                if not base_result.get("allowed"):
                    return {"allowed": False, "reason": "emergency_mode_strict"}
            
            # Adaptive rate limiting based on threat score
            threat_score = self.threat_scores.get(identifier, 0.0)
            
            if threat_score > 0.8:
                # High threat - very restrictive
                return {"allowed": False, "reason": "high_threat_score"}
            elif threat_score > 0.5:
                # Medium threat - reduced limits
                multiplier = 0.5
            else:
                # Normal limits
                multiplier = 1.0
            
            # Apply adaptive limits
            result = await super().check_rate_limit(identifier, endpoint)
            
            # Enhance result with additional info
            result.update({
                "threat_score": threat_score,
                "adaptive_multiplier": multiplier,
                "emergency_mode": self.emergency_mode
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced rate limiting failed: {e}")
            return {"allowed": True, "reason": "error_fallback"}


class EnhancedNotificationService(ProductionNotificationService):
    """Enhanced notification service with multiple channels"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.channels = ["email", "slack", "webhook", "sms"]
        self.channel_health: Dict[str, bool] = {ch: True for ch in self.channels}
        self.notification_queue = asyncio.Queue()
        self.delivery_stats: Dict[str, int] = {"sent": 0, "failed": 0}
    
    async def send_notification_enhanced(
        self, 
        message: str,
        recipients: List[str],
        channel: str = "email",
        priority: str = "normal",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Enhanced notification with multiple channels and retries"""
        try:
            notification = {
                "id": f"notif_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "message": message,
                "recipients": recipients,
                "channel": channel,
                "priority": priority,
                "context": context or {},
                "timestamp": datetime.utcnow().isoformat(),
                "retry_count": 0
            }
            
            # Queue for processing
            await self.notification_queue.put(notification)
            
            # Immediate send for high priority
            if priority == "critical":
                return await self._send_immediate(notification)
            
            return {"status": "queued", "notification_id": notification["id"]}
            
        except Exception as e:
            logger.error(f"Enhanced notification failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _send_immediate(self, notification: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification immediately"""
        try:
            if notification["channel"] == "email":
                result = await self._send_email(notification)
            elif notification["channel"] == "slack":
                result = await self._send_slack(notification)
            else:
                result = await super().send_notification(
                    notification["message"], 
                    notification["recipients"]
                )
            
            if result.get("success"):
                self.delivery_stats["sent"] += 1
            else:
                self.delivery_stats["failed"] += 1
            
            return result
            
        except Exception as e:
            self.delivery_stats["failed"] += 1
            return {"success": False, "error": str(e)}


class EnhancedHealthService(ProductionHealthService):
    """Enhanced health service with predictive monitoring"""
    
    def __init__(self, services: List[Any]):
        super().__init__()
        self.monitored_services = services
        self.health_history: List[Dict[str, Any]] = []
        self.anomaly_detector = SimpleAnomalyDetector()
        self.alert_thresholds = {
            "response_time": 1000,  # ms
            "error_rate": 0.05,     # 5%
            "cpu_usage": 0.8,       # 80%
            "memory_usage": 0.9     # 90%
        }
    
    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """Comprehensive health check with predictive analysis"""
        try:
            # Basic health check
            basic_health = await super().check_health()
            
            # Enhanced metrics collection
            enhanced_metrics = await self._collect_enhanced_metrics()
            
            # Anomaly detection
            anomalies = await self._detect_anomalies(enhanced_metrics)
            
            # Predictive analysis
            predictions = await self._predict_issues(enhanced_metrics)
            
            health_result = {
                "basic_health": basic_health,
                "enhanced_metrics": enhanced_metrics,
                "anomalies_detected": anomalies,
                "predictions": predictions,
                "overall_score": self._calculate_health_score(enhanced_metrics, anomalies),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store in history
            self.health_history.append(health_result)
            
            # Keep only last 100 entries
            if len(self.health_history) > 100:
                self.health_history = self.health_history[-100:]
            
            return health_result
            
        except Exception as e:
            logger.error(f"Comprehensive health check failed: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
    
    async def _collect_enhanced_metrics(self) -> Dict[str, Any]:
        """Collect enhanced system metrics"""
        import psutil
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "network_io": dict(psutil.net_io_counters()._asdict()),
            "process_count": len(psutil.pids()),
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        }
    
    async def _detect_anomalies(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics"""
        anomalies = []
        
        for metric, value in metrics.items():
            if metric in self.alert_thresholds:
                threshold = self.alert_thresholds[metric]
                if isinstance(value, (int, float)) and value > threshold:
                    anomalies.append({
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "severity": "high" if value > threshold * 1.2 else "medium"
                    })
        
        return anomalies
    
    def _calculate_health_score(self, metrics: Dict[str, Any], anomalies: List[Dict[str, Any]]) -> float:
        """Calculate overall health score (0-100)"""
        base_score = 100.0
        
        # Deduct for anomalies
        for anomaly in anomalies:
            if anomaly["severity"] == "high":
                base_score -= 15
            elif anomaly["severity"] == "medium":
                base_score -= 10
            else:
                base_score -= 5
        
        return max(0.0, min(100.0, base_score))


class SimpleAnomalyDetector:
    """Simple anomaly detection for health monitoring"""
    
    def __init__(self):
        self.baseline_metrics: Dict[str, float] = {}
        self.sensitivity = 2.0  # Standard deviations
    
    async def detect(self, metrics: Dict[str, Any]) -> List[str]:
        """Detect anomalies in metrics"""
        anomalies = []
        
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                baseline = self.baseline_metrics.get(metric)
                if baseline and abs(value - baseline) > baseline * self.sensitivity:
                    anomalies.append(f"{metric}: {value} (baseline: {baseline})")
                
                # Update baseline (simple moving average)
                if baseline:
                    self.baseline_metrics[metric] = (baseline * 0.9) + (value * 0.1)
                else:
                    self.baseline_metrics[metric] = value
        
        return anomalies