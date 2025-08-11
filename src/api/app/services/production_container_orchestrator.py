"""
Production Container and Service Orchestrator
Unified dependency injection container with service lifecycle management
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Type, Callable
from datetime import datetime
import inspect

from .production_service_implementations import (
    ProductionAuthenticationService, ProductionAuthorizationService,
    ProductionPTaaSService, ProductionHealthService, create_service_instances
)
from .production_intelligence_service import ProductionThreatIntelligenceService
from .interfaces import *

logger = logging.getLogger(__name__)


class ProductionServiceContainer:
    """
    Production-ready dependency injection container with advanced service management
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._services = {}
        self._service_instances = {}
        self._initialization_order = []
        self._is_initialized = False
        self._redis_client = None
        self._db_client = None
        
        # Service health tracking
        self._service_health = {}
        self._last_health_check = None
        
        # Register core services
        self._register_core_services()
    
    def _register_core_services(self):
        """Register all core service implementations"""
        
        # Authentication and Authorization
        self.register_service(
            "authentication_service",
            ProductionAuthenticationService,
            AuthenticationService,
            dependencies=["redis_client"],
            singleton=True
        )
        
        self.register_service(
            "authorization_service", 
            ProductionAuthorizationService,
            AuthorizationService,
            dependencies=["redis_client"],
            singleton=True
        )
        
        # PTaaS Services
        self.register_service(
            "ptaas_service",
            ProductionPTaaSService,
            PTaaSService,
            dependencies=["redis_client"],
            singleton=True
        )
        
        # Threat Intelligence
        self.register_service(
            "threat_intelligence_service",
            ProductionThreatIntelligenceService,
            ThreatIntelligenceService,
            dependencies=["redis_client"],
            singleton=True
        )
        
        # Health Service
        self.register_service(
            "health_service",
            ProductionHealthService,
            HealthService,
            dependencies=["redis_client", "db_client"],
            singleton=True
        )
        
        # Additional services
        self._register_advanced_services()
    
    def _register_advanced_services(self):
        """Register advanced service implementations"""
        
        # Mock implementations for services not yet fully implemented
        
        # Rate Limiting Service
        self.register_service(
            "rate_limiting_service",
            MockRateLimitingService,
            RateLimitingService,
            dependencies=["redis_client"],
            singleton=True
        )
        
        # Notification Service
        self.register_service(
            "notification_service",
            MockNotificationService,
            NotificationService,
            dependencies=[],
            singleton=True
        )
        
        # Security Service
        self.register_service(
            "security_service",
            MockSecurityService,
            SecurityService,
            dependencies=[],
            singleton=True
        )
        
        # Embedding Service
        self.register_service(
            "embedding_service",
            MockEmbeddingService,
            EmbeddingService,
            dependencies=[],
            singleton=True
        )
        
        # User and Organization Services
        self.register_service(
            "user_service",
            MockUserService,
            UserService,
            dependencies=["db_client"],
            singleton=True
        )
        
        self.register_service(
            "organization_service",
            MockOrganizationService,
            OrganizationService,
            dependencies=["db_client"],
            singleton=True
        )
        
        # Tenant Service
        self.register_service(
            "tenant_service",
            MockTenantService,
            TenantService,
            dependencies=["db_client"],
            singleton=True
        )
        
        # Discovery Service
        self.register_service(
            "discovery_service",
            MockDiscoveryService,
            DiscoveryService,
            dependencies=["redis_client"],
            singleton=True
        )
        
        # Security Orchestration
        self.register_service(
            "security_orchestration_service",
            MockSecurityOrchestrationService,
            SecurityOrchestrationService,
            dependencies=["redis_client"],
            singleton=True
        )
        
        # Compliance Service
        self.register_service(
            "compliance_service",
            MockComplianceService,
            ComplianceService,
            dependencies=["db_client"],
            singleton=True
        )
        
        # Security Monitoring
        self.register_service(
            "security_monitoring_service",
            MockSecurityMonitoringService,
            SecurityMonitoringService,
            dependencies=["redis_client"],
            singleton=True
        )
    
    def register_service(
        self,
        name: str,
        implementation: Type,
        interface: Type,
        dependencies: List[str] = None,
        singleton: bool = True,
        factory: Callable = None
    ):
        """Register a service implementation"""
        self._services[name] = {
            "implementation": implementation,
            "interface": interface,
            "dependencies": dependencies or [],
            "singleton": singleton,
            "factory": factory,
            "instance": None,
            "initialized": False
        }
        
        # Add to initialization order based on dependencies
        self._update_initialization_order(name)
    
    def _update_initialization_order(self, service_name: str):
        """Update service initialization order based on dependencies"""
        if service_name in self._initialization_order:
            return
        
        service_info = self._services[service_name]
        
        # Add dependencies first
        for dep in service_info["dependencies"]:
            if dep in self._services and dep not in self._initialization_order:
                self._update_initialization_order(dep)
        
        # Add this service
        self._initialization_order.append(service_name)
    
    async def initialize_all_services(self, redis_client=None, db_client=None):
        """Initialize all registered services"""
        try:
            logger.info("🔧 Initializing Production Service Container...")
            
            self._redis_client = redis_client
            self._db_client = db_client
            
            # Initialize infrastructure services first
            await self._initialize_infrastructure()
            
            # Initialize services in dependency order
            initialized_count = 0
            failed_count = 0
            
            for service_name in self._initialization_order:
                try:
                    await self._initialize_service(service_name)
                    initialized_count += 1
                    logger.info(f"✅ Initialized service: {service_name}")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize service {service_name}: {e}")
                    failed_count += 1
            
            self._is_initialized = True
            
            logger.info(f"📦 Service Container Initialization Complete")
            logger.info(f"   • Services Initialized: {initialized_count}")
            logger.info(f"   • Services Failed: {failed_count}")
            logger.info(f"   • Total Services: {len(self._services)}")
            
            return {
                "initialized": initialized_count,
                "failed": failed_count,
                "total": len(self._services)
            }
            
        except Exception as e:
            logger.error(f"❌ Service container initialization failed: {e}")
            raise
    
    async def _initialize_infrastructure(self):
        """Initialize infrastructure components"""
        logger.info("🔧 Initializing infrastructure components...")
        
        # Redis client setup
        if self._redis_client:
            try:
                await self._redis_client.ping()
                logger.info("✅ Redis client connected")
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed: {e}")
                self._redis_client = None
        
        # Database client setup
        if self._db_client:
            logger.info("✅ Database client configured")
        else:
            logger.warning("⚠️ Database client not configured")
    
    async def _initialize_service(self, service_name: str):
        """Initialize a single service"""
        service_info = self._services[service_name]
        
        if service_info["initialized"]:
            return service_info["instance"]
        
        # Resolve dependencies
        dependencies = {}
        for dep_name in service_info["dependencies"]:
            if dep_name == "redis_client":
                dependencies["redis_client"] = self._redis_client
            elif dep_name == "db_client":
                dependencies["db_client"] = self._db_client
            elif dep_name in self._services:
                dep_instance = await self._initialize_service(dep_name)
                dependencies[dep_name] = dep_instance
            else:
                logger.warning(f"⚠️ Dependency {dep_name} not found for service {service_name}")
        
        # Create service instance
        if service_info["factory"]:
            instance = service_info["factory"](**dependencies)
        else:
            # Use constructor with dependency injection
            constructor_params = self._get_constructor_params(
                service_info["implementation"], dependencies
            )
            instance = service_info["implementation"](**constructor_params)
        
        # Store instance if singleton
        if service_info["singleton"]:
            service_info["instance"] = instance
        
        service_info["initialized"] = True
        
        # Perform service-specific initialization if needed
        if hasattr(instance, "initialize"):
            await instance.initialize()
        
        return instance
    
    def _get_constructor_params(self, implementation: Type, dependencies: Dict[str, Any]) -> Dict[str, Any]:
        """Get constructor parameters based on available dependencies"""
        try:
            sig = inspect.signature(implementation.__init__)
            params = {}
            
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                
                if param_name in dependencies:
                    params[param_name] = dependencies[param_name]
                elif param_name == "config":
                    params["config"] = self.config
                elif param.default != inspect.Parameter.empty:
                    # Use default value
                    continue
                else:
                    logger.warning(f"⚠️ Missing required parameter: {param_name}")
            
            return params
            
        except Exception as e:
            logger.error(f"❌ Failed to resolve constructor params: {e}")
            return {}
    
    def get_service(self, service_name: str) -> Any:
        """Get a service instance by name"""
        if not self._is_initialized:
            raise RuntimeError("Container not initialized")
        
        if service_name not in self._services:
            raise ValueError(f"Service not registered: {service_name}")
        
        service_info = self._services[service_name]
        
        if service_info["singleton"]:
            return service_info["instance"]
        else:
            # Create new instance for non-singleton services
            dependencies = self._resolve_dependencies(service_info["dependencies"])
            constructor_params = self._get_constructor_params(
                service_info["implementation"], dependencies
            )
            return service_info["implementation"](**constructor_params)
    
    def _resolve_dependencies(self, dependency_names: List[str]) -> Dict[str, Any]:
        """Resolve dependencies by name"""
        dependencies = {}
        
        for dep_name in dependency_names:
            if dep_name == "redis_client":
                dependencies["redis_client"] = self._redis_client
            elif dep_name == "db_client":
                dependencies["db_client"] = self._db_client
            elif dep_name in self._services:
                dependencies[dep_name] = self.get_service(dep_name)
        
        return dependencies
    
    async def health_check_all_services(self) -> Dict[str, Any]:
        """Perform health check on all services"""
        try:
            health_results = {
                "overall_status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "services": {},
                "total_services": len(self._services),
                "healthy_services": 0,
                "unhealthy_services": 0
            }
            
            for service_name, service_info in self._services.items():
                try:
                    if not service_info["initialized"]:
                        health_results["services"][service_name] = {
                            "status": "not_initialized",
                            "message": "Service not initialized"
                        }
                        continue
                    
                    instance = service_info["instance"]
                    
                    # Check if service has health check method
                    if hasattr(instance, "health_check"):
                        health_result = await instance.health_check()
                    elif hasattr(instance, "check_service_health"):
                        health_result = await instance.check_service_health(service_name)
                    else:
                        # Default health check - just verify instance exists
                        health_result = {
                            "status": "healthy",
                            "message": "Service instance active"
                        }
                    
                    health_results["services"][service_name] = health_result
                    
                    if health_result.get("status") == "healthy":
                        health_results["healthy_services"] += 1
                    else:
                        health_results["unhealthy_services"] += 1
                        
                except Exception as e:
                    logger.error(f"Health check failed for {service_name}: {e}")
                    health_results["services"][service_name] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    health_results["unhealthy_services"] += 1
            
            # Determine overall status
            if health_results["unhealthy_services"] > 0:
                if health_results["healthy_services"] == 0:
                    health_results["overall_status"] = "unhealthy"
                else:
                    health_results["overall_status"] = "degraded"
            
            self._last_health_check = datetime.utcnow()
            return health_results
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "overall_status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status"""
        return {
            "container_initialized": self._is_initialized,
            "registered_services": len(self._services),
            "initialized_services": len([s for s in self._services.values() if s["initialized"]]),
            "services": {
                name: {
                    "initialized": info["initialized"],
                    "singleton": info["singleton"],
                    "has_instance": info["instance"] is not None
                }
                for name, info in self._services.items()
            },
            "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None
        }
    
    async def shutdown_all_services(self) -> Dict[str, Any]:
        """Shutdown all services gracefully"""
        try:
            logger.info("🛑 Shutting down Production Service Container...")
            
            shutdown_count = 0
            failed_count = 0
            
            # Shutdown services in reverse order
            for service_name in reversed(self._initialization_order):
                try:
                    service_info = self._services[service_name]
                    
                    if service_info["initialized"] and service_info["instance"]:
                        instance = service_info["instance"]
                        
                        # Call shutdown method if available
                        if hasattr(instance, "shutdown"):
                            await instance.shutdown()
                        elif hasattr(instance, "close"):
                            await instance.close()
                        
                        # Clear instance
                        service_info["instance"] = None
                        service_info["initialized"] = False
                        
                        shutdown_count += 1
                        logger.info(f"✅ Shutdown service: {service_name}")
                        
                except Exception as e:
                    logger.error(f"❌ Failed to shutdown service {service_name}: {e}")
                    failed_count += 1
            
            self._is_initialized = False
            
            logger.info(f"📦 Service Container Shutdown Complete")
            logger.info(f"   • Services Shutdown: {shutdown_count}")
            logger.info(f"   • Shutdown Failures: {failed_count}")
            
            return {
                "shutdown": shutdown_count,
                "failed": failed_count,
                "total": len(self._services)
            }
            
        except Exception as e:
            logger.error(f"❌ Service container shutdown failed: {e}")
            return {
                "shutdown": 0,
                "failed": len(self._services),
                "error": str(e)
            }


# Mock service implementations for services not yet fully implemented

class MockRateLimitingService(RateLimitingService):
    """Mock implementation of rate limiting service"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
    
    async def check_rate_limit(self, key: str, rule_name: str = "api_global", 
                              tenant_id: Optional[UUID] = None, 
                              user_role: Optional[str] = None) -> RateLimitInfo:
        # Mock implementation - always allow
        return RateLimitInfo(
            allowed=True,
            remaining=1000,
            reset_time=time.time() + 3600,
            limit=1000
        )
    
    async def increment_usage(self, key: str, rule_name: str = "api_global",
                             tenant_id: Optional[UUID] = None, cost: int = 1) -> bool:
        return True
    
    async def get_usage_stats(self, key: str, tenant_id: Optional[UUID] = None,
                             time_range_hours: int = 24) -> UsageStats:
        return UsageStats(
            current_usage=10,
            limit=1000,
            reset_time=time.time() + 3600,
            time_range_hours=time_range_hours
        )


class MockNotificationService(NotificationService):
    """Mock implementation of notification service"""
    
    async def send_notification(self, recipient: str, channel: str, message: str,
                               subject: Optional[str] = None, priority: str = "normal",
                               variables: Optional[Dict[str, Any]] = None,
                               attachments: Optional[List[Dict[str, Any]]] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> str:
        logger.info(f"Mock notification sent to {recipient}: {message}")
        return str(uuid4())
    
    async def send_webhook(self, url: str, payload: Dict[str, Any],
                          headers: Optional[Dict[str, str]] = None,
                          secret: Optional[str] = None, retry_count: int = 3) -> bool:
        logger.info(f"Mock webhook sent to {url}")
        return True


class MockSecurityService(SecurityService):
    """Mock implementation of security service"""
    
    async def analyze_security_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "analysis_id": str(uuid4()),
            "risk_score": 0.3,
            "findings": ["No significant threats detected"],
            "recommendations": ["Continue monitoring"]
        }
    
    async def assess_risk(self, context: Dict[str, Any]) -> float:
        return 0.3  # Low risk


class MockEmbeddingService(EmbeddingService):
    """Mock implementation of embedding service"""
    
    async def generate_embeddings(self, texts: List[str], model: str, input_type: str,
                                 user: User, org: Organization) -> EmbeddingResult:
        # Mock embeddings
        embeddings = [[0.1, 0.2, 0.3] * 100 for _ in texts]  # 300-dimensional mock embeddings
        
        return EmbeddingResult(
            embeddings=embeddings,
            model=model,
            input_type=input_type,
            usage={"tokens": len(texts) * 10}
        )
    
    async def compute_similarity(self, text1: str, text2: str, model: str, user: User) -> float:
        # Mock similarity
        return 0.85
    
    async def batch_embeddings(self, texts: List[str], model: str, batch_size: int,
                              input_type: str, user: User, org: Organization) -> EmbeddingResult:
        return await self.generate_embeddings(texts, model, input_type, user, org)
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        return [
            {"name": "text-embedding-ada-002", "dimensions": 1536},
            {"name": "sentence-transformers", "dimensions": 384}
        ]


class MockUserService(UserService):
    """Mock implementation of user service"""
    
    def __init__(self, db_client=None):
        self.db_client = db_client
    
    async def create_user(self, username: str, email: str, password: str, roles: List[str]) -> User:
        return User(
            id=uuid4(),
            username=username,
            email=email,
            roles=roles,
            is_active=True,
            created_at=datetime.utcnow()
        )
    
    async def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        # Mock user
        return User(
            id=user_id,
            username="mock_user",
            email="user@example.com",
            roles=["user"],
            is_active=True,
            created_at=datetime.utcnow()
        )
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        if username == "admin":
            return User(
                id=uuid4(),
                username="admin",
                email="admin@example.com",
                roles=["admin"],
                is_active=True,
                created_at=datetime.utcnow()
            )
        return None
    
    async def update_user(self, user: User) -> User:
        return user
    
    async def deactivate_user(self, user_id: UUID) -> bool:
        return True


class MockOrganizationService(OrganizationService):
    """Mock implementation of organization service"""
    
    def __init__(self, db_client=None):
        self.db_client = db_client
    
    async def create_organization(self, name: str, plan_type: str, owner: User) -> Organization:
        return Organization(
            id=uuid4(),
            name=name,
            plan_type=plan_type,
            owner_id=owner.id if hasattr(user, 'id') else uuid4(),
            created_at=datetime.utcnow()
        )
    
    async def get_organization_by_id(self, org_id: UUID) -> Optional[Organization]:
        return Organization(
            id=org_id,
            name="Mock Organization",
            plan_type="enterprise",
            owner_id=uuid4(),
            created_at=datetime.utcnow()
        )
    
    async def update_organization(self, organization: Organization) -> Organization:
        return organization
    
    async def get_user_organizations(self, user: User) -> List[Organization]:
        return [
            Organization(
                id=uuid4(),
                name="Default Organization",
                plan_type="enterprise",
                owner_id=user.id if hasattr(user, 'id') else uuid4(),
                created_at=datetime.utcnow()
            )
        ]


class MockTenantService(TenantService):
    """Mock implementation of tenant service"""
    
    def __init__(self, db_client=None):
        self.db_client = db_client
    
    async def create_tenant(self, name: str, slug: str, plan_type: TenantPlan,
                           settings: Dict[str, Any] = None) -> Tenant:
        return Tenant(
            id=uuid4(),
            name=name,
            slug=slug,
            plan=plan_type,
            status=TenantStatus.ACTIVE,
            settings=settings or {},
            created_at=datetime.utcnow()
        )
    
    async def get_tenant_by_id(self, tenant_id: UUID) -> Optional[Tenant]:
        return Tenant(
            id=tenant_id,
            name="Mock Tenant",
            slug="mock-tenant",
            plan=TenantPlan.ENTERPRISE,
            status=TenantStatus.ACTIVE,
            settings={},
            created_at=datetime.utcnow()
        )
    
    async def get_tenant_by_slug(self, slug: str) -> Optional[Tenant]:
        return Tenant(
            id=uuid4(),
            name="Mock Tenant",
            slug=slug,
            plan=TenantPlan.ENTERPRISE,
            status=TenantStatus.ACTIVE,
            settings={},
            created_at=datetime.utcnow()
        )
    
    async def update_tenant(self, tenant: Tenant) -> Tenant:
        return tenant
    
    async def deactivate_tenant(self, tenant_id: UUID) -> bool:
        return True


class MockDiscoveryService(DiscoveryService):
    """Mock implementation of discovery service"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
    
    async def start_discovery(self, domain: str, user: User, org: Organization) -> DiscoveryWorkflow:
        return DiscoveryWorkflow(
            id=str(uuid4()),
            domain=domain,
            status="running",
            user_id=user.id if hasattr(user, 'id') else str(uuid4()),
            org_id=org.id if hasattr(org, 'id') else str(uuid4()),
            created_at=datetime.utcnow()
        )
    
    async def get_discovery_results(self, workflow_id: str, user: User) -> Optional[DiscoveryWorkflow]:
        return DiscoveryWorkflow(
            id=workflow_id,
            domain="example.com",
            status="completed",
            user_id=user.id if hasattr(user, 'id') else str(uuid4()),
            org_id=str(uuid4()),
            results={"subdomains": ["www.example.com", "mail.example.com"]},
            created_at=datetime.utcnow()
        )
    
    async def get_user_workflows(self, user: User, limit: int = 50, offset: int = 0) -> List[DiscoveryWorkflow]:
        return [
            DiscoveryWorkflow(
                id=str(uuid4()),
                domain="example.com",
                status="completed",
                user_id=user.id if hasattr(user, 'id') else str(uuid4()),
                org_id=str(uuid4()),
                created_at=datetime.utcnow()
            )
        ]


class MockSecurityOrchestrationService(SecurityOrchestrationService):
    """Mock implementation of security orchestration service"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
    
    async def create_workflow(self, workflow_definition: Dict[str, Any], 
                             user: User, org: Organization) -> Dict[str, Any]:
        return {
            "workflow_id": str(uuid4()),
            "status": "created",
            "definition": workflow_definition
        }
    
    async def execute_workflow(self, workflow_id: str, parameters: Dict[str, Any], 
                              user: User) -> Dict[str, Any]:
        return {
            "execution_id": str(uuid4()),
            "workflow_id": workflow_id,
            "status": "running",
            "parameters": parameters
        }
    
    async def get_workflow_status(self, execution_id: str, user: User) -> Dict[str, Any]:
        return {
            "execution_id": execution_id,
            "status": "completed",
            "progress": 100,
            "results": {"success": True}
        }
    
    async def schedule_recurring_scan(self, targets: List[str], schedule: str,
                                     scan_config: Dict[str, Any], user: User) -> Dict[str, Any]:
        return {
            "schedule_id": str(uuid4()),
            "targets": targets,
            "schedule": schedule,
            "next_run": datetime.utcnow() + timedelta(hours=24)
        }


class MockComplianceService(ComplianceService):
    """Mock implementation of compliance service"""
    
    def __init__(self, db_client=None):
        self.db_client = db_client
    
    async def validate_compliance(self, framework: str, scan_results: Dict[str, Any],
                                 organization: Organization) -> Dict[str, Any]:
        return {
            "framework": framework,
            "compliance_score": 0.85,
            "passed_controls": 15,
            "failed_controls": 3,
            "recommendations": ["Update security controls", "Implement additional monitoring"]
        }
    
    async def generate_compliance_report(self, framework: str, time_period: str,
                                        organization: Organization) -> Dict[str, Any]:
        return {
            "report_id": str(uuid4()),
            "framework": framework,
            "time_period": time_period,
            "compliance_score": 0.85,
            "summary": "Organization meets most compliance requirements"
        }
    
    async def get_compliance_gaps(self, framework: str, 
                                 current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {
                "control_id": "AC-1",
                "description": "Access Control Policy",
                "gap": "Policy needs updating",
                "remediation": "Update access control policy"
            }
        ]
    
    async def track_remediation_progress(self, compliance_issues: List[str],
                                        organization: Organization) -> Dict[str, Any]:
        return {
            "total_issues": len(compliance_issues),
            "resolved_issues": len(compliance_issues) // 2,
            "progress": 50.0,
            "estimated_completion": datetime.utcnow() + timedelta(days=30)
        }


class MockSecurityMonitoringService(SecurityMonitoringService):
    """Mock implementation of security monitoring service"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
    
    async def start_real_time_monitoring(self, targets: List[str], 
                                        monitoring_config: Dict[str, Any],
                                        user: User) -> Dict[str, Any]:
        return {
            "monitoring_id": str(uuid4()),
            "targets": targets,
            "status": "active",
            "started_at": datetime.utcnow()
        }
    
    async def get_security_alerts(self, organization: Organization,
                                 severity_filter: Optional[str] = None,
                                 limit: int = 100) -> List[Dict[str, Any]]:
        return [
            {
                "alert_id": str(uuid4()),
                "severity": "medium",
                "title": "Suspicious network activity detected",
                "description": "Unusual traffic patterns observed",
                "timestamp": datetime.utcnow()
            }
        ]
    
    async def create_alert_rule(self, rule_definition: Dict[str, Any],
                               organization: Organization, user: User) -> Dict[str, Any]:
        return {
            "rule_id": str(uuid4()),
            "definition": rule_definition,
            "status": "active"
        }
    
    async def investigate_incident(self, incident_id: str,
                                  investigation_parameters: Dict[str, Any],
                                  user: User) -> Dict[str, Any]:
        return {
            "investigation_id": str(uuid4()),
            "incident_id": incident_id,
            "status": "in_progress",
            "findings": ["Initial analysis shows no immediate threat"]
        }


# Container factory function
async def create_production_container(config: Dict[str, Any] = None, 
                                    redis_client=None, db_client=None) -> ProductionServiceContainer:
    """Factory function to create and initialize production container"""
    container = ProductionServiceContainer(config)
    await container.initialize_all_services(redis_client, db_client)
    return container