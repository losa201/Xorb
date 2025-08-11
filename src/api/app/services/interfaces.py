"""
Service interfaces - Define contracts for business operations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from uuid import UUID

from ..domain.entities import (
    User, Organization, EmbeddingRequest, EmbeddingResult, 
    DiscoveryWorkflow, AuthToken
)
from ..domain.value_objects import UsageStats, RateLimitInfo
from ..domain.tenant_entities import Tenant, TenantPlan, TenantStatus


class AuthenticationService(ABC):
    """Interface for unified authentication operations"""
    
    @abstractmethod
    async def authenticate_user(self, credentials) -> Any:
        """Authenticate user with various credential types"""
        raise NotImplementedError("authenticate_user must be implemented by subclass")
    
    @abstractmethod
    async def validate_token(self, token: str) -> Any:
        """Validate access token and return validation result"""
        raise NotImplementedError("validate_token must be implemented by subclass")
    
    @abstractmethod
    async def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Generate new access token from refresh token"""
        raise NotImplementedError("refresh_access_token must be implemented by subclass")
    
    @abstractmethod
    async def logout_user(self, session_id: str) -> bool:
        """Logout user and invalidate session"""
        raise NotImplementedError("logout_user must be implemented by subclass")
    
    @abstractmethod
    def hash_password(self, password: str) -> str:
        """Hash password securely"""
        raise NotImplementedError("hash_password must be implemented by subclass")
    
    @abstractmethod
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        raise NotImplementedError("verify_password must be implemented by subclass")


class AuthorizationService(ABC):
    """Interface for authorization operations"""
    
    @abstractmethod
    async def check_permission(self, user: User, resource: str, action: str) -> bool:
        """Check if user has permission for resource action"""
        raise NotImplementedError("check_permission must be implemented by subclass")
    
    @abstractmethod
    async def get_user_permissions(self, user: User) -> Dict[str, List[str]]:
        """Get all permissions for user"""
        raise NotImplementedError("get_user_permissions must be implemented by subclass")


class EmbeddingService(ABC):
    """Interface for embedding operations"""
    
    @abstractmethod
    async def generate_embeddings(
        self,
        texts: List[str],
        model: str,
        input_type: str,
        user: User,
        org: Organization
    ) -> EmbeddingResult:
        """Generate embeddings for texts"""
        raise NotImplementedError("generate_embeddings must be implemented by subclass")
    
    @abstractmethod
    async def compute_similarity(
        self,
        text1: str,
        text2: str,
        model: str,
        user: User
    ) -> float:
        """Compute similarity between two texts"""
        raise NotImplementedError("compute_similarity must be implemented by subclass")
    
    @abstractmethod
    async def batch_embeddings(
        self,
        texts: List[str],
        model: str,
        batch_size: int,
        input_type: str,
        user: User,
        org: Organization
    ) -> EmbeddingResult:
        """Process large batches of texts"""
        raise NotImplementedError("batch_embeddings must be implemented by subclass")
    
    @abstractmethod
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available embedding models"""
        raise NotImplementedError("get_available_models must be implemented by subclass")


class DiscoveryService(ABC):
    """Interface for discovery operations"""
    
    @abstractmethod
    async def start_discovery(
        self,
        domain: str,
        user: User,
        org: Organization
    ) -> DiscoveryWorkflow:
        """Start a new discovery workflow"""
        raise NotImplementedError("start_discovery must be implemented by subclass")
    
    @abstractmethod
    async def get_discovery_results(
        self,
        workflow_id: str,
        user: User
    ) -> Optional[DiscoveryWorkflow]:
        """Get results from discovery workflow"""
        raise NotImplementedError("get_discovery_results must be implemented by subclass")
    
    @abstractmethod
    async def get_user_workflows(
        self,
        user: User,
        limit: int = 50,
        offset: int = 0
    ) -> List[DiscoveryWorkflow]:
        """Get workflows for user"""
        raise NotImplementedError("get_user_workflows must be implemented by subclass")


class RateLimitService(ABC):
    """Interface for rate limiting operations"""
    
    @abstractmethod
    async def check_rate_limit(
        self,
        org: Organization,
        resource_type: str,
        action: str
    ) -> RateLimitInfo:
        """Check rate limit for organization and resource"""
        raise NotImplementedError("check_rate_limit must be implemented by subclass")
    
    @abstractmethod
    async def increment_usage(
        self,
        org: Organization,
        resource_type: str,
        amount: int = 1
    ) -> None:
        """Increment resource usage"""
        raise NotImplementedError("increment_usage must be implemented by subclass")
    
    @abstractmethod
    async def get_usage_stats(
        self,
        org: Organization
    ) -> Dict[str, Any]:
        """Get usage statistics for organization"""
        raise NotImplementedError("get_usage_stats must be implemented by subclass")


class UserService(ABC):
    """Interface for user management operations"""
    
    @abstractmethod
    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: List[str]
    ) -> User:
        """Create a new user"""
        raise NotImplementedError("create_user must be implemented by subclass")
    
    @abstractmethod
    async def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID"""
        raise NotImplementedError("get_user_by_id must be implemented by subclass")
    
    @abstractmethod
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        raise NotImplementedError("get_user_by_username must be implemented by subclass")
    
    @abstractmethod
    async def update_user(self, user: User) -> User:
        """Update user information"""
        raise NotImplementedError("update_user must be implemented by subclass")
    
    @abstractmethod
    async def deactivate_user(self, user_id: UUID) -> bool:
        """Deactivate a user"""
        raise NotImplementedError("deactivate_user must be implemented by subclass")


class OrganizationService(ABC):
    """Interface for organization management operations"""
    
    @abstractmethod
    async def create_organization(
        self,
        name: str,
        plan_type: str,
        owner: User
    ) -> Organization:
        """Create a new organization"""
        raise NotImplementedError("create_organization must be implemented by subclass")
    
    @abstractmethod
    async def get_organization_by_id(self, org_id: UUID) -> Optional[Organization]:
        """Get organization by ID"""
        raise NotImplementedError("get_organization_by_id must be implemented by subclass")
    
    @abstractmethod
    async def update_organization(self, organization: Organization) -> Organization:
        """Update organization information"""
        raise NotImplementedError("update_organization must be implemented by subclass")
    
    @abstractmethod
    async def get_user_organizations(self, user: User) -> List[Organization]:
        """Get organizations for user"""
        raise NotImplementedError("get_user_organizations must be implemented by subclass")


class SecurityService(ABC):
    """Interface for security-related operations"""
    
    @abstractmethod
    async def analyze_security_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze security data and return insights"""
        raise NotImplementedError("analyze_security_data must be implemented by subclass")
    
    @abstractmethod
    async def assess_risk(self, context: Dict[str, Any]) -> float:
        """Assess risk level for given context"""
        raise NotImplementedError("assess_risk must be implemented by subclass")


class NotificationService(ABC):
    """Interface for notification operations"""
    
    @abstractmethod
    async def send_notification(
        self,
        user: User,
        message: str,
        notification_type: str
    ) -> bool:
        """Send notification to user"""
        raise NotImplementedError("send_notification must be implemented by subclass")
    
    @abstractmethod
    async def send_webhook(
        self,
        org: Organization,
        event: str,
        data: Dict[str, Any]
    ) -> bool:
        """Send webhook to organization"""
        raise NotImplementedError("send_webhook must be implemented by subclass")


class TenantService(ABC):
    """Interface for tenant management operations"""
    
    @abstractmethod
    async def create_tenant(
        self,
        name: str,
        slug: str,
        plan_type: TenantPlan,
        settings: Dict[str, Any] = None
    ) -> Tenant:
        """Create a new tenant"""
        raise NotImplementedError("create_tenant must be implemented by subclass")
    
    @abstractmethod
    async def get_tenant_by_id(self, tenant_id: UUID) -> Optional[Tenant]:
        """Get tenant by ID"""
        raise NotImplementedError("get_tenant_by_id must be implemented by subclass")
    
    @abstractmethod
    async def get_tenant_by_slug(self, slug: str) -> Optional[Tenant]:
        """Get tenant by slug"""
        raise NotImplementedError("get_tenant_by_slug must be implemented by subclass")
    
    @abstractmethod
    async def update_tenant(self, tenant: Tenant) -> Tenant:
        """Update tenant information"""
        raise NotImplementedError("update_tenant must be implemented by subclass")
    
    @abstractmethod
    async def deactivate_tenant(self, tenant_id: UUID) -> bool:
        """Deactivate a tenant"""
        raise NotImplementedError("deactivate_tenant must be implemented by subclass")


class HealthService(ABC):
    """Interface for health check operations"""
    
    @abstractmethod
    async def check_service_health(self, service_name: str) -> Dict[str, Any]:
        """Check health of a specific service"""
        raise NotImplementedError("check_service_health must be implemented by subclass")
    
    @abstractmethod
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        raise NotImplementedError("get_system_health must be implemented by subclass")


class PTaaSService(ABC):
    """Interface for PTaaS (Penetration Testing as a Service) operations"""
    
    @abstractmethod
    async def create_scan_session(
        self,
        targets: List[Dict[str, Any]],
        scan_type: str,
        user: User,
        org: Organization,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new PTaaS scan session"""
        raise NotImplementedError("create_scan_session must be implemented by subclass")
    
    @abstractmethod
    async def get_scan_status(
        self,
        session_id: str,
        user: User
    ) -> Dict[str, Any]:
        """Get status of a scan session"""
        raise NotImplementedError("get_scan_status must be implemented by subclass")
    
    @abstractmethod
    async def get_scan_results(
        self,
        session_id: str,
        user: User
    ) -> Dict[str, Any]:
        """Get results from a completed scan"""
        raise NotImplementedError("get_scan_results must be implemented by subclass")
    
    @abstractmethod
    async def cancel_scan(
        self,
        session_id: str,
        user: User
    ) -> bool:
        """Cancel an active scan session"""
        raise NotImplementedError("cancel_scan must be implemented by subclass")
    
    @abstractmethod
    async def get_available_scan_profiles(self) -> List[Dict[str, Any]]:
        """Get available scan profiles and their configurations"""
        raise NotImplementedError("get_available_scan_profiles must be implemented by subclass")
    
    @abstractmethod
    async def create_compliance_scan(
        self,
        targets: List[str],
        compliance_framework: str,
        user: User,
        org: Organization
    ) -> Dict[str, Any]:
        """Create compliance-specific scan"""
        raise NotImplementedError("create_compliance_scan must be implemented by subclass")


class ThreatIntelligenceService(ABC):
    """Interface for threat intelligence operations"""
    
    @abstractmethod
    async def analyze_indicators(
        self,
        indicators: List[str],
        context: Dict[str, Any],
        user: User
    ) -> Dict[str, Any]:
        """Analyze threat indicators using AI"""
        raise NotImplementedError("analyze_indicators must be implemented by subclass")
    
    @abstractmethod
    async def correlate_threats(
        self,
        scan_results: Dict[str, Any],
        threat_feeds: List[str] = None
    ) -> Dict[str, Any]:
        """Correlate scan results with threat intelligence"""
        raise NotImplementedError("correlate_threats must be implemented by subclass")
    
    @abstractmethod
    async def get_threat_prediction(
        self,
        environment_data: Dict[str, Any],
        timeframe: str = "24h"
    ) -> Dict[str, Any]:
        """Get AI-powered threat predictions"""
        raise NotImplementedError("get_threat_prediction must be implemented by subclass")
    
    @abstractmethod
    async def generate_threat_report(
        self,
        analysis_results: Dict[str, Any],
        report_format: str = "json"
    ) -> Dict[str, Any]:
        """Generate comprehensive threat intelligence report"""
        raise NotImplementedError("generate_threat_report must be implemented by subclass")


class SecurityOrchestrationService(ABC):
    """Interface for security orchestration and automation"""
    
    @abstractmethod
    async def create_workflow(
        self,
        workflow_definition: Dict[str, Any],
        user: User,
        org: Organization
    ) -> Dict[str, Any]:
        """Create security automation workflow"""
        raise NotImplementedError("create_workflow must be implemented by subclass")
    
    @abstractmethod
    async def execute_workflow(
        self,
        workflow_id: str,
        parameters: Dict[str, Any],
        user: User
    ) -> Dict[str, Any]:
        """Execute a security workflow"""
        raise NotImplementedError("execute_workflow must be implemented by subclass")
    
    @abstractmethod
    async def get_workflow_status(
        self,
        execution_id: str,
        user: User
    ) -> Dict[str, Any]:
        """Get status of workflow execution"""
        raise NotImplementedError("get_workflow_status must be implemented by subclass")
    
    @abstractmethod
    async def schedule_recurring_scan(
        self,
        targets: List[str],
        schedule: str,
        scan_config: Dict[str, Any],
        user: User
    ) -> Dict[str, Any]:
        """Schedule recurring security scans"""
        raise NotImplementedError("schedule_recurring_scan must be implemented by subclass")


class ComplianceService(ABC):
    """Interface for compliance management operations"""
    
    @abstractmethod
    async def validate_compliance(
        self,
        framework: str,
        scan_results: Dict[str, Any],
        organization: Organization
    ) -> Dict[str, Any]:
        """Validate compliance against specific framework"""
        raise NotImplementedError("validate_compliance must be implemented by subclass")
    
    @abstractmethod
    async def generate_compliance_report(
        self,
        framework: str,
        time_period: str,
        organization: Organization
    ) -> Dict[str, Any]:
        """Generate compliance report for specified period"""
        raise NotImplementedError("generate_compliance_report must be implemented by subclass")
    
    @abstractmethod
    async def get_compliance_gaps(
        self,
        framework: str,
        current_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify compliance gaps and remediation steps"""
        raise NotImplementedError("get_compliance_gaps must be implemented by subclass")
    
    @abstractmethod
    async def track_remediation_progress(
        self,
        compliance_issues: List[str],
        organization: Organization
    ) -> Dict[str, Any]:
        """Track progress of compliance remediation efforts"""
        raise NotImplementedError("track_remediation_progress must be implemented by subclass")


class SecurityMonitoringService(ABC):
    """Interface for real-time security monitoring"""
    
    @abstractmethod
    async def start_real_time_monitoring(
        self,
        targets: List[str],
        monitoring_config: Dict[str, Any],
        user: User
    ) -> Dict[str, Any]:
        """Start real-time security monitoring"""
        raise NotImplementedError("start_real_time_monitoring must be implemented by subclass")
    
    @abstractmethod
    async def get_security_alerts(
        self,
        organization: Organization,
        severity_filter: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent security alerts"""
        raise NotImplementedError("get_security_alerts must be implemented by subclass")
    
    @abstractmethod
    async def create_alert_rule(
        self,
        rule_definition: Dict[str, Any],
        organization: Organization,
        user: User
    ) -> Dict[str, Any]:
        """Create custom security alert rule"""
        raise NotImplementedError("create_alert_rule must be implemented by subclass")
    
    @abstractmethod
    async def investigate_incident(
        self,
        incident_id: str,
        investigation_parameters: Dict[str, Any],
        user: User
    ) -> Dict[str, Any]:
        """Perform automated incident investigation"""
        raise NotImplementedError("investigate_incident must be implemented by subclass")


class RateLimitingService(ABC):
    """Interface for rate limiting operations"""
    
    @abstractmethod
    async def check_rate_limit(
        self,
        key: str,
        rule_name: str = "api_global",
        tenant_id: Optional[UUID] = None,
        user_role: Optional[str] = None
    ) -> RateLimitInfo:
        """Check if request is within rate limits"""
        raise NotImplementedError("check_rate_limit must be implemented by subclass")
    
    @abstractmethod
    async def increment_usage(
        self,
        key: str,
        rule_name: str = "api_global",
        tenant_id: Optional[UUID] = None,
        cost: int = 1
    ) -> bool:
        """Increment usage counter for rate limiting"""
        raise NotImplementedError("increment_usage must be implemented by subclass")
    
    @abstractmethod
    async def get_usage_stats(
        self,
        key: str,
        tenant_id: Optional[UUID] = None,
        time_range_hours: int = 24
    ) -> UsageStats:
        """Get usage statistics for a key"""
        raise NotImplementedError("get_usage_stats must be implemented by subclass")


class NotificationService(ABC):
    """Interface for notification operations"""
    
    @abstractmethod
    async def send_notification(
        self,
        recipient: str,
        channel: str,
        message: str,
        subject: Optional[str] = None,
        priority: str = "normal",
        variables: Optional[Dict[str, Any]] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Send a notification"""
        raise NotImplementedError("send_notification must be implemented by subclass")
    
    @abstractmethod
    async def send_webhook(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        secret: Optional[str] = None,
        retry_count: int = 3
    ) -> bool:
        """Send webhook notification"""
        raise NotImplementedError("send_webhook must be implemented by subclass")


class IntelligenceService(ABC):
    """Interface for intelligence and analytics operations"""
    
    @abstractmethod
    async def analyze_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data and return intelligence insights"""
        raise NotImplementedError("analyze_data must be implemented by subclass")
    
    @abstractmethod
    async def get_intelligence_report(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intelligence report based on query"""
        raise NotImplementedError("get_intelligence_report must be implemented by subclass")