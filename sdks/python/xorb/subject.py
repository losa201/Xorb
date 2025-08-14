import re
from typing import Optional

class SubjectBuilder:
    """Builds and validates XORB subjects with format: xorb.<tenant>.<domain>.<service>.<event>"""
    
    PATTERN = r'^xorb\.[a-z0-9-]+\.[a-z0-9-]+\.[a-z0-9-]+\.[a-z0-9-]+$'
    
    def __init__(self):
        self.tenant = None
        self.domain = None
        self.service = None
        self.event = None
        
    def with_tenant(self, tenant: str) -> 'SubjectBuilder':
        """Set the tenant component (alphanumeric + hyphens)"""
        if not re.match(r'^[a-z0-9-]+$', tenant):
            raise ValueError("Tenant must be alphanumeric with hyphens only")
        self.tenant = tenant
        return self
        
    def with_domain(self, domain: str) -> 'SubjectBuilder':
        """Set the domain component (alphanumeric + hyphens)"""
        if not re.match(r'^[a-z0-9-]+$', domain):
            raise ValueError("Domain must be alphanumeric with hyphens only")
        self.domain = domain
        return self
        
    def with_service(self, service: str) -> 'SubjectBuilder':
        """Set the service component (alphanumeric + hyphens)"""
        if not re.match(r'^[a-z0-9-]+$', service):
            raise ValueError("Service must be alphanumeric with hyphens only")
        self.service = service
        return self
        
    def with_event(self, event: str) -> 'SubjectBuilder':
        """Set the event component (alphanumeric + hyphens)"""
        if not re.match(r'^[a-z0-9-]+$', event):
            raise ValueError("Event must be alphanumeric with hyphens only")
        self.event = event
        return self
        
    def build(self) -> str:
        """Build the subject string"""
        if not all([self.tenant, self.domain, self.service, self.event]):
            raise ValueError("All components (tenant, domain, service, event) must be set")
        return f"xorb.{self.tenant}.{self.domain}.{self.service}.{self.event}"
        
    @staticmethod
    def validate(subject: str) -> bool:
        """Validate a subject string against the pattern"""
        return bool(re.match(SubjectBuilder.PATTERN, subject))
        
    @staticmethod
    def parse(subject: str) -> Optional[dict]:
        """Parse a subject string into its components"""
        match = re.match(SubjectBuilder.PATTERN, subject)
        if not match:
            return None
        
        parts = subject.split('.')
        return {
            'tenant': parts[1],
            'domain': parts[2],
            'service': parts[3],
            'event': parts[4]
        }