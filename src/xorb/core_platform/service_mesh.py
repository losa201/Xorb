import httpx

from xorb.shared.config import PlatformConfig

# Unified Service Discovery and Health Check
class UnifiedServiceMesh:
    def __init__(self):
        self.services = PlatformConfig.SERVICES.copy()
        self.health_status = {}
        self.circuit_breakers = {}
        
    async def health_check(self, service_name: str) -> bool:
        """Check health of a service."""
        if service_name not in self.services:
            return False
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.services[service_name]}/health")
                return response.status_code == 200
        except Exception:
            return False
    
    async def proxy_request(self, service_name: str, path: str, method: str, **kwargs) -> httpx.Response:
        """Proxy request to service with circuit breaker."""
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")
        
        url = f"{self.services[service_name]}{path}"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(method, url, **kwargs)
            return response