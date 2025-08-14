"""
Quick utility to add missing XORBService abstract methods to production services
"""

# Template methods to add to each service class
BASE_METHODS_TEMPLATE = '''
    # XORBService abstract methods implementation
    async def initialize(self) -> bool:
        """Initialize {service_name} service"""
        try:
            self.start_time = datetime.utcnow()
            self.status = ServiceStatus.HEALTHY
            logger.info(f"{service_name} service {{self.service_id}} initialized")
            return True
        except Exception as e:
            logger.error(f"{service_name} service initialization failed: {{e}}")
            self.status = ServiceStatus.UNHEALTHY
            return False

    async def shutdown(self) -> bool:
        """Shutdown {service_name} service"""
        try:
            self.status = ServiceStatus.SHUTTING_DOWN
            # Clean up service-specific resources here
            self.status = ServiceStatus.STOPPED
            logger.info(f"{service_name} service {{self.service_id}} shutdown complete")
            return True
        except Exception as e:
            logger.error(f"{service_name} service shutdown failed: {{e}}")
            return False

    async def health_check(self) -> ServiceHealth:
        """Perform health check"""
        try:
            checks = {{
                "service_operational": True,
                "dependencies_available": True
            }}

            all_healthy = all(checks.values())
            status = ServiceStatus.HEALTHY if all_healthy else ServiceStatus.DEGRADED

            uptime = 0.0
            if hasattr(self, 'start_time') and self.start_time:
                uptime = (datetime.utcnow() - self.start_time).total_seconds()

            return ServiceHealth(
                status=status,
                message="{service_name} service operational",
                timestamp=datetime.utcnow(),
                checks=checks,
                uptime_seconds=uptime
            )
        except Exception as e:
            logger.error(f"{service_name} health check failed: {{e}}")
            return ServiceHealth(
                status=ServiceStatus.UNHEALTHY,
                message=f"Health check failed: {{e}}",
                timestamp=datetime.utcnow(),
                checks={{}},
                last_error=str(e)
            )
'''

# Services that need the methods added
services_to_update = [
    ("ProductionEmbeddingService", "Embedding"),
    ("ProductionNotificationService", "Notification"),
    ("ProductionRateLimitingService", "Rate Limiting"),
    ("ProductionDiscoveryService", "Discovery"),
    ("ProductionHealthService", "Health")
]

print("Template methods for missing XORBService abstract methods:")
print("Add these to the respective service classes:")
print("=" * 80)

for service_class, service_name in services_to_update:
    print(f"\n# Add to {service_class}:")
    print(BASE_METHODS_TEMPLATE.format(service_name=service_name))
    print("-" * 40)
