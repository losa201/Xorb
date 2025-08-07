from xorb.core.config.loader import ConfigLoader
from typing import Dict, Any

class RegionalConfig:
    """Regional configuration manager with fallback to default settings"""
    
    def __init__(self, region: str = "global"):
        self.region = region
        self.config_loader = ConfigLoader()
        self.region_config = self._load_region_config()
    
    def _load_region_config(self) -> Dict[str, Any]:
        """Load region-specific configuration"""
        return self.config_loader.load_config(f"region_{self.region}")
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration for the region"""
        return self.region_config.get("security", {})
    
    def get_compliance_config(self) -> Dict[str, Any]:
        """Get compliance configuration for the region"""
        return self.region_config.get("compliance", {})
    
    def get_service_config(self, service_name: str) -> Dict[str, Any]:
        """Get service-specific configuration for the region"""
        return self.region_config.get("services", {}).get(service_name, {})