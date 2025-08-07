from typing import Dict, Any

class SecurityManager:
    """Regional security manager with fallback to default settings"""
    
    def __init__(self, region: str = "global"):
        self.region = region
        self.encryption = self.configure_encryption()
        self.authentication = self.configure_authentication()
        self.compliance = self.configure_compliance()
        self.audit = self.configure_audit()
    
    def configure_encryption(self) -> Dict[str, Any]:
        """Configure encryption settings based on region"""
        encryption = {
            "data": "AES-128-GCM",
            "key_rotation": "180d",
            "quantum_safe": False
        }
        
        if self.region in ["EU", "Government"]:
            encryption["data"] = "AES-256-GCM"
            encryption["key_rotation"] = "90d"
            encryption["quantum_safe"] = (self.region == "Government")
        
        return encryption
    
    def configure_authentication(self) -> Dict[str, Any]:
        """Configure authentication settings based on region"""
        return {
            "mfa": self.region in ["EU", "Healthcare", "Finance"],
            "biometric": self.region in ["Government", "Defense"],
            "session_timeout": "15m" if self.region in ["EU", "Healthcare"] else "1h"
        }
    
    def configure_compliance(self) -> Dict[str, Any]:
        """Configure compliance settings based on region"""
        return {
            "gdpr": self.region in ["EU", "UK"],
            "ccpa": self.region in ["US"],
            "data_localization": self.region in ["EU", "China", "Russia"]
        }
    
    def configure_audit(self) -> Dict[str, Any]:
        """Configure audit settings based on region"""
        return {
            "level": "full" if self.region in ["EU", "Government"] else "basic",
            "retention": "5y" if self.region in ["EU", "Healthcare"] else "1y",
            "encryption": self.region in ["EU", "Government"]
        }
    
    def get_encryption_settings(self) -> Dict[str, Any]:
        """Get encryption settings"""
        return self.encryption
    
    def get_authentication_settings(self) -> Dict[str, Any]:
        """Get authentication settings"""
        return self.authentication
    
    def get_compliance_settings(self) -> Dict[str, Any]:
        """Get compliance settings"""
        return self.compliance
    
    def get_audit_settings(self) -> Dict[str, Any]:
        """Get audit settings"""
        return self.audit
    
    def is_compliant_region(self) -> bool:
        """Check if region has compliance requirements"""
        return any([self.compliance["gdpr"], self.compliance["ccpa"], self.compliance["data_localization"]])
    
    def requires_data_localization(self) -> bool:
        """Check if data localization is required"""
        return self.compliance["data_localization"]
    
    def requires_mfa(self) -> bool:
        """Check if MFA is required"""
        return self.authentication["mfa"]
    
    def requires_biometric_auth(self) -> bool:
        """Check if biometric authentication is required"""
        return self.authentication["biometric"]
    
    def requires_encrypted_audit(self) -> bool:
        """Check if audit logs need to be encrypted"""
        return self.audit["encryption"]
    
    def get_key_rotation_period(self) -> str:
        """Get key rotation period"""
        return self.encryption["key_rotation"]
    
    def is_quantum_safe(self) -> bool:
        """Check if quantum-safe encryption is required"""
        return self.encryption["quantum_safe"]
    
    def get_session_timeout(self) -> str:
        """Get session timeout setting"""
        return self.authentication["session_timeout"]
    
    def get_audit_retention(self) -> str:
        """Get audit retention period"""
        return self.audit["retention"]
    
    def get_audit_level(self) -> str:
        """Get audit level"""
        return self.audit["level"]
    
    def get_encryption_type(self) -> str:
        """Get encryption type"""
        return self.encryption["data"]
    
    def get_region(self) -> str:
        """Get current region"""
        return self.region
    
    def set_region(self, region: str) -> None:
        """Set new region and update settings"""
        self.region = region
        self.encryption = self.configure_encryption()
        self.authentication = self.configure_authentication()
        self.compliance = self.configure_compliance()
        self.audit = self.configure_audit()