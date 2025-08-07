import logging
import json
from datetime import datetime
from cryptography.fernet import Fernet
from typing import Dict, Any

class SecureAuditLogger:
    """Secure audit logging with optional encryption"""
    
    def __init__(self, encryption_key: str, compliance_enabled: bool = True):
        self.logger = logging.getLogger('xorb_audit')
        self.compliance_enabled = compliance_enabled
        
        # Configure logging format
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Initialize encryption if compliance is enabled
        self.cipher = Fernet(encryption_key) if compliance_enabled else None

    def log_event(self, user_id: str, event_type: str, details: Dict[str, Any]) -> None:
        """Log event with optional encryption based on compliance settings"""
        audit_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'event_type': event_type,
            'details': details
        }
        
        if self.compliance_enabled and self.cipher:
            # Encrypt the record if compliance is enabled
            encrypted_record = self.cipher.encrypt(json.dumps(audit_record).encode())
            self.logger.info(encrypted_record.decode())
        else:
            # Log in plain text if compliance is not required
            self.logger.info(json.dumps(audit_record))

    def set_level(self, level: int) -> None:
        """Set the logging level"""
        self.logger.setLevel(level)