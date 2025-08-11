"""
Enterprise-grade encryption at rest and in transit
Implements AES-256-GCM for data at rest and TLS 1.3 for data in transit
"""

import os
import base64
import secrets
import hashlib
from typing import Optional, Union, Dict, Any, Tuple
import ipaddress
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.x509.oid import NameOID
from cryptography import x509
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class EncryptionManager:
    """Centralized encryption management for XORB platform"""
    
    def __init__(self, master_key: Optional[str] = None):
        self.master_key = master_key or self._get_master_key()
        self.fernet = self._create_fernet_instance()
        self.aes_gcm = AESGCM(self._derive_aes_key())
        
    def _get_master_key(self) -> str:
        """Get or generate master encryption key"""
        # In production, this should come from a secure key management service
        key = os.getenv("ENCRYPTION_MASTER_KEY")
        if not key:
            # Generate a new key for development
            key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
            logger.warning("Generated new master key for development use")
        return key
        
    def _create_fernet_instance(self) -> Fernet:
        """Create Fernet instance for symmetric encryption"""
        key_bytes = base64.urlsafe_b64decode(self.master_key.encode())
        return Fernet(base64.urlsafe_b64encode(key_bytes[:32]))
        
    def _derive_aes_key(self) -> bytes:
        """Derive AES key from master key"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'xorb_encryption_salt',  # In production, use random salt per operation
            iterations=100000,
        )
        return kdf.derive(self.master_key.encode())
        
    def encrypt_data(self, data: Union[str, bytes], context: str = "") -> str:
        """Encrypt data using Fernet (symmetric encryption)"""
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        # Add context for authenticated encryption
        if context:
            data = f"{context}|{data.decode('utf-8')}".encode('utf-8')
            
        encrypted = self.fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')
        
    def decrypt_data(self, encrypted_data: str, context: str = "") -> str:
        """Decrypt data using Fernet"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            decrypted = self.fernet.decrypt(encrypted_bytes)
            
            if context:
                decrypted_str = decrypted.decode('utf-8')
                if not decrypted_str.startswith(f"{context}|"):
                    raise ValueError("Invalid context for decryption")
                return decrypted_str[len(context)+1:]
            
            return decrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Decryption failed")
            
    def encrypt_aes_gcm(self, data: Union[str, bytes], associated_data: Optional[bytes] = None) -> Dict[str, str]:
        """Encrypt data using AES-GCM for authenticated encryption"""
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        # Generate random nonce
        nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
        
        # Encrypt with authentication
        ciphertext = self.aes_gcm.encrypt(nonce, data, associated_data)
        
        return {
            'ciphertext': base64.urlsafe_b64encode(ciphertext).decode('utf-8'),
            'nonce': base64.urlsafe_b64encode(nonce).decode('utf-8'),
            'algorithm': 'AES-256-GCM'
        }
        
    def decrypt_aes_gcm(self, encrypted_dict: Dict[str, str], associated_data: Optional[bytes] = None) -> str:
        """Decrypt AES-GCM encrypted data"""
        try:
            ciphertext = base64.urlsafe_b64decode(encrypted_dict['ciphertext'].encode('utf-8'))
            nonce = base64.urlsafe_b64decode(encrypted_dict['nonce'].encode('utf-8'))
            
            plaintext = self.aes_gcm.decrypt(nonce, ciphertext, associated_data)
            return plaintext.decode('utf-8')
        except Exception as e:
            logger.error(f"AES-GCM decryption failed: {e}")
            raise ValueError("Decryption failed")
            
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Dict[str, str]:
        """Hash password using PBKDF2 with SHA-256"""
        if salt is None:
            salt = secrets.token_bytes(32)
            
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = kdf.derive(password.encode('utf-8'))
        
        return {
            'hash': base64.urlsafe_b64encode(key).decode('utf-8'),
            'salt': base64.urlsafe_b64encode(salt).decode('utf-8'),
            'algorithm': 'PBKDF2-SHA256',
            'iterations': 100000
        }
        
    def verify_password(self, password: str, password_hash: Dict[str, str]) -> bool:
        """Verify password against hash"""
        try:
            salt = base64.urlsafe_b64decode(password_hash['salt'].encode('utf-8'))
            stored_hash = base64.urlsafe_b64decode(password_hash['hash'].encode('utf-8'))
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=password_hash.get('iterations', 100000),
            )
            
            computed_hash = kdf.derive(password.encode('utf-8'))
            return secrets.compare_digest(stored_hash, computed_hash)
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False


class CertificateManager:
    """Manages TLS certificates for secure communication"""
    
    def __init__(self):
        self.cert_directory = "secrets/ssl"
        os.makedirs(self.cert_directory, exist_ok=True)
        
    def generate_self_signed_cert(
        self, 
        common_name: str = "localhost",
        validity_days: int = 365
    ) -> Tuple[str, str]:
        """Generate self-signed certificate for development"""
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
        )
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "State"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "City"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "XORB"),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=validity_days)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(common_name),
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        # Serialize to PEM format
        cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode('utf-8')
        key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')
        
        return cert_pem, key_pem
        
    def save_certificate(self, cert_pem: str, key_pem: str, name: str):
        """Save certificate and key to secure location"""
        cert_path = os.path.join(self.cert_directory, f"{name}.crt")
        key_path = os.path.join(self.cert_directory, f"{name}.key")
        
        # Save certificate (world readable)
        with open(cert_path, 'w') as f:
            f.write(cert_pem)
        os.chmod(cert_path, 0o644)
        
        # Save private key (owner only)
        with open(key_path, 'w') as f:
            f.write(key_pem)
        os.chmod(key_path, 0o600)
        
        logger.info(f"Certificate saved: {cert_path}")
        logger.info(f"Private key saved: {key_path}")
        
    def get_tls_config(self) -> Dict[str, Any]:
        """Get TLS configuration for secure communication"""
        return {
            "tls_version": "TLSv1.3",
            "cipher_suites": [
                "TLS_AES_256_GCM_SHA384",
                "TLS_AES_128_GCM_SHA256",
                "TLS_CHACHA20_POLY1305_SHA256"
            ],
            "certificate_path": os.path.join(self.cert_directory, "xorb.crt"),
            "private_key_path": os.path.join(self.cert_directory, "xorb.key"),
            "require_client_cert": False,
            "verify_mode": "CERT_REQUIRED"
        }


class DatabaseEncryption:
    """Database-level encryption for sensitive fields"""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        
    def encrypt_field(self, value: Any, field_name: str, table_name: str) -> str:
        """Encrypt a database field with context"""
        if value is None:
            return None
            
        context = f"{table_name}.{field_name}"
        return self.encryption_manager.encrypt_data(str(value), context)
        
    def decrypt_field(self, encrypted_value: str, field_name: str, table_name: str) -> str:
        """Decrypt a database field with context"""
        if not encrypted_value:
            return None
            
        context = f"{table_name}.{field_name}"
        return self.encryption_manager.decrypt_data(encrypted_value, context)
        
    def encrypt_json_field(self, json_data: Dict[str, Any], table_name: str) -> str:
        """Encrypt JSON data for database storage"""
        import json
        json_str = json.dumps(json_data, sort_keys=True)
        return self.encryption_manager.encrypt_aes_gcm(
            json_str, 
            associated_data=table_name.encode('utf-8')
        )
        
    def decrypt_json_field(self, encrypted_data: Dict[str, str], table_name: str) -> Dict[str, Any]:
        """Decrypt JSON data from database"""
        import json
        json_str = self.encryption_manager.decrypt_aes_gcm(
            encrypted_data,
            associated_data=table_name.encode('utf-8')
        )
        return json.loads(json_str)


# Global encryption instances
encryption_manager = EncryptionManager()
certificate_manager = CertificateManager()
database_encryption = DatabaseEncryption(encryption_manager)


def encrypt_sensitive_data(data: str, context: str = "") -> str:
    """Convenience function for encrypting sensitive data"""
    return encryption_manager.encrypt_data(data, context)


def decrypt_sensitive_data(encrypted_data: str, context: str = "") -> str:
    """Convenience function for decrypting sensitive data"""
    return encryption_manager.decrypt_data(encrypted_data, context)


def setup_development_certificates():
    """Setup self-signed certificates for development"""
    try:
        cert_pem, key_pem = certificate_manager.generate_self_signed_cert("xorb-dev")
        certificate_manager.save_certificate(cert_pem, key_pem, "xorb-dev")
        logger.info("Development certificates generated successfully")
    except Exception as e:
        logger.error(f"Failed to generate development certificates: {e}")


if __name__ == "__main__":
    # Generate development certificates
    setup_development_certificates()
    
    # Test encryption
    test_data = "sensitive information"
    encrypted = encrypt_sensitive_data(test_data, "test")
    decrypted = decrypt_sensitive_data(encrypted, "test")
    
    print(f"Original: {test_data}")
    print(f"Encrypted: {encrypted}")
    print(f"Decrypted: {decrypted}")
    print(f"Encryption working: {test_data == decrypted}")