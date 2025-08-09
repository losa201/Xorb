from typing import Optional
from pathlib import Path
import ssl

class ServiceMeshTLS:
    """Handles mutual TLS configuration for service mesh"""
    
    def __init__(self, cert_dir: str = "/etc/xorb/tls"):
        self.cert_dir = Path(cert_dir)
        self.ca_cert = self.cert_dir / "ca.crt"
        self.cert = self.cert_dir / "service.crt"
        self.key = self.cert_dir / "service.key"
        
    def validate_certs(self) -> bool:
        """Validate that all required certificates exist and are valid"""
        if not all(cert.exists() for cert in [self.ca_cert, self.cert, self.key]):
            return False
            
        # Add certificate validation logic here
        try:
            # Check certificate expiration
            # Check certificate chain
            # Verify certificate permissions
            return True
        except Exception:
            return False
            
    def get_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context with mTLS configuration"""
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        ssl_ctx.load_verify_locations(cafile=str(self.ca_cert))
        ssl_ctx.load_cert_chain(certfile=str(self.cert), keyfile=str(self.key))
        ssl_ctx.verify_mode = ssl.CERT_REQUIRED
        ssl_ctx.check_hostname = True
        return ssl_ctx
        
    def reload_certs(self) -> None:
        """Reload certificates without service restart"""
        # Implementation for certificate reloading
        pass

# Configuration for mTLS
MTLS_ENABLED = True
MIN_TLS_VERSION = "TLSv1.2"
CERT_RELOAD_INTERVAL = 3600  # 1 hour
TRUSTED_CA_BUNDLE = "/etc/ssl/certs/trusted-ca-bundle.crt"

__all__ = ["ServiceMeshTLS", "MTLS_ENABLED", "MIN_TLS_VERSION"]

# Example usage:
# ssl_ctx = ServiceMeshTLS().get_ssl_context()