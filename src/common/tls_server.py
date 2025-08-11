"""
TLS/mTLS Server Implementation for XORB Platform
Provides secure server configuration with client certificate verification
"""

import ssl
import logging
import asyncio
from typing import Optional, Dict, Any, Callable, List
from pathlib import Path
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response
import cryptography.x509
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class TLSServerConfig:
    """TLS Server Configuration for secure FastAPI applications"""
    
    def __init__(
        self,
        cert_file: str,
        key_file: str,
        ca_file: Optional[str] = None,
        require_client_cert: bool = False,
        allowed_client_subjects: Optional[List[str]] = None,
        min_tls_version: str = "TLSv1.2",
        ciphers: Optional[str] = None
    ):
        self.cert_file = cert_file
        self.key_file = key_file
        self.ca_file = ca_file
        self.require_client_cert = require_client_cert
        self.allowed_client_subjects = allowed_client_subjects or []
        self.min_tls_version = min_tls_version
        self.ciphers = ciphers or (
            "TLS_AES_256_GCM_SHA384:"
            "TLS_CHACHA20_POLY1305_SHA256:"
            "TLS_AES_128_GCM_SHA256:"
            "ECDHE-ECDSA-AES256-GCM-SHA384:"
            "ECDHE-ECDSA-CHACHA20-POLY1305:"
            "ECDHE-ECDSA-AES128-GCM-SHA256:"
            "ECDHE-RSA-AES256-GCM-SHA384:"
            "ECDHE-RSA-CHACHA20-POLY1305:"
            "ECDHE-RSA-AES128-GCM-SHA256"
        )
        
    def create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for the server"""
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        
        # Set TLS version constraints
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.maximum_version = ssl.TLSVersion.TLSv1_3
        
        # Configure cipher suites
        context.set_ciphers(self.ciphers)
        
        # Set ECDH curve
        context.set_ecdh_curve("prime256v1")
        
        # Load server certificate and key
        if not Path(self.cert_file).exists():
            raise FileNotFoundError(f"Server certificate not found: {self.cert_file}")
        if not Path(self.key_file).exists():
            raise FileNotFoundError(f"Server private key not found: {self.key_file}")
            
        context.load_cert_chain(certfile=self.cert_file, keyfile=self.key_file)
        logger.info(f"Loaded server certificate: {self.cert_file}")
        
        # Configure client certificate verification
        if self.require_client_cert:
            context.verify_mode = ssl.CERT_REQUIRED
            
            if self.ca_file and Path(self.ca_file).exists():
                context.load_verify_locations(cafile=self.ca_file)
                logger.info(f"Loaded CA certificate for client verification: {self.ca_file}")
            else:
                logger.warning("Client certificate required but no CA file provided")
        else:
            context.verify_mode = ssl.CERT_NONE
            
        # Security settings
        context.options |= ssl.OP_NO_SSLv2
        context.options |= ssl.OP_NO_SSLv3
        context.options |= ssl.OP_NO_TLSv1
        context.options |= ssl.OP_NO_TLSv1_1
        context.options |= ssl.OP_SINGLE_DH_USE
        context.options |= ssl.OP_SINGLE_ECDH_USE
        context.options |= ssl.OP_CIPHER_SERVER_PREFERENCE
        
        return context
        
    def validate(self) -> bool:
        """Validate the TLS configuration"""
        try:
            # Check if certificate files exist
            if not Path(self.cert_file).exists():
                logger.error(f"Certificate file not found: {self.cert_file}")
                return False
                
            if not Path(self.key_file).exists():
                logger.error(f"Private key file not found: {self.key_file}")
                return False
                
            if self.ca_file and not Path(self.ca_file).exists():
                logger.error(f"CA file not found: {self.ca_file}")
                return False
                
            # Try to create SSL context
            self.create_ssl_context()
            logger.info("TLS configuration validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"TLS configuration validation failed: {e}")
            return False


class ClientCertificateMiddleware(BaseHTTPMiddleware):
    """Middleware to handle client certificate authentication"""
    
    def __init__(
        self,
        app,
        allowed_subjects: Optional[List[str]] = None,
        require_client_cert: bool = True
    ):
        super().__init__(app)
        self.allowed_subjects = allowed_subjects or []
        self.require_client_cert = require_client_cert
        
    async def dispatch(
        self, 
        request: Request, 
        call_next: RequestResponseEndpoint
    ) -> Response:
        # Extract client certificate information from headers (set by Envoy)
        client_subject = request.headers.get("x-tls-subject")
        client_serial = request.headers.get("x-tls-cert-serial")
        client_fingerprint = request.headers.get("x-tls-cert-fingerprint")
        
        # Store certificate info in request state
        request.state.client_cert = {
            "subject": client_subject,
            "serial": client_serial,
            "fingerprint": client_fingerprint,
            "verified": False
        }
        
        # Verify client certificate if required
        if self.require_client_cert:
            if not client_subject:
                logger.warning("Client certificate required but not provided")
                raise HTTPException(
                    status_code=401,
                    detail="Client certificate required"
                )
                
            # Check if subject is in allowed list
            if self.allowed_subjects and client_subject not in self.allowed_subjects:
                logger.warning(f"Client certificate subject not allowed: {client_subject}")
                raise HTTPException(
                    status_code=403,
                    detail="Client certificate not authorized"
                )
                
            request.state.client_cert["verified"] = True
            logger.info(f"Client authenticated with certificate: {client_subject}")
        
        # Add security headers
        response = await call_next(request)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response


def get_client_certificate_info(request: Request) -> Dict[str, Any]:
    """Extract client certificate information from request"""
    if hasattr(request.state, "client_cert"):
        return request.state.client_cert
    return {
        "subject": None,
        "serial": None,
        "fingerprint": None,
        "verified": False
    }


def require_client_cert(request: Request) -> Dict[str, Any]:
    """Dependency to require valid client certificate"""
    cert_info = get_client_certificate_info(request)
    if not cert_info.get("verified", False):
        raise HTTPException(
            status_code=401,
            detail="Valid client certificate required"
        )
    return cert_info


class SecureUvicornServer:
    """Secure Uvicorn server with TLS configuration"""
    
    def __init__(
        self,
        app: FastAPI,
        host: str = "0.0.0.0",
        port: int = 8000,
        tls_config: Optional[TLSServerConfig] = None,
        workers: int = 1
    ):
        self.app = app
        self.host = host
        self.port = port
        self.tls_config = tls_config
        self.workers = workers
        
    def run(self):
        """Run the secure server"""
        config_kwargs = {
            "app": self.app,
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "loop": "asyncio",
            "http": "httptools",
            "log_level": "info",
            "access_log": True,
            "server_header": False,
            "date_header": False
        }
        
        # Add TLS configuration if provided
        if self.tls_config:
            if not self.tls_config.validate():
                raise ValueError("Invalid TLS configuration")
                
            ssl_context = self.tls_config.create_ssl_context()
            config_kwargs.update({
                "ssl_context": ssl_context,
                "ssl_version": ssl.PROTOCOL_TLS_SERVER,
                "ssl_cert_reqs": ssl.CERT_REQUIRED if self.tls_config.require_client_cert else ssl.CERT_NONE,
                "ssl_ca_certs": self.tls_config.ca_file
            })
            
            logger.info(
                f"Starting secure server on {self.host}:{self.port} "
                f"(TLS, client cert: {self.tls_config.require_client_cert})"
            )
        else:
            logger.warning(f"Starting INSECURE server on {self.host}:{self.port}")
            
        uvicorn.run(**config_kwargs)


def create_tls_server_config_from_env() -> Optional[TLSServerConfig]:
    """Create TLS server configuration from environment variables"""
    import os
    
    cert_file = os.getenv("API_TLS_CERT")
    key_file = os.getenv("API_TLS_KEY")
    
    if not cert_file or not key_file:
        logger.warning("TLS certificate/key not configured via environment variables")
        return None
        
    ca_file = os.getenv("API_TLS_CA")
    require_client_cert = os.getenv("API_REQUIRE_CLIENT_CERT", "false").lower() == "true"
    
    # Parse allowed client subjects from environment
    allowed_subjects = []
    subjects_env = os.getenv("API_ALLOWED_CLIENT_SUBJECTS", "")
    if subjects_env:
        allowed_subjects = [s.strip() for s in subjects_env.split(",") if s.strip()]
    
    return TLSServerConfig(
        cert_file=cert_file,
        key_file=key_file,
        ca_file=ca_file,
        require_client_cert=require_client_cert,
        allowed_client_subjects=allowed_subjects,
        min_tls_version=os.getenv("API_TLS_MIN_VERSION", "TLSv1.2")
    )


# Example usage for FastAPI application
def create_secure_app() -> FastAPI:
    """Create a FastAPI application with TLS middleware"""
    app = FastAPI(
        title="XORB Secure API",
        description="XORB Platform API with mTLS security",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add client certificate middleware
    allowed_subjects = [
        "CN=orchestrator.xorb.local,OU=Services,O=XORB Platform,L=San Francisco,ST=CA,C=US",
        "CN=agent.xorb.local,OU=Services,O=XORB Platform,L=San Francisco,ST=CA,C=US"
    ]
    
    app.add_middleware(
        ClientCertificateMiddleware,
        allowed_subjects=allowed_subjects,
        require_client_cert=True
    )
    
    @app.get("/api/v1/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "tls": "enabled"}
    
    @app.get("/api/v1/secure")
    async def secure_endpoint(
        request: Request,
        cert_info: Dict[str, Any] = Depends(require_client_cert)
    ):
        """Secure endpoint requiring client certificate"""
        return {
            "message": "Authenticated via client certificate",
            "client_subject": cert_info.get("subject"),
            "client_serial": cert_info.get("serial")
        }
    
    return app


if __name__ == "__main__":
    # Example: Run secure server
    app = create_secure_app()
    tls_config = create_tls_server_config_from_env()
    
    server = SecureUvicornServer(
        app=app,
        host="0.0.0.0",
        port=8000,
        tls_config=tls_config
    )
    
    server.run()