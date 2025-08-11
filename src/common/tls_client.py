"""
TLS/mTLS Client Implementation for XORB Platform
Provides secure client connections with certificate verification
"""

import ssl
import logging
import asyncio
from typing import Optional, Dict, Any, Union
from pathlib import Path
import aiohttp
import redis.asyncio as aioredis
from aiohttp import ClientSession, TCPConnector
from aiohttp.client_exceptions import ClientError

logger = logging.getLogger(__name__)


class TLSConfig:
    """TLS Configuration for secure connections"""
    
    def __init__(
        self,
        cert_file: Optional[str] = None,
        key_file: Optional[str] = None,
        ca_file: Optional[str] = None,
        verify_hostname: bool = True,
        check_hostname: bool = True,
        min_tls_version: str = "TLSv1.2"
    ):
        self.cert_file = cert_file
        self.key_file = key_file
        self.ca_file = ca_file
        self.verify_hostname = verify_hostname
        self.check_hostname = check_hostname
        self.min_tls_version = getattr(ssl, f"PROTOCOL_{min_tls_version.replace('.', '_')}")
        
    def create_ssl_context(self, purpose: ssl.Purpose = ssl.Purpose.SERVER_AUTH) -> ssl.SSLContext:
        """Create SSL context with security best practices"""
        context = ssl.create_default_context(purpose=purpose)
        
        # Set minimum TLS version
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.maximum_version = ssl.TLSVersion.TLSv1_3
        
        # Configure cipher suites for security
        context.set_ciphers(
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
        
        # Set ECDH curves
        context.set_ecdh_curve("prime256v1")
        
        # Hostname verification
        context.check_hostname = self.check_hostname
        context.verify_mode = ssl.CERT_REQUIRED if self.verify_hostname else ssl.CERT_NONE
        
        # Load CA certificate if provided
        if self.ca_file:
            if Path(self.ca_file).exists():
                context.load_verify_locations(cafile=self.ca_file)
                logger.debug(f"Loaded CA certificate from {self.ca_file}")
            else:
                logger.warning(f"CA file not found: {self.ca_file}")
                
        # Load client certificate if provided (for mTLS)
        if self.cert_file and self.key_file:
            if Path(self.cert_file).exists() and Path(self.key_file).exists():
                context.load_cert_chain(certfile=self.cert_file, keyfile=self.key_file)
                logger.debug(f"Loaded client certificate from {self.cert_file}")
            else:
                logger.warning(f"Client cert/key files not found: {self.cert_file}, {self.key_file}")
                
        return context


class SecureHTTPClient:
    """Secure HTTP client with mTLS support"""
    
    def __init__(self, tls_config: TLSConfig, timeout: int = 30):
        self.tls_config = tls_config
        self.timeout = timeout
        self._session: Optional[ClientSession] = None
        
    async def __aenter__(self):
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def start(self):
        """Initialize the HTTP client session"""
        if self._session is None:
            ssl_context = self.tls_config.create_ssl_context()
            connector = TCPConnector(
                ssl=ssl_context,
                limit=100,
                limit_per_host=30,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "User-Agent": "XORB-Platform/1.0",
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                }
            )
            logger.debug("Secure HTTP client session started")
            
    async def close(self):
        """Close the HTTP client session"""
        if self._session:
            await self._session.close()
            self._session = None
            logger.debug("Secure HTTP client session closed")
            
    async def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Union[str, bytes, Dict[str, Any]]] = None,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None
    ) -> aiohttp.ClientResponse:
        """Make a secure HTTP request"""
        if not self._session:
            await self.start()
            
        try:
            async with self._session.request(
                method=method,
                url=url,
                headers=headers,
                data=data,
                json=json,
                params=params
            ) as response:
                # Log the request for audit purposes
                logger.info(
                    f"HTTP {method} {url} -> {response.status}",
                    extra={
                        "method": method,
                        "url": url,
                        "status_code": response.status,
                        "response_headers": dict(response.headers)
                    }
                )
                return response
                
        except ClientError as e:
            logger.error(f"HTTP request failed: {method} {url} - {e}")
            raise
            
    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """GET request"""
        return await self.request("GET", url, **kwargs)
        
    async def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """POST request"""
        return await self.request("POST", url, **kwargs)
        
    async def put(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """PUT request"""
        return await self.request("PUT", url, **kwargs)
        
    async def delete(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """DELETE request"""
        return await self.request("DELETE", url, **kwargs)


class SecureRedisClient:
    """Secure Redis client with TLS support"""
    
    def __init__(
        self,
        host: str = "redis",
        port: int = 6379,
        tls_config: Optional[TLSConfig] = None,
        password: Optional[str] = None,
        db: int = 0
    ):
        self.host = host
        self.port = port
        self.tls_config = tls_config
        self.password = password
        self.db = db
        self._pool: Optional[aioredis.ConnectionPool] = None
        self._redis: Optional[aioredis.Redis] = None
        
    async def __aenter__(self):
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def connect(self):
        """Connect to Redis with TLS"""
        connection_kwargs = {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "decode_responses": True,
            "socket_keepalive": True,
            "socket_keepalive_options": {
                1: 1,  # TCP_KEEPIDLE
                2: 3,  # TCP_KEEPINTVL
                3: 5   # TCP_KEEPCNT
            },
            "health_check_interval": 30
        }
        
        if self.password:
            connection_kwargs["password"] = self.password
            
        # Configure TLS if provided
        if self.tls_config:
            ssl_context = self.tls_config.create_ssl_context()
            connection_kwargs.update({
                "ssl": True,
                "ssl_context": ssl_context,
                "ssl_check_hostname": self.tls_config.check_hostname
            })
            
        try:
            self._pool = aioredis.ConnectionPool(**connection_kwargs)
            self._redis = aioredis.Redis(connection_pool=self._pool)
            
            # Test connection
            await self._redis.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port} (TLS: {bool(self.tls_config)})")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
            
    async def close(self):
        """Close Redis connection"""
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            
        if self._pool:
            await self._pool.aclose()
            self._pool = None
            
        logger.debug("Redis connection closed")
        
    @property
    def redis(self) -> aioredis.Redis:
        """Get Redis client instance"""
        if not self._redis:
            raise RuntimeError("Redis client not connected. Call connect() first.")
        return self._redis


def create_tls_config_from_env() -> TLSConfig:
    """Create TLS configuration from environment variables"""
    import os
    
    return TLSConfig(
        cert_file=os.getenv("TLS_CERT_FILE"),
        key_file=os.getenv("TLS_KEY_FILE"),
        ca_file=os.getenv("TLS_CA_FILE", "/run/tls/ca/ca.pem"),
        verify_hostname=os.getenv("TLS_VERIFY_HOSTNAME", "true").lower() == "true",
        check_hostname=os.getenv("TLS_CHECK_HOSTNAME", "true").lower() == "true",
        min_tls_version=os.getenv("TLS_MIN_VERSION", "TLSv1.2")
    )


async def create_secure_http_client(
    cert_file: Optional[str] = None,
    key_file: Optional[str] = None,
    ca_file: Optional[str] = None,
    **kwargs
) -> SecureHTTPClient:
    """Factory function to create a secure HTTP client"""
    tls_config = TLSConfig(
        cert_file=cert_file,
        key_file=key_file,
        ca_file=ca_file,
        **kwargs
    )
    
    client = SecureHTTPClient(tls_config)
    await client.start()
    return client


async def create_secure_redis_client(
    host: str = "redis",
    port: int = 6379,
    cert_file: Optional[str] = None,
    key_file: Optional[str] = None,
    ca_file: Optional[str] = None,
    password: Optional[str] = None,
    **kwargs
) -> SecureRedisClient:
    """Factory function to create a secure Redis client"""
    tls_config = TLSConfig(
        cert_file=cert_file,
        key_file=key_file,
        ca_file=ca_file,
        **kwargs
    ) if cert_file or ca_file else None
    
    client = SecureRedisClient(
        host=host,
        port=port,
        tls_config=tls_config,
        password=password
    )
    await client.connect()
    return client