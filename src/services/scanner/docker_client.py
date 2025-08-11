"""
Secure Docker Client for Scanner Executor
Connects to Docker-in-Docker with TLS client certificates
"""

import asyncio
import logging
import ssl
from typing import Dict, List, Optional, Any
from pathlib import Path
import docker
from docker.client import DockerClient
from docker.errors import DockerException, APIError
import os

logger = logging.getLogger(__name__)


class SecureDockerClient:
    """Secure Docker client with TLS certificate authentication"""
    
    def __init__(
        self,
        host: str = "tcp://dind:2376",
        cert_path: str = "/certs/client",
        ca_file: Optional[str] = None,
        verify_tls: bool = True
    ):
        self.host = host
        self.cert_path = cert_path
        self.ca_file = ca_file or f"{cert_path}/../ca/ca.pem"
        self.verify_tls = verify_tls
        self._client: Optional[DockerClient] = None
        
    def __enter__(self):
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def connect(self):
        """Connect to Docker daemon with TLS"""
        try:
            # Prepare TLS configuration
            tls_config = None
            if self.verify_tls:
                cert_file = f"{self.cert_path}/cert.pem"
                key_file = f"{self.cert_path}/key.pem"
                
                # Verify certificate files exist
                for file_path in [cert_file, key_file, self.ca_file]:
                    if not Path(file_path).exists():
                        raise FileNotFoundError(f"TLS file not found: {file_path}")
                
                tls_config = docker.tls.TLSConfig(
                    client_cert=(cert_file, key_file),
                    ca_cert=self.ca_file,
                    verify=True,
                    ssl_version=ssl.PROTOCOL_TLS,
                    assert_hostname=False  # Using IP addresses in Docker networking
                )
                
                logger.info(f"Connecting to Docker daemon with TLS: {self.host}")
            else:
                logger.warning(f"Connecting to Docker daemon WITHOUT TLS: {self.host}")
            
            # Create Docker client
            self._client = docker.DockerClient(
                base_url=self.host,
                tls=tls_config,
                timeout=30
            )
            
            # Test connection
            version_info = self._client.version()
            logger.info(
                f"Connected to Docker daemon: {version_info.get('Version')} "
                f"(API: {version_info.get('ApiVersion')})"
            )
            
        except Exception as e:
            logger.error(f"Failed to connect to Docker daemon: {e}")
            raise
            
    def close(self):
        """Close Docker client connection"""
        if self._client:
            self._client.close()
            self._client = None
            logger.debug("Docker client connection closed")
            
    @property
    def client(self) -> DockerClient:
        """Get Docker client instance"""
        if not self._client:
            raise RuntimeError("Docker client not connected. Call connect() first.")
        return self._client
    
    def run_security_scan(
        self,
        image: str,
        command: List[str],
        target: str,
        environment: Optional[Dict[str, str]] = None,
        volumes: Optional[Dict[str, Dict[str, str]]] = None,
        network_mode: str = "bridge",
        timeout: int = 300
    ) -> Dict[str, Any]:
        """Run a security scanning container"""
        try:
            container_name = f"xorb-scan-{image.replace('/', '-').replace(':', '-')}-{os.urandom(4).hex()}"
            
            # Prepare environment variables
            env = environment or {}
            env.update({
                "SCAN_TARGET": target,
                "SCAN_TIMESTAMP": str(asyncio.get_event_loop().time()),
                "XORB_SCAN_ID": container_name
            })
            
            # Default volumes for scan results
            scan_volumes = volumes or {}
            scan_volumes.setdefault("/tmp/scan-results", {"bind": "/results", "mode": "rw"})
            
            logger.info(f"Starting security scan: {image} -> {target}")
            
            # Run container
            container = self._client.containers.run(
                image=image,
                command=command,
                name=container_name,
                environment=env,
                volumes=scan_volumes,
                network_mode=network_mode,
                detach=True,
                remove=True,
                mem_limit="512m",
                cpu_quota=50000,  # 50% CPU limit
                security_opt=["no-new-privileges:true"],
                cap_drop=["ALL"],
                cap_add=["NET_RAW", "NET_ADMIN"],  # Required for network scanning
                user="1000:1000"  # Non-root user
            )
            
            # Wait for completion with timeout
            result = container.wait(timeout=timeout)
            logs = container.logs(stdout=True, stderr=True).decode('utf-8')
            
            logger.info(f"Scan completed: {container_name} (exit code: {result['StatusCode']})")
            
            return {
                "container_id": container.id,
                "container_name": container_name,
                "exit_code": result["StatusCode"],
                "logs": logs,
                "target": target,
                "image": image,
                "command": command
            }
            
        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            raise
    
    def run_nmap_scan(
        self,
        target: str,
        scan_type: str = "default",
        ports: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run Nmap security scan"""
        nmap_commands = {
            "quick": ["-sS", "-O", "-F"],
            "default": ["-sS", "-sV", "-O", "-A"],
            "comprehensive": ["-sS", "-sV", "-sC", "-O", "-A", "--script=vuln"],
            "stealth": ["-sS", "-f", "-T2", "--randomize-hosts"]
        }
        
        command = ["nmap"] + nmap_commands.get(scan_type, nmap_commands["default"])
        
        if ports:
            command.extend(["-p", ports])
            
        command.extend(["-oX", "/results/nmap.xml", target])
        
        return self.run_security_scan(
            image="instrumentisto/nmap:latest",
            command=command,
            target=target,
            timeout=600
        )
    
    def run_nuclei_scan(
        self,
        target: str,
        templates: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run Nuclei vulnerability scan"""
        command = ["nuclei", "-target", target, "-json", "-o", "/results/nuclei.json"]
        
        if templates:
            for template in templates:
                command.extend(["-t", template])
        else:
            command.extend(["-t", "cves/", "-t", "vulnerabilities/"])
            
        return self.run_security_scan(
            image="projectdiscovery/nuclei:latest",
            command=command,
            target=target,
            timeout=900
        )
    
    def run_nikto_scan(self, target: str) -> Dict[str, Any]:
        """Run Nikto web vulnerability scan"""
        command = [
            "nikto",
            "-h", target,
            "-Format", "json",
            "-output", "/results/nikto.json"
        ]
        
        return self.run_security_scan(
            image="sullo/nikto:latest",
            command=command,
            target=target,
            timeout=600
        )
    
    def run_sslscan(self, target: str) -> Dict[str, Any]:
        """Run SSL/TLS configuration scan"""
        command = [
            "sslscan",
            "--xml=/results/sslscan.xml",
            target
        ]
        
        return self.run_security_scan(
            image="drwetter/testssl.sh:latest",
            command=command,
            target=target,
            timeout=300
        )
    
    def cleanup_old_containers(self, max_age_hours: int = 24):
        """Clean up old scanning containers"""
        try:
            import datetime
            cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=max_age_hours)
            
            containers = self._client.containers.list(
                all=True,
                filters={"name": "xorb-scan-"}
            )
            
            cleaned = 0
            for container in containers:
                created_time = datetime.datetime.fromisoformat(
                    container.attrs["Created"].replace("Z", "+00:00")
                ).replace(tzinfo=None)
                
                if created_time < cutoff_time:
                    try:
                        container.remove(force=True)
                        cleaned += 1
                        logger.debug(f"Cleaned up old container: {container.name}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up container {container.name}: {e}")
            
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} old scanning containers")
                
        except Exception as e:
            logger.error(f"Container cleanup failed: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get Docker daemon system information"""
        try:
            info = self._client.info()
            version = self._client.version()
            
            return {
                "docker_version": version.get("Version"),
                "api_version": version.get("ApiVersion"),
                "architecture": info.get("Architecture"),
                "operating_system": info.get("OperatingSystem"),
                "total_memory": info.get("MemTotal"),
                "containers_running": info.get("ContainersRunning"),
                "containers_total": info.get("Containers"),
                "images": info.get("Images"),
                "server_version": info.get("ServerVersion"),
                "security_options": info.get("SecurityOptions", [])
            }
            
        except Exception as e:
            logger.error(f"Failed to get Docker system info: {e}")
            return {}


def create_secure_docker_client_from_env() -> SecureDockerClient:
    """Create secure Docker client from environment variables"""
    host = os.getenv("DOCKER_HOST", "tcp://dind:2376")
    cert_path = os.getenv("DOCKER_CERT_PATH", "/certs/client")
    verify_tls = os.getenv("DOCKER_TLS_VERIFY", "1") == "1"
    
    return SecureDockerClient(
        host=host,
        cert_path=cert_path,
        verify_tls=verify_tls
    )


# Example usage
async def example_security_scan():
    """Example of running security scans with secure Docker client"""
    with create_secure_docker_client_from_env() as docker_client:
        target = "scanme.nmap.org"
        
        # Run multiple scans
        scans = [
            docker_client.run_nmap_scan(target, "quick"),
            docker_client.run_nuclei_scan(target),
            docker_client.run_nikto_scan(f"http://{target}"),
        ]
        
        for scan_result in scans:
            print(f"Scan completed: {scan_result['container_name']}")
            print(f"Exit code: {scan_result['exit_code']}")
            if scan_result['exit_code'] != 0:
                print(f"Error logs: {scan_result['logs']}")


if __name__ == "__main__":
    asyncio.run(example_security_scan())