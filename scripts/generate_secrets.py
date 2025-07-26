#!/usr/bin/env python3
"""
Security Hardening: Generate secure secrets for production deployment
"""

import os
import secrets
import string
from cryptography.fernet import Fernet
import base64

def generate_strong_password(length=32):
    """Generate cryptographically secure password"""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def generate_jwt_secret():
    """Generate JWT secret key"""
    return base64.urlsafe_b64encode(os.urandom(64)).decode('utf-8')

def generate_encryption_key():
    """Generate Fernet encryption key"""
    return Fernet.generate_key().decode('utf-8')

def create_production_secrets():
    """Create production-ready secrets"""
    
    secrets_config = {
        # Database credentials
        "POSTGRES_PASSWORD": generate_strong_password(),
        "POSTGRES_USER": "xorb_prod",
        "POSTGRES_DB": "xorb_production",
        
        # Redis auth
        "REDIS_PASSWORD": generate_strong_password(),
        
        # JWT secrets
        "JWT_SECRET_KEY": generate_jwt_secret(),
        "JWT_REFRESH_SECRET": generate_jwt_secret(),
        
        # Encryption keys
        "DATA_ENCRYPTION_KEY": generate_encryption_key(),
        "CONFIG_ENCRYPTION_KEY": generate_encryption_key(),
        
        # API keys (placeholders - replace with real values)
        "NVIDIA_API_KEY": "REPLACE_WITH_REAL_NVIDIA_KEY",
        "OPENROUTER_API_KEY": "REPLACE_WITH_REAL_OPENROUTER_KEY",
        
        # Neo4j credentials
        "NEO4J_PASSWORD": generate_strong_password(),
        "NEO4J_USER": "xorb_prod",
        
        # Temporal secrets
        "TEMPORAL_TLS_CERT": "REPLACE_WITH_TLS_CERT",
        "TEMPORAL_TLS_KEY": "REPLACE_WITH_TLS_KEY",
    }
    
    return secrets_config

def write_kubernetes_secrets(secrets_config):
    """Write Kubernetes secret manifests"""
    
    k8s_secrets = f"""---
apiVersion: v1
kind: Secret
metadata:
  name: xorb-database-secrets
  namespace: xorb-system
type: Opaque
stringData:
  POSTGRES_PASSWORD: "{secrets_config['POSTGRES_PASSWORD']}"
  POSTGRES_USER: "{secrets_config['POSTGRES_USER']}"
  POSTGRES_DB: "{secrets_config['POSTGRES_DB']}"
  REDIS_PASSWORD: "{secrets_config['REDIS_PASSWORD']}"
  NEO4J_PASSWORD: "{secrets_config['NEO4J_PASSWORD']}"
  NEO4J_USER: "{secrets_config['NEO4J_USER']}"

---
apiVersion: v1
kind: Secret
metadata:
  name: xorb-api-secrets
  namespace: xorb-system
type: Opaque
stringData:
  JWT_SECRET_KEY: "{secrets_config['JWT_SECRET_KEY']}"
  JWT_REFRESH_SECRET: "{secrets_config['JWT_REFRESH_SECRET']}"
  DATA_ENCRYPTION_KEY: "{secrets_config['DATA_ENCRYPTION_KEY']}"
  CONFIG_ENCRYPTION_KEY: "{secrets_config['CONFIG_ENCRYPTION_KEY']}"

---
apiVersion: v1
kind: Secret
metadata:
  name: xorb-integration-secrets
  namespace: xorb-system
type: Opaque
stringData:
  NVIDIA_API_KEY: "{secrets_config['NVIDIA_API_KEY']}"
  OPENROUTER_API_KEY: "{secrets_config['OPENROUTER_API_KEY']}"
"""
    
    with open("/root/Xorb/kubernetes/secrets/production-secrets.yaml", "w") as f:
        f.write(k8s_secrets)
    
    print("✅ Kubernetes secrets written to kubernetes/secrets/production-secrets.yaml")

def write_docker_env(secrets_config):
    """Write Docker environment file"""
    
    env_content = "\n".join([f"{key}={value}" for key, value in secrets_config.items()])
    
    with open("/root/Xorb/.env.production.secure", "w") as f:
        f.write(env_content)
    
    os.chmod("/root/Xorb/.env.production.secure", 0o600)  # Restrict permissions
    print("✅ Docker environment written to .env.production.secure")

def main():
    """Generate and write all production secrets"""
    print("🔒 Generating production-grade secrets...")
    
    # Create directories
    os.makedirs("/root/Xorb/kubernetes/secrets", exist_ok=True)
    
    # Generate secrets
    secrets_config = create_production_secrets()
    
    # Write to different formats
    write_kubernetes_secrets(secrets_config)
    write_docker_env(secrets_config)
    
    print("\n🎯 Security Hardening Summary:")
    print("   ✅ Strong passwords generated (32+ character complexity)")
    print("   ✅ JWT secrets with 64-byte entropy")
    print("   ✅ Fernet encryption keys for data protection")
    print("   ✅ Separate credentials for each service")
    print("   ✅ Production Kubernetes secrets manifest")
    print("   ✅ Secure Docker environment file")
    
    print("\n⚠️  Next Steps:")
    print("   1. Replace placeholder API keys with real values")
    print("   2. Generate TLS certificates for Temporal")
    print("   3. Apply secrets to Kubernetes cluster")
    print("   4. Update application configuration")
    print("   5. Test authentication flows")

if __name__ == "__main__":
    main()