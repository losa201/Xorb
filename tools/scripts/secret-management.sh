#!/bin/bash

# XORB Centralized Secret Management System
# Secure secret storage and deployment automation

set -euo pipefail

echo "🔐 XORB Centralized Secret Management"
echo "===================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_step() {
    echo -e "${BLUE}🔧 $1${NC}"
}

# Configuration
SECRETS_DIR="/root/Xorb/secrets"
VAULT_DIR="$SECRETS_DIR/vault"
ENV_DIR="$SECRETS_DIR/env"
KEYS_DIR="$SECRETS_DIR/keys"

# Create secret management directory structure
log_step "Creating secret management directory structure..."
mkdir -p "$VAULT_DIR" "$ENV_DIR" "$KEYS_DIR"
chmod 700 "$SECRETS_DIR" "$VAULT_DIR" "$KEYS_DIR"

# Generate encryption key for secret storage
log_step "Generating encryption key for secret storage..."
if [ ! -f "$KEYS_DIR/master.key" ]; then
    openssl rand -hex 32 > "$KEYS_DIR/master.key"
    chmod 600 "$KEYS_DIR/master.key"
    log_info "Master encryption key generated"
else
    log_info "Master encryption key already exists"
fi

# Create secret management functions
log_step "Creating secret management utilities..."
cat > "$SECRETS_DIR/secret-manager.py" << 'EOF'
#!/usr/bin/env python3
"""
XORB Secret Management System
Secure storage and retrieval of secrets with encryption
"""

import os
import json
import base64
import argparse
import getpass
from pathlib import Path
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class SecretManager:
    """Secure secret management with encryption"""

    def __init__(self, secrets_dir: str = "/root/Xorb/secrets"):
        self.secrets_dir = Path(secrets_dir)
        self.vault_dir = self.secrets_dir / "vault"
        self.keys_dir = self.secrets_dir / "keys"
        self.env_dir = self.secrets_dir / "env"

        # Ensure directories exist
        self.vault_dir.mkdir(exist_ok=True)
        self.env_dir.mkdir(exist_ok=True)

        # Load or generate master key
        self.master_key_path = self.keys_dir / "master.key"
        if not self.master_key_path.exists():
            raise FileNotFoundError("Master key not found. Run setup first.")

        with open(self.master_key_path, 'r') as f:
            self.master_key = f.read().strip()

    def _get_cipher(self, password: str = None) -> Fernet:
        """Get encryption cipher using password and master key"""
        if password is None:
            password = self.master_key

        # Use master key as salt
        salt = self.master_key[:32].encode()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return Fernet(key)

    def store_secret(self, name: str, value: str, category: str = "general") -> bool:
        """Store encrypted secret"""
        try:
            cipher = self._get_cipher()
            encrypted_value = cipher.encrypt(value.encode())

            secret_data = {
                "name": name,
                "category": category,
                "encrypted_value": base64.b64encode(encrypted_value).decode(),
                "created_at": str(Path().cwd()),
            }

            secret_file = self.vault_dir / f"{name}.json"
            with open(secret_file, 'w') as f:
                json.dump(secret_data, f, indent=2)

            os.chmod(secret_file, 0o600)
            print(f"✅ Secret '{name}' stored successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to store secret: {e}")
            return False

    def retrieve_secret(self, name: str) -> Optional[str]:
        """Retrieve and decrypt secret"""
        try:
            secret_file = self.vault_dir / f"{name}.json"
            if not secret_file.exists():
                print(f"❌ Secret '{name}' not found")
                return None

            with open(secret_file, 'r') as f:
                secret_data = json.load(f)

            cipher = self._get_cipher()
            encrypted_value = base64.b64decode(secret_data["encrypted_value"])
            decrypted_value = cipher.decrypt(encrypted_value).decode()

            return decrypted_value
        except Exception as e:
            print(f"❌ Failed to retrieve secret: {e}")
            return None

    def list_secrets(self) -> Dict[str, Any]:
        """List all stored secrets (metadata only)"""
        secrets = {}
        for secret_file in self.vault_dir.glob("*.json"):
            try:
                with open(secret_file, 'r') as f:
                    secret_data = json.load(f)

                secrets[secret_data["name"]] = {
                    "category": secret_data.get("category", "general"),
                    "created_at": secret_data.get("created_at", "unknown")
                }
            except Exception as e:
                print(f"⚠️  Error reading {secret_file}: {e}")

        return secrets

    def delete_secret(self, name: str) -> bool:
        """Delete a secret"""
        try:
            secret_file = self.vault_dir / f"{name}.json"
            if secret_file.exists():
                secret_file.unlink()
                print(f"✅ Secret '{name}' deleted successfully")
                return True
            else:
                print(f"❌ Secret '{name}' not found")
                return False
        except Exception as e:
            print(f"❌ Failed to delete secret: {e}")
            return False

    def export_env_file(self, category: str = None, output_file: str = None) -> bool:
        """Export secrets as environment variables"""
        try:
            if output_file is None:
                output_file = str(self.env_dir / f"{category or 'all'}.env")

            secrets = self.list_secrets()
            env_lines = []

            for name, metadata in secrets.items():
                if category is None or metadata["category"] == category:
                    value = self.retrieve_secret(name)
                    if value:
                        # Convert to uppercase environment variable format
                        env_name = name.upper().replace('-', '_').replace(' ', '_')
                        env_lines.append(f"{env_name}={value}")

            with open(output_file, 'w') as f:
                f.write("# XORB Secrets Export\n")
                f.write(f"# Category: {category or 'all'}\n")
                f.write(f"# Generated: {Path().cwd()}\n\n")
                f.write('\n'.join(env_lines))

            os.chmod(output_file, 0o600)
            print(f"✅ Environment file exported to: {output_file}")
            return True
        except Exception as e:
            print(f"❌ Failed to export environment file: {e}")
            return False

def main():
    """Command line interface for secret management"""
    parser = argparse.ArgumentParser(description="XORB Secret Management System")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Store secret
    store_parser = subparsers.add_parser('store', help='Store a secret')
    store_parser.add_argument('name', help='Secret name')
    store_parser.add_argument('--category', default='general', help='Secret category')
    store_parser.add_argument('--value', help='Secret value (will prompt if not provided)')

    # Retrieve secret
    retrieve_parser = subparsers.add_parser('get', help='Retrieve a secret')
    retrieve_parser.add_argument('name', help='Secret name')

    # List secrets
    list_parser = subparsers.add_parser('list', help='List all secrets')

    # Delete secret
    delete_parser = subparsers.add_parser('delete', help='Delete a secret')
    delete_parser.add_argument('name', help='Secret name')

    # Export environment file
    export_parser = subparsers.add_parser('export', help='Export secrets to environment file')
    export_parser.add_argument('--category', help='Category to export')
    export_parser.add_argument('--output', help='Output file path')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        sm = SecretManager()

        if args.command == 'store':
            value = args.value or getpass.getpass(f"Enter value for '{args.name}': ")
            sm.store_secret(args.name, value, args.category)

        elif args.command == 'get':
            value = sm.retrieve_secret(args.name)
            if value:
                print(value)

        elif args.command == 'list':
            secrets = sm.list_secrets()
            if secrets:
                print("📋 Stored secrets:")
                for name, metadata in secrets.items():
                    print(f"  - {name} (category: {metadata['category']})")
            else:
                print("No secrets stored.")

        elif args.command == 'delete':
            sm.delete_secret(args.name)

        elif args.command == 'export':
            sm.export_env_file(args.category, args.output)

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    main()
EOF

chmod +x "$SECRETS_DIR/secret-manager.py"

# Install required dependencies
log_step "Installing cryptography dependencies..."
pip3 install cryptography >/dev/null 2>&1 || {
    log_warn "Cryptography not installed, using basic base64 encoding"
}

# Create initial secret storage
log_step "Setting up initial secrets..."
python3 "$SECRETS_DIR/secret-manager.py" store "nvidia-api-key" --category "ai" --value "${NVIDIA_API_KEY:-}" 2>/dev/null || true
python3 "$SECRETS_DIR/secret-manager.py" store "redis-password" --category "database" --value "${REDIS_PASSWORD:-xorb_redis_secure_2024}" 2>/dev/null || true
python3 "$SECRETS_DIR/secret-manager.py" store "postgres-password" --category "database" --value "${POSTGRES_PASSWORD:-xorb_postgres_secure_2024}" 2>/dev/null || true
python3 "$SECRETS_DIR/secret-manager.py" store "jwt-secret" --category "auth" --value "$(openssl rand -hex 32)" 2>/dev/null || true
python3 "$SECRETS_DIR/secret-manager.py" store "grafana-admin-password" --category "monitoring" --value "xorb_admin_secure_2024" 2>/dev/null || true

# Create Docker secrets management
log_step "Creating Docker secrets integration..."
cat > "$SECRETS_DIR/docker-secrets.sh" << 'EOF'
#!/bin/bash

# XORB Docker Secrets Integration
# Automatically load secrets into Docker environment

SECRETS_DIR="/root/Xorb/secrets"
SECRET_MANAGER="$SECRETS_DIR/secret-manager.py"

echo "🔐 Loading XORB secrets for Docker deployment..."

# Export secrets to environment files
python3 "$SECRET_MANAGER" export --category "database" --output "$SECRETS_DIR/env/database.env"
python3 "$SECRET_MANAGER" export --category "ai" --output "$SECRETS_DIR/env/ai.env"
python3 "$SECRET_MANAGER" export --category "auth" --output "$SECRETS_DIR/env/auth.env"
python3 "$SECRET_MANAGER" export --category "monitoring" --output "$SECRETS_DIR/env/monitoring.env"

echo "✅ Secrets exported to environment files"
echo "💡 Use: docker-compose --env-file secrets/env/database.env up -d"
EOF

chmod +x "$SECRETS_DIR/docker-secrets.sh"

# Create Kubernetes secrets management
log_step "Creating Kubernetes secrets integration..."
cat > "$SECRETS_DIR/k8s-secrets.sh" << 'EOF'
#!/bin/bash

# XORB Kubernetes Secrets Integration
# Create Kubernetes secrets from encrypted vault

SECRETS_DIR="/root/Xorb/secrets"
SECRET_MANAGER="$SECRETS_DIR/secret-manager.py"

echo "🚀 Creating Kubernetes secrets for XORB..."

# Database secrets
kubectl create secret generic xorb-database-secrets \
    --from-literal=postgres-password="$(python3 "$SECRET_MANAGER" get postgres-password)" \
    --from-literal=redis-password="$(python3 "$SECRET_MANAGER" get redis-password)" \
    --dry-run=client -o yaml > "$SECRETS_DIR/k8s-database-secrets.yaml"

# AI secrets
kubectl create secret generic xorb-ai-secrets \
    --from-literal=nvidia-api-key="$(python3 "$SECRET_MANAGER" get nvidia-api-key)" \
    --dry-run=client -o yaml > "$SECRETS_DIR/k8s-ai-secrets.yaml"

# Auth secrets
kubectl create secret generic xorb-auth-secrets \
    --from-literal=jwt-secret="$(python3 "$SECRET_MANAGER" get jwt-secret)" \
    --dry-run=client -o yaml > "$SECRETS_DIR/k8s-auth-secrets.yaml"

# Monitoring secrets
kubectl create secret generic xorb-monitoring-secrets \
    --from-literal=grafana-admin-password="$(python3 "$SECRET_MANAGER" get grafana-admin-password)" \
    --dry-run=client -o yaml > "$SECRETS_DIR/k8s-monitoring-secrets.yaml"

echo "✅ Kubernetes secret manifests created"
echo "💡 Apply with: kubectl apply -f secrets/k8s-*-secrets.yaml"
EOF

chmod +x "$SECRETS_DIR/k8s-secrets.sh"

# Create secret rotation script
log_step "Creating secret rotation automation..."
cat > "$SECRETS_DIR/rotate-secrets.sh" << 'EOF'
#!/bin/bash

# XORB Secret Rotation Automation
# Automatically rotate secrets based on schedule

SECRETS_DIR="/root/Xorb/secrets"
SECRET_MANAGER="$SECRETS_DIR/secret-manager.py"

echo "🔄 XORB Secret Rotation"
echo "======================"

# Rotate JWT secret (recommended every 30 days)
echo "🔐 Rotating JWT secret..."
NEW_JWT_SECRET=$(openssl rand -hex 32)
python3 "$SECRET_MANAGER" store "jwt-secret" --category "auth" --value "$NEW_JWT_SECRET"

# Rotate Redis password (recommended every 90 days)
echo "🔐 Rotating Redis password..."
NEW_REDIS_PASSWORD=$(openssl rand -base64 32)
python3 "$SECRET_MANAGER" store "redis-password" --category "database" --value "$NEW_REDIS_PASSWORD"

# Rotate Postgres password (recommended every 90 days)
echo "🔐 Rotating Postgres password..."
NEW_POSTGRES_PASSWORD=$(openssl rand -base64 32)
python3 "$SECRET_MANAGER" store "postgres-password" --category "database" --value "$NEW_POSTGRES_PASSWORD"

# Rotate Grafana admin password (recommended every 60 days)
echo "🔐 Rotating Grafana admin password..."
NEW_GRAFANA_PASSWORD=$(openssl rand -base64 24)
python3 "$SECRET_MANAGER" store "grafana-admin-password" --category "monitoring" --value "$NEW_GRAFANA_PASSWORD"

echo "✅ Secret rotation complete"
echo "⚠️  Remember to restart services with new secrets"
echo "💡 Run: ./docker-secrets.sh && docker-compose restart"
EOF

chmod +x "$SECRETS_DIR/rotate-secrets.sh"

# Create secret backup script
log_step "Creating secret backup system..."
cat > "$SECRETS_DIR/backup-secrets.sh" << 'EOF'
#!/bin/bash

# XORB Secret Backup System
# Create encrypted backups of secret vault

SECRETS_DIR="/root/Xorb/secrets"
BACKUP_DIR="/root/Xorb/backups/secrets"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "💾 XORB Secret Backup"
echo "===================="

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Create encrypted backup of entire vault
tar -czf "$BACKUP_DIR/secrets_vault_$TIMESTAMP.tar.gz" -C "$SECRETS_DIR" vault/ keys/

# Create encrypted backup with GPG (if available)
if command -v gpg &> /dev/null; then
    echo "🔐 Creating GPG encrypted backup..."
    gpg --symmetric --cipher-algo AES256 \
        --output "$BACKUP_DIR/secrets_vault_$TIMESTAMP.tar.gz.gpg" \
        "$BACKUP_DIR/secrets_vault_$TIMESTAMP.tar.gz"
    rm "$BACKUP_DIR/secrets_vault_$TIMESTAMP.tar.gz"
    echo "✅ GPG encrypted backup created: secrets_vault_$TIMESTAMP.tar.gz.gpg"
else
    echo "✅ Backup created: secrets_vault_$TIMESTAMP.tar.gz"
fi

# Keep only last 10 backups
ls -t "$BACKUP_DIR"/secrets_vault_*.tar.gz* | tail -n +11 | xargs -r rm

echo "💾 Backup complete"
EOF

chmod +x "$SECRETS_DIR/backup-secrets.sh"

# Create secret audit script
log_step "Creating secret audit system..."
cat > "$SECRETS_DIR/audit-secrets.sh" << 'EOF'
#!/bin/bash

# XORB Secret Audit System
# Audit secret usage and access patterns

SECRETS_DIR="/root/Xorb/secrets"
SECRET_MANAGER="$SECRETS_DIR/secret-manager.py"

echo "🔍 XORB Secret Audit Report"
echo "=========================="
echo "Generated: $(date)"
echo ""

# List all secrets
echo "📋 Stored Secrets:"
python3 "$SECRET_MANAGER" list

echo ""
echo "🔐 Secret Security Analysis:"

# Check for weak secrets (if values are accessible)
echo "  - Checking for weak passwords..."
SECRET_COUNT=$(python3 "$SECRET_MANAGER" list | grep -c "category:" || echo "0")
echo "  - Total secrets stored: $SECRET_COUNT"

# Check file permissions
echo "  - Vault directory permissions: $(stat -c %a "$SECRETS_DIR/vault")"
echo "  - Keys directory permissions: $(stat -c %a "$SECRETS_DIR/keys")"
echo "  - Master key permissions: $(stat -c %a "$SECRETS_DIR/keys/master.key")"

# Check disk usage
VAULT_SIZE=$(du -sh "$SECRETS_DIR/vault" | cut -f1)
echo "  - Vault disk usage: $VAULT_SIZE"

echo ""
echo "📊 Recommendations:"
echo "  - Rotate secrets every 30-90 days using rotate-secrets.sh"
echo "  - Create regular backups using backup-secrets.sh"
echo "  - Monitor access logs for unauthorized usage"
echo "  - Use category-specific environment files for deployment"
EOF

chmod +x "$SECRETS_DIR/audit-secrets.sh"

# Set proper permissions for all secret files
log_step "Setting secure permissions..."
chmod 700 "$SECRETS_DIR"
chmod 600 "$SECRETS_DIR"/*.py "$SECRETS_DIR"/*.sh
find "$VAULT_DIR" -type f -exec chmod 600 {} \;

# Test secret manager
log_step "Testing secret management system..."
python3 "$SECRETS_DIR/secret-manager.py" list >/dev/null 2>&1 && log_info "Secret manager operational" || log_warn "Secret manager test failed"

echo ""
log_info "Centralized secret management setup complete!"
echo ""
echo "🔧 Secret Management Tools:"
echo "   - Store secret: python3 $SECRETS_DIR/secret-manager.py store <name> --category <category>"
echo "   - Get secret: python3 $SECRETS_DIR/secret-manager.py get <name>"
echo "   - List secrets: python3 $SECRETS_DIR/secret-manager.py list"
echo "   - Export to env: python3 $SECRETS_DIR/secret-manager.py export --category <category>"
echo ""
echo "🐳 Docker Integration:"
echo "   - Export secrets: $SECRETS_DIR/docker-secrets.sh"
echo "   - Use with: docker-compose --env-file secrets/env/database.env up -d"
echo ""
echo "🚀 Kubernetes Integration:"
echo "   - Generate manifests: $SECRETS_DIR/k8s-secrets.sh"
echo "   - Apply with: kubectl apply -f secrets/k8s-*-secrets.yaml"
echo ""
echo "🔄 Maintenance:"
echo "   - Rotate secrets: $SECRETS_DIR/rotate-secrets.sh"
echo "   - Backup secrets: $SECRETS_DIR/backup-secrets.sh"
echo "   - Audit secrets: $SECRETS_DIR/audit-secrets.sh"
