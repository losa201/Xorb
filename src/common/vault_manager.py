#!/usr/bin/env python3
"""
XORB Vault Management CLI

This script provides management capabilities for the Vault integration:
1. Health checks and connection testing
2. Secret management and rotation
3. Backup and restore operations
4. Integration with existing Vault infrastructure

Usage:
    python vault_manager.py health
    python vault_manager.py list-secrets
    python vault_manager.py get-secret xorb/config
    python vault_manager.py rotate-jwt-key
    python vault_manager.py backup secrets-backup.json
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from vault_client import VaultClient, get_vault_client


class VaultManager:
    """CLI interface for Vault management operations"""

    def __init__(self):
        self.vault_client = None

    async def initialize(self):
        """Initialize Vault client"""
        try:
            self.vault_client = await get_vault_client()
            print(f"🔐 Connected to Vault at {self.vault_client.vault_url}")
        except Exception as e:
            print(f"❌ Failed to connect to Vault: {e}")
            sys.exit(1)

    async def health_check(self):
        """Check Vault health and display status"""
        health = await self.vault_client.health_check()

        print("🏥 Vault Health Check:")
        print(f"   Status: {health['status']}")
        print(f"   URL: {health['vault_url']}")
        print(f"   Authenticated: {health['authenticated']}")
        print(f"   Fallback Mode: {health['fallback_mode']}")

        if "message" in health:
            print(f"   Message: {health['message']}")
        if "error" in health:
            print(f"   Error: {health['error']}")

    async def list_secrets(self, path: str = ""):
        """List available secrets"""
        try:
            secrets = await self.vault_client.list_secrets(path)
            print(f"📋 Secrets at '{path or 'root'}':")

            if "keys" in secrets:
                for key in secrets["keys"]:
                    print(f"   - {key}")
            else:
                print("   No secrets found")

        except Exception as e:
            print(f"❌ Failed to list secrets: {e}")

    async def get_secret(self, path: str, version: int = None):
        """Get and display a secret"""
        try:
            if version:
                secret_data = await self.vault_client.get_secret_version(path, version)
                print(f"🔑 Secret '{path}' (version {version}):")
            else:
                secret_data = await self.vault_client.get_secret(path)
                print(f"🔑 Secret '{path}':")

            # Mask sensitive values
            masked_data = {}
            for key, value in secret_data.items():
                if any(sensitive in key.lower() for sensitive in ['secret', 'key', 'password', 'token']):
                    masked_data[key] = "*" * min(len(str(value)), 8) if value else ""
                else:
                    masked_data[key] = value

            print(json.dumps(masked_data, indent=2))

        except Exception as e:
            print(f"❌ Failed to get secret: {e}")

    async def rotate_jwt_key(self):
        """Rotate JWT signing key"""
        print("🔄 Rotating JWT signing key...")
        success = await self.vault_client.rotate_jwt_key()
        if success:
            print("✅ JWT key rotation completed")
        else:
            print("❌ JWT key rotation failed")

    async def backup_secrets(self, output_file: str):
        """Backup secrets to file"""
        print(f"💾 Backing up secrets to {output_file}...")
        success = await self.vault_client.backup_secrets(output_file)
        if success:
            print("✅ Secrets backup completed")
        else:
            print("❌ Secrets backup failed")

    async def test_integration(self):
        """Test integration with existing Vault infrastructure"""
        print("🧪 Testing Vault integration...")

        # Test health
        print("\n1. Health Check:")
        await self.health_check()

        # Test secret retrieval
        print("\n2. Configuration Secrets:")
        await self.get_secret("xorb/config")

        print("\n3. External API Secrets:")
        await self.get_secret("xorb/external")

        # Test convenience functions
        print("\n4. Convenience Functions Test:")
        try:
            from vault_client import get_jwt_secret, get_xorb_api_key

            jwt_secret = await get_jwt_secret()
            api_key = await get_xorb_api_key()

            print(f"   JWT Secret: {'*' * 8}")
            print(f"   API Key: {'*' * 8}")
            print("   ✅ Convenience functions working")

        except Exception as e:
            print(f"   ❌ Convenience function error: {e}")

    async def setup_development_secrets(self):
        """Setup development secrets (requires write access)"""
        print("🔧 Setting up development secrets...")

        try:
            # This would require write permissions to Vault
            print("⚠️  Development secret setup requires Vault write permissions")
            print("   Run the infra/vault/setup-vault-dev.sh script instead")

        except Exception as e:
            print(f"❌ Failed to setup development secrets: {e}")


async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="XORB Vault Management")
    parser.add_argument("command", choices=[
        "health", "list-secrets", "get-secret", "rotate-jwt-key",
        "backup", "test", "setup-dev"
    ], help="Command to execute")

    parser.add_argument("--path", help="Secret path (for get-secret, list-secrets)")
    parser.add_argument("--version", type=int, help="Secret version (for get-secret)")
    parser.add_argument("--output", help="Output file (for backup)")

    args = parser.parse_args()

    manager = VaultManager()
    await manager.initialize()

    try:
        if args.command == "health":
            await manager.health_check()

        elif args.command == "list-secrets":
            await manager.list_secrets(args.path or "")

        elif args.command == "get-secret":
            if not args.path:
                print("❌ --path required for get-secret")
                sys.exit(1)
            await manager.get_secret(args.path, args.version)

        elif args.command == "rotate-jwt-key":
            await manager.rotate_jwt_key()

        elif args.command == "backup":
            if not args.output:
                print("❌ --output required for backup")
                sys.exit(1)
            await manager.backup_secrets(args.output)

        elif args.command == "test":
            await manager.test_integration()

        elif args.command == "setup-dev":
            await manager.setup_development_secrets()

    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
