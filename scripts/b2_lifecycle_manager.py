#!/usr/bin/env python3
"""
Backblaze B2 Lifecycle Policy Manager
Manages B2 bucket lifecycle policies for automated cost optimization
"""

import asyncio
import json
import logging
import os
from datetime import datetime

import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class B2LifecycleManager:
    """Manages B2 bucket lifecycle policies"""

    def __init__(self):
        self.account_id = os.getenv("B2_ACCOUNT_ID")
        self.application_key = os.getenv("B2_APPLICATION_KEY")
        self.bucket_name = os.getenv("B2_BUCKET_NAME", "xorb-backup")

        # B2 API endpoints
        self.api_url = "https://api.backblazeb2.com"
        self.auth_token = None
        self.api_url_authorized = None

        # Lifecycle policies
        self.lifecycle_rules = self.define_lifecycle_rules()

    def define_lifecycle_rules(self) -> list[dict]:
        """Define lifecycle rules for different backup types"""

        return [
            {
                "fileNamePrefix": "primary/",
                "daysFromHidingToDeleting": 30,
                "daysFromUploadingToHiding": 90
            },
            {
                "fileNamePrefix": "database/",
                "daysFromHidingToDeleting": 7,
                "daysFromUploadingToHiding": 30
            },
            {
                "fileNamePrefix": "files/",
                "daysFromHidingToDeleting": 14,
                "daysFromUploadingToHiding": 60
            },
            {
                "fileNamePrefix": "config/",
                "daysFromHidingToDeleting": 90,
                "daysFromUploadingToHiding": 365
            },
            {
                "fileNamePrefix": "archive/",
                "daysFromHidingToDeleting": 365,
                "daysFromUploadingToHiding": 1095  # 3 years
            }
        ]

    async def authenticate(self) -> bool:
        """Authenticate with B2 API"""
        try:
            auth_url = f"{self.api_url}/b2api/v2/b2_authorize_account"

            # Prepare basic auth
            import base64
            credentials = base64.b64encode(f"{self.account_id}:{self.application_key}".encode()).decode()

            headers = {
                "Authorization": f"Basic {credentials}"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(auth_url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.auth_token = data["authorizationToken"]
                        self.api_url_authorized = data["apiUrl"]
                        logger.info("B2 authentication successful")
                        return True
                    else:
                        logger.error(f"B2 authentication failed: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"B2 authentication error: {e}")
            return False

    async def get_bucket_id(self) -> str | None:
        """Get bucket ID by name"""
        try:
            url = f"{self.api_url_authorized}/b2api/v2/b2_list_buckets"
            headers = {
                "Authorization": self.auth_token
            }

            payload = {
                "accountId": self.account_id
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()

                        for bucket in data["buckets"]:
                            if bucket["bucketName"] == self.bucket_name:
                                return bucket["bucketId"]

                        logger.error(f"Bucket {self.bucket_name} not found")
                        return None
                    else:
                        logger.error(f"Failed to list buckets: {response.status}")
                        return None

        except Exception as e:
            logger.error(f"Error getting bucket ID: {e}")
            return None

    async def update_lifecycle_rules(self, bucket_id: str) -> bool:
        """Update bucket lifecycle rules"""
        try:
            url = f"{self.api_url_authorized}/b2api/v2/b2_update_bucket"
            headers = {
                "Authorization": self.auth_token
            }

            payload = {
                "accountId": self.account_id,
                "bucketId": bucket_id,
                "lifecycleRules": self.lifecycle_rules
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info("Lifecycle rules updated successfully")
                        return True
                    else:
                        error_data = await response.json()
                        logger.error(f"Failed to update lifecycle rules: {error_data}")
                        return False

        except Exception as e:
            logger.error(f"Error updating lifecycle rules: {e}")
            return False

    async def get_current_lifecycle_rules(self, bucket_id: str) -> list[dict] | None:
        """Get current lifecycle rules for bucket"""
        try:
            url = f"{self.api_url_authorized}/b2api/v2/b2_list_buckets"
            headers = {
                "Authorization": self.auth_token
            }

            payload = {
                "accountId": self.account_id,
                "bucketId": bucket_id
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()

                        for bucket in data["buckets"]:
                            if bucket["bucketId"] == bucket_id:
                                return bucket.get("lifecycleRules", [])

                        return []
                    else:
                        logger.error(f"Failed to get lifecycle rules: {response.status}")
                        return None

        except Exception as e:
            logger.error(f"Error getting lifecycle rules: {e}")
            return None

    async def calculate_storage_costs(self, bucket_id: str) -> dict:
        """Calculate storage costs with current lifecycle policies"""
        try:
            url = f"{self.api_url_authorized}/b2api/v2/b2_list_file_names"
            headers = {
                "Authorization": self.auth_token
            }

            payload = {
                "bucketId": bucket_id,
                "maxFileCount": 10000
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()

                        total_size = 0
                        file_count = 0
                        prefix_stats = {}

                        for file_info in data["files"]:
                            size = file_info["contentLength"]
                            name = file_info["fileName"]

                            total_size += size
                            file_count += 1

                            # Categorize by prefix
                            prefix = name.split("/")[0] if "/" in name else "other"
                            if prefix not in prefix_stats:
                                prefix_stats[prefix] = {"size": 0, "files": 0}

                            prefix_stats[prefix]["size"] += size
                            prefix_stats[prefix]["files"] += 1

                        # Calculate costs (B2 pricing: $0.005/GB/month)
                        monthly_cost = (total_size / (1024**3)) * 0.005

                        return {
                            "total_size_bytes": total_size,
                            "total_size_gb": total_size / (1024**3),
                            "file_count": file_count,
                            "estimated_monthly_cost": monthly_cost,
                            "prefix_breakdown": prefix_stats
                        }
                    else:
                        logger.error(f"Failed to list files: {response.status}")
                        return {}

        except Exception as e:
            logger.error(f"Error calculating storage costs: {e}")
            return {}

    async def optimize_lifecycle_policies(self) -> dict:
        """Analyze and optimize lifecycle policies"""

        if not await self.authenticate():
            return {"error": "Authentication failed"}

        bucket_id = await self.get_bucket_id()
        if not bucket_id:
            return {"error": "Bucket not found"}

        # Get current rules
        current_rules = await self.get_current_lifecycle_rules(bucket_id)

        # Get storage statistics
        storage_stats = await self.calculate_storage_costs(bucket_id)

        # Update lifecycle rules
        success = await self.update_lifecycle_rules(bucket_id)

        return {
            "bucket_id": bucket_id,
            "bucket_name": self.bucket_name,
            "current_rules": current_rules,
            "new_rules": self.lifecycle_rules,
            "storage_statistics": storage_stats,
            "update_success": success,
            "optimization_summary": self.generate_optimization_summary(storage_stats)
        }

    def generate_optimization_summary(self, storage_stats: dict) -> dict:
        """Generate optimization summary and recommendations"""

        if not storage_stats:
            return {}

        total_gb = storage_stats.get("total_size_gb", 0)
        monthly_cost = storage_stats.get("estimated_monthly_cost", 0)

        # Estimate savings from lifecycle policies
        # Assuming 30% of data gets moved to cheaper tiers
        estimated_savings = monthly_cost * 0.3

        recommendations = []

        if total_gb > 100:
            recommendations.append("Consider implementing more aggressive archiving for old backups")

        if monthly_cost > 50:
            recommendations.append("High storage costs detected - review retention policies")

        prefix_breakdown = storage_stats.get("prefix_breakdown", {})
        largest_prefix = max(prefix_breakdown.items(), key=lambda x: x[1]["size"]) if prefix_breakdown else None

        if largest_prefix:
            recommendations.append(f"'{largest_prefix[0]}' prefix uses most storage - optimize retention")

        return {
            "current_monthly_cost": monthly_cost,
            "estimated_monthly_savings": estimated_savings,
            "annual_savings_projection": estimated_savings * 12,
            "recommendations": recommendations,
            "largest_storage_category": largest_prefix[0] if largest_prefix else None
        }

    async def setup_bucket_encryption(self, bucket_id: str) -> bool:
        """Setup server-side encryption for backup bucket"""
        try:
            url = f"{self.api_url_authorized}/b2api/v2/b2_update_bucket"
            headers = {
                "Authorization": self.auth_token
            }

            payload = {
                "accountId": self.account_id,
                "bucketId": bucket_id,
                "defaultServerSideEncryption": {
                    "mode": "SSE-B2",
                    "algorithm": "AES256"
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        logger.info("Bucket encryption enabled successfully")
                        return True
                    else:
                        logger.error(f"Failed to enable encryption: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Error setting up encryption: {e}")
            return False

    async def generate_cost_report(self) -> dict:
        """Generate comprehensive cost analysis report"""

        if not await self.authenticate():
            return {"error": "Authentication failed"}

        bucket_id = await self.get_bucket_id()
        if not bucket_id:
            return {"error": "Bucket not found"}

        storage_stats = await self.calculate_storage_costs(bucket_id)
        current_rules = await self.get_current_lifecycle_rules(bucket_id)

        report = {
            "report_generated": datetime.now().isoformat(),
            "bucket_info": {
                "name": self.bucket_name,
                "id": bucket_id
            },
            "storage_analysis": storage_stats,
            "current_lifecycle_rules": current_rules,
            "recommended_rules": self.lifecycle_rules,
            "cost_optimization": self.generate_optimization_summary(storage_stats)
        }

        return report

async def main():
    """Main function for B2 lifecycle management"""
    import argparse

    parser = argparse.ArgumentParser(description="B2 Lifecycle Policy Manager")
    parser.add_argument("--optimize", action="store_true", help="Optimize lifecycle policies")
    parser.add_argument("--report", action="store_true", help="Generate cost report")
    parser.add_argument("--setup-encryption", action="store_true", help="Setup bucket encryption")

    args = parser.parse_args()

    manager = B2LifecycleManager()

    if args.optimize:
        result = await manager.optimize_lifecycle_policies()
        print(json.dumps(result, indent=2))

    elif args.report:
        report = await manager.generate_cost_report()
        print(json.dumps(report, indent=2))

    elif args.setup_encryption:
        if await manager.authenticate():
            bucket_id = await manager.get_bucket_id()
            if bucket_id:
                success = await manager.setup_bucket_encryption(bucket_id)
                print(f"Encryption setup: {'Success' if success else 'Failed'}")

    else:
        print("Use --optimize, --report, or --setup-encryption")

if __name__ == "__main__":
    asyncio.run(main())
