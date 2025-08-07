from dataclasses import dataclass

#!/usr/bin/env python3
"""
Disaster Recovery - Backup Status Checker
Verifies the status and integrity of all backup systems
"""

import asyncio
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BackupStatusChecker:
    """Comprehensive backup status verification"""

    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "backups": {},
            "issues": [],
            "recommendations": []
        }

    async def check_postgres_wal_backup(self) -> dict:
        """Check PostgreSQL WAL-G backup status"""
        logger.info("Checking PostgreSQL WAL-G backup status...")

        try:
            # Check latest base backup
            result = subprocess.run(
                ["wal-g", "backup-list", "--json"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                backups = json.loads(result.stdout)
                if backups:
                    latest_backup = backups[0]
                    backup_time = datetime.fromisoformat(latest_backup["time"].replace("Z", "+00:00"))
                    age_hours = (datetime.now() - backup_time.replace(tzinfo=None)).total_seconds() / 3600

                    status = {
                        "service": "postgresql_wal",
                        "status": "healthy" if age_hours < 24 else "stale",
                        "latest_backup": latest_backup["backup_name"],
                        "backup_time": backup_time.isoformat(),
                        "age_hours": round(age_hours, 2),
                        "size_compressed": latest_backup.get("compressed_size", "unknown"),
                        "size_uncompressed": latest_backup.get("uncompressed_size", "unknown")
                    }

                    # Check WAL continuity
                    wal_result = subprocess.run(
                        ["wal-g", "wal-verify", "timeline", "1"],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )

                    status["wal_continuity"] = "verified" if wal_result.returncode == 0 else "broken"

                    if age_hours > 24:
                        self.results["issues"].append(f"PostgreSQL backup is {age_hours:.1f} hours old")

                    return status
                else:
                    self.results["issues"].append("No PostgreSQL backups found")
                    return {"service": "postgresql_wal", "status": "no_backups"}
            else:
                self.results["issues"].append(f"WAL-G error: {result.stderr}")
                return {"service": "postgresql_wal", "status": "error", "error": result.stderr}

        except Exception as e:
            logger.error(f"PostgreSQL backup check failed: {e}")
            self.results["issues"].append(f"PostgreSQL backup check failed: {e}")
            return {"service": "postgresql_wal", "status": "error", "error": str(e)}

    async def check_redis_backup(self) -> dict:
        """Check Redis backup status"""
        logger.info("Checking Redis backup status...")

        try:
            # Check if Redis dump exists
            dump_path = Path("/data/redis/dump.rdb")
            if dump_path.exists():
                stat = dump_path.stat()
                age_hours = (time.time() - stat.st_mtime) / 3600

                status = {
                    "service": "redis",
                    "status": "healthy" if age_hours < 24 else "stale",
                    "backup_file": str(dump_path),
                    "backup_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "age_hours": round(age_hours, 2),
                    "size_bytes": stat.st_size
                }

                # Check AOF if enabled
                aof_path = Path("/data/redis/appendonly.aof")
                if aof_path.exists():
                    aof_stat = aof_path.stat()
                    status["aof_backup"] = {
                        "exists": True,
                        "size_bytes": aof_stat.st_size,
                        "modified": datetime.fromtimestamp(aof_stat.st_mtime).isoformat()
                    }

                if age_hours > 24:
                    self.results["issues"].append(f"Redis backup is {age_hours:.1f} hours old")

                return status
            else:
                self.results["issues"].append("Redis backup file not found")
                return {"service": "redis", "status": "no_backup"}

        except Exception as e:
            logger.error(f"Redis backup check failed: {e}")
            self.results["issues"].append(f"Redis backup check failed: {e}")
            return {"service": "redis", "status": "error", "error": str(e)}

    async def check_filesystem_backup(self) -> dict:
        """Check filesystem backup via restic"""
        logger.info("Checking filesystem backup status...")

        try:
            # Check restic snapshots
            result = subprocess.run(
                ["restic", "snapshots", "--json", "--last"],
                capture_output=True,
                text=True,
                timeout=30,
                env={"RESTIC_REPOSITORY": "/backups/restic", "RESTIC_PASSWORD": "xorb_backup_2024"}
            )

            if result.returncode == 0:
                snapshots = json.loads(result.stdout)
                if snapshots:
                    latest_snapshot = snapshots[0]
                    snapshot_time = datetime.fromisoformat(latest_snapshot["time"].replace("Z", "+00:00"))
                    age_hours = (datetime.now() - snapshot_time.replace(tzinfo=None)).total_seconds() / 3600

                    status = {
                        "service": "filesystem",
                        "status": "healthy" if age_hours < 168 else "stale",  # Weekly backup
                        "snapshot_id": latest_snapshot["id"][:8],
                        "snapshot_time": snapshot_time.isoformat(),
                        "age_hours": round(age_hours, 2),
                        "paths": latest_snapshot.get("paths", []),
                        "hostname": latest_snapshot.get("hostname", "unknown")
                    }

                    if age_hours > 168:  # 1 week
                        self.results["issues"].append(f"Filesystem backup is {age_hours/24:.1f} days old")

                    return status
                else:
                    self.results["issues"].append("No filesystem snapshots found")
                    return {"service": "filesystem", "status": "no_snapshots"}
            else:
                self.results["issues"].append(f"Restic error: {result.stderr}")
                return {"service": "filesystem", "status": "error", "error": result.stderr}

        except Exception as e:
            logger.error(f"Filesystem backup check failed: {e}")
            self.results["issues"].append(f"Filesystem backup check failed: {e}")
            return {"service": "filesystem", "status": "error", "error": str(e)}

    async def check_backup_storage_space(self) -> dict:
        """Check backup storage space availability"""
        logger.info("Checking backup storage space...")

        try:
            # Check local backup storage
            result = subprocess.run(
                ["df", "-h", "/backups"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    fields = lines[1].split()
                    total = fields[1]
                    used = fields[2]
                    available = fields[3]
                    use_percent = int(fields[4].rstrip('%'))

                    status = {
                        "service": "backup_storage",
                        "status": "healthy" if use_percent < 80 else "warning" if use_percent < 95 else "critical",
                        "total_space": total,
                        "used_space": used,
                        "available_space": available,
                        "use_percentage": use_percent
                    }

                    if use_percent >= 95:
                        self.results["issues"].append(f"Backup storage critically low: {use_percent}% used")
                    elif use_percent >= 80:
                        self.results["issues"].append(f"Backup storage high usage: {use_percent}% used")

                    return status

            return {"service": "backup_storage", "status": "error", "error": "Unable to check storage"}

        except Exception as e:
            logger.error(f"Storage check failed: {e}")
            return {"service": "backup_storage", "status": "error", "error": str(e)}

    async def verify_backup_integrity(self, service: str = None) -> dict:
        """Verify backup integrity through test restore"""
        logger.info("Verifying backup integrity...")

        integrity_results = {}

        # PostgreSQL integrity check
        if not service or service == "postgresql":
            try:
                # Quick integrity check using pg_verifybackup (if available)
                result = subprocess.run(
                    ["wal-g", "backup-fetch", "/tmp/pg_integrity_test", "LATEST", "--extract"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                if result.returncode == 0:
                    # Check if control file exists and is valid
                    control_file = Path("/tmp/pg_integrity_test/global/pg_control")
                    if control_file.exists():
                        integrity_results["postgresql"] = {
                            "status": "verified",
                            "method": "test_extract",
                            "verified_at": datetime.now().isoformat()
                        }
                    else:
                        integrity_results["postgresql"] = {
                            "status": "failed",
                            "error": "Missing control file in backup"
                        }

                    # Cleanup
                    await asyncio.create_subprocess_exec(["rm", "-rf", "/tmp/pg_integrity_test"], capture_output=True)
                else:
                    integrity_results["postgresql"] = {
                        "status": "failed",
                        "error": result.stderr
                    }

            except Exception as e:
                integrity_results["postgresql"] = {
                    "status": "error",
                    "error": str(e)
                }

        # Redis integrity check
        if not service or service == "redis":
            try:
                dump_path = Path("/data/redis/dump.rdb")
                if dump_path.exists():
                    # Use redis-check-rdb to verify dump integrity
                    result = subprocess.run(
                        ["redis-check-rdb", str(dump_path)],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )

                    if result.returncode == 0:
                        integrity_results["redis"] = {
                            "status": "verified",
                            "method": "redis-check-rdb",
                            "verified_at": datetime.now().isoformat()
                        }
                    else:
                        integrity_results["redis"] = {
                            "status": "failed",
                            "error": result.stderr
                        }
                else:
                    integrity_results["redis"] = {
                        "status": "no_backup",
                        "error": "Dump file not found"
                    }

            except Exception as e:
                integrity_results["redis"] = {
                    "status": "error",
                    "error": str(e)
                }

        return {"service": "integrity_verification", "results": integrity_results}

    async def generate_recommendations(self):
        """Generate recommendations based on backup status"""

        # Check for old backups
        postgres_status = self.results["backups"].get("postgresql_wal", {})
        if postgres_status.get("age_hours", 0) > 24:
            self.results["recommendations"].append("Schedule immediate PostgreSQL backup")

        redis_status = self.results["backups"].get("redis", {})
        if redis_status.get("age_hours", 0) > 24:
            self.results["recommendations"].append("Schedule immediate Redis backup")

        # Check storage space
        storage_status = self.results["backups"].get("backup_storage", {})
        if storage_status.get("use_percentage", 0) > 80:
            self.results["recommendations"].append("Clean up old backups or expand storage")

        # Check backup continuity
        if postgres_status.get("wal_continuity") == "broken":
            self.results["recommendations"].append("URGENT: Fix PostgreSQL WAL continuity")

        # General recommendations
        if not self.results["recommendations"]:
            self.results["recommendations"].append("All backup systems are healthy")

    async def run_comprehensive_check(self, verify_integrity: bool = False) -> dict:
        """Run comprehensive backup status check"""
        logger.info("Starting comprehensive backup status check...")

        # Run all backup checks concurrently
        backup_checks = await asyncio.gather(
            self.check_postgres_wal_backup(),
            self.check_redis_backup(),
            self.check_filesystem_backup(),
            self.check_backup_storage_space(),
            return_exceptions=True
        )

        # Process results
        for check_result in backup_checks:
            if isinstance(check_result, Exception):
                logger.error(f"Backup check failed: {check_result}")
                self.results["issues"].append(f"Check failed: {check_result}")
            else:
                service_name = check_result["service"]
                self.results["backups"][service_name] = check_result

        # Verify integrity if requested
        if verify_integrity:
            integrity_result = await self.verify_backup_integrity()
            self.results["backups"]["integrity"] = integrity_result

        # Generate recommendations
        await self.generate_recommendations()

        # Determine overall status
        critical_issues = [issue for issue in self.results["issues"]
                          if "critical" in issue.lower() or "urgent" in issue.lower()]
        warning_issues = [issue for issue in self.results["issues"]
                         if "warning" in issue.lower() or "stale" in issue.lower()]

        if critical_issues:
            self.results["overall_status"] = "critical"
        elif warning_issues or self.results["issues"]:
            self.results["overall_status"] = "warning"
        else:
            self.results["overall_status"] = "healthy"

        logger.info(f"Backup status check completed. Overall status: {self.results['overall_status']}")
        return self.results

async def main():
    """Main function for script execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Xorb Disaster Recovery - Backup Status Checker")
    parser.add_argument("--verify-integrity", action="store_true",
                       help="Verify backup integrity (slower)")
    parser.add_argument("--service", choices=["postgresql", "redis", "filesystem"],
                       help="Check specific service only")
    parser.add_argument("--json", action="store_true",
                       help="Output results in JSON format")
    parser.add_argument("--alert-threshold", type=int, default=24,
                       help="Alert threshold in hours for backup age")

    args = parser.parse_args()

    checker = BackupStatusChecker()
    results = await checker.run_comprehensive_check(verify_integrity=args.verify_integrity)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        # Human-readable output
        print("\n🔍 Xorb Backup Status Report")
        print(f"📅 Generated: {results['timestamp']}")
        print(f"🎯 Overall Status: {results['overall_status'].upper()}")

        print("\n📊 Backup Services:")
        for service, status in results["backups"].items():
            if service == "integrity":
                continue
            emoji = "✅" if status.get("status") == "healthy" else "⚠️" if status.get("status") == "warning" else "❌"
            print(f"  {emoji} {service}: {status.get('status', 'unknown')}")
            if "age_hours" in status:
                print(f"     Last backup: {status['age_hours']:.1f} hours ago")

        if results["issues"]:
            print("\n⚠️  Issues Found:")
            for issue in results["issues"]:
                print(f"  • {issue}")

        if results["recommendations"]:
            print("\n💡 Recommendations:")
            for rec in results["recommendations"]:
                print(f"  • {rec}")

    # Exit with appropriate code
    if results["overall_status"] == "critical":
        exit(2)
    elif results["overall_status"] == "warning":
        exit(1)
    else:
        exit(0)

if __name__ == "__main__":
    asyncio.run(main())
