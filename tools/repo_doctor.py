#!/usr/bin/env python3
"""
Repository Doctor Script for XORB Monorepo

This script performs various sanitation checks and generates audit reports:
- Duplicate file detection by hash
- Large file detection (>5MB)
- Proto file mapping
- Docker compose variant mapping

Usage:
    python3 tools/repo_doctor.py
"""

import os
import hashlib
import csv
import sys
from collections import defaultdict
from pathlib import Path
import argparse


def get_file_hash(file_path):
    """Calculate SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception:
        return None


def find_duplicates(root_path):
    """Find duplicate files by calculating hashes."""
    file_hashes = defaultdict(list)

    # Walk through all files in the repository
    for root, dirs, files in os.walk(root_path):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

        for file in files:
            file_path = os.path.join(root, file)
            # Skip if file is too large to hash efficiently
            try:
                if os.path.getsize(file_path) > 100 * 1024 * 1024:  # Skip files > 100MB
                    continue
            except OSError:
                continue

            file_hash = get_file_hash(file_path)
            if file_hash:
                file_hashes[file_hash].append(os.path.relpath(file_path, root_path))

    # Filter to only duplicates (more than one file with same hash)
    duplicates = {hash_val: paths for hash_val, paths in file_hashes.items() if len(paths) > 1}

    return duplicates


def find_large_files(root_path, size_threshold=5*1024*1024):  # 5MB threshold
    """Find files larger than the specified threshold."""
    large_files = []

    for root, dirs, files in os.walk(root_path):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

        for file in files:
            file_path = os.path.join(root, file)
            try:
                size = os.path.getsize(file_path)
                if size > size_threshold:
                    relative_path = os.path.relpath(file_path, root_path)
                    large_files.append((relative_path, size))
            except OSError:
                continue

    return large_files


def find_proto_files(root_path):
    """Find all .proto files and their service mappings."""
    proto_files = []

    for root, dirs, files in os.walk(root_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for file in files:
            if file.endswith('.proto'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, root_path)

                # Extract service names from proto file
                services = []
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if 'service ' in line and not line.strip().startswith('//'):
                                # Extract service name (simple parsing)
                                parts = line.split()
                                if len(parts) >= 2 and parts[0] == 'service':
                                    service_name = parts[1].rstrip('{').strip()
                                    services.append(service_name)
                except Exception:
                    pass

                proto_files.append((relative_path, services))

    return proto_files


def find_docker_compose_files(root_path):
    """Find all docker-compose variants."""
    compose_files = []

    for root, dirs, files in os.walk(root_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for file in files:
            if 'docker-compose' in file and file.endswith(('.yml', '.yaml')):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, root_path)
                compose_files.append(relative_path)

    return compose_files


def write_duplicates_report(duplicates, output_path):
    """Write duplicates report to CSV."""
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File Hash', 'File Paths'])
        for file_hash, paths in duplicates.items():
            writer.writerow([file_hash, '; '.join(paths)])


def write_size_report(large_files, output_path):
    """Write size report to CSV."""
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File Path', 'Size (bytes)', 'Size (MB)'])
        for path, size in large_files:
            size_mb = size / (1024 * 1024)
            writer.writerow([path, size, f"{size_mb:.2f}"])


def write_proto_map(proto_files, output_path):
    """Write proto map to markdown."""
    with open(output_path, 'w') as mdfile:
        mdfile.write("# Proto File Service Mapping\n\n")
        mdfile.write("| Proto File | Services |\n")
        mdfile.write("|------------|----------|\n")
        for path, services in proto_files:
            services_str = ', '.join(services) if services else 'None'
            mdfile.write(f"| {path} | {services_str} |\n")


def write_compose_map(compose_files, output_path):
    """Write compose map to markdown."""
    with open(output_path, 'w') as mdfile:
        mdfile.write("# Docker Compose Variants\n\n")
        mdfile.write("| Compose File | Path |\n")
        mdfile.write("|--------------|------|\n")
        for path in compose_files:
            mdfile.write(f"| {os.path.basename(path)} | {path} |\n")


def check_for_redis_pubsub_usage(root_path):
    """Check for Redis pubsub/subscribe usage (ADR-002 guard)."""
    redis_usage_files = []

    for root, dirs, files in os.walk(root_path):
        # Skip hidden directories and some known directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', '.git']]

        for file in files:
            if file.endswith(('.py', '.js', '.ts', '.go')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Check for Redis pub/sub patterns
                        if any(pattern in content.lower() for pattern in ['pubsub', 'publish', 'subscribe', 'psubscribe']):
                            # Additional check to avoid false positives
                            if any(redis_pattern in content.lower() for redis_pattern in ['redis', 'rediss']):
                                relative_path = os.path.relpath(file_path, root_path)
                                redis_usage_files.append(relative_path)
                except Exception:
                    continue

    return redis_usage_files


def main():
    """Main function to run all checks and generate reports."""
    parser = argparse.ArgumentParser(description="Repository Doctor for XORB Monorepo")
    parser.add_argument("--fail-on-redis-pubsub", action="store_true",
                        help="Exit with code 1 if Redis pubsub usage is found")
    parser.add_argument("--fail-on-dup", action="store_true",
                        help="Exit with code 1 if duplicate files are found")

    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.absolute()
    audit_dir = repo_root / 'tools' / 'repo_audit'

    # Create audit directory if it doesn't exist
    audit_dir.mkdir(exist_ok=True)

    print(f"Repository Doctor scanning: {repo_root}")

    # 1. Find duplicates
    print("Scanning for duplicate files...")
    duplicates = find_duplicates(repo_root)
    write_duplicates_report(duplicates, audit_dir / 'duplicates.csv')
    print(f"Found {len(duplicates)} sets of duplicate files")

    # 2. Find large files
    print("Scanning for large files...")
    large_files = find_large_files(repo_root)
    write_size_report(large_files, audit_dir / 'size_report.csv')
    print(f"Found {len(large_files)} files larger than 5MB")

    # 3. Find proto files
    print("Scanning for proto files...")
    proto_files = find_proto_files(repo_root)
    write_proto_map(proto_files, audit_dir / 'proto_map.md')
    print(f"Found {len(proto_files)} proto files")

    # 4. Find docker-compose files
    print("Scanning for docker-compose files...")
    compose_files = find_docker_compose_files(repo_root)
    write_compose_map(compose_files, audit_dir / 'compose_map.md')
    print(f"Found {len(compose_files)} docker-compose files")

    # 5. Check for Redis pubsub usage (ADR-002 guard)
    print("Checking for Redis pubsub usage...")
    redis_files = check_for_redis_pubsub_usage(repo_root)
    if redis_files:
        print(f"WARNING: Found Redis pubsub usage in {len(redis_files)} files:")
        for file in redis_files:
            print(f"  - {file}")
        # For CI purposes, we might want to create a report
        with open(audit_dir / 'redis_pubsub_usage.txt', 'w') as f:
            for file in redis_files:
                f.write(f"{file}\n")
    else:
        print("No Redis pubsub usage found")
        # Create empty file if none found
        (audit_dir / 'redis_pubsub_usage.txt').touch()

    print(f"All reports generated in: {audit_dir}")

    # Handle failure conditions
    exit_code = 0

    if args.fail_on_redis_pubsub and redis_files:
        print("\n❌ FAIL: Redis pubsub usage detected (ADR-002 violation)")
        exit_code = 1

    if args.fail_on_dup and duplicates:
        print("\n❌ FAIL: Duplicate files detected")
        exit_code = 1

    if exit_code != 0:
        print("Summary:")
        if args.fail_on_redis_pubsub and redis_files:
            print(f"  - Redis pubsub violations: {len(redis_files)} files")
        if args.fail_on_dup and duplicates:
            print(f"  - Duplicate files: {len(duplicates)} sets")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
