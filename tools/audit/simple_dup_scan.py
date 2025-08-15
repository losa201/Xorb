#!/usr/bin/env python3
"""Simple duplication scanner for XORB monorepo audit."""

import json
import hashlib
from pathlib import Path
from collections import defaultdict


def find_exact_duplicates(root_path):
    """Find exact file duplicates using file hashes."""
    root = Path(root_path)
    file_hashes = defaultdict(list)

    for file_path in root.rglob("*.py"):
        if any(
            skip in str(file_path)
            for skip in [".git", "__pycache__", "node_modules", ".venv", "htmlcov"]
        ):
            continue

        try:
            with open(file_path, "rb") as f:
                content = f.read()
                if len(content) > 100:  # Skip very small files
                    file_hash = hashlib.sha256(content).hexdigest()
                    file_hashes[file_hash].append(str(file_path.relative_to(root)))
        except (IOError, OSError):
            continue

    # Return only files that have duplicates
    duplicates = {h: files for h, files in file_hashes.items() if len(files) > 1}
    return duplicates


def find_similar_names(root_path):
    """Find files with similar names that might be duplicates."""
    root = Path(root_path)
    similar_names = defaultdict(list)

    for file_path in root.rglob("*.py"):
        if any(
            skip in str(file_path)
            for skip in [".git", "__pycache__", "node_modules", ".venv", "htmlcov"]
        ):
            continue

        filename = file_path.stem.lower()
        # Group by similar base names
        base_name = filename.replace("_", "").replace("-", "")
        similar_names[base_name].append(str(file_path.relative_to(root)))

    # Return only groups with multiple files
    similar_files = {
        name: files for name, files in similar_names.items() if len(files) > 1
    }
    return similar_files


def find_large_files(root_path):
    """Find large files that might be candidates for deduplication."""
    root = Path(root_path)
    large_files = []

    for file_path in root.rglob("*"):
        if file_path.is_file() and not any(
            skip in str(file_path)
            for skip in [".git", "__pycache__", "node_modules", ".venv"]
        ):
            try:
                size = file_path.stat().st_size
                if size > 50000:  # Files larger than 50KB
                    large_files.append(
                        {
                            "file": str(file_path.relative_to(root)),
                            "size": size,
                            "size_mb": round(size / (1024 * 1024), 2),
                        }
                    )
            except (OSError, IOError):
                continue

    return sorted(large_files, key=lambda x: x["size"], reverse=True)[:20]


def analyze_router_duplication(root_path):
    """Analyze potential router duplication."""
    root = Path(root_path)
    router_files = []

    # Find router-like files
    router_patterns = ["router", "ptaas", "api", "endpoint", "handler"]

    for file_path in root.rglob("*.py"):
        if any(
            skip in str(file_path)
            for skip in [".git", "__pycache__", "node_modules", ".venv"]
        ):
            continue

        filename = file_path.name.lower()
        if any(pattern in filename for pattern in router_patterns):
            try:
                size = file_path.stat().st_size
                router_files.append(
                    {
                        "file": str(file_path.relative_to(root)),
                        "size": size,
                        "directory": str(file_path.parent.relative_to(root)),
                    }
                )
            except (OSError, IOError):
                continue

    return router_files


if __name__ == "__main__":
    import sys

    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    print(f"Running simplified duplication analysis on: {root_dir}")

    # Run simplified scans
    exact_duplicates = find_exact_duplicates(root_dir)
    similar_names = find_similar_names(root_dir)
    large_files = find_large_files(root_dir)
    router_files = analyze_router_duplication(root_dir)

    # Create impact analysis
    total_duplicate_files = sum(len(files) - 1 for files in exact_duplicates.values())
    high_impact_duplicates = [
        files for files in exact_duplicates.values() if len(files) > 3
    ]

    # Compile results
    duplication_data = {
        "exact_duplicates": exact_duplicates,
        "similar_names": similar_names,
        "large_files": large_files,
        "router_analysis": router_files,
        "summary": {
            "exact_duplicate_groups": len(exact_duplicates),
            "total_duplicate_files": total_duplicate_files,
            "similar_name_groups": len(similar_names),
            "large_files_count": len(large_files),
            "router_files_count": len(router_files),
            "high_impact_groups": len(high_impact_duplicates),
        },
        "impact_analysis": {
            "high_impact_duplicates": high_impact_duplicates,
            "router_proliferation": len(
                [f for f in router_files if "router" in f["file"]]
            ),
            "largest_files": large_files[:5],
        },
    }

    # Save results
    with open("docs/audit/catalog/duplication_analysis.json", "w") as f:
        json.dump(duplication_data, f, indent=2)

    print("Duplication analysis completed:")
    print(f"  Exact duplicate groups: {len(exact_duplicates)}")
    print(f"  Total duplicate files: {total_duplicate_files}")
    print(f"  Similar name groups: {len(similar_names)}")
    print(f"  Large files (>50KB): {len(large_files)}")
    print(f"  Router-like files: {len(router_files)}")
    print(f"  High impact groups: {len(high_impact_duplicates)}")
