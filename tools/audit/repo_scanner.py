#!/usr/bin/env python3
"""Repository topology scanner for XORB monorepo audit."""

import hashlib
import json
import csv
from pathlib import Path
from collections import defaultdict


def calculate_sha256(filepath):
    """Calculate SHA256 hash of a file."""
    try:
        with open(filepath, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except (IOError, OSError):
        return "ERROR"


def get_file_extension(path):
    """Get normalized file extension."""
    ext = Path(path).suffix.lower()
    return ext if ext else "(no_ext)"


def estimate_loc(filepath):
    """Estimate lines of code for text files."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for line in f if line.strip())
    except (IOError, OSError, UnicodeDecodeError):
        return 0


def is_binary_file(filepath):
    """Check if file is binary."""
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(8192)
            return b"\0" in chunk
    except (IOError, OSError):
        return False


def is_generated_file(path):
    """Detect generated/vendor files."""
    path_str = str(path).lower()
    generated_patterns = [
        "node_modules/",
        "venv/",
        ".venv/",
        "__pycache__/",
        ".git/",
        ".pytest_cache/",
        "htmlcov/",
        ".coverage",
        "dist/",
        "build/",
        "target/",
        "pkg/",
        ".lock",
        "package-lock.json",
        "yarn.lock",
        "requirements.lock",
        "poetry.lock",
        ".pb.go",
        "_pb2.py",
        ".pb.h",
        ".pb.cc",
        "vendor/",
        "third_party/",
        "external/",
        ".min.js",
        ".min.css",
        "bundle.js",
        "bundle.css",
        ".d.ts",
        ".map",
    ]
    return any(pattern in path_str for pattern in generated_patterns)


def scan_repository(root_path):
    """Scan repository and gather file information."""
    root = Path(root_path)
    file_index = []
    language_stats = defaultdict(lambda: {"files": 0, "loc": 0, "size": 0})
    large_files = []
    duplicates = defaultdict(list)

    # Language mappings
    lang_map = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".go": "Go",
        ".rs": "Rust",
        ".java": "Java",
        ".c": "C",
        ".cpp": "C++",
        ".cc": "C++",
        ".cxx": "C++",
        ".h": "C Header",
        ".hpp": "C++ Header",
        ".sql": "SQL",
        ".sh": "Shell",
        ".bash": "Shell",
        ".yml": "YAML",
        ".yaml": "YAML",
        ".json": "JSON",
        ".xml": "XML",
        ".html": "HTML",
        ".css": "CSS",
        ".md": "Markdown",
        ".rst": "reStructuredText",
        ".toml": "TOML",
        ".ini": "INI",
        ".cfg": "Config",
        ".dockerfile": "Docker",
        ".proto": "Protobuf",
        ".tf": "Terraform",
        ".hcl": "HCL",
    }

    for file_path in root.rglob("*"):
        if file_path.is_file():
            try:
                stat_info = file_path.stat()
                size = stat_info.st_size
                rel_path = file_path.relative_to(root)
                ext = get_file_extension(file_path)

                # Calculate hash for duplicate detection
                file_hash = calculate_sha256(file_path)
                if file_hash != "ERROR":
                    duplicates[file_hash].append(str(rel_path))

                # Check if binary
                is_binary = is_binary_file(file_path)
                is_generated = is_generated_file(rel_path)

                # Estimate LOC for text files
                loc = 0 if is_binary or is_generated else estimate_loc(file_path)

                # Add to file index
                file_index.append(
                    {
                        "path": str(rel_path),
                        "size": size,
                        "ext": ext,
                        "sha256": file_hash,
                        "is_binary": is_binary,
                        "is_generated": is_generated,
                        "loc": loc,
                    }
                )

                # Language statistics
                lang = lang_map.get(ext, f"Other({ext})")
                if not is_generated:
                    language_stats[lang]["files"] += 1
                    language_stats[lang]["loc"] += loc
                    language_stats[lang]["size"] += size

                # Track large files (>1MB)
                if size > 1024 * 1024:
                    large_files.append(
                        {
                            "path": str(rel_path),
                            "size": size,
                            "size_mb": round(size / (1024 * 1024), 2),
                        }
                    )

            except (OSError, IOError):
                continue

    return {
        "file_index": file_index,
        "language_stats": dict(language_stats),
        "large_files": sorted(large_files, key=lambda x: x["size"], reverse=True)[:50],
        "duplicates": {h: paths for h, paths in duplicates.items() if len(paths) > 1},
    }


def generate_tree_summary(root_path):
    """Generate tree structure summary."""
    root = Path(root_path)
    tree_summary = {}

    for item in root.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            dir_stats = {"files": 0, "size": 0, "subdirs": 0}
            for file_path in item.rglob("*"):
                if file_path.is_file():
                    try:
                        dir_stats["files"] += 1
                        dir_stats["size"] += file_path.stat().st_size
                    except (OSError, IOError):
                        continue
                elif file_path.is_dir():
                    dir_stats["subdirs"] += 1
            tree_summary[item.name] = dir_stats

    return tree_summary


def find_notable_files(root_path):
    """Find notable configuration and build files."""
    root = Path(root_path)
    notable_patterns = {
        "Docker": ["Dockerfile*", "docker-compose*.yml", "docker-compose*.yaml"],
        "Build": ["Makefile", "CMakeLists.txt", "build.gradle", "pom.xml"],
        "CI/CD": [
            ".github/workflows/*.yml",
            ".github/workflows/*.yaml",
            ".gitlab-ci.yml",
        ],
        "Config": ["*.toml", "*.ini", "*.cfg", "config.*"],
        "Package": ["package.json", "requirements*.txt", "Cargo.toml", "go.mod"],
        "Kubernetes": ["*.yaml", "*.yml"],
        "Terraform": ["*.tf", "*.tfvars"],
    }

    notable_files = defaultdict(list)

    for category, patterns in notable_patterns.items():
        for pattern in patterns:
            for file_path in root.glob(pattern):
                if file_path.is_file():
                    notable_files[category].append(str(file_path.relative_to(root)))
            for file_path in root.rglob(pattern):
                if file_path.is_file():
                    rel_path = str(file_path.relative_to(root))
                    if rel_path not in notable_files[category]:
                        notable_files[category].append(rel_path)

    return dict(notable_files)


if __name__ == "__main__":
    import sys

    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    print(f"Scanning repository: {root_dir}")

    # Scan repository
    scan_results = scan_repository(root_dir)
    tree_summary = generate_tree_summary(root_dir)
    notable_files = find_notable_files(root_dir)

    # Save file index
    with open("docs/audit/catalog/file_index.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "size",
                "ext",
                "sha256",
                "is_binary",
                "is_generated",
                "loc",
            ],
        )
        writer.writeheader()
        writer.writerows(scan_results["file_index"])

    # Save language stats
    with open("docs/audit/catalog/language_stats.json", "w") as f:
        json.dump(scan_results["language_stats"], f, indent=2)

    # Save large files
    with open("docs/audit/catalog/large_files.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "size", "size_mb"])
        writer.writeheader()
        writer.writerows(scan_results["large_files"])

    # Save tree summary
    with open("docs/audit/catalog/tree_summary.json", "w") as f:
        json.dump(tree_summary, f, indent=2)

    # Save notable files
    with open("docs/audit/catalog/notable_files.json", "w") as f:
        json.dump(notable_files, f, indent=2)

    # Save duplicates
    with open("docs/audit/catalog/duplicates.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hash", "file_count", "paths"])
        for file_hash, paths in scan_results["duplicates"].items():
            writer.writerow([file_hash, len(paths), "|".join(paths)])

    print("Generated catalogs in docs/audit/catalog/")
    print(f"Total files scanned: {len(scan_results['file_index'])}")
    print(f"Languages detected: {len(scan_results['language_stats'])}")
    print(f"Large files (>1MB): {len(scan_results['large_files'])}")
    print(f"Duplicate file groups: {len(scan_results['duplicates'])}")
