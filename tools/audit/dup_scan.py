#!/usr/bin/env python3
"""Duplication analysis scanner for XORB monorepo audit."""

import os
import json
import hashlib
import ast
import re
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher


def calculate_file_hash(filepath):
    """Calculate SHA256 hash of file content."""
    try:
        with open(filepath, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except (IOError, OSError):
        return None


def normalize_function_content(content):
    """Normalize function content for comparison."""
    # Remove comments and docstrings
    content = re.sub(r"#.*", "", content)
    content = re.sub(r'""".*?"""', "", content, flags=re.DOTALL)
    content = re.sub(r"'''.*?'''", "", content, flags=re.DOTALL)

    # Normalize whitespace
    content = re.sub(r"\s+", " ", content)
    content = content.strip()

    return content


def extract_functions_and_classes(file_path):
    """Extract functions and classes from Python files."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        tree = ast.parse(content)
        functions = []
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_content = ast.get_source_segment(content, node)
                if func_content:
                    normalized = normalize_function_content(func_content)
                    functions.append(
                        {
                            "name": node.name,
                            "line": node.lineno,
                            "content": func_content,
                            "normalized": normalized,
                            "hash": hashlib.sha256(normalized.encode()).hexdigest(),
                        }
                    )

            elif isinstance(node, ast.ClassDef):
                class_content = ast.get_source_segment(content, node)
                if class_content:
                    normalized = normalize_function_content(class_content)
                    classes.append(
                        {
                            "name": node.name,
                            "line": node.lineno,
                            "content": class_content,
                            "normalized": normalized,
                            "hash": hashlib.sha256(normalized.encode()).hexdigest(),
                        }
                    )

        return functions, classes

    except (SyntaxError, IOError, OSError, UnicodeDecodeError):
        return [], []


def find_exact_duplicates(root_path):
    """Find exact file duplicates (Type-1 clones)."""
    root = Path(root_path)
    file_hashes = defaultdict(list)

    for file_path in root.rglob("*"):
        if file_path.is_file() and not any(
            skip in str(file_path)
            for skip in [".git", "__pycache__", "node_modules", ".venv"]
        ):
            file_hash = calculate_file_hash(file_path)
            if file_hash:
                file_hashes[file_hash].append(str(file_path.relative_to(root)))

    # Return only duplicates (more than one file with same hash)
    duplicates = {h: files for h, files in file_hashes.items() if len(files) > 1}

    return duplicates


def find_function_clones(root_path):
    """Find function-level clones."""
    root = Path(root_path)
    function_hashes = defaultdict(list)

    for py_file in root.rglob("*.py"):
        if any(
            skip in str(py_file)
            for skip in [".git", "__pycache__", "node_modules", ".venv"]
        ):
            continue

        functions, _ = extract_functions_and_classes(py_file)

        for func in functions:
            function_hashes[func["hash"]].append(
                {
                    "file": str(py_file.relative_to(root)),
                    "name": func["name"],
                    "line": func["line"],
                    "content_preview": func["content"][:200] + "..."
                    if len(func["content"]) > 200
                    else func["content"],
                }
            )

    # Return only clones (more than one function with same hash)
    clones = {h: funcs for h, funcs in function_hashes.items() if len(funcs) > 1}

    return clones


def find_near_duplicates(root_path, similarity_threshold=0.8):
    """Find near-duplicate files (Type-2/3 clones)."""
    root = Path(root_path)
    files_content = {}

    # Read file contents for comparison
    for file_path in root.rglob("*.py"):
        if any(
            skip in str(file_path)
            for skip in [".git", "__pycache__", "node_modules", ".venv"]
        ):
            continue

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Normalize content for comparison
            normalized = normalize_function_content(content)
            if len(normalized) > 100:  # Skip very short files
                files_content[str(file_path.relative_to(root))] = normalized
        except (IOError, OSError, UnicodeDecodeError):
            continue

    near_duplicates = []

    # Compare all pairs of files
    file_list = list(files_content.items())
    for i, (file1, content1) in enumerate(file_list):
        for file2, content2 in file_list[i + 1 :]:
            similarity = SequenceMatcher(None, content1, content2).ratio()

            if similarity >= similarity_threshold:
                near_duplicates.append(
                    {
                        "file1": file1,
                        "file2": file2,
                        "similarity": round(similarity, 3),
                        "size1": len(content1),
                        "size2": len(content2),
                    }
                )

    return sorted(near_duplicates, key=lambda x: x["similarity"], reverse=True)


def find_constant_duplicates(root_path):
    """Find duplicated constants and configuration."""
    root = Path(root_path)
    constants = defaultdict(list)

    # Common constant patterns
    constant_patterns = [
        r"([A-Z_]+)\s*=\s*['\"]([^'\"]+)['\"]",  # String constants
        r"([A-Z_]+)\s*=\s*(\d+)",  # Numeric constants
        r"DEFAULT_([A-Z_]+)\s*=\s*['\"]([^'\"]+)['\"]",  # Default values
        r"([A-Z_]+)_URL\s*=\s*['\"]([^'\"]+)['\"]",  # URL constants
        r"([A-Z_]+)_PORT\s*=\s*(\d+)",  # Port constants
    ]

    for py_file in root.rglob("*.py"):
        if any(
            skip in str(py_file)
            for skip in [".git", "__pycache__", "node_modules", ".venv"]
        ):
            continue

        try:
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            for pattern in constant_patterns:
                matches = re.finditer(pattern, content, re.MULTILINE)
                for match in matches:
                    const_name = match.group(1)
                    const_value = match.group(2)

                    constants[f"{const_name}={const_value}"].append(
                        {
                            "file": str(py_file.relative_to(root)),
                            "line": content[: match.start()].count("\n") + 1,
                            "name": const_name,
                            "value": const_value,
                        }
                    )

        except (IOError, OSError, UnicodeDecodeError):
            continue

    # Return only duplicated constants
    duplicated_constants = {k: v for k, v in constants.items() if len(v) > 1}

    return duplicated_constants


def find_config_key_duplicates(root_path):
    """Find duplicated configuration keys."""
    root = Path(root_path)
    config_keys = defaultdict(list)

    # Configuration file patterns
    config_files = []
    config_patterns = ["*.yml", "*.yaml", "*.json", "*.toml", "*.ini", "*.cfg"]

    for pattern in config_patterns:
        config_files.extend(root.rglob(pattern))

    for config_file in config_files:
        if any(skip in str(config_file) for skip in [".git", "node_modules", ".venv"]):
            continue

        try:
            with open(config_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Extract key-value patterns
            key_patterns = [
                r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*(.+)$",  # General key: value
                r'"([^"]+)"\s*:\s*"([^"]+)"',  # JSON string keys
                r"'([^']+)'\s*:\s*'([^']+)'",  # YAML string keys
            ]

            for pattern in key_patterns:
                matches = re.finditer(pattern, content, re.MULTILINE)
                for match in matches:
                    key = match.group(1)
                    value = match.group(2).strip()

                    if len(key) > 2 and len(value) > 0:  # Skip very short keys
                        config_keys[key].append(
                            {
                                "file": str(config_file.relative_to(root)),
                                "line": content[: match.start()].count("\n") + 1,
                                "key": key,
                                "value": value[:100] + "..."
                                if len(value) > 100
                                else value,
                            }
                        )

        except (IOError, OSError, UnicodeDecodeError):
            continue

    # Return only duplicated keys
    duplicated_keys = {k: v for k, v in config_keys.items() if len(v) > 1}

    return duplicated_keys


def analyze_duplication_impact(duplicates_data):
    """Analyze the impact and priority of duplications."""
    impact_analysis = {
        "high_impact": [],
        "medium_impact": [],
        "low_impact": [],
        "total_duplicate_files": 0,
        "total_wasted_space": 0,
        "refactoring_opportunities": [],
    }

    # Analyze exact duplicates
    for file_hash, files in duplicates_data.get("exact_duplicates", {}).items():
        if len(files) > 1:
            # Estimate file size (approximate)
            try:
                file_size = (
                    os.path.getsize(files[0]) if os.path.exists(files[0]) else 1000
                )
                wasted_space = file_size * (len(files) - 1)

                impact_analysis["total_duplicate_files"] += len(files) - 1
                impact_analysis["total_wasted_space"] += wasted_space

                duplicate_info = {
                    "type": "exact_duplicate",
                    "files": files,
                    "count": len(files),
                    "wasted_space": wasted_space,
                }

                if len(files) > 5 or wasted_space > 10000:
                    impact_analysis["high_impact"].append(duplicate_info)
                elif len(files) > 2 or wasted_space > 1000:
                    impact_analysis["medium_impact"].append(duplicate_info)
                else:
                    impact_analysis["low_impact"].append(duplicate_info)

            except (OSError, IOError):
                pass

    # Analyze function clones
    function_clones = duplicates_data.get("function_clones", {})
    for func_hash, functions in function_clones.items():
        if len(functions) > 2:
            impact_analysis["refactoring_opportunities"].append(
                {
                    "type": "function_clone",
                    "function_name": functions[0]["name"],
                    "occurrences": len(functions),
                    "files": [f["file"] for f in functions],
                }
            )

    # Analyze near duplicates
    near_duplicates = duplicates_data.get("near_duplicates", [])
    for near_dup in near_duplicates:
        if near_dup["similarity"] > 0.9:
            impact_analysis["refactoring_opportunities"].append(
                {
                    "type": "near_duplicate",
                    "files": [near_dup["file1"], near_dup["file2"]],
                    "similarity": near_dup["similarity"],
                }
            )

    return impact_analysis


if __name__ == "__main__":
    import sys

    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    print(f"Running duplication analysis on: {root_dir}")

    # Run all duplication scans
    exact_duplicates = find_exact_duplicates(root_dir)
    function_clones = find_function_clones(root_dir)
    near_duplicates = find_near_duplicates(root_dir)
    constant_duplicates = find_constant_duplicates(root_dir)
    config_key_duplicates = find_config_key_duplicates(root_dir)

    # Compile duplication data
    duplicates_data = {
        "exact_duplicates": exact_duplicates,
        "function_clones": function_clones,
        "near_duplicates": near_duplicates,
        "constant_duplicates": constant_duplicates,
        "config_key_duplicates": config_key_duplicates,
        "summary": {
            "exact_duplicate_groups": len(exact_duplicates),
            "function_clone_groups": len(function_clones),
            "near_duplicate_pairs": len(near_duplicates),
            "constant_duplicates": len(constant_duplicates),
            "config_key_duplicates": len(config_key_duplicates),
        },
    }

    # Analyze impact
    impact_analysis = analyze_duplication_impact(duplicates_data)
    duplicates_data["impact_analysis"] = impact_analysis

    # Save duplication analysis
    with open("docs/audit/catalog/duplication_analysis.json", "w") as f:
        json.dump(duplicates_data, f, indent=2)

    print("Duplication analysis completed:")
    print(f"  Exact duplicate groups: {len(exact_duplicates)}")
    print(f"  Function clone groups: {len(function_clones)}")
    print(f"  Near duplicate pairs: {len(near_duplicates)}")
    print(f"  Constant duplicates: {len(constant_duplicates)}")
    print(f"  Config key duplicates: {len(config_key_duplicates)}")
    print(f"  High impact issues: {len(impact_analysis['high_impact'])}")
    print(f"  Total wasted space: {impact_analysis['total_wasted_space']} bytes")
