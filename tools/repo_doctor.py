#!/usr/bin/env python3
"""
Repository Doctor - Comprehensive duplicate detection and analysis tool.
Principal Codebase Surgeon tooling for XORB monorepo deduplication.
"""

import os
import csv
import json
import hashlib
import ast
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

@dataclass
class DuplicateGroup:
    """Represents a group of duplicate files."""
    hash_value: str
    files: List[str]
    size: int
    file_type: str
    canonical_path: Optional[str] = None
    action: str = "pending"  # pending, migrate, delete, merge

@dataclass
class NearDuplicate:
    """Represents near-duplicate files."""
    file1: str
    file2: str
    similarity: float
    size1: int
    size2: int
    canonical_path: Optional[str] = None
    action: str = "pending"

class RepoDoctorConfig:
    """Configuration for canonical locations."""
    
    CANONICAL_LOCATIONS = {
        "python_code": "src/",
        "orchestrator": "src/orchestrator/",
        "ptaas": "src/api/app/routers/",
        "ui_web": "ui/",
        "docs": "docs/",
        "reports": "docs/reports/",
        "protobuf": "proto/",
        "tests": "tests/",
        "tools": "tools/",
        "configs": "configs/",
        "scripts": "scripts/"
    }
    
    SKIP_PATHS = {
        ".git", "__pycache__", "node_modules", ".venv", "htmlcov",
        ".pytest_cache", ".mypy_cache", "build", "dist", ".tox"
    }
    
    SIMILARITY_THRESHOLD = 0.85
    MIN_FILE_SIZE = 100  # bytes

class RepoDoctor:
    """Main repository doctor class."""
    
    def __init__(self, root_path: str = "."):
        self.root = Path(root_path).resolve()
        self.config = RepoDoctorConfig()
        self.exact_duplicates: Dict[str, DuplicateGroup] = {}
        self.near_duplicates: List[NearDuplicate] = []
        self.file_hashes: Dict[str, str] = {}
        self.file_sizes: Dict[str, int] = {}
        
    def calculate_file_hash(self, file_path: Path) -> Optional[str]:
        """Calculate SHA256 hash of file content."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        except (IOError, OSError, PermissionError):
            return None
    
    def normalize_content(self, content: str, file_type: str) -> str:
        """Normalize content for similarity comparison."""
        if file_type == "python":
            # Remove comments, docstrings, and normalize whitespace
            try:
                tree = ast.parse(content)
                # Extract just the structure without docstrings
                normalized = ast.unparse(tree) if hasattr(ast, 'unparse') else content
            except SyntaxError:
                normalized = content
        elif file_type in ["markdown", "text"]:
            # Remove excess whitespace and normalize line endings
            normalized = '\n'.join(line.strip() for line in content.split('\n') if line.strip())
        else:
            normalized = content
        
        return normalized.strip()
    
    def get_file_type(self, file_path: Path) -> str:
        """Determine file type from extension."""
        suffix = file_path.suffix.lower()
        type_map = {
            '.py': 'python',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.md': 'markdown',
            '.rst': 'markdown',
            '.txt': 'text',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.toml': 'toml',
            '.ini': 'ini',
            '.cfg': 'config',
            '.proto': 'protobuf',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'css',
            '.sql': 'sql',
            '.sh': 'shell',
            '.bash': 'shell',
            '.dockerfile': 'docker',
        }
        return type_map.get(suffix, 'binary')
    
    def should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        path_str = str(file_path)
        
        # Skip paths in SKIP_PATHS
        if any(skip in path_str for skip in self.config.SKIP_PATHS):
            return True
        
        # Skip binary files that are likely build artifacts
        if file_path.suffix.lower() in {'.pyc', '.pyo', '.so', '.dylib', '.dll', '.exe'}:
            return True
        
        # Skip very small files
        try:
            if file_path.stat().st_size < self.config.MIN_FILE_SIZE:
                return True
        except (OSError, IOError):
            return True
        
        return False
    
    def find_exact_duplicates(self):
        """Find exact duplicate files using hash comparison."""
        print("🔍 Finding exact duplicates...")
        
        hash_to_files = defaultdict(list)
        
        for file_path in self.root.rglob("*"):
            if not file_path.is_file() or self.should_skip_file(file_path):
                continue
            
            file_hash = self.calculate_file_hash(file_path)
            if not file_hash:
                continue
            
            rel_path = str(file_path.relative_to(self.root))
            self.file_hashes[rel_path] = file_hash
            self.file_sizes[rel_path] = file_path.stat().st_size
            hash_to_files[file_hash].append(rel_path)
        
        # Create duplicate groups
        for file_hash, files in hash_to_files.items():
            if len(files) > 1:
                file_type = self.get_file_type(Path(files[0]))
                size = self.file_sizes[files[0]]
                
                group = DuplicateGroup(
                    hash_value=file_hash,
                    files=sorted(files),
                    size=size,
                    file_type=file_type
                )
                
                # Determine canonical path
                group.canonical_path = self.choose_canonical_path(files, file_type)
                self.exact_duplicates[file_hash] = group
        
        print(f"✅ Found {len(self.exact_duplicates)} exact duplicate groups")
    
    def choose_canonical_path(self, files: List[str], file_type: str) -> str:
        """Choose the canonical path for a group of duplicates."""
        # Priority order for different file types
        priority_prefixes = {
            "python": ["src/", "packages/", "tools/", "scripts/"],
            "typescript": ["ui/", "frontend/", "web/"],
            "javascript": ["ui/", "frontend/", "web/"],
            "markdown": ["docs/", "README"],
            "protobuf": ["proto/"],
            "yaml": ["configs/", ".github/"],
            "json": ["configs/", "package"],
            "shell": ["scripts/", "tools/"],
            "docker": ["docker/", "Dockerfile"],
        }
        
        file_priorities = priority_prefixes.get(file_type, ["src/", "tools/"])
        
        # Score files based on priority prefixes
        scored_files = []
        for file_path in files:
            score = 100  # Default score
            
            for i, prefix in enumerate(file_priorities):
                if file_path.startswith(prefix):
                    score = i  # Lower score = higher priority
                    break
            
            # Prefer shorter paths (less nested)
            path_depth = file_path.count('/')
            score += path_depth * 0.1
            
            # Prefer non-test files for non-test content
            if '/test' in file_path and file_type != 'test':
                score += 50
            
            scored_files.append((score, file_path))
        
        # Return the file with the lowest score (highest priority)
        scored_files.sort()
        return scored_files[0][1]
    
    def find_near_duplicates(self):
        """Find near-duplicate files using content similarity."""
        print("🔍 Finding near-duplicates...")
        
        # Group files by type and base name for efficient comparison
        files_by_type = defaultdict(list)
        
        for file_path in self.root.rglob("*"):
            if not file_path.is_file() or self.should_skip_file(file_path):
                continue
            
            file_type = self.get_file_type(file_path)
            if file_type in {"python", "typescript", "javascript", "markdown"}:
                rel_path = str(file_path.relative_to(self.root))
                
                # Skip files that are exact duplicates
                file_hash = self.file_hashes.get(rel_path)
                if file_hash and file_hash in self.exact_duplicates:
                    continue
                
                files_by_type[file_type].append((rel_path, file_path))
        
        # Compare files within each type
        for file_type, files in files_by_type.items():
            print(f"  Analyzing {len(files)} {file_type} files...")
            
            for i, (path1, file1) in enumerate(files):
                for path2, file2 in files[i+1:]:
                    similarity = self.calculate_similarity(file1, file2, file_type)
                    
                    if similarity >= self.config.SIMILARITY_THRESHOLD:
                        size1 = self.file_sizes.get(path1, 0)
                        size2 = self.file_sizes.get(path2, 0)
                        
                        near_dup = NearDuplicate(
                            file1=path1,
                            file2=path2,
                            similarity=similarity,
                            size1=size1,
                            size2=size2
                        )
                        
                        # Choose canonical path
                        near_dup.canonical_path = self.choose_canonical_path(
                            [path1, path2], file_type
                        )
                        
                        self.near_duplicates.append(near_dup)
        
        # Sort by similarity (highest first)
        self.near_duplicates.sort(key=lambda x: x.similarity, reverse=True)
        print(f"✅ Found {len(self.near_duplicates)} near-duplicate pairs")
    
    def calculate_similarity(self, file1: Path, file2: Path, file_type: str) -> float:
        """Calculate similarity between two files."""
        try:
            with open(file1, 'r', encoding='utf-8', errors='ignore') as f:
                content1 = f.read()
            with open(file2, 'r', encoding='utf-8', errors='ignore') as f:
                content2 = f.read()
            
            # Normalize content
            norm1 = self.normalize_content(content1, file_type)
            norm2 = self.normalize_content(content2, file_type)
            
            return SequenceMatcher(None, norm1, norm2).ratio()
            
        except (IOError, OSError, UnicodeDecodeError):
            return 0.0
    
    def save_duplicates_csv(self, output_path: str = "tools/repo_audit/duplicates.csv"):
        """Save exact duplicates to CSV."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'hash', 'canonical_path', 'duplicate_paths', 'file_type', 
                'size_bytes', 'action', 'priority'
            ])
            
            for group in self.exact_duplicates.values():
                duplicates = [f for f in group.files if f != group.canonical_path]
                priority = "HIGH" if len(group.files) > 3 or group.size > 10000 else "MEDIUM"
                
                writer.writerow([
                    group.hash_value[:16],  # Shortened hash
                    group.canonical_path,
                    '; '.join(duplicates),
                    group.file_type,
                    group.size,
                    group.action,
                    priority
                ])
    
    def save_near_duplicates_csv(self, output_path: str = "tools/repo_audit/near_duplicates.csv"):
        """Save near duplicates to CSV."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'canonical_path', 'duplicate_path', 'similarity', 'size1', 'size2', 
                'action', 'priority'
            ])
            
            for near_dup in self.near_duplicates:
                canonical = near_dup.canonical_path
                duplicate = near_dup.file2 if canonical == near_dup.file1 else near_dup.file1
                
                priority = "HIGH" if near_dup.similarity > 0.95 else "MEDIUM"
                
                writer.writerow([
                    canonical,
                    duplicate,
                    f"{near_dup.similarity:.3f}",
                    near_dup.size1,
                    near_dup.size2,
                    near_dup.action,
                    priority
                ])
    
    def generate_summary_report(self) -> Dict:
        """Generate summary statistics."""
        total_duplicate_files = sum(len(g.files) - 1 for g in self.exact_duplicates.values())
        total_wasted_space = sum(g.size * (len(g.files) - 1) for g in self.exact_duplicates.values())
        
        file_type_stats = defaultdict(int)
        for group in self.exact_duplicates.values():
            file_type_stats[group.file_type] += len(group.files) - 1
        
        return {
            "exact_duplicate_groups": len(self.exact_duplicates),
            "total_duplicate_files": total_duplicate_files,
            "total_wasted_space_bytes": total_wasted_space,
            "near_duplicate_pairs": len(self.near_duplicates),
            "file_type_breakdown": dict(file_type_stats),
            "high_priority_exact": len([g for g in self.exact_duplicates.values() 
                                       if len(g.files) > 3 or g.size > 10000]),
            "high_priority_near": len([n for n in self.near_duplicates 
                                      if n.similarity > 0.95])
        }
    
    def run_analysis(self):
        """Run complete duplicate analysis."""
        print("🏥 Repository Doctor - Starting comprehensive analysis...")
        
        self.find_exact_duplicates()
        self.find_near_duplicates()
        
        # Save results
        self.save_duplicates_csv()
        self.save_near_duplicates_csv()
        
        # Generate summary
        summary = self.generate_summary_report()
        
        with open("tools/repo_audit/summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print("\n📊 Analysis Summary:")
        print(f"  Exact duplicate groups: {summary['exact_duplicate_groups']}")
        print(f"  Total duplicate files: {summary['total_duplicate_files']}")
        print(f"  Wasted space: {summary['total_wasted_space_bytes']:,} bytes")
        print(f"  Near-duplicate pairs: {summary['near_duplicate_pairs']}")
        print(f"  High priority exact: {summary['high_priority_exact']}")
        print(f"  High priority near: {summary['high_priority_near']}")
        
        print(f"\n📁 Results saved:")
        print(f"  tools/repo_audit/duplicates.csv")
        print(f"  tools/repo_audit/near_duplicates.csv") 
        print(f"  tools/repo_audit/summary.json")

if __name__ == "__main__":
    import sys
    root_path = sys.argv[1] if len(sys.argv) > 1 else "."
    
    doctor = RepoDoctor(root_path)
    doctor.run_analysis()