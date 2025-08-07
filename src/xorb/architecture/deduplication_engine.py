#!/usr/bin/env python3
"""
XORB Advanced Deduplication Engine
Intelligent elimination of redundant code, configurations, and functionalities
"""

import asyncio
import logging
import os
import ast
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
from datetime import datetime
import difflib

from xorb.architecture.observability import get_observability, trace

logger = logging.getLogger(__name__)

class RedundancyType(Enum):
    CODE_DUPLICATION = "code_duplication"
    CONFIG_DUPLICATION = "config_duplication"
    FUNCTION_SIMILARITY = "function_similarity"
    CLASS_SIMILARITY = "class_similarity"
    IMPORT_REDUNDANCY = "import_redundancy"
    DOCUMENTATION_DUPLICATION = "documentation_duplication"
    DEPLOYMENT_REDUNDANCY = "deployment_redundancy"
    DATABASE_REDUNDANCY = "database_redundancy"

class DeduplicationAction(Enum):
    ELIMINATE = "eliminate"           # Remove completely redundant items
    CONSOLIDATE = "consolidate"       # Merge similar items
    EXTRACT_COMMON = "extract_common" # Extract to shared module
    STANDARDIZE = "standardize"       # Standardize variations
    REFACTOR = "refactor"            # Refactor to remove duplication

@dataclass
class RedundancyItem:
    """Represents a redundant item found in the codebase."""
    type: RedundancyType
    source_files: List[str]
    similarity_score: float
    content_hash: str
    content_preview: str
    size_bytes: int
    recommended_action: DeduplicationAction
    consolidation_target: Optional[str] = None
    estimated_savings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeduplicationPlan:
    """Plan for eliminating redundancy."""
    redundancy_item: RedundancyItem
    action: DeduplicationAction
    target_location: str
    affected_files: List[str]
    implementation_steps: List[str]
    risk_level: str
    estimated_effort: str
    validation_criteria: List[str]

class XORBDeduplicationEngine:
    """Advanced deduplication engine for XORB platform."""
    
    def __init__(self, root_path: str = "/root/Xorb"):
        self.root_path = Path(root_path)
        self.observability = None
        
        # Analysis results
        self.redundant_items: List[RedundancyItem] = []
        self.deduplication_plans: List[DeduplicationPlan] = []
        
        # Configuration
        self.similarity_threshold = 0.85
        self.min_content_size = 50  # Minimum size to consider for deduplication
        
        # File patterns to analyze
        self.python_patterns = ["**/*.py"]
        self.config_patterns = ["**/*.json", "**/*.yaml", "**/*.yml", "**/*.toml"]
        self.docker_patterns = ["**/Dockerfile*", "**/docker-compose*.yml"]
        self.docs_patterns = ["**/*.md", "**/*.rst", "**/*.txt"]
        
        # Exclusion patterns
        self.exclude_patterns = [
            "__pycache__",
            ".git",
            ".pytest_cache",
            "node_modules",
            "venv",
            "env"
        ]
    
    async def initialize(self):
        """Initialize the deduplication engine."""
        self.observability = await get_observability()
        logger.info("XORB Deduplication Engine initialized")
    
    @trace("comprehensive_deduplication_analysis")
    async def analyze_comprehensive_redundancy(self) -> Dict[str, Any]:
        """Perform comprehensive redundancy analysis."""
        logger.info("🔍 Starting comprehensive redundancy analysis")
        
        analysis_results = {}
        
        # Analyze different types of redundancy
        analysis_results["code_duplication"] = await self._analyze_code_duplication()
        analysis_results["config_duplication"] = await self._analyze_config_duplication()
        analysis_results["function_similarity"] = await self._analyze_function_similarity()
        analysis_results["class_similarity"] = await self._analyze_class_similarity()
        analysis_results["import_redundancy"] = await self._analyze_import_redundancy()
        analysis_results["documentation_duplication"] = await self._analyze_documentation_duplication()
        analysis_results["deployment_redundancy"] = await self._analyze_deployment_redundancy()
        analysis_results["database_redundancy"] = await self._analyze_database_redundancy()
        
        # Generate deduplication plans
        await self._generate_deduplication_plans()
        
        # Calculate impact metrics
        impact_metrics = await self._calculate_impact_metrics()
        
        return {
            "analysis_results": analysis_results,
            "redundancy_items": [item.__dict__ for item in self.redundant_items],
            "deduplication_plans": [plan.__dict__ for plan in self.deduplication_plans],
            "impact_metrics": impact_metrics,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _analyze_code_duplication(self) -> Dict[str, Any]:
        """Analyze code duplication across Python files."""
        logger.info("Analyzing code duplication...")
        
        python_files = []
        for pattern in self.python_patterns:
            python_files.extend(self.root_path.glob(pattern))
        
        # Filter out excluded paths
        python_files = [f for f in python_files if not any(exc in str(f) for exc in self.exclude_patterns)]
        
        code_blocks = {}
        duplicates = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract significant code blocks
                blocks = self._extract_code_blocks(content, str(file_path))
                
                for block in blocks:
                    content_hash = hashlib.md5(block['normalized_content'].encode()).hexdigest()
                    
                    if content_hash in code_blocks:
                        # Found duplicate
                        existing = code_blocks[content_hash]
                        
                        if existing['file'] != str(file_path):  # Different files
                            duplicate_item = RedundancyItem(
                                type=RedundancyType.CODE_DUPLICATION,
                                source_files=[existing['file'], str(file_path)],
                                similarity_score=1.0,  # Exact match
                                content_hash=content_hash,
                                content_preview=block['content'][:200] + "...",
                                size_bytes=len(block['content']),
                                recommended_action=DeduplicationAction.EXTRACT_COMMON,
                                estimated_savings={
                                    "lines_of_code": block['content'].count('\n'),
                                    "maintenance_reduction": "high"
                                }
                            )
                            duplicates.append(duplicate_item)
                    else:
                        code_blocks[content_hash] = {
                            'file': str(file_path),
                            'content': block['content'],
                            'line_start': block['line_start'],
                            'line_end': block['line_end']
                        }
            
            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")
        
        self.redundant_items.extend(duplicates)
        
        return {
            "files_analyzed": len(python_files),
            "duplicates_found": len(duplicates),
            "total_duplicate_lines": sum(item.estimated_savings.get("lines_of_code", 0) for item in duplicates)
        }
    
    def _extract_code_blocks(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Extract significant code blocks from Python content."""
        blocks = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    # Extract function/class content
                    lines = content.split('\n')
                    start_line = node.lineno - 1
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
                    
                    block_content = '\n'.join(lines[start_line:end_line])
                    
                    if len(block_content) >= self.min_content_size:
                        # Normalize content for comparison
                        normalized = self._normalize_code_block(block_content)
                        
                        blocks.append({
                            'content': block_content,
                            'normalized_content': normalized,
                            'line_start': start_line + 1,
                            'line_end': end_line,
                            'type': 'function' if isinstance(node, ast.FunctionDef) else 'class',
                            'name': node.name
                        })
        
        except SyntaxError:
            # If parsing fails, try line-based analysis
            lines = content.split('\n')
            for i in range(0, len(lines) - 10, 5):  # Sliding window of 10 lines
                block_lines = lines[i:i+10]
                block_content = '\n'.join(block_lines)
                
                if len(block_content.strip()) >= self.min_content_size:
                    blocks.append({
                        'content': block_content,
                        'normalized_content': self._normalize_code_block(block_content),
                        'line_start': i + 1,
                        'line_end': i + 10,
                        'type': 'block',
                        'name': f'block_{i}'
                    })
        
        return blocks
    
    def _normalize_code_block(self, content: str) -> str:
        """Normalize code block for comparison."""
        lines = content.split('\n')
        normalized_lines = []
        
        for line in lines:
            # Remove leading/trailing whitespace
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Remove inline comments
            if '#' in line:
                line = line[:line.index('#')].strip()
            
            # Normalize string literals
            import re
            line = re.sub(r'"[^"]*"', '""', line)
            line = re.sub(r"'[^']*'", "''", line)
            
            normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    async def _analyze_config_duplication(self) -> Dict[str, Any]:
        """Analyze configuration file duplication."""
        logger.info("Analyzing configuration duplication...")
        
        config_files = []
        for pattern in self.config_patterns:
            config_files.extend(self.root_path.glob(pattern))
        
        config_files = [f for f in config_files if not any(exc in str(f) for exc in self.exclude_patterns)]
        
        config_contents = {}
        duplicates = []
        
        for file_path in config_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse configuration based on file type
                parsed_config = self._parse_config_file(content, file_path.suffix)
                
                if parsed_config:
                    config_hash = self._hash_config(parsed_config)
                    
                    if config_hash in config_contents:
                        existing_file = config_contents[config_hash]
                        
                        duplicate_item = RedundancyItem(
                            type=RedundancyType.CONFIG_DUPLICATION,
                            source_files=[existing_file, str(file_path)],
                            similarity_score=1.0,
                            content_hash=config_hash,
                            content_preview=content[:200] + "...",
                            size_bytes=len(content),
                            recommended_action=DeduplicationAction.CONSOLIDATE,
                            estimated_savings={
                                "config_files": 1,
                                "maintenance_reduction": "medium"
                            }
                        )
                        duplicates.append(duplicate_item)
                    else:
                        config_contents[config_hash] = str(file_path)
            
            except Exception as e:
                logger.warning(f"Failed to analyze config {file_path}: {e}")
        
        self.redundant_items.extend(duplicates)
        
        return {
            "files_analyzed": len(config_files),
            "duplicates_found": len(duplicates)
        }
    
    def _parse_config_file(self, content: str, file_extension: str) -> Optional[Any]:
        """Parse configuration file based on type."""
        try:
            if file_extension in ['.json']:
                return json.loads(content)
            elif file_extension in ['.yml', '.yaml']:
                import yaml
                return yaml.safe_load(content)
            elif file_extension == '.toml':
                import toml
                return toml.loads(content)
        except Exception:
            return None
        
        return content  # Return raw content if parsing fails
    
    def _hash_config(self, config: Any) -> str:
        """Create hash of configuration for comparison."""
        if isinstance(config, dict):
            # Sort keys for consistent hashing
            sorted_config = json.dumps(config, sort_keys=True)
            return hashlib.md5(sorted_config.encode()).hexdigest()
        elif isinstance(config, str):
            return hashlib.md5(config.encode()).hexdigest()
        else:
            return hashlib.md5(str(config).encode()).hexdigest()
    
    async def _analyze_function_similarity(self) -> Dict[str, Any]:
        """Analyze similar functions that could be consolidated."""
        logger.info("Analyzing function similarity...")
        
        functions = await self._extract_all_functions()
        similar_groups = []
        
        # Compare functions pairwise
        for i, func1 in enumerate(functions):
            for j, func2 in enumerate(functions[i+1:], i+1):
                similarity = self._calculate_function_similarity(func1, func2)
                
                if similarity >= self.similarity_threshold:
                    # Check if already in a group
                    existing_group = None
                    for group in similar_groups:
                        if func1 in group or func2 in group:
                            existing_group = group
                            break
                    
                    if existing_group:
                        existing_group.add(func1)
                        existing_group.add(func2)
                    else:
                        similar_groups.append({func1, func2})
        
        # Create redundancy items for similar function groups
        for group in similar_groups:
            if len(group) >= 2:
                func_list = list(group)
                source_files = list(set(func['file'] for func in func_list))
                
                similarity_item = RedundancyItem(
                    type=RedundancyType.FUNCTION_SIMILARITY,
                    source_files=source_files,
                    similarity_score=self.similarity_threshold,
                    content_hash=hashlib.md5(str(group).encode()).hexdigest(),
                    content_preview=f"Similar functions: {[f['name'] for f in func_list]}",
                    size_bytes=sum(len(f['content']) for f in func_list),
                    recommended_action=DeduplicationAction.CONSOLIDATE,
                    estimated_savings={
                        "functions": len(func_list) - 1,
                        "lines_of_code": sum(f['content'].count('\n') for f in func_list) // 2
                    }
                )
                self.redundant_items.append(similarity_item)
        
        return {
            "functions_analyzed": len(functions),
            "similar_groups": len(similar_groups),
            "consolidation_opportunities": sum(len(group) - 1 for group in similar_groups)
        }
    
    async def _extract_all_functions(self) -> List[Dict[str, Any]]:
        """Extract all functions from Python files."""
        functions = []
        
        for file_path in self.root_path.glob("**/*.py"):
            if any(exc in str(file_path) for exc in self.exclude_patterns):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                lines = content.split('\n')
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        start_line = node.lineno - 1
                        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
                        
                        func_content = '\n'.join(lines[start_line:end_line])
                        
                        functions.append({
                            'name': node.name,
                            'file': str(file_path),
                            'content': func_content,
                            'normalized_content': self._normalize_function(func_content),
                            'args': [arg.arg for arg in node.args.args],
                            'line_start': start_line + 1,
                            'line_end': end_line
                        })
            
            except Exception as e:
                logger.warning(f"Failed to extract functions from {file_path}: {e}")
        
        return functions
    
    def _normalize_function(self, content: str) -> str:
        """Normalize function content for similarity comparison."""
        lines = content.split('\n')
        normalized = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Remove variable names and string literals for structural comparison
            import re
            line = re.sub(r'\b\w+\b', 'VAR', line)
            line = re.sub(r'"[^"]*"', '""', line)
            line = re.sub(r"'[^']*'", "''", line)
            line = re.sub(r'\d+', 'NUM', line)
            
            normalized.append(line)
        
        return '\n'.join(normalized)
    
    def _calculate_function_similarity(self, func1: Dict[str, Any], func2: Dict[str, Any]) -> float:
        """Calculate similarity between two functions."""
        if func1['file'] == func2['file']:
            return 0.0  # Don't compare functions in the same file
        
        # Compare normalized content
        content1 = func1['normalized_content']
        content2 = func2['normalized_content']
        
        # Use difflib for similarity calculation
        similarity = difflib.SequenceMatcher(None, content1, content2).ratio()
        
        # Bonus for similar argument patterns
        args1 = set(func1['args'])
        args2 = set(func2['args'])
        
        if args1 and args2:
            arg_similarity = len(args1.intersection(args2)) / len(args1.union(args2))
            similarity = (similarity * 0.8) + (arg_similarity * 0.2)
        
        return similarity
    
    async def _analyze_class_similarity(self) -> Dict[str, Any]:
        """Analyze similar classes that could be consolidated."""
        logger.info("Analyzing class similarity...")
        
        # Similar to function analysis but for classes
        classes = await self._extract_all_classes()
        similar_groups = []
        
        for i, class1 in enumerate(classes):
            for j, class2 in enumerate(classes[i+1:], i+1):
                similarity = self._calculate_class_similarity(class1, class2)
                
                if similarity >= self.similarity_threshold:
                    existing_group = None
                    for group in similar_groups:
                        if class1 in group or class2 in group:
                            existing_group = group
                            break
                    
                    if existing_group:
                        existing_group.add(class1)
                        existing_group.add(class2)
                    else:
                        similar_groups.append({class1, class2})
        
        # Create redundancy items
        for group in similar_groups:
            if len(group) >= 2:
                class_list = list(group)
                source_files = list(set(cls['file'] for cls in class_list))
                
                similarity_item = RedundancyItem(
                    type=RedundancyType.CLASS_SIMILARITY,
                    source_files=source_files,
                    similarity_score=self.similarity_threshold,
                    content_hash=hashlib.md5(str(group).encode()).hexdigest(),
                    content_preview=f"Similar classes: {[c['name'] for c in class_list]}",
                    size_bytes=sum(len(c['content']) for c in class_list),
                    recommended_action=DeduplicationAction.CONSOLIDATE,
                    estimated_savings={
                        "classes": len(class_list) - 1,
                        "lines_of_code": sum(c['content'].count('\n') for c in class_list) // 2
                    }
                )
                self.redundant_items.append(similarity_item)
        
        return {
            "classes_analyzed": len(classes),
            "similar_groups": len(similar_groups)
        }
    
    async def _extract_all_classes(self) -> List[Dict[str, Any]]:
        """Extract all classes from Python files."""
        classes = []
        
        for file_path in self.root_path.glob("**/*.py"):
            if any(exc in str(file_path) for exc in self.exclude_patterns):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                lines = content.split('\n')
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        start_line = node.lineno - 1
                        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 20
                        
                        class_content = '\n'.join(lines[start_line:end_line])
                        
                        # Extract method names
                        methods = []
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                methods.append(item.name)
                        
                        classes.append({
                            'name': node.name,
                            'file': str(file_path),
                            'content': class_content,
                            'normalized_content': self._normalize_class(class_content),
                            'methods': methods,
                            'base_classes': [base.id for base in node.bases if hasattr(base, 'id')],
                            'line_start': start_line + 1,
                            'line_end': end_line
                        })
            
            except Exception as e:
                logger.warning(f"Failed to extract classes from {file_path}: {e}")
        
        return classes
    
    def _normalize_class(self, content: str) -> str:
        """Normalize class content for similarity comparison."""
        # Similar to function normalization but for classes
        return self._normalize_function(content)
    
    def _calculate_class_similarity(self, class1: Dict[str, Any], class2: Dict[str, Any]) -> float:
        """Calculate similarity between two classes."""
        if class1['file'] == class2['file']:
            return 0.0
        
        # Compare normalized content
        content_similarity = difflib.SequenceMatcher(
            None, 
            class1['normalized_content'], 
            class2['normalized_content']
        ).ratio()
        
        # Compare method names
        methods1 = set(class1['methods'])
        methods2 = set(class2['methods'])
        
        if methods1 and methods2:
            method_similarity = len(methods1.intersection(methods2)) / len(methods1.union(methods2))
        else:
            method_similarity = 0.0
        
        # Weighted combination
        return (content_similarity * 0.7) + (method_similarity * 0.3)
    
    async def _analyze_import_redundancy(self) -> Dict[str, Any]:
        """Analyze redundant import statements."""
        logger.info("Analyzing import redundancy...")
        
        import_analysis = {}
        redundant_imports = []
        
        for file_path in self.root_path.glob("**/*.py"):
            if any(exc in str(file_path) for exc in self.exclude_patterns):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                imports = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            for alias in node.names:
                                imports.append(f"{node.module}.{alias.name}")
                
                import_analysis[str(file_path)] = imports
                
                # Check for unused imports (simplified)
                # In a real implementation, this would use AST analysis to check usage
                
            except Exception as e:
                logger.warning(f"Failed to analyze imports in {file_path}: {e}")
        
        return {
            "files_analyzed": len(import_analysis),
            "redundant_imports": len(redundant_imports)
        }
    
    async def _analyze_documentation_duplication(self) -> Dict[str, Any]:
        """Analyze documentation duplication."""
        logger.info("Analyzing documentation duplication...")
        
        doc_files = []
        for pattern in self.docs_patterns:
            doc_files.extend(self.root_path.glob(pattern))
        
        doc_contents = {}
        duplicates = []
        
        for file_path in doc_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create content hash
                content_hash = hashlib.md5(content.encode()).hexdigest()
                
                if content_hash in doc_contents and len(content) > self.min_content_size:
                    existing_file = doc_contents[content_hash]
                    
                    duplicate_item = RedundancyItem(
                        type=RedundancyType.DOCUMENTATION_DUPLICATION,
                        source_files=[existing_file, str(file_path)],
                        similarity_score=1.0,
                        content_hash=content_hash,
                        content_preview=content[:200] + "...",
                        size_bytes=len(content),
                        recommended_action=DeduplicationAction.CONSOLIDATE,
                        estimated_savings={
                            "doc_files": 1,
                            "maintenance_reduction": "low"
                        }
                    )
                    duplicates.append(duplicate_item)
                else:
                    doc_contents[content_hash] = str(file_path)
            
            except Exception as e:
                logger.warning(f"Failed to analyze documentation {file_path}: {e}")
        
        self.redundant_items.extend(duplicates)
        
        return {
            "files_analyzed": len(doc_files),
            "duplicates_found": len(duplicates)
        }
    
    async def _analyze_deployment_redundancy(self) -> Dict[str, Any]:
        """Analyze deployment configuration redundancy."""
        logger.info("Analyzing deployment redundancy...")
        
        deployment_files = []
        for pattern in self.docker_patterns:
            deployment_files.extend(self.root_path.glob(pattern))
        
        # Add other deployment files
        deployment_files.extend(self.root_path.glob("**/k8s/**/*.yaml"))
        deployment_files.extend(self.root_path.glob("**/kubernetes/**/*.yaml"))
        
        redundancies = []
        
        # Analyze for similar deployment configurations
        for file_path in deployment_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for patterns that indicate redundancy
                # This is a simplified analysis
                
            except Exception as e:
                logger.warning(f"Failed to analyze deployment {file_path}: {e}")
        
        return {
            "files_analyzed": len(deployment_files),
            "redundancies_found": len(redundancies)
        }
    
    async def _analyze_database_redundancy(self) -> Dict[str, Any]:
        """Analyze database schema and model redundancy."""
        logger.info("Analyzing database redundancy...")
        
        # Look for database models and schema files
        model_files = list(self.root_path.glob("**/models.py"))
        model_files.extend(self.root_path.glob("**/models/**/*.py"))
        
        redundancies = []
        
        # Analyze for duplicate models or similar table structures
        # This would require more sophisticated analysis
        
        return {
            "files_analyzed": len(model_files),
            "redundancies_found": len(redundancies)
        }
    
    async def _generate_deduplication_plans(self):
        """Generate actionable deduplication plans."""
        logger.info("Generating deduplication plans...")
        
        for item in self.redundant_items:
            plan = self._create_deduplication_plan(item)
            self.deduplication_plans.append(plan)
    
    def _create_deduplication_plan(self, item: RedundancyItem) -> DeduplicationPlan:
        """Create a deduplication plan for a redundancy item."""
        
        # Determine target location based on item type
        if item.type == RedundancyType.CODE_DUPLICATION:
            target_location = "src/xorb/shared/common.py"
        elif item.type == RedundancyType.CONFIG_DUPLICATION:
            target_location = "config/shared_config.json"
        elif item.type in [RedundancyType.FUNCTION_SIMILARITY, RedundancyType.CLASS_SIMILARITY]:
            target_location = f"src/xorb/shared/{item.type.value}_consolidated.py"
        else:
            target_location = "src/xorb/shared/"
        
        # Generate implementation steps
        steps = self._generate_implementation_steps(item)
        
        # Assess risk and effort
        risk_level = self._assess_risk_level(item)
        effort = self._estimate_effort(item)
        
        return DeduplicationPlan(
            redundancy_item=item,
            action=item.recommended_action,
            target_location=target_location,
            affected_files=item.source_files,
            implementation_steps=steps,
            risk_level=risk_level,
            estimated_effort=effort,
            validation_criteria=[
                "All functionality preserved",
                "No breaking changes introduced",
                "Tests pass successfully",
                "Performance maintained or improved"
            ]
        )
    
    def _generate_implementation_steps(self, item: RedundancyItem) -> List[str]:
        """Generate implementation steps for deduplication."""
        if item.recommended_action == DeduplicationAction.EXTRACT_COMMON:
            return [
                "Extract common code to shared module",
                "Update imports in source files",
                "Remove duplicate code from source files",
                "Add tests for shared module",
                "Update documentation"
            ]
        elif item.recommended_action == DeduplicationAction.CONSOLIDATE:
            return [
                "Analyze differences between similar items",
                "Design unified interface",
                "Implement consolidated version",
                "Migrate all usages to consolidated version",
                "Remove obsolete implementations"
            ]
        elif item.recommended_action == DeduplicationAction.ELIMINATE:
            return [
                "Verify item is truly redundant",
                "Check for any dependencies",
                "Remove redundant item",
                "Update references if necessary",
                "Validate no functionality lost"
            ]
        else:
            return [
                "Analyze redundancy pattern",
                "Design solution approach",
                "Implement deduplication",
                "Test thoroughly",
                "Deploy changes"
            ]
    
    def _assess_risk_level(self, item: RedundancyItem) -> str:
        """Assess risk level of deduplication."""
        if len(item.source_files) > 5:
            return "high"
        elif item.type in [RedundancyType.CODE_DUPLICATION, RedundancyType.FUNCTION_SIMILARITY]:
            return "medium"
        else:
            return "low"
    
    def _estimate_effort(self, item: RedundancyItem) -> str:
        """Estimate effort required for deduplication."""
        if item.size_bytes > 10000:  # Large items
            return "high"
        elif item.size_bytes > 1000:
            return "medium"
        else:
            return "low"
    
    async def _calculate_impact_metrics(self) -> Dict[str, Any]:
        """Calculate the impact of deduplication."""
        total_lines_saved = sum(
            item.estimated_savings.get("lines_of_code", 0) 
            for item in self.redundant_items
        )
        
        total_files_eliminated = sum(
            item.estimated_savings.get("config_files", 0) + 
            item.estimated_savings.get("doc_files", 0)
            for item in self.redundant_items
        )
        
        maintenance_reduction_items = [
            item for item in self.redundant_items
            if item.estimated_savings.get("maintenance_reduction") == "high"
        ]
        
        return {
            "total_redundant_items": len(self.redundant_items),
            "estimated_lines_saved": total_lines_saved,
            "estimated_files_eliminated": total_files_eliminated,
            "maintenance_reduction_opportunities": len(maintenance_reduction_items),
            "deduplication_plans_generated": len(self.deduplication_plans),
            "estimated_storage_savings_bytes": sum(item.size_bytes for item in self.redundant_items),
            "complexity_reduction_score": self._calculate_complexity_reduction()
        }
    
    def _calculate_complexity_reduction(self) -> float:
        """Calculate complexity reduction score."""
        # Simplified complexity reduction calculation
        base_score = len(self.redundant_items) * 0.1
        
        # Weight by type of redundancy
        type_weights = {
            RedundancyType.CODE_DUPLICATION: 0.3,
            RedundancyType.FUNCTION_SIMILARITY: 0.25,
            RedundancyType.CLASS_SIMILARITY: 0.25,
            RedundancyType.CONFIG_DUPLICATION: 0.1,
            RedundancyType.DOCUMENTATION_DUPLICATION: 0.05
        }
        
        weighted_score = sum(
            type_weights.get(item.type, 0.1) 
            for item in self.redundant_items
        )
        
        return min(10.0, base_score + weighted_score)

# Global deduplication engine instance
deduplication_engine: Optional[XORBDeduplicationEngine] = None

async def initialize_deduplication_engine(root_path: str = "/root/Xorb") -> XORBDeduplicationEngine:
    """Initialize the global deduplication engine."""
    global deduplication_engine
    deduplication_engine = XORBDeduplicationEngine(root_path)
    await deduplication_engine.initialize()
    return deduplication_engine

async def get_deduplication_engine() -> Optional[XORBDeduplicationEngine]:
    """Get the global deduplication engine."""
    return deduplication_engine