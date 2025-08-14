#!/usr/bin/env python3
"""
Shared utilities for contract compatibility checking.
"""

import json
import os
import subprocess
import tempfile
import yaml
from typing import Any, Dict, List, Tuple


def checkout_file_from_main(file_path: str) -> str:
    """
    Checkout a file from the main branch to a temporary location.
    
    Args:
        file_path: Path to the file relative to repository root
        
    Returns:
        Path to the temporary file containing the main branch version
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
    try:
        # Get the file content from main branch
        result = subprocess.run(
            ['git', 'show', f'origin/main:{file_path}'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode == 0:
            temp_file.write(result.stdout)
            temp_file.flush()
            return temp_file.name
        else:
            # If failed to get from origin/main, try main
            result = subprocess.run(
                ['git', 'show', f'main:{file_path}'],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            if result.returncode == 0:
                temp_file.write(result.stdout)
                temp_file.flush()
                return temp_file.name
            else:
                # If both failed, return empty temp file
                return temp_file.name
    finally:
        temp_file.close()


def load_yaml_file(file_path: str) -> Dict[str, Any]:
    """
    Load a YAML file into a dictionary.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Dictionary representation of the YAML file
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file into a dictionary.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary representation of the JSON file
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def write_json_file(file_path: str, data: Dict[str, Any]) -> None:
    """
    Write data to a JSON file.
    
    Args:
        file_path: Path to the JSON file
        data: Data to write
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def write_markdown_table(file_path: str, headers: List[str], rows: List[List[str]]) -> None:
    """
    Write a markdown table to a file.
    
    Args:
        file_path: Path to the markdown file
        headers: Table headers
        rows: Table rows
    """
    with open(file_path, 'w') as f:
        # Write headers
        f.write('| ' + ' | '.join(headers) + ' |\n')
        # Write separator
        f.write('| ' + ' | '.join(['---'] * len(headers)) + ' |\n')
        # Write rows
        for row in rows:
            f.write('| ' + ' | '.join(row) + ' |\n')


def format_breaking_changes_as_markdown(changes: List[Dict[str, Any]], title: str) -> str:
    """
    Format breaking changes as a markdown table.
    
    Args:
        changes: List of breaking changes
        title: Title for the section
        
    Returns:
        Markdown formatted string
    """
    if not changes:
        return f"## {title}\n\nNo breaking changes detected.\n"
    
    markdown = f"## {title}\n\n"
    markdown += "| What | Where | Why | Suggested Fix |\n"
    markdown += "|------|-------|-----|---------------|\n"
    
    for change in changes:
        what = change.get('what', '')
        where = change.get('where', '')
        why = change.get('why', '')
        fix = change.get('fix', '')
        markdown += f"| {what} | {where} | {why} | {fix} |\n"
    
    return markdown


def git_diff_files(file1: str, file2: str) -> bool:
    """
    Check if two files are different using git diff.
    
    Args:
        file1: Path to first file
        file2: Path to second file
        
    Returns:
        True if files are different, False otherwise
    """
    try:
        result = subprocess.run(
            ['git', 'diff', '--no-index', '--quiet', file1, file2],
            capture_output=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        return result.returncode != 0
    except Exception:
        # If git diff fails, fallback to file comparison
        try:
            with open(file1, 'r') as f1, open(file2, 'r') as f2:
                return f1.read() != f2.read()
        except Exception:
            return True