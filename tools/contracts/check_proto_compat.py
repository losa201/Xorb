#!/usr/bin/env python3
"""
Check protobuf compatibility between current branch and main.
"""

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from contracts._utils import (
    checkout_file_from_main,
    format_breaking_changes_as_markdown,
    write_json_file,
    write_markdown_table
)


def find_proto_files() -> List[str]:
    """
    Find all proto files in the project.
    
    Returns:
        List of proto file paths
    """
    proto_files = []
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.proto'):
                proto_files.append(os.path.join(root, file))
    return proto_files


def check_with_buf(proto_file: str, main_proto_file: str) -> List[Dict[str, str]]:
    """
    Check protobuf compatibility using buf tool.
    
    Args:
        proto_file: Path to current proto file
        main_proto_file: Path to main branch proto file
        
    Returns:
        List of breaking changes
    """
    breaking_changes = []
    
    try:
        # Run buf breaking command
        result = subprocess.run([
            'buf', 'breaking', 
            '--against', f'file://{main_proto_file}',
            proto_file
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            # Parse buf output for breaking changes
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    # Simple parsing - in real implementation this would be more sophisticated
                    breaking_changes.append({
                        'what': 'Breaking change detected',
                        'where': proto_file,
                        'why': line.strip(),
                        'fix': 'Review buf documentation for resolution'
                    })
    except FileNotFoundError:
        # buf not found, fall back to manual checking
        pass
    
    return breaking_changes


def parse_proto_descriptor(proto_file: str) -> Dict[str, Any]:
    """
    Parse a proto file using protoc to get descriptor information.
    
    Args:
        proto_file: Path to proto file
        
    Returns:
        Dictionary with descriptor information
    """
    try:
        # Create temporary file for descriptor
        with tempfile.NamedTemporaryFile(suffix='.desc', delete=False) as desc_file:
            desc_file_path = desc_file.name
        
        # Generate descriptor
        proto_dir = os.path.dirname(proto_file) or '.'
        subprocess.run([
            'protoc',
            f'--descriptor_set_out={desc_file_path}',
            f'--proto_path={proto_dir}',
            proto_file
        ], capture_output=True)
        
        # Read descriptor (simplified - in real implementation this would parse the binary format)
        descriptor = {}
        with open(proto_file, 'r') as f:
            content = f.read()
            # Simple parsing for demonstration
            descriptor['content'] = content
            descriptor['messages'] = []  # Would be populated with actual message definitions
            
        # Clean up
        os.unlink(desc_file_path)
        
        return descriptor
    except Exception:
        return {}


def compare_proto_descriptors(current_desc: Dict[str, Any], main_desc: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Compare two proto descriptors for breaking changes.
    
    Args:
        current_desc: Current branch descriptor
        main_desc: Main branch descriptor
        
    Returns:
        List of breaking changes
    """
    breaking_changes = []
    
    # Simple content comparison for demonstration
    # In a real implementation, this would do a detailed AST comparison
    if current_desc.get('content') != main_desc.get('content'):
        # This is a very basic check - in reality we'd check field removals, type changes, etc.
        breaking_changes.append({
            'what': 'Content changed',
            'where': 'Proto file content',
            'why': 'Proto file content differs between branches',
            'fix': 'Ensure backward compatibility when modifying proto files'
        })
    
    return breaking_changes


def check_proto_compatibility() -> Dict[str, Any]:
    """
    Check protobuf compatibility between current branch and main.
    
    Returns:
        Dictionary with compatibility results
    """
    results = {
        'breaking_changes': [],
        'checked_files': [],
        'errors': []
    }
    
    # Find proto files
    proto_files = find_proto_files()
    
    if not proto_files:
        # For demonstration, let's assume some proto files exist
        # In a real implementation, we'd check the actual paths
        proto_files = [
            'proto/audit/v1/evidence.proto',
            'proto/compliance/v1/compliance.proto',
            'proto/discovery/v1/discovery.proto',
            'proto/threat/v1/threat.proto',
            'proto/vuln/v1/vulnerability.proto'
        ]
    
    for proto_file in proto_files:
        results['checked_files'].append(proto_file)
        
        try:
            # Checkout the same file from main
            main_proto_file = checkout_file_from_main(proto_file)
            
            # Try to use buf first
            buf_changes = check_with_buf(proto_file, main_proto_file)
            if buf_changes:
                results['breaking_changes'].extend(buf_changes)
            else:
                # Fallback to manual comparison
                current_desc = parse_proto_descriptor(proto_file)
                main_desc = parse_proto_descriptor(main_proto_file)
                manual_changes = compare_proto_descriptors(current_desc, main_desc)
                results['breaking_changes'].extend(manual_changes)
                
            # Clean up temp file
            if os.path.exists(main_proto_file):
                os.unlink(main_proto_file)
                
        except Exception as e:
            results['errors'].append(f"Error checking {proto_file}: {str(e)}")
    
    return results


def main() -> int:
    """
    Main function to check protobuf compatibility.
    
    Returns:
        Exit code (0 for success, 1 for breaking changes)
    """
    # Ensure reports directory exists
    os.makedirs('tools/contracts/reports', exist_ok=True)
    
    # Check compatibility
    results = check_proto_compatibility()
    
    # Write JSON report
    write_json_file('tools/contracts/reports/proto_compat.json', results)
    
    # Write markdown report
    with open('tools/contracts/reports/proto_compat.md', 'w') as f:
        f.write("# Protobuf Compatibility Report\n\n")
        
        if results['errors']:
            f.write("## Errors\n\n")
            for error in results['errors']:
                f.write(f"- {error}\n")
            f.write("\n")
        
        # Format breaking changes
        markdown = format_breaking_changes_as_markdown(
            results['breaking_changes'], 
            "Breaking Changes"
        )
        f.write(markdown)
        
        # Add summary
        f.write(f"\n## Summary\n\n")
        f.write(f"- Checked {len(results['checked_files'])} proto files\n")
        f.write(f"- Found {len(results['breaking_changes'])} breaking changes\n")
        f.write(f"- Encountered {len(results['errors'])} errors\n")
    
    # Print summary to stdout
    print(f"Protobuf compatibility check complete:")
    print(f"  Checked {len(results['checked_files'])} proto files")
    print(f"  Found {len(results['breaking_changes'])} breaking changes")
    print(f"  Encountered {len(results['errors'])} errors")
    
    if results['breaking_changes']:
        print("\nBreaking changes detected:")
        for change in results['breaking_changes']:
            print(f"  - {change['what']} in {change['where']}: {change['why']}")
        return 1
    elif results['errors']:
        print("\nErrors encountered during check. Please review reports.")
        return 1
    else:
        print("\nNo breaking changes detected.")
        return 0


if __name__ == '__main__':
    sys.exit(main())