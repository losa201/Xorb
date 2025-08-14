#!/usr/bin/env python3
"""
Check OpenAPI compatibility between current branch and main.
"""

import os
import sys
from typing import Any, Dict, List

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from contracts._utils import (
    checkout_file_from_main,
    format_breaking_changes_as_markdown,
    load_yaml_file,
    write_json_file
)


# OpenAPI file path
OPENAPI_FILE = 'docs/api/xorb-openapi-spec.yaml'


def compare_openapi_specs(current_spec: Dict[str, Any], main_spec: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Compare two OpenAPI specifications for breaking changes.

    Args:
        current_spec: Current branch OpenAPI spec
        main_spec: Main branch OpenAPI spec

    Returns:
        List of breaking changes
    """
    breaking_changes = []

    # Compare paths
    current_paths = current_spec.get('paths', {})
    main_paths = main_spec.get('paths', {})

    # Check for removed paths
    for path, methods in main_paths.items():
        if path not in current_paths:
            breaking_changes.append({
                'what': 'Removed endpoint',
                'where': f'PATH: {path}',
                'why': 'Endpoint removed from API',
                'fix': 'Deprecate endpoint instead of removing or provide migration path'
            })
            continue

        # Check for removed methods
        current_methods = current_paths[path]
        for method in methods:
            if method not in current_methods:
                breaking_changes.append({
                    'what': f'Removed method {method.upper()}',
                    'where': f'PATH: {path} METHOD: {method}',
                    'why': 'HTTP method removed from endpoint',
                    'fix': 'Deprecate method instead of removing or provide alternative'
                })

    # Check for removed or narrowed response schemas
    for path, methods in current_paths.items():
        main_methods = main_paths.get(path, {})
        for method, details in methods.items():
            main_details = main_methods.get(method, {})

            # Check responses
            current_responses = details.get('responses', {})
            main_responses = main_details.get('responses', {})

            # Check for removed response codes
            for code, response in main_responses.items():
                if code not in current_responses:
                    breaking_changes.append({
                        'what': f'Removed response code {code}',
                        'where': f'PATH: {path} METHOD: {method}',
                        'why': 'Response code removed from endpoint',
                        'fix': 'Maintain backward compatibility by keeping all response codes'
                    })

            # Check for required fields added to request bodies
            current_request_body = details.get('requestBody', {})
            main_request_body = main_details.get('requestBody', {})

            if current_request_body and main_request_body:
                current_required = get_required_fields(current_request_body)
                main_required = get_required_fields(main_request_body)

                # Check for newly required fields
                for field in current_required:
                    if field not in main_required:
                        breaking_changes.append({
                            'what': f'Added required field: {field}',
                            'where': f'PATH: {path} METHOD: {method}',
                            'why': 'Newly required field breaks existing clients',
                            'fix': 'Make field optional or provide default value'
                        })

    # Check for enum narrowing
    current_schemas = current_spec.get('components', {}).get('schemas', {})
    main_schemas = main_spec.get('components', {}).get('schemas', {})

    for schema_name, current_schema in current_schemas.items():
        main_schema = main_schemas.get(schema_name)
        if main_schema:
            enum_changes = compare_schema_enums(current_schema, main_schema)
            for change in enum_changes:
                change['where'] = f'SCHEMA: {schema_name} {change["where"]}'
                breaking_changes.append(change)

    return breaking_changes


def get_required_fields(request_body: Dict[str, Any]) -> List[str]:
    """
    Extract required fields from a request body.

    Args:
        request_body: Request body specification

    Returns:
        List of required field names
    """
    required_fields = []

    content = request_body.get('content', {})
    for media_type, media_details in content.items():
        schema = media_details.get('schema', {})
        if '$ref' in schema:
            # For simplicity, we're not resolving refs in this example
            continue
        required_fields.extend(schema.get('required', []))

    return required_fields


def compare_schema_enums(current_schema: Dict[str, Any], main_schema: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Compare enums in two schemas for narrowing.

    Args:
        current_schema: Current schema
        main_schema: Main schema

    Returns:
        List of enum narrowing changes
    """
    changes = []

    # Check enum properties
    current_props = current_schema.get('properties', {})
    main_props = main_schema.get('properties', {})

    for prop_name, current_prop in current_props.items():
        main_prop = main_props.get(prop_name)
        if main_prop and 'enum' in main_prop and 'enum' in current_prop:
            main_enum = set(main_prop['enum'])
            current_enum = set(current_prop['enum'])

            # Check if enum was narrowed (values removed)
            removed_values = main_enum - current_enum
            if removed_values:
                changes.append({
                    'what': f'Enum narrowed: removed values {list(removed_values)}',
                    'where': f'PROPERTY: {prop_name}',
                    'why': 'Removing enum values breaks existing clients',
                    'fix': 'Add new values but do not remove existing ones'
                })

    return changes


def check_openapi_compatibility() -> Dict[str, Any]:
    """
    Check OpenAPI compatibility between current branch and main.

    Returns:
        Dictionary with compatibility results
    """
    results = {
        'breaking_changes': [],
        'checked_files': [OPENAPI_FILE],
        'errors': []
    }

    try:
        # Load current OpenAPI spec
        if not os.path.exists(OPENAPI_FILE):
            results['errors'].append(f"OpenAPI spec not found: {OPENAPI_FILE}")
            return results

        current_spec = load_yaml_file(OPENAPI_FILE)

        # Checkout the same file from main
        main_openapi_file = checkout_file_from_main(OPENAPI_FILE)

        # Load main OpenAPI spec
        main_spec = load_yaml_file(main_openapi_file)

        # Compare specs
        breaking_changes = compare_openapi_specs(current_spec, main_spec)
        results['breaking_changes'].extend(breaking_changes)

        # Clean up temp file
        if os.path.exists(main_openapi_file):
            os.unlink(main_openapi_file)

    except Exception as e:
        results['errors'].append(f"Error checking OpenAPI compatibility: {str(e)}")

    return results


def main() -> int:
    """
    Main function to check OpenAPI compatibility.

    Returns:
        Exit code (0 for success, 1 for breaking changes)
    """
    # Ensure reports directory exists
    os.makedirs('tools/contracts/reports', exist_ok=True)

    # Check compatibility
    results = check_openapi_compatibility()

    # Write JSON report
    write_json_file('tools/contracts/reports/openapi_compat.json', results)

    # Write markdown report
    with open('tools/contracts/reports/openapi_compat.md', 'w') as f:
        f.write("# OpenAPI Compatibility Report\n\n")

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
        f.write(f"- Checked {len(results['checked_files'])} OpenAPI files\n")
        f.write(f"- Found {len(results['breaking_changes'])} breaking changes\n")
        f.write(f"- Encountered {len(results['errors'])} errors\n")

    # Print summary to stdout
    print(f"OpenAPI compatibility check complete:")
    print(f"  Checked {len(results['checked_files'])} OpenAPI files")
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
