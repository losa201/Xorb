#  Targets Configuration

The `targets.json` file defines the target assets to be analyzed and processed by the XORB system. This configuration is used across multiple modules including security analysis, service fusion, and monitoring.

##  Configuration Structure

```json
[
  {
    "target_type": "domain",
    "value": "example.com",
    "scope": "external",
    "metadata": {
      "priority": "high",
      "scan_type": "discovery"
    }
  }
]
```

##  Field Definitions

| Field | Type | Description | Example Values |
|-------|------|-------------|----------------|
| `target_type` | string | Type of target asset | `domain`, `ip`, `service`, `api` |
| `value` | string | Target identifier | `example.com`, `192.168.1.0/24` |
| `scope` | string | Target scope classification | `external`, `internal`, `partner` |
| `metadata.priority` | string | Operational priority | `high`, `medium`, `low` |
| `metadata.scan_type` | string | Type of analysis required | `discovery`, `deep`, `compliance` |

##  Usage Notes
- This configuration drives the scope of all security and optimization operations
- Targets should be reviewed and updated regularly to reflect current assets
- Priority and scan type metadata informs resource allocation decisions
- For large-scale deployments, consider splitting into multiple configuration files

##  Best Practices
1. Maintain version control for target configurations
2. Use consistent naming conventions for target values
3. Regularly audit scope classifications
4. Align priority settings with business criticality

This sample configuration can be extended to include multiple targets with varying characteristics.