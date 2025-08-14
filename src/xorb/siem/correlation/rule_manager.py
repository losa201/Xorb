"""
Correlation rule management system
Handles rule CRUD operations, validation, and persistence
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from .correlation_engine import CorrelationRule, CorrelationRuleType, CorrelationSeverity


class RuleValidationError(Exception):
    """Exception raised for rule validation errors"""
    pass


class RuleManager:
    """Manages correlation rules with persistence and validation"""

    def __init__(self, rules_directory: str = "data/correlation_rules"):
        self.rules_directory = Path(rules_directory)
        self.rules_directory.mkdir(parents=True, exist_ok=True)
        self.rules: Dict[str, CorrelationRule] = {}

        # Load existing rules
        self.load_rules_from_disk()

    def create_rule(self, rule_data: Dict[str, Any]) -> CorrelationRule:
        """Create new correlation rule"""
        # Validate rule data
        self._validate_rule_data(rule_data)

        # Convert string enums to actual enums
        rule_data['rule_type'] = CorrelationRuleType(rule_data['rule_type'])
        rule_data['severity'] = CorrelationSeverity(rule_data['severity'])

        # Create rule object
        rule = CorrelationRule(**rule_data)

        # Store in memory and persist
        self.rules[rule.rule_id] = rule
        self._persist_rule(rule)

        return rule

    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> CorrelationRule:
        """Update existing correlation rule"""
        if rule_id not in self.rules:
            raise ValueError(f"Rule {rule_id} not found")

        rule = self.rules[rule_id]

        # Apply updates
        for field, value in updates.items():
            if hasattr(rule, field):
                if field == 'rule_type' and isinstance(value, str):
                    value = CorrelationRuleType(value)
                elif field == 'severity' and isinstance(value, str):
                    value = CorrelationSeverity(value)

                setattr(rule, field, value)

        rule.last_modified = datetime.utcnow()

        # Validate updated rule
        self._validate_rule(rule)

        # Persist changes
        self._persist_rule(rule)

        return rule

    def delete_rule(self, rule_id: str) -> bool:
        """Delete correlation rule"""
        if rule_id not in self.rules:
            return False

        # Remove from memory
        del self.rules[rule_id]

        # Remove from disk
        rule_file = self.rules_directory / f"{rule_id}.json"
        if rule_file.exists():
            rule_file.unlink()

        return True

    def get_rule(self, rule_id: str) -> Optional[CorrelationRule]:
        """Get specific correlation rule"""
        return self.rules.get(rule_id)

    def get_all_rules(self) -> List[CorrelationRule]:
        """Get all correlation rules"""
        return list(self.rules.values())

    def get_enabled_rules(self) -> List[CorrelationRule]:
        """Get only enabled correlation rules"""
        return [rule for rule in self.rules.values() if rule.enabled]

    def get_rules_by_type(self, rule_type: CorrelationRuleType) -> List[CorrelationRule]:
        """Get rules by type"""
        return [rule for rule in self.rules.values() if rule.rule_type == rule_type]

    def get_rules_by_severity(self, severity: CorrelationSeverity) -> List[CorrelationRule]:
        """Get rules by severity"""
        return [rule for rule in self.rules.values() if rule.severity == severity]

    def enable_rule(self, rule_id: str) -> bool:
        """Enable correlation rule"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            self.rules[rule_id].last_modified = datetime.utcnow()
            self._persist_rule(self.rules[rule_id])
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """Disable correlation rule"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            self.rules[rule_id].last_modified = datetime.utcnow()
            self._persist_rule(self.rules[rule_id])
            return True
        return False

    def import_rules(self, rules_data: List[Dict[str, Any]]) -> List[str]:
        """Import multiple rules from data"""
        imported_rules = []

        for rule_data in rules_data:
            try:
                rule = self.create_rule(rule_data)
                imported_rules.append(rule.rule_id)
            except Exception as e:
                print(f"Failed to import rule {rule_data.get('rule_id', 'unknown')}: {e}")

        return imported_rules

    def export_rules(self, rule_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Export rules as JSON data"""
        if rule_ids is None:
            rules_to_export = self.rules.values()
        else:
            rules_to_export = [self.rules[rid] for rid in rule_ids if rid in self.rules]

        exported_data = []
        for rule in rules_to_export:
            rule_dict = self._rule_to_dict(rule)
            exported_data.append(rule_dict)

        return exported_data

    def validate_rule_syntax(self, rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate rule syntax and return validation results"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        try:
            self._validate_rule_data(rule_data)
        except RuleValidationError as e:
            validation_result["valid"] = False
            validation_result["errors"].append(str(e))

        return validation_result

    def load_rules_from_disk(self):
        """Load all rules from disk"""
        if not self.rules_directory.exists():
            return

        for rule_file in self.rules_directory.glob("*.json"):
            try:
                with open(rule_file, 'r') as f:
                    rule_data = json.load(f)

                # Convert enum strings back to enums
                rule_data['rule_type'] = CorrelationRuleType(rule_data['rule_type'])
                rule_data['severity'] = CorrelationSeverity(rule_data['severity'])

                # Convert datetime strings back to datetime objects
                if 'created_at' in rule_data:
                    rule_data['created_at'] = datetime.fromisoformat(rule_data['created_at'])
                if 'last_modified' in rule_data:
                    rule_data['last_modified'] = datetime.fromisoformat(rule_data['last_modified'])

                rule = CorrelationRule(**rule_data)
                self.rules[rule.rule_id] = rule

            except Exception as e:
                print(f"Failed to load rule from {rule_file}: {e}")

    def _persist_rule(self, rule: CorrelationRule):
        """Persist rule to disk"""
        rule_file = self.rules_directory / f"{rule.rule_id}.json"
        rule_data = self._rule_to_dict(rule)

        with open(rule_file, 'w') as f:
            json.dump(rule_data, f, indent=2, default=str)

    def _rule_to_dict(self, rule: CorrelationRule) -> Dict[str, Any]:
        """Convert rule to dictionary for serialization"""
        return {
            "rule_id": rule.rule_id,
            "name": rule.name,
            "description": rule.description,
            "rule_type": rule.rule_type.value,
            "severity": rule.severity.value,
            "enabled": rule.enabled,
            "event_filters": rule.event_filters,
            "time_window": rule.time_window.total_seconds(),
            "threshold_count": rule.threshold_count,
            "group_by_fields": rule.group_by_fields,
            "sequence_pattern": rule.sequence_pattern,
            "statistical_baseline": rule.statistical_baseline,
            "geographic_constraints": rule.geographic_constraints,
            "created_at": rule.created_at.isoformat(),
            "last_modified": rule.last_modified.isoformat(),
            "author": rule.author,
            "tags": rule.tags
        }

    def _validate_rule_data(self, rule_data: Dict[str, Any]):
        """Validate rule data before creation"""
        required_fields = ['rule_id', 'name', 'description', 'rule_type', 'severity']

        for field in required_fields:
            if field not in rule_data:
                raise RuleValidationError(f"Missing required field: {field}")

        # Validate rule_id uniqueness
        if rule_data['rule_id'] in self.rules:
            raise RuleValidationError(f"Rule ID {rule_data['rule_id']} already exists")

        # Validate enums
        try:
            CorrelationRuleType(rule_data['rule_type'])
        except ValueError:
            raise RuleValidationError(f"Invalid rule_type: {rule_data['rule_type']}")

        try:
            CorrelationSeverity(rule_data['severity'])
        except ValueError:
            raise RuleValidationError(f"Invalid severity: {rule_data['severity']}")

        # Validate rule-specific requirements
        rule_type = CorrelationRuleType(rule_data['rule_type'])

        if rule_type == CorrelationRuleType.FREQUENCY:
            if 'threshold_count' not in rule_data:
                raise RuleValidationError("Frequency rules require threshold_count")

        elif rule_type == CorrelationRuleType.SEQUENCE:
            if 'sequence_pattern' not in rule_data or not rule_data['sequence_pattern']:
                raise RuleValidationError("Sequence rules require sequence_pattern")

        elif rule_type == CorrelationRuleType.STATISTICAL:
            if 'statistical_baseline' not in rule_data:
                raise RuleValidationError("Statistical rules require statistical_baseline")

    def _validate_rule(self, rule: CorrelationRule):
        """Validate rule object"""
        if not rule.rule_id:
            raise RuleValidationError("Rule ID cannot be empty")

        if not rule.name:
            raise RuleValidationError("Rule name cannot be empty")

        if rule.threshold_count < 1:
            raise RuleValidationError("Threshold count must be at least 1")

    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about managed rules"""
        total_rules = len(self.rules)
        enabled_rules = len([r for r in self.rules.values() if r.enabled])

        rules_by_type = {}
        for rule_type in CorrelationRuleType:
            rules_by_type[rule_type.value] = len(self.get_rules_by_type(rule_type))

        rules_by_severity = {}
        for severity in CorrelationSeverity:
            rules_by_severity[severity.value] = len(self.get_rules_by_severity(severity))

        return {
            "total_rules": total_rules,
            "enabled_rules": enabled_rules,
            "disabled_rules": total_rules - enabled_rules,
            "rules_by_type": rules_by_type,
            "rules_by_severity": rules_by_severity
        }
