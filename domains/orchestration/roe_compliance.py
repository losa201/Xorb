#!/usr/bin/env python3

import re
import ipaddress
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import urlparse
import asyncio
import aiohttp


@dataclass
class RoERule:
    rule_id: str
    rule_type: str  # "allow", "deny", "require_approval"
    pattern: str
    description: str
    severity: str = "medium"
    enabled: bool = True


class RoEValidator:
    def __init__(self):
        self.rules: List[RoERule] = []
        self.logger = logging.getLogger(__name__)
        self._initialize_default_rules()

    def _initialize_default_rules(self):
        default_rules = [
            RoERule(
                rule_id="deny_internal_networks",
                rule_type="deny",
                pattern="^(10\\.|172\\.(1[6-9]|2[0-9]|3[0-1])\\.|192\\.168\\.)",
                description="Deny access to internal/private networks",
                severity="critical"
            ),
            RoERule(
                rule_id="deny_localhost",
                rule_type="deny", 
                pattern="^(127\\.|localhost|::1)",
                description="Deny access to localhost/loopback",
                severity="critical"
            ),
            RoERule(
                rule_id="deny_government_domains",
                rule_type="deny",
                pattern="\\.(gov|mil)$",
                description="Deny access to government domains",
                severity="critical"
            ),
            RoERule(
                rule_id="require_https_financial",
                rule_type="require_approval",
                pattern="\\.(bank|finance|financial|payment)\\.",
                description="Require approval for financial services",
                severity="high"
            ),
            RoERule(
                rule_id="respect_robots_txt",
                rule_type="require_approval",
                pattern=".*",
                description="Check robots.txt before proceeding",
                severity="medium"
            ),
            RoERule(
                rule_id="deny_educational_institutions",
                rule_type="deny",
                pattern="\\.(edu|ac\\.))",
                description="Deny access to educational institutions",
                severity="high"
            )
        ]
        
        self.rules.extend(default_rules)
        self.logger.info(f"Initialized {len(default_rules)} default RoE rules")

    async def validate_target(self, target) -> bool:
        hostname = target.hostname
        ip_address = target.ip_address
        
        validation_results = {
            "hostname_valid": await self._validate_hostname(hostname),
            "ip_valid": await self._validate_ip_address(ip_address) if ip_address else True,
            "robots_txt_compliant": await self._check_robots_txt(hostname),
            "scope_valid": target.scope == "in-scope"
        }
        
        is_valid = all(validation_results.values())
        
        if not is_valid:
            failed_checks = [k for k, v in validation_results.items() if not v]
            self.logger.warning(f"Target {hostname} failed RoE validation: {failed_checks}")
        
        return is_valid

    async def _validate_hostname(self, hostname: str) -> bool:
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            if re.search(rule.pattern, hostname, re.IGNORECASE):
                if rule.rule_type == "deny":
                    self.logger.warning(f"Hostname {hostname} denied by rule {rule.rule_id}: {rule.description}")
                    return False
                elif rule.rule_type == "require_approval":
                    approval_granted = await self._request_approval(hostname, rule)
                    if not approval_granted:
                        self.logger.warning(f"Hostname {hostname} requires approval (rule {rule.rule_id})")
                        return False
        
        return True

    async def _validate_ip_address(self, ip_address: str) -> bool:
        try:
            ip = ipaddress.ip_address(ip_address)
            
            if ip.is_private:
                self.logger.warning(f"IP address {ip_address} is private/internal")
                return False
            
            if ip.is_loopback:
                self.logger.warning(f"IP address {ip_address} is loopback")
                return False
            
            if ip.is_multicast or ip.is_reserved:
                self.logger.warning(f"IP address {ip_address} is multicast/reserved")
                return False
            
        except ValueError:
            self.logger.error(f"Invalid IP address format: {ip_address}")
            return False
        
        return True

    async def _check_robots_txt(self, hostname: str) -> bool:
        try:
            robots_url = f"https://{hostname}/robots.txt"
            timeout = aiohttp.ClientTimeout(total=10)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(robots_url) as response:
                    if response.status == 200:
                        robots_content = await response.text()
                        return self._parse_robots_txt(robots_content, hostname)
                    else:
                        self.logger.debug(f"No robots.txt found for {hostname} (status: {response.status})")
                        return True
                        
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout checking robots.txt for {hostname}")
            return True
        except Exception as e:
            self.logger.debug(f"Error checking robots.txt for {hostname}: {e}")
            return True

    def _parse_robots_txt(self, content: str, hostname: str) -> bool:
        user_agent_section = False
        disallowed_paths = []
        
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if line.lower().startswith('user-agent:'):
                agent = line.split(':', 1)[1].strip()
                user_agent_section = agent == '*' or 'xorb' in agent.lower()
            elif user_agent_section and line.lower().startswith('disallow:'):
                path = line.split(':', 1)[1].strip()
                if path:
                    disallowed_paths.append(path)
        
        if disallowed_paths and '/' in disallowed_paths:
            self.logger.warning(f"robots.txt for {hostname} disallows all crawling")
            return False
        
        self.logger.debug(f"robots.txt for {hostname} allows crawling with restrictions: {disallowed_paths}")
        return True

    async def _request_approval(self, target: str, rule: RoERule) -> bool:
        self.logger.info(f"Approval required for target {target} (rule: {rule.description})")
        
        approval_request = {
            "target": target,
            "rule_id": rule.rule_id,
            "rule_description": rule.description,
            "severity": rule.severity,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        return await self._simulate_approval_process(approval_request)

    async def _simulate_approval_process(self, approval_request: Dict[str, Any]) -> bool:
        if approval_request["severity"] in ["low", "medium"]:
            self.logger.info(f"Auto-approving {approval_request['severity']} severity request for {approval_request['target']}")
            return True
        else:
            self.logger.warning(f"Manual approval required for {approval_request['severity']} severity request for {approval_request['target']}")
            return False

    def add_rule(self, rule: RoERule):
        self.rules.append(rule)
        self.logger.info(f"Added RoE rule: {rule.rule_id}")

    def remove_rule(self, rule_id: str) -> bool:
        original_count = len(self.rules)
        self.rules = [rule for rule in self.rules if rule.rule_id != rule_id]
        removed = len(self.rules) < original_count
        
        if removed:
            self.logger.info(f"Removed RoE rule: {rule_id}")
        else:
            self.logger.warning(f"Rule not found: {rule_id}")
        
        return removed

    def enable_rule(self, rule_id: str) -> bool:
        for rule in self.rules:
            if rule.rule_id == rule_id:
                rule.enabled = True
                self.logger.info(f"Enabled RoE rule: {rule_id}")
                return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        for rule in self.rules:
            if rule.rule_id == rule_id:
                rule.enabled = False
                self.logger.info(f"Disabled RoE rule: {rule_id}")
                return True
        return False

    def list_rules(self, enabled_only: bool = False) -> List[Dict[str, Any]]:
        rules_list = []
        for rule in self.rules:
            if enabled_only and not rule.enabled:
                continue
            
            rules_list.append({
                "rule_id": rule.rule_id,
                "rule_type": rule.rule_type,
                "pattern": rule.pattern,
                "description": rule.description,
                "severity": rule.severity,
                "enabled": rule.enabled
            })
        
        return rules_list

    async def validate_url_list(self, urls: List[str]) -> Dict[str, bool]:
        results = {}
        
        for url in urls:
            try:
                parsed = urlparse(url)
                hostname = parsed.hostname
                
                if hostname:
                    mock_target = type('Target', (), {
                        'hostname': hostname,
                        'ip_address': None,
                        'scope': 'in-scope'
                    })()
                    
                    results[url] = await self.validate_target(mock_target)
                else:
                    results[url] = False
                    
            except Exception as e:
                self.logger.error(f"Error validating URL {url}: {e}")
                results[url] = False
        
        return results

    def get_compliance_summary(self) -> Dict[str, Any]:
        enabled_rules = [rule for rule in self.rules if rule.enabled]
        
        rule_types = {}
        severities = {}
        
        for rule in enabled_rules:
            rule_types[rule.rule_type] = rule_types.get(rule.rule_type, 0) + 1
            severities[rule.severity] = severities.get(rule.severity, 0) + 1
        
        return {
            "total_rules": len(self.rules),
            "enabled_rules": len(enabled_rules),
            "disabled_rules": len(self.rules) - len(enabled_rules),
            "rule_types": rule_types,
            "severities": severities,
            "compliance_level": "strict" if len(enabled_rules) >= 5 else "basic"
        }