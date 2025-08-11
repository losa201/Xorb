#!/usr/bin/env python3
"""
Purple Environment Generator (PEG)
Infrastructure-as-Code orchestrator for synthetic enterprise environments
"""

import os
import json
import yaml
import random
import asyncio
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import tempfile
import shutil
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnvironmentBlueprint:
    name: str
    industry: str
    description: str
    baseline_topology: Dict[str, Any]
    default_services: List[Dict]
    compliance_controls: List[str]
    vulnerability_templates: List[str]
    data_patterns: Dict[str, Any]

@dataclass
class DeploymentConfiguration:
    deployment_id: str
    blueprint_name: str
    randomization_seed: int
    target_infrastructure: str  # aws, azure, gcp, vmware
    network_configuration: Dict[str, Any]
    service_manifest: List[Dict]
    vulnerability_profile: Dict[str, Any]
    synthetic_data_config: Dict[str, Any]
    instrumentation_config: Dict[str, Any]
    mutation_schedule: Dict[str, Any]

@dataclass
class VulnerabilityDefinition:
    vuln_id: str
    name: str
    severity: str  # critical, high, medium, low
    cve_id: Optional[str]
    category: str  # injection, authentication, crypto, etc.
    affected_services: List[str]
    exploitation_complexity: str  # low, medium, high
    detection_difficulty: str  # easy, medium, hard
    injection_method: str  # terraform, ansible, custom
    remediation_steps: List[str]

class PurpleEnvironmentGenerator:
    def __init__(self, base_path: str = "/root/Xorb/wargame-enterprise"):
        self.base_path = Path(base_path)
        self.blueprints_path = self.base_path / "purple-orchestrator" / "blueprints"
        self.templates_path = self.base_path / "purple-orchestrator" / "templates"
        self.vulnerability_db_path = self.base_path / "purple-orchestrator" / "vulnerability-db"
        self.deployments_path = self.base_path / "deployments"
        
        # Core components
        self.environment_blueprints = {}
        self.vulnerability_database = {}
        self.synthetic_data_generators = {}
        self.active_deployments = {}
        
        # Infrastructure tools
        self.terraform_path = "terraform"
        self.ansible_path = "ansible-playbook"
        self.helm_path = "helm"
        self.kubectl_path = "kubectl"
        
        # Initialize PEG
        self.initialize_peg()

    def initialize_peg(self):
        """Initialize Purple Environment Generator components"""
        logger.info("Initializing Purple Environment Generator (PEG)")
        
        # Create directory structure
        self.create_directory_structure()
        
        # Load environment blueprints
        self.load_environment_blueprints()
        
        # Load vulnerability database
        self.load_vulnerability_database()
        
        # Initialize synthetic data generators
        self.initialize_synthetic_data_generators()
        
        logger.info("PEG initialization complete")

    def create_directory_structure(self):
        """Create PEG directory structure"""
        directories = [
            self.blueprints_path,
            self.templates_path,
            self.vulnerability_db_path,
            self.deployments_path,
            self.templates_path / "terraform",
            self.templates_path / "ansible",
            self.templates_path / "helm",
            self.templates_path / "kubernetes"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def load_environment_blueprints(self):
        """Load industry-specific environment blueprints"""
        self.environment_blueprints = {
            "fintech": EnvironmentBlueprint(
                name="Financial Technology Platform",
                industry="fintech",
                description="Core banking app with payment gateway and API integrations",
                baseline_topology={
                    "dmz": {"subnets": ["10.1.0.0/24"], "services": ["web", "api_gateway", "load_balancer"]},
                    "internal": {"subnets": ["10.2.0.0/24", "10.3.0.0/24"], "services": ["app_servers", "databases"]},
                    "cloud": {"provider": "aws", "services": ["s3", "rds", "lambda"]}
                },
                default_services=[
                    {"name": "core_banking_app", "type": "web_application", "framework": "spring_boot"},
                    {"name": "payment_gateway", "type": "api_service", "framework": "nodejs"},
                    {"name": "customer_portal", "type": "spa", "framework": "react"},
                    {"name": "admin_console", "type": "web_application", "framework": "django"},
                    {"name": "transaction_db", "type": "database", "engine": "postgresql"},
                    {"name": "cache_layer", "type": "cache", "engine": "redis"},
                    {"name": "message_queue", "type": "messaging", "engine": "rabbitmq"}
                ],
                compliance_controls=["PCI-DSS", "SOX", "GDPR"],
                vulnerability_templates=["sql_injection", "weak_authentication", "insecure_api", "crypto_weakness"],
                data_patterns={
                    "customer_records": 10000,
                    "transactions": 50000,
                    "api_keys": 25,
                    "admin_accounts": 5
                }
            ),
            
            "healthcare": EnvironmentBlueprint(
                name="Healthcare Management System",
                industry="healthcare",
                description="EMR system with patient portal and HIPAA compliance",
                baseline_topology={
                    "dmz": {"subnets": ["192.168.1.0/24"], "services": ["patient_portal", "api_gateway"]},
                    "internal": {"subnets": ["192.168.10.0/24", "192.168.20.0/24"], "services": ["emr_system", "pacs"]},
                    "cloud": {"provider": "azure", "services": ["blob_storage", "sql_database", "key_vault"]}
                },
                default_services=[
                    {"name": "emr_system", "type": "web_application", "framework": "dotnet"},
                    {"name": "patient_portal", "type": "web_application", "framework": "angular"},
                    {"name": "pacs_viewer", "type": "specialized_app", "framework": "java"},
                    {"name": "billing_system", "type": "web_application", "framework": "php"},
                    {"name": "patient_db", "type": "database", "engine": "sql_server"},
                    {"name": "image_storage", "type": "storage", "engine": "azure_blob"},
                    {"name": "audit_service", "type": "logging", "engine": "elasticsearch"}
                ],
                compliance_controls=["HIPAA", "HITECH", "SOC2"],
                vulnerability_templates=["phi_exposure", "weak_encryption", "access_control", "audit_bypass"],
                data_patterns={
                    "patient_records": 25000,
                    "medical_images": 5000,
                    "staff_accounts": 200,
                    "audit_logs": 100000
                }
            ),
            
            "saas_startup": EnvironmentBlueprint(
                name="Multi-tenant SaaS Platform",
                industry="technology",
                description="Scalable web platform with CI/CD and cloud storage APIs",
                baseline_topology={
                    "dmz": {"subnets": ["172.16.1.0/24"], "services": ["cdn", "api_gateway", "auth_service"]},
                    "internal": {"subnets": ["172.16.10.0/24", "172.16.20.0/24"], "services": ["app_cluster", "data_layer"]},
                    "cloud": {"provider": "gcp", "services": ["gcs", "cloud_sql", "gke"]}
                },
                default_services=[
                    {"name": "saas_platform", "type": "microservices", "framework": "kubernetes"},
                    {"name": "auth_service", "type": "identity", "framework": "oauth2"},
                    {"name": "tenant_api", "type": "api_service", "framework": "golang"},
                    {"name": "analytics_engine", "type": "data_processing", "framework": "spark"},
                    {"name": "platform_db", "type": "database", "engine": "mongodb"},
                    {"name": "file_storage", "type": "storage", "engine": "gcs"},
                    {"name": "ci_cd_pipeline", "type": "devops", "engine": "jenkins"}
                ],
                compliance_controls=["SOC2", "ISO27001", "GDPR"],
                vulnerability_templates=["tenant_isolation", "api_abuse", "container_escape", "ci_cd_poisoning"],
                data_patterns={
                    "tenant_accounts": 500,
                    "user_accounts": 50000,
                    "api_tokens": 1000,
                    "deployment_artifacts": 200
                }
            ),
            
            "gov_defense": EnvironmentBlueprint(
                name="Government Defense Network",
                industry="government",
                description="Classified document repository with restricted networks and PKI",
                baseline_topology={
                    "unclassified": {"subnets": ["10.100.1.0/24"], "services": ["public_web", "email_gateway"]},
                    "classified": {"subnets": ["10.200.1.0/24", "10.200.2.0/24"], "services": ["doc_repo", "secure_comms"]},
                    "cloud": {"provider": "govcloud", "services": ["secure_storage", "encrypted_compute"]}
                },
                default_services=[
                    {"name": "document_repository", "type": "secure_storage", "framework": "custom"},
                    {"name": "classification_system", "type": "security_service", "framework": "java"},
                    {"name": "secure_messaging", "type": "communication", "framework": "signal_protocol"},
                    {"name": "pki_infrastructure", "type": "crypto_service", "framework": "openssl"},
                    {"name": "classified_db", "type": "database", "engine": "oracle"},
                    {"name": "audit_system", "type": "monitoring", "engine": "splunk"},
                    {"name": "vpn_gateway", "type": "network", "engine": "ipsec"}
                ],
                compliance_controls=["FISMA", "NIST", "FedRAMP"],
                vulnerability_templates=["classification_bypass", "pki_weakness", "network_isolation", "covert_channel"],
                data_patterns={
                    "classified_documents": 5000,
                    "security_clearances": 100,
                    "crypto_keys": 50,
                    "access_logs": 250000
                }
            )
        }

    def load_vulnerability_database(self):
        """Load comprehensive vulnerability database"""
        self.vulnerability_database = {
            "sql_injection": VulnerabilityDefinition(
                vuln_id="VULN_SQL_001",
                name="SQL Injection in User Search",
                severity="high",
                cve_id="CVE-2024-1234",
                category="injection",
                affected_services=["web_application", "api_service"],
                exploitation_complexity="low",
                detection_difficulty="medium",
                injection_method="ansible",
                remediation_steps=["Input validation", "Parameterized queries", "WAF rules"]
            ),
            
            "weak_authentication": VulnerabilityDefinition(
                vuln_id="VULN_AUTH_001",
                name="Default Administrative Credentials",
                severity="critical",
                cve_id=None,
                category="authentication",
                affected_services=["admin_console", "database"],
                exploitation_complexity="low",
                detection_difficulty="easy",
                injection_method="terraform",
                remediation_steps=["Change default passwords", "Implement MFA", "Account lockout"]
            ),
            
            "insecure_api": VulnerabilityDefinition(
                vuln_id="VULN_API_001",
                name="Unprotected API Endpoint",
                severity="medium",
                cve_id=None,
                category="authorization",
                affected_services=["api_gateway", "microservices"],
                exploitation_complexity="medium",
                detection_difficulty="hard",
                injection_method="helm",
                remediation_steps=["API authentication", "Rate limiting", "Input validation"]
            ),
            
            "crypto_weakness": VulnerabilityDefinition(
                vuln_id="VULN_CRYPTO_001",
                name="Weak Cryptographic Implementation",
                severity="high",
                cve_id="CVE-2024-5678",
                category="cryptography",
                affected_services=["auth_service", "payment_gateway"],
                exploitation_complexity="high",
                detection_difficulty="hard",
                injection_method="custom",
                remediation_steps=["Upgrade crypto libraries", "Strong algorithms", "Key rotation"]
            ),
            
            "container_escape": VulnerabilityDefinition(
                vuln_id="VULN_CONTAINER_001",
                name="Container Escape via Privileged Mode",
                severity="critical",
                cve_id="CVE-2024-9999",
                category="container_security",
                affected_services=["kubernetes", "docker"],
                exploitation_complexity="medium",
                detection_difficulty="medium",
                injection_method="helm",
                remediation_steps=["Remove privileged mode", "Security contexts", "Pod security policies"]
            ),
            
            "phi_exposure": VulnerabilityDefinition(
                vuln_id="VULN_PHI_001",
                name="Patient Health Information Exposure",
                severity="critical",
                cve_id=None,
                category="data_exposure",
                affected_services=["patient_portal", "emr_system"],
                exploitation_complexity="low",
                detection_difficulty="easy",
                injection_method="ansible",
                remediation_steps=["Access controls", "Data encryption", "Audit logging"]
            ),
            
            "tenant_isolation": VulnerabilityDefinition(
                vuln_id="VULN_TENANT_001",
                name="Multi-tenant Data Isolation Bypass",
                severity="high",
                cve_id=None,
                category="authorization",
                affected_services=["saas_platform", "tenant_api"],
                exploitation_complexity="medium",
                detection_difficulty="hard",
                injection_method="terraform",
                remediation_steps=["Tenant ID validation", "Database isolation", "API boundaries"]
            ),
            
            "classification_bypass": VulnerabilityDefinition(
                vuln_id="VULN_CLASS_001",
                name="Security Classification Bypass",
                severity="critical",
                cve_id=None,
                category="access_control",
                affected_services=["document_repository", "classification_system"],
                exploitation_complexity="high",
                detection_difficulty="medium",
                injection_method="custom",
                remediation_steps=["Classification enforcement", "Mandatory access controls", "Audit trails"]
            )
        }

    def initialize_synthetic_data_generators(self):
        """Initialize synthetic data generation capabilities"""
        self.synthetic_data_generators = {
            "identity_generator": self.generate_synthetic_identities,
            "workflow_simulator": self.simulate_synthetic_workflows,
            "document_generator": self.generate_synthetic_documents,
            "traffic_simulator": self.simulate_baseline_traffic
        }

    async def generate_environment(self, blueprint_name: str, 
                                 customizations: Dict = None,
                                 target_infrastructure: str = "aws") -> str:
        """Generate complete synthetic environment from blueprint"""
        
        if blueprint_name not in self.environment_blueprints:
            raise ValueError(f"Blueprint '{blueprint_name}' not found")
        
        blueprint = self.environment_blueprints[blueprint_name]
        deployment_id = f"peg_{blueprint_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Generating environment: {deployment_id}")
        
        # Create randomization configuration
        randomization_seed = random.randint(1000, 9999)
        random.seed(randomization_seed)
        
        # Generate deployment configuration
        deployment_config = await self.create_deployment_configuration(
            deployment_id, blueprint, customizations or {}, target_infrastructure, randomization_seed
        )
        
        # Create deployment workspace
        deployment_path = self.deployments_path / deployment_id
        deployment_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Phase 1: Generate Infrastructure as Code
            logger.info(f"Phase 1: Generating infrastructure for {deployment_id}")
            await self.generate_infrastructure_code(deployment_config, deployment_path)
            
            # Phase 2: Deploy Infrastructure
            logger.info(f"Phase 2: Deploying infrastructure for {deployment_id}")
            await self.deploy_infrastructure(deployment_config, deployment_path)
            
            # Phase 3: Deploy Services
            logger.info(f"Phase 3: Deploying services for {deployment_id}")
            await self.deploy_services(deployment_config, deployment_path)
            
            # Phase 4: Inject Vulnerabilities
            logger.info(f"Phase 4: Injecting vulnerabilities for {deployment_id}")
            await self.inject_vulnerabilities(deployment_config, deployment_path)
            
            # Phase 5: Seed Synthetic Data
            logger.info(f"Phase 5: Seeding synthetic data for {deployment_id}")
            await self.seed_synthetic_data(deployment_config, deployment_path)
            
            # Phase 6: Setup Instrumentation
            logger.info(f"Phase 6: Setting up instrumentation for {deployment_id}")
            await self.setup_instrumentation(deployment_config, deployment_path)
            
            # Phase 7: Start Traffic Simulation
            logger.info(f"Phase 7: Starting traffic simulation for {deployment_id}")
            await self.start_traffic_simulation(deployment_config, deployment_path)
            
            # Store deployment info
            self.active_deployments[deployment_id] = {
                "config": deployment_config,
                "status": "active",
                "created_time": datetime.now(),
                "path": str(deployment_path)
            }
            
            # Generate environment manifest for Red/Blue teams
            await self.generate_environment_manifest(deployment_config, deployment_path)
            
            logger.info(f"Environment {deployment_id} generation complete")
            return deployment_id
            
        except Exception as e:
            logger.error(f"Error generating environment {deployment_id}: {e}")
            # Cleanup on failure
            await self.cleanup_failed_deployment(deployment_id, deployment_path)
            raise

    async def create_deployment_configuration(self, deployment_id: str, 
                                            blueprint: EnvironmentBlueprint,
                                            customizations: Dict,
                                            target_infrastructure: str,
                                            randomization_seed: int) -> DeploymentConfiguration:
        """Create comprehensive deployment configuration"""
        
        # Randomize network configuration
        network_config = self.randomize_network_topology(blueprint.baseline_topology)
        
        # Randomize service versions and configurations
        service_manifest = self.randomize_service_manifest(blueprint.default_services, customizations)
        
        # Select and configure vulnerabilities
        vulnerability_profile = self.create_vulnerability_profile(blueprint.vulnerability_templates)
        
        # Configure synthetic data generation
        synthetic_data_config = self.configure_synthetic_data(blueprint.data_patterns, customizations)
        
        # Setup instrumentation configuration
        instrumentation_config = self.configure_instrumentation(blueprint.industry)
        
        # Create mutation schedule
        mutation_schedule = self.create_mutation_schedule(customizations)
        
        return DeploymentConfiguration(
            deployment_id=deployment_id,
            blueprint_name=blueprint.name,
            randomization_seed=randomization_seed,
            target_infrastructure=target_infrastructure,
            network_configuration=network_config,
            service_manifest=service_manifest,
            vulnerability_profile=vulnerability_profile,
            synthetic_data_config=synthetic_data_config,
            instrumentation_config=instrumentation_config,
            mutation_schedule=mutation_schedule
        )

    def randomize_network_topology(self, baseline_topology: Dict) -> Dict:
        """Randomize network topology while maintaining functionality"""
        
        randomized = {}
        
        for zone, config in baseline_topology.items():
            randomized[zone] = {
                "subnets": self.randomize_ip_ranges(config.get("subnets", [])),
                "services": config.get("services", []),
                "security_groups": self.generate_security_groups(zone),
                "exposed_ports": self.randomize_exposed_ports(zone)
            }
        
        return randomized

    def randomize_ip_ranges(self, base_subnets: List[str]) -> List[str]:
        """Randomize IP ranges while avoiding conflicts"""
        
        private_ranges = [
            "10.{}.0.0/24",
            "172.{}.0.0/24", 
            "192.168.{}.0/24"
        ]
        
        randomized_subnets = []
        for _ in range(len(base_subnets)):
            range_template = random.choice(private_ranges)
            octet = random.randint(1, 254)
            subnet = range_template.format(octet)
            randomized_subnets.append(subnet)
        
        return randomized_subnets

    def generate_security_groups(self, zone: str) -> List[Dict]:
        """Generate security group configurations for network zone"""
        
        security_groups = []
        
        if zone == "dmz":
            security_groups.append({
                "name": f"sg_{zone}_web",
                "rules": [
                    {"protocol": "tcp", "port": 80, "source": "0.0.0.0/0"},
                    {"protocol": "tcp", "port": 443, "source": "0.0.0.0/0"},
                    {"protocol": "tcp", "port": 22, "source": "admin_ips"}
                ]
            })
        elif zone == "internal":
            security_groups.append({
                "name": f"sg_{zone}_app",
                "rules": [
                    {"protocol": "tcp", "port": 8080, "source": "dmz"},
                    {"protocol": "tcp", "port": 3306, "source": "app_servers"},
                    {"protocol": "tcp", "port": 22, "source": "bastion"}
                ]
            })
        
        return security_groups

    def randomize_exposed_ports(self, zone: str) -> List[int]:
        """Randomize exposed ports with some intentional misconfigurations"""
        
        standard_ports = {
            "dmz": [80, 443, 22],
            "internal": [8080, 3306, 5432, 6379],
            "cloud": [443, 22]
        }
        
        ports = standard_ports.get(zone, [80, 443])
        
        # Randomly add some potentially problematic ports
        risky_ports = [21, 23, 135, 139, 445, 1433, 3389, 5900]
        if random.random() < 0.3:  # 30% chance
            ports.append(random.choice(risky_ports))
        
        return ports

    def randomize_service_manifest(self, default_services: List[Dict], 
                                 customizations: Dict) -> List[Dict]:
        """Randomize service versions and configurations"""
        
        randomized_services = []
        
        for service in default_services:
            randomized_service = service.copy()
            
            # Randomize versions
            randomized_service["version"] = self.get_random_version(service["type"])
            
            # Add random configurations
            randomized_service["config"] = self.generate_service_config(service)
            
            # Sometimes add intentional misconfigurations
            if random.random() < 0.2:  # 20% chance
                randomized_service["misconfigurations"] = self.add_misconfigurations(service)
            
            randomized_services.append(randomized_service)
        
        return randomized_services

    def get_random_version(self, service_type: str) -> str:
        """Get random (potentially outdated) version for service type"""
        
        version_ranges = {
            "web_application": ["2.4.1", "2.5.0", "2.6.2", "3.0.1", "3.1.0"],
            "database": ["5.7.32", "8.0.25", "8.0.30", "8.0.33"],
            "api_service": ["1.18.0", "1.19.5", "1.20.2", "1.21.0"],
            "microservices": ["v1.24.0", "v1.25.3", "v1.26.1", "v1.27.0"]
        }
        
        versions = version_ranges.get(service_type, ["1.0.0", "2.0.0", "3.0.0"])
        return random.choice(versions)

    def generate_service_config(self, service: Dict) -> Dict:
        """Generate service-specific configuration"""
        
        configs = {
            "debug_mode": random.choice([True, False]),
            "logging_level": random.choice(["DEBUG", "INFO", "WARN", "ERROR"]),
            "max_connections": random.randint(50, 500),
            "timeout_seconds": random.randint(30, 300)
        }
        
        # Add service-specific configs
        if service["type"] == "database":
            configs.update({
                "query_logging": random.choice([True, False]),
                "remote_access": random.choice([True, False])
            })
        elif service["type"] == "web_application":
            configs.update({
                "cors_enabled": random.choice([True, False]),
                "csrf_protection": random.choice([True, False])
            })
        
        return configs

    def add_misconfigurations(self, service: Dict) -> List[str]:
        """Add intentional misconfigurations for testing"""
        
        misconfigs = []
        
        if service["type"] == "web_application":
            misconfigs.extend([
                "debug_endpoints_enabled",
                "permissive_cors",
                "weak_session_config"
            ])
        elif service["type"] == "database":
            misconfigs.extend([
                "default_credentials",
                "remote_root_access",
                "unencrypted_connections"
            ])
        elif service["type"] == "api_service":
            misconfigs.extend([
                "no_rate_limiting",
                "verbose_error_messages", 
                "weak_token_validation"
            ])
        
        # Return 1-2 random misconfigurations
        return random.sample(misconfigs, min(2, len(misconfigs)))

    def create_vulnerability_profile(self, vulnerability_templates: List[str]) -> Dict:
        """Create vulnerability profile for environment"""
        
        profile = {
            "total_vulnerabilities": random.randint(3, 8),
            "severity_distribution": {
                "critical": random.randint(0, 2),
                "high": random.randint(1, 3),
                "medium": random.randint(1, 4),
                "low": random.randint(0, 2)
            },
            "selected_vulnerabilities": [],
            "custom_vulnerabilities": []
        }
        
        # Select vulnerabilities from templates
        available_vulns = [
            vuln for vuln_id, vuln in self.vulnerability_database.items()
            if any(template in vuln_id.lower() for template in vulnerability_templates)
        ]
        
        # Randomly select vulnerabilities
        selected_count = min(profile["total_vulnerabilities"], len(available_vulns))
        selected_vulns = random.sample(available_vulns, selected_count)
        
        profile["selected_vulnerabilities"] = [asdict(vuln) for vuln in selected_vulns]
        
        # Generate custom logic flaws
        profile["custom_vulnerabilities"] = self.generate_custom_vulnerabilities()
        
        return profile

    def generate_custom_vulnerabilities(self) -> List[Dict]:
        """Generate custom logic flaws and misconfigurations"""
        
        custom_vulns = []
        
        # Business logic flaws
        if random.random() < 0.4:
            custom_vulns.append({
                "name": "Price Manipulation Vulnerability",
                "description": "Negative quantity allows price manipulation",
                "severity": "high",
                "location": "payment_processing",
                "exploitation": "Submit negative quantity values"
            })
        
        # Authentication bypasses
        if random.random() < 0.3:
            custom_vulns.append({
                "name": "JWT Secret Exposure",
                "description": "JWT signing key exposed in configuration",
                "severity": "critical",
                "location": "auth_service",
                "exploitation": "Forge JWT tokens with exposed secret"
            })
        
        # Authorization flaws
        if random.random() < 0.5:
            custom_vulns.append({
                "name": "Horizontal Privilege Escalation",
                "description": "User ID parameter manipulation",
                "severity": "medium",
                "location": "user_profile_api",
                "exploitation": "Modify user_id parameter to access other accounts"
            })
        
        return custom_vulns

    def configure_synthetic_data(self, data_patterns: Dict, customizations: Dict) -> Dict:
        """Configure synthetic data generation"""
        
        config = {
            "identity_generation": {
                "employee_count": data_patterns.get("employee_count", 100),
                "department_count": random.randint(5, 12),
                "naming_patterns": ["US_names", "international_mix"],
                "email_domains": ["company.com", "company.org"]
            },
            "document_generation": {
                "document_count": data_patterns.get("document_count", 1000),
                "sensitivity_levels": ["public", "internal", "confidential", "restricted"],
                "file_types": [".pdf", ".docx", ".xlsx", ".txt", ".csv"],
                "content_categories": ["financial", "personal", "technical", "legal"]
            },
            "workflow_simulation": {
                "business_processes": ["order_processing", "customer_onboarding", "incident_response"],
                "automation_level": random.uniform(0.3, 0.8),
                "peak_hours": "09:00-17:00",
                "weekend_activity": random.uniform(0.1, 0.3)
            },
            "traffic_simulation": {
                "baseline_users": random.randint(50, 200),
                "peak_multiplier": random.uniform(2.0, 5.0),
                "geographic_distribution": ["US_East", "US_West", "Europe", "Asia"],
                "device_types": ["desktop", "mobile", "tablet", "api_client"]
            }
        }
        
        return config

    def configure_instrumentation(self, industry: str) -> Dict:
        """Configure telemetry and monitoring instrumentation"""
        
        config = {
            "network_monitoring": {
                "pcap_collection": True,
                "netflow_enabled": True,
                "dns_logging": True,
                "full_packet_retention_days": 7
            },
            "application_monitoring": {
                "access_logs": True,
                "error_logs": True,
                "performance_metrics": True,
                "user_activity_tracking": True
            },
            "security_monitoring": {
                "authentication_events": True,
                "authorization_failures": True,
                "privilege_escalations": True,
                "data_access_logging": True
            },
            "compliance_logging": {
                "audit_trail": True,
                "data_retention_days": self.get_compliance_retention(industry),
                "encryption_at_rest": True,
                "log_integrity_protection": True
            },
            "honeypots": {
                "fake_admin_panels": random.randint(2, 5),
                "decoy_databases": random.randint(1, 3),
                "canary_tokens": random.randint(10, 25)
            }
        }
        
        return config

    def get_compliance_retention(self, industry: str) -> int:
        """Get compliance-based log retention requirements"""
        
        retention_map = {
            "fintech": 2555,      # 7 years for financial
            "healthcare": 2190,   # 6 years for healthcare
            "government": 2555,   # 7 years for government
            "technology": 1095    # 3 years for general tech
        }
        
        return retention_map.get(industry, 1095)

    def create_mutation_schedule(self, customizations: Dict) -> Dict:
        """Create environment mutation schedule"""
        
        schedule = {
            "mutation_enabled": customizations.get("enable_mutations", True),
            "triggers": [
                {
                    "type": "time_based",
                    "interval_hours": random.randint(24, 168),  # 1-7 days
                    "mutations": ["service_updates", "config_changes", "new_vulnerabilities"]
                },
                {
                    "type": "performance_based",
                    "blue_success_threshold": 0.8,
                    "red_success_threshold": 0.8,
                    "mutations": ["difficulty_adjustment", "new_attack_vectors"]
                },
                {
                    "type": "scenario_based",
                    "trigger_events": ["incident_response", "compliance_audit", "penetration_test"],
                    "mutations": ["security_hardening", "monitoring_enhancement"]
                }
            ],
            "mutation_types": [
                "add_new_service",
                "update_service_version",
                "modify_network_topology",
                "inject_new_vulnerability",
                "change_user_permissions",
                "update_security_policies"
            ]
        }
        
        return schedule

    async def generate_infrastructure_code(self, config: DeploymentConfiguration, 
                                         deployment_path: Path):
        """Generate Terraform infrastructure code"""
        
        terraform_path = deployment_path / "terraform"
        terraform_path.mkdir(exist_ok=True)
        
        # Generate main.tf
        main_tf = self.generate_terraform_main(config)
        (terraform_path / "main.tf").write_text(main_tf)
        
        # Generate variables.tf
        variables_tf = self.generate_terraform_variables(config)
        (terraform_path / "variables.tf").write_text(variables_tf)
        
        # Generate terraform.tfvars
        tfvars = self.generate_terraform_tfvars(config)
        (terraform_path / "terraform.tfvars").write_text(tfvars)
        
        logger.info(f"Generated Terraform code for {config.deployment_id}")

    def generate_terraform_main(self, config: DeploymentConfiguration) -> str:
        """Generate main Terraform configuration"""
        
        terraform_config = f"""
# Generated by Purple Environment Generator
# Deployment: {config.deployment_id}
# Blueprint: {config.blueprint_name}

terraform {{
  required_version = ">= 1.0"
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }}
  }}
}}

provider "aws" {{
  region = var.aws_region
  
  default_tags {{
    tags = {{
      Environment = "wargame"
      Deployment  = "{config.deployment_id}"
      Blueprint   = "{config.blueprint_name}"
      Generator   = "PEG"
    }}
  }}
}}

# VPC and Networking
resource "aws_vpc" "main" {{
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {{
    Name = "{config.deployment_id}-vpc"
  }}
}}

# Internet Gateway
resource "aws_internet_gateway" "main" {{
  vpc_id = aws_vpc.main.id
  
  tags = {{
    Name = "{config.deployment_id}-igw"
  }}
}}
"""
        
        # Add subnets for each network zone
        for zone, zone_config in config.network_configuration.items():
            for i, subnet_cidr in enumerate(zone_config["subnets"]):
                terraform_config += f"""
# {zone.title()} Subnet {i+1}
resource "aws_subnet" "{zone}_subnet_{i+1}" {{
  vpc_id            = aws_vpc.main.id
  cidr_block        = "{subnet_cidr}"
  availability_zone = data.aws_availability_zones.available.names[{i % 3}]
  
  map_public_ip_on_launch = {str(zone == "dmz").lower()}
  
  tags = {{
    Name = "{config.deployment_id}-{zone}-subnet-{i+1}"
    Zone = "{zone}"
  }}
}}
"""
        
        terraform_config += """
# Data source for availability zones
data "aws_availability_zones" "available" {
  state = "available"
}

# Route table for public subnets
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  
  tags = {
    Name = "public-rt"
  }
}
"""
        
        return terraform_config

    def generate_terraform_variables(self, config: DeploymentConfiguration) -> str:
        """Generate Terraform variables"""
        
        return f"""
# Variables for {config.deployment_id}

variable "aws_region" {{
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}}

variable "vpc_cidr" {{
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}}

variable "environment" {{
  description = "Environment name"
  type        = string
  default     = "{config.deployment_id}"
}}

variable "blueprint" {{
  description = "Environment blueprint"
  type        = string
  default     = "{config.blueprint_name}"
}}
"""

    def generate_terraform_tfvars(self, config: DeploymentConfiguration) -> str:
        """Generate Terraform variable values"""
        
        return f"""
# Terraform variables for {config.deployment_id}
aws_region  = "us-east-1"
environment = "{config.deployment_id}"
blueprint   = "{config.blueprint_name}"
"""

    async def deploy_infrastructure(self, config: DeploymentConfiguration, 
                                  deployment_path: Path):
        """Deploy infrastructure using Terraform"""
        
        terraform_path = deployment_path / "terraform"
        
        # Initialize Terraform
        await self.run_command(
            [self.terraform_path, "init"],
            cwd=terraform_path,
            description="Terraform init"
        )
        
        # Plan deployment
        await self.run_command(
            [self.terraform_path, "plan", "-out=tfplan"],
            cwd=terraform_path,
            description="Terraform plan"
        )
        
        # Apply deployment
        await self.run_command(
            [self.terraform_path, "apply", "-auto-approve", "tfplan"],
            cwd=terraform_path,
            description="Terraform apply"
        )
        
        logger.info(f"Infrastructure deployed for {config.deployment_id}")

    async def deploy_services(self, config: DeploymentConfiguration, 
                            deployment_path: Path):
        """Deploy services using Helm and Kubernetes"""
        
        helm_path = deployment_path / "helm"
        helm_path.mkdir(exist_ok=True)
        
        # Generate Helm charts for each service
        for service in config.service_manifest:
            chart_name = service["name"].replace("_", "-")
            chart_path = helm_path / chart_name
            
            # Create Helm chart
            await self.run_command(
                [self.helm_path, "create", chart_name],
                cwd=helm_path,
                description=f"Create Helm chart for {service['name']}"
            )
            
            # Customize chart values
            await self.customize_helm_chart(service, chart_path)
            
            # Deploy chart
            await self.run_command(
                [self.helm_path, "install", chart_name, str(chart_path)],
                cwd=helm_path,
                description=f"Deploy {service['name']}"
            )
        
        logger.info(f"Services deployed for {config.deployment_id}")

    async def customize_helm_chart(self, service: Dict, chart_path: Path):
        """Customize Helm chart based on service configuration"""
        
        values_yaml = chart_path / "values.yaml"
        
        # Read existing values
        if values_yaml.exists():
            with open(values_yaml, 'r') as f:
                values = yaml.safe_load(f) or {}
        else:
            values = {}
        
        # Update values based on service config
        values.update({
            "image": {
                "repository": f"synthetic/{service['name']}",
                "tag": service.get("version", "latest")
            },
            "service": {
                "type": "ClusterIP",
                "port": service.get("port", 8080)
            },
            "resources": {
                "limits": {
                    "cpu": "500m",
                    "memory": "512Mi"
                },
                "requests": {
                    "cpu": "250m", 
                    "memory": "256Mi"
                }
            },
            "config": service.get("config", {})
        })
        
        # Add misconfigurations if present
        if "misconfigurations" in service:
            values["misconfigurations"] = service["misconfigurations"]
        
        # Write updated values
        with open(values_yaml, 'w') as f:
            yaml.dump(values, f, default_flow_style=False)

    async def inject_vulnerabilities(self, config: DeploymentConfiguration, 
                                   deployment_path: Path):
        """Inject vulnerabilities using various methods"""
        
        vuln_path = deployment_path / "vulnerabilities"
        vuln_path.mkdir(exist_ok=True)
        
        for vuln in config.vulnerability_profile["selected_vulnerabilities"]:
            logger.info(f"Injecting vulnerability: {vuln['name']}")
            
            if vuln["injection_method"] == "terraform":
                await self.inject_vulnerability_terraform(vuln, deployment_path)
            elif vuln["injection_method"] == "ansible":
                await self.inject_vulnerability_ansible(vuln, deployment_path)
            elif vuln["injection_method"] == "helm":
                await self.inject_vulnerability_helm(vuln, deployment_path)
            elif vuln["injection_method"] == "custom":
                await self.inject_vulnerability_custom(vuln, deployment_path)
        
        # Generate vulnerability manifest
        vuln_manifest = {
            "deployment_id": config.deployment_id,
            "injected_vulnerabilities": config.vulnerability_profile["selected_vulnerabilities"],
            "custom_vulnerabilities": config.vulnerability_profile["custom_vulnerabilities"],
            "injection_timestamp": datetime.now().isoformat()
        }
        
        with open(vuln_path / "vulnerability_manifest.json", "w") as f:
            json.dump(vuln_manifest, f, indent=2)
        
        logger.info(f"Vulnerabilities injected for {config.deployment_id}")

    async def inject_vulnerability_terraform(self, vuln: Dict, deployment_path: Path):
        """Inject vulnerability via Terraform configuration modification"""
        
        # This would modify Terraform files to include vulnerable configurations
        terraform_path = deployment_path / "terraform"
        vuln_tf_path = terraform_path / f"vuln_{vuln['vuln_id'].lower()}.tf"
        
        # Example: weak security group for authentication vulnerability
        if vuln["category"] == "authentication":
            vuln_config = f"""
# Vulnerability injection: {vuln['name']}
resource "aws_security_group_rule" "weak_auth_access" {{
  type              = "ingress"
  from_port         = 22
  to_port           = 22
  protocol          = "tcp"
  cidr_blocks       = ["0.0.0.0/0"]  # Intentionally permissive
  security_group_id = aws_security_group.main.id
}}
"""
            vuln_tf_path.write_text(vuln_config)

    async def inject_vulnerability_ansible(self, vuln: Dict, deployment_path: Path):
        """Inject vulnerability via Ansible playbook"""
        
        ansible_path = deployment_path / "ansible"
        ansible_path.mkdir(exist_ok=True)
        
        playbook_path = ansible_path / f"inject_{vuln['vuln_id'].lower()}.yml"
        
        # Example playbook for SQL injection vulnerability
        if vuln["category"] == "injection":
            playbook = f"""
---
- name: Inject {vuln['name']}
  hosts: web_servers
  become: yes
  tasks:
    - name: Deploy vulnerable code
      copy:
        content: |
          # Vulnerable SQL query - {vuln['name']}
          query = "SELECT * FROM users WHERE id = " + user_input
        dest: /var/www/html/vulnerable_search.php
        mode: '0644'
    
    - name: Restart web service
      service:
        name: apache2
        state: restarted
"""
            playbook_path.write_text(playbook)
            
            # Execute playbook
            await self.run_command(
                [self.ansible_path, str(playbook_path)],
                cwd=ansible_path,
                description=f"Inject vulnerability {vuln['vuln_id']}"
            )

    async def inject_vulnerability_helm(self, vuln: Dict, deployment_path: Path):
        """Inject vulnerability via Helm chart modification"""
        
        # This would modify Helm charts to include vulnerable configurations
        pass  # Implementation would modify specific Helm values

    async def inject_vulnerability_custom(self, vuln: Dict, deployment_path: Path):
        """Inject vulnerability via custom scripts"""
        
        # Custom vulnerability injection for complex scenarios
        pass  # Implementation would run custom scripts

    async def seed_synthetic_data(self, config: DeploymentConfiguration, 
                                deployment_path: Path):
        """Seed environment with synthetic data"""
        
        data_path = deployment_path / "synthetic_data"
        data_path.mkdir(exist_ok=True)
        
        # Generate synthetic identities
        identities = await self.generate_synthetic_identities(config.synthetic_data_config)
        with open(data_path / "identities.json", "w") as f:
            json.dump(identities, f, indent=2)
        
        # Generate synthetic documents
        documents = await self.generate_synthetic_documents(config.synthetic_data_config)
        with open(data_path / "documents_manifest.json", "w") as f:
            json.dump(documents, f, indent=2)
        
        # Setup workflow simulation
        workflows = await self.setup_workflow_simulation(config.synthetic_data_config)
        with open(data_path / "workflows.json", "w") as f:
            json.dump(workflows, f, indent=2)
        
        logger.info(f"Synthetic data seeded for {config.deployment_id}")

    async def generate_synthetic_identities(self, data_config: Dict) -> Dict:
        """Generate synthetic employee identities and accounts"""
        
        import random
        import string
        
        first_names = ["John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Lisa", "James", "Ashley"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
        departments = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations", "Legal", "IT"]
        
        identities = {
            "employees": [],
            "departments": departments,
            "total_count": data_config["identity_generation"]["employee_count"]
        }
        
        for i in range(data_config["identity_generation"]["employee_count"]):
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            department = random.choice(departments)
            
            employee = {
                "employee_id": f"EMP{i+1:04d}",
                "first_name": first_name,
                "last_name": last_name,
                "email": f"{first_name.lower()}.{last_name.lower()}@company.com",
                "department": department,
                "title": self.generate_job_title(department),
                "manager_id": f"EMP{random.randint(1, min(i+1, 20)):04d}" if i > 0 else None,
                "hire_date": self.generate_random_date(),
                "access_level": random.choice(["basic", "standard", "elevated", "admin"]),
                "weak_password": random.random() < 0.1  # 10% have weak passwords
            }
            
            identities["employees"].append(employee)
        
        return identities

    def generate_job_title(self, department: str) -> str:
        """Generate realistic job titles by department"""
        
        titles = {
            "Engineering": ["Software Engineer", "Senior Engineer", "Engineering Manager", "DevOps Engineer"],
            "Sales": ["Sales Representative", "Account Manager", "Sales Director", "Business Development"],
            "Marketing": ["Marketing Specialist", "Content Manager", "Marketing Director", "Digital Marketer"],
            "HR": ["HR Specialist", "HR Manager", "Recruiter", "HR Director"],
            "Finance": ["Financial Analyst", "Accountant", "Finance Manager", "Controller"],
            "Operations": ["Operations Manager", "Project Manager", "Operations Coordinator", "Program Manager"],
            "Legal": ["Legal Counsel", "Paralegal", "Compliance Officer", "General Counsel"],
            "IT": ["IT Specialist", "System Administrator", "IT Manager", "Security Analyst"]
        }
        
        return random.choice(titles.get(department, ["Specialist", "Manager", "Director"]))

    def generate_random_date(self) -> str:
        """Generate random hire date within last 5 years"""
        
        start_date = datetime.now() - timedelta(days=365*5)
        random_days = random.randint(0, 365*5)
        hire_date = start_date + timedelta(days=random_days)
        
        return hire_date.strftime("%Y-%m-%d")

    async def generate_synthetic_documents(self, data_config: Dict) -> Dict:
        """Generate synthetic document repository"""
        
        document_types = {
            "financial": ["budget.xlsx", "revenue_report.pdf", "expense_summary.xlsx"],
            "personal": ["employee_records.xlsx", "performance_reviews.pdf", "salary_data.xlsx"],
            "technical": ["system_architecture.pdf", "api_documentation.md", "database_schema.sql"],
            "legal": ["contracts.pdf", "privacy_policy.md", "compliance_report.pdf"]
        }
        
        documents = {
            "total_documents": data_config["document_generation"]["document_count"],
            "categories": list(document_types.keys()),
            "documents": []
        }
        
        for i in range(data_config["document_generation"]["document_count"]):
            category = random.choice(list(document_types.keys()))
            template = random.choice(document_types[category])
            
            document = {
                "document_id": f"DOC{i+1:06d}",
                "filename": f"{template.split('.')[0]}_{i+1:04d}.{template.split('.')[1]}",
                "category": category,
                "sensitivity": random.choice(["public", "internal", "confidential", "restricted"]),
                "created_date": self.generate_random_date(),
                "owner": f"EMP{random.randint(1, 100):04d}",
                "size_bytes": random.randint(1024, 10*1024*1024),  # 1KB to 10MB
                "contains_pii": random.random() < 0.3,  # 30% contain PII
                "contains_secrets": random.random() < 0.05  # 5% contain secrets
            }
            
            documents["documents"].append(document)
        
        return documents

    async def setup_workflow_simulation(self, data_config: Dict) -> Dict:
        """Setup automated workflow simulation"""
        
        workflows = {
            "business_processes": data_config["workflow_simulation"]["business_processes"],
            "automation_level": data_config["workflow_simulation"]["automation_level"],
            "scheduled_tasks": []
        }
        
        # Create scheduled tasks for realistic activity
        task_templates = [
            {"name": "daily_backup", "frequency": "daily", "time": "02:00"},
            {"name": "log_rotation", "frequency": "weekly", "time": "03:00"},
            {"name": "security_scan", "frequency": "daily", "time": "01:00"},
            {"name": "database_maintenance", "frequency": "weekly", "time": "04:00"},
            {"name": "report_generation", "frequency": "daily", "time": "06:00"}
        ]
        
        for template in task_templates:
            task = template.copy()
            task["enabled"] = random.random() < 0.8  # 80% of tasks enabled
            task["last_run"] = self.generate_random_date()
            workflows["scheduled_tasks"].append(task)
        
        return workflows

    async def setup_instrumentation(self, config: DeploymentConfiguration, 
                                  deployment_path: Path):
        """Setup comprehensive instrumentation and monitoring"""
        
        instrumentation_path = deployment_path / "instrumentation"
        instrumentation_path.mkdir(exist_ok=True)
        
        # Setup network monitoring
        await self.setup_network_monitoring(config, instrumentation_path)
        
        # Setup application monitoring
        await self.setup_application_monitoring(config, instrumentation_path)
        
        # Setup security monitoring
        await self.setup_security_monitoring(config, instrumentation_path)
        
        # Deploy honeypots
        await self.deploy_honeypots(config, instrumentation_path)
        
        logger.info(f"Instrumentation setup complete for {config.deployment_id}")

    async def setup_network_monitoring(self, config: DeploymentConfiguration, 
                                     instrumentation_path: Path):
        """Setup network traffic monitoring and PCAP collection"""
        
        # Generate network monitoring configuration
        monitoring_config = {
            "pcap_collection": {
                "enabled": config.instrumentation_config["network_monitoring"]["pcap_collection"],
                "interfaces": ["eth0", "eth1"],
                "rotation_size": "100MB",
                "retention_days": config.instrumentation_config["network_monitoring"]["full_packet_retention_days"]
            },
            "netflow": {
                "enabled": config.instrumentation_config["network_monitoring"]["netflow_enabled"],
                "collectors": ["10.0.1.100:2055"],
                "sampling_rate": 1000
            },
            "dns_logging": {
                "enabled": config.instrumentation_config["network_monitoring"]["dns_logging"],
                "log_queries": True,
                "log_responses": True
            }
        }
        
        with open(instrumentation_path / "network_monitoring.json", "w") as f:
            json.dump(monitoring_config, f, indent=2)

    async def setup_application_monitoring(self, config: DeploymentConfiguration, 
                                         instrumentation_path: Path):
        """Setup application-level monitoring and logging"""
        
        app_monitoring_config = {
            "log_aggregation": {
                "enabled": True,
                "targets": ["elasticsearch:9200"],
                "log_levels": ["INFO", "WARN", "ERROR"],
                "structured_logging": True
            },
            "performance_monitoring": {
                "enabled": config.instrumentation_config["application_monitoring"]["performance_metrics"],
                "metrics_endpoint": "/metrics",
                "collection_interval": 30
            },
            "user_activity_tracking": {
                "enabled": config.instrumentation_config["application_monitoring"]["user_activity_tracking"],
                "track_logins": True,
                "track_api_calls": True,
                "track_data_access": True
            }
        }
        
        with open(instrumentation_path / "application_monitoring.json", "w") as f:
            json.dump(app_monitoring_config, f, indent=2)

    async def setup_security_monitoring(self, config: DeploymentConfiguration, 
                                      instrumentation_path: Path):
        """Setup security event monitoring and alerting"""
        
        security_monitoring_config = {
            "authentication_monitoring": {
                "enabled": config.instrumentation_config["security_monitoring"]["authentication_events"],
                "failed_login_threshold": 5,
                "suspicious_patterns": ["brute_force", "credential_stuffing"]
            },
            "authorization_monitoring": {
                "enabled": config.instrumentation_config["security_monitoring"]["authorization_failures"],
                "track_privilege_escalation": True,
                "track_unauthorized_access": True
            },
            "data_access_monitoring": {
                "enabled": config.instrumentation_config["security_monitoring"]["data_access_logging"],
                "sensitive_data_access": True,
                "bulk_operations": True
            }
        }
        
        with open(instrumentation_path / "security_monitoring.json", "w") as f:
            json.dump(security_monitoring_config, f, indent=2)

    async def deploy_honeypots(self, config: DeploymentConfiguration, 
                             instrumentation_path: Path):
        """Deploy honeypots and canary tokens"""
        
        honeypot_config = {
            "fake_admin_panels": [],
            "decoy_databases": [],
            "canary_tokens": []
        }
        
        # Generate fake admin panels
        for i in range(config.instrumentation_config["honeypots"]["fake_admin_panels"]):
            admin_panel = {
                "id": f"admin_panel_{i+1:02d}",
                "url": f"/admin{random.randint(1000, 9999)}/",
                "type": "fake_login",
                "monitoring": True
            }
            honeypot_config["fake_admin_panels"].append(admin_panel)
        
        # Generate decoy databases
        for i in range(config.instrumentation_config["honeypots"]["decoy_databases"]):
            decoy_db = {
                "id": f"decoy_db_{i+1:02d}",
                "connection_string": f"postgresql://readonly:guest123@10.0.{random.randint(1,254)}.99:5432/decoy_db_{i+1}",
                "monitoring": True
            }
            honeypot_config["decoy_databases"].append(decoy_db)
        
        # Generate canary tokens
        for i in range(config.instrumentation_config["honeypots"]["canary_tokens"]):
            canary_token = {
                "id": f"canary_{i+1:03d}",
                "type": random.choice(["aws_key", "db_credential", "api_token", "ssh_key"]),
                "location": random.choice(["config_file", "source_code", "database", "document"]),
                "monitoring": True
            }
            honeypot_config["canary_tokens"].append(canary_token)
        
        with open(instrumentation_path / "honeypots.json", "w") as f:
            json.dump(honeypot_config, f, indent=2)

    async def start_traffic_simulation(self, config: DeploymentConfiguration, 
                                     deployment_path: Path):
        """Start baseline traffic simulation"""
        
        traffic_path = deployment_path / "traffic_simulation"
        traffic_path.mkdir(exist_ok=True)
        
        # Generate traffic simulation configuration
        traffic_config = {
            "baseline_users": config.synthetic_data_config["traffic_simulation"]["baseline_users"],
            "peak_multiplier": config.synthetic_data_config["traffic_simulation"]["peak_multiplier"],
            "simulation_patterns": [
                {
                    "name": "business_hours",
                    "schedule": "09:00-17:00",
                    "user_activity": "high",
                    "api_calls_per_minute": random.randint(100, 500)
                },
                {
                    "name": "after_hours",
                    "schedule": "18:00-08:00",
                    "user_activity": "low",
                    "api_calls_per_minute": random.randint(10, 50)
                },
                {
                    "name": "weekend",
                    "schedule": "saturday-sunday",
                    "user_activity": "minimal",
                    "api_calls_per_minute": random.randint(5, 25)
                }
            ]
        }
        
        with open(traffic_path / "traffic_config.json", "w") as f:
            json.dump(traffic_config, f, indent=2)
        
        # Start traffic simulation script
        simulation_script = traffic_path / "start_simulation.sh"
        simulation_script.write_text(f"""#!/bin/bash
# Traffic simulation for {config.deployment_id}
echo "Starting traffic simulation..."
python3 traffic_simulator.py --config traffic_config.json &
echo "Traffic simulation started"
""")
        simulation_script.chmod(0o755)
        
        logger.info(f"Traffic simulation started for {config.deployment_id}")

    async def generate_environment_manifest(self, config: DeploymentConfiguration, 
                                          deployment_path: Path):
        """Generate environment manifest for Red/Blue team consumption"""
        
        manifest = {
            "deployment_info": {
                "deployment_id": config.deployment_id,
                "blueprint_name": config.blueprint_name,
                "created_timestamp": datetime.now().isoformat(),
                "randomization_seed": config.randomization_seed,
                "target_infrastructure": config.target_infrastructure
            },
            "network_topology": config.network_configuration,
            "services": config.service_manifest,
            "vulnerabilities": {
                "known_vulnerabilities": config.vulnerability_profile["selected_vulnerabilities"],
                "custom_vulnerabilities": config.vulnerability_profile["custom_vulnerabilities"],
                "total_count": config.vulnerability_profile["total_vulnerabilities"]
            },
            "synthetic_data": {
                "identities_location": "synthetic_data/identities.json",
                "documents_location": "synthetic_data/documents_manifest.json",
                "workflows_location": "synthetic_data/workflows.json"
            },
            "instrumentation": {
                "network_monitoring": "instrumentation/network_monitoring.json",
                "application_monitoring": "instrumentation/application_monitoring.json",
                "security_monitoring": "instrumentation/security_monitoring.json",
                "honeypots": "instrumentation/honeypots.json"
            },
            "access_points": {
                "external_endpoints": self.extract_external_endpoints(config),
                "internal_services": self.extract_internal_services(config),
                "admin_interfaces": self.extract_admin_interfaces(config)
            },
            "intelligence_gathering": {
                "osint_targets": self.generate_osint_targets(config),
                "reconnaissance_hints": self.generate_recon_hints(config)
            }
        }
        
        # Save manifest
        with open(deployment_path / "environment_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Environment manifest generated for {config.deployment_id}")
        return manifest

    def extract_external_endpoints(self, config: DeploymentConfiguration) -> List[Dict]:
        """Extract external-facing endpoints for Red Team targeting"""
        
        endpoints = []
        
        for service in config.service_manifest:
            if service.get("exposure", "internal") in ["external", "public"]:
                endpoint = {
                    "service_name": service["name"],
                    "url": f"https://{service['name'].replace('_', '-')}.{config.deployment_id}.local",
                    "type": service["type"],
                    "framework": service.get("framework", "unknown"),
                    "version": service.get("version", "unknown")
                }
                endpoints.append(endpoint)
        
        return endpoints

    def extract_internal_services(self, config: DeploymentConfiguration) -> List[Dict]:
        """Extract internal services for lateral movement targets"""
        
        services = []
        
        for service in config.service_manifest:
            if service.get("exposure", "internal") == "internal":
                internal_service = {
                    "service_name": service["name"],
                    "type": service["type"],
                    "framework": service.get("framework", "unknown"),
                    "network_zone": "internal",
                    "potential_vulnerabilities": service.get("misconfigurations", [])
                }
                services.append(internal_service)
        
        return services

    def extract_admin_interfaces(self, config: DeploymentConfiguration) -> List[Dict]:
        """Extract administrative interfaces for privilege escalation"""
        
        admin_interfaces = []
        
        for service in config.service_manifest:
            if "admin" in service["name"] or service["type"] == "admin_console":
                admin_interface = {
                    "service_name": service["name"],
                    "access_method": "web",
                    "authentication": service.get("auth_method", "username_password"),
                    "default_credentials": service.get("default_creds", False)
                }
                admin_interfaces.append(admin_interface)
        
        return admin_interfaces

    def generate_osint_targets(self, config: DeploymentConfiguration) -> List[str]:
        """Generate OSINT targets for reconnaissance"""
        
        targets = [
            f"{config.deployment_id}.local",
            f"www.{config.deployment_id}.local",
            f"mail.{config.deployment_id}.local",
            f"api.{config.deployment_id}.local"
        ]
        
        return targets

    def generate_recon_hints(self, config: DeploymentConfiguration) -> List[str]:
        """Generate reconnaissance hints for Red Team"""
        
        hints = [
            f"Organization follows {config.blueprint_name} industry patterns",
            f"Network uses {len(config.network_configuration)} security zones",
            f"Environment contains {len(config.service_manifest)} primary services",
            f"Vulnerability count: {config.vulnerability_profile['total_vulnerabilities']} known issues"
        ]
        
        return hints

    async def run_command(self, command: List[str], cwd: Path, 
                         description: str) -> subprocess.CompletedProcess:
        """Run shell command with proper error handling"""
        
        logger.info(f"Running: {description}")
        
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.debug(f"Command output: {result.stdout}")
            return result
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {description}")
            logger.error(f"Error: {e.stderr}")
            raise

    async def cleanup_failed_deployment(self, deployment_id: str, deployment_path: Path):
        """Clean up failed deployment resources"""
        
        logger.info(f"Cleaning up failed deployment: {deployment_id}")
        
        try:
            # Destroy Terraform resources
            terraform_path = deployment_path / "terraform"
            if terraform_path.exists():
                await self.run_command(
                    [self.terraform_path, "destroy", "-auto-approve"],
                    cwd=terraform_path,
                    description="Terraform destroy"
                )
            
            # Remove deployment directory
            if deployment_path.exists():
                shutil.rmtree(deployment_path)
            
            # Remove from active deployments
            if deployment_id in self.active_deployments:
                del self.active_deployments[deployment_id]
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Example usage
async def main():
    """Example usage of Purple Environment Generator"""
    
    peg = PurpleEnvironmentGenerator()
    
    # Generate a fintech environment
    deployment_id = await peg.generate_environment(
        blueprint_name="fintech",
        customizations={
            "enable_mutations": True,
            "vulnerability_density": "high",
            "compliance_mode": "PCI-DSS"
        },
        target_infrastructure="aws"
    )
    
    print(f"Generated environment: {deployment_id}")

if __name__ == "__main__":
    asyncio.run(main())