"""
XORB Platform Demo Sample Data Generator
Creates comprehensive sample datasets for demonstration purposes
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
from pathlib import Path

# Sample data templates and configurations
COMPANY_NAMES = [
    "Acme Financial Corp", "SecureBank Holdings", "TechFlow Industries",
    "Global Manufacturing Inc", "HealthCare Solutions", "RetailMax Chain",
    "DataFlow Systems", "CyberShield Corp", "InnovTech Labs", "FutureNet Inc"
]

ATTACK_VECTORS = [
    "malware", "phishing", "brute_force", "sql_injection", "xss",
    "ransomware", "ddos", "insider_threat", "privilege_escalation",
    "social_engineering", "zero_day", "man_in_middle"
]

SERVICE_NAMES = [
    "web-api", "auth-service", "database-cluster", "cache-redis",
    "message-queue", "file-storage", "cdn-edge", "monitoring-stack",
    "backup-service", "logging-aggregator"
]

COMPLIANCE_FRAMEWORKS = ["SOC2", "GDPR", "HIPAA", "PCI-DSS", "ISO27001", "NIST"]

VULNERABILITY_TYPES = [
    "SQL Injection", "Cross-Site Scripting", "Buffer Overflow",
    "Authentication Bypass", "Privilege Escalation", "Information Disclosure",
    "Denial of Service", "Remote Code Execution", "Path Traversal",
    "Weak Encryption", "Insecure Configuration", "Missing Security Headers"
]

class DemoDataGenerator:
    """Generate comprehensive sample data for XORB platform demos"""
    
    def __init__(self, output_dir: str = "demo/sample_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def generate_all_datasets(self):
        """Generate all demo datasets"""
        self.logger.info("Generating comprehensive demo datasets")
        
        # Generate core datasets
        self.generate_security_incidents()
        self.generate_vulnerability_assessments()
        self.generate_compliance_reports()
        self.generate_performance_metrics()
        self.generate_threat_intelligence()
        self.generate_user_behavior_data()
        self.generate_network_topology()
        self.generate_audit_logs()
        self.generate_business_metrics()
        self.generate_ptaas_scenarios()
        
        # Generate configuration data
        self.generate_demo_configurations()
        
        self.logger.info("All demo datasets generated successfully")
    
    def generate_security_incidents(self):
        """Generate realistic security incident data"""
        incidents = []
        
        for i in range(150):  # Generate 150 incidents over past month
            incident_time = datetime.now() - timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            severity = random.choices(
                ["critical", "high", "medium", "low"],
                weights=[5, 15, 50, 30]
            )[0]
            
            status = random.choices(
                ["resolved", "investigating", "mitigated", "false_positive"],
                weights=[60, 10, 25, 5]
            )[0]
            
            attack_vector = random.choice(ATTACK_VECTORS)
            
            incident = {
                "incident_id": f"INC-{datetime.now().year}-{1000 + i:04d}",
                "timestamp": incident_time.isoformat(),
                "severity": severity,
                "status": status,
                "attack_vector": attack_vector,
                "source_ip": f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
                "target_service": random.choice(SERVICE_NAMES),
                "description": f"{attack_vector.replace('_', ' ').title()} attack detected on {random.choice(SERVICE_NAMES)}",
                "analyst_assigned": f"analyst_{random.randint(1, 8)}",
                "detection_time_seconds": random.uniform(1.5, 45.2),
                "containment_time_minutes": random.uniform(2.1, 15.8),
                "resolution_time_hours": random.uniform(0.5, 8.3) if status == "resolved" else None,
                "indicators_of_compromise": [
                    f"hash_{uuid.uuid4().hex[:16]}",
                    f"domain_{uuid.uuid4().hex[:8]}.suspicious.com"
                ],
                "mitre_attack_technique": f"T{random.randint(1000, 1699)}.{random.randint(1, 999):03d}",
                "false_positive": status == "false_positive",
                "customer_impact": random.choice(["none", "minimal", "moderate", "significant"]),
                "estimated_damage_usd": random.randint(0, 50000) if severity in ["critical", "high"] else 0
            }
            incidents.append(incident)
        
        self._save_dataset("security_incidents.json", incidents)
        self.logger.info(f"Generated {len(incidents)} security incidents")
    
    def generate_vulnerability_assessments(self):
        """Generate vulnerability assessment data"""
        assessments = []
        
        for i in range(75):  # Generate 75 vulnerability assessments
            assessment_date = datetime.now() - timedelta(days=random.randint(0, 90))
            
            vuln_type = random.choice(VULNERABILITY_TYPES)
            cvss_score = round(random.uniform(2.1, 9.8), 1)
            
            if cvss_score >= 9.0:
                severity = "critical"
            elif cvss_score >= 7.0:
                severity = "high"
            elif cvss_score >= 4.0:
                severity = "medium"
            else:
                severity = "low"
            
            assessment = {
                "assessment_id": f"VULN-{assessment_date.year}-{2000 + i:04d}",
                "discovered_date": assessment_date.isoformat(),
                "vulnerability_type": vuln_type,
                "cve_id": f"CVE-{assessment_date.year}-{random.randint(10000, 99999)}",
                "cvss_score": cvss_score,
                "severity": severity,
                "affected_service": random.choice(SERVICE_NAMES),
                "affected_component": f"component_{random.choice(['web', 'api', 'db', 'cache', 'auth'])}",
                "description": f"{vuln_type} vulnerability in {random.choice(SERVICE_NAMES)}",
                "remediation_status": random.choices(
                    ["patched", "mitigated", "accepted_risk", "in_progress"],
                    weights=[50, 25, 5, 20]
                )[0],
                "remediation_date": (assessment_date + timedelta(days=random.randint(1, 30))).isoformat() if random.choice([True, False]) else None,
                "scanner_source": random.choice(["nmap", "nuclei", "custom_scanner", "manual_review"]),
                "exploitability": random.choice(["high", "medium", "low"]),
                "business_impact": random.choice(["critical", "high", "medium", "low"]),
                "vendor_acknowledgment": random.choice([True, False]),
                "patch_available": random.choice([True, False]),
                "workaround_available": random.choice([True, False])
            }
            assessments.append(assessment)
        
        self._save_dataset("vulnerability_assessments.json", assessments)
        self.logger.info(f"Generated {len(assessments)} vulnerability assessments")
    
    def generate_compliance_reports(self):
        """Generate compliance report data"""
        reports = {}
        
        for framework in COMPLIANCE_FRAMEWORKS:
            report_date = datetime.now() - timedelta(days=random.randint(1, 30))
            
            # Generate framework-specific metrics
            if framework == "SOC2":
                controls_data = {
                    "security": round(random.uniform(95.0, 99.5), 1),
                    "availability": round(random.uniform(94.0, 99.0), 1),
                    "processing_integrity": round(random.uniform(96.0, 99.8), 1),
                    "confidentiality": round(random.uniform(93.0, 98.0), 1),
                    "privacy": round(random.uniform(94.0, 98.5), 1)
                }
                overall_score = sum(controls_data.values()) / len(controls_data)
                
                report_data = {
                    "framework": framework,
                    "assessment_date": report_date.isoformat(),
                    "overall_score": round(overall_score, 1),
                    "control_categories": controls_data,
                    "total_controls": 17,
                    "automated_controls": 12,
                    "manual_controls": 5,
                    "control_deficiencies": random.randint(0, 3),
                    "audit_readiness": random.choice([True, False]),
                    "remediation_timeline_days": random.randint(7, 45) if random.choice([True, False]) else None
                }
            elif framework == "GDPR":
                report_data = {
                    "framework": framework,
                    "assessment_date": report_date.isoformat(),
                    "overall_score": round(random.uniform(85.0, 96.0), 1),
                    "privacy_impact_assessments": random.randint(5, 12),
                    "data_subject_requests": random.randint(15, 45),
                    "consent_management_score": round(random.uniform(90.0, 98.0), 1),
                    "data_retention_compliance": round(random.uniform(94.0, 99.5), 1),
                    "breach_notifications": random.randint(0, 2),
                    "data_processing_agreements": random.randint(8, 15),
                    "right_to_be_forgotten_requests": random.randint(3, 12)
                }
            else:
                report_data = {
                    "framework": framework,
                    "assessment_date": report_date.isoformat(),
                    "overall_score": round(random.uniform(80.0, 95.0), 1),
                    "controls_implemented": random.randint(75, 95),
                    "controls_tested": random.randint(65, 90),
                    "findings": random.randint(2, 8),
                    "recommendations": random.randint(5, 12),
                    "compliance_gaps": random.randint(1, 5)
                }
            
            reports[framework] = report_data
        
        self._save_dataset("compliance_reports.json", reports)
        self.logger.info(f"Generated compliance reports for {len(reports)} frameworks")
    
    def generate_performance_metrics(self):
        """Generate system performance metrics"""
        metrics = []
        
        # Generate hourly metrics for past 7 days
        for hour in range(7 * 24):
            timestamp = datetime.now() - timedelta(hours=hour)
            
            # Simulate daily patterns (higher load during business hours)
            hour_of_day = timestamp.hour
            if 9 <= hour_of_day <= 17:  # Business hours
                load_multiplier = random.uniform(1.2, 1.8)
            elif 6 <= hour_of_day <= 8 or 18 <= hour_of_day <= 22:  # Peak hours
                load_multiplier = random.uniform(0.8, 1.3)
            else:  # Off hours
                load_multiplier = random.uniform(0.3, 0.7)
            
            metric = {
                "timestamp": timestamp.isoformat(),
                "api_response_time_ms": round(random.uniform(15, 85) * load_multiplier, 2),
                "throughput_rps": int(random.randint(800, 1500) * load_multiplier),
                "error_rate_percent": round(random.uniform(0.1, 3.5) / load_multiplier, 3),
                "cpu_utilization_percent": round(random.uniform(25, 75) * load_multiplier, 1),
                "memory_utilization_percent": round(random.uniform(35, 80) * load_multiplier, 1),
                "disk_utilization_percent": round(random.uniform(20, 60), 1),
                "network_latency_ms": round(random.uniform(3.2, 18.7), 2),
                "active_connections": int(random.randint(150, 800) * load_multiplier),
                "database_performance": {
                    "avg_query_time_ms": round(random.uniform(8.5, 55.2) * load_multiplier, 2),
                    "connection_pool_usage_percent": round(random.uniform(10, 70) * load_multiplier, 1),
                    "slow_queries_count": int(random.randint(0, 12) * load_multiplier)
                },
                "cache_performance": {
                    "hit_rate_percent": round(random.uniform(85.0, 98.5), 2),
                    "memory_usage_percent": round(random.uniform(25, 75), 1),
                    "operations_per_second": int(random.randint(5000, 25000) * load_multiplier)
                }
            }
            metrics.append(metric)
        
        self._save_dataset("performance_metrics.json", metrics)
        self.logger.info(f"Generated {len(metrics)} performance metrics")
    
    def generate_threat_intelligence(self):
        """Generate threat intelligence data"""
        intelligence = []
        
        threat_types = ["malware", "botnet", "c2_server", "phishing_domain", "exploit_kit"]
        confidence_levels = ["high", "medium", "low"]
        
        for i in range(200):
            discovery_date = datetime.now() - timedelta(days=random.randint(0, 60))
            
            threat = {
                "threat_id": f"TI-{uuid.uuid4().hex[:12].upper()}",
                "discovery_date": discovery_date.isoformat(),
                "threat_type": random.choice(threat_types),
                "confidence_level": random.choice(confidence_levels),
                "source": random.choice(["osint", "commercial_feed", "internal_analysis", "partner_sharing"]),
                "ioc_type": random.choice(["ip_address", "domain", "file_hash", "url", "email"]),
                "indicator_value": self._generate_ioc(random.choice(["ip_address", "domain", "file_hash", "url", "email"])),
                "threat_actor": f"APT{random.randint(1, 50)}" if random.choice([True, False]) else f"Criminal Group {random.randint(1, 20)}",
                "target_industries": random.sample(["financial", "healthcare", "government", "technology", "manufacturing"], k=random.randint(1, 3)),
                "attack_techniques": random.sample(ATTACK_VECTORS, k=random.randint(1, 4)),
                "severity_score": random.randint(1, 10),
                "active_status": random.choice([True, False]),
                "first_seen": (discovery_date - timedelta(days=random.randint(1, 30))).isoformat(),
                "last_seen": discovery_date.isoformat(),
                "geographic_origin": random.choice(["CN", "RU", "KP", "IR", "US", "Unknown"]),
                "mitigation_recommendations": [
                    "Block indicator at network boundary",
                    "Monitor for lateral movement",
                    "Update endpoint detection rules"
                ]
            }
            intelligence.append(threat)
        
        self._save_dataset("threat_intelligence.json", intelligence)
        self.logger.info(f"Generated {len(intelligence)} threat intelligence entries")
    
    def generate_user_behavior_data(self):
        """Generate user behavior analytics data"""
        users = []
        
        for i in range(50):  # 50 users
            user_id = f"user_{uuid.uuid4().hex[:8]}"
            join_date = datetime.now() - timedelta(days=random.randint(30, 365))
            
            # Generate activity patterns
            login_frequency = random.uniform(0.5, 8.5)  # logins per day
            avg_session_duration = random.uniform(15, 240)  # minutes
            
            user = {
                "user_id": user_id,
                "email": f"{user_id}@{random.choice(['company', 'enterprise', 'corp'])}.com",
                "role": random.choice(["admin", "analyst", "user", "viewer"]),
                "join_date": join_date.isoformat(),
                "last_login": (datetime.now() - timedelta(days=random.randint(0, 7))).isoformat(),
                "login_frequency_daily": round(login_frequency, 2),
                "avg_session_duration_minutes": round(avg_session_duration, 1),
                "total_sessions": int(login_frequency * (datetime.now() - join_date).days),
                "failed_login_attempts": random.randint(0, 15),
                "risk_score": round(random.uniform(0.1, 9.8), 1),
                "anomaly_count": random.randint(0, 8),
                "locations_accessed": [
                    f"{random.choice(['New York', 'London', 'Tokyo', 'San Francisco', 'Berlin'])}, {random.choice(['US', 'UK', 'JP', 'DE'])}"
                    for _ in range(random.randint(1, 4))
                ],
                "device_count": random.randint(1, 5),
                "feature_usage": {
                    "security_dashboard": random.randint(5, 150),
                    "incident_management": random.randint(2, 80),
                    "threat_intelligence": random.randint(1, 45),
                    "compliance_reports": random.randint(0, 25),
                    "analytics": random.randint(3, 60)
                },
                "permissions": random.sample([
                    "view_dashboard", "manage_incidents", "admin_users", 
                    "export_data", "configure_alerts", "view_reports"
                ], k=random.randint(2, 5))
            }
            users.append(user)
        
        self._save_dataset("user_behavior_data.json", users)
        self.logger.info(f"Generated user behavior data for {len(users)} users")
    
    def generate_network_topology(self):
        """Generate network topology and asset inventory"""
        assets = []
        
        asset_types = ["server", "workstation", "network_device", "iot_device", "mobile_device"]
        operating_systems = ["Windows Server 2019", "Ubuntu 20.04", "CentOS 8", "Windows 10", "macOS", "iOS", "Android"]
        
        for i in range(150):  # 150 network assets
            asset_type = random.choice(asset_types)
            
            asset = {
                "asset_id": f"ASSET-{uuid.uuid4().hex[:12].upper()}",
                "hostname": f"{asset_type.replace('_', '-')}-{random.randint(100, 999)}",
                "ip_address": f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}",
                "mac_address": ":".join([f"{random.randint(0, 255):02x}" for _ in range(6)]),
                "asset_type": asset_type,
                "operating_system": random.choice(operating_systems),
                "os_version": f"{random.randint(1, 10)}.{random.randint(0, 9)}.{random.randint(0, 99)}",
                "last_seen": (datetime.now() - timedelta(days=random.randint(0, 7))).isoformat(),
                "criticality": random.choice(["critical", "high", "medium", "low"]),
                "owner": random.choice(COMPANY_NAMES),
                "location": random.choice(["Data Center 1", "Office Floor 2", "Remote", "Cloud - AWS", "Cloud - Azure"]),
                "open_ports": random.sample([22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 3389, 5432, 6379], k=random.randint(2, 8)),
                "services_running": random.sample([
                    "ssh", "http", "https", "smtp", "dns", "ftp", 
                    "database", "web_server", "cache", "monitoring"
                ], k=random.randint(1, 5)),
                "vulnerabilities_count": random.randint(0, 25),
                "patch_level": random.choice(["current", "outdated", "critical_updates_needed"]),
                "antivirus_status": random.choice(["active", "inactive", "not_installed"]),
                "firewall_status": random.choice(["enabled", "disabled"]),
                "encryption_status": random.choice(["encrypted", "partially_encrypted", "unencrypted"]),
                "compliance_status": random.choice(["compliant", "non_compliant", "unknown"]),
                "risk_score": round(random.uniform(1.0, 9.5), 1)
            }
            assets.append(asset)
        
        self._save_dataset("network_topology.json", assets)
        self.logger.info(f"Generated network topology with {len(assets)} assets")
    
    def generate_audit_logs(self):
        """Generate security audit logs"""
        logs = []
        
        actions = [
            "user_login", "user_logout", "password_change", "permission_granted",
            "permission_revoked", "data_access", "data_export", "configuration_change",
            "policy_update", "system_restart", "backup_created", "alert_dismissed"
        ]
        
        # Generate logs for past 30 days
        for i in range(1000):
            log_time = datetime.now() - timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            log_entry = {
                "log_id": f"LOG-{uuid.uuid4().hex[:16].upper()}",
                "timestamp": log_time.isoformat(),
                "action": random.choice(actions),
                "user_id": f"user_{uuid.uuid4().hex[:8]}",
                "source_ip": f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
                "user_agent": random.choice([
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                    "XORB-API-Client/1.0"
                ]),
                "resource_accessed": random.choice([
                    "/api/v1/security/incidents", "/api/v1/analytics/dashboard",
                    "/api/v1/users/profile", "/api/v1/compliance/reports"
                ]),
                "result": random.choices(["success", "failure"], weights=[85, 15])[0],
                "session_id": f"sess_{uuid.uuid4().hex[:16]}",
                "risk_level": random.choice(["low", "medium", "high"]),
                "geolocation": {
                    "country": random.choice(["US", "UK", "DE", "CA", "AU"]),
                    "city": random.choice(["New York", "London", "Berlin", "Toronto", "Sydney"]),
                    "coordinates": {
                        "lat": round(random.uniform(-90, 90), 4),
                        "lon": round(random.uniform(-180, 180), 4)
                    }
                },
                "details": {
                    "method": random.choice(["GET", "POST", "PUT", "DELETE"]),
                    "status_code": random.choice([200, 201, 400, 401, 403, 404, 500]),
                    "response_time_ms": round(random.uniform(10, 500), 2)
                }
            }
            logs.append(log_entry)
        
        self._save_dataset("audit_logs.json", logs)
        self.logger.info(f"Generated {len(logs)} audit log entries")
    
    def generate_business_metrics(self):
        """Generate business and operational metrics"""
        metrics = []
        
        # Generate daily business metrics for past 90 days
        for day in range(90):
            metric_date = datetime.now() - timedelta(days=day)
            
            # Simulate weekly patterns (lower on weekends)
            day_of_week = metric_date.weekday()
            if day_of_week >= 5:  # Weekend
                business_multiplier = random.uniform(0.3, 0.6)
            else:  # Weekday
                business_multiplier = random.uniform(0.8, 1.2)
            
            daily_metric = {
                "date": metric_date.date().isoformat(),
                "total_customers": int(random.randint(450, 650) * business_multiplier),
                "active_users": int(random.randint(380, 580) * business_multiplier),
                "new_signups": int(random.randint(5, 25) * business_multiplier),
                "customer_satisfaction": round(random.uniform(4.1, 4.9), 1),
                "support_tickets": {
                    "created": int(random.randint(10, 40) * business_multiplier),
                    "resolved": int(random.randint(8, 38) * business_multiplier),
                    "escalated": int(random.randint(0, 5) * business_multiplier)
                },
                "revenue_metrics": {
                    "daily_revenue": round(random.uniform(15000, 45000) * business_multiplier, 2),
                    "mrr": round(random.uniform(150000, 350000), 2),  # Monthly Recurring Revenue
                    "churn_rate_percent": round(random.uniform(1.2, 4.8), 2)
                },
                "feature_adoption": {
                    "security_dashboard": round(random.uniform(75, 95), 1),
                    "threat_intelligence": round(random.uniform(55, 85), 1),
                    "compliance_automation": round(random.uniform(45, 75), 1),
                    "incident_response": round(random.uniform(65, 90), 1),
                    "analytics_reporting": round(random.uniform(35, 65), 1)
                },
                "system_usage": {
                    "total_api_calls": int(random.randint(50000, 150000) * business_multiplier),
                    "data_processed_gb": round(random.uniform(100, 500) * business_multiplier, 2),
                    "peak_concurrent_users": int(random.randint(50, 200) * business_multiplier)
                }
            }
            metrics.append(daily_metric)
        
        self._save_dataset("business_metrics.json", metrics)
        self.logger.info(f"Generated {len(metrics)} business metrics entries")
    
    def generate_ptaas_scenarios(self):
        """Generate PTaaS (Penetration Testing as a Service) scenarios"""
        scenarios = []
        
        target_types = ["web_application", "network_infrastructure", "mobile_app", "api_endpoints", "cloud_services"]
        test_types = ["black_box", "white_box", "gray_box"]
        methodologies = ["OWASP", "NIST", "PTES", "Custom"]
        
        for i in range(25):  # 25 PTaaS scenarios
            start_date = datetime.now() - timedelta(days=random.randint(1, 90))
            duration_days = random.randint(3, 21)
            
            scenario = {
                "scenario_id": f"PTAAS-{start_date.year}-{3000 + i:04d}",
                "scenario_name": f"Penetration Test - {random.choice(COMPANY_NAMES)}",
                "target_type": random.choice(target_types),
                "test_type": random.choice(test_types),
                "methodology": random.choice(methodologies),
                "start_date": start_date.isoformat(),
                "end_date": (start_date + timedelta(days=duration_days)).isoformat(),
                "duration_days": duration_days,
                "status": random.choice(["completed", "in_progress", "scheduled", "cancelled"]),
                "tester_assigned": f"pentester_{random.randint(1, 5)}",
                "client_company": random.choice(COMPANY_NAMES),
                "scope": {
                    "target_urls": [f"https://{random.choice(['app', 'api', 'admin'])}.example{i}.com" for _ in range(random.randint(1, 5))],
                    "ip_ranges": [f"192.168.{random.randint(1, 254)}.0/24"],
                    "exclusions": ["192.168.1.100", "critical-production-db"]
                },
                "findings": {
                    "critical": random.randint(0, 3),
                    "high": random.randint(1, 8),
                    "medium": random.randint(3, 15),
                    "low": random.randint(5, 25),
                    "informational": random.randint(8, 30)
                },
                "vulnerabilities_discovered": [
                    {
                        "vulnerability": random.choice(VULNERABILITY_TYPES),
                        "severity": random.choice(["critical", "high", "medium", "low"]),
                        "cvss_score": round(random.uniform(2.1, 9.8), 1),
                        "exploited": random.choice([True, False]),
                        "remediation_effort": random.choice(["low", "medium", "high"])
                    }
                    for _ in range(random.randint(5, 20))
                ],
                "tools_used": random.sample([
                    "Nmap", "Nuclei", "Burp Suite", "OWASP ZAP", "Metasploit",
                    "Nikto", "SQLMap", "Gobuster", "Custom Scripts"
                ], k=random.randint(3, 7)),
                "compliance_validation": {
                    "framework": random.choice(COMPLIANCE_FRAMEWORKS),
                    "controls_tested": random.randint(15, 45),
                    "controls_passed": random.randint(12, 42),
                    "compliance_score": round(random.uniform(75.0, 98.0), 1)
                },
                "executive_summary": {
                    "overall_risk_rating": random.choice(["low", "medium", "high", "critical"]),
                    "business_impact": random.choice(["minimal", "moderate", "significant", "severe"]),
                    "recommendations": [
                        "Implement multi-factor authentication",
                        "Update security policies",
                        "Patch critical vulnerabilities",
                        "Enhance monitoring capabilities"
                    ][:random.randint(2, 4)]
                }
            }
            scenarios.append(scenario)
        
        self._save_dataset("ptaas_scenarios.json", scenarios)
        self.logger.info(f"Generated {len(scenarios)} PTaaS scenarios")
    
    def generate_demo_configurations(self):
        """Generate demo environment configurations"""
        config = {
            "demo_environment": {
                "name": "XORB Enterprise Demo Environment",
                "version": "3.0.0",
                "last_updated": datetime.now().isoformat(),
                "description": "Comprehensive cybersecurity platform demonstration environment"
            },
            "sample_companies": [
                {
                    "company_id": f"comp_{uuid.uuid4().hex[:8]}",
                    "name": name,
                    "industry": random.choice(["financial", "healthcare", "manufacturing", "technology", "retail"]),
                    "size": random.choice(["startup", "small", "medium", "large", "enterprise"]),
                    "employees": random.randint(50, 50000),
                    "security_maturity": random.choice(["basic", "intermediate", "advanced", "expert"])
                }
                for name in COMPANY_NAMES
            ],
            "demo_users": [
                user
                for i in range(1, 4)
                for user in [
                    {
                        "username": f"demo_admin_{i}",
                        "email": f"admin{i}@xorb-demo.com",
                        "role": "administrator",
                        "permissions": ["all"]
                    },
                    {
                        "username": f"demo_analyst_{i}",
                        "email": f"analyst{i}@xorb-demo.com",
                        "role": "security_analyst",
                        "permissions": ["view_incidents", "manage_threats", "generate_reports"]
                    },
                    {
                        "username": f"demo_viewer_{i}",
                        "email": f"viewer{i}@xorb-demo.com",
                        "role": "read_only",
                        "permissions": ["view_dashboard", "view_reports"]
                    }
                ]
            ],
            "demo_scenarios": [
                {
                    "scenario_name": "Fortune 500 Financial Institution",
                    "description": "Large bank with complex infrastructure and strict compliance requirements",
                    "use_cases": ["SOC2 compliance", "threat intelligence", "incident response", "executive reporting"],
                    "duration_minutes": 45
                },
                {
                    "scenario_name": "Healthcare Provider Network",
                    "description": "Multi-location healthcare network with HIPAA compliance needs",
                    "use_cases": ["HIPAA compliance", "vulnerability management", "user behavior analytics"],
                    "duration_minutes": 30
                },
                {
                    "scenario_name": "Manufacturing Enterprise",
                    "description": "Global manufacturer with OT/IT convergence security challenges",
                    "use_cases": ["industrial security", "supply chain monitoring", "anomaly detection"],
                    "duration_minutes": 35
                }
            ],
            "dashboard_presets": {
                "executive": ["key_metrics", "security_overview", "business_intelligence", "compliance_status"],
                "security_analyst": ["threat_dashboard", "incident_management", "vulnerability_tracking", "forensics"],
                "compliance_officer": ["compliance_dashboards", "audit_reports", "policy_tracking", "risk_assessment"]
            }
        }
        
        self._save_dataset("demo_configurations.json", config)
        self.logger.info("Generated demo environment configurations")
    
    def _generate_ioc(self, ioc_type: str) -> str:
        """Generate realistic indicators of compromise"""
        if ioc_type == "ip_address":
            return f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
        elif ioc_type == "domain":
            return f"{uuid.uuid4().hex[:8]}.{random.choice(['com', 'net', 'org', 'info'])}"
        elif ioc_type == "file_hash":
            return uuid.uuid4().hex + uuid.uuid4().hex[:8]  # SHA256-like
        elif ioc_type == "url":
            return f"http://{uuid.uuid4().hex[:8]}.{random.choice(['com', 'net'])}/malicious"
        elif ioc_type == "email":
            return f"{uuid.uuid4().hex[:8]}@{uuid.uuid4().hex[:8]}.com"
        else:
            return "unknown_ioc"
    
    def _save_dataset(self, filename: str, data: Any):
        """Save dataset to JSON file"""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        self.logger.info(f"Saved dataset to {filepath}")


# Main execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    generator = DemoDataGenerator()
    generator.generate_all_datasets()
    
    print("✅ All demo datasets generated successfully!")
    print(f"📁 Output directory: {generator.output_dir}")
    print("📊 Generated datasets:")
    
    for file in sorted(generator.output_dir.glob("*.json")):
        size_kb = file.stat().st_size / 1024
        print(f"   • {file.name} ({size_kb:.1f} KB)")