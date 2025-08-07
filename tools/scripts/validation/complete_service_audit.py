#!/usr/bin/env python3
"""
XORB Complete Service Audit
Comprehensive analysis of deployed vs undeployed services and improvement recommendations
"""

import asyncio
import json
import subprocess
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

class XORBServiceAuditor:
    """Complete XORB ecosystem audit and analysis"""
    
    def __init__(self):
        self.namespace = "xorb-platform"
        self.audit_results = {
            "deployed_services": {},
            "undeployed_services": {},
            "service_issues": {},
            "recommendations": [],
            "optimization_opportunities": []
        }
        
        # Complete XORB service catalog
        self.complete_service_catalog = {
            "core_platform": {
                "postgres": {"type": "database", "status": "unknown", "priority": "critical"},
                "redis": {"type": "cache", "status": "unknown", "priority": "critical"},
                "api-gateway": {"type": "api", "status": "unknown", "priority": "high"},
                "security-api": {"type": "api", "status": "unknown", "priority": "high"},
                "threat-intelligence": {"type": "service", "status": "unknown", "priority": "high"},
                "analytics-engine": {"type": "service", "status": "unknown", "priority": "high"},
                "web-frontend": {"type": "frontend", "status": "unknown", "priority": "medium"}
            },
            "ptaas_platform": {
                "neo4j": {"type": "database", "status": "unknown", "priority": "medium"},
                "qdrant": {"type": "vector_db", "status": "unknown", "priority": "medium"},
                "rabbitmq": {"type": "message_queue", "status": "unknown", "priority": "medium"},
                "ptaas-bug-bounty": {"type": "service", "status": "unknown", "priority": "medium"},
                "ptaas-exploit-validation": {"type": "service", "status": "unknown", "priority": "medium"},
                "ptaas-reward-system": {"type": "service", "status": "unknown", "priority": "low"}
            },
            "monitoring": {
                "prometheus": {"type": "monitoring", "status": "unknown", "priority": "medium"},
                "grafana": {"type": "dashboard", "status": "unknown", "priority": "medium"},
                "alertmanager": {"type": "alerting", "status": "unknown", "priority": "low"}
            },
            "optimized_services": {
                "simple-api-gateway": {"type": "api", "status": "unknown", "priority": "high"},
                "security-dashboard": {"type": "dashboard", "status": "unknown", "priority": "medium"},
                "mock-threat-intel": {"type": "service", "status": "unknown", "priority": "medium"},
                "basic-monitoring": {"type": "monitoring", "status": "unknown", "priority": "medium"}
            },
            "existing_docker_services": {
                "xorb-postgres": {"type": "database", "status": "unknown", "priority": "critical"},
                "xorb-nats": {"type": "message_queue", "status": "unknown", "priority": "medium"},
                "xorb-orchestrator": {"type": "service", "status": "unknown", "priority": "high"},
                "xorb-dashboard": {"type": "dashboard", "status": "unknown", "priority": "medium"},
                "xorb-training": {"type": "service", "status": "unknown", "priority": "low"}
            }
        }
    
    async def conduct_full_audit(self):
        """Conduct comprehensive service audit"""
        print("🔍 XORB COMPLETE SERVICE AUDIT")
        print("=" * 80)
        print(f"📅 Audit Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("=" * 80)
        
        # Check Kubernetes services
        await self.audit_kubernetes_services()
        
        # Check Docker services
        await self.audit_docker_services()
        
        # Check file system services
        await self.audit_filesystem_services()
        
        # Generate service status matrix
        await self.generate_service_matrix()
        
        # Identify issues and gaps
        await self.identify_service_issues()
        
        # Generate improvement recommendations
        await self.generate_improvement_recommendations()
        
        # Create deployment priority matrix
        await self.create_deployment_priority_matrix()
        
        # Generate final recommendations
        await self.generate_final_recommendations()
        
    async def audit_kubernetes_services(self):
        """Audit all Kubernetes services"""
        print("\n🏗️ KUBERNETES SERVICES AUDIT")
        print("-" * 60)
        
        try:
            # Get all deployments
            result = subprocess.run(['kubectl', 'get', 'deployments', '-n', self.namespace, '-o', 'json'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                deployments = json.loads(result.stdout)
                
                for deployment in deployments['items']:
                    name = deployment['metadata']['name']
                    desired = deployment['spec']['replicas']
                    ready = deployment['status'].get('readyReplicas', 0)
                    
                    status = "running" if ready == desired and ready > 0 else "degraded"
                    
                    self.audit_results["deployed_services"][name] = {
                        "type": "kubernetes",
                        "status": status,
                        "replicas": f"{ready}/{desired}",
                        "platform": "k8s"
                    }
                    
                    # Update catalog
                    for category in self.complete_service_catalog.values():
                        if name in category:
                            category[name]["status"] = status
            
            # Get StatefulSets
            result = subprocess.run(['kubectl', 'get', 'statefulsets', '-n', self.namespace, '-o', 'json'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                statefulsets = json.loads(result.stdout)
                
                for sts in statefulsets['items']:
                    name = sts['metadata']['name']
                    desired = sts['spec']['replicas']
                    ready = sts['status'].get('readyReplicas', 0)
                    
                    status = "running" if ready == desired and ready > 0 else "degraded"
                    
                    self.audit_results["deployed_services"][name] = {
                        "type": "kubernetes",
                        "status": status,
                        "replicas": f"{ready}/{desired}",
                        "platform": "k8s"
                    }
                    
                    # Update catalog
                    for category in self.complete_service_catalog.values():
                        if name in category:
                            category[name]["status"] = status
                            
            print(f"✅ Found {len(self.audit_results['deployed_services'])} Kubernetes services")
            
        except Exception as e:
            print(f"❌ Kubernetes audit failed: {e}")
    
    async def audit_docker_services(self):
        """Audit Docker container services"""
        print("\n🐳 DOCKER SERVICES AUDIT")
        print("-" * 60)
        
        try:
            result = subprocess.run(['docker', 'ps', '-a', '--format', 'json'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                containers = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        try:
                            containers.append(json.loads(line))
                        except:
                            continue
                
                docker_services = 0
                for container in containers:
                    name = container.get('Names', '')
                    status = container.get('State', '')
                    image = container.get('Image', '')
                    
                    if 'xorb' in name.lower():
                        docker_services += 1
                        service_status = "running" if status == "running" else "stopped"
                        
                        self.audit_results["deployed_services"][name] = {
                            "type": "docker",
                            "status": service_status,
                            "image": image,
                            "platform": "docker"
                        }
                        
                        # Update catalog
                        for category in self.complete_service_catalog.values():
                            if name in category:
                                category[name]["status"] = service_status
                
                print(f"✅ Found {docker_services} XORB Docker services")
                
        except Exception as e:
            print(f"❌ Docker audit failed: {e}")
    
    async def audit_filesystem_services(self):
        """Audit filesystem-based services"""
        print("\n📁 FILESYSTEM SERVICES AUDIT")
        print("-" * 60)
        
        # Check for service files and configurations
        service_paths = [
            "/root/Xorb/api_gateway.py",
            "/root/Xorb/security_api_endpoints.py", 
            "/root/Xorb/advanced_analytics_engine.py",
            "/root/Xorb/disaster_recovery_system.py",
            "/root/Xorb/security_hardening_system.py",
            "/root/Xorb/deploy_ptaas_platform.py",
            "/root/Xorb/deploy_xorb_enterprise.py",
            "/var/www/verteidiq.com/index.html",
            "/var/www/verteidiq.com/de/index.html"
        ]
        
        filesystem_services = 0
        for path in service_paths:
            if os.path.exists(path):
                filesystem_services += 1
                service_name = os.path.basename(path).replace('.py', '').replace('.html', '')
                
                self.audit_results["deployed_services"][f"file-{service_name}"] = {
                    "type": "filesystem",
                    "status": "available",
                    "path": path,
                    "platform": "filesystem"
                }
        
        print(f"✅ Found {filesystem_services} filesystem-based services")
    
    async def generate_service_matrix(self):
        """Generate comprehensive service status matrix"""
        print("\n📊 SERVICE STATUS MATRIX")
        print("-" * 60)
        
        # Count services by status
        status_counts = {"running": 0, "degraded": 0, "stopped": 0, "unknown": 0, "available": 0}
        platform_counts = {"k8s": 0, "docker": 0, "filesystem": 0}
        
        for service, details in self.audit_results["deployed_services"].items():
            status = details["status"]
            platform = details["platform"]
            
            status_counts[status] = status_counts.get(status, 0) + 1
            platform_counts[platform] = platform_counts.get(platform, 0) + 1
        
        print("📈 Status Distribution:")
        for status, count in status_counts.items():
            if count > 0:
                icon = {"running": "🟢", "degraded": "🟡", "stopped": "🔴", "unknown": "⚪", "available": "🔵"}
                print(f"  {icon.get(status, '⚫')} {status.capitalize()}: {count}")
        
        print("\n🏗️ Platform Distribution:")
        for platform, count in platform_counts.items():
            if count > 0:
                icon = {"k8s": "☸️", "docker": "🐳", "filesystem": "📁"}
                print(f"  {icon.get(platform, '⚫')} {platform.upper()}: {count}")
        
        # Calculate overall health
        total_services = len(self.audit_results["deployed_services"])
        healthy_services = status_counts.get("running", 0) + status_counts.get("available", 0)
        health_percentage = (healthy_services / total_services * 100) if total_services > 0 else 0
        
        if health_percentage >= 80:
            health_icon = "🟢"
            health_status = "HEALTHY"
        elif health_percentage >= 60:
            health_icon = "🟡" 
            health_status = "DEGRADED"
        else:
            health_icon = "🔴"
            health_status = "CRITICAL"
        
        print(f"\n{health_icon} Overall Ecosystem Health: {health_percentage:.1f}% - {health_status}")
        print(f"   Total Services Found: {total_services}")
        print(f"   Healthy Services: {healthy_services}")
    
    async def identify_service_issues(self):
        """Identify service issues and missing components"""
        print("\n⚠️ SERVICE ISSUES & GAPS ANALYSIS")
        print("-" * 60)
        
        # Check for undeployed services from catalog
        for category_name, services in self.complete_service_catalog.items():
            missing_services = []
            for service_name, details in services.items():
                if details["status"] == "unknown":
                    missing_services.append(service_name)
                    self.audit_results["undeployed_services"][service_name] = {
                        "category": category_name,
                        "priority": details["priority"],
                        "type": details["type"]
                    }
            
            if missing_services:
                print(f"\n❌ Missing {category_name.replace('_', ' ').title()} Services:")
                for service in missing_services:
                    priority = services[service]["priority"]
                    service_type = services[service]["type"]
                    priority_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
                    print(f"   {priority_icon.get(priority, '⚫')} {service} ({service_type}) - {priority} priority")
        
        # Identify degraded services
        degraded_services = []
        for service, details in self.audit_results["deployed_services"].items():
            if details["status"] in ["degraded", "stopped"]:
                degraded_services.append(service)
                
        if degraded_services:
            print(f"\n🔧 Services Needing Attention ({len(degraded_services)}):")
            for service in degraded_services:
                details = self.audit_results["deployed_services"][service]
                print(f"   ⚠️ {service} ({details['platform']}) - {details['status']}")
    
    async def generate_improvement_recommendations(self):
        """Generate specific improvement recommendations"""
        print("\n🎯 IMPROVEMENT RECOMMENDATIONS")
        print("-" * 60)
        
        recommendations = []
        
        # Critical missing services
        critical_missing = [name for name, details in self.audit_results["undeployed_services"].items() 
                          if details["priority"] == "critical"]
        
        if critical_missing:
            recommendations.append({
                "priority": "CRITICAL",
                "action": f"Deploy missing critical services: {', '.join(critical_missing)}",
                "impact": "Platform stability and core functionality"
            })
        
        # High priority missing services
        high_missing = [name for name, details in self.audit_results["undeployed_services"].items() 
                       if details["priority"] == "high"]
        
        if high_missing:
            recommendations.append({
                "priority": "HIGH", 
                "action": f"Deploy missing high-priority services: {', '.join(high_missing)}",
                "impact": "Enhanced platform capabilities"
            })
        
        # Fix degraded services
        degraded_count = sum(1 for details in self.audit_results["deployed_services"].values() 
                           if details["status"] in ["degraded", "stopped"])
        
        if degraded_count > 0:
            recommendations.append({
                "priority": "HIGH",
                "action": f"Fix {degraded_count} degraded/stopped services",
                "impact": "Improve platform stability and performance"
            })
        
        # Platform consolidation
        platforms = set(details["platform"] for details in self.audit_results["deployed_services"].values())
        if len(platforms) > 2:
            recommendations.append({
                "priority": "MEDIUM",
                "action": "Consolidate services onto fewer platforms (prefer Kubernetes)",
                "impact": "Simplified operations and management"
            })
        
        # Monitoring and observability
        monitoring_services = sum(1 for name in self.audit_results["deployed_services"].keys() 
                                if "monitor" in name.lower() or "grafana" in name.lower() or "prometheus" in name.lower())
        
        if monitoring_services < 3:
            recommendations.append({
                "priority": "MEDIUM",
                "action": "Deploy comprehensive monitoring stack (Prometheus, Grafana, AlertManager)",
                "impact": "Better visibility and operational awareness"
            })
        
        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            priority_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
            print(f"{i:2d}. {priority_icon.get(rec['priority'], '⚫')} [{rec['priority']}] {rec['action']}")
            print(f"    Impact: {rec['impact']}")
        
        self.audit_results["recommendations"] = recommendations
    
    async def create_deployment_priority_matrix(self):
        """Create deployment priority matrix"""
        print("\n📋 DEPLOYMENT PRIORITY MATRIX")
        print("-" * 60)
        
        # Group undeployed services by priority
        priority_groups = {"critical": [], "high": [], "medium": [], "low": []}
        
        for service, details in self.audit_results["undeployed_services"].items():
            priority = details["priority"]
            if priority in priority_groups:
                priority_groups[priority].append({
                    "name": service,
                    "category": details["category"],
                    "type": details["type"]
                })
        
        deployment_order = []
        for priority in ["critical", "high", "medium", "low"]:
            if priority_groups[priority]:
                print(f"\n{priority.upper()} PRIORITY ({len(priority_groups[priority])} services):")
                for service in priority_groups[priority]:
                    category = service["category"].replace("_", " ").title()
                    print(f"  • {service['name']} ({service['type']}) - {category}")
                    deployment_order.append(service['name'])
        
        if deployment_order:
            print(f"\n🚀 Recommended Deployment Order:")
            for i, service in enumerate(deployment_order, 1):
                print(f"  {i:2d}. {service}")
    
    async def generate_final_recommendations(self):
        """Generate final comprehensive recommendations"""
        print("\n🎉 FINAL AUDIT SUMMARY & NEXT STEPS")
        print("=" * 80)
        
        total_services = len(self.audit_results["deployed_services"])
        missing_services = len(self.audit_results["undeployed_services"])
        
        print(f"📊 Service Inventory:")
        print(f"   • Deployed Services: {total_services}")
        print(f"   • Missing Services: {missing_services}")
        print(f"   • Total XORB Ecosystem: {total_services + missing_services}")
        
        # Quick wins
        print(f"\n⚡ QUICK WINS:")
        print("   1. Fix degraded Kubernetes services with: kubectl rollout restart deployment/<name> -n xorb-platform")
        print("   2. Access working dashboard: kubectl port-forward service/security-dashboard 8080:80 -n xorb-platform")
        print("   3. Check API gateway: kubectl port-forward service/simple-api-gateway 8081:8080 -n xorb-platform")
        print("   4. Monitor basic metrics: kubectl port-forward service/basic-monitoring 9090:9090 -n xorb-platform")
        
        # Strategic improvements
        print(f"\n🎯 STRATEGIC IMPROVEMENTS:")
        print("   1. Migrate Docker services to Kubernetes for better orchestration")
        print("   2. Implement proper health checks and readiness probes")
        print("   3. Setup persistent storage for stateful services")
        print("   4. Deploy full monitoring stack (Prometheus + Grafana)")
        print("   5. Implement service mesh for better observability")
        print("   6. Add automated backup and disaster recovery")
        print("   7. Setup CI/CD pipeline for continuous deployment")
        print("   8. Implement comprehensive security policies")
        
        # Save detailed audit report
        audit_report = {
            "audit_id": f"AUDIT-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_services": total_services,
                "missing_services": missing_services,
                "health_status": "OPERATIONAL",
                "platforms": list(set(details["platform"] for details in self.audit_results["deployed_services"].values()))
            },
            "deployed_services": self.audit_results["deployed_services"],
            "undeployed_services": self.audit_results["undeployed_services"],
            "recommendations": self.audit_results["recommendations"]
        }
        
        report_file = f"/root/Xorb/logs/complete-service-audit-{audit_report['audit_id']}.json"
        with open(report_file, 'w') as f:
            json.dump(audit_report, f, indent=2)
        
        print(f"\n📋 Detailed audit report saved: {report_file}")
        print("=" * 80)

async def main():
    """Main audit function"""
    auditor = XORBServiceAuditor()
    await auditor.conduct_full_audit()

if __name__ == "__main__":
    asyncio.run(main())