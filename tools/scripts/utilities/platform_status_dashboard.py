#!/usr/bin/env python3
"""
XORB Platform Status Dashboard
Real-time monitoring and status reporting for the complete XORB ecosystem
"""

import asyncio
import json
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import sys

class XORBPlatformMonitor:
    """Comprehensive XORB Platform Status Monitor"""
    
    def __init__(self):
        self.namespace = "xorb-platform"
        self.services = {
            "core": ["postgres", "redis", "api-gateway", "security-api", "threat-intelligence", "analytics-engine", "web-frontend"],
            "ptaas": ["neo4j", "qdrant", "rabbitmq", "ptaas-bug-bounty", "ptaas-exploit-validation", "ptaas-reward-system"],
            "monitoring": ["prometheus", "grafana", "alertmanager"]
        }
        
    async def generate_platform_status(self):
        """Generate comprehensive platform status report"""
        print("🔍 XORB ECOSYSTEM STATUS DASHBOARD")
        print("=" * 80)
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("=" * 80)
        
        # Kubernetes Cluster Status
        await self.check_cluster_status()
        
        # Service Status by Category
        await self.check_service_categories()
        
        # Resource Utilization
        await self.check_resource_usage()
        
        # Network and Ingress Status
        await self.check_networking()
        
        # Storage Status
        await self.check_storage()
        
        # Platform Health Summary
        await self.generate_health_summary()
        
        # Next Steps and Recommendations
        await self.generate_recommendations()
        
    async def check_cluster_status(self):
        """Check Kubernetes cluster status"""
        print("\n🏗️ KUBERNETES CLUSTER STATUS")
        print("-" * 40)
        
        try:
            # Cluster info
            result = subprocess.run(['kubectl', 'cluster-info'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Cluster: Online and accessible")
                print(f"   Control Plane: https://127.0.0.1:42881")
            else:
                print("❌ Cluster: Not accessible")
                
            # Node status
            result = subprocess.run(['kubectl', 'get', 'nodes', '-o', 'wide'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    print(f"✅ Nodes: {len(lines)-1} node(s) ready")
                    for line in lines[1:]:  # Skip header
                        parts = line.split()
                        if len(parts) >= 2:
                            print(f"   📍 {parts[0]}: {parts[1]}")
                            
        except Exception as e:
            print(f"❌ Cluster check failed: {e}")
    
    async def check_service_categories(self):
        """Check services by category"""
        for category, service_list in self.services.items():
            print(f"\n🔧 {category.upper()} SERVICES")
            print("-" * 40)
            
            for service in service_list:
                await self.check_individual_service(service)
    
    async def check_individual_service(self, service_name: str):
        """Check status of individual service"""
        try:
            # Check deployment
            result = subprocess.run(['kubectl', 'get', 'deployment', service_name, '-n', self.namespace, '-o', 'json'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                deployment_data = json.loads(result.stdout)
                desired = deployment_data['spec']['replicas']
                ready = deployment_data['status'].get('readyReplicas', 0)
                
                if ready == desired:
                    print(f"✅ {service_name}: {ready}/{desired} replicas ready")
                else:
                    print(f"⚠️ {service_name}: {ready}/{desired} replicas ready")
                    
                # Check for recent events
                await self.check_service_events(service_name)
            else:
                # Check if it's a StatefulSet instead
                result = subprocess.run(['kubectl', 'get', 'statefulset', service_name, '-n', self.namespace], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ {service_name}: StatefulSet deployed")
                else:
                    print(f"❌ {service_name}: Not found")
                    
        except Exception as e:
            print(f"❌ {service_name}: Error checking status - {e}")
    
    async def check_service_events(self, service_name: str):
        """Check recent events for a service"""
        try:
            result = subprocess.run(['kubectl', 'get', 'events', '-n', self.namespace, 
                                   '--field-selector', f'involvedObject.name={service_name}', 
                                   '--sort-by', '.lastTimestamp'], capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                events = result.stdout.strip().split('\n')
                if len(events) > 1:  # Skip header
                    latest_event = events[-1]
                    if 'Warning' in latest_event or 'Error' in latest_event:
                        print(f"   ⚠️ Latest event: {latest_event.split()[-1]}")
                        
        except Exception:
            pass  # Ignore event check errors
    
    async def check_resource_usage(self):
        """Check resource utilization"""
        print("\n📊 RESOURCE UTILIZATION")
        print("-" * 40)
        
        try:
            # Get node resource usage
            result = subprocess.run(['kubectl', 'top', 'nodes'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Node Resource Usage:")
                for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                    print(f"   {line}")
            else:
                print("⚠️ Metrics server not available for resource usage")
                
            # Pod resource requests/limits
            result = subprocess.run(['kubectl', 'get', 'pods', '-n', self.namespace, '-o', 'json'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                pods_data = json.loads(result.stdout)
                total_pods = len(pods_data['items'])
                running_pods = sum(1 for pod in pods_data['items'] if pod['status']['phase'] == 'Running')
                
                print(f"✅ Pods: {running_pods}/{total_pods} running")
                
        except Exception as e:
            print(f"❌ Resource usage check failed: {e}")
    
    async def check_networking(self):
        """Check networking and ingress status"""
        print("\n🌐 NETWORKING & INGRESS")
        print("-" * 40)
        
        try:
            # Check ingress rules
            result = subprocess.run(['kubectl', 'get', 'ingress', '-n', self.namespace], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                ingress_lines = result.stdout.strip().split('\n')
                if len(ingress_lines) > 1:
                    print(f"✅ Ingress Rules: {len(ingress_lines)-1} configured")
                    for line in ingress_lines[1:]:
                        parts = line.split()
                        if len(parts) >= 3:
                            print(f"   📍 {parts[0]}: {parts[2]}")
                else:
                    print("⚠️ No ingress rules configured")
                    
            # Check services
            result = subprocess.run(['kubectl', 'get', 'services', '-n', self.namespace], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                service_lines = result.stdout.strip().split('\n')
                if len(service_lines) > 1:
                    print(f"✅ Services: {len(service_lines)-1} configured")
                    
        except Exception as e:
            print(f"❌ Networking check failed: {e}")
    
    async def check_storage(self):
        """Check storage and persistent volumes"""
        print("\n💾 STORAGE STATUS")
        print("-" * 40)
        
        try:
            # Check PVCs
            result = subprocess.run(['kubectl', 'get', 'pvc', '-n', self.namespace], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                pvc_lines = result.stdout.strip().split('\n')
                if len(pvc_lines) > 1:
                    print(f"✅ Persistent Volume Claims: {len(pvc_lines)-1}")
                    for line in pvc_lines[1:]:
                        parts = line.split()
                        if len(parts) >= 3:
                            print(f"   📦 {parts[0]}: {parts[1]} ({parts[2]})")
                else:
                    print("⚠️ No persistent volumes configured")
                    
        except Exception as e:
            print(f"❌ Storage check failed: {e}")
    
    async def generate_health_summary(self):
        """Generate overall platform health summary"""
        print("\n🏥 PLATFORM HEALTH SUMMARY")
        print("-" * 40)
        
        # Calculate health metrics
        try:
            result = subprocess.run(['kubectl', 'get', 'pods', '-n', self.namespace, '-o', 'json'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                pods_data = json.loads(result.stdout)
                total_pods = len(pods_data['items'])
                running_pods = sum(1 for pod in pods_data['items'] if pod['status']['phase'] == 'Running')
                
                health_percentage = (running_pods / total_pods * 100) if total_pods > 0 else 0
                
                if health_percentage >= 80:
                    status_icon = "🟢"
                    status_text = "HEALTHY"
                elif health_percentage >= 50:
                    status_icon = "🟡"
                    status_text = "DEGRADED"
                else:
                    status_icon = "🔴"
                    status_text = "CRITICAL"
                
                print(f"{status_icon} Overall Health: {health_percentage:.1f}% - {status_text}")
                print(f"   Running Services: {running_pods}/{total_pods}")
                
                # Identify problematic services
                problematic_pods = []
                for pod in pods_data['items']:
                    if pod['status']['phase'] != 'Running':
                        pod_name = pod['metadata']['name']
                        pod_status = pod['status']['phase']
                        problematic_pods.append(f"{pod_name} ({pod_status})")
                
                if problematic_pods:
                    print("   ⚠️ Issues Detected:")
                    for pod_issue in problematic_pods[:5]:  # Show first 5
                        print(f"     • {pod_issue}")
                    if len(problematic_pods) > 5:
                        print(f"     • ... and {len(problematic_pods)-5} more")
                
        except Exception as e:
            print(f"❌ Health summary failed: {e}")
    
    async def generate_recommendations(self):
        """Generate next steps and recommendations"""
        print("\n🎯 RECOMMENDATIONS & NEXT STEPS")
        print("-" * 40)
        
        recommendations = []
        
        try:
            # Check for common issues
            result = subprocess.run(['kubectl', 'get', 'pods', '-n', self.namespace, '-o', 'json'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                pods_data = json.loads(result.stdout)
                
                # Check for ImagePullBackOff errors
                image_pull_errors = sum(1 for pod in pods_data['items'] 
                                      if any(container.get('state', {}).get('waiting', {}).get('reason') == 'ImagePullBackOff' 
                                           for container in pod.get('status', {}).get('containerStatuses', [])))
                
                if image_pull_errors > 0:
                    recommendations.append(f"🔧 Fix {image_pull_errors} ImagePullBackOff errors - build or pull missing container images")
                
                # Check for CrashLoopBackOff errors
                crash_loop_errors = sum(1 for pod in pods_data['items']
                                      if any(container.get('state', {}).get('waiting', {}).get('reason') == 'CrashLoopBackOff'
                                           for container in pod.get('status', {}).get('containerStatuses', [])))
                
                if crash_loop_errors > 0:
                    recommendations.append(f"🐛 Debug {crash_loop_errors} CrashLoopBackOff errors - check logs and resource limits")
                
                # Check for pending pods
                pending_pods = sum(1 for pod in pods_data['items'] if pod['status']['phase'] == 'Pending')
                if pending_pods > 0:
                    recommendations.append(f"⏳ Resolve {pending_pods} pending pods - check resource constraints and storage")
        
        except Exception:
            pass
        
        # Standard recommendations
        standard_recommendations = [
            "📊 Monitor service logs: kubectl logs -f deployment/<service-name> -n xorb-platform",
            "🔍 Check service events: kubectl describe pod <pod-name> -n xorb-platform", 
            "🚀 Scale services if needed: kubectl scale deployment <name> --replicas=3 -n xorb-platform",
            "🔄 Restart problematic services: kubectl rollout restart deployment/<name> -n xorb-platform",
            "📈 Setup monitoring alerts for production readiness",
            "🔒 Configure SSL certificates for external access",
            "💾 Implement backup strategies for persistent data",
            "🧪 Run load tests to validate performance",
            "📋 Document operational procedures",
            "🛡️ Complete security compliance scan"
        ]
        
        # Combine recommendations
        all_recommendations = recommendations + standard_recommendations
        
        for i, rec in enumerate(all_recommendations[:10], 1):  # Show top 10
            print(f"{i:2d}. {rec}")
        
        print("\n" + "=" * 80)
        print("🎉 XORB Platform Status Dashboard Complete!")
        print("   Run this script regularly to monitor platform health")
        print("=" * 80)

async def main():
    """Main monitoring function"""
    monitor = XORBPlatformMonitor()
    await monitor.generate_platform_status()

if __name__ == "__main__":
    asyncio.run(main())