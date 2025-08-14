#!/usr/bin/env python3
"""
XORB Platform Status & Performance Monitor
Real-time platform health and performance validation
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import sys

class XORBPlatformMonitor:
    def __init__(self):
        self.services = {
            'api': 'http://localhost:8000',
            'temporal': 'http://localhost:8233',
            'prometheus': 'http://localhost:9090',
            'grafana': 'http://localhost:3000',
            'postgres': 'localhost:5432',
            'redis': 'localhost:6380'
        }

        self.health_status = {}
        self.performance_metrics = {}

    async def check_service_health(self, service_name: str, url: str) -> Dict[str, Any]:
        """Check individual service health"""
        start_time = time.time()

        try:
            if service_name in ['postgres', 'redis']:
                # For databases, just check if port is accessible
                import socket
                host, port = url.split(':')
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, int(port)))
                sock.close()

                if result == 0:
                    response_time = (time.time() - start_time) * 1000
                    return {
                        'status': 'healthy',
                        'response_time_ms': round(response_time, 2),
                        'message': f'{service_name} port accessible'
                    }
                else:
                    return {
                        'status': 'unhealthy',
                        'response_time_ms': None,
                        'message': f'{service_name} port not accessible'
                    }
            else:
                # For HTTP services
                async with aiohttp.ClientSession() as session:
                    health_url = f"{url}/health" if service_name == 'api' else url

                    async with session.get(health_url, timeout=10) as response:
                        response_time = (time.time() - start_time) * 1000

                        if response.status == 200:
                            return {
                                'status': 'healthy',
                                'response_time_ms': round(response_time, 2),
                                'message': f'{service_name} responding normally'
                            }
                        else:
                            return {
                                'status': 'degraded',
                                'response_time_ms': round(response_time, 2),
                                'message': f'{service_name} returned status {response.status}'
                            }

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return {
                'status': 'unhealthy',
                'response_time_ms': round(response_time, 2),
                'message': f'{service_name} error: {str(e)[:100]}'
            }

    async def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """Run comprehensive platform health check"""
        print("🏥 Running XORB Platform Health Check...")

        # Check all services concurrently
        tasks = []
        for service_name, url in self.services.items():
            task = self.check_service_health(service_name, url)
            tasks.append((service_name, task))

        results = {}
        for service_name, task in tasks:
            try:
                result = await task
                results[service_name] = result

                # Print status
                status_icon = "✅" if result['status'] == 'healthy' else "⚠️" if result['status'] == 'degraded' else "❌"
                response_time = f" ({result['response_time_ms']}ms)" if result['response_time_ms'] else ""
                print(f"   {status_icon} {service_name.upper()}: {result['status']}{response_time}")

            except Exception as e:
                results[service_name] = {
                    'status': 'error',
                    'response_time_ms': None,
                    'message': str(e)
                }
                print(f"   ❌ {service_name.upper()}: error - {str(e)}")

        # Calculate overall health
        healthy_count = sum(1 for r in results.values() if r['status'] == 'healthy')
        total_count = len(results)
        health_percentage = (healthy_count / total_count) * 100

        overall_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': health_percentage,
            'services_healthy': healthy_count,
            'total_services': total_count,
            'service_details': results
        }

        return overall_status

    async def generate_platform_report(self) -> Dict[str, Any]:
        """Generate comprehensive platform status report"""
        health_check = await self.run_comprehensive_health_check()

        # Platform information
        platform_info = {
            'platform_name': 'XORB Cybersecurity Platform',
            'version': '2.1.0',
            'deployment_mode': 'development',
            'region': 'eu-central'
        }

        # Performance summary
        avg_response_time = None
        healthy_services = [s for s in health_check['service_details'].values()
                          if s['status'] == 'healthy' and s['response_time_ms']]

        if healthy_services:
            avg_response_time = sum(s['response_time_ms'] for s in healthy_services) / len(healthy_services)

        performance_summary = {
            'average_response_time_ms': round(avg_response_time, 2) if avg_response_time else None,
            'fastest_service': min(healthy_services, key=lambda x: x['response_time_ms'])['response_time_ms'] if healthy_services else None,
            'slowest_service': max(healthy_services, key=lambda x: x['response_time_ms'])['response_time_ms'] if healthy_services else None
        }

        # Recommendations
        recommendations = []
        if health_check['overall_health'] < 100:
            unhealthy_services = [name for name, details in health_check['service_details'].items()
                                if details['status'] != 'healthy']
            recommendations.append(f"Check unhealthy services: {', '.join(unhealthy_services)}")

        if avg_response_time and avg_response_time > 1000:
            recommendations.append("Consider performance optimization - high response times detected")

        if not recommendations:
            recommendations.append("Platform operating optimally - no issues detected")

        return {
            'platform_info': platform_info,
            'health_check': health_check,
            'performance_summary': performance_summary,
            'recommendations': recommendations,
            'report_generated': datetime.now().isoformat()
        }

    def print_status_report(self, report: Dict[str, Any]):
        """Print formatted status report"""
        print("\n" + "="*80)
        print("🛡️  XORB PLATFORM STATUS REPORT")
        print("="*80)

        # Platform info
        info = report['platform_info']
        print(f"Platform: {info['platform_name']} v{info['version']}")
        print(f"Mode: {info['deployment_mode']} | Region: {info['region']}")

        # Health summary
        health = report['health_check']
        health_icon = "🟢" if health['overall_health'] >= 90 else "🟡" if health['overall_health'] >= 70 else "🔴"
        print(f"\n{health_icon} Overall Health: {health['overall_health']:.1f}% ({health['services_healthy']}/{health['total_services']} services)")

        # Performance summary
        perf = report['performance_summary']
        if perf['average_response_time_ms']:
            print(f"⚡ Average Response Time: {perf['average_response_time_ms']:.2f}ms")
            print(f"🏃 Fastest: {perf['fastest_service']:.2f}ms | 🐌 Slowest: {perf['slowest_service']:.2f}ms")

        # Recommendations
        print(f"\n📋 Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")

        print("\n" + "="*80)
        print(f"Report generated: {report['report_generated']}")
        print("="*80)

async def main():
    """Main execution function"""
    monitor = XORBPlatformMonitor()

    try:
        # Generate comprehensive report
        report = await monitor.generate_platform_report()

        # Print formatted report
        monitor.print_status_report(report)

        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"xorb_platform_status_{timestamp}.json"

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n💾 Detailed report saved: {report_file}")

        # Exit with appropriate code
        health_percentage = report['health_check']['overall_health']
        if health_percentage >= 90:
            print("✅ Platform status: EXCELLENT")
            sys.exit(0)
        elif health_percentage >= 70:
            print("⚠️  Platform status: GOOD")
            sys.exit(0)
        else:
            print("❌ Platform status: NEEDS ATTENTION")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error running platform check: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
