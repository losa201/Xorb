#!/usr/bin/env python3
"""
XORB Deployment Readiness Check
Real-time validation of live deployment functionality
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('XORB-READINESS-CHECK')

class XORBReadinessChecker:
    """Live deployment readiness validation"""

    def __init__(self):
        self.check_id = f"READINESS-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = {
            'check_id': self.check_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'running',
            'live_tests': {},
            'summary': {}
        }
        logger.info(f"🚀 Starting deployment readiness check: {self.check_id}")

    def run_command(self, command: str, timeout: int = 30, env: dict = None) -> tuple[bool, str]:
        """Run shell command with optional environment"""
        try:
            cmd_env = os.environ.copy()
            if env:
                cmd_env.update(env)

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=cmd_env
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout} seconds"
        except Exception as e:
            return False, str(e)

    def check_docker_availability(self) -> dict:
        """Check Docker engine availability"""
        logger.info("🐳 Checking Docker availability...")

        checks = {}

        # Check Docker daemon
        success, output = self.run_command("docker info")
        checks['docker_daemon'] = {
            'status': 'pass' if success else 'fail',
            'message': 'Docker daemon running' if success else 'Docker daemon not available',
            'details': 'Available' if success else output[:200]
        }

        # Check Docker Compose
        success, output = self.run_command("docker-compose --version")
        checks['docker_compose'] = {
            'status': 'pass' if success else 'fail',
            'message': 'Docker Compose available' if success else 'Docker Compose not found',
            'details': output.strip() if success else 'Not installed'
        }

        return checks

    def test_core_services_deployment(self) -> dict:
        """Test deployment of core services"""
        logger.info("🚀 Testing core services deployment...")

        checks = {}
        test_env = {'NVIDIA_API_KEY': 'test-deployment-key'}

        # Stop any existing services first
        logger.info("Cleaning up existing containers...")
        success, output = self.run_command("docker-compose down --remove-orphans", env=test_env)

        # Start core services
        logger.info("Starting PostgreSQL and Redis...")
        success, output = self.run_command("docker-compose up -d postgres redis", env=test_env)
        checks['core_services_start'] = {
            'status': 'pass' if success else 'fail',
            'message': 'Core services started successfully' if success else 'Failed to start core services',
            'details': output.strip()
        }

        if success:
            # Wait for services to be ready
            time.sleep(8)

            # Test PostgreSQL connectivity
            success, output = self.run_command("docker exec $(docker ps -q -f name=postgres) pg_isready -U temporal")
            checks['postgres_health'] = {
                'status': 'pass' if success and 'accepting connections' in output else 'fail',
                'message': 'PostgreSQL healthy' if success and 'accepting connections' in output else 'PostgreSQL not responding',
                'details': output.strip()
            }

            # Test Redis connectivity
            success, output = self.run_command("docker exec $(docker ps -q -f name=redis) redis-cli ping")
            checks['redis_health'] = {
                'status': 'pass' if success and 'PONG' in output else 'fail',
                'message': 'Redis healthy' if success and 'PONG' in output else 'Redis not responding',
                'details': output.strip()
            }

            # Check container status
            success, output = self.run_command("docker ps --format 'table {{.Names}}\\t{{.Status}}'")
            running_containers = len([line for line in output.split('\n') if 'xorb_' in line and 'Up' in line])
            checks['container_status'] = {
                'status': 'pass' if running_containers >= 2 else 'fail',
                'message': f'{running_containers} containers running' if running_containers >= 2 else f'Only {running_containers} containers running',
                'details': output.strip()
            }

        return checks

    def test_service_configuration(self) -> dict:
        """Test service configuration and networking"""
        logger.info("🔧 Testing service configuration...")

        checks = {}
        test_env = {'NVIDIA_API_KEY': 'test-deployment-key'}

        # Test Docker Compose configuration validation
        success, output = self.run_command("docker-compose config --services", env=test_env)
        if success:
            services = [s.strip() for s in output.split('\n') if s.strip()]
            checks['compose_validation'] = {
                'status': 'pass',
                'message': f'Configuration valid ({len(services)} services)',
                'details': ', '.join(services)
            }
        else:
            checks['compose_validation'] = {
                'status': 'fail',
                'message': 'Docker Compose configuration invalid',
                'details': output
            }

        # Test network connectivity between services
        if self.results['live_tests'].get('deployment', {}).get('postgres_health', {}).get('status') == 'pass':
            # Test network connectivity from Redis to PostgreSQL
            success, output = self.run_command(
                "docker exec $(docker ps -q -f name=redis) nc -z $(docker ps -q -f name=postgres) 5432"
            )
            checks['inter_service_network'] = {
                'status': 'pass' if success else 'warn',
                'message': 'Inter-service networking functional' if success else 'Network connectivity issues detected',
                'details': 'Services can communicate' if success else output.strip()
            }

        return checks

    def test_environment_configuration(self) -> dict:
        """Test environment variable handling"""
        logger.info("🔐 Testing environment configuration...")

        checks = {}

        # Test .env.example exists and is readable
        if os.path.exists('.env.example'):
            with open('.env.example') as f:
                env_content = f.read()

            required_vars = ['NVIDIA_API_KEY', 'POSTGRES_PASSWORD', 'REDIS_PASSWORD']
            missing_vars = [var for var in required_vars if var not in env_content]

            checks['env_template'] = {
                'status': 'pass' if not missing_vars else 'warn',
                'message': 'Environment template complete' if not missing_vars else f'Missing variables: {", ".join(missing_vars)}'
            }
        else:
            checks['env_template'] = {
                'status': 'fail',
                'message': 'Environment template (.env.example) not found'
            }

        # Test environment variable injection
        test_env = {'NVIDIA_API_KEY': 'test-injection-key'}
        success, output = self.run_command("docker-compose config", env=test_env)

        if success and 'test-injection-key' in output:
            checks['env_injection'] = {
                'status': 'pass',
                'message': 'Environment variable injection working',
                'details': 'Variables properly substituted in configuration'
            }
        else:
            checks['env_injection'] = {
                'status': 'warn',
                'message': 'Environment variable injection may have issues',
                'details': 'Variables not found in configuration output'
            }

        return checks

    def cleanup_test_environment(self) -> dict:
        """Clean up test deployment"""
        logger.info("🧹 Cleaning up test environment...")

        checks = {}
        test_env = {'NVIDIA_API_KEY': 'test-deployment-key'}

        # Stop and remove test containers
        success, output = self.run_command("docker-compose down --remove-orphans", env=test_env)
        checks['cleanup'] = {
            'status': 'pass' if success else 'warn',
            'message': 'Test environment cleaned up' if success else 'Cleanup completed with warnings',
            'details': output.strip()
        }

        # Verify no XORB containers are running
        success, output = self.run_command("docker ps -q -f name=xorb_")
        if not output.strip():
            checks['cleanup_verification'] = {
                'status': 'pass',
                'message': 'No test containers remaining',
                'details': 'Clean state verified'
            }
        else:
            checks['cleanup_verification'] = {
                'status': 'warn',
                'message': 'Some containers may still be running',
                'details': f'Found containers: {output.strip()}'
            }

        return checks

    def generate_readiness_summary(self) -> dict:
        """Generate overall readiness assessment"""
        all_checks = {}
        for category in self.results['live_tests'].values():
            all_checks.update(category)

        total = len(all_checks)
        passed = sum(1 for check in all_checks.values() if check['status'] == 'pass')
        warnings = sum(1 for check in all_checks.values() if check['status'] == 'warn')
        failed = sum(1 for check in all_checks.values() if check['status'] == 'fail')

        success_rate = round((passed / total) * 100, 1) if total > 0 else 0

        # Determine readiness level
        if failed == 0 and warnings <= 1:
            readiness = 'READY'
        elif failed == 0:
            readiness = 'CAUTION'
        elif failed <= 2:
            readiness = 'NEEDS_ATTENTION'
        else:
            readiness = 'NOT_READY'

        return {
            'total_checks': total,
            'passed': passed,
            'warnings': warnings,
            'failed': failed,
            'success_rate': success_rate,
            'readiness_level': readiness,
            'deployment_ready': failed == 0,
            'production_ready': failed == 0 and warnings <= 1
        }

    def run_readiness_check(self) -> dict:
        """Execute complete readiness validation"""
        try:
            # Run all readiness tests
            self.results['live_tests']['docker'] = self.check_docker_availability()
            self.results['live_tests']['deployment'] = self.test_core_services_deployment()
            self.results['live_tests']['configuration'] = self.test_service_configuration()
            self.results['live_tests']['environment'] = self.test_environment_configuration()
            self.results['live_tests']['cleanup'] = self.cleanup_test_environment()

            # Generate summary
            self.results['summary'] = self.generate_readiness_summary()
            self.results['status'] = self.results['summary']['readiness_level'].lower()

            logger.info(f"✅ Readiness check complete: {self.results['status']}")
            return self.results

        except Exception as e:
            logger.error(f"❌ Readiness check failed: {str(e)}")
            self.results['status'] = 'error'
            self.results['error'] = str(e)
            return self.results

    def print_results(self):
        """Print formatted readiness results"""
        print("\n" + "=" * 80)
        print("🚀 XORB DEPLOYMENT READINESS CHECK RESULTS")
        print("=" * 80)
        print(f"Check ID: {self.results['check_id']}")
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"Readiness Level: {self.results['summary']['readiness_level']}")
        print()

        # Print summary
        summary = self.results['summary']
        print("📊 READINESS SUMMARY:")
        print(f"   Total Checks: {summary['total_checks']}")
        print(f"   ✅ Passed: {summary['passed']}")
        print(f"   🟡 Warnings: {summary['warnings']}")
        print(f"   ❌ Failed: {summary['failed']}")
        print(f"   Success Rate: {summary['success_rate']}%")
        print(f"   Deployment Ready: {'✅ YES' if summary['deployment_ready'] else '❌ NO'}")
        print(f"   Production Ready: {'✅ YES' if summary['production_ready'] else '❌ NO'}")
        print()

        # Print detailed results
        for category_name, category_checks in self.results['live_tests'].items():
            print(f"🔍 {category_name.upper()} TESTS:")
            for check_name, check_result in category_checks.items():
                status_icon = {'pass': '✅', 'warn': '🟡', 'fail': '❌'}.get(check_result['status'], '❓')
                print(f"   {status_icon} {check_result['message']}")
                if 'details' in check_result and check_result['details']:
                    print(f"      Details: {check_result['details']}")
            print()

        print("=" * 80)

    def save_results(self, filename: str = None):
        """Save results to JSON file"""
        if not filename:
            filename = f"/root/Xorb/deployment_readiness_{self.check_id}.json"

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"📝 Results saved to: {filename}")
        return filename

def main():
    """Main execution function"""
    checker = XORBReadinessChecker()
    results = checker.run_readiness_check()
    checker.print_results()
    checker.save_results()

    # Exit with appropriate code
    if results['summary']['deployment_ready']:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
