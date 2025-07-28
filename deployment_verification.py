#!/usr/bin/env python3
"""
XORB Ecosystem - Comprehensive Deployment Verification
Production-Ready Testing and Validation Suite
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/root/Xorb/deployment_verification.log')
    ]
)
logger = logging.getLogger('XORB-DEPLOYMENT-VERIFICATION')

class XORBDeploymentVerifier:
    """Comprehensive XORB deployment verification and testing suite"""

    def __init__(self):
        self.verification_id = f"DEPLOY-VERIFY-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = {
            'verification_id': self.verification_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'running',
            'checks': {},
            'summary': {},
            'recommendations': []
        }
        logger.info(f"🚀 Starting XORB deployment verification: {self.verification_id}")

    def run_command(self, command: str, timeout: int = 30) -> tuple[bool, str]:
        """Run a shell command and return success status and output"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout} seconds"
        except Exception as e:
            return False, str(e)

    def check_repository_status(self) -> dict:
        """Verify repository and git status"""
        logger.info("🔍 Checking repository status...")

        checks = {}

        # Check git status
        success, output = self.run_command("git status --porcelain")
        checks['git_clean'] = {
            'status': 'pass' if success and not output.strip() else 'fail',
            'message': 'Working directory clean' if success and not output.strip() else 'Uncommitted changes detected',
            'details': output.strip() if output.strip() else 'No pending changes'
        }

        # Check current branch
        success, output = self.run_command("git branch --show-current")
        checks['current_branch'] = {
            'status': 'pass' if success else 'fail',
            'message': f"Current branch: {output.strip()}" if success else 'Failed to get current branch',
            'details': output.strip()
        }

        # Check remote status
        success, output = self.run_command("git remote -v")
        checks['remote_configured'] = {
            'status': 'pass' if success and 'origin' in output else 'fail',
            'message': 'Remote origin configured' if success and 'origin' in output else 'No remote origin found',
            'details': output.strip()
        }

        # Check latest commits
        success, output = self.run_command("git log --oneline -5")
        checks['recent_commits'] = {
            'status': 'pass' if success else 'fail',
            'message': 'Recent commits available' if success else 'Failed to get commit history',
            'details': output.strip()
        }

        return checks

    def check_security_compliance(self) -> dict:
        """Verify security compliance and secret management"""
        logger.info("🔒 Checking security compliance...")

        checks = {}

        # Check for hardcoded secrets
        success, output = self.run_command(
            'grep -r "nvapi-[A-Za-z0-9_-]" . --exclude-dir=.git --exclude-dir=backups --exclude="*.example" --exclude="*.md" || echo "NO_SECRETS_FOUND"'
        )
        checks['no_hardcoded_secrets'] = {
            'status': 'pass' if 'NO_SECRETS_FOUND' in output else 'fail',
            'message': 'No hardcoded API keys found' if 'NO_SECRETS_FOUND' in output else 'Hardcoded secrets detected',
            'details': output.strip() if 'NO_SECRETS_FOUND' not in output else 'Security scan clean'
        }

        # Check .env.example exists
        checks['env_example_exists'] = {
            'status': 'pass' if os.path.exists('.env.example') else 'fail',
            'message': 'Environment template exists' if os.path.exists('.env.example') else 'Missing .env.example template'
        }

        # Check .gitignore for secrets
        if os.path.exists('.gitignore'):
            with open('.gitignore') as f:
                gitignore_content = f.read()
            has_secret_patterns = any(pattern in gitignore_content for pattern in ['*api_key*', '*token*', '.env.*'])
            checks['gitignore_secrets'] = {
                'status': 'pass' if has_secret_patterns else 'warn',
                'message': 'Gitignore excludes secrets' if has_secret_patterns else 'Gitignore may not exclude all secret patterns'
            }

        return checks

    def check_ci_cd_pipeline(self) -> dict:
        """Verify GitHub Actions CI/CD pipeline"""
        logger.info("🔄 Checking CI/CD pipeline status...")

        checks = {}

        # Check CI/CD workflow file exists
        ci_file_path = '.github/workflows/ci.yml'
        checks['ci_workflow_exists'] = {
            'status': 'pass' if os.path.exists(ci_file_path) else 'fail',
            'message': 'CI/CD workflow configured' if os.path.exists(ci_file_path) else 'Missing CI/CD workflow'
        }

        # Check workflow content for security scanning
        if os.path.exists(ci_file_path):
            with open(ci_file_path) as f:
                workflow_content = f.read()
            has_security_scan = 'hardcoded secrets' in workflow_content.lower()
            checks['security_scanning_enabled'] = {
                'status': 'pass' if has_security_scan else 'warn',
                'message': 'Security scanning configured' if has_security_scan else 'No security scanning detected in CI'
            }

        # Test GitHub API access (if available)
        try:
            response = requests.get(
                "https://api.github.com/repos/losa201/Xorb/actions/runs",
                timeout=10,
                params={'per_page': 1}
            )
            if response.status_code == 200:
                runs_data = response.json()
                if runs_data.get('workflow_runs'):
                    latest_run = runs_data['workflow_runs'][0]
                    checks['latest_ci_run'] = {
                        'status': 'pass' if latest_run.get('conclusion') == 'success' else 'warn',
                        'message': f"Latest CI run: {latest_run.get('conclusion', 'unknown')}",
                        'details': f"Run #{latest_run.get('run_number', 'unknown')} on {latest_run.get('head_branch', 'unknown')}"
                    }
                else:
                    checks['latest_ci_run'] = {
                        'status': 'warn',
                        'message': 'No CI runs found'
                    }
            else:
                checks['github_api_access'] = {
                    'status': 'warn',
                    'message': f'GitHub API returned {response.status_code}'
                }
        except Exception as e:
            checks['github_api_access'] = {
                'status': 'warn',
                'message': f'GitHub API access failed: {str(e)}'
            }

        return checks

    def check_docker_deployment(self) -> dict:
        """Verify Docker deployment configuration"""
        logger.info("🐳 Checking Docker deployment...")

        checks = {}

        # Check docker-compose.yml exists
        checks['docker_compose_exists'] = {
            'status': 'pass' if os.path.exists('docker-compose.yml') else 'fail',
            'message': 'Docker Compose configuration exists' if os.path.exists('docker-compose.yml') else 'Missing docker-compose.yml'
        }

        # Validate docker-compose configuration
        success, output = self.run_command("docker-compose config --services")
        if success:
            services = output.strip().split('\n')
            checks['docker_compose_valid'] = {
                'status': 'pass',
                'message': f'Docker Compose valid ({len(services)} services)',
                'details': ', '.join(services)
            }
        else:
            checks['docker_compose_valid'] = {
                'status': 'fail',
                'message': 'Docker Compose configuration invalid',
                'details': output
            }

        # Check if Docker is running
        success, output = self.run_command("docker info")
        checks['docker_running'] = {
            'status': 'pass' if success else 'fail',
            'message': 'Docker daemon running' if success else 'Docker daemon not accessible',
            'details': 'Docker available' if success else output
        }

        return checks

    def check_documentation(self) -> dict:
        """Verify documentation completeness"""
        logger.info("📚 Checking documentation...")

        checks = {}

        # Check key documentation files
        docs = {
            'README.md': 'Main documentation',
            'CLAUDE.md': 'Development guide',
            'SECURITY_DEPLOYMENT_COMPLETE.md': 'Security documentation',
            'GITHUB_DEPLOYMENT_COMPLETE.md': 'Deployment guide'
        }

        for doc_file, description in docs.items():
            exists = os.path.exists(doc_file)
            checks[f'doc_{doc_file.lower().replace(".", "_")}'] = {
                'status': 'pass' if exists else 'warn',
                'message': f'{description} exists' if exists else f'Missing {description}',
                'details': doc_file
            }

        # Check README.md for key sections
        if os.path.exists('README.md'):
            with open('README.md') as f:
                readme_content = f.read().lower()

            key_sections = ['quick start', 'installation', 'security', 'contributing']
            missing_sections = [section for section in key_sections if section not in readme_content]

            checks['readme_completeness'] = {
                'status': 'pass' if not missing_sections else 'warn',
                'message': 'README contains key sections' if not missing_sections else f'README missing: {", ".join(missing_sections)}'
            }

        return checks

    def check_project_structure(self) -> dict:
        """Verify project structure and organization"""
        logger.info("📁 Checking project structure...")

        checks = {}

        # Check key directories
        key_dirs = [
            'packages/xorb_core',
            'services',
            'gitops',
            'scripts',
            'tests'
        ]

        missing_dirs = [dir_path for dir_path in key_dirs if not os.path.exists(dir_path)]
        checks['project_structure'] = {
            'status': 'pass' if not missing_dirs else 'warn',
            'message': 'Core directories present' if not missing_dirs else f'Missing directories: {", ".join(missing_dirs)}'
        }

        # Check for Python package structure
        xorb_core_init = 'packages/xorb_core/xorb_core/__init__.py'
        checks['python_package_structure'] = {
            'status': 'pass' if os.path.exists(xorb_core_init) else 'warn',
            'message': 'Python package structure valid' if os.path.exists(xorb_core_init) else 'Missing Python package __init__.py'
        }

        return checks

    def generate_recommendations(self) -> list[str]:
        """Generate deployment recommendations based on check results"""
        recommendations = []

        # Check overall health
        all_checks = {}
        for category in self.results['checks'].values():
            all_checks.update(category)

        fail_count = sum(1 for check in all_checks.values() if check['status'] == 'fail')
        warn_count = sum(1 for check in all_checks.values() if check['status'] == 'warn')

        if fail_count > 0:
            recommendations.append(f"🔴 CRITICAL: {fail_count} critical issues need immediate attention")

        if warn_count > 0:
            recommendations.append(f"🟡 WARNING: {warn_count} warnings should be addressed")

        # Specific recommendations
        if any(check['status'] == 'fail' for check in self.results['checks'].get('security', {}).values()):
            recommendations.append("🔒 Review and fix security compliance issues before production deployment")

        if any(check['status'] == 'fail' for check in self.results['checks'].get('docker', {}).values()):
            recommendations.append("🐳 Fix Docker configuration issues for reliable container deployment")

        if fail_count == 0 and warn_count <= 2:
            recommendations.append("✅ Deployment appears ready for production")
            recommendations.append("🚀 Consider running integration tests and load testing")

        return recommendations

    def run_verification(self) -> dict:
        """Run complete deployment verification"""
        logger.info("🛡️ XORB Ecosystem - Deployment Verification Starting...")

        try:
            # Run all verification checks
            self.results['checks']['repository'] = self.check_repository_status()
            self.results['checks']['security'] = self.check_security_compliance()
            self.results['checks']['cicd'] = self.check_ci_cd_pipeline()
            self.results['checks']['docker'] = self.check_docker_deployment()
            self.results['checks']['documentation'] = self.check_documentation()
            self.results['checks']['structure'] = self.check_project_structure()

            # Calculate summary
            all_checks = {}
            for category in self.results['checks'].values():
                all_checks.update(category)

            total_checks = len(all_checks)
            passed = sum(1 for check in all_checks.values() if check['status'] == 'pass')
            warnings = sum(1 for check in all_checks.values() if check['status'] == 'warn')
            failed = sum(1 for check in all_checks.values() if check['status'] == 'fail')

            self.results['summary'] = {
                'total_checks': total_checks,
                'passed': passed,
                'warnings': warnings,
                'failed': failed,
                'success_rate': round((passed / total_checks) * 100, 1) if total_checks > 0 else 0
            }

            # Generate recommendations
            self.results['recommendations'] = self.generate_recommendations()

            # Determine overall status
            if failed == 0 and warnings <= 2:
                self.results['status'] = 'ready'
            elif failed == 0:
                self.results['status'] = 'caution'
            else:
                self.results['status'] = 'blocked'

            logger.info(f"✅ Verification complete: {self.results['status']}")
            return self.results

        except Exception as e:
            logger.error(f"❌ Verification failed: {str(e)}")
            self.results['status'] = 'error'
            self.results['error'] = str(e)
            return self.results

    def print_results(self):
        """Print formatted verification results"""
        print("\n" + "=" * 80)
        print("🛡️  XORB ECOSYSTEM - DEPLOYMENT VERIFICATION REPORT")
        print("=" * 80)
        print(f"Verification ID: {self.results['verification_id']}")
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"Overall Status: {self.results['status'].upper()}")
        print()

        # Print summary
        summary = self.results['summary']
        print("📊 SUMMARY:")
        print(f"   Total Checks: {summary['total_checks']}")
        print(f"   ✅ Passed: {summary['passed']}")
        print(f"   🟡 Warnings: {summary['warnings']}")
        print(f"   ❌ Failed: {summary['failed']}")
        print(f"   Success Rate: {summary['success_rate']}%")
        print()

        # Print detailed results by category
        for category_name, category_checks in self.results['checks'].items():
            print(f"🔍 {category_name.upper()} CHECKS:")
            for check_name, check_result in category_checks.items():
                status_icon = {'pass': '✅', 'warn': '🟡', 'fail': '❌'}.get(check_result['status'], '❓')
                print(f"   {status_icon} {check_result['message']}")
                if 'details' in check_result and check_result['details']:
                    print(f"      Details: {check_result['details']}")
            print()

        # Print recommendations
        if self.results['recommendations']:
            print("💡 RECOMMENDATIONS:")
            for recommendation in self.results['recommendations']:
                print(f"   {recommendation}")
            print()

        print("=" * 80)

    def save_results(self, filename: str = None):
        """Save verification results to JSON file"""
        if not filename:
            filename = f"/root/Xorb/deployment_verification_{self.verification_id}.json"

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"📝 Results saved to: {filename}")
        return filename

def main():
    """Main execution function"""
    verifier = XORBDeploymentVerifier()
    results = verifier.run_verification()
    verifier.print_results()
    verifier.save_results()

    # Exit with appropriate code
    if results['status'] in ['ready', 'caution']:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
