#!/usr/bin/env python3
"""
XORB Environment Validation Script

This script validates that the development environment is properly set up
and all required dependencies are available.
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")

def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_info(message: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")

def check_python_version() -> bool:
    """Check Python version"""
    print_header("Python Environment")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version >= (3, 10):
        print_success(f"Python version: {version_str}")
        return True
    else:
        print_error(f"Python version {version_str} is too old. Required: 3.10+")
        return False

def check_virtual_environment() -> bool:
    """Check if running in virtual environment"""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )

    if in_venv:
        print_success(f"Virtual environment: {sys.prefix}")
        return True
    else:
        print_warning("Not running in virtual environment")
        print_info("Consider creating one: python3 -m venv venv && source venv/bin/activate")
        return False

def check_required_packages() -> bool:
    """Check if required Python packages are installed"""
    print_header("Python Dependencies")

    required_packages = {
        'fastapi': '0.115.0',
        'uvicorn': '0.30.6',
        'pydantic': '2.9.2',
        'redis': '5.1.0',
        'temporalio': '1.6.0',
        'asyncpg': '0.30.0',
        'sqlalchemy': None,
        'alembic': None,
        'prometheus_client': '0.21.0',
        'structlog': None,
        'opentelemetry': None,  # Check opentelemetry-api
    }

    missing_packages = []
    outdated_packages = []

    for package_name, required_version in required_packages.items():
        try:
            if package_name == 'opentelemetry':
                try:
                    module = importlib.import_module('opentelemetry.api')
                except ImportError:
                    print_warning(f"opentelemetry: not installed (optional for telemetry)")
                    continue
            else:
                module = importlib.import_module(package_name.replace('-', '_'))

            if hasattr(module, '__version__'):
                installed_version = module.__version__
                if required_version and installed_version != required_version:
                    outdated_packages.append(f"{package_name}: {installed_version} (want {required_version})")
                    print_warning(f"{package_name}: {installed_version} (recommended: {required_version})")
                else:
                    print_success(f"{package_name}: {installed_version}")
            else:
                print_success(f"{package_name}: installed")

        except ImportError:
            missing_packages.append(package_name)
            print_error(f"{package_name}: not installed")

    if missing_packages:
        print_error(f"Missing packages: {', '.join(missing_packages)}")
        print_info("Install with: pip install -r requirements.txt")
        return False

    if outdated_packages:
        print_warning(f"Outdated packages detected: {len(outdated_packages)}")

    return True

def check_docker() -> bool:
    """Check Docker installation and status"""
    print_header("Docker Environment")

    try:
        # Check docker command
        result = subprocess.run(['docker', '--version'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print_success(f"Docker: {result.stdout.strip()}")
        else:
            print_error("Docker command failed")
            return False

        # Check docker-compose command
        result = subprocess.run(['docker-compose', '--version'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print_success(f"Docker Compose: {result.stdout.strip()}")
        else:
            print_error("Docker Compose not available")
            return False

        # Check if Docker daemon is running
        result = subprocess.run(['docker', 'info'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print_success("Docker daemon is running")
            return True
        else:
            print_error("Docker daemon is not running")
            return False

    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_error("Docker not installed or not accessible")
        return False

def check_file_structure() -> bool:
    """Check if required files and directories exist"""
    print_header("Project Structure")

    project_root = Path(__file__).parent.parent.parent

    required_paths = {
        'src/api/app/main.py': 'API main module',
        'src/api/requirements.txt': 'API requirements',
        'src/orchestrator/main.py': 'Orchestrator main',
        'PTaaS/package.json': 'Frontend package.json',
        'requirements.txt': 'Main requirements',
        'pytest.ini': 'Test configuration',
        'infra/docker-compose.yml': 'Docker Compose config',
        'CLAUDE.md': 'Claude guidance',
    }

    all_exist = True

    for path_str, description in required_paths.items():
        path = project_root / path_str
        if path.exists():
            print_success(f"{description}: {path_str}")
        else:
            print_error(f"Missing {description}: {path_str}")
            all_exist = False

    return all_exist

def check_environment_variables() -> bool:
    """Check important environment variables"""
    print_header("Environment Variables")

    env_vars = {
        # Optional but recommended
        'DATABASE_URL': False,
        'REDIS_URL': False,
        'TEMPORAL_HOST': False,
        'JWT_SECRET': False,
        'NVIDIA_API_KEY': False,
        'OPENROUTER_API_KEY': False,
        # Development settings
        'LOG_LEVEL': False,
        'ENVIRONMENT': False,
    }

    all_configured = True

    for var_name, required in env_vars.items():
        value = os.getenv(var_name)
        if value:
            # Don't print sensitive values
            if 'KEY' in var_name or 'SECRET' in var_name or 'PASSWORD' in var_name:
                print_success(f"{var_name}: [CONFIGURED]")
            else:
                print_success(f"{var_name}: {value}")
        else:
            if required:
                print_error(f"{var_name}: not set (required)")
                all_configured = False
            else:
                print_warning(f"{var_name}: not set (optional)")

    return all_configured

def test_api_import() -> bool:
    """Test if the FastAPI app can be imported"""
    print_header("API Import Test")

    try:
        # Change to API directory
        api_dir = Path(__file__).parent.parent.parent / 'src' / 'api'
        original_cwd = os.getcwd()
        os.chdir(api_dir)

        # Add to Python path
        sys.path.insert(0, str(api_dir))

        from app.main import app
        print_success("FastAPI app imported successfully")

        # Test app configuration
        print_success(f"App title: {app.title}")
        print_success(f"App version: {app.version}")
        print_info(f"Routes registered: {len(app.routes)}")

        return True

    except Exception as e:
        print_error(f"Failed to import FastAPI app: {str(e)}")
        return False
    finally:
        # Restore working directory
        os.chdir(original_cwd)
        if str(api_dir) in sys.path:
            sys.path.remove(str(api_dir))

def check_frontend_dependencies() -> bool:
    """Check frontend dependencies"""
    print_header("Frontend Dependencies")

    frontend_dir = Path(__file__).parent.parent.parent / 'PTaaS'
    package_json = frontend_dir / 'package.json'

    if not package_json.exists():
        print_error("PTaaS/package.json not found")
        return False

    print_success("package.json exists")

    # Check if node_modules exists
    node_modules = frontend_dir / 'node_modules'
    if node_modules.exists():
        print_success("node_modules directory exists")
    else:
        print_warning("node_modules not found")
        print_info("Run: cd PTaaS && npm install")

    # Check for npm
    try:
        result = subprocess.run(['npm', '--version'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print_success(f"npm version: {result.stdout.strip()}")
            return True
        else:
            print_error("npm command failed")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_error("npm not installed")
        return False

def generate_summary(results: Dict[str, bool]) -> None:
    """Generate validation summary"""
    print_header("Validation Summary")

    total_checks = len(results)
    passed_checks = sum(1 for passed in results.values() if passed)
    failed_checks = total_checks - passed_checks

    print(f"\n{Colors.BOLD}Results:{Colors.END}")
    print(f"  {Colors.GREEN}✅ Passed: {passed_checks}{Colors.END}")
    print(f"  {Colors.RED}❌ Failed: {failed_checks}{Colors.END}")
    print(f"  {Colors.BLUE}📊 Total:  {total_checks}{Colors.END}")

    if failed_checks == 0:
        print(f"\n{Colors.BOLD}{Colors.GREEN}🎉 All checks passed! Your environment is ready for development.{Colors.END}")
        return True
    else:
        print(f"\n{Colors.BOLD}{Colors.RED}⚠️  {failed_checks} checks failed. Please address the issues above.{Colors.END}")

        # Provide next steps
        print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Install frontend dependencies: cd PTaaS && npm install")
        print("3. Set up environment variables in .env file")
        print("4. Ensure Docker is running if using containers")

        return False

def main():
    """Main validation function"""
    print(f"{Colors.BOLD}{Colors.MAGENTA}")
    print("🚀 XORB Environment Validator")
    print("Checking development environment setup...")
    print(f"{Colors.END}")

    # Run all validation checks
    results = {
        'Python Version': check_python_version(),
        'Virtual Environment': check_virtual_environment(),
        'Python Dependencies': check_required_packages(),
        'Docker Environment': check_docker(),
        'Project Structure': check_file_structure(),
        'Environment Variables': check_environment_variables(),
        'API Import': test_api_import(),
        'Frontend Dependencies': check_frontend_dependencies(),
    }

    # Generate summary
    success = generate_summary(results)

    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
