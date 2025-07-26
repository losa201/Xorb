#!/usr/bin/env python3
"""
XORB Auto-Configuration Demo

Demonstrates the auto-configuration system capabilities without requiring
full system dependencies. Shows how different hardware profiles are detected
and configured.

Author: XORB DevOps AI
Version: 2.0.0
"""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

# Mock the required modules for demo
class MockPsutil:
    @staticmethod
    def cpu_count(logical=True):
        return 8 if logical else 4
    
    @staticmethod
    def virtual_memory():
        class Memory:
            total = 8 * 1024**3  # 8GB
            available = 6 * 1024**3  # 6GB available
        return Memory()
    
    @staticmethod
    def disk_usage(path):
        class Disk:
            free = 50 * 1024**3  # 50GB free
        return Disk()
    
    @staticmethod
    def getloadavg():
        return [0.5, 0.4, 0.3]
    
    @staticmethod
    def net_if_addrs():
        return {'eth0': [], 'lo': []}

class MockDocker:
    @staticmethod
    def from_env():
        class Client:
            @staticmethod
            def version():
                return {'Version': '24.0.7'}
        return Client()

class MockPlatform:
    @staticmethod
    def system():
        return "Linux"
    
    @staticmethod
    def release():
        return "6.8.0-64-generic"
    
    @staticmethod
    def machine():
        return "x86_64"

# Patch imports
import sys
sys.modules['psutil'] = MockPsutil
sys.modules['docker'] = MockDocker
sys.modules['platform'] = MockPlatform

# Now import our configurator
from autoconfigure import XorbAutoConfigurator, SystemProfile, XorbMode, SystemCapabilities


def demo_hardware_profiles():
    """Demonstrate configuration for different hardware profiles"""
    
    print("🤖 XORB Auto-Configuration Demo")
    print("=" * 50)
    print("Simulating different hardware configurations...\n")
    
    configurator = XorbAutoConfigurator()
    
    # Define test scenarios
    scenarios = [
        {
            "name": "Raspberry Pi 5",
            "cpu_cores": 4,
            "ram_gb": 4.0,
            "is_arm": True,
            "is_virtualized": False
        },
        {
            "name": "Cloud Micro Instance",
            "cpu_cores": 1,
            "ram_gb": 2.0,
            "is_arm": False,
            "is_virtualized": True
        },
        {
            "name": "Cloud Medium Instance",
            "cpu_cores": 4,
            "ram_gb": 8.0,
            "is_arm": False,
            "is_virtualized": True
        },
        {
            "name": "Bare Metal Server",
            "cpu_cores": 16,
            "ram_gb": 32.0,
            "is_arm": False,
            "is_virtualized": False
        },
        {
            "name": "EPYC Server",
            "cpu_cores": 64,
            "ram_gb": 128.0,
            "is_arm": False,
            "is_virtualized": False
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"📊 Testing: {scenario['name']}")
        print("-" * 30)
        
        # Classify system profile
        profile = configurator._classify_system_profile(
            scenario['cpu_cores'],
            scenario['ram_gb'],
            scenario['is_arm'],
            scenario['is_virtualized']
        )
        
        # Create mock capabilities
        capabilities = SystemCapabilities(
            os_type="Linux",
            os_version="6.8.0",
            architecture="aarch64" if scenario['is_arm'] else "x86_64",
            cpu_cores=scenario['cpu_cores'],
            cpu_threads=scenario['cpu_cores'],
            cpu_frequency=2400.0,
            ram_total_gb=scenario['ram_gb'],
            ram_available_gb=scenario['ram_gb'] * 0.8,
            disk_space_gb=50.0,
            is_arm=scenario['is_arm'],
            is_virtualized=scenario['is_virtualized'],
            docker_version="24.0.7",
            docker_buildkit=True,
            docker_compose_version="2.21.0",
            podman_available=False,
            network_interfaces=["eth0", "lo"],
            dns_servers=["8.8.8.8"],
            system_load=0.5,
            profile=profile
        )
        
        # Generate configuration
        config = configurator.generate_configuration(capabilities)
        
        # Display results
        print(f"   Profile: {profile.value}")
        print(f"   Mode: {config.mode.value}")
        print(f"   Agent Concurrency: {config.agent_concurrency}")
        print(f"   Max Missions: {config.max_concurrent_missions}")
        print(f"   Monitoring: {'Enabled' if config.monitoring_enabled else 'Disabled'}")
        print(f"   Services: {len(config.services_enabled)}")
        print(f"   Memory Limit: {config.memory_limit_mb}MB")
        
        # Show key services
        key_services = ['postgres', 'api', 'prometheus', 'grafana', 'temporal']
        enabled_key_services = [s for s in key_services if s in config.services_enabled]
        print(f"   Key Services: {', '.join(enabled_key_services)}")
        
        results.append({
            'scenario': scenario['name'],
            'profile': profile.value,
            'config': asdict(config)
        })
        
        print()
    
    return results


def demo_environment_generation():
    """Demonstrate environment file generation"""
    print("📝 Environment File Generation Demo")
    print("-" * 40)
    
    configurator = XorbAutoConfigurator()
    
    # Mock cloud medium configuration
    profile = configurator._classify_system_profile(4, 8.0, False, True)
    
    capabilities = SystemCapabilities(
        os_type="Linux",
        os_version="6.8.0",
        architecture="x86_64",
        cpu_cores=4,
        cpu_threads=8,
        cpu_frequency=2400.0,
        ram_total_gb=8.0,
        ram_available_gb=6.4,
        disk_space_gb=50.0,
        is_arm=False,
        is_virtualized=True,
        docker_version="24.0.7",
        docker_buildkit=True,
        docker_compose_version="2.21.0",
        podman_available=False,
        network_interfaces=["eth0", "lo"],
        dns_servers=["8.8.8.8"],
        system_load=0.5,
        profile=profile
    )
    
    config = configurator.generate_configuration(capabilities)
    
    print("Sample .xorb.env content:")
    print("=" * 25)
    
    # Generate sample environment content
    content = [
        "# XORB Auto-Generated Environment Configuration",
        f"# Generated: {datetime.now().isoformat()}",
        f"# System Profile: {config.system_profile.value}",
        f"# XORB Mode: {config.mode.value}",
        "",
    ]
    
    # Add key environment variables
    for key, value in sorted(config.environment_variables.items())[:10]:
        content.append(f"{key}={value}")
    
    content.extend([
        "",
        "# Database Configuration",
        "POSTGRES_USER=xorb",
        "POSTGRES_PASSWORD=xorb_secure_2024",
        "POSTGRES_DB=xorb",
    ])
    
    for line in content:
        print(line)


def demo_resource_optimization():
    """Demonstrate resource optimization"""
    print("\n🔧 Resource Optimization Demo")
    print("-" * 35)
    
    configurator = XorbAutoConfigurator()
    
    # Test different profiles
    test_cases = [
        ("Raspberry Pi", SystemProfile.RPI, 4, 4.0),
        ("Cloud Small", SystemProfile.CLOUD_SMALL, 2, 4.0),
        ("EPYC Server", SystemProfile.EPYC_SERVER, 64, 128.0)
    ]
    
    for name, profile, cores, ram in test_cases:
        print(f"\n{name} ({cores} cores, {ram:.0f}GB RAM):")
        
        # Performance tuning values
        cycle_time = configurator._get_cycle_time(profile)
        db_pool = configurator._get_db_pool_size(profile)
        redis_pool = configurator._get_redis_pool_size(profile)
        
        print(f"  Orchestration Cycle: {cycle_time}ms")
        print(f"  DB Pool Size: {db_pool}")
        print(f"  Redis Pool Size: {redis_pool}")
        
        # Calculate memory allocation
        memory_limit = int(ram * 1024 * 0.8)  # 80% of RAM
        print(f"  Memory Limit: {memory_limit}MB")


def main():
    """Run the demo"""
    try:
        # Demo different hardware profiles
        results = demo_hardware_profiles()
        
        # Demo environment generation
        demo_environment_generation()
        
        # Demo resource optimization
        demo_resource_optimization()
        
        print("\n✅ Demo completed successfully!")
        print("\nTo run the actual auto-configurator:")
        print("  python autoconfigure.py")
        print("  ./configure_environment.sh")
        
        # Save demo results
        demo_file = Path("demo_results.json")
        with open(demo_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'demo_results': results
            }, f, indent=2)
        
        print(f"\n📊 Demo results saved to: {demo_file}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)