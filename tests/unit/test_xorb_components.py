#!/usr/bin/env python3
"""
Comprehensive test suite for EPYC-optimized Xorb 2.0 components
"""

import asyncio
import sys

# Add package to path
sys.path.insert(0, 'packages/xorb_core')

def test_imports():
    """Test all component imports"""
    print("=== Testing Component Imports ===")

    try:
        from xorb_core.orchestration.dqn_agent_selector import (
            DQNAgentSelector,
            EPYCOptimizedDQN,
        )
        print("✅ DQN Agent Selector import successful")

        from xorb_core.orchestration.epyc_numa_optimizer import (
            EPYCNUMAOptimizer,
            EPYCTopology,
        )
        print("✅ NUMA Optimizer import successful")

        from xorb_core.orchestration.advanced_metrics_collector import (
            AdvancedMetricsCollector,
        )
        print("✅ Advanced Metrics Collector import successful")

        from xorb_core.orchestration.epyc_backpressure_controller import (
            EPYCBackpressureController,
        )
        print("✅ EPYC Backpressure Controller import successful")

        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_dqn_agent_selector():
    """Test DQN Agent Selector functionality"""
    print("\n=== Testing DQN Agent Selector ===")

    try:
        import torch

        from xorb_core.orchestration.dqn_agent_selector import DQNAgentSelector

        # Initialize with EPYC configuration
        selector = DQNAgentSelector(
            state_size=128,
            action_size=10,
            epyc_cores=64,
            learning_rate=0.001
        )

        print(f"✅ DQN initialized: {selector.action_size} actions")
        print(f"✅ Device: {selector.device}")
        print(f"✅ EPYC cores: {selector.epyc_cores}")

        # Test forward pass
        test_state = torch.randn(1, 128)
        with torch.no_grad():
            q_values = selector.q_network(test_state)

        print(f"✅ Q-values shape: {q_values.shape}")
        assert q_values.shape == (1, 10), "Q-values shape mismatch"

        # Test action selection
        action = selector.select_action(test_state.numpy(), epsilon=0.1)
        print(f"✅ Selected action: {action}")
        assert 0 <= action < 10, "Action out of range"

        return True
    except Exception as e:
        print(f"❌ DQN test failed: {e}")
        return False

def test_numa_optimizer():
    """Test NUMA Optimizer functionality"""
    print("\n=== Testing NUMA Optimizer ===")

    try:
        from xorb_core.orchestration.epyc_numa_optimizer import (
            EPYCNUMAOptimizer,
        )

        # Initialize optimizer
        optimizer = EPYCNUMAOptimizer()

        print(f"✅ NUMA nodes: {optimizer.topology.numa_nodes}")
        print(f"✅ Total cores: {optimizer.topology.total_cores}")
        print(f"✅ Total threads: {optimizer.topology.total_threads}")
        print(f"✅ CCX per NUMA: {optimizer.topology.ccx_per_numa}")
        print(f"✅ NUMA available: {optimizer.numa_available}")

        # Test topology properties
        assert optimizer.topology.threads_per_numa == 64, "Threads per NUMA mismatch"
        assert optimizer.topology.cores_per_numa == 32, "Cores per NUMA mismatch"

        return True
    except Exception as e:
        print(f"❌ NUMA optimizer test failed: {e}")
        return False

async def test_async_components():
    """Test async functionality"""
    print("\n=== Testing Async Components ===")

    try:
        from xorb_core.orchestration.epyc_numa_optimizer import EPYCNUMAOptimizer

        optimizer = EPYCNUMAOptimizer()

        # Test resource allocation
        allocation = await optimizer.allocate_process_resources(
            process_id="test_process",
            process_type="orchestrator",
            cpu_requirement=0.25,
            memory_requirement=2 * 1024 * 1024 * 1024  # 2GB
        )

        print(f"✅ Process allocated to NUMA node: {allocation.numa_affinity.numa_node}")
        print(f"✅ CPU list: {allocation.numa_affinity.cpu_list[:5]}...")  # Show first 5
        print(f"✅ Memory policy: {allocation.numa_affinity.memory_policy}")

        # Test stats
        stats = await optimizer.get_numa_utilization_stats()
        print(f"✅ Active processes: {stats['active_processes']}")

        # Cleanup
        await optimizer.deallocate_process_resources("test_process")
        print("✅ Process deallocated")

        return True
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        return False

def test_metrics_collector():
    """Test metrics collector"""
    print("\n=== Testing Metrics Collector ===")

    try:
        from xorb_core.orchestration.advanced_metrics_collector import (
            AdvancedMetricsCollector,
        )

        collector = AdvancedMetricsCollector()
        print("✅ Metrics collector initialized")
        print(f"✅ EPYC cores: {collector.epyc_cores}")
        print(f"✅ NUMA nodes: {collector.numa_nodes}")

        # Test metric collection
        rl_metrics = collector.collect_rl_performance_metrics()
        print(f"✅ RL metrics collected: {type(rl_metrics)}")

        epyc_metrics = collector.collect_epyc_utilization_metrics()
        print(f"✅ EPYC metrics collected: {len(epyc_metrics)} metrics")

        return True
    except Exception as e:
        print(f"❌ Metrics collector test failed: {e}")
        return False

def test_kubernetes_integration():
    """Test Kubernetes integration"""
    print("\n=== Testing Kubernetes Integration ===")

    try:
        from xorb_core.orchestration.epyc_numa_optimizer import (
            KubernetesNUMAIntegration,
            NUMAAffinityMapping,
            ProcessResourceAllocation,
        )

        # Create mock allocation
        affinity = NUMAAffinityMapping(
            numa_node=0,
            cpu_list=[0, 1, 2, 3],
            memory_policy='bind',
            expected_locality_ratio=0.95
        )

        allocation = ProcessResourceAllocation(
            process_id="test",
            numa_affinity=affinity,
            memory_limit=4 * 1024 * 1024 * 1024,  # 4GB
            cpu_shares=512,
            io_priority=1,
            process_type="orchestrator"
        )

        # Generate pod spec
        k8s_integration = KubernetesNUMAIntegration()
        pod_spec = k8s_integration.generate_numa_aware_pod_spec(allocation)

        print("✅ Kubernetes pod spec generated")
        print(f"✅ CPU requests: {pod_spec['spec']['containers'][0]['resources']['requests']['cpu']}")
        print(f"✅ Memory requests: {pod_spec['spec']['containers'][0]['resources']['requests']['memory']}")
        print(f"✅ NUMA annotations: {len(pod_spec['metadata']['annotations'])} annotations")

        return True
    except Exception as e:
        print(f"❌ Kubernetes integration test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 EPYC-Optimized Xorb 2.0 Component Test Suite")
    print("=" * 60)

    results = []

    # Test imports
    results.append(test_imports())

    if results[-1]:  # Only continue if imports work
        results.append(test_dqn_agent_selector())
        results.append(test_numa_optimizer())
        results.append(await test_async_components())
        results.append(test_metrics_collector())
        results.append(test_kubernetes_integration())

    # Summary
    print("\n" + "=" * 60)
    print("🏁 Test Summary")
    passed = sum(results)
    total = len(results)

    print(f"✅ Passed: {passed}/{total} tests")

    if passed == total:
        print("🎉 All tests passed! Xorb 2.0 EPYC optimizations are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
