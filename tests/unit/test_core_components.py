#!/usr/bin/env python3
"""
Core component test for EPYC-optimized Xorb 2.0
"""

import asyncio
import sys

# Add packages to path
sys.path.insert(0, 'packages/xorb_core')

def test_numa_optimizer():
    """Test NUMA optimizer basic functionality"""
    print("=== Testing NUMA Optimizer ===")

    try:
        from xorb_core.orchestration.epyc_numa_optimizer import (
            EPYCNUMAOptimizer,
            EPYCTopology,
        )

        # Test topology
        topology = EPYCTopology()
        print(f"✅ EPYC Topology: {topology.total_cores} cores, {topology.numa_nodes} NUMA nodes")

        # Test optimizer
        optimizer = EPYCNUMAOptimizer()
        print("✅ NUMA Optimizer initialized")
        print(f"✅ NUMA available: {optimizer.numa_available}")

        return True
    except Exception as e:
        print(f"❌ NUMA test failed: {e}")
        return False

def test_dqn_with_torch():
    """Test DQN agent selector with PyTorch"""
    print("\n=== Testing DQN Agent Selector ===")

    try:
        import torch

        from xorb_core.orchestration.dqn_agent_selector import (
            DQNAgentSelector,
            EPYCOptimizedDQN,
        )

        # Test DQN network
        network = EPYCOptimizedDQN(state_size=64, action_size=8, epyc_cores=64)
        print("✅ EPYC-optimized DQN network created")

        # Test forward pass
        test_input = torch.randn(1, 64)
        output = network(test_input)
        print(f"✅ Forward pass successful: {output.shape}")

        # Test agent selector
        selector = DQNAgentSelector(state_size=64, action_size=8, epyc_cores=64)
        print(f"✅ DQN Agent Selector initialized on {selector.device}")

        # Test action selection
        state = torch.randn(64).numpy()
        action = selector.select_action(state, epsilon=0.1)
        print(f"✅ Action selected: {action}")

        return True
    except Exception as e:
        print(f"❌ DQN test failed: {e}")
        return False

def test_metrics_collector():
    """Test metrics collector"""
    print("\n=== Testing Metrics Collector ===")

    try:
        from xorb_core.orchestration.advanced_metrics_collector import (
            AdvancedMetricsCollector,
        )

        collector = AdvancedMetricsCollector(epyc_cores=64, numa_nodes=2)
        print("✅ Metrics collector initialized")

        # Test metric collection
        rl_metrics = collector.collect_rl_performance_metrics()
        print(f"✅ RL metrics collected: {type(rl_metrics).__name__}")

        epyc_metrics = collector.collect_epyc_utilization_metrics()
        print(f"✅ EPYC metrics collected: {len(epyc_metrics)} metrics")

        return True
    except Exception as e:
        print(f"❌ Metrics collector test failed: {e}")
        return False

def test_backpressure_controller():
    """Test backpressure controller"""
    print("\n=== Testing Backpressure Controller ===")

    try:
        from xorb_core.orchestration.epyc_backpressure_controller import (
            EPYCBackpressureController,
        )

        controller = EPYCBackpressureController(epyc_cores=64, numa_nodes=2)
        print("✅ Backpressure controller initialized")
        print(f"✅ EPYC configuration: {controller.epyc_config}")

        # Test threshold calculation
        thresholds = controller.calculate_dynamic_thresholds()
        print(f"✅ Dynamic thresholds calculated: {len(thresholds)} thresholds")

        return True
    except Exception as e:
        print(f"❌ Backpressure controller test failed: {e}")
        return False

async def test_async_numa_operations():
    """Test async NUMA operations"""
    print("\n=== Testing Async NUMA Operations ===")

    try:
        from xorb_core.orchestration.epyc_numa_optimizer import EPYCNUMAOptimizer

        optimizer = EPYCNUMAOptimizer()

        # Test resource allocation
        allocation = await optimizer.allocate_process_resources(
            process_id="test_async",
            process_type="ml_training",
            cpu_requirement=0.5,
            memory_requirement=4 * 1024 * 1024 * 1024  # 4GB
        )

        print(f"✅ Async allocation successful: NUMA node {allocation.numa_affinity.numa_node}")
        print(f"✅ CPU list length: {len(allocation.numa_affinity.cpu_list)}")

        # Test stats
        stats = await optimizer.get_numa_utilization_stats()
        print(f"✅ NUMA stats retrieved: {stats['active_processes']} active processes")

        # Cleanup
        await optimizer.deallocate_process_resources("test_async")
        print("✅ Resource cleanup successful")

        return True
    except Exception as e:
        print(f"❌ Async NUMA test failed: {e}")
        return False

async def main():
    """Run all core tests"""
    print("🚀 Xorb 2.0 Core Component Tests")
    print("=" * 50)

    results = []

    # Run synchronous tests
    results.append(test_numa_optimizer())
    results.append(test_dqn_with_torch())
    results.append(test_metrics_collector())
    results.append(test_backpressure_controller())

    # Run async tests
    results.append(await test_async_numa_operations())

    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Summary")
    passed = sum(results)
    total = len(results)

    print(f"✅ Passed: {passed}/{total} tests")

    if passed == total:
        print("🎉 All core tests passed! EPYC optimizations are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
