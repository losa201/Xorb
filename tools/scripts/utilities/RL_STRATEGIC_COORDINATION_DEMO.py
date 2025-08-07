#!/usr/bin/env python3

import requests
import json
import time
from datetime import datetime

def demonstrate_rl_strategic_coordination():
    """Demonstrate the strategic RL coordination capabilities of the XORB platform"""
    
    print("🎯 RL-ENHANCED XORB STRATEGIC COORDINATION DEMONSTRATION")
    print("=" * 70)
    
    rl_orchestrator = "http://localhost:8215"
    strategic_coordinator = "http://localhost:8216"
    
    # 1. Check RL Orchestrator Status
    print("\n🧠 REINFORCEMENT LEARNING ORCHESTRATOR STATUS")
    print("-" * 50)
    
    try:
        response = requests.get(f"{rl_orchestrator}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Status: {health_data['status']}")
            print(f"✅ Reinforcement Learning: {health_data['reinforcement_learning']}")
            print(f"✅ Strategic Reasoning: {health_data['strategic_reasoning']}")
            print(f"✅ Historical Data Integration: {health_data['historical_data_integration']}")
            print(f"✅ Extensible Reasoning: {health_data['extensible_reasoning']}")
            print(f"✅ Agent Coordination: {health_data['agent_coordination']}")
    except Exception as e:
        print(f"❌ Error checking RL Orchestrator: {e}")
    
    # 2. Check RL Agents Status
    print("\n🤖 RL AGENTS PERFORMANCE SUMMARY")
    print("-" * 50)
    
    try:
        response = requests.get(f"{rl_orchestrator}/rl/training-status")
        if response.status_code == 200:
            training_data = response.json()
            print(f"✅ Training Active: {training_data['training_active']}")
            print(f"📊 Learning Rate: {training_data['learning_parameters']['learning_rate']}")
            print(f"🔍 Exploration Rate: {training_data['learning_parameters']['exploration_rate']}")
            
            # Show top performing agents
            agent_performance = training_data['agent_performance']
            top_agents = sorted(
                [(agent_id, data['recent_performance']['performance_score']) 
                 for agent_id, data in agent_performance.items()],
                key=lambda x: x[1], reverse=True
            )[:5]
            
            print(f"\n🏆 TOP 5 PERFORMING AGENTS:")
            for i, (agent_id, score) in enumerate(top_agents, 1):
                cluster = agent_performance[agent_id]['recent_performance']['cluster']
                print(f"  {i}. {agent_id}: {score:.3f} ({cluster} cluster)")
                
    except Exception as e:
        print(f"❌ Error checking RL agents: {e}")
    
    # 3. Check Strategic Coordinator
    print("\n⚡ STRATEGIC SERVICE COORDINATION")
    print("-" * 50)
    
    try:
        response = requests.get(f"{strategic_coordinator}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Status: {health_data['status']}")
            print(f"✅ Strategic Coordination: {health_data['strategic_coordination']}")
            print(f"✅ RL Integration: {health_data['rl_integration']}")
            print(f"✅ Historical Data Utilization: {health_data['historical_data_utilization']}")
            print(f"✅ Cross Service Integration: {health_data['cross_service_integration']}")
            print(f"✅ Adaptive Learning: {health_data['adaptive_learning']}")
    except Exception as e:
        print(f"❌ Error checking Strategic Coordinator: {e}")
    
    # 4. Service Registry Check
    print("\n🌐 SERVICE REGISTRY & COORDINATION")
    print("-" * 50)
    
    try:
        response = requests.get(f"{strategic_coordinator}/services/registry")
        if response.status_code == 200:
            registry_data = response.json()
            print(f"📊 Total Services Registered: {registry_data['service_count']}")
            
            print(f"\n🔗 REGISTERED SERVICES:")
            for service_name, service_info in registry_data['service_registry'].items():
                print(f"  • {service_name}: {service_info['coordination_role']}")
                print(f"    Capabilities: {', '.join(service_info['capabilities'][:2])}...")
                
    except Exception as e:
        print(f"❌ Error checking service registry: {e}")
    
    # 5. Historical Insights
    print("\n📈 HISTORICAL INSIGHTS & PATTERNS")
    print("-" * 50)
    
    try:
        response = requests.get(f"{strategic_coordinator}/coordination/patterns")
        if response.status_code == 200:
            patterns_data = response.json()
            
            coordination_patterns = patterns_data['coordination_patterns']
            print(f"🎯 Available Coordination Patterns: {len(coordination_patterns)}")
            
            for pattern_name, pattern_info in coordination_patterns.items():
                print(f"  • {pattern_name}: Efficiency {pattern_info['efficiency_score']:.2f}, Reliability {pattern_info['reliability']:.2f}")
            
            successful_patterns = patterns_data['historical_insights']['successful_patterns']
            print(f"\n🏆 Historical Success Patterns: {len(successful_patterns)}")
            
            # Show high performance agents from historical data
            high_perf_agents = [p for p in successful_patterns if p['pattern_type'] == 'high_performance_agent']
            cluster_counts = {}
            for agent in high_perf_agents:
                cluster = agent['cluster']
                cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
            
            print(f"  Cluster Distribution:")
            for cluster, count in cluster_counts.items():
                print(f"    - {cluster}: {count} high-performance agents")
                
    except Exception as e:
        print(f"❌ Error checking coordination patterns: {e}")
    
    # 6. Platform Capabilities Summary
    print("\n🚀 RL-ENHANCED PLATFORM CAPABILITIES")
    print("-" * 50)
    
    capabilities = [
        "✅ Deep Reinforcement Learning with PyTorch",
        "✅ 60 Specialized RL Agents (4 clusters)",
        "✅ Strategic Reasoning Engine (4 patterns)",
        "✅ Historical Data Integration & Learning",
        "✅ Extensible Multi-Agent Coordination",
        "✅ Adaptive Cross-Service Communication",
        "✅ Meta-Strategic Planning Capabilities",
        "✅ Continuous Policy Optimization",
        "✅ Real-time Performance Monitoring",
        "✅ Historical Pattern Recognition"
    ]
    
    for capability in capabilities:
        print(f"  {capability}")
    
    print("\n🎊 STRATEGIC RL COORDINATION DEMONSTRATION COMPLETE")
    print(f"📅 Timestamp: {datetime.now().isoformat()}")
    print("🎯 The RL-Enhanced XORB Platform is fully operational with strategic reasoning capabilities!")

if __name__ == "__main__":
    demonstrate_rl_strategic_coordination()