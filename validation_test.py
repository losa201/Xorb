import asyncio
from xorbfw.core.simulation import SimulationEngine
from xorbfw.agents.base_agent import RedTeamAgent, BlueTeamAgent
from xorbfw.learning.rl_agent import RLAgent
from xorbfw.telemetry.telemetry import TelemetrySystem
from xorbfw.orchestration.orchestrator import MissionOrchestrator
from xorbfw.orchestration.emergent_behavior import SwarmBehavior

async def run_validation_test():
    # Initialize core components
    sim_engine = SimulationEngine()
    telemetry = TelemetrySystem()
    orchestrator = MissionOrchestrator()
    
    # Create agents
    red_agent = RedTeamAgent("R-001", sim_engine)
    blue_agent = BlueTeamAgent("B-001", sim_engine)
    rl_agent = RLAgent("RL-001", sim_engine)
    
    # Register agents with simulation
    sim_engine.register_agent(red_agent)
    sim_engine.register_agent(blue_agent)
    sim_engine.register_agent(rl_agent)
    
    # Set up mission
    mission = {
        "id": "M-001",
        "type": "reconnaissance",
        "priority": 0.8,
        "objectives": [{
            "type": "data_exfiltration",
            "target": "secure_server",
            "constraints": {
                "stealth_budget": 0.7,
                "resource_limit": 0.6
            }
        }]
    }
    
    # Assign mission
    orchestrator.assign_mission(rl_agent.id, mission)
    
    # Set up swarm behavior
    swarm = SwarmBehavior([red_agent, blue_agent, rl_agent])
    
    # Run simulation
    print("Starting simulation...")
    for i in range(10):  # Run for 10 steps
        print(f"\n--- Step {i} ---")
        
        # Advance simulation
        sim_engine.step()
        
        # Collect telemetry
        telemetry_data = telemetry.collect_telemetry()
        telemetry.process_telemetry(telemetry_data)
        
        # Show agent states
        print("\nAgent States:")
        for agent in [red_agent, blue_agent, rl_agent]:
            print(f"{agent.agent_type} {agent.id}:")
            print(f"  Position: {agent.position}")
            print(f"  Resource Level: {agent.resource_level:.2f}")
            print(f"  Detection Risk: {agent.detection_risk:.2f}")
            print(f"  Mission Progress: {agent.mission_progress:.2f}")
            print(f"  Policy: {agent.current_policy}")
        
        # Check for emergent behavior
        if i % 3 == 0:
            print("\nChecking for emergent behavior...")
            swarm_behavior = swarm.coordinate_attack()
            print(f"Emergent behavior pattern: {swarm_behavior}")
        
        # Wait for next step
        await asyncio.sleep(0.5)
    
    # Generate validation report
    print("\n=== Validation Report ===")
    print(f"Simulation Steps Completed: {sim_engine.step_count}")
    print(f"Total Agents Simulated: {len(sim_engine.agents)}")
    print(f"Total Missions Assigned: {len(orchestrator.assigned_missions)}")
    print(f"Total Telemetry Collected: {len(telemetry_data)} metrics")
    print(f"Resource Utilization: {sim_engine.calculate_resource_utilization():.2f}")
    print(f"Average Detection Risk: {sim_engine.calculate_average_detection_risk():.2f}")
    print(f"Mission Success Rate: {sim_engine.calculate_mission_success_rate():.2f}")
    
    return {
        "step_count": sim_engine.step_count,
        "agents_simulated": len(sim_engine.agents),
        "missions_assigned": len(orchestrator.assigned_missions),
        "telemetry_metrics": len(telemetry_data),
        "resource_utilization": sim_engine.calculate_resource_utilization(),
        "detection_risk": sim_engine.calculate_average_detection_risk(),
        "mission_success_rate": sim_engine.calculate_mission_success_rate()
    }

if __name__ == "__main__":
    asyncio.run(run_validation_test())