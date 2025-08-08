import numpy as np
import matplotlib.pyplot as plt
from xorbfw.core.simulation import SimulationEngine
from xorbfw.agents.base_agent import Agent, RedTeamAgent
from xorbfw.agents.blue_team_agent import BlueTeamAgent
from xorbfw.agents.rl_agent import RLAgent
from xorbfw.telemetry.telemetry import TelemetrySystem
from xorbfw.orchestration.orchestrator import Orchestrator
from xorbfw.orchestration.emergent_behavior import SwarmBehavior
import time

class SimulationVisualizer:
    """Handles visualization of simulation state and telemetry data."""
    def __init__(self, sim):
        self.sim = sim
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8))
        self._init_plots()
        
    def _init_plots(self):
        """Initialize plot elements and labels."""
        # Agent position plot
        self.ax1.set_title('Agent Positions')
        self.ax1.set_xlabel('X Position')
        self.ax1.set_ylabel('Y Position')
        self.ax1.set_xlim(0, 100)
        self.ax1.set_ylim(0, 100)
        
        # Telemetry metrics plot
        self.ax2.set_title('Telemetry Metrics')
        self.ax2.set_xlabel('Steps')
        self.ax2.set_ylabel('Metric Value')
        self.ax2.grid(True)
        
        # Create line objects for efficient updates
        self.agent_plots = {}
        self.metric_lines = {}
        
    def update(self):
        """Update visualization with current simulation state."""
        self._update_agents()
        self._update_telemetry()
        plt.tight_layout()
        plt.draw()
        plt.pause(0.05)
        
    def _update_agents(self):
        """Update agent position visualization."""
        # Clear previous agent positions
        for line in self.agent_plots.values():
            line.remove()
        self.agent_plots.clear()
        
        # Plot current agent positions
        for agent_id, agent in self.sim.agents.items():
            x, y = agent.position
            if isinstance(agent, RedTeamAgent):
                self.agent_plots[agent_id] = self.ax1.plot(x, y, 'ro', markersize=10)[0]
            elif isinstance(agent, BlueTeamAgent):
                self.agent_plots[agent_id] = self.ax1.plot(x, y, 'bo', markersize=10)[0]
            elif isinstance(agent, RLAgent):
                self.agent_plots[agent_id] = self.ax1.plot(x, y, 'go', markersize=12, marker='s')[0]
        
    def _update_telemetry(self):
        """Update telemetry metrics visualization."""
        telemetry = self.sim.get_telemetry()
        
        # Update detection risk
        steps = list(range(len(telemetry['detection_risk'])))
        if 'detection_risk' in self.metric_lines:
            self.metric_lines['detection_risk'].set_data(steps, telemetry['detection_risk'])
        else:
            self.metric_lines['detection_risk'], = self.ax2.plot(
                steps, telemetry['detection_risk'], 'r-', label='Detection Risk'
            )
        
        # Update resource levels
        for aid, levels in telemetry['resource_levels'].items():
            key = f'resource_{aid}'
            levels = levels[-50:]
            if key in self.metric_lines:
                self.metric_lines[key].set_data(range(len(levels)), levels)
            else:
                self.metric_lines[key], = self.ax2.plot(
                    range(len(levels)), levels, label=f'Resource {aid}'
                )
        
        # Update mission progress
        for aid, progress in telemetry['mission_progress'].items():
            key = f'progress_{aid}'
            progress = progress[-50:]
            if key in self.metric_lines:
                self.metric_lines[key].set_data(range(len(progress)), progress)
            else:
                self.metric_lines[key], = self.ax2.plot(
                    range(len(progress)), progress, '--', label=f'Progress {aid}'
                )
        
        # Update legend and limits
        self.ax2.legend()
        self.ax2.relim()
        self.ax2.autoscale_view()

def run_demonstration():
    """Run a demonstration of the adversarial simulation scenario."""
    try:
        # Initialize simulation
        sim = SimulationEngine(
            name="XORB Adversarial Demonstration",
            world_size=(100, 100),
            max_steps=200
        )
        
        # Create orchestrator
        orchestrator = Orchestrator(sim)
        
        # Create and add agents
        red_agents = []
        blue_agents = []
        
        # Create red team agents (adversarial)
        for i in range(5):
            red_agent = RedTeamAgent(
                id=f"R-{i}",
                position=(np.random.uniform(0, 100), np.random.uniform(0, 100)),
                stealth_budget=0.8,
                resource_level=0.7
            )
            red_agents.append(red_agent)
            sim.add_agent(red_agent)
        
        # Create blue team agents (defensive)
        for i in range(5):
            blue_agent = BlueTeamAgent(
                id=f"B-{i}",
                position=(np.random.uniform(0, 100), np.random.uniform(0, 100)),
                detection_radius=15,
                resource_level=0.9
            )
            blue_agents.append(blue_agent)
            sim.add_agent(blue_agent)
        
        # Create RL agent for adaptive behavior
        rl_agent = RLAgent(
            id="RL-1",
            position=(50, 50),
            resource_level=0.85,
            observation_space=10,
            action_space=5
        )
        sim.add_agent(rl_agent)
        
        # Assign missions through orchestrator
        orchestrator.assign_mission(rl_agent.id, {
            'type': 'reconnaissance',
            'target': (75, 75),
            'priority': 0.8
        })
        
        # Create swarm behavior for red team
        swarm = SwarmBehavior(red_agents)
        
        # Initialize visualization
        visualizer = SimulationVisualizer(sim)
        
        # Run simulation loop
        for step in range(sim.max_steps):
            # Update simulation
            sim.step()
            
            # Swarm behavior coordination
            swarm.coordinate()
            
            # RL agent learning step
            obs = rl_agent.perceive(sim)
            action = rl_agent.select_action(obs)
            reward, done = rl_agent.execute_action(sim, action)
            rl_agent.update_policy(reward, done)
            
            # Update visualization every 5 steps
            if step % 5 == 0:
                visualizer.update()
                visualizer.fig.suptitle(f'Simulation Step: {step}')
            
            # Check for mission completion
            if done:
                print(f"Mission completed at step {step}")
                break
                
    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        # Clean up visualization
        plt.close('all')
        print("Simulation demonstration completed")
    
    # Final analysis
    print("\nFinal Simulation Analysis:")
    print(f"Total Steps: {step + 1}")
    print(f"Final Detection Risk: {telemetry['detection_risk'][-1]:.2f}")
    print(f"Average Resource Level: {np.mean([level[-1] for level in resource_levels.values()]):.2f}")
    print(f"Final Mission Progress: {telemetry['mission_progress'][rl_agent.id][-1]:.2f}")
    print(f"Total Adversarial Encounters: {len(telemetry['adversarial_encounters'])}")
    
    # Save telemetry data
    np.save('telemetry_data.npy', telemetry)
    
    # Show final plot
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_demonstration()