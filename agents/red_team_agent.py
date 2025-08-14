from xorbfw.agents.base_agent import RedTeamAgent

from xorbfw.agents.base_agent import RedTeamAgent
from xorbfw.core.attack_patterns import TTP, AttackChain
from xorbfw.telemetry.telemetry import TelemetrySystem
import numpy as np
import random

class RedTeamAgent:
    """Sophisticated adversarial agent implementing realistic attack patterns."""
    def __init__(self, id, position, stealth_budget, resource_level, skill_level=0.5):
        self.id = id
        self.position = position
        self.stealth_budget = stealth_budget
        self.resource_level = resource_level
        self.skill_level = skill_level  # 0-1 scale of attacker proficiency
        self.type = "red"
        self.attack_chain = AttackChain()
        self.detected = False
        self.last_action_time = 0
        self.operation_phase = "reconnaissance"  # Initial phase
        self.knowledge_base = {}

    def move(self, new_position):
        """Update agent position with potential detection risk."""
        distance = np.linalg.norm(np.array(self.position) - np.array(new_position))
        self.position = new_position

        # Moving consumes resources and may trigger detection
        move_cost = distance * 0.01
        self.resource_level = max(0, self.resource_level - move_cost)

        # Random chance of detection during movement
        if random.random() < (0.05 * self.skill_level):
            self.detected = True
            return False  # Movement detected

        return True  # Movement successful

    def plan_attack(self, target_info):
        """Develop attack plan based on target information."""
        # Update knowledge base with new target info
        self.knowledge_base.update(target_info)

        # Select appropriate TTPs based on target profile
        available_ttps = self._select_applicable_ttps(target_info)

        # Build attack chain using MITRE ATT&CK framework
        self.attack_chain = self._build_attack_chain(available_ttps)

        # Update operational phase
        self.operation_phase = self.attack_chain.current_phase()

        return self.attack_chain.summary()

    def _select_applicable_ttps(self, target_info):
        """Identify relevant TTPs based on target characteristics."""
        applicable_ttps = []

        # Example logic - would be more sophisticated in real implementation
        if "windows" in target_info.get("os", "").lower():
            applicable_ttps.append(TTP("T1059", "Command and Scripting Interpreter"))
            applicable_ttps.append(TTP("T1047", "Windows Management Instrumentation"))

        if "linux" in target_info.get("os", "").lower():
            applicable_ttps.append(TTP("T1059", "Command and Scripting Interpreter"))
            applicable_ttps.append(TTP("T1163", "Securityd Memory Extraction"))

        # Add privilege escalation techniques
        if self.skill_level > 0.7:
            applicable_ttps.append(TTP("T1068", "Exploitation for Privilege Escalation"))
            applicable_ttps.append(TTP("T1055", "Process Injection"))

        return applicable_ttps

    def _build_attack_chain(self, ttps):
        """Construct an attack chain from selected TTPs."""
        attack_chain = AttackChain()

        # Add reconnaissance phase
        recon_ttps = [t for t in ttps if t.id in ["T1059", "T1047", "T1163"]]
        if recon_ttps:
            attack_chain.add_phase("reconnaissance", recon_ttps[:2])

        # Add initial access techniques
        initial_access = [t for t in ttps if t.id in ["T1071", "T1190", "T1195"]]
        if initial_access:
            attack_chain.add_phase("initial_access", initial_access[:1])

        # Add execution techniques
        execution = [t for t in ttps if t.id in ["T1059", "T1047", "T1055"]]
        if execution:
            attack_chain.add_phase("execution", execution[:2])

        # Add privilege escalation techniques
        priv_esc = [t for t in ttps if t.id in ["T1068", "T1055"]]
        if priv_esc:
            attack_chain.add_phase("privilege_escalation", priv_esc[:1])

        # Add persistence techniques
        persistence = [t for t in ttps if t.id in ["T1067", "T1082", "T1176"]]
        if persistence:
            attack_chain.add_phase("persistence", persistence[:1])

        return attack_chain

    def execute_attack_step(self, environment):
        """Execute the next step in the attack chain."""
        if not self.attack_chain.current_phase:
            return {"success": False, "message": "No active attack chain"}

        current_ttp = self.attack_chain.get_next_ttp()
        if not current_ttp:
            return {"success": False, "message": "No TTP to execute"}

        # Calculate success probability based on skill level and environment factors
        base_success_rate = 0.6
        skill_factor = self.skill_level * 0.4
        detection_factor = 1 - (self.stealth_budget * 0.5)

        success_chance = base_success_rate + skill_factor - detection_factor
        success_chance = max(0.2, min(0.9, success_chance))  # Clamp between 0.2 and 0.9

        # Execute the TTP
        if random.random() < success_chance:
            # Successful execution
            result = {
                "success": True,
                "ttp_id": current_ttp.id,
                "message": f"Successfully executed {current_ttp.name}",
                "phase": self.attack_chain.current_phase
            }

            # Update agent state
            self.last_action_time = environment.current_time
            self.resource_level = max(0, self.resource_level - 0.05)
            self.stealth_budget = max(0, self.stealth_budget - 0.1)

            # Potentially escalate phase
            if random.random() < 0.3:
                self.attack_chain.advance_phase()

            return result
        else:
            # Failed execution
            return {
                "success": False,
                "ttp_id": current_ttp.id,
                "message": f"Failed to execute {current_ttp.name}",
                "detected": self.detected
            }

    def evade_detection(self, detection_data):
        """Implement evasion techniques when detected."""
        # Analyze detection patterns
        evasion_success = self._analyze_detection_patterns(detection_data)

        if evasion_success:
            self.detected = False
            return {"success": True, "message": "Successfully evaded detection"}
        else:
            return {"success": False, "message": "Failed to evade detection"}

    def _analyze_detection_patterns(self, detection_data):
        """Analyze detection patterns to develop evasion strategy."""
        # Update knowledge base with detection patterns
        self.knowledge_base["detection_patterns"] = detection_data

        # Calculate evasion success probability
        evasion_chance = 0.4 + (self.skill_level * 0.5) - (detection_data.get("confidence", 0.5) * 0.3)
        evasion_chance = max(0.2, min(0.8, evasion_chance))  # Clamp between 0.2 and 0.8

        return random.random() < evasion_chance

    def get_telemetry(self):
        """Return telemetry data for this agent."""
        return {
            "agent_id": self.id,
            "type": self.type,
            "position": self.position,
            "stealth_budget": self.stealth_budget,
            "resource_level": self.resource_level,
            "operation_phase": self.operation_phase,
            "detected": self.detected,
            "attack_chain": self.attack_chain.to_dict()
        }
