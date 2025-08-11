#!/usr/bin/env python3
"""
NVIDIA-Powered Scenario Orchestrator
Advanced AI orchestrator that presents dynamic scenarios while agents learn and adapt
"""

import os
import json
import asyncio
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import openai
from openai import OpenAI
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScenarioContext:
    scenario_id: str
    name: str
    description: str
    difficulty_level: int  # 1-10
    target_environment: str
    learning_objectives: List[str]
    success_metrics: Dict[str, Any]
    duration_minutes: int
    threat_actor_profile: str
    industry_context: str

@dataclass
class AgentLearningState:
    agent_id: str
    agent_type: str  # red, blue, purple
    current_skill_level: float  # 0.0-1.0
    learning_rate: float
    strengths: List[str]
    weaknesses: List[str]
    recent_performance: Dict[str, float]
    adaptation_history: List[Dict]

@dataclass
class ScenarioPresentation:
    timestamp: str
    scenario_context: ScenarioContext
    narrative: str
    technical_briefing: str
    learning_hints: List[str]
    adaptive_challenges: List[str]
    expected_agent_responses: Dict[str, List[str]]

class NVIDIAScenarioOrchestrator:
    def __init__(self):
        self.nvidia_api_key = os.getenv('NVIDIA_API_KEY')
        self.openai_client = None
        self.active_scenarios = {}
        self.agent_learning_states = {}
        self.scenario_templates = {}
        self.learning_analytics = {}
        
        # Initialize NVIDIA API client
        self.setup_nvidia_client()
        
        # Load scenario templates
        self.load_scenario_templates()
        
        # Initialize agent learning tracking
        self.initialize_agent_learning()

    def setup_nvidia_client(self):
        """Setup NVIDIA API client for advanced AI capabilities"""
        if not self.nvidia_api_key:
            logger.warning("NVIDIA_API_KEY not found, using OpenAI as fallback")
            # Fallback to OpenAI with NVIDIA-compatible interface
            self.openai_client = OpenAI(
                api_key=os.getenv('OPENROUTER_API_KEY', 'fallback'),
                base_url="https://openrouter.ai/api/v1"
            )
        else:
            # Use NVIDIA's NIM (NVIDIA Inference Microservices)
            self.openai_client = OpenAI(
                api_key=self.nvidia_api_key,
                base_url="https://integrate.api.nvidia.com/v1"
            )
        
        logger.info("NVIDIA/OpenAI client initialized for scenario orchestration")

    def load_scenario_templates(self):
        """Load comprehensive scenario templates for different learning objectives"""
        self.scenario_templates = {
            "apt_infiltration": {
                "name": "Advanced Persistent Threat Infiltration",
                "base_difficulty": 8,
                "learning_objectives": [
                    "Stealth reconnaissance techniques",
                    "Social engineering tactics", 
                    "Persistence mechanism deployment",
                    "Defense evasion strategies"
                ],
                "threat_profile": "Nation-state APT group",
                "industry_contexts": ["finance", "healthcare", "government", "technology"],
                "adaptive_elements": [
                    "Detection system sophistication",
                    "Network segmentation complexity",
                    "User awareness levels",
                    "Incident response capabilities"
                ]
            },
            
            "ransomware_outbreak": {
                "name": "Ransomware Outbreak Response",
                "base_difficulty": 7,
                "learning_objectives": [
                    "Rapid threat containment",
                    "Forensic analysis techniques",
                    "Business continuity planning", 
                    "Crisis communication"
                ],
                "threat_profile": "Ransomware-as-a-Service group",
                "industry_contexts": ["manufacturing", "retail", "education", "logistics"],
                "adaptive_elements": [
                    "Encryption speed",
                    "Lateral movement vectors",
                    "Backup system targeting",
                    "Payment pressure tactics"
                ]
            },
            
            "insider_threat_detection": {
                "name": "Malicious Insider Detection",
                "base_difficulty": 6,
                "learning_objectives": [
                    "Behavioral analysis techniques",
                    "Data access monitoring",
                    "Privilege abuse detection",
                    "Investigation methodologies"
                ],
                "threat_profile": "Disgruntled employee/contractor",
                "industry_contexts": ["finance", "technology", "defense", "pharmaceuticals"],
                "adaptive_elements": [
                    "Insider access levels",
                    "Data sensitivity",
                    "Monitoring capabilities",
                    "Legal constraints"
                ]
            },
            
            "supply_chain_compromise": {
                "name": "Supply Chain Attack Scenario",
                "base_difficulty": 9,
                "learning_objectives": [
                    "Third-party risk assessment",
                    "Software integrity verification",
                    "Incident attribution",
                    "Recovery strategies"
                ],
                "threat_profile": "Supply chain infiltration group",
                "industry_contexts": ["software", "manufacturing", "automotive", "aerospace"],
                "adaptive_elements": [
                    "Compromise sophistication",
                    "Vendor trust relationships", 
                    "Detection timeframes",
                    "Impact scope"
                ]
            }
        }

    def initialize_agent_learning(self):
        """Initialize learning state tracking for all agent types"""
        agent_types = ["red_team", "blue_team", "purple_team"]
        
        for agent_type in agent_types:
            for i in range(3):  # 3 agents per type
                agent_id = f"{agent_type}_agent_{i+1:02d}"
                self.agent_learning_states[agent_id] = AgentLearningState(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    current_skill_level=0.5,  # Start at intermediate
                    learning_rate=0.1,
                    strengths=[],
                    weaknesses=["scenario_adaptation", "cross_domain_knowledge"],
                    recent_performance={},
                    adaptation_history=[]
                )

    async def present_dynamic_scenario(self, scenario_template: str, 
                                     target_agents: List[str],
                                     custom_parameters: Dict = None) -> str:
        """Present a dynamic scenario using NVIDIA AI to create adaptive narratives"""
        
        if scenario_template not in self.scenario_templates:
            raise ValueError(f"Scenario template '{scenario_template}' not found")
        
        template = self.scenario_templates[scenario_template]
        scenario_id = f"scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Analyze target agent capabilities for adaptive difficulty
        agent_analysis = await self.analyze_target_agents(target_agents)
        
        # Generate scenario context with NVIDIA AI
        scenario_context = await self.generate_scenario_context(
            template, agent_analysis, custom_parameters or {}
        )
        
        # Create narrative presentation
        presentation = await self.create_scenario_presentation(
            scenario_context, agent_analysis
        )
        
        # Store active scenario
        self.active_scenarios[scenario_id] = {
            "context": scenario_context,
            "presentation": presentation,
            "target_agents": target_agents,
            "start_time": datetime.now(),
            "status": "active",
            "agent_responses": {},
            "learning_metrics": {}
        }
        
        logger.info(f"Presenting scenario {scenario_id} to agents: {target_agents}")
        
        # Present scenario to agents
        await self.deliver_scenario_to_agents(scenario_id, target_agents)
        
        return scenario_id

    async def analyze_target_agents(self, target_agents: List[str]) -> Dict[str, Any]:
        """Analyze target agent capabilities for scenario adaptation"""
        
        agent_analysis = {
            "average_skill_level": 0.0,
            "collective_strengths": [],
            "collective_weaknesses": [],
            "recommended_difficulty": 5,
            "learning_focus_areas": []
        }
        
        if not target_agents:
            return agent_analysis
        
        # Collect agent learning states
        agent_states = [
            self.agent_learning_states.get(agent_id) 
            for agent_id in target_agents
            if agent_id in self.agent_learning_states
        ]
        
        if not agent_states:
            return agent_analysis
        
        # Calculate collective metrics
        agent_analysis["average_skill_level"] = sum(
            state.current_skill_level for state in agent_states
        ) / len(agent_states)
        
        # Aggregate strengths and weaknesses
        all_strengths = [strength for state in agent_states for strength in state.strengths]
        all_weaknesses = [weakness for state in agent_states for weakness in state.weaknesses]
        
        agent_analysis["collective_strengths"] = list(set(all_strengths))
        agent_analysis["collective_weaknesses"] = list(set(all_weaknesses))
        
        # Calculate recommended difficulty
        base_difficulty = 5
        skill_modifier = (agent_analysis["average_skill_level"] - 0.5) * 4
        agent_analysis["recommended_difficulty"] = max(1, min(10, base_difficulty + skill_modifier))
        
        # Identify learning focus areas
        agent_analysis["learning_focus_areas"] = agent_analysis["collective_weaknesses"][:3]
        
        return agent_analysis

    async def generate_scenario_context(self, template: Dict, 
                                      agent_analysis: Dict,
                                      custom_parameters: Dict) -> ScenarioContext:
        """Generate dynamic scenario context using NVIDIA AI"""
        
        # Create prompt for NVIDIA AI
        prompt = f"""
        Generate a cybersecurity scenario based on the following template and agent analysis:
        
        Template: {template['name']}
        Learning Objectives: {template['learning_objectives']}
        Threat Profile: {template['threat_profile']}
        
        Agent Analysis:
        - Average Skill Level: {agent_analysis['average_skill_level']:.2f}
        - Collective Weaknesses: {agent_analysis['collective_weaknesses']}
        - Recommended Difficulty: {agent_analysis['recommended_difficulty']}
        
        Custom Parameters: {custom_parameters}
        
        Create a detailed scenario that:
        1. Challenges the agents appropriately
        2. Focuses on their learning needs
        3. Provides realistic industry context
        4. Includes specific success metrics
        
        Return a JSON structure with:
        - Scenario name and description
        - Industry context selection
        - Specific difficulty adjustments
        - Learning objectives tailored to agent weaknesses
        - Success metrics
        """
        
        try:
            response = await self.query_nvidia_ai(prompt, "scenario_generation")
            
            # Parse AI response
            ai_scenario = json.loads(response)
            
            # Create scenario context
            scenario_context = ScenarioContext(
                scenario_id=f"nvidia_scenario_{datetime.now().strftime('%H%M%S')}",
                name=ai_scenario.get("name", template["name"]),
                description=ai_scenario.get("description", "Dynamic AI-generated scenario"),
                difficulty_level=int(ai_scenario.get("difficulty_level", agent_analysis["recommended_difficulty"])),
                target_environment=custom_parameters.get("environment", "hybrid_enterprise"),
                learning_objectives=ai_scenario.get("learning_objectives", template["learning_objectives"]),
                success_metrics=ai_scenario.get("success_metrics", {}),
                duration_minutes=custom_parameters.get("duration", 60),
                threat_actor_profile=ai_scenario.get("threat_profile", template["threat_profile"]),
                industry_context=ai_scenario.get("industry_context", "technology")
            )
            
            return scenario_context
            
        except Exception as e:
            logger.error(f"Error generating scenario context: {e}")
            # Fallback to template-based generation
            return self.generate_fallback_scenario_context(template, agent_analysis, custom_parameters)

    async def create_scenario_presentation(self, scenario_context: ScenarioContext,
                                         agent_analysis: Dict) -> ScenarioPresentation:
        """Create comprehensive scenario presentation with NVIDIA AI"""
        
        prompt = f"""
        Create a comprehensive presentation for this cybersecurity scenario:
        
        Scenario: {scenario_context.name}
        Description: {scenario_context.description}
        Difficulty Level: {scenario_context.difficulty_level}/10
        Industry: {scenario_context.industry_context}
        Threat Actor: {scenario_context.threat_actor_profile}
        
        Agent Capabilities:
        - Skill Level: {agent_analysis['average_skill_level']:.2f}
        - Focus Areas: {agent_analysis['learning_focus_areas']}
        
        Create:
        1. Engaging narrative introduction (2-3 paragraphs)
        2. Technical briefing with specific details
        3. Learning hints tailored to agent weaknesses
        4. Adaptive challenges that scale with performance
        5. Expected agent response patterns
        
        Make the presentation immersive and educational while maintaining realism.
        """
        
        try:
            response = await self.query_nvidia_ai(prompt, "presentation_creation")
            ai_presentation = json.loads(response)
            
            presentation = ScenarioPresentation(
                timestamp=datetime.now().isoformat(),
                scenario_context=scenario_context,
                narrative=ai_presentation.get("narrative", "Default scenario narrative"),
                technical_briefing=ai_presentation.get("technical_briefing", "Default technical briefing"),
                learning_hints=ai_presentation.get("learning_hints", []),
                adaptive_challenges=ai_presentation.get("adaptive_challenges", []),
                expected_agent_responses=ai_presentation.get("expected_responses", {})
            )
            
            return presentation
            
        except Exception as e:
            logger.error(f"Error creating presentation: {e}")
            return self.create_fallback_presentation(scenario_context)

    async def query_nvidia_ai(self, prompt: str, task_type: str) -> str:
        """Query NVIDIA AI models for scenario generation and analysis"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="nvidia/llama-3.1-nemotron-70b-instruct",  # NVIDIA's flagship model
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert cybersecurity scenario designer and AI orchestrator. "
                                 "Generate realistic, educational scenarios that challenge and teach cybersecurity professionals. "
                                 "Always respond with valid JSON when requested."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000,
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"NVIDIA AI query failed for {task_type}: {e}")
            # Fallback to basic response
            return self.generate_fallback_ai_response(task_type)

    async def deliver_scenario_to_agents(self, scenario_id: str, target_agents: List[str]):
        """Deliver scenario presentation to target agents"""
        
        scenario_data = self.active_scenarios[scenario_id]
        presentation = scenario_data["presentation"]
        
        # Create agent-specific briefings
        for agent_id in target_agents:
            agent_briefing = await self.create_agent_specific_briefing(
                agent_id, presentation
            )
            
            # Simulate delivery to agent
            logger.info(f"Delivering scenario briefing to {agent_id}")
            
            # Store agent briefing for learning replay
            scenario_data["agent_responses"][agent_id] = {
                "briefing_delivered": datetime.now().isoformat(),
                "custom_briefing": agent_briefing,
                "status": "briefed",
                "learning_state_snapshot": asdict(self.agent_learning_states.get(agent_id))
            }

    async def create_agent_specific_briefing(self, agent_id: str, 
                                           presentation: ScenarioPresentation) -> Dict:
        """Create agent-specific briefing based on their learning needs"""
        
        agent_state = self.agent_learning_states.get(agent_id)
        if not agent_state:
            return {"error": f"Agent {agent_id} not found"}
        
        # Customize briefing based on agent type and learning state
        agent_type = agent_state.agent_type
        
        custom_briefing = {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "scenario_narrative": presentation.narrative,
            "role_specific_objectives": self.get_role_specific_objectives(agent_type, presentation),
            "personalized_hints": self.filter_learning_hints(agent_state, presentation.learning_hints),
            "skill_based_challenges": self.adapt_challenges_to_skill(agent_state, presentation.adaptive_challenges),
            "expected_learning_outcomes": presentation.scenario_context.learning_objectives
        }
        
        return custom_briefing

    def get_role_specific_objectives(self, agent_type: str, 
                                   presentation: ScenarioPresentation) -> List[str]:
        """Get role-specific objectives for different agent types"""
        
        objectives_map = {
            "red_team": [
                "Identify and exploit vulnerabilities",
                "Maintain persistence and stealth",
                "Achieve scenario-specific attack objectives",
                "Document attack vectors and IOCs"
            ],
            "blue_team": [
                "Detect and respond to threat activities", 
                "Implement defensive countermeasures",
                "Minimize impact and contain threats",
                "Conduct forensic analysis"
            ],
            "purple_team": [
                "Facilitate red and blue team collaboration",
                "Validate detection and response capabilities",
                "Optimize security controls",
                "Generate improvement recommendations"
            ]
        }
        
        return objectives_map.get(agent_type, ["General scenario participation"])

    async def monitor_agent_learning(self, scenario_id: str):
        """Monitor agent performance and adaptation during scenario execution"""
        
        if scenario_id not in self.active_scenarios:
            return
        
        scenario_data = self.active_scenarios[scenario_id]
        
        while scenario_data["status"] == "active":
            # Collect agent performance metrics
            for agent_id in scenario_data["target_agents"]:
                performance_metrics = await self.collect_agent_performance(agent_id, scenario_id)
                
                # Update learning state
                await self.update_agent_learning_state(agent_id, performance_metrics)
                
                # Provide adaptive guidance if needed
                if performance_metrics.get("struggling", False):
                    await self.provide_adaptive_guidance(agent_id, scenario_id)
            
            # Wait before next monitoring cycle
            await asyncio.sleep(30)  # Monitor every 30 seconds

    async def collect_agent_performance(self, agent_id: str, scenario_id: str) -> Dict:
        """Collect real-time agent performance metrics"""
        
        # Simulate performance data collection
        # In real implementation, this would interface with agent execution systems
        
        performance = {
            "agent_id": agent_id,
            "scenario_id": scenario_id,
            "timestamp": datetime.now().isoformat(),
            "actions_taken": 5,  # Simulated
            "success_rate": 0.7,  # Simulated
            "learning_indicators": {
                "adaptation_speed": 0.8,
                "technique_mastery": 0.6,
                "collaboration_effectiveness": 0.7
            },
            "struggling": False  # Would be determined by actual metrics
        }
        
        return performance

    async def update_agent_learning_state(self, agent_id: str, performance_metrics: Dict):
        """Update agent learning state based on performance"""
        
        if agent_id not in self.agent_learning_states:
            return
        
        agent_state = self.agent_learning_states[agent_id]
        
        # Update skill level based on performance
        success_rate = performance_metrics.get("success_rate", 0.5)
        learning_adjustment = (success_rate - 0.5) * agent_state.learning_rate
        agent_state.current_skill_level = max(0.0, min(1.0, 
            agent_state.current_skill_level + learning_adjustment))
        
        # Update recent performance
        agent_state.recent_performance[performance_metrics["timestamp"]] = success_rate
        
        # Track adaptation
        adaptation_record = {
            "timestamp": performance_metrics["timestamp"],
            "performance_change": learning_adjustment,
            "new_skill_level": agent_state.current_skill_level
        }
        agent_state.adaptation_history.append(adaptation_record)
        
        # Limit history size
        if len(agent_state.adaptation_history) > 100:
            agent_state.adaptation_history = agent_state.adaptation_history[-100:]

    async def provide_adaptive_guidance(self, agent_id: str, scenario_id: str):
        """Provide adaptive guidance to struggling agents"""
        
        agent_state = self.agent_learning_states.get(agent_id)
        scenario_data = self.active_scenarios.get(scenario_id)
        
        if not agent_state or not scenario_data:
            return
        
        # Generate guidance using NVIDIA AI
        guidance_prompt = f"""
        An agent is struggling in a cybersecurity scenario. Provide adaptive guidance:
        
        Agent: {agent_id} ({agent_state.agent_type})
        Current Skill Level: {agent_state.current_skill_level:.2f}
        Weaknesses: {agent_state.weaknesses}
        Scenario: {scenario_data['context'].name}
        
        Generate:
        1. Specific tactical advice (2-3 actionable items)
        2. Learning resources or references
        3. Adjusted difficulty recommendations
        4. Encouragement and motivation
        
        Keep guidance concise and immediately actionable.
        """
        
        try:
            guidance = await self.query_nvidia_ai(guidance_prompt, "adaptive_guidance")
            
            # Deliver guidance to agent
            logger.info(f"Providing adaptive guidance to {agent_id}")
            
            # Store guidance in scenario data
            if "adaptive_guidance" not in scenario_data:
                scenario_data["adaptive_guidance"] = {}
            
            scenario_data["adaptive_guidance"][agent_id] = {
                "timestamp": datetime.now().isoformat(),
                "guidance": guidance,
                "trigger": "performance_struggle"
            }
            
        except Exception as e:
            logger.error(f"Error providing adaptive guidance: {e}")

    def generate_fallback_scenario_context(self, template: Dict, agent_analysis: Dict, 
                                         custom_parameters: Dict) -> ScenarioContext:
        """Generate fallback scenario context when AI is unavailable"""
        
        return ScenarioContext(
            scenario_id=f"fallback_{datetime.now().strftime('%H%M%S')}",
            name=template["name"],
            description=f"Adaptive {template['name']} scenario",
            difficulty_level=int(agent_analysis["recommended_difficulty"]),
            target_environment=custom_parameters.get("environment", "enterprise"),
            learning_objectives=template["learning_objectives"],
            success_metrics={"completion_rate": 0.8, "learning_improvement": 0.2},
            duration_minutes=custom_parameters.get("duration", 60),
            threat_actor_profile=template["threat_profile"],
            industry_context="technology"
        )

    def create_fallback_presentation(self, scenario_context: ScenarioContext) -> ScenarioPresentation:
        """Create fallback presentation when AI is unavailable"""
        
        return ScenarioPresentation(
            timestamp=datetime.now().isoformat(),
            scenario_context=scenario_context,
            narrative=f"You are about to engage in {scenario_context.name}. This scenario will test your skills in a realistic cybersecurity environment.",
            technical_briefing="Standard technical briefing for scenario execution.",
            learning_hints=["Focus on methodical approach", "Document all actions", "Collaborate effectively"],
            adaptive_challenges=["Time pressure", "Limited information", "Evolving threats"],
            expected_agent_responses={"red_team": ["reconnaissance", "exploitation"], "blue_team": ["detection", "response"]}
        )

    def generate_fallback_ai_response(self, task_type: str) -> str:
        """Generate fallback AI response when NVIDIA API is unavailable"""
        
        fallback_responses = {
            "scenario_generation": json.dumps({
                "name": "Adaptive Cybersecurity Scenario",
                "description": "Dynamic scenario generated for learning",
                "difficulty_level": 5,
                "learning_objectives": ["threat detection", "incident response"],
                "success_metrics": {"completion": 0.8}
            }),
            "presentation_creation": json.dumps({
                "narrative": "A sophisticated threat has been detected in your environment.",
                "technical_briefing": "Investigate and respond to the security incident.",
                "learning_hints": ["Use systematic approach", "Document findings"],
                "adaptive_challenges": ["Time constraints", "Limited visibility"]
            }),
            "adaptive_guidance": "Focus on fundamentals and methodical approach. Review relevant documentation and collaborate with team members."
        }
        
        return fallback_responses.get(task_type, '{"status": "fallback_response"}')

    def filter_learning_hints(self, agent_state: AgentLearningState, all_hints: List[str]) -> List[str]:
        """Filter learning hints based on agent's specific needs"""
        
        # Simple filtering based on weaknesses
        relevant_hints = []
        for hint in all_hints:
            for weakness in agent_state.weaknesses:
                if weakness.lower() in hint.lower():
                    relevant_hints.append(hint)
                    break
        
        # Return top 3 most relevant hints
        return relevant_hints[:3] if relevant_hints else all_hints[:3]

    def adapt_challenges_to_skill(self, agent_state: AgentLearningState, 
                                challenges: List[str]) -> List[str]:
        """Adapt challenges based on agent's skill level"""
        
        skill_level = agent_state.current_skill_level
        
        if skill_level < 0.3:  # Beginner
            return [f"[BEGINNER] {challenge}" for challenge in challenges[:2]]
        elif skill_level < 0.7:  # Intermediate
            return [f"[INTERMEDIATE] {challenge}" for challenge in challenges]
        else:  # Advanced
            return [f"[ADVANCED] {challenge}" for challenge in challenges] + ["Additional complexity: Multi-vector attack"]

# Example usage and testing
async def main():
    """Example usage of the NVIDIA Scenario Orchestrator"""
    
    orchestrator = NVIDIAScenarioOrchestrator()
    
    # Present an APT scenario to mixed agent team
    target_agents = ["red_team_agent_01", "blue_team_agent_01", "purple_team_agent_01"]
    
    scenario_id = await orchestrator.present_dynamic_scenario(
        scenario_template="apt_infiltration",
        target_agents=target_agents,
        custom_parameters={
            "environment": "financial_services",
            "duration": 90
        }
    )
    
    print(f"Scenario {scenario_id} presented to agents")
    
    # Start monitoring agent learning
    await orchestrator.monitor_agent_learning(scenario_id)

if __name__ == "__main__":
    asyncio.run(main())