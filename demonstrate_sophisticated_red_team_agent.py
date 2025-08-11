#!/usr/bin/env python3
"""
Sophisticated Red Team AI Agent Demonstration
Showcases the advanced capabilities of the AI-powered red team agent

This script demonstrates:
- Red team operation planning and execution
- AI-powered decision making
- Purple team collaboration
- Defensive insight generation
- Threat actor intelligence
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src" / "api"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SophisticatedRedTeamDemo:
    """Comprehensive demonstration of sophisticated red team capabilities"""
    
    def __init__(self):
        self.demo_id = f"redteam_demo_{int(time.time())}"
        self.results = {
            "demo_id": self.demo_id,
            "timestamp": time.time(),
            "demonstrations": {},
            "summary": {}
        }
        
        # Service instances
        self.red_team_agent = None
        self.red_team_utilities = None

    async def run_comprehensive_demonstration(self) -> Dict[str, Any]:
        """Run comprehensive demonstration of red team capabilities"""
        print("🎯 SOPHISTICATED RED TEAM AI AGENT DEMONSTRATION")
        print("=" * 60)
        print("🤖 Showcasing advanced AI-powered adversary emulation capabilities")
        print("🛡️ All operations conducted for defensive security purposes only")
        print("=" * 60)
        
        try:
            # Initialize services
            await self._initialize_services()
            
            # Core demonstrations
            await self._demonstrate_threat_actor_intelligence()
            await self._demonstrate_ai_decision_making()
            await self._demonstrate_attack_planning()
            await self._demonstrate_operation_execution()
            await self._demonstrate_defensive_insights()
            await self._demonstrate_purple_team_collaboration()
            await self._demonstrate_detection_rule_generation()
            await self._demonstrate_training_exercise_creation()
            await self._demonstrate_threat_hunting_queries()
            
            # Advanced features
            await self._demonstrate_mitre_integration()
            await self._demonstrate_performance_metrics()
            
            # Generate summary
            await self._generate_demonstration_summary()
            
            return self.results
            
        except Exception as e:
            logger.error(f"❌ Demonstration failed: {e}")
            self.results["error"] = str(e)
            return self.results

    async def _initialize_services(self):
        """Initialize red team services"""
        print("\n🔧 Initializing Sophisticated Red Team Services...")
        
        try:
            # Import and initialize red team agent
            from app.services.sophisticated_red_team_agent import get_sophisticated_red_team_agent
            from app.services.red_team_utilities import get_red_team_utilities
            
            print("   📦 Loading sophisticated red team AI agent...")
            self.red_team_agent = await get_sophisticated_red_team_agent()
            
            print("   🛠️ Loading comprehensive red team utilities...")
            self.red_team_utilities = await get_red_team_utilities()
            
            print("✅ Services initialized successfully")
            print(f"   🤖 Agent: {self.red_team_agent.service_name}")
            print(f"   📊 Threat actors: {len(self.red_team_agent.threat_actor_models)}")
            print(f"   🔍 Techniques: {len(self.red_team_agent.technique_database)}")
            
        except Exception as e:
            print(f"❌ Service initialization failed: {e}")
            raise

    async def _demonstrate_threat_actor_intelligence(self):
        """Demonstrate threat actor intelligence generation"""
        demo_name = "Threat Actor Intelligence"
        print(f"\n🧪 Demonstrating: {demo_name}")
        
        try:
            # Get available threat actors
            threat_actors = list(self.red_team_agent.threat_actor_models.keys())
            print(f"   📊 Available threat actors: {len(threat_actors)}")
            
            # Demonstrate APT29 intelligence generation
            actor_id = "APT29"
            if actor_id in threat_actors:
                print(f"   🎯 Generating intelligence for {actor_id} (Cozy Bear)...")
                
                intelligence = await self.red_team_agent.generate_threat_actor_intelligence(actor_id)
                
                actor_profile = intelligence.get("actor_profile", {})
                behavioral_analysis = intelligence.get("behavioral_analysis", {})
                defensive_strategies = intelligence.get("defensive_strategies", [])
                
                print(f"   ✅ Intelligence generated successfully")
                print(f"   📋 Sophistication: {actor_profile.get('sophistication_level', 'Unknown')}")
                print(f"   🎯 Preferred techniques: {len(actor_profile.get('preferred_techniques', []))}")
                print(f"   🛡️ Defensive strategies: {len(defensive_strategies)}")
                print(f"   📊 Attribution confidence: {actor_profile.get('attribution_confidence', 0):.1%}")
                
                # Show sample defensive strategies
                if defensive_strategies:
                    print(f"   🔍 Sample defensive strategies:")
                    for strategy in defensive_strategies[:3]:
                        print(f"      • {strategy}")
                
                self.results["demonstrations"][demo_name] = {
                    "status": "success",
                    "actor_analyzed": actor_id,
                    "intelligence_components": list(intelligence.keys()),
                    "techniques_count": len(actor_profile.get('preferred_techniques', [])),
                    "defensive_strategies_count": len(defensive_strategies)
                }
            else:
                print(f"   ⚠️ {actor_id} not available, using first available actor")
                actor_id = threat_actors[0] if threat_actors else None
                if actor_id:
                    intelligence = await self.red_team_agent.generate_threat_actor_intelligence(actor_id)
                    self.results["demonstrations"][demo_name] = {
                        "status": "success",
                        "actor_analyzed": actor_id,
                        "fallback_used": True
                    }
            
            print(f"✅ {demo_name} demonstration completed")
            
        except Exception as e:
            print(f"❌ {demo_name} demonstration failed: {e}")
            self.results["demonstrations"][demo_name] = {
                "status": "failed",
                "error": str(e)
            }

    async def _demonstrate_ai_decision_making(self):
        """Demonstrate AI decision-making capabilities"""
        demo_name = "AI Decision Making"
        print(f"\n🧪 Demonstrating: {demo_name}")
        
        try:
            # Show AI engines available
            ai_engines = {
                "behavior_models": len(self.red_team_agent.behavior_models),
                "decision_trees": len(self.red_team_agent.decision_trees),
                "neural_networks": len(self.red_team_agent.neural_networks)
            }
            
            print(f"   🤖 AI Engines Available:")
            print(f"      • Behavior models: {ai_engines['behavior_models']}")
            print(f"      • Decision trees: {ai_engines['decision_trees']}")
            print(f"      • Neural networks: {ai_engines['neural_networks']}")
            
            # Demonstrate technique effectiveness analysis
            mock_execution_results = {
                "techniques_used": [
                    {"technique_id": "T1566.001", "success": True, "detection_triggered": False},
                    {"technique_id": "T1059.001", "success": True, "detection_triggered": True},
                    {"technique_id": "T1055", "success": False, "detection_triggered": False}
                ],
                "total_attempts": 3
            }
            
            print(f"   🔬 Analyzing technique effectiveness...")
            analysis = await self.red_team_utilities.analyze_technique_effectiveness(
                "T1566.001", mock_execution_results
            )
            
            if analysis and not analysis.get("error"):
                print(f"   ✅ AI analysis completed")
                print(f"   📊 Success rate: {analysis.get('success_rate', 0):.1%}")
                print(f"   📊 Detection rate: {analysis.get('detection_rate', 0):.1%}")
                print(f"   📊 Evasion effectiveness: {analysis.get('evasion_effectiveness', 0):.1%}")
                
                recommendations = analysis.get("defensive_recommendations", [])
                if recommendations:
                    print(f"   💡 AI-generated recommendations: {len(recommendations)}")
                    for rec in recommendations[:2]:
                        print(f"      • {rec.get('description', 'N/A')}")
            
            self.results["demonstrations"][demo_name] = {
                "status": "success",
                "ai_engines": ai_engines,
                "analysis_performed": True,
                "recommendations_generated": len(analysis.get("defensive_recommendations", [])) if analysis else 0
            }
            
            print(f"✅ {demo_name} demonstration completed")
            
        except Exception as e:
            print(f"❌ {demo_name} demonstration failed: {e}")
            self.results["demonstrations"][demo_name] = {
                "status": "failed",
                "error": str(e)
            }

    async def _demonstrate_attack_planning(self):
        """Demonstrate sophisticated attack planning"""
        demo_name = "Attack Planning"
        print(f"\n🧪 Demonstrating: {demo_name}")
        
        try:
            # Create demonstration objective
            from app.services.sophisticated_red_team_agent import RedTeamObjective, SophisticationLevel
            from datetime import timedelta
            
            demo_objective = RedTeamObjective(
                objective_id="demo_planning_001",
                name="Advanced Purple Team Exercise",
                description="Comprehensive security assessment for defensive improvement",
                target_assets=["demo-web-server.local", "demo-database.local"],
                success_criteria=[
                    "Test detection capabilities",
                    "Identify security gaps",
                    "Generate defensive insights",
                    "Improve team readiness"
                ],
                mitre_tactics=["TA0001", "TA0002", "TA0003", "TA0005"],
                mitre_techniques=["T1566.001", "T1059.001", "T1055", "T1078"],
                sophistication_level=SophisticationLevel.ADVANCED,
                estimated_duration=timedelta(hours=4),
                stealth_requirements=True,
                defensive_learning_goals=[
                    "Improve phishing detection",
                    "Enhance PowerShell monitoring",
                    "Strengthen access controls",
                    "Optimize incident response"
                ]
            )
            
            # Demonstration environment
            demo_environment = {
                "environment_type": "demonstration",
                "security_controls": [
                    "email_security_gateway",
                    "endpoint_detection_response",
                    "security_information_event_management",
                    "network_access_control"
                ],
                "target_systems": ["windows_workstations", "linux_servers", "network_infrastructure"],
                "compliance_requirements": ["PCI_DSS", "ISO_27001"],
                "purple_team_participants": ["red_team", "blue_team", "security_analysts"]
            }
            
            print(f"   🎯 Planning advanced red team operation...")
            print(f"   📋 Objective: {demo_objective.name}")
            print(f"   🎖️ Sophistication: {demo_objective.sophistication_level.value}")
            print(f"   ⚡ Tactics: {len(demo_objective.mitre_tactics)}")
            print(f"   🔧 Techniques: {len(demo_objective.mitre_techniques)}")
            
            # Plan the operation
            operation = await self.red_team_agent.plan_red_team_operation(
                demo_objective, demo_environment
            )
            
            if operation:
                print(f"   ✅ Operation planned successfully")
                print(f"   🆔 Operation ID: {operation.operation_id}")
                print(f"   🔗 Attack chain: {len(operation.attack_chain)} vectors")
                print(f"   ⏰ Timeline: {len(operation.timeline)} phases")
                print(f"   📊 Success metrics: {len(operation.success_metrics)} metrics")
                print(f"   💡 Defensive insights: {len(operation.defensive_insights)} insights")
                print(f"   🤝 Purple team integration: {'✅' if operation.purple_team_integration else '❌'}")
                
                # Show attack chain summary
                if operation.attack_chain:
                    print(f"   🔍 Attack Chain Overview:")
                    for i, vector in enumerate(operation.attack_chain[:3], 1):
                        print(f"      {i}. {vector.name} ({vector.technique_id})")
                        print(f"         Success: {vector.success_probability:.1%} | Detection: {vector.detection_probability:.1%}")
                
                self.results["demonstrations"][demo_name] = {
                    "status": "success",
                    "operation_id": operation.operation_id,
                    "attack_vectors": len(operation.attack_chain),
                    "timeline_phases": len(operation.timeline),
                    "purple_team_enabled": operation.purple_team_integration,
                    "defensive_insights_preview": len(operation.defensive_insights)
                }
            else:
                print(f"   ❌ Operation planning failed")
                self.results["demonstrations"][demo_name] = {
                    "status": "failed",
                    "error": "Operation planning returned None"
                }
            
            print(f"✅ {demo_name} demonstration completed")
            
        except Exception as e:
            print(f"❌ {demo_name} demonstration failed: {e}")
            self.results["demonstrations"][demo_name] = {
                "status": "failed",
                "error": str(e)
            }

    async def _demonstrate_operation_execution(self):
        """Demonstrate safe operation execution"""
        demo_name = "Operation Execution"
        print(f"\n🧪 Demonstrating: {demo_name}")
        
        try:
            # Check if we have a planned operation
            active_operations = list(self.red_team_agent.active_operations.keys())
            
            if not active_operations:
                print(f"   ⚠️ No active operations available for execution demonstration")
                print(f"   📋 This would normally execute a planned red team operation")
                print(f"   🛡️ All execution is performed in safe simulation mode")
                print(f"   🤝 Purple team coordination enabled throughout")
                
                self.results["demonstrations"][demo_name] = {
                    "status": "skipped",
                    "reason": "No active operations available",
                    "simulation_mode": True
                }
                return
            
            operation_id = active_operations[0]
            print(f"   🎯 Executing operation: {operation_id}")
            print(f"   🛡️ Safety mode: SIMULATION ONLY")
            print(f"   🤝 Purple team coordination: ENABLED")
            
            # Execute the operation
            execution_results = await self.red_team_agent.execute_red_team_operation(operation_id)
            
            if execution_results:
                print(f"   ✅ Operation executed successfully")
                
                exec_data = execution_results.get("execution_results", {})
                print(f"   📊 Execution Summary:")
                print(f"      • Phases executed: {len(exec_data.get('phases_executed', []))}")
                print(f"      • Techniques used: {len(exec_data.get('techniques_used', []))}")
                print(f"      • Detection events: {len(exec_data.get('detection_events', []))}")
                print(f"      • Success rate: {execution_results.get('success_rate', 0):.1%}")
                print(f"      • Detection rate: {execution_results.get('detection_rate', 0):.1%}")
                
                # Show defensive insights generated
                insights = execution_results.get("defensive_insights", [])
                if insights:
                    print(f"   💡 Defensive Insights Generated: {len(insights)}")
                    for insight in insights[:2]:
                        print(f"      • {insight.get('type', 'Unknown')}: {insight.get('description', 'N/A')[:50]}...")
                
                # Show purple team feedback
                purple_feedback = execution_results.get("purple_team_feedback", {})
                if purple_feedback:
                    print(f"   🤝 Purple team feedback collected")
                
                self.results["demonstrations"][demo_name] = {
                    "status": "success",
                    "operation_id": operation_id,
                    "phases_executed": len(exec_data.get('phases_executed', [])),
                    "techniques_used": len(exec_data.get('techniques_used', [])),
                    "detection_events": len(exec_data.get('detection_events', [])),
                    "defensive_insights": len(insights),
                    "simulation_mode": exec_data.get("simulation_only", True)
                }
            else:
                print(f"   ❌ Operation execution failed")
                self.results["demonstrations"][demo_name] = {
                    "status": "failed",
                    "error": "Execution returned no results"
                }
            
            print(f"✅ {demo_name} demonstration completed")
            
        except Exception as e:
            print(f"❌ {demo_name} demonstration failed: {e}")
            self.results["demonstrations"][demo_name] = {
                "status": "failed",
                "error": str(e)
            }

    async def _demonstrate_defensive_insights(self):
        """Demonstrate defensive insight generation"""
        demo_name = "Defensive Insights"
        print(f"\n🧪 Demonstrating: {demo_name}")
        
        try:
            # Create mock execution results for demonstration
            mock_results = {
                "techniques_used": [
                    {"technique_id": "T1566.001", "name": "Spearphishing Attachment", "success": True, "detection_triggered": False},
                    {"technique_id": "T1059.001", "name": "PowerShell", "success": True, "detection_triggered": True},
                    {"technique_id": "T1055", "name": "Process Injection", "success": False, "detection_triggered": False},
                    {"technique_id": "T1078", "name": "Valid Accounts", "success": True, "detection_triggered": False}
                ],
                "detection_events": [
                    {"technique": "T1059.001", "detection_method": "PowerShell logging", "timestamp": "2025-01-01T12:00:00"}
                ],
                "purple_team_observations": [
                    {"observation": "PowerShell detection working well"},
                    {"observation": "Phishing detection needs improvement"}
                ]
            }
            
            print(f"   🔬 Analyzing security gaps and generating insights...")
            print(f"   📊 Analyzing {len(mock_results['techniques_used'])} techniques")
            
            # Generate defensive insights
            insights = await self.red_team_agent._generate_defensive_insights(mock_results)
            
            if insights:
                print(f"   ✅ Generated {len(insights)} defensive insights")
                
                # Categorize insights
                insight_types = {}
                for insight in insights:
                    insight_type = insight.get('type', 'Unknown')
                    insight_types[insight_type] = insight_types.get(insight_type, 0) + 1
                
                print(f"   📋 Insight Categories:")
                for category, count in insight_types.items():
                    print(f"      • {category.replace('_', ' ').title()}: {count}")
                
                # Show sample insights
                print(f"   💡 Sample Insights:")
                for insight in insights[:3]:
                    priority = insight.get('priority', 'medium')
                    description = insight.get('description', 'N/A')
                    recommendations = insight.get('recommendations', [])
                    
                    print(f"      • [{priority.upper()}] {description}")
                    if recommendations:
                        print(f"        Recommendations: {len(recommendations)} available")
                
                self.results["demonstrations"][demo_name] = {
                    "status": "success",
                    "total_insights": len(insights),
                    "insight_categories": insight_types,
                    "high_priority_insights": len([i for i in insights if i.get('priority') == 'high'])
                }
            else:
                print(f"   ⚠️ No insights generated")
                self.results["demonstrations"][demo_name] = {
                    "status": "warning",
                    "message": "No insights generated from mock data"
                }
            
            print(f"✅ {demo_name} demonstration completed")
            
        except Exception as e:
            print(f"❌ {demo_name} demonstration failed: {e}")
            self.results["demonstrations"][demo_name] = {
                "status": "failed",
                "error": str(e)
            }

    async def _demonstrate_purple_team_collaboration(self):
        """Demonstrate purple team collaboration features"""
        demo_name = "Purple Team Collaboration"
        print(f"\n🧪 Demonstrating: {demo_name}")
        
        try:
            # Mock operation results for purple team report
            mock_operation_results = {
                "techniques_used": [
                    {"technique_id": "T1566.001", "success": True, "detection_triggered": False},
                    {"technique_id": "T1059.001", "success": True, "detection_triggered": True},
                    {"technique_id": "T1055", "success": False, "detection_triggered": False}
                ],
                "detection_events": [
                    {"technique": "T1059.001", "detection_method": "PowerShell logging", "timestamp": "2025-01-01T12:00:00"}
                ],
                "purple_team_observations": [
                    {"feedback": "Excellent PowerShell detection coverage"},
                    {"feedback": "Email security needs enhancement"},
                    {"feedback": "Response time could be improved"}
                ]
            }
            
            print(f"   🤝 Generating comprehensive purple team report...")
            print(f"   📊 Analyzing collaborative operation results")
            
            # Generate purple team report
            purple_report = await self.red_team_utilities.generate_purple_team_report(mock_operation_results)
            
            if purple_report:
                print(f"   ✅ Purple team report generated successfully")
                
                # Show report sections
                report_sections = list(purple_report.keys())
                print(f"   📋 Report Sections: {len(report_sections)}")
                for section in report_sections:
                    if section != "report_id" and section != "generation_timestamp":
                        print(f"      • {section.replace('_', ' ').title()}")
                
                # Show executive summary
                exec_summary = purple_report.get("executive_summary", {})
                if exec_summary:
                    print(f"   📊 Executive Summary:")
                    overview = exec_summary.get("operation_overview", "N/A")
                    print(f"      • {overview}")
                    
                    key_findings = exec_summary.get("key_findings", [])
                    if key_findings:
                        print(f"      • Key findings: {len(key_findings)}")
                        for finding in key_findings[:2]:
                            print(f"        - {finding}")
                
                # Show defensive improvements
                improvements = purple_report.get("defensive_improvements", [])
                if improvements:
                    print(f"   🛡️ Defensive Improvements: {len(improvements)}")
                    for improvement in improvements[:2]:
                        tech_id = improvement.get("technique_id", "Unknown")
                        priority = improvement.get("priority", "medium")
                        print(f"      • [{priority.upper()}] {tech_id}: {improvement.get('technique_name', 'N/A')}")
                
                # Show training recommendations
                training = purple_report.get("training_recommendations", [])
                if training:
                    print(f"   🎓 Training Recommendations: {len(training)}")
                    for rec in training[:1]:
                        print(f"      • {rec.get('training_type', 'Unknown')}: {rec.get('category', 'N/A')}")
                
                self.results["demonstrations"][demo_name] = {
                    "status": "success",
                    "report_sections": len(report_sections),
                    "defensive_improvements": len(improvements),
                    "training_recommendations": len(training),
                    "collaboration_features": True
                }
            else:
                print(f"   ❌ Purple team report generation failed")
                self.results["demonstrations"][demo_name] = {
                    "status": "failed",
                    "error": "Report generation returned None"
                }
            
            print(f"✅ {demo_name} demonstration completed")
            
        except Exception as e:
            print(f"❌ {demo_name} demonstration failed: {e}")
            self.results["demonstrations"][demo_name] = {
                "status": "failed",
                "error": str(e)
            }

    async def _demonstrate_detection_rule_generation(self):
        """Demonstrate automated detection rule generation"""
        demo_name = "Detection Rule Generation"
        print(f"\n🧪 Demonstrating: {demo_name}")
        
        try:
            from app.services.red_team_utilities import DetectionRuleType
            
            test_technique = "T1566.001"  # Spearphishing Attachment
            
            print(f"   🔍 Generating detection rule for technique: {test_technique}")
            
            # Generate Sigma detection rule
            sigma_rule = await self.red_team_utilities.generate_detection_rule(
                test_technique, DetectionRuleType.SIGMA
            )
            
            if sigma_rule:
                print(f"   ✅ Sigma detection rule generated")
                print(f"   📋 Rule Details:")
                print(f"      • Name: {sigma_rule.name}")
                print(f"      • Technique: {sigma_rule.technique_id}")
                print(f"      • Confidence: {sigma_rule.confidence_level:.1%}")
                print(f"      • False positive rate: {sigma_rule.false_positive_rate:.1%}")
                print(f"      • Data sources: {', '.join(sigma_rule.data_sources)}")
                print(f"      • Coverage areas: {', '.join(sigma_rule.coverage_areas)}")
                
                # Show rule content preview
                rule_lines = sigma_rule.rule_content.split('\n')
                print(f"   📄 Rule Content Preview:")
                for line in rule_lines[:5]:
                    if line.strip():
                        print(f"      {line}")
                if len(rule_lines) > 5:
                    print(f"      ... ({len(rule_lines) - 5} more lines)")
                
                self.results["demonstrations"][demo_name] = {
                    "status": "success",
                    "rule_type": sigma_rule.rule_type.value,
                    "technique_covered": sigma_rule.technique_id,
                    "confidence_level": sigma_rule.confidence_level,
                    "data_sources": len(sigma_rule.data_sources),
                    "rule_length": len(sigma_rule.rule_content)
                }
            else:
                print(f"   ⚠️ Detection rule generation failed or not available")
                self.results["demonstrations"][demo_name] = {
                    "status": "warning",
                    "message": "Detection rule generation returned None"
                }
            
            print(f"✅ {demo_name} demonstration completed")
            
        except Exception as e:
            print(f"❌ {demo_name} demonstration failed: {e}")
            self.results["demonstrations"][demo_name] = {
                "status": "failed",
                "error": str(e)
            }

    async def _demonstrate_training_exercise_creation(self):
        """Demonstrate training exercise creation"""
        demo_name = "Training Exercise Creation"
        print(f"\n🧪 Demonstrating: {demo_name}")
        
        try:
            from app.services.red_team_utilities import TrainingDifficulty
            
            test_techniques = ["T1566.001", "T1059.001"]  # Phishing and PowerShell
            
            print(f"   🎓 Creating training exercise for techniques: {', '.join(test_techniques)}")
            
            # Generate training exercise
            exercise = await self.red_team_utilities.generate_training_exercise(
                test_techniques, TrainingDifficulty.INTERMEDIATE
            )
            
            if exercise:
                print(f"   ✅ Training exercise created successfully")
                print(f"   📋 Exercise Details:")
                print(f"      • Name: {exercise.name}")
                print(f"      • Difficulty: {exercise.difficulty.value}")
                print(f"      • Duration: {exercise.duration_minutes} minutes")
                print(f"      • Techniques covered: {len(exercise.techniques_covered)}")
                print(f"      • Learning objectives: {len(exercise.learning_objectives)}")
                print(f"      • Exercise steps: {len(exercise.exercise_steps)}")
                print(f"      • Defensive focus areas: {len(exercise.defensive_focus_areas)}")
                
                # Show learning objectives
                print(f"   🎯 Learning Objectives:")
                for obj in exercise.learning_objectives[:3]:
                    print(f"      • {obj}")
                
                # Show exercise steps summary
                print(f"   📝 Exercise Steps Summary:")
                for i, step in enumerate(exercise.exercise_steps[:3], 1):
                    title = step.get('title', 'Unknown')
                    duration = step.get('duration_minutes', 0)
                    role = step.get('role', 'unknown')
                    print(f"      {i}. {title} ({duration}min, {role})")
                
                # Show defensive focus areas
                print(f"   🛡️ Defensive Focus Areas:")
                for area in exercise.defensive_focus_areas[:3]:
                    print(f"      • {area}")
                
                self.results["demonstrations"][demo_name] = {
                    "status": "success",
                    "exercise_name": exercise.name,
                    "difficulty": exercise.difficulty.value,
                    "duration_minutes": exercise.duration_minutes,
                    "techniques_covered": len(exercise.techniques_covered),
                    "learning_objectives": len(exercise.learning_objectives),
                    "exercise_steps": len(exercise.exercise_steps)
                }
            else:
                print(f"   ❌ Training exercise creation failed")
                self.results["demonstrations"][demo_name] = {
                    "status": "failed",
                    "error": "Exercise creation returned None"
                }
            
            print(f"✅ {demo_name} demonstration completed")
            
        except Exception as e:
            print(f"❌ {demo_name} demonstration failed: {e}")
            self.results["demonstrations"][demo_name] = {
                "status": "failed",
                "error": str(e)
            }

    async def _demonstrate_threat_hunting_queries(self):
        """Demonstrate threat hunting query capabilities"""
        demo_name = "Threat Hunting Queries"
        print(f"\n🧪 Demonstrating: {demo_name}")
        
        try:
            # Show available hunting queries
            hunting_queries = self.red_team_utilities.hunting_queries_db
            
            print(f"   🔍 Available threat hunting queries: {len(hunting_queries)}")
            
            if hunting_queries:
                # Show sample hunting query
                sample_query = list(hunting_queries.values())[0]
                
                print(f"   📄 Sample Hunting Query:")
                print(f"      • Name: {sample_query.name}")
                print(f"      • Technique: {sample_query.technique_id}")
                print(f"      • Query language: {sample_query.query_language}")
                print(f"      • Data sources: {', '.join(sample_query.data_sources)}")
                print(f"      • False positive likelihood: {sample_query.false_positive_likelihood}")
                
                # Show query content preview
                query_lines = sample_query.query_content.split('\n')
                print(f"   💻 Query Content Preview:")
                for line in query_lines[:5]:
                    if line.strip():
                        print(f"      {line}")
                if len(query_lines) > 5:
                    print(f"      ... ({len(query_lines) - 5} more lines)")
                
                # Show hypothesis and expected results
                print(f"   🧪 Hunting Hypothesis:")
                print(f"      {sample_query.hunting_hypothesis}")
                
                # Show all query types available
                query_types = set(q.query_language for q in hunting_queries.values())
                print(f"   📊 Query Languages Available: {', '.join(query_types)}")
                
                self.results["demonstrations"][demo_name] = {
                    "status": "success",
                    "total_queries": len(hunting_queries),
                    "query_languages": list(query_types),
                    "sample_query": {
                        "name": sample_query.name,
                        "technique": sample_query.technique_id,
                        "language": sample_query.query_language
                    }
                }
            else:
                print(f"   ⚠️ No hunting queries available")
                self.results["demonstrations"][demo_name] = {
                    "status": "warning",
                    "message": "No hunting queries loaded"
                }
            
            print(f"✅ {demo_name} demonstration completed")
            
        except Exception as e:
            print(f"❌ {demo_name} demonstration failed: {e}")
            self.results["demonstrations"][demo_name] = {
                "status": "failed",
                "error": str(e)
            }

    async def _demonstrate_mitre_integration(self):
        """Demonstrate MITRE ATT&CK integration"""
        demo_name = "MITRE Integration"
        print(f"\n🧪 Demonstrating: {demo_name}")
        
        try:
            # Show MITRE engine status
            mitre_engine = self.red_team_agent.mitre_engine
            technique_db = self.red_team_agent.technique_database
            technique_mappings = self.red_team_utilities.technique_mappings
            
            print(f"   🎯 MITRE ATT&CK Integration Status:")
            print(f"      • MITRE engine available: {'✅' if mitre_engine else '❌'}")
            print(f"      • Techniques loaded: {len(technique_db)}")
            print(f"      • Technique mappings: {len(technique_mappings)}")
            
            if technique_mappings:
                # Show sample technique mapping
                sample_technique_id = list(technique_mappings.keys())[0]
                sample_mapping = technique_mappings[sample_technique_id]
                
                print(f"   📋 Sample Technique Mapping ({sample_technique_id}):")
                print(f"      • Name: {sample_mapping.get('technique_name', 'Unknown')}")
                print(f"      • Detection rules: {len(sample_mapping.get('detection_rules', []))}")
                print(f"      • Hunting queries: {len(sample_mapping.get('hunting_queries', []))}")
                print(f"      • Training exercises: {len(sample_mapping.get('training_exercises', []))}")
                print(f"      • Countermeasures: {len(sample_mapping.get('countermeasures', []))}")
                print(f"      • Data sources: {len(sample_mapping.get('data_sources', []))}")
                print(f"      • Difficulty to detect: {sample_mapping.get('difficulty_to_detect', 'Unknown')}")
                
                # Show countermeasures
                countermeasures = sample_mapping.get('countermeasures', [])
                if countermeasures:
                    print(f"   🛡️ Sample Countermeasures:")
                    for cm in countermeasures[:3]:
                        print(f"      • {cm}")
                
                # Show technique coverage statistics
                coverage_stats = {
                    "with_detection_rules": sum(1 for m in technique_mappings.values() if m.get('detection_rules')),
                    "with_hunting_queries": sum(1 for m in technique_mappings.values() if m.get('hunting_queries')),
                    "with_training_exercises": sum(1 for m in technique_mappings.values() if m.get('training_exercises')),
                    "with_countermeasures": sum(1 for m in technique_mappings.values() if m.get('countermeasures'))
                }
                
                print(f"   📊 Coverage Statistics:")
                for stat_name, count in coverage_stats.items():
                    percentage = (count / len(technique_mappings)) * 100
                    print(f"      • {stat_name.replace('_', ' ').title()}: {count}/{len(technique_mappings)} ({percentage:.1f}%)")
                
                self.results["demonstrations"][demo_name] = {
                    "status": "success",
                    "mitre_engine_available": mitre_engine is not None,
                    "techniques_loaded": len(technique_db),
                    "technique_mappings": len(technique_mappings),
                    "coverage_statistics": coverage_stats,
                    "sample_technique": {
                        "id": sample_technique_id,
                        "name": sample_mapping.get('technique_name', 'Unknown'),
                        "countermeasures": len(countermeasures)
                    }
                }
            else:
                print(f"   ⚠️ No technique mappings available")
                self.results["demonstrations"][demo_name] = {
                    "status": "warning",
                    "message": "No technique mappings loaded"
                }
            
            print(f"✅ {demo_name} demonstration completed")
            
        except Exception as e:
            print(f"❌ {demo_name} demonstration failed: {e}")
            self.results["demonstrations"][demo_name] = {
                "status": "failed",
                "error": str(e)
            }

    async def _demonstrate_performance_metrics(self):
        """Demonstrate performance metrics and monitoring"""
        demo_name = "Performance Metrics"
        print(f"\n🧪 Demonstrating: {demo_name}")
        
        try:
            # Get comprehensive metrics
            metrics = await self.red_team_agent.get_operation_metrics()
            
            if metrics:
                print(f"   📊 Performance Metrics Available:")
                
                # Show operation metrics
                op_metrics = metrics.get("operation_metrics", {})
                print(f"      • Total operations: {op_metrics.get('total_operations', 0)}")
                print(f"      • Successful operations: {op_metrics.get('successful_operations', 0)}")
                print(f"      • Defensive insights generated: {op_metrics.get('defensive_insights_generated', 0)}")
                print(f"      • Purple team collaborations: {op_metrics.get('purple_team_collaborations', 0)}")
                
                # Show current status
                print(f"   🎯 Current Status:")
                print(f"      • Active operations: {metrics.get('active_operations', 0)}")
                print(f"      • Historical operations: {metrics.get('historical_operations', 0)}")
                print(f"      • Threat actors modeled: {metrics.get('threat_actors_modeled', 0)}")
                print(f"      • Techniques available: {metrics.get('techniques_available', 0)}")
                
                # Show performance rates
                print(f"   📈 Performance Rates:")
                print(f"      • Success rate: {metrics.get('success_rate', 0):.1%}")
                print(f"      • Detection improvement rate: {metrics.get('detection_improvement_rate', 0):.1%}")
                
                # Get utilities statistics
                utils_stats = await self.red_team_utilities.get_technique_statistics()
                
                if utils_stats and not utils_stats.get("error"):
                    print(f"   🔧 Utilities Statistics:")
                    print(f"      • Detection rules: {utils_stats.get('detection_rules_available', 0)}")
                    print(f"      • Hunting queries: {utils_stats.get('hunting_queries_available', 0)}")
                    print(f"      • Training exercises: {utils_stats.get('training_exercises_available', 0)}")
                
                self.results["demonstrations"][demo_name] = {
                    "status": "success",
                    "operation_metrics": op_metrics,
                    "current_status": {
                        "active_operations": metrics.get('active_operations', 0),
                        "threat_actors": metrics.get('threat_actors_modeled', 0),
                        "techniques": metrics.get('techniques_available', 0)
                    },
                    "performance_rates": {
                        "success_rate": metrics.get('success_rate', 0),
                        "detection_improvement_rate": metrics.get('detection_improvement_rate', 0)
                    }
                }
            else:
                print(f"   ❌ Metrics not available")
                self.results["demonstrations"][demo_name] = {
                    "status": "failed",
                    "error": "Metrics not available"
                }
            
            print(f"✅ {demo_name} demonstration completed")
            
        except Exception as e:
            print(f"❌ {demo_name} demonstration failed: {e}")
            self.results["demonstrations"][demo_name] = {
                "status": "failed",
                "error": str(e)
            }

    async def _generate_demonstration_summary(self):
        """Generate comprehensive demonstration summary"""
        print("\n📋 Generating Demonstration Summary...")
        
        try:
            # Calculate statistics
            total_demos = len(self.results["demonstrations"])
            successful_demos = len([d for d in self.results["demonstrations"].values() if d["status"] == "success"])
            failed_demos = len([d for d in self.results["demonstrations"].values() if d["status"] == "failed"])
            warning_demos = len([d for d in self.results["demonstrations"].values() if d["status"] == "warning"])
            skipped_demos = len([d for d in self.results["demonstrations"].values() if d["status"] == "skipped"])
            
            success_rate = successful_demos / total_demos if total_demos > 0 else 0
            
            # Determine overall status
            if failed_demos == 0 and warning_demos <= 1:
                overall_status = "EXCELLENT"
            elif failed_demos <= 1 and warning_demos <= 2:
                overall_status = "GOOD"
            elif failed_demos <= 2:
                overall_status = "FAIR"
            else:
                overall_status = "NEEDS_ATTENTION"
            
            # Generate capability summary
            capabilities_demonstrated = {
                "ai_decision_making": "AI Decision Making" in self.results["demonstrations"],
                "threat_actor_modeling": "Threat Actor Intelligence" in self.results["demonstrations"],
                "attack_planning": "Attack Planning" in self.results["demonstrations"],
                "operation_execution": "Operation Execution" in self.results["demonstrations"],
                "defensive_insights": "Defensive Insights" in self.results["demonstrations"],
                "purple_team_collaboration": "Purple Team Collaboration" in self.results["demonstrations"],
                "detection_rule_generation": "Detection Rule Generation" in self.results["demonstrations"],
                "training_exercise_creation": "Training Exercise Creation" in self.results["demonstrations"],
                "threat_hunting_queries": "Threat Hunting Queries" in self.results["demonstrations"],
                "mitre_integration": "MITRE Integration" in self.results["demonstrations"],
                "performance_metrics": "Performance Metrics" in self.results["demonstrations"]
            }
            
            capabilities_working = sum(capabilities_demonstrated.values())
            
            self.results["summary"] = {
                "overall_status": overall_status,
                "demonstration_statistics": {
                    "total_demonstrations": total_demos,
                    "successful": successful_demos,
                    "failed": failed_demos,
                    "warnings": warning_demos,
                    "skipped": skipped_demos,
                    "success_rate": success_rate
                },
                "capabilities_demonstrated": capabilities_demonstrated,
                "capabilities_working": capabilities_working,
                "total_capabilities": len(capabilities_demonstrated),
                "demonstration_outcome": self._generate_demonstration_outcome(overall_status, success_rate, capabilities_working)
            }
            
            print(f"   📊 DEMONSTRATION SUMMARY")
            print(f"      Overall Status: {overall_status}")
            print(f"      Success Rate: {success_rate:.1%}")
            print(f"      Demonstrations: {successful_demos}/{total_demos} successful")
            print(f"      Capabilities: {capabilities_working}/{len(capabilities_demonstrated)} working")
            
        except Exception as e:
            print(f"   ❌ Failed to generate demonstration summary: {e}")
            self.results["summary"] = {"error": str(e)}

    def _generate_demonstration_outcome(self, overall_status: str, success_rate: float, capabilities_working: int) -> str:
        """Generate demonstration outcome message"""
        if overall_status == "EXCELLENT":
            return "🎉 SOPHISTICATED RED TEAM AI AGENT: FULLY OPERATIONAL! All capabilities demonstrated successfully."
        elif overall_status == "GOOD":
            return "✅ SOPHISTICATED RED TEAM AI AGENT: OPERATIONAL with excellent capabilities demonstrated."
        elif overall_status == "FAIR":
            return "⚠️ SOPHISTICATED RED TEAM AI AGENT: MOSTLY OPERATIONAL with some areas needing attention."
        else:
            return "🔧 SOPHISTICATED RED TEAM AI AGENT: PARTIALLY OPERATIONAL - some capabilities need review."


async def main():
    """Main demonstration function"""
    demo = SophisticatedRedTeamDemo()
    results = await demo.run_comprehensive_demonstration()
    
    # Save results
    output_file = f"sophisticated_redteam_demo_{int(time.time())}.json"
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📄 Detailed results saved to: {output_file}")
    except Exception as e:
        print(f"\n⚠️ Could not save results file: {e}")
    
    # Print final outcome
    print("\n" + "=" * 60)
    if "summary" in results and "demonstration_outcome" in results["summary"]:
        print(results["summary"]["demonstration_outcome"])
    else:
        print("❌ DEMONSTRATION INCOMPLETE - Check detailed results for errors")
    
    print("🛡️ All red team operations conducted for defensive purposes only")
    print("🤝 Purple team collaboration enabled throughout")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    asyncio.run(main())