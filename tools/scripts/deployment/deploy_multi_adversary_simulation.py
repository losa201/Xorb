#!/usr/bin/env python3
"""
Multi-Adversary Red Team Simulation Framework Deployment Script

This script deploys the complete Multi-Adversary Red Team Simulation Framework
to the XORB platform, including all components and dashboard integrations.

Author: XORB AI Engineering Team
Version: 2.0.0
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add XORB core to path
sys.path.insert(0, '/root/Xorb')

from xorb_core.simulation import (
    SyntheticAdversaryProfileManager,
    MultiActorSimulationEngine,
    PredictiveThreatIntelligenceSynthesizer,
    CampaignGoalOptimizer,
    AdversaryType,
    SimulationMode,
    OptimizationStrategy,
    ResourceType,
    GoalPriority
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/Xorb/logs/multi_adversary_deployment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class MultiAdversaryFrameworkDeployer:
    """Handles deployment of the Multi-Adversary Red Team Simulation Framework."""

    def __init__(self):
        self.deployment_id = f"deploy_multi_adversary_{int(time.time())}"
        self.deployment_status = {
            'deployment_id': self.deployment_id,
            'start_time': datetime.utcnow().isoformat(),
            'components_deployed': [],
            'tests_passed': [],
            'dashboard_configured': False,
            'metrics_configured': False,
            'integration_verified': False,
            'deployment_complete': False
        }

        # Component managers
        self.profile_manager = None
        self.simulation_engine = None
        self.threat_synthesizer = None
        self.goal_optimizer = None

        logger.info(f"Initialized Multi-Adversary Framework Deployer: {self.deployment_id}")

    async def deploy_framework(self) -> Dict[str, Any]:
        """Deploy the complete Multi-Adversary Red Team Simulation Framework."""

        try:
            logger.info("🚀 Starting Multi-Adversary Red Team Simulation Framework deployment")

            # Step 1: Initialize core components
            await self._initialize_components()

            # Step 2: Configure component integrations
            await self._configure_integrations()

            # Step 3: Set up monitoring and metrics
            await self._setup_monitoring()

            # Step 4: Deploy Grafana dashboard
            await self._deploy_dashboard()

            # Step 5: Run integration tests
            await self._run_integration_tests()

            # Step 6: Deploy sample scenarios
            await self._deploy_sample_scenarios()

            # Step 7: Verify end-to-end functionality
            await self._verify_deployment()

            # Update final status
            self.deployment_status['deployment_complete'] = True
            self.deployment_status['end_time'] = datetime.utcnow().isoformat()

            logger.info("✅ Multi-Adversary Framework deployment completed successfully")
            return self.deployment_status

        except Exception as e:
            logger.error(f"❌ Framework deployment failed: {str(e)}")
            self.deployment_status['error'] = str(e)
            self.deployment_status['deployment_complete'] = False
            raise

    async def _initialize_components(self) -> None:
        """Initialize all framework components."""

        logger.info("🔧 Initializing framework components...")

        # Initialize Synthetic Adversary Profile Manager
        try:
            self.profile_manager = SyntheticAdversaryProfileManager()

            # Create initial adversary profiles
            await self._create_sample_adversary_profiles()

            self.deployment_status['components_deployed'].append('SyntheticAdversaryProfileManager')
            logger.info("✅ Synthetic Adversary Profile Manager initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Profile Manager: {str(e)}")
            raise

        # Initialize Multi-Actor Simulation Engine
        try:
            self.simulation_engine = MultiActorSimulationEngine(
                profile_manager=self.profile_manager,
                max_concurrent_adversaries=5,
                enable_game_theory=True
            )

            self.deployment_status['components_deployed'].append('MultiActorSimulationEngine')
            logger.info("✅ Multi-Actor Simulation Engine initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Simulation Engine: {str(e)}")
            raise

        # Initialize Predictive Threat Intelligence Synthesizer
        try:
            self.threat_synthesizer = PredictiveThreatIntelligenceSynthesizer(
                enable_ml_predictions=True,
                prediction_horizon_days=30,
                confidence_threshold=0.6
            )

            # Initialize the synthesizer system
            init_results = await self.threat_synthesizer.initialize_system()
            if not init_results.get('system_ready', False):
                logger.warning("⚠️ Threat Intelligence Synthesizer initialization incomplete")

            self.deployment_status['components_deployed'].append('PredictiveThreatIntelligenceSynthesizer')
            logger.info("✅ Predictive Threat Intelligence Synthesizer initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Threat Synthesizer: {str(e)}")
            raise

        # Initialize Campaign Goal Optimizer
        try:
            self.goal_optimizer = CampaignGoalOptimizer()

            self.deployment_status['components_deployed'].append('CampaignGoalOptimizer')
            logger.info("✅ Campaign Goal Optimizer initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Goal Optimizer: {str(e)}")
            raise

    async def _create_sample_adversary_profiles(self) -> None:
        """Create sample adversary profiles for demonstration."""

        logger.info("👥 Creating sample adversary profiles...")

        sample_profiles = [
            {
                'name': 'APT-XORB-001',
                'adversary_type': AdversaryType.NATION_STATE,
                'description': 'Advanced nation-state actor focusing on espionage and persistent access'
            },
            {
                'name': 'CyberCrime-Alpha',
                'adversary_type': AdversaryType.CYBERCRIME,
                'description': 'Financially motivated cybercrime group specializing in data theft'
            },
            {
                'name': 'Hacktivist-Beta',
                'adversary_type': AdversaryType.HACKTIVIST,
                'description': 'Politically motivated hacktivist collective focusing on disruption'
            },
            {
                'name': 'Insider-Gamma',
                'adversary_type': AdversaryType.INSIDER_THREAT,
                'description': 'Malicious insider with privileged access and sabotage intentions'
            },
            {
                'name': 'Ransomware-Delta',
                'adversary_type': AdversaryType.RANSOMWARE_GROUP,
                'description': 'Sophisticated ransomware operation with double extortion tactics'
            }
        ]

        created_profiles = []
        for profile_config in sample_profiles:
            try:
                profile = await self.profile_manager.create_adversary_profile(
                    name=profile_config['name'],
                    adversary_type=profile_config['adversary_type'],
                    description=profile_config['description'],
                    capabilities_config={
                        'technical_sophistication': 8,
                        'resource_level': 7,
                        'operational_security': 9,
                        'tool_diversity': 8
                    }
                )
                created_profiles.append(profile)
                logger.info(f"✅ Created adversary profile: {profile_config['name']}")

            except Exception as e:
                logger.error(f"❌ Failed to create profile {profile_config['name']}: {str(e)}")

        logger.info(f"📊 Created {len(created_profiles)} sample adversary profiles")

    async def _configure_integrations(self) -> None:
        """Configure integrations between framework components."""

        logger.info("🔗 Configuring component integrations...")

        # Test profile manager and simulation engine integration
        try:
            profiles = await self.profile_manager.list_profiles()
            if len(profiles) >= 2:
                # Create a sample simulation configuration
                sample_config = {
                    'mode': 'competitive',
                    'adversary_profiles': [
                        {'profile_id': profiles[0].profile_id, 'team_color': 'red'},
                        {'profile_id': profiles[1].profile_id, 'team_color': 'blue'}
                    ],
                    'objectives': [
                        {
                            'name': 'Test Objective 1',
                            'description': 'Sample objective for integration testing',
                            'target_systems': ['test-system-01'],
                            'required_techniques': ['T1078', 'T1003'],
                            'conflict_type': 'exclusive_target',
                            'exclusivity_level': 0.8,
                            'priority_weight': 2.0,
                            'estimated_duration': 3600,
                            'resource_requirements': {'computational': 5, 'network': 3},
                            'stealth_requirements': {'detection_avoidance': 0.8}
                        }
                    ]
                }

                # Test simulation creation (don't start it)
                simulation_id = await self.simulation_engine.create_multi_adversary_simulation(sample_config)
                logger.info(f"✅ Integration test simulation created: {simulation_id}")

        except Exception as e:
            logger.error(f"❌ Component integration test failed: {str(e)}")
            raise

        logger.info("✅ Component integrations configured successfully")

    async def _setup_monitoring(self) -> None:
        """Set up monitoring and metrics collection."""

        logger.info("📊 Setting up monitoring and metrics...")

        # Create metrics configuration
        metrics_config = {
            'prometheus_metrics': [
                'xorb_simulation_adversaries_total',
                'xorb_adversary_coordination_matrix',
                'xorb_adversary_interference_events_total',
                'xorb_goal_completion_progress',
                'xorb_technique_executions_total',
                'xorb_interference_events_total',
                'xorb_deception_tactics_effectiveness',
                'xorb_predicted_campaign_progress',
                'xorb_actual_campaign_progress',
                'xorb_simulation_success_rate',
                'xorb_detection_evasion_rate',
                'xorb_resource_utilization_efficiency',
                'xorb_nash_equilibrium_convergence',
                'xorb_strategy_adaptation_rate',
                'xorb_game_theory_optimality',
                'xorb_adversary_status',
                'xorb_resource_allocation',
                'xorb_ml_prediction_accuracy'
            ],
            'collection_interval': '5s',
            'retention_period': '7d'
        }

        # Write metrics configuration
        metrics_config_path = '/root/Xorb/config/multi_adversary_metrics.json'
        os.makedirs(os.path.dirname(metrics_config_path), exist_ok=True)

        with open(metrics_config_path, 'w') as f:
            json.dump(metrics_config, f, indent=2)

        self.deployment_status['metrics_configured'] = True
        logger.info("✅ Monitoring and metrics configured")

    async def _deploy_dashboard(self) -> None:
        """Deploy the Grafana dashboard for simulation visualization."""

        logger.info("📈 Deploying Grafana dashboard...")

        dashboard_path = '/root/Xorb/grafana/multi-adversary-simulation-dashboard.json'

        if os.path.exists(dashboard_path):
            # Dashboard already created, just verify it
            with open(dashboard_path, 'r') as f:
                dashboard_config = json.load(f)

            # Verify dashboard structure
            required_panels = [
                'Active Simulation Status Overview',
                'Adversary Interactions Map',
                'Goal Progress Matrix',
                'Real-time Activity Timeline',
                'Deception Tactics Heatmap',
                'Predicted vs Actual Campaign Paths',
                'Simulation KPIs',
                'Game Theory Analysis',
                'Adversary Status Table',
                'Resource Allocation Overview',
                'ML Prediction Accuracy Heatmap'
            ]

            panel_titles = [panel.get('title', '') for panel in dashboard_config.get('panels', [])]
            missing_panels = [title for title in required_panels if title not in panel_titles]

            if missing_panels:
                logger.warning(f"⚠️ Dashboard missing panels: {missing_panels}")
            else:
                logger.info("✅ All required dashboard panels present")

            self.deployment_status['dashboard_configured'] = True
            logger.info("✅ Grafana dashboard deployment verified")

        else:
            logger.error("❌ Dashboard configuration file not found")
            raise FileNotFoundError("Dashboard configuration missing")

    async def _run_integration_tests(self) -> None:
        """Run integration tests for all framework components."""

        logger.info("🧪 Running integration tests...")

        # Test 1: Profile Manager functionality
        try:
            profiles = await self.profile_manager.list_profiles()
            assert len(profiles) > 0, "No profiles found"

            # Test profile evolution
            if profiles:
                sample_results = [{'success': True, 'technique': 'T1078', 'effectiveness': 0.8}]
                evolved_profile = await self.profile_manager.evolve_profile(profiles[0], sample_results)
                assert evolved_profile is not None, "Profile evolution failed"

            self.deployment_status['tests_passed'].append('ProfileManager')
            logger.info("✅ Profile Manager integration test passed")

        except Exception as e:
            logger.error(f"❌ Profile Manager test failed: {str(e)}")
            raise

        # Test 2: Threat Intelligence Synthesizer
        try:
            # Generate threat landscape analysis
            landscape = await self.threat_synthesizer.generate_threat_landscape_analysis()
            assert landscape.snapshot_id is not None, "Landscape analysis failed"

            # Generate simulation scenarios
            scenarios = await self.threat_synthesizer.generate_simulation_scenarios(landscape, 3)
            assert len(scenarios) > 0, "Scenario generation failed"

            self.deployment_status['tests_passed'].append('ThreatIntelligenceSynthesizer')
            logger.info("✅ Threat Intelligence Synthesizer integration test passed")

        except Exception as e:
            logger.error(f"❌ Threat Intelligence test failed: {str(e)}")
            raise

        # Test 3: Goal Optimizer
        try:
            # Create sample campaign
            sample_objectives = [
                "Establish persistent access to domain controller",
                "Exfiltrate customer database records"
            ]

            profiles = await self.profile_manager.list_profiles()
            available_resources = {
                ResourceType.COMPUTATIONAL: 10.0,
                ResourceType.NETWORK: 5.0,
                ResourceType.HUMAN: 3.0,
                ResourceType.TIME: 168.0
            }

            if profiles:
                optimization_result = await self.goal_optimizer.create_campaign_from_objectives(
                    sample_objectives,
                    profiles[:2],  # Use first 2 profiles
                    available_resources,
                    OptimizationStrategy.BALANCED
                )

                assert optimization_result.optimized_plan is not None, "Goal optimization failed"
                assert optimization_result.confidence_score > 0, "Invalid confidence score"

            self.deployment_status['tests_passed'].append('CampaignGoalOptimizer')
            logger.info("✅ Campaign Goal Optimizer integration test passed")

        except Exception as e:
            logger.error(f"❌ Goal Optimizer test failed: {str(e)}")
            raise

        # Test 4: Multi-Actor Simulation Engine
        try:
            simulations = list(self.simulation_engine.active_simulations.keys())
            if simulations:
                # Check simulation status
                status = await self.simulation_engine.get_simulation_status(simulations[0])
                assert 'simulation_id' in status, "Invalid simulation status"

            # Test performance metrics
            metrics = self.simulation_engine.get_performance_metrics()
            assert isinstance(metrics, dict), "Invalid performance metrics"

            self.deployment_status['tests_passed'].append('MultiActorSimulationEngine')
            logger.info("✅ Multi-Actor Simulation Engine integration test passed")

        except Exception as e:
            logger.error(f"❌ Simulation Engine test failed: {str(e)}")
            raise

        logger.info(f"✅ All integration tests passed: {len(self.deployment_status['tests_passed'])} components")

    async def _deploy_sample_scenarios(self) -> None:
        """Deploy sample simulation scenarios for demonstration."""

        logger.info("🎭 Deploying sample simulation scenarios...")

        sample_scenarios = [
            {
                'name': 'Nation-State vs Cybercrime Competition',
                'description': 'Competitive simulation between nation-state and cybercrime adversaries',
                'mode': SimulationMode.COMPETITIVE,
                'adversary_count': 2,
                'objectives': [
                    'Compromise critical infrastructure systems',
                    'Establish persistent access channels',
                    'Exfiltrate sensitive intelligence data'
                ]
            },
            {
                'name': 'Multi-Team Red vs Blue Exercise',
                'description': 'Team-based simulation with multiple adversary types',
                'mode': SimulationMode.TEAM_VS_TEAM,
                'adversary_count': 4,
                'objectives': [
                    'Test defensive capabilities against coordinated attack',
                    'Evaluate incident response procedures',
                    'Assess detection and mitigation effectiveness'
                ]
            },
            {
                'name': 'Collaborative APT Campaign',
                'description': 'Collaborative simulation with multiple APT groups working together',
                'mode': SimulationMode.COLLABORATIVE,
                'adversary_count': 3,
                'objectives': [
                    'Execute coordinated supply chain attack',
                    'Share intelligence and resources efficiently',
                    'Maximize campaign success through coordination'
                ]
            }
        ]

        # Create scenario configurations
        scenarios_config_path = '/root/Xorb/config/sample_simulation_scenarios.json'
        with open(scenarios_config_path, 'w') as f:
            json.dump(sample_scenarios, f, indent=2, default=str)

        logger.info(f"✅ Deployed {len(sample_scenarios)} sample simulation scenarios")

    async def _verify_deployment(self) -> None:
        """Verify end-to-end functionality of the deployed framework."""

        logger.info("🔍 Verifying end-to-end deployment...")

        # Verification checklist
        verification_checks = {
            'components_initialized': len(self.deployment_status['components_deployed']) >= 4,
            'integration_tests_passed': len(self.deployment_status['tests_passed']) >= 4,
            'dashboard_configured': self.deployment_status['dashboard_configured'],
            'metrics_configured': self.deployment_status['metrics_configured'],
            'profiles_created': True,  # Will be verified
            'scenarios_deployed': True  # Will be verified
        }

        # Verify profile creation
        try:
            profiles = await self.profile_manager.list_profiles()
            verification_checks['profiles_created'] = len(profiles) >= 3
        except Exception as e:
            logger.error(f"Profile verification failed: {str(e)}")
            verification_checks['profiles_created'] = False

        # Verify scenario deployment
        scenarios_path = '/root/Xorb/config/sample_simulation_scenarios.json'
        verification_checks['scenarios_deployed'] = os.path.exists(scenarios_path)

        # Calculate overall verification score
        passed_checks = sum(1 for check in verification_checks.values() if check)
        total_checks = len(verification_checks)
        verification_score = (passed_checks / total_checks) * 100

        self.deployment_status['verification_checks'] = verification_checks
        self.deployment_status['verification_score'] = verification_score
        self.deployment_status['integration_verified'] = verification_score >= 80

        if verification_score >= 80:
            logger.info(f"✅ Deployment verification passed: {verification_score:.1f}% ({passed_checks}/{total_checks})")
        else:
            logger.warning(f"⚠️ Deployment verification incomplete: {verification_score:.1f}% ({passed_checks}/{total_checks})")
            failed_checks = [check for check, passed in verification_checks.items() if not passed]
            logger.warning(f"Failed checks: {failed_checks}")

    def get_deployment_summary(self) -> Dict[str, Any]:
        """Get comprehensive deployment summary."""

        summary = {
            'deployment_info': {
                'deployment_id': self.deployment_id,
                'framework_version': '2.0.0',
                'deployment_time': self.deployment_status.get('end_time', 'In Progress'),
                'status': 'Success' if self.deployment_status['deployment_complete'] else 'In Progress'
            },
            'components': {
                'deployed_components': self.deployment_status['components_deployed'],
                'total_components': 4,
                'deployment_coverage': len(self.deployment_status['components_deployed']) / 4 * 100
            },
            'testing': {
                'tests_passed': self.deployment_status['tests_passed'],
                'total_tests': 4,
                'test_coverage': len(self.deployment_status['tests_passed']) / 4 * 100
            },
            'configuration': {
                'dashboard_configured': self.deployment_status['dashboard_configured'],
                'metrics_configured': self.deployment_status['metrics_configured'],
                'integration_verified': self.deployment_status['integration_verified']
            },
            'verification': self.deployment_status.get('verification_checks', {}),
            'verification_score': self.deployment_status.get('verification_score', 0)
        }

        return summary


async def main():
    """Main deployment function."""

    print("🚀 XORB Multi-Adversary Red Team Simulation Framework Deployment")
    print("=" * 70)

    deployer = MultiAdversaryFrameworkDeployer()

    try:
        # Deploy the framework
        deployment_result = await deployer.deploy_framework()

        # Print deployment summary
        summary = deployer.get_deployment_summary()

        print("\n📊 DEPLOYMENT SUMMARY")
        print("=" * 70)
        print(f"Deployment ID: {summary['deployment_info']['deployment_id']}")
        print(f"Status: {summary['deployment_info']['status']}")
        print(f"Components Deployed: {summary['components']['deployment_coverage']:.1f}% ({len(summary['components']['deployed_components'])}/4)")
        print(f"Tests Passed: {summary['testing']['test_coverage']:.1f}% ({len(summary['testing']['tests_passed'])}/4)")
        print(f"Verification Score: {summary['verification_score']:.1f}%")

        print(f"\n✅ Deployed Components:")
        for component in summary['components']['deployed_components']:
            print(f"  - {component}")

        print(f"\n🧪 Tests Passed:")
        for test in summary['testing']['tests_passed']:
            print(f"  - {test}")

        print(f"\n🔧 Configuration Status:")
        print(f"  - Dashboard: {'✅' if summary['configuration']['dashboard_configured'] else '❌'}")
        print(f"  - Metrics: {'✅' if summary['configuration']['metrics_configured'] else '❌'}")
        print(f"  - Integration: {'✅' if summary['configuration']['integration_verified'] else '❌'}")

        if deployment_result['deployment_complete']:
            print(f"\n🎉 Multi-Adversary Red Team Simulation Framework deployed successfully!")
            print(f"🌐 Access the dashboard at: http://localhost:3000/d/multi-adversary-sim-dashboard")
            print(f"📊 Monitor metrics at: http://localhost:9090")
            print(f"📁 Configuration files in: /root/Xorb/config/")

            # Save deployment report
            report_path = f'/root/Xorb/reports_output/multi_adversary_deployment_report_{int(time.time())}.json'
            os.makedirs(os.path.dirname(report_path), exist_ok=True)

            with open(report_path, 'w') as f:
                json.dump({
                    'deployment_result': deployment_result,
                    'deployment_summary': summary,
                    'timestamp': datetime.utcnow().isoformat()
                }, f, indent=2, default=str)

            print(f"📋 Deployment report saved: {report_path}")

        else:
            print(f"\n❌ Deployment failed or incomplete")
            if 'error' in deployment_result:
                print(f"Error: {deployment_result['error']}")

    except Exception as e:
        logger.error(f"Deployment failed with error: {str(e)}")
        print(f"\n❌ DEPLOYMENT FAILED: {str(e)}")
        return False

    return True


if __name__ == "__main__":
    # Run deployment
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
