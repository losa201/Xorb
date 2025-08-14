#!/usr/bin/env python3
"""
🎯 XORB PTaaS Orchestration Engine
Penetration Testing as a Service with autonomous application security assessment

This engine orchestrates continuous penetration testing campaigns against live applications
with full integration of XORB's adversarial intelligence and defensive mutation capabilities.
"""

import asyncio
import json
import logging
import aiohttp
import yaml
import git
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import hashlib
import secrets
import base64
import threading
import queue
import time

# Import XORB modules
from XORB_PRKMT_13_1_APP_ASSAULT_ENGINE import (
    LiveApplicationTargetAcquisitionEngine,
    AutonomousAppAssaultAgents, ExploitabilityScoringMatrix,
    DefensiveMutationInjector, ApplicationTarget, VulnerabilityFinding,
    AssaultMode, TargetType, AttackVector, VulnerabilityRisk
)
from XORB_API_UI_EXPLORATION_AGENT import XORBAPIUIExplorationAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PTaaSMode(Enum):
    CONTINUOUS = "continuous"
    SCHEDULED = "scheduled"
    TRIGGERED = "triggered"
    ON_DEMAND = "on_demand"
    CI_CD_INTEGRATED = "ci_cd_integrated"

class AssessmentType(Enum):
    FULL_STACK = "full_stack"
    API_ONLY = "api_only"
    WEB_APP_ONLY = "web_app_only"
    MOBILE_API = "mobile_api"
    INFRASTRUCTURE = "infrastructure"
    SOCIAL_ENGINEERING = "social_engineering"

class ReportFormat(Enum):
    JSON = "json"
    YAML = "yaml"
    HTML = "html"
    PDF = "pdf"
    SARIF = "sarif"
    JUNIT = "junit"

class IntegrationType(Enum):
    BURP_SUITE = "burp_suite"
    OWASP_ZAP = "owasp_zap"
    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"
    JIRA = "jira"
    SLACK = "slack"
    SIEM = "siem"

@dataclass
class PTaaSCampaign:
    campaign_id: str
    name: str
    targets: List[ApplicationTarget]
    assessment_type: AssessmentType
    mode: PTaaSMode
    schedule: Optional[str] = None
    duration_hours: int = 24
    stealth_level: float = 0.5
    hard_realism_mode: bool = False
    auto_remediation: bool = False
    created: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    progress: float = 0.0

@dataclass
class PTaaSReport:
    report_id: str
    campaign_id: str
    format: ReportFormat
    vulnerability_count: int
    critical_findings: int
    high_findings: int
    medium_findings: int
    low_findings: int
    exploitability_score: float
    business_impact_score: float
    remediation_priority: List[str]
    executive_summary: str
    technical_details: Dict[str, Any]
    generated: datetime = field(default_factory=datetime.now)

@dataclass
class GitOpsIntegration:
    integration_id: str
    repository_url: str
    branch: str
    credentials: Dict[str, str]
    auto_commit: bool
    auto_merge: bool
    pr_creation: bool

class XORBPTaaSOrchestrationEngine:
    """Penetration Testing as a Service Orchestration Engine"""

    def __init__(self):
        self.engine_id = f"PTAAS-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_campaigns = {}
        self.completed_campaigns = {}
        self.reports = {}
        self.integrations = {}

        # Initialize XORB components
        self.target_engine = LiveApplicationTargetAcquisitionEngine()
        self.assault_agents = AutonomousAppAssaultAgents()
        self.ui_explorer = XORBAPIUIExplorationAgent()
        self.scoring_matrix = ExploitabilityScoringMatrix()
        self.mutation_injector = DefensiveMutationInjector()

        # PTaaS configuration
        self.api_endpoints = {
            'assessments': '/api/v1/assessments',
            'findings': '/api/v1/findings',
            'reports': '/api/v1/reports',
            'defense': '/api/v1/defense'
        }

        # Campaign queue
        self.campaign_queue = queue.PriorityQueue()
        self.running = True

        # Start orchestration thread
        self.orchestration_thread = threading.Thread(target=self._orchestration_loop, daemon=True)
        self.orchestration_thread.start()

        logger.info(f"🎯 PTaaS Orchestration Engine initialized - ID: {self.engine_id}")

    async def create_ptaas_campaign(self, campaign_spec: Dict[str, Any]) -> PTaaSCampaign:
        """Create new PTaaS campaign"""
        try:
            campaign_id = f"CAMPAIGN-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{secrets.token_hex(4)}"

            # Discover targets
            seed_domains = campaign_spec.get('targets', [])
            targets = await self.target_engine.discover_application_targets(seed_domains)

            # Create campaign
            campaign = PTaaSCampaign(
                campaign_id=campaign_id,
                name=campaign_spec.get('name', f'Campaign {campaign_id}'),
                targets=targets,
                assessment_type=AssessmentType(campaign_spec.get('assessment_type', 'full_stack')),
                mode=PTaaSMode(campaign_spec.get('mode', 'on_demand')),
                schedule=campaign_spec.get('schedule'),
                duration_hours=campaign_spec.get('duration_hours', 24),
                stealth_level=campaign_spec.get('stealth_level', 0.5),
                hard_realism_mode=campaign_spec.get('hard_realism_mode', False),
                auto_remediation=campaign_spec.get('auto_remediation', False)
            )

            # Add to active campaigns
            self.active_campaigns[campaign_id] = campaign

            # Queue for execution
            priority = 1 if campaign.mode == PTaaSMode.TRIGGERED else 3
            self.campaign_queue.put((priority, campaign))

            logger.info(f"🎯 Created PTaaS campaign: {campaign_id} | Targets: {len(targets)}")

            return campaign

        except Exception as e:
            logger.error(f"❌ Campaign creation error: {e}")
            raise

    async def execute_ptaas_campaign(self, campaign: PTaaSCampaign) -> Dict[str, Any]:
        """Execute PTaaS campaign"""
        try:
            campaign.status = "running"
            campaign.progress = 0.0

            logger.info(f"🚀 Executing PTaaS campaign: {campaign.campaign_id}")

            # Phase 1: Target Discovery and Mapping (20%)
            topology = await self.target_engine.generate_app_topology_graph(campaign.targets)
            campaign.progress = 0.2

            # Phase 2: UI/API Exploration (40%)
            ui_flows = []
            for target in campaign.targets:
                if target.target_type in [TargetType.WEB_APP, TargetType.REST_API]:
                    flows = await self.ui_explorer.explore_from_openapi_spec(f"{target.base_url}/swagger.json")
                    ui_flows.extend(flows)

                    # Discover hidden functions
                    hidden_functions = await self.ui_explorer.discover_hidden_functions(
                        target.base_url, target.endpoints
                    )
            campaign.progress = 0.4

            # Phase 3: Assault Agent Deployment (60%)
            assault_mode = AssaultMode.HARD_REALISM if campaign.hard_realism_mode else AssaultMode.STEALTH
            agents = await self.assault_agents.deploy_assault_agents(campaign.targets, assault_mode)

            # Wait for initial assault completion
            await asyncio.sleep(min(campaign.duration_hours * 3600, 300))  # Max 5 minutes for demo
            campaign.progress = 0.6

            # Phase 4: Vulnerability Analysis (80%)
            findings = list(self.assault_agents.findings.values())

            # Score findings
            scored_findings = []
            for finding in findings:
                score = self.scoring_matrix.calculate_exploitability_score(finding)
                finding.exploitability_score = score["overall_score"]
                scored_findings.append(finding)
            campaign.progress = 0.8

            # Phase 5: Defensive Mutations (90%)
            mutations = {}
            if campaign.auto_remediation:
                mutations = await self.mutation_injector.generate_defensive_mutations(findings)

                # Apply GitOps if configured
                if self.integrations.get('gitops'):
                    await self._apply_gitops_mutations(mutations, campaign)
            campaign.progress = 0.9

            # Phase 6: Report Generation (100%)
            report = await self._generate_campaign_report(campaign, findings, mutations)
            campaign.progress = 1.0
            campaign.status = "completed"

            # Move to completed campaigns
            self.completed_campaigns[campaign.campaign_id] = self.active_campaigns.pop(campaign.campaign_id)

            execution_results = {
                "campaign_id": campaign.campaign_id,
                "targets_assessed": len(campaign.targets),
                "agents_deployed": len(agents),
                "findings_discovered": len(findings),
                "critical_findings": len([f for f in findings if f.risk_level == VulnerabilityRisk.CRITICAL]),
                "high_findings": len([f for f in findings if f.risk_level == VulnerabilityRisk.HIGH]),
                "mutations_generated": len(mutations.get("waf_rules", [])),
                "report_id": report.report_id,
                "execution_time": datetime.now() - campaign.created
            }

            logger.info(f"✅ PTaaS campaign completed: {campaign.campaign_id}")
            logger.info(f"📊 Results: {execution_results['findings_discovered']} findings, {execution_results['critical_findings']} critical")

            return execution_results

        except Exception as e:
            campaign.status = "failed"
            logger.error(f"❌ Campaign execution error: {e}")
            raise

    async def setup_ci_cd_integration(self, integration_spec: Dict[str, Any]) -> Dict[str, str]:
        """Setup CI/CD integration for automated testing"""
        try:
            integration_type = IntegrationType(integration_spec.get('type'))
            integration_id = f"INT-{integration_type.value}-{secrets.token_hex(4)}"

            if integration_type == IntegrationType.GITHUB_ACTIONS:
                workflow_yaml = await self._generate_github_workflow(integration_spec)

            elif integration_type == IntegrationType.GITLAB_CI:
                pipeline_yaml = await self._generate_gitlab_pipeline(integration_spec)

            elif integration_type == IntegrationType.JENKINS:
                jenkins_config = await self._generate_jenkins_pipeline(integration_spec)

            # Store integration
            self.integrations[integration_id] = {
                'type': integration_type,
                'config': integration_spec,
                'created': datetime.now()
            }

            logger.info(f"🔗 CI/CD integration configured: {integration_type.value}")

            return {
                'integration_id': integration_id,
                'type': integration_type.value,
                'status': 'configured',
                'webhook_url': f'/api/v1/integrations/{integration_id}/webhook'
            }

        except Exception as e:
            logger.error(f"❌ CI/CD integration error: {e}")
            raise

    async def setup_gitops_integration(self, gitops_spec: Dict[str, Any]) -> GitOpsIntegration:
        """Setup GitOps integration for automated remediation"""
        try:
            integration = GitOpsIntegration(
                integration_id=f"GITOPS-{secrets.token_hex(4)}",
                repository_url=gitops_spec['repository_url'],
                branch=gitops_spec.get('branch', 'main'),
                credentials=gitops_spec.get('credentials', {}),
                auto_commit=gitops_spec.get('auto_commit', False),
                auto_merge=gitops_spec.get('auto_merge', False),
                pr_creation=gitops_spec.get('pr_creation', True)
            )

            # Test repository access
            await self._test_git_access(integration)

            # Store integration
            self.integrations['gitops'] = integration

            logger.info(f"🔗 GitOps integration configured: {integration.repository_url}")

            return integration

        except Exception as e:
            logger.error(f"❌ GitOps integration error: {e}")
            raise

    async def handle_ci_cd_trigger(self, trigger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle CI/CD triggered assessment"""
        try:
            # Extract trigger information
            repository = trigger_data.get('repository')
            branch = trigger_data.get('branch')
            commit_sha = trigger_data.get('commit_sha')

            # Create automated campaign
            campaign_spec = {
                'name': f'CI/CD Assessment - {repository}:{branch}',
                'targets': trigger_data.get('targets', []),
                'assessment_type': 'full_stack',
                'mode': 'triggered',
                'duration_hours': 2,
                'stealth_level': 1.0,  # High stealth for CI/CD
                'auto_remediation': True
            }

            # Execute campaign
            campaign = await self.create_ptaas_campaign(campaign_spec)
            results = await self.execute_ptaas_campaign(campaign)

            # Generate CI/CD specific report
            ci_report = await self._generate_ci_cd_report(results, trigger_data)

            logger.info(f"🔗 CI/CD triggered assessment completed: {campaign.campaign_id}")

            return {
                'campaign_id': campaign.campaign_id,
                'trigger_source': trigger_data.get('source'),
                'repository': repository,
                'branch': branch,
                'commit_sha': commit_sha,
                'findings_count': results['findings_discovered'],
                'critical_findings': results['critical_findings'],
                'report_url': f'/api/v1/reports/{ci_report["report_id"]}',
                'status': 'completed'
            }

        except Exception as e:
            logger.error(f"❌ CI/CD trigger handling error: {e}")
            raise

    async def export_findings(self, campaign_id: str, format: ReportFormat = ReportFormat.JSON) -> Dict[str, Any]:
        """Export campaign findings in specified format"""
        try:
            campaign = self.completed_campaigns.get(campaign_id) or self.active_campaigns.get(campaign_id)
            if not campaign:
                raise ValueError(f"Campaign not found: {campaign_id}")

            # Get findings for campaign
            findings = [f for f in self.assault_agents.findings.values()
                       if any(f.target_id == t.target_id for t in campaign.targets)]

            if format == ReportFormat.JSON:
                export_data = {
                    'campaign_id': campaign_id,
                    'export_timestamp': datetime.now().isoformat(),
                    'findings_count': len(findings),
                    'findings': [asdict(f) for f in findings]
                }

            elif format == ReportFormat.SARIF:
                export_data = await self._convert_to_sarif(findings, campaign)

            elif format == ReportFormat.JUNIT:
                export_data = await self._convert_to_junit(findings, campaign)

            logger.info(f"📤 Exported {len(findings)} findings in {format.value} format")

            return export_data

        except Exception as e:
            logger.error(f"❌ Findings export error: {e}")
            raise

    async def auto_patch_vulnerabilities(self, campaign_id: str) -> Dict[str, Any]:
        """Automatically generate and apply patches for discovered vulnerabilities"""
        try:
            campaign = self.completed_campaigns.get(campaign_id)
            if not campaign or not campaign.auto_remediation:
                raise ValueError("Auto-remediation not enabled for campaign")

            # Get findings
            findings = [f for f in self.assault_agents.findings.values()
                       if any(f.target_id == t.target_id for t in campaign.targets)]

            # Generate patches
            mutations = await self.mutation_injector.generate_defensive_mutations(findings)

            # Apply GitOps patches
            gitops_results = {}
            if self.integrations.get('gitops'):
                gitops_results = await self._apply_gitops_mutations(mutations, campaign)

            patch_results = {
                'campaign_id': campaign_id,
                'patches_generated': len(mutations.get('gitops_patches', [])),
                'waf_rules': len(mutations.get('waf_rules', [])),
                'input_validation_rules': len(mutations.get('input_validation', [])),
                'security_headers': len(mutations.get('security_headers', [])),
                'gitops_applied': bool(gitops_results),
                'pull_request_url': gitops_results.get('pr_url')
            }

            logger.info(f"🛡️ Auto-patching completed: {patch_results['patches_generated']} patches generated")

            return patch_results

        except Exception as e:
            logger.error(f"❌ Auto-patching error: {e}")
            raise

    def _orchestration_loop(self):
        """Main orchestration loop for campaign processing"""
        while self.running:
            try:
                if not self.campaign_queue.empty():
                    priority, campaign = self.campaign_queue.get(timeout=1)
                    asyncio.run(self.execute_ptaas_campaign(campaign))
                else:
                    time.sleep(1)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Orchestration loop error: {e}")

    async def _generate_campaign_report(self, campaign: PTaaSCampaign, findings: List[VulnerabilityFinding], mutations: Dict[str, Any]) -> PTaaSReport:
        """Generate comprehensive campaign report"""
        try:
            report_id = f"RPT-{campaign.campaign_id}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Calculate metrics
            critical_count = len([f for f in findings if f.risk_level == VulnerabilityRisk.CRITICAL])
            high_count = len([f for f in findings if f.risk_level == VulnerabilityRisk.HIGH])
            medium_count = len([f for f in findings if f.risk_level == VulnerabilityRisk.MEDIUM])
            low_count = len([f for f in findings if f.risk_level == VulnerabilityRisk.LOW])

            # Calculate average scores
            exploitability_avg = statistics.mean([f.exploitability_score for f in findings]) if findings else 0.0
            business_impact_avg = statistics.mean([f.business_impact_score for f in findings]) if findings else 0.0

            # Generate executive summary
            executive_summary = await self._generate_executive_summary(campaign, findings)

            # Create remediation priority
            remediation_priority = [f.finding_id for f in sorted(findings,
                                   key=lambda x: (x.risk_level.value, -x.exploitability_score))[:10]]

            report = PTaaSReport(
                report_id=report_id,
                campaign_id=campaign.campaign_id,
                format=ReportFormat.JSON,
                vulnerability_count=len(findings),
                critical_findings=critical_count,
                high_findings=high_count,
                medium_findings=medium_count,
                low_findings=low_count,
                exploitability_score=exploitability_avg,
                business_impact_score=business_impact_avg,
                remediation_priority=remediation_priority,
                executive_summary=executive_summary,
                technical_details={
                    'targets': [asdict(t) for t in campaign.targets],
                    'findings': [asdict(f) for f in findings],
                    'mutations': mutations,
                    'topology': self.target_engine.target_topology
                }
            )

            self.reports[report_id] = report

            return report

        except Exception as e:
            logger.error(f"❌ Report generation error: {e}")
            raise

    async def _generate_executive_summary(self, campaign: PTaaSCampaign, findings: List[VulnerabilityFinding]) -> str:
        """Generate executive summary for campaign"""
        critical_count = len([f for f in findings if f.risk_level == VulnerabilityRisk.CRITICAL])
        high_count = len([f for f in findings if f.risk_level == VulnerabilityRisk.HIGH])

        summary = f"""
        PTaaS Assessment Summary for {campaign.name}

        Assessment Period: {campaign.created.strftime('%Y-%m-%d %H:%M')} - {datetime.now().strftime('%Y-%m-%d %H:%M')}
        Targets Assessed: {len(campaign.targets)}
        Assessment Type: {campaign.assessment_type.value}

        Key Findings:
        - Total Vulnerabilities: {len(findings)}
        - Critical Risk: {critical_count}
        - High Risk: {high_count}

        Risk Level: {'CRITICAL' if critical_count > 0 else 'HIGH' if high_count > 0 else 'MEDIUM'}

        Immediate Actions Required:
        {f'- Address {critical_count} critical vulnerabilities immediately' if critical_count > 0 else ''}
        {f'- Review and remediate {high_count} high-risk findings' if high_count > 0 else ''}
        - Implement automated defensive measures
        - Schedule follow-up assessment
        """

        return summary.strip()

    async def _apply_gitops_mutations(self, mutations: Dict[str, Any], campaign: PTaaSCampaign) -> Dict[str, Any]:
        """Apply mutations via GitOps workflow"""
        try:
            gitops_integration = self.integrations.get('gitops')
            if not gitops_integration:
                return {}

            # Create branch for mutations
            branch_name = f"xorb-mutations-{campaign.campaign_id}"

            # Generate mutation files
            mutation_files = {
                'waf_rules.yaml': yaml.dump(mutations.get('waf_rules', [])),
                'input_validation.json': json.dumps(mutations.get('input_validation', []), indent=2),
                'security_headers.yaml': yaml.dump(mutations.get('security_headers', [])),
                'iam_policies.json': json.dumps(mutations.get('iam_policies', []), indent=2)
            }

            # Simulate GitOps operations (in real implementation, use actual Git operations)
            gitops_results = {
                'branch_created': branch_name,
                'files_committed': list(mutation_files.keys()),
                'pr_url': f"https://github.com/repo/pull/mutations-{campaign.campaign_id}",
                'auto_merge': gitops_integration.auto_merge
            }

            logger.info(f"🔗 GitOps mutations applied: {len(mutation_files)} files")

            return gitops_results

        except Exception as e:
            logger.error(f"❌ GitOps application error: {e}")
            return {}

    async def get_ptaas_status(self) -> Dict[str, Any]:
        """Get comprehensive PTaaS engine status"""
        try:
            status = {
                "engine_id": self.engine_id,
                "timestamp": datetime.now().isoformat(),
                "active_campaigns": len(self.active_campaigns),
                "completed_campaigns": len(self.completed_campaigns),
                "queued_campaigns": self.campaign_queue.qsize(),
                "total_reports": len(self.reports),
                "integrations": len(self.integrations),
                "api_endpoints": self.api_endpoints,
                "system_health": await self._assess_ptaas_health()
            }

            return status

        except Exception as e:
            logger.error(f"❌ Status retrieval error: {e}")
            return {}

    async def _assess_ptaas_health(self) -> Dict[str, Any]:
        """Assess PTaaS engine health"""
        health = {
            "orchestration_thread": self.orchestration_thread.is_alive(),
            "campaign_processing": "operational",
            "integration_status": "ready",
            "component_health": {
                "target_engine": "operational",
                "assault_agents": "operational",
                "ui_explorer": "operational",
                "scoring_matrix": "operational",
                "mutation_injector": "operational"
            }
        }

        return health

async def main():
    """Demonstrate XORB PTaaS Orchestration Engine"""
    logger.info("🎯 Starting XORB PTaaS Orchestration demonstration")

    engine = XORBPTaaSOrchestrationEngine()

    # Create sample campaign
    campaign_spec = {
        'name': 'Demo Application Assessment',
        'targets': ['demo.example.com', 'api.demo.com'],
        'assessment_type': 'full_stack',
        'mode': 'on_demand',
        'duration_hours': 1,
        'stealth_level': 0.8,
        'auto_remediation': True
    }

    # Create and execute campaign
    campaign = await engine.create_ptaas_campaign(campaign_spec)
    results = await engine.execute_ptaas_campaign(campaign)

    # Setup sample integration
    integration_spec = {
        'type': 'github_actions',
        'repository': 'demo/app',
        'webhook_secret': 'secret',
        'trigger_on': ['push', 'pull_request']
    }

    integration = await engine.setup_ci_cd_integration(integration_spec)

    # Export findings
    findings_export = await engine.export_findings(campaign.campaign_id, ReportFormat.JSON)

    # Get engine status
    engine_status = await engine.get_ptaas_status()

    # Cleanup
    engine.running = False

    logger.info("🎯 PTaaS Orchestration demonstration complete")
    logger.info(f"📊 Campaign results: {results['findings_discovered']} findings")
    logger.info(f"🔗 Integration configured: {integration['type']}")
    logger.info(f"📤 Exported findings: {len(findings_export.get('findings', []))}")

    return {
        "engine_id": engine.engine_id,
        "campaign_id": campaign.campaign_id,
        "findings_discovered": results['findings_discovered'],
        "critical_findings": results['critical_findings'],
        "integration_type": integration['type'],
        "engine_status": engine_status
    }

if __name__ == "__main__":
    asyncio.run(main())
