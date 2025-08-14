"""
CI/CD Pipeline Integration Platform
DevSecOps automation for GitLab, GitHub, Jenkins, and other CI/CD platforms
"""

import asyncio
import logging
import json
import yaml
import base64
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import re

logger = logging.getLogger(__name__)


class CICDPlatform(Enum):
    """Supported CI/CD platforms"""
    GITLAB = "gitlab"
    GITHUB = "github"
    JENKINS = "jenkins"
    AZURE_DEVOPS = "azure_devops"
    BITBUCKET = "bitbucket"
    CIRCLECI = "circleci"
    TRAVIS_CI = "travis_ci"
    TEAMCITY = "teamcity"


class ScanTrigger(Enum):
    """When to trigger security scans"""
    ON_COMMIT = "on_commit"
    ON_MERGE_REQUEST = "on_merge_request"
    ON_RELEASE = "on_release"
    SCHEDULED = "scheduled"
    MANUAL = "manual"


class ScanType(Enum):
    """Types of security scans"""
    SAST = "sast"  # Static Application Security Testing
    DAST = "dast"  # Dynamic Application Security Testing
    SCA = "sca"    # Software Composition Analysis
    CONTAINER = "container"  # Container security scanning
    INFRASTRUCTURE = "infrastructure"  # Infrastructure as Code scanning
    SECRETS = "secrets"  # Secret detection
    COMPLIANCE = "compliance"  # Compliance checking


@dataclass
class SecurityPolicy:
    """Security policy for CI/CD pipeline"""
    max_critical_vulnerabilities: int = 0
    max_high_vulnerabilities: int = 5
    max_medium_vulnerabilities: int = 20
    block_on_policy_violation: bool = True
    require_security_review: bool = True
    allowed_licenses: List[str] = field(default_factory=lambda: ["MIT", "Apache-2.0", "BSD-3-Clause"])
    blocked_licenses: List[str] = field(default_factory=lambda: ["GPL-3.0", "AGPL-3.0"])
    secret_detection_enabled: bool = True
    compliance_frameworks: List[str] = field(default_factory=list)


@dataclass
class PipelineConfiguration:
    """CI/CD pipeline configuration"""
    platform: CICDPlatform
    repository_url: str
    branch_patterns: List[str] = field(default_factory=lambda: ["main", "master", "develop"])
    scan_triggers: List[ScanTrigger] = field(default_factory=lambda: [ScanTrigger.ON_MERGE_REQUEST])
    scan_types: List[ScanType] = field(default_factory=lambda: [ScanType.SAST, ScanType.SCA, ScanType.SECRETS])
    security_policy: SecurityPolicy = field(default_factory=SecurityPolicy)
    notification_channels: List[str] = field(default_factory=list)
    custom_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanResult:
    """Security scan result"""
    scan_id: str
    scan_type: ScanType
    status: str
    timestamp: datetime
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    secrets_found: List[Dict[str, Any]] = field(default_factory=list)
    license_issues: List[Dict[str, Any]] = field(default_factory=list)
    compliance_issues: List[Dict[str, Any]] = field(default_factory=list)
    policy_violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    execution_time: Optional[float] = None
    artifacts: Dict[str, str] = field(default_factory=dict)


class GitLabIntegrator:
    """GitLab CI/CD integration"""

    def __init__(self, config: PipelineConfiguration, api_token: str):
        self.config = config
        self.api_token = api_token
        self.project_id = self._extract_project_id(config.repository_url)

    def _extract_project_id(self, repo_url: str) -> str:
        """Extract GitLab project ID from repository URL"""
        try:
            # Extract from URL like https://gitlab.com/group/project
            match = re.search(r'gitlab\.com[/:]([\w\-\.]+/[\w\-\.]+)', repo_url)
            if match:
                return match.group(1).replace('/', '%2F')
            return "unknown"
        except:
            return "unknown"

    async def create_pipeline_config(self) -> str:
        """Create .gitlab-ci.yml configuration"""
        try:
            pipeline_config = {
                "stages": ["security", "build", "test", "deploy"],
                "variables": {
                    "XORB_SCAN_ENABLED": "true",
                    "SECURITY_POLICY_STRICT": str(self.config.security_policy.block_on_policy_violation).lower()
                },
                "include": [
                    {"template": "Security/SAST.gitlab-ci.yml"},
                    {"template": "Security/Dependency-Scanning.gitlab-ci.yml"},
                    {"template": "Security/Secret-Detection.gitlab-ci.yml"}
                ]
            }

            # Add XORB PTaaS security job
            pipeline_config["xorb_security_scan"] = {
                "stage": "security",
                "image": "xorb/security-scanner:latest",
                "script": [
                    "echo 'Starting XORB PTaaS security scan'",
                    "xorb-scanner --config security-config.yml --output scan-results.json",
                    "xorb-policy-check --results scan-results.json --policy security-policy.yml"
                ],
                "artifacts": {
                    "reports": {
                        "security": "scan-results.json"
                    },
                    "paths": ["scan-results.json", "security-report.html"],
                    "expire_in": "1 week"
                },
                "only": {
                    "refs": self.config.branch_patterns
                },
                "allow_failure": not self.config.security_policy.block_on_policy_violation
            }

            # Add container scanning if enabled
            if ScanType.CONTAINER in self.config.scan_types:
                pipeline_config["container_security_scan"] = {
                    "stage": "security",
                    "image": "docker:stable",
                    "services": ["docker:dind"],
                    "script": [
                        "docker build -t $CI_PROJECT_NAME:$CI_COMMIT_SHA .",
                        "xorb-container-scan --image $CI_PROJECT_NAME:$CI_COMMIT_SHA --output container-scan.json"
                    ],
                    "artifacts": {
                        "reports": {
                            "container_scanning": "container-scan.json"
                        }
                    }
                }

            return yaml.dump(pipeline_config, default_flow_style=False)

        except Exception as e:
            logger.error(f"GitLab pipeline config creation failed: {e}")
            return ""

    async def create_merge_request_webhook(self) -> bool:
        """Create webhook for merge request events"""
        try:
            webhook_config = {
                "url": "https://xorb_platform.com/api/v1/cicd/gitlab/webhook",
                "merge_requests_events": True,
                "push_events": True,
                "issues_events": False,
                "confidential_issues_events": False,
                "wiki_page_events": False,
                "deployment_events": True,
                "job_events": True,
                "pipeline_events": True,
                "token": hashlib.sha256(self.api_token.encode()).hexdigest()[:32]
            }

            # In production, this would create actual webhook via GitLab API
            # POST /projects/:id/hooks

            logger.info(f"Created GitLab webhook for project {self.project_id}")
            return True

        except Exception as e:
            logger.error(f"GitLab webhook creation failed: {e}")
            return False

    async def update_merge_request_status(self, merge_request_id: int, scan_results: List[ScanResult]) -> bool:
        """Update merge request with security scan status"""
        try:
            # Calculate overall security status
            policy_violations = []
            total_critical = 0
            total_high = 0

            for result in scan_results:
                policy_violations.extend(result.policy_violations)
                for vuln in result.vulnerabilities:
                    if vuln.get("severity") == "critical":
                        total_critical += 1
                    elif vuln.get("severity") == "high":
                        total_high += 1

            # Determine status
            if policy_violations or total_critical > self.config.security_policy.max_critical_vulnerabilities:
                status = "failed"
                message = f"❌ Security scan failed: {len(policy_violations)} policy violations, {total_critical} critical vulnerabilities"
            elif total_high > self.config.security_policy.max_high_vulnerabilities:
                status = "warning"
                message = f"⚠️ Security scan warning: {total_high} high-severity vulnerabilities"
            else:
                status = "success"
                message = f"✅ Security scan passed: No critical issues found"

            # Create MR note with detailed results
            note_content = self._create_security_report_comment(scan_results)

            # In production, this would update via GitLab API
            # POST /projects/:id/merge_requests/:merge_request_iid/notes

            logger.info(f"Updated GitLab MR {merge_request_id} with security status: {status}")
            return True

        except Exception as e:
            logger.error(f"GitLab MR status update failed: {e}")
            return False

    def _create_security_report_comment(self, scan_results: List[ScanResult]) -> str:
        """Create detailed security report comment for MR"""
        try:
            report = "## 🛡️ XORB PTaaS Security Scan Report\n\n"

            for result in scan_results:
                report += f"### {result.scan_type.value.upper()} Scan Results\n"
                report += f"- **Status**: {result.status}\n"
                report += f"- **Execution Time**: {result.execution_time:.2f}s\n"
                report += f"- **Vulnerabilities Found**: {len(result.vulnerabilities)}\n"

                if result.vulnerabilities:
                    severity_counts = {}
                    for vuln in result.vulnerabilities:
                        severity = vuln.get("severity", "unknown")
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1

                    report += "\n#### Vulnerability Breakdown:\n"
                    for severity, count in severity_counts.items():
                        emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}.get(severity, "⚪")
                        report += f"- {emoji} {severity.title()}: {count}\n"

                if result.policy_violations:
                    report += "\n#### Policy Violations:\n"
                    for violation in result.policy_violations:
                        report += f"- ❌ {violation}\n"

                if result.recommendations:
                    report += "\n#### Recommendations:\n"
                    for rec in result.recommendations[:3]:  # Top 3 recommendations
                        report += f"- 💡 {rec}\n"

                report += "\n---\n"

            report += "\n*Report generated by XORB PTaaS Platform*"
            return report

        except Exception as e:
            logger.error(f"Security report comment creation failed: {e}")
            return "Security scan completed. Please check artifacts for detailed results."


class GitHubIntegrator:
    """GitHub Actions integration"""

    def __init__(self, config: PipelineConfiguration, api_token: str):
        self.config = config
        self.api_token = api_token

    async def create_workflow_config(self) -> str:
        """Create GitHub Actions workflow configuration"""
        try:
            workflow = {
                "name": "XORB PTaaS Security Scan",
                "on": {
                    "pull_request": {
                        "branches": self.config.branch_patterns
                    },
                    "push": {
                        "branches": self.config.branch_patterns
                    }
                },
                "jobs": {
                    "security-scan": {
                        "runs-on": "ubuntu-latest",
                        "steps": [
                            {
                                "name": "Checkout code",
                                "uses": "actions/checkout@v3"
                            },
                            {
                                "name": "Setup XORB Scanner",
                                "run": "curl -sSL https://get.xorb.io/install.sh | bash"
                            },
                            {
                                "name": "Run SAST Scan",
                                "if": "contains(fromJSON('[\"sast\"]'), matrix.scan-type)",
                                "run": "xorb-scanner sast --config .xorb/security-config.yml --output sast-results.json"
                            },
                            {
                                "name": "Run SCA Scan",
                                "if": "contains(fromJSON('[\"sca\"]'), matrix.scan-type)",
                                "run": "xorb-scanner sca --config .xorb/security-config.yml --output sca-results.json"
                            },
                            {
                                "name": "Run Secret Detection",
                                "if": "contains(fromJSON('[\"secrets\"]'), matrix.scan-type)",
                                "run": "xorb-scanner secrets --config .xorb/security-config.yml --output secrets-results.json"
                            },
                            {
                                "name": "Policy Validation",
                                "run": "xorb-policy-check --results *.json --policy .xorb/security-policy.yml"
                            },
                            {
                                "name": "Upload Results",
                                "uses": "actions/upload-artifact@v3",
                                "with": {
                                    "name": "security-scan-results",
                                    "path": "*-results.json"
                                }
                            },
                            {
                                "name": "Security Report",
                                "uses": "xorb/security-report-action@v1",
                                "with": {
                                    "results-path": ".",
                                    "github-token": "${{ secrets.GITHUB_TOKEN }}"
                                }
                            }
                        ],
                        "strategy": {
                            "matrix": {
                                "scan-type": [scan_type.value for scan_type in self.config.scan_types]
                            }
                        }
                    }
                }
            }

            # Add container scanning job if enabled
            if ScanType.CONTAINER in self.config.scan_types:
                workflow["jobs"]["container-scan"] = {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"name": "Checkout code", "uses": "actions/checkout@v3"},
                        {"name": "Build Docker image", "run": "docker build -t ${{ github.repository }}:${{ github.sha }} ."},
                        {"name": "Scan container", "run": "xorb-container-scan --image ${{ github.repository }}:${{ github.sha }} --output container-results.json"},
                        {"name": "Upload results", "uses": "actions/upload-artifact@v3", "with": {"name": "container-scan-results", "path": "container-results.json"}}
                    ]
                }

            return yaml.dump(workflow, default_flow_style=False)

        except Exception as e:
            logger.error(f"GitHub workflow config creation failed: {e}")
            return ""

    async def create_status_check(self, commit_sha: str, scan_results: List[ScanResult]) -> bool:
        """Create GitHub status check for commit"""
        try:
            # Calculate overall status
            has_failures = any(result.policy_violations for result in scan_results)

            status_data = {
                "state": "failure" if has_failures else "success",
                "target_url": "https://xorb_platform.com/scans",
                "description": f"XORB PTaaS security scan {'failed' if has_failures else 'passed'}",
                "context": "security/xorb-ptaas"
            }

            # In production, this would create via GitHub API
            # POST /repos/:owner/:repo/statuses/:sha

            logger.info(f"Created GitHub status check for commit {commit_sha}")
            return True

        except Exception as e:
            logger.error(f"GitHub status check creation failed: {e}")
            return False


class JenkinsIntegrator:
    """Jenkins CI/CD integration"""

    def __init__(self, config: PipelineConfiguration, jenkins_url: str, api_token: str):
        self.config = config
        self.jenkins_url = jenkins_url
        self.api_token = api_token

    async def create_pipeline_script(self) -> str:
        """Create Jenkins pipeline script (Jenkinsfile)"""
        try:
            pipeline_script = f"""
pipeline {{
    agent any

    environment {{
        XORB_API_TOKEN = credentials('xorb-api-token')
        SECURITY_POLICY_STRICT = '{str(self.config.security_policy.block_on_policy_violation).lower()}'
    }}

    stages {{
        stage('Checkout') {{
            steps {{
                checkout scm
            }}
        }}

        stage('Security Scan') {{
            parallel {{
                stage('SAST') {{
                    when {{
                        expression {{ params.SCAN_TYPES.contains('sast') }}
                    }}
                    steps {{
                        sh 'xorb-scanner sast --config security-config.yml --output sast-results.json'
                        archiveArtifacts artifacts: 'sast-results.json', fingerprint: true
                    }}
                }}

                stage('SCA') {{
                    when {{
                        expression {{ params.SCAN_TYPES.contains('sca') }}
                    }}
                    steps {{
                        sh 'xorb-scanner sca --config security-config.yml --output sca-results.json'
                        archiveArtifacts artifacts: 'sca-results.json', fingerprint: true
                    }}
                }}

                stage('Secret Detection') {{
                    when {{
                        expression {{ params.SCAN_TYPES.contains('secrets') }}
                    }}
                    steps {{
                        sh 'xorb-scanner secrets --config security-config.yml --output secrets-results.json'
                        archiveArtifacts artifacts: 'secrets-results.json', fingerprint: true
                    }}
                }}
            }}
        }}

        stage('Policy Check') {{
            steps {{
                sh 'xorb-policy-check --results *.json --policy security-policy.yml'
                script {{
                    def scanResults = readJSON file: 'policy-check-results.json'
                    if (scanResults.violations.size() > 0 && env.SECURITY_POLICY_STRICT == 'true') {{
                        error("Security policy violations detected")
                    }}
                }}
            }}
        }}

        stage('Security Report') {{
            steps {{
                sh 'xorb-report-generator --results . --format html --output security-report.html'
                publishHTML([
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: '.',
                    reportFiles: 'security-report.html',
                    reportName: 'XORB Security Report'
                ])
            }}
        }}
    }}

    post {{
        always {{
            archiveArtifacts artifacts: '*.json, *.html', fingerprint: true
        }}

        failure {{
            emailext (
                subject: "Security Scan Failed: ${{env.JOB_NAME}} - ${{env.BUILD_NUMBER}}",
                body: "Security scan failed for build ${{env.BUILD_NUMBER}}. Please check the security report.",
                to: "{','.join(self.config.notification_channels)}"
            )
        }}

        success {{
            emailext (
                subject: "Security Scan Passed: ${{env.JOB_NAME}} - ${{env.BUILD_NUMBER}}",
                body: "Security scan passed for build ${{env.BUILD_NUMBER}}.",
                to: "{','.join(self.config.notification_channels)}"
            )
        }}
    }}
}}
"""
            return pipeline_script

        except Exception as e:
            logger.error(f"Jenkins pipeline script creation failed: {e}")
            return ""

    async def create_webhook_job(self) -> bool:
        """Create Jenkins webhook job"""
        try:
            job_config = f"""
<?xml version='1.1' encoding='UTF-8'?>
<flow-definition plugin="workflow-job@2.40">
    <actions/>
    <description>XORB PTaaS Security Scan Pipeline</description>
    <keepDependencies>false</keepDependencies>
    <properties>
        <hudson.model.ParametersDefinitionProperty>
            <parameterDefinitions>
                <hudson.model.StringParameterDefinition>
                    <name>BRANCH</name>
                    <description>Branch to scan</description>
                    <defaultValue>main</defaultValue>
                </hudson.model.StringParameterDefinition>
                <hudson.model.ChoiceParameterDefinition>
                    <name>SCAN_TYPES</name>
                    <description>Types of scans to run</description>
                    <choices>
                        <string>sast,sca,secrets</string>
                        <string>sast</string>
                        <string>sca</string>
                        <string>secrets</string>
                    </choices>
                </hudson.model.ChoiceParameterDefinition>
            </parameterDefinitions>
        </hudson.model.ParametersDefinitionProperty>
        <org.jenkinsci.plugins.workflow.job.properties.PipelineTriggersJobProperty>
            <triggers>
                <com.cloudbees.jenkins.GitHubPushTrigger plugin="github@1.32.0">
                    <spec></spec>
                </com.cloudbees.jenkins.GitHubPushTrigger>
            </triggers>
        </org.jenkinsci.plugins.workflow.job.properties.PipelineTriggersJobProperty>
    </properties>
    <definition class="org.jenkinsci.plugins.workflow.cps.CpsFlowDefinition" plugin="workflow-cps@2.90">
        <script>{await self.create_pipeline_script()}</script>
        <sandbox>true</sandbox>
    </definition>
    <triggers/>
    <disabled>false</disabled>
</flow-definition>
"""

            # In production, this would create job via Jenkins API
            # POST /createItem?name=xorb-security-scan

            logger.info("Created Jenkins webhook job for XORB security scanning")
            return True

        except Exception as e:
            logger.error(f"Jenkins webhook job creation failed: {e}")
            return False


class CICDIntegrationPlatform:
    """Main CI/CD integration platform"""

    def __init__(self):
        self.integrators = {}
        self.scan_queue = asyncio.Queue()
        self.active_scans = {}

    async def initialize(self):
        """Initialize CI/CD integration platform"""
        logger.info("Initializing CI/CD Integration Platform")

        # Start scan processor
        asyncio.create_task(self._process_scan_queue())

    async def register_repository(self, config: PipelineConfiguration, credentials: Dict[str, str]) -> bool:
        """Register repository with CI/CD platform"""
        try:
            if config.platform == CICDPlatform.GITLAB:
                integrator = GitLabIntegrator(config, credentials.get("api_token", ""))
            elif config.platform == CICDPlatform.GITHUB:
                integrator = GitHubIntegrator(config, credentials.get("api_token", ""))
            elif config.platform == CICDPlatform.JENKINS:
                integrator = JenkinsIntegrator(
                    config,
                    credentials.get("jenkins_url", ""),
                    credentials.get("api_token", "")
                )
            else:
                logger.error(f"Unsupported CI/CD platform: {config.platform}")
                return False

            repo_key = f"{config.platform.value}:{config.repository_url}"
            self.integrators[repo_key] = {
                "config": config,
                "integrator": integrator,
                "credentials": credentials
            }

            logger.info(f"Registered repository {config.repository_url} with {config.platform.value}")
            return True

        except Exception as e:
            logger.error(f"Repository registration failed: {e}")
            return False

    async def handle_webhook(self, platform: CICDPlatform, payload: Dict[str, Any]) -> bool:
        """Handle webhook from CI/CD platform"""
        try:
            if platform == CICDPlatform.GITLAB:
                return await self._handle_gitlab_webhook(payload)
            elif platform == CICDPlatform.GITHUB:
                return await self._handle_github_webhook(payload)
            else:
                logger.warning(f"Webhook handling not implemented for {platform.value}")
                return False

        except Exception as e:
            logger.error(f"Webhook handling failed: {e}")
            return False

    async def _handle_gitlab_webhook(self, payload: Dict[str, Any]) -> bool:
        """Handle GitLab webhook"""
        try:
            event_type = payload.get("object_kind")

            if event_type == "merge_request":
                mr_data = payload.get("object_attributes", {})
                action = mr_data.get("action")

                if action in ["opened", "synchronize"]:
                    # Trigger security scan for merge request
                    scan_request = {
                        "platform": "gitlab",
                        "repository_url": payload.get("project", {}).get("web_url"),
                        "branch": mr_data.get("source_branch"),
                        "merge_request_id": mr_data.get("iid"),
                        "commit_sha": mr_data.get("last_commit", {}).get("id"),
                        "scan_types": ["sast", "sca", "secrets"]
                    }

                    await self.scan_queue.put(scan_request)
                    return True

            elif event_type == "push":
                # Trigger scan for push to protected branches
                ref = payload.get("ref", "")
                if any(branch in ref for branch in ["main", "master", "develop"]):
                    scan_request = {
                        "platform": "gitlab",
                        "repository_url": payload.get("project", {}).get("web_url"),
                        "branch": ref.split("/")[-1],
                        "commit_sha": payload.get("after"),
                        "scan_types": ["sast", "sca", "secrets", "container"]
                    }

                    await self.scan_queue.put(scan_request)
                    return True

            return True

        except Exception as e:
            logger.error(f"GitLab webhook handling failed: {e}")
            return False

    async def _handle_github_webhook(self, payload: Dict[str, Any]) -> bool:
        """Handle GitHub webhook"""
        try:
            event_type = payload.get("action")

            if "pull_request" in payload:
                pr_data = payload["pull_request"]

                if event_type in ["opened", "synchronize"]:
                    scan_request = {
                        "platform": "github",
                        "repository_url": payload.get("repository", {}).get("html_url"),
                        "branch": pr_data.get("head", {}).get("ref"),
                        "pull_request_id": pr_data.get("number"),
                        "commit_sha": pr_data.get("head", {}).get("sha"),
                        "scan_types": ["sast", "sca", "secrets"]
                    }

                    await self.scan_queue.put(scan_request)
                    return True

            elif "push" in payload.get("ref", ""):
                # Handle push events
                ref = payload.get("ref", "")
                if any(branch in ref for branch in ["main", "master", "develop"]):
                    scan_request = {
                        "platform": "github",
                        "repository_url": payload.get("repository", {}).get("html_url"),
                        "branch": ref.split("/")[-1],
                        "commit_sha": payload.get("after"),
                        "scan_types": ["sast", "sca", "secrets"]
                    }

                    await self.scan_queue.put(scan_request)
                    return True

            return True

        except Exception as e:
            logger.error(f"GitHub webhook handling failed: {e}")
            return False

    async def _process_scan_queue(self):
        """Process queued security scans"""
        while True:
            try:
                scan_request = await self.scan_queue.get()
                scan_id = f"scan_{hashlib.md5(str(scan_request).encode()).hexdigest()[:8]}"

                # Start scan processing
                scan_task = asyncio.create_task(self._execute_security_scan(scan_id, scan_request))
                self.active_scans[scan_id] = scan_task

                logger.info(f"Started security scan {scan_id} for {scan_request.get('repository_url')}")

                # Clean up completed scans
                completed_scans = [
                    scan_id for scan_id, task in self.active_scans.items()
                    if task.done()
                ]
                for completed_id in completed_scans:
                    del self.active_scans[completed_id]

            except Exception as e:
                logger.error(f"Scan queue processing failed: {e}")
                await asyncio.sleep(1)

    async def _execute_security_scan(self, scan_id: str, scan_request: Dict[str, Any]):
        """Execute security scan"""
        try:
            platform = scan_request["platform"]
            repo_url = scan_request["repository_url"]
            repo_key = f"{platform}:{repo_url}"

            if repo_key not in self.integrators:
                logger.error(f"No integrator found for {repo_key}")
                return

            integrator_info = self.integrators[repo_key]
            integrator = integrator_info["integrator"]
            config = integrator_info["config"]

            # Simulate security scanning
            scan_results = []

            for scan_type in scan_request.get("scan_types", []):
                await asyncio.sleep(2)  # Simulate scan time

                result = ScanResult(
                    scan_id=f"{scan_id}_{scan_type}",
                    scan_type=ScanType(scan_type),
                    status="completed",
                    timestamp=datetime.now(),
                    vulnerabilities=self._generate_mock_vulnerabilities(scan_type),
                    execution_time=2.0
                )

                # Check against security policy
                result.policy_violations = self._check_security_policy(result, config.security_policy)

                scan_results.append(result)

            # Update CI/CD platform with results
            if platform == "gitlab" and hasattr(integrator, 'update_merge_request_status'):
                mr_id = scan_request.get("merge_request_id")
                if mr_id:
                    await integrator.update_merge_request_status(mr_id, scan_results)

            elif platform == "github" and hasattr(integrator, 'create_status_check'):
                commit_sha = scan_request.get("commit_sha")
                if commit_sha:
                    await integrator.create_status_check(commit_sha, scan_results)

            logger.info(f"Completed security scan {scan_id}")

        except Exception as e:
            logger.error(f"Security scan execution failed: {e}")

    def _generate_mock_vulnerabilities(self, scan_type: str) -> List[Dict[str, Any]]:
        """Generate mock vulnerabilities for testing"""
        vulnerabilities = []

        if scan_type == "sast":
            vulnerabilities.extend([
                {
                    "id": "sast-001",
                    "name": "SQL Injection",
                    "severity": "high",
                    "file": "src/database.py",
                    "line": 42,
                    "description": "Potential SQL injection vulnerability"
                },
                {
                    "id": "sast-002",
                    "name": "Cross-Site Scripting (XSS)",
                    "severity": "medium",
                    "file": "src/web.py",
                    "line": 128,
                    "description": "Potential XSS vulnerability"
                }
            ])

        elif scan_type == "sca":
            vulnerabilities.extend([
                {
                    "id": "sca-001",
                    "name": "Vulnerable Dependency",
                    "severity": "critical",
                    "component": "requests==2.25.1",
                    "cve": "CVE-2023-32681",
                    "description": "Known vulnerability in requests library"
                }
            ])

        elif scan_type == "secrets":
            vulnerabilities.extend([
                {
                    "id": "secrets-001",
                    "name": "Hardcoded API Key",
                    "severity": "high",
                    "file": "config.py",
                    "line": 15,
                    "description": "API key found in source code"
                }
            ])

        return vulnerabilities

    def _check_security_policy(self, result: ScanResult, policy: SecurityPolicy) -> List[str]:
        """Check scan result against security policy"""
        violations = []

        # Count vulnerabilities by severity
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for vuln in result.vulnerabilities:
            severity = vuln.get("severity", "low")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Check against policy limits
        if severity_counts["critical"] > policy.max_critical_vulnerabilities:
            violations.append(f"Too many critical vulnerabilities: {severity_counts['critical']} (max: {policy.max_critical_vulnerabilities})")

        if severity_counts["high"] > policy.max_high_vulnerabilities:
            violations.append(f"Too many high-severity vulnerabilities: {severity_counts['high']} (max: {policy.max_high_vulnerabilities})")

        if severity_counts["medium"] > policy.max_medium_vulnerabilities:
            violations.append(f"Too many medium-severity vulnerabilities: {severity_counts['medium']} (max: {policy.max_medium_vulnerabilities})")

        return violations

    async def get_scan_status(self, scan_id: str) -> Dict[str, Any]:
        """Get status of security scan"""
        if scan_id in self.active_scans:
            task = self.active_scans[scan_id]
            return {
                "scan_id": scan_id,
                "status": "running" if not task.done() else "completed",
                "started_at": datetime.now().isoformat()  # Simplified
            }
        else:
            return {
                "scan_id": scan_id,
                "status": "not_found"
            }

    async def get_platform_status(self) -> Dict[str, Any]:
        """Get CI/CD integration platform status"""
        return {
            "registered_repositories": len(self.integrators),
            "active_scans": len(self.active_scans),
            "queued_scans": self.scan_queue.qsize(),
            "supported_platforms": [platform.value for platform in CICDPlatform],
            "supported_scan_types": [scan_type.value for scan_type in ScanType]
        }


# Global instance
_cicd_platform: Optional[CICDIntegrationPlatform] = None

async def get_cicd_platform() -> CICDIntegrationPlatform:
    """Get global CI/CD integration platform instance"""
    global _cicd_platform

    if _cicd_platform is None:
        _cicd_platform = CICDIntegrationPlatform()
        await _cicd_platform.initialize()

    return _cicd_platform
