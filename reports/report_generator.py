#!/usr/bin/env python3

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

import jinja2
from pydantic import BaseModel, Field, validator


class SeverityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class CVSSVersion(str, Enum):
    V3_1 = "3.1"
    V3_0 = "3.0"
    V2_0 = "2.0"


class ReportFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"
    PDF = "pdf"


@dataclass
class CVSSMetrics:
    # Base Metrics
    attack_vector: str  # Network, Adjacent, Local, Physical
    attack_complexity: str  # Low, High
    privileges_required: str  # None, Low, High
    user_interaction: str  # None, Required
    scope: str  # Unchanged, Changed
    confidentiality_impact: str  # None, Low, High
    integrity_impact: str  # None, Low, High
    availability_impact: str  # None, Low, High
    
    # Temporal Metrics (Optional)
    exploit_code_maturity: Optional[str] = None
    remediation_level: Optional[str] = None
    report_confidence: Optional[str] = None
    
    # Environmental Metrics (Optional)
    modified_attack_vector: Optional[str] = None
    modified_attack_complexity: Optional[str] = None
    modified_privileges_required: Optional[str] = None
    modified_user_interaction: Optional[str] = None
    modified_scope: Optional[str] = None
    modified_confidentiality: Optional[str] = None
    modified_integrity: Optional[str] = None
    modified_availability: Optional[str] = None


class Finding(BaseModel):
    id: str
    title: str
    description: str
    severity: SeverityLevel
    cvss_score: Optional[float] = None
    cvss_vector: Optional[str] = None
    cvss_metrics: Optional[Dict[str, Any]] = None
    
    affected_targets: List[str] = Field(default_factory=list)
    proof_of_concept: Optional[str] = None
    remediation: str = ""
    references: List[str] = Field(default_factory=list)
    
    discovered_at: datetime = Field(default_factory=datetime.utcnow)
    agent_id: Optional[str] = None
    campaign_id: Optional[str] = None
    
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    
    @validator('severity')
    def validate_severity(cls, v):
        if isinstance(v, str):
            return SeverityLevel(v.lower())
        return v

    @validator('cvss_score')
    def validate_cvss_score(cls, v):
        if v is not None and (v < 0.0 or v > 10.0):
            raise ValueError('CVSS score must be between 0.0 and 10.0')
        return v


class ReportMetadata(BaseModel):
    title: str
    campaign_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generated_by: str = "XORB Security Platform"
    version: str = "1.0.0"
    
    targets: List[str] = Field(default_factory=list)
    scope: str = ""
    methodology: str = ""
    
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_hours: Optional[float] = None
    
    executive_summary: str = ""
    recommendations: List[str] = Field(default_factory=list)
    
    technical_details: Dict[str, Any] = Field(default_factory=dict)


class SecurityReport:
    def __init__(self, metadata: ReportMetadata):
        self.metadata = metadata
        self.findings: List[Finding] = []
        self.logger = logging.getLogger(__name__)

    def add_finding(self, finding: Finding):
        """Add a finding to the report"""
        self.findings.append(finding)
        self.logger.debug(f"Added finding: {finding.title}")

    def calculate_risk_summary(self) -> Dict[str, Any]:
        """Calculate risk summary statistics"""
        severity_counts = {severity.value: 0 for severity in SeverityLevel}
        total_cvss = 0.0
        cvss_count = 0
        
        for finding in self.findings:
            severity_counts[finding.severity.value] += 1
            
            if finding.cvss_score:
                total_cvss += finding.cvss_score
                cvss_count += 1
        
        avg_cvss = total_cvss / cvss_count if cvss_count > 0 else 0.0
        
        # Calculate overall risk level
        if severity_counts["critical"] > 0:
            overall_risk = "CRITICAL"
        elif severity_counts["high"] > 0:
            overall_risk = "HIGH"
        elif severity_counts["medium"] > 0:
            overall_risk = "MEDIUM"
        elif severity_counts["low"] > 0:
            overall_risk = "LOW"
        else:
            overall_risk = "INFO"
        
        return {
            "overall_risk": overall_risk,
            "total_findings": len(self.findings),
            "severity_distribution": severity_counts,
            "average_cvss": round(avg_cvss, 1),
            "cvss_findings": cvss_count
        }


class CVSSCalculator:
    def __init__(self, version: CVSSVersion = CVSSVersion.V3_1):
        self.version = version
        self.logger = logging.getLogger(__name__)
        
        # CVSS 3.1 scoring values
        self.base_scores = {
            "attack_vector": {"network": 0.85, "adjacent": 0.62, "local": 0.55, "physical": 0.2},
            "attack_complexity": {"low": 0.77, "high": 0.44},
            "privileges_required": {"none": 0.85, "low": 0.62, "high": 0.27},
            "user_interaction": {"none": 0.85, "required": 0.62},
            "confidentiality_impact": {"high": 0.56, "low": 0.22, "none": 0.0},
            "integrity_impact": {"high": 0.56, "low": 0.22, "none": 0.0},
            "availability_impact": {"high": 0.56, "low": 0.22, "none": 0.0}
        }

    def calculate_base_score(self, metrics: CVSSMetrics) -> float:
        """Calculate CVSS base score from metrics"""
        try:
            # Exploitability Score
            av = self.base_scores["attack_vector"][metrics.attack_vector.lower()]
            ac = self.base_scores["attack_complexity"][metrics.attack_complexity.lower()]
            pr = self.base_scores["privileges_required"][metrics.privileges_required.lower()]
            ui = self.base_scores["user_interaction"][metrics.user_interaction.lower()]
            
            # Adjust PR for scope
            if metrics.scope.lower() == "changed":
                if metrics.privileges_required.lower() == "low":
                    pr = 0.68
                elif metrics.privileges_required.lower() == "high":
                    pr = 0.5
            
            exploitability = 8.22 * av * ac * pr * ui
            
            # Impact Score
            c = self.base_scores["confidentiality_impact"][metrics.confidentiality_impact.lower()]
            i = self.base_scores["integrity_impact"][metrics.integrity_impact.lower()]
            a = self.base_scores["availability_impact"][metrics.availability_impact.lower()]
            
            impact_base = 1 - ((1 - c) * (1 - i) * (1 - a))
            
            if metrics.scope.lower() == "unchanged":
                impact = 6.42 * impact_base
            else:  # Changed
                impact = 7.52 * (impact_base - 0.029) - 3.25 * pow(impact_base - 0.02, 15)
            
            # Base Score
            if impact <= 0:
                base_score = 0.0
            else:
                if metrics.scope.lower() == "unchanged":
                    base_score = min(impact + exploitability, 10.0)
                else:  # Changed
                    base_score = min(1.08 * (impact + exploitability), 10.0)
            
            # Round up to nearest 0.1
            base_score = round(base_score * 10) / 10.0
            
            return base_score
            
        except Exception as e:
            self.logger.error(f"Error calculating CVSS score: {e}")
            return 0.0

    def generate_vector_string(self, metrics: CVSSMetrics) -> str:
        """Generate CVSS vector string"""
        vector_parts = [
            f"CVSS:3.1",
            f"AV:{metrics.attack_vector[0].upper()}",
            f"AC:{metrics.attack_complexity[0].upper()}",
            f"PR:{metrics.privileges_required[0].upper()}",
            f"UI:{metrics.user_interaction[0].upper()}" if metrics.user_interaction.lower() != "none" else "UI:N",
            f"S:{metrics.scope[0].upper()}",
            f"C:{metrics.confidentiality_impact[0].upper()}",
            f"I:{metrics.integrity_impact[0].upper()}",
            f"A:{metrics.availability_impact[0].upper()}"
        ]
        
        return "/".join(vector_parts)

    def severity_from_score(self, score: float) -> SeverityLevel:
        """Convert CVSS score to severity level"""
        if score >= 9.0:
            return SeverityLevel.CRITICAL
        elif score >= 7.0:
            return SeverityLevel.HIGH
        elif score >= 4.0:
            return SeverityLevel.MEDIUM
        elif score > 0.0:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.INFO


class ReportGenerator:
    def __init__(self, output_dir: str = "./reports_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cvss_calculator = CVSSCalculator()
        self.logger = logging.getLogger(__name__)
        
        # Setup Jinja2 for templating
        template_dir = Path(__file__).parent / "templates"
        template_dir.mkdir(exist_ok=True)
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        self._create_default_templates()

    def generate_report(self, report: SecurityReport, formats: List[ReportFormat] = None) -> Dict[str, str]:
        """Generate report in specified formats"""
        if formats is None:
            formats = [ReportFormat.MARKDOWN, ReportFormat.JSON]
        
        generated_files = {}
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        campaign_id = report.metadata.campaign_id
        
        try:
            for format_type in formats:
                filename = f"security_report_{campaign_id}_{timestamp}.{format_type.value}"
                filepath = self.output_dir / filename
                
                if format_type == ReportFormat.MARKDOWN:
                    content = self._generate_markdown_report(report)
                elif format_type == ReportFormat.JSON:
                    content = self._generate_json_report(report)
                elif format_type == ReportFormat.HTML:
                    content = self._generate_html_report(report)
                else:
                    self.logger.warning(f"Unsupported format: {format_type}")
                    continue
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                generated_files[format_type.value] = str(filepath)
                self.logger.info(f"Generated {format_type.value} report: {filepath}")
        
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            raise
        
        return generated_files

    def add_cvss_to_finding(self, finding: Finding, metrics: CVSSMetrics) -> Finding:
        """Calculate and add CVSS information to a finding"""
        try:
            base_score = self.cvss_calculator.calculate_base_score(metrics)
            vector_string = self.cvss_calculator.generate_vector_string(metrics)
            severity = self.cvss_calculator.severity_from_score(base_score)
            
            finding.cvss_score = base_score
            finding.cvss_vector = vector_string
            finding.cvss_metrics = {
                "attack_vector": metrics.attack_vector,
                "attack_complexity": metrics.attack_complexity,
                "privileges_required": metrics.privileges_required,
                "user_interaction": metrics.user_interaction,
                "scope": metrics.scope,
                "confidentiality_impact": metrics.confidentiality_impact,
                "integrity_impact": metrics.integrity_impact,
                "availability_impact": metrics.availability_impact
            }
            finding.severity = severity
            
            return finding
            
        except Exception as e:
            self.logger.error(f"Error adding CVSS to finding {finding.id}: {e}")
            return finding

    def _generate_markdown_report(self, report: SecurityReport) -> str:
        """Generate Markdown report"""
        try:
            template = self.jinja_env.get_template("security_report.md")
            risk_summary = report.calculate_risk_summary()
            
            return template.render(
                metadata=report.metadata,
                findings=report.findings,
                risk_summary=risk_summary,
                generated_at=datetime.utcnow()
            )
        except Exception as e:
            self.logger.error(f"Error generating Markdown report: {e}")
            return self._fallback_markdown_report(report)

    def _generate_json_report(self, report: SecurityReport) -> str:
        """Generate JSON report"""
        try:
            risk_summary = report.calculate_risk_summary()
            
            report_data = {
                "metadata": report.metadata.dict(),
                "risk_summary": risk_summary,
                "findings": [finding.dict() for finding in report.findings],
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return json.dumps(report_data, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Error generating JSON report: {e}")
            return json.dumps({"error": str(e)})

    def _generate_html_report(self, report: SecurityReport) -> str:
        """Generate HTML report"""
        try:
            template = self.jinja_env.get_template("security_report.html")
            risk_summary = report.calculate_risk_summary()
            
            return template.render(
                metadata=report.metadata,
                findings=report.findings,
                risk_summary=risk_summary,
                generated_at=datetime.utcnow()
            )
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {e}")
            return f"<html><body><h1>Error</h1><p>{e}</p></body></html>"

    def _create_default_templates(self):
        """Create default report templates"""
        template_dir = Path(__file__).parent / "templates"
        
        # Markdown template
        md_template = """# Security Assessment Report

## Executive Summary

**Campaign:** {{ metadata.campaign_id }}  
**Generated:** {{ generated_at.strftime('%Y-%m-%d %H:%M:%S UTC') }}  
**Overall Risk Level:** {{ risk_summary.overall_risk }}

{{ metadata.executive_summary }}

## Risk Summary

- **Total Findings:** {{ risk_summary.total_findings }}
- **Critical:** {{ risk_summary.severity_distribution.critical }}
- **High:** {{ risk_summary.severity_distribution.high }}
- **Medium:** {{ risk_summary.severity_distribution.medium }}
- **Low:** {{ risk_summary.severity_distribution.low }}
- **Info:** {{ risk_summary.severity_distribution.info }}

{% if risk_summary.cvss_findings > 0 %}
- **Average CVSS Score:** {{ risk_summary.average_cvss }}
{% endif %}

## Methodology

{{ metadata.methodology }}

## Findings

{% for finding in findings %}
### {{ loop.index }}. {{ finding.title }}

**Severity:** {{ finding.severity.value.upper() }}{% if finding.cvss_score %} (CVSS: {{ finding.cvss_score }}){% endif %}  
**Affected Targets:** {{ finding.affected_targets | join(', ') }}  
**Discovered:** {{ finding.discovered_at.strftime('%Y-%m-%d %H:%M:%S') }}

#### Description
{{ finding.description }}

{% if finding.proof_of_concept %}
#### Proof of Concept
```
{{ finding.proof_of_concept }}
```
{% endif %}

#### Remediation
{{ finding.remediation }}

{% if finding.cvss_vector %}
#### CVSS Vector
{{ finding.cvss_vector }}
{% endif %}

{% if finding.references %}
#### References
{% for ref in finding.references %}
- {{ ref }}
{% endfor %}
{% endif %}

---
{% endfor %}

## Recommendations

{% for recommendation in metadata.recommendations %}
- {{ recommendation }}
{% endfor %}

## Technical Details

{{ metadata.technical_details | tojson(indent=2) }}

---

*Report generated by {{ metadata.generated_by }} v{{ metadata.version }}*
"""

        # HTML template
        html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Security Assessment Report - {{ metadata.campaign_id }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { border-bottom: 2px solid #333; padding-bottom: 20px; }
        .risk-summary { background: #f5f5f5; padding: 20px; margin: 20px 0; }
        .finding { border-left: 4px solid #ddd; padding-left: 20px; margin: 30px 0; }
        .critical { border-left-color: #d32f2f; }
        .high { border-left-color: #f57c00; }
        .medium { border-left-color: #fbc02d; }
        .low { border-left-color: #388e3c; }
        .info { border-left-color: #1976d2; }
        .cvss-score { font-weight: bold; color: #d32f2f; }
        pre { background: #f8f8f8; padding: 10px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Security Assessment Report</h1>
        <p><strong>Campaign:</strong> {{ metadata.campaign_id }}</p>
        <p><strong>Generated:</strong> {{ generated_at.strftime('%Y-%m-%d %H:%M:%S UTC') }}</p>
        <p><strong>Overall Risk:</strong> <span class="risk-{{ risk_summary.overall_risk.lower() }}">{{ risk_summary.overall_risk }}</span></p>
    </div>

    <div class="risk-summary">
        <h2>Risk Summary</h2>
        <ul>
            <li><strong>Total Findings:</strong> {{ risk_summary.total_findings }}</li>
            <li><strong>Critical:</strong> {{ risk_summary.severity_distribution.critical }}</li>
            <li><strong>High:</strong> {{ risk_summary.severity_distribution.high }}</li>
            <li><strong>Medium:</strong> {{ risk_summary.severity_distribution.medium }}</li>
            <li><strong>Low:</strong> {{ risk_summary.severity_distribution.low }}</li>
            <li><strong>Info:</strong> {{ risk_summary.severity_distribution.info }}</li>
            {% if risk_summary.cvss_findings > 0 %}
            <li><strong>Average CVSS:</strong> <span class="cvss-score">{{ risk_summary.average_cvss }}</span></li>
            {% endif %}
        </ul>
    </div>

    <h2>Findings</h2>
    {% for finding in findings %}
    <div class="finding {{ finding.severity.value }}">
        <h3>{{ loop.index }}. {{ finding.title }}</h3>
        <p><strong>Severity:</strong> {{ finding.severity.value.upper() }}{% if finding.cvss_score %} <span class="cvss-score">(CVSS: {{ finding.cvss_score }})</span>{% endif %}</p>
        <p><strong>Affected Targets:</strong> {{ finding.affected_targets | join(', ') }}</p>
        
        <h4>Description</h4>
        <p>{{ finding.description }}</p>

        {% if finding.proof_of_concept %}
        <h4>Proof of Concept</h4>
        <pre>{{ finding.proof_of_concept }}</pre>
        {% endif %}

        <h4>Remediation</h4>
        <p>{{ finding.remediation }}</p>

        {% if finding.cvss_vector %}
        <p><strong>CVSS Vector:</strong> <code>{{ finding.cvss_vector }}</code></p>
        {% endif %}
    </div>
    {% endfor %}

    <footer>
        <p><em>Report generated by {{ metadata.generated_by }} v{{ metadata.version }}</em></p>
    </footer>
</body>
</html>"""

        # Write templates to disk
        with open(template_dir / "security_report.md", 'w') as f:
            f.write(md_template)
        
        with open(template_dir / "security_report.html", 'w') as f:
            f.write(html_template)

    def _fallback_markdown_report(self, report: SecurityReport) -> str:
        """Fallback markdown report generation"""
        content = [
            f"# Security Assessment Report",
            f"",
            f"**Campaign:** {report.metadata.campaign_id}",
            f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"",
            f"## Findings ({len(report.findings)} total)",
            f""
        ]
        
        for i, finding in enumerate(report.findings, 1):
            content.extend([
                f"### {i}. {finding.title}",
                f"**Severity:** {finding.severity.value.upper()}",
                f"**Description:** {finding.description}",
                f"**Remediation:** {finding.remediation}",
                f""
            ])
        
        return "\n".join(content)

    def create_finding_from_dict(self, finding_data: Dict[str, Any]) -> Finding:
        """Create Finding object from dictionary data"""
        return Finding(**finding_data)

    def batch_generate_reports(self, reports: List[SecurityReport], formats: List[ReportFormat] = None) -> Dict[str, List[str]]:
        """Generate multiple reports in batch"""
        results = {"successful": [], "failed": []}
        
        for report in reports:
            try:
                files = self.generate_report(report, formats)
                results["successful"].append({
                    "campaign_id": report.metadata.campaign_id,
                    "files": files
                })
            except Exception as e:
                self.logger.error(f"Failed to generate report for {report.metadata.campaign_id}: {e}")
                results["failed"].append({
                    "campaign_id": report.metadata.campaign_id,
                    "error": str(e)
                })
        
        return results