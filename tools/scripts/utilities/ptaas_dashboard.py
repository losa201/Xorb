#!/usr/bin/env python3
"""
XORB PTaaS Dashboard and Reporting System
Advanced web dashboard for penetration testing management and reporting
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import base64
import io
from dataclasses import dataclass
import uuid
import aiohttp
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors as rl_colors

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="XORB PTaaS Dashboard",
    description="Advanced penetration testing dashboard and reporting system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates
templates = Jinja2Templates(directory="templates")

@dataclass
class TestResult:
    test_id: str
    name: str
    target: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    findings: List[Dict[str, Any]]
    agents_used: List[str]
    methodology: str
    client_id: str

class PTaaSDashboard:
    """Advanced PTaaS dashboard with comprehensive reporting"""

    def __init__(self):
        self.test_results: Dict[str, TestResult] = {}
        self.active_connections = set()
        self.dashboard_metrics = {
            "total_tests": 0,
            "active_tests": 0,
            "completed_tests": 0,
            "failed_tests": 0,
            "total_findings": 0,
            "critical_findings": 0,
            "high_findings": 0,
            "medium_findings": 0,
            "low_findings": 0,
            "agent_utilization": 0.0
        }

        # Initialize with sample data
        self._initialize_sample_data()

    def _initialize_sample_data(self):
        """Initialize dashboard with sample test data"""
        sample_tests = [
            {
                "test_id": "PTAAS-1736075642-A1B2C3D4",
                "name": "E-commerce Platform Security Assessment",
                "target": "https://shop.example.com",
                "status": "completed",
                "start_time": datetime.now() - timedelta(hours=25),
                "end_time": datetime.now() - timedelta(hours=1),
                "methodology": "OWASP",
                "client_id": "CLIENT-001",
                "agents_used": ["AGENT-WEB-APP-002", "AGENT-NETWORK-RECON-001"],
                "findings": [
                    {
                        "finding_id": "VULN-1736075642-1",
                        "title": "SQL Injection in Login Form",
                        "severity": "critical",
                        "description": "Critical SQL injection vulnerability allows database access",
                        "cvss_score": 9.8,
                        "location": "/login.php",
                        "impact": "Complete database compromise possible"
                    },
                    {
                        "finding_id": "VULN-1736075642-2",
                        "title": "Stored XSS in User Comments",
                        "severity": "high",
                        "description": "Persistent XSS vulnerability in comment system",
                        "cvss_score": 8.1,
                        "location": "/product/comments",
                        "impact": "Session hijacking and malicious script execution"
                    },
                    {
                        "finding_id": "VULN-1736075642-3",
                        "title": "Weak SSL/TLS Configuration",
                        "severity": "medium",
                        "description": "Server supports weak cipher suites",
                        "cvss_score": 5.3,
                        "location": "SSL/TLS Configuration",
                        "impact": "Potential man-in-the-middle attacks"
                    }
                ]
            },
            {
                "test_id": "PTAAS-1736075643-E5F6G7H8",
                "name": "Corporate API Security Test",
                "target": "https://api.corporate.com",
                "status": "running",
                "start_time": datetime.now() - timedelta(hours=2),
                "end_time": None,
                "methodology": "NIST",
                "client_id": "CLIENT-002",
                "agents_used": ["AGENT-WEB-APP-002"],
                "findings": [
                    {
                        "finding_id": "VULN-1736075643-1",
                        "title": "API Rate Limiting Not Implemented",
                        "severity": "medium",
                        "description": "No rate limiting on API endpoints",
                        "cvss_score": 4.3,
                        "location": "/api/v1/*",
                        "impact": "Potential DoS attacks"
                    }
                ]
            }
        ]

        for test_data in sample_tests:
            test_result = TestResult(
                test_id=test_data["test_id"],
                name=test_data["name"],
                target=test_data["target"],
                status=test_data["status"],
                start_time=test_data["start_time"],
                end_time=test_data["end_time"],
                findings=test_data["findings"],
                agents_used=test_data["agents_used"],
                methodology=test_data["methodology"],
                client_id=test_data["client_id"]
            )
            self.test_results[test_data["test_id"]] = test_result

        self._update_metrics()

    def _update_metrics(self):
        """Update dashboard metrics based on current test results"""
        self.dashboard_metrics["total_tests"] = len(self.test_results)
        self.dashboard_metrics["active_tests"] = len([t for t in self.test_results.values() if t.status == "running"])
        self.dashboard_metrics["completed_tests"] = len([t for t in self.test_results.values() if t.status == "completed"])
        self.dashboard_metrics["failed_tests"] = len([t for t in self.test_results.values() if t.status == "failed"])

        # Count findings by severity
        all_findings = []
        for test in self.test_results.values():
            all_findings.extend(test.findings)

        self.dashboard_metrics["total_findings"] = len(all_findings)
        self.dashboard_metrics["critical_findings"] = len([f for f in all_findings if f.get("severity") == "critical"])
        self.dashboard_metrics["high_findings"] = len([f for f in all_findings if f.get("severity") == "high"])
        self.dashboard_metrics["medium_findings"] = len([f for f in all_findings if f.get("severity") == "medium"])
        self.dashboard_metrics["low_findings"] = len([f for f in all_findings if f.get("severity") == "low"])

    def get_dashboard_overview(self) -> Dict[str, Any]:
        """Get comprehensive dashboard overview"""
        self._update_metrics()

        recent_tests = sorted(
            self.test_results.values(),
            key=lambda x: x.start_time,
            reverse=True
        )[:5]

        return {
            "metrics": self.dashboard_metrics,
            "recent_tests": [
                {
                    "test_id": test.test_id,
                    "name": test.name,
                    "target": test.target,
                    "status": test.status,
                    "start_time": test.start_time.isoformat(),
                    "findings_count": len(test.findings),
                    "critical_findings": len([f for f in test.findings if f.get("severity") == "critical"])
                } for test in recent_tests
            ],
            "severity_distribution": {
                "critical": self.dashboard_metrics["critical_findings"],
                "high": self.dashboard_metrics["high_findings"],
                "medium": self.dashboard_metrics["medium_findings"],
                "low": self.dashboard_metrics["low_findings"]
            }
        }

    def generate_test_chart(self, chart_type: str = "severity") -> str:
        """Generate charts for dashboard visualization"""
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "severity":
            # Severity distribution pie chart
            severities = ["Critical", "High", "Medium", "Low"]
            counts = [
                self.dashboard_metrics["critical_findings"],
                self.dashboard_metrics["high_findings"],
                self.dashboard_metrics["medium_findings"],
                self.dashboard_metrics["low_findings"]
            ]
            colors = ['#dc3545', '#fd7e14', '#ffc107', '#28a745']

            # Filter out zero values
            non_zero_data = [(sev, count, color) for sev, count, color in zip(severities, counts, colors) if count > 0]
            if non_zero_data:
                severities, counts, colors = zip(*non_zero_data)
                ax.pie(counts, labels=severities, autopct='%1.1f%%', colors=colors, startangle=90)
                ax.set_title('Findings by Severity', fontsize=16, fontweight='bold', color='white')
            else:
                ax.text(0.5, 0.5, 'No findings yet', ha='center', va='center', fontsize=14, color='white')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

        elif chart_type == "test_status":
            # Test status bar chart
            statuses = ["Completed", "Running", "Failed"]
            counts = [
                self.dashboard_metrics["completed_tests"],
                self.dashboard_metrics["active_tests"],
                self.dashboard_metrics["failed_tests"]
            ]
            colors = ['#28a745', '#ffc107', '#dc3545']

            bars = ax.bar(statuses, counts, color=colors)
            ax.set_title('Test Status Distribution', fontsize=16, fontweight='bold', color='white')
            ax.set_ylabel('Number of Tests', color='white')

            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                       f'{count}', ha='center', va='bottom', color='white')

        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', facecolor='#1a1a1a', edgecolor='none')
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return f"data:image/png;base64,{chart_data}"

    async def generate_detailed_report(self, test_id: str, format: str = "pdf") -> bytes:
        """Generate detailed test report in PDF format"""
        if test_id not in self.test_results:
            raise ValueError(f"Test {test_id} not found")

        test = self.test_results[test_id]

        if format == "pdf":
            return self._generate_pdf_report(test)
        elif format == "json":
            return json.dumps({
                "test_id": test.test_id,
                "name": test.name,
                "target": test.target,
                "status": test.status,
                "start_time": test.start_time.isoformat(),
                "end_time": test.end_time.isoformat() if test.end_time else None,
                "findings": test.findings,
                "agents_used": test.agents_used,
                "methodology": test.methodology,
                "client_id": test.client_id
            }, indent=2).encode()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_pdf_report(self, test: TestResult) -> bytes:
        """Generate comprehensive PDF report"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            spaceAfter=30,
            textColor=rl_colors.darkblue
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=rl_colors.darkblue
        )

        # Title
        story.append(Paragraph(f"Penetration Test Report: {test.name}", title_style))
        story.append(Spacer(1, 12))

        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))

        exec_summary = f"""
        This report contains the results of a comprehensive penetration test conducted on {test.target}.
        The assessment was performed using the {test.methodology} methodology and identified {len(test.findings)} security findings.

        <b>Test Details:</b><br/>
        • Target: {test.target}<br/>
        • Test ID: {test.test_id}<br/>
        • Start Time: {test.start_time.strftime('%Y-%m-%d %H:%M:%S')}<br/>
        • End Time: {test.end_time.strftime('%Y-%m-%d %H:%M:%S') if test.end_time else 'In Progress'}<br/>
        • Status: {test.status.title()}<br/>
        • Methodology: {test.methodology}<br/>
        """

        story.append(Paragraph(exec_summary, styles['Normal']))
        story.append(Spacer(1, 20))

        # Findings Summary
        story.append(Paragraph("Findings Summary", heading_style))

        # Count findings by severity
        severity_counts = {}
        for finding in test.findings:
            severity = finding.get("severity", "unknown")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        summary_data = [
            ["Severity", "Count", "Description"],
            ["Critical", severity_counts.get("critical", 0), "Immediate attention required"],
            ["High", severity_counts.get("high", 0), "High priority remediation"],
            ["Medium", severity_counts.get("medium", 0), "Medium priority remediation"],
            ["Low", severity_counts.get("low", 0), "Low priority remediation"],
        ]

        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), rl_colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), rl_colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, rl_colors.black)
        ]))

        story.append(summary_table)
        story.append(Spacer(1, 20))

        # Detailed Findings
        story.append(Paragraph("Detailed Findings", heading_style))

        for i, finding in enumerate(test.findings, 1):
            # Finding header
            finding_title = f"Finding {i}: {finding.get('title', 'Unknown')}"
            story.append(Paragraph(finding_title, styles['Heading3']))

            # Finding details
            severity = finding.get("severity", "unknown").title()
            cvss_score = finding.get("cvss_score", "N/A")
            location = finding.get("location", "N/A")

            finding_details = f"""
            <b>Severity:</b> {severity}<br/>
            <b>CVSS Score:</b> {cvss_score}<br/>
            <b>Location:</b> {location}<br/>
            <b>Description:</b> {finding.get('description', 'No description provided')}<br/>
            <b>Impact:</b> {finding.get('impact', 'Impact not specified')}<br/>
            """

            story.append(Paragraph(finding_details, styles['Normal']))
            story.append(Spacer(1, 12))

        # Recommendations
        story.append(PageBreak())
        story.append(Paragraph("Recommendations", heading_style))

        recommendations = """
        Based on the findings identified during this penetration test, we recommend the following actions:

        1. <b>Critical and High Severity Issues:</b> Address immediately with priority given to issues that could lead to data breach or system compromise.

        2. <b>Input Validation:</b> Implement comprehensive input validation and output encoding to prevent injection attacks.

        3. <b>Authentication and Authorization:</b> Strengthen authentication mechanisms and implement proper access controls.

        4. <b>SSL/TLS Configuration:</b> Update SSL/TLS configuration to use modern, secure cipher suites.

        5. <b>Security Headers:</b> Implement security headers to protect against common web vulnerabilities.

        6. <b>Regular Security Testing:</b> Conduct regular penetration tests and security assessments.
        """

        story.append(Paragraph(recommendations, styles['Normal']))
        story.append(Spacer(1, 20))

        # Footer
        story.append(Paragraph("Report generated by XORB PTaaS Platform", styles['Normal']))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))

        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

# Global dashboard instance
dashboard = PTaaSDashboard()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ptaas_dashboard",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(dashboard.active_connections),
        "total_tests": dashboard.dashboard_metrics["total_tests"]
    }

@app.get("/", response_class=HTMLResponse)
async def dashboard_home():
    """Main dashboard page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>XORB PTaaS Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            body { background-color: #1a1a1a; color: #ffffff; }
            .card { background-color: #2d2d2d; border: 1px solid #404040; }
            .navbar { background-color: #0d1117; }
            .metric-card { transition: transform 0.2s; }
            .metric-card:hover { transform: translateY(-5px); }
            .severity-critical { color: #dc3545; }
            .severity-high { color: #fd7e14; }
            .severity-medium { color: #ffc107; }
            .severity-low { color: #28a745; }
            .chart-container { height: 400px; }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-dark">
            <div class="container-fluid">
                <a class="navbar-brand" href="#">
                    <i class="fas fa-shield-alt"></i> XORB PTaaS Dashboard
                </a>
                <div class="d-flex">
                    <span class="navbar-text">
                        <i class="fas fa-clock"></i> <span id="current-time"></span>
                    </span>
                </div>
            </div>
        </nav>

        <div class="container-fluid mt-4">
            <!-- Metrics Row -->
            <div class="row mb-4" id="metrics-row">
                <!-- Metrics will be loaded here -->
            </div>

            <!-- Charts Row -->
            <div class="row mb-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h5><i class="fas fa-chart-pie"></i> Findings by Severity</h5>
                        </div>
                        <div class="card-body text-center">
                            <img id="severity-chart" src="" alt="Severity Chart" class="img-fluid">
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h5><i class="fas fa-chart-bar"></i> Test Status Distribution</h5>
                        </div>
                        <div class="card-body text-center">
                            <img id="status-chart" src="" alt="Status Chart" class="img-fluid">
                        </div>
                    </div>
                </div>
            </div>

            <!-- Recent Tests -->
            <div class="row mb-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <h5><i class="fas fa-list"></i> Recent Penetration Tests</h5>
                        </div>
                        <div class="card-body">
                            <div class="table-responsive">
                                <table class="table table-dark table-striped" id="tests-table">
                                    <thead>
                                        <tr>
                                            <th>Test ID</th>
                                            <th>Name</th>
                                            <th>Target</th>
                                            <th>Status</th>
                                            <th>Findings</th>
                                            <th>Critical</th>
                                            <th>Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody id="tests-body">
                                        <!-- Test data will be loaded here -->
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // Update current time
            function updateTime() {
                const now = new Date();
                document.getElementById('current-time').textContent = now.toLocaleString();
            }
            setInterval(updateTime, 1000);
            updateTime();

            // Load dashboard data
            async function loadDashboardData() {
                try {
                    const response = await fetch('/api/dashboard/overview');
                    const data = await response.json();

                    // Update metrics
                    updateMetrics(data.metrics);

                    // Update recent tests
                    updateRecentTests(data.recent_tests);

                    // Load charts
                    loadCharts();

                } catch (error) {
                    console.error('Error loading dashboard data:', error);
                }
            }

            function updateMetrics(metrics) {
                const metricsHtml = `
                    <div class="col-md-3">
                        <div class="card metric-card">
                            <div class="card-body text-center">
                                <i class="fas fa-vial fa-2x text-primary mb-2"></i>
                                <h3>${metrics.total_tests}</h3>
                                <p class="text-muted">Total Tests</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card metric-card">
                            <div class="card-body text-center">
                                <i class="fas fa-play-circle fa-2x text-warning mb-2"></i>
                                <h3>${metrics.active_tests}</h3>
                                <p class="text-muted">Active Tests</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card metric-card">
                            <div class="card-body text-center">
                                <i class="fas fa-bug fa-2x severity-critical mb-2"></i>
                                <h3>${metrics.total_findings}</h3>
                                <p class="text-muted">Total Findings</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card metric-card">
                            <div class="card-body text-center">
                                <i class="fas fa-exclamation-triangle fa-2x severity-critical mb-2"></i>
                                <h3>${metrics.critical_findings}</h3>
                                <p class="text-muted">Critical Findings</p>
                            </div>
                        </div>
                    </div>
                `;
                document.getElementById('metrics-row').innerHTML = metricsHtml;
            }

            function updateRecentTests(tests) {
                const tbody = document.getElementById('tests-body');
                tbody.innerHTML = tests.map(test => `
                    <tr>
                        <td><code>${test.test_id}</code></td>
                        <td>${test.name}</td>
                        <td>${test.target}</td>
                        <td>
                            <span class="badge bg-${getStatusColor(test.status)}">${test.status}</span>
                        </td>
                        <td>${test.findings_count}</td>
                        <td>
                            <span class="severity-critical">${test.critical_findings}</span>
                        </td>
                        <td>
                            <button class="btn btn-sm btn-outline-primary" onclick="viewReport('${test.test_id}')">
                                <i class="fas fa-eye"></i> View
                            </button>
                            <button class="btn btn-sm btn-outline-success" onclick="downloadReport('${test.test_id}')">
                                <i class="fas fa-download"></i> PDF
                            </button>
                        </td>
                    </tr>
                `).join('');
            }

            function getStatusColor(status) {
                switch(status) {
                    case 'completed': return 'success';
                    case 'running': return 'warning';
                    case 'failed': return 'danger';
                    default: return 'secondary';
                }
            }

            async function loadCharts() {
                try {
                    // Load severity chart
                    const severityResponse = await fetch('/api/charts/severity');
                    if (severityResponse.ok) {
                        const severityChart = await severityResponse.text();
                        document.getElementById('severity-chart').src = severityChart;
                    }

                    // Load status chart
                    const statusResponse = await fetch('/api/charts/test_status');
                    if (statusResponse.ok) {
                        const statusChart = await statusResponse.text();
                        document.getElementById('status-chart').src = statusChart;
                    }
                } catch (error) {
                    console.error('Error loading charts:', error);
                }
            }

            function viewReport(testId) {
                window.open(`/reports/${testId}`, '_blank');
            }

            function downloadReport(testId) {
                window.open(`/reports/${testId}/download?format=pdf`, '_blank');
            }

            // Load data on page load and refresh every 30 seconds
            loadDashboardData();
            setInterval(loadDashboardData, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/dashboard/overview")
async def get_dashboard_overview():
    """Get dashboard overview data"""
    return dashboard.get_dashboard_overview()

@app.get("/api/charts/{chart_type}")
async def get_chart(chart_type: str):
    """Get dashboard charts"""
    try:
        chart_data = dashboard.generate_test_chart(chart_type)
        return Response(content=chart_data, media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/tests/{test_id}")
async def get_test_details(test_id: str):
    """Get detailed information about a specific test"""
    if test_id not in dashboard.test_results:
        raise HTTPException(status_code=404, detail="Test not found")

    test = dashboard.test_results[test_id]
    return {
        "test_id": test.test_id,
        "name": test.name,
        "target": test.target,
        "status": test.status,
        "start_time": test.start_time.isoformat(),
        "end_time": test.end_time.isoformat() if test.end_time else None,
        "findings": test.findings,
        "agents_used": test.agents_used,
        "methodology": test.methodology,
        "client_id": test.client_id
    }

@app.get("/reports/{test_id}")
async def view_test_report(test_id: str):
    """View test report in browser"""
    if test_id not in dashboard.test_results:
        raise HTTPException(status_code=404, detail="Test not found")

    test = dashboard.test_results[test_id]

    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Test Report: {test.name}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ background-color: #1a1a1a; color: #ffffff; }}
            .card {{ background-color: #2d2d2d; border: 1px solid #404040; }}
            .severity-critical {{ color: #dc3545; font-weight: bold; }}
            .severity-high {{ color: #fd7e14; font-weight: bold; }}
            .severity-medium {{ color: #ffc107; font-weight: bold; }}
            .severity-low {{ color: #28a745; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container mt-4">
            <div class="row">
                <div class="col-12">
                    <h1>Penetration Test Report</h1>
                    <h2 class="text-muted">{test.name}</h2>
                    <hr>
                </div>
            </div>

            <div class="row mb-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h5>Test Information</h5>
                        </div>
                        <div class="card-body">
                            <p><strong>Test ID:</strong> {test.test_id}</p>
                            <p><strong>Target:</strong> {test.target}</p>
                            <p><strong>Status:</strong> <span class="badge bg-{"success" if test.status == "completed" else "warning"}">{test.status}</span></p>
                            <p><strong>Start Time:</strong> {test.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                            <p><strong>End Time:</strong> {test.end_time.strftime('%Y-%m-%d %H:%M:%S') if test.end_time else 'In Progress'}</p>
                            <p><strong>Methodology:</strong> {test.methodology}</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h5>Findings Summary</h5>
                        </div>
                        <div class="card-body">
                            <p><strong>Total Findings:</strong> {len(test.findings)}</p>
                            <p><strong>Critical:</strong> <span class="severity-critical">{len([f for f in test.findings if f.get('severity') == 'critical'])}</span></p>
                            <p><strong>High:</strong> <span class="severity-high">{len([f for f in test.findings if f.get('severity') == 'high'])}</span></p>
                            <p><strong>Medium:</strong> <span class="severity-medium">{len([f for f in test.findings if f.get('severity') == 'medium'])}</span></p>
                            <p><strong>Low:</strong> <span class="severity-low">{len([f for f in test.findings if f.get('severity') == 'low'])}</span></p>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <h5>Detailed Findings</h5>
                        </div>
                        <div class="card-body">
    """

    for i, finding in enumerate(test.findings, 1):
        severity_class = f"severity-{finding.get('severity', 'low')}"
        html_content += f"""
                            <div class="mb-4">
                                <h6>Finding {i}: {finding.get('title', 'Unknown')}</h6>
                                <p><strong>Severity:</strong> <span class="{severity_class}">{finding.get('severity', 'unknown').title()}</span></p>
                                <p><strong>CVSS Score:</strong> {finding.get('cvss_score', 'N/A')}</p>
                                <p><strong>Location:</strong> <code>{finding.get('location', 'N/A')}</code></p>
                                <p><strong>Description:</strong> {finding.get('description', 'No description provided')}</p>
                                <p><strong>Impact:</strong> {finding.get('impact', 'Impact not specified')}</p>
                                <hr>
                            </div>
        """

    html_content += """
                        </div>
                    </div>
                </div>
            </div>

            <div class="row mt-4">
                <div class="col-12">
                    <a href="/reports/{}/download?format=pdf" class="btn btn-success">
                        <i class="fas fa-download"></i> Download PDF Report
                    </a>
                    <a href="/" class="btn btn-secondary">
                        <i class="fas fa-arrow-left"></i> Back to Dashboard
                    </a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """.format(test_id)

    return HTMLResponse(content=html_content)

@app.get("/reports/{test_id}/download")
async def download_test_report(test_id: str, format: str = "pdf"):
    """Download test report in specified format"""
    try:
        report_data = await dashboard.generate_detailed_report(test_id, format)

        if format == "pdf":
            return Response(
                content=report_data,
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=report_{test_id}.pdf"}
            )
        elif format == "json":
            return Response(
                content=report_data,
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=report_{test_id}.json"}
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "ptaas_dashboard:app",
        host="0.0.0.0",
        port=8085,
        reload=False,
        log_level="info"
    )
