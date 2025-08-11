"""
Compliance Report Exporter
Handles conversion of compliance reports to various formats
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, Union
from uuid import UUID

from jinja2 import Environment, FileSystemLoader, Template
from weasyprint import HTML, CSS
from xorb.shared.config import settings

logger = logging.getLogger(__name__)

class ReportExporter:
    """
    Handles export of compliance reports to various formats
    """
    
    def __init__(self):
        self.templates_dir = settings.REPORT_TEMPLATES_DIR
        self.output_dir = settings.REPORT_OUTPUT_DIR
        self.default_template = settings.DEFAULT_REPORT_TEMPLATE
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load templates
        self.env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            autoescape=True
        )
        
    def _load_template(self, template_name: Optional[str] = None) -> Template:
        """
        Load a template by name or use the default
        """
        template_name = template_name or self.default_template
        try:
            return self.env.get_template(template_name)
        except Exception as e:
            logger.error(f"Failed to load template {template_name}: {e}")
            raise
    
    def _render_template(self, template: Template, report_data: Dict[str, Any]) -> str:
        """
        Render a template with report data
        """
        try:
            return template.render(
                report=report_data,
                current_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        except Exception as e:
            logger.error(f"Template rendering error: {e}")
            raise
    
    def export_to_html(self, report_data: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Export compliance report to HTML format
        """
        try:
            # Load template
            template = self._load_template("compliance_report_html.j2")
            
            # Render template
            html_content = self._render_template(template, report_data)
            
            # Save to file if output path provided
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(html_content)
                return output_path
            
            # Otherwise return raw content
            return html_content
            
        except Exception as e:
            logger.error(f"HTML export error: {e}")
            raise
    
    def export_to_pdf(self, report_data: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Export compliance report to PDF format
        """
        try:
            # First convert to HTML
            html_content = self.export_to_html(report_data)
            
            # Convert HTML to PDF
            pdf_content = HTML(string=html_content).write_pdf(
                stylesheets=[CSS(string='''
                    @page {
                        size: A4;
                        margin: 2cm;
                    }
                    body {
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                    }
                    h1, h2, h3 {
                        color: #2c3e50;
                    }
                    .requirement {
                        margin-bottom: 1em;
                        padding: 0.5em;
                        background-color: #f8f9fa;
                        border-left: 4px solid #3498db;
                    }
                    .finding {
                        margin-bottom: 1em;
                        padding: 0.5em;
                        background-color: #fefefe;
                        border-left: 4px solid #e74c3c;
                    }
                ''')]
            )
            
            # Save to file if output path provided
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(pdf_content)
                return output_path
            
            # Otherwise return base64 encoded content
            return pdf_content
            
        except Exception as e:
            logger.error(f"PDF export error: {e}")
            raise
    
    def export_to_json(self, report_data: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Export compliance report to JSON format
        """
        try:
            # Convert to JSON
            json_content = json.dumps(report_data, indent=2)
            
            # Save to file if output path provided
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(json_content)
                return output_path
            
            # Otherwise return raw content
            return json_content
            
        except Exception as e:
            logger.error(f"JSON export error: {e}")
            raise
    
    def export_to_markdown(self, report_data: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Export compliance report to Markdown format
        """
        try:
            # Load template
            template = self._load_template("compliance_report_md.j2")
            
            # Render template
            md_content = self._render_template(template, report_data)
            
            # Save to file if output path provided
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(md_content)
                return output_path
            
            # Otherwise return raw content
            return md_content
            
        except Exception as e:
            logger.error(f"Markdown export error: {e}")
            raise
    
    def export_to(self, report_data: Dict[str, Any], format: str, output_path: Optional[str] = None) -> str:
        """
        Export compliance report to the specified format
        """
        exporters = {
            "html": self.export_to_html,
            "pdf": self.export_to_pdf,
            "json": self.export_to_json,
            "markdown": self.export_to_markdown
        }
        
        exporter = exporters.get(format.lower())
        if not exporter:
            raise ValueError(f"Unsupported export format: {format}")
        
        return exporter(report_data, output_path)

# Example usage
if __name__ == '__main__':
    import asyncio
    from xorb.core.compliance_orchestrator import ComplianceOrchestrator
    
    async def main():
        exporter = ReportExporter()
        orchestrator = ComplianceOrchestrator()
        
        # Get a sample report (this would typically be a completed validation)
        sample_report = await orchestrator.run_compliance_validation(
            standard="nist",
            targets=["scanme.nmap.org"],
            scan_profile="comprehensive"
        )
        
        # Export to different formats
        html_path = exporter.export_to_html(sample_report, "sample_report.html")
        pdf_path = exporter.export_to_pdf(sample_report, "sample_report.pdf")
        json_path = exporter.export_to_json(sample_report, "sample_report.json")
        md_path = exporter.export_to_markdown(sample_report, "sample_report.md")
        
        print(f"Reports exported to:")
        print(f"- HTML: {html_path}")
        print(f"- PDF: {pdf_path}")
        print(f"- JSON: {json_path}")
        print(f"- Markdown: {md_path}")
    
    # Run the example
    asyncio.run(main())
    