from typing import Dict, List, Any, Optional

#!/usr/bin/env python3
"""
Business Intelligence and Reporting Test Script
Tests comprehensive reporting capabilities and dashboard generation
"""

import asyncio
import sys
import os
import aiofiles
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from xorb_core.reporting.business_intelligence import (
    BusinessIntelligenceEngine, ReportType, ReportFormat,
    ReportingDashboard, demo_business_intelligence
)
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_report_generation() -> None:
    """Test comprehensive report generation"""
    logger.info("=== Testing Report Generation ===")
    
    bi_engine = BusinessIntelligenceEngine()
    
    # Test different report types
    report_types = [
        ReportType.EXECUTIVE_SUMMARY,
        ReportType.OPERATIONAL_DASHBOARD,
        ReportType.COMPLIANCE_AUDIT
    ]
    
    for report_type in report_types:
        logger.info(f"Generating {report_type.value} report...")
        
        report = await bi_engine.generate_report(report_type)
        
        # Verify report structure
        assert report.report_id is not None, "Report should have ID"
        assert report.title is not None, "Report should have title"
        assert len(report.sections) > 0, "Report should have sections"
        assert report.executive_summary is not None, "Report should have executive summary"
        
        logger.info(f"✅ Generated {report_type.value} with {len(report.sections)} sections")
    
    logger.info("✅ Report generation test passed")

async def test_report_export() -> None:
    """Test report export to different formats"""
    logger.info("=== Testing Report Export ===")
    
    bi_engine = BusinessIntelligenceEngine()
    
    # Generate a test report
    report = await bi_engine.generate_report(ReportType.EXECUTIVE_SUMMARY)
    
    # Test different export formats
    formats = [ReportFormat.JSON, ReportFormat.HTML, ReportFormat.CSV]
    
    for format in formats:
        logger.info(f"Exporting report to {format.value}...")
        
        output_path = await bi_engine.export_report(report, format)
        
        # Verify file was created
        assert Path(output_path).exists(), f"Export file {output_path} should exist"
        assert Path(output_path).stat().st_size > 0, f"Export file should not be empty"
        
        logger.info(f"✅ Exported to {output_path}")
    
    logger.info("✅ Report export test passed")

async def test_data_source_integration() -> None:
    """Test data source registration and integration"""
    logger.info("=== Testing Data Source Integration ===")
    
    bi_engine = BusinessIntelligenceEngine()
    
    # Mock data sources
    mock_sources = {
        "threat_db": {"threats": [{"id": "t1", "type": "malware"}]},
        "campaign_db": {"campaigns": [{"id": "c1", "status": "completed"}]},
        "compliance_db": {"audits": [{"standard": "SOC2", "score": 0.95}]}
    }
    
    # Register data sources
    for name, source in mock_sources.items():
        await bi_engine.register_data_source(name, source)
    
    # Verify registration
    assert len(bi_engine.data_sources) == 3, "Should have 3 registered data sources"
    assert "threat_db" in bi_engine.data_sources, "Threat DB should be registered"
    
    logger.info("✅ Data source integration test passed")

async def test_scheduled_reports() -> None:
    """Test scheduled report functionality"""
    logger.info("=== Testing Scheduled Reports ===")
    
    bi_engine = BusinessIntelligenceEngine()
    
    # Schedule different types of reports
    await bi_engine.schedule_report(
        ReportType.OPERATIONAL_DASHBOARD,
        "0 8 * * *",  # Daily at 8 AM
        ["ops@company.com"]
    )
    
    await bi_engine.schedule_report(
        ReportType.EXECUTIVE_SUMMARY,
        "0 9 * * MON",  # Weekly on Monday at 9 AM
        ["exec@company.com", "ciso@company.com"]
    )
    
    # Verify scheduling
    assert len(bi_engine.scheduled_reports) == 2, "Should have 2 scheduled reports"
    
    daily_report = bi_engine.scheduled_reports[0]
    assert daily_report["report_type"] == ReportType.OPERATIONAL_DASHBOARD
    assert daily_report["schedule"] == "0 8 * * *"
    
    logger.info("✅ Scheduled reports test passed")

async def test_dashboard_functionality() -> None:
    """Test interactive dashboard functionality"""
    logger.info("=== Testing Dashboard Functionality ===")
    
    bi_engine = BusinessIntelligenceEngine()
    dashboard = ReportingDashboard(bi_engine)
    
    # Test dashboard data retrieval
    dashboard_data = await dashboard.get_dashboard_data()
    
    # Verify dashboard data structure
    assert "timestamp" in dashboard_data, "Dashboard should have timestamp"
    assert "metrics" in dashboard_data, "Dashboard should have metrics"
    assert "alerts" in dashboard_data, "Dashboard should have alerts"
    assert "status" in dashboard_data, "Dashboard should have status"
    
    logger.info(f"Dashboard data: {json.dumps(dashboard_data, indent=2, default=str)}")
    logger.info("✅ Dashboard functionality test passed")

async def test_metric_calculations() -> None:
    """Test metric calculation and status determination"""
    logger.info("=== Testing Metric Calculations ===")
    
    bi_engine = BusinessIntelligenceEngine()
    
    # Generate report to test metrics
    report = await bi_engine.generate_report(ReportType.EXECUTIVE_SUMMARY)
    
    # Check metrics in security posture section
    security_section = None
    for section in report.sections:
        if "Security Posture" in section.title:
            security_section = section
            break
    
    assert security_section is not None, "Should have security posture section"
    assert len(security_section.metrics) > 0, "Security section should have metrics"
    
    # Verify metric properties
    for metric in security_section.metrics:
        assert metric.name is not None, "Metric should have name"
        assert metric.value is not None, "Metric should have value"
        assert metric.status in ["normal", "warning", "critical"], "Metric should have valid status"
    
    logger.info(f"Verified {len(security_section.metrics)} metrics in security section")
    logger.info("✅ Metric calculations test passed")

async def test_chart_generation() -> None:
    """Test chart and visualization generation"""
    logger.info("=== Testing Chart Generation ===")
    
    bi_engine = BusinessIntelligenceEngine()
    
    # Generate report with charts
    report = await bi_engine.generate_report(ReportType.EXECUTIVE_SUMMARY)
    
    chart_count = 0
    for section in report.sections:
        chart_count += len(section.charts)
    
    assert chart_count > 0, "Report should contain charts"
    
    # Verify chart structure
    for section in report.sections:
        for chart in section.charts:
            assert "type" in chart, "Chart should have type"
            assert "title" in chart, "Chart should have title"
            assert "data" in chart, "Chart should have data"
    
    logger.info(f"Generated {chart_count} charts across all sections")
    logger.info("✅ Chart generation test passed")

async def test_compliance_reporting() -> None:
    """Test compliance-specific reporting features"""
    logger.info("=== Testing Compliance Reporting ===")
    
    bi_engine = BusinessIntelligenceEngine()
    
    # Generate compliance audit report
    compliance_report = await bi_engine.generate_report(ReportType.COMPLIANCE_AUDIT)
    
    # Verify compliance-specific content
    assert "Compliance" in compliance_report.title, "Should be compliance report"
    
    compliance_sections = [s for s in compliance_report.sections if "compliance" in s.title.lower()]
    assert len(compliance_sections) > 0, "Should have compliance sections"
    
    # Check for compliance metrics
    compliance_metrics = []
    for section in compliance_report.sections:
        compliance_metrics.extend([m for m in section.metrics if "compliance" in m.name.lower()])
    
    assert len(compliance_metrics) > 0, "Should have compliance metrics"
    
    logger.info(f"Generated compliance report with {len(compliance_sections)} sections")
    logger.info("✅ Compliance reporting test passed")

async def test_performance_metrics() -> None:
    """Test performance and operational metrics"""
    logger.info("=== Testing Performance Metrics ===")
    
    bi_engine = BusinessIntelligenceEngine()
    
    # Generate operational dashboard
    ops_report = await bi_engine.generate_report(ReportType.OPERATIONAL_DASHBOARD)
    
    # Look for performance-related metrics
    performance_metrics = []
    for section in ops_report.sections:
        for metric in section.metrics:
            if any(term in metric.name.lower() for term in ["time", "rate", "duration", "utilization"]):
                performance_metrics.append(metric)
    
    assert len(performance_metrics) > 0, "Should have performance metrics"
    
    logger.info(f"Found {len(performance_metrics)} performance metrics")
    logger.info("✅ Performance metrics test passed")

async def test_recommendation_engine() -> None:
    """Test recommendation generation"""
    logger.info("=== Testing Recommendation Engine ===")
    
    bi_engine = BusinessIntelligenceEngine()
    
    # Generate report to test recommendations
    report = await bi_engine.generate_report(ReportType.EXECUTIVE_SUMMARY)
    
    # Verify recommendations exist
    assert len(report.recommendations) > 0, "Report should have recommendations"
    
    # Check section-level recommendations
    section_recs = 0
    for section in report.sections:
        section_recs += len(section.recommendations)
    
    assert section_recs > 0, "Sections should have recommendations"
    
    logger.info(f"Generated {len(report.recommendations)} top-level recommendations")
    logger.info(f"Generated {section_recs} section-level recommendations")
    logger.info("✅ Recommendation engine test passed")

async def test_report_caching() -> None:
    """Test report caching functionality"""
    logger.info("=== Testing Report Caching ===")
    
    bi_engine = BusinessIntelligenceEngine()
    
    # Generate report (should be cached)
    report1 = await bi_engine.generate_report(ReportType.EXECUTIVE_SUMMARY)
    
    # Verify it's in cache
    assert report1.report_id in bi_engine.report_cache, "Report should be cached"
    
    # Retrieve from cache
    cached_report = bi_engine.report_cache[report1.report_id]
    assert cached_report.report_id == report1.report_id, "Cached report should match"
    
    logger.info(f"Report {report1.report_id} successfully cached and retrieved")
    logger.info("✅ Report caching test passed")

async def main() -> None:
    """Run all business intelligence tests"""
    logger.info("Starting Business Intelligence and Reporting Tests")
    logger.info("=" * 70)
    
    try:
        # Run comprehensive test suite
        await test_report_generation()
        await test_report_export()
        await test_data_source_integration()
        await test_scheduled_reports()
        await test_dashboard_functionality()
        await test_metric_calculations()
        await test_chart_generation()
        await test_compliance_reporting()
        await test_performance_metrics()
        await test_recommendation_engine()
        await test_report_caching()
        
        # Run the demo
        logger.info("=== Running Business Intelligence Demo ===")
        await demo_business_intelligence()
        
        logger.info("=" * 70)
        logger.info("🎉 All business intelligence tests passed!")
        
        # Cleanup test files
        import shutil
        if Path("reports").exists():
            shutil.rmtree("reports")
            logger.info("🧹 Cleaned up test report files")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())