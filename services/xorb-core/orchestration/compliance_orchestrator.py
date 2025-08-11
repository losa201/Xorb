"""
Compliance Orchestration System
Integrates compliance validation with security scanning evidence
"""
import asyncio
import logging
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from uuid import uuid4

from xorb.security.compliance import ComplianceFramework, ComplianceStandard
from xorb.security.evidence_collector import EvidenceCollector, EvidenceSource
from xorb.execution_engine.scanner import MultiEngineScanner
from xorb.shared.execution_models import ScanResult, ComplianceReport
from xorb.shared.redis_client import get_redis_client
from xorb.shared.config import settings

logger = logging.getLogger(__name__)

class ComplianceOrchestrator:
    """
    Orchestrates compliance validation workflows with security scanning evidence
    """
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.scanner = MultiEngineScanner(self.redis_client)
        self.evidence_collector = EvidenceCollector()
        self.compliance_engine = ComplianceFramework(ComplianceStandard.NIST)
        self.work_dir = settings.WORK_DIR
        self.max_concurrent_scans = settings.MAX_CONCURRENT_COMPLIANCE_SCANS
        self.scan_timeout = settings.COMPLIANCE_SCAN_TIMEOUT  # 2 hours
        self.evidence_timeout = settings.EVIDENCE_COLLECTION_TIMEOUT  # 1 hour
        
        # Create work directory if it doesn't exist
        os.makedirs(self.work_dir, exist_ok=True)
    
    async def run_compliance_validation(self, 
                                       standard: ComplianceStandard,
                                       targets: List[str],
                                       scan_profile: str = "comprehensive",
                                       policy: Optional[Dict[str, Any]] = None,
                                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run complete compliance validation workflow
        """
        try:
            # Initialize validation
            validation_id = f"compliance-{uuid4()}"
            start_time = time.time()
            
            # Set up validation context
            context = {
                'validation_id': validation_id,
                'standard': standard.value,
                'targets': targets,
                'scan_profile': scan_profile,
                'start_time': datetime.now().isoformat(),
                'status': 'in_progress',
                'findings': [],
                'evidence': [],
                'reports': [],
                'policy': policy or {},
                'metadata': metadata or {}
            }
            
            # Store initial context in Redis
            await self.redis_client.setex(
                f"compliance:{validation_id}",
                int(self.scan_timeout * 2),
                json.dumps(context)
            )
            
            # Update compliance engine with target standard
            self.compliance_engine = ComplianceFramework(standard)
            
            # Step 1: Validate policy against standard
            if policy:
                policy_validation = self.compliance_engine.validate_policy(policy)
                context['policy_validation'] = policy_validation
                
                # Save policy validation report
                policy_report_path = os.path.join(self.work_dir, f"{validation_id}_policy_report.json")
                async with aiofiles.open(policy_report_path, 'w') as f:
                    await f.write(json.dumps(policy_validation, indent=2))
                
                context['policy_report'] = policy_report_path
                
                # Update Redis with policy validation
                await self.redis_client.hset(
                    f"compliance:{validation_id}:reports",
                    "policy_validation",
                    json.dumps(policy_validation)
                )
            
            # Step 2: Collect evidence from security scans
            evidence = await self._collect_security_evidence(
                targets=targets,
                scan_profile=scan_profile,
                validation_id=validation_id,
                context=context
            )
            context['evidence'] = evidence
            
            # Step 3: Generate compliance report
            compliance_report = await self._generate_compliance_report(
                validation_id=validation_id,
                standard=standard,
                policy=policy,
                evidence=evidence,
                context=context
            )
            
            # Step 4: Finalize validation
            context['status'] = 'completed'
            context['end_time'] = datetime.now().isoformat()
            context['duration'] = time.time() - start_time
            context['compliance_report'] = compliance_report
            
            # Update Redis with final status
            await self.redis_client.setex(
                f"compliance:{validation_id}",
                int(self.scan_timeout * 2),
                json.dumps(context)
            )
            
            # Publish completion event
            await self.redis_client.publish(
                "compliance_validation_complete",
                json.dumps({
                    'validation_id': validation_id,
                    'standard': standard.value,
                    'compliant': compliance_report['summary']['compliant'],
                    'coverage_percentage': compliance_report['summary']['coverage_percentage']
                })
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Compliance validation error: {e}", exc_info=True)
            await self._handle_error(validation_id, str(e))
            raise
    
    async def _collect_security_evidence(self,
                                        targets: List[str],
                                        scan_profile: str,
                                        validation_id: str,
                                        context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Collect security evidence from multiple sources
        """
        evidence = []
        
        # Create tasks for evidence collection
        tasks = []
        for target in targets:
            # Add Nmap scan task
            tasks.append(
                self._run_nmap_evidence(
                    target=target,
                    scan_profile=scan_profile,
                    validation_id=validation_id,
                    context=context
                )
            )
            
            # Add Nuclei scan task
            tasks.append(
                self._run_nuclei_evidence(
                    target=target,
                    scan_profile=scan_profile,
                    validation_id=validation_id,
                    context=context
                )
            )
            
            # Add ZAP scan task for web targets
            if "://" in target or "://" in target:  # Simple URL detection
                tasks.append(
                    self._run_zap_evidence(
                        target=target,
                        scan_profile=scan_profile,
                        validation_id=validation_id,
                        context=context
                    )
                )
        
        # Run all evidence collection tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Collect all evidence
        for result in results:
            if result:
                evidence.extend(result)
        
        return evidence
    
    async def _run_nmap_evidence(self,
                               target: str,
                               scan_profile: str,
                               validation_id: str,
                               context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run Nmap scan and collect evidence
        """
        try:
            # Determine scan type based on profile
            if scan_profile == "discovery":
                scan_type = "discovery"
            elif scan_profile == "port_scan":
                scan_type = "port_scan"
            else:
                scan_type = "comprehensive"
            
            # Run Nmap scan
            nmap_result = await self.scanner.run_nmap_scan(target, scan_type)
            
            # Create evidence record
            evidence = {
                'source': EvidenceSource.NMAP.value,
                'target': target,
                'timestamp': datetime.now().isoformat(),
                'data': nmap_result,
                'validation_id': validation_id
            }
            
            # Save evidence to file
            evidence_path = os.path.join(
                self.work_dir, 
                f"{validation_id}_nmap_{target.replace(':', '_').replace('/', '_')}.json"
            )
            async with aiofiles.open(evidence_path, 'w') as f:
                await f.write(json.dumps(evidence, indent=2))
            
            # Add to context
            context['evidence'].append(evidence_path)
            
            # Update Redis
            await self.redis_client.hset(
                f"compliance:{validation_id}:evidence",
                f"nmap:{target}",
                json.dumps(evidence)
            )
            
            return [evidence]
            
        except Exception as e:
            logger.error(f"Nmap evidence collection error: {e}")
            return []
    
    async def _run_nuclei_evidence(self,
                                target: str,
                                scan_profile: str,
                                validation_id: str,
                                context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run Nuclei scan and collect evidence
        """
        try:
            # Determine templates based on profile
            templates = []
            if scan_profile == "comprehensive":
                templates = ["cves/", "vulnerabilities/", "misconfiguration/", "exposures/", "panels/"]
            elif scan_profile == "web":
                templates = ["cves/", "vulnerabilities/", "panels/"]
            else:
                templates = ["cves/"]
            
            # Run Nuclei scan
            nuclei_result = await self.scanner.run_nuclei_scan(target, templates)
            
            # Create evidence record
            evidence = {
                'source': EvidenceSource.NUCLEI.value,
                'target': target,
                'timestamp': datetime.now().isoformat(),
                'data': nuclei_result,
                'validation_id': validation_id
            }
            
            # Save evidence to file
            evidence_path = os.path.join(
                self.work_dir, 
                f"{validation_id}_nuclei_{target.replace(':', '_').replace('/', '_')}.json"
            )
            async with aiofiles.open(evidence_path, 'w') as f:
                await f.write(json.dumps(evidence, indent=2))
            
            # Add to context
            context['evidence'].append(evidence_path)
            
            # Update Redis
            await self.redis_client.hset(
                f"compliance:{validation_id}:evidence",
                f"nuclei:{target}",
                json.dumps(evidence)
            )
            
            return [evidence]
            
        except Exception as e:
            logger.error(f"Nuclei evidence collection error: {e}")
            return []
    
    async def _run_zap_evidence(self,
                              target: str,
                              scan_profile: str,
                              validation_id: str,
                              context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run ZAP scan and collect evidence
        """
        try:
            # Determine scan type based on profile
            zap_scan_type = "baseline"
            if scan_profile == "comprehensive":
                zap_scan_type = "full"
            
            # Run ZAP scan
            zap_result = await self.scanner.run_zap_scan(target, zap_scan_type)
            
            # Create evidence record
            evidence = {
                'source': EvidenceSource.ZAP.value,
                'target': target,
                'timestamp': datetime.now().isoformat(),
                'data': zap_result,
                'validation_id': validation_id
            }
            
            # Save evidence to file
            evidence_path = os.path.join(
                self.work_dir, 
                f"{validation_id}_zap_{target.replace(':', '_').replace('/', '_')}.json"
            )
            async with aiofiles.open(evidence_path, 'w') as f:
                await f.write(json.dumps(evidence, indent=2))
            
            # Add to context
            context['evidence'].append(evidence_path)
            
            # Update Redis
            await self.redis_client.hset(
                f"compliance:{validation_id}:evidence",
                f"zap:{target}",
                json.dumps(evidence)
            )
            
            return [evidence]
            
        except Exception as e:
            logger.error(f"ZAP evidence collection error: {e}")
            return []
    
    async def _generate_compliance_report(self,
                                        validation_id: str,
                                        standard: ComplianceStandard,
                                        policy: Optional[Dict[str, Any]],
                                        evidence: List[Dict[str, Any]],
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report with evidence
        """
        try:
            # Initialize compliance engine with target standard
            compliance_engine = ComplianceFramework(standard)
            
            # If policy is provided, use it for validation
            if policy:
                validation_result = compliance_engine.validate_policy(policy)
            else:
                # Create a synthetic policy based on evidence
                synthetic_policy = self._create_synthetic_policy(evidence)
                validation_result = compliance_engine.validate_policy(synthetic_policy)
            
            # Create detailed report
            report = {
                'header': {
                    'standard': validation_result['metadata'].get('name', standard.value),
                    'version': validation_result['metadata'].get('version', 'N/A'),
                    'scope': validation_result['metadata'].get('scope', 'N/A'),
                    'validation_id': validation_id,
                    'date': datetime.now().isoformat()
                },
                'summary': {
                    'total_requirements': validation_result['total_requirements'],
                    'implemented': validation_result['implemented'],
                    'coverage_percentage': validation_result['coverage_percentage'],
                    'compliant': validation_result['compliant']
                },
                'details': {
                    'missing_requirements': validation_result['missing_requirements'],
                    'recommendations': validation_result['recommendations'],
                    'evidence_summary': self._summarize_evidence(evidence)
                },
                'raw_validation': validation_result,
                'evidence': [e['source'] for e in evidence]
            }
            
            # Save report to file
            report_path = os.path.join(self.work_dir, f"{validation_id}_report.json")
            async with aiofiles.open(report_path, 'w') as f:
                await f.write(json.dumps(report, indent=2))
            
            # Update Redis with report
            await self.redis_client.hset(
                f"compliance:{validation_id}:reports",
                "compliance_report",
                json.dumps(report)
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Compliance report generation error: {e}")
            raise
    
    def _create_synthetic_policy(self, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a synthetic policy based on collected evidence
        """
        policy = {
            'controls': {}
        }
        
        # Analyze evidence to create policy
        for evidence_record in evidence:
            source = evidence_record['source']
            data = evidence_record['data']
            
            # Process Nmap evidence
            if source == EvidenceSource.NMAP.value:
                self._process_nmap_evidence(data, policy)
            
            # Process Nuclei evidence
            elif source == EvidenceSource.NUCLEI.value:
                self._process_nuclei_evidence(data, policy)
            
            # Process ZAP evidence
            elif source == EvidenceSource.ZAP.value:
                self._process_zap_evidence(data, policy)
        
        return policy
    
    def _process_nmap_evidence(self, data: Dict[str, Any], policy: Dict[str, Any]):
        """
        Process Nmap evidence to create policy elements
        """
        # Extract information from Nmap scan
        if 'findings' in data:
            findings = data['findings']
            for finding in findings:
                # Process findings to identify implemented controls
                pass
        
        # Process raw output
        if 'raw_output' in data:
            raw_output = data['raw_output']
            # Analyze raw output to identify implemented controls
            
    
    def _process_nuclei_evidence(self, data: Dict[str, Any], policy: Dict[str, Any]):
        """
        Process Nuclei evidence to create policy elements
        """
        # Extract information from Nuclei scan
        if 'findings' in data:
            findings = data['findings']
            for finding in findings:
                # Process findings to identify implemented controls
                pass
        
    
    def _process_zap_evidence(self, data: Dict[str, Any], policy: Dict[str, Any]):
        """
        Process ZAP evidence to create policy elements
        """
        # Extract information from ZAP scan
        if 'zap_results' in data:
            zap_results = data['zap_results']
            # Process results to identify implemented controls
            
    
    def _summarize_evidence(self, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create summary of collected evidence
        """
        summary = {
            'total_evidence': len(evidence),
            'by_source': {},
            'by_target': {},
            'findings_summary': {}
        }
        
        for record in evidence:
            # Summarize by source
            source = record['source']
            summary['by_source'][source] = summary['by_source'].get(source, 0) + 1
            
            # Summarize by target
            target = record['target']
            summary['by_target'][target] = summary['by_target'].get(target, 0) + 1
            
            # Summarize findings
            data = record.get('data', {})
            
            # Count findings by source
            if source == EvidenceSource.NMAP.value:
                self._summarize_nmap_findings(data, summary)
            elif source == EvidenceSource.NUCLEI.value:
                self._summarize_nuclei_findings(data, summary)
            elif source == EvidenceSource.ZAP.value:
                self._summarize_zap_findings(data, summary)
        
        return summary
    
    def _summarize_nmap_findings(self, data: Dict[str, Any], summary: Dict[str, Any]):
        """
        Summarize Nmap findings
        """
        # Extract information from Nmap scan
        if 'findings' in data:
            findings = data['findings']
            # Process findings to create summary
            
    
    def _summarize_nuclei_findings(self, data: Dict[str, Any], summary: Dict[str, Any]):
        """
        Summarize Nuclei findings
        """
        # Extract information from Nuclei scan
        if 'findings' in data:
            findings = data['findings']
            # Process findings to create summary
            
    
    def _summarize_zap_findings(self, data: Dict[str, Any], summary: Dict[str, Any]):
        """
        Summarize ZAP findings
        """
        # Extract information from ZAP scan
        if 'zap_results' in data:
            zap_results = data['zap_results']
            # Process results to create summary
            
    
    async def _handle_error(self, validation_id: str, error: str):
        """
        Handle errors during compliance validation
        """
        try:
            # Update Redis with error status
            await self.redis_client.hset(
                f"compliance:{validation_id}",
                "status",
                "error"
            )
            await self.redis_client.hset(
                f"compliance:{validation_id}",
                "error",
                error
            )
            
            # Publish error event
            await self.redis_client.publish(
                "compliance_validation_error",
                json.dumps({
                    'validation_id': validation_id,
                    'error': error
                })
            )
            
        except Exception as e:
            logger.error(f"Error handling failed: {e}")
    
    async def get_validation_status(self, validation_id: str) -> Dict[str, Any]:
        """
        Get status of a compliance validation
        """
        try:
            # Get validation context from Redis
            context = await self.redis_client.get(f"compliance:{validation_id}")
            if context:
                return json.loads(context)
            return {"error": "Validation not found"}
            
        except Exception as e:
            logger.error(f"Get validation status error: {e}")
            return {"error": str(e)}
    
    async def cancel_validation(self, validation_id: str) -> bool:
        """
        Cancel a running compliance validation
        """
        try:
            # Set validation status to cancelled
            await self.redis_client.hset(
                f"compliance:{validation_id}",
                "status",
                "cancelled"
            )
            
            # Implement cancellation logic
            # This would typically involve:
            # 1. Sending cancellation signals to running tasks
            # 2. Cleaning up resources
            # 3. Preserving partial results
            
            return True
            
        except Exception as e:
            logger.error(f"Validation cancellation error: {e}")
            return False

# Example usage
if __name__ == '__main__':
    import asyncio
    
    async def main():
        orchestrator = ComplianceOrchestrator()
        
        # Example compliance validation
        result = await orchestrator.run_compliance_validation(
            standard=ComplianceStandard.NIST,
            targets=["scanme.nmap.org"],
            scan_profile="comprehensive"
        )
        
        print(f"Compliance validation completed: {result['status']}")
        print(f"Validation ID: {result['validation_id']}")
        print(f"Coverage: {result['compliance_report']['summary']['coverage_percentage']}%")
        print(f"Compliant: {result['compliance_report']['summary']['compliant']}")
        
    # Run the example
    asyncio.run(main())
    