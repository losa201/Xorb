#!/usr/bin/env python3
"""
Xorb Enhanced Triage Service with Vector Store Integration
Phase 5.1 - Smart Triage Optimization with MiniLM + FAISS + GPT-4o
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

import nats
from nats.js import JetStreamContext
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog

from vector_store_service import VectorStoreService, DeduplicationResult

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("xorb.enhanced_triage")

# Phase 5.1 Required Metrics
triage_dedupe_saved_tokens_total = Counter(
    'triage_dedupe_saved_tokens_total', 
    'Total tokens saved through deduplication'
)
triage_false_positive_score = Gauge(
    'triage_false_positive_score', 
    'Current false positive detection score'
)

# Additional metrics
enhanced_triage_requests = Counter(
    'xorb_enhanced_triage_requests_total', 
    'Total enhanced triage requests', 
    ['status']
)
enhanced_triage_duration = Histogram(
    'xorb_enhanced_triage_duration_seconds', 
    'Enhanced triage processing duration'
)
duplicates_detected_enhanced = Counter(
    'xorb_enhanced_triage_duplicates_total', 
    'Enhanced duplicate findings detected',
    ['method']
)
vector_store_operations = Counter(
    'xorb_vector_store_operations_total',
    'Vector store operations',
    ['operation', 'result']
)
queue_size = Gauge(
    'xorb_enhanced_triage_queue_size', 
    'Number of findings in enhanced triage queue'
)

class EnhancedTriageService:
    """Enhanced AI-powered triage service with vector store deduplication"""
    
    def __init__(self):
        self.nats_client = None
        self.js = None
        self.vector_store = VectorStoreService()
        
        # Configuration
        self.enable_vector_deduplication = True
        self.enable_gpt_fallback = True
        self.batch_size = 10
        
        # Statistics
        self.processed_count = 0
        self.duplicate_count = 0
        self.false_positive_estimates = []
        
    async def initialize(self):
        """Initialize enhanced triage service"""
        try:
            # Initialize vector store
            logger.info("Initializing vector store service...")
            await self.vector_store.initialize()
            
            # Initialize NATS connection
            self.nats_client = await nats.connect(
                servers=[os.getenv("NATS_URL", "nats://localhost:4222")],
                name="xorb-enhanced-triage"
            )
            self.js = self.nats_client.jetstream()
            
            # Subscribe to scan results
            await self.js.subscribe(
                "scans.results",
                cb=self.handle_triage_request,
                queue="enhanced-triage-workers"
            )
            
            # Subscribe to direct triage requests
            await self.js.subscribe(
                "triage.request",
                cb=self.handle_direct_triage_request,
                queue="enhanced-triage-workers"
            )
            
            logger.info("Enhanced triage service initialized", 
                       vector_store_enabled=self.enable_vector_deduplication)
            
        except Exception as e:
            logger.error("Failed to initialize enhanced triage service", error=str(e))
            raise
    
    async def handle_triage_request(self, msg):
        """Handle incoming triage requests from scanner"""
        try:
            scan_results = json.loads(msg.data.decode())
            scan_id = scan_results.get('scan_id')
            
            logger.info("Processing enhanced triage request", scan_id=scan_id)
            
            queue_size.inc()
            
            with enhanced_triage_duration.time():
                # Perform enhanced triage analysis
                triaged_results = await self.perform_enhanced_triage(scan_results)
            
            # Publish triaged results
            await self.publish_triaged_results(scan_id, triaged_results)
            
            # Acknowledge message
            await msg.ack()
            
            queue_size.dec()
            enhanced_triage_requests.labels(status='success').inc()
            
        except Exception as e:
            logger.error("Enhanced triage request failed", error=str(e))
            enhanced_triage_requests.labels(status='error').inc()
            await msg.nak()
        finally:
            queue_size.dec()
    
    async def handle_direct_triage_request(self, msg):
        """Handle direct triage requests for individual vulnerabilities"""
        try:
            vulnerability_data = json.loads(msg.data.decode())
            
            logger.info("Processing direct triage request", 
                       vulnerability_id=vulnerability_data.get('id'))
            
            with enhanced_triage_duration.time():
                result = await self.triage_single_vulnerability(vulnerability_data)
            
            # Publish result
            await self.js.publish(
                "triage.completed",
                json.dumps(result).encode(),
                headers={'vulnerability_id': vulnerability_data.get('id')}
            )
            
            await msg.ack()
            enhanced_triage_requests.labels(status='success').inc()
            
        except Exception as e:
            logger.error("Direct triage request failed", error=str(e))
            enhanced_triage_requests.labels(status='error').inc()
            await msg.nak()
    
    async def perform_enhanced_triage(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive enhanced triage analysis"""
        findings = []
        
        # Extract all vulnerabilities from scan results
        for tool_name, tool_results in scan_results.get('tools', {}).items():
            if 'vulnerabilities' in tool_results:
                for vuln in tool_results['vulnerabilities']:
                    vuln['tool'] = tool_name
                    findings.append(vuln)
        
        if not findings:
            return scan_results
        
        logger.info("Enhanced triaging findings", count=len(findings))
        
        # Process findings in batches for better performance
        triaged_findings = []
        duplicate_findings = []
        
        for i in range(0, len(findings), self.batch_size):
            batch = findings[i:i + self.batch_size]
            batch_results = await asyncio.gather(
                *[self.triage_single_vulnerability(finding) for finding in batch],
                return_exceptions=True
            )
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error("Enhanced triage batch failed", error=str(result))
                    continue
                
                if result.get('deduplication_result', {}).get('is_duplicate'):
                    duplicate_findings.append(result)
                    duplicates_detected_enhanced.labels(method='vector_store').inc()
                else:
                    triaged_findings.append(result)
        
        # Update statistics
        self.processed_count += len(findings)
        self.duplicate_count += len(duplicate_findings)
        
        # Calculate false positive score
        await self._update_false_positive_score(triaged_findings)
        
        # Update scan results with enhanced triage data
        scan_results['enhanced_triage'] = {
            'triaged_findings': triaged_findings,
            'duplicate_findings': duplicate_findings,
            'triage_timestamp': datetime.utcnow().isoformat(),
            'summary': self._generate_enhanced_summary(triaged_findings, duplicate_findings),
            'vector_store_stats': await self.vector_store.get_statistics()
        }
        
        return scan_results
    
    async def triage_single_vulnerability(self, vulnerability_data: Dict[str, Any]) -> Dict[str, Any]:
        """Triage single vulnerability with vector store deduplication"""
        vuln_id = vulnerability_data.get('id', f"vuln_{int(time.time() * 1000)}")
        title = vulnerability_data.get('name', vulnerability_data.get('title', 'Unknown Vulnerability'))
        description = vulnerability_data.get('description', '')
        severity = vulnerability_data.get('severity', 'info')
        target = vulnerability_data.get('url', vulnerability_data.get('target', ''))
        
        # Enhanced vulnerability data
        enhanced_vuln = {
            **vulnerability_data,
            'vulnerability_id': vuln_id,
            'normalized_title': title,
            'normalized_description': description,
            'normalized_severity': severity,
            'normalized_target': target,
            'triage_timestamp': datetime.utcnow().isoformat()
        }
        
        # Perform vector store deduplication if enabled
        deduplication_result = None
        if self.enable_vector_deduplication:
            try:
                # Add to vector store (for future similarity searches)
                await self.vector_store.add_vulnerability_vector(
                    vulnerability_id=vuln_id,
                    title=title,
                    description=description,
                    severity=severity,
                    target=target,
                    metadata={
                        'tool': vulnerability_data.get('tool'),
                        'cvss_score': vulnerability_data.get('cvss_score'),
                        'cwe_id': vulnerability_data.get('cwe_id')
                    }
                )
                vector_store_operations.labels(operation='add_vector', result='success').inc()
                
                # Perform deduplication analysis
                dedupe_result = await self.vector_store.detect_duplicate(
                    vulnerability_id=vuln_id,
                    title=title,
                    description=description,
                    severity=severity,
                    target=target,
                    use_gpt_fallback=self.enable_gpt_fallback
                )
                
                deduplication_result = {
                    'is_duplicate': dedupe_result.is_duplicate,
                    'confidence': dedupe_result.confidence,
                    'duplicate_of': dedupe_result.duplicate_of,
                    'similar_findings_count': len(dedupe_result.similar_findings),
                    'gpt_analysis': dedupe_result.gpt_analysis,
                    'tokens_saved': dedupe_result.tokens_saved,
                    'processing_time': dedupe_result.processing_time,
                    'reasoning': dedupe_result.reasoning,
                    'similar_findings': [
                        {
                            'id': sf.vulnerability_id,
                            'title': sf.title,
                            'similarity_score': sf.similarity_score,
                            'target': sf.target
                        } for sf in dedupe_result.similar_findings[:5]  # Top 5 only
                    ]
                }
                
                # Update metrics
                if dedupe_result.tokens_saved > 0:
                    triage_dedupe_saved_tokens_total.inc(dedupe_result.tokens_saved)
                
                vector_store_operations.labels(operation='deduplication', result='success').inc()
                
            except Exception as e:
                logger.error("Vector store deduplication failed", 
                           vulnerability_id=vuln_id, error=str(e))
                vector_store_operations.labels(operation='deduplication', result='error').inc()
                
                deduplication_result = {
                    'is_duplicate': False,
                    'confidence': 0.0,
                    'error': str(e),
                    'reasoning': f"Deduplication failed: {str(e)}"
                }
        
        # Add deduplication result to enhanced vulnerability data
        enhanced_vuln['deduplication_result'] = deduplication_result
        
        return enhanced_vuln
    
    async def _update_false_positive_score(self, triaged_findings: List[Dict[str, Any]]):
        """Update false positive detection score"""
        if not triaged_findings:
            return
        
        # Simple heuristic: high similarity but not marked as duplicate
        # indicates potential false positive detection
        fp_indicators = []
        
        for finding in triaged_findings:
            dedupe_result = finding.get('deduplication_result', {})
            similar_findings = dedupe_result.get('similar_findings', [])
            
            if similar_findings and not dedupe_result.get('is_duplicate'):
                # High similarity but not duplicate - potential FP detection
                max_similarity = max(sf.get('similarity_score', 0) for sf in similar_findings)
                if max_similarity > 0.8:  # High similarity threshold
                    fp_indicators.append(1.0 - max_similarity)  # Inverse as FP score
        
        if fp_indicators:
            avg_fp_score = sum(fp_indicators) / len(fp_indicators)
            self.false_positive_estimates.append(avg_fp_score)
            
            # Keep only recent estimates
            if len(self.false_positive_estimates) > 100:
                self.false_positive_estimates = self.false_positive_estimates[-100:]
            
            # Update metric
            current_fp_score = sum(self.false_positive_estimates) / len(self.false_positive_estimates)
            triage_false_positive_score.set(current_fp_score)
    
    def _generate_enhanced_summary(
        self, 
        triaged_findings: List[Dict[str, Any]], 
        duplicate_findings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate enhanced triage summary"""
        total_findings = len(triaged_findings) + len(duplicate_findings)
        
        # Severity distribution for unique findings
        severity_distribution = {}
        for finding in triaged_findings:
            severity = finding.get('normalized_severity', 'info')
            severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
        
        # Calculate token savings
        total_tokens_saved = sum(
            finding.get('deduplication_result', {}).get('tokens_saved', 0)
            for finding in duplicate_findings
        )
        
        # GPT analysis usage
        gpt_analyses = sum(
            1 for finding in triaged_findings + duplicate_findings
            if finding.get('deduplication_result', {}).get('gpt_analysis')
        )
        
        return {
            'total_processed': total_findings,
            'unique_findings': len(triaged_findings),
            'duplicate_findings': len(duplicate_findings),
            'deduplication_rate': len(duplicate_findings) / total_findings if total_findings > 0 else 0,
            'severity_distribution': severity_distribution,
            'tokens_saved': total_tokens_saved,
            'gpt_analyses_used': gpt_analyses,
            'efficiency_metrics': {
                'processing_enabled': self.enable_vector_deduplication,
                'gpt_fallback_enabled': self.enable_gpt_fallback,
                'false_positive_score': triage_false_positive_score._value._value
            }
        }
    
    async def publish_triaged_results(self, scan_id: str, results: Dict[str, Any]):
        """Publish enhanced triaged results"""
        try:
            await self.js.publish(
                "triage.enhanced.completed",
                json.dumps(results).encode(),
                headers={'scan_id': scan_id}
            )
            
            # Also publish to legacy topic for backward compatibility
            await self.js.publish(
                "triage.results",
                json.dumps(results).encode(),
                headers={'scan_id': scan_id}
            )
            
            logger.info("Enhanced triaged results published", scan_id=scan_id)
            
        except Exception as e:
            logger.error("Failed to publish enhanced triaged results", 
                        scan_id=scan_id, error=str(e))
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get service health and statistics"""
        vector_stats = await self.vector_store.get_statistics()
        
        return {
            'service': 'enhanced_triage',
            'status': 'healthy',
            'configuration': {
                'vector_deduplication_enabled': self.enable_vector_deduplication,
                'gpt_fallback_enabled': self.enable_gpt_fallback,
                'batch_size': self.batch_size
            },
            'statistics': {
                'processed_count': self.processed_count,
                'duplicate_count': self.duplicate_count,
                'deduplication_rate': self.duplicate_count / self.processed_count if self.processed_count > 0 else 0,
                'false_positive_score': triage_false_positive_score._value._value
            },
            'vector_store': vector_stats
        }

async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "enhanced_triage"}

async def main():
    """Main service entry point"""
    # Start Prometheus metrics server
    start_http_server(8006)  # Different port from basic triage
    
    # Initialize enhanced triage service
    triage = EnhancedTriageService()
    await triage.initialize()
    
    logger.info("🧠 Xorb Enhanced Triage Service with Vector Store started", 
               service_version="5.1",
               features=['minilm_embeddings', 'faiss_similarity', 'gpt4_reranking'])
    
    try:
        # Keep service running
        while True:
            await asyncio.sleep(60)
            
            # Periodic health check and stats logging
            stats = await triage.get_service_health()
            logger.info("Service health check", **stats['statistics'])
            
    except KeyboardInterrupt:
        logger.info("Shutting down enhanced triage service")
    finally:
        if triage.nats_client:
            await triage.nats_client.close()

if __name__ == "__main__":
    asyncio.run(main())