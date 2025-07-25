"""
Xorb PTaaS AI Triage Service
GPT-4o powered duplicate detection and severity analysis
Optimized for AMD EPYC single-node deployment
"""

import asyncio
import json
import logging
import os
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import openai
import anthropic
import nats
from nats.js import JetStreamContext
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

logger = structlog.get_logger("xorb.triage")

# Prometheus metrics
TRIAGE_REQUESTS = Counter('xorb_triage_requests_total', 'Total triage requests', ['status'])
TRIAGE_DURATION = Histogram('xorb_triage_duration_seconds', 'Time spent on triage')
DUPLICATES_DETECTED = Counter('xorb_triage_duplicates_total', 'Duplicate findings detected')
SEVERITY_CHANGES = Counter('xorb_triage_severity_changes_total', 'Severity adjustments', ['from', 'to'])
QUEUE_SIZE = Gauge('xorb_triage_queue_size', 'Number of findings in triage queue')

class AITriageService:
    """AI-powered triage service for vulnerability analysis"""
    
    def __init__(self):
        self.nats_client = None
        self.js = None
        self.openai_client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.anthropic_client = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        
        # In-memory cache for similarity analysis
        self.finding_cache = {}
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.vectors = None
        self.finding_texts = []
        
    async def initialize(self):
        """Initialize NATS connection and JetStream"""
        try:
            self.nats_client = await nats.connect(
                servers=[os.getenv("NATS_URL", "nats://localhost:4222")],
                name="xorb-triage"
            )
            self.js = self.nats_client.jetstream()
            
            # Subscribe to scan results for triage
            await self.js.subscribe(
                "scans.results",
                cb=self.handle_triage_request,
                queue="triage-workers"
            )
            
            logger.info("Triage service initialized", nats_connected=True)
            
        except Exception as e:
            logger.error("Failed to initialize triage service", error=str(e))
            raise
    
    async def handle_triage_request(self, msg):
        """Handle incoming triage requests from scanner"""
        try:
            scan_results = json.loads(msg.data.decode())
            scan_id = scan_results.get('scan_id')
            
            logger.info("Processing triage request", scan_id=scan_id)
            
            QUEUE_SIZE.inc()
            
            with TRIAGE_DURATION.time():
                # Perform triage analysis
                triaged_results = await self.perform_triage(scan_results)
            
            # Publish triaged results
            await self.publish_triaged_results(scan_id, triaged_results)
            
            # Acknowledge message
            await msg.ack()
            
            QUEUE_SIZE.dec()
            TRIAGE_REQUESTS.labels(status='success').inc()
            
        except Exception as e:
            logger.error("Triage request failed", error=str(e))
            TRIAGE_REQUESTS.labels(status='error').inc()
            await msg.nak()
        finally:
            QUEUE_SIZE.dec()
    
    async def perform_triage(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive AI triage analysis"""
        findings = []
        
        # Extract all vulnerabilities from scan results
        for tool_name, tool_results in scan_results.get('tools', {}).items():
            if 'vulnerabilities' in tool_results:
                for vuln in tool_results['vulnerabilities']:
                    vuln['tool'] = tool_name
                    findings.append(vuln)
        
        if not findings:
            return scan_results
        
        logger.info("Triaging findings", count=len(findings))
        
        # Process findings in batches for better performance on EPYC
        batch_size = 10
        triaged_findings = []
        
        for i in range(0, len(findings), batch_size):
            batch = findings[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[self.triage_finding(finding) for finding in batch],
                return_exceptions=True
            )
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error("Triage batch failed", error=str(result))
                    continue
                triaged_findings.append(result)
        
        # Update scan results with triaged findings
        scan_results['triaged_findings'] = triaged_findings
        scan_results['triage_timestamp'] = datetime.utcnow().isoformat()
        scan_results['triage_summary'] = self.generate_triage_summary(triaged_findings)
        
        return scan_results
    
    async def triage_finding(self, finding: Dict[str, Any]) -> Dict[str, Any]:
        """Triage individual finding with AI analysis"""
        # Create finding fingerprint for duplicate detection
        fingerprint = self.create_finding_fingerprint(finding)
        
        # Check for duplicates
        duplicate_info = await self.check_duplicates(finding, fingerprint)
        
        # AI severity analysis
        ai_analysis = await self.analyze_severity(finding)
        
        # Combine original finding with triage results
        triaged_finding = {
            **finding,
            'fingerprint': fingerprint,
            'duplicate_info': duplicate_info,
            'ai_analysis': ai_analysis,
            'triage_timestamp': datetime.utcnow().isoformat()
        }
        
        # Track metrics
        if duplicate_info['is_duplicate']:
            DUPLICATES_DETECTED.inc()
        
        original_severity = finding.get('severity', 'info')
        ai_severity = ai_analysis.get('suggested_severity', original_severity)
        if original_severity != ai_severity:
            SEVERITY_CHANGES.labels(from=original_severity, to=ai_severity).inc()
        
        return triaged_finding
    
    def create_finding_fingerprint(self, finding: Dict[str, Any]) -> str:
        """Create unique fingerprint for finding"""
        # Use key attributes to create fingerprint
        key_data = {
            'name': finding.get('name', ''),
            'description': finding.get('description', ''),
            'id': finding.get('id', ''),
            'package': finding.get('package', ''),
            'url_path': self.normalize_url_path(finding.get('url', ''))
        }
        
        # Create hash from normalized data
        data_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def normalize_url_path(self, url: str) -> str:
        """Normalize URL path for consistent fingerprinting"""
        if not url:
            return ""
        
        # Remove query parameters and fragments
        if '?' in url:
            url = url.split('?')[0]
        if '#' in url:
            url = url.split('#')[0]
        
        # Remove common dynamic parts
        import re
        url = re.sub(r'/\d+/', '/[id]/', url)  # Replace numeric IDs
        url = re.sub(r'[a-f0-9]{8,}', '[hash]', url)  # Replace hashes
        
        return url
    
    async def check_duplicates(self, finding: Dict[str, Any], fingerprint: str) -> Dict[str, Any]:
        """Check for duplicate findings using similarity analysis"""
        # Check exact fingerprint match first
        if fingerprint in self.finding_cache:
            return {
                'is_duplicate': True,
                'duplicate_type': 'exact',
                'confidence': 1.0,
                'original_finding': self.finding_cache[fingerprint]
            }
        
        # Semantic similarity check
        finding_text = self.create_finding_text(finding)
        similarity_result = await self.check_semantic_similarity(finding_text)
        
        # Cache the finding
        self.finding_cache[fingerprint] = {
            'fingerprint': fingerprint,
            'text': finding_text,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return similarity_result
    
    def create_finding_text(self, finding: Dict[str, Any]) -> str:
        """Create text representation of finding for similarity analysis"""
        parts = []
        
        if finding.get('name'):
            parts.append(finding['name'])
        if finding.get('description'):
            parts.append(finding['description'])
        if finding.get('id'):
            parts.append(finding['id'])
        
        return ' '.join(parts)
    
    async def check_semantic_similarity(self, finding_text: str) -> Dict[str, Any]:
        """Check semantic similarity with existing findings"""
        if not self.finding_texts:
            self.finding_texts.append(finding_text)
            return {
                'is_duplicate': False,
                'duplicate_type': 'none',
                'confidence': 0.0
            }
        
        try:
            # Add new finding to texts
            texts = self.finding_texts + [finding_text]
            
            # Vectorize texts
            vectors = self.vectorizer.fit_transform(texts)
            
            # Calculate similarity with existing findings
            new_vector = vectors[-1]
            existing_vectors = vectors[:-1]
            
            similarities = cosine_similarity(new_vector, existing_vectors).flatten()
            max_similarity = np.max(similarities) if len(similarities) > 0 else 0.0
            
            # Update cache
            self.finding_texts.append(finding_text)
            
            # Determine if duplicate based on threshold
            threshold = 0.8
            is_duplicate = max_similarity > threshold
            
            return {
                'is_duplicate': is_duplicate,
                'duplicate_type': 'semantic' if is_duplicate else 'none',
                'confidence': float(max_similarity),
                'threshold': threshold
            }
            
        except Exception as e:
            logger.warning("Similarity check failed", error=str(e))
            return {
                'is_duplicate': False,
                'duplicate_type': 'error',
                'confidence': 0.0,
                'error': str(e)
            }
    
    async def analyze_severity(self, finding: Dict[str, Any]) -> Dict[str, Any]:
        """AI-powered severity analysis using GPT-4o"""
        
        # Prepare context for AI analysis
        context = self.prepare_ai_context(finding)
        
        try:
            # Use GPT-4o for primary analysis
            gpt_analysis = await self.gpt4_analysis(context)
            
            # Use Claude as secondary opinion for critical findings
            claude_analysis = None
            if gpt_analysis.get('suggested_severity') in ['critical', 'high']:
                claude_analysis = await self.claude_analysis(context)
            
            return {
                'gpt4_analysis': gpt_analysis,
                'claude_analysis': claude_analysis,
                'suggested_severity': gpt_analysis.get('suggested_severity'),
                'confidence': gpt_analysis.get('confidence', 0.5),
                'reasoning': gpt_analysis.get('reasoning', ''),
                'false_positive_probability': gpt_analysis.get('false_positive_probability', 0.1)
            }
            
        except Exception as e:
            logger.error("AI severity analysis failed", error=str(e))
            return {
                'suggested_severity': finding.get('severity', 'info'),
                'confidence': 0.1,
                'error': str(e)
            }
    
    def prepare_ai_context(self, finding: Dict[str, Any]) -> str:
        """Prepare context for AI analysis"""
        context_parts = [
            f"Finding ID: {finding.get('id', 'Unknown')}",
            f"Name: {finding.get('name', 'Unknown')}",
            f"Tool: {finding.get('tool', 'Unknown')}",
            f"Original Severity: {finding.get('severity', 'Unknown')}",
        ]
        
        if finding.get('description'):
            context_parts.append(f"Description: {finding['description']}")
        
        if finding.get('url'):
            context_parts.append(f"URL: {finding['url']}")
        
        if finding.get('package'):
            context_parts.append(f"Package: {finding['package']}")
        
        if finding.get('version'):
            context_parts.append(f"Version: {finding['version']}")
        
        return '\n'.join(context_parts)
    
    async def gpt4_analysis(self, context: str) -> Dict[str, Any]:
        """GPT-4o analysis of vulnerability severity"""
        
        prompt = f"""
        Analyze this security finding and provide a severity assessment:

        {context}

        Please analyze:
        1. Actual exploitability and impact
        2. Whether this is a false positive
        3. Appropriate severity level (critical/high/medium/low/info)
        4. Confidence in your assessment (0.0-1.0)

        Respond in JSON format:
        {{
            "suggested_severity": "critical|high|medium|low|info",
            "confidence": 0.0-1.0,
            "reasoning": "explanation of your assessment",
            "false_positive_probability": 0.0-1.0,
            "exploitability": "description of how this could be exploited",
            "impact": "description of potential impact"
        }}
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",  # Use GPT-4 for better analysis
                messages=[
                    {"role": "system", "content": "You are a cybersecurity expert specializing in vulnerability assessment and triage."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content
            
            # Try to parse JSON response
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "suggested_severity": "medium",
                    "confidence": 0.5,
                    "reasoning": response_text,
                    "false_positive_probability": 0.2
                }
                
        except Exception as e:
            logger.error("GPT-4 analysis failed", error=str(e))
            raise
    
    async def claude_analysis(self, context: str) -> Dict[str, Any]:
        """Claude analysis for second opinion on critical findings"""
        
        prompt = f"""
        Provide a second opinion on this security finding:

        {context}

        Focus on:
        1. Confirming the severity assessment
        2. Identifying any factors that might affect severity
        3. Assessing false positive likelihood

        Respond in JSON format with your analysis.
        """
        
        try:
            response = await self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=300,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = response.content[0].text
            
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                return {
                    "analysis": response_text,
                    "confidence": 0.5
                }
                
        except Exception as e:
            logger.warning("Claude analysis failed", error=str(e))
            return None
    
    def generate_triage_summary(self, triaged_findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of triage results"""
        total_findings = len(triaged_findings)
        duplicates = sum(1 for f in triaged_findings if f.get('duplicate_info', {}).get('is_duplicate', False))
        severity_distribution = {}
        
        for finding in triaged_findings:
            severity = finding.get('ai_analysis', {}).get('suggested_severity', 'info')
            severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
        
        return {
            'total_findings': total_findings,
            'unique_findings': total_findings - duplicates,
            'duplicates_detected': duplicates,
            'severity_distribution': severity_distribution,
            'triage_completion_rate': 1.0 if total_findings > 0 else 0.0
        }
    
    async def publish_triaged_results(self, scan_id: str, results: Dict[str, Any]):
        """Publish triaged results for further processing"""
        try:
            await self.js.publish(
                "triage.results",
                json.dumps(results).encode(),
                headers={'scan_id': scan_id}
            )
            logger.info("Triaged results published", scan_id=scan_id)
        except Exception as e:
            logger.error("Failed to publish triaged results", scan_id=scan_id, error=str(e))

async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "triage"}

async def main():
    """Main service entry point"""
    # Start Prometheus metrics server
    start_http_server(8005)
    
    # Initialize triage service
    triage = AITriageService()
    await triage.initialize()
    
    logger.info("Xorb AI Triage service started", 
               epyc_optimized=True,
               ai_providers=['openai', 'anthropic'])
    
    try:
        # Keep service running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down triage service")
    finally:
        await triage.nats_client.close()

if __name__ == "__main__":
    asyncio.run(main())