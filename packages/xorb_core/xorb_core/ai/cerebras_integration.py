#!/usr/bin/env python3
"""
Cerebras AI Integration for Xorb 2.0 EPYC-Optimized Platform

This module provides integration with Cerebras AI models for enhanced
security intelligence generation, threat analysis, and automated reporting
with EPYC-specific optimizations for high-performance inference.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
from dataclasses import dataclass
from datetime import datetime
import json
import time

try:
    from cerebras.cloud.sdk import Cerebras
    CEREBRAS_AVAILABLE = True
except ImportError:
    CEREBRAS_AVAILABLE = False
    logging.warning("Cerebras Cloud SDK not available. Install with: pip install cerebras-cloud-sdk")

from ..orchestration.epyc_numa_optimizer import EPYCNUMAOptimizer


@dataclass
class CerebrasConfig:
    """Cerebras AI configuration for EPYC-optimized deployment"""
    api_key: str
    model: str = "llama3.1-70b"  # Default model
    max_completion_tokens: int = 40000
    temperature: float = 0.6
    top_p: float = 0.95
    stream: bool = True
    epyc_parallel_requests: int = 8  # Optimized for EPYC CCX count
    request_timeout: int = 120
    retry_attempts: int = 3
    numa_aware_batching: bool = True


@dataclass
class SecurityAnalysisRequest:
    """Request for security analysis using Cerebras AI"""
    analysis_type: str  # 'vulnerability', 'threat_intel', 'incident_response', 'payload_analysis'
    target_data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    priority: str = "medium"  # 'low', 'medium', 'high', 'critical'
    user_id: str = "system"


@dataclass
class CerebrasResponse:
    """Response from Cerebras AI analysis"""
    request_id: str
    analysis_type: str
    content: str
    confidence_score: float
    processing_time: float
    tokens_used: int
    model_used: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_id': self.request_id,
            'analysis_type': self.analysis_type,
            'content': self.content,
            'confidence_score': self.confidence_score,
            'processing_time': self.processing_time,
            'tokens_used': self.tokens_used,
            'model_used': self.model_used,
            'timestamp': self.timestamp.isoformat()
        }


class EPYCOptimizedCerebrasClient:
    """
    EPYC-optimized Cerebras AI client with NUMA-aware processing
    and high-performance inference capabilities
    """
    
    def __init__(self, config: CerebrasConfig, numa_optimizer: EPYCNUMAOptimizer = None):
        if not CEREBRAS_AVAILABLE:
            raise ImportError("Cerebras Cloud SDK not available")
            
        self.config = config
        self.numa_optimizer = numa_optimizer or EPYCNUMAOptimizer()
        self.logger = logging.getLogger(__name__)
        
        # Initialize Cerebras client
        self.client = Cerebras(api_key=config.api_key)
        
        # EPYC-specific optimizations
        self.request_semaphore = asyncio.Semaphore(config.epyc_parallel_requests)
        self.numa_request_pools = self._initialize_numa_pools()
        
        # Performance tracking
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'tokens_processed': 0,
            'numa_locality_ratio': 0.0
        }
        
        self.logger.info(f"EPYC-optimized Cerebras client initialized with {config.epyc_parallel_requests} parallel requests")
    
    def _initialize_numa_pools(self) -> Dict[int, List]:
        """Initialize NUMA-aware request pools for load balancing"""
        pools = {}
        for numa_node in range(self.numa_optimizer.topology.numa_nodes):
            pools[numa_node] = []
        return pools
    
    async def analyze_security_data(self, request: SecurityAnalysisRequest) -> CerebrasResponse:
        """Analyze security data using Cerebras AI with EPYC optimization"""
        start_time = time.time()
        request_id = f"cerebras_{int(time.time() * 1000)}"
        
        try:
            # Acquire semaphore for rate limiting
            async with self.request_semaphore:
                # Get optimal NUMA node for this request
                numa_node = await self._get_optimal_numa_node()
                
                # Prepare system prompt based on analysis type
                system_prompt = self._get_system_prompt(request.analysis_type)
                
                # Prepare user message with context
                user_message = self._prepare_user_message(request)
                
                # Make Cerebras API call with EPYC optimization
                response_content, tokens_used = await self._make_cerebras_request(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    numa_node=numa_node
                )
                
                # Calculate confidence score based on response quality
                confidence_score = self._calculate_confidence_score(response_content, request.analysis_type)
                
                processing_time = time.time() - start_time
                
                # Update performance metrics
                await self._update_performance_metrics(processing_time, tokens_used, numa_node)
                
                # Create response object
                response = CerebrasResponse(
                    request_id=request_id,
                    analysis_type=request.analysis_type,
                    content=response_content,
                    confidence_score=confidence_score,
                    processing_time=processing_time,
                    tokens_used=tokens_used,
                    model_used=self.config.model,
                    timestamp=datetime.utcnow()
                )
                
                self.logger.info(f"Cerebras analysis completed: {request_id} in {processing_time:.2f}s")
                return response
                
        except Exception as e:
            self.performance_metrics['failed_requests'] += 1
            self.logger.error(f"Cerebras analysis failed for {request_id}: {e}")
            raise
    
    def _get_system_prompt(self, analysis_type: str) -> str:
        """Get system prompt based on analysis type"""
        prompts = {
            'vulnerability': """You are an expert cybersecurity analyst specializing in vulnerability assessment. 
            Analyze the provided data for security vulnerabilities, assess their severity, and provide 
            actionable remediation recommendations. Focus on technical accuracy and practical solutions.
            
            Provide your analysis in a structured format with:
            1. Executive Summary
            2. Vulnerability Details
            3. Risk Assessment (Critical/High/Medium/Low)
            4. Technical Impact Analysis
            5. Remediation Steps
            6. Prevention Measures""",
            
            'threat_intel': """You are a threat intelligence analyst with expertise in cyber threat landscape analysis.
            Analyze the provided threat data, identify threat actors, tactics, techniques, and procedures (TTPs),
            and provide intelligence-driven insights for proactive defense.
            
            Structure your analysis with:
            1. Threat Overview
            2. Attribution Analysis
            3. TTP Mapping (MITRE ATT&CK)
            4. Indicators of Compromise (IoCs)
            5. Threat Hunting Recommendations
            6. Defensive Countermeasures""",
            
            'incident_response': """You are an incident response specialist analyzing security incidents.
            Provide comprehensive incident analysis, timeline reconstruction, and response recommendations
            based on the provided incident data.
            
            Format your response with:
            1. Incident Classification
            2. Timeline Analysis
            3. Attack Vector Assessment
            4. Impact Analysis
            5. Containment Recommendations
            6. Recovery Steps
            7. Lessons Learned""",
            
            'payload_analysis': """You are a malware analyst specializing in payload analysis and reverse engineering.
            Analyze the provided payload data for malicious behavior, evasion techniques, and potential impact.
            
            Provide analysis covering:
            1. Payload Classification
            2. Behavioral Analysis
            3. Evasion Techniques
            4. Network Communications
            5. Persistence Mechanisms
            6. Detection Signatures
            7. Mitigation Strategies""",
            
            'campaign_optimization': """You are a security testing expert specializing in penetration testing campaign optimization.
            Analyze the provided campaign data and provide recommendations for improving test coverage,
            methodology, and effectiveness.
            
            Structure your recommendations with:
            1. Campaign Assessment
            2. Coverage Analysis
            3. Methodology Improvements
            4. Target Prioritization
            5. Testing Efficiency
            6. Risk-Based Recommendations"""
        }
        
        return prompts.get(analysis_type, prompts['vulnerability'])
    
    def _prepare_user_message(self, request: SecurityAnalysisRequest) -> str:
        """Prepare user message with context"""
        message_parts = [f"Analysis Type: {request.analysis_type.title()}"]
        
        # Add target data
        if request.target_data:
            message_parts.append(f"\nTarget Data:\n{json.dumps(request.target_data, indent=2)}")
        
        # Add context if provided
        if request.context:
            message_parts.append(f"\nAdditional Context:\n{json.dumps(request.context, indent=2)}")
        
        # Add priority and requirements
        message_parts.append(f"\nPriority Level: {request.priority.title()}")
        message_parts.append(f"User ID: {request.user_id}")
        
        # Add specific instructions based on priority
        if request.priority == "critical":
            message_parts.append("\nThis is a CRITICAL analysis. Provide immediate actionable insights and urgent recommendations.")
        elif request.priority == "high":
            message_parts.append("\nThis is a HIGH priority analysis. Focus on key risks and primary mitigation strategies.")
        
        return "\n".join(message_parts)
    
    async def _make_cerebras_request(self, system_prompt: str, user_message: str, numa_node: int) -> tuple[str, int]:
        """Make optimized Cerebras API request with NUMA awareness"""
        
        # Allocate process resources on optimal NUMA node
        if self.config.numa_aware_batching:
            process_allocation = await self.numa_optimizer.allocate_process_resources(
                process_id=f"cerebras_request_{int(time.time() * 1000)}",
                process_type="ai_inference",
                cpu_requirement=0.125,  # 1/8 of total for parallel processing
                memory_requirement=2 * 1024 * 1024 * 1024,  # 2GB
                locality_preference=f"node_{numa_node}"
            )
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            if self.config.stream:
                # Streaming response for better performance
                response_content = ""
                tokens_used = 0
                
                stream = self.client.chat.completions.create(
                    messages=messages,
                    model=self.config.model,
                    stream=True,
                    max_completion_tokens=self.config.max_completion_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        response_content += chunk.choices[0].delta.content
                        tokens_used += 1  # Approximate token count
                
            else:
                # Non-streaming response
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.config.model,
                    stream=False,
                    max_completion_tokens=self.config.max_completion_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p
                )
                
                response_content = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0
            
            return response_content, tokens_used
            
        finally:
            # Cleanup NUMA allocation
            if self.config.numa_aware_batching and 'process_allocation' in locals():
                await self.numa_optimizer.deallocate_process_resources(process_allocation.process_id)
    
    async def _get_optimal_numa_node(self) -> int:
        """Get optimal NUMA node for request processing"""
        if not self.config.numa_aware_batching:
            return 0
        
        # Use NUMA optimizer to select optimal node
        stats = await self.numa_optimizer.get_numa_utilization_stats()
        utilization = stats.get('utilization', {})
        
        # Select least utilized NUMA node
        if utilization:
            return min(utilization.keys(), key=lambda k: utilization[k])
        
        return 0
    
    def _calculate_confidence_score(self, content: str, analysis_type: str) -> float:
        """Calculate confidence score based on response quality"""
        base_score = 0.8
        
        # Check for structured response
        if any(marker in content.lower() for marker in ['1.', '2.', '3.', 'summary', 'analysis']):
            base_score += 0.1
        
        # Check for specific keywords based on analysis type
        type_keywords = {
            'vulnerability': ['cve', 'severity', 'cvss', 'remediation', 'exploit'],
            'threat_intel': ['ttp', 'mitre', 'apt', 'campaign', 'attribution'],
            'incident_response': ['timeline', 'containment', 'recovery', 'forensics'],
            'payload_analysis': ['malware', 'behavior', 'signature', 'hash', 'detection']
        }
        
        keywords = type_keywords.get(analysis_type, [])
        keyword_matches = sum(1 for keyword in keywords if keyword in content.lower())
        keyword_score = min(0.1, keyword_matches * 0.02)
        
        # Check content length (longer responses generally more comprehensive)
        length_score = min(0.05, len(content) / 10000)
        
        return min(1.0, base_score + keyword_score + length_score)
    
    async def _update_performance_metrics(self, processing_time: float, tokens_used: int, numa_node: int):
        """Update performance metrics with NUMA awareness"""
        self.performance_metrics['total_requests'] += 1
        self.performance_metrics['successful_requests'] += 1
        self.performance_metrics['tokens_processed'] += tokens_used
        
        # Update average response time
        total_requests = self.performance_metrics['total_requests']
        current_avg = self.performance_metrics['average_response_time']
        self.performance_metrics['average_response_time'] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )
        
        # Track NUMA locality (simplified metric)
        if numa_node == 0:  # Prefer NUMA node 0 for consistency
            locality_improvement = 0.1
        else:
            locality_improvement = -0.05
        
        current_locality = self.performance_metrics['numa_locality_ratio']
        self.performance_metrics['numa_locality_ratio'] = max(0.0, min(1.0, 
            current_locality + locality_improvement / total_requests
        ))
    
    async def batch_analyze(self, requests: List[SecurityAnalysisRequest]) -> List[CerebrasResponse]:
        """Batch analyze multiple requests with EPYC optimization"""
        if not requests:
            return []
        
        self.logger.info(f"Starting batch analysis of {len(requests)} requests")
        
        # Group requests by NUMA node for optimal processing
        numa_groups = await self._group_requests_by_numa(requests)
        
        # Process groups concurrently
        tasks = []
        for numa_node, node_requests in numa_groups.items():
            task = self._process_numa_group(numa_node, node_requests)
            tasks.append(task)
        
        # Wait for all groups to complete
        group_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        results = []
        for group_result in group_results:
            if isinstance(group_result, list):
                results.extend(group_result)
            elif isinstance(group_result, Exception):
                self.logger.error(f"Batch processing error: {group_result}")
        
        self.logger.info(f"Batch analysis completed: {len(results)} successful responses")
        return results
    
    async def _group_requests_by_numa(self, requests: List[SecurityAnalysisRequest]) -> Dict[int, List[SecurityAnalysisRequest]]:
        """Group requests by optimal NUMA node"""
        groups = {i: [] for i in range(self.numa_optimizer.topology.numa_nodes)}
        
        for i, request in enumerate(requests):
            # Simple round-robin assignment for now
            numa_node = i % self.numa_optimizer.topology.numa_nodes
            groups[numa_node].append(request)
        
        return groups
    
    async def _process_numa_group(self, numa_node: int, requests: List[SecurityAnalysisRequest]) -> List[CerebrasResponse]:
        """Process a group of requests on a specific NUMA node"""
        results = []
        
        # Process requests with limited concurrency per NUMA node
        semaphore = asyncio.Semaphore(self.config.epyc_parallel_requests // self.numa_optimizer.topology.numa_nodes)
        
        async def process_single_request(request):
            async with semaphore:
                try:
                    return await self.analyze_security_data(request)
                except Exception as e:
                    self.logger.error(f"Request processing failed: {e}")
                    return None
        
        tasks = [process_single_request(request) for request in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for response in responses:
            if isinstance(response, CerebrasResponse):
                results.append(response)
        
        return results
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        numa_stats = await self.numa_optimizer.get_numa_utilization_stats()
        
        return {
            **self.performance_metrics,
            'numa_stats': numa_stats,
            'config': {
                'model': self.config.model,
                'parallel_requests': self.config.epyc_parallel_requests,
                'numa_aware_batching': self.config.numa_aware_batching
            }
        }
    
    async def optimize_for_workload(self, workload_type: str):
        """Optimize client configuration for specific workload"""
        if workload_type == "high_throughput":
            self.config.epyc_parallel_requests = 16
            self.config.temperature = 0.3
            self.config.stream = True
        elif workload_type == "high_quality":
            self.config.epyc_parallel_requests = 4
            self.config.temperature = 0.8
            self.config.max_completion_tokens = 60000
        elif workload_type == "real_time":
            self.config.epyc_parallel_requests = 8
            self.config.temperature = 0.5
            self.config.stream = True
            self.config.max_completion_tokens = 20000
        
        self.logger.info(f"Optimized configuration for {workload_type} workload")


class CerebrasSecurityAnalyzer:
    """High-level security analyzer using Cerebras AI"""
    
    def __init__(self, cerebras_client: EPYCOptimizedCerebrasClient):
        self.client = cerebras_client
        self.logger = logging.getLogger(__name__)
    
    async def analyze_vulnerability_scan(self, scan_results: Dict[str, Any], 
                                       context: Dict[str, Any] = None) -> CerebrasResponse:
        """Analyze vulnerability scan results"""
        request = SecurityAnalysisRequest(
            analysis_type='vulnerability',
            target_data=scan_results,
            context=context,
            priority='high'
        )
        
        return await self.client.analyze_security_data(request)
    
    async def analyze_threat_intelligence(self, threat_data: Dict[str, Any],
                                        context: Dict[str, Any] = None) -> CerebrasResponse:
        """Analyze threat intelligence data"""
        request = SecurityAnalysisRequest(
            analysis_type='threat_intel',
            target_data=threat_data,
            context=context,
            priority='medium'
        )
        
        return await self.client.analyze_security_data(request)
    
    async def analyze_security_incident(self, incident_data: Dict[str, Any],
                                      context: Dict[str, Any] = None) -> CerebrasResponse:
        """Analyze security incident"""
        request = SecurityAnalysisRequest(
            analysis_type='incident_response',
            target_data=incident_data,
            context=context,
            priority='critical'
        )
        
        return await self.client.analyze_security_data(request)
    
    async def analyze_malware_payload(self, payload_data: Dict[str, Any],
                                    context: Dict[str, Any] = None) -> CerebrasResponse:
        """Analyze malware payload"""
        request = SecurityAnalysisRequest(
            analysis_type='payload_analysis',
            target_data=payload_data,
            context=context,
            priority='high'
        )
        
        return await self.client.analyze_security_data(request)
    
    async def optimize_campaign(self, campaign_data: Dict[str, Any],
                              performance_metrics: Dict[str, Any] = None) -> CerebrasResponse:
        """Optimize security testing campaign"""
        request = SecurityAnalysisRequest(
            analysis_type='campaign_optimization',
            target_data=campaign_data,
            context={'performance_metrics': performance_metrics} if performance_metrics else None,
            priority='medium'
        )
        
        return await self.client.analyze_security_data(request)


# Factory function for easy initialization
def create_cerebras_client(api_key: str = None, model: str = "llama3.1-70b") -> EPYCOptimizedCerebrasClient:
    """Create EPYC-optimized Cerebras client"""
    if not CEREBRAS_AVAILABLE:
        raise ImportError("Cerebras Cloud SDK not available. Install with: pip install cerebras-cloud-sdk")
    
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError("CEREBRAS_API_KEY not provided and not found in environment")
    
    config = CerebrasConfig(
        api_key=api_key,
        model=model,
        epyc_parallel_requests=8  # Optimized for EPYC CCX count
    )
    
    numa_optimizer = EPYCNUMAOptimizer()
    
    return EPYCOptimizedCerebrasClient(config, numa_optimizer)


# Example usage
async def main():
    """Example usage of Cerebras integration"""
    
    # Initialize client
    client = create_cerebras_client(
        api_key="csk-3dnfdhv685envept86ycrfnckvxmndwm3rvphmjx52c3pf94",
        model="llama3.1-70b"
    )
    
    # Create security analyzer
    analyzer = CerebrasSecurityAnalyzer(client)
    
    # Example vulnerability analysis
    scan_results = {
        "target": "https://example.com",
        "vulnerabilities": [
            {
                "type": "SQL Injection",
                "severity": "High",
                "location": "/login.php",
                "parameter": "username"
            },
            {
                "type": "XSS",
                "severity": "Medium", 
                "location": "/search.php",
                "parameter": "query"
            }
        ],
        "scan_date": "2025-07-23",
        "scanner": "Xorb 2.0"
    }
    
    try:
        # Analyze vulnerability scan
        response = await analyzer.analyze_vulnerability_scan(
            scan_results=scan_results,
            context={"customer": "Enterprise Client", "environment": "Production"}
        )
        
        print(f"Analysis completed in {response.processing_time:.2f}s")
        print(f"Confidence: {response.confidence_score:.2f}")
        print(f"Tokens used: {response.tokens_used}")
        print(f"\nAnalysis:\n{response.content}")
        
        # Get performance metrics
        metrics = await client.get_performance_metrics()
        print(f"\nPerformance Metrics: {metrics}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())