"""
Domain Services - Business Logic that doesn't belong to a single entity

Domain services encapsulate business logic that involves multiple entities
or doesn't naturally fit within any single entity.
"""

from __future__ import annotations

from decimal import Decimal
from typing import List, Optional, Set, Tuple

from .models import (
    Agent,
    AgentCapability,
    BudgetLimit,
    Campaign,
    CampaignStatus,
    Embedding,
    Finding,
    KnowledgeAtom,
    Severity,
    Target
)

__all__ = [
    "AgentSelectionService",
    "BudgetManagementService", 
    "SimilarityService",
    "TriageService",
    "CampaignOrchestratorService"
]


class AgentSelectionService:
    """Domain service for selecting optimal agents for campaigns"""
    
    @staticmethod
    def select_agents_for_target(
        target: Target,
        available_agents: List[Agent],
        required_capabilities: Set[AgentCapability],
        budget_limit: BudgetLimit,
        max_agents: int = 5
    ) -> List[Agent]:
        """
        Select optimal agents for a target based on capabilities and budget
        
        Args:
            target: The target to test
            available_agents: Available agents to choose from
            required_capabilities: Required capabilities for the campaign
            budget_limit: Budget constraints
            max_agents: Maximum number of agents to select
            
        Returns:
            List of selected agents ordered by priority
        """
        # Filter agents by capabilities and activity status
        capable_agents = [
            agent for agent in available_agents
            if agent.is_active and any(
                agent.can_handle_capability(cap) for cap in required_capabilities
            )
        ]
        
        # Calculate cost efficiency (capabilities per dollar)
        def agent_efficiency(agent: Agent) -> float:
            matching_caps = len(
                required_capabilities.intersection(agent.capabilities)
            )
            if agent.cost_per_execution == 0:
                return float('inf')
            return matching_caps / float(agent.cost_per_execution)
        
        # Sort by efficiency (descending) then by cost (ascending)
        capable_agents.sort(
            key=lambda a: (-agent_efficiency(a), a.cost_per_execution)
        )
        
        # Select agents within budget
        selected_agents = []
        total_cost = Decimal('0')
        
        for agent in capable_agents:
            if len(selected_agents) >= max_agents:
                break
                
            projected_cost = total_cost + agent.cost_per_execution
            if projected_cost <= budget_limit.max_cost_usd:
                selected_agents.append(agent)
                total_cost = projected_cost
        
        return selected_agents


class BudgetManagementService:
    """Domain service for managing campaign budgets"""
    
    @staticmethod
    def calculate_projected_cost(
        agents: List[Agent],
        estimated_executions_per_agent: int = 1
    ) -> Decimal:
        """Calculate projected cost for agent executions"""
        total_cost = Decimal('0')
        
        for agent in agents:
            agent_cost = agent.cost_per_execution * estimated_executions_per_agent
            total_cost += agent_cost
        
        return total_cost
    
    @staticmethod
    def is_budget_exceeded(
        campaign: Campaign,
        current_cost: Decimal,
        duration_hours: int,
        api_calls: int
    ) -> Tuple[bool, List[str]]:
        """
        Check if budget limits are exceeded
        
        Returns:
            Tuple of (is_exceeded, list_of_violations)
        """
        violations = []
        
        if current_cost > campaign.budget.max_cost_usd:
            violations.append(
                f"Cost ${current_cost} exceeds limit ${campaign.budget.max_cost_usd}"
            )
        
        if duration_hours > campaign.budget.max_duration_hours:
            violations.append(
                f"Duration {duration_hours}h exceeds limit {campaign.budget.max_duration_hours}h"
            )
        
        if api_calls > campaign.budget.max_api_calls:
            violations.append(
                f"API calls {api_calls} exceeds limit {campaign.budget.max_api_calls}"
            )
        
        return len(violations) > 0, violations
    
    @staticmethod
    def calculate_budget_utilization(
        campaign: Campaign,
        current_cost: Decimal,
        duration_hours: int,
        api_calls: int
    ) -> dict[str, float]:
        """Calculate budget utilization percentages"""
        return {
            "cost_percentage": float(current_cost / campaign.budget.max_cost_usd * 100),
            "duration_percentage": duration_hours / campaign.budget.max_duration_hours * 100,
            "api_calls_percentage": api_calls / campaign.budget.max_api_calls * 100
        }


class SimilarityService:
    """Domain service for computing semantic similarity between entities"""
    
    @staticmethod
    def compute_cosine_similarity(
        embedding1: Embedding, 
        embedding2: Embedding
    ) -> float:
        """Compute cosine similarity between two embeddings"""
        if embedding1.dimension != embedding2.dimension:
            raise ValueError("Embeddings must have the same dimension")
        
        return embedding1.similarity_cosine(embedding2)
    
    @staticmethod
    def find_similar_findings(
        target_finding: Finding,
        candidate_findings: List[Finding],
        similarity_threshold: float = 0.8,
        max_results: int = 10
    ) -> List[Tuple[Finding, float]]:
        """
        Find findings similar to the target finding
        
        Args:
            target_finding: Finding to compare against
            candidate_findings: List of candidate findings
            similarity_threshold: Minimum similarity score
            max_results: Maximum number of results to return
            
        Returns:
            List of tuples (finding, similarity_score) ordered by similarity
        """
        if not target_finding.embedding:
            return []
        
        similar_findings = []
        
        for candidate in candidate_findings:
            if candidate.id == target_finding.id or not candidate.embedding:
                continue
            
            similarity = SimilarityService.compute_cosine_similarity(
                target_finding.embedding,
                candidate.embedding
            )
            
            if similarity >= similarity_threshold:
                similar_findings.append((candidate, similarity))
        
        # Sort by similarity (descending) and limit results
        similar_findings.sort(key=lambda x: x[1], reverse=True)
        return similar_findings[:max_results]
    
    @staticmethod
    def find_similar_knowledge_atoms(
        target_atom: KnowledgeAtom,
        candidate_atoms: List[KnowledgeAtom],
        similarity_threshold: float = 0.7,
        max_results: int = 5
    ) -> List[Tuple[KnowledgeAtom, float]]:
        """Find knowledge atoms similar to the target atom"""
        if not target_atom.embedding:
            return []
        
        similar_atoms = []
        
        for candidate in candidate_atoms:
            if candidate.id == target_atom.id or not candidate.embedding:
                continue
            
            similarity = SimilarityService.compute_cosine_similarity(
                target_atom.embedding,
                candidate.embedding
            )
            
            if similarity >= similarity_threshold:
                similar_atoms.append((candidate, similarity))
        
        similar_atoms.sort(key=lambda x: x[1], reverse=True)
        return similar_atoms[:max_results]


class TriageService:
    """Domain service for triaging security findings"""
    
    @staticmethod
    def calculate_priority_score(finding: Finding) -> float:
        """
        Calculate priority score for finding triage
        
        Score based on severity, confidence, and other factors
        Returns score between 0.0 and 1.0
        """
        # Base score from severity
        severity_scores = {
            Severity.CRITICAL: 1.0,
            Severity.HIGH: 0.8,
            Severity.MEDIUM: 0.6,
            Severity.LOW: 0.4,
            Severity.INFO: 0.2
        }
        
        base_score = severity_scores[finding.severity]
        
        # Adjust based on evidence quality (number of evidence items)
        evidence_bonus = min(len(finding.evidence) * 0.05, 0.2)
        
        return min(base_score + evidence_bonus, 1.0)
    
    @staticmethod
    def is_likely_duplicate(
        finding: Finding,
        existing_findings: List[Finding],
        similarity_threshold: float = 0.9
    ) -> Optional[Finding]:
        """
        Check if finding is likely a duplicate of an existing finding
        
        Returns the original finding if duplicate detected, None otherwise
        """
        similar_findings = SimilarityService.find_similar_findings(
            finding,
            existing_findings,
            similarity_threshold,
            max_results=1
        )
        
        if similar_findings:
            original_finding, similarity = similar_findings[0]
            # Additional checks for duplicate detection
            if (finding.title.lower() == original_finding.title.lower() or 
                similarity > 0.95):
                return original_finding
        
        return None
    
    @staticmethod
    def suggest_remediation_priority(
        findings: List[Finding]
    ) -> List[Finding]:
        """
        Sort findings by suggested remediation priority
        
        Priority based on severity, exploitability, and impact
        """
        def priority_key(finding: Finding) -> Tuple[float, int]:
            priority_score = TriageService.calculate_priority_score(finding)
            # Secondary sort by creation time (newer first)
            timestamp_score = int(finding.created_at.timestamp())
            return (-priority_score, -timestamp_score)
        
        return sorted(findings, key=priority_key)


class CampaignOrchestratorService:
    """Domain service for orchestrating campaign execution"""
    
    @staticmethod
    def can_start_campaign(campaign: Campaign) -> Tuple[bool, List[str]]:
        """
        Check if campaign can be started
        
        Returns:
            Tuple of (can_start, list_of_blocking_issues)
        """
        issues = []
        
        if campaign.status != CampaignStatus.QUEUED:
            issues.append(f"Campaign status is {campaign.status}, must be QUEUED")
        
        if not campaign.scheduled_agents:
            issues.append("No agents scheduled for campaign")
        
        if not campaign.target.scope.domains and not campaign.target.scope.ip_ranges:
            issues.append("Target has no domains or IP ranges in scope")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def estimate_campaign_duration(
        agents: List[Agent],
        parallel_execution: bool = True
    ) -> int:
        """
        Estimate campaign duration in minutes
        
        Args:
            agents: List of agents to execute
            parallel_execution: Whether agents run in parallel
            
        Returns:
            Estimated duration in minutes
        """
        if not agents:
            return 0
        
        if parallel_execution:
            # Duration is the longest-running agent
            return max(agent.average_duration_minutes for agent in agents)
        else:
            # Duration is sum of all agents
            return sum(agent.average_duration_minutes for agent in agents)
    
    @staticmethod
    def should_continue_campaign(
        campaign: Campaign,
        current_cost: Decimal,
        duration_hours: int,
        api_calls: int,
        error_rate: float = 0.0
    ) -> Tuple[bool, str]:
        """
        Determine if campaign should continue execution
        
        Returns:
            Tuple of (should_continue, reason)
        """
        # Check budget limits
        is_exceeded, violations = BudgetManagementService.is_budget_exceeded(
            campaign, current_cost, duration_hours, api_calls
        )
        
        if is_exceeded:
            return False, f"Budget exceeded: {'; '.join(violations)}"
        
        # Check error rate
        if error_rate > 0.5:  # More than 50% errors
            return False, f"High error rate: {error_rate:.1%}"
        
        # Check if campaign is in valid running state
        if campaign.status != CampaignStatus.RUNNING:
            return False, f"Campaign status is {campaign.status}"
        
        return True, "Continue execution"