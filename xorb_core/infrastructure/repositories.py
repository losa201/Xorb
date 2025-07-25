"""
Repository Implementations - Database Adapters

Concrete implementations of repository ports using PostgreSQL and Neo4j.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import asyncpg
import structlog
from neo4j import AsyncGraphDatabase

from ..application.ports import (
    AgentRepository,
    CampaignRepository,
    FindingRepository,
    KnowledgeAtomRepository,
    TargetRepository
)
from ..domain import (
    Agent,
    AgentCapability,
    AgentId,
    AtomId,
    AtomType,
    BudgetLimit,
    Campaign,
    CampaignId,
    CampaignStatus,
    Embedding,
    Finding,
    FindingId,
    FindingStatus,
    KnowledgeAtom,
    Severity,
    Target,
    TargetId,
    TargetScope
)

__all__ = [
    "PostgreSQLTargetRepository",
    "PostgreSQLAgentRepository",
    "PostgreSQLCampaignRepository",
    "PostgreSQLFindingRepository",
    "Neo4jKnowledgeAtomRepository"
]

log = structlog.get_logger(__name__)


class PostgreSQLTargetRepository(TargetRepository):
    """PostgreSQL implementation of TargetRepository"""
    
    def __init__(self, connection_pool: asyncpg.Pool) -> None:
        self._pool = connection_pool
    
    async def save(self, target: Target) -> None:
        """Save a target to PostgreSQL"""
        
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO targets (id, name, scope_domains, scope_ip_ranges, 
                                   excluded_domains, excluded_ips, created_at, updated_at, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    scope_domains = EXCLUDED.scope_domains,
                    scope_ip_ranges = EXCLUDED.scope_ip_ranges,
                    excluded_domains = EXCLUDED.excluded_domains,
                    excluded_ips = EXCLUDED.excluded_ips,
                    updated_at = EXCLUDED.updated_at,
                    metadata = EXCLUDED.metadata
            """,
                str(target.id),
                target.name,
                list(target.scope.domains),
                list(target.scope.ip_ranges),
                list(target.scope.excluded_domains),
                list(target.scope.excluded_ips),
                target.created_at,
                target.updated_at,
                json.dumps(target.metadata)
            )
        
        log.info("Target saved", target_id=str(target.id), name=target.name)
    
    async def find_by_id(self, target_id: TargetId) -> Optional[Target]:
        """Find target by ID"""
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM targets WHERE id = $1",
                str(target_id)
            )
            
            if not row:
                return None
            
            return self._row_to_target(row)
    
    async def find_by_name(self, name: str) -> Optional[Target]:
        """Find target by name"""
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM targets WHERE name = $1",
                name
            )
            
            if not row:
                return None
            
            return self._row_to_target(row)
    
    async def find_all(self) -> List[Target]:
        """Find all targets"""
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM targets ORDER BY created_at DESC")
            
            return [self._row_to_target(row) for row in rows]
    
    async def delete(self, target: Target) -> None:
        """Delete a target"""
        
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM targets WHERE id = $1", str(target.id))
        
        log.info("Target deleted", target_id=str(target.id))
    
    def _row_to_target(self, row: asyncpg.Record) -> Target:
        """Convert database row to Target entity"""
        
        scope = TargetScope(
            domains=set(row['scope_domains']),
            ip_ranges=set(row['scope_ip_ranges']),
            excluded_domains=set(row['excluded_domains']),
            excluded_ips=set(row['excluded_ips'])
        )
        
        return Target(
            id_=TargetId.from_string(row['id']),
            name=row['name'],
            scope=scope,
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )


class PostgreSQLAgentRepository(AgentRepository):
    """PostgreSQL implementation of AgentRepository"""
    
    def __init__(self, connection_pool: asyncpg.Pool) -> None:
        self._pool = connection_pool
    
    async def save(self, agent: Agent) -> None:
        """Save an agent to PostgreSQL"""
        
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO agents (id, name, capabilities, cost_per_execution,
                                  average_duration_minutes, created_at, is_active, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    capabilities = EXCLUDED.capabilities,
                    cost_per_execution = EXCLUDED.cost_per_execution,
                    average_duration_minutes = EXCLUDED.average_duration_minutes,
                    is_active = EXCLUDED.is_active,
                    metadata = EXCLUDED.metadata
            """,
                str(agent.id),
                agent.name,
                [cap.value for cap in agent.capabilities],
                agent.cost_per_execution,
                agent.average_duration_minutes,
                agent.created_at,
                agent.is_active,
                json.dumps(agent.metadata)
            )
        
        log.info("Agent saved", agent_id=str(agent.id), name=agent.name)
    
    async def find_by_id(self, agent_id: AgentId) -> Optional[Agent]:
        """Find agent by ID"""
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM agents WHERE id = $1",
                str(agent_id)
            )
            
            if not row:
                return None
            
            return self._row_to_agent(row)
    
    async def find_active_agents(self) -> List[Agent]:
        """Find all active agents"""
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM agents WHERE is_active = true ORDER BY name"
            )
            
            return [self._row_to_agent(row) for row in rows]
    
    async def find_by_capabilities(self, capabilities: List[str]) -> List[Agent]:
        """Find agents with specific capabilities"""
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM agents 
                WHERE is_active = true 
                AND capabilities && $1
                ORDER BY name
            """, capabilities)
            
            return [self._row_to_agent(row) for row in rows]
    
    async def delete(self, agent: Agent) -> None:
        """Delete an agent"""
        
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM agents WHERE id = $1", str(agent.id))
        
        log.info("Agent deleted", agent_id=str(agent.id))
    
    def _row_to_agent(self, row: asyncpg.Record) -> Agent:
        """Convert database row to Agent entity"""
        
        capabilities = {AgentCapability(cap) for cap in row['capabilities']}
        
        return Agent(
            id_=AgentId.from_string(row['id']),
            name=row['name'],
            capabilities=capabilities,
            cost_per_execution=row['cost_per_execution'],
            average_duration_minutes=row['average_duration_minutes'],
            created_at=row['created_at'],
            is_active=row['is_active'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )


class PostgreSQLCampaignRepository(CampaignRepository):
    """PostgreSQL implementation of CampaignRepository"""
    
    def __init__(
        self, 
        connection_pool: asyncpg.Pool,
        target_repository: TargetRepository
    ) -> None:
        self._pool = connection_pool
        self._target_repository = target_repository
    
    async def save(self, campaign: Campaign) -> None:
        """Save a campaign to PostgreSQL"""
        
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO campaigns (id, name, target_id, max_cost_usd, max_duration_hours,
                                     max_api_calls, created_at, status, scheduled_agents, 
                                     findings, started_at, completed_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    status = EXCLUDED.status,
                    scheduled_agents = EXCLUDED.scheduled_agents,
                    findings = EXCLUDED.findings,
                    started_at = EXCLUDED.started_at,
                    completed_at = EXCLUDED.completed_at
            """,
                str(campaign.id),
                campaign.name,
                str(campaign.target.id),
                campaign.budget.max_cost_usd,
                campaign.budget.max_duration_hours,
                campaign.budget.max_api_calls,
                campaign.created_at,
                campaign.status.value,
                [str(agent_id) for agent_id in campaign.scheduled_agents],
                [str(finding_id) for finding_id in campaign.findings],
                campaign.started_at,
                campaign.completed_at
            )
        
        log.info("Campaign saved", campaign_id=str(campaign.id), name=campaign.name)
    
    async def find_by_id(self, campaign_id: CampaignId) -> Optional[Campaign]:
        """Find campaign by ID"""
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM campaigns WHERE id = $1",
                str(campaign_id)
            )
            
            if not row:
                return None
            
            return await self._row_to_campaign(row)
    
    async def find_by_status(self, status: str) -> List[Campaign]:
        """Find campaigns by status"""
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM campaigns WHERE status = $1 ORDER BY created_at DESC",
                status
            )
            
            campaigns = []
            for row in rows:
                campaign = await self._row_to_campaign(row)
                if campaign:
                    campaigns.append(campaign)
            
            return campaigns
    
    async def find_active_campaigns(self) -> List[Campaign]:
        """Find all running campaigns"""
        
        return await self.find_by_status(CampaignStatus.RUNNING.value)
    
    async def delete(self, campaign: Campaign) -> None:
        """Delete a campaign"""
        
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM campaigns WHERE id = $1", str(campaign.id))
        
        log.info("Campaign deleted", campaign_id=str(campaign.id))
    
    async def _row_to_campaign(self, row: asyncpg.Record) -> Optional[Campaign]:
        """Convert database row to Campaign entity"""
        
        # Fetch target
        target = await self._target_repository.find_by_id(
            TargetId.from_string(row['target_id'])
        )
        
        if not target:
            log.warning("Target not found for campaign", campaign_id=row['id'])
            return None
        
        budget = BudgetLimit(
            max_cost_usd=row['max_cost_usd'],
            max_duration_hours=row['max_duration_hours'],
            max_api_calls=row['max_api_calls']
        )
        
        scheduled_agents = [AgentId.from_string(id_) for id_ in row['scheduled_agents']]
        findings = [FindingId.from_string(id_) for id_ in row['findings']]
        
        campaign = Campaign(
            id_=CampaignId.from_string(row['id']),
            name=row['name'],
            target=target,
            budget=budget,
            created_at=row['created_at'],
            status=CampaignStatus(row['status']),
            scheduled_agents=scheduled_agents,
            findings=findings
        )
        
        campaign.started_at = row['started_at']
        campaign.completed_at = row['completed_at']
        
        return campaign


class PostgreSQLFindingRepository(FindingRepository):
    """PostgreSQL implementation of FindingRepository"""
    
    def __init__(self, connection_pool: asyncpg.Pool) -> None:
        self._pool = connection_pool
    
    async def save(self, finding: Finding) -> None:
        """Save a finding to PostgreSQL with vector embedding"""
        
        embedding_vector = None
        embedding_model = None
        embedding_dimension = None
        
        if finding.embedding:
            embedding_vector = list(finding.embedding.vector)
            embedding_model = finding.embedding.model
            embedding_dimension = finding.embedding.dimension
        
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO findings (id, campaign_id, agent_id, title, description,
                                    severity, created_at, status, evidence, updated_at,
                                    embedding, embedding_model, embedding_dimension)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    severity = EXCLUDED.severity,
                    status = EXCLUDED.status,
                    evidence = EXCLUDED.evidence,
                    updated_at = EXCLUDED.updated_at,
                    embedding = EXCLUDED.embedding,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_dimension = EXCLUDED.embedding_dimension
            """,
                str(finding.id),
                str(finding.campaign_id),
                str(finding.agent_id),
                finding.title,
                finding.description,
                finding.severity.value,
                finding.created_at,
                finding.status.value,
                json.dumps(finding.evidence),
                finding.updated_at,
                embedding_vector,
                embedding_model,
                embedding_dimension
            )
        
        log.info("Finding saved", finding_id=str(finding.id), title=finding.title)
    
    async def find_by_id(self, finding_id: FindingId) -> Optional[Finding]:
        """Find finding by ID"""
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM findings WHERE id = $1",
                str(finding_id)
            )
            
            if not row:
                return None
            
            return self._row_to_finding(row)
    
    async def find_by_campaign(self, campaign_id: CampaignId) -> List[Finding]:
        """Find findings for a campaign"""
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM findings WHERE campaign_id = $1 ORDER BY created_at DESC",
                str(campaign_id)
            )
            
            return [self._row_to_finding(row) for row in rows]
    
    async def find_by_severity(self, severity: str) -> List[Finding]:
        """Find findings by severity"""
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM findings WHERE severity = $1 ORDER BY created_at DESC",
                severity
            )
            
            return [self._row_to_finding(row) for row in rows]
    
    async def find_similar(
        self, 
        embedding: Embedding, 
        threshold: float = 0.8,
        limit: int = 10
    ) -> List[Tuple[Finding, float]]:
        """Find similar findings using vector similarity (pgvector)"""
        
        if not embedding.vector:
            return []
        
        async with self._pool.acquire() as conn:
            # Use pgvector cosine similarity
            rows = await conn.fetch("""
                SELECT *, 1 - (embedding <=> $1::vector) AS similarity
                FROM findings 
                WHERE embedding IS NOT NULL
                AND 1 - (embedding <=> $1::vector) >= $2
                ORDER BY similarity DESC
                LIMIT $3
            """, 
                list(embedding.vector),
                threshold,
                limit
            )
            
            results = []
            for row in rows:
                finding = self._row_to_finding(row)
                similarity = float(row['similarity'])
                results.append((finding, similarity))
            
            return results
    
    async def delete(self, finding: Finding) -> None:
        """Delete a finding"""
        
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM findings WHERE id = $1", str(finding.id))
        
        log.info("Finding deleted", finding_id=str(finding.id))
    
    def _row_to_finding(self, row: asyncpg.Record) -> Finding:
        """Convert database row to Finding entity"""
        
        embedding = None
        if row['embedding'] and row['embedding_model']:
            embedding = Embedding.from_list(
                vector=row['embedding'],
                model=row['embedding_model']
            )
        
        return Finding(
            id_=FindingId.from_string(row['id']),
            campaign_id=CampaignId.from_string(row['campaign_id']),
            agent_id=AgentId.from_string(row['agent_id']),
            title=row['title'],
            description=row['description'],
            severity=Severity(row['severity']),
            created_at=row['created_at'],
            status=FindingStatus(row['status']),
            evidence=json.loads(row['evidence']) if row['evidence'] else {},
            embedding=embedding
        )


class Neo4jKnowledgeAtomRepository(KnowledgeAtomRepository):
    """Neo4j implementation of KnowledgeAtomRepository for graph relationships"""
    
    def __init__(self, driver: AsyncGraphDatabase) -> None:
        self._driver = driver
    
    async def save(self, atom: KnowledgeAtom) -> None:
        """Save a knowledge atom to Neo4j"""
        
        embedding_data = None
        if atom.embedding:
            embedding_data = {
                'vector': list(atom.embedding.vector),
                'model': atom.embedding.model,
                'dimension': atom.embedding.dimension
            }
        
        async with self._driver.session() as session:
            await session.run("""
                MERGE (a:KnowledgeAtom {id: $id})
                SET a.content = $content,
                    a.atom_type = $atom_type,
                    a.confidence = $confidence,
                    a.created_at = $created_at,
                    a.updated_at = $updated_at,
                    a.tags = $tags,
                    a.source = $source,
                    a.embedding = $embedding
            """,
                id=str(atom.id),
                content=atom.content,
                atom_type=atom.atom_type.value,
                confidence=atom.confidence,
                created_at=atom.created_at.isoformat(),
                updated_at=atom.updated_at.isoformat(),
                tags=list(atom.tags),
                source=atom.source,
                embedding=embedding_data
            )
        
        log.info("Knowledge atom saved", atom_id=str(atom.id), atom_type=atom.atom_type.value)
    
    async def find_by_id(self, atom_id: AtomId) -> Optional[KnowledgeAtom]:
        """Find knowledge atom by ID"""
        
        async with self._driver.session() as session:
            result = await session.run(
                "MATCH (a:KnowledgeAtom {id: $id}) RETURN a",
                id=str(atom_id)
            )
            
            record = await result.single()
            if not record:
                return None
            
            return self._record_to_atom(record['a'])
    
    async def find_by_type(self, atom_type: str) -> List[KnowledgeAtom]:
        """Find atoms by type"""
        
        async with self._driver.session() as session:
            result = await session.run(
                "MATCH (a:KnowledgeAtom {atom_type: $atom_type}) RETURN a ORDER BY a.created_at DESC",
                atom_type=atom_type
            )
            
            atoms = []
            async for record in result:
                atom = self._record_to_atom(record['a'])
                atoms.append(atom)
            
            return atoms
    
    async def find_by_tags(self, tags: List[str]) -> List[KnowledgeAtom]:
        """Find atoms with specific tags"""
        
        async with self._driver.session() as session:
            result = await session.run("""
                MATCH (a:KnowledgeAtom)
                WHERE ANY(tag IN $tags WHERE tag IN a.tags)
                RETURN a ORDER BY a.confidence DESC
            """, tags=tags)
            
            atoms = []
            async for record in result:
                atom = self._record_to_atom(record['a'])
                atoms.append(atom)
            
            return atoms
    
    async def find_similar(
        self,
        embedding: Embedding,
        threshold: float = 0.7,
        limit: int = 5
    ) -> List[Tuple[KnowledgeAtom, float]]:
        """Find similar knowledge atoms using vector similarity"""
        
        # This would require a vector similarity plugin for Neo4j
        # For now, we'll fetch all atoms and compute similarity in memory
        # In production, would use Neo4j vector search capabilities
        
        async with self._driver.session() as session:
            result = await session.run(
                "MATCH (a:KnowledgeAtom) WHERE a.embedding IS NOT NULL RETURN a"
            )
            
            similar_atoms = []
            async for record in result:
                atom = self._record_to_atom(record['a'])
                if atom.embedding:
                    similarity = embedding.similarity_cosine(atom.embedding)
                    
                    if similarity >= threshold:
                        similar_atoms.append((atom, similarity))
            
            # Sort by similarity and limit results
            similar_atoms.sort(key=lambda x: x[1], reverse=True)
            return similar_atoms[:limit]
    
    async def delete(self, atom: KnowledgeAtom) -> None:
        """Delete a knowledge atom"""
        
        async with self._driver.session() as session:
            await session.run(
                "MATCH (a:KnowledgeAtom {id: $id}) DELETE a",
                id=str(atom.id)
            )
        
        log.info("Knowledge atom deleted", atom_id=str(atom.id))
    
    def _record_to_atom(self, record: Any) -> KnowledgeAtom:
        """Convert Neo4j record to KnowledgeAtom entity"""
        
        embedding = None
        if record.get('embedding'):
            embedding_data = record['embedding']
            embedding = Embedding.from_list(
                vector=embedding_data['vector'],
                model=embedding_data['model']
            )
        
        atom = KnowledgeAtom(
            id_=AtomId.from_string(record['id']),
            content=record['content'],
            atom_type=AtomType(record['atom_type']),
            confidence=record['confidence'],
            created_at=datetime.fromisoformat(record['created_at']),
            embedding=embedding,
            tags=set(record.get('tags', [])),
            source=record.get('source')
        )
        
        atom.updated_at = datetime.fromisoformat(record['updated_at'])
        
        return atom