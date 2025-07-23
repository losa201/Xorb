#!/usr/bin/env python3

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union
from collections import defaultdict

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, and_, or_, desc, func

from .atom import KnowledgeAtom, AtomType, ConfidenceLevel, Source, ValidationResult
from .sharding import AtomShardManager
from .ml_predictor import ConfidencePredictor
from .models import AtomModel, RelationshipModel, create_tables


class KnowledgeFabric:
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379/0",
                 database_url: str = "sqlite+aiosqlite:///./xorb_knowledge.db"):
        
        self.redis_client = redis.from_url(redis_url)
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
        
        self.shard_manager = AtomShardManager()
        self.confidence_predictor = ConfidencePredictor()
        
        self.logger = logging.getLogger(__name__)
        
        self.hot_cache_ttl = 3600  # 1 hour
        self.warm_cache_ttl = 86400  # 24 hours
        
        self._relationship_cache: Dict[str, Set[str]] = defaultdict(set)

    async def initialize(self):
        await create_tables(self.engine)
        await self.redis_client.ping()
        await self.confidence_predictor.initialize()
        
        self.logger.info("Knowledge Fabric initialized successfully")

    async def add_atom(self, atom: KnowledgeAtom) -> str:
        try:
            await self._store_atom_hot(atom)
            await self._store_atom_warm(atom)
            
            if atom.related_atoms:
                await self._update_relationships(atom)
            
            await self.confidence_predictor.train_on_atom(atom)
            
            self.logger.debug(f"Added atom {atom.id} to knowledge fabric")
            return atom.id
            
        except Exception as e:
            self.logger.error(f"Failed to add atom {atom.id}: {e}")
            raise

    async def get_atom(self, atom_id: str) -> Optional[KnowledgeAtom]:
        atom = await self._get_atom_hot(atom_id)
        if atom:
            atom.record_usage(success=True)
            await self._store_atom_hot(atom)
            return atom
        
        atom = await self._get_atom_warm(atom_id)
        if atom:
            await self._store_atom_hot(atom)
            atom.record_usage(success=True)
            return atom
        
        return None

    async def search_atoms(self, 
                          query: Optional[str] = None,
                          atom_type: Optional[AtomType] = None,
                          tags: Optional[Set[str]] = None,
                          min_confidence: float = 0.0,
                          max_results: int = 100) -> List[KnowledgeAtom]:
        
        search_results = []
        
        # Search hot cache first
        hot_results = await self._search_hot_cache(query, atom_type, tags, min_confidence)
        search_results.extend(hot_results[:max_results])
        
        if len(search_results) < max_results:
            # Search warm storage
            remaining = max_results - len(search_results)
            warm_results = await self._search_warm_storage(query, atom_type, tags, min_confidence, remaining)
            search_results.extend(warm_results)
        
        # Sort by predictive score and confidence
        search_results.sort(key=lambda a: (a.predictive_score, a.confidence), reverse=True)
        return search_results[:max_results]

    async def update_atom(self, atom_id: str, updates: Dict[str, Any], source: Optional[Source] = None) -> bool:
        atom = await self.get_atom(atom_id)
        if not atom:
            return False
        
        atom.update_content(updates, source)
        await self.add_atom(atom)  # This will update existing atom
        
        return True

    async def validate_atom(self, atom_id: str, validation_result: ValidationResult) -> bool:
        atom = await self.get_atom(atom_id)
        if not atom:
            return False
        
        atom.add_validation(validation_result)
        await self.add_atom(atom)
        
        return True

    async def create_relationship(self, atom1_id: str, atom2_id: str, relationship_type: str = "related") -> bool:
        try:
            async with self.async_session() as session:
                relationship = RelationshipModel(
                    source_atom_id=atom1_id,
                    target_atom_id=atom2_id,
                    relationship_type=relationship_type,
                    strength=1.0,
                    created_at=datetime.utcnow()
                )
                session.add(relationship)
                await session.commit()
            
            self._relationship_cache[atom1_id].add(atom2_id)
            self._relationship_cache[atom2_id].add(atom1_id)
            
            self.logger.debug(f"Created relationship between {atom1_id} and {atom2_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create relationship: {e}")
            return False

    async def get_related_atoms(self, atom_id: str, max_depth: int = 2) -> List[KnowledgeAtom]:
        related_ids = await self._get_related_atom_ids(atom_id, max_depth)
        
        related_atoms = []
        for related_id in related_ids:
            atom = await self.get_atom(related_id)
            if atom:
                related_atoms.append(atom)
        
        return related_atoms

    async def analyze_knowledge_gaps(self, atom_type: Optional[AtomType] = None) -> Dict[str, Any]:
        async with self.async_session() as session:
            query = select(AtomModel)
            if atom_type:
                query = query.where(AtomModel.atom_type == atom_type.value)
            
            result = await session.execute(query)
            atoms = result.scalars().all()
        
        total_atoms = len(atoms)
        low_confidence_atoms = len([a for a in atoms if a.confidence < 0.5])
        expired_atoms = len([a for a in atoms if a.expires_at and a.expires_at < datetime.utcnow()])
        
        confidence_distribution = {level.value: 0 for level in ConfidenceLevel}
        type_distribution = {atype.value: 0 for atype in AtomType}
        
        for atom in atoms:
            if atom.confidence <= 0.2:
                confidence_distribution["very_low"] += 1
            elif atom.confidence <= 0.4:
                confidence_distribution["low"] += 1
            elif atom.confidence <= 0.6:
                confidence_distribution["medium"] += 1
            elif atom.confidence <= 0.8:
                confidence_distribution["high"] += 1
            else:
                confidence_distribution["very_high"] += 1
            
            type_distribution[atom.atom_type] += 1
        
        return {
            "total_atoms": total_atoms,
            "low_confidence_count": low_confidence_atoms,
            "expired_count": expired_atoms,
            "confidence_distribution": confidence_distribution,
            "type_distribution": type_distribution,
            "coverage_score": 1.0 - (low_confidence_atoms / max(total_atoms, 1)),
            "freshness_score": 1.0 - (expired_atoms / max(total_atoms, 1))
        }

    async def get_high_value_atoms(self, limit: int = 50) -> List[KnowledgeAtom]:
        async with self.async_session() as session:
            query = select(AtomModel).where(
                and_(
                    AtomModel.confidence >= 0.7,
                    AtomModel.predictive_score >= 0.6,
                    or_(AtomModel.expires_at.is_(None), AtomModel.expires_at > datetime.utcnow())
                )
            ).order_by(desc(AtomModel.predictive_score), desc(AtomModel.confidence)).limit(limit)
            
            result = await session.execute(query)
            atom_models = result.scalars().all()
        
        atoms = []
        for model in atom_models:
            atom = await self._model_to_atom(model)
            atoms.append(atom)
        
        return atoms

    async def cleanup_expired_atoms(self) -> int:
        cleanup_count = 0
        
        async with self.async_session() as session:
            query = select(AtomModel).where(
                and_(
                    AtomModel.expires_at.is_not(None),
                    AtomModel.expires_at < datetime.utcnow()
                )
            )
            result = await session.execute(query)
            expired_models = result.scalars().all()
            
            for model in expired_models:
                await self._remove_atom_hot(model.id)
                await session.delete(model)
                cleanup_count += 1
            
            await session.commit()
        
        if cleanup_count > 0:
            self.logger.info(f"Cleaned up {cleanup_count} expired atoms")
        
        return cleanup_count

    async def _store_atom_hot(self, atom: KnowledgeAtom):
        atom_data = json.dumps(atom.to_dict(), default=str)
        await self.redis_client.setex(f"atom:{atom.id}", self.hot_cache_ttl, atom_data)
        
        for tag in atom.tags:
            await self.redis_client.sadd(f"tag:{tag}", atom.id)
            await self.redis_client.expire(f"tag:{tag}", self.hot_cache_ttl)
        
        await self.redis_client.zadd(f"type:{atom.atom_type.value}", {atom.id: atom.predictive_score})

    async def _get_atom_hot(self, atom_id: str) -> Optional[KnowledgeAtom]:
        atom_data = await self.redis_client.get(f"atom:{atom_id}")
        if atom_data:
            try:
                data = json.loads(atom_data)
                return KnowledgeAtom.from_dict(data)
            except Exception as e:
                self.logger.error(f"Failed to deserialize atom {atom_id} from hot cache: {e}")
        return None

    async def _remove_atom_hot(self, atom_id: str):
        await self.redis_client.delete(f"atom:{atom_id}")

    async def _store_atom_warm(self, atom: KnowledgeAtom):
        try:
            async with self.async_session() as session:
                existing = await session.get(AtomModel, atom.id)
                if existing:
                    await self._update_atom_model(existing, atom)
                else:
                    model = await self._atom_to_model(atom)
                    session.add(model)
                
                await session.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store atom {atom.id} in warm storage: {e}")
            raise

    async def _get_atom_warm(self, atom_id: str) -> Optional[KnowledgeAtom]:
        async with self.async_session() as session:
            model = await session.get(AtomModel, atom_id)
            if model:
                return await self._model_to_atom(model)
        return None

    async def _search_hot_cache(self, query: Optional[str], atom_type: Optional[AtomType], tags: Optional[Set[str]], min_confidence: float) -> List[KnowledgeAtom]:
        results = []
        
        if atom_type:
            atom_ids = await self.redis_client.zrevrange(f"type:{atom_type.value}", 0, -1)
            for atom_id in atom_ids:
                atom = await self._get_atom_hot(atom_id.decode())
                if atom and atom.confidence >= min_confidence:
                    if not tags or tags.intersection(atom.tags):
                        results.append(atom)
        
        return results

    async def _search_warm_storage(self, query: Optional[str], atom_type: Optional[AtomType], tags: Optional[Set[str]], min_confidence: float, limit: int) -> List[KnowledgeAtom]:
        results = []
        
        async with self.async_session() as session:
            sql_query = select(AtomModel).where(AtomModel.confidence >= min_confidence)
            
            if atom_type:
                sql_query = sql_query.where(AtomModel.atom_type == atom_type.value)
            
            sql_query = sql_query.order_by(desc(AtomModel.predictive_score)).limit(limit)
            
            result = await session.execute(sql_query)
            models = result.scalars().all()
            
            for model in models:
                atom = await self._model_to_atom(model)
                if not tags or tags.intersection(atom.tags):
                    results.append(atom)
        
        return results

    async def _update_relationships(self, atom: KnowledgeAtom):
        for related_id in atom.related_atoms:
            await self.create_relationship(atom.id, related_id)

    async def _get_related_atom_ids(self, atom_id: str, max_depth: int) -> Set[str]:
        if max_depth <= 0:
            return set()
        
        if atom_id in self._relationship_cache:
            direct_related = self._relationship_cache[atom_id].copy()
        else:
            async with self.async_session() as session:
                query = select(RelationshipModel).where(
                    or_(
                        RelationshipModel.source_atom_id == atom_id,
                        RelationshipModel.target_atom_id == atom_id
                    )
                )
                result = await session.execute(query)
                relationships = result.scalars().all()
                
                direct_related = set()
                for rel in relationships:
                    if rel.source_atom_id == atom_id:
                        direct_related.add(rel.target_atom_id)
                    else:
                        direct_related.add(rel.source_atom_id)
                
                self._relationship_cache[atom_id] = direct_related
        
        all_related = direct_related.copy()
        if max_depth > 1:
            for related_id in direct_related:
                indirect = await self._get_related_atom_ids(related_id, max_depth - 1)
                all_related.update(indirect)
        
        return all_related

    async def _atom_to_model(self, atom: KnowledgeAtom) -> AtomModel:
        return AtomModel(
            id=atom.id,
            atom_type=atom.atom_type.value,
            title=atom.title,
            content=json.dumps(atom.content, default=str),
            confidence=atom.confidence,
            predictive_score=atom.predictive_score,
            tags=json.dumps(list(atom.tags)),
            sources=json.dumps([s.to_dict() if hasattr(s, 'to_dict') else {
                'name': s.name, 'type': s.type, 'version': s.version,
                'timestamp': s.timestamp.isoformat(), 'reliability_score': s.reliability_score,
                'metadata': s.metadata
            } for s in atom.sources], default=str),
            created_at=atom.created_at,
            updated_at=atom.updated_at,
            expires_at=atom.expires_at,
            usage_count=atom.usage_count,
            success_rate=atom.success_rate
        )

    async def _model_to_atom(self, model: AtomModel) -> KnowledgeAtom:
        sources = []
        try:
            source_data = json.loads(model.sources)
            for s_data in source_data:
                source = Source(
                    name=s_data['name'],
                    type=s_data['type'],
                    version=s_data.get('version'),
                    timestamp=datetime.fromisoformat(s_data['timestamp']),
                    reliability_score=s_data.get('reliability_score', 0.5),
                    metadata=s_data.get('metadata', {})
                )
                sources.append(source)
        except:
            pass
        
        return KnowledgeAtom(
            id=model.id,
            atom_type=AtomType(model.atom_type),
            title=model.title,
            content=json.loads(model.content),
            confidence=model.confidence,
            predictive_score=model.predictive_score,
            tags=set(json.loads(model.tags)),
            sources=sources,
            created_at=model.created_at,
            updated_at=model.updated_at,
            expires_at=model.expires_at,
            usage_count=model.usage_count,
            success_rate=model.success_rate
        )

    async def _update_atom_model(self, model: AtomModel, atom: KnowledgeAtom):
        model.title = atom.title
        model.content = json.dumps(atom.content, default=str)
        model.confidence = atom.confidence
        model.predictive_score = atom.predictive_score
        model.tags = json.dumps(list(atom.tags))
        model.sources = json.dumps([{
            'name': s.name, 'type': s.type, 'version': s.version,
            'timestamp': s.timestamp.isoformat(), 'reliability_score': s.reliability_score,
            'metadata': s.metadata
        } for s in atom.sources], default=str)
        model.updated_at = atom.updated_at
        model.expires_at = atom.expires_at
        model.usage_count = atom.usage_count
        model.success_rate = atom.success_rate

    async def shutdown(self):
        await self.redis_client.close()
        await self.engine.dispose()
        self.logger.info("Knowledge Fabric shutdown complete")