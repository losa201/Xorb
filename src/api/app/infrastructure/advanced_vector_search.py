"""Advanced Vector Search Engine with AI/ML Enhancements"""
import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_async_session, get_database_connection
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    EXACT_MATCH = "exact_match"
    SEMANTIC_SEARCH = "semantic_search" 
    HYBRID_SEARCH = "hybrid_search"
    GRAPH_SEARCH = "graph_search"
    CONTEXTUAL_SEARCH = "contextual_search"


class QueryType(Enum):
    THREAT_INTELLIGENCE = "threat_intelligence"
    EVIDENCE_ANALYSIS = "evidence_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"
    ANOMALY_DETECTION = "anomaly_detection"
    SIMILARITY_CLUSTERING = "similarity_clustering"


@dataclass
class SearchQuery:
    text: Optional[str] = None
    vector: Optional[List[float]] = None
    filters: Dict[str, Any] = None
    strategy: SearchStrategy = SearchStrategy.SEMANTIC_SEARCH
    query_type: QueryType = QueryType.THREAT_INTELLIGENCE
    limit: int = 10
    similarity_threshold: float = 0.7
    boost_recent: bool = True
    include_metadata: bool = True
    tenant_id: Optional[UUID] = None
    

@dataclass
class SearchResult:
    id: UUID
    content: str
    similarity_score: float
    metadata: Dict[str, Any]
    source_type: str
    source_id: UUID
    embedding_model: str
    created_at: datetime
    relevance_explanation: Optional[str] = None


@dataclass
class EnhancedSearchResponse:
    query: SearchQuery
    results: List[SearchResult]
    total_found: int
    search_time_ms: float
    strategy_used: SearchStrategy
    query_analysis: Dict[str, Any]
    suggestions: List[str]
    related_concepts: List[str]


class AdvancedVectorSearchEngine:
    """Enhanced vector search with AI-powered query optimization"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.query_cache = {}  # LRU cache for frequent queries
        self.search_analytics = {}
        self.concept_graphs = {}  # For graph-based search
        
    async def search(self, query: SearchQuery) -> EnhancedSearchResponse:
        """Execute advanced search with strategy optimization"""
        start_time = datetime.utcnow()
        
        try:
            # Analyze query to select optimal strategy
            optimized_query = await self._optimize_query(query)
            
            # Execute search based on strategy
            results = await self._execute_search_strategy(optimized_query)
            
            # Post-process results
            enhanced_results = await self._enhance_results(results, optimized_query)
            
            # Generate search analytics
            search_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            query_analysis = await self._analyze_query_performance(optimized_query, enhanced_results)
            
            # Generate suggestions and related concepts
            suggestions = await self._generate_suggestions(optimized_query, enhanced_results)
            related_concepts = await self._find_related_concepts(optimized_query)
            
            return EnhancedSearchResponse(
                query=optimized_query,
                results=enhanced_results,
                total_found=len(enhanced_results),
                search_time_ms=search_time,
                strategy_used=optimized_query.strategy,
                query_analysis=query_analysis,
                suggestions=suggestions,
                related_concepts=related_concepts
            )
            
        except Exception as e:
            logger.error(f"Advanced search failed: {e}")
            # Fallback to basic search
            return await self._fallback_search(query)
    
    async def _optimize_query(self, query: SearchQuery) -> SearchQuery:
        """Optimize query based on content analysis and historical performance"""
        optimized = SearchQuery(**asdict(query))
        
        if query.text:
            # Analyze query text for optimal strategy selection
            query_features = await self._analyze_query_text(query.text)
            
            # Select strategy based on query characteristics
            if query_features.get('has_technical_terms', False):
                optimized.strategy = SearchStrategy.EXACT_MATCH
            elif query_features.get('is_conceptual', False):
                optimized.strategy = SearchStrategy.SEMANTIC_SEARCH
            elif query_features.get('has_relationships', False):
                optimized.strategy = SearchStrategy.GRAPH_SEARCH
            else:
                optimized.strategy = SearchStrategy.HYBRID_SEARCH
                
            # Adjust similarity threshold based on query complexity
            if query_features.get('complexity_score', 0.5) > 0.8:
                optimized.similarity_threshold = max(0.6, query.similarity_threshold - 0.1)
                
        return optimized
    
    async def _execute_search_strategy(self, query: SearchQuery) -> List[Dict[str, Any]]:
        """Execute search based on selected strategy"""
        if query.strategy == SearchStrategy.EXACT_MATCH:
            return await self._exact_match_search(query)
        elif query.strategy == SearchStrategy.SEMANTIC_SEARCH:
            return await self._semantic_search(query)
        elif query.strategy == SearchStrategy.HYBRID_SEARCH:
            return await self._hybrid_search(query)
        elif query.strategy == SearchStrategy.GRAPH_SEARCH:
            return await self._graph_search(query)
        elif query.strategy == SearchStrategy.CONTEXTUAL_SEARCH:
            return await self._contextual_search(query)
        else:
            return await self._semantic_search(query)  # Default fallback
    
    async def _exact_match_search(self, query: SearchQuery) -> List[Dict[str, Any]]:
        """Exact keyword matching with advanced scoring"""
        if not query.text:
            return []
        
        # Use full-text search with ranking
        search_sql = """
        SELECT 
            v.id, v.source_type, v.source_id, v.content_hash,
            v.embedding_model, v.metadata, v.created_at,
            ts_rank_cd(to_tsvector('english', coalesce(v.metadata->>'content', '')), 
                      plainto_tsquery('english', :search_text)) as rank_score,
            1.0 as similarity
        FROM embedding_vectors v
        WHERE v.tenant_id = :tenant_id
        AND to_tsvector('english', coalesce(v.metadata->>'content', '')) @@ 
            plainto_tsquery('english', :search_text)
        ORDER BY rank_score DESC, v.created_at DESC
        LIMIT :limit
        """
        
        params = {
            "search_text": query.text,
            "tenant_id": query.tenant_id,
            "limit": query.limit
        }
        
        results = []
        try:
            async with get_database_connection() as conn:
                await conn.execute("SELECT set_config('app.tenant_id', $1, false)", str(query.tenant_id))
                rows = await conn.fetch(search_sql, **params)
                
                for row in rows:
                    results.append({
                        "id": row["id"],
                        "source_type": row["source_type"],
                        "source_id": row["source_id"],
                        "content_hash": row["content_hash"],
                        "embedding_model": row["embedding_model"],
                        "metadata": row["metadata"],
                        "similarity": float(row["similarity"]),
                        "created_at": row["created_at"]
                    })
                    
        except Exception as e:
            logger.error(f"Exact match search failed: {e}")
        
        return results
    
    async def _semantic_search(self, query: SearchQuery) -> List[Dict[str, Any]]:
        """Enhanced semantic search with query expansion"""
        if query.vector:
            query_vector = query.vector
        elif query.text:
            # Generate embedding for text query
            query_vector = await self._generate_query_embedding(query.text)
        else:
            return []
        
        # Enhanced vector search with additional scoring factors
        enhanced_params = {
            "query_vector": query_vector,
            "tenant_id": query.tenant_id,
            "similarity_threshold": query.similarity_threshold,
            "limit": query.limit * 2  # Get more results for re-ranking
        }
        
        # Add temporal boost if requested
        time_boost_sql = ""
        if query.boost_recent:
            time_boost_sql = """
            * (1 + GREATEST(0, 1 - EXTRACT(EPOCH FROM (NOW() - v.created_at)) / 2592000)) -- 30 days boost
            """
        
        search_sql = f"""
        WITH vector_search AS (
            SELECT 
                v.id, v.source_type, v.source_id, v.content_hash,
                v.embedding_model, v.metadata, v.created_at,
                1 - (v.embedding <=> :query_vector::vector) as base_similarity,
                -- Metadata boost based on source type and quality
                CASE 
                    WHEN v.source_type = 'threat_intelligence' THEN 1.2
                    WHEN v.source_type = 'evidence' THEN 1.1
                    ELSE 1.0
                END as source_boost,
                -- Content length normalization
                CASE 
                    WHEN length(coalesce(v.metadata->>'content', '')) > 1000 THEN 1.1
                    WHEN length(coalesce(v.metadata->>'content', '')) < 100 THEN 0.9
                    ELSE 1.0
                END as content_boost
            FROM embedding_vectors v
            WHERE v.tenant_id = :tenant_id
            AND 1 - (v.embedding <=> :query_vector::vector) >= :similarity_threshold
        )
        SELECT *,
            base_similarity * source_boost * content_boost {time_boost_sql} as final_similarity
        FROM vector_search
        ORDER BY final_similarity DESC
        LIMIT :limit
        """
        
        results = []
        try:
            async with get_database_connection() as conn:
                await conn.execute("SELECT set_config('app.tenant_id', $1, false)", str(query.tenant_id))
                rows = await conn.fetch(search_sql, **enhanced_params)
                
                for row in rows:
                    results.append({
                        "id": row["id"],
                        "source_type": row["source_type"], 
                        "source_id": row["source_id"],
                        "content_hash": row["content_hash"],
                        "embedding_model": row["embedding_model"],
                        "metadata": row["metadata"],
                        "similarity": float(row["final_similarity"]),
                        "created_at": row["created_at"]
                    })
                    
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
        
        return results[:query.limit]  # Return only requested limit after re-ranking
    
    async def _hybrid_search(self, query: SearchQuery) -> List[Dict[str, Any]]:
        """Combine exact match and semantic search with intelligent weighting"""
        if not query.text:
            return await self._semantic_search(query)
        
        # Execute both search strategies
        exact_results = await self._exact_match_search(query)
        semantic_results = await self._semantic_search(query)
        
        # Merge and re-rank results
        combined_results = {}
        
        # Add exact match results with keyword boost
        for result in exact_results:
            result_id = result["id"]
            combined_results[result_id] = result.copy()
            combined_results[result_id]["keyword_score"] = result.get("similarity", 0.0)
            combined_results[result_id]["semantic_score"] = 0.0
            
        # Add semantic results
        for result in semantic_results:
            result_id = result["id"]
            if result_id in combined_results:
                combined_results[result_id]["semantic_score"] = result["similarity"]
            else:
                combined_results[result_id] = result.copy()
                combined_results[result_id]["keyword_score"] = 0.0
                combined_results[result_id]["semantic_score"] = result["similarity"]
        
        # Calculate hybrid score (weighted combination)
        for result in combined_results.values():
            keyword_weight = 0.4
            semantic_weight = 0.6
            
            result["similarity"] = (
                keyword_weight * result["keyword_score"] + 
                semantic_weight * result["semantic_score"]
            )
        
        # Sort by hybrid score and return top results
        sorted_results = sorted(
            combined_results.values(), 
            key=lambda x: x["similarity"], 
            reverse=True
        )
        
        return sorted_results[:query.limit]
    
    async def _graph_search(self, query: SearchQuery) -> List[Dict[str, Any]]:
        """Graph-based search using concept relationships"""
        # Start with semantic search as base
        base_results = await self._semantic_search(query)
        
        if not base_results:
            return base_results
            
        # Expand search using concept graphs
        expanded_results = base_results.copy()
        
        # Find related concepts for top results
        for result in base_results[:3]:  # Expand top 3 results
            related_vectors = await self._find_related_vectors(
                result["id"], 
                query.tenant_id,
                max_depth=2
            )
            
            for related in related_vectors:
                if related["id"] not in [r["id"] for r in expanded_results]:
                    # Add relationship score
                    related["similarity"] = related["similarity"] * 0.8  # Discount for indirect match
                    expanded_results.append(related)
        
        # Re-sort and limit
        sorted_results = sorted(
            expanded_results,
            key=lambda x: x["similarity"],
            reverse=True
        )
        
        return sorted_results[:query.limit]
    
    async def _contextual_search(self, query: SearchQuery) -> List[Dict[str, Any]]:
        """Context-aware search using query history and user patterns"""
        # Get user's recent search context if available
        user_context = await self._get_user_search_context(query.tenant_id)
        
        # Enhance query with contextual information
        enhanced_query = query
        if user_context and query.text:
            # Add context keywords to search
            context_terms = user_context.get("frequent_terms", [])
            if context_terms:
                enhanced_text = f"{query.text} {' '.join(context_terms[:3])}"
                enhanced_query.text = enhanced_text
        
        # Execute semantic search with enhanced query
        results = await self._semantic_search(enhanced_query)
        
        # Re-rank based on user's interaction history
        if user_context:
            preferred_types = user_context.get("preferred_source_types", [])
            for result in results:
                if result["source_type"] in preferred_types:
                    result["similarity"] *= 1.2  # Boost preferred types
        
        return results
    
    async def _enhance_results(self, results: List[Dict[str, Any]], query: SearchQuery) -> List[SearchResult]:
        """Convert raw results to enhanced SearchResult objects"""
        enhanced_results = []
        
        for result in results:
            # Extract content from metadata
            content = result.get("metadata", {}).get("content", "")
            
            # Generate relevance explanation
            explanation = await self._generate_relevance_explanation(result, query)
            
            search_result = SearchResult(
                id=result["id"],
                content=content,
                similarity_score=result["similarity"],
                metadata=result.get("metadata", {}),
                source_type=result["source_type"],
                source_id=result["source_id"],
                embedding_model=result["embedding_model"],
                created_at=result["created_at"],
                relevance_explanation=explanation
            )
            
            enhanced_results.append(search_result)
        
        return enhanced_results
    
    async def _analyze_query_text(self, text: str) -> Dict[str, Any]:
        """Analyze query text to determine characteristics"""
        features = {
            "has_technical_terms": False,
            "is_conceptual": False,
            "has_relationships": False,
            "complexity_score": 0.5,
            "intent": "general"
        }
        
        text_lower = text.lower()
        
        # Technical terms detection
        technical_terms = ['ip', 'hash', 'malware', 'vulnerability', 'exploit', 'cve', 'ioc']
        features["has_technical_terms"] = any(term in text_lower for term in technical_terms)
        
        # Conceptual query detection
        conceptual_words = ['how', 'what', 'why', 'explain', 'understand', 'analysis']
        features["is_conceptual"] = any(word in text_lower for word in conceptual_words)
        
        # Relationship detection
        relationship_words = ['related', 'connected', 'similar', 'like', 'associated']
        features["has_relationships"] = any(word in text_lower for word in relationship_words)
        
        # Complexity scoring
        word_count = len(text.split())
        unique_words = len(set(text.lower().split()))
        features["complexity_score"] = min(1.0, (word_count + unique_words) / 20)
        
        # Intent detection
        if features["has_technical_terms"]:
            features["intent"] = "technical_lookup"
        elif features["is_conceptual"]:
            features["intent"] = "conceptual_understanding"
        elif features["has_relationships"]:
            features["intent"] = "relationship_discovery"
        
        return features
    
    async def _generate_query_embedding(self, text: str) -> List[float]:
        """Generate embedding for query text"""
        # This would integrate with the embedding service
        # For now, return a mock embedding
        import random
        return [random.random() for _ in range(1536)]  # OpenAI embedding dimension
    
    async def _find_related_vectors(self, vector_id: UUID, tenant_id: UUID, max_depth: int = 2) -> List[Dict]:
        """Find vectors related to a given vector through graph relationships"""
        related = []
        
        try:
            # Get the source vector
            source_vector_sql = """
            SELECT embedding, metadata, source_type 
            FROM embedding_vectors 
            WHERE id = :vector_id AND tenant_id = :tenant_id
            """
            
            async with get_database_connection() as conn:
                await conn.execute("SELECT set_config('app.tenant_id', $1, false)", str(tenant_id))
                source_result = await conn.fetchrow(source_vector_sql, vector_id=vector_id, tenant_id=tenant_id)
                
                if not source_result:
                    return related
                
                source_embedding = source_result["embedding"]
                source_metadata = source_result["metadata"] or {}
                
                # Find similar vectors using cosine similarity
                similar_vectors_sql = """
                WITH similar_vectors AS (
                    SELECT 
                        id, embedding, metadata, source_type, source_id,
                        1 - (embedding <=> :source_embedding::vector) as similarity,
                        created_at
                    FROM embedding_vectors
                    WHERE tenant_id = :tenant_id 
                    AND id != :vector_id
                    AND 1 - (embedding <=> :source_embedding::vector) >= 0.7
                    ORDER BY similarity DESC
                    LIMIT 10
                )
                SELECT * FROM similar_vectors
                """
                
                similar_rows = await conn.fetch(
                    similar_vectors_sql,
                    source_embedding=source_embedding,
                    vector_id=vector_id,
                    tenant_id=tenant_id
                )
                
                for row in similar_rows:
                    related_vector = {
                        "id": row["id"],
                        "similarity": float(row["similarity"]),
                        "source_type": row["source_type"],
                        "source_id": row["source_id"],
                        "metadata": row["metadata"] or {},
                        "created_at": row["created_at"],
                        "relationship_type": "semantic_similarity"
                    }
                    related.append(related_vector)
                
                # If we have depth > 1, find vectors related to the similar vectors
                if max_depth > 1 and related:
                    for related_vector in related[:3]:  # Limit to top 3 to prevent explosion
                        second_level = await self._find_related_vectors(
                            related_vector["id"], 
                            tenant_id, 
                            max_depth=1
                        )
                        
                        # Add second level relationships with reduced similarity
                        for second_rel in second_level[:2]:  # Limit second level
                            second_rel["similarity"] *= 0.8  # Reduce similarity for indirect relations
                            second_rel["relationship_type"] = "indirect_similarity"
                            
                            # Avoid duplicates
                            if not any(r["id"] == second_rel["id"] for r in related):
                                related.append(second_rel)
                
                # Sort by similarity and limit results
                related.sort(key=lambda x: x["similarity"], reverse=True)
                return related[:20]  # Return top 20 related vectors
                
        except Exception as e:
            logger.error(f"Failed to find related vectors: {e}")
            return []
    
    async def _get_user_search_context(self, tenant_id: UUID) -> Dict[str, Any]:
        """Get user's search context and preferences"""
        # This would analyze user's search history and preferences
        return {
            "frequent_terms": [],
            "preferred_source_types": ["threat_intelligence", "evidence"],
            "typical_search_patterns": []
        }
    
    async def _generate_relevance_explanation(self, result: Dict[str, Any], query: SearchQuery) -> str:
        """Generate explanation for why this result is relevant"""
        similarity = result.get("similarity", 0.0)
        source_type = result.get("source_type", "unknown")
        
        if similarity > 0.9:
            return f"Highly relevant {source_type} with {similarity:.1%} similarity"
        elif similarity > 0.8:
            return f"Very relevant {source_type} with {similarity:.1%} similarity"
        elif similarity > 0.7:
            return f"Relevant {source_type} with {similarity:.1%} similarity"
        else:
            return f"Potentially relevant {source_type} with {similarity:.1%} similarity"
    
    async def _analyze_query_performance(self, query: SearchQuery, results: List[SearchResult]) -> Dict[str, Any]:
        """Analyze query performance for optimization"""
        return {
            "strategy_effectiveness": len(results) / query.limit if query.limit > 0 else 0,
            "avg_similarity": np.mean([r.similarity_score for r in results]) if results else 0,
            "result_diversity": len(set(r.source_type for r in results)),
            "query_complexity": len(query.text.split()) if query.text else 0
        }
    
    async def _generate_suggestions(self, query: SearchQuery, results: List[SearchResult]) -> List[str]:
        """Generate search suggestions based on results"""
        suggestions = []
        
        if not results:
            suggestions.append("Try broader search terms")
            suggestions.append("Check spelling and try synonyms")
        elif len(results) < query.limit // 2:
            suggestions.append("Try using fewer specific terms")
            suggestions.append("Consider related concepts")
        
        # Add suggestions based on successful result patterns
        if results:
            common_types = {}
            for result in results:
                source_type = result.source_type
                common_types[source_type] = common_types.get(source_type, 0) + 1
            
            most_common = max(common_types, key=common_types.get)
            suggestions.append(f"Explore more {most_common} sources")
        
        return suggestions[:3]  # Limit to top 3 suggestions
    
    async def _find_related_concepts(self, query: SearchQuery) -> List[str]:
        """Find concepts related to the query"""
        if not query.text:
            return []
        
        # This would use NLP to find related concepts
        # For now, return mock related concepts
        mock_concepts = {
            "malware": ["virus", "trojan", "ransomware", "botnet"],
            "vulnerability": ["exploit", "cve", "patch", "security flaw"],
            "threat": ["attack", "compromise", "breach", "incident"]
        }
        
        related = []
        for term, concepts in mock_concepts.items():
            if term in query.text.lower():
                related.extend(concepts)
        
        return related[:5]  # Return top 5 related concepts
    
    async def _fallback_search(self, query: SearchQuery) -> EnhancedSearchResponse:
        """Fallback to basic search when advanced search fails"""
        try:
            if query.vector:
                basic_results = await self.vector_store.search_similar(
                    query.vector, query.tenant_id, query.limit, 
                    similarity_threshold=query.similarity_threshold
                )
            else:
                basic_results = []
            
            enhanced_results = []
            for result in basic_results:
                enhanced_results.append(SearchResult(
                    id=result["id"],
                    content=result.get("metadata", {}).get("content", ""),
                    similarity_score=result["similarity"],
                    metadata=result.get("metadata", {}),
                    source_type=result["source_type"],
                    source_id=result["source_id"],
                    embedding_model=result["embedding_model"],
                    created_at=datetime.utcnow(),  # Fallback timestamp
                    relevance_explanation="Basic similarity match"
                ))
            
            return EnhancedSearchResponse(
                query=query,
                results=enhanced_results,
                total_found=len(enhanced_results),
                search_time_ms=0,
                strategy_used=SearchStrategy.SEMANTIC_SEARCH,
                query_analysis={"fallback": True},
                suggestions=["Try a different search approach"],
                related_concepts=[]
            )
            
        except Exception as e:
            logger.error(f"Fallback search also failed: {e}")
            return EnhancedSearchResponse(
                query=query,
                results=[],
                total_found=0,
                search_time_ms=0,
                strategy_used=SearchStrategy.SEMANTIC_SEARCH,
                query_analysis={"error": str(e)},
                suggestions=["Please try again with different terms"],
                related_concepts=[]
            )


# Global search engine instance
_search_engine: Optional[AdvancedVectorSearchEngine] = None


async def get_advanced_search_engine() -> AdvancedVectorSearchEngine:
    """Get global advanced search engine instance"""
    global _search_engine
    
    if _search_engine is None:
        from .vector_store import get_vector_store
        vector_store = get_vector_store()
        _search_engine = AdvancedVectorSearchEngine(vector_store)
    
    return _search_engine