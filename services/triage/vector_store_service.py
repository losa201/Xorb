#!/usr/bin/env python3
"""
Xorb Vector Store Service for Intelligent Triage Deduplication
MiniLM embeddings + FAISS vector search + GPT-4o reranking
Phase 5.1 - Smart Triage Optimization
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import faiss
import asyncpg
import aioredis
import openai
from sentence_transformers import SentenceTransformer
from prometheus_client import Counter, Histogram, Gauge
import structlog

# Configure structured logging
logger = structlog.get_logger("xorb.vector_store")

# Prometheus metrics for Phase 5.1 requirements
triage_dedupe_saved_tokens_total = Counter(
    'triage_dedupe_saved_tokens_total', 
    'Total tokens saved through deduplication'
)
triage_false_positive_score = Gauge(
    'triage_false_positive_score', 
    'Current false positive detection score'
)
embedding_generation_duration = Histogram(
    'embedding_generation_duration_seconds', 
    'Time to generate embeddings'
)
faiss_similarity_search_duration = Histogram(
    'faiss_similarity_search_duration_seconds', 
    'FAISS similarity search duration'
)
vector_store_cache_hits = Counter(
    'vector_store_cache_hits_total', 
    'Cache hits in vector store operations'
)
gpt_rerank_operations = Counter(
    'gpt_rerank_operations_total', 
    'GPT reranking operations', 
    ['result']
)

@dataclass
class VulnerabilityVector:
    """Vulnerability vector representation for similarity search"""
    id: str
    title: str
    description: str
    severity: str
    target: str
    embedding: np.ndarray
    metadata: Dict
    created_at: datetime
    fingerprint: str

@dataclass
class SimilarityResult:
    """Similarity search result"""
    vulnerability_id: str
    similarity_score: float
    title: str
    description: str
    severity: str
    target: str
    created_at: datetime
    metadata: Dict

@dataclass
class DeduplicationResult:
    """Deduplication analysis result"""
    is_duplicate: bool
    confidence: float
    duplicate_of: Optional[str]
    similar_findings: List[SimilarityResult]
    gpt_analysis: Optional[str]
    tokens_saved: int
    processing_time: float
    reasoning: str

class VectorStoreService:
    """High-performance vector store for vulnerability deduplication"""
    
    def __init__(self):
        self.db_pool = None
        self.redis = None
        self.embedding_model = None
        self.faiss_index = None
        self.openai_client = None
        
        # Configuration
        self.embedding_dim = 384  # MiniLM L6 v2 dimension
        self.similarity_threshold = 0.85
        self.top_k_similar = 10
        self.cache_ttl = 86400 * 7  # 7 days
        
        # In-memory state
        self.vulnerability_vectors: Dict[str, VulnerabilityVector] = {}
        self.id_to_index_mapping: Dict[str, int] = {}
        self.index_to_id_mapping: Dict[int, str] = {}
        self.next_index = 0
        
        # Cache for embeddings
        self.embedding_cache = {}
        
    async def initialize(self):
        """Initialize vector store service"""
        logger.info("Initializing Vector Store Service...")
        
        # Database connection
        database_url = os.getenv("DATABASE_URL", "postgresql://xorb:xorb_secure_2024@postgres:5432/xorb_ptaas")
        self.db_pool = await asyncpg.create_pool(database_url, min_size=5, max_size=20)
        
        # Redis connection
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        self.redis = await aioredis.from_url(redis_url)
        
        # Initialize MiniLM model
        logger.info("Loading MiniLM embedding model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize FAISS index with inner product for cosine similarity
        logger.info("Initializing FAISS index...")
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Initialize OpenAI client
        self.openai_client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create database tables
        await self._create_vector_tables()
        
        # Load existing vectors
        await self._load_existing_vectors()
        
        logger.info(f"Vector Store initialized with {len(self.vulnerability_vectors)} vectors")
        
    async def _create_vector_tables(self):
        """Create database tables for vector storage"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS vulnerability_vectors (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    vulnerability_id VARCHAR(255) UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    severity VARCHAR(50) NOT NULL,
                    target TEXT NOT NULL,
                    embedding BYTEA NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    fingerprint VARCHAR(64) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_vulnerability_vectors_vuln_id 
                ON vulnerability_vectors(vulnerability_id);
                
                CREATE INDEX IF NOT EXISTS idx_vulnerability_vectors_fingerprint 
                ON vulnerability_vectors(fingerprint);
                
                CREATE INDEX IF NOT EXISTS idx_vulnerability_vectors_severity 
                ON vulnerability_vectors(severity);
                
                CREATE INDEX IF NOT EXISTS idx_vulnerability_vectors_created_at 
                ON vulnerability_vectors(created_at);
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS deduplication_results (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    vulnerability_id VARCHAR(255) NOT NULL,
                    is_duplicate BOOLEAN NOT NULL,
                    confidence FLOAT NOT NULL,
                    duplicate_of VARCHAR(255),
                    similar_findings_count INTEGER DEFAULT 0,
                    gpt_analysis TEXT,
                    tokens_saved INTEGER DEFAULT 0,
                    processing_time FLOAT NOT NULL,
                    reasoning TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_deduplication_results_vuln_id 
                ON deduplication_results(vulnerability_id);
                
                CREATE INDEX IF NOT EXISTS idx_deduplication_results_is_duplicate 
                ON deduplication_results(is_duplicate);
                
                CREATE INDEX IF NOT EXISTS idx_deduplication_results_created_at 
                ON deduplication_results(created_at);
            """)
    
    async def _load_existing_vectors(self):
        """Load existing vectors from database and rebuild FAISS index"""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT vulnerability_id, title, description, severity, target, 
                       embedding, metadata, fingerprint, created_at
                FROM vulnerability_vectors
                ORDER BY created_at DESC
                LIMIT 50000
            """)
            
            if not rows:
                logger.info("No existing vectors found")
                return
                
            logger.info(f"Loading {len(rows)} existing vectors...")
            
            embeddings_list = []
            
            for row in rows:
                # Deserialize embedding
                embedding = pickle.loads(row['embedding'])
                
                vuln_vector = VulnerabilityVector(
                    id=row['vulnerability_id'],
                    title=row['title'],
                    description=row['description'],
                    severity=row['severity'],
                    target=row['target'],
                    embedding=embedding,
                    metadata=row['metadata'] or {},
                    created_at=row['created_at'],
                    fingerprint=row['fingerprint']
                )
                
                # Add to in-memory storage
                self.vulnerability_vectors[row['vulnerability_id']] = vuln_vector
                self.id_to_index_mapping[row['vulnerability_id']] = self.next_index
                self.index_to_id_mapping[self.next_index] = row['vulnerability_id']
                
                embeddings_list.append(embedding)
                self.next_index += 1
            
            # Build FAISS index
            if embeddings_list:
                embeddings_matrix = np.array(embeddings_list).astype('float32')
                # Normalize for cosine similarity
                faiss.normalize_L2(embeddings_matrix)
                self.faiss_index.add(embeddings_matrix)
                
            logger.info(f"FAISS index built with {len(embeddings_list)} vectors")
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text with caching"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        # Check local cache first
        if text_hash in self.embedding_cache:
            vector_store_cache_hits.inc()
            return self.embedding_cache[text_hash]
        
        # Check Redis cache
        try:
            cached_embedding = await self.redis.get(f"embedding:{text_hash}")
            if cached_embedding:
                embedding = pickle.loads(cached_embedding)
                self.embedding_cache[text_hash] = embedding
                vector_store_cache_hits.inc()
                return embedding
        except Exception as e:
            logger.warning(f"Redis cache error: {e}")
        
        # Generate new embedding
        with embedding_generation_duration.time():
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        
        # Cache the embedding
        self.embedding_cache[text_hash] = embedding
        try:
            await self.redis.setex(
                f"embedding:{text_hash}", 
                self.cache_ttl, 
                pickle.dumps(embedding)
            )
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
        
        return embedding
    
    def _create_vulnerability_fingerprint(self, title: str, description: str, target: str) -> str:
        """Create fingerprint for vulnerability"""
        normalized_data = {
            'title': title.lower().strip(),
            'description': description.lower().strip()[:500],  # First 500 chars
            'target': self._normalize_target(target)
        }
        data_str = json.dumps(normalized_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _normalize_target(self, target: str) -> str:
        """Normalize target for consistent fingerprinting"""
        import re
        target = target.lower()
        # Remove port numbers
        target = re.sub(r':\d+', '', target)
        # Remove protocol
        target = re.sub(r'^https?://', '', target)
        # Remove www
        target = re.sub(r'^www\.', '', target)
        return target
    
    def _create_vulnerability_text(self, title: str, description: str, target: str, severity: str) -> str:
        """Create text representation for embedding"""
        return f"{title} {description} {target} {severity}"
    
    async def add_vulnerability_vector(
        self, 
        vulnerability_id: str,
        title: str,
        description: str,
        severity: str,
        target: str,
        metadata: Dict = None
    ) -> VulnerabilityVector:
        """Add vulnerability vector to store"""
        if metadata is None:
            metadata = {}
            
        # Create fingerprint
        fingerprint = self._create_vulnerability_fingerprint(title, description, target)
        
        # Generate embedding
        vuln_text = self._create_vulnerability_text(title, description, target, severity)
        embedding = await self.generate_embedding(vuln_text)
        
        # Create vulnerability vector
        vuln_vector = VulnerabilityVector(
            id=vulnerability_id,
            title=title,
            description=description,
            severity=severity,
            target=target,
            embedding=embedding,
            metadata=metadata,
            created_at=datetime.utcnow(),
            fingerprint=fingerprint
        )
        
        # Store in database
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO vulnerability_vectors 
                (vulnerability_id, title, description, severity, target, embedding, metadata, fingerprint)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (vulnerability_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    severity = EXCLUDED.severity,
                    target = EXCLUDED.target,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    fingerprint = EXCLUDED.fingerprint,
                    updated_at = NOW()
            """, vulnerability_id, title, description, severity, target, 
                pickle.dumps(embedding), json.dumps(metadata), fingerprint)
        
        # Add to in-memory storage
        self.vulnerability_vectors[vulnerability_id] = vuln_vector
        
        # Add to FAISS index
        if vulnerability_id not in self.id_to_index_mapping:
            index = self.next_index
            self.id_to_index_mapping[vulnerability_id] = index
            self.index_to_id_mapping[index] = vulnerability_id
            self.next_index += 1
        else:
            index = self.id_to_index_mapping[vulnerability_id]
        
        # Normalize and add to FAISS
        embedding_normalized = embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(embedding_normalized)
        
        if index >= self.faiss_index.ntotal:
            self.faiss_index.add(embedding_normalized)
        else:
            # Update existing vector in FAISS (reconstruct index)
            await self._rebuild_faiss_index()
        
        logger.info(f"Added vulnerability vector: {vulnerability_id}")
        return vuln_vector
    
    async def _rebuild_faiss_index(self):
        """Rebuild FAISS index from scratch"""
        logger.info("Rebuilding FAISS index...")
        
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        embeddings_list = []
        
        # Collect all embeddings in order
        for i in range(self.next_index):
            if i in self.index_to_id_mapping:
                vuln_id = self.index_to_id_mapping[i]
                if vuln_id in self.vulnerability_vectors:
                    embeddings_list.append(self.vulnerability_vectors[vuln_id].embedding)
                else:
                    # Fill with zero vector for missing entries
                    embeddings_list.append(np.zeros(self.embedding_dim))
        
        if embeddings_list:
            embeddings_matrix = np.array(embeddings_list).astype('float32')
            faiss.normalize_L2(embeddings_matrix)
            self.faiss_index.add(embeddings_matrix)
        
        logger.info(f"FAISS index rebuilt with {len(embeddings_list)} vectors")
    
    async def find_similar_vulnerabilities(
        self, 
        vulnerability_id: str,
        title: str,
        description: str,
        severity: str,
        target: str,
        k: int = None
    ) -> List[SimilarityResult]:
        """Find similar vulnerabilities using FAISS search"""
        if k is None:
            k = self.top_k_similar
            
        if self.faiss_index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_text = self._create_vulnerability_text(title, description, target, severity)
        query_embedding = await self.generate_embedding(query_text)
        
        with faiss_similarity_search_duration.time():
            # Normalize query embedding
            query_embedding_norm = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_embedding_norm)
            
            # Search FAISS index
            similarities, indices = self.faiss_index.search(query_embedding_norm, k + 1)  # +1 to exclude self
            
            similar_results = []
            for similarity, index in zip(similarities[0], indices[0]):
                if index == -1:  # Invalid index
                    continue
                    
                if index in self.index_to_id_mapping:
                    similar_vuln_id = self.index_to_id_mapping[index]
                    
                    # Skip self
                    if similar_vuln_id == vulnerability_id:
                        continue
                    
                    if similar_vuln_id in self.vulnerability_vectors:
                        vuln_vector = self.vulnerability_vectors[similar_vuln_id]
                        
                        similar_result = SimilarityResult(
                            vulnerability_id=similar_vuln_id,
                            similarity_score=float(similarity),
                            title=vuln_vector.title,
                            description=vuln_vector.description,
                            severity=vuln_vector.severity,
                            target=vuln_vector.target,
                            created_at=vuln_vector.created_at,
                            metadata=vuln_vector.metadata
                        )
                        similar_results.append(similar_result)
            
            return similar_results[:k]  # Ensure we don't exceed k results
    
    async def detect_duplicate(
        self,
        vulnerability_id: str,
        title: str,
        description: str,
        severity: str,
        target: str,
        use_gpt_fallback: bool = True
    ) -> DeduplicationResult:
        """Detect if vulnerability is duplicate with GPT-4o fallback"""
        start_time = time.time()
        tokens_saved = 0
        
        # Check fingerprint-based exact duplicates first
        fingerprint = self._create_vulnerability_fingerprint(title, description, target)
        exact_duplicate = await self._check_fingerprint_duplicate(fingerprint, vulnerability_id)
        
        if exact_duplicate:
            processing_time = time.time() - start_time
            return DeduplicationResult(
                is_duplicate=True,
                confidence=1.0,
                duplicate_of=exact_duplicate,
                similar_findings=[],
                gpt_analysis="Exact fingerprint match",
                tokens_saved=500,  # Estimated tokens saved
                processing_time=processing_time,
                reasoning="Exact duplicate detected via fingerprint matching"
            )
        
        # Find similar vulnerabilities
        similar_findings = await self.find_similar_vulnerabilities(
            vulnerability_id, title, description, severity, target
        )
        
        if not similar_findings:
            processing_time = time.time() - start_time
            return DeduplicationResult(
                is_duplicate=False,
                confidence=0.0,
                duplicate_of=None,
                similar_findings=[],
                gpt_analysis=None,
                tokens_saved=0,
                processing_time=processing_time,
                reasoning="No similar vulnerabilities found"
            )
        
        # Check if top similarity exceeds threshold
        top_similar = similar_findings[0]
        if top_similar.similarity_score > self.similarity_threshold:
            # High confidence duplicate based on similarity alone
            tokens_saved = 300  # Estimated tokens saved by avoiding GPT
            triage_dedupe_saved_tokens_total.inc(tokens_saved)
            
            processing_time = time.time() - start_time
            return DeduplicationResult(
                is_duplicate=True,
                confidence=float(top_similar.similarity_score),
                duplicate_of=top_similar.vulnerability_id,
                similar_findings=similar_findings,
                gpt_analysis=None,
                tokens_saved=tokens_saved,
                processing_time=processing_time,
                reasoning=f"High similarity ({top_similar.similarity_score:.3f}) to existing vulnerability"
            )
        
        # Use GPT-4o for borderline cases if enabled
        gpt_analysis = None
        final_confidence = top_similar.similarity_score
        is_duplicate = False
        
        if use_gpt_fallback and top_similar.similarity_score > 0.70:  # GPT threshold
            try:
                gpt_result = await self._gpt_analyze_duplicates(
                    title, description, target, similar_findings[:3]
                )
                gpt_analysis = gpt_result['analysis']
                
                if gpt_result['is_duplicate']:
                    is_duplicate = True
                    final_confidence = max(final_confidence, gpt_result['confidence'])
                    gpt_rerank_operations.labels(result='duplicate').inc()
                else:
                    gpt_rerank_operations.labels(result='unique').inc()
                    
            except Exception as e:
                logger.error(f"GPT analysis failed: {e}")
                gpt_rerank_operations.labels(result='error').inc()
        
        # Calculate tokens saved
        if is_duplicate:
            tokens_saved = 400  # Estimated tokens saved
            triage_dedupe_saved_tokens_total.inc(tokens_saved)
        
        processing_time = time.time() - start_time
        
        result = DeduplicationResult(
            is_duplicate=is_duplicate,
            confidence=final_confidence,
            duplicate_of=top_similar.vulnerability_id if is_duplicate else None,
            similar_findings=similar_findings,
            gpt_analysis=gpt_analysis,
            tokens_saved=tokens_saved,
            processing_time=processing_time,
            reasoning=self._build_reasoning(is_duplicate, top_similar, gpt_analysis, final_confidence)
        )
        
        # Store result in database
        await self._store_deduplication_result(result)
        
        return result
    
    async def _check_fingerprint_duplicate(self, fingerprint: str, exclude_id: str) -> Optional[str]:
        """Check for exact fingerprint duplicates"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT vulnerability_id 
                FROM vulnerability_vectors 
                WHERE fingerprint = $1 AND vulnerability_id != $2
                LIMIT 1
            """, fingerprint, exclude_id)
            
            return row['vulnerability_id'] if row else None
    
    async def _gpt_analyze_duplicates(
        self, 
        title: str, 
        description: str, 
        target: str, 
        similar_findings: List[SimilarityResult]
    ) -> Dict:
        """Use GPT-4o to analyze potential duplicates"""
        
        prompt = f"""
        Analyze if this vulnerability is a duplicate of any similar findings:
        
        NEW VULNERABILITY:
        Title: {title}
        Description: {description}
        Target: {target}
        
        SIMILAR FINDINGS:
        """
        
        for i, finding in enumerate(similar_findings, 1):
            prompt += f"""
        {i}. Title: {finding.title}
           Description: {finding.description}
           Target: {finding.target}
           Similarity: {finding.similarity_score:.3f}
        """
        
        prompt += """
        
        Determine if the new vulnerability is a duplicate of any similar finding.
        Consider:
        1. Technical details and root cause
        2. Affected components and scope
        3. Exploitation methods
        4. Impact and severity
        
        Respond in JSON format:
        {
            "is_duplicate": boolean,
            "confidence": float (0.0-1.0),
            "duplicate_of_index": int or null (1-based),
            "analysis": "brief explanation"
        }
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a cybersecurity expert analyzing vulnerability duplicates."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # If GPT identified a duplicate, map back to vulnerability ID
            if result['is_duplicate'] and result.get('duplicate_of_index'):
                idx = result['duplicate_of_index'] - 1  # Convert to 0-based
                if 0 <= idx < len(similar_findings):
                    result['duplicate_of'] = similar_findings[idx].vulnerability_id
            
            return result
            
        except Exception as e:
            logger.error(f"GPT analysis failed: {e}")
            return {
                "is_duplicate": False,
                "confidence": 0.0,
                "analysis": f"Analysis failed: {str(e)}"
            }
    
    def _build_reasoning(
        self, 
        is_duplicate: bool, 
        top_similar: SimilarityResult, 
        gpt_analysis: Optional[str],
        confidence: float
    ) -> str:
        """Build human-readable reasoning"""
        if is_duplicate:
            reason = f"Duplicate detected (confidence: {confidence:.3f})"
            if gpt_analysis:
                reason += f" - GPT analysis: {gpt_analysis}"
            else:
                reason += f" - High similarity to '{top_similar.title}'"
        else:
            reason = f"Unique vulnerability (highest similarity: {top_similar.similarity_score:.3f})"
            if gpt_analysis:
                reason += f" - GPT confirmed uniqueness: {gpt_analysis}"
        
        return reason
    
    async def _store_deduplication_result(self, result: DeduplicationResult):
        """Store deduplication result in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO deduplication_results 
                    (vulnerability_id, is_duplicate, confidence, duplicate_of, 
                     similar_findings_count, gpt_analysis, tokens_saved, 
                     processing_time, reasoning)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, 
                result.vulnerability_id, result.is_duplicate, result.confidence,
                result.duplicate_of, len(result.similar_findings), result.gpt_analysis,
                result.tokens_saved, result.processing_time, result.reasoning)
        except Exception as e:
            logger.error(f"Failed to store deduplication result: {e}")
    
    async def get_statistics(self) -> Dict:
        """Get vector store statistics"""
        async with self.db_pool.acquire() as conn:
            # Get general stats
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_vectors,
                    COUNT(DISTINCT fingerprint) as unique_fingerprints,
                    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as recent_vectors
                FROM vulnerability_vectors
            """)
            
            # Get deduplication stats
            dedupe_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_analyses,
                    COUNT(*) FILTER (WHERE is_duplicate = true) as duplicates_found,
                    AVG(confidence) as avg_confidence,
                    SUM(tokens_saved) as total_tokens_saved,
                    AVG(processing_time) as avg_processing_time
                FROM deduplication_results
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """)
            
            return {
                'vector_store': {
                    'total_vectors': stats['total_vectors'],
                    'unique_fingerprints': stats['unique_fingerprints'],
                    'recent_vectors': stats['recent_vectors'],
                    'faiss_index_size': self.faiss_index.ntotal,
                    'embedding_cache_size': len(self.embedding_cache)
                },
                'deduplication_24h': {
                    'total_analyses': dedupe_stats['total_analyses'] or 0,
                    'duplicates_found': dedupe_stats['duplicates_found'] or 0,
                    'avg_confidence': float(dedupe_stats['avg_confidence'] or 0),
                    'total_tokens_saved': dedupe_stats['total_tokens_saved'] or 0,
                    'avg_processing_time': float(dedupe_stats['avg_processing_time'] or 0)
                },
                'configuration': {
                    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                    'embedding_dimension': self.embedding_dim,
                    'similarity_threshold': self.similarity_threshold,
                    'top_k_similar': self.top_k_similar
                }
            }

# Export main service class
__all__ = ['VectorStoreService', 'DeduplicationResult', 'SimilarityResult', 'VulnerabilityVector']