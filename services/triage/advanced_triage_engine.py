#!/usr/bin/env python3
"""
Xorb Advanced Triage Engine 2.0
Intelligent vulnerability triage using MiniLM embeddings, FAISS vector search, and GPT reranking
"""

import asyncio
import json
import logging
import math
import os
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import faiss
import asyncpg
import aioredis
import openai
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nats.aio.client import Client as NATS
from prometheus_client import Counter, Histogram, Gauge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
triage_operations = Counter('xorb_triage_operations_total', 'Total triage operations', ['operation', 'result'])
triage_duration = Histogram('xorb_triage_processing_duration_seconds', 'Triage processing duration')
embedding_cache_hits = Counter('xorb_triage_embedding_cache_hits_total', 'Embedding cache hits')
faiss_search_duration = Histogram('xorb_triage_faiss_search_duration_seconds', 'FAISS search duration')
gpt_rerank_duration = Histogram('xorb_triage_gpt_rerank_duration_seconds', 'GPT reranking duration')
duplicate_detection_accuracy = Gauge('xorb_triage_duplicate_detection_accuracy', 'Duplicate detection accuracy')

class SeverityLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class TriageStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Vulnerability:
    """Vulnerability finding data structure"""
    id: str
    title: str
    description: str
    severity: SeverityLevel
    cvss_score: Optional[float]
    cwe_id: Optional[str]
    target: str
    evidence: Dict
    tags: List[str]
    created_at: datetime
    raw_output: str
    
    def to_text(self) -> str:
        """Convert vulnerability to text for embedding"""
        return f"{self.title} {self.description} {' '.join(self.tags)} {self.target}"

@dataclass
class TriageResult:
    """Triage analysis result"""
    vulnerability_id: str
    is_duplicate: bool
    duplicate_of: Optional[str]
    confidence_score: float
    severity_adjusted: Optional[SeverityLevel]
    priority_score: float
    similar_findings: List[Dict]
    gpt_analysis: Optional[str]
    processing_time: float
    reasoning: str

class EmbeddingCache:
    """High-performance embedding cache with Redis backend"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.local_cache = {}
        self.cache_ttl = 86400 * 7  # 7 days
    
    async def get_embedding(self, text: str, text_hash: str) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        
        # Check local cache first
        if text_hash in self.local_cache:
            embedding_cache_hits.inc()
            return self.local_cache[text_hash]
        
        # Check Redis cache
        try:
            cached_data = await self.redis.get(f"embedding:{text_hash}")
            if cached_data:
                embedding = pickle.loads(cached_data)
                self.local_cache[text_hash] = embedding
                embedding_cache_hits.inc()
                return embedding
        except Exception as e:
            logger.error(f"Redis cache error: {e}")
        
        return None
    
    async def store_embedding(self, text: str, text_hash: str, embedding: np.ndarray):
        """Store embedding in cache"""
        
        # Store in local cache
        self.local_cache[text_hash] = embedding
        
        # Store in Redis
        try:
            await self.redis.setex(
                f"embedding:{text_hash}", 
                self.cache_ttl, 
                pickle.dumps(embedding)
            )
        except Exception as e:
            logger.error(f"Failed to cache embedding: {e}")

class AdvancedTriageEngine:
    """Advanced triage engine with ML-powered deduplication and prioritization"""
    
    def __init__(self):
        self.db_pool = None
        self.redis = None
        self.nats = None
        
        # ML Models
        self.embedding_model = None
        self.faiss_index = None
        self.embedding_cache = None
        self.tfidf_vectorizer = None
        
        # OpenAI client
        self.openai_client = None
        
        # Configuration
        self.embedding_dim = 384  # MiniLM dimension
        self.similarity_threshold = 0.85
        self.faiss_k = 10  # Top-K for FAISS search
        
        # State
        self.vulnerability_embeddings = {}
        self.vulnerability_metadata = {}
        self.processed_vulnerabilities = set()
    
    async def initialize(self):
        """Initialize the triage engine"""
        logger.info("Initializing Advanced Triage Engine 2.0...")
        
        # Database connection
        database_url = os.getenv("DATABASE_URL", "postgresql://xorb:xorb_secure_2024@postgres:5432/xorb_ptaas")
        self.db_pool = await asyncpg.create_pool(database_url, min_size=5, max_size=20)
        
        # Redis connection
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        self.redis = await aioredis.create_redis_pool(redis_url)
        self.embedding_cache = EmbeddingCache(self.redis)
        
        # NATS connection
        self.nats = NATS()
        await self.nats.connect(os.getenv("NATS_URL", "nats://nats:4222"))
        
        # Initialize ML models
        await self.initialize_ml_models()
        
        # Load existing vulnerability database
        await self.load_vulnerability_database()
        
        # Subscribe to triage requests
        await self.nats.subscribe("triage.request", cb=self.handle_triage_request)
        
        # Initialize OpenAI
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        logger.info("Advanced Triage Engine initialized successfully")
    
    async def initialize_ml_models(self):
        """Initialize machine learning models"""
        try:
            # Load MiniLM embedding model
            logger.info("Loading MiniLM embedding model...")
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Initialize FAISS index
            logger.info("Initializing FAISS index...")
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            
            # Initialize TF-IDF vectorizer for fallback
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
            raise
    
    async def load_vulnerability_database(self):
        """Load existing vulnerabilities and build FAISS index"""
        try:
            async with self.db_pool.acquire() as conn:
                # Load vulnerabilities from the last 90 days
                rows = await conn.fetch("""
                    SELECT id, title, description, severity, target, tags, evidence, raw_output, created_at
                    FROM vulnerabilities 
                    WHERE created_at > NOW() - INTERVAL '90 days'
                    ORDER BY created_at DESC
                    LIMIT 10000
                """)
                
                if not rows:
                    logger.info("No existing vulnerabilities found")
                    return
                
                logger.info(f"Loading {len(rows)} vulnerabilities for indexing...")
                
                # Process vulnerabilities and build embeddings
                vulnerabilities = []
                texts = []
                
                for row in rows:
                    vuln = Vulnerability(
                        id=row['id'],
                        title=row['title'],
                        description=row['description'],
                        severity=SeverityLevel(row['severity']),
                        target=row['target'],
                        tags=row['tags'] or [],
                        evidence=row['evidence'] or {},
                        raw_output=row['raw_output'] or "",
                        created_at=row['created_at'],
                        cvss_score=None,
                        cwe_id=None
                    )
                    
                    vulnerabilities.append(vuln)
                    texts.append(vuln.to_text())
                    self.vulnerability_metadata[row['id']] = vuln
                
                # Generate embeddings in batches
                batch_size = 100
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_embeddings = await self.generate_embeddings_batch(batch_texts)
                    all_embeddings.extend(batch_embeddings)
                
                # Build FAISS index
                if all_embeddings:
                    embeddings_matrix = np.array(all_embeddings).astype('float32')
                    # Normalize for cosine similarity
                    faiss.normalize_L2(embeddings_matrix)
                    self.faiss_index.add(embeddings_matrix)
                    
                    # Store embeddings mapping
                    for i, vuln in enumerate(vulnerabilities):
                        self.vulnerability_embeddings[vuln.id] = all_embeddings[i]
                
                logger.info(f"Built FAISS index with {len(all_embeddings)} vulnerability embeddings")
                
        except Exception as e:
            logger.error(f"Failed to load vulnerability database: {e}")
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts"""
        try:
            # Check cache first
            embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                text_hash = str(hash(text))
                cached_embedding = await self.embedding_cache.get_embedding(text, text_hash)
                
                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                else:
                    embeddings.append(None)
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                new_embeddings = self.embedding_model.encode(uncached_texts)
                
                # Store in cache and update results
                for j, embedding in enumerate(new_embeddings):
                    idx = uncached_indices[j]
                    text = uncached_texts[j]
                    text_hash = str(hash(text))
                    
                    embeddings[idx] = embedding
                    await self.embedding_cache.store_embedding(text, text_hash, embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return [np.zeros(self.embedding_dim) for _ in texts]
    
    async def handle_triage_request(self, msg):
        """Handle incoming triage requests"""
        try:
            data = json.loads(msg.data.decode())
            
            vulnerability = Vulnerability(
                id=data['id'],
                title=data['title'],
                description=data['description'],
                severity=SeverityLevel(data['severity']),
                target=data['target'],
                tags=data.get('tags', []),
                evidence=data.get('evidence', {}),
                raw_output=data.get('raw_output', ''),
                created_at=datetime.fromisoformat(data['created_at']),
                cvss_score=data.get('cvss_score'),
                cwe_id=data.get('cwe_id')
            )
            
            # Process triage
            result = await self.triage_vulnerability(vulnerability)
            
            # Publish result
            await self.publish_triage_result(result)
            
        except Exception as e:
            logger.error(f"Failed to handle triage request: {e}")
    
    async def triage_vulnerability(self, vulnerability: Vulnerability) -> TriageResult:
        """Perform comprehensive triage analysis"""
        start_time = time.time()
        
        with triage_duration.time():
            try:
                logger.info(f"Triaging vulnerability: {vulnerability.id}")
                
                # Generate embedding for the new vulnerability
                vuln_text = vulnerability.to_text()
                vuln_embeddings = await self.generate_embeddings_batch([vuln_text])
                vuln_embedding = vuln_embeddings[0]
                
                # FAISS similarity search
                similar_findings = await self.find_similar_vulnerabilities(
                    vuln_embedding, vulnerability
                )
                
                # Duplicate detection
                is_duplicate, duplicate_of, confidence = await self.detect_duplicate(
                    vulnerability, similar_findings
                )
                
                # Severity adjustment based on context
                adjusted_severity = await self.adjust_severity(
                    vulnerability, similar_findings
                )
                
                # Priority scoring
                priority_score = await self.calculate_priority_score(
                    vulnerability, similar_findings, is_duplicate
                )
                
                # GPT-powered analysis and reranking
                gpt_analysis = await self.gpt_analyze_vulnerability(
                    vulnerability, similar_findings
                )
                
                # Build reasoning
                reasoning = self.build_triage_reasoning(
                    vulnerability, similar_findings, is_duplicate, 
                    confidence, adjusted_severity, priority_score
                )
                
                processing_time = time.time() - start_time
                
                result = TriageResult(
                    vulnerability_id=vulnerability.id,
                    is_duplicate=is_duplicate,
                    duplicate_of=duplicate_of,
                    confidence_score=confidence,
                    severity_adjusted=adjusted_severity,
                    priority_score=priority_score,
                    similar_findings=similar_findings,
                    gpt_analysis=gpt_analysis,
                    processing_time=processing_time,
                    reasoning=reasoning
                )
                
                # Update internal state
                await self.update_vulnerability_index(vulnerability, vuln_embedding)
                
                # Update metrics
                triage_operations.labels(
                    operation="triage", 
                    result="duplicate" if is_duplicate else "unique"
                ).inc()
                
                logger.info(f"Triage completed for {vulnerability.id} in {processing_time:.2f}s")
                return result
                
            except Exception as e:
                logger.error(f"Triage failed for {vulnerability.id}: {e}")
                triage_operations.labels(operation="triage", result="error").inc()
                
                return TriageResult(
                    vulnerability_id=vulnerability.id,
                    is_duplicate=False,
                    duplicate_of=None,
                    confidence_score=0.0,
                    severity_adjusted=vulnerability.severity,
                    priority_score=50.0,
                    similar_findings=[],
                    gpt_analysis=None,
                    processing_time=time.time() - start_time,
                    reasoning=f"Triage failed: {str(e)}"
                )
    
    async def find_similar_vulnerabilities(
        self, 
        query_embedding: np.ndarray, 
        vulnerability: Vulnerability
    ) -> List[Dict]:
        """Find similar vulnerabilities using FAISS"""
        
        if self.faiss_index.ntotal == 0:
            return []
        
        try:
            with faiss_search_duration.time():
                # Normalize query embedding
                query_embedding = query_embedding.reshape(1, -1).astype('float32')
                faiss.normalize_L2(query_embedding)
                
                # Search FAISS index
                similarities, indices = self.faiss_index.search(query_embedding, self.faiss_k)
                
                similar_findings = []
                for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                    if idx == -1:  # Invalid index
                        continue
                    
                    # Get vulnerability metadata
                    vuln_ids = list(self.vulnerability_metadata.keys())
                    if idx < len(vuln_ids):
                        vuln_id = vuln_ids[idx]
                        similar_vuln = self.vulnerability_metadata[vuln_id]
                        
                        similar_findings.append({
                            'id': vuln_id,
                            'title': similar_vuln.title,
                            'description': similar_vuln.description,
                            'severity': similar_vuln.severity.value,
                            'target': similar_vuln.target,
                            'similarity_score': float(similarity),
                            'created_at': similar_vuln.created_at.isoformat()
                        })
                
                return similar_findings
                
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    async def detect_duplicate(
        self, 
        vulnerability: Vulnerability, 
        similar_findings: List[Dict]
    ) -> Tuple[bool, Optional[str], float]:
        """Detect if vulnerability is a duplicate"""
        
        if not similar_findings:
            return False, None, 0.0
        
        # Check top similarity
        top_similar = similar_findings[0]
        similarity_score = top_similar['similarity_score']
        
        # Additional checks for duplicate detection
        confidence_factors = []
        
        # Similarity threshold
        if similarity_score > self.similarity_threshold:
            confidence_factors.append(similarity_score)
        
        # Same target check
        if top_similar['target'] == vulnerability.target:
            confidence_factors.append(0.9)
        
        # Title similarity (fuzzy matching)
        title_similarity = self.calculate_text_similarity(
            vulnerability.title, top_similar['title']
        )
        if title_similarity > 0.8:
            confidence_factors.append(title_similarity)
        
        # Time proximity check (recent duplicates more likely)
        time_diff = datetime.now() - datetime.fromisoformat(top_similar['created_at'])
        if time_diff.days < 7:
            confidence_factors.append(0.8)
        
        # Calculate overall confidence
        if confidence_factors:
            confidence = np.mean(confidence_factors)
            is_duplicate = confidence > 0.75
            duplicate_of = top_similar['id'] if is_duplicate else None
            
            return is_duplicate, duplicate_of, confidence
        
        return False, None, 0.0
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using TF-IDF"""
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    async def adjust_severity(
        self, 
        vulnerability: Vulnerability, 
        similar_findings: List[Dict]
    ) -> Optional[SeverityLevel]:
        """Adjust severity based on context and similar findings"""
        
        if not similar_findings:
            return None
        
        # Analyze severity patterns in similar findings
        severity_counts = {}
        for finding in similar_findings[:5]:  # Top 5 similar
            severity = finding['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Check if there's a strong pattern different from original
        if severity_counts:
            most_common_severity = max(severity_counts, key=severity_counts.get)
            if (severity_counts[most_common_severity] >= 3 and 
                most_common_severity != vulnerability.severity.value):
                
                return SeverityLevel(most_common_severity)
        
        return None
    
    async def calculate_priority_score(
        self, 
        vulnerability: Vulnerability, 
        similar_findings: List[Dict], 
        is_duplicate: bool
    ) -> float:
        """Calculate priority score (0-100)"""
        
        score = 50.0  # Base score
        
        # Severity impact
        severity_scores = {
            SeverityLevel.CRITICAL: 40,
            SeverityLevel.HIGH: 30,
            SeverityLevel.MEDIUM: 20,
            SeverityLevel.LOW: 10,
            SeverityLevel.INFO: 5
        }
        score += severity_scores.get(vulnerability.severity, 20)
        
        # Duplicate penalty
        if is_duplicate:
            score -= 30
        
        # Uniqueness bonus
        if not similar_findings:
            score += 15
        
        # Target criticality (heuristic)
        if any(keyword in vulnerability.target.lower() 
               for keyword in ['admin', 'login', 'api', 'database']):
            score += 10
        
        # Evidence richness
        if vulnerability.evidence and len(vulnerability.evidence) > 3:
            score += 5
        
        # CVSS score integration
        if vulnerability.cvss_score:
            score += (vulnerability.cvss_score / 10) * 20
        
        return min(100.0, max(0.0, score))
    
    async def gpt_analyze_vulnerability(
        self, 
        vulnerability: Vulnerability, 
        similar_findings: List[Dict]
    ) -> Optional[str]:
        """Use GPT for advanced vulnerability analysis and reranking"""
        
        if not openai.api_key:
            return None
        
        try:
            with gpt_rerank_duration.time():
                # Prepare context for GPT
                context = f"""
                Vulnerability Analysis Request:
                
                Title: {vulnerability.title}
                Description: {vulnerability.description}
                Severity: {vulnerability.severity.value}
                Target: {vulnerability.target}
                Tags: {', '.join(vulnerability.tags)}
                
                Similar Findings Found:
                """
                
                for i, finding in enumerate(similar_findings[:3]):
                    context += f"""
                {i+1}. {finding['title']} (Similarity: {finding['similarity_score']:.2f})
                   Target: {finding['target']}
                   Severity: {finding['severity']}
                """
                
                prompt = context + """
                
                Please analyze this vulnerability and provide:
                1. Risk assessment and business impact
                2. Likelihood of exploitation
                3. Recommended priority level (Critical/High/Medium/Low)
                4. Any false positive indicators
                5. Suggested remediation approach
                
                Keep analysis under 200 words and be specific.
                """
                
                response = await asyncio.wait_for(
                    openai.ChatCompletion.acreate(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a cybersecurity expert analyzing vulnerability findings."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=300,
                        temperature=0.3
                    ),
                    timeout=30.0
                )
                
                return response.choices[0].message.content.strip()
                
        except Exception as e:
            logger.error(f"GPT analysis failed: {e}")
            return None
    
    def build_triage_reasoning(
        self, 
        vulnerability: Vulnerability,
        similar_findings: List[Dict],
        is_duplicate: bool,
        confidence: float,
        adjusted_severity: Optional[SeverityLevel],
        priority_score: float
    ) -> str:
        """Build human-readable reasoning for triage decision"""
        
        reasons = []
        
        if is_duplicate:
            reasons.append(f"Identified as duplicate with {confidence:.1%} confidence")
            if similar_findings:
                reasons.append(f"Most similar to: {similar_findings[0]['title']}")
        else:
            reasons.append("Unique vulnerability - no duplicates found")
        
        if adjusted_severity:
            reasons.append(f"Severity adjusted from {vulnerability.severity.value} to {adjusted_severity.value}")
        
        reasons.append(f"Priority score: {priority_score:.1f}/100")
        
        if similar_findings:
            reasons.append(f"Found {len(similar_findings)} similar findings for context")
        
        if vulnerability.cvss_score:
            reasons.append(f"CVSS Score: {vulnerability.cvss_score}")
        
        return "; ".join(reasons)
    
    async def update_vulnerability_index(self, vulnerability: Vulnerability, embedding: np.ndarray):
        """Update FAISS index with new vulnerability"""
        try:
            # Add to metadata
            self.vulnerability_metadata[vulnerability.id] = vulnerability
            self.vulnerability_embeddings[vulnerability.id] = embedding
            
            # Add to FAISS index
            embedding_normalized = embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(embedding_normalized)
            self.faiss_index.add(embedding_normalized)
            
            # Store in database
            await self.store_vulnerability_embedding(vulnerability.id, embedding)
            
        except Exception as e:
            logger.error(f"Failed to update vulnerability index: {e}")
    
    async def store_vulnerability_embedding(self, vuln_id: str, embedding: np.ndarray):
        """Store vulnerability embedding in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO vulnerability_embeddings (vulnerability_id, embedding, created_at)
                    VALUES ($1, $2, NOW())
                    ON CONFLICT (vulnerability_id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        updated_at = NOW()
                """, vuln_id, embedding.tobytes())
        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")
    
    async def publish_triage_result(self, result: TriageResult):
        """Publish triage result to NATS"""
        try:
            message = {
                'vulnerability_id': result.vulnerability_id,
                'is_duplicate': result.is_duplicate,
                'duplicate_of': result.duplicate_of,
                'confidence_score': result.confidence_score,
                'severity_adjusted': result.severity_adjusted.value if result.severity_adjusted else None,
                'priority_score': result.priority_score,
                'similar_findings_count': len(result.similar_findings),
                'gpt_analysis': result.gpt_analysis,
                'processing_time': result.processing_time,
                'reasoning': result.reasoning,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.nats.publish("triage.completed", json.dumps(message).encode())
            
        except Exception as e:
            logger.error(f"Failed to publish triage result: {e}")
    
    async def get_triage_stats(self) -> Dict:
        """Get triage engine statistics"""
        try:
            async with self.db_pool.acquire() as conn:
                stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_processed,
                        COUNT(*) FILTER (WHERE is_duplicate = true) as duplicates_found,
                        AVG(confidence_score) as avg_confidence,
                        AVG(priority_score) as avg_priority,
                        AVG(processing_time) as avg_processing_time
                    FROM triage_results 
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                """)
                
                return {
                    'faiss_index_size': self.faiss_index.ntotal,
                    'cached_embeddings': len(self.vulnerability_embeddings),
                    'similarity_threshold': self.similarity_threshold,
                    'last_24h_stats': dict(stats) if stats else {},
                    'model_info': {
                        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                        'embedding_dimension': self.embedding_dim,
                        'faiss_index_type': 'IndexFlatIP'
                    }
                }
        except Exception as e:
            logger.error(f"Failed to get triage stats: {e}")
            return {}

async def main():
    """Main function for the advanced triage engine"""
    engine = AdvancedTriageEngine()
    
    try:
        await engine.initialize()
        logger.info("🧠 Advanced Triage Engine 2.0 with MiniLM + FAISS + GPT running")
        
        # Keep running
        while True:
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Triage engine stopped by user")
    except Exception as e:
        logger.error(f"Triage engine failed: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())