#!/usr/bin/env python3
"""
LLM-Enhanced Knowledge Fabric for XORB Supreme
Integrates LLM-generated payloads and tactics with provenance tracking
"""

import asyncio
import logging
import json
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel

from xorb_core.knowledge_fabric.core import KnowledgeFabric
from xorb_core.knowledge_fabric.atom import KnowledgeAtom, AtomType, Source
from xorb_core.llm.intelligent_client import IntelligentLLMClient, LLMResponse, TaskType
from xorb_core.llm.payload_generator import PayloadGenerator, GeneratedPayload, PayloadCategory

logger = logging.getLogger(__name__)

class LLMProvenanceType(Enum):
    GENERATED = "llm_generated"
    ENHANCED = "llm_enhanced"
    VALIDATED = "llm_validated"
    SUMMARIZED = "llm_summarized"

@dataclass
class LLMProvenance:
    """Tracks the origin and quality of LLM-generated content"""
    model_name: str
    provider: str
    generated_at: datetime
    confidence_score: float
    tokens_used: int
    cost_usd: float
    prompt_hash: str  # Hash of the prompt used
    provenance_type: LLMProvenanceType
    validation_status: Optional[str] = None
    human_reviewed: bool = False

class LLMAtom(KnowledgeAtom):
    """Extended KnowledgeAtom with LLM-specific metadata"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_provenance: Optional[LLMProvenance] = None
        self.payload_data: Optional[GeneratedPayload] = None
        self.enhancement_history: List[Dict[str, Any]] = []

class LLMKnowledgeFabric(KnowledgeFabric):
    """Enhanced Knowledge Fabric with LLM integration"""
    
    def __init__(self, redis_url: str, database_url: str, llm_client: IntelligentLLMClient):
        super().__init__(redis_url, database_url)
        self.llm_client = llm_client
        self.payload_generator = PayloadGenerator(llm_client)
        
        # LLM-specific storage
        self.llm_atoms: Dict[str, LLMAtom] = {}
        self.provenance_index: Dict[str, List[str]] = {}  # model -> atom_ids
        self.confidence_threshold = 0.7
        
    async def initialize(self):
        """Initialize the enhanced knowledge fabric"""
        await super().initialize()
        await self.llm_client.start()
        logger.info("LLM Knowledge Fabric initialized")
    
    async def store_llm_payload(
        self,
        payload: GeneratedPayload,
        llm_response: LLMResponse,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store LLM-generated payload in knowledge fabric"""
        
        # Create provenance record
        provenance = LLMProvenance(
            model_name=llm_response.model_used,
            provider=llm_response.provider.value,
            generated_at=llm_response.generated_at,
            confidence_score=llm_response.confidence_score,
            tokens_used=llm_response.tokens_used,
            cost_usd=llm_response.cost_usd,
            prompt_hash=hashlib.sha256(str(context or {}).encode()).hexdigest()[:16],
            provenance_type=LLMProvenanceType.GENERATED
        )
        
        # Create enhanced atom
        atom = LLMAtom(
            title=f"LLM Generated Payload: {payload.category.value}",
            content={
                "payload_string": payload.payload,
                "category": payload.category.value,
                "complexity": payload.complexity.value,
                "description": payload.description,
                "target_parameter": payload.target_parameter,
                "expected_result": payload.expected_result,
                "detection_difficulty": payload.detection_difficulty,
                "remediation": payload.remediation,
                "references": payload.references,
                "context": context or {}
            },
            atom_type=AtomType.PAYLOAD,
            sources=[Source(name="llm_generator", type="llm", reliability_score=0.8)],
            confidence=payload.success_probability
        )
        
        # Add LLM-specific data
        atom.llm_provenance = provenance
        atom.payload_data = payload
        
        # Generate unique ID
        atom_id = self._generate_atom_id(atom)
        
        # Store in enhanced fabric
        self.llm_atoms[atom_id] = atom
        await self._store_atom_persistent(atom_id, atom)
        
        # Update provenance index
        if provenance.model_name not in self.provenance_index:
            self.provenance_index[provenance.model_name] = []
        self.provenance_index[provenance.model_name].append(atom_id)
        
        logger.info(f"Stored LLM payload atom {atom_id} from {provenance.model_name}")
        return atom_id
    
    async def generate_and_store_payloads(
        self,
        category: PayloadCategory,
        target_context: Dict[str, Any],
        count: int = 5
    ) -> List[str]:
        """Generate payloads with LLM and store them"""
        
        from llm.payload_generator import TargetContext, PayloadComplexity
        
        # Convert context
        context = TargetContext(**target_context)
        
        # Generate payloads
        payloads = await self.payload_generator.generate_contextual_payloads(
            category=category,
            target_context=context,
            complexity=PayloadComplexity.INTERMEDIATE,
            count=count
        )
        
        # Store each payload
        atom_ids = []
        for payload in payloads:
            # Create mock LLM response for provenance
            mock_response = LLMResponse(
                content=payload.payload,
                model_used="payload_generator",
                provider=self.llm_client.models[self.llm_client.select_optimal_model(
                    self.llm_client.LLMRequest(task_type=TaskType.PAYLOAD_GENERATION, prompt="")
                )].provider,
                tokens_used=100,
                cost_usd=0.001,
                confidence_score=payload.success_probability,
                generated_at=datetime.now(timezone.utc),
                request_id=f"gen_{int(datetime.now(timezone.utc).timestamp())}"
            )
            
            atom_id = await self.store_llm_payload(payload, mock_response, target_context)
            atom_ids.append(atom_id)
        
        return atom_ids
    
    async def enhance_existing_atom(self, atom_id: str, enhancement_context: Dict[str, Any]) -> str:
        """Enhance existing atom with LLM"""
        
        if atom_id not in self.llm_atoms:
            raise ValueError(f"Atom {atom_id} not found")
        
        original_atom = self.llm_atoms[atom_id]
        
        if not original_atom.payload_data:
            raise ValueError(f"Atom {atom_id} is not a payload atom")
        
        # Enhance with LLM
        enhanced_payload = await self.payload_generator.enhance_payload_with_context(
            original_atom.payload_data,
            enhancement_context
        )
        
        # Create enhanced atom
        enhanced_atom = LLMAtom(
            title=f"LLM Enhanced Payload: {enhanced_payload.category.value}",
            content={
                "payload": enhanced_payload.payload,
                "category": enhanced_payload.category.value,
                "complexity": enhanced_payload.complexity.value,
                "description": enhanced_payload.description,
                "target_parameter": enhanced_payload.target_parameter,
                "expected_result": enhanced_payload.expected_result,
                "detection_difficulty": enhanced_payload.detection_difficulty,
                "remediation": enhanced_payload.remediation,
                "references": enhanced_payload.references,
                "enhanced_from": atom_id,
                "enhancement_context": enhancement_context,
                "original_confidence": original_atom.confidence
            },
            atom_type=AtomType.PAYLOAD,
            sources=[Source(name="llm_enhancer", type="llm", reliability_score=0.9)],
            confidence=enhanced_payload.success_probability
        )
        
        # Update metadata
        enhanced_atom.metadata.update({
            "enhanced_from": atom_id,
            "enhancement_context": enhancement_context,
            "original_confidence": original_atom.confidence
        })
        
        # Add enhancement history
        enhanced_atom.enhancement_history = original_atom.enhancement_history.copy()
        enhanced_atom.enhancement_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": enhancement_context,
            "confidence_change": enhanced_payload.success_probability - original_atom.confidence
        })
        
        # Create provenance for enhancement
        enhanced_atom.llm_provenance = LLMProvenance(
            model_name="enhancement_generator",
            provider="internal",
            generated_at=datetime.now(timezone.utc),
            confidence_score=enhanced_payload.success_probability,
            tokens_used=0,
            cost_usd=0.0,
            prompt_hash=hashlib.sha256(str(enhancement_context).encode()).hexdigest()[:16],
            provenance_type=LLMProvenanceType.ENHANCED
        )
        
        enhanced_atom.payload_data = enhanced_payload
        
        # Store enhanced atom
        enhanced_id = self._generate_atom_id(enhanced_atom)
        self.llm_atoms[enhanced_id] = enhanced_atom
        await self._store_atom_persistent(enhanced_id, enhanced_atom)
        
        logger.info(f"Enhanced atom {atom_id} -> {enhanced_id}")
        return enhanced_id
    
    async def query_llm_atoms(
        self,
        category: Optional[PayloadCategory] = None,
        min_confidence: float = 0.0,
        model_name: Optional[str] = None,
        limit: int = 10
    ) -> List[Tuple[str, LLMAtom]]:
        """Query LLM-generated atoms with filters"""
        
        results = []
        
        for atom_id, atom in self.llm_atoms.items():
            # Apply filters
            if category and atom.metadata.get("category") != category.value:
                continue
            
            if atom.confidence < min_confidence:
                continue
            
            if model_name and atom.llm_provenance and atom.llm_provenance.model_name != model_name:
                continue
            
            results.append((atom_id, atom))
        
        # Sort by confidence (descending)
        results.sort(key=lambda x: x[1].confidence, reverse=True)
        
        return results[:limit]
    
    async def analyze_with_llm(self, content: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze content using LLM and store results"""
        
        analysis_prompts = {
            "vulnerability_assessment": f"""
            Analyze this security finding for:
            - Severity (Critical/High/Medium/Low)
            - CVSS v3.1 score
            - Exploitability assessment
            - Business impact
            - Remediation priority
            
            Content: {content}
            """,
            
            "payload_effectiveness": f"""
            Evaluate this security payload for:
            - Success probability (0.0-1.0)
            - Detection difficulty (1-5)
            - Potential damage assessment
            - Variations and improvements
            
            Payload: {content}
            """,
            
            "threat_intelligence": f"""
            Extract threat intelligence from this data:
            - IoCs (indicators of compromise)
            - TTPs (tactics, techniques, procedures)
            - Attribution indicators
            - Related campaigns
            
            Data: {content}
            """
        }
        
        if analysis_type not in analysis_prompts:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        from llm.intelligent_client import LLMRequest
        request = LLMRequest(
            task_type=TaskType.VULNERABILITY_ANALYSIS,
            prompt=analysis_prompts[analysis_type],
            max_tokens=2000,
            temperature=0.3,  # Lower temperature for analysis
            structured_output=True
        )
        
        try:
            response = await self.llm_client.generate_payload(request)
            
            # Parse analysis results
            analysis_results = self._parse_analysis_response(response, analysis_type)
            
            # Store analysis as atom
            analysis_atom = LLMAtom(
                title=f"LLM Analysis: {analysis_type}",
                content={
                    "raw_content": content,
                    "analysis_type": analysis_type,
                    "results": analysis_results,
                    "analyzed_at": datetime.now(timezone.utc).isoformat()
                },
                atom_type=AtomType.INTELLIGENCE,  # Use existing enum value
                sources=[Source(name="llm_analyzer", type="llm", reliability_score=0.85)],
                confidence=response.confidence_score
            )
            
            analysis_atom.llm_provenance = LLMProvenance(
                model_name=response.model_used,
                provider=response.provider.value,
                generated_at=response.generated_at,
                confidence_score=response.confidence_score,
                tokens_used=response.tokens_used,
                cost_usd=response.cost_usd,
                prompt_hash=hashlib.sha256(analysis_prompts[analysis_type].encode()).hexdigest()[:16],
                provenance_type=LLMProvenanceType.VALIDATED
            )
            
            # Store analysis atom
            analysis_id = self._generate_atom_id(analysis_atom)
            self.llm_atoms[analysis_id] = analysis_atom
            await self._store_atom_persistent(analysis_id, analysis_atom)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {"error": str(e), "analysis_type": analysis_type}
    
    def _parse_analysis_response(self, response: LLMResponse, analysis_type: str) -> Dict[str, Any]:
        """Parse LLM analysis response"""
        try:
            # Try JSON parsing first
            if response.content.strip().startswith('{'):
                return json.loads(response.content)
        except:
            pass
        
        # Fallback text parsing
        return {
            "raw_analysis": response.content,
            "confidence": response.confidence_score,
            "model_used": response.model_used,
            "analysis_type": analysis_type
        }
    
    async def get_llm_fabric_stats(self) -> Dict[str, Any]:
        """Get statistics about LLM-generated content"""
        
        stats = {
            "total_llm_atoms": len(self.llm_atoms),
            "atoms_by_model": {},
            "atoms_by_category": {},
            "atoms_by_confidence": {"high": 0, "medium": 0, "low": 0},
            "total_cost": 0.0,
            "avg_confidence": 0.0
        }
        
        total_confidence = 0.0
        
        for atom in self.llm_atoms.values():
            # Count by model
            if atom.llm_provenance:
                model = atom.llm_provenance.model_name
                stats["atoms_by_model"][model] = stats["atoms_by_model"].get(model, 0) + 1
                stats["total_cost"] += atom.llm_provenance.cost_usd
            
            # Count by category
            category = atom.metadata.get("category", "unknown")
            stats["atoms_by_category"][category] = stats["atoms_by_category"].get(category, 0) + 1
            
            # Count by confidence
            if atom.confidence >= 0.8:
                stats["atoms_by_confidence"]["high"] += 1
            elif atom.confidence >= 0.5:
                stats["atoms_by_confidence"]["medium"] += 1
            else:
                stats["atoms_by_confidence"]["low"] += 1
            
            total_confidence += atom.confidence
        
        if len(self.llm_atoms) > 0:
            stats["avg_confidence"] = total_confidence / len(self.llm_atoms)
        
        return stats
    
    async def cleanup_low_quality_atoms(self, min_confidence: float = 0.3, max_age_days: int = 30):
        """Clean up low-quality or old atoms"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        atoms_to_remove = []
        
        for atom_id, atom in self.llm_atoms.items():
            should_remove = False
            
            # Remove low confidence atoms
            if atom.confidence < min_confidence:
                should_remove = True
            
            # Remove old atoms with low confidence
            if (atom.llm_provenance and 
                atom.llm_provenance.generated_at < cutoff_date and 
                atom.confidence < 0.6):
                should_remove = True
            
            if should_remove:
                atoms_to_remove.append(atom_id)
        
        # Remove atoms
        for atom_id in atoms_to_remove:
            del self.llm_atoms[atom_id]
            await self._remove_atom_persistent(atom_id)
        
        logger.info(f"Cleaned up {len(atoms_to_remove)} low-quality atoms")
        return len(atoms_to_remove)
    
    async def _store_atom_persistent(self, atom_id: str, atom: LLMAtom):
        """Store atom in persistent storage"""
        # Serialize atom data
        atom_data = {
            "content": atom.content,
            "atom_type": atom.atom_type.value,
            "confidence": atom.confidence,
            "created_at": atom.created_at.isoformat() if atom.created_at else None,
            "llm_provenance": {
                "model_name": atom.llm_provenance.model_name,
                "provider": atom.llm_provenance.provider,
                "generated_at": atom.llm_provenance.generated_at.isoformat(),
                "confidence_score": atom.llm_provenance.confidence_score,
                "tokens_used": atom.llm_provenance.tokens_used,
                "cost_usd": atom.llm_provenance.cost_usd,
                "prompt_hash": atom.llm_provenance.prompt_hash,
                "provenance_type": atom.llm_provenance.provenance_type.value
            } if atom.llm_provenance else None,
            "enhancement_history": atom.enhancement_history
        }
        
        # Store in Redis
        await self.redis_client.hset("llm_atoms", atom_id, json.dumps(atom_data))
        
        # Store in database if available
        try:
            if hasattr(self, 'session') and self.session:
                # Would implement SQLAlchemy storage here
                pass
        except Exception as e:
            logger.warning(f"Failed to store atom in database: {e}")
    
    async def _remove_atom_persistent(self, atom_id: str):
        """Remove atom from persistent storage"""
        await self.redis.hdel("llm_atoms", atom_id)
    
    def _generate_atom_id(self, atom: LLMAtom) -> str:
        """Generate unique ID for atom"""
        content_hash = hashlib.sha256(json.dumps(atom.content, sort_keys=True).encode()).hexdigest()[:16]
        timestamp = int(datetime.now(timezone.utc).timestamp())
        return f"llm_{atom.atom_type.value}_{content_hash}_{timestamp}"