#!/usr/bin/env python3

import asyncio
import logging
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..integrations.openrouter_client import OpenRouterClient, LLMRequest, ModelProvider
from ..knowledge_fabric.core import KnowledgeFabric
from ..knowledge_fabric.atom import KnowledgeAtom, AtomType, Source, ValidationResult


class LLMKnowledgeUpdater:
    def __init__(self, 
                 openrouter_client: OpenRouterClient,
                 knowledge_fabric: KnowledgeFabric,
                 config: Optional[Dict[str, Any]] = None):
        
        self.openrouter = openrouter_client
        self.knowledge_fabric = knowledge_fabric
        self.config = config or {}
        
        self.logger = logging.getLogger(__name__)
        
        self.update_interval = self.config.get('update_interval', 3600)  # 1 hour
        self.max_daily_requests = self.config.get('max_daily_requests', 100)
        self.batch_size = self.config.get('batch_size', 5)
        
        self.running = False
        self._update_task: Optional[asyncio.Task] = None
        
        self.daily_request_count = 0
        self.last_reset_date = datetime.utcnow().date()
        
        # Load prompt templates
        self._load_prompt_templates()

    def _load_prompt_templates(self):
        """Load rotating prompts for knowledge gathering"""
        self.prompt_templates = {
            "vulnerability_research": [
                "Describe the latest web application vulnerabilities discovered in 2024, focusing on technical details and detection methods.",
                "What are the most common API security vulnerabilities and how can they be identified during security assessments?",
                "Explain advanced SQL injection techniques that bypass modern web application firewalls.",
                "Detail the latest techniques for exploiting server-side request forgery (SSRF) vulnerabilities.",
                "What are the emerging attack vectors in cloud infrastructure security?"
            ],
            "technique_discovery": [
                "Describe advanced reconnaissance techniques used in red team operations for information gathering.",
                "What are the latest techniques for bypassing endpoint detection and response (EDR) systems?",
                "Explain modern privilege escalation techniques in Linux environments.",
                "Detail advanced persistent threat (APT) techniques for maintaining access in enterprise networks.",
                "What are the latest techniques for lateral movement in Active Directory environments?"
            ],
            "payload_research": [
                "Provide examples of modern XSS payloads that bypass content security policy (CSP) protections.",
                "What are effective payloads for testing command injection vulnerabilities in different environments?",
                "Describe payload techniques for exploiting deserialization vulnerabilities in web applications.",
                "What are the latest techniques for creating undetectable reverse shell payloads?",
                "Provide examples of payloads for testing LDAP injection vulnerabilities."
            ],
            "defensive_intelligence": [
                "What are the most effective detection rules for identifying SQL injection attacks in web logs?",
                "Describe indicators of compromise (IOCs) for detecting advanced persistent threats in network traffic.",
                "What are the best practices for detecting and preventing privilege escalation attacks?",
                "How can organizations detect lateral movement activities in their networks?",
                "What are effective methods for detecting command and control (C2) communications?"
            ],
            "threat_intelligence": [
                "What are the latest tactics, techniques, and procedures (TTPs) used by cybercriminal groups?",
                "Describe recent trends in ransomware attacks and their technical characteristics.",
                "What are the emerging threats in cloud security and how are they being exploited?",
                "Detail the latest phishing techniques and how they bypass security controls.",
                "What are the current trends in supply chain attacks and their detection methods?"
            ]
        }

    async def start_continuous_updates(self):
        """Start continuous knowledge updates"""
        if self.running:
            self.logger.warning("LLM updater is already running")
            return
        
        self.running = True
        self._update_task = asyncio.create_task(self._update_loop())
        
        self.logger.info("LLM knowledge updater started")

    async def stop(self):
        """Stop continuous updates"""
        if not self.running:
            return
        
        self.running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("LLM knowledge updater stopped")

    async def single_update_cycle(self):
        """Perform a single update cycle"""
        self.logger.info("Starting single LLM knowledge update cycle")
        
        if not await self._check_daily_limit():
            self.logger.warning("Daily request limit reached, skipping update")
            return
        
        try:
            # Generate prompts for this cycle
            selected_prompts = self._select_prompts_for_cycle()
            
            # Process prompts in batches
            for batch in self._batch_prompts(selected_prompts):
                await self._process_prompt_batch(batch)
                await asyncio.sleep(2)  # Brief pause between batches
            
            self.logger.info(f"Completed update cycle, processed {len(selected_prompts)} prompts")
            
        except Exception as e:
            self.logger.error(f"Error in update cycle: {e}")

    async def update_specific_topic(self, topic: str, custom_prompt: Optional[str] = None):
        """Update knowledge on a specific topic"""
        if not await self._check_daily_limit():
            self.logger.warning("Daily request limit reached")
            return
        
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = f"Provide detailed technical information about {topic} from a cybersecurity perspective, focusing on defensive techniques and threat detection."
        
        try:
            # Use structured output based on topic
            if "vulnerability" in topic.lower() or "cve" in topic.lower():
                result = await self.openrouter.analyze_vulnerability(prompt, {"topic": topic})
                await self._store_vulnerability_knowledge(result, topic)
            
            elif "threat" in topic.lower() or "apt" in topic.lower():
                result = await self.openrouter.gather_threat_intelligence(prompt, {"topic": topic})
                await self._store_threat_intelligence(result, topic)
            
            else:
                result = await self.openrouter.generate_security_research(prompt, {"topic": topic})
                await self._store_security_research(result, topic)
            
            self.daily_request_count += 1
            self.logger.info(f"Updated knowledge for topic: {topic}")
            
        except Exception as e:
            self.logger.error(f"Failed to update topic {topic}: {e}")

    async def validate_existing_knowledge(self, max_atoms: int = 50):
        """Validate existing knowledge atoms using LLM"""
        self.logger.info("Starting knowledge validation process")
        
        try:
            # Get high-value atoms that haven't been validated recently
            atoms = await self.knowledge_fabric.get_high_value_atoms(limit=max_atoms)
            
            validation_count = 0
            for atom in atoms:
                if not await self._check_daily_limit():
                    break
                
                # Skip recently validated atoms
                recent_validations = [v for v in atom.validation_results 
                                    if (datetime.utcnow() - v.validation_timestamp).days < 30]
                if recent_validations:
                    continue
                
                await self._validate_atom_with_llm(atom)
                validation_count += 1
                
                await asyncio.sleep(1)  # Rate limiting
            
            self.logger.info(f"Validated {validation_count} knowledge atoms")
            
        except Exception as e:
            self.logger.error(f"Error in knowledge validation: {e}")

    async def _update_loop(self):
        """Main update loop"""
        self.logger.debug("LLM update loop started")
        
        while self.running:
            try:
                await self.single_update_cycle()
                
                # Reset daily counter if needed
                if datetime.utcnow().date() > self.last_reset_date:
                    self.daily_request_count = 0
                    self.last_reset_date = datetime.utcnow().date()
                    self.logger.info("Reset daily request counter")
                
                # Wait for next cycle
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    def _select_prompts_for_cycle(self) -> List[Dict[str, str]]:
        """Select prompts for the current update cycle"""
        selected = []
        
        for category, prompts in self.prompt_templates.items():
            # Select 1-2 prompts per category
            count = min(2, len(prompts))
            selected_prompts = random.sample(prompts, count)
            
            for prompt in selected_prompts:
                selected.append({
                    "category": category,
                    "prompt": prompt,
                    "model": self._select_model_for_category(category)
                })
        
        return selected[:self.batch_size]

    def _select_model_for_category(self, category: str) -> ModelProvider:
        """Select appropriate model for category"""
        model_preferences = {
            "vulnerability_research": ModelProvider.ANTHROPIC_CLAUDE,
            "technique_discovery": ModelProvider.ANTHROPIC_CLAUDE,
            "payload_research": ModelProvider.KIMI_K2,
            "defensive_intelligence": ModelProvider.GOOGLE_GEMINI,
            "threat_intelligence": ModelProvider.ANTHROPIC_CLAUDE
        }
        
        return model_preferences.get(category, ModelProvider.ANTHROPIC_CLAUDE)

    def _batch_prompts(self, prompts: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
        """Split prompts into batches"""
        batches = []
        for i in range(0, len(prompts), self.batch_size):
            batches.append(prompts[i:i + self.batch_size])
        return batches

    async def _process_prompt_batch(self, batch: List[Dict[str, str]]):
        """Process a batch of prompts"""
        try:
            for prompt_data in batch:
                if not await self._check_daily_limit():
                    break
                
                category = prompt_data["category"]
                prompt = prompt_data["prompt"]
                model = prompt_data["model"]
                
                # Use structured output based on category
                if category == "vulnerability_research":
                    result = await self.openrouter.analyze_vulnerability(prompt)
                    await self._store_vulnerability_knowledge(result, category)
                
                elif category == "threat_intelligence":
                    result = await self.openrouter.gather_threat_intelligence(prompt)
                    await self._store_threat_intelligence(result, category)
                
                else:
                    result = await self.openrouter.generate_security_research(prompt)
                    await self._store_security_research(result, category)
                
                self.daily_request_count += 1
                await asyncio.sleep(1)  # Rate limiting
                
        except Exception as e:
            self.logger.error(f"Error processing prompt batch: {e}")

    async def _store_vulnerability_knowledge(self, vuln_info, source_category: str):
        """Store vulnerability information as knowledge atoms"""
        try:
            atom = KnowledgeAtom(
                id="",
                atom_type=AtomType.VULNERABILITY,
                title=vuln_info.title,
                content={
                    "cve_id": vuln_info.cve_id,
                    "description": vuln_info.description,
                    "affected_systems": vuln_info.affected_systems,
                    "exploitation_complexity": vuln_info.exploitation_complexity,
                    "proof_of_concept": vuln_info.proof_of_concept,
                    "remediation": vuln_info.remediation,
                    "cvss_score": vuln_info.cvss_score
                },
                confidence=0.6,  # LLM-generated content gets medium confidence
                tags={source_category, "vulnerability", "llm_generated"}
            )
            
            # Add source information
            source = Source(
                name="openrouter_llm",
                type="llm",
                reliability_score=0.6,
                metadata={"category": source_category, "model": "structured_output"}
            )
            atom.add_source(source)
            
            await self.knowledge_fabric.add_atom(atom)
            self.logger.debug(f"Stored vulnerability knowledge: {vuln_info.title}")
            
        except Exception as e:
            self.logger.error(f"Failed to store vulnerability knowledge: {e}")

    async def _store_threat_intelligence(self, intel_info, source_category: str):
        """Store threat intelligence as knowledge atoms"""
        try:
            atom = KnowledgeAtom(
                id="",
                atom_type=AtomType.INTELLIGENCE,
                title=f"Threat Intelligence: {intel_info.threat_type}",
                content={
                    "threat_type": intel_info.threat_type,
                    "iocs": intel_info.iocs,
                    "ttps": intel_info.ttps,
                    "attribution": intel_info.attribution,
                    "timeline": intel_info.timeline,
                    "impact": intel_info.impact
                },
                confidence=0.5,
                tags={source_category, "threat_intelligence", "llm_generated"},
                expires_at=datetime.utcnow() + timedelta(days=30)  # Intel expires faster
            )
            
            source = Source(
                name="openrouter_llm",
                type="llm",
                reliability_score=0.5,
                metadata={"category": source_category, "type": "threat_intel"}
            )
            atom.add_source(source)
            
            await self.knowledge_fabric.add_atom(atom)
            self.logger.debug(f"Stored threat intelligence: {intel_info.threat_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to store threat intelligence: {e}")

    async def _store_security_research(self, research_info, source_category: str):
        """Store security research as knowledge atoms"""
        try:
            atom = KnowledgeAtom(
                id="",
                atom_type=AtomType.TECHNIQUE,
                title=research_info.technique_name,
                content={
                    "description": research_info.description,
                    "attack_vectors": research_info.attack_vectors,
                    "payloads": research_info.payloads,
                    "mitigation": research_info.mitigation,
                    "severity": research_info.severity,
                    "references": research_info.references
                },
                confidence=research_info.confidence,
                tags={source_category, "technique", "llm_generated"}
            )
            
            source = Source(
                name="openrouter_llm",
                type="llm",
                reliability_score=research_info.confidence,
                metadata={"category": source_category, "technique_type": research_info.technique_name}
            )
            atom.add_source(source)
            
            await self.knowledge_fabric.add_atom(atom)
            self.logger.debug(f"Stored security research: {research_info.technique_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to store security research: {e}")

    async def _validate_atom_with_llm(self, atom: KnowledgeAtom):
        """Validate an existing atom using LLM"""
        try:
            validation_prompt = f"""
            Please validate the following security information for accuracy and relevance:
            
            Title: {atom.title}
            Type: {atom.atom_type.value}
            Content: {json.dumps(atom.content, indent=2)}
            
            Provide a validation assessment focusing on:
            1. Technical accuracy
            2. Current relevance
            3. Completeness
            4. Any corrections needed
            
            Rate the overall quality from 0.0 to 1.0.
            """
            
            request = LLMRequest(
                prompt=validation_prompt,
                model=ModelProvider.ANTHROPIC_CLAUDE,
                temperature=0.3,  # Lower temperature for validation
                max_tokens=2000
            )
            
            response = await self.openrouter.chat_completion(request)
            
            # Parse validation result
            is_valid = "accurate" in response.content.lower() or "valid" in response.content.lower()
            confidence_adjustment = 0.0
            
            # Try to extract confidence score from response
            import re
            confidence_match = re.search(r'(\d+\.?\d*)', response.content)
            if confidence_match:
                try:
                    confidence_score = float(confidence_match.group(1))
                    if confidence_score <= 1.0:
                        confidence_adjustment = confidence_score - atom.confidence
                    elif confidence_score <= 10.0:
                        confidence_adjustment = (confidence_score / 10.0) - atom.confidence
                except ValueError:
                    pass
            
            validation_result = ValidationResult(
                is_valid=is_valid,
                confidence_adjustment=confidence_adjustment,
                validation_method="llm_validation",
                notes=response.content[:500]  # Truncate notes
            )
            
            await self.knowledge_fabric.validate_atom(atom.id, validation_result)
            self.daily_request_count += 1
            
            self.logger.debug(f"Validated atom {atom.id}: valid={is_valid}")
            
        except Exception as e:
            self.logger.error(f"Failed to validate atom {atom.id}: {e}")

    async def _check_daily_limit(self) -> bool:
        """Check if within daily request limit"""
        if datetime.utcnow().date() > self.last_reset_date:
            self.daily_request_count = 0
            self.last_reset_date = datetime.utcnow().date()
        
        return self.daily_request_count < self.max_daily_requests

    def get_updater_stats(self) -> Dict[str, Any]:
        """Get updater statistics"""
        return {
            "running": self.running,
            "daily_requests": self.daily_request_count,
            "max_daily_requests": self.max_daily_requests,
            "last_reset_date": self.last_reset_date.isoformat(),
            "update_interval": self.update_interval,
            "prompt_categories": list(self.prompt_templates.keys()),
            "total_prompts": sum(len(prompts) for prompts in self.prompt_templates.values())
        }


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def main():
        # Initialize components
        openrouter_client = OpenRouterClient(
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        
        knowledge_fabric = KnowledgeFabric()
        await knowledge_fabric.initialize()
        
        updater = LLMKnowledgeUpdater(openrouter_client, knowledge_fabric)
        
        try:
            if "--continuous" in sys.argv:
                await updater.start_continuous_updates()
                # Keep running
                while updater.running:
                    await asyncio.sleep(10)
            else:
                await updater.single_update_cycle()
        
        finally:
            await updater.stop()
            await openrouter_client.close()
            await knowledge_fabric.shutdown()
    
    import sys
    asyncio.run(main())