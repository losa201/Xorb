import hashlib
import json
import time
import sys
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

# Add parent directory to path for service imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'api', 'app'))
try:
    from services.base_service import SecurityService, ServiceHealth, ServiceStatus
except ImportError:
    # Fallback for when base service is not available
    SecurityService = None
    ServiceHealth = None
    ServiceStatus = None

logger = logging.getLogger("ForensicsEngine")

@dataclass
class EvidenceMetadata:
    """Metadata for forensic evidence collection"""
    case_id: str
    evidence_type: str
    source: str
    collection_method: str
    collector: str
    timestamp: str = datetime.now().isoformat()
    
@dataclass
class ChainOfCustodyEntry:
    """Record of evidence handling in the chain of custody"""
    evidence_id: str
    action: str  # "collected", "transferred", "analyzed", "stored", etc.
    actor: str
    location: str
    timestamp: str = datetime.now().isoformat()
    next_custodian: Optional[str] = None
    
    def to_dict(self):
        return {
            "evidence_id": self.evidence_id,
            "action": self.action,
            "actor": self.actor,
            "timestamp": self.timestamp,
            "location": self.location,
            "next_custodian": self.next_custodian
        }

class EvidenceChain:
    """Represents the complete chain of custody for a piece of evidence"""
    def __init__(self, evidence_id: str):
        self.evidence_id = evidence_id
        self.chain: List[Dict[str, Any]] = []
        self.current_custodian = None
        self.locked = False
        
    def add_entry(self, entry: ChainOfCustodyEntry) -> bool:
        """Add a new entry to the chain of custody"""
        if self.locked:
            logger.error(f"Chain of custody for {self.evidence_id} is locked")
            return False
            
        if entry.evidence_id != self.evidence_id:
            logger.error(f"Entry evidence ID {entry.evidence_id} does not match chain ID {self.evidence_id}")
            return False
            
        # Add hash of previous entry for integrity verification
        entry_dict = entry.to_dict()
        if self.chain:
            last_hash = self.chain[-1]["hash"]
            entry_dict["previous_hash"] = last_hash
        else:
            entry_dict["previous_hash"] = "genesis"
            
        # Calculate hash of current entry
        entry_json = json.dumps(entry_dict, sort_keys=True)
        current_hash = hashlib.sha256(entry_json.encode()).hexdigest()
        entry_dict["hash"] = current_hash
        
        self.chain.append(entry_dict)
        self.current_custodian = entry.next_custodian
        return True
        
    def lock_chain(self) -> str:
        """Lock the chain of custody with a final hash"""
        if not self.chain:
            logger.error("Cannot lock empty chain")
            return ""
            
        final_hash = self.chain[-1]["hash"]
        self.locked = True
        return final_hash
        
    def verify_chain(self) -> bool:
        """Verify the integrity of the chain of custody"""
        if not self.chain:
            return False
            
        # Verify each entry in the chain
        previous_hash = "genesis"
        
        for entry in self.chain:
            if entry["previous_hash"] != previous_hash:
                logger.error(f"Chain integrity violation at {entry['timestamp']}")
                return False
                
            # Recalculate hash
            entry_copy = entry.copy()
            current_hash = entry_copy.pop("hash", "")
            entry_json = json.dumps(entry_copy, sort_keys=True)
            calculated_hash = hashlib.sha256(entry_json.encode()).hexdigest()
            
            if calculated_hash != current_hash:
                logger.error(f"Hash mismatch at {entry['timestamp']}")
                return False
                
            previous_hash = current_hash
            
        return True

class ForensicsEngine(SecurityService if SecurityService else object):
    """Main forensics engine for automated evidence collection and tracking"""
    def __init__(self, **kwargs):
        if SecurityService:
            super().__init__(**kwargs)
        self.evidence_store = {}  # In-memory store for active evidence
        self.chain_store = {}    # In-memory store for evidence chains
        self.storage_backend = None  # For persistent storage integration
        if hasattr(super(), 'logger'):
            self.logger = super().logger
        else:
            self.logger = logging.getLogger("ForensicsEngine")
        
    def set_storage_backend(self, backend):
        """Set the persistent storage backend"""
        self.storage_backend = backend
        
    def collect_evidence(self, metadata: EvidenceMetadata, data: Dict[str, Any]) -> str:
        """Collect new evidence and store it with proper metadata"""
        # Generate unique evidence ID
        evidence_id = f"EVID-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{metadata.case_id[:4]}"
        
        # Create evidence record
        evidence_record = {
            "id": evidence_id,
            "metadata": metadata.__dict__,
            "data": data,
            "timestamp": metadata.timestamp,
            "status": "collected"
        }
        
        # Store in memory
        self.evidence_store[evidence_id] = evidence_record
        
        # Store persistently if backend is available
        if self.storage_backend:
            try:
                self.storage_backend.store_evidence(evidence_id, evidence_record)
            except Exception as e:
                self.logger.error(f"Failed to store evidence persistently: {e}")
                return ""
                
        self.logger.info(f"Collected evidence {evidence_id}")
        return evidence_id
        
    def create_chain_of_custody(self, evidence_id: str, initial_entry: ChainOfCustodyEntry) -> bool:
        """Create a new chain of custody for evidence"""
        if evidence_id not in self.evidence_store:
            self.logger.error(f"Evidence {evidence_id} not found")
            return False
            
        if evidence_id in self.chain_store:
            self.logger.error(f"Chain of custody already exists for {evidence_id}")
            return False
            
        # Create new chain
        chain = EvidenceChain(evidence_id)
        if not chain.add_entry(initial_entry):
            return False
            
        self.chain_store[evidence_id] = chain
        
        # Store persistently if backend available
        if self.storage_backend:
            try:
                self.storage_backend.store_chain_of_custody(evidence_id, chain.chain)
            except Exception as e:
                self.logger.error(f"Failed to store chain of custody: {e}")
                return False
                
        self.logger.info(f"Created chain of custody for {evidence_id}")
        return True
        
    def add_to_chain_of_custody(self, evidence_id: str, entry: ChainOfCustodyEntry) -> bool:
        """Add a new entry to an existing chain of custody"""
        if evidence_id not in self.evidence_store:
            self.logger.error(f"Evidence {evidence_id} not found")
            return False
            
        if evidence_id not in self.chain_store:
            self.logger.error(f"Chain of custody not found for {evidence_id}")
            return False
            
        chain = self.chain_store[evidence_id]
        if chain.locked:
            self.logger.error(f"Chain of custody for {evidence_id} is locked")
            return False
            
        if entry.evidence_id != evidence_id:
            self.logger.error(f"Evidence ID mismatch: {entry.evidence_id} vs {evidence_id}")
            return False
            
        # Add entry to chain
        success = chain.add_entry(entry)
        if not success:
            return False
            
        # Update persistent storage
        if self.storage_backend:
            try:
                self.storage_backend.store_chain_of_custody(evidence_id, chain.chain)
            except Exception as e:
                self.logger.error(f"Failed to update chain of custody: {e}")
                return False
                
        self.logger.info(f"Added entry to chain of custody for {evidence_id}")
        return True
        
    def lock_chain(self, evidence_id: str) -> bool:
        """Lock a chain of custody to prevent further modifications"""
        if evidence_id not in self.chain_store:
            self.logger.error(f"Chain of custody not found for {evidence_id}")
            return False
            
        chain = self.chain_store[evidence_id]
        final_hash = chain.lock_chain()
        
        # Update persistent storage
        if self.storage_backend:
            try:
                self.storage_backend.store_chain_of_custody(evidence_id, chain.chain)
            except Exception as e:
                self.logger.error(f"Failed to lock chain of custody: {e}")
                return False
                
        self.logger.info(f"Locked chain of custody for {evidence_id} with final hash {final_hash}")
        return True
        
    def verify_evidence(self, evidence_id: str) -> Dict[str, Any]:
        """Verify the integrity of evidence and its chain of custody"""
        result = {
            "evidence_id": evidence_id,
            "evidence_exists": False,
            "chain_exists": False,
            "chain_valid": False,
            "chain_locked": False
        }
        
        # Check if evidence exists
        if evidence_id in self.evidence_store:
            result["evidence_exists"] = True
            
        # Check chain of custody
        if evidence_id in self.chain_store:
            result["chain_exists"] = True
            result["chain_locked"] = self.chain_store[evidence_id].locked
            result["chain_valid"] = self.chain_store[evidence_id].verify_chain()
            
        return result
        
    def get_evidence(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve evidence by ID"""
        return self.evidence_store.get(evidence_id)
        
    async def initialize(self) -> bool:
        """Initialize the forensics engine"""
        try:
            self.logger.info("ForensicsEngine initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize ForensicsEngine: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the forensics engine"""
        try:
            # Clear sensitive data
            self.evidence_store.clear()
            self.chain_store.clear()
            self.logger.info("ForensicsEngine shutdown successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to shutdown ForensicsEngine: {e}")
            return False
    
    async def health_check(self) -> 'ServiceHealth':
        """Perform health check"""
        try:
            checks = {
                "evidence_count": len(self.evidence_store),
                "chain_count": len(self.chain_store),
                "storage_backend_available": self.storage_backend is not None
            }
            
            status = ServiceStatus.HEALTHY if ServiceStatus else "healthy"
            health = ServiceHealth(
                status=status,
                message="ForensicsEngine is operational",
                timestamp=datetime.utcnow(),
                checks=checks
            ) if ServiceHealth else {
                "status": status,
                "message": "ForensicsEngine is operational",
                "timestamp": datetime.utcnow(),
                "checks": checks
            }
            
            return health
        except Exception as e:
            status = ServiceStatus.UNHEALTHY if ServiceStatus else "unhealthy"
            return ServiceHealth(
                status=status,
                message=f"Health check failed: {e}",
                timestamp=datetime.utcnow(),
                checks={"error": str(e)}
            ) if ServiceHealth else {
                "status": status,
                "message": f"Health check failed: {e}",
                "timestamp": datetime.utcnow(),
                "checks": {"error": str(e)}
            }
    
    async def process_security_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process a security event for forensic evidence collection"""
        try:
            if "incident_id" not in event or "evidence_data" not in event:
                return {"error": "Missing required fields: incident_id, evidence_data"}
            
            # Create metadata from event
            metadata = EvidenceMetadata(
                case_id=event["incident_id"],
                evidence_type=event.get("evidence_type", "security_event"),
                source=event.get("source", "unknown"),
                collection_method="automated",
                collector="forensics_engine"
            )
            
            # Collect evidence
            evidence_id = self.collect_evidence(metadata, event["evidence_data"])
            
            # Create chain of custody
            initial_entry = ChainOfCustodyEntry(
                evidence_id=evidence_id,
                action="collected",
                actor="forensics_engine",
                location="automated_collection",
                next_custodian="security_analyst"
            )
            
            self.create_chain_of_custody(evidence_id, initial_entry)
            
            return {
                "evidence_id": evidence_id,
                "status": "collected",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to process security event: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
    
    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get security-specific metrics"""
        try:
            # Calculate chain integrity metrics
            total_chains = len(self.chain_store)
            valid_chains = sum(1 for chain in self.chain_store.values() if chain.verify_chain())
            locked_chains = sum(1 for chain in self.chain_store.values() if chain.locked)
            
            return {
                "total_evidence": len(self.evidence_store),
                "total_chains": total_chains,
                "valid_chains": valid_chains,
                "locked_chains": locked_chains,
                "chain_integrity_rate": valid_chains / total_chains if total_chains > 0 else 1.0,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to get security metrics: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
        
    def get_chain_of_custody(self, evidence_id: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve chain of custody for evidence"""
        if evidence_id in self.chain_store:
            return self.chain_store[evidence_id].chain
        return None

# Example usage
if __name__ == "__main__":
    # Initialize forensics engine
    engine = ForensicsEngine()
    
    # Collect evidence
    metadata = EvidenceMetadata(
        case_id="CASE-2023-001",
        evidence_type="network_traffic",
        source="192.168.1.100",
        collection_method="pcap",
        collector="analyst_john"
    )
    
    data = {
        "pcap_file": "base64_encoded_data",
        "start_time": "2023-01-01T10:00:00",
        "end_time": "2023-01-01T10:05:00",
        "size_bytes": 1048576
    }
    
    evidence_id = engine.collect_evidence(metadata, data)
    print(f"Collected evidence ID: {evidence_id}")
    
    # Create chain of custody
    initial_entry = ChainOfCustodyEntry(
        evidence_id=evidence_id,
        action="collected",
        actor="analyst_john",
        timestamp=datetime.now().isoformat(),
        location="HQ-Forensics-Lab",
        next_custodian="supervisor_mary"
    )
    
    engine.create_chain_of_custody(evidence_id, initial_entry)
    print("Chain of custody created")
    
    # Add to chain of custody
    analysis_entry = ChainOfCustodyEntry(
        evidence_id=evidence_id,
        action="analyzed",
        actor="analyst_sara",
        timestamp=datetime.now().isoformat(),
        location="HQ-Forensics-Lab",
        next_custodian="analyst_john"
    )
    
    engine.add_to_chain_of_custody(evidence_id, analysis_entry)
    print("Added analysis entry to chain of custody")
    
    # Lock chain
    engine.lock_chain(evidence_id)
    print("Chain of custody locked")
    
    # Verify evidence
    verification = engine.verify_evidence(evidence_id)
    print("\nEvidence Verification:")
    for key, value in verification.items():
        print(f"{key}: {value}")
    
    # Get chain of custody
    chain = engine.get_chain_of_custody(evidence_id)
    print("\nChain of Custody:")
    for entry in chain:
        print(f"{entry['timestamp']} - {entry['action']} by {entry['actor']}")
    
    # Get evidence
    evidence = engine.get_evidence(evidence_id)
    print("\nEvidence Metadata:")
    print(json.dumps(evidence["metadata"], indent=2))
    
    print("\nForensics engine test completed")
    