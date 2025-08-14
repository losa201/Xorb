#!/usr/bin/env python3
"""
XORB Phase G7 Merkle Tree Roll-up Job
Weekly job to create Merkle tree roll-ups of evidence for immutable audit trails.

Creates cryptographic proof of all evidence in a time period, enabling
efficient verification of large evidence sets without storing all individual proofs.
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import argparse

# Merkle tree implementation
class MerkleNode:
    """Node in a Merkle tree."""
    
    def __init__(self, hash_value: str, left: Optional['MerkleNode'] = None, right: Optional['MerkleNode'] = None):
        self.hash = hash_value
        self.left = left
        self.right = right
        self.is_leaf = left is None and right is None
    
    def __str__(self):
        return f"Node({self.hash[:8]}...)"


@dataclass
class EvidenceLeaf:
    """Evidence entry for Merkle tree."""
    evidence_id: str
    tenant_id: str
    content_hash: str
    created_at: datetime
    signature_hash: str  # Hash of Ed25519 signature
    
    def to_hash_input(self) -> str:
        """Convert to string for hashing."""
        return f"{self.evidence_id}:{self.tenant_id}:{self.content_hash}:{self.created_at.isoformat()}:{self.signature_hash}"


@dataclass
class MerkleProof:
    """Merkle proof for evidence inclusion."""
    evidence_id: str
    leaf_hash: str
    root_hash: str
    proof_hashes: List[str]  # Sibling hashes from leaf to root
    proof_positions: List[bool]  # True = right sibling, False = left sibling
    
    def verify(self, root_hash: str) -> bool:
        """Verify this proof against a known root hash."""
        current_hash = self.leaf_hash
        
        for i, (proof_hash, is_right) in enumerate(zip(self.proof_hashes, self.proof_positions)):
            if is_right:
                # Current hash is left, proof hash is right
                combined = current_hash + proof_hash
            else:
                # Current hash is right, proof hash is left  
                combined = proof_hash + current_hash
            
            current_hash = hashlib.sha256(combined.encode()).hexdigest()
        
        return current_hash == root_hash


class MerkleTree:
    """Merkle tree implementation for evidence integrity."""
    
    def __init__(self, evidence_leaves: List[EvidenceLeaf]):
        self.leaves = evidence_leaves
        self.root = None
        self.leaf_nodes = {}  # evidence_id -> MerkleNode
        self._build_tree()
    
    def _build_tree(self):
        """Build the Merkle tree from evidence leaves."""
        if not self.leaves:
            self.root = MerkleNode("empty_tree")
            return
        
        # Create leaf nodes
        current_level = []
        for leaf in self.leaves:
            leaf_hash = hashlib.sha256(leaf.to_hash_input().encode()).hexdigest()
            node = MerkleNode(leaf_hash)
            current_level.append(node)
            self.leaf_nodes[leaf.evidence_id] = node
        
        # Build tree bottom-up
        while len(current_level) > 1:
            next_level = []
            
            # Process pairs
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                
                if i + 1 < len(current_level):
                    right = current_level[i + 1]
                else:
                    # Odd number of nodes - duplicate the last one
                    right = left
                
                # Create parent node
                combined_hash = left.hash + right.hash
                parent_hash = hashlib.sha256(combined_hash.encode()).hexdigest()
                parent = MerkleNode(parent_hash, left, right)
                
                next_level.append(parent)
            
            current_level = next_level
        
        self.root = current_level[0]
    
    def get_root_hash(self) -> str:
        """Get the root hash of the tree."""
        return self.root.hash if self.root else "empty_tree"
    
    def generate_proof(self, evidence_id: str) -> Optional[MerkleProof]:
        """Generate Merkle proof for evidence inclusion."""
        if evidence_id not in self.leaf_nodes:
            return None
        
        leaf_node = self.leaf_nodes[evidence_id]
        proof_hashes = []
        proof_positions = []
        
        current = leaf_node
        
        # Traverse up to root, collecting sibling hashes
        while True:
            # Find parent that contains this node
            parent = self._find_parent(current)
            if not parent:
                break
            
            # Get sibling
            if parent.left == current:
                # Current is left child, sibling is right
                sibling = parent.right
                proof_positions.append(True)  # Sibling is on the right
            else:
                # Current is right child, sibling is left
                sibling = parent.left
                proof_positions.append(False)  # Sibling is on the left
            
            proof_hashes.append(sibling.hash)
            current = parent
        
        return MerkleProof(
            evidence_id=evidence_id,
            leaf_hash=leaf_node.hash,
            root_hash=self.get_root_hash(),
            proof_hashes=proof_hashes,
            proof_positions=proof_positions
        )
    
    def _find_parent(self, node: MerkleNode) -> Optional[MerkleNode]:
        """Find parent of a node (inefficient but works for demo)."""
        return self._find_parent_recursive(self.root, node)
    
    def _find_parent_recursive(self, current: MerkleNode, target: MerkleNode) -> Optional[MerkleNode]:
        """Recursively find parent of target node."""
        if current.is_leaf:
            return None
        
        if current.left == target or current.right == target:
            return current
        
        # Search left subtree
        if current.left:
            result = self._find_parent_recursive(current.left, target)
            if result:
                return result
        
        # Search right subtree  
        if current.right:
            result = self._find_parent_recursive(current.right, target)
            if result:
                return result
        
        return None


@dataclass
class RollupSummary:
    """Summary of a Merkle roll-up operation."""
    rollup_id: str
    period_start: datetime
    period_end: datetime
    total_evidence: int
    tenant_counts: Dict[str, int]
    merkle_root: str
    created_at: datetime
    storage_locations: List[str]


class G7MerkleRollupService:
    """Service for creating weekly Merkle roll-ups."""
    
    def __init__(self, evidence_storage_path: str = "evidence_storage", rollup_storage_path: str = "rollup_storage"):
        self.evidence_storage_path = Path(evidence_storage_path)
        self.rollup_storage_path = Path(rollup_storage_path)
        self.rollup_storage_path.mkdir(exist_ok=True)
    
    async def collect_evidence_for_period(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[EvidenceLeaf]:
        """Collect all evidence created in the specified time period."""
        evidence_leaves = []
        
        if not self.evidence_storage_path.exists():
            print("⚠️ Evidence storage path does not exist")
            return evidence_leaves
        
        # Scan all tenant directories
        for tenant_dir in self.evidence_storage_path.iterdir():
            if not tenant_dir.is_dir():
                continue
            
            tenant_id = tenant_dir.name
            
            # Scan evidence files in tenant directory
            for evidence_file in tenant_dir.glob("*.json"):
                try:
                    with open(evidence_file, 'r') as f:
                        evidence_data = json.load(f)
                    
                    metadata = evidence_data.get("metadata", {})
                    signature = evidence_data.get("signature", {})
                    
                    created_at = datetime.fromisoformat(metadata.get("created_at", ""))
                    
                    # Check if evidence is in our time period
                    if start_date <= created_at <= end_date:
                        leaf = EvidenceLeaf(
                            evidence_id=metadata.get("evidence_id", ""),
                            tenant_id=tenant_id,
                            content_hash=metadata.get("content_hash", ""),
                            created_at=created_at,
                            signature_hash=hashlib.sha256(signature.get("signature", "").encode()).hexdigest()
                        )
                        evidence_leaves.append(leaf)
                        
                except Exception as e:
                    print(f"⚠️ Failed to process evidence file {evidence_file}: {e}")
                    continue
        
        print(f"📋 Collected {len(evidence_leaves)} evidence items for period {start_date.date()} to {end_date.date()}")
        return evidence_leaves
    
    async def create_weekly_rollup(self, week_offset: int = 0) -> RollupSummary:
        """Create Merkle roll-up for a specific week."""
        
        # Calculate week boundaries
        now = datetime.now(timezone.utc)
        week_start = now - timedelta(days=now.weekday() + 7 * week_offset + 7)  # Previous week
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        week_end = week_start + timedelta(days=7) - timedelta(microseconds=1)
        
        rollup_id = f"rollup_{week_start.strftime('%Y_%m_%d')}_to_{week_end.strftime('%Y_%m_%d')}"
        
        print(f"🌳 Creating Merkle roll-up: {rollup_id}")
        print(f"📅 Period: {week_start.date()} to {week_end.date()}")
        
        # Collect evidence for the period
        evidence_leaves = await self.collect_evidence_for_period(week_start, week_end)
        
        if not evidence_leaves:
            print("⚠️ No evidence found for this period")
            # Create empty rollup
            return RollupSummary(
                rollup_id=rollup_id,
                period_start=week_start,
                period_end=week_end,
                total_evidence=0,
                tenant_counts={},
                merkle_root="empty_tree",
                created_at=datetime.now(timezone.utc),
                storage_locations=[]
            )
        
        # Sort evidence by created_at for consistent ordering
        evidence_leaves.sort(key=lambda x: x.created_at)
        
        # Build Merkle tree
        print("🌳 Building Merkle tree...")
        merkle_tree = MerkleTree(evidence_leaves)
        merkle_root = merkle_tree.get_root_hash()
        
        print(f"🌳 Merkle root: {merkle_root}")
        
        # Calculate tenant statistics
        tenant_counts = {}
        for leaf in evidence_leaves:
            tenant_counts[leaf.tenant_id] = tenant_counts.get(leaf.tenant_id, 0) + 1
        
        # Create rollup data
        rollup_data = {
            "rollup_id": rollup_id,
            "version": "g7.1.0",
            "period": {
                "start": week_start.isoformat(),
                "end": week_end.isoformat()
            },
            "statistics": {
                "total_evidence": len(evidence_leaves),
                "tenant_counts": tenant_counts,
                "evidence_types": self._count_by_type(evidence_leaves)
            },
            "merkle_tree": {
                "root_hash": merkle_root,
                "leaf_count": len(evidence_leaves),
                "tree_height": self._calculate_tree_height(len(evidence_leaves))
            },
            "evidence_list": [
                {
                    "evidence_id": leaf.evidence_id,
                    "tenant_id": leaf.tenant_id,
                    "content_hash": leaf.content_hash,
                    "created_at": leaf.created_at.isoformat(),
                    "leaf_hash": hashlib.sha256(leaf.to_hash_input().encode()).hexdigest()
                }
                for leaf in evidence_leaves
            ],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": "g7_merkle_rollup_job"
        }
        
        # Generate Merkle proofs for all evidence
        print("🔍 Generating Merkle proofs...")
        proofs = {}
        for leaf in evidence_leaves:
            proof = merkle_tree.generate_proof(leaf.evidence_id)
            if proof:
                proofs[leaf.evidence_id] = {
                    "leaf_hash": proof.leaf_hash,
                    "root_hash": proof.root_hash,
                    "proof_hashes": proof.proof_hashes,
                    "proof_positions": proof.proof_positions
                }
        
        rollup_data["merkle_proofs"] = proofs
        
        # Store rollup data
        storage_locations = await self._store_rollup(rollup_id, rollup_data)
        
        # Create summary
        summary = RollupSummary(
            rollup_id=rollup_id,
            period_start=week_start,
            period_end=week_end,
            total_evidence=len(evidence_leaves),
            tenant_counts=tenant_counts,
            merkle_root=merkle_root,
            created_at=datetime.now(timezone.utc),
            storage_locations=storage_locations
        )
        
        print(f"✅ Merkle roll-up completed: {rollup_id}")
        print(f"📊 Total evidence: {len(evidence_leaves)}")
        print(f"🏢 Tenants: {len(tenant_counts)}")
        print(f"🌳 Merkle root: {merkle_root[:16]}...")
        
        return summary
    
    def _count_by_type(self, evidence_leaves: List[EvidenceLeaf]) -> Dict[str, int]:
        """Count evidence by type (simplified - would need to load full metadata)."""
        # In a real implementation, this would analyze the evidence metadata
        return {"various_types": len(evidence_leaves)}
    
    def _calculate_tree_height(self, leaf_count: int) -> int:
        """Calculate height of Merkle tree."""
        if leaf_count <= 1:
            return 0
        
        import math
        return math.ceil(math.log2(leaf_count))
    
    async def _store_rollup(self, rollup_id: str, rollup_data: Dict[str, Any]) -> List[str]:
        """Store rollup data in multiple locations."""
        storage_locations = []
        
        # 1. Local filesystem storage
        rollup_file = self.rollup_storage_path / f"{rollup_id}.json"
        with open(rollup_file, 'w') as f:
            json.dump(rollup_data, f, indent=2, default=str)
        
        storage_locations.append(f"file://{rollup_file}")
        print(f"💾 Rollup stored locally: {rollup_file}")
        
        # 2. IPFS storage (if available)
        try:
            import ipfshttpclient
            ipfs_client = ipfshttpclient.connect("/ip4/127.0.0.1/tcp/5001")
            
            rollup_json = json.dumps(rollup_data, default=str)
            ipfs_result = ipfs_client.add(rollup_json)
            ipfs_hash = ipfs_result["Hash"]
            
            storage_locations.append(f"ipfs://{ipfs_hash}")
            print(f"📦 Rollup stored in IPFS: {ipfs_hash}")
            
        except Exception as e:
            print(f"⚠️ IPFS storage failed: {e}")
        
        # 3. Blockchain storage (mock implementation)
        try:
            # In production, this would submit to a blockchain
            blockchain_tx = f"0x{hashlib.sha256(rollup_id.encode()).hexdigest()[:32]}"
            storage_locations.append(f"blockchain://{blockchain_tx}")
            print(f"⛓️ Rollup hash submitted to blockchain (mock): {blockchain_tx}")
            
        except Exception as e:
            print(f"⚠️ Blockchain submission failed: {e}")
        
        return storage_locations
    
    async def verify_rollup(self, rollup_id: str, evidence_id: str) -> Dict[str, Any]:
        """Verify that specific evidence is included in a rollup."""
        rollup_file = self.rollup_storage_path / f"{rollup_id}.json"
        
        if not rollup_file.exists():
            return {"error": f"Rollup {rollup_id} not found"}
        
        try:
            with open(rollup_file, 'r') as f:
                rollup_data = json.load(f)
            
            # Get Merkle proof for evidence
            proof_data = rollup_data.get("merkle_proofs", {}).get(evidence_id)
            if not proof_data:
                return {"error": f"Evidence {evidence_id} not found in rollup"}
            
            # Create MerkleProof object
            proof = MerkleProof(
                evidence_id=evidence_id,
                leaf_hash=proof_data["leaf_hash"],
                root_hash=proof_data["root_hash"],
                proof_hashes=proof_data["proof_hashes"],
                proof_positions=proof_data["proof_positions"]
            )
            
            # Verify proof
            merkle_root = rollup_data["merkle_tree"]["root_hash"]
            is_valid = proof.verify(merkle_root)
            
            return {
                "rollup_id": rollup_id,
                "evidence_id": evidence_id,
                "merkle_root": merkle_root,
                "proof_valid": is_valid,
                "proof_steps": len(proof.proof_hashes),
                "verified_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {"error": f"Verification failed: {str(e)}"}


async def main():
    """Main function for running Merkle roll-up job."""
    parser = argparse.ArgumentParser(description="XORB G7 Merkle Tree Roll-up Job")
    parser.add_argument("--weeks-back", type=int, default=0, help="Number of weeks back to process (0 = last week)")
    parser.add_argument("--evidence-storage", default="evidence_storage", help="Evidence storage directory")
    parser.add_argument("--rollup-storage", default="rollup_storage", help="Rollup storage directory")
    parser.add_argument("--verify", help="Verify evidence in rollup (format: rollup_id:evidence_id)")
    parser.add_argument("--list-rollups", action="store_true", help="List all available rollups")
    
    args = parser.parse_args()
    
    service = G7MerkleRollupService(
        evidence_storage_path=args.evidence_storage,
        rollup_storage_path=args.rollup_storage
    )
    
    if args.verify:
        # Verify specific evidence in rollup
        try:
            rollup_id, evidence_id = args.verify.split(":", 1)
            result = await service.verify_rollup(rollup_id, evidence_id)
            print(json.dumps(result, indent=2))
        except ValueError:
            print("❌ Verify format should be 'rollup_id:evidence_id'")
        return
    
    if args.list_rollups:
        # List all available rollups
        rollup_files = list(service.rollup_storage_path.glob("*.json"))
        if not rollup_files:
            print("📋 No rollups found")
        else:
            print(f"📋 Found {len(rollup_files)} rollups:")
            for rollup_file in sorted(rollup_files):
                rollup_id = rollup_file.stem
                print(f"  • {rollup_id}")
        return
    
    # Create weekly rollup
    try:
        summary = await service.create_weekly_rollup(week_offset=args.weeks_back)
        
        print("\n" + "=" * 60)
        print("📊 XORB G7 Merkle Roll-up Summary")
        print("=" * 60)
        print(f"🆔 Rollup ID: {summary.rollup_id}")
        print(f"📅 Period: {summary.period_start.date()} to {summary.period_end.date()}")
        print(f"📋 Total Evidence: {summary.total_evidence}")
        print(f"🏢 Tenants: {len(summary.tenant_counts)}")
        
        if summary.tenant_counts:
            print("📊 Evidence by Tenant:")
            for tenant_id, count in summary.tenant_counts.items():
                print(f"  • {tenant_id}: {count} evidence items")
        
        print(f"🌳 Merkle Root: {summary.merkle_root}")
        print(f"💾 Storage Locations: {len(summary.storage_locations)}")
        
        for location in summary.storage_locations:
            print(f"  • {location}")
        
        print(f"🕐 Created: {summary.created_at.isoformat()}")
        print("\n✅ Weekly Merkle roll-up completed successfully")
        
    except Exception as e:
        print(f"❌ Merkle roll-up failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)