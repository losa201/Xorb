"""
Evidence Collection System
Implements secure evidence collection for compliance validation
"""
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging
import os
import json
from enum import Enum
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

class EvidenceType(Enum):
    """
    Types of evidence that can be collected
    """
    LOG_FILE = "log_file"
    CONFIG_FILE = "config_file"
    SCAN_RESULT = "scan_result"
    SYSTEM_INFO = "system_info"
    NETWORK_TRAFFIC = "network_traffic"
    USER_ACTIVITY = "user_activity"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_REPORT = "compliance_report"

class EvidenceSource(Enum):
    """
    Sources of evidence
    """
    SYSTEM_LOGS = "system_logs"
    SECURITY_TOOLS = "security_tools"
    NETWORK_DEVICES = "network_devices"
    APPLICATION_LOGS = "application_logs"
    CLOUD_SERVICES = "cloud_services"
    ENDPOINT_AGENTS = "endpoint_agents"
    MANUAL_UPLOAD = "manual_upload"

class EvidenceCollector:
    """
    Enterprise evidence collection implementation
    Provides secure collection, storage, and management of compliance evidence
    """

    def __init__(self, storage_path: str = "/var/xorb/evidence"):
        self.storage_path = storage_path
        self._initialize_storage()

    def _initialize_storage(self) -> None:
        """
        Initialize evidence storage directory
        """
        try:
            Path(self.storage_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Evidence storage initialized at {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to initialize evidence storage: {str(e)}")
            raise

    def collect_evidence(self,
                        evidence_type: EvidenceType,
                        source: EvidenceSource,
                        content: Union[str, bytes, Dict[str, Any]],
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Collect and store evidence with metadata
        Returns evidence record with storage information
        """
        try:
            # Create evidence record
            evidence_record = {
                "id": self._generate_evidence_id(),
                "type": evidence_type.value,
                "source": source.value,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
                "storage_info": {}
            }

            # Handle different content types
            if isinstance(content, dict):
                content_type = "json"
                content_data = json.dumps(content).encode()
            elif isinstance(content, str):
                content_type = "text"
                content_data = content.encode()
            elif isinstance(content, bytes):
                content_type = "binary"
                content_data = content
            else:
                raise ValueError("Unsupported content type")

            # Generate file path
            file_path = self._generate_file_path(evidence_record["id"], content_type)

            # Store evidence securely
            with open(file_path, "wb") as f:
                f.write(content_data)

            # Calculate hashes
            evidence_hash = self._calculate_hash(content_data)

            # Update storage info
            evidence_record["storage_info"] = {
                "file_path": str(file_path),
                "file_size": len(content_data),
                "hash": evidence_hash,
                "content_type": content_type
            }

            # Store evidence record
            self._store_evidence_record(evidence_record)

            logger.info(f"Collected evidence {evidence_record['id']} of type {evidence_type.value}")
            return evidence_record

        except Exception as e:
            logger.error(f"Failed to collect evidence: {str(e)}")
            raise

    def _generate_evidence_id(self) -> str:
        """
        Generate unique evidence ID
        """
        import uuid
        return f"EVID-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

    def _generate_file_path(self, evidence_id: str, content_type: str) -> Path:
        """
        Generate secure file path for evidence storage
        """
        # Create directory structure based on date
        date_dir = datetime.now().strftime("%Y/%m/%d")
        return Path(self.storage_path) / date_dir / f"{evidence_id}.{content_type}"

    def _calculate_hash(self, data: bytes) -> str:
        """
        Calculate SHA-256 hash of evidence data
        """
        return hashlib.sha256(data).hexdigest()

    def _store_evidence_record(self, evidence_record: Dict[str, Any]) -> None:
        """
        Store evidence record metadata
        """
        # Create record file path
        record_path = Path(self.storage_path) / "records" / f"{evidence_record['id']}.json"

        # Ensure record directory exists
        record_path.parent.mkdir(parents=True, exist_ok=True)

        # Store record
        with open(record_path, "w") as f:
            json.dump(evidence_record, f, indent=2)

    def get_evidence(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve evidence record by ID
        """
        try:
            record_path = Path(self.storage_path) / "records" / f"{evidence_id}.json"

            if not record_path.exists():
                logger.warning(f"Evidence record {evidence_id} not found")
                return None

            with open(record_path, "r") as f:
                evidence_record = json.load(f)

            # Add content preview if available
            if "storage_info" in evidence_record and "file_path" in evidence_record["storage_info"]:
                file_path = Path(evidence_record["storage_info"]["file_path"])
                if file_path.exists():
                    try:
                        # Add first 1024 bytes as preview
                        with open(file_path, "rb") as f:
                            evidence_record["content_preview"] = f.read(1024).decode(errors="ignore")
                    except Exception as e:
                        logger.warning(f"Failed to read content preview: {str(e)}")

            return evidence_record

        except Exception as e:
            logger.error(f"Failed to retrieve evidence: {str(e)}")
            raise

    def search_evidence(self,
                        evidence_type: Optional[EvidenceType] = None,
                        source: Optional[EvidenceSource] = None,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for evidence based on criteria
        """
        try:
            results = []
            records_path = Path(self.storage_path) / "records"

            if not records_path.exists():
                return []

            # Search all record files
            for record_file in records_path.glob("*.json"):
                with open(record_file, "r") as f:
                    evidence_record = json.load(f)

                # Apply filters
                if evidence_type and evidence_record.get("type") != evidence_type.value:
                    continue

                if source and evidence_record.get("source") != source.value:
                    continue

                if start_date:
                    record_time = datetime.fromisoformat(evidence_record.get("timestamp", ""))
                    if record_time < start_date:
                        continue

                if end_date:
                    record_time = datetime.fromisoformat(evidence_record.get("timestamp", ""))
                    if record_time > end_date:
                        continue

                if metadata_filter:
                    matches = True
                    for key, value in metadata_filter.items():
                        if evidence_record.get("metadata", {}).get(key) != value:
                            matches = False
                            break
                    if not matches:
                        continue

                results.append(evidence_record)

            return results

        except Exception as e:
            logger.error(f"Failed to search evidence: {str(e)}")
            raise

    def delete_evidence(self, evidence_id: str) -> bool:
        """
        Delete evidence record and associated files
        """
        try:
            # Get evidence record
            evidence_record = self.get_evidence(evidence_id)
            if not evidence_record:
                return False

            # Delete storage file
            if "storage_info" in evidence_record and "file_path" in evidence_record["storage_info"]:
                file_path = Path(evidence_record["storage_info"]["file_path"])
                if file_path.exists():
                    file_path.unlink()

            # Delete record file
            record_path = Path(self.storage_path) / "records" / f"{evidence_id}.json"
            if record_path.exists():
                record_path.unlink()

            logger.info(f"Deleted evidence {evidence_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete evidence: {str(e)}")
            return False

    def verify_evidence_integrity(self, evidence_id: str) -> Dict[str, Any]:
        """
        Verify the integrity of stored evidence
        """
        try:
            # Get evidence record
            evidence_record = self.get_evidence(evidence_id)
            if not evidence_record:
                return {
                    "valid": False,
                    "error": f"Evidence {evidence_id} not found"
                }

            # Get stored hash
            stored_hash = evidence_record.get("storage_info", {}).get("hash")
            if not stored_hash:
                return {
                    "valid": False,
                    "error": "No hash found in record"
                }

            # Read file and calculate hash
            file_path = Path(evidence_record["storage_info"]["file_path"])
            if not file_path.exists():
                return {
                    "valid": False,
                    "error": "Evidence file not found"
                }

            with open(file_path, "rb") as f:
                file_data = f.read()

            calculated_hash = self._calculate_hash(file_data)

            # Compare hashes
            valid = stored_hash == calculated_hash

            return {
                "valid": valid,
                "stored_hash": stored_hash,
                "calculated_hash": calculated_hash,
                "match": valid,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to verify evidence integrity: {str(e)}")
            return {
                "valid": False,
                "error": str(e)
            }

    def export_evidence(self, evidence_id: str, export_path: str) -> bool:
        """
        Export evidence to specified path
        """
        try:
            # Get evidence record
            evidence_record = self.get_evidence(evidence_id)
            if not evidence_record:
                return False

            # Get file path
            file_path = Path(evidence_record["storage_info"]["file_path"])
            if not file_path.exists():
                logger.warning(f"Evidence file {file_path} not found")
                return False

            # Create export path
            export_file = Path(export_path) / file_path.name

            # Copy file
            import shutil
            shutil.copy2(file_path, export_file)

            logger.info(f"Exported evidence {evidence_id} to {export_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to export evidence: {str(e)}")
            return False

# Example usage
if __name__ == '__main__':
    # Create evidence collector instance
    collector = EvidenceCollector()

    # Example metadata
    metadata = {
        "collector": "system_audit",
        "system": "firewall",
        "component": "access_control",
        "compliance_standard": "NIST"
    }

    # Example log content
    log_content = """
2025-08-10 12:00:00 INFO User admin accessed firewall configuration
2025-08-10 12:05:00 WARNING Failed login attempt from 192.168.1.100
2025-08-10 12:10:00 INFO Firewall rules updated
    """

    # Collect system log evidence
    evidence_record = collector.collect_evidence(
        evidence_type=EvidenceType.LOG_FILE,
        source=EvidenceSource.SYSTEM_LOGS,
        content=log_content,
        metadata=metadata
    )

    print("Collected Evidence:", evidence_record)

    # Verify evidence integrity
    verification = collector.verify_evidence_integrity(evidence_record["id"])
    print("\nEvidence Verification:", verification)

    # Search for evidence
    search_results = collector.search_evidence(
        evidence_type=EvidenceType.LOG_FILE,
        source=EvidenceSource.SYSTEM_LOGS
    )
    print(f"\nSearch Results ({len(search_results)} items):")
    for result in search_results:
        print(f"- {result['id']} ({result['timestamp']})")

    # Export evidence
    export_success = collector.export_evidence(evidence_record["id"], ".")
    print("\nExport Success:", export_success)

    # Delete evidence
    delete_success = collector.delete_evidence(evidence_record["id"])
    print("\nDelete Success:", delete_success)
