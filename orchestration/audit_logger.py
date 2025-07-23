#!/usr/bin/env python3

import json
import logging
import asyncio
import hashlib
import hmac
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import aiofiles
from cryptography.fernet import Fernet
import os


class AuditLogger:
    def __init__(self, log_path: str = "./logs/audit.log", encrypt_logs: bool = True, signing_key: Optional[str] = None):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.encrypt_logs = encrypt_logs
        self.signing_key = signing_key or os.getenv("LOG_SIGNING_KEY", "default-xorb-signing-key")
        
        if self.encrypt_logs:
            encryption_key = os.getenv("LOG_ENCRYPTION_KEY")
            if not encryption_key:
                encryption_key = Fernet.generate_key()
                self.logger = logging.getLogger(__name__)
                self.logger.warning("Generated new encryption key. Set LOG_ENCRYPTION_KEY environment variable.")
            else:
                encryption_key = encryption_key.encode()
            
            self.cipher_suite = Fernet(encryption_key)
        else:
            self.cipher_suite = None
        
        self.logger = logging.getLogger(__name__)
        self._setup_file_logger()

    def _setup_file_logger(self):
        file_handler = logging.FileHandler(self.log_path.with_suffix('.system.log'))
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    async def log_event(self, event_type: str, event_data: Dict[str, Any], severity: str = "INFO") -> str:
        event_id = self._generate_event_id()
        timestamp = datetime.utcnow()
        
        log_entry = {
            "event_id": event_id,
            "timestamp": timestamp.isoformat(),
            "event_type": event_type,
            "severity": severity,
            "data": event_data,
            "source": "xorb_orchestrator",
            "version": "1.0.0"
        }
        
        log_entry["signature"] = self._sign_entry(log_entry)
        
        try:
            await self._write_log_entry(log_entry)
            self.logger.debug(f"Logged audit event: {event_type} ({event_id})")
            return event_id
        except Exception as e:
            self.logger.error(f"Failed to write audit log entry: {e}")
            return ""

    async def log_security_event(self, event_type: str, event_data: Dict[str, Any], risk_level: str = "MEDIUM") -> str:
        enhanced_data = {
            **event_data,
            "risk_level": risk_level,
            "security_context": True,
            "requires_review": risk_level in ["HIGH", "CRITICAL"]
        }
        
        return await self.log_event(event_type, enhanced_data, severity="WARNING" if risk_level in ["HIGH", "CRITICAL"] else "INFO")

    async def log_roe_violation(self, violation_type: str, target: str, details: Dict[str, Any]) -> str:
        roe_data = {
            "violation_type": violation_type,
            "target": target,
            "details": details,
            "action_taken": "operation_blocked",
            "compliance_status": "violation_detected"
        }
        
        return await self.log_security_event("roe_violation", roe_data, risk_level="CRITICAL")

    async def log_finding_submission(self, campaign_id: str, finding_id: str, platform: str, status: str) -> str:
        submission_data = {
            "campaign_id": campaign_id,
            "finding_id": finding_id,
            "platform": platform,
            "submission_status": status,
            "submission_time": datetime.utcnow().isoformat()
        }
        
        return await self.log_event("finding_submission", submission_data)

    async def log_agent_activity(self, agent_id: str, agent_type: str, activity: str, target: str, result: Dict[str, Any]) -> str:
        agent_data = {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "activity": activity,
            "target": target,
            "result": result,
            "execution_time": result.get("execution_time", 0)
        }
        
        severity = "WARNING" if result.get("error") else "INFO"
        return await self.log_event("agent_activity", agent_data, severity=severity)

    async def query_logs(self, event_type: Optional[str] = None, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, severity: Optional[str] = None) -> list:
        results = []
        
        try:
            async with aiofiles.open(self.log_path, 'r') as file:
                async for line in file:
                    try:
                        log_entry = await self._parse_log_entry(line.strip())
                        
                        if self._matches_query_criteria(log_entry, event_type, start_time, end_time, severity):
                            results.append(log_entry)
                    except Exception as e:
                        self.logger.warning(f"Failed to parse log entry: {e}")
                        continue
        except FileNotFoundError:
            self.logger.info("Audit log file not found")
        
        return results

    async def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        logs = await self.query_logs(start_time=start_time, end_time=end_time)
        
        summary = {
            "total_events": len(logs),
            "security_events": 0,
            "roe_violations": 0,
            "findings_submitted": 0,
            "failed_operations": 0,
            "agents_active": set(),
            "campaigns_active": set(),
            "risk_events": {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        }
        
        for log_entry in logs:
            event_data = log_entry.get("data", {})
            
            if log_entry.get("event_type") == "roe_violation":
                summary["roe_violations"] += 1
            
            if log_entry.get("event_type") == "finding_submission":
                summary["findings_submitted"] += 1
            
            if event_data.get("security_context"):
                summary["security_events"] += 1
            
            if log_entry.get("severity") in ["ERROR", "CRITICAL"]:
                summary["failed_operations"] += 1
            
            risk_level = event_data.get("risk_level")
            if risk_level and risk_level in summary["risk_events"]:
                summary["risk_events"][risk_level] += 1
            
            if "agent_id" in event_data:
                summary["agents_active"].add(event_data["agent_id"])
            
            if "campaign_id" in event_data:
                summary["campaigns_active"].add(event_data["campaign_id"])
        
        summary["agents_active"] = len(summary["agents_active"])
        summary["campaigns_active"] = len(summary["campaigns_active"])
        
        return summary

    async def verify_log_integrity(self) -> Dict[str, Any]:
        verification_results = {
            "total_entries": 0,
            "valid_signatures": 0,
            "invalid_signatures": 0,
            "corrupted_entries": 0,
            "integrity_score": 0.0
        }
        
        try:
            async with aiofiles.open(self.log_path, 'r') as file:
                async for line in file:
                    verification_results["total_entries"] += 1
                    try:
                        log_entry = await self._parse_log_entry(line.strip())
                        
                        if self._verify_signature(log_entry):
                            verification_results["valid_signatures"] += 1
                        else:
                            verification_results["invalid_signatures"] += 1
                    except Exception:
                        verification_results["corrupted_entries"] += 1
        except FileNotFoundError:
            pass
        
        if verification_results["total_entries"] > 0:
            verification_results["integrity_score"] = verification_results["valid_signatures"] / verification_results["total_entries"]
        
        return verification_results

    def _generate_event_id(self) -> str:
        import uuid
        return str(uuid.uuid4())

    def _sign_entry(self, log_entry: Dict[str, Any]) -> str:
        entry_copy = {k: v for k, v in log_entry.items() if k != "signature"}
        entry_string = json.dumps(entry_copy, sort_keys=True, default=str)
        
        return hmac.new(
            self.signing_key.encode(),
            entry_string.encode(),
            hashlib.sha256
        ).hexdigest()

    def _verify_signature(self, log_entry: Dict[str, Any]) -> bool:
        if "signature" not in log_entry:
            return False
        
        provided_signature = log_entry.pop("signature")
        expected_signature = self._sign_entry(log_entry)
        log_entry["signature"] = provided_signature
        
        return hmac.compare_digest(provided_signature, expected_signature)

    async def _write_log_entry(self, log_entry: Dict[str, Any]):
        log_line = json.dumps(log_entry, default=str, separators=(',', ':'))
        
        if self.encrypt_logs and self.cipher_suite:
            log_line = self.cipher_suite.encrypt(log_line.encode()).decode()
        
        async with aiofiles.open(self.log_path, 'a') as file:
            await file.write(log_line + '\n')

    async def _parse_log_entry(self, line: str) -> Dict[str, Any]:
        if self.encrypt_logs and self.cipher_suite:
            try:
                decrypted_line = self.cipher_suite.decrypt(line.encode()).decode()
                return json.loads(decrypted_line)
            except Exception as e:
                raise ValueError(f"Failed to decrypt log entry: {e}")
        else:
            return json.loads(line)

    def _matches_query_criteria(self, log_entry: Dict[str, Any], event_type: Optional[str], start_time: Optional[datetime], end_time: Optional[datetime], severity: Optional[str]) -> bool:
        if event_type and log_entry.get("event_type") != event_type:
            return False
        
        if severity and log_entry.get("severity") != severity:
            return False
        
        entry_time = datetime.fromisoformat(log_entry.get("timestamp", ""))
        if start_time and entry_time < start_time:
            return False
        
        if end_time and entry_time > end_time:
            return False
        
        return True