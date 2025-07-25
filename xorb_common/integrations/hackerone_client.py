#!/usr/bin/env python3

import asyncio
import logging
import json
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import aiohttp
from pydantic import BaseModel, Field


class SubmissionStatus(str):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    TRIAGED = "triaged"
    RESOLVED = "resolved"
    DUPLICATE = "duplicate"
    INFORMATIVE = "informative"
    NOT_APPLICABLE = "not-applicable"
    SPAM = "spam"


@dataclass
class HackerOneProgram:
    id: str
    handle: str
    name: str
    url: str
    state: str
    submission_state: str
    triage_active: bool
    policy: str
    scopes: List[Dict[str, Any]]
    bounty_enabled: bool
    average_bounty_lower_amount: Optional[float] = None
    average_bounty_upper_amount: Optional[float] = None


class VulnerabilitySubmission(BaseModel):
    title: str
    description: str
    impact: str
    severity_rating: str  # none, low, medium, high, critical
    
    program_handle: str
    weakness_id: Optional[int] = None  # CWE ID
    
    proof_of_concept: Optional[str] = None
    steps_to_reproduce: Optional[str] = None
    supporting_material: List[str] = Field(default_factory=list)
    
    cvss_vector: Optional[str] = None
    cvss_score: Optional[float] = None
    
    structured_scope_id: Optional[str] = None
    asset_identifier: Optional[str] = None
    
    collaboration: bool = False
    team_members: List[str] = Field(default_factory=list)


class HackerOneClient:
    def __init__(self, api_key: str, username: str = "xorb", base_url: str = "https://api.hackerone.com/v1"):
        self.api_key = api_key
        self.username = username
        self.base_url = base_url
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.rate_limit = 100  # requests per hour
        self.rate_window = 3600  # seconds
        self.request_times: List[datetime] = []
        
        # Submission tracking
        self.submitted_reports: Dict[str, Dict[str, Any]] = {}
        self.earnings_total = 0.0

    async def start(self):
        """Initialize the HTTP session"""
        if not self.session:
            auth_string = base64.b64encode(f"{self.username}:{self.api_key}".encode()).decode()
            
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Basic {auth_string}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "XORB Security Platform/1.0"
                },
                timeout=aiohttp.ClientTimeout(total=60)
            )
            
            self.logger.info("HackerOne client session started")

    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def get_programs(self, eligible_only: bool = True) -> List[HackerOneProgram]:
        """Get list of available bug bounty programs"""
        if not await self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        if not self.session:
            await self.start()
        
        try:
            params = {}
            if eligible_only:
                params["filter[submission_state]"] = "open"
            
            async with self.session.get(f"{self.base_url}/programs", params=params) as response:
                await self._handle_response_errors(response)
                data = await response.json()
                
                programs = []
                for program_data in data.get("data", []):
                    attributes = program_data.get("attributes", {})
                    relationships = program_data.get("relationships", {})
                    
                    # Extract structured scopes
                    scopes = []
                    if "structured_scopes" in relationships:
                        scope_data = relationships["structured_scopes"].get("data", [])
                        scopes = [scope.get("attributes", {}) for scope in scope_data]
                    
                    program = HackerOneProgram(
                        id=program_data.get("id", ""),
                        handle=attributes.get("handle", ""),
                        name=attributes.get("name", ""),
                        url=attributes.get("url", ""),
                        state=attributes.get("state", ""),
                        submission_state=attributes.get("submission_state", ""),
                        triage_active=attributes.get("triage_active", False),
                        policy=attributes.get("policy", ""),
                        scopes=scopes,
                        bounty_enabled=attributes.get("offers_bounties", False),
                        average_bounty_lower_amount=attributes.get("average_bounty_lower_amount"),
                        average_bounty_upper_amount=attributes.get("average_bounty_upper_amount")
                    )
                    programs.append(program)
                
                self.logger.info(f"Retrieved {len(programs)} programs")
                return programs
                
        except Exception as e:
            self.logger.error(f"Failed to get programs: {e}")
            raise

    async def get_program_scopes(self, program_handle: str) -> List[Dict[str, Any]]:
        """Get scopes for a specific program"""
        if not await self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        if not self.session:
            await self.start()
        
        try:
            async with self.session.get(f"{self.base_url}/programs/{program_handle}") as response:
                await self._handle_response_errors(response)
                data = await response.json()
                
                relationships = data.get("data", {}).get("relationships", {})
                
                scopes = []
                if "structured_scopes" in relationships:
                    scope_ids = [item["id"] for item in relationships["structured_scopes"].get("data", [])]
                    
                    # Get detailed scope information
                    for scope_id in scope_ids:
                        scope_details = await self._get_scope_details(scope_id)
                        if scope_details:
                            scopes.append(scope_details)
                
                self.logger.info(f"Retrieved {len(scopes)} scopes for program {program_handle}")
                return scopes
                
        except Exception as e:
            self.logger.error(f"Failed to get scopes for {program_handle}: {e}")
            return []

    async def submit_report(self, submission: VulnerabilitySubmission) -> Dict[str, Any]:
        """Submit a vulnerability report"""
        if not await self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        if not self.session:
            await self.start()
        
        try:
            # Prepare submission payload
            payload = {
                "data": {
                    "type": "report",
                    "attributes": {
                        "title": submission.title,
                        "vulnerability_information": submission.description,
                        "impact": submission.impact,
                        "severity_rating": submission.severity_rating,
                        "weakness": {"id": submission.weakness_id} if submission.weakness_id else None
                    },
                    "relationships": {
                        "program": {
                            "data": {
                                "type": "program",
                                "attributes": {"handle": submission.program_handle}
                            }
                        }
                    }
                }
            }
            
            # Add optional fields
            if submission.structured_scope_id:
                payload["data"]["relationships"]["structured_scope"] = {
                    "data": {"type": "structured-scope", "id": submission.structured_scope_id}
                }
            
            if submission.cvss_vector:
                payload["data"]["attributes"]["cvss_vector"] = submission.cvss_vector
            
            async with self.session.post(f"{self.base_url}/reports", json=payload) as response:
                await self._handle_response_errors(response)
                data = await response.json()
                
                report_data = data.get("data", {})
                report_id = report_data.get("id")
                attributes = report_data.get("attributes", {})
                
                result = {
                    "report_id": report_id,
                    "state": attributes.get("state"),
                    "title": attributes.get("title"),
                    "submitted_at": attributes.get("created_at"),
                    "program_handle": submission.program_handle,
                    "url": f"https://hackerone.com/reports/{report_id}"
                }
                
                # Track submission
                self.submitted_reports[report_id] = {
                    "submission": submission.dict(),
                    "result": result,
                    "submitted_at": datetime.utcnow()
                }
                
                self.logger.info(f"Successfully submitted report {report_id} to {submission.program_handle}")
                return result
                
        except Exception as e:
            self.logger.error(f"Failed to submit report to {submission.program_handle}: {e}")
            raise

    async def get_report_status(self, report_id: str) -> Dict[str, Any]:
        """Get status of a submitted report"""
        if not await self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        if not self.session:
            await self.start()
        
        try:
            async with self.session.get(f"{self.base_url}/reports/{report_id}") as response:
                await self._handle_response_errors(response)
                data = await response.json()
                
                attributes = data.get("data", {}).get("attributes", {})
                relationships = data.get("data", {}).get("relationships", {})
                
                # Extract bounty information if available
                bounty_info = None
                if "bounties" in relationships:
                    bounty_data = relationships["bounties"].get("data", [])
                    if bounty_data:
                        # Get bounty details (simplified - would need additional API calls for full details)
                        bounty_info = {
                            "awarded": len(bounty_data) > 0,
                            "count": len(bounty_data)
                        }
                
                return {
                    "report_id": report_id,
                    "state": attributes.get("state"),
                    "title": attributes.get("title"),
                    "created_at": attributes.get("created_at"),
                    "updated_at": attributes.get("updated_at"),
                    "triaged_at": attributes.get("triaged_at"),
                    "closed_at": attributes.get("closed_at"),
                    "bounty": bounty_info,
                    "activity_count": attributes.get("activities_count", 0)
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get report status for {report_id}: {e}")
            return {}

    async def get_earnings_summary(self) -> Dict[str, Any]:
        """Get earnings summary from bounties"""
        try:
            # This would require additional API endpoints and proper authentication
            # For now, return tracked earnings
            total_submissions = len(self.submitted_reports)
            
            return {
                "total_submissions": total_submissions,
                "total_earnings": self.earnings_total,
                "average_per_report": self.earnings_total / max(1, total_submissions),
                "submitted_programs": list(set([
                    report["submission"]["program_handle"] 
                    for report in self.submitted_reports.values()
                ])),
                "last_submission": max([
                    report["submitted_at"] 
                    for report in self.submitted_reports.values()
                ], default=None)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get earnings summary: {e}")
            return {}

    async def validate_scope_match(self, program_handle: str, target: str) -> bool:
        """Validate if a target matches program scope"""
        try:
            scopes = await self.get_program_scopes(program_handle)
            
            for scope in scopes:
                asset_identifier = scope.get("asset_identifier", "")
                asset_type = scope.get("asset_type", "")
                eligible_for_submission = scope.get("eligible_for_submission", False)
                
                if not eligible_for_submission:
                    continue
                
                # Basic scope matching logic
                if asset_type == "DOMAIN":
                    if target in asset_identifier or asset_identifier in target:
                        return True
                elif asset_type == "URL":
                    if target.startswith(asset_identifier) or asset_identifier.startswith(target):
                        return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to validate scope for {target} in {program_handle}: {e}")
            return False

    async def find_matching_programs(self, target: str) -> List[str]:
        """Find programs that might accept reports for a given target"""
        try:
            programs = await self.get_programs(eligible_only=True)
            matching_programs = []
            
            for program in programs:
                if await self.validate_scope_match(program.handle, target):
                    matching_programs.append(program.handle)
            
            self.logger.info(f"Found {len(matching_programs)} matching programs for {target}")
            return matching_programs
            
        except Exception as e:
            self.logger.error(f"Failed to find matching programs for {target}: {e}")
            return []

    async def create_submission_from_finding(self, finding_data: Dict[str, Any], program_handle: str) -> VulnerabilitySubmission:
        """Create HackerOne submission from XORB finding"""
        # Map XORB severity to HackerOne severity
        severity_mapping = {
            "critical": "critical",
            "high": "high", 
            "medium": "medium",
            "low": "low",
            "info": "none"
        }
        
        severity = severity_mapping.get(finding_data.get("severity", "medium").lower(), "medium")
        
        # Extract CWE from finding if available
        weakness_id = None
        if "cwe" in finding_data.get("tags", []):
            # Try to extract CWE ID from tags or content
            pass
        
        submission = VulnerabilitySubmission(
            title=finding_data.get("title", "Security Vulnerability"),
            description=finding_data.get("description", ""),
            impact=finding_data.get("impact", "Security vulnerability that could affect confidentiality, integrity, or availability."),
            severity_rating=severity,
            program_handle=program_handle,
            weakness_id=weakness_id,
            proof_of_concept=finding_data.get("proof_of_concept"),
            cvss_vector=finding_data.get("cvss_vector"),
            cvss_score=finding_data.get("cvss_score"),
            asset_identifier=finding_data.get("affected_targets", [""])[0] if finding_data.get("affected_targets") else None
        )
        
        return submission

    async def batch_submit_findings(self, findings: List[Dict[str, Any]], target_programs: Dict[str, str] = None) -> Dict[str, Any]:
        """Submit multiple findings to appropriate programs"""
        results = {"successful": [], "failed": [], "skipped": []}
        
        for finding in findings:
            try:
                # Find target programs if not specified
                if target_programs and finding.get("id") in target_programs:
                    program_handle = target_programs[finding["id"]]
                else:
                    # Auto-detect program based on targets
                    targets = finding.get("affected_targets", [])
                    if not targets:
                        results["skipped"].append({
                            "finding_id": finding.get("id"),
                            "reason": "no_targets"
                        })
                        continue
                    
                    matching_programs = await self.find_matching_programs(targets[0])
                    if not matching_programs:
                        results["skipped"].append({
                            "finding_id": finding.get("id"),
                            "reason": "no_matching_programs"
                        })
                        continue
                    
                    program_handle = matching_programs[0]  # Use first match
                
                # Create and submit
                submission = await self.create_submission_from_finding(finding, program_handle)
                result = await self.submit_report(submission)
                
                results["successful"].append({
                    "finding_id": finding.get("id"),
                    "report_id": result["report_id"],
                    "program": program_handle
                })
                
                # Rate limiting between submissions
                await asyncio.sleep(5)
                
            except Exception as e:
                results["failed"].append({
                    "finding_id": finding.get("id"),
                    "error": str(e)
                })
        
        return results

    async def _get_scope_details(self, scope_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed scope information"""
        try:
            async with self.session.get(f"{self.base_url}/structured_scopes/{scope_id}") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", {}).get("attributes", {})
        except Exception:
            pass
        return None

    async def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = datetime.utcnow()
        
        # Remove old requests outside the window
        cutoff = now - timedelta(seconds=self.rate_window)
        self.request_times = [t for t in self.request_times if t > cutoff]
        
        if len(self.request_times) >= self.rate_limit:
            self.logger.warning("Rate limit reached, request denied")
            return False
        
        self.request_times.append(now)
        return True

    async def _handle_response_errors(self, response: aiohttp.ClientResponse):
        """Handle API response errors"""
        if response.status >= 400:
            error_text = await response.text()
            self.logger.error(f"API error {response.status}: {error_text}")
            
            if response.status == 401:
                raise Exception("Authentication failed - check API credentials")
            elif response.status == 403:
                raise Exception("Access forbidden - insufficient permissions")
            elif response.status == 404:
                raise Exception("Resource not found")
            elif response.status == 422:
                raise Exception(f"Validation error: {error_text}")
            elif response.status == 429:
                raise Exception("Rate limit exceeded")
            else:
                raise Exception(f"API error {response.status}: {error_text}")

    def get_client_stats(self) -> Dict[str, Any]:
        """Get client usage statistics"""
        return {
            "total_submissions": len(self.submitted_reports),
            "total_earnings": self.earnings_total,
            "recent_requests": len(self.request_times),
            "rate_limit": self.rate_limit,
            "submitted_programs": list(set([
                report["submission"]["program_handle"] 
                for report in self.submitted_reports.values()
            ])),
            "session_active": self.session is not None
        }