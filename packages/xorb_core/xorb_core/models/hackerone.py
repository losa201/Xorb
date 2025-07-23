from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional

class HackerOneProgram(BaseModel):
    id: str
    handle: str
    name: str
    url: HttpUrl
    state: str
    submission_state: str
    triage_active: bool
    policy: str
    scopes: List[dict] = Field(default_factory=list)
    offers_bounties: bool
    average_bounty_lower_amount: Optional[float] = None
    average_bounty_upper_amount: Optional[float] = None

    class Config:
        populate_by_name = True
        alias_generator = lambda x: x.replace("_", "-")

class VulnerabilitySubmission(BaseModel):
    title: str
    description: str
    impact: str
    severity_rating: str
    program_handle: str
    weakness_id: Optional[int] = None
    proof_of_concept: Optional[str] = None
    steps_to_reproduce: Optional[str] = None
    cvss_vector: Optional[str] = None
    asset_identifier: Optional[str] = None
