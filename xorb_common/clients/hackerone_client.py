import httpx
from pydantic import BaseSettings, SecretStr
from typing import List, Optional

from ..models.hackerone import HackerOneProgram, VulnerabilitySubmission

class HackerOneSettings(BaseSettings):
    hackerone_username: str
    hackerone_api_key: SecretStr

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class HackerOneClient:
    def __init__(self, settings: HackerOneSettings, base_url: str = "https://api.hackerone.com/v1"):
        self.base_url = base_url
        self.settings = settings
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            auth=(self.settings.hackerone_username, self.settings.hackerone_api_key.get_secret_value()),
            headers={
                "Accept": "application/json",
                "User-Agent": "Xorb/2.0",
            },
        )

    async def get_programs(self, eligible_only: bool = True) -> List[HackerOneProgram]:
        params = {}
        if eligible_only:
            params["filter[submission_state]"] = "open"

        response = await self.client.get("/programs", params=params)
        response.raise_for_status()
        data = response.json()["data"]
        return [HackerOneProgram(**item["attributes"]) for item in data]

    async def submit_report(self, submission: VulnerabilitySubmission) -> dict:
        payload = {
            "data": {
                "type": "report",
                "attributes": submission.dict(exclude_none=True),
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
        response = await self.client.post("/reports", json=payload)
        response.raise_for_status()
        return response.json()

    async def close(self):
        await self.client.aclose()
