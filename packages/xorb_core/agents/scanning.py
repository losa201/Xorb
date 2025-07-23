
import httpx
from typing import List, Set
from urllib.parse import urlparse
from xorb_core.models.agents import DiscoveryTarget, Finding

class SecurityHeadersAgent:
    """A simple vulnerability scanner that checks for common security headers."""

    @property
    def name(self) -> str:
        return "web_security_headers_check"

    @property
    def description(self) -> str:
        return "Checks a web page for the presence of essential security headers."

    @property
    def accepted_target_types(self) -> Set[str]:
        return {"url"}

    async def run(self, target: DiscoveryTarget) -> List[Finding]:
        if target.target_type not in self.accepted_target_types:
            return []

        # Basic URL validation
        parsed_url = urlparse(target.value)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            return []

        findings = []
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(target.value)

                missing_headers = []
                # More comprehensive list of modern security headers
                security_headers = {
                    "Strict-Transport-Security": "critical",
                    "Content-Security-Policy": "high",
                    "X-Frame-Options": "medium",
                    "X-Content-Type-Options": "medium",
                    "Referrer-Policy": "low",
                    "Permissions-Policy": "low",
                }

                for header, severity in security_headers.items():
                    if header.lower() not in [h.lower() for h in response.headers]:
                        missing_headers.append((header, severity))

                for header, severity in missing_headers:
                    findings.append(
                        Finding(
                            title=f"Missing Security Header: {header}",
                            description=f"The essential security header '{header}' was not found on {target.value}.",
                            target=target.value,
                            finding_type="missing_security_header",
                            severity=severity,
                            confidence=0.9,
                            evidence={"missing_header": header}
                        )
                    )
        except httpx.RequestError as e:
            # Handle network-level errors (DNS, connection refused, etc.)
            print(f"Request error for {target.value}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while scanning {target.value}: {e}")
            
        return findings
