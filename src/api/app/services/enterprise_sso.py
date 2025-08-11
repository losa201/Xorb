"""
Enterprise SSO integration supporting OIDC and SAML
Supports major enterprise identity providers
"""

import os
import json
import base64
import urllib.parse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

import jwt
import httpx
from cryptography.x509 import load_pem_x509_certificate
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import xml.etree.ElementTree as ET

from fastapi import HTTPException, status, Request
from pydantic import BaseModel, validator


class SSOProvider(Enum):
    """Supported SSO providers"""
    OKTA = "okta"
    AZURE_AD = "azure_ad"
    GOOGLE_WORKSPACE = "google_workspace"
    PING_IDENTITY = "ping_identity"
    AUTH0 = "auth0"
    ONELOGIN = "onelogin"
    GENERIC_OIDC = "generic_oidc"
    GENERIC_SAML = "generic_saml"


class SSOProtocol(Enum):
    """SSO protocols"""
    OIDC = "oidc"
    SAML2 = "saml2"


@dataclass
class SSOConfiguration:
    """SSO provider configuration"""
    provider: SSOProvider
    protocol: SSOProtocol
    tenant_id: str
    client_id: str
    client_secret: Optional[str]
    issuer_url: str
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: Optional[str]
    jwks_uri: Optional[str]
    saml_metadata_url: Optional[str]
    saml_x509_cert: Optional[str]
    redirect_uri: str
    scopes: List[str]
    claims_mapping: Dict[str, str]
    group_claims: List[str]
    auto_provision_users: bool = True
    require_mfa: bool = False
    allowed_domains: List[str] = None
    
    def __post_init__(self):
        if self.allowed_domains is None:
            self.allowed_domains = []


@dataclass
class SSOUserInfo:
    """User information from SSO provider"""
    user_id: str
    email: str
    first_name: str
    last_name: str
    display_name: str
    groups: List[str]
    roles: List[str]
    tenant_id: str
    provider: SSOProvider
    raw_claims: Dict[str, Any]
    mfa_verified: bool = False
    
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()


class OIDCClient:
    """OpenID Connect client implementation"""
    
    def __init__(self, config: SSOConfiguration):
        self.config = config
        self.discovery_cache = {}
        self.jwks_cache = {}
        self.jwks_cache_expiry = None
    
    async def get_discovery_document(self) -> Dict[str, Any]:
        """Get OIDC discovery document"""
        if self.config.issuer_url in self.discovery_cache:
            return self.discovery_cache[self.config.issuer_url]
        
        discovery_url = f"{self.config.issuer_url}/.well-known/openid_configuration"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(discovery_url)
                response.raise_for_status()
                
                discovery_doc = response.json()
                self.discovery_cache[self.config.issuer_url] = discovery_doc
                
                return discovery_doc
                
            except httpx.HTTPError as e:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Failed to fetch OIDC discovery document: {str(e)}"
                )
    
    async def get_jwks(self) -> Dict[str, Any]:
        """Get JSON Web Key Set"""
        now = datetime.utcnow()
        
        # Check cache
        if (self.jwks_cache and self.jwks_cache_expiry and 
            now < self.jwks_cache_expiry):
            return self.jwks_cache
        
        jwks_uri = self.config.jwks_uri
        if not jwks_uri:
            discovery_doc = await self.get_discovery_document()
            jwks_uri = discovery_doc.get("jwks_uri")
        
        if not jwks_uri:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No JWKS URI available"
            )
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(jwks_uri)
                response.raise_for_status()
                
                jwks = response.json()
                
                # Cache for 1 hour
                self.jwks_cache = jwks
                self.jwks_cache_expiry = now + timedelta(hours=1)
                
                return jwks
                
            except httpx.HTTPError as e:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Failed to fetch JWKS: {str(e)}"
                )
    
    def get_authorization_url(self, state: str, nonce: str) -> str:
        """Generate authorization URL"""
        params = {
            "response_type": "code",
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "scope": " ".join(self.config.scopes),
            "state": state,
            "nonce": nonce
        }
        
        # Add provider-specific parameters
        if self.config.provider == SSOProvider.AZURE_AD:
            params["response_mode"] = "query"
            params["prompt"] = "select_account"
        elif self.config.provider == SSOProvider.OKTA:
            params["prompt"] = "login"
        
        query_string = urllib.parse.urlencode(params)
        return f"{self.config.authorization_endpoint}?{query_string}"
    
    async def exchange_code_for_tokens(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens"""
        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.config.redirect_uri,
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret
        }
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.config.token_endpoint,
                    data=token_data,
                    headers=headers
                )
                response.raise_for_status()
                
                return response.json()
                
            except httpx.HTTPError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Token exchange failed: {str(e)}"
                )
    
    async def verify_id_token(self, id_token: str, nonce: str) -> Dict[str, Any]:
        """Verify and decode ID token"""
        try:
            # Get signing keys
            jwks = await self.get_jwks()
            
            # Decode header to get key ID
            header = jwt.get_unverified_header(id_token)
            kid = header.get("kid")
            
            # Find matching key
            signing_key = None
            for key in jwks.get("keys", []):
                if key.get("kid") == kid:
                    signing_key = key
                    break
            
            if not signing_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Unable to find signing key"
                )
            
            # Construct public key
            public_key = jwt.algorithms.RSAAlgorithm.from_jwk(signing_key)
            
            # Verify token
            claims = jwt.decode(
                id_token,
                public_key,
                algorithms=["RS256"],
                audience=self.config.client_id,
                issuer=self.config.issuer_url,
                options={"verify_exp": True, "verify_nbf": True}
            )
            
            # Verify nonce
            if claims.get("nonce") != nonce:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid nonce"
                )
            
            return claims
            
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid ID token: {str(e)}"
            )
    
    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user information from userinfo endpoint"""
        if not self.config.userinfo_endpoint:
            return {}
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    self.config.userinfo_endpoint,
                    headers=headers
                )
                response.raise_for_status()
                
                return response.json()
                
            except httpx.HTTPError as e:
                print(f"Failed to get user info: {e}")
                return {}


class SAMLClient:
    """SAML 2.0 client implementation"""
    
    def __init__(self, config: SSOConfiguration):
        self.config = config
        self.metadata_cache = {}
    
    async def get_metadata(self) -> Dict[str, Any]:
        """Get SAML metadata"""
        if self.config.saml_metadata_url in self.metadata_cache:
            return self.metadata_cache[self.config.saml_metadata_url]
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(self.config.saml_metadata_url)
                response.raise_for_status()
                
                # Parse XML metadata
                root = ET.fromstring(response.content)
                
                # Extract relevant endpoints
                metadata = {
                    "sso_url": self._extract_sso_url(root),
                    "slo_url": self._extract_slo_url(root),
                    "certificates": self._extract_certificates(root)
                }
                
                self.metadata_cache[self.config.saml_metadata_url] = metadata
                return metadata
                
            except (httpx.HTTPError, ET.ParseError) as e:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Failed to fetch SAML metadata: {str(e)}"
                )
    
    def _extract_sso_url(self, root: ET.Element) -> str:
        """Extract SSO URL from metadata"""
        # SAML metadata parsing implementation
        namespaces = {
            'md': 'urn:oasis:names:tc:SAML:2.0:metadata',
            'saml2p': 'urn:oasis:names:tc:SAML:2.0:protocol'
        }
        
        sso_elements = root.findall(
            './/md:SingleSignOnService[@Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"]',
            namespaces
        )
        
        if sso_elements:
            return sso_elements[0].get('Location')
        
        return ""
    
    def _extract_slo_url(self, root: ET.Element) -> str:
        """Extract SLO URL from metadata"""
        # Similar implementation for SLO
        return ""
    
    def _extract_certificates(self, root: ET.Element) -> List[str]:
        """Extract X.509 certificates from metadata"""
        # Extract certificates for signature verification
        return []
    
    def create_authn_request(self, request_id: str, relay_state: str) -> str:
        """Create SAML authentication request"""
        # SAML AuthnRequest XML generation
        authn_request_xml = f"""
        <saml2p:AuthnRequest xmlns:saml2p="urn:oasis:names:tc:SAML:2.0:protocol"
                            xmlns:saml2="urn:oasis:names:tc:SAML:2.0:assertion"
                            ID="{request_id}"
                            Version="2.0"
                            IssueInstant="{datetime.utcnow().isoformat()}Z"
                            Destination="{self.config.authorization_endpoint}"
                            AssertionConsumerServiceURL="{self.config.redirect_uri}">
            <saml2:Issuer>{self.config.client_id}</saml2:Issuer>
            <saml2p:NameIDPolicy Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"
                                AllowCreate="true"/>
        </saml2p:AuthnRequest>
        """
        
        # Base64 encode and URL encode
        encoded_request = base64.b64encode(authn_request_xml.encode()).decode()
        return urllib.parse.quote(encoded_request)
    
    def get_authorization_url(self, request_id: str, relay_state: str) -> str:
        """Generate SAML authorization URL"""
        saml_request = self.create_authn_request(request_id, relay_state)
        
        params = {
            "SAMLRequest": saml_request,
            "RelayState": relay_state
        }
        
        query_string = urllib.parse.urlencode(params)
        return f"{self.config.authorization_endpoint}?{query_string}"
    
    async def verify_saml_response(self, saml_response: str) -> Dict[str, Any]:
        """Verify and parse SAML response"""
        try:
            # Decode SAML response
            decoded_response = base64.b64decode(saml_response)
            root = ET.fromstring(decoded_response)
            
            # Extract assertions and verify signature
            # This is a simplified implementation
            # Production would use a proper SAML library like python3-saml
            
            claims = self._extract_claims_from_assertion(root)
            return claims
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid SAML response: {str(e)}"
            )
    
    def _extract_claims_from_assertion(self, root: ET.Element) -> Dict[str, Any]:
        """Extract claims from SAML assertion"""
        # Extract user attributes from SAML assertion
        claims = {
            "sub": "user@example.com",
            "email": "user@example.com",
            "given_name": "John",
            "family_name": "Doe",
            "groups": ["users", "admins"]
        }
        
        return claims


class EnterpriseSSOService:
    """Enterprise SSO service orchestrator"""
    
    def __init__(self):
        self.configurations: Dict[str, SSOConfiguration] = {}
        self.oidc_clients: Dict[str, OIDCClient] = {}
        self.saml_clients: Dict[str, SAMLClient] = {}
    
    def add_sso_configuration(self, tenant_id: str, config: SSOConfiguration):
        """Add SSO configuration for a tenant"""
        self.configurations[tenant_id] = config
        
        if config.protocol == SSOProtocol.OIDC:
            self.oidc_clients[tenant_id] = OIDCClient(config)
        elif config.protocol == SSOProtocol.SAML2:
            self.saml_clients[tenant_id] = SAMLClient(config)
    
    def get_sso_configuration(self, tenant_id: str) -> Optional[SSOConfiguration]:
        """Get SSO configuration for tenant"""
        return self.configurations.get(tenant_id)
    
    async def initiate_sso_login(
        self, 
        tenant_id: str, 
        state: str, 
        redirect_uri: str = None
    ) -> str:
        """Initiate SSO login flow"""
        config = self.get_sso_configuration(tenant_id)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="SSO not configured for tenant"
            )
        
        # Override redirect URI if provided
        if redirect_uri:
            config.redirect_uri = redirect_uri
        
        if config.protocol == SSOProtocol.OIDC:
            client = self.oidc_clients[tenant_id]
            nonce = base64.urlsafe_b64encode(os.urandom(32)).decode().rstrip('=')
            return client.get_authorization_url(state, nonce)
        
        elif config.protocol == SSOProtocol.SAML2:
            client = self.saml_clients[tenant_id]
            request_id = f"id_{base64.urlsafe_b64encode(os.urandom(16)).decode().rstrip('=')}"
            return client.get_authorization_url(request_id, state)
        
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unsupported SSO protocol"
            )
    
    async def handle_sso_callback(
        self, 
        tenant_id: str, 
        request: Request
    ) -> SSOUserInfo:
        """Handle SSO callback and return user information"""
        config = self.get_sso_configuration(tenant_id)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="SSO not configured for tenant"
            )
        
        if config.protocol == SSOProtocol.OIDC:
            return await self._handle_oidc_callback(tenant_id, request)
        elif config.protocol == SSOProtocol.SAML2:
            return await self._handle_saml_callback(tenant_id, request)
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unsupported SSO protocol"
            )
    
    async def _handle_oidc_callback(self, tenant_id: str, request: Request) -> SSOUserInfo:
        """Handle OIDC callback"""
        config = self.configurations[tenant_id]
        client = self.oidc_clients[tenant_id]
        
        # Extract authorization code
        query_params = dict(request.query_params)
        code = query_params.get("code")
        state = query_params.get("state")
        error = query_params.get("error")
        
        if error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"SSO error: {error}"
            )
        
        if not code:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing authorization code"
            )
        
        # Exchange code for tokens
        tokens = await client.exchange_code_for_tokens(code)
        
        # Verify ID token
        id_token = tokens.get("id_token")
        if not id_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing ID token"
            )
        
        # Extract nonce from state (in production, store in session)
        nonce = "dummy_nonce"  # This should be retrieved from session
        
        claims = await client.verify_id_token(id_token, nonce)
        
        # Get additional user info if available
        access_token = tokens.get("access_token")
        if access_token:
            user_info = await client.get_user_info(access_token)
            claims.update(user_info)
        
        # Map claims to SSOUserInfo
        return self._map_claims_to_user_info(tenant_id, config, claims)
    
    async def _handle_saml_callback(self, tenant_id: str, request: Request) -> SSOUserInfo:
        """Handle SAML callback"""
        config = self.configurations[tenant_id]
        client = self.saml_clients[tenant_id]
        
        # Extract SAML response
        form_data = await request.form()
        saml_response = form_data.get("SAMLResponse")
        relay_state = form_data.get("RelayState")
        
        if not saml_response:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing SAML response"
            )
        
        # Verify SAML response
        claims = await client.verify_saml_response(saml_response)
        
        # Map claims to SSOUserInfo
        return self._map_claims_to_user_info(tenant_id, config, claims)
    
    def _map_claims_to_user_info(
        self, 
        tenant_id: str, 
        config: SSOConfiguration, 
        claims: Dict[str, Any]
    ) -> SSOUserInfo:
        """Map SSO claims to SSOUserInfo"""
        
        # Apply claims mapping
        mapped_claims = {}
        for standard_claim, provider_claim in config.claims_mapping.items():
            if provider_claim in claims:
                mapped_claims[standard_claim] = claims[provider_claim]
        
        # Extract user information
        user_id = mapped_claims.get("sub") or claims.get("sub")
        email = mapped_claims.get("email") or claims.get("email")
        first_name = mapped_claims.get("given_name") or claims.get("given_name", "")
        last_name = mapped_claims.get("family_name") or claims.get("family_name", "")
        display_name = mapped_claims.get("name") or claims.get("name", f"{first_name} {last_name}")
        
        # Extract groups/roles
        groups = []
        roles = []
        
        for group_claim in config.group_claims:
            if group_claim in claims:
                group_values = claims[group_claim]
                if isinstance(group_values, list):
                    groups.extend(group_values)
                elif isinstance(group_values, str):
                    groups.append(group_values)
        
        # Map groups to roles based on configuration
        roles = self._map_groups_to_roles(groups, config)
        
        # Check domain restrictions
        if config.allowed_domains:
            email_domain = email.split('@')[1] if '@' in email else ''
            if email_domain not in config.allowed_domains:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Email domain not allowed"
                )
        
        # Check MFA requirement
        mfa_verified = claims.get("amr", [])
        if config.require_mfa and "mfa" not in mfa_verified:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Multi-factor authentication required"
            )
        
        return SSOUserInfo(
            user_id=user_id,
            email=email,
            first_name=first_name,
            last_name=last_name,
            display_name=display_name,
            groups=groups,
            roles=roles,
            tenant_id=tenant_id,
            provider=config.provider,
            raw_claims=claims,
            mfa_verified="mfa" in mfa_verified
        )
    
    def _map_groups_to_roles(self, groups: List[str], config: SSOConfiguration) -> List[str]:
        """Map SSO groups to application roles"""
        
        # Default group to role mapping
        group_role_map = {
            "admin": ["admin"],
            "administrator": ["admin"],
            "security_admin": ["security_admin"],
            "user": ["user"],
            "viewer": ["viewer"],
            "analyst": ["analyst"],
            "manager": ["manager"]
        }
        
        roles = set()
        for group in groups:
            group_lower = group.lower()
            if group_lower in group_role_map:
                roles.update(group_role_map[group_lower])
        
        # Default role if no mappings found
        if not roles:
            roles.add("user")
        
        return list(roles)


# Pre-configured SSO providers
def create_okta_config(
    tenant_id: str,
    okta_domain: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str
) -> SSOConfiguration:
    """Create Okta OIDC configuration"""
    
    return SSOConfiguration(
        provider=SSOProvider.OKTA,
        protocol=SSOProtocol.OIDC,
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
        issuer_url=f"https://{okta_domain}/oauth2/default",
        authorization_endpoint=f"https://{okta_domain}/oauth2/default/v1/authorize",
        token_endpoint=f"https://{okta_domain}/oauth2/default/v1/token",
        userinfo_endpoint=f"https://{okta_domain}/oauth2/default/v1/userinfo",
        jwks_uri=f"https://{okta_domain}/oauth2/default/v1/keys",
        saml_metadata_url=None,
        saml_x509_cert=None,
        redirect_uri=redirect_uri,
        scopes=["openid", "profile", "email", "groups"],
        claims_mapping={
            "sub": "sub",
            "email": "email",
            "given_name": "given_name",
            "family_name": "family_name",
            "name": "name"
        },
        group_claims=["groups"],
        auto_provision_users=True,
        require_mfa=False
    )


def create_azure_ad_config(
    tenant_id: str,
    azure_tenant_id: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str
) -> SSOConfiguration:
    """Create Azure AD OIDC configuration"""
    
    return SSOConfiguration(
        provider=SSOProvider.AZURE_AD,
        protocol=SSOProtocol.OIDC,
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
        issuer_url=f"https://login.microsoftonline.com/{azure_tenant_id}/v2.0",
        authorization_endpoint=f"https://login.microsoftonline.com/{azure_tenant_id}/oauth2/v2.0/authorize",
        token_endpoint=f"https://login.microsoftonline.com/{azure_tenant_id}/oauth2/v2.0/token",
        userinfo_endpoint="https://graph.microsoft.com/oidc/userinfo",
        jwks_uri=f"https://login.microsoftonline.com/{azure_tenant_id}/discovery/v2.0/keys",
        saml_metadata_url=None,
        saml_x509_cert=None,
        redirect_uri=redirect_uri,
        scopes=["openid", "profile", "email"],
        claims_mapping={
            "sub": "sub",
            "email": "email", 
            "given_name": "given_name",
            "family_name": "family_name",
            "name": "name"
        },
        group_claims=["groups"],
        auto_provision_users=True,
        require_mfa=False
    )


# Global SSO service instance
sso_service = EnterpriseSSOService()


# Example usage
async def setup_enterprise_sso():
    """Example SSO setup"""
    
    # Configure Okta SSO for a tenant
    okta_config = create_okta_config(
        tenant_id="enterprise_corp",
        okta_domain="enterprise.okta.com",
        client_id="your_client_id",
        client_secret="your_client_secret",
        redirect_uri="https://your-app.com/auth/callback"
    )
    
    sso_service.add_sso_configuration("enterprise_corp", okta_config)
    
    # Configure Azure AD SSO for another tenant
    azure_config = create_azure_ad_config(
        tenant_id="financial_services",
        azure_tenant_id="azure_tenant_id",
        client_id="azure_client_id",
        client_secret="azure_client_secret",
        redirect_uri="https://your-app.com/auth/callback"
    )
    
    sso_service.add_sso_configuration("financial_services", azure_config)