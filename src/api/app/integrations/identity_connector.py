"""
Identity Provider Integration Connectors
Focused connectors for identity management platforms (Active Directory, Okta, Azure AD)
"""

import json
import logging
import base64
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from .base_connector import (
    BaseConnector,
    IntegrationRequest,
    IntegrationResponse,
    AuthenticationType
)

logger = logging.getLogger(__name__)


class ActiveDirectoryConnector(BaseConnector):
    """Active Directory integration connector via LDAP/Graph API"""
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for AD Graph API requests"""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    async def health_check(self) -> bool:
        """Check Active Directory connectivity"""
        try:
            # Test with a simple organization query
            request = IntegrationRequest(
                endpoint_id=self.endpoint.integration_id,
                method="GET",
                path="/v1.0/organization"
            )
            
            response = await self.make_authenticated_request(request)
            return response.success
            
        except Exception as e:
            logger.error(f"Active Directory health check failed: {e}")
            return False
    
    async def get_user_info(self, user_id: str) -> IntegrationResponse:
        """Get user information from Active Directory"""
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="GET",
            path=f"/v1.0/users/{user_id}"
        )
        
        return await self.make_authenticated_request(request)
    
    async def search_users(self, search_query: str) -> IntegrationResponse:
        """Search for users in Active Directory"""
        params = {
            "$search": f"\"displayName:{search_query}\" OR \"mail:{search_query}\"",
            "$select": "id,displayName,mail,userPrincipalName,accountEnabled"
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="GET",
            path="/v1.0/users",
            params=params,
            custom_headers={"ConsistencyLevel": "eventual"}
        )
        
        return await self.make_authenticated_request(request)
    
    async def disable_user(self, user_id: str, reason: str = "XORB Security Action") -> IntegrationResponse:
        """Disable user account"""
        update_data = {
            "accountEnabled": False
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="PATCH",
            path=f"/v1.0/users/{user_id}",
            data=update_data
        )
        
        response = await self.make_authenticated_request(request)
        
        if response.success:
            # Log the action
            await self._log_security_action(user_id, "disable_account", reason)
        
        return response
    
    async def enable_user(self, user_id: str, reason: str = "XORB Security Action") -> IntegrationResponse:
        """Enable user account"""
        update_data = {
            "accountEnabled": True
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="PATCH",
            path=f"/v1.0/users/{user_id}",
            data=update_data
        )
        
        response = await self.make_authenticated_request(request)
        
        if response.success:
            await self._log_security_action(user_id, "enable_account", reason)
        
        return response
    
    async def reset_user_password(self, user_id: str) -> IntegrationResponse:
        """Force password reset for user"""
        password_profile = {
            "forceChangePasswordNextSignIn": True,
            "password": self._generate_temp_password()
        }
        
        update_data = {
            "passwordProfile": password_profile
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="PATCH",
            path=f"/v1.0/users/{user_id}",
            data=update_data
        )
        
        response = await self.make_authenticated_request(request)
        
        if response.success:
            await self._log_security_action(user_id, "force_password_reset", "XORB Security Action")
        
        return response
    
    async def get_user_sign_in_logs(
        self, 
        user_id: str, 
        start_time: Optional[datetime] = None
    ) -> IntegrationResponse:
        """Get sign-in logs for user"""
        filters = [f"userId eq '{user_id}'"]
        
        if start_time:
            time_filter = f"createdDateTime ge {start_time.isoformat()}"
            filters.append(time_filter)
        
        params = {
            "$filter": " and ".join(filters),
            "$orderby": "createdDateTime desc",
            "$top": "100"
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="GET",
            path="/v1.0/auditLogs/signIns",
            params=params
        )
        
        return await self.make_authenticated_request(request)
    
    async def revoke_user_sessions(self, user_id: str) -> IntegrationResponse:
        """Revoke all user sessions"""
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path=f"/v1.0/users/{user_id}/revokeSignInSessions",
            data={}
        )
        
        response = await self.make_authenticated_request(request)
        
        if response.success:
            await self._log_security_action(user_id, "revoke_sessions", "XORB Security Action")
        
        return response
    
    def _generate_temp_password(self) -> str:
        """Generate temporary password"""
        import secrets
        import string
        
        # Generate complex temporary password
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        password = ''.join(secrets.choice(alphabet) for _ in range(16))
        return password
    
    async def _log_security_action(self, user_id: str, action: str, reason: str):
        """Log security action (could be sent to audit system)"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,
            "reason": reason,
            "source": "XORB Platform",
            "integration": self.endpoint.integration_id
        }
        
        logger.info(f"AD Security Action: {json.dumps(log_entry)}")


class OktaConnector(BaseConnector):
    """Okta identity provider integration connector"""
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for Okta requests"""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    async def health_check(self) -> bool:
        """Check Okta connectivity"""
        try:
            request = IntegrationRequest(
                endpoint_id=self.endpoint.integration_id,
                method="GET",
                path="/api/v1/org"
            )
            
            response = await self.make_authenticated_request(request)
            return response.success
            
        except Exception as e:
            logger.error(f"Okta health check failed: {e}")
            return False
    
    async def get_user_info(self, user_id: str) -> IntegrationResponse:
        """Get user information from Okta"""
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="GET",
            path=f"/api/v1/users/{user_id}"
        )
        
        return await self.make_authenticated_request(request)
    
    async def search_users(self, search_query: str) -> IntegrationResponse:
        """Search for users in Okta"""
        params = {
            "q": search_query,
            "limit": "100"
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="GET",
            path="/api/v1/users",
            params=params
        )
        
        return await self.make_authenticated_request(request)
    
    async def suspend_user(self, user_id: str) -> IntegrationResponse:
        """Suspend user account in Okta"""
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path=f"/api/v1/users/{user_id}/lifecycle/suspend"
        )
        
        response = await self.make_authenticated_request(request)
        
        if response.success:
            await self._log_security_action(user_id, "suspend_user", "XORB Security Action")
        
        return response
    
    async def unsuspend_user(self, user_id: str) -> IntegrationResponse:
        """Unsuspend user account in Okta"""
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path=f"/api/v1/users/{user_id}/lifecycle/unsuspend"
        )
        
        response = await self.make_authenticated_request(request)
        
        if response.success:
            await self._log_security_action(user_id, "unsuspend_user", "XORB Security Action")
        
        return response
    
    async def deactivate_user(self, user_id: str) -> IntegrationResponse:
        """Deactivate user account in Okta"""
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path=f"/api/v1/users/{user_id}/lifecycle/deactivate"
        )
        
        response = await self.make_authenticated_request(request)
        
        if response.success:
            await self._log_security_action(user_id, "deactivate_user", "XORB Security Action")
        
        return response
    
    async def reset_user_password(self, user_id: str) -> IntegrationResponse:
        """Reset user password in Okta"""
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path=f"/api/v1/users/{user_id}/lifecycle/reset_password",
            params={"sendEmail": "true"}
        )
        
        response = await self.make_authenticated_request(request)
        
        if response.success:
            await self._log_security_action(user_id, "reset_password", "XORB Security Action")
        
        return response
    
    async def clear_user_sessions(self, user_id: str) -> IntegrationResponse:
        """Clear all user sessions in Okta"""
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="DELETE",
            path=f"/api/v1/users/{user_id}/sessions"
        )
        
        response = await self.make_authenticated_request(request)
        
        if response.success:
            await self._log_security_action(user_id, "clear_sessions", "XORB Security Action")
        
        return response
    
    async def get_user_logs(
        self, 
        user_id: str, 
        start_time: Optional[datetime] = None
    ) -> IntegrationResponse:
        """Get audit logs for user"""
        params = {
            "filter": f'target.id eq "{user_id}"',
            "sortOrder": "DESCENDING",
            "limit": "100"
        }
        
        if start_time:
            params["since"] = start_time.isoformat()
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="GET",
            path="/api/v1/logs",
            params=params
        )
        
        return await self.make_authenticated_request(request)
    
    async def add_user_to_group(self, user_id: str, group_id: str) -> IntegrationResponse:
        """Add user to group"""
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="PUT",
            path=f"/api/v1/groups/{group_id}/users/{user_id}"
        )
        
        return await self.make_authenticated_request(request)
    
    async def remove_user_from_group(self, user_id: str, group_id: str) -> IntegrationResponse:
        """Remove user from group"""
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="DELETE",
            path=f"/api/v1/groups/{group_id}/users/{user_id}"
        )
        
        return await self.make_authenticated_request(request)
    
    async def _log_security_action(self, user_id: str, action: str, reason: str):
        """Log security action"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,
            "reason": reason,
            "source": "XORB Platform",
            "integration": self.endpoint.integration_id
        }
        
        logger.info(f"Okta Security Action: {json.dumps(log_entry)}")


class AzureADConnector(BaseConnector):
    """Azure Active Directory integration connector"""
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for Azure AD requests"""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    async def health_check(self) -> bool:
        """Check Azure AD connectivity"""
        try:
            request = IntegrationRequest(
                endpoint_id=self.endpoint.integration_id,
                method="GET",
                path="/v1.0/organization"
            )
            
            response = await self.make_authenticated_request(request)
            return response.success
            
        except Exception as e:
            logger.error(f"Azure AD health check failed: {e}")
            return False
    
    async def _refresh_oauth2_token(self) -> bool:
        """Refresh OAuth2 token for Azure AD"""
        credentials = self.endpoint.credentials.credentials
        tenant_id = credentials.get("tenant_id")
        client_id = credentials.get("client_id")
        client_secret = credentials.get("client_secret")
        refresh_token = credentials.get("refresh_token")
        
        if not all([tenant_id, client_id, client_secret, refresh_token]):
            logger.error("Missing required credentials for token refresh")
            return False
        
        token_data = {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "scope": "https://graph.microsoft.com/.default"
        }
        
        # Create a temporary session for token refresh
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
                data=token_data
            ) as response:
                if response.status == 200:
                    token_response = await response.json()
                    
                    # Update credentials
                    credentials["access_token"] = token_response["access_token"]
                    if "refresh_token" in token_response:
                        credentials["refresh_token"] = token_response["refresh_token"]
                    
                    # Set expiration time
                    expires_in = token_response.get("expires_in", 3600)
                    self.endpoint.credentials.expires_at = datetime.utcnow() + timedelta(seconds=expires_in - 300)
                    
                    return True
        
        return False
    
    async def get_user_info(self, user_id: str) -> IntegrationResponse:
        """Get user information from Azure AD"""
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="GET",
            path=f"/v1.0/users/{user_id}"
        )
        
        return await self.make_authenticated_request(request)
    
    async def get_user_risk_detections(self, user_id: str) -> IntegrationResponse:
        """Get risk detections for user"""
        params = {
            "$filter": f"userId eq '{user_id}'",
            "$orderby": "detectedDateTime desc"
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="GET",
            path="/v1.0/identityProtection/riskDetections",
            params=params
        )
        
        return await self.make_authenticated_request(request)
    
    async def confirm_user_compromised(self, user_id: str) -> IntegrationResponse:
        """Mark user as compromised in Azure AD Identity Protection"""
        confirm_data = {
            "userIds": [user_id]
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/v1.0/identityProtection/riskyUsers/confirmCompromised",
            data=confirm_data
        )
        
        response = await self.make_authenticated_request(request)
        
        if response.success:
            await self._log_security_action(user_id, "confirm_compromised", "XORB Security Action")
        
        return response
    
    async def dismiss_user_risk(self, user_id: str) -> IntegrationResponse:
        """Dismiss user risk in Azure AD Identity Protection"""
        dismiss_data = {
            "userIds": [user_id]
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/v1.0/identityProtection/riskyUsers/dismiss",
            data=dismiss_data
        )
        
        response = await self.make_authenticated_request(request)
        
        if response.success:
            await self._log_security_action(user_id, "dismiss_risk", "XORB Security Action")
        
        return response
    
    async def block_user_sign_in(self, user_id: str) -> IntegrationResponse:
        """Block user sign-in"""
        update_data = {
            "accountEnabled": False
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="PATCH",
            path=f"/v1.0/users/{user_id}",
            data=update_data
        )
        
        response = await self.make_authenticated_request(request)
        
        if response.success:
            await self._log_security_action(user_id, "block_signin", "XORB Security Action")
        
        return response
    
    async def get_conditional_access_policies(self) -> IntegrationResponse:
        """Get conditional access policies"""
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="GET",
            path="/v1.0/identity/conditionalAccess/policies"
        )
        
        return await self.make_authenticated_request(request)
    
    async def _log_security_action(self, user_id: str, action: str, reason: str):
        """Log security action"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,
            "reason": reason,
            "source": "XORB Platform",
            "integration": self.endpoint.integration_id
        }
        
        logger.info(f"Azure AD Security Action: {json.dumps(log_entry)}")