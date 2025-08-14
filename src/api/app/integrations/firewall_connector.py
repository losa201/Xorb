"""
Firewall Integration Connectors
Focused connectors for firewall management platforms
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from .base_connector import (
    BaseConnector,
    IntegrationRequest,
    IntegrationResponse,
    AuthenticationType
)

logger = logging.getLogger(__name__)


class PaloAltoConnector(BaseConnector):
    """Palo Alto Networks firewall integration connector"""
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for Palo Alto requests"""
        return {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/xml"
        }
    
    async def health_check(self) -> bool:
        """Check Palo Alto firewall connectivity and health"""
        try:
            request = IntegrationRequest(
                endpoint_id=self.endpoint.integration_id,
                method="GET",
                path="/api/",
                params={
                    "type": "op",
                    "cmd": "<show><system><info></info></system></show>"
                }
            )
            
            response = await self.make_authenticated_request(request)
            return response.success
            
        except Exception as e:
            logger.error(f"Palo Alto health check failed: {e}")
            return False
    
    async def _prepare_auth_headers(self) -> Dict[str, str]:
        """Prepare Palo Alto-specific authentication"""
        # Palo Alto uses API key in URL parameters, not headers
        return {}
    
    async def block_ip(self, ip_address: str, reason: str = "XORB Threat Block") -> IntegrationResponse:
        """Block IP address on Palo Alto firewall"""
        api_key = self.endpoint.credentials.credentials.get("api_key")
        
        # Create address object
        address_name = f"XORB_BLOCK_{ip_address.replace('.', '_')}"
        create_address_cmd = f"""
        <set>
            <address>
                <entry name="{address_name}">
                    <ip-netmask>{ip_address}/32</ip-netmask>
                    <description>{reason}</description>
                    <tag>
                        <member>XORB-Blocked</member>
                    </tag>
                </entry>
            </address>
        </set>
        """
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/api/",
            params={
                "type": "config",
                "action": "set",
                "xpath": "/config/devices/entry[@name='localhost.localdomain']/vsys/entry[@name='vsys1']/address",
                "element": create_address_cmd,
                "key": api_key
            }
        )
        
        response = await self.make_authenticated_request(request)
        
        if response.success:
            # Add to block policy
            await self._add_to_block_policy(address_name, api_key)
        
        return response
    
    async def unblock_ip(self, ip_address: str) -> IntegrationResponse:
        """Unblock IP address on Palo Alto firewall"""
        api_key = self.endpoint.credentials.credentials.get("api_key")
        address_name = f"XORB_BLOCK_{ip_address.replace('.', '_')}"
        
        # Remove from block policy first
        await self._remove_from_block_policy(address_name, api_key)
        
        # Delete address object
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/api/",
            params={
                "type": "config",
                "action": "delete",
                "xpath": f"/config/devices/entry[@name='localhost.localdomain']/vsys/entry[@name='vsys1']/address/entry[@name='{address_name}']",
                "key": api_key
            }
        )
        
        return await self.make_authenticated_request(request)
    
    async def create_security_rule(
        self, 
        rule_name: str, 
        source_zones: List[str],
        destination_zones: List[str],
        action: str = "deny"
    ) -> IntegrationResponse:
        """Create security rule"""
        api_key = self.endpoint.credentials.credentials.get("api_key")
        
        rule_xml = f"""
        <entry name="{rule_name}">
            <from>
                {''.join(f'<member>{zone}</member>' for zone in source_zones)}
            </from>
            <to>
                {''.join(f'<member>{zone}</member>' for zone in destination_zones)}
            </to>
            <source>
                <member>any</member>
            </source>
            <destination>
                <member>any</member>
            </destination>
            <service>
                <member>any</member>
            </service>
            <application>
                <member>any</member>
            </application>
            <action>{action}</action>
            <description>XORB Security Rule</description>
        </entry>
        """
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/api/",
            params={
                "type": "config",
                "action": "set",
                "xpath": "/config/devices/entry[@name='localhost.localdomain']/vsys/entry[@name='vsys1']/rulebase/security/rules",
                "element": rule_xml,
                "key": api_key
            }
        )
        
        return await self.make_authenticated_request(request)
    
    async def commit_changes(self) -> IntegrationResponse:
        """Commit configuration changes"""
        api_key = self.endpoint.credentials.credentials.get("api_key")
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/api/",
            params={
                "type": "commit",
                "cmd": "<commit></commit>",
                "key": api_key
            }
        )
        
        return await self.make_authenticated_request(request)
    
    async def _add_to_block_policy(self, address_name: str, api_key: str) -> IntegrationResponse:
        """Add address to existing block policy"""
        # This would need to be customized based on your specific policy structure
        block_policy_name = self.endpoint.metadata.get("block_policy_name", "XORB_Block_Policy")
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/api/",
            params={
                "type": "config",
                "action": "set",
                "xpath": f"/config/devices/entry[@name='localhost.localdomain']/vsys/entry[@name='vsys1']/rulebase/security/rules/entry[@name='{block_policy_name}']/source",
                "element": f"<member>{address_name}</member>",
                "key": api_key
            }
        )
        
        return await self.make_authenticated_request(request)
    
    async def _remove_from_block_policy(self, address_name: str, api_key: str) -> IntegrationResponse:
        """Remove address from block policy"""
        block_policy_name = self.endpoint.metadata.get("block_policy_name", "XORB_Block_Policy")
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/api/",
            params={
                "type": "config",
                "action": "delete",
                "xpath": f"/config/devices/entry[@name='localhost.localdomain']/vsys/entry[@name='vsys1']/rulebase/security/rules/entry[@name='{block_policy_name}']/source/member[text()='{address_name}']",
                "key": api_key
            }
        )
        
        return await self.make_authenticated_request(request)


class FortiGateConnector(BaseConnector):
    """Fortinet FortiGate firewall integration connector"""
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for FortiGate requests"""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    async def health_check(self) -> bool:
        """Check FortiGate connectivity and health"""
        try:
            request = IntegrationRequest(
                endpoint_id=self.endpoint.integration_id,
                method="GET",
                path="/api/v2/monitor/system/status"
            )
            
            response = await self.make_authenticated_request(request)
            return response.success
            
        except Exception as e:
            logger.error(f"FortiGate health check failed: {e}")
            return False
    
    async def block_ip(self, ip_address: str, reason: str = "XORB Threat Block") -> IntegrationResponse:
        """Block IP address on FortiGate firewall"""
        address_name = f"XORB_BLOCK_{ip_address.replace('.', '_')}"
        
        # Create address object
        address_data = {
            "name": address_name,
            "type": "ipmask",
            "subnet": f"{ip_address}/32",
            "comment": reason
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/api/v2/cmdb/firewall/address",
            data=address_data
        )
        
        response = await self.make_authenticated_request(request)
        
        if response.success:
            # Add to address group
            await self._add_to_address_group(address_name)
        
        return response
    
    async def unblock_ip(self, ip_address: str) -> IntegrationResponse:
        """Unblock IP address on FortiGate firewall"""
        address_name = f"XORB_BLOCK_{ip_address.replace('.', '_')}"
        
        # Remove from address group
        await self._remove_from_address_group(address_name)
        
        # Delete address object
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="DELETE",
            path=f"/api/v2/cmdb/firewall/address/{address_name}"
        )
        
        return await self.make_authenticated_request(request)
    
    async def create_policy(
        self, 
        policy_name: str,
        source_interface: str,
        destination_interface: str,
        action: str = "deny"
    ) -> IntegrationResponse:
        """Create firewall policy"""
        policy_data = {
            "name": policy_name,
            "srcintf": [{"name": source_interface}],
            "dstintf": [{"name": destination_interface}],
            "srcaddr": [{"name": "all"}],
            "dstaddr": [{"name": "all"}],
            "service": [{"name": "ALL"}],
            "action": action,
            "comments": "XORB Security Policy",
            "status": "enable"
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/api/v2/cmdb/firewall/policy",
            data=policy_data
        )
        
        return await self.make_authenticated_request(request)
    
    async def get_policies(self) -> IntegrationResponse:
        """Get all firewall policies"""
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="GET",
            path="/api/v2/cmdb/firewall/policy"
        )
        
        return await self.make_authenticated_request(request)
    
    async def _add_to_address_group(self, address_name: str) -> IntegrationResponse:
        """Add address to XORB block group"""
        group_name = self.endpoint.metadata.get("block_group_name", "XORB_BLOCKED_IPS")
        
        # First, get current group members
        get_request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="GET",
            path=f"/api/v2/cmdb/firewall/addrgrp/{group_name}"
        )
        
        get_response = await self.make_authenticated_request(get_request)
        
        if get_response.success:
            current_members = get_response.data.get("results", [{}])[0].get("member", [])
            current_members.append({"name": address_name})
            
            update_data = {"member": current_members}
            
            request = IntegrationRequest(
                endpoint_id=self.endpoint.integration_id,
                method="PUT",
                path=f"/api/v2/cmdb/firewall/addrgrp/{group_name}",
                data=update_data
            )
            
            return await self.make_authenticated_request(request)
        else:
            # Create group if it doesn't exist
            group_data = {
                "name": group_name,
                "member": [{"name": address_name}],
                "comment": "XORB Blocked IPs"
            }
            
            request = IntegrationRequest(
                endpoint_id=self.endpoint.integration_id,
                method="POST",
                path="/api/v2/cmdb/firewall/addrgrp",
                data=group_data
            )
            
            return await self.make_authenticated_request(request)
    
    async def _remove_from_address_group(self, address_name: str) -> IntegrationResponse:
        """Remove address from block group"""
        group_name = self.endpoint.metadata.get("block_group_name", "XORB_BLOCKED_IPS")
        
        # Get current group members
        get_request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="GET",
            path=f"/api/v2/cmdb/firewall/addrgrp/{group_name}"
        )
        
        get_response = await self.make_authenticated_request(get_request)
        
        if get_response.success:
            current_members = get_response.data.get("results", [{}])[0].get("member", [])
            updated_members = [m for m in current_members if m.get("name") != address_name]
            
            update_data = {"member": updated_members}
            
            request = IntegrationRequest(
                endpoint_id=self.endpoint.integration_id,
                method="PUT",
                path=f"/api/v2/cmdb/firewall/addrgrp/{group_name}",
                data=update_data
            )
            
            return await self.make_authenticated_request(request)
        
        return IntegrationResponse(success=False, status_code=404, error="Address group not found")


class CheckPointConnector(BaseConnector):
    """Check Point firewall integration connector"""
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for Check Point requests"""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    async def health_check(self) -> bool:
        """Check Check Point connectivity and health"""
        try:
            request = IntegrationRequest(
                endpoint_id=self.endpoint.integration_id,
                method="POST",
                path="/web_api/show-api-versions",
                data={}
            )
            
            response = await self.make_authenticated_request(request)
            return response.success
            
        except Exception as e:
            logger.error(f"Check Point health check failed: {e}")
            return False
    
    async def login(self) -> IntegrationResponse:
        """Login to Check Point management server"""
        login_data = {
            "user": self.endpoint.credentials.credentials.get("username"),
            "password": self.endpoint.credentials.credentials.get("password")
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/web_api/login",
            data=login_data
        )
        
        response = await self.make_authenticated_request(request)
        
        if response.success and response.data:
            # Store session ID for subsequent requests
            session_id = response.data.get("sid")
            if session_id:
                self.endpoint.credentials.credentials["session_id"] = session_id
        
        return response
    
    async def logout(self) -> IntegrationResponse:
        """Logout from Check Point management server"""
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/web_api/logout",
            data={}
        )
        
        return await self.make_authenticated_request(request)
    
    async def block_ip(self, ip_address: str, reason: str = "XORB Threat Block") -> IntegrationResponse:
        """Block IP address on Check Point firewall"""
        # Ensure we're logged in
        if "session_id" not in self.endpoint.credentials.credentials:
            login_response = await self.login()
            if not login_response.success:
                return login_response
        
        host_name = f"XORB_BLOCK_{ip_address.replace('.', '_')}"
        
        # Create host object
        host_data = {
            "name": host_name,
            "ip-address": ip_address,
            "comments": reason
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/web_api/add-host",
            data=host_data
        )
        
        response = await self.make_authenticated_request(request)
        
        if response.success:
            # Add to blocked group
            await self._add_to_group(host_name, "XORB_BLOCKED_HOSTS")
            # Publish changes
            await self._publish()
        
        return response
    
    async def unblock_ip(self, ip_address: str) -> IntegrationResponse:
        """Unblock IP address on Check Point firewall"""
        host_name = f"XORB_BLOCK_{ip_address.replace('.', '_')}"
        
        # Remove from group
        await self._remove_from_group(host_name, "XORB_BLOCKED_HOSTS")
        
        # Delete host object
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/web_api/delete-host",
            data={"name": host_name}
        )
        
        response = await self.make_authenticated_request(request)
        
        if response.success:
            await self._publish()
        
        return response
    
    async def _add_to_group(self, object_name: str, group_name: str) -> IntegrationResponse:
        """Add object to group"""
        group_data = {
            "name": group_name,
            "members": {"add": object_name}
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/web_api/set-group",
            data=group_data
        )
        
        return await self.make_authenticated_request(request)
    
    async def _remove_from_group(self, object_name: str, group_name: str) -> IntegrationResponse:
        """Remove object from group"""
        group_data = {
            "name": group_name,
            "members": {"remove": object_name}
        }
        
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/web_api/set-group",
            data=group_data
        )
        
        return await self.make_authenticated_request(request)
    
    async def _publish(self) -> IntegrationResponse:
        """Publish configuration changes"""
        request = IntegrationRequest(
            endpoint_id=self.endpoint.integration_id,
            method="POST",
            path="/web_api/publish",
            data={}
        )
        
        return await self.make_authenticated_request(request)
    
    async def _prepare_auth_headers(self) -> Dict[str, str]:
        """Prepare Check Point-specific authentication headers"""
        headers = {}
        
        session_id = self.endpoint.credentials.credentials.get("session_id")
        if session_id:
            headers["X-chkp-sid"] = session_id
        
        return headers