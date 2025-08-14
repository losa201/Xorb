
# incident_response.py
# Core incident response orchestration

class IncidentResponseOrchestrator:
    def __init__(self):
        self.playbooks = {}
        self.severity_thresholds = {
            'low': 1,
            'medium': 3,
            'high': 5,
            'critical': 7
        }

    def register_playbook(self, incident_type, severity_level, playbook):
        """Register a playbook for a specific incident type and severity level"""
        if incident_type not in self.playbooks:
            self.playbooks[incident_type] = {}
        self.playbooks[incident_type][severity_level] = playbook

    def evaluate_incident(self, incident_type, threat_score):
        """Evaluate incident and determine appropriate response level"""
        if incident_type not in self.playbooks:
            return 'no_playbook', 0

        # Determine severity level based on threat score
        severity_level = 'low'
        for level, threshold in self.severity_thresholds.items():
            if threat_score >= threshold:
                severity_level = level

        # Find the most appropriate playbook
        if severity_level in self.playbooks[incident_type]:
            return severity_level, self.playbooks[incident_type][severity_level]

        # Fallback to highest available playbook
        available_levels = sorted(self.playbooks[incident_type].keys(),
                                key=lambda x: self.severity_thresholds.get(x, 0),
                                reverse=True)
        if available_levels:
            return available_levels[0], self.playbooks[incident_type][available_levels[0]]

        return 'no_playbook', 0

    def execute_response(self, incident_type, threat_score):
        """Execute appropriate response based on incident type and threat score"""
        severity_level, playbook = self.evaluate_incident(incident_type, threat_score)

        if playbook:
            print(f"Executing {severity_level} severity response for {incident_type}")
            return playbook.execute()
        else:
            print(f"No playbook found for {incident_type}")
            return {'status': 'failed', 'reason': 'no_playbook'}

# Example playbook implementation
class Playbook:
    def __init__(self, name, description, actions, escalation_threshold=None):
        self.name = name
        self.description = description
        self.actions = actions
        self.escalation_threshold = escalation_threshold

    def execute(self):
        """Execute the playbook actions"""
        results = []
        for action in self.actions:
            try:
                result = action.execute()
                results.append({
                    'action': action.name,
                    'status': 'success',
                    'result': result
                })
            except Exception as e:
                results.append({
                    'action': action.name,
                    'status': 'failed',
                    'error': str(e)
                })
                # Stop execution on critical failure unless action specifies continue_on_failure
                if not getattr(action, 'continue_on_failure', False):
                    return {
                        'status': 'partial',
                        'results': results
                    }

        return {
            'status': 'complete',
            'results': results
        }

# Example action implementation
class Action:
    def __init__(self, name, description, system_interface, parameters,
                 continue_on_failure=False):
        self.name = name
        self.description = description
        self.system_interface = system_interface
        self.parameters = parameters
        self.continue_on_failure = continue_on_failure

    def execute(self):
        """Execute the action through the system interface"""
        return self.system_interface.execute(self.parameters)

# Example system interface
class SystemInterface:
    def __init__(self, system_name, connection_info):
        self.system_name = system_name
        self.connection_info = connection_info
        self.connected = False

    def connect(self):
        """Establish connection to the system"""
        # In a real implementation, this would establish a connection
        # to the actual system using the connection_info
        self.connected = True
        return {'status': 'connected', 'system': self.system_name}

    def execute(self, parameters):
        """Execute command on the system"""
        if not self.connected:
            raise Exception(f"Not connected to {self.system_name}")

        # In a real implementation, this would execute the command
        # on the connected system
        return {
            'status': 'success',
            'command': parameters.get('command'),
            'output': f"Executed {parameters.get('command')} on {self.system_name}"
        }

    def disconnect(self):
        """Close connection to the system"""
        self.connected = False
        return {'status': 'disconnected', 'system': self.system_name}

# Example SOAR platform integration
class SOARIntegration:
    def __init__(self, platform_url, api_key):
        self.platform_url = platform_url
        self.api_key = api_key

    def create_incident(self, incident_data):
        """Create incident in SOAR platform"""
        # In a real implementation, this would make an API call to the SOAR platform
        return {
            'status': 'success',
            'incident_id': 'SOAR-2025-001',
            'url': f"{self.platform_url}/incidents/SOAR-2025-001"
        }

    def update_incident(self, incident_id, update_data):
        """Update incident in SOAR platform"""
        # In a real implementation, this would make an API call to the SOAR platform
        return {
            'status': 'success',
            'incident_id': incident_id,
            'update': update_data
        }

    def execute_playbook(self, incident_id, playbook_name):
        """Execute playbook in SOAR platform"""
        # In a real implementation, this would make an API call to the SOAR platform
        return {
            'status': 'success',
            'incident_id': incident_id,
            'playbook_executed': playbook_name,
            'result': 'Playbook completed successfully'
        }

# Example usage
if __name__ == '__main__':
    # Create orchestrator
    orchestrator = IncidentResponseOrchestrator()

    # Create system interfaces
    firewall_interface = SystemInterface('firewall', {'host': '192.168.1.1'})
    siem_interface = SystemInterface('siem', {'host': '192.168.1.2'})

    # Define actions for malware playbook
    isolate_host_action = Action(
        name='isolate_host',
        description='Isolate infected host from network',
        system_interface=firewall_interface,
        parameters={'command': 'isolate', 'host': 'malicious_host'}
    )

    collect_logs_action = Action(
        name='collect_logs',
        description='Collect logs from SIEM system',
        system_interface=siem_interface,
        parameters={'command': 'search', 'query': 'malicious_activity'}
    )

    # Create malware playbook
    malware_playbook = Playbook(
        name='malware_response',
        description='Standard response to malware incidents',
        actions=[isolate_host_action, collect_logs_action]
    )

    # Register playbook with orchestrator
    orchestrator.register_playbook('malware', 'medium', malware_playbook)

    # Execute response
    result = orchestrator.execute_response('malware', 4)
    print("\nResponse Result:", result)

# This implementation provides:
# - Automated playbook execution based on incident type and severity
# - Integration with security systems through standardized interfaces
# - Flexible action definition with error handling
# - SOAR platform compatibility
# - Extensible architecture for additional playbooks and systems

# The code is ready to be extended with real system integrations and additional playbooks.
