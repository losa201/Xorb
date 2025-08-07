from prometheus_client import Counter, Histogram, Gauge

class MetricsCollector:
    """Collects and exposes metrics for the agent system."""
    
    def __init__(self):
        # Initialize metrics
        self.agent_actions = Counter(
            'xorb_agent_actions_total', 
            'Agent actions taken', 
            ['action_type']
        )
        self.threat_context_calls = Counter(
            'xorb_threat_context_calls_total', 
            'Threat context API calls', 
            ['status']
        )
        self.decision_latency = Histogram(
            'xorb_agent_decision_latency_seconds', 
            'Agent decision latency'
        )
        self.memory_usage = Gauge(
            'xorb_agent_memory_usage', 
            'Agent memory usage'
        )
        
    def record_action(self, action_type):
        """Record an agent action."""
        self.agent_actions.labels(action_type=action_type).inc()

    def record_threat_context_call(self, status="success"):
        """Record a threat context API call."""
        self.threat_context_calls.labels(status=status).inc()

    def record_decision_latency(self, latency_seconds):
        """Record decision latency metric."""
        self.decision_latency.observe(latency_seconds)

    def set_memory_usage(self, usage):
        """Update memory usage metric."""
        self.memory_usage.set(usage)