from prometheus_client import Counter, Histogram, Gauge

class MetricsCollector:
    def __init__(self):
        self.request_counter = Counter(
            'xorb_requests_total', 
            'Requests by endpoint and status',
            ['endpoint', 'status']
        )
        self.latency_histogram = Histogram(
            'xorb_request_latency_seconds', 
            'Request latency by endpoint',
            ['endpoint']
        )
        self.error_gauge = Gauge(
            'xorb_current_errors', 
            'Current errors by type',
            ['error_type']
        )
    
    def record_request(self, endpoint: str, status: str, latency: float) -> None:
        self.request_counter.labels(endpoint=endpoint, status=status).inc()
        self.latency_histogram.labels(endpoint=endpoint).observe(latency)
    
    def record_error(self, error_type: str) -> None:
        self.error_gauge.labels(error_type=error_type).inc()