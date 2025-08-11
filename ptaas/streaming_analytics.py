from kafka import KafkaProducer, KafkaConsumer
from redis import Redis
from redis.commands.json.path import Path
import json
import threading
import time
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StreamingAnalyticsEngine')

class StreamingAnalyticsEngine:
    """Real-time streaming analytics engine for PTaaS platform"""
    
    def __init__(self, kafka_bootstrap='localhost:9092', redis_host='localhost', redis_port=6379):
        # Initialize Kafka producer and consumer
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        self.kafka_consumer = KafkaConsumer(
            'security_events',
            bootstrap_servers=kafka_bootstrap,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        # Initialize Redis Streams connection
        self.redis_client = Redis(host=redis_host, port=redis_port)
        
        # Time window configuration (in seconds)
        self.time_window = 300  # 5 minutes
        
    def produce_event(self, event_type, source, details):
        """Produce a security event to Kafka"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'source': source,
            'details': details
        }
        
        try:
            self.kafka_producer.send('security_events', value=event)
            logger.info(f"Produced event: {event_type} from {source}")
        except Exception as e:
            logger.error(f"Error producing event: {e}")

    def consume_and_aggregate(self):
        """Consume events and perform time-window aggregations"""
        for message in self.kafka_consumer:
            event = message.value
            event_time = datetime.fromisoformat(event['timestamp'])
            
            # Store event in Redis Streams
            self.redis_client.xadd('security_stream', {
                'timestamp': event_time.timestamp(),
                'event_type': event['event_type'],
                'source': event['source'],
                'details': json.dumps(event['details'])
            })
            
            # Maintain time window
            cutoff = time.time() - self.time_window
            self.redis_client.xtrim('security_stream', maxlen=1000, approximate=True)
            
            # Update time-window aggregations
            self._update_aggregations(event_time)

    def _update_aggregations(self, current_time):
        """Update time-window aggregations and detect anomalies"""
        # Get events in current window
        window_start = current_time - timedelta(seconds=self.time_window)
        
        # Get all events in window
        events = self.redis_client.xrange(
            'security_stream', 
            min=f'({window_start.timestamp()}',
            max=f'({current_time.timestamp()}'
        )
        
        # Count events by type
        event_counts = {}
        for _, event_data in events:
            event_type = event_data[b'event_type'].decode('utf-8')
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Store aggregated counts
        self.redis_client.json().set('event_counts', Path.root_path(), event_counts)
        
        # Detect anomalies (example: sudden spike in auth failures)
        auth_failures = event_counts.get('auth_failure', 0)
        if auth_failures > 100:  # Threshold for demonstration
            self._trigger_alert('auth_failure_spike', {
                'count': auth_failures,
                'threshold': 100,
                'time_window': self.time_window
            })

    def _trigger_alert(self, alert_type, details):
        """Trigger an alert and send to Kafka"""
        alert = {
            'timestamp': datetime.utcnow().isoformat(),
            'alert_type': alert_type,
            'details': details
        }
        
        try:
            self.kafka_producer.send('security_alerts', value=alert)
            logger.info(f"Triggered alert: {alert_type}")
        except Exception as e:
            logger.error(f"Error triggering alert: {e}")

    def start(self):
        """Start the streaming analytics engine"""
        # Create consumer thread
        consumer_thread = threading.Thread(target=self.consume_and_aggregate)
        consumer_thread.daemon = True
        consumer_thread.start()
        
        logger.info("Streaming analytics engine started")
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down streaming analytics engine")
            self.kafka_producer.close()

# Example usage
if __name__ == '__main__':
    engine = StreamingAnalyticsEngine()
    
    # Start the engine
    engine.start()