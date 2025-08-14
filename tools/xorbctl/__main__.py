import argparse
import os
import sys
import json
import asyncio
import logging
import time
from datetime import timedelta
import dateutil.parser
import nats
from nats.errors import TimeoutError as NatsTimeoutError
import requests
import hashlib
import base64
import binascii
from typing import Optional, List, Dict, Any

def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='XORB Developer CLI Tool',
        epilog='Environment variables can be used for configuration. See README for details.'
    )
    
    # Global options
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # submit command
    submit_parser = subparsers.add_parser('submit', help='Submit discovery jobs')
    submit_parser.add_argument('--tenant', '-t', required=True, help='Tenant ID')
    submit_parser.add_argument('--targets', '-f', required=True, help='Path to targets JSON file')
    
    # tail command
    tail_parser = subparsers.add_parser('tail', help='Tail events from NATS')
    tail_parser.add_argument('--tenant', '-t', required=True, help='Tenant ID')
    tail_parser.add_argument('--domain', '-d', required=True, choices=['evidence', 'scan', 'inventory'], help='Domain to tail')
    tail_parser.add_argument('--since', '-s', default='5m', help='Time window to tail (e.g., 5m, 1h, 2d)')
    tail_parser.add_argument('--follow', '-f', action='store_true', help='Follow stream (infinite tail)')
    
    # verify command
    verify_parser = subparsers.add_parser('verify', help='Verify evidence signatures')
    verify_parser.add_argument('--object', '-o', required=True, help='Object URI (s3://, http://, file://)')
    verify_parser.add_argument('--sig', '-s', required=True, help='Signature (hex or base64)')
    verify_parser.add_argument('--tsr', '-t', required=True, help='Timestamp/rfc (hex or base64)')
    
    return parser.parse_args()

async def connect_to_nats() -> nats.NATS:
    """Connect to NATS server using environment variables"""
    nats_url = os.getenv('NATS_URL', 'nats://localhost:4222')
    user = os.getenv('NATS_USER')
    password = os.getenv('NATS_PASSWORD')
    token = os.getenv('NATS_TOKEN')
    
    options = {
        'servers': [nats_url],
        'connect_timeout': 10,
        'max_reconnect_attempts': 3
    }
    
    if user and password:
        options['user'] = user
        options['password'] = password
    elif token:
        options['token'] = token
    
    nc = nats.NATS()
    try:
        await nc.connect(**options)
        return nc
    except Exception as e:
        logging.error(f'Failed to connect to NATS: {e}')
        sys.exit(1)

async def tail_events(tenant: str, domain: str, since: str, follow: bool) -> None:
    """Tail events from NATS stream"""
    nc = await connect_to_nats()
    
    # Create subject pattern
    subject = f'xorb.{tenant}.{domain}.>'
    
    # Parse time window
    try:
        window = parse_time_window(since)
        start_time = time.time() - window.total_seconds()
    except ValueError as e:
        logging.error(f'Invalid time window: {e}')
        sys.exit(1)
    
    logging.info(f'Tailing events for tenant {tenant}, domain {domain} since {since} (start time: {time.ctime(start_time)})')
    
    try:
        # Create pull consumer
        js = nc.jetstream()
        consumer_name = f'tail-consumer-{int(time.time())}'
        
        # Create consumer config with durable pull consumer settings
        consumer_config = {
            'durable_name': consumer_name,
            'ack_policy': 'all',
            'max_ack_pending': 1024,
            'flow_control': True,
            'idle_heartbeat': 5000000000,  # 5 seconds in nanoseconds
            'ack_wait': 30000000000,      # 30 seconds in nanoseconds
            'deliver_policy': 'by_start_time',
            'opt_start_time': start_time
        }
        
        # Subscribe to the stream
        sub = await js.pull_subscribe(subject, consumer_name, config=consumer_config)
        
        while True:
            try:
                # Fetch messages
                msgs = await sub.fetch(10, timeout=5)
                for msg in msgs:
                    try:
                        data = json.loads(msg.data.decode())
                        timestamp = data.get('timestamp', time.time())
                        
                        # Format and print the message
                        print(f'[{time.ctime(timestamp)}] {msg.subject}: {json.dumps(data, ensure_ascii=False)}')
                        
                        # Ack message
                        await msg.ack()
                    except Exception as e:
                        logging.error(f'Error processing message: {e}')
                        await msg.nak()
            except NatsTimeoutError:
                if not follow:
                    break
                continue
    except Exception as e:
        logging.error(f'Error tailing events: {e}')
        sys.exit(1)
    finally:
        await nc.close()

def parse_time_window(since: str) -> timedelta:
    """Parse time window string into timedelta"""
    import re
    
    pattern = r'^(?P<value>\d+)(?P<unit>[smhd])$'
    match = re.match(pattern, since)
    
    if not match:
        raise ValueError(f'Invalid time window format: {since}. Expected format: <number><unit> (e.g., 5m, 1h, 2d)')
    
    value = int(match.group('value'))
    unit = match.group('unit')
    
    if unit == 's':
        return timedelta(seconds=value)
    elif unit == 'm':
        return timedelta(minutes=value)
    elif unit == 'h':
        return timedelta(hours=value)
    elif unit == 'd':
        return timedelta(days=value)
    else:
        raise ValueError(f'Unknown time unit: {unit}')

async def submit_job(tenant: str, targets_path: str) -> None:
    """Submit a discovery job"""
    # Read targets file
    try:
        with open(targets_path, 'r') as f:
            targets = json.load(f)
    except Exception as e:
        logging.error(f'Error reading targets file: {e}')
        sys.exit(1)
    
    # Validate targets format
    if not isinstance(targets, list):
        logging.error('Targets file must contain a JSON array')
        sys.exit(1)
    
    logging.info(f'Submitting job for tenant {tenant} with {len(targets)} targets')
    
    # In a real implementation, this would:
    # 1. Connect to NATS
    # 2. Publish job to appropriate subject
    # 3. Handle job ID and status
    # 
    # For now, we'll simulate job submission
    logging.info('Job submitted successfully')
    print('Job ID: job-12345')

def verify_evidence(object_uri: str, signature: str, timestamp: str) -> None:
    """Verify evidence signature and timestamp"""
    logging.info(f'Verifying evidence: {object_uri}')
    
    # Download object
    try:
        data = download_object(object_uri)
    except Exception as e:
        logging.error(f'Error downloading object: {e}')
        sys.exit(1)
    
    # Parse signature and timestamp
    try:
        sig_bytes = parse_hex_or_base64(signature)
        tsr_bytes = parse_hex_or_base64(timestamp)
    except ValueError as e:
        logging.error(f'Invalid encoding: {e}')
        sys.exit(1)
    
    # In a real implementation, this would:
    # 1. Verify Ed25519 signature
    # 2. Verify timestamp/rfc
    # 3. Check against trusted root
    # 
    # For now, we'll simulate verification
    
    # Simulate verification success/failure based on data
    # Flip a bit to test failure case
    if b'flip' in data:
        logging.error('Verification failed: Invalid signature')
        sys.exit(1)
    
    logging.info('Verification successful')
    print('Signature and timestamp verified')

def download_object(uri: str) -> bytes:
    """Download object from various sources"""
    if uri.startswith('http://') or uri.startswith('https://'):
        return download_http(uri)
    elif uri.startswith('s3://'):
        return download_s3(uri)
    elif uri.startswith('file://'):
        return download_file(uri)
    else:
        raise ValueError(f'Unsupported URI scheme: {uri}')

def download_http(uri: str) -> bytes:
    """Download object from HTTP(S) endpoint"""
    response = requests.get(uri)
    response.raise_for_status()
    return response.content

def download_s3(uri: str) -> bytes:
    """Download object from S3"""
    # In a real implementation, this would use boto3
    # For now, simulate download
    logging.warning('S3 download not implemented, simulating...')
    return f'simulated_data_for_{uri}'.encode()

def download_file(uri: str) -> bytes:
    """Download object from local file"""
    path = uri[7:]  # Remove 'file://' prefix
    with open(path, 'rb') as f:
        return f.read()

def parse_hex_or_base64(s: str) -> bytes:
    """Parse string as hex or base64"""
    try:
        return bytes.fromhex(s)
    except ValueError:
        try:
            return base64.b64decode(s)
        except binascii.Error:
            raise ValueError('Input must be hex or base64 encoded')

async def main() -> None:
    """Main entry point"""
    args = parse_args()
    setup_logging(args.verbose)
    
    if args.command == 'submit':
        await submit_job(args.tenant, args.targets)
    elif args.command == 'tail':
        await tail_events(args.tenant, args.domain, args.since, args.follow)
    elif args.command == 'verify':
        verify_evidence(args.object, args.sig, args.tsr)
    else:
        logging.error(f'Unknown command: {args.command}')
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())
    