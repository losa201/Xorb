import os
from typing import Dict

class PlatformConfig:
    # EPYC Optimization
    NUMA_NODES = 2
    CPU_CORES = 16
    MEMORY_GB = 32
    
    # Security
    JWT_SECRET = os.getenv("JWT_SECRET")
    if not JWT_SECRET:
        raise ValueError("JWT_SECRET environment variable must be set")
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRY_HOURS = 24
    
    # Rate limiting (per EPYC capabilities)
    RATE_LIMIT_REQUESTS = 1000  # per minute
    RATE_LIMIT_BURST = 50       # burst capacity
    
    # Service mesh
    SERVICES = {
        "intelligence-engine": "http://xorb-intelligence-engine:8001",
        "execution-engine": "http://xorb-execution-engine:8002",
        "ml-defense": "http://xorb-ml-defense:8003",
    }