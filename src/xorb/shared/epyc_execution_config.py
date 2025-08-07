class EPYCExecutionConfig:
    CPU_CORES = 16
    MAX_CONCURRENT_SCANS = 8  # Utilize half cores for scanning
    MAX_CONCURRENT_EXPLOITS = 4  # Reserve cores for exploitation
    MEMORY_PER_SCAN = 512 * 1024 * 1024  # 512MB per scan
    
    # Tool configurations
    NMAP_THREADS = 16
    ZAP_MEMORY = "2g"
    NUCLEI_CONCURRENCY = 16
    
    # Stealth configurations
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    ]
    
    DELAY_RANGES = {
        'stealth': (2, 5),     # 2-5 seconds between requests
        'normal': (0.5, 2),    # 0.5-2 seconds
        'aggressive': (0.1, 0.5)  # 0.1-0.5 seconds
    }