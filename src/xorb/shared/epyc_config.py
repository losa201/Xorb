class EPYCConfig:
    CPU_CORES = 16
    NUMA_NODES = 2
    MEMORY_GB = 32
    
    # Thread optimization for EPYC
    OMP_NUM_THREADS = 16
    MKL_NUM_THREADS = 16
    TORCH_NUM_THREADS = 16
    
    # ML model configuration
    MAX_MODELS_MEMORY = 8 * 1024 * 1024 * 1024  # 8GB for models
    EMBEDDING_DIM = 768
    MAX_SEQUENCE_LENGTH = 512