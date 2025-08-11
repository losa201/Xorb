#!/usr/bin/env python3
"""
Quick multi-round wargame demonstration
"""

import sys
import os
sys.path.append('/root/Xorb/wargame')

from wargame_orchestrator import WargameOrchestrator

def main():
    print("🚀 XORB Multi-Round Wargame Demonstration")
    print("=" * 60)
    
    # Initialize the orchestrator
    orchestrator = WargameOrchestrator()
    
    # Run 2 quick rounds with minimal delay
    orchestrator.run_continuous_wargame(max_rounds=2, round_delay=2)

if __name__ == "__main__":
    main()