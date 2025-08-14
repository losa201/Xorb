#!/usr/bin/env python3
"""
Demonstration of a single wargame round
"""

import sys
import os
sys.path.append('/root/Xorb/wargame')

from wargame_orchestrator import WargameOrchestrator

def main():
    print("🚀 XORB Red vs Blue Wargame Demonstration")
    print("=" * 60)
    
    # Initialize the orchestrator
    orchestrator = WargameOrchestrator()
    
    # Execute a single round
    print("Executing demonstration round...")
    round_summary = orchestrator.execute_wargame_round()
    
    print("\n🎯 Demonstration Complete!")
    print("Check the following locations for detailed reports:")
    print("- Red Team Actions: /root/Xorb/wargame/reports/red/")
    print("- Blue Team Defenses: /root/Xorb/wargame/reports/blue/")
    print("- Purple Environment: /root/Xorb/wargame/reports/purple/")

if __name__ == "__main__":
    main()