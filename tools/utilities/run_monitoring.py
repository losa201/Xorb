from xorb.monitoring import fusion_monitor
import asyncio

if __name__ == "__main__":
    fusion_monitor.display_banner()
    asyncio.run(fusion_monitor.main())
