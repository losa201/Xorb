from xorb.common.config import settings

def main():
    print("XORB PTaaS Engine")
    print("===============")
    print(f"NVIDIA_API_KEY: {settings.NVIDIA_API_KEY[:5]}...")
    print(f"OPENROUTER_API_KEY: {settings.OPENROUTER_API_KEY[:5]}...")

if __name__ == "__main__":
    main()
