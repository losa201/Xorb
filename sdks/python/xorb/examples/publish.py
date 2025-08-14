from xorb.bus import NATSClient
from xorb.signing import Ed25519Signer
from xorb.subject import SubjectBuilder
import asyncio
import os

async def main():
    # Initialize clients
    nats_client = NATSClient(servers=[os.getenv('NATS_URL', 'nats://localhost:4222')])
    await nats_client.connect()

    signer = Ed25519Signer(os.getenv('SIGNING_KEY'))

    # Build subject
    subject = SubjectBuilder() \
        .with_tenant("example-tenant") \
        .with_domain("inventory") \
        .with_service("warehouse") \
        .with_event("item.created") \
        .build()

    # Create payload
    payload = {
        "item_id": "12345",
        "timestamp": 1718389200,
        "data": {"quantity": 100}
    }

    # Sign payload
    signature = signer.sign(payload)

    # Publish message
    await nats_client.publish(subject, payload, signature)
    print(f"Published to {subject} with signature {signature}")

if __name__ == "__main__":
    asyncio.run(main())

# Example output:
# Published to xorb.example-tenant.inventory.warehouse.item.created with signature 3a7d...8e2f
# (signature format will be hex string representation of Ed25519 signature)
