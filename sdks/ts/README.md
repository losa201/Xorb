# XORB TypeScript SDK

## Installation

```bash
npm install @xorb/sdk
```

## Usage

### Subject Builder

```typescript
import { SubjectBuilder } from '@xorb/sdk';

const builder = new SubjectBuilder('mytenant');
const subject = builder.build('mydomain', 'myservice', 'myevent');
console.log(subject); // outputs: xorb.mytenant.mydomain.myservice.myevent
```

### Subject Validation

```typescript
import { is_valid_subject } from '@xorb/sdk';

console.log(is_valid_subject('xorb.mytenant.mydomain.myservice.myevent')); // true
console.log(is_valid_subject('invalid.subject')); // false
```

### Signing

```typescript
import { Ed25519Signer } from '@xorb/sdk';

const signer = new Ed25519Signer();
const data = new TextEncoder().encode('test data');
const signature = signer.sign(data);
console.log('Signature:', Buffer.from(signature).toString('hex'));

const isValid = signer.verify(data, signature);
console.log('Signature valid:', isValid); // true
```

### NATS Bus

```typescript
import { NatsBus } from '@xorb/sdk';

const bus = new NatsBus('nats://localhost:4222');

// Publish
await bus.publish('xorb.mytenant.mydomain.myservice.myevent', Buffer.from('hello world'));

// Subscribe
await bus.subscribe('xorb.mytenant.>', async (err, msg) => {
  if (err) throw err;
  console.log(`Received: ${Buffer.from(msg.data)}`);
});
```

## Configuration

The SDK can be configured with environment variables:

- `XORB_NATS_URL`: The NATS server URL (default: `nats://localhost:4222`)
- `XORB_SUBJECT_PREFIX`: The subject prefix (default: `xorb`)

## Testing

```bash
npm test
```

## Linting

```bash
npm run lint
```