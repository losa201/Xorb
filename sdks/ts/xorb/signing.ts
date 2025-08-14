import nacl from 'tweetnacl';
import { decodeUTF8, encodeBase64 } from 'tweetnacl-js-helper';

export interface EvidenceSigner {
  signEvidence(evidence: string): Promise<string>;
  verifySignature(evidence: string, signature: string): Promise<boolean>;
}

export class Ed25519Signer implements EvidenceSigner {
  private readonly signingKeyPair: nacl.SignKeyPair;

  constructor(private readonly seed: Uint8Array) {
    this.signingKeyPair = nacl.sign.keyPair.fromSeed(seed);
  }

  async signEvidence(evidence: string): Promise<string> {
    const evidenceBytes = decodeUTF8(evidence);
    const signatureBytes = nacl.sign(evidenceBytes, this.signingKeyPair.secretKey);
    return encodeBase64(signatureBytes);
  }

  async verifySignature(evidence: string, signature: string): Promise<boolean> {
    try {
      const evidenceBytes = decodeUTF8(evidence);
      const signatureBytes = decodeUTF8(signature);
      return nacl.sign.open(signatureBytes, evidenceBytes, this.signingKeyPair.publicKey) !== null;
    } catch (error) {
      return false;
    }
  }

  getPublicKey(): string {
    return encodeBase64(this.signingKeyPair.publicKey);
  }
}

// Placeholder timestamp hook
export async function getTimestamp(): Promise<number> {
  // In a real implementation, this would use a trusted time source
  return Date.now();
}