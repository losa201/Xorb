import { describe, it, expect } from 'vitest';
import { buildSubject, validateSubject } from './subject';

describe('Subject Builder/Validator', () => {
  const validParts = {
    tenant: 'acme',
    domain: 'auth',
    service: 'usersvc',
    event: 'created'
  };

  it('should build valid subjects', () => {
    const subject = buildSubject(validParts);
    expect(subject).toBe('xorb.acme.auth.usersvc.created');
    expect(validateSubject(subject)).toBe(true);
  });

  it('should reject missing parts', () => {
    expect(() => buildSubject({ ...validParts, tenant: '' })).toThrow();
    expect(validateSubject('xorb..auth.usersvc.created')).toBe(false);
  });

  it('should reject invalid characters', () => {
    expect(validateSubject('xorb.acme!auth.usersvc.created')).toBe(false);
  });
});

// Run tests with: npx vitest subject.test.ts