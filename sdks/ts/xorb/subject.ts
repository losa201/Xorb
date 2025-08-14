import { describe, it, expect } from 'vitest';

// Regex pattern for valid subject format and components
const SUBJECT_PATTERN = /^xorb\.([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)\.([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)\.([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)\.([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)$/;
const COMPONENT_PATTERN = /^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$/;

export class SubjectBuilder {
  private tenant?: string;
  private domain?: string;
  private service?: string;
  private event?: string;

  setTenant(tenant: string): this {
    if (!COMPONENT_PATTERN.test(tenant)) {
      throw new Error(`Invalid tenant format: ${tenant}`);
    }
    this.tenant = tenant;
    return this;
  }

  setDomain(domain: string): this {
    if (!COMPONENT_PATTERN.test(domain)) {
      throw new Error(`Invalid domain format: ${domain}`);
    }
    this.domain = domain;
    return this;
  }

  setService(service: string): this {
    if (!COMPONENT_PATTERN.test(service)) {
      throw new Error(`Invalid service format: ${service}`);
    }
    this.service = service;
    return this;
  }

  setEvent(event: string): this {
    if (!COMPONENT_PATTERN.test(event)) {
      throw new Error(`Invalid event format: ${event}`);
    }
    this.event = event;
    return this;
  }

  build(): string {
    if (!this.tenant || !this.domain || !this.service || !this.event) {
      throw new Error('All components (tenant, domain, service, event) must be set');
    }
    return `xorb.${this.tenant}.${this.domain}.${this.service}.${this.event}`;
  }
}

export function isValidSubject(subject: string): boolean {
  return SUBJECT_PATTERN.test(subject);
}

// ------------------------------
// Unit Tests
// ------------------------------
describe('SubjectBuilder', () => {
  it('should build a valid subject', () => {
    const subject = new SubjectBuilder()
      .setTenant('acme')
      .setDomain('auth')
      .setService('user')
      .setEvent('created')
      .build();

    expect(subject).toBe('xorb.acme.auth.user.created');
    expect(isValidSubject(subject)).toBe(true);
  });

  it('should reject invalid components', () => {
    expect(() => new SubjectBuilder().setTenant('invalid@name').build())
      .toThrow('Invalid tenant format: invalid@name');

    expect(() => new SubjectBuilder().setDomain('my_domain').build())
      .toThrow('Invalid domain format: my_domain');
  });

  it('should reject missing components', () => {
    expect(() => new SubjectBuilder().build())
      .toThrow('All components (tenant, domain, service, event) must be set');
  });

  it('should validate subject format', () => {
    expect(isValidSubject('xorb.acme.auth.user.created')).toBe(true);
    expect(isValidSubject('xorb.acme.auth.user')).toBe(false); // Missing event
    expect(isValidSubject('xorb..auth.user.created')).toBe(false); // Empty tenant
    expect(isValidSubject('xorb.acme-auth.auth.user.created')).toBe(true);
    expect(isValidSubject('xorb.acme--auth.auth.user.created')).toBe(false); // Double hyphen
  });
});
