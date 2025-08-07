// content/types/site.types.ts
export interface SiteConfig {
  id: string;
  name: string;
  domain: string;
  description: string;
  cta: string;
  logo: string;
  metadata: {
    createdAt: Date;
    updatedAt: Date;
    version: number;
  };
}

export interface NavLink {
  id: string;
  label: string;
  href: string;
  metadata: {
    order: number;
    isVisible: boolean;
  };
}

export interface Feature {
  id: string;
  title: string;
  description: string;
  icon: string;
  metadata: {
    order: number;
    category: 'ai' | 'security' | 'compliance' | 'infrastructure';
    isVisible: boolean;
  };
}

export interface TrustedCompany {
  id: string;
  name: string;
  logo: string;
  metadata: {
    order: number;
    isVisible: boolean;
  };
}