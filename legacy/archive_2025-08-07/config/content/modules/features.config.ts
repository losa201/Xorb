import { Feature } from '../types/feature.types';

export const FEATURES: Feature[] = [
  {
    id: 'ai-testing',
    title: 'AI-Driven Testing',
    description: 'Leverage advanced AI algorithms for thorough, efficient testing.',
    icon: '🤖',
    metadata: {
      order: 1,
      category: 'ai',
      isVisible: true
    }
  },
  {
    id: 'continuous-monitoring',
    title: 'Continuous Monitoring',
    description: 'Keep an eye on your security posture 24/7.',
    icon: '📈',
    metadata: {
      order: 2,
      category: 'security',
      isVisible: true
    }
  },
  {
    id: 'compliance-automation',
    title: 'Compliance Automation',
    description: 'Stay compliant with automated checks for regulatory standards.',
    icon: '✅',
    metadata: {
      order: 3,
      category: 'compliance',
      isVisible: true
    }
  },
  {
    id: 'multi-tenant',
    title: 'Multi-Tenant Support',
    description: 'Manage multiple clients and projects within one platform.',
    icon: '👥',
    metadata: {
      order: 4,
      category: 'infrastructure',
      isVisible: true
    }
  },
  {
    id: 'real-time-reporting',
    title: 'Real-Time Reporting',
    description: 'Instant insights with detailed, live reports.',
    icon: '📊',
    metadata: {
      order: 5,
      category: 'analytics',
      isVisible: true
    }
  },
  {
    id: 'seamless-integrations',
    title: 'Seamless Integrations',
    description: 'Easily connect to your existing tools and workflows.',
    icon: '🔗',
    metadata: {
      order: 6,
      category: 'infrastructure',
      isVisible: true
    }
  }
];

export const FEATURE_CATEGORIES = [
  'ai',
  'security',
  'compliance',
  'infrastructure',
  'analytics'
] as const;

export type FeatureCategory = typeof FEATURE_CATEGORIES[number];

export const getFeaturesByCategory = (category: FeatureCategory): Feature[] => {
  return FEATURES.filter(feature => feature.metadata.category === category);
};

export const getVisibleFeatures = (): Feature[] => {
  return FEATURES.filter(feature => feature.metadata.isVisible);
};

export const getFeatureById = (id: string): Feature | undefined => {
  return FEATURES.find(feature => feature.id === id);
};

export const FEATURE_TAGS = {
  ai: {
    name: 'AI & Machine Learning',
    description: 'Artificial Intelligence and Machine Learning capabilities'
  },
  security: {
    name: 'Security',
    description: 'Security monitoring and protection features'
  },
  compliance: {
    name: 'Compliance',
    description: 'Regulatory compliance and audit features'
  },
  infrastructure: {
    name: 'Infrastructure',
    description: 'Platform infrastructure and management features'
  },
  analytics: {
    name: 'Analytics',
    description: 'Data analysis and reporting capabilities'
  }
} as const;