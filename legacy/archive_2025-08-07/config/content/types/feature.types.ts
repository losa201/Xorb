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

export interface FeatureCategory {
  id: string;
  name: string;
  description: string;
  icon: string;
  features: Feature[];
}