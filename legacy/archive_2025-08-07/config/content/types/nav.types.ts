export interface NavLink {
  id: string;
  label: string;
  href: string;
  metadata: {
    order: number;
    isVisible: boolean;
  };
}

export interface NavGroup {
  id: string;
  title: string;
  links: NavLink[];
  metadata: {
    order: number;
    isVisible: boolean;
  };
}