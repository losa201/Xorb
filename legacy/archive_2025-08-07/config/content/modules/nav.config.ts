import { NavLink } from '../types/site.types';

export const NAV_CONFIG: NavLink[] = [
  { 
    id: 'features',
    label: 'Features', 
    href: '#features',
    metadata: {
      order: 1,
      isVisible: true
    }
  },
  { 
    id: 'pricing',
    label: 'Pricing', 
    href: '#pricing',
    metadata: {
      order: 2,
      isVisible: true
    }
  },
  { 
    id: 'about',
    label: 'About', 
    href: '#about',
    metadata: {
      order: 3,
      isVisible: true
    }
  },
  { 
    id: 'login',
    label: 'Login', 
    href: '/login',
    metadata: {
      order: 4,
      isVisible: true
    }
  }
];

export default NAV_CONFIG;