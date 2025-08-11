#  PTaaS Frontend Service

##  Overview
PTaaS (Penetration Testing as a Service) serves as the primary web interface for the XORB cybersecurity ecosystem. This service provides enterprise-grade penetration testing capabilities through a modern React-based dashboard.

##  Architecture
- **Frontend Framework**: React 18.3.1 with TypeScript 5.5.3
- **Build Tool**: Vite 5.4.1 for fast development and optimized builds
- **Styling**: Tailwind CSS 3.4.11 with Radix UI components
- **State Management**: React Query with custom hooks
- **Data Visualization**: Recharts for security metrics and analytics

##  Service Structure
```
ptaas/
├── web/                    # React application
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── pages/          # Route components
│   │   ├── services/       # API integration
│   │   ├── hooks/          # React hooks
│   │   └── utils/          # Utilities
│   ├── public/             # Static assets
│   └── dist/               # Production build
├── api/                    # PTaaS-specific APIs
├── docs/                   # Service documentation
└── deployment/             # Deployment configurations
```

##  Integration with XORB
PTaaS integrates with the XORB platform through:
- **XORB API Gateway** (Port 8000) for backend services
- **WebSocket connections** for real-time updates
- **Unified authentication** via JWT tokens
- **Multi-tenant architecture** for enterprise deployments

##  Development
```bash
cd services/ptaas/web
npm install
npm run dev
```

##  Production Deployment
```bash
npm run build
#  Deploy to CDN (Vercel, Netlify, etc.)
```