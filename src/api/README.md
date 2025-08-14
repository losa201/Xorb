# API Service

The API service is the core interface for the Xorb Cybersecurity Platform, providing RESTful endpoints for all security operations and integrations.

## Architecture

Built using FastAPI with clean architecture principles:
- **Controllers**: HTTP request handlers
- **Services**: Business logic layer
- **Repositories**: Data access layer
- **Domain**: Core entities and models
- **Infrastructure**: External service implementations
- **Middleware**: Security and request processing
- **Routers**: API route definitions

## Features

- **Authentication**: JWT-based security with role-based access control
- **Threat Detection**: Real-time security monitoring endpoints
- **Response Orchestration**: Integration with automated response workflows
- **Monitoring**: Prometheus metrics and health checks
- **Security**: Rate limiting, request validation, and secure headers

## Endpoints

### Authentication
- `POST /auth/login` - User authentication
- `POST /auth/logout` - Session termination
- `GET /auth/me` - Current user information

### Threat Detection
- `GET /threats` - List detected threats
- `GET /threats/{id}` - Threat details
- `POST /threats/scan` - Initiate new scan

### Response Orchestration
- `GET /responses` - List response workflows
- `POST /responses/execute` - Execute response workflow
- `GET /responses/{id}` - Workflow status

### Monitoring & Health
- `GET /health` - Service health check
- `GET /readiness` - Dependency readiness (Redis, Postgres, Temporal)
- `GET /metrics` - Prometheus metrics
- `GET /status` - System status overview

## Environment Variables

| Variable | Description | Default |
|---------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://temporal:temporal@postgres:5432/temporal` |
| `REDIS_URL` | Redis connection string | `redis://redis:6379/0` |
| `TEMPORAL_HOST` | Temporal server address | `temporal:7233` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `JWT_SECRET` | Secret for JWT signing | `xorb-secret-key` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token expiration time | `30` |

## Usage

### Development
```bash
cd src/api
uvicorn app.main:app --reload
```

### Production
```bash
cd src/api
docker build -t xorb-api .
docker run -p 8000:8000 xorb-api
```

Create a local dev token (DEV_MODE=true) and call a protected endpoint:
```bash
curl -s -X POST http://localhost:8000/auth/dev-token | jq -r .access_token > token.txt
AUTH="Authorization: Bearer $(cat token.txt)"
curl -H "$AUTH" http://localhost:8000/agents/
```

## Dependencies

- FastAPI
- Pydantic
- SQLAlchemy
- JWT
- Python-Multipart
- Passlib
- Temporalio

## Testing

Run tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=app
```

## Security

- All endpoints require proper authentication
- Input validation with Pydantic models
- Rate limiting on authentication endpoints
- Secure headers middleware
- JWT token expiration and refresh mechanism

## Monitoring

Metrics available at `/metrics` endpoint:
- Request count by endpoint and method
- Request latency
- Error rates
- Database connection status
- External service health

## API Documentation

Interactive documentation available at:
- http://localhost:8000/docs
- http://localhost:8000/redoc

## Contributing

1. Follow clean architecture principles
2. Write tests for new features
3. Use dependency injection for service registration
4. Maintain clear separation of concerns
5. Document all public APIs
