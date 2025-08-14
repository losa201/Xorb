# Unified Authentication API Documentation

## Overview

The XORB Platform's Unified Authentication API provides comprehensive authentication, authorization, and session management capabilities. This API consolidates all authentication functionality into a single, secure, and scalable service.

## Base URL
```
Production: https://api.verteidiq.com/auth
Staging: https://staging-api.verteidiq.com/auth
Development: http://localhost:8000/auth
```

## Authentication Methods

The API supports multiple authentication methods:
- **Username/Password**: Traditional credential-based authentication
- **JWT Tokens**: Bearer token authentication for API access
- **API Keys**: Long-lived keys for service-to-service authentication
- **SSO Integration**: Azure AD, Google, Okta, GitHub SSO
- **Multi-Factor Authentication**: TOTP, WebAuthn, SMS

## Security Features

### Zero Trust Architecture
- Device fingerprinting and verification
- Behavioral analysis and anomaly detection
- Geolocation monitoring
- Adaptive authentication based on risk assessment

### Account Protection
- Progressive account lockout after failed attempts
- Rate limiting on authentication endpoints
- Brute force attack protection
- Session management and token blacklisting

### Cryptographic Security
- Argon2 password hashing with enhanced parameters
- Cryptographically secure random token generation
- JWT tokens with proper expiration and signature verification
- Secure API key generation and validation

---

## Endpoints

### Authentication

#### POST /auth/login
Authenticate user with username and password.

**Request Body:**
```json
{
  "username": "string",
  "password": "string",
  "client_ip": "string",
  "device_fingerprint": "string",
  "remember_me": false
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 1800,
  "expires_at": "2024-01-15T14:30:00Z",
  "user": {
    "id": "uuid",
    "username": "string",
    "email": "string",
    "roles": ["user", "admin"],
    "permissions": ["agent:read", "task:submit"]
  },
  "security_context": {
    "requires_mfa": false,
    "trusted_device": true,
    "risk_score": 0.1
  }
}
```

**Response (401 Unauthorized):**
```json
{
  "success": false,
  "error": "invalid_credentials",
  "message": "Invalid username or password",
  "attempts_remaining": 3
}
```

**Response (423 Locked):**
```json
{
  "success": false,
  "error": "account_locked",
  "message": "Account locked due to too many failed attempts",
  "lockout_until": "2024-01-15T15:00:00Z"
}
```

#### POST /auth/refresh
Refresh access token using refresh token.

**Request Body:**
```json
{
  "refresh_token": "string"
}
```

**Response (200 OK):**
```json
{
  "access_token": "string",
  "token_type": "Bearer",
  "expires_in": 1800,
  "expires_at": "2024-01-15T14:30:00Z"
}
```

#### POST /auth/logout
Logout user and revoke tokens.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "revoke_all_sessions": false
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Successfully logged out"
}
```

#### POST /auth/validate
Validate access token and return user information.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```json
{
  "valid": true,
  "user": {
    "id": "uuid",
    "username": "string",
    "email": "string",
    "roles": ["user"],
    "permissions": ["agent:read"]
  },
  "token_info": {
    "issued_at": "2024-01-15T13:00:00Z",
    "expires_at": "2024-01-15T14:30:00Z",
    "type": "access"
  }
}
```

### API Key Management

#### POST /auth/api-keys
Create a new API key.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "name": "My API Key",
  "scopes": ["read", "write"],
  "expires_in_days": 365
}
```

**Response (201 Created):**
```json
{
  "api_key": "xorb_1234567890abcdef...",
  "key_id": "uuid",
  "name": "My API Key",
  "scopes": ["read", "write"],
  "created_at": "2024-01-15T13:00:00Z",
  "expires_at": "2025-01-15T13:00:00Z"
}
```

#### GET /auth/api-keys
List user's API keys.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```json
{
  "api_keys": [
    {
      "key_id": "uuid",
      "name": "My API Key",
      "scopes": ["read", "write"],
      "created_at": "2024-01-15T13:00:00Z",
      "last_used": "2024-01-15T14:00:00Z",
      "usage_count": 150,
      "expires_at": "2025-01-15T13:00:00Z"
    }
  ]
}
```

#### DELETE /auth/api-keys/{key_id}
Revoke an API key.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "API key revoked successfully"
}
```

### Multi-Factor Authentication

#### POST /auth/mfa/setup
Setup MFA for user account.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "method": "totp",
  "device_name": "My Phone"
}
```

**Response (200 OK):**
```json
{
  "method": "totp",
  "secret": "BASE32SECRET",
  "qr_code": "data:image/png;base64,iVBOR...",
  "backup_codes": ["12345678", "87654321"],
  "setup_token": "temp_setup_token"
}
```

#### POST /auth/mfa/verify
Verify MFA setup or login challenge.

**Request Body:**
```json
{
  "method": "totp",
  "code": "123456",
  "setup_token": "temp_setup_token"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "MFA verified successfully"
}
```

### SSO Authentication

#### GET /auth/sso/{provider}/authorize
Initiate SSO authentication flow.

**Parameters:**
- `provider`: One of `azure`, `google`, `okta`, `github`
- `redirect_uri`: Post-authentication redirect URL
- `state`: CSRF protection state parameter

**Response (302 Redirect):**
Redirects to SSO provider authorization URL.

#### POST /auth/sso/{provider}/callback
Handle SSO callback and complete authentication.

**Request Body:**
```json
{
  "code": "authorization_code",
  "state": "csrf_state_token"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "access_token": "string",
  "refresh_token": "string",
  "user": {
    "id": "uuid",
    "username": "string",
    "email": "string",
    "sso_provider": "azure",
    "sso_id": "external_user_id"
  }
}
```

### Password Management

#### POST /auth/password/change
Change user password.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "current_password": "string",
  "new_password": "string"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Password changed successfully"
}
```

#### POST /auth/password/reset/request
Request password reset.

**Request Body:**
```json
{
  "email": "user@example.com"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Password reset email sent"
}
```

#### POST /auth/password/reset/confirm
Confirm password reset with token.

**Request Body:**
```json
{
  "reset_token": "string",
  "new_password": "string"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Password reset successfully"
}
```

---

## Security Considerations

### Password Requirements
- Minimum 12 characters
- Must contain uppercase, lowercase, number, and special character
- Cannot be common passwords or dictionary words
- Cannot contain personal information

### Token Security
- Access tokens expire in 30 minutes (configurable)
- Refresh tokens expire in 7 days (configurable)
- Tokens are cryptographically signed with HS256
- Revoked tokens are immediately blacklisted

### Rate Limiting
- Authentication attempts: 5 per minute per IP
- Password reset requests: 3 per hour per email
- API key creation: 10 per hour per user
- General API calls: 1000 per hour per user

### Account Lockout
- Account locked after 5 failed login attempts
- Lockout duration: 30 minutes (configurable)
- Progressive delays for repeated failures
- Email notification on account lockout

---

## Error Codes

| Code | Description |
|------|-------------|
| `invalid_credentials` | Username/password incorrect |
| `account_locked` | Account locked due to failed attempts |
| `token_expired` | JWT token has expired |
| `token_invalid` | JWT token is malformed or invalid |
| `insufficient_permissions` | User lacks required permissions |
| `mfa_required` | Multi-factor authentication required |
| `password_weak` | Password doesn't meet requirements |
| `rate_limit_exceeded` | Too many requests |
| `sso_error` | SSO authentication failed |
| `api_key_invalid` | API key is invalid or revoked |

---

## SDKs and Examples

### Python SDK Example
```python
from xorb_sdk import XORBClient

# Initialize client
client = XORBClient(
    base_url="https://api.verteidiq.com",
    api_key="xorb_your_api_key_here"
)

# Authenticate with username/password
auth_result = await client.auth.login(
    username="user@example.com",
    password="SecurePassword123!"
)

# Use access token for API calls
client.set_access_token(auth_result.access_token)

# Create API key
api_key = await client.auth.create_api_key(
    name="Backend Service",
    scopes=["read", "write"]
)
```

### JavaScript SDK Example
```javascript
import { XORBClient } from '@xorb/sdk';

// Initialize client
const client = new XORBClient({
  baseURL: 'https://api.verteidiq.com',
  apiKey: 'xorb_your_api_key_here'
});

// Authenticate
const authResult = await client.auth.login({
  username: 'user@example.com',
  password: 'SecurePassword123!'
});

// Set access token
client.setAccessToken(authResult.accessToken);

// Validate token
const userInfo = await client.auth.validate();
console.log('Authenticated as:', userInfo.user.username);
```

### cURL Examples
```bash
# Login
curl -X POST https://api.verteidiq.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "user@example.com",
    "password": "SecurePassword123!"
  }'

# Use access token
curl -X GET https://api.verteidiq.com/auth/validate \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# Create API key
curl -X POST https://api.verteidiq.com/auth/api-keys \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My API Key",
    "scopes": ["read", "write"]
  }'
```

---

## Monitoring and Observability

### Metrics
The authentication service exposes the following Prometheus metrics:

- `xorb_auth_requests_total` - Total authentication requests
- `xorb_auth_success_rate` - Authentication success rate
- `xorb_auth_lockouts_total` - Total account lockouts
- `xorb_auth_token_validations_total` - Token validation requests
- `xorb_auth_api_key_usage_total` - API key usage count

### Health Check
```
GET /auth/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T14:00:00Z",
  "version": "1.0.0",
  "checks": {
    "database": "healthy",
    "redis": "healthy",
    "vault": "healthy"
  }
}
```

### Audit Logging
All authentication events are logged with the following information:
- User ID and username
- IP address and user agent
- Timestamp and event type
- Success/failure status
- Risk assessment data
- Device fingerprint

---

## Migration Guide

### From Legacy Auth Services

If migrating from the legacy authentication services:

1. **Update imports:**
   ```python
   # Old
   from app.services.auth_security_service import AuthSecurityService

   # New
   from app.services.unified_auth_service_consolidated import UnifiedAuthService
   ```

2. **Update service instantiation:**
   ```python
   # Old
   auth_service = AuthSecurityService(user_repo, redis_client)

   # New
   auth_service = UnifiedAuthService(
       user_repository=user_repo,
       token_repository=token_repo,
       redis_client=redis_client,
       secret_key=config.security.jwt_secret
   )
   ```

3. **Update method calls:**
   ```python
   # Method names remain the same
   result = await auth_service.authenticate_user(credentials)
   token = await auth_service.create_access_token(user)
   ```

### Configuration Migration

Update environment variables:
```bash
# Old scattered configuration
AUTH_SECRET_KEY=...
PASSWORD_MIN_LENGTH=...
MAX_LOGIN_ATTEMPTS=...

# New unified configuration
JWT_SECRET=...
PASSWORD_MIN_LENGTH=12
MAX_LOGIN_ATTEMPTS=5
LOCKOUT_DURATION_MINUTES=30
```

---

## Support and Resources

- **Documentation**: https://docs.verteidiq.com/auth
- **API Reference**: https://api-docs.verteidiq.com
- **SDKs**: https://github.com/verteidiq/sdks
- **Examples**: https://github.com/verteidiq/examples
- **Support**: support@verteidiq.com
