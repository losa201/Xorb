# Environment Configuration

The `.env` file contains critical configuration parameters for the XORB system. Below is a detailed breakdown of each variable:

## Database Configuration
- `POSTGRES_DB=xorb`
  - PostgreSQL database name
  - Default: "xorb"

- `POSTGRES_USER=xorb`
  - PostgreSQL database username
  - Default: "xorb"

- `POSTGRES_PASSWORD=secure_password`
  - PostgreSQL database password
  - **Security Note:** Should be changed from default "secure_password" in production

## Security Configuration
- `JWT_SECRET=your-secret-key`
  - Secret key for JWT token signing/verification
  - **Security Note:** Must be kept confidential and rotated periodically

- `CORS_ORIGINS=*`
  - Comma-separated list of allowed CORS origins
  - Default: "*" (allow all) - should be restricted in production

## Environment Settings
- `ENVIRONMENT=production`
  - Current environment mode
  - Options: "production", "development", "staging"

## Monitoring Configuration
- `GRAFANA_PASSWORD=xorb-admin-2024`
  - Grafana admin password
  - **Security Note:** Default should be changed in production

- `GRAFANA_USER=admin`
  - Grafana admin username
  - Default: "admin"

## Best Practices
1. Never commit `.env` files to version control
2. Use secret management systems in production (e.g., HashiCorp Vault)
3. Rotate secrets periodically
4. Restrict CORS origins to trusted domains only
5. Set appropriate environment mode for security settings

## Example Usage
```python
import os

postgres_config = {
    'db': os.environ['POSTGRES_DB'],
    'user': os.environ['POSTGRES_USER'],
    'password': os.environ['POSTGRES_PASSWORD']
}
```
