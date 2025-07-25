# Xorb Secrets Directory

This directory contains sensitive configuration files for the Xorb platform.

## Required Files

### nvidia_api_key
Your NVIDIA API key for embedding services.
```bash
echo "YOUR_NVIDIA_API_KEY_HERE" > nvidia_api_key
```

### postgres_password
PostgreSQL database password.
```bash
openssl rand -base64 32 > postgres_password
```

## Security Notes

- All files in this directory should have 600 permissions
- Never commit secrets to version control
- Use environment-specific secrets for different deployments
- Rotate secrets regularly

## File Permissions

```bash
chmod 700 .secrets/
chmod 600 .secrets/*
```