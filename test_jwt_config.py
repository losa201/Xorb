#!/usr/bin/env python3
"""
Test JWT configuration to debug pydantic_settings issue
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Set test environment variable
os.environ['JWT_SECRET'] = 'test-jwt-secret-for-validation-12345678901234567890'

print("Environment variable JWT_SECRET:", repr(os.environ.get('JWT_SECRET')))

# Test 1: Minimal working configuration
class MinimalConfig(BaseSettings):
    jwt_secret: str

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=None
    )

print("\n=== Test 1: Minimal working config ===")
try:
    config = MinimalConfig()
    print("SUCCESS:", len(config.jwt_secret), "characters")
except Exception as e:
    print("FAILED:", str(e))

# Test 2: With explicit env field name mapping  
class ExplicitConfig(BaseSettings):
    jwt_secret_key: str = Field(alias="jwt_secret")

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=None
    )

print("\n=== Test 2: With alias ===")
try:
    config = ExplicitConfig()
    print("SUCCESS:", len(config.jwt_secret_key), "characters")
except Exception as e:
    print("FAILED:", str(e))

# Test 3: Direct env configuration
class DirectEnvConfig(BaseSettings):
    jwt_secret_key: str

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=None,
        # Map JWT_SECRET to jwt_secret_key
        extra='ignore'
    )

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        # Custom environment source
        from pydantic_settings.sources import EnvSettingsSource
        
        class CustomEnvSource(EnvSettingsSource):
            def get_field_value(self, field_info, field_name: str):
                if field_name == 'jwt_secret_key':
                    return os.environ.get('JWT_SECRET'), 'JWT_SECRET', True
                return super().get_field_value(field_info, field_name)
        
        return (
            init_settings,
            CustomEnvSource(settings_cls),
        )

print("\n=== Test 3: Custom env source ===")
try:
    config = DirectEnvConfig()
    print("SUCCESS:", len(config.jwt_secret_key), "characters")
except Exception as e:
    print("FAILED:", str(e))
    import traceback
    traceback.print_exc()