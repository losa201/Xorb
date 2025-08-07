import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    NVIDIA_API_KEY: str
    OPENROUTER_API_KEY: str

    class Config:
        env_file = ".env"

settings = Settings()
