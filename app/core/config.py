import os
from typing import Dict, List, Optional, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings."""

    PROJECT_NAME: str = "Autonomous AI Research Agent"
    API_V1_STR: str = "/api/v1"

    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")

    # Vector DB settings
    INDEX_NAME: str = os.getenv("PINECONE_INDEX", "research-agent")
    DIMENSION: int = 1536  # OpenAI embeddings dimension
    INDEX_SPEC: Dict = {
        "cloud": "aws",
        "region": "us-east-1",  # Use exact region from screenshot
    }

    # Default namespace to use for vector storage
    DEFAULT_NAMESPACE: str = os.getenv(
        "DEFAULT_NAMESPACE", ""
    )  # Empty string by default, unique UUIDs are preferred

    # Additional settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_RESULTS_PER_SOURCE: int = int(os.getenv("MAX_RESULTS_PER_SOURCE", "5"))

    # Scalability settings
    WORKER_CONCURRENCY: int = int(os.getenv("WORKER_CONCURRENCY", "2"))
    REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))
    USE_CACHE: bool = os.getenv("USE_CACHE", "True").lower() == "true"
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1 hour
    ENABLE_RATE_LIMITING: bool = (
        os.getenv("ENABLE_RATE_LIMITING", "True").lower() == "true"
    )

    # Notification system (generic replacement for Slack)
    ENABLE_NOTIFICATIONS: bool = (
        os.getenv("ENABLE_NOTIFICATIONS", "False").lower() == "true"
    )
    NOTIFICATION_PROVIDER: str = os.getenv("NOTIFICATION_PROVIDER", "email")
    NOTIFICATION_CONFIG: Dict = {
        "from_email": os.getenv("NOTIFICATION_EMAIL_FROM", ""),
        "smtp_server": os.getenv("NOTIFICATION_SMTP_SERVER", ""),
        "smtp_port": int(os.getenv("NOTIFICATION_SMTP_PORT", "587")),
        "smtp_username": os.getenv("NOTIFICATION_SMTP_USERNAME", ""),
        "smtp_password": os.getenv("NOTIFICATION_SMTP_PASSWORD", ""),
    }

    # CORS settings
    ALLOWED_HOSTS: List[str] = ["*"]

    @field_validator(
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_ENVIRONMENT",
        "NEWS_API_KEY",
        "GITHUB_TOKEN",
    )
    def validate_required_api_keys(cls, v, field):
        if not v and not os.getenv("TESTING", "False").lower() == "true":
            raise ValueError(f"{field.info['title']} is required")
        return v


# Create a global settings object
settings = Settings()
