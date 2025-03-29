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
    SLACK_BOT_TOKEN: Optional[str] = os.getenv("SLACK_BOT_TOKEN", "")

    # Vector DB settings
    INDEX_NAME: str = os.getenv("PINECONE_INDEX", "research-agent")
    DIMENSION: int = 1536  # OpenAI embeddings dimension
    INDEX_SPEC: Dict = {
        "cloud": "aws",
        "region": os.getenv("PINECONE_ENVIRONMENT", "us-east-1"),
    }

    # Additional settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_RESULTS_PER_SOURCE: int = int(os.getenv("MAX_RESULTS_PER_SOURCE", "5"))

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
