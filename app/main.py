from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys

from app.core.config import settings
from app.api.endpoints import api_router

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    level=settings.LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)


def create_application() -> FastAPI:
    """Create FastAPI application."""
    application = FastAPI(
        title=settings.PROJECT_NAME,
        debug=settings.DEBUG,
    )

    # Set all CORS enabled origins
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add API routes
    application.include_router(api_router, prefix=settings.API_V1_STR)

    @application.get("/")
    def read_root():
        return {"message": f"Welcome to the {settings.PROJECT_NAME} API"}

    @application.get("/health")
    def health_check():
        return {"status": "healthy"}

    return application


app = create_application()

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting {settings.PROJECT_NAME} API server")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
