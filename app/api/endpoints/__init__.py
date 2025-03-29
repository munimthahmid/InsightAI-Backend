from fastapi import APIRouter
from app.api.routes import (
    research_router,
    history_router,
    advanced_router,
    templates_router,
    slack_router,
)

# Create main router
api_router = APIRouter()

# Include all routers
api_router.include_router(research_router, prefix="/research", tags=["Research"])
api_router.include_router(
    history_router, prefix="/research/history", tags=["Research History"]
)
api_router.include_router(
    advanced_router, prefix="/research/advanced", tags=["Advanced Research"]
)
api_router.include_router(templates_router, prefix="/templates", tags=["Templates"])
api_router.include_router(slack_router, prefix="/slack", tags=["Slack Integration"])
