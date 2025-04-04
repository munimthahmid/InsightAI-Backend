from app.api.routes.research import router as research_router
from app.api.routes.history import router as history_router
from app.api.routes.advanced import router as advanced_router
from app.api.routes.templates import router as templates_router
from app.api.routes.notification import router as notification_router

__all__ = [
    "research_router",
    "history_router",
    "advanced_router",
    "templates_router",
    "notification_router",
]
