from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger

from app.api.models.notification import NotificationRequest, NotificationResponse
from app.api.routes.helpers.research_helpers import _conduct_and_notify

# Create router
router = APIRouter()


@router.post("", response_model=NotificationResponse)
async def send_notification(
    request: NotificationRequest, background_tasks: BackgroundTasks
):
    """
    Endpoint to conduct research and send a notification with the results.

    This operates asynchronously - it will start the research and immediately return success,
    then send a notification when the research is complete.
    """
    try:
        # Validate the request
        if not request.query or len(request.query.strip()) < 3:
            raise HTTPException(
                status_code=400, detail="Query must be at least 3 characters"
            )

        # Add the research task to background tasks
        background_tasks.add_task(
            _conduct_and_notify,
            query=request.query,
            recipient=request.recipient,
            max_results_per_source=request.max_results_per_source,
        )

        logger.info(
            f"Started background research with notification for {request.recipient} on topic: {request.query}"
        )

        return {
            "success": True,
            "recipient": request.recipient,
            "research_query": request.query,
        }
    except Exception as e:
        logger.error(f"Error starting research with notification: {str(e)}")
        return {
            "success": False,
            "recipient": request.recipient,
            "research_query": request.query,
            "error": str(e),
        }
