from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger

from app.api.models import SlackResearchRequest, SlackResearchResponse
from app.api.routes.helpers.research_helpers import _conduct_and_post_research

# Create router
router = APIRouter()


@router.post("", response_model=SlackResearchResponse)
async def slack_research(
    request: SlackResearchRequest, background_tasks: BackgroundTasks
):
    """
    Endpoint to conduct research and post the results to a Slack channel.

    This operates asynchronously - it will start the research and immediately return success,
    then post to Slack when the research is complete.
    """
    try:
        # Validate the request
        if not request.query or len(request.query.strip()) < 3:
            raise HTTPException(
                status_code=400, detail="Query must be at least 3 characters"
            )

        # Add the research task to background tasks
        background_tasks.add_task(
            _conduct_and_post_research,
            query=request.query,
            channel=request.channel,
            max_results_per_source=request.max_results_per_source,
        )

        logger.info(
            f"Started background research for Slack channel {request.channel} on topic: {request.query}"
        )

        return {
            "success": True,
            "slack_channel": request.channel,
            "research_query": request.query,
        }
    except Exception as e:
        logger.error(f"Error starting Slack research: {str(e)}")
        return {
            "success": False,
            "slack_channel": request.channel,
            "research_query": request.query,
            "error": str(e),
        }
