from loguru import logger

from app.services.research.agent import ResearchAgent
from app.services.notification_service import NotificationService

# Initialize services
research_agent = ResearchAgent()
notification_service = NotificationService()


async def _conduct_and_notify(
    query: str, recipient: str, max_results_per_source: int = None
):
    """
    Helper function to conduct research and send a notification.

    Args:
        query: The research query
        recipient: The notification recipient
        max_results_per_source: Maximum number of results to fetch per source
    """
    try:
        # Conduct research
        result = await research_agent.conduct_research(
            query=query, max_results_per_source=max_results_per_source
        )

        # Send notification
        await notification_service.send_research_notification(
            recipient=recipient, query=query, report=result["report"]
        )

        logger.info(
            f"Sent research results notification to {recipient} for topic: {query}"
        )
    except Exception as e:
        logger.error(f"Error in background research task: {str(e)}")
        # Try to send error notification
        try:
            await notification_service.send_error_notification(
                recipient=recipient, query=query, error=str(e)
            )
        except Exception:
            logger.error(f"Failed to send error notification to {recipient}")
