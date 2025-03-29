from loguru import logger

from app.services.research_agent import ResearchAgent
from app.services.slack_bot import SlackBot

# Initialize services
research_agent = ResearchAgent()
slack_bot = SlackBot()


async def _conduct_and_post_research(
    query: str, channel: str, max_results_per_source: int = None
):
    """
    Helper function to conduct research and post to Slack.

    Args:
        query: The research query
        channel: The Slack channel to post to
        max_results_per_source: Maximum number of results to fetch per source
    """
    try:
        # Conduct research
        result = await research_agent.research_topic(
            query=query, max_results_per_source=max_results_per_source
        )

        # Post to Slack
        await slack_bot.post_research_report(
            channel=channel, query=query, report=result["report"]
        )

        logger.info(
            f"Posted research results to Slack channel {channel} for topic: {query}"
        )
    except Exception as e:
        logger.error(f"Error in background research task: {str(e)}")
        # Try to post error to Slack
        try:
            await slack_bot.post_error_message(
                channel=channel, query=query, error=str(e)
            )
        except Exception:
            logger.error(f"Failed to post error message to Slack channel {channel}")
