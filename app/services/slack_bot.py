import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from typing import Dict, Any, Optional, List
from loguru import logger

from app.core.config import settings


class SlackBot:
    """Integration with Slack for delivering research reports."""

    def __init__(self):
        """Initialize the Slack bot client."""
        self.token = settings.SLACK_BOT_TOKEN
        self.initialized = bool(self.token)

        if not self.initialized:
            logger.warning(
                "Slack bot token not provided, Slack integration will not work"
            )
            return

        self.client = WebClient(token=self.token)
        logger.info("Slack bot initialized successfully")

    def _check_initialized(self):
        """Check if the Slack bot is initialized."""
        if not self.initialized:
            raise ValueError(
                "Slack bot not initialized. Check SLACK_BOT_TOKEN environment variable."
            )

    async def post_research_report(
        self, channel: str, query: str, report: str
    ) -> Dict[str, Any]:
        """
        Post a research report to a Slack channel.

        Args:
            channel: The Slack channel to post to
            query: The research query
            report: The research report

        Returns:
            Dictionary with the result of the operation
        """
        self._check_initialized()

        try:
            # Format the message for Slack
            # Slack limits messages to 40K characters, so we may need to split
            if len(report) > 38000:  # Leave room for headers and formatting
                logger.warning(
                    f"Research report too long for Slack ({len(report)} chars), truncating"
                )
                report = (
                    report[:38000]
                    + "...\n\n*Note: This report was truncated due to Slack message size limits.*"
                )

            # Create message blocks
            blocks = self._create_report_blocks(query, report)

            # Post the message to the specified channel
            response = self.client.chat_postMessage(
                channel=channel,
                blocks=blocks,
                text=f"Research Report: {query}",  # Fallback text
            )

            logger.info(f"Research report posted to Slack channel: {channel}")

            return {
                "success": True,
                "timestamp": response["ts"],
                "channel": response["channel"],
            }

        except SlackApiError as e:
            logger.error(f"Error posting to Slack: {str(e)}")
            return {"success": False, "error": str(e)}

    async def post_error_message(
        self, channel: str, query: str, error: str
    ) -> Dict[str, Any]:
        """
        Post an error message to a Slack channel.

        Args:
            channel: The Slack channel to post to
            query: The research query
            error: The error message

        Returns:
            Dictionary with the result of the operation
        """
        self._check_initialized()

        try:
            # Create message blocks
            blocks = [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": f"Research Error: {query}"},
                },
                {"type": "divider"},
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Error occurred during research:*\n```{error}```",
                    },
                },
            ]

            # Post the message to the specified channel
            response = self.client.chat_postMessage(
                channel=channel,
                blocks=blocks,
                text=f"Research Error: {query}",  # Fallback text
            )

            logger.info(f"Error message posted to Slack channel: {channel}")

            return {
                "success": True,
                "timestamp": response["ts"],
                "channel": response["channel"],
            }

        except SlackApiError as e:
            logger.error(f"Error posting error message to Slack: {str(e)}")
            return {"success": False, "error": str(e)}

    def _create_report_blocks(self, query: str, report: str) -> List[Dict[str, Any]]:
        """
        Create Slack message blocks for a research report.

        Args:
            query: The research query
            report: The research report

        Returns:
            List of Slack message blocks
        """
        # Split the report into sections based on markdown headers
        sections = []
        current_section = ""
        for line in report.split("\n"):
            if line.startswith("# ") or line.startswith("## "):
                if current_section:
                    sections.append(current_section)
                current_section = line
            else:
                current_section += "\n" + line

        if current_section:
            sections.append(current_section)

        # Create message blocks
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"Research Report: {query}"},
            },
            {"type": "divider"},
        ]

        # Add sections
        if sections:
            for section in sections:
                # Ensure section doesn't exceed Slack's text limit (3000 chars)
                if len(section) > 2900:
                    section = section[:2900] + "...(truncated)"

                blocks.append(
                    {"type": "section", "text": {"type": "mrkdwn", "text": section}}
                )
        else:
            # If no sections were identified, add the whole report
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": report[:2900] if len(report) > 2900 else report,
                    },
                }
            )

        # Add footer
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "Generated by *Autonomous AI Research Agent*",
                    }
                ],
            }
        )

        return blocks
