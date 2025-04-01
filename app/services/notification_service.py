"""
Notification service for the research agent.
This service provides a generic way to send notifications about research results.
Multiple providers can be supported through a pluggable architecture.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from loguru import logger

from app.core.config import settings


class NotificationProvider(ABC):
    """Abstract base class for notification providers."""

    @abstractmethod
    async def send_notification(
        self, recipient: str, subject: str, message: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Send a notification to the recipient.

        Args:
            recipient: The recipient of the notification
            subject: The subject of the notification
            message: The content of the notification
            **kwargs: Additional provider-specific arguments

        Returns:
            Dictionary with the result of the operation
        """
        pass


class EmailNotificationProvider(NotificationProvider):
    """Email notification provider."""

    def __init__(self):
        """Initialize the email notification provider."""
        self.from_email = settings.NOTIFICATION_CONFIG["from_email"]
        self.smtp_server = settings.NOTIFICATION_CONFIG["smtp_server"]
        self.smtp_port = settings.NOTIFICATION_CONFIG["smtp_port"]
        self.smtp_username = settings.NOTIFICATION_CONFIG["smtp_username"]
        self.smtp_password = settings.NOTIFICATION_CONFIG["smtp_password"]

        # Check if the provider is properly configured
        self.initialized = bool(
            self.from_email
            and self.smtp_server
            and self.smtp_username
            and self.smtp_password
        )

        if not self.initialized:
            logger.warning(
                "Email notification provider not properly configured, notifications will not work"
            )

    async def send_notification(
        self, recipient: str, subject: str, message: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Send an email notification.

        Args:
            recipient: Email address of the recipient
            subject: Email subject
            message: Email content (HTML format supported)
            **kwargs: Additional arguments
                html_format: Boolean indicating if the message is in HTML format

        Returns:
            Dictionary with the result of the operation
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "Email notification provider not properly configured",
            }

        try:
            # Create a multi-part message
            msg = MIMEMultipart()
            msg["From"] = self.from_email
            msg["To"] = recipient
            msg["Subject"] = subject

            # Determine content type from kwargs
            is_html = kwargs.get("html_format", False)
            content_type = "html" if is_html else "plain"

            # Add message content
            msg.attach(MIMEText(message, content_type))

            # Connect to SMTP server and send the email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            logger.info(f"Email notification sent to: {recipient}")
            return {"success": True}

        except Exception as e:
            logger.error(f"Error sending email notification: {str(e)}")
            return {"success": False, "error": str(e)}


class ConsoleNotificationProvider(NotificationProvider):
    """Console notification provider for development purposes."""

    async def send_notification(
        self, recipient: str, subject: str, message: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Print notification to console (for development/testing).

        Args:
            recipient: The recipient identifier
            subject: The notification subject
            message: The notification content
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with the result of the operation
        """
        logger.info(f"NOTIFICATION for {recipient}")
        logger.info(f"Subject: {subject}")
        logger.info(
            f"Message: {message[:100]}..."
            if len(message) > 100
            else f"Message: {message}"
        )
        return {"success": True}


class NotificationService:
    """Service for sending notifications about research results."""

    def __init__(self):
        """Initialize the notification service with the configured provider."""
        self.enabled = settings.ENABLE_NOTIFICATIONS

        if not self.enabled:
            logger.info("Notification service disabled by configuration")
            return

        # Initialize the appropriate provider based on configuration
        provider_type = settings.NOTIFICATION_PROVIDER.lower()

        if provider_type == "email":
            self.provider = EmailNotificationProvider()
        elif provider_type == "console":
            self.provider = ConsoleNotificationProvider()
        else:
            logger.warning(f"Unknown notification provider: {provider_type}")
            self.enabled = False
            return

        logger.info(f"Notification service initialized with provider: {provider_type}")

    async def send_research_notification(
        self, recipient: str, query: str, report: str
    ) -> Dict[str, Any]:
        """
        Send a notification about research results.

        Args:
            recipient: The recipient of the notification
            query: The research query
            report: The research report

        Returns:
            Dictionary with the result of the operation
        """
        if not self.enabled:
            return {"success": False, "error": "Notification service is disabled"}

        subject = f"Research Report: {query}"

        # Format report for readability
        formatted_report = self._format_report_for_notification(report)

        return await self.provider.send_notification(
            recipient=recipient,
            subject=subject,
            message=formatted_report,
            html_format=True,
        )

    async def send_error_notification(
        self, recipient: str, query: str, error: str
    ) -> Dict[str, Any]:
        """
        Send a notification about an error during research.

        Args:
            recipient: The recipient of the notification
            query: The research query
            error: The error message

        Returns:
            Dictionary with the result of the operation
        """
        if not self.enabled:
            return {"success": False, "error": "Notification service is disabled"}

        subject = f"Research Error: {query}"
        message = f"<h2>Error occurred during research</h2><pre>{error}</pre>"

        return await self.provider.send_notification(
            recipient=recipient,
            subject=subject,
            message=message,
            html_format=True,
        )

    def _format_report_for_notification(self, report: str) -> str:
        """
        Format a research report for notification.

        This converts markdown to HTML for better readability in emails.

        Args:
            report: The markdown research report

        Returns:
            Formatted HTML content
        """
        # Very basic markdown to HTML conversion
        html = "<html><body>"

        # Split the report by lines
        lines = report.split("\n")

        for line in lines:
            if line.startswith("# "):
                html += f"<h1>{line[2:]}</h1>"
            elif line.startswith("## "):
                html += f"<h2>{line[3:]}</h2>"
            elif line.startswith("### "):
                html += f"<h3>{line[4:]}</h3>"
            elif line.startswith("- "):
                html += f"<li>{line[2:]}</li>"
            elif line.startswith("1. "):
                html += f"<ol><li>{line[3:]}</li></ol>"
            elif line.startswith("```"):
                html += "<pre><code>"
                # Skip the closing code block
                if line == "```":
                    html += "</code></pre>"
            elif line == "":
                html += "<br/>"
            else:
                html += f"<p>{line}</p>"

        html += "<hr/><footer>Generated by <strong>Autonomous AI Research Agent</strong></footer>"
        html += "</body></html>"

        return html
