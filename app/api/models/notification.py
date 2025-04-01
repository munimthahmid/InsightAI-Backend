from pydantic import BaseModel, EmailStr
from typing import Optional, List


class NotificationRequest(BaseModel):
    """Model for notification request."""

    query: str
    recipient: str
    max_results_per_source: Optional[int] = None


class NotificationResponse(BaseModel):
    """Model for notification response."""

    success: bool
    recipient: str
    research_query: str
    error: Optional[str] = None
