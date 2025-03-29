from pydantic import BaseModel
from typing import Optional


class SlackResearchRequest(BaseModel):
    query: str
    channel: str
    max_results_per_source: Optional[int] = None


class SlackResearchResponse(BaseModel):
    success: bool
    slack_channel: str
    research_query: str
    error: Optional[str] = None
