"""
Research service package with components for AI research.
"""

from app.services.research.agent import ResearchAgent
from app.services.research.core import ResearchAgent as CoreResearchAgent
from app.services.research.focused_reports import FocusedReportGenerator
from app.services.research.literature_review import LiteratureReviewGenerator
from app.services.research.history import ResearchHistoryManager
from app.services.research.report import ReportGenerator
from app.services.research.utils import create_empty_result, get_source_counts

__all__ = [
    "ResearchAgent",
    "CoreResearchAgent",
    "FocusedReportGenerator",
    "LiteratureReviewGenerator",
    "ResearchHistoryManager",
    "ReportGenerator",
    "create_empty_result",
    "get_source_counts",
]
