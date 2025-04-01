"""
Templates module for managing research report templates.
"""

from app.services.templates.models import ResearchTemplate
from app.services.templates.manager import TemplateManager

__all__ = [
    "ResearchTemplate",
    "TemplateManager",
]
