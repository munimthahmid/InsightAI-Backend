"""
Models for research templates.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator
import uuid


class TemplateSectionSchema(BaseModel):
    """Schema for a section in a research template."""

    section: str = Field(..., description="Name of the section")
    description: str = Field(
        ..., description="Description of what this section should contain"
    )
    required: bool = Field(True, description="Whether this section is required")


class ResearchTemplate(BaseModel):
    """
    Research template model for standardized research reports.
    Templates define the structure and sections of generated research reports.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    prompt_template: str = Field(
        ..., description="The template prompt to guide the LLM in generating the report"
    )
    report_structure: List[TemplateSectionSchema] = Field(
        default_factory=list, description="The structure of sections in the report"
    )
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None
    is_default: bool = False
    tags: List[str] = Field(default_factory=list)

    @validator("report_structure")
    def validate_structure(cls, v):
        """Ensure that the structure contains at least one section."""
        if not v:
            raise ValueError("Report structure must contain at least one section")
        return v

    @validator("prompt_template")
    def validate_prompt_template(cls, v):
        """Ensure that the prompt template contains required placeholders."""
        if "{query}" not in v:
            raise ValueError("Prompt template must contain the {query} placeholder")
        return v

    class Config:
        schema_extra = {
            "example": {
                "name": "Standard Research Report",
                "description": "A standard template for general research reports",
                "prompt_template": (
                    "You are researching the topic: {query}. "
                    "Create a comprehensive research report that analyzes the provided information."
                ),
                "report_structure": [
                    {
                        "section": "Introduction",
                        "description": "Overview of the topic and research questions",
                        "required": True,
                    },
                    {
                        "section": "Methodology",
                        "description": "Approach used for the research",
                        "required": True,
                    },
                    {
                        "section": "Findings",
                        "description": "Main results of the research",
                        "required": True,
                    },
                    {
                        "section": "Analysis",
                        "description": "Interpretation of the findings",
                        "required": True,
                    },
                    {
                        "section": "Conclusion",
                        "description": "Summary of key points and recommendations",
                        "required": True,
                    },
                ],
                "is_default": True,
                "tags": ["general", "standard"],
            }
        }
