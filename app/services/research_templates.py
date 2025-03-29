"""
Research templates for specific domains and report formats.
"""

from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel
import json
import os
from loguru import logger


class ResearchTemplate(BaseModel):
    """Model for research templates."""

    template_id: str
    name: str
    description: str
    domain: str
    focus_areas: List[str]
    report_structure: List[Dict[str, Any]]
    prompt_template: str
    default_sources: List[str] = ["arxiv", "news", "github", "wikipedia"]


class TemplateManager:
    """Manager for research templates."""

    def __init__(self, templates_file: str = "research_templates.json"):
        """Initialize the template manager with templates file."""
        self.templates_file = templates_file
        self.templates = self._load_templates()

    def _load_templates(self) -> List[ResearchTemplate]:
        """Load templates from file or create default templates if file doesn't exist."""
        try:
            if os.path.exists(self.templates_file):
                with open(self.templates_file, "r") as f:
                    templates_data = json.load(f)
                    return [ResearchTemplate(**template) for template in templates_data]
            else:
                # Create and save default templates
                default_templates = self._create_default_templates()
                self._save_templates(default_templates)
                return default_templates
        except Exception as e:
            logger.error(f"Error loading templates: {str(e)}")
            return self._create_default_templates()

    def _save_templates(self, templates: List[ResearchTemplate]) -> None:
        """Save templates to file."""
        try:
            with open(self.templates_file, "w") as f:
                json.dump([template.dict() for template in templates], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving templates: {str(e)}")

    def _create_default_templates(self) -> List[ResearchTemplate]:
        """Create default templates for various domains."""

        # Academic Literature Review Template
        academic_template = ResearchTemplate(
            template_id="academic-review",
            name="Academic Literature Review",
            description="Comprehensive literature review with academic focus, methodological analysis, and gap identification",
            domain="Academic",
            focus_areas=["Literature Review", "Gap Analysis", "Methodology Assessment"],
            report_structure=[
                {
                    "section": "Introduction",
                    "description": "Overview of the research topic and its importance",
                },
                {
                    "section": "Methodology",
                    "description": "Approach used for literature selection and review",
                },
                {
                    "section": "Literature Review",
                    "description": "Detailed analysis of relevant papers and studies",
                },
                {
                    "section": "Gaps and Opportunities",
                    "description": "Identification of research gaps",
                },
                {
                    "section": "Future Directions",
                    "description": "Suggestions for future research",
                },
                {
                    "section": "References",
                    "description": "List of all sources in proper academic format",
                },
            ],
            prompt_template="""
            You are an academic researcher writing a comprehensive literature review.
            
            Research Topic: "{query}"
            
            Create a detailed academic literature review that:
            1. Introduces the topic and its significance
            2. Explains your methodology for selecting and reviewing literature
            3. Thoroughly analyzes the current state of research on this topic
            4. Identifies gaps in the existing literature
            5. Suggests directions for future research
            6. Includes a properly formatted reference list
            
            Use an academic tone and structure throughout the report.
            """,
        )

        # Market Analysis Template
        market_template = ResearchTemplate(
            template_id="market-analysis",
            name="Market Analysis Report",
            description="Comprehensive market analysis with trends, competitors, opportunities, and recommendations",
            domain="Business",
            focus_areas=[
                "Market Trends",
                "Competitive Analysis",
                "SWOT Analysis",
                "Strategic Recommendations",
            ],
            report_structure=[
                {
                    "section": "Executive Summary",
                    "description": "Brief overview of key findings and recommendations",
                },
                {
                    "section": "Market Overview",
                    "description": "Current state and size of the market",
                },
                {
                    "section": "Key Trends",
                    "description": "Analysis of emerging and current market trends",
                },
                {
                    "section": "Competitive Landscape",
                    "description": "Analysis of key competitors and market players",
                },
                {
                    "section": "Opportunities & Threats",
                    "description": "SWOT analysis focusing on market opportunities",
                },
                {
                    "section": "Strategic Recommendations",
                    "description": "Actionable recommendations based on findings",
                },
                {
                    "section": "Sources",
                    "description": "List of data sources and references",
                },
            ],
            prompt_template="""
            You are a market analyst creating a comprehensive market research report.
            
            Market/Industry: "{query}"
            
            Create a detailed market analysis that:
            1. Provides a concise executive summary of key findings
            2. Analyzes the current market size, growth, and overall landscape
            3. Identifies and explains key market trends and drivers
            4. Examines the competitive landscape and key players
            5. Conducts a brief SWOT analysis focusing on market opportunities
            6. Offers strategic, actionable recommendations
            
            Use a professional business tone with concrete data points when available.
            """,
        )

        # Technology Assessment Template
        tech_template = ResearchTemplate(
            template_id="tech-assessment",
            name="Technology Assessment Report",
            description="Technical analysis of emerging technologies, their applications, maturity, and implementation considerations",
            domain="Technology",
            focus_areas=[
                "Technical Analysis",
                "Implementation Assessment",
                "Risk Evaluation",
                "Future Projections",
            ],
            report_structure=[
                {
                    "section": "Technology Overview",
                    "description": "Introduction to the technology and its core concepts",
                },
                {
                    "section": "Current State",
                    "description": "Analysis of the technology's current development and adoption",
                },
                {
                    "section": "Use Cases",
                    "description": "Practical applications and implementation examples",
                },
                {
                    "section": "Technical Assessment",
                    "description": "Evaluation of capabilities, limitations, and architecture",
                },
                {
                    "section": "Implementation Considerations",
                    "description": "Practical aspects of adoption and integration",
                },
                {
                    "section": "Future Development",
                    "description": "Projected evolution and roadmap",
                },
                {
                    "section": "References",
                    "description": "Technical resources and sources",
                },
            ],
            prompt_template="""
            You are a technology analyst evaluating an emerging or established technology.
            
            Technology: "{query}"
            
            Create a comprehensive technology assessment that:
            1. Explains the technology and its core concepts in clear terms
            2. Analyzes the current state of development and market adoption
            3. Explores practical use cases and implementation examples
            4. Provides a technical assessment of capabilities and limitations
            5. Discusses implementation considerations and challenges
            6. Projects future developments and potential impact
            
            Balance technical accuracy with accessibility, providing enough detail for technical audiences while keeping the report understandable.
            """,
        )

        return [academic_template, market_template, tech_template]

    def get_all_templates(self) -> List[ResearchTemplate]:
        """Get all available templates."""
        return self.templates

    def get_template_by_id(self, template_id: str) -> Optional[ResearchTemplate]:
        """Get a template by ID."""
        for template in self.templates:
            if template.template_id == template_id:
                return template
        return None

    def get_template(self, template_id: str) -> Optional[ResearchTemplate]:
        """Alias for get_template_by_id for backward compatibility."""
        return self.get_template_by_id(template_id)

    def get_templates_by_domain(self, domain: str) -> List[ResearchTemplate]:
        """Get all templates for a specific domain."""
        return [t for t in self.templates if t.domain.lower() == domain.lower()]

    def add_template(self, template: ResearchTemplate) -> bool:
        """Add a new template."""
        try:
            # Check if template with same ID already exists
            if any(t.template_id == template.template_id for t in self.templates):
                logger.error(f"Template with ID {template.template_id} already exists")
                return False

            self.templates.append(template)
            self._save_templates(self.templates)
            return True
        except Exception as e:
            logger.error(f"Error adding template: {str(e)}")
            return False

    def update_template(
        self, template_id: str, updated_template: ResearchTemplate
    ) -> bool:
        """Update an existing template."""
        try:
            for i, template in enumerate(self.templates):
                if template.template_id == template_id:
                    # Ensure the ID remains the same
                    updated_template.template_id = template_id
                    self.templates[i] = updated_template
                    self._save_templates(self.templates)
                    return True

            logger.error(f"Template with ID {template_id} not found")
            return False
        except Exception as e:
            logger.error(f"Error updating template: {str(e)}")
            return False

    def delete_template(self, template_id: str) -> bool:
        """Delete a template."""
        try:
            original_count = len(self.templates)
            self.templates = [t for t in self.templates if t.template_id != template_id]

            if len(self.templates) == original_count:
                logger.error(f"Template with ID {template_id} not found")
                return False

            self._save_templates(self.templates)
            return True
        except Exception as e:
            logger.error(f"Error deleting template: {str(e)}")
            return False
