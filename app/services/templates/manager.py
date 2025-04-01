"""
Template management for research templates.
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
from loguru import logger

from app.services.templates.models import ResearchTemplate, TemplateSectionSchema


class TemplateManager:
    """Manager for research templates."""

    def __init__(self, templates_dir: str = "templates"):
        """
        Initialize the template manager.

        Args:
            templates_dir: Directory to store templates
        """
        self.templates_dir = templates_dir
        self.templates_file = os.path.join(templates_dir, "research_templates.json")
        self.templates = {}
        self._load_templates()

        # Create default templates if none exist
        if not self.templates:
            self._create_default_templates()

    def _load_templates(self) -> None:
        """Load templates from the templates file."""
        try:
            # Create templates directory if it doesn't exist
            os.makedirs(self.templates_dir, exist_ok=True)

            # Load templates if file exists
            if os.path.exists(self.templates_file):
                with open(self.templates_file, "r") as f:
                    templates_data = json.load(f)

                # Convert to ResearchTemplate objects
                for template_dict in templates_data:
                    # Convert report structure to TemplateSectionSchema objects
                    if "report_structure" in template_dict:
                        template_dict["report_structure"] = [
                            TemplateSectionSchema(**section)
                            for section in template_dict["report_structure"]
                        ]

                    # Parse datetime strings to datetime objects
                    if "created_at" in template_dict and isinstance(
                        template_dict["created_at"], str
                    ):
                        template_dict["created_at"] = datetime.fromisoformat(
                            template_dict["created_at"]
                        )
                    if "updated_at" in template_dict and isinstance(
                        template_dict["updated_at"], str
                    ):
                        template_dict["updated_at"] = datetime.fromisoformat(
                            template_dict["updated_at"]
                        )

                    # Create template object
                    template = ResearchTemplate(**template_dict)
                    self.templates[template.id] = template

                logger.info(f"Loaded {len(self.templates)} templates")
            else:
                logger.info("Templates file not found, will create default templates")

        except Exception as e:
            logger.error(f"Error loading templates: {str(e)}")
            # Proceed with empty templates dictionary

    def _save_templates(self) -> None:
        """Save templates to the templates file."""
        try:
            # Create templates directory if it doesn't exist
            os.makedirs(self.templates_dir, exist_ok=True)

            # Convert templates to dictionaries
            templates_data = []
            for template in self.templates.values():
                template_dict = template.dict()

                # Convert datetime objects to strings
                if "created_at" in template_dict and isinstance(
                    template_dict["created_at"], datetime
                ):
                    template_dict["created_at"] = template_dict[
                        "created_at"
                    ].isoformat()
                if "updated_at" in template_dict and isinstance(
                    template_dict["updated_at"], datetime
                ):
                    template_dict["updated_at"] = template_dict[
                        "updated_at"
                    ].isoformat()

                templates_data.append(template_dict)

            # Save to file
            with open(self.templates_file, "w") as f:
                json.dump(templates_data, f, indent=2)

            logger.info(f"Saved {len(templates_data)} templates")

        except Exception as e:
            logger.error(f"Error saving templates: {str(e)}")

    def _create_default_templates(self) -> None:
        """Create default templates."""
        # Standard Research Report
        standard_template = ResearchTemplate(
            id=str(uuid.uuid4()),
            name="Standard Research Report",
            description="A comprehensive general-purpose research report template",
            prompt_template=(
                "You are researching the topic: {query}. "
                "Create a comprehensive research report that analyzes the provided information."
            ),
            report_structure=[
                TemplateSectionSchema(
                    section="Introduction",
                    description="Overview of the topic and research questions",
                    required=True,
                ),
                TemplateSectionSchema(
                    section="Methodology",
                    description="Approach used for the research",
                    required=True,
                ),
                TemplateSectionSchema(
                    section="Findings",
                    description="Main results of the research",
                    required=True,
                ),
                TemplateSectionSchema(
                    section="Analysis",
                    description="Interpretation of the findings",
                    required=True,
                ),
                TemplateSectionSchema(
                    section="Conclusion",
                    description="Summary of key points and recommendations",
                    required=True,
                ),
            ],
            is_default=True,
            tags=["general", "standard"],
        )

        # Technical Deep Dive
        technical_template = ResearchTemplate(
            id=str(uuid.uuid4()),
            name="Technical Deep Dive",
            description="A technical analysis focused on implementation details",
            prompt_template=(
                "You are a technical expert analyzing: {query}. "
                "Create a detailed technical report examining implementation, architecture, "
                "and technical considerations of this topic."
            ),
            report_structure=[
                TemplateSectionSchema(
                    section="Technical Overview",
                    description="High-level technical description",
                    required=True,
                ),
                TemplateSectionSchema(
                    section="Architecture",
                    description="System architecture and components",
                    required=True,
                ),
                TemplateSectionSchema(
                    section="Implementation Details",
                    description="Specific implementation approaches",
                    required=True,
                ),
                TemplateSectionSchema(
                    section="Performance Considerations",
                    description="Analysis of performance aspects",
                    required=True,
                ),
                TemplateSectionSchema(
                    section="Trade-offs",
                    description="Technical trade-offs and design decisions",
                    required=True,
                ),
                TemplateSectionSchema(
                    section="Future Directions",
                    description="Potential technical improvements and research",
                    required=True,
                ),
            ],
            is_default=True,
            tags=["technical", "implementation", "deep-dive"],
        )

        # Trend Analysis
        trend_template = ResearchTemplate(
            id=str(uuid.uuid4()),
            name="Trend Analysis",
            description="Analysis of current trends and future directions",
            prompt_template=(
                "You are analyzing trends related to: {query}. "
                "Create a report that identifies current trends, historical context, "
                "and makes predictions about future developments."
            ),
            report_structure=[
                TemplateSectionSchema(
                    section="Executive Summary",
                    description="Brief overview of key trends",
                    required=True,
                ),
                TemplateSectionSchema(
                    section="Historical Context",
                    description="How the trend has evolved over time",
                    required=True,
                ),
                TemplateSectionSchema(
                    section="Current Landscape",
                    description="Analysis of present state",
                    required=True,
                ),
                TemplateSectionSchema(
                    section="Key Players",
                    description="Important organizations and individuals",
                    required=True,
                ),
                TemplateSectionSchema(
                    section="Future Projections",
                    description="Predictions for future developments",
                    required=True,
                ),
                TemplateSectionSchema(
                    section="Strategic Implications",
                    description="What these trends mean for stakeholders",
                    required=True,
                ),
            ],
            is_default=True,
            tags=["trends", "forecasting", "analysis"],
        )

        # Add templates to the manager
        self.templates[standard_template.id] = standard_template
        self.templates[technical_template.id] = technical_template
        self.templates[trend_template.id] = trend_template

        # Save templates
        self._save_templates()
        logger.info("Created default templates")

    def get_template(self, template_id: str) -> Optional[ResearchTemplate]:
        """
        Get a template by ID.

        Args:
            template_id: ID of the template

        Returns:
            Template if found, None otherwise
        """
        return self.templates.get(template_id)

    def get_default_template(self) -> Optional[ResearchTemplate]:
        """
        Get the default template.

        Returns:
            Default template if found, None otherwise
        """
        for template in self.templates.values():
            if template.is_default:
                return template

        # If no default template is found, return the first template or None
        if self.templates:
            return list(self.templates.values())[0]
        return None

    def get_all_templates(self) -> List[ResearchTemplate]:
        """
        Get all templates.

        Returns:
            List of all templates
        """
        return list(self.templates.values())

    def create_template(self, template_data: Dict[str, Any]) -> ResearchTemplate:
        """
        Create a new template.

        Args:
            template_data: Dictionary with template data

        Returns:
            Created template
        """
        # Generate ID if not provided
        if "id" not in template_data:
            template_data["id"] = str(uuid.uuid4())

        # Set created_at timestamp
        template_data["created_at"] = datetime.now()

        # Convert report_structure to TemplateSectionSchema if provided as dict
        if "report_structure" in template_data and isinstance(
            template_data["report_structure"], list
        ):
            template_data["report_structure"] = [
                (
                    item
                    if isinstance(item, TemplateSectionSchema)
                    else TemplateSectionSchema(**item)
                )
                for item in template_data["report_structure"]
            ]

        # Create template
        template = ResearchTemplate(**template_data)

        # Add to templates
        self.templates[template.id] = template

        # Save templates
        self._save_templates()

        return template

    def update_template(
        self, template_id: str, template_data: Dict[str, Any]
    ) -> Optional[ResearchTemplate]:
        """
        Update an existing template.

        Args:
            template_id: ID of the template to update
            template_data: Dictionary with updated template data

        Returns:
            Updated template if found, None otherwise
        """
        # Check if template exists
        if template_id not in self.templates:
            return None

        # Get existing template
        existing_template = self.templates[template_id]

        # Convert to dict for updating
        template_dict = existing_template.dict()

        # Update fields
        for key, value in template_data.items():
            if key in template_dict and key != "id":  # Don't update ID
                template_dict[key] = value

        # Set updated_at timestamp
        template_dict["updated_at"] = datetime.now()

        # Convert report_structure to TemplateSectionSchema if provided as dict
        if "report_structure" in template_dict and isinstance(
            template_dict["report_structure"], list
        ):
            template_dict["report_structure"] = [
                (
                    item
                    if isinstance(item, TemplateSectionSchema)
                    else TemplateSectionSchema(**item)
                )
                for item in template_dict["report_structure"]
            ]

        # Create updated template
        updated_template = ResearchTemplate(**template_dict)

        # Update templates
        self.templates[template_id] = updated_template

        # Save templates
        self._save_templates()

        return updated_template

    def delete_template(self, template_id: str) -> bool:
        """
        Delete a template by ID.

        Args:
            template_id: ID of the template to delete

        Returns:
            True if deleted, False otherwise
        """
        # Check if template exists
        if template_id not in self.templates:
            return False

        # Check if it's the only default template
        is_default = self.templates[template_id].is_default
        if is_default:
            # Count default templates
            default_count = sum(1 for t in self.templates.values() if t.is_default)

            # Don't delete if it's the only default template
            if default_count <= 1:
                logger.warning("Cannot delete the only default template")
                return False

        # Delete template
        del self.templates[template_id]

        # Save templates
        self._save_templates()

        return True

    def get_templates_by_tag(self, tag: str) -> List[ResearchTemplate]:
        """
        Get templates by tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of templates with the specified tag
        """
        return [t for t in self.templates.values() if tag in t.tags]

    def set_default_template(self, template_id: str) -> bool:
        """
        Set a template as the default.

        Args:
            template_id: ID of the template to set as default

        Returns:
            True if successful, False otherwise
        """
        # Check if template exists
        if template_id not in self.templates:
            return False

        # Update default status
        for tid, template in self.templates.items():
            if tid == template_id:
                template.is_default = True
            else:
                template.is_default = False

        # Save templates
        self._save_templates()

        return True
