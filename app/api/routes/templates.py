from fastapi import APIRouter, HTTPException, Path
from loguru import logger

from app.api.models import (
    TemplateResponse,
    TemplatesResponse,
)
from app.services.templates.manager import TemplateManager

# Initialize service
template_manager = TemplateManager()

# Create router
router = APIRouter()


@router.get("", response_model=TemplatesResponse)
async def get_templates():
    """
    Get all available research templates.

    Returns a list of templates that can be used for specialized research.
    """
    try:
        templates = template_manager.get_all_templates()

        # Transform to expected response format
        template_responses = []
        for template in templates:
            # Extract section names from report_structure
            structure = [section.section for section in template.report_structure]

            template_responses.append(
                TemplateResponse(
                    id=template.id,
                    name=template.name,
                    description=template.description,
                    domain=template.tags[0] if template.tags else "general",
                    structure=structure,  # Use the extracted section names
                    default_sources=[],  # Set default value since it might not exist in new model
                )
            )

        return {"templates": template_responses}
    except Exception as e:
        logger.error(f"Error retrieving templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{template_id}", response_model=TemplateResponse)
async def get_template(
    template_id: str = Path(..., description="ID of the template to retrieve")
):
    """
    Get a specific template by ID.
    """
    try:
        template = template_manager.get_template(template_id)

        if not template:
            raise HTTPException(
                status_code=404, detail=f"Template with ID {template_id} not found"
            )

        # Extract section names from report_structure
        structure = [section.section for section in template.report_structure]

        return TemplateResponse(
            id=template.id,
            name=template.name,
            description=template.description,
            domain=template.tags[0] if template.tags else "general",
            structure=structure,  # Use the extracted section names
            default_sources=[],  # Set default value since it might not exist in new model
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving template: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
