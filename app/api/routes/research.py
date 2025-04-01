from fastapi import APIRouter, HTTPException, Path
from loguru import logger
from typing import Dict, Optional

from app.api.models import (
    ResearchRequest,
    ResearchResponse,
    ResearchTemplateRequest,
)
from app.services.research.agent import ResearchAgent
from app.services.templates.manager import TemplateManager

# Initialize services
research_agent = ResearchAgent()
template_manager = TemplateManager()

# Create router
router = APIRouter()


@router.post("", response_model=ResearchResponse)
async def conduct_research(request: ResearchRequest):
    """
    Endpoint to conduct research on a topic.

    This will:
    1. Fetch data from multiple sources (ArXiv, News, GitHub, Wikipedia)
    2. Process and store the data in a vector database
    3. Generate a comprehensive research report
    """
    try:
        logger.info(f"Conducting research on topic: {request.query}")

        # Conduct research
        result = await research_agent.conduct_research(
            query=request.query,
            max_results_per_source=request.max_results_per_source,
        )

        logger.info(f"Research completed for topic: {request.query}")
        return result
    except Exception as e:
        logger.error(f"Error conducting research: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/template", response_model=ResearchResponse)
async def conduct_templated_research(request: ResearchTemplateRequest):
    """
    Endpoint to conduct research using a predefined template.

    Templates provide specialized structure and sources for different types of research.
    """
    try:
        # Get the template
        template = template_manager.get_template(request.template_id)
        if not template:
            raise HTTPException(
                status_code=404,
                detail=f"Template with ID {request.template_id} not found",
            )

        logger.info(
            f"Conducting templated research on topic: {request.query} using template: {template.name}"
        )

        # Conduct research with template
        result = await research_agent.conduct_research(
            query=request.query,
            template_id=request.template_id,
            max_results_per_source=request.max_results_per_source,
        )

        logger.info(
            f"Templated research completed for topic: {request.query} using template: {template.name}"
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error conducting templated research: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
