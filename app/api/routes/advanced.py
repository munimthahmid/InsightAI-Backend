from fastapi import APIRouter, HTTPException
from loguru import logger

from app.api.models import (
    LiteratureReviewRequest,
    LiteratureReviewResponse,
    ComparisonRequest,
    ComparisonResponse,
)
from app.services.research.agent import ResearchAgent

# Initialize services
research_agent = ResearchAgent()

# Create router
router = APIRouter()


@router.post("/literature-review", response_model=LiteratureReviewResponse)
async def generate_literature_review(request: LiteratureReviewRequest):
    """
    Generate a formal literature review based on previously conducted research.

    This creates an academic-style literature review with proper citations, organization, and analysis.
    """
    try:
        result = await research_agent.generate_literature_review(
            research_id=request.research_id,
            format_type=request.format_type,
            section_format=request.section_format,
            max_length=request.max_length,
        )

        return result
    except ValueError as e:
        logger.error(f"Error generating literature review: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating literature review: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=ComparisonResponse)
async def compare_research(request: ComparisonRequest):
    """
    Compare multiple research topics or papers.

    This performs an in-depth comparison between two or more previously conducted research topics,
    highlighting similarities, differences, and providing actionable insights.
    """
    try:
        result = await research_agent.compare_research(
            research_ids=request.research_ids,
            comparison_aspects=request.comparison_aspects,
            include_visualization=request.include_visualization,
        )

        return result
    except ValueError as e:
        logger.error(f"Error in comparison request: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error comparing research: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
