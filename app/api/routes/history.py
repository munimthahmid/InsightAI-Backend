from fastapi import APIRouter, HTTPException, Path
from loguru import logger
from typing import Dict, Optional

from app.api.models import (
    ResearchHistoryResponse,
    ResearchHistoryItem,
    ResearchResponse,
)
from app.services.research_agent import ResearchAgent

# Initialize services
research_agent = ResearchAgent()

# Create router
router = APIRouter()


@router.get("", response_model=ResearchHistoryResponse)
async def get_research_history(limit: Optional[int] = 50):
    """
    Get the history of research queries.

    Returns a list of previous research queries with metadata.
    """
    try:
        history = await research_agent.get_research_history(limit=limit)

        # Transform to expected response format
        history_items = []
        for item in history:
            history_items.append(
                ResearchHistoryItem(
                    research_id=item.get("research_id", ""),
                    query=item.get("query", ""),
                    saved_at=item.get("saved_at", ""),
                    metadata=item.get("metadata", {}),
                    sources=item.get("sources", {}),
                )
            )

        return {"items": history_items, "total": len(history_items)}
    except Exception as e:
        logger.error(f"Error retrieving research history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{research_id}", response_model=ResearchResponse)
async def get_research_by_id(
    research_id: str = Path(..., description="ID of the research to retrieve")
):
    """
    Get a specific research by ID.

    Returns the full research data including the report.
    """
    try:
        research = await research_agent.get_research_by_id(research_id)

        if not research:
            raise HTTPException(
                status_code=404, detail=f"Research with ID {research_id} not found"
            )

        return research
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving research by ID: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{research_id}", response_model=Dict[str, bool])
async def delete_research_by_id(
    research_id: str = Path(..., description="ID of the research to delete")
):
    """
    Delete a specific research by ID.
    """
    try:
        success = await research_agent.delete_research_by_id(research_id)

        if not success:
            raise HTTPException(
                status_code=404, detail=f"Research with ID {research_id} not found"
            )

        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting research by ID: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
