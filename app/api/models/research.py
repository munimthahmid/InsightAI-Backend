from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union, Literal


# Request models
class ResearchRequest(BaseModel):
    query: str
    max_results_per_source: Optional[int] = None
    save_history: bool = True


class ResearchTemplateRequest(BaseModel):
    query: str
    template_id: str
    max_results_per_source: Optional[int] = None
    save_history: bool = True


class LiteratureReviewRequest(BaseModel):
    research_id: str
    format_type: Literal["APA", "MLA", "Chicago", "IEEE"] = "APA"
    section_format: Literal["chronological", "thematic", "methodological"] = "thematic"
    max_length: Optional[int] = None


class ComparisonRequest(BaseModel):
    research_ids: List[str]
    comparison_aspects: Optional[List[str]] = None
    include_visualization: bool = True


# Response models
class Citation(BaseModel):
    chunk_id: str
    source_type: str
    title: str
    url: Optional[str] = None
    text: str
    additional_info: Dict[str, Any] = {}


class SourceInfo(BaseModel):
    type: str
    count: int
    items: List[Dict[str, Any]] = []


class ResearchResponse(BaseModel):
    query: str
    report: str
    sources: Dict[str, int]
    research_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    citations: Optional[Dict[str, Citation]] = None
    contradictions: Optional[Dict[str, List[Dict[str, Any]]]] = None


class ResearchHistoryItem(BaseModel):
    research_id: str
    query: str
    saved_at: str
    metadata: Optional[Dict[str, Any]] = None
    sources: Dict[str, int]


class ResearchHistoryResponse(BaseModel):
    items: List[ResearchHistoryItem]
    total: int


class LiteratureReviewResponse(BaseModel):
    research_id: str
    literature_review: str
    format: Dict[str, str]
    generated_at: str


class TemplateResponse(BaseModel):
    id: str
    name: str
    description: str
    domain: str
    structure: List[str]
    default_sources: List[str]


class TemplatesResponse(BaseModel):
    templates: List[TemplateResponse]


class ComparisonResponse(BaseModel):
    topics: List[str]
    research_ids: List[str]
    comparison_aspects: List[str]
    comparison_result: str
    generated_at: str
