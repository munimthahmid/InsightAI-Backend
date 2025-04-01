"""
Default templates for research.
"""

from typing import List
from app.services.templates.models import ResearchTemplate


def create_default_templates() -> List[ResearchTemplate]:
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
                "section": "Risk Analysis",
                "description": "Potential challenges and mitigation strategies",
            },
            {
                "section": "Future Outlook",
                "description": "Predictions for future development and adoption",
            },
            {
                "section": "References",
                "description": "Technical sources and documentation",
            },
        ],
        prompt_template="""
        You are a technology analyst creating a comprehensive assessment report.
        
        Technology: "{query}"
        
        Create a detailed technology assessment that:
        1. Explains the core concepts and components of the technology
        2. Analyzes its current state of development and adoption
        3. Explores real-world applications and use cases
        4. Evaluates technical capabilities, limitations, and architecture
        5. Discusses implementation considerations and requirements
        6. Identifies potential risks and challenges with mitigation strategies
        7. Provides insight into future development trajectories
        
        Focus on technical accuracy while making the content accessible to technical decision-makers.
        """,
    )

    # General Research Template
    general_template = ResearchTemplate(
        template_id="general-research",
        name="General Research Report",
        description="Comprehensive research report on any topic with factual information and balanced analysis",
        domain="General",
        focus_areas=[
            "Factual Information",
            "Balanced Analysis",
            "Multiple Perspectives",
        ],
        report_structure=[
            {
                "section": "Introduction",
                "description": "Overview of the topic and key questions to be addressed",
            },
            {
                "section": "Background",
                "description": "Essential context and historical information",
            },
            {
                "section": "Main Findings",
                "description": "Primary research results organized by subtopic",
            },
            {
                "section": "Analysis",
                "description": "Interpretation of findings with multiple perspectives",
            },
            {
                "section": "Conclusion",
                "description": "Summary of key points and their significance",
            },
            {
                "section": "Sources",
                "description": "References and citations",
            },
        ],
        prompt_template="""
        You are a researcher creating a comprehensive research report.
        
        Research Topic: "{query}"
        
        Create a detailed, factual research report that:
        1. Introduces the topic clearly and outlines key questions
        2. Provides necessary background and context
        3. Presents main findings in a well-organized, logical structure
        4. Analyzes the information from multiple perspectives
        5. Summarizes key points and their significance
        6. Includes citations for all sources
        
        Maintain factual accuracy and a balanced, neutral tone throughout the report.
        """,
    )

    return [academic_template, market_template, tech_template, general_template]
