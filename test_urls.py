import asyncio
from app.services.research.agent import ResearchAgent
from loguru import logger


async def test_urls_in_references():
    """Test if URLs are properly included in the references section."""
    logger.info("Starting URL reference test")
    agent = ResearchAgent()

    # Run a test query
    query = "AI Ethics and Accountability"
    result = await agent.conduct_research(query, max_results_per_source=3)

    # Check if report has references section
    if "## References" not in result["report"]:
        logger.error("No References section found in report")
        return

    # Extract the references section
    references_section = result["report"].split("## References")[1]

    # Check for URLs in references
    url_count = references_section.count("http")
    generic_source_count = references_section.count("Source document")

    logger.info(f"References section contains {url_count} URLs")
    logger.info(
        f"References section contains {generic_source_count} generic 'Source document' entries"
    )

    # Print an excerpt of the references section
    logger.info("References section excerpt:")
    print(
        references_section[:500] + "..."
        if len(references_section) > 500
        else references_section
    )

    # Check if we have more URLs than generic sources
    if url_count > generic_source_count:
        logger.info("SUCCESS: More URLs than generic sources in references")
    else:
        logger.warning("ISSUE: Still more generic sources than URLs in references")


if __name__ == "__main__":
    logger.info("Running URL reference test script")
    asyncio.run(test_urls_in_references())
