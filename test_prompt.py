from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from app.core.config import settings
from loguru import logger


async def test_report_prompt():
    """Test if the LLM properly includes URLs in the references section."""
    logger.info("Testing report prompt for URL inclusion")

    # Initialize the language model
    llm = ChatOpenAI(
        openai_api_key=settings.OPENAI_API_KEY,
        model_name="gpt-3.5-turbo-16k",
        temperature=0.2,
    )

    # Sample document context with URLs
    context = """
    DOCUMENT 1 (Source: web - AI Ethics Guide (URL: https://example.com/ai-ethics)):
    Title: AI Ethics Guide
    URL: https://example.com/ai-ethics
    
    AI ethics is a set of principles and guidelines for the responsible development and use of artificial intelligence systems.
    
    DOCUMENT 2 (Source: wikipedia - Ethics of artificial intelligence (URL: https://en.wikipedia.org/wiki/Ethics_of_artificial_intelligence)):
    Title: Ethics of artificial intelligence
    URL: https://en.wikipedia.org/wiki/Ethics_of_artificial_intelligence
    
    The ethics of artificial intelligence is the branch of the ethics of technology specific to artificially intelligent systems.
    """

    # Create prompt template
    prompt_template = """
    You are an advanced research assistant that creates comprehensive research reports. 
    Use the provided documents to create a detailed, well-structured report on the topic.
    
    TOPIC: AI Ethics
    
    DOCUMENTS:
    {context}
    
    Create a comprehensive research report on the topic above. Your report should:
    
    1. Include a clear introduction explaining the topic's importance
    2. Organize findings into logical sections with headings
    3. Present a balanced view considering multiple perspectives
    4. Include specific data, examples, and evidence from the documents
    5. Identify patterns, trends, and key insights
    6. Include a conclusion summarizing the main findings
    7. Use formal, academic language and proper citations
    
    FORMAT REQUIREMENTS:
    - Use Markdown formatting for structure (headers, lists, etc.)
    - Include citations to specific documents in the format [Document X]
    - CRITICAL: Each source in the References section MUST include its URL if available
    - Do NOT use generic "Source document" entries in References without a URL
    - Organize content into logical sections with appropriate headings
    - End with a detailed References section listing all cited sources with their full URLs
    
    YOUR REPORT:
    """

    # Create the LLM chain
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain
    try:
        response = await chain.arun(context=context)

        # Check if references section contains URLs
        if "## References" not in response:
            logger.error("No References section found in report")
            return

        # Extract references section
        references_section = response.split("## References")[1]

        # Check for URLs in references
        url_count = references_section.count("http")
        generic_source_count = references_section.count("Source document")

        logger.info(f"References section contains {url_count} URLs")
        logger.info(
            f"References section contains {generic_source_count} generic 'Source document' entries"
        )

        # Print the references section
        logger.info("References section:")
        print(references_section)

        # Success criteria
        if url_count >= 2:  # We provided 2 documents with URLs
            logger.info("SUCCESS: URLs are included in references")
        else:
            logger.warning("ISSUE: URLs are missing from references")

    except Exception as e:
        logger.error(f"Error testing prompt: {str(e)}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_report_prompt())
