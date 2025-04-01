"""
Report generation for research results.
"""

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from typing import Dict, Any, List, Optional, Tuple
import re
import math
from loguru import logger
from urllib.parse import urlparse

from app.core.config import settings
from app.services.templates.models import ResearchTemplate
from app.services.research._get_source_title import get_source_title


class ReportGenerator:
    """Generates research reports from retrieved documents."""

    def __init__(self):
        """Initialize the report generator with language model."""
        # Initialize the language model
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name="gpt-4o",  # Use GPT-4o for higher quality reports
            temperature=0.2,  # Lower temperature for more focused, fact-based responses
        )

    async def generate_standard_report(
        self, query: str, relevant_docs: Dict[str, Any]
    ) -> str:
        """
        Generate a standard research report from relevant documents.

        Args:
            query: The research query
            relevant_docs: Relevant documents from vector search

        Returns:
            Generated research report
        """
        # Prepare the document chunks for input to the LLM
        chunks = relevant_docs.get("matches", [])

        # Handle case where no matches were found
        if not chunks:
            logger.warning(f"No document matches found for query: {query}")
            return f"""# Research Report: {query}

## Summary
No relevant documents were found for this query. This could be due to:
- The topic may be too specific or niche
- There might be limited data available in the current sources
- The search terms might need refinement

## Recommendations
- Try broadening your search terms
- Consider using different keywords related to your topic
- Explore alternative data sources

*Note: This is an automatically generated placeholder report due to insufficient data.*
"""

        # Create context from chunks
        context_parts = []
        source_types_seen = set()

        # First pass to collect source types for better formatting
        for chunk in chunks:
            metadata = chunk["metadata"]
            source_type = metadata.get("source_type", "unknown")
            source_types_seen.add(source_type)

        # Log the variety of sources available
        logger.info(
            f"Found {len(source_types_seen)} different source types: {', '.join(source_types_seen)}"
        )

        for i, chunk in enumerate(chunks):
            # Extract content and source info
            content = chunk["metadata"].get("page_content", "")
            if (
                not content
            ):  # Fallback to text or full metadata if content not available
                content = chunk["metadata"].get("text", str(chunk["metadata"]))

            # Get source info
            metadata = chunk["metadata"]
            source_type = metadata.get("source_type", "unknown")
            source_title = get_source_title(chunk)
            source_url = metadata.get("url", "")

            # If no URL was found in metadata, try to extract it from the content
            if not source_url and "URL:" in content:
                url_match = re.search(r"URL: (https?://\S+)", content)
                if url_match:
                    source_url = url_match.group(1)

            # Make the document source clearly visible
            document_header = f"DOCUMENT {i+1} [{source_type.upper()}]"
            if source_title:
                document_header += f": {source_title}"

            # Add URL information prominently
            if source_url:
                document_header += f"\nURL: {source_url}"

            # Format the content more distinctively
            formatted_content = (
                f"{document_header}\n"
                f"SOURCE TYPE: {source_type}\n"
                f"CONTENT:\n{content}\n"
            )

            # Make each source type visually distinct
            separator = "=" * 40
            formatted_content = f"{separator}\n{formatted_content}\n{separator}"

            context_parts.append(formatted_content)

        # Join all context parts with clear separation
        full_context = "\n\n".join(context_parts)

        # Add a summary of available sources to emphasize diversity
        source_summary = "AVAILABLE SOURCES SUMMARY:\n"
        for source_type in source_types_seen:
            count = sum(
                1 for c in chunks if c["metadata"].get("source_type") == source_type
            )
            source_summary += f"- {source_type.upper()}: {count} documents\n"

        full_context = f"{source_summary}\n\n{full_context}"

        # Log some context statistics
        logger.info(
            f"Prepared context with {len(chunks)} documents for {len(source_types_seen)} different source types"
        )

        # Create prompt template
        prompt_template = """
        You are an advanced research assistant that creates comprehensive research reports. 
        Use the provided documents to create a detailed, well-structured report on the topic.
        
        TOPIC: {query}
        
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
        - IMPORTANT: Use citations from a WIDE VARIETY of the provided documents, not just one or two sources
        - DISTRIBUTE citations evenly throughout your report
        - AVOID citing the same source repeatedly for different points - use diverse sources
        - USE AT LEAST ONE SOURCE from each available source type (web, wikipedia, arxiv, github, etc.)
        
        YOUR REPORT:
        """

        # Create the LLM chain
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Run the chain
        try:
            response = await chain.arun(query=query, context=full_context)
            logger.info(f"Generated standard report for query: {query}")
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating standard report: {str(e)}")
            return f"Error generating report: {str(e)}"

    async def generate_template_report(
        self, query: str, relevant_docs: Dict[str, Any], template: ResearchTemplate
    ) -> str:
        """
        Generate a report using a specific template.

        Args:
            query: The research query
            relevant_docs: Relevant documents from vector search
            template: The template to use for the report

        Returns:
            Generated research report
        """
        # Prepare the document chunks for input to the LLM
        chunks = relevant_docs["matches"]

        # Create context from chunks
        context_parts = []
        source_types_seen = set()

        # First pass to collect source types for better formatting
        for chunk in chunks:
            metadata = chunk["metadata"]
            source_type = metadata.get("source_type", "unknown")
            source_types_seen.add(source_type)

        # Log the variety of sources available
        logger.info(
            f"Found {len(source_types_seen)} different source types: {', '.join(source_types_seen)}"
        )

        for i, chunk in enumerate(chunks):
            # Extract content and source info
            content = chunk["metadata"].get("page_content", "")
            if (
                not content
            ):  # Fallback to text or full metadata if content not available
                content = chunk["metadata"].get("text", str(chunk["metadata"]))

            # Get source info
            metadata = chunk["metadata"]
            source_type = metadata.get("source_type", "unknown")
            source_title = get_source_title(chunk)
            source_url = metadata.get("url", "")

            # If no URL was found in metadata, try to extract it from the content
            if not source_url and "URL:" in content:
                url_match = re.search(r"URL: (https?://\S+)", content)
                if url_match:
                    source_url = url_match.group(1)

            # Make the document source clearly visible
            document_header = f"DOCUMENT {i+1} [{source_type.upper()}]"
            if source_title:
                document_header += f": {source_title}"

            # Add URL information prominently
            if source_url:
                document_header += f"\nURL: {source_url}"

            # Format the content more distinctively
            formatted_content = (
                f"{document_header}\n"
                f"SOURCE TYPE: {source_type}\n"
                f"CONTENT:\n{content}\n"
            )

            # Make each source type visually distinct
            separator = "=" * 40
            formatted_content = f"{separator}\n{formatted_content}\n{separator}"

            context_parts.append(formatted_content)

        # Join all context parts with clear separation
        full_context = "\n\n".join(context_parts)

        # Add a summary of available sources to emphasize diversity
        source_summary = "AVAILABLE SOURCES SUMMARY:\n"
        for source_type in source_types_seen:
            count = sum(
                1 for c in chunks if c["metadata"].get("source_type") == source_type
            )
            source_summary += f"- {source_type.upper()}: {count} documents\n"

        full_context = f"{source_summary}\n\n{full_context}"

        # Log some context statistics
        logger.info(
            f"Prepared context with {len(chunks)} documents for {len(source_types_seen)} different source types"
        )

        # Extract report structure for instructions
        structure_sections = "\n".join(
            [f"- {s['section']}: {s['description']}" for s in template.report_structure]
        )

        # Create the full system prompt
        prompt_template = f"""
        {template.prompt_template}
        
        REPORT STRUCTURE:
        {structure_sections}
        
        DOCUMENTS:
        {{context}}
        
        FORMAT REQUIREMENTS:
        - Use Markdown formatting for structure (headers, lists, etc.)
        - Include citations to specific documents in the format [Document X]
        - CRITICAL: Each source in the References section MUST include its URL if available
        - Do NOT use generic "Source document" entries in References without a URL  
        - Follow the report structure outlined above
        - End with a detailed References section listing all cited sources with their full URLs
        - IMPORTANT: Use citations from a WIDE VARIETY of the provided documents, not just one or two sources
        - DISTRIBUTE citations evenly throughout your report
        - AVOID citing the same source repeatedly for different points - use diverse sources
        - USE AT LEAST ONE SOURCE from each available source type (web, wikipedia, arxiv, github, etc.)
        
        YOUR REPORT:
        """

        # Create the LLM chain
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Run the chain
        try:
            # Use the query in the template
            formatted_template = prompt_template.replace("{query}", query)
            response = await chain.arun(context=full_context)
            logger.info(f"Generated template report for query: {query}")
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating template report: {str(e)}")
            return f"Error generating report: {str(e)}"

    async def enhance_report_with_citations(
        self,
        report: str,
        evidence_chunks: List[Dict[str, Any]],
        sources_dict: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        """
        Enhance a report with detailed citations.

        Args:
            report: The research report
            evidence_chunks: List of evidence chunks
            sources_dict: Dictionary mapping source types to sources

        Returns:
            Report with enhanced citations
        """
        try:
            # Check if we have evidence chunks to work with
            if not evidence_chunks:
                logger.warning("No evidence chunks available for citation enhancement")
                # Add a placeholder references section if there are none
                if "## References" not in report:
                    report += "\n\n## References\n\n*No specific sources were found for this query.*"
                return report

            # Log the raw data for debugging
            logger.info(f"Sources dict keys: {list(sources_dict.keys())}")
            for source_type, sources in sources_dict.items():
                if sources:
                    sample = sources[0]
                    logger.info(
                        f"Sample source of type {source_type}: Keys: {list(sample.keys())}"
                    )
                    if "url" in sample:
                        logger.info(f"  Has URL: {sample['url']}")
                    if "metadata" in sample:
                        logger.info(
                            f"  Metadata keys: {list(sample['metadata'].keys())}"
                        )
                        if "url" in sample["metadata"]:
                            logger.info(f"  Metadata URL: {sample['metadata']['url']}")

            # Log evidence chunks more thoroughly
            logger.info(f"Evidence chunks count: {len(evidence_chunks)}")
            for i, chunk in enumerate(
                evidence_chunks[:5]
            ):  # First 5 chunks for more detail
                metadata = chunk.get("metadata", {})
                logger.info(
                    f"Evidence chunk {i}: source_type={metadata.get('source_type')}, "
                    f"has_url={'url' in metadata}, "
                    f"title={metadata.get('title', 'No title')}"
                )
                if "url" in metadata:
                    logger.info(f"  URL: {metadata['url']}")

                # Log more nested metadata
                if "metadata" in chunk:
                    logger.info(f"  Nested metadata: {chunk['metadata']}")

                # Log the content to see if URLs might be hidden in the content
                content = metadata.get("page_content", "")
                if content:
                    content_preview = (
                        content[:200] + "..." if len(content) > 200 else content
                    )
                    logger.info(f"  Content preview: {content_preview}")

            # Find all document citations in the form [Document X]
            citation_pattern = r"\[Document (\d+)\]"
            citations = re.findall(citation_pattern, report)

            logger.info(f"Found {len(citations)} citations in report")

            # If no citations found, return the original report with a basic references section
            if not citations:
                if "## References" not in report:
                    report += "\n\n## References\n\n*No specific citations were identified in this report.*"
                return report

            # Create a mapping of document numbers to actual sources
            citation_mapping = {}

            # Create a sources lookup for faster access - combine all sources by title
            url_lookup = {}

            # First, build a comprehensive URL lookup from all sources
            for source_type, sources in sources_dict.items():
                for source in sources:
                    # Try multiple places for URLs
                    url = None

                    # Direct URL in source
                    if "url" in source:
                        url = source.get("url", "")

                    # URL in metadata
                    if not url and "metadata" in source and "url" in source["metadata"]:
                        url = source["metadata"]["url"]

                    # URL in html_url (GitHub)
                    if not url and "html_url" in source:
                        url = source["html_url"]

                    # URL in link field
                    if not url and "link" in source:
                        url = source["link"]

                    title = (
                        source.get("title", "").lower() if source.get("title") else ""
                    )

                    if url and title:
                        url_lookup[title] = url

                        # Also store by source_type and title combination for more specific matching
                        url_lookup[f"{source_type}:{title}"] = url

                    # Also index by URL domain if available
                    if url:
                        try:
                            domain = urlparse(url).netloc
                            if domain:
                                url_lookup[domain] = url
                        except Exception:
                            pass

            logger.info(f"Created URL lookup with {len(url_lookup)} entries")

            # Process each citation to get its source information
            for citation in citations:
                doc_num = int(citation)
                if doc_num <= len(evidence_chunks):
                    chunk = evidence_chunks[doc_num - 1]
                    metadata = chunk.get("metadata", {})
                    source_type = metadata.get("source_type", "unknown")

                    # Get title from either the chunk or our source title utility
                    title = get_source_title(chunk)

                    # Get URL - try multiple sources
                    url = ""

                    # 1. Direct URL in metadata
                    if "url" in metadata:
                        url = metadata.get("url", "")

                    # 2. URL in page_content for web results
                    if not url and "page_content" in chunk:
                        content = chunk["page_content"]
                        url_match = re.search(r"URL: (https?://\S+)", content)
                        if url_match:
                            url = url_match.group(1)
                            logger.info(f"Found URL in page_content: {url}")

                    # 3. Look for matching title in url_lookup
                    if not url and title.lower() in url_lookup:
                        url = url_lookup[title.lower()]
                        logger.info(f"Found URL from title match: {url}")

                    # 4. Try source_type:title combination
                    if not url and f"{source_type}:{title.lower()}" in url_lookup:
                        url = url_lookup[f"{source_type}:{title.lower()}"]
                        logger.info(f"Found URL from source_type:title match: {url}")

                    # 5. Look in raw_data by title or content match
                    if not url and source_type in sources_dict:
                        logger.info(
                            f"Looking for URL in raw_data for source_type {source_type}"
                        )
                        chunk_title = metadata.get("title", "").lower()
                        chunk_content = metadata.get("text", "").lower()

                        if not chunk_content and "page_content" in metadata:
                            chunk_content = metadata["page_content"].lower()

                        # Try to match by title
                        for source in sources_dict.get(source_type, []):
                            source_title = source.get("title", "").lower()

                            # Try multiple places for URLs
                            source_url = source.get("url", "")
                            if not source_url and "metadata" in source:
                                source_url = source["metadata"].get("url", "")
                            if not source_url and "html_url" in source:
                                source_url = source.get("html_url", "")
                            if not source_url and "link" in source:
                                source_url = source.get("link", "")

                            # Check if titles match or content contains title
                            if source_url and (
                                source_title == chunk_title
                                or (
                                    chunk_content
                                    and source_title in chunk_content[:200]
                                )
                            ):
                                url = source_url
                                logger.info(f"Found URL from content match: {url}")
                                break

                    # Log if URL was found or not
                    if url:
                        logger.info(f"Citation {citation} has URL: {url}")
                    else:
                        logger.warning(
                            f"No URL found for citation {citation}, type {source_type}, title {title}"
                        )

                    # Create a unique key for this citation
                    citation_key = f"[{source_type}-{doc_num}]"

                    # Store the citation mapping
                    citation_mapping[f"[Document {citation}]"] = {
                        "key": citation_key,
                        "source_type": source_type,
                        "title": title,
                        "url": url,
                    }

            # Replace [Document X] citations with more specific ones like [arxiv-1]
            enhanced_report = report
            for doc_citation, citation_info in citation_mapping.items():
                enhanced_report = enhanced_report.replace(
                    doc_citation, citation_info["key"]
                )

            # Remove any existing References section
            references_section_pattern = r"\n## References\n[\s\S]*?($|\n# |\n## [^R])"
            enhanced_report = re.sub(
                references_section_pattern, r"\n\1", enhanced_report
            )

            # Add a references section
            if citation_mapping:
                # Sort references by source type and citation number
                references = sorted(
                    citation_mapping.values(),
                    key=lambda x: (x["source_type"], x["key"]),
                )

                # Create the references section
                references_section = "\n\n## References\n\n"

                # Group references by source type for better organization
                source_types = sorted(set(ref["source_type"] for ref in references))

                # Create references by source type
                for source_type in source_types:
                    # Add a subheading for this source type
                    references_section += f"### {source_type.capitalize()} Sources\n\n"

                    # Filter references for this source type
                    source_refs = [
                        ref for ref in references if ref["source_type"] == source_type
                    ]

                    for ref in source_refs:
                        ref_text = f"{ref['key']} "

                        if ref.get("title"):
                            ref_text += f"**{ref['title']}**"
                        else:
                            ref_text += "Source document"  # Avoid "Untitled Source"

                        # Always include URL if available - make sure it's properly formatted as a markdown link
                        if ref.get("url"):
                            url = ref["url"]
                            # Make sure URL has a scheme
                            if not url.startswith(("http://", "https://")):
                                url = "https://" + url
                            ref_text += f". [{url}]({url})"
                        else:
                            # Log citations without URLs
                            logger.warning(
                                f"Reference {ref['key']} has no URL in the final output"
                            )

                        references_section += ref_text + "\n\n"

                # Add references to the report
                enhanced_report += references_section

            # Log the input and output to LLM for debugging
            logger.info("Original report excerpt (first 500 chars):")
            logger.info(report[:500] + "...")
            logger.info("Enhanced report excerpt (first 500 chars):")
            logger.info(enhanced_report[:500] + "...")

            return enhanced_report
        except Exception as e:
            logger.error(f"Error enhancing report with citations: {str(e)}")
            return report

    async def generate_literature_review(
        self,
        research_data: Dict[str, Any],
        format_type: str = "APA",
        section_format: str = "thematic",
        max_length: Optional[int] = None,
    ) -> str:
        """
        Generate a formal literature review from research results.

        Args:
            research_data: The complete research result
            format_type: Citation format (APA, MLA, Chicago, IEEE)
            section_format: Organization method (chronological, thematic, methodological)
            max_length: Maximum length of the review

        Returns:
            Formatted literature review
        """
        if not research_data:
            return "Research data not found"

        query = research_data.get("query", "Unknown Topic")
        report = research_data.get("report", "")
        relevant_docs = research_data.get("relevant_docs", {})
        chunks = relevant_docs.get("matches", [])

        # Prepare the document chunks for input to the LLM
        context_parts = []
        source_count = 0

        # Focus primarily on academic sources
        academic_chunks = [
            chunk
            for chunk in chunks
            if chunk["metadata"].get("source_type") in ["arxiv", "semantic_scholar"]
        ]

        # Use academic sources first, then others if needed
        for i, chunk in enumerate(academic_chunks or chunks):
            if source_count >= 15:  # Limit to prevent exceeding context window
                break

            # Extract content and source info
            content = chunk["metadata"].get("page_content", "")
            if not content:  # Fallback to full metadata if content not available
                content = str(chunk["metadata"])

            # Get source info
            source_type = chunk["metadata"].get("source_type", "unknown")
            title = get_source_title(chunk)
            authors = chunk["metadata"].get("authors", "Unknown Author")
            year = chunk["metadata"].get("published", "Unknown Date")

            if isinstance(year, str) and len(year) >= 4:
                year = year[:4]  # Extract just the year

            # Create citation info
            citation_info = f"SOURCE {i+1}:\nTitle: {title}\nAuthors: {authors}\nYear: {year}\nType: {source_type}\n"

            # Format this chunk with source information
            context_parts.append(f"{citation_info}\nContent: {content}\n")
            source_count += 1

        # Join all context parts
        full_context = "\n\n".join(context_parts)

        # Create literature review prompt
        prompt_template = f"""
        You are an academic researcher creating a formal literature review.
        
        TOPIC: {{query}}
        
        FORMAT TYPE: {format_type} citation style
        SECTION FORMAT: {section_format} organization
        
        SOURCES:
        {{context}}
        
        Create a formal literature review that:
        1. Begins with an introduction that defines the research question and its significance
        2. Describes your methodology for selecting and analyzing the literature
        3. Organizes the review in a {section_format} structure
        4. Critically analyzes the literature, identifying patterns, gaps, and contradictions
        5. Properly cites all sources using {format_type} format
        6. Ends with a conclusion summarizing key findings and suggesting directions for future research
        7. Includes a complete list of references in {format_type} format
        
        The literature review should be scholarly in tone, well-structured, and comprehensive.
        """

        # Create the LLM chain
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Run the chain
        try:
            response = await chain.arun(query=query, context=full_context)

            # Limit length if requested
            if max_length and len(response) > max_length:
                response = (
                    response[:max_length]
                    + "...\n\n[Note: Review truncated due to length constraints]"
                )

            logger.info(f"Generated literature review for topic: {query}")
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating literature review: {str(e)}")
            return f"Error generating literature review: {str(e)}"

    async def generate_cluster_report(
        self,
        query: str,
        relevant_docs: Dict[str, Any],
        template_id: Optional[str] = None,
    ) -> str:
        """
        Generate a report using clustered document data.

        This method takes advantage of the document clustering to produce
        a more organized, comprehensive report that addresses different
        aspects or perspectives of the research topic.

        Args:
            query: The research query
            relevant_docs: Relevant documents from vector search with clustering information
            template_id: Optional template ID to use for structure (not yet implemented)

        Returns:
            Generated research report with cluster-aware organization
        """
        # Verify that we have clustered data
        if "cluster_stats" not in relevant_docs:
            logger.warning(
                "Cluster stats not found in results, falling back to standard report"
            )
            return await self.generate_standard_report(query, relevant_docs)

        # Get matches and cluster information
        matches = relevant_docs.get("matches", [])
        num_clusters = relevant_docs.get("num_clusters", 0)
        cluster_stats = relevant_docs.get("cluster_stats", {})

        if not matches or num_clusters == 0:
            logger.warning(
                "No matches or clusters found, falling back to standard report"
            )
            return await self.generate_standard_report(query, relevant_docs)

        logger.info(f"Generating cluster-based report with {num_clusters} clusters")

        # Organize documents by cluster
        clustered_docs = {}

        # First pass to identify cluster labels
        for doc in matches:
            cluster = doc.get("cluster", -1)
            if cluster == -1:
                # If document doesn't have a cluster label, try to determine from metadata
                metadata = doc.get("metadata", {})
                cluster = metadata.get("cluster", -1)

            if cluster not in clustered_docs:
                clustered_docs[cluster] = []
            clustered_docs[cluster].append(doc)

        # If clustering wasn't preserved in the documents, try to cluster them now
        if len(clustered_docs) <= 1:
            logger.warning(
                "Documents don't have cluster labels, treating as a single cluster"
            )
            clustered_docs = {0: matches}

        # Now create a prompt that leverages the cluster information
        prompt_parts = [
            f"# Research for query: {query}",
            f"\nThe documents have been organized into {len(clustered_docs)} different thematic clusters.",
            "\n## Clusters Overview:\n",
        ]

        # Add cluster statistics
        for cluster_id, docs in clustered_docs.items():
            prompt_parts.append(f"- Cluster {cluster_id}: {len(docs)} documents")

        # For each cluster, prepare document context
        for cluster_id, docs in clustered_docs.items():
            prompt_parts.append(f"\n\n## DOCUMENTS FROM CLUSTER {cluster_id}\n")

            # Add context from this cluster's documents
            for i, doc in enumerate(docs):
                metadata = doc.get("metadata", {})
                content = metadata.get("text", "")
                score = doc.get("score", 0)

                # Get source information
                source_type = metadata.get("source_type", "unknown")
                source_title = get_source_title(doc)
                source_url = metadata.get("url", "")

                # Format document header
                doc_header = (
                    f"DOCUMENT {i+1} (CLUSTER {cluster_id}) [{source_type.upper()}]"
                )
                if source_title:
                    doc_header += f": {source_title}"

                if source_url:
                    doc_header += f"\nURL: {source_url}"

                # Format content
                formatted_content = (
                    f"{doc_header}\n"
                    f"RELEVANCE: {score:.4f}\n"
                    f"SOURCE TYPE: {source_type}\n"
                    f"CONTENT:\n{content}\n"
                )

                # Add separator
                separator = "=" * 40
                formatted_content = f"{separator}\n{formatted_content}\n{separator}"

                prompt_parts.append(formatted_content)

        # Create the full context
        full_context = "\n\n".join(prompt_parts)

        # Create cluster-aware prompt template
        cluster_prompt_template = """
        You are an advanced research assistant that creates comprehensive cluster-based research reports.
        The documents provided have been automatically clustered by semantic similarity into distinct thematic groups.
        
        RESEARCH QUERY: {query}
        
        CLUSTERED DOCUMENTS:
        {context}
        
        Create a comprehensive research report that takes advantage of this thematic clustering. Your report should:
        
        1. Include an introduction explaining the query and the major themes discovered
        2. Organize the report into SEPARATE SECTIONS FOR EACH CLUSTER, with each section covering the theme/perspective of that cluster
        3. For each cluster section:
           - Identify the main theme or perspective represented by this cluster
           - Summarize the key information and insights from the documents in this cluster
           - Include specific data, examples, and evidence with proper citations
           - Note any relationships to other clusters/themes
        4. After covering all clusters, include a synthesis section that connects the different themes
        5. End with a conclusion that answers the research query comprehensively
        
        FORMAT REQUIREMENTS:
        - Use Markdown formatting with clear hierarchical structure
        - Include citations to specific documents in the format [Cluster X, Document Y]
        - Each section should represent a coherent theme from one cluster
        - Include a "Synthesis and Integration" section that connects insights across clusters
        - End with a comprehensive References section organized by cluster
        - IMPORTANT: Balance coverage across all clusters, don't focus only on one cluster
        
        YOUR REPORT:
        """

        # Create the LLM chain
        prompt = ChatPromptTemplate.from_template(cluster_prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Run the chain
        try:
            response = await chain.arun(query=query, context=full_context)
            logger.info(
                f"Generated cluster-based report for query: {query} with {num_clusters} clusters"
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating cluster-based report: {str(e)}")
            # Fall back to standard report
            logger.info("Falling back to standard report after cluster report failure")
            return await self.generate_standard_report(query, relevant_docs)
