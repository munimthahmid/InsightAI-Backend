"""
Synthesis agent for generating comprehensive research reports.
"""

from typing import Dict, List, Any, Optional
import json
import time
from loguru import logger
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

from app.core.config import settings
from app.services.research.agents.base_agent import BaseAgent
from app.services.research.orchestration.schemas import TaskSchema
from app.services.templates.manager import TemplateManager
from app.services.research._get_source_title import get_source_title


class SynthesisAgent(BaseAgent):
    """
    Specializes in synthesizing research data into coherent reports.
    Generates well-structured research reports based on analysis.
    """

    def __init__(
        self,
        context_manager=None,
        task_queue=None,
        agent_id=None,
        template_manager=None,
    ):
        """
        Initialize the synthesis agent.

        Args:
            context_manager: Shared research context manager
            task_queue: Task queue for asynchronous operations
            agent_id: Optional unique identifier
            template_manager: Optional template manager for report templates
        """
        super().__init__(
            agent_id=agent_id,
            agent_type="synthesis",
            context_manager=context_manager,
            task_queue=task_queue,
        )

        # Register task handler
        if task_queue:
            task_queue.register_handler("synthesis_task", self.execute_task)

        # Initialize template manager
        self.template_manager = template_manager or TemplateManager()

        # Initialize language model for report generation
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name="gpt-4o",  # Use GPT-4o for higher quality reports
            temperature=0.3,  # Slightly higher temperature for creativity in synthesis
        )

        logger.info(f"SynthesisAgent initialized with ID: {self.agent_id}")

    async def execute_task(self, task: TaskSchema) -> Dict[str, Any]:
        """
        Execute a synthesis task to generate a research report.

        Args:
            task: The task containing query and template parameters

        Returns:
            Dictionary with the generated report and metadata
        """
        params = task.params
        query = params.get("query", "")
        template_id = params.get("template_id")

        await self.log_activity(
            "start_synthesis",
            {"task_id": task.task_id, "query": query, "template_id": template_id},
        )

        try:
            # Get research context
            research_id = await self.get_context("research_id")
            sources = await self.get_context("sources") or []
            raw_data = await self.get_context("raw_data") or {}
            analysis_results = await self.get_context("analysis_results") or {}
            analysis_summary = await self.get_context("analysis_summary") or ""
            vector_namespace = await self.get_context("vector_namespace")
            cluster_results = await self.get_context("cluster_results") or {}

            # Convert sources to dictionary if needed
            sources_dict = {}
            if isinstance(sources, list):
                for source in sources:
                    sources_dict[source] = 1  # Default count
            elif isinstance(sources, dict):
                sources_dict = sources
            else:
                # Fallback if sources is neither list nor dict
                sources_dict = {"placeholder": 1}

            # Extract key information from analysis results
            entities = analysis_results.get("entities", [])
            topics = analysis_results.get("topics", [])
            claims = analysis_results.get("claims", [])
            contradictions = analysis_results.get("contradictions", [])
            concept_map = analysis_results.get("concept_map", {})

            # Update status for progress tracking
            await self.context_manager.update_status("generating_report")

            # Generate the appropriate report based on available information
            report = ""

            # Check if a template should be used
            if template_id:
                template = self.template_manager.get_template(template_id)
                if template:
                    report = await self._generate_template_report(
                        query=query,
                        template=template,
                        analysis_results=analysis_results,
                        analysis_summary=analysis_summary,
                        raw_data=raw_data,
                        cluster_results=cluster_results,
                    )
                else:
                    logger.warning(
                        f"Template {template_id} not found, using standard report"
                    )
                    report = await self._generate_standard_report(
                        query=query,
                        analysis_results=analysis_results,
                        analysis_summary=analysis_summary,
                        raw_data=raw_data,
                        cluster_results=cluster_results,
                    )
            else:
                # Use standard report format
                report = await self._generate_standard_report(
                    query=query,
                    analysis_results=analysis_results,
                    analysis_summary=analysis_summary,
                    raw_data=raw_data,
                    cluster_results=cluster_results,
                )

            # Enhance with citations
            enhanced_report = await self._enhance_with_citations(
                report=report,
                raw_data=raw_data,
                sources_dict=sources_dict,
            )

            # Store report in context
            await self.set_context("report", enhanced_report)

            # Log completion
            await self.log_activity(
                "synthesis_complete",
                {
                    "task_id": task.task_id,
                    "query": query,
                    "template_id": template_id,
                    "report_length": len(enhanced_report),
                },
            )

            # Return results
            return {
                "success": True,
                "report": enhanced_report,
                "research_id": research_id,
                "query": query,
                "timestamp": time.time(),
                "template_id": template_id,
                "sources": sources_dict,
                "sources_used": sources,
                "sources_dict": sources_dict,
                "result_count": len(raw_data),
                "vector_namespace": vector_namespace,
            }

        except Exception as e:
            # Log failure
            logger.error(f"Error in synthesis task: {str(e)}")
            await self.log_activity(
                "synthesis_failed",
                {"task_id": task.task_id, "error": str(e)},
            )

            # Add error to context
            await self.context_manager.add_error(str(e), "synthesis_agent")

            # Re-raise for task queue error handling
            raise

    async def _generate_standard_report(
        self,
        query: str,
        analysis_results: Dict[str, Any],
        analysis_summary: str,
        raw_data: Dict[str, Any],
        cluster_results: Dict[str, Any],
    ) -> str:
        """
        Generate a standard research report.

        Args:
            query: Research query
            analysis_results: Results from analysis
            analysis_summary: Summary from analysis
            raw_data: Raw data from sources
            cluster_results: Optional clustering results

        Returns:
            Generated report text
        """
        # Extract key information
        entities = analysis_results.get("entities", [])
        topics = analysis_results.get("topics", [])
        claims = analysis_results.get("claims", [])
        contradictions = analysis_results.get("contradictions", [])
        concept_map = analysis_results.get("concept_map", {})

        # Use analysis summary if available, otherwise create from scratch
        if analysis_summary:
            logger.info("Using existing analysis summary as basis for report")
            content_for_synthesis = {
                "analysis_summary": analysis_summary,
                "entities": entities[:10],  # Limit to top 10
                "topics": topics,
                "claims": claims[:10],  # Limit to top 10
                "contradictions": contradictions,
            }

            prompt_template = """
            You are an expert research report writer. Your task is to create a comprehensive, well-structured research report
            based on the analysis summary and key elements provided.
            
            RESEARCH QUERY: {query}
            
            ANALYSIS SUMMARY:
            {analysis_summary}
            
            KEY ENTITIES:
            {entities_text}
            
            KEY TOPICS:
            {topics_text}
            
            MAIN CLAIMS:
            {claims_text}
            
            CONTRADICTIONS:
            {contradictions_text}
            
            Based on this information, create a comprehensive research report. Your report should:
            
            1. Have a clear structure with an introduction, body sections, and conclusion
            2. Organize information logically by topics or themes
            3. Present a balanced view of different perspectives
            4. Include specific data, examples, and evidence
            5. Cite sources appropriately using numbered citations like [1], [2], etc.
            6. Use formal, academic language
            7. Include recommendations or next steps when appropriate
            
            CITATION AND REFERENCE GUIDELINES:
            - When citing a source, use a numbered reference style with square brackets: [1], [2], etc.
            - DO NOT use "Document X" style references - use numbered citations instead
            - Include exact URL links in your references section when available
            - For each reference, include title, source type, and URL (if available)
            
            FORMAT REQUIREMENTS:
            - Use Markdown formatting with hierarchical headers
            - Create a table of contents at the beginning
            - Include clear section headers that reflect the content
            - Use bullet points or numbered lists where appropriate
            - End with a References section that includes numbered references with titles and URLs
            - IMPORTANT: Make the report comprehensive and detailed - at least 1000-1500 words
            
            THE REPORT:
            """

            # Create the LLM chain
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = LLMChain(llm=self.llm, prompt=prompt)

            # Run the chain
            try:
                # Convert content to text for the prompt
                entities_text = json.dumps(content_for_synthesis["entities"], indent=2)
                topics_text = json.dumps(content_for_synthesis["topics"], indent=2)
                claims_text = json.dumps(content_for_synthesis["claims"], indent=2)
                contradictions_text = json.dumps(
                    content_for_synthesis["contradictions"], indent=2
                )

                response = await chain.arun(
                    query=query,
                    analysis_summary=analysis_summary,
                    entities_text=entities_text,
                    topics_text=topics_text,
                    claims_text=claims_text,
                    contradictions_text=contradictions_text,
                )
                logger.info(
                    f"Generated standard report from analysis summary for query: {query}"
                )
                return response.strip()
            except Exception as e:
                logger.error(f"Error generating report from analysis: {str(e)}")
                # Fall back to basic report
                return self._generate_fallback_report(query, raw_data)

        else:
            # Create report from raw data if no analysis is available
            logger.info(
                "No analysis summary available, generating report from raw data"
            )
            return await self._generate_report_from_raw_data(query, raw_data)

    async def _generate_template_report(
        self,
        query: str,
        template: Any,
        analysis_results: Dict[str, Any],
        analysis_summary: str,
        raw_data: Dict[str, Any],
        cluster_results: Dict[str, Any],
    ) -> str:
        """
        Generate a report using a specific template.

        Args:
            query: Research query
            template: Report template to use
            analysis_results: Results from analysis
            analysis_summary: Summary from analysis
            raw_data: Raw data from sources
            cluster_results: Optional clustering results

        Returns:
            Generated report using the template
        """
        # Extract template information
        template_content = template.content
        template_name = template.name
        template_description = template.description

        # Extract key information
        entities = analysis_results.get("entities", [])
        topics = analysis_results.get("topics", [])
        claims = analysis_results.get("claims", [])
        contradictions = analysis_results.get("contradictions", [])

        # Use template with analysis data
        prompt_template = """
        You are an expert research report writer. Your task is to create a research report
        following a specific template format, based on the analysis provided.
        
        RESEARCH QUERY: {query}
        
        TEMPLATE NAME: {template_name}
        
        TEMPLATE DESCRIPTION: {template_description}
        
        TEMPLATE FORMAT:
        ```
        {template_content}
        ```
        
        ANALYSIS SUMMARY:
        {analysis_summary}
        
        KEY ENTITIES:
        {entities_text}
        
        KEY TOPICS:
        {topics_text}
        
        MAIN CLAIMS:
        {claims_text}
        
        Based on this information, create a comprehensive research report following the template format.
        Your report should:
        
        1. Strictly follow the structure and formatting of the template
        2. Replace placeholders in the template with relevant research content
        3. Maintain the sections and organization of the template
        4. Ensure the content is comprehensive, accurate, and based on the analysis
        5. Cite sources appropriately using numbered citations like [1], [2], etc.
        
        CITATION AND REFERENCE GUIDELINES:
        - When citing a source, use a numbered reference style with square brackets: [1], [2], etc.
        - DO NOT use "Document X" style references - use numbered citations instead
        - Include exact URL links in your references section when available
        - For each reference, include title, source type, and URL (if available)
        
        FORMAT REQUIREMENTS:
        - Follow the exact Markdown formatting of the template
        - Replace placeholders like {{Title}} with appropriate content
        - Fill in all sections of the template with substantive information
        - Keep section headers exactly as they appear in the template
        - Include a proper References section at the end with numbered references and links
        - IMPORTANT: Make the report comprehensive and detailed, maintaining proper academic tone
        
        THE REPORT:
        """

        # Create the LLM chain
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Run the chain
        try:
            # Convert content to text for the prompt
            entities_text = json.dumps(entities[:10], indent=2)  # Limit to top 10
            topics_text = json.dumps(topics, indent=2)
            claims_text = json.dumps(claims[:10], indent=2)  # Limit to top 10

            response = await chain.arun(
                query=query,
                template_name=template_name,
                template_description=template_description,
                template_content=template_content,
                analysis_summary=analysis_summary,
                entities_text=entities_text,
                topics_text=topics_text,
                claims_text=claims_text,
            )
            logger.info(
                f"Generated template report for query: {query} using template: {template_name}"
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating template report: {str(e)}")
            # Fall back to standard report
            return await self._generate_standard_report(
                query=query,
                analysis_results=analysis_results,
                analysis_summary=analysis_summary,
                raw_data=raw_data,
                cluster_results=cluster_results,
            )

    async def _generate_report_from_raw_data(
        self,
        query: str,
        raw_data: Dict[str, Any],
    ) -> str:
        """
        Generate a report directly from raw data when no analysis is available.

        Args:
            query: Research query
            raw_data: Raw data from sources

        Returns:
            Generated report text
        """
        # Format raw data for the prompt
        formatted_sources = []

        # Process each source type
        for source_type, docs in raw_data.items():
            # Add up to 3 documents per source type
            for i, doc in enumerate(docs[:3]):
                # Extract content
                content = ""
                if "page_content" in doc:
                    content = doc["page_content"]
                elif "content" in doc:
                    content = doc["content"]
                elif "text" in doc:
                    content = doc["text"]

                # Get metadata
                metadata = doc.get("metadata", {})
                title = (
                    doc.get("title", "")
                    or metadata.get("title", "")
                    or f"Document {i} from {source_type}"
                )

                url = (
                    doc.get("url", "") or doc.get("link", "") or metadata.get("url", "")
                )

                # Format the source
                formatted_source = f"SOURCE {source_type.upper()} - {title}"
                if url:
                    formatted_source += f"\nURL: {url}"
                formatted_source += f"\n\n{content[:1000]}..."  # Truncate long content

                formatted_sources.append(formatted_source)

        # Join sources with separators
        sources_text = "\n\n---\n\n".join(formatted_sources)

        # Create prompt for report generation
        prompt_template = """
        You are an expert research report writer. Your task is to create a comprehensive research report
        based on the sources provided, without prior analysis.
        
        RESEARCH QUERY: {query}
        
        SOURCES:
        {sources_text}
        
        Based on these sources, create a comprehensive research report. Your report should:
        
        1. Have a clear structure with an introduction, body sections, and conclusion
        2. Organize information logically by topics or themes
        3. Present a balanced view of different perspectives
        4. Include specific data, examples, and evidence from the sources
        5. Cite sources appropriately using numbered citations like [1], [2], etc.
        6. Use formal, academic language
        7. Include recommendations or next steps when appropriate
        
        CITATION AND REFERENCE GUIDELINES:
        - When citing a source, use a numbered reference style with square brackets: [1], [2], etc.
        - DO NOT use "Document X" style references - use numbered citations instead
        - Include exact URL links in your references section when available
        - For each reference, include title, source type, and URL (if available)
        
        FORMAT REQUIREMENTS:
        - Use Markdown formatting with hierarchical headers
        - Create a table of contents at the beginning
        - Include clear section headers that reflect the content
        - Use bullet points or numbered lists where appropriate
        - End with a References section that includes numbered references with titles and URLs
        - IMPORTANT: Make the report comprehensive and detailed
        
        THE REPORT:
        """

        # Create the LLM chain
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Run the chain
        try:
            response = await chain.arun(
                query=query,
                sources_text=sources_text,
            )
            logger.info(f"Generated report from raw data for query: {query}")
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating report from raw data: {str(e)}")
            # Fall back to basic report
            return self._generate_fallback_report(query, raw_data)

    def _generate_fallback_report(self, query: str, raw_data: Dict[str, Any]) -> str:
        """
        Generate a basic fallback report when other methods fail.

        Args:
            query: Research query
            raw_data: Raw data from sources

        Returns:
            Basic report text
        """
        # Create a simple report structure
        source_types = list(raw_data.keys())
        source_count = sum(len(docs) for docs in raw_data.values())

        report = f"""# Research Report: {query}

## Introduction

This report presents findings related to the query: "{query}". The research gathered information from {source_count} sources across {len(source_types)} different source types.

## Sources Overview

The following source types were used in this research:

"""

        # Add source types
        for source_type, docs in raw_data.items():
            report += f"- {source_type.capitalize()}: {len(docs)} documents\n"

        # Add placeholder sections
        report += f"""
## Key Findings

Based on the collected information, the following key points emerged:

1. Information related to {query} was gathered from multiple sources [1]
2. The sources include {', '.join(source_types)} [2, 3]
3. A total of {source_count} documents were processed

## Conclusion

This research gathered preliminary information on {query}. For a more comprehensive analysis, additional research with advanced analysis methods is recommended.

## References

"""

        # Add better formatted references
        ref_index = 1
        for source_type, docs in raw_data.items():
            for i, doc in enumerate(docs[:5]):  # List up to 5 docs per source
                metadata = doc.get("metadata", {})
                title = get_source_title(doc)

                url = (
                    doc.get("url", "")
                    or doc.get("link", "")
                    or metadata.get("url", "")
                    or metadata.get("link", "")
                    or metadata.get("html_url", "")
                )

                reference = f"{ref_index}. {title}"
                if url:
                    reference += f" - [{url}]({url})"
                reference += f" ({source_type})"

                report += reference + "\n"
                ref_index += 1

        return report

    async def _enhance_with_citations(
        self,
        report: str,
        raw_data: Dict[str, Any],
        sources_dict: Dict[str, Any],
    ) -> str:
        """
        Enhance the report with detailed citations and references.

        Args:
            report: Generated report text
            raw_data: Raw source data
            sources_dict: Dictionary of sources

        Returns:
            Report enhanced with proper citations
        """
        # If no raw data, return the original report
        if not raw_data:
            return report

        # Create a mapping of sources
        source_map = {}
        reference_entries = []

        # Check for generic document references that need replacement
        document_references = {}

        # Process each source
        ref_index = 1
        for source_type, docs in raw_data.items():
            for i, doc in enumerate(docs):
                metadata = doc.get("metadata", {})

                # Get source title
                title = get_source_title(doc)
                if not title:
                    title = f"Source from {source_type}"

                # Get URL
                url = (
                    doc.get("url", "")
                    or doc.get("link", "")
                    or metadata.get("url", "")
                    or metadata.get("link", "")
                    or metadata.get("html_url", "")
                )

                # Create reference key
                ref_key = f"{source_type}_{i}"
                citation_number = ref_index

                # Track document number to citation mapping for replacement
                document_references[f"Document {i+1}"] = citation_number

                # Store in map
                source_map[ref_key] = {
                    "number": citation_number,
                    "title": title,
                    "url": url,
                    "source_type": source_type,
                }

                # Create reference entry with proper formatting
                entry = f"{citation_number}. {title}"
                if url:
                    entry += f" - [{url}]({url})"
                else:
                    # Add additional source identifiers if no URL is available
                    additional_info = []
                    if metadata.get("authors"):
                        additional_info.append(f"Authors: {metadata.get('authors')}")
                    if metadata.get("published"):
                        additional_info.append(
                            f"Published: {metadata.get('published')}"
                        )
                    if additional_info:
                        entry += f" ({', '.join(additional_info)})"

                entry += f" ({source_type})"
                reference_entries.append(entry)

                ref_index += 1

        # Update any references in the report that use generic "Document X" format
        for doc_ref, citation_num in document_references.items():
            # Replace in a way that preserves markdown formatting
            # Only replace if it's a clear reference and not part of a URL or code
            report = (
                report.replace(f"{doc_ref}:", f"[{citation_num}]:")
                .replace(f"{doc_ref}.", f"[{citation_num}].")
                .replace(f"{doc_ref},", f"[{citation_num}],")
                .replace(f"{doc_ref} ", f"[{citation_num}] ")
            )

        # Fix any instances where documents are mentioned in a list
        for doc_ref in sorted(document_references.keys(), key=len, reverse=True):
            citation_num = document_references[doc_ref]
            document_pattern = f"Documents {doc_ref.split(' ')[1]}"
            report = report.replace(document_pattern, f"[{citation_num}]")

        # Check for "Documents X and Y" patterns
        import re

        doc_pattern = r"Documents (\d+) and (\d+)"
        for match in re.finditer(doc_pattern, report):
            doc1, doc2 = match.groups()
            if (
                f"Document {doc1}" in document_references
                and f"Document {doc2}" in document_references
            ):
                cite1 = document_references[f"Document {doc1}"]
                cite2 = document_references[f"Document {doc2}"]
                report = report.replace(
                    f"Documents {doc1} and {doc2}", f"[{cite1}] and [{cite2}]"
                )

        # Check if report already has a references section
        has_references = "## References" in report or "# References" in report

        # If no references section, add one
        if not has_references and reference_entries:
            report += "\n\n## References\n\n"
            report += "\n".join(reference_entries)
        # If it has a references section but might contain generic references, update it
        elif has_references and reference_entries:
            # Find the References section
            references_index = max(
                report.find("## References"), report.find("# References")
            )

            if references_index >= 0:
                # Replace the entire references section
                report_without_refs = report[:references_index]
                report = f"{report_without_refs}\n\n## References\n\n{chr(10).join(reference_entries)}"

        return report
