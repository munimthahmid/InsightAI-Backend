from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from typing import Dict, Any, List, Optional, Literal, Tuple
import uuid
from loguru import logger
import asyncio
import time
import re
import json
from datetime import datetime
import math

from app.core.config import settings
from app.services.data_sources import DataSources
from app.services.embeddings import VectorStorage
from app.services.research_templates import TemplateManager, ResearchTemplate


class ResearchAgent:
    """AI agent for autonomous research."""

    def __init__(self):
        """Initialize the research agent components."""
        # Initialize the language model
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name="gpt-3.5-turbo-16k",  # Use the 16k context window model for longer reports
            temperature=0.2,  # Lower temperature for more focused, fact-based responses
        )

        # Initialize data sources and vector storage
        self.data_sources = DataSources()
        self.vector_storage = VectorStorage()

        # Initialize storage for research history
        self.research_history_file = "research_history.json"

        # Initialize template manager
        self.template_manager = TemplateManager()

    async def save_research_history(self, research_result: Dict[str, Any]) -> str:
        """
        Save research results to history for later retrieval.

        Args:
            research_result: The complete research result dictionary

        Returns:
            ID of the saved research
        """
        try:
            # Generate a unique ID for this research if not already present
            research_id = research_result.get("metadata", {}).get(
                "session_id", str(uuid.uuid4())
            )

            # Add timestamp and research_id
            research_result["saved_at"] = datetime.now().isoformat()
            research_result["research_id"] = research_id

            # Load existing history
            try:
                with open(self.research_history_file, "r") as f:
                    history = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                history = []

            # Add new research and save
            history.append(research_result)

            with open(self.research_history_file, "w") as f:
                json.dump(history, f, indent=2)

            logger.info(
                f"Saved research '{research_result['query']}' to history with ID: {research_id}"
            )
            return research_id

        except Exception as e:
            logger.error(f"Error saving research history: {str(e)}")
            raise

    async def get_research_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve research history.

        Args:
            limit: Maximum number of history items to return

        Returns:
            List of research history items (most recent first)
        """
        try:
            try:
                with open(self.research_history_file, "r") as f:
                    history = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return []

            # Sort by saved_at (most recent first) and limit
            history.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
            return history[:limit]

        except Exception as e:
            logger.error(f"Error retrieving research history: {str(e)}")
            return []

    async def get_research_by_id(self, research_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific research by ID.

        Args:
            research_id: ID of the research to retrieve

        Returns:
            Research data if found, None otherwise
        """
        try:
            try:
                with open(self.research_history_file, "r") as f:
                    history = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return None

            # Find research with matching ID
            for research in history:
                if research.get("research_id") == research_id:
                    return research

            return None

        except Exception as e:
            logger.error(f"Error retrieving research by ID: {str(e)}")
            return None

    async def delete_research_by_id(self, research_id: str) -> bool:
        """
        Delete a specific research by ID.

        Args:
            research_id: ID of the research to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            try:
                with open(self.research_history_file, "r") as f:
                    history = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return False

            # Filter out the research with matching ID
            original_length = len(history)
            history = [r for r in history if r.get("research_id") != research_id]

            if len(history) == original_length:
                # No research was removed
                return False

            # Save updated history
            with open(self.research_history_file, "w") as f:
                json.dump(history, f, indent=2)

            logger.info(f"Deleted research with ID: {research_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting research by ID: {str(e)}")
            return False

    async def research_topic(
        self,
        query: str,
        max_results_per_source: Optional[int] = None,
        save_history: bool = True,
    ) -> Dict[str, Any]:
        """
        Research a topic using multiple sources, process the data, and generate a report.

        Args:
            query: Research topic or query
            max_results_per_source: Maximum number of results to fetch per source
            save_history: Whether to save the research to history

        Returns:
            Dictionary containing the research report and metadata
        """
        if not query or len(query.strip()) < 3:
            raise ValueError("Query must be at least 3 characters")

        logger.info(f"Starting research on topic: {query}")
        start_time = time.time()

        # Create a session-specific namespace for vectors
        namespace = str(uuid.uuid4())

        # Step 1: Fetch data from multiple sources
        data_fetch_start = time.time()

        # Process the query for different data sources
        # Some APIs work better with broader terms, others with specific ones
        cleaned_query = query.strip()

        # For Wikipedia and News API, we want to process the query differently
        # Remove special characters except spaces
        basic_query = re.sub(r"[^\w\s]", " ", cleaned_query)
        # Remove extra whitespace
        basic_query = " ".join(basic_query.split())

        # For Wikipedia, extract key concepts (first 2-3 words)
        wiki_query_parts = basic_query.split()[:3]
        wiki_query = " ".join(wiki_query_parts)

        # For News API, focus on key terms and limit to 2-3 words
        # News API sometimes works better with fewer, more focused terms
        news_query = " ".join(basic_query.split()[:2])

        async with DataSources() as sources:
            # First try the original query for all sources
            logger.info(f"Fetching data with primary query: '{cleaned_query}'")
            data = await sources.fetch_all_sources(
                query=cleaned_query, max_results_per_source=max_results_per_source
            )

            # For any source that returned no results, try alternative queries
            tasks = []

            # Try simplified query for Wikipedia if no results
            if not data.get("wikipedia") and wiki_query != cleaned_query:
                logger.info(
                    f"No Wikipedia results with original query, trying simplified query: '{wiki_query}'"
                )
                tasks.append(
                    sources.fetch_wikipedia_info(
                        query=wiki_query, max_results=max_results_per_source
                    )
                )

            # Try shorter query for News API if no results
            if not data.get("news") and news_query != cleaned_query:
                logger.info(
                    f"No News API results with original query, trying simplified query: '{news_query}'"
                )
                tasks.append(
                    sources.fetch_news_articles(
                        query=news_query, max_results=max_results_per_source
                    )
                )

            # If we have any secondary queries to run, run them concurrently
            if tasks:
                secondary_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process secondary results
                if len(tasks) == 2:  # Both Wikipedia and News were tried
                    if not isinstance(secondary_results[0], Exception):
                        data["wikipedia"] = secondary_results[0]
                    if not isinstance(secondary_results[1], Exception):
                        data["news"] = secondary_results[1]
                elif not data.get("wikipedia") and not data.get("news"):
                    # Only one was tried, figure out which one
                    if wiki_query != cleaned_query and not isinstance(
                        secondary_results[0], Exception
                    ):
                        data["wikipedia"] = secondary_results[0]
                    elif news_query != cleaned_query and not isinstance(
                        secondary_results[0], Exception
                    ):
                        data["news"] = secondary_results[0]

        data_fetch_time = time.time() - data_fetch_start
        logger.info(f"Data fetching completed in {data_fetch_time:.2f} seconds")

        # Log what we got from each source
        for source_type, source_data in data.items():
            count = len(source_data) if source_data else 0
            logger.info(f"Fetched {count} items from {source_type}")

        # Step 2: Process and store in vector DB using the session ID as namespace
        process_start = time.time()
        source_counts = {}

        for source_type, data in data.items():
            count = self.vector_storage.process_and_store(
                data=data, source_type=source_type, namespace=namespace
            )
            source_counts[source_type] = count

        process_time = time.time() - process_start
        logger.info(f"Vector processing completed in {process_time:.2f} seconds")

        # Step 3: Query the vector store with the original query
        query_start = time.time()
        relevant_docs = self.vector_storage.query(
            query_text=query,
            top_k=15,  # Retrieve more documents for better context
            namespace=namespace,
        )

        query_time = time.time() - query_start
        logger.info(f"Vector querying completed in {query_time:.2f} seconds")

        # Step 4: Generate the research report
        report_start = time.time()
        research_report = await self._generate_report(query, relevant_docs)

        report_time = time.time() - report_start
        logger.info(f"Report generation completed in {report_time:.2f} seconds")

        # Step 5: Clean up - delete the temporary namespace
        try:
            self.vector_storage.delete_namespace(namespace)
        except Exception as e:
            logger.error(f"Error deleting namespace {namespace}: {str(e)}")

        total_time = time.time() - start_time
        logger.info(f"Research completed in {total_time:.2f} seconds")

        # Return the research results
        result = {
            "query": query,
            "report": research_report,
            "sources": source_counts,
            "metadata": {
                "session_id": namespace,
                "processing_time": {
                    "fetch_time": round(data_fetch_time, 2),
                    "process_time": round(process_time, 2),
                    "query_time": round(query_time, 2),
                    "report_time": round(report_time, 2),
                    "total_time": round(total_time, 2),
                },
            },
        }

        # Save to history if requested
        if save_history:
            await self.save_research_history(result)

        return result

    async def _generate_report(self, query: str, relevant_docs: Dict[str, Any]) -> str:
        """
        Generate a structured research report based on the retrieved documents.

        Args:
            query: The research query
            relevant_docs: Results from vector store query

        Returns:
            Formatted research report as markdown text
        """
        # Extract text and metadata from relevant documents
        context_texts = []
        sources_dict = {
            "arxiv": [],
            "news": [],
            "github": [],
            "wikipedia": [],
            "semantic_scholar": [],
        }
        evidence_chunks = []

        # Process the matches
        for match in relevant_docs.get("matches", []):
            metadata = match.get("metadata", {})
            text = metadata.get("text", "")
            source = metadata.get("source", "unknown")

            # Skip if no text
            if not text:
                continue

            # Store original text chunk for evidence tracing
            chunk_id = f"{source}-{metadata.get('chunk_id', uuid.uuid4())}"
            evidence_chunk = {
                "id": chunk_id,
                "text": text,
                "source": source,
                "metadata": metadata,
                "score": match.get("score", 0),
            }
            evidence_chunks.append(evidence_chunk)

            # Format based on source type and add to appropriate list
            if source == "arxiv":
                title = metadata.get("title", "Untitled Paper")
                authors = metadata.get("authors", "Unknown Authors")
                url = metadata.get("url", "")
                published = metadata.get("published", "")

                formatted_text = f"RESEARCH PAPER: {title}\nAuthors: {authors}\nPublished: {published}\nURL: {url}\nChunk ID: {chunk_id}\n\n{text}"
                context_texts.append(formatted_text)

                # Add to sources if not already there
                paper_info = {
                    "title": title,
                    "authors": authors,
                    "url": url,
                    "published": published,
                    "chunk_id": chunk_id,
                }
                if not any(
                    p.get("title") == title and p.get("authors") == authors
                    for p in sources_dict["arxiv"]
                ):
                    sources_dict["arxiv"].append(paper_info)

            elif source == "news":
                title = metadata.get("title", "Untitled Article")
                news_source = metadata.get("news_source", "Unknown Source")
                url = metadata.get("url", "")
                published = metadata.get("published", "")

                formatted_text = f"NEWS ARTICLE: {title}\nSource: {news_source}\nPublished: {published}\nURL: {url}\nChunk ID: {chunk_id}\n\n{text}"
                context_texts.append(formatted_text)

                article_info = {
                    "title": title,
                    "source": news_source,
                    "url": url,
                    "published": published,
                    "chunk_id": chunk_id,
                }
                if not any(
                    a.get("title") == title and a.get("url") == url
                    for a in sources_dict["news"]
                ):
                    sources_dict["news"].append(article_info)

            elif source == "github":
                name = metadata.get("full_name", "Unknown Repository")
                description = (
                    text.split("\n\n")[1].replace("Description: ", "")
                    if "\n\n" in text
                    else ""
                )
                url = metadata.get("url", "")
                stars = metadata.get("stars", 0)
                language = metadata.get("language", "")

                formatted_text = f"GITHUB REPOSITORY: {name}\nStars: {stars}\nLanguage: {language}\nURL: {url}\nChunk ID: {chunk_id}\n\n{text}"
                context_texts.append(formatted_text)

                repo_info = {
                    "name": name,
                    "url": url,
                    "stars": stars,
                    "language": language,
                    "chunk_id": chunk_id,
                }
                if not any(
                    r.get("name") == name and r.get("url") == url
                    for r in sources_dict["github"]
                ):
                    sources_dict["github"].append(repo_info)

            elif source == "wikipedia":
                title = metadata.get("title", "Untitled Wikipedia Article")
                url = metadata.get("url", "")

                formatted_text = f"WIKIPEDIA ARTICLE: {title}\nURL: {url}\nChunk ID: {chunk_id}\n\n{text}"
                context_texts.append(formatted_text)

                wiki_info = {"title": title, "url": url, "chunk_id": chunk_id}
                if not any(
                    w.get("title") == title and w.get("url") == url
                    for w in sources_dict["wikipedia"]
                ):
                    sources_dict["wikipedia"].append(wiki_info)

            elif source == "semantic_scholar":
                title = metadata.get("title", "Untitled Paper")
                authors = metadata.get("authors", "Unknown Authors")
                url = metadata.get("url", "")
                year = metadata.get("year", "")
                venue = metadata.get("venue", "")

                formatted_text = f"SEMANTIC SCHOLAR PAPER: {title}\nAuthors: {authors}\nYear: {year}\nVenue: {venue}\nURL: {url}\nChunk ID: {chunk_id}\n\n{text}"
                context_texts.append(formatted_text)

                scholar_info = {
                    "title": title,
                    "authors": authors,
                    "url": url,
                    "year": year,
                    "venue": venue,
                    "chunk_id": chunk_id,
                }
                if not any(
                    s.get("title") == title and s.get("authors") == authors
                    for s in sources_dict["semantic_scholar"]
                ):
                    sources_dict["semantic_scholar"].append(scholar_info)

        # Combine all context
        context = "\n\n---\n\n".join(context_texts)

        # Create a formatted list of sources for citation
        sources_text = self._format_sources_for_citation(sources_dict)

        # Detect contradictions and find supporting evidence
        contradictions, consensus = await self._detect_contradictions(
            evidence_chunks, query
        )

        # Include contradiction information in the prompt
        contradiction_text = ""
        if contradictions:
            contradiction_text = "## Note on Contradictory Information\n\n"
            for topic, items in contradictions.items():
                contradiction_text += f"### {topic}\n\n"
                for view in items:
                    contradiction_text += f"- {view['statement']}\n"
                    contradiction_text += (
                        f"  - Source: {view['source_type']}, {view['source_title']}\n"
                    )
                contradiction_text += "\n"

        # Create prompt template with improved instructions for more comprehensive research report
        template = """
        You are an expert AI research assistant with a PhD-level understanding of the subject matter.
        Your task is to create a comprehensive, authoritative research report on the following topic:
        
        TOPIC: {query}
        
        Based on the provided information:
        
        {context}
        
        Generate a detailed, well-structured research report in markdown format with the following sections:
        
        # Executive Summary
        A concise overview of the key findings (2-3 paragraphs) that highlights the most significant aspects of the topic.
        
        # Key Findings and Insights
        Bullet points highlighting the most important insights, breakthroughs, and current state of knowledge.
        
        # Research Analysis
        ## Current State of Academic Research
        Detailed summary of academic papers, their methodologies, contributions, and how they relate to each other.
        Include any conflicting viewpoints or competing theories if they exist.
        
        ## Recent Developments and Trends
        Analysis of recent news, breakthroughs, and emerging trends in this field.
        Identify patterns of development and where the field is heading.
        
        ## Top Researchers and Research Groups
        Identify leading researchers, professors, labs, and institutions working in this field.
        Highlight their main contributions and focus areas.
        
        ## Notable Projects and Implementations
        Overview of relevant GitHub repositories, commercial products, or other implementations.
        Evaluate their technical approach, strengths, and limitations.
        
        # Technical Analysis
        ## Methodologies and Approaches
        Analysis of the main technical approaches, algorithms, or methodologies used in this field.
        Compare different approaches when multiple exist.
        
        ## Challenges and Limitations
        Discussion of current technical challenges, limitations, and unsolved problems.
        
        # Practical Applications
        Existing and potential real-world applications of this technology or research area.
        Industry adoption and practical implementations.
        
        # Future Research Directions
        ## Emerging Research Areas
        Identify promising new research directions and unexplored territories.
        
        ## Potential Breakthroughs
        Discuss what breakthroughs might be expected in the near to medium term.
        
        ## Open Questions
        List significant unanswered questions in the field that require further research.
        
        # Conclusions
        Summary of the current state and potential future developments.
        
        {contradictions}
        
        # Sources
        {sources}
        
        IMPORTANT INSTRUCTIONS FOR EVIDENCE CITATIONS:
        1. For EVERY important fact, claim, or finding you include, add a citation using the format [Claim](chunk_id).
           For example: "Large language models demonstrate strong performance on reasoning tasks [Study shows GPT-4 outperforms humans on logic puzzles](arxiv-123)"
        2. When multiple sources support a claim, include all of them: [Claim](chunk_id1)(chunk_id2)
        3. When sources have conflicting information, present all viewpoints and clearly indicate the source of each.
        4. Present a balanced view when there are conflicting perspectives.
        5. Use your judgment to synthesize information, but always make it clear which source supports each claim.
        
        IMPORTANT GUIDELINES:
        1. Use proper markdown formatting with headers, bullet points, and emphasis where appropriate.
        2. Draw insights by synthesizing information from multiple sources when available.
        3. Be objective and fact-based, avoiding speculation unless clearly indicated as such.
        4. Provide specific technical details relevant to the topic.
        5. When comparing research papers or approaches, highlight their differences, similarities, and relative advantages.
        6. If certain sections have insufficient information, briefly mention this limitation rather than inventing details.
        7. Use concise, clear language that would be suitable for an academic or professional audience.
        8. NEVER make up information. If there's not enough data for a section, acknowledge the gap.
        9. Identify contradictions or conflicts in the research when they exist.
        10. For technical topics, include formulas, algorithms, or pseudocode when relevant.
        """

        # Format the prompt template
        prompt = ChatPromptTemplate.from_template(template)

        # Generate the report using a more capable model for complex research topics
        chain = LLMChain(
            llm=ChatOpenAI(
                openai_api_key=settings.OPENAI_API_KEY,
                model_name="gpt-4",  # Use most capable model for in-depth research
                temperature=0.2,  # Keep temperature low for factual reporting
            ),
            prompt=prompt,
        )

        result = await chain.arun(
            query=query,
            context=context,
            sources=sources_text,
            contradictions=contradiction_text,
        )

        # Process the result to enhance with evidence links and source verification
        enhanced_report = await self._enhance_report_with_citations(
            result, evidence_chunks, sources_dict
        )

        return enhanced_report

    async def _detect_contradictions(
        self, evidence_chunks: List[Dict[str, Any]], query: str
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
        """
        Detect contradictory information in the evidence chunks.

        Args:
            evidence_chunks: List of evidence chunks
            query: The research query

        Returns:
            Tuple of contradictions and consensus information
        """
        if len(evidence_chunks) < 2:
            return {}, {}

        # First, we need to extract key claims from each chunk
        extraction_prompt = """
        Extract the key factual claims from the following text related to the topic: {query}
        
        Text: {text}
        
        Return ONLY a JSON array of claims, with each claim being a single factual statement.
        The claims should be self-contained statements that can be verified as true or false.
        Format: [{"claim": "statement 1"}, {"claim": "statement 2"}]
        """

        claim_template = ChatPromptTemplate.from_template(extraction_prompt)
        claim_chain = LLMChain(
            llm=ChatOpenAI(
                openai_api_key=settings.OPENAI_API_KEY,
                model_name="gpt-3.5-turbo",
                temperature=0.0,
            ),
            prompt=claim_template,
        )

        # Process chunks in batches to extract claims
        all_claims = []
        for chunk in evidence_chunks:
            try:
                claims_text = await claim_chain.arun(query=query, text=chunk["text"])
                # Parse the JSON response
                try:
                    claims = json.loads(claims_text)
                    for claim in claims:
                        if isinstance(claim, dict) and "claim" in claim:
                            all_claims.append(
                                {
                                    "claim": claim["claim"],
                                    "chunk_id": chunk["id"],
                                    "source_type": chunk["source"],
                                    "source_title": self._get_source_title(chunk),
                                    "score": chunk["score"],
                                }
                            )
                except:
                    logger.warning(f"Failed to parse claims from chunk {chunk['id']}")
                    continue
            except Exception as e:
                logger.error(f"Error extracting claims: {str(e)}")
                continue

        # If we have too few claims, no need to continue
        if len(all_claims) < 3:
            return {}, {}

        # Calculate embeddings for all claims to compare them
        try:
            # Use OpenAI embeddings for claim comparison
            embeddings = OpenAIEmbeddings(
                openai_api_key=settings.OPENAI_API_KEY, model="text-embedding-ada-002"
            )

            claim_texts = [claim["claim"] for claim in all_claims]
            claim_embeddings = embeddings.embed_documents(claim_texts)

            # Cluster similar claims
            clusters = {}
            for i, (claim, embedding) in enumerate(zip(all_claims, claim_embeddings)):
                added = False
                for cluster_key, cluster_items in clusters.items():
                    # Compare with the first item in each cluster
                    similarity = self._cosine_similarity(
                        embedding, claim_embeddings[cluster_items[0]["index"]]
                    )
                    if similarity > 0.85:  # High similarity threshold
                        clusters[cluster_key].append({"index": i, "claim": claim})
                        added = True
                        break

                if not added:
                    # Create a new cluster
                    cluster_key = f"cluster_{i}"
                    clusters[cluster_key] = [{"index": i, "claim": claim}]

            # Analyze clusters for contradictions
            contradictions = {}
            consensus = {}

            for cluster_key, cluster_items in clusters.items():
                # Only consider clusters with multiple claims
                if len(cluster_items) < 2:
                    continue

                # Check if the claims in this cluster contradict each other
                if len(cluster_items) >= 2:
                    # Use a language model to check for contradictions
                    contradiction_prompt = """
                    Analyze the following claims about the same topic and determine if they contradict each other or are consistent.
                    
                    Claims:
                    {claims}
                    
                    First, identify the main topic these claims are about.
                    Then, determine if there are ANY contradictions or inconsistencies between these claims.
                    
                    Return ONLY a JSON object with the following format:
                    {{"topic": "brief description of what these claims are about", 
                      "contradictory": true/false, 
                      "explanation": "brief explanation of the contradiction or consistency"}}
                    """

                    claims_text = "\n".join(
                        [
                            f"{i+1}. {item['claim']['claim']}"
                            for i, item in enumerate(cluster_items)
                        ]
                    )

                    contradiction_template = ChatPromptTemplate.from_template(
                        contradiction_prompt
                    )
                    contradiction_chain = LLMChain(
                        llm=ChatOpenAI(
                            openai_api_key=settings.OPENAI_API_KEY,
                            model_name="gpt-4",
                            temperature=0.0,
                        ),
                        prompt=contradiction_template,
                    )

                    try:
                        result_text = await contradiction_chain.arun(claims=claims_text)
                        result = json.loads(result_text)

                        topic = result.get("topic", cluster_key)
                        is_contradictory = result.get("contradictory", False)

                        if is_contradictory:
                            if topic not in contradictions:
                                contradictions[topic] = []

                            for item in cluster_items:
                                claim_data = item["claim"]
                                contradictions[topic].append(
                                    {
                                        "statement": claim_data["claim"],
                                        "source_type": claim_data["source_type"],
                                        "source_title": claim_data["source_title"],
                                        "chunk_id": claim_data["chunk_id"],
                                    }
                                )
                        else:
                            # Add to consensus if not contradictory
                            if topic not in consensus:
                                consensus[topic] = []

                            for item in cluster_items:
                                claim_data = item["claim"]
                                consensus[topic].append(
                                    {
                                        "statement": claim_data["claim"],
                                        "source_type": claim_data["source_type"],
                                        "source_title": claim_data["source_title"],
                                        "chunk_id": claim_data["chunk_id"],
                                    }
                                )
                    except Exception as e:
                        logger.error(f"Error analyzing contradictions: {str(e)}")
                        continue

        except Exception as e:
            logger.error(f"Error in contradiction analysis: {str(e)}")
            return {}, {}

        return contradictions, consensus

    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        if magnitude1 * magnitude2 == 0:
            return 0
        return dot_product / (magnitude1 * magnitude2)

    def _get_source_title(self, chunk):
        """Get a readable title for the source of a chunk"""
        metadata = chunk["metadata"]
        source = chunk["source"]

        if source == "arxiv":
            return metadata.get("title", "Research paper")
        elif source == "news":
            return f"{metadata.get('title', 'News article')} ({metadata.get('news_source', 'Unknown source')})"
        elif source == "github":
            return metadata.get("full_name", "GitHub repository")
        elif source == "wikipedia":
            return metadata.get("title", "Wikipedia article")
        elif source == "semantic_scholar":
            return metadata.get("title", "Research paper")
        else:
            return "Unknown source"

    async def _enhance_report_with_citations(
        self,
        report: str,
        evidence_chunks: List[Dict[str, Any]],
        sources_dict: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        """
        Enhance the report with citations, evidence links, and source verification.

        Args:
            report: The generated research report
            evidence_chunks: List of evidence chunks
            sources_dict: Dictionary of sources

        Returns:
            Enhanced report with citations and evidence links
        """
        # Create a citation index to add tooltips and details
        citation_index = {}

        # Build a citation index from all sources
        for source_type, sources in sources_dict.items():
            for source in sources:
                chunk_id = source.get("chunk_id")
                if not chunk_id:
                    continue

                # Find the corresponding evidence chunk
                chunk = next((c for c in evidence_chunks if c["id"] == chunk_id), None)
                if not chunk:
                    continue

                citation_data = {
                    "source_type": source_type,
                    "text": chunk["text"],
                    "metadata": source,
                }

                citation_index[chunk_id] = citation_data

        # Find citation patterns like [text](chunk_id) in the report
        citation_pattern = r"\[(.*?)\]\(([^)]+)\)"

        # Replace citations with enhanced versions that include tooltips and source verification
        enhanced_report = report

        # Add source verification section at the end
        source_verification = "\n\n## Evidence and Citations\n\n"
        source_verification += "Each claim in this report is linked to its source material. Click on the citation links to see the supporting evidence.\n\n"

        # Add footnotes for each citation
        footnotes = {}
        matches = re.findall(citation_pattern, report)

        for match in matches:
            text, chunk_id = match
            # Skip if the chunk_id is not a valid citation ID
            if chunk_id not in citation_index:
                continue

            citation = citation_index[chunk_id]
            source_type = citation["source_type"]

            # Create detailed footnote
            if chunk_id not in footnotes:
                source_info = ""
                if source_type == "arxiv":
                    title = citation["metadata"].get("title", "")
                    authors = citation["metadata"].get("authors", "")
                    source_info = f"{title} by {authors}"
                elif source_type == "news":
                    title = citation["metadata"].get("title", "")
                    source = citation["metadata"].get("source", "")
                    source_info = f"{title} from {source}"
                elif source_type == "github":
                    name = citation["metadata"].get("name", "")
                    stars = citation["metadata"].get("stars", "")
                    source_info = f"{name} ({stars} stars)"
                elif source_type == "wikipedia":
                    title = citation["metadata"].get("title", "")
                    source_info = f"Wikipedia: {title}"
                elif source_type == "semantic_scholar":
                    title = citation["metadata"].get("title", "")
                    authors = citation["metadata"].get("authors", "")
                    source_info = f"{title} by {authors}"

                evidence_text = citation["text"]

                # Add to footnotes
                footnote = f"**Source ({source_type})**: {source_info}\n\n"
                footnote += f"**Evidence**: {evidence_text}"

                footnotes[chunk_id] = footnote

        # Add footnotes to the verification section
        if footnotes:
            for chunk_id, footnote in footnotes.items():
                source_verification += f"### Citation {chunk_id}\n\n{footnote}\n\n"

            # Add the verification section to the end of the report
            enhanced_report += source_verification

        return enhanced_report

    def _format_sources_for_citation(
        self, sources_dict: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """
        Format sources for citation in the research report.

        Args:
            sources_dict: Dictionary of sources organized by type

        Returns:
            Formatted sources text for the report
        """
        sources_text = "List all sources used in this report in the following format:\n"

        # Format research papers
        if sources_dict["arxiv"]:
            sources_text += "\n## Research Papers\n"
            for i, paper in enumerate(sources_dict["arxiv"], 1):
                title = paper.get("title", "Untitled")
                authors = paper.get("authors", "Unknown Authors")
                url = paper.get("url", "")
                sources_text += f"{i}. [{title}]({url}) - {authors}\n"

        # Format news articles
        if sources_dict["news"]:
            sources_text += "\n## News Articles\n"
            for i, article in enumerate(sources_dict["news"], 1):
                title = article.get("title", "Untitled")
                source = article.get("source", "Unknown Source")
                url = article.get("url", "")
                sources_text += f"{i}. [{title}]({url}) - {source}\n"

        # Format GitHub repositories
        if sources_dict["github"]:
            sources_text += "\n## GitHub Repositories\n"
            for i, repo in enumerate(sources_dict["github"], 1):
                name = repo.get("name", "Unknown Repository")
                stars = repo.get("stars", 0)
                language = repo.get("language", "")
                url = repo.get("url", "")
                sources_text += f"{i}. [{name}]({url}) - {language}, {stars} stars\n"

        # Format Wikipedia articles
        if sources_dict["wikipedia"]:
            sources_text += "\n## Wikipedia Articles\n"
            for i, wiki in enumerate(sources_dict["wikipedia"], 1):
                title = wiki.get("title", "Untitled")
                url = wiki.get("url", "")
                sources_text += f"{i}. [{title}]({url})\n"

        return sources_text

    async def generate_literature_review(
        self,
        research_id: str,
        format_type: Literal["APA", "MLA", "Chicago", "IEEE"] = "APA",
        section_format: Literal[
            "chronological", "thematic", "methodological"
        ] = "thematic",
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a formal literature review based on previously conducted research.

        Args:
            research_id: ID of existing research to use as basis
            format_type: Citation format to use
            section_format: How to organize the literature review sections
            max_length: Maximum word count (approximate)

        Returns:
            Dictionary containing the literature review and metadata
        """
        # Get the existing research
        research = await self.get_research_by_id(research_id)
        if not research:
            raise ValueError(f"Research with ID {research_id} not found")

        query = research["query"]
        report = research["report"]
        sources = research.get("sources", {})

        logger.info(
            f"Generating literature review for research: {query} (ID: {research_id})"
        )

        # Create prompt for literature review generation
        template = """
        You are an academic research assistant specializing in writing literature reviews.
        
        Please create a formal literature review on the topic: "{query}"
        
        Format requirements:
        - Use {format_type} citation format
        - Organize sections {section_format}ally
        - {length_constraint}
        
        Here is the existing research report to build upon:
        ```
        {report}
        ```
        
        Your literature review should:
        1. Start with a proper introduction stating the purpose and scope
        2. Include a methodology section explaining how literature was selected
        3. Organize the main body {section_format}ally
        4. Identify gaps in existing research
        5. Conclude with recommendations for future research
        6. Include a properly formatted bibliography/references section
        
        The literature review should be well-structured, critical, and analytical rather than merely summarizing sources.
        """

        length_constraint = (
            f"Maximum length of approximately {max_length} words"
            if max_length
            else "No specific length constraint"
        )

        prompt = ChatPromptTemplate.from_template(template)

        # Create a dedicated chain for literature review generation
        literature_review_chain = LLMChain(
            llm=ChatOpenAI(
                openai_api_key=settings.OPENAI_API_KEY,
                model_name="gpt-4",  # Use more capable model for academic writing
                temperature=0.2,
            ),
            prompt=prompt,
        )

        # Generate the literature review
        try:
            literature_review = await literature_review_chain.arun(
                query=query,
                report=report,
                format_type=format_type,
                section_format=section_format,
                length_constraint=length_constraint,
            )

            # Create result dictionary
            result = {
                "research_id": research_id,
                "query": query,
                "literature_review": literature_review,
                "format": {
                    "citation_style": format_type,
                    "organization": section_format,
                    "max_length": max_length,
                },
                "generated_at": datetime.now().isoformat(),
            }

            logger.info(f"Generated literature review for research ID: {research_id}")
            return result

        except Exception as e:
            logger.error(f"Error generating literature review: {str(e)}")
            raise

    async def compare_research(
        self,
        research_ids: List[str],
        comparison_aspects: Optional[List[str]] = None,
        include_visualization: bool = True,
    ) -> Dict[str, Any]:
        """
        Compare multiple research topics or papers.

        Args:
            research_ids: List of research IDs to compare
            comparison_aspects: Specific aspects to compare (e.g., methodology, findings, data sources)
            include_visualization: Whether to include markdown-based visualization data

        Returns:
            Dictionary containing comparison results
        """
        if len(research_ids) < 2:
            raise ValueError("At least two research IDs are required for comparison")

        # Get all research data
        research_data = []
        topics = []

        for research_id in research_ids:
            research = await self.get_research_by_id(research_id)
            if not research:
                raise ValueError(f"Research with ID {research_id} not found")

            research_data.append(research)
            topics.append(research["query"])

        logger.info(f"Comparing {len(topics)} research topics: {', '.join(topics)}")

        # Determine aspects to compare
        if not comparison_aspects:
            comparison_aspects = [
                "Main findings and discoveries",
                "Research methodologies and approaches",
                "Theoretical frameworks",
                "Data sources and quality",
                "Technological implementations",
                "Key research contributors and groups",
                "Historical development and trends",
                "Current limitations and challenges",
                "Practical applications and industry relevance",
                "Future research directions",
                "Interdisciplinary connections",
                "Societal and ethical implications",
            ]

        # Format research data for comparison
        research_context = []
        for i, research in enumerate(research_data):
            # Extract sources information
            sources_info = ""
            if "sources" in research:
                sources_info = "\n".join(
                    [
                        f"- {source}: {count} results"
                        for source, count in research["sources"].items()
                    ]
                )

            # Include any metadata about templates if available
            template_info = ""
            if "metadata" in research and "template" in research["metadata"]:
                template = research["metadata"]["template"]
                template_info = f"\nResearch Template: {template.get('name', 'Unknown')} (Domain: {template.get('domain', 'Unknown')})"

            research_context.append(
                f"""
            RESEARCH TOPIC #{i+1}: {research['query']}
            {template_info}
            
            FULL REPORT:
            ```
            {research['report']}
            ```
            
            SOURCES:
            {sources_info}
            """
            )

        # Create enhanced prompt for comparison
        template = """
        You are a world-class research scientist with expertise in comparative analysis and synthesis across multiple domains.
        
        Your task is to conduct a detailed, insightful comparison between the following research topics:
        {topics_list}
        
        Please compare them specifically on these aspects:
        {aspects_list}
        
        Here is the comprehensive data for each research topic:
        
        {research_context}
        
        INSTRUCTIONS FOR COMPARATIVE ANALYSIS:
        
        1. Begin with an "Executive Summary" that highlights the most significant similarities and differences
           between the topics and your key insights from the comparison.
        
        2. For each comparison aspect:
           - Analyze each topic individually first
           - Then directly compare and contrast them
           - Identify patterns, relationships, and underlying principles
           - Note any conflicting findings or approaches
           - Assess the relative strengths and limitations of each
        
        3. In a "Synthesis" section:
           - Explore how these topics relate to or complement each other
           - Identify potential synergies or collaborative opportunities between the fields
           - Discuss how insights from one topic might address limitations in another
        
        4. In a "Research Gaps and Opportunities" section:
           - Identify questions that remain unanswered across these topics
           - Suggest specific new research directions that emerge from the comparison
           - Propose potential methodologies for these new directions
        
        5. Conclude with "Key Takeaways" that capture the most valuable insights from this comparative analysis
        
        Format your response as a well-structured markdown document with clear hierarchical headings.
        
        {visualization_request}
        
        IMPORTANT:
        - Be precise, specific, and technically accurate
        - Support claims with evidence from the research reports
        - Identify genuine connections rather than superficial similarities
        - Note both complementary and contradictory elements between the topics
        - Provide nuanced analysis that goes beyond simply summarizing each topic
        - Aim for insights that wouldn't be obvious from examining each topic in isolation
        """

        topics_list = "\n".join([f"- {topic}" for topic in topics])
        aspects_list = "\n".join([f"- {aspect}" for aspect in comparison_aspects])

        visualization_request = (
            """
        Additionally, include a "Comparative Visualization" section with:
        
        1. A comprehensive markdown comparison table that summarizes key points for each aspect across all topics
        
        2. Where relevant, include at least one more advanced visualization such as:
           - A conceptual relationship diagram (using ASCII/markdown)
           - A technology/methodology evolution timeline
           - A capabilities comparison chart
           - A matrix showing how approaches from different topics could be combined
        
        Make sure these visualizations highlight the most important insights from your analysis.
        """
            if include_visualization
            else ""
        )

        prompt = ChatPromptTemplate.from_template(template)

        # Create a dedicated chain for comparison
        comparison_chain = LLMChain(
            llm=ChatOpenAI(
                openai_api_key=settings.OPENAI_API_KEY,
                model_name="gpt-4",  # Use more capable model for complex comparison
                temperature=0.1,  # Lower temperature for more precise analysis
            ),
            prompt=prompt,
        )

        # Generate the comparison
        try:
            comparison_result = await comparison_chain.arun(
                topics_list=topics_list,
                aspects_list=aspects_list,
                research_context="\n\n".join(research_context),
                visualization_request=visualization_request,
            )

            # Create result dictionary
            result = {
                "topics": topics,
                "research_ids": research_ids,
                "comparison_aspects": comparison_aspects,
                "comparison_result": comparison_result,
                "generated_at": datetime.now().isoformat(),
            }

            logger.info(
                f"Generated enhanced comparison for research topics: {', '.join(topics)}"
            )
            return result

        except Exception as e:
            logger.error(f"Error generating comparison: {str(e)}")
            raise

    async def research_topic_with_template(
        self,
        query: str,
        template_id: str,
        max_results_per_source: Optional[int] = None,
        save_history: bool = True,
    ) -> Dict[str, Any]:
        """
        Research a topic using a specific template.

        Args:
            query: Research topic or query
            template_id: ID of the template to use
            max_results_per_source: Maximum number of results to fetch per source
            save_history: Whether to save the research to history

        Returns:
            Dictionary containing the research report and metadata
        """
        # Get the template
        template = self.template_manager.get_template_by_id(template_id)
        if not template:
            raise ValueError(f"Template with ID {template_id} not found")

        logger.info(f"Using template '{template.name}' for research on: {query}")

        # First collect data as in regular research
        if not query or len(query.strip()) < 3:
            raise ValueError("Query must be at least 3 characters")

        logger.info(f"Starting template-based research on topic: {query}")
        start_time = time.time()

        # Create a session-specific namespace for vectors
        namespace = str(uuid.uuid4())

        # Step 1: Fetch data from sources specified in the template
        data_fetch_start = time.time()
        cleaned_query = query.strip()

        # Use the template's default sources if specified
        sources_to_use = template.default_sources

        async with DataSources() as sources:
            logger.info(f"Fetching data from sources: {', '.join(sources_to_use)}")
            # Modify to only fetch from specified sources
            data = {}

            # Fetch from all requested sources
            tasks = []
            if "arxiv" in sources_to_use:
                tasks.append(
                    sources.fetch_arxiv_papers(
                        query=cleaned_query, max_results=max_results_per_source
                    )
                )
            if "news" in sources_to_use:
                tasks.append(
                    sources.fetch_news_articles(
                        query=cleaned_query, max_results=max_results_per_source
                    )
                )
            if "github" in sources_to_use:
                tasks.append(
                    sources.fetch_github_repos(
                        query=cleaned_query, max_results=max_results_per_source
                    )
                )
            if "wikipedia" in sources_to_use:
                tasks.append(
                    sources.fetch_wikipedia_info(
                        query=cleaned_query, max_results=max_results_per_source
                    )
                )

            # Run all fetch tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, source in enumerate(sources_to_use):
                if i < len(results) and not isinstance(results[i], Exception):
                    data[source] = results[i]
                else:
                    data[source] = []

        data_fetch_time = time.time() - data_fetch_start
        logger.info(f"Data fetching completed in {data_fetch_time:.2f} seconds")

        # Step 2: Process and store in vector DB using the session ID as namespace
        process_start = time.time()
        source_counts = {}

        for source_type, source_data in data.items():
            count = self.vector_storage.process_and_store(
                data=source_data, source_type=source_type, namespace=namespace
            )
            source_counts[source_type] = count

        process_time = time.time() - process_start
        logger.info(f"Vector processing completed in {process_time:.2f} seconds")

        # Step 3: Query the vector store with the original query
        query_start = time.time()
        relevant_docs = self.vector_storage.query(
            query_text=query,
            top_k=15,  # Retrieve more documents for better context
            namespace=namespace,
        )

        query_time = time.time() - query_start
        logger.info(f"Vector querying completed in {query_time:.2f} seconds")

        # Step 4: Generate the template-specific research report
        report_start = time.time()
        research_report = await self._generate_template_report(
            query, relevant_docs, template
        )

        report_time = time.time() - report_start
        logger.info(
            f"Template-based report generation completed in {report_time:.2f} seconds"
        )

        # Step 5: Clean up - delete the temporary namespace
        try:
            self.vector_storage.delete_namespace(namespace)
        except Exception as e:
            logger.error(f"Error deleting namespace {namespace}: {str(e)}")

        total_time = time.time() - start_time
        logger.info(f"Template-based research completed in {total_time:.2f} seconds")

        # Return the research results
        result = {
            "query": query,
            "report": research_report,
            "sources": source_counts,
            "template": {
                "id": template.template_id,
                "name": template.name,
                "domain": template.domain,
            },
            "metadata": {
                "session_id": namespace,
                "processing_time": {
                    "fetch_time": round(data_fetch_time, 2),
                    "process_time": round(process_time, 2),
                    "query_time": round(query_time, 2),
                    "report_time": round(report_time, 2),
                    "total_time": round(total_time, 2),
                },
            },
        }

        # Save to history if requested
        if save_history:
            await self.save_research_history(result)

        return result

    async def _generate_template_report(
        self, query: str, relevant_docs: Dict[str, Any], template: ResearchTemplate
    ) -> str:
        """
        Generate a research report based on a specific template.

        Args:
            query: The research query
            relevant_docs: Results from vector store query
            template: The template to use

        Returns:
            Formatted research report as markdown text
        """
        # Extract text and metadata from relevant documents
        context_texts = []
        sources_dict = {"arxiv": [], "news": [], "github": [], "wikipedia": []}

        # Process the matches
        for match in relevant_docs.get("matches", []):
            metadata = match.get("metadata", {})
            text = metadata.get("text", "")
            source = metadata.get("source", "unknown")

            # Skip if no text
            if not text:
                continue

            # Add to context
            context_texts.append(f"SOURCE ({source.upper()}): {text}")

            # Also collect source information
            if source == "arxiv":
                title = metadata.get("title", "Untitled Paper")
                authors = metadata.get("authors", "Unknown Authors")
                url = metadata.get("url", "")
                published = metadata.get("published", "")

                paper_info = {"title": title, "authors": authors, "url": url}
                if paper_info not in sources_dict["arxiv"]:
                    sources_dict["arxiv"].append(paper_info)

            elif source == "news":
                title = metadata.get("title", "Untitled Article")
                news_source = metadata.get("news_source", "Unknown Source")
                url = metadata.get("url", "")
                published = metadata.get("published", "")

                article_info = {"title": title, "source": news_source, "url": url}
                if article_info not in sources_dict["news"]:
                    sources_dict["news"].append(article_info)

            elif source == "github":
                name = metadata.get("full_name", "Unknown Repository")
                url = metadata.get("url", "")

                repo_info = {"name": name, "url": url}
                if repo_info not in sources_dict["github"]:
                    sources_dict["github"].append(repo_info)

            elif source == "wikipedia":
                title = metadata.get("title", "Untitled Page")
                url = metadata.get("url", "")

                wiki_info = {"title": title, "url": url}
                if wiki_info not in sources_dict["wikipedia"]:
                    sources_dict["wikipedia"].append(wiki_info)

        # Create context
        context = "\n\n".join(context_texts)

        # Format sources for citation
        sources_text = self._format_sources_for_citation(sources_dict)

        # Get template-specific prompt
        template_prompt = template.prompt_template

        # Create a chain for the template-specific report generation
        report_prompt = ChatPromptTemplate.from_template(
            f"""
        {template_prompt}
        
        Here is the information collected on this topic:
        
        {context}
        
        Based on this information, generate a comprehensive report following the specified structure.
        
        At the end of the report, include a "Sources" section with:
        
        {sources_text}
        
        Format the entire report in well-structured markdown.
        """
        )

        # Create the chain for report generation
        report_chain = LLMChain(
            llm=self.llm,
            prompt=report_prompt,
        )

        # Generate the report
        report = await report_chain.arun(query=query)

        return report.strip()
