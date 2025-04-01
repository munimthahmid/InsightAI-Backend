"""
Analysis agent responsible for analyzing collected data.
"""

import json
import time
from typing import Dict, List, Any, Optional
import numpy as np
from loguru import logger
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

from app.core.config import settings
from app.services.research.agents.base_agent import BaseAgent
from app.services.research.orchestration.schemas import TaskSchema
from app.services.vector_db.clustering.kmeans import KMeansClustering
from app.services.vector_db.clustering.hdbscan_cluster import HDBSCANClustering
from app.services.vector_db.clustering.mmr import MaximumMarginalRelevance
from app.services.vector_db.vector_operations import VectorOperations
from langchain.embeddings import OpenAIEmbeddings


class AnalysisAgent(BaseAgent):
    """
    Analysis agent that processes collected information.
    """

    def __init__(
        self,
        context_manager=None,
        task_queue=None,
        agent_id=None,
    ):
        """Initialize the analysis agent."""
        super().__init__(
            agent_id=agent_id,
            agent_type="analysis",
            context_manager=context_manager,
            task_queue=task_queue,
        )

        # Register this agent's task handler
        if task_queue:
            task_queue.register_handler("analysis_task", self.execute_task)

        # Initialize analysis components
        self.kmeans_clustering = KMeansClustering()
        self.hdbscan_clustering = HDBSCANClustering()
        self.mmr = MaximumMarginalRelevance()

        # Initialize language model for analysis
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name="gpt-4o",  # Use GPT-4o for better analysis
            temperature=0.2,  # Low temperature for focused, fact-based analysis
        )

        logger.info(f"AnalysisAgent initialized with ID: {self.agent_id}")

    async def execute_task(self, task: TaskSchema) -> Dict[str, Any]:
        """
        Execute an analysis task.

        This method analyzes the documents collected by the acquisition agent,
        extracts key information, identifies topics, and creates a structured
        analysis of the research question.

        Args:
            task: The task containing query parameters

        Returns:
            Dictionary with analysis results
        """
        # Extract task parameters
        params = task.params
        query = params.get("query", "")

        await self.log_activity(
            "start_analysis",
            {"task_id": task.task_id, "query": query},
        )

        try:
            # Retrieve collected documents from context
            raw_data = await self.get_context("raw_data")
            processed_docs = await self.get_context("processed_docs")
            if processed_docs:
                logger.info(f"Found {len(processed_docs)} processed documents")
            vector_namespace = await self.get_context("vector_namespace")

            if not processed_docs:
                logger.warning("No processed documents found in context")
                return {
                    "success": False,
                    "error": "No documents to analyze",
                    "query": query,
                }

            # Log the document counts
            await self.log_activity(
                "documents_for_analysis",
                {"doc_count": len(processed_docs), "task_id": task.task_id},
            )

            # 1. Organize documents by source type
            sources_map = self._organize_by_source(processed_docs)

            # 2. Extract vectors for clustering (if available)
            try:
                print(" processed docs")
                print(processed_docs)
                doc_vectors = self._extract_vectors(processed_docs)

                # 3. Cluster documents if we have vectors
                if len(doc_vectors) > 5:  # Only cluster if we have enough documents
                    cluster_results = self._cluster_documents(
                        vectors=doc_vectors, documents=processed_docs, method="kmeans"
                    )

                    # Store cluster information
                    await self.set_context(
                        "cluster_results", self._make_serializable(cluster_results)
                    )
                    await self.log_activity(
                        "document_clustering",
                        {
                            "num_clusters": cluster_results.get("num_clusters", 0),
                            "task_id": task.task_id,
                        },
                    )
                else:
                    cluster_results = None
            except Exception as e:
                logger.error(f"Error during document clustering: {str(e)}")
                cluster_results = None

            # 4. Perform various analyses on the documents
            analysis_results = {}

            # 4.1 Extract key entities
            entities = await self._extract_entities(processed_docs, query)
            analysis_results["entities"] = entities

            # 4.2 Perform topic modeling
            if cluster_results:
                topics = await self._analyze_topics(cluster_results, query)
                analysis_results["topics"] = topics
            else:
                # Fallback topic extraction without clustering
                topics = await self._extract_topics(processed_docs, query)
                analysis_results["topics"] = topics

            # 4.3 Identify main claims and evidence
            claims = await self._extract_claims(processed_docs, query)
            analysis_results["claims"] = claims

            # 4.4 Identify potential contradictions
            contradictions = await self._identify_contradictions(processed_docs, claims)
            analysis_results["contradictions"] = contradictions

            # 4.5 Generate concept map
            concept_map = await self._generate_concept_map(
                entities=entities, topics=topics, claims=claims, query=query
            )
            analysis_results["concept_map"] = concept_map

            # 5. Create summary analysis
            summary = await self._generate_summary_analysis(
                query=query,
                entities=entities,
                topics=topics,
                claims=claims,
                contradictions=contradictions,
                concept_map=concept_map,
                cluster_results=cluster_results,
            )

            # Store analysis in context for other agents
            await self.set_context(
                "analysis_results", self._make_serializable(analysis_results)
            )
            await self.set_context("analysis_summary", summary)

            # If we have cluster results, add to context
            if cluster_results:
                await self.set_context(
                    "document_clusters",
                    self._make_serializable(cluster_results.get("clusters", {})),
                )

            # Add sources information
            sources_info = {}
            for source_type, docs in sources_map.items():
                sources_info[source_type] = len(docs)

            # Log success
            await self.log_activity(
                "analysis_complete",
                {
                    "task_id": task.task_id,
                    "topics": len(topics),
                    "entities": len(entities),
                    "claims": len(claims),
                    "source_types": list(sources_info.keys()),
                },
            )

            # Return results
            return {
                "success": True,
                "query": query,
                "analysis_summary": summary,
                "entities": entities,
                "topics": topics,
                "claims": claims,
                "contradictions": contradictions,
                "concept_map": concept_map,
                "cluster_info": (
                    cluster_results.get("cluster_stats", {}) if cluster_results else {}
                ),
                "sources": sources_info,
                "timestamp": time.time(),
            }

        except Exception as e:
            # Log failure
            logger.error(f"Error in analysis task: {str(e)}")
            await self.log_activity(
                "analysis_failed",
                {"task_id": task.task_id, "error": str(e)},
            )

            # Add error to context
            await self.context_manager.add_error(str(e), "analysis_agent")

            # Re-raise for task queue error handling
            raise

    def _organize_by_source(
        self, documents: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Organize documents by source type.

        Args:
            documents: List of documents

        Returns:
            Dictionary mapping source type to list of documents
        """
        sources_map = {}

        for doc in documents:
            metadata = doc.get("metadata", {})
            source_type = metadata.get("source_type", "unknown")

            if source_type not in sources_map:
                sources_map[source_type] = []

            sources_map[source_type].append(doc)

        return sources_map

    def _extract_vectors(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract vectors from documents if available, or retrieve them from vector storage.

        Args:
            documents: List of documents

        Returns:
            Array of document vectors
        """
        vectors = []

        # First try to get vectors directly from documents if they're already there
        for doc in documents:
            # Look for vector in standard locations
            if "vector" in doc:
                vectors.append(doc["vector"])
            elif "embedding" in doc:
                vectors.append(doc["embedding"])
            elif "metadata" in doc and "vector" in doc["metadata"]:
                vectors.append(doc["metadata"]["vector"])
            elif "metadata" in doc and "embedding" in doc["metadata"]:
                vectors.append(doc["metadata"]["embedding"])

        # If no vectors found, try to retrieve them from the vector storage
        if not vectors:
            try:
                # Initialize the embeddings model
                embeddings = OpenAIEmbeddings(
                    openai_api_key=settings.OPENAI_API_KEY,
                )

                # Extract text content from documents for embedding
                texts = []
                for doc in documents:
                    if "page_content" in doc:
                        texts.append(doc["page_content"])
                    elif "text" in doc:
                        texts.append(doc["text"])
                    elif "metadata" in doc and "text" in doc["metadata"]:
                        texts.append(doc["metadata"]["text"])
                    else:
                        # Use a default representation
                        texts.append(str(doc.get("metadata", {})))

                # Generate embeddings for all documents
                if texts:
                    logger.info(f"Generating embeddings for {len(texts)} documents")
                    vectors = embeddings.embed_documents(texts)

                    # Add embeddings back to documents for future use
                    for i, doc in enumerate(documents):
                        if i < len(vectors):
                            if "metadata" not in doc:
                                doc["metadata"] = {}
                            doc["metadata"]["vector"] = vectors[i]

            except Exception as e:
                logger.error(f"Error retrieving vectors from storage: {str(e)}")

        # If we still have no vectors, generate placeholder vectors for testing
        if not vectors:
            logger.warning(
                "No document vectors found, generating placeholder vectors for testing"
            )

            # Generate random embeddings with a fixed seed for consistent results
            # Use a low-dimensional space (10) for efficiency during testing
            import random

            random.seed(42)  # Fixed seed for reproducibility
            for _ in range(len(documents)):
                # Create a simple 10-dimensional vector
                vector = [random.uniform(-1, 1) for _ in range(10)]
                vectors.append(vector)

            # If there are still no vectors, raise an error
            if not vectors:
                raise ValueError(
                    "No document vectors found and couldn't create placeholders"
                )

        return np.array(vectors)

    def _cluster_documents(
        self,
        vectors: np.ndarray,
        documents: List[Dict[str, Any]],
        method: str = "kmeans",
    ) -> Dict[str, Any]:
        """
        Cluster documents based on their vectors.

        Args:
            vectors: Document vectors
            documents: List of documents
            method: Clustering method to use

        Returns:
            Clustering results
        """
        # Default to a reasonable number of clusters
        # Use square root of document count as a heuristic
        num_clusters = min(10, max(2, int(np.sqrt(len(documents)) / 2)))

        # Apply the selected clustering method
        if method == "kmeans":
            labels, actual_clusters = self.kmeans_clustering.cluster(
                vectors=vectors, k=num_clusters
            )
            cluster_stats = self.kmeans_clustering.get_cluster_statistics(labels)
        elif method == "hdbscan":
            labels, actual_clusters = self.hdbscan_clustering.cluster(vectors)
            cluster_stats = self.hdbscan_clustering.get_cluster_statistics(labels)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Organize documents by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if i >= len(documents):
                continue

            if label not in clusters:
                clusters[int(label)] = []

            # Add cluster label to document metadata
            doc = documents[i].copy()
            if "metadata" not in doc:
                doc["metadata"] = {}
            doc["metadata"]["cluster"] = int(label)

            clusters[int(label)].append(doc)

        # Get top documents for each cluster using MMR
        exemplars = {}
        for label, cluster_docs in clusters.items():
            if len(cluster_docs) > 3:
                # Get document vectors for this cluster
                cluster_indices = np.where(labels == label)[0]
                cluster_vectors = vectors[cluster_indices]

                # Use MMR to select diverse top documents
                mmr = MaximumMarginalRelevance(diversity_weight=0.3)
                selected_indices = mmr.select_diverse_subset(
                    query_vector=np.mean(
                        cluster_vectors, axis=0
                    ),  # Use cluster center as query
                    doc_vectors=cluster_vectors,
                    k=3,  # Get top 3 documents
                )

                # Map back to original indices
                exemplars[label] = [cluster_docs[idx] for idx in selected_indices]
            else:
                # For small clusters, use all documents
                exemplars[label] = cluster_docs

        return {
            "labels": labels.tolist(),
            "num_clusters": actual_clusters,
            "cluster_stats": cluster_stats,
            "clusters": clusters,
            "exemplars": exemplars,
        }

    async def _extract_entities(
        self,
        documents: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract key entities from documents.

        Args:
            documents: List of documents
            query: Research query

        Returns:
            List of entities with metadata
        """
        # Prepare document context (use a subset to avoid token limits)
        doc_sample = documents[:15]  # Limit to 15 documents

        # Format document content
        doc_texts = []
        for i, doc in enumerate(doc_sample):
            content = ""
            metadata = doc.get("metadata", {})

            # Get text content
            if "page_content" in doc:
                content = doc["page_content"]
            elif "text" in metadata:
                content = metadata["text"]
            else:
                content = str(metadata)

            # Include source information
            source_type = metadata.get("source_type", "unknown")
            title = metadata.get("title", "")
            url = metadata.get("url", "")

            # Format as numbered document
            formatted_doc = f"DOCUMENT {i+1} [{source_type.upper()}]"
            if title:
                formatted_doc += f": {title}"
            if url:
                formatted_doc += f"\nURL: {url}"
            formatted_doc += f"\n{content[:500]}..."  # Truncate for brevity

            doc_texts.append(formatted_doc)

        # Join with clear separators
        context = "\n\n---\n\n".join(doc_texts)

        # Create prompt for entity extraction - more explicitly formatted to ensure JSON response
        prompt_template = """You are an expert research analyst tasked with identifying key entities in a set of documents related to a research query.

RESEARCH QUERY: {query}

DOCUMENTS:
{context}

Based on these documents, identify the key entities related to the research query. 
These should include:
- People (researchers, historical figures, experts, etc.)
- Organizations (companies, research institutions, government bodies, etc.)
- Concepts (theories, methods, paradigms, etc.)
- Technologies (tools, systems, platforms, etc.)
- Locations (if relevant)
- Time periods (if relevant)

For each entity:
1. Provide the entity name
2. Categorize the entity type
3. Give a brief description (1-2 sentences)
4. Note which document(s) mention this entity
5. Assign a relevance score (1-10) based on importance to the research query

FORMAT YOUR RESPONSE AS A VALID JSON LIST with the following structure:
[
  {{
    "name": "Entity name",
    "type": "Entity type",
    "description": "Brief description",
    "mentioned_in": [document numbers],
    "relevance": relevance score
  }},
  {{
    "name": "Second entity",
    "type": "Entity type",
    "description": "Brief description",
    "mentioned_in": [document numbers],
    "relevance": relevance score
  }}
]

Identify at least 5-10 entities if possible, focusing on those most relevant to the query.
IMPORTANT: Only include the JSON list in your response, with no other text, explanation, or formatting outside the JSON.
Make sure the output is a properly formatted JSON array that can be parsed by json.loads().
"""

        # Create the LLM chain
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Run the chain with improved error handling and JSON validation
        try:
            response = await chain.arun(query=query, context=context)

            # Clean up the response to ensure valid JSON
            cleaned_response = response.strip()
            # Remove any potential markdown code block formatting
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response.replace("```json", "", 1)
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.replace("```", "", 1)
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]

            cleaned_response = cleaned_response.strip()

            # Parse JSON response
            entities = json.loads(cleaned_response)
            logger.info(f"Extracted {len(entities)} entities")
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            # Return minimal fallback entities
            return [
                {
                    "name": query,
                    "type": "Topic",
                    "description": "The main research query",
                    "mentioned_in": [1],
                    "relevance": 10,
                }
            ]

    async def _analyze_topics(
        self,
        cluster_results: Dict[str, Any],
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Analyze topics based on document clusters.

        Args:
            cluster_results: Results from document clustering
            query: Research query

        Returns:
            List of topics with metadata
        """
        # Extract clusters and exemplars
        clusters = cluster_results.get("clusters", {})
        exemplars = cluster_results.get("exemplars", {})

        if not clusters:
            logger.warning("No clusters found for topic analysis")
            return []

        # Prepare cluster information for analysis
        cluster_texts = []

        for cluster_id, docs in exemplars.items():
            # Format exemplar documents
            doc_texts = []
            for i, doc in enumerate(docs):
                content = ""
                metadata = doc.get("metadata", {})

                # Get text content
                if "page_content" in doc:
                    content = doc["page_content"]
                elif "text" in metadata:
                    content = metadata["text"]
                else:
                    content = str(metadata)

                # Format as exemplar
                doc_texts.append(f"Exemplar {i+1}: {content[:300]}...")

            # Create cluster summary
            cluster_summary = (
                f"CLUSTER {cluster_id} ({len(clusters.get(cluster_id, []))} documents)\n\n"
                f"EXEMPLARS:\n{chr(10).join(doc_texts)}\n"
            )

            cluster_texts.append(cluster_summary)

        # Join cluster texts
        context = "\n\n---\n\n".join(cluster_texts)

        # Create prompt for topic analysis - escape JSON curly braces
        prompt_template = """
        You are an expert research analyst analyzing document clusters to identify topics.
        
        RESEARCH QUERY: {query}
        
        DOCUMENT CLUSTERS:
        {context}
        
        For each cluster, identify the main topic represented by that cluster. The documents in each cluster
        are semantically similar, so they should represent a coherent topic or theme.
        
        For each topic (cluster):
        1. Provide a concise topic name/title
        2. Write a brief description of the topic
        3. List key terms/concepts associated with this topic
        4. Explain how this topic relates to the research query
        5. Assign a relevance score (1-10) based on importance to the research query
        
        FORMAT YOUR RESPONSE AS A JSON LIST with the following format:
        [
            {{
                "cluster_id": cluster number,
                "topic_name": "Topic name",
                "description": "Brief description",
                "key_terms": ["term1", "term2", "term3"...],
                "relation_to_query": "How this relates to the research query",
                "relevance": relevance score
            }},
            ... more topics ...
        ]
        
        Only include the JSON list in your response, with no other text.
        """

        # Create the LLM chain
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Run the chain
        try:
            response = await chain.arun(query=query, context=context)

            # Clean up the response to ensure valid JSON
            cleaned_response = response.strip()
            # Remove any potential markdown code block formatting
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response.replace("```json", "", 1)
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.replace("```", "", 1)
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]

            cleaned_response = cleaned_response.strip()

            # Check if response is empty
            if not cleaned_response:
                logger.warning("Empty response from topic analysis")
                return self._create_default_topic(query)

            # Parse JSON response
            try:
                topics = json.loads(cleaned_response)
                logger.info(f"Analyzed {len(topics)} topics from clusters")
                return topics
            except json.JSONDecodeError as json_err:
                logger.error(
                    f"JSON decode error in topic analysis: {str(json_err)} - Response: {cleaned_response[:100]}..."
                )
                return self._create_default_topic(query)
        except Exception as e:
            logger.error(f"Error analyzing topics: {str(e)}")
            return self._create_default_topic(query)

    def _create_default_topic(self, query: str) -> List[Dict[str, Any]]:
        """Create a default topic when extraction fails."""
        return [
            {
                "cluster_id": 0,
                "topic_name": query,
                "description": "The main research query",
                "key_terms": [query],
                "relation_to_query": "This is the main research topic",
                "relevance": 10,
            }
        ]

    async def _extract_topics(
        self,
        documents: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract topics from documents without clustering.
        Fallback method when clustering is not available.

        Args:
            documents: List of documents
            query: Research query

        Returns:
            List of topics with metadata
        """
        # Prepare document context (use a subset to avoid token limits)
        doc_sample = documents[:15]  # Limit to 15 documents

        # Format document content
        doc_texts = []
        for i, doc in enumerate(doc_sample):
            content = ""
            metadata = doc.get("metadata", {})

            # Get text content
            if "page_content" in doc:
                content = doc["page_content"]
            elif "text" in metadata:
                content = metadata["text"]
            else:
                content = str(metadata)

            # Format as numbered document
            doc_texts.append(f"DOCUMENT {i+1}:\n{content[:300]}...")

        # Join with clear separators
        context = "\n\n---\n\n".join(doc_texts)

        # Create prompt for topic extraction
        prompt_template = """
        You are an expert research analyst tasked with identifying key topics in a set of documents related to a research query.
        
        RESEARCH QUERY: {query}
        
        DOCUMENTS:
        {context}
        
        Based on these documents, identify 5-7 main topics relevant to the research query.
        
        For each topic:
        1. Provide a concise topic name/title
        2. Write a brief description of the topic
        3. List key terms/concepts associated with this topic
        4. Explain how this topic relates to the research query
        5. Assign a relevance score (1-10) based on importance to the research query
        
        FORMAT YOUR RESPONSE AS A JSON LIST with the following format:
        [
            {{
                "topic_name": "Topic name",
                "description": "Brief description",
                "key_terms": ["term1", "term2", "term3"...],
                "relation_to_query": "How this relates to the research query",
                "relevance": relevance score
            }},
            ... more topics ...
        ]
        
        Only include the JSON list in your response, with no other text.
        """

        # Create the LLM chain
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Run the chain
        try:
            response = await chain.arun(query=query, context=context)

            # Clean up the response to ensure valid JSON
            cleaned_response = response.strip()
            # Remove any potential markdown code block formatting
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response.replace("```json", "", 1)
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.replace("```", "", 1)
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]

            cleaned_response = cleaned_response.strip()

            # Check if response is empty
            if not cleaned_response:
                logger.warning("Empty response from topic extraction")
                return self._create_default_topic(query)

            # Parse JSON response
            try:
                topics = json.loads(cleaned_response)
                logger.info(f"Extracted {len(topics)} topics")
                return topics
            except json.JSONDecodeError as json_err:
                logger.error(
                    f"JSON decode error in topic extraction: {str(json_err)} - Response: {cleaned_response[:100]}..."
                )
                return self._create_default_topic(query)
        except Exception as e:
            logger.error(f"Error extracting topics: {str(e)}")
            return self._create_default_topic(query)

    async def _extract_claims(
        self,
        documents: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract main claims and evidence from documents.

        Args:
            documents: List of documents
            query: Research query

        Returns:
            List of claims with supporting evidence
        """
        # Prepare document context (use a subset to avoid token limits)
        doc_sample = documents[:15]  # Limit to 15 documents

        # Format document content
        doc_texts = []
        for i, doc in enumerate(doc_sample):
            content = ""
            metadata = doc.get("metadata", {})

            # Get text content
            if "page_content" in doc:
                content = doc["page_content"]
            elif "text" in metadata:
                content = metadata["text"]
            else:
                content = str(metadata)

            # Include source information
            source_type = metadata.get("source_type", "unknown")
            title = metadata.get("title", "")
            url = metadata.get("url", "")

            # Format as numbered document
            formatted_doc = f"DOCUMENT {i+1} [{source_type.upper()}]"
            if title:
                formatted_doc += f": {title}"
            if url:
                formatted_doc += f"\nURL: {url}"
            formatted_doc += f"\n{content[:500]}..."  # Truncate for brevity

            doc_texts.append(formatted_doc)

        # Join with clear separators
        context = "\n\n---\n\n".join(doc_texts)

        # Create prompt for claim extraction
        prompt_template = """
        You are an expert research analyst tasked with identifying key claims and supporting evidence in a set of documents.
        
        RESEARCH QUERY: {query}
        
        DOCUMENTS:
        {context}
        
        Based on these documents, identify the main claims or assertions relevant to the research query.
        
        For each claim:
        1. State the claim clearly
        2. Note the document(s) where this claim is made or supported
        3. Provide a brief excerpt of supporting evidence from the document(s)
        4. Rate the claim strength (1-10) based on evidence quality and consensus
        5. Note if there are any conflicting perspectives on this claim
        
        FORMAT YOUR RESPONSE AS A JSON LIST with the following format:
        [
            {{
                "claim": "The claim statement",
                "source_documents": [list of document numbers],
                "evidence": "Brief excerpt of supporting evidence",
                "strength": strength rating,
                "conflicting_perspectives": "Description of any conflicting views" OR null if none
            }},
            ... more claims ...
        ]
        
        Identify at least 5-10 significant claims related to the research query.
        Only include the JSON list in your response, with no other text.
        """

        # Create the LLM chain
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Run the chain
        try:
            response = await chain.arun(query=query, context=context)

            # Clean up the response to ensure valid JSON
            cleaned_response = response.strip()
            # Remove any potential markdown code block formatting
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response.replace("```json", "", 1)
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.replace("```", "", 1)
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]

            cleaned_response = cleaned_response.strip()

            # Check if response is empty
            if not cleaned_response:
                logger.warning("Empty response from claim extraction")
                return self._create_default_claims(query)

            # Parse JSON response
            try:
                claims = json.loads(cleaned_response)
                logger.info(f"Extracted {len(claims)} claims")
                return claims
            except json.JSONDecodeError as json_err:
                logger.error(
                    f"JSON decode error in claim extraction: {str(json_err)} - Response: {cleaned_response[:100]}..."
                )
                return self._create_default_claims(query)
        except Exception as e:
            logger.error(f"Error extracting claims: {str(e)}")
            return self._create_default_claims(query)

    def _create_default_claims(self, query: str) -> List[Dict[str, Any]]:
        """Create default claims when extraction fails."""
        return [
            {
                "claim": f"Information related to {query} was found",
                "source_documents": [1],
                "evidence": "General information in the documents",
                "strength": 5,
                "conflicting_perspectives": None,
            }
        ]

    async def _identify_contradictions(
        self,
        documents: List[Dict[str, Any]],
        claims: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Identify potential contradictions in the research materials.

        Args:
            documents: List of documents
            claims: List of claims extracted from documents

        Returns:
            List of contradictions with evidence
        """
        if not claims:
            return []

        # Create prompt to analyze contradictions
        claims_text = json.dumps(claims, indent=2)

        prompt_template = """
        You are an expert research analyst tasked with identifying contradictions and conflicts in research claims.
        
        CLAIMS IDENTIFIED IN DOCUMENTS:
        {claims_text}
        
        Based on these claims, identify any significant contradictions, conflicts, or inconsistencies in the research.
        
        For each contradiction:
        1. Describe the contradiction clearly
        2. Identify the conflicting claims by referencing their content
        3. Assess the significance of this contradiction (1-10)
        4. Suggest possible explanations for the contradiction
        
        FORMAT YOUR RESPONSE AS A JSON LIST with the following format:
        [
            {{
                "contradiction": "Description of the contradiction",
                "conflicting_elements": "The specific claims or statements that conflict",
                "significance": significance rating,
                "possible_explanations": "Possible reasons for this contradiction"
            }},
            ... more contradictions ...
        ]
        
        If no significant contradictions are found, return an empty list.
        Only include the JSON list in your response, with no other text.
        """

        # Create the LLM chain
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Run the chain
        try:
            response = await chain.arun(claims_text=claims_text)

            # Clean up the response to ensure valid JSON
            cleaned_response = response.strip()
            # Remove any potential markdown code block formatting
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response.replace("```json", "", 1)
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.replace("```", "", 1)
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]

            cleaned_response = cleaned_response.strip()

            # Check if response is empty
            if not cleaned_response:
                logger.warning("Empty response from contradiction identification")
                return []

            # Parse JSON response
            try:
                contradictions = json.loads(cleaned_response)
                logger.info(f"Identified {len(contradictions)} contradictions")
                return contradictions
            except json.JSONDecodeError as json_err:
                logger.error(
                    f"JSON decode error in contradiction identification: {str(json_err)} - Response: {cleaned_response[:100]}..."
                )
                return []
        except Exception as e:
            logger.error(f"Error identifying contradictions: {str(e)}")
            return []

    async def _generate_concept_map(
        self,
        entities: List[Dict[str, Any]],
        topics: List[Dict[str, Any]],
        claims: List[Dict[str, Any]],
        query: str,
    ) -> Dict[str, Any]:
        """
        Generate a concept map connecting entities, topics, and claims.

        Args:
            entities: Extracted entities
            topics: Extracted topics
            claims: Extracted claims
            query: Research query

        Returns:
            Concept map as a structured dictionary
        """
        # Create input data for analysis
        entities_text = json.dumps(entities[:10], indent=2)  # Limit to top 10
        topics_text = json.dumps(topics, indent=2)
        claims_text = json.dumps(claims[:10], indent=2)  # Limit to top 10

        prompt_template = """
        You are an expert research analyst tasked with creating a concept map connecting key elements of a research topic.
        
        RESEARCH QUERY: {query}
        
        ENTITIES:
        {entities_text}
        
        TOPICS:
        {topics_text}
        
        CLAIMS:
        {claims_text}
        
        Based on this information, create a concept map that shows relationships between key elements of the research.
        
        The concept map should:
        1. Include the most important entities, topics and claims
        2. Define relationships between elements (uses, supports, contradicts, is part of, etc.)
        3. Create a hierarchical organization with the main research query at the center
        
        FORMAT YOUR RESPONSE AS A JSON OBJECT with the following format:
        {{
            "central_node": {{
                "id": "central",
                "label": "The research query",
                "type": "query"
            }},
            "nodes": [
                {{
                    "id": "unique_id",
                    "label": "Node label",
                    "type": "entity|topic|claim",
                    "importance": 1-10
                }},
                ... more nodes ...
            ],
            "relationships": [
                {{
                    "source": "source_node_id",
                    "target": "target_node_id",
                    "type": "relationship type",
                    "description": "Brief description of relationship"
                }},
                ... more relationships ...
            ]
        }}
        
        Include around 10-15 nodes (beyond the central node) and their relationships.
        Only include the JSON object in your response, with no other text.
        """

        # Create the LLM chain
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Run the chain
        try:
            response = await chain.arun(
                query=query,
                entities_text=entities_text,
                topics_text=topics_text,
                claims_text=claims_text,
            )

            # Clean up the response to ensure valid JSON
            cleaned_response = response.strip()
            # Remove any potential markdown code block formatting
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response.replace("```json", "", 1)
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.replace("```", "", 1)
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]

            cleaned_response = cleaned_response.strip()

            # Check if response is empty
            if not cleaned_response:
                logger.warning("Empty response from concept map generation")
                return self._create_default_concept_map(query)

            # Parse JSON response
            try:
                concept_map = json.loads(cleaned_response)
                logger.info(
                    f"Generated concept map with {len(concept_map.get('nodes', []))} nodes"
                )
                return concept_map
            except json.JSONDecodeError as json_err:
                logger.error(
                    f"JSON decode error in concept map generation: {str(json_err)} - Response: {cleaned_response[:100]}..."
                )
                return self._create_default_concept_map(query)
        except Exception as e:
            logger.error(f"Error generating concept map: {str(e)}")
            return self._create_default_concept_map(query)

    def _create_default_concept_map(self, query: str) -> Dict[str, Any]:
        """Create a default concept map when generation fails."""
        return {
            "central_node": {"id": "central", "label": query, "type": "query"},
            "nodes": [],
            "relationships": [],
        }

    async def _generate_summary_analysis(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        topics: List[Dict[str, Any]],
        claims: List[Dict[str, Any]],
        contradictions: List[Dict[str, Any]],
        concept_map: Dict[str, Any],
        cluster_results: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a summary analysis of all findings.

        Args:
            query: Research query
            entities: Extracted entities
            topics: Extracted topics
            claims: Extracted claims
            contradictions: Identified contradictions
            concept_map: Generated concept map
            cluster_results: Optional clustering results

        Returns:
            Summary analysis text
        """
        # Create input data for summary
        entities_text = json.dumps(entities[:10], indent=2)  # Limit to top 10
        topics_text = json.dumps(topics, indent=2)
        claims_text = json.dumps(claims[:10], indent=2)  # Limit to top 10
        contradictions_text = json.dumps(contradictions, indent=2)

        # Cluster information
        cluster_info = "No clustering information available."
        if cluster_results:
            cluster_stats = cluster_results.get("cluster_stats", {})
            num_clusters = cluster_results.get("num_clusters", 0)
            cluster_info = (
                f"Documents were organized into {num_clusters} thematic clusters."
            )

            if "cluster_sizes" in cluster_stats:
                sizes = cluster_stats["cluster_sizes"]
                cluster_info += f" Cluster sizes: {sizes}"

        prompt_template = """
        You are an expert research analyst tasked with creating a comprehensive summary of research findings.
        
        RESEARCH QUERY: {query}
        
        CLUSTERING INFO:
        {cluster_info}
        
        ENTITIES:
        {entities_text}
        
        TOPICS:
        {topics_text}
        
        CLAIMS:
        {claims_text}
        
        CONTRADICTIONS:
        {contradictions_text}
        
        Based on this analysis, create a comprehensive summary of the research findings. Your summary should:
        
        1. Provide an overview of the main topics and themes found
        2. Highlight the most significant entities and their relationships
        3. Summarize the key claims and evidence
        4. Address any contradictions or areas of debate
        5. Suggest gaps in the current research or areas for further investigation
        6. Provide an overall assessment of the research landscape on this topic
        
        Format your response as Markdown with clear sections and bullet points where appropriate.
        """

        # Create the LLM chain
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Run the chain
        try:
            response = await chain.arun(
                query=query,
                cluster_info=cluster_info,
                entities_text=entities_text,
                topics_text=topics_text,
                claims_text=claims_text,
                contradictions_text=contradictions_text,
            )
            logger.info(f"Generated summary analysis for query: {query}")
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating summary analysis: {str(e)}")
            # Return minimal fallback
            return f"# Analysis Summary for: {query}\n\nInformation was collected and analyzed, but a detailed summary could not be generated."
