"""
Prompt engineering for clustered document results.

This module provides functionality to create specialized prompts based on
document clusters, enhancing the quality and diversity of research results.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger


class ClusterPromptGenerator:
    """
    Generates specialized prompts based on document clusters.

    This class creates prompts that focus on specific themes or topics
    discovered through document clustering, allowing for more targeted
    and comprehensive research.
    """

    def __init__(self, max_docs_per_cluster: int = 3, max_prompt_length: int = 4000):
        """
        Initialize the cluster prompt generator.

        Args:
            max_docs_per_cluster: Maximum number of documents to include per cluster
            max_prompt_length: Maximum length of generated prompts in characters
        """
        self.max_docs_per_cluster = max_docs_per_cluster
        self.max_prompt_length = max_prompt_length

    def _get_cluster_summary(self, cluster_docs: List[Dict[str, Any]]) -> str:
        """
        Create a summary of documents in a cluster.

        Args:
            cluster_docs: List of documents in the cluster

        Returns:
            Summary text for the cluster
        """
        if not cluster_docs:
            return "No documents in cluster"

        # Select top documents based on score
        top_docs = sorted(cluster_docs, key=lambda x: x.get("score", 0), reverse=True)[
            : self.max_docs_per_cluster
        ]

        # Extract common themes from document titles/content
        titles = []
        for doc in top_docs:
            metadata = doc.get("metadata", {})
            title = metadata.get("title", "")
            if title:
                titles.append(title)

        # Create a summary based on document content
        summary = f"Cluster with {len(cluster_docs)} documents"
        if titles:
            summary += f" including: {', '.join(titles[:3])}"
            if len(titles) > 3:
                summary += f" and {len(titles) - 3} more"

        return summary

    def create_cluster_specific_prompt(
        self,
        base_query: str,
        cluster_docs: List[Dict[str, Any]],
        cluster_id: int,
        cluster_stats: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a prompt focused on a specific cluster's theme.

        Args:
            base_query: Original research query
            cluster_docs: Documents in this cluster
            cluster_id: Identifier for the cluster
            cluster_stats: Optional statistics about the cluster

        Returns:
            Prompt text focusing on this cluster's theme
        """
        if not cluster_docs:
            return f"Research the query: {base_query}"

        # Get top documents
        top_docs = sorted(cluster_docs, key=lambda x: x.get("score", 0), reverse=True)[
            : self.max_docs_per_cluster
        ]

        # Extract text excerpts
        excerpts = []
        for doc in top_docs:
            metadata = doc.get("metadata", {})
            text = metadata.get("text", "")
            title = metadata.get("title", "")
            url = metadata.get("url", "")

            # Format excerpt
            excerpt = f"Document: {title}\n" if title else "Document:\n"
            excerpt += f"URL: {url}\n" if url else ""
            excerpt += (
                f"Excerpt: {text[:500]}..." if len(text) > 500 else f"Excerpt: {text}"
            )
            excerpts.append(excerpt)

        # Create the prompt
        prompt = f"""Research Query: {base_query}

Focus on the following theme from Cluster {cluster_id}:
{self._get_cluster_summary(cluster_docs)}

Use the following relevant sources:

"""

        for i, excerpt in enumerate(excerpts, 1):
            prompt += f"Source {i}:\n{excerpt}\n\n"

        prompt += f"""
Based on these sources from Cluster {cluster_id}, analyze and extract key information related to the research query.
Identify important facts, concepts, and insights specific to this thematic cluster.
"""

        # Truncate if too long
        if len(prompt) > self.max_prompt_length:
            prompt = (
                prompt[: self.max_prompt_length - 100]
                + "...\n[Prompt truncated due to length]"
            )

        return prompt

    def create_multi_cluster_synthesis_prompt(
        self,
        base_query: str,
        cluster_summaries: List[Dict[str, Any]],
        exemplar_docs: List[Dict[str, Any]],
    ) -> str:
        """
        Create a prompt for synthesizing information across multiple clusters.

        Args:
            base_query: Original research query
            cluster_summaries: Summaries of the clusters
            exemplar_docs: Representative documents from each cluster

        Returns:
            Prompt for synthesizing across clusters
        """
        # Create the prompt
        prompt = f"""Research Synthesis for Query: {base_query}

The research has identified {len(cluster_summaries)} main thematic clusters:

"""

        # Add cluster summaries
        for i, summary in enumerate(cluster_summaries, 1):
            cluster_id = summary.get("cluster_id", i)
            description = summary.get("description", f"Cluster {cluster_id}")
            doc_count = summary.get("doc_count", "unknown")

            prompt += f"Cluster {cluster_id}: {description} ({doc_count} documents)\n"

        prompt += "\nKey exemplar documents from each cluster:\n\n"

        # Add exemplar documents
        for i, doc in enumerate(exemplar_docs, 1):
            metadata = doc.get("metadata", {})
            text = metadata.get("text", "")
            title = metadata.get("title", "")
            url = metadata.get("url", "")
            cluster = doc.get("cluster", i)

            prompt += f"Exemplar {i} (Cluster {cluster}):\n"
            prompt += f"Title: {title}\n" if title else ""
            prompt += f"URL: {url}\n" if url else ""
            prompt += (
                f"Excerpt: {text[:300]}...\n\n"
                if len(text) > 300
                else f"Excerpt: {text}\n\n"
            )

        prompt += f"""
Synthesize the information from these different thematic clusters to create a comprehensive answer to the research query.
- Identify how the different themes relate to each other
- Note similarities and differences between the perspectives in different clusters
- Consider how the different themes collectively address the research query
- Highlight key insights from each cluster
"""

        # Truncate if too long
        if len(prompt) > self.max_prompt_length:
            prompt = (
                prompt[: self.max_prompt_length - 100]
                + "...\n[Prompt truncated due to length]"
            )

        return prompt

    def create_cluster_based_prompts(
        self,
        base_query: str,
        clustered_docs: Dict[int, List[Dict[str, Any]]],
        cluster_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Create a complete set of prompts for cluster-based research.

        Args:
            base_query: Original research query
            clustered_docs: Documents organized by cluster
            cluster_stats: Optional statistics about the clusters

        Returns:
            Dictionary mapping prompt types to prompt texts:
            - "cluster_{id}": Prompt for each individual cluster
            - "synthesis": Prompt for synthesizing across clusters
        """
        prompts = {}

        # Skip if no clusters
        if not clustered_docs:
            prompts["default"] = f"Research the query: {base_query}"
            return prompts

        # Create cluster summary for synthesis
        cluster_summaries = []
        exemplar_docs = []

        # Create individual cluster prompts
        for cluster_id, docs in clustered_docs.items():
            # Create prompt for this cluster
            prompt_key = f"cluster_{cluster_id}"
            prompts[prompt_key] = self.create_cluster_specific_prompt(
                base_query=base_query,
                cluster_docs=docs,
                cluster_id=cluster_id,
                cluster_stats=cluster_stats,
            )

            # Add to summaries for synthesis
            if docs:
                summary = {
                    "cluster_id": cluster_id,
                    "description": self._get_cluster_summary(docs),
                    "doc_count": len(docs),
                }
                cluster_summaries.append(summary)

                # Get an exemplar document
                top_doc = sorted(docs, key=lambda x: x.get("score", 0), reverse=True)[0]
                top_doc["cluster"] = cluster_id
                exemplar_docs.append(top_doc)

        # Create synthesis prompt
        prompts["synthesis"] = self.create_multi_cluster_synthesis_prompt(
            base_query=base_query,
            cluster_summaries=cluster_summaries,
            exemplar_docs=exemplar_docs[:5],  # Limit to top 5 exemplars
        )

        return prompts
