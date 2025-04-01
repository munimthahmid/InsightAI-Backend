"""
Maximum Marginal Relevance implementation for diverse document selection.

This module provides Maximum Marginal Relevance (MMR) functionality to
select a diverse subset of documents from a larger set, balancing relevance
to a query with diversity among the selected documents.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from loguru import logger


class MaximumMarginalRelevance:
    """
    Maximum Marginal Relevance (MMR) implementation for diverse document selection.

    MMR balances relevance to the query with diversity in the result set.
    The algorithm iteratively selects documents that are relevant to the query
    but also different from already-selected documents.
    """

    def __init__(self, diversity_weight: float = 0.3, use_cosine: bool = True):
        """
        Initialize MaximumMarginalRelevance.

        Args:
            diversity_weight: Weight for diversity (lambda in MMR formula)
                              0.0 = maximum relevance, 1.0 = maximum diversity
            use_cosine: Whether to use cosine similarity (True) or dot product (False)
        """
        if not 0 <= diversity_weight <= 1:
            raise ValueError("diversity_weight must be between 0 and 1")

        self.diversity_weight = diversity_weight
        self.use_cosine = use_cosine

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors to unit length for cosine similarity.

        Args:
            vectors: Document vectors

        Returns:
            Normalized vectors
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Handle zero-norm vectors
        norms[norms == 0] = 1.0
        return vectors / norms

    def _similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score (cosine similarity or dot product)
        """
        if self.use_cosine:
            # Cosine similarity for normalized vectors is just the dot product
            return float(np.dot(vec1, vec2))
        else:
            # Dot product for unnormalized vectors
            return float(np.dot(vec1, vec2))

    def select_diverse_subset(
        self,
        query_vector: np.ndarray,
        doc_vectors: np.ndarray,
        doc_scores: Optional[np.ndarray] = None,
        k: int = 10,
    ) -> List[int]:
        """
        Select a diverse subset of documents using MMR.

        Args:
            query_vector: Query vector
            doc_vectors: Document vectors
            doc_scores: Optional relevance scores for documents (if None, computed from vectors)
            k: Number of documents to select

        Returns:
            List of indices of selected documents
        """
        if len(doc_vectors) == 0:
            return []

        # Cap k at the number of available documents
        k = min(k, len(doc_vectors))

        # Normalize vectors if using cosine similarity
        if self.use_cosine:
            query_vector = query_vector / np.linalg.norm(query_vector)
            doc_vectors = self._normalize_vectors(doc_vectors)

        # If scores not provided, compute relevance scores as similarity to query
        if doc_scores is None:
            doc_scores = np.array(
                [self._similarity(query_vector, doc_vec) for doc_vec in doc_vectors]
            )

        # Track which docs are selected
        selected_indices = []
        remaining_indices = list(range(len(doc_vectors)))

        # First, select the most relevant document
        if remaining_indices:
            best_idx = remaining_indices[np.argmax(doc_scores[remaining_indices])]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        # Then iteratively select documents using MMR
        while len(selected_indices) < k and remaining_indices:
            mmr_scores = []

            for idx in remaining_indices:
                # Relevance component
                relevance = doc_scores[idx]

                # Diversity component (max similarity to any selected document)
                max_similarity = 0
                if selected_indices:  # If we have selected documents
                    similarities = [
                        self._similarity(doc_vectors[idx], doc_vectors[sel_idx])
                        for sel_idx in selected_indices
                    ]
                    max_similarity = max(similarities)

                # MMR score: (1-λ)×relevance - λ×max_similarity
                mmr_score = (
                    1 - self.diversity_weight
                ) * relevance - self.diversity_weight * max_similarity
                mmr_scores.append(mmr_score)

            # Select document with highest MMR score
            best_mmr_idx = remaining_indices[np.argmax(mmr_scores)]
            selected_indices.append(best_mmr_idx)
            remaining_indices.remove(best_mmr_idx)

        logger.info(
            f"Selected {len(selected_indices)} diverse documents using MMR (diversity_weight={self.diversity_weight})"
        )
        return selected_indices

    def select_diverse_documents(
        self,
        query_vector: np.ndarray,
        documents: List[Dict[str, Any]],
        doc_vectors: np.ndarray,
        doc_scores: Optional[np.ndarray] = None,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Select a diverse subset of document objects using MMR.

        Args:
            query_vector: Query vector
            documents: List of document dictionaries
            doc_vectors: Document vectors (same order as documents)
            doc_scores: Optional relevance scores (if None, computed from vectors)
            k: Number of documents to select

        Returns:
            List of selected document dictionaries
        """
        if len(documents) != len(doc_vectors):
            raise ValueError("Number of documents must match number of vectors")

        # Get diverse indices
        selected_indices = self.select_diverse_subset(
            query_vector=query_vector,
            doc_vectors=doc_vectors,
            doc_scores=doc_scores,
            k=k,
        )

        # Return selected documents
        return [documents[idx] for idx in selected_indices]

    def rerank_by_clusters_then_mmr(
        self,
        query_vector: np.ndarray,
        documents: List[Dict[str, Any]],
        doc_vectors: np.ndarray,
        cluster_labels: np.ndarray,
        doc_scores: Optional[np.ndarray] = None,
        k: int = 10,
        docs_per_cluster: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents by first selecting from each cluster, then using MMR.

        This approach ensures representation from each cluster, then uses MMR
        for diversity within each cluster's selection.

        Args:
            query_vector: Query vector
            documents: List of document dictionaries
            doc_vectors: Document vectors
            cluster_labels: Cluster label for each document
            doc_scores: Optional relevance scores
            k: Total number of documents to select
            docs_per_cluster: Number of docs to select per cluster (if None, computed based on cluster sizes)

        Returns:
            List of selected document dictionaries
        """
        if len(documents) != len(doc_vectors) or len(doc_vectors) != len(
            cluster_labels
        ):
            raise ValueError("Number of documents, vectors, and labels must match")

        # If no documents, return empty list
        if len(documents) == 0:
            return []

        # Get unique clusters
        unique_clusters = np.unique(cluster_labels)
        num_clusters = len(unique_clusters)

        # Skip noise points (label = -1) if needed
        if -1 in unique_clusters and num_clusters > 1:
            valid_clusters = [c for c in unique_clusters if c != -1]
        else:
            valid_clusters = unique_clusters

        # If docs_per_cluster not specified, distribute based on cluster sizes
        if docs_per_cluster is None:
            # Calculate how many documents to select from each cluster
            cluster_sizes = {c: np.sum(cluster_labels == c) for c in valid_clusters}
            total_docs = sum(cluster_sizes.values())

            # Distribute based on cluster sizes, with at least 1 per cluster
            docs_per_cluster = {
                c: max(1, min(int(k * size / total_docs), size))
                for c, size in cluster_sizes.items()
            }

            # Adjust if we've allocated more or less than k
            allocated = sum(docs_per_cluster.values())
            if allocated < k:
                # Prioritize larger clusters for extra docs
                sorted_clusters = sorted(
                    [(c, size) for c, size in cluster_sizes.items()],
                    key=lambda x: x[1],
                    reverse=True,
                )
                for c, _ in sorted_clusters:
                    if allocated >= k:
                        break
                    if docs_per_cluster[c] < cluster_sizes[c]:
                        docs_per_cluster[c] += 1
                        allocated += 1
            elif allocated > k:
                # Remove from smallest clusters first
                sorted_clusters = sorted(
                    [(c, size) for c, size in cluster_sizes.items()], key=lambda x: x[1]
                )
                for c, _ in sorted_clusters:
                    if allocated <= k:
                        break
                    if docs_per_cluster[c] > 1:
                        docs_per_cluster[c] -= 1
                        allocated -= 1
        else:
            # Use the same value for all clusters
            docs_per_cluster = {
                c: min(docs_per_cluster, np.sum(cluster_labels == c))
                for c in valid_clusters
            }

        # Select documents from each cluster using MMR
        selected_documents = []

        for cluster in valid_clusters:
            # Get documents in this cluster
            cluster_indices = np.where(cluster_labels == cluster)[0]
            cluster_doc_vectors = doc_vectors[cluster_indices]

            # Get scores for this cluster if available
            cluster_scores = (
                doc_scores[cluster_indices] if doc_scores is not None else None
            )

            # Apply MMR to this cluster
            num_to_select = (
                docs_per_cluster[cluster]
                if isinstance(docs_per_cluster, dict)
                else docs_per_cluster
            )
            selected_cluster_indices = self.select_diverse_subset(
                query_vector=query_vector,
                doc_vectors=cluster_doc_vectors,
                doc_scores=cluster_scores,
                k=num_to_select,
            )

            # Map back to original indices and add to selection
            for idx in selected_cluster_indices:
                selected_documents.append(documents[cluster_indices[idx]])

        # If we need more documents to reach k (e.g., due to rounding)
        if len(selected_documents) < k and -1 in unique_clusters:
            # Fill with noise points if available
            noise_indices = np.where(cluster_labels == -1)[0]
            noise_doc_vectors = doc_vectors[noise_indices]
            noise_scores = doc_scores[noise_indices] if doc_scores is not None else None

            remaining_to_select = k - len(selected_documents)
            selected_noise_indices = self.select_diverse_subset(
                query_vector=query_vector,
                doc_vectors=noise_doc_vectors,
                doc_scores=noise_scores,
                k=remaining_to_select,
            )

            # Add selected noise points
            for idx in selected_noise_indices:
                selected_documents.append(documents[noise_indices[idx]])

        logger.info(
            f"Selected {len(selected_documents)} documents using cluster-based MMR from {num_clusters} clusters"
        )
        return selected_documents
