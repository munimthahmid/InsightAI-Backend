"""
K-means clustering implementation for document vectors.

This module provides K-means clustering functionality for semantic grouping
of document vectors, enabling topic-based organization of retrieval results.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from loguru import logger


class KMeansClustering:
    """
    K-means clustering for document embeddings.

    Provides methods to cluster document vectors and extract
    representative documents from each cluster.
    """

    def __init__(
        self,
        min_clusters: int = 2,
        max_clusters: int = 10,
        auto_tune: bool = True,
        random_state: int = 42,
    ):
        """
        Initialize KMeansClustering.

        Args:
            min_clusters: Minimum number of clusters to consider
            max_clusters: Maximum number of clusters to consider
            auto_tune: Whether to automatically determine optimal cluster count
            random_state: Random seed for reproducibility
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.auto_tune = auto_tune
        self.random_state = random_state
        self.model = None
        self.optimal_k = None

    def cluster(
        self, vectors: np.ndarray, k: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Cluster document vectors using K-means.

        Args:
            vectors: Document vectors as a numpy array
            k: Number of clusters (if None, auto-determined based on silhouette score)

        Returns:
            Tuple of (cluster labels, number of clusters used)
        """
        if vectors.shape[0] <= 1:
            # Can't cluster a single document
            return np.zeros(vectors.shape[0], dtype=int), 1

        # Determine number of clusters
        if k is None and self.auto_tune:
            k = self._determine_optimal_clusters(vectors)
        elif k is None:
            # Default to min_clusters if not provided and auto_tune is False
            k = min(self.min_clusters, vectors.shape[0])
        else:
            # Ensure k doesn't exceed number of documents
            k = min(k, vectors.shape[0])

        # Fit model
        self.model = KMeans(n_clusters=k, random_state=self.random_state)
        labels = self.model.fit_predict(vectors)
        self.optimal_k = k

        logger.info(f"Clustered {vectors.shape[0]} documents into {k} clusters")
        return labels, k

    def _determine_optimal_clusters(self, vectors: np.ndarray) -> int:
        """
        Determine optimal number of clusters using silhouette score.

        Args:
            vectors: Document vectors as a numpy array

        Returns:
            Optimal number of clusters
        """
        n_samples = vectors.shape[0]

        # Adjust min/max clusters based on number of samples
        min_k = min(self.min_clusters, n_samples - 1)
        max_k = min(self.max_clusters, n_samples - 1)

        # Need at least 2 samples and min_k >= 2 for clustering
        if n_samples < 4 or min_k < 2:
            return min(2, n_samples)

        best_score = -1
        best_k = min_k

        # Try different cluster counts and evaluate
        for k in range(min_k, max_k + 1):
            # Skip if we don't have enough samples for k clusters
            if n_samples <= k:
                continue

            # Fit model and score
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            labels = kmeans.fit_predict(vectors)

            # Silhouette score requires at least 2 clusters with >1 sample each
            if len(np.unique(labels)) < 2:
                continue

            try:
                score = silhouette_score(vectors, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except ValueError as e:
                # Handle case where clustering failed
                logger.warning(f"Silhouette score calculation failed for k={k}: {e}")
                continue

        logger.info(
            f"Determined optimal number of clusters: {best_k} with score {best_score:.4f}"
        )
        return best_k

    def get_cluster_centers(self) -> np.ndarray:
        """
        Get the cluster centers.

        Returns:
            Numpy array of cluster centers
        """
        if self.model is None:
            raise ValueError("Model has not been fit. Call cluster() first.")
        return self.model.cluster_centers_

    def get_nearest_to_centers(
        self, vectors: np.ndarray, labels: np.ndarray
    ) -> List[int]:
        """
        Find the indices of documents nearest to each cluster center.

        Args:
            vectors: Document vectors
            labels: Cluster labels for each document

        Returns:
            List of document indices (one per cluster, nearest to center)
        """
        if self.model is None:
            raise ValueError("Model has not been fit. Call cluster() first.")

        centers = self.model.cluster_centers_
        nearest_indices = []

        for i in range(len(centers)):
            # Get documents in this cluster
            cluster_vectors = vectors[labels == i]
            cluster_indices = np.where(labels == i)[0]

            if len(cluster_indices) == 0:
                continue

            # Find document closest to center
            distances = np.linalg.norm(cluster_vectors - centers[i], axis=1)
            nearest_in_cluster = cluster_indices[np.argmin(distances)]
            nearest_indices.append(nearest_in_cluster)

        return nearest_indices

    def organize_documents_by_cluster(
        self, documents: List[Dict[str, Any]], labels: np.ndarray
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Organize documents into clusters based on labels.

        Args:
            documents: List of document dictionaries
            labels: Cluster labels for each document

        Returns:
            Dictionary mapping cluster IDs to lists of documents
        """
        if len(documents) != len(labels):
            raise ValueError("Number of documents must match number of labels")

        clusters = {}
        for i, doc in enumerate(documents):
            label = int(labels[i])
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(doc)

        return clusters

    def get_cluster_statistics(self, labels: np.ndarray) -> Dict[str, Any]:
        """
        Calculate statistics about the clustering.

        Args:
            labels: Cluster labels

        Returns:
            Dictionary with statistics (counts per cluster, etc.)
        """
        unique_labels, counts = np.unique(labels, return_counts=True)

        stats = {
            "num_clusters": len(unique_labels),
            "cluster_sizes": {
                int(label): int(count) for label, count in zip(unique_labels, counts)
            },
            "min_cluster_size": int(min(counts)),
            "max_cluster_size": int(max(counts)),
            "avg_cluster_size": float(np.mean(counts)),
        }

        return stats
