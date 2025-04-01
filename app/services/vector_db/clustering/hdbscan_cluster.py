"""
HDBSCAN clustering implementation for document vectors.

This module provides density-based clustering for document vectors,
which can identify clusters of varying shapes without requiring a
preset number of clusters.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import sklearn.cluster as sklearn_cluster

try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
from loguru import logger


class HDBSCANClustering:
    """
    HDBSCAN clustering for document embeddings.

    Provides methods to cluster document vectors using density-based clustering,
    which can identify noise points and clusters of arbitrary shapes.
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        metric: str = "euclidean",
        fallback_to_kmeans: bool = True,
    ):
        """
        Initialize HDBSCANClustering.

        Args:
            min_cluster_size: Minimum size of clusters
            min_samples: Minimum number of samples in neighborhood for core points
            metric: Distance metric to use
            fallback_to_kmeans: Whether to fall back to K-means if HDBSCAN is not available
        """
        self.min_cluster_size = min_cluster_size
        # Default min_samples to min_cluster_size if not specified
        self.min_samples = min_samples if min_samples is not None else min_cluster_size
        self.metric = metric
        self.fallback_to_kmeans = fallback_to_kmeans
        self.model = None

    def cluster(self, vectors: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Cluster document vectors using HDBSCAN.

        Args:
            vectors: Document vectors as a numpy array

        Returns:
            Tuple of (cluster labels, number of clusters found)
        """
        if vectors.shape[0] <= 1:
            # Can't cluster a single document
            return np.zeros(vectors.shape[0], dtype=int), 1

        # Adjust parameters based on number of samples
        adjusted_min_cluster_size = min(
            self.min_cluster_size, max(2, vectors.shape[0] // 5)
        )
        adjusted_min_samples = min(self.min_samples, adjusted_min_cluster_size)

        # Try HDBSCAN if available
        if HDBSCAN_AVAILABLE:
            try:
                logger.info(
                    f"Using HDBSCAN with min_cluster_size={adjusted_min_cluster_size}, min_samples={adjusted_min_samples}"
                )
                self.model = hdbscan.HDBSCAN(
                    min_cluster_size=adjusted_min_cluster_size,
                    min_samples=adjusted_min_samples,
                    metric=self.metric,
                    gen_min_span_tree=True,
                )
                labels = self.model.fit_predict(vectors)
                num_clusters = len(np.unique(labels[labels >= 0]))
                logger.info(
                    f"HDBSCAN identified {num_clusters} clusters and {np.sum(labels == -1)} noise points"
                )
                return labels, num_clusters
            except Exception as e:
                logger.error(f"HDBSCAN clustering failed: {e}")
                if not self.fallback_to_kmeans:
                    raise
        elif not self.fallback_to_kmeans:
            raise ImportError(
                "HDBSCAN is not available. Install with 'pip install hdbscan'."
            )

        # Fallback to KMeans if HDBSCAN is unavailable or failed
        logger.info("Falling back to KMeans clustering")
        # Estimate a reasonable number of clusters
        k = min(10, max(2, vectors.shape[0] // 5))
        kmeans = sklearn_cluster.KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(vectors)
        self.model = kmeans
        logger.info(f"KMeans fallback identified {k} clusters")
        return labels, k

    def get_exemplars(self, vectors: np.ndarray, labels: np.ndarray) -> List[int]:
        """
        Get exemplar documents for each cluster.

        Args:
            vectors: Document vectors
            labels: Cluster labels for each document

        Returns:
            List of exemplar document indices, one per cluster
        """
        if self.model is None:
            raise ValueError("Model has not been fit. Call cluster() first.")

        exemplars = []

        # For HDBSCAN, use exemplars if available
        if HDBSCAN_AVAILABLE and isinstance(self.model, hdbscan.HDBSCAN):
            if hasattr(self.model, "exemplars_") and self.model.exemplars_:
                # Find closest point to each exemplar
                for exemplar in self.model.exemplars_:
                    # Calculate distance to exemplar for all points
                    distances = np.linalg.norm(vectors - exemplar, axis=1)
                    # Find index of closest point
                    exemplars.append(int(np.argmin(distances)))
                return exemplars

        # Otherwise, find points closest to cluster centers (for each cluster)
        unique_labels = np.unique(labels)
        for label in unique_labels:
            # Skip noise points (label = -1)
            if label == -1:
                continue

            # Get documents in this cluster
            cluster_vectors = vectors[labels == label]
            cluster_indices = np.where(labels == label)[0]

            if len(cluster_indices) == 0:
                continue

            # Find centroid of this cluster
            centroid = np.mean(cluster_vectors, axis=0)

            # Find document closest to centroid
            distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
            nearest_in_cluster = cluster_indices[np.argmin(distances)]
            exemplars.append(int(nearest_in_cluster))

        return exemplars

    def organize_documents_by_cluster(
        self,
        documents: List[Dict[str, Any]],
        labels: np.ndarray,
        include_noise: bool = True,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Organize documents into clusters based on labels.

        Args:
            documents: List of document dictionaries
            labels: Cluster labels for each document
            include_noise: Whether to include noise points (label = -1)

        Returns:
            Dictionary mapping cluster IDs to lists of documents
        """
        if len(documents) != len(labels):
            raise ValueError("Number of documents must match number of labels")

        clusters = {}
        for i, doc in enumerate(documents):
            label = int(labels[i])
            # Skip noise points if not including them
            if label == -1 and not include_noise:
                continue
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
        unique_labels = np.unique(labels)
        noise_count = np.sum(labels == -1)
        cluster_counts = {
            int(label): int(np.sum(labels == label))
            for label in unique_labels
            if label != -1
        }

        # Skip noise points (-1) in cluster size calculations
        cluster_sizes = [count for label, count in cluster_counts.items()]

        stats = {
            "num_clusters": len(unique_labels) - (1 if -1 in unique_labels else 0),
            "noise_points": int(noise_count),
            "noise_percentage": (
                float(noise_count / len(labels)) if len(labels) > 0 else 0.0
            ),
            "cluster_sizes": cluster_counts,
        }

        if cluster_sizes:
            stats.update(
                {
                    "min_cluster_size": int(min(cluster_sizes)),
                    "max_cluster_size": int(max(cluster_sizes)),
                    "avg_cluster_size": float(np.mean(cluster_sizes)),
                }
            )

        return stats
