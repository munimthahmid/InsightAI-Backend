"""
Clustering modules for enhancing vector retrieval with document grouping.

This package provides different clustering algorithms for semantic similarity
clustering of document vectors, improving retrieval quality and diversity.
"""

__all__ = ["kmeans", "hdbscan_cluster", "mmr"]

# Re-export main classes for easier imports
from app.services.vector_db.clustering.kmeans import KMeansClustering
from app.services.vector_db.clustering.hdbscan_cluster import HDBSCANClustering
from app.services.vector_db.clustering.mmr import MaximumMarginalRelevance
