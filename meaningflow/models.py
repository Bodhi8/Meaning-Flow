"""
Data models for MeaningFlow.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Cluster:
    """
    A single topic cluster discovered by the SemanticGraph pipeline.

    Attributes
    ----------
    id : int
        Cluster label assigned by HDBSCAN.
    size : int
        Number of texts assigned to this cluster.
    top_terms : list[str]
        Most representative texts, ranked by proximity to the cluster centroid.
    centroid : np.ndarray
        Mean embedding vector of all cluster members.
    texts : list[str]
        All texts assigned to this cluster.
    volume : int
        Total search volume if volume data was provided; otherwise equals size.
    """

    id: int
    size: int
    top_terms: List[str]
    centroid: np.ndarray
    texts: List[str]
    volume: int = 0

    def __repr__(self) -> str:
        terms_preview = ", ".join(self.top_terms[:3])
        return f"Cluster(id={self.id}, size={self.size}, top=[{terms_preview}])"


@dataclass
class GapCluster:
    """
    A demand cluster with insufficient coverage in the supply corpus.

    Attributes
    ----------
    id : int
        Cluster label from the demand graph.
    size : int
        Number of queries in this demand cluster.
    top_terms : list[str]
        Most representative queries from this cluster.
    volume : int
        Total search volume of queries in this cluster.
    centroid : np.ndarray
        Mean embedding of cluster members.
    nearest_supply : str
        Label or top term of the closest supply cluster.
    nearest_similarity : float
        Cosine similarity to the nearest supply cluster centroid.
    """

    id: int
    size: int
    top_terms: List[str]
    volume: int
    centroid: np.ndarray
    nearest_supply: str = ""
    nearest_similarity: float = 0.0

    def __repr__(self) -> str:
        terms_preview = ", ".join(self.top_terms[:3])
        return (
            f"GapCluster(id={self.id}, size={self.size}, volume={self.volume}, "
            f"top=[{terms_preview}], nearest_sim={self.nearest_similarity:.3f})"
        )
