"""
MeaningFlow: Semantic Content Modeling and Coverage Gap Analysis

A framework for measuring how thoroughly your content covers the topics
your audience is searching for, and finding where the gaps are.

Quickstart:
    >>> from meaningflow import SemanticGraph
    >>> demand = SemanticGraph(texts=queries, embedder="all-MiniLM-L6-v2")
    >>> demand.fit()
    >>> supply = SemanticGraph(texts=content, embedder="all-MiniLM-L6-v2")
    >>> supply.fit()
    >>> gaps = demand.coverage_gaps(reference=supply)
"""

__version__ = "0.2.0"

from .core import SemanticGraph
from .models import Cluster, GapCluster

__all__ = [
    "SemanticGraph",
    "Cluster",
    "GapCluster",
    "__version__",
]
