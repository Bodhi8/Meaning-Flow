"""
MeaningFlow: Semantic Market Modeling Framework

A framework for understanding how meaning propagates through high-dimensional 
systems and how that structure translates into visibility, demand, and outcomes.
"""

__version__ = "0.1.0"

from .embeddings import SemanticEmbedder
from .graph import SemanticGraph
from .coverage import CoverageAnalyzer
from .demand import DemandMapper

__all__ = [
    "SemanticEmbedder",
    "SemanticGraph", 
    "CoverageAnalyzer",
    "DemandMapper",
    "__version__",
]
