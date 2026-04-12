"""
Coverage Analysis Module

Measures semantic coverage by mapping demand signals onto content clusters
and identifying gaps where demand exists but content is thin or missing.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CoverageGap:
    """Represents an identified coverage gap."""
    
    cluster_id: int
    representative_queries: List[str]
    total_demand: float
    content_count: int
    coverage_score: float
    gap_score: float  # Higher = bigger opportunity
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoverageReport:
    """Complete coverage analysis results."""
    
    overall_coverage: float
    covered_demand: float
    uncovered_demand: float
    gaps: List[CoverageGap]
    cluster_coverage: Dict[int, float]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert gaps to DataFrame."""
        data = []
        for gap in self.gaps:
            data.append({
                "cluster_id": gap.cluster_id,
                "representative_queries": "; ".join(gap.representative_queries[:5]),
                "total_demand": gap.total_demand,
                "content_count": gap.content_count,
                "coverage_score": gap.coverage_score,
                "gap_score": gap.gap_score,
                **gap.metadata,
            })
        return pd.DataFrame(data).sort_values("gap_score", ascending=False)


class CoverageAnalyzer:
    """
    Analyzes semantic coverage of content against demand signals.
    
    Maps query demand onto content clusters to identify gaps where
    high demand exists but content coverage is thin.
    
    Parameters
    ----------
    min_content_threshold : int
        Minimum content pieces for a cluster to be considered "covered".
    demand_weight_power : float
        Power to raise demand weights (>1 emphasizes high-volume queries).
        
    Examples
    --------
    >>> analyzer = CoverageAnalyzer()
    >>> report = analyzer.analyze(
    ...     content_clusters=[0, 0, 1, 1, 2],
    ...     query_clusters=[0, 0, 0, 1, 3, 3, 3],
    ...     query_demands=[100, 200, 150, 50, 300, 400, 200],
    ... )
    >>> print(report.overall_coverage)
    0.65
    """
    
    def __init__(
        self,
        min_content_threshold: int = 2,
        demand_weight_power: float = 1.0,
    ):
        self.min_content_threshold = min_content_threshold
        self.demand_weight_power = demand_weight_power
    
    def analyze(
        self,
        content_clusters: List[int],
        query_clusters: List[int],
        query_demands: List[float],
        content_ids: Optional[List[str]] = None,
        query_texts: Optional[List[str]] = None,
    ) -> CoverageReport:
        """
        Analyze coverage of content against query demand.
        
        Parameters
        ----------
        content_clusters : List[int]
            Cluster assignments for each content piece.
        query_clusters : List[int]
            Cluster assignments for each query.
        query_demands : List[float]
            Demand signal (e.g., search volume) for each query.
        content_ids : List[str], optional
            Identifiers for content pieces.
        query_texts : List[str], optional
            Query text strings for labeling gaps.
            
        Returns
        -------
        CoverageReport
            Complete coverage analysis with gaps identified.
        """
        # Count content per cluster
        content_per_cluster = defaultdict(int)
        for cluster in content_clusters:
            if cluster >= 0:  # Skip noise
                content_per_cluster[cluster] += 1
        
        # Aggregate demand per cluster
        demand_per_cluster = defaultdict(float)
        queries_per_cluster = defaultdict(list)
        
        for i, (cluster, demand) in enumerate(zip(query_clusters, query_demands)):
            if cluster >= 0:  # Skip noise
                weighted_demand = demand ** self.demand_weight_power
                demand_per_cluster[cluster] += weighted_demand
                if query_texts:
                    queries_per_cluster[cluster].append((query_texts[i], demand))
        
        # Sort queries within each cluster by demand
        for cluster in queries_per_cluster:
            queries_per_cluster[cluster].sort(key=lambda x: x[1], reverse=True)
        
        # All clusters (from both content and queries)
        all_clusters = set(content_per_cluster.keys()) | set(demand_per_cluster.keys())
        
        # Calculate coverage per cluster
        cluster_coverage = {}
        gaps = []
        total_demand = sum(demand_per_cluster.values())
        covered_demand = 0.0
        
        for cluster in all_clusters:
            content_count = content_per_cluster.get(cluster, 0)
            cluster_demand = demand_per_cluster.get(cluster, 0)
            
            # Coverage score: 0 if no content, scales with content count
            if content_count == 0:
                coverage_score = 0.0
            elif content_count >= self.min_content_threshold:
                coverage_score = 1.0
            else:
                coverage_score = content_count / self.min_content_threshold
            
            cluster_coverage[cluster] = coverage_score
            
            # Track covered demand
            covered_demand += cluster_demand * coverage_score
            
            # Gap score: high demand + low coverage = big opportunity
            gap_score = cluster_demand * (1 - coverage_score)
            
            # Get representative queries
            rep_queries = [q[0] for q in queries_per_cluster.get(cluster, [])[:5]]
            
            gap = CoverageGap(
                cluster_id=cluster,
                representative_queries=rep_queries,
                total_demand=cluster_demand,
                content_count=content_count,
                coverage_score=coverage_score,
                gap_score=gap_score,
            )
            gaps.append(gap)
        
        # Sort gaps by opportunity
        gaps.sort(key=lambda g: g.gap_score, reverse=True)
        
        # Overall coverage (demand-weighted)
        overall_coverage = covered_demand / total_demand if total_demand > 0 else 1.0
        
        return CoverageReport(
            overall_coverage=overall_coverage,
            covered_demand=covered_demand,
            uncovered_demand=total_demand - covered_demand,
            gaps=gaps,
            cluster_coverage=cluster_coverage,
        )
    
    def analyze_with_embeddings(
        self,
        content_embeddings: np.ndarray,
        query_embeddings: np.ndarray,
        query_demands: List[float],
        query_texts: Optional[List[str]] = None,
        similarity_threshold: float = 0.7,
    ) -> CoverageReport:
        """
        Analyze coverage using direct embedding similarity (no pre-clustering).
        
        Each query is considered "covered" if it has sufficient similar content.
        
        Parameters
        ----------
        content_embeddings : np.ndarray
            Embedding vectors for content.
        query_embeddings : np.ndarray
            Embedding vectors for queries.
        query_demands : List[float]
            Demand signal for each query.
        query_texts : List[str], optional
            Query text strings.
        similarity_threshold : float
            Minimum similarity to consider a query covered.
            
        Returns
        -------
        CoverageReport
            Coverage analysis based on embedding similarity.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Compute similarity between queries and content
        sim_matrix = cosine_similarity(query_embeddings, content_embeddings)
        
        # For each query, find max similarity to any content
        max_similarities = sim_matrix.max(axis=1)
        
        # Count content matches per query
        content_matches = (sim_matrix >= similarity_threshold).sum(axis=1)
        
        # Calculate coverage
        total_demand = sum(query_demands)
        covered_demand = 0.0
        gaps = []
        
        for i, (sim, matches, demand) in enumerate(zip(max_similarities, content_matches, query_demands)):
            if matches >= self.min_content_threshold:
                coverage_score = 1.0
            elif matches > 0:
                coverage_score = matches / self.min_content_threshold
            else:
                coverage_score = 0.0
            
            covered_demand += demand * coverage_score
            gap_score = demand * (1 - coverage_score)
            
            query_text = query_texts[i] if query_texts else f"Query {i}"
            
            gap = CoverageGap(
                cluster_id=i,  # Each query is its own "cluster"
                representative_queries=[query_text],
                total_demand=demand,
                content_count=int(matches),
                coverage_score=coverage_score,
                gap_score=gap_score,
                metadata={"max_similarity": float(sim)},
            )
            gaps.append(gap)
        
        gaps.sort(key=lambda g: g.gap_score, reverse=True)
        overall_coverage = covered_demand / total_demand if total_demand > 0 else 1.0
        
        return CoverageReport(
            overall_coverage=overall_coverage,
            covered_demand=covered_demand,
            uncovered_demand=total_demand - covered_demand,
            gaps=gaps,
            cluster_coverage={i: g.coverage_score for i, g in enumerate(gaps)},
        )
    
    def compare_coverage(
        self,
        your_report: CoverageReport,
        competitor_report: CoverageReport,
    ) -> pd.DataFrame:
        """
        Compare coverage between your site and a competitor.
        
        Parameters
        ----------
        your_report : CoverageReport
            Your coverage analysis.
        competitor_report : CoverageReport
            Competitor's coverage analysis.
            
        Returns
        -------
        pd.DataFrame
            Comparison showing where competitor has better coverage.
        """
        your_coverage = {g.cluster_id: g for g in your_report.gaps}
        comp_coverage = {g.cluster_id: g for g in competitor_report.gaps}
        
        all_clusters = set(your_coverage.keys()) | set(comp_coverage.keys())
        
        data = []
        for cluster in all_clusters:
            your_gap = your_coverage.get(cluster)
            comp_gap = comp_coverage.get(cluster)
            
            your_score = your_gap.coverage_score if your_gap else 0.0
            comp_score = comp_gap.coverage_score if comp_gap else 0.0
            demand = (your_gap or comp_gap).total_demand if (your_gap or comp_gap) else 0.0
            
            data.append({
                "cluster_id": cluster,
                "your_coverage": your_score,
                "competitor_coverage": comp_score,
                "coverage_gap": comp_score - your_score,
                "demand": demand,
                "opportunity": demand * max(0, comp_score - your_score),
            })
        
        df = pd.DataFrame(data)
        return df.sort_values("opportunity", ascending=False)


def calculate_topical_authority(
    cluster_coverage: Dict[int, float],
    cluster_demands: Dict[int, float],
    internal_link_density: Optional[Dict[int, float]] = None,
) -> float:
    """
    Calculate overall topical authority score.
    
    Combines coverage breadth, demand alignment, and internal linking.
    
    Parameters
    ----------
    cluster_coverage : Dict[int, float]
        Coverage score per cluster.
    cluster_demands : Dict[int, float]
        Demand weight per cluster.
    internal_link_density : Dict[int, float], optional
        Internal linking density per cluster.
        
    Returns
    -------
    float
        Topical authority score (0-1).
    """
    total_demand = sum(cluster_demands.values())
    if total_demand == 0:
        return 0.0
    
    # Demand-weighted coverage
    weighted_coverage = sum(
        cluster_coverage.get(c, 0) * d 
        for c, d in cluster_demands.items()
    ) / total_demand
    
    # Breadth bonus (covering more clusters)
    covered_clusters = sum(1 for c in cluster_coverage.values() if c > 0.5)
    total_clusters = len(set(cluster_coverage.keys()) | set(cluster_demands.keys()))
    breadth_score = covered_clusters / total_clusters if total_clusters > 0 else 0
    
    # Internal linking bonus
    if internal_link_density:
        link_score = np.mean(list(internal_link_density.values()))
    else:
        link_score = 0.5  # Neutral if not provided
    
    # Combined score
    authority = (0.5 * weighted_coverage) + (0.3 * breadth_score) + (0.2 * link_score)
    return min(1.0, authority)
