"""
Demand Mapping Module

Maps search demand signals (volume, trends, CTR) onto the semantic graph
and content clusters to identify high-value opportunities.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DemandSignal:
    """Represents a demand signal (typically a search query)."""
    
    query: str
    volume: float
    embedding: Optional[np.ndarray] = None
    trend: Optional[float] = None  # Growth rate
    cpc: Optional[float] = None  # Cost per click (competition proxy)
    difficulty: Optional[float] = None  # Ranking difficulty
    intent: Optional[str] = None  # informational, transactional, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def opportunity_score(self) -> float:
        """Calculate basic opportunity score from available signals."""
        score = self.volume
        
        # Boost for growing queries
        if self.trend and self.trend > 0:
            score *= (1 + min(self.trend, 1.0))
        
        # Adjust for difficulty (lower difficulty = higher opportunity)
        if self.difficulty is not None:
            score *= (1 - self.difficulty * 0.5)
        
        return score


@dataclass
class DemandCluster:
    """Aggregated demand for a semantic cluster."""
    
    cluster_id: int
    total_volume: float
    query_count: int
    avg_difficulty: Optional[float]
    dominant_intent: Optional[str]
    representative_queries: List[str]
    trend_direction: str  # 'growing', 'stable', 'declining'
    signals: List[DemandSignal] = field(default_factory=list)


class DemandMapper:
    """
    Maps demand signals onto semantic space and content clusters.
    
    Aggregates query-level signals into cluster-level demand metrics
    for strategic prioritization.
    
    Examples
    --------
    >>> mapper = DemandMapper()
    >>> mapper.add_signal("how to do seo", volume=5000, intent="informational")
    >>> mapper.add_signal("seo tools", volume=8000, intent="commercial")
    >>> demand_by_cluster = mapper.aggregate_by_cluster(cluster_labels)
    """
    
    def __init__(self):
        self.signals: List[DemandSignal] = []
        self._embeddings: Optional[np.ndarray] = None
    
    def add_signal(
        self,
        query: str,
        volume: float,
        embedding: Optional[np.ndarray] = None,
        trend: Optional[float] = None,
        cpc: Optional[float] = None,
        difficulty: Optional[float] = None,
        intent: Optional[str] = None,
        **metadata,
    ) -> None:
        """
        Add a demand signal (query with associated metrics).
        
        Parameters
        ----------
        query : str
            Search query text.
        volume : float
            Search volume or demand metric.
        embedding : np.ndarray, optional
            Semantic embedding of the query.
        trend : float, optional
            Growth rate (-1 to +inf, where 0 = stable).
        cpc : float, optional
            Cost per click (competition indicator).
        difficulty : float, optional
            Ranking difficulty (0-1).
        intent : str, optional
            Search intent category.
        **metadata
            Additional signal attributes.
        """
        signal = DemandSignal(
            query=query,
            volume=volume,
            embedding=embedding,
            trend=trend,
            cpc=cpc,
            difficulty=difficulty,
            intent=intent,
            metadata=metadata,
        )
        self.signals.append(signal)
    
    def load_from_dataframe(
        self,
        df: pd.DataFrame,
        query_col: str = "query",
        volume_col: str = "volume",
        trend_col: Optional[str] = None,
        cpc_col: Optional[str] = None,
        difficulty_col: Optional[str] = None,
        intent_col: Optional[str] = None,
    ) -> int:
        """
        Load demand signals from a DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing query data.
        query_col : str
            Column name for query text.
        volume_col : str
            Column name for search volume.
        trend_col : str, optional
            Column name for trend data.
        cpc_col : str, optional
            Column name for CPC data.
        difficulty_col : str, optional
            Column name for difficulty scores.
        intent_col : str, optional
            Column name for intent labels.
            
        Returns
        -------
        int
            Number of signals loaded.
        """
        count = 0
        for _, row in df.iterrows():
            self.add_signal(
                query=row[query_col],
                volume=row[volume_col],
                trend=row.get(trend_col) if trend_col else None,
                cpc=row.get(cpc_col) if cpc_col else None,
                difficulty=row.get(difficulty_col) if difficulty_col else None,
                intent=row.get(intent_col) if intent_col else None,
            )
            count += 1
        return count
    
    def embed_queries(self, embedder) -> np.ndarray:
        """
        Generate embeddings for all queries using provided embedder.
        
        Parameters
        ----------
        embedder : SemanticEmbedder
            Embedder instance to use.
            
        Returns
        -------
        np.ndarray
            Array of query embeddings.
        """
        queries = [s.query for s in self.signals]
        result = embedder.embed(queries, show_progress=True)
        
        for i, signal in enumerate(self.signals):
            signal.embedding = result.vectors[i]
        
        self._embeddings = result.vectors
        return result.vectors
    
    def aggregate_by_cluster(
        self,
        cluster_labels: List[int],
        top_queries: int = 5,
    ) -> Dict[int, DemandCluster]:
        """
        Aggregate demand signals by cluster.
        
        Parameters
        ----------
        cluster_labels : List[int]
            Cluster assignment for each signal.
        top_queries : int
            Number of top queries to include as representatives.
            
        Returns
        -------
        Dict[int, DemandCluster]
            Cluster ID mapped to aggregated demand metrics.
        """
        if len(cluster_labels) != len(self.signals):
            raise ValueError(
                f"Cluster labels ({len(cluster_labels)}) must match "
                f"signal count ({len(self.signals)})"
            )
        
        # Group signals by cluster
        clusters: Dict[int, List[DemandSignal]] = {}
        for label, signal in zip(cluster_labels, self.signals):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(signal)
        
        # Aggregate each cluster
        demand_clusters = {}
        for cluster_id, signals in clusters.items():
            if cluster_id == -1:  # Skip noise
                continue
            
            total_volume = sum(s.volume for s in signals)
            
            # Sort by volume for representative queries
            sorted_signals = sorted(signals, key=lambda s: s.volume, reverse=True)
            rep_queries = [s.query for s in sorted_signals[:top_queries]]
            
            # Average difficulty
            difficulties = [s.difficulty for s in signals if s.difficulty is not None]
            avg_difficulty = np.mean(difficulties) if difficulties else None
            
            # Dominant intent
            intents = [s.intent for s in signals if s.intent]
            if intents:
                from collections import Counter
                dominant_intent = Counter(intents).most_common(1)[0][0]
            else:
                dominant_intent = None
            
            # Trend direction
            trends = [s.trend for s in signals if s.trend is not None]
            if trends:
                avg_trend = np.mean(trends)
                if avg_trend > 0.1:
                    trend_direction = "growing"
                elif avg_trend < -0.1:
                    trend_direction = "declining"
                else:
                    trend_direction = "stable"
            else:
                trend_direction = "unknown"
            
            demand_clusters[cluster_id] = DemandCluster(
                cluster_id=cluster_id,
                total_volume=total_volume,
                query_count=len(signals),
                avg_difficulty=avg_difficulty,
                dominant_intent=dominant_intent,
                representative_queries=rep_queries,
                trend_direction=trend_direction,
                signals=signals,
            )
        
        return demand_clusters
    
    def find_nearest_content(
        self,
        content_embeddings: np.ndarray,
        content_ids: List[str],
        top_k: int = 3,
    ) -> pd.DataFrame:
        """
        For each demand signal, find the nearest content pieces.
        
        Parameters
        ----------
        content_embeddings : np.ndarray
            Embeddings for content pieces.
        content_ids : List[str]
            Identifiers for content pieces.
        top_k : int
            Number of nearest content pieces to return.
            
        Returns
        -------
        pd.DataFrame
            Query-to-content mapping with similarity scores.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        if self._embeddings is None:
            raise ValueError("Must call embed_queries() first")
        
        sim_matrix = cosine_similarity(self._embeddings, content_embeddings)
        
        data = []
        for i, signal in enumerate(self.signals):
            top_indices = np.argsort(sim_matrix[i])[-top_k:][::-1]
            
            for rank, idx in enumerate(top_indices):
                data.append({
                    "query": signal.query,
                    "volume": signal.volume,
                    "content_id": content_ids[idx],
                    "similarity": sim_matrix[i, idx],
                    "rank": rank + 1,
                })
        
        return pd.DataFrame(data)
    
    def prioritize_opportunities(
        self,
        cluster_labels: List[int],
        content_coverage: Dict[int, float],
    ) -> pd.DataFrame:
        """
        Rank clusters by opportunity (high demand, low coverage).
        
        Parameters
        ----------
        cluster_labels : List[int]
            Cluster assignments for queries.
        content_coverage : Dict[int, float]
            Coverage score (0-1) for each cluster.
            
        Returns
        -------
        pd.DataFrame
            Clusters ranked by opportunity score.
        """
        demand_clusters = self.aggregate_by_cluster(cluster_labels)
        
        data = []
        for cluster_id, demand in demand_clusters.items():
            coverage = content_coverage.get(cluster_id, 0.0)
            gap = 1 - coverage
            
            # Opportunity = demand * gap
            opportunity = demand.total_volume * gap
            
            # Boost for growing trends
            if demand.trend_direction == "growing":
                opportunity *= 1.2
            elif demand.trend_direction == "declining":
                opportunity *= 0.8
            
            data.append({
                "cluster_id": cluster_id,
                "total_volume": demand.total_volume,
                "query_count": demand.query_count,
                "coverage": coverage,
                "gap": gap,
                "opportunity_score": opportunity,
                "trend": demand.trend_direction,
                "dominant_intent": demand.dominant_intent,
                "representative_queries": "; ".join(demand.representative_queries),
            })
        
        df = pd.DataFrame(data)
        return df.sort_values("opportunity_score", ascending=False)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Export all signals to DataFrame."""
        data = []
        for s in self.signals:
            data.append({
                "query": s.query,
                "volume": s.volume,
                "trend": s.trend,
                "cpc": s.cpc,
                "difficulty": s.difficulty,
                "intent": s.intent,
                "opportunity_score": s.opportunity_score,
                **s.metadata,
            })
        return pd.DataFrame(data)
    
    def __len__(self) -> int:
        return len(self.signals)
    
    def __repr__(self) -> str:
        total_volume = sum(s.volume for s in self.signals)
        return f"DemandMapper(signals={len(self.signals)}, total_volume={total_volume:,.0f})"
