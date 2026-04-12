"""
Semantic Graph Module

Constructs and analyzes semantic content graphs where nodes represent 
content clusters or entities and edges denote their relationships.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class GraphNode:
    """Represents a node in the semantic graph."""
    
    id: str
    label: str
    node_type: str  # 'content', 'cluster', 'entity', 'query'
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class GraphEdge:
    """Represents an edge in the semantic graph."""
    
    source: str
    target: str
    edge_type: str  # 'semantic', 'structural', 'hierarchical'
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticGraph:
    """
    Semantic content graph for modeling meaning relationships.
    
    Nodes can be content pieces, topic clusters, entities, or queries.
    Edges represent semantic similarity, structural links, or hierarchical
    relationships.
    
    Parameters
    ----------
    similarity_threshold : float
        Minimum cosine similarity for automatic edge creation.
        
    Examples
    --------
    >>> graph = SemanticGraph()
    >>> graph.add_node("doc1", "How to SEO", "content", embedding=vec1)
    >>> graph.add_node("doc2", "SEO Guide", "content", embedding=vec2)
    >>> graph.build_similarity_edges()
    >>> graph.get_connected_components()
    """
    
    def __init__(self, similarity_threshold: float = 0.5):
        self.similarity_threshold = similarity_threshold
        self._graph = nx.DiGraph()
        self._nodes: Dict[str, GraphNode] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        
    @property
    def graph(self) -> nx.DiGraph:
        """Access the underlying NetworkX graph."""
        return self._graph
    
    def add_node(
        self,
        node_id: str,
        label: str,
        node_type: str,
        embedding: Optional[np.ndarray] = None,
        **metadata,
    ) -> None:
        """
        Add a node to the semantic graph.
        
        Parameters
        ----------
        node_id : str
            Unique identifier for the node.
        label : str
            Human-readable label.
        node_type : str
            Type of node: 'content', 'cluster', 'entity', or 'query'.
        embedding : np.ndarray, optional
            Semantic embedding vector.
        **metadata
            Additional node attributes.
        """
        node = GraphNode(
            id=node_id,
            label=label,
            node_type=node_type,
            embedding=embedding,
            metadata=metadata,
        )
        self._nodes[node_id] = node
        
        if embedding is not None:
            self._embeddings[node_id] = embedding
            
        self._graph.add_node(
            node_id,
            label=label,
            node_type=node_type,
            **metadata,
        )
    
    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str = "semantic",
        weight: float = 1.0,
        **metadata,
    ) -> None:
        """
        Add an edge between two nodes.
        
        Parameters
        ----------
        source : str
            Source node ID.
        target : str
            Target node ID.
        edge_type : str
            Type of relationship: 'semantic', 'structural', 'hierarchical'.
        weight : float
            Edge weight (e.g., similarity score).
        **metadata
            Additional edge attributes.
        """
        self._graph.add_edge(
            source,
            target,
            edge_type=edge_type,
            weight=weight,
            **metadata,
        )
    
    def build_similarity_edges(
        self,
        threshold: Optional[float] = None,
        node_types: Optional[List[str]] = None,
    ) -> int:
        """
        Create edges between nodes based on embedding similarity.
        
        Parameters
        ----------
        threshold : float, optional
            Minimum similarity for edge creation. Uses instance default if None.
        node_types : List[str], optional
            Only connect nodes of these types. Connects all if None.
            
        Returns
        -------
        int
            Number of edges created.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        threshold = threshold or self.similarity_threshold
        
        # Filter nodes with embeddings
        nodes_with_embeddings = [
            (nid, emb) for nid, emb in self._embeddings.items()
            if node_types is None or self._nodes[nid].node_type in node_types
        ]
        
        if len(nodes_with_embeddings) < 2:
            return 0
        
        node_ids = [n[0] for n in nodes_with_embeddings]
        embeddings = np.array([n[1] for n in nodes_with_embeddings])
        
        # Compute similarity matrix
        sim_matrix = cosine_similarity(embeddings)
        
        edges_created = 0
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                if sim_matrix[i, j] >= threshold:
                    self.add_edge(
                        node_ids[i],
                        node_ids[j],
                        edge_type="semantic",
                        weight=float(sim_matrix[i, j]),
                    )
                    # Add reverse edge for undirected similarity
                    self.add_edge(
                        node_ids[j],
                        node_ids[i],
                        edge_type="semantic",
                        weight=float(sim_matrix[i, j]),
                    )
                    edges_created += 2
                    
        return edges_created
    
    def add_cluster_hierarchy(
        self,
        cluster_labels: np.ndarray,
        node_ids: List[str],
        cluster_prefix: str = "cluster_",
    ) -> None:
        """
        Add cluster nodes and connect content nodes to their clusters.
        
        Parameters
        ----------
        cluster_labels : np.ndarray
            Cluster assignment for each node.
        node_ids : List[str]
            Node IDs corresponding to cluster labels.
        cluster_prefix : str
            Prefix for cluster node IDs.
        """
        unique_clusters = set(cluster_labels)
        
        # Create cluster nodes
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise cluster from HDBSCAN
                continue
            cluster_node_id = f"{cluster_prefix}{cluster_id}"
            
            # Compute cluster centroid if embeddings exist
            member_ids = [
                node_ids[i] for i, c in enumerate(cluster_labels) 
                if c == cluster_id
            ]
            member_embeddings = [
                self._embeddings[nid] for nid in member_ids 
                if nid in self._embeddings
            ]
            
            centroid = None
            if member_embeddings:
                centroid = np.mean(member_embeddings, axis=0)
            
            self.add_node(
                cluster_node_id,
                label=f"Cluster {cluster_id}",
                node_type="cluster",
                embedding=centroid,
                member_count=len(member_ids),
            )
            
            # Connect members to cluster
            for member_id in member_ids:
                self.add_edge(
                    member_id,
                    cluster_node_id,
                    edge_type="hierarchical",
                    weight=1.0,
                )
    
    def compute_pagerank(self, **kwargs) -> Dict[str, float]:
        """
        Compute PageRank centrality for all nodes.
        
        Returns
        -------
        Dict[str, float]
            Node IDs mapped to PageRank scores.
        """
        return nx.pagerank(self._graph, **kwargs)
    
    def compute_authority_scores(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute HITS hub and authority scores.
        
        Returns
        -------
        Tuple[Dict, Dict]
            (hub_scores, authority_scores) dictionaries.
        """
        hubs, authorities = nx.hits(self._graph)
        return hubs, authorities
    
    def find_bridges(self) -> List[Tuple[str, str]]:
        """
        Find bridge edges that connect different communities.
        
        Returns
        -------
        List[Tuple[str, str]]
            List of (source, target) edge tuples that are bridges.
        """
        # Convert to undirected for bridge detection
        undirected = self._graph.to_undirected()
        return list(nx.bridges(undirected))
    
    def find_bottlenecks(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Find nodes that act as bottlenecks (high betweenness centrality).
        
        Parameters
        ----------
        top_n : int
            Number of top bottleneck nodes to return.
            
        Returns
        -------
        List[Tuple[str, float]]
            List of (node_id, betweenness_score) tuples.
        """
        betweenness = nx.betweenness_centrality(self._graph)
        sorted_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_n]
    
    def find_isolated_nodes(self) -> List[str]:
        """
        Find nodes with no connections.
        
        Returns
        -------
        List[str]
            List of isolated node IDs.
        """
        return list(nx.isolates(self._graph))
    
    def get_connected_components(self) -> List[Set[str]]:
        """
        Get all connected components in the graph.
        
        Returns
        -------
        List[Set[str]]
            List of node ID sets, one per component.
        """
        undirected = self._graph.to_undirected()
        return [set(c) for c in nx.connected_components(undirected)]
    
    def get_subgraph(self, node_ids: List[str]) -> "SemanticGraph":
        """
        Extract a subgraph containing only specified nodes.
        
        Parameters
        ----------
        node_ids : List[str]
            Node IDs to include in subgraph.
            
        Returns
        -------
        SemanticGraph
            New graph containing only specified nodes and their edges.
        """
        subgraph = SemanticGraph(similarity_threshold=self.similarity_threshold)
        
        for nid in node_ids:
            if nid in self._nodes:
                node = self._nodes[nid]
                subgraph.add_node(
                    nid,
                    node.label,
                    node.node_type,
                    embedding=node.embedding,
                    **node.metadata,
                )
        
        for u, v, data in self._graph.edges(data=True):
            if u in node_ids and v in node_ids:
                subgraph.add_edge(u, v, **data)
                
        return subgraph
    
    def to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Export graph to node and edge DataFrames.
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (nodes_df, edges_df) tuple.
        """
        # Nodes
        nodes_data = []
        for nid, node in self._nodes.items():
            row = {
                "id": nid,
                "label": node.label,
                "node_type": node.node_type,
                **node.metadata,
            }
            nodes_data.append(row)
        nodes_df = pd.DataFrame(nodes_data)
        
        # Edges
        edges_data = []
        for u, v, data in self._graph.edges(data=True):
            row = {"source": u, "target": v, **data}
            edges_data.append(row)
        edges_df = pd.DataFrame(edges_data)
        
        return nodes_df, edges_df
    
    def export_graphml(self, filepath: str) -> None:
        """Export graph to GraphML format."""
        nx.write_graphml(self._graph, filepath)
    
    def export_gexf(self, filepath: str) -> None:
        """Export graph to GEXF format (for Gephi)."""
        nx.write_gexf(self._graph, filepath)
    
    def __len__(self) -> int:
        return len(self._nodes)
    
    def __repr__(self) -> str:
        return f"SemanticGraph(nodes={len(self._nodes)}, edges={self._graph.number_of_edges()})"
