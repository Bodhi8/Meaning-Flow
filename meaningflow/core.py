"""
Core SemanticGraph implementation.

This is the primary entry point for MeaningFlow. It wraps the full
embed -> reduce -> cluster -> graph pipeline into a single object
with a clean API.
"""

import logging
from typing import List, Optional, Union

import numpy as np
from dataclasses import dataclass

from .models import Cluster, GapCluster

log = logging.getLogger(__name__)


class SemanticGraph:
    """
    Semantic content graph built from a text corpus.

    Embeds texts using a Sentence-BERT model, reduces dimensionality with
    UMAP, clusters with HDBSCAN, and builds a NetworkX graph over the
    resulting clusters. Supports coverage gap analysis against a reference
    graph.

    Parameters
    ----------
    texts : list[str]
        The text corpus to analyze.
    embedder : str
        Name of a Sentence-Transformers model. Default: 'all-MiniLM-L6-v2'.
    embeddings : np.ndarray, optional
        Pre-computed embeddings. If provided, the embedding step is skipped.
    volumes : list[int], optional
        Search volume per text. Used for volume-weighted gap analysis.
        Must be the same length as texts.
    min_cluster_size : int
        HDBSCAN minimum cluster size. Default: 30.
    min_samples : int
        HDBSCAN minimum samples for core points. Default: 10.
    umap_n_neighbors : int
        UMAP local neighborhood size. Default: 30.
    umap_n_components : int
        UMAP target dimensionality. Default: 10.
    umap_metric : str
        UMAP distance metric. Default: 'cosine'.
    similarity_threshold : float
        Minimum cosine similarity for inter-cluster graph edges. Default: 0.3.
    top_n_terms : int
        Number of representative terms to store per cluster. Default: 10.
    random_state : int, optional
        Random seed for reproducibility. Default: 42.

    Examples
    --------
    >>> sg = SemanticGraph(texts=my_queries, embedder="all-MiniLM-L6-v2")
    >>> sg.fit()
    >>> print(f"Found {sg.n_clusters} clusters, noise ratio: {sg.noise_ratio:.2%}")
    >>> for c in sg.clusters[:5]:
    ...     print(c.top_terms[:3], c.size)
    """

    def __init__(
        self,
        texts: List[str],
        embedder: str = "all-MiniLM-L6-v2",
        embeddings: Optional[np.ndarray] = None,
        volumes: Optional[List[int]] = None,
        min_cluster_size: int = 30,
        min_samples: int = 10,
        umap_n_neighbors: int = 30,
        umap_n_components: int = 10,
        umap_metric: str = "cosine",
        similarity_threshold: float = 0.3,
        top_n_terms: int = 10,
        random_state: int = 42,
    ):
        self.texts = texts
        self.embedder_name = embedder
        self._precomputed_embeddings = embeddings
        self.volumes = volumes if volumes is not None else [1] * len(texts)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_n_components = umap_n_components
        self.umap_metric = umap_metric
        self.similarity_threshold = similarity_threshold
        self.top_n_terms = top_n_terms
        self.random_state = random_state

        # Populated after fit()
        self.embeddings: Optional[np.ndarray] = None
        self.reduced: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.clusters: List[Cluster] = []
        self.graph = None  # nx.Graph
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> "SemanticGraph":
        """
        Run the full pipeline: embed -> reduce -> cluster -> build graph.

        Returns self for method chaining.
        """
        log.info("Starting MeaningFlow pipeline on %d texts", len(self.texts))

        self._embed()
        self._reduce()
        self._cluster()
        self._build_graph()

        self._is_fitted = True
        log.info(
            "Pipeline complete: %d clusters, %.1f%% noise",
            self.n_clusters,
            self.noise_ratio * 100,
        )
        return self

    @property
    def n_clusters(self) -> int:
        """Number of clusters found (excluding the noise bucket)."""
        if self.labels is None:
            return 0
        unique = set(self.labels)
        unique.discard(-1)
        return len(unique)

    @property
    def noise_ratio(self) -> float:
        """Fraction of texts assigned to the noise bucket (-1)."""
        if self.labels is None:
            return 0.0
        return float((self.labels == -1).sum()) / len(self.labels)

    def coverage_gaps(
        self,
        reference: "SemanticGraph",
        similarity_threshold: float = 0.55,
    ) -> List[GapCluster]:
        """
        Find clusters in this graph that have no close match in a reference graph.

        This graph is treated as "demand" (what users want). The reference
        graph is "supply" (what content exists). A demand cluster is a gap
        if no supply cluster centroid has cosine similarity >= the threshold.

        Parameters
        ----------
        reference : SemanticGraph
            A fitted SemanticGraph representing the supply side.
        similarity_threshold : float
            Minimum cosine similarity for a supply cluster to count as
            covering a demand cluster. Default: 0.55.

        Returns
        -------
        list[GapCluster]
            Gap clusters sorted by volume descending.
        """
        self._check_fitted("coverage_gaps")
        reference._check_fitted("coverage_gaps (reference)")

        if not reference.clusters:
            # No supply at all — everything is a gap
            return [
                GapCluster(
                    id=c.id,
                    size=c.size,
                    top_terms=c.top_terms,
                    volume=c.volume,
                    centroid=c.centroid,
                    nearest_supply="(none)",
                    nearest_similarity=0.0,
                )
                for c in sorted(self.clusters, key=lambda x: x.volume, reverse=True)
            ]

        # Stack supply centroids for vectorized comparison
        supply_centroids = np.array([c.centroid for c in reference.clusters])
        supply_norms = np.linalg.norm(supply_centroids, axis=1, keepdims=True)
        supply_centroids_normed = supply_centroids / np.maximum(supply_norms, 1e-10)

        gaps = []
        for cluster in self.clusters:
            # Cosine similarity to all supply centroids
            demand_normed = cluster.centroid / max(
                np.linalg.norm(cluster.centroid), 1e-10
            )
            similarities = supply_centroids_normed @ demand_normed

            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])
            best_supply = reference.clusters[best_idx]

            if best_sim < similarity_threshold:
                gaps.append(
                    GapCluster(
                        id=cluster.id,
                        size=cluster.size,
                        top_terms=cluster.top_terms,
                        volume=cluster.volume,
                        centroid=cluster.centroid,
                        nearest_supply=(
                            best_supply.top_terms[0] if best_supply.top_terms else ""
                        ),
                        nearest_similarity=best_sim,
                    )
                )

        gaps.sort(key=lambda g: g.volume, reverse=True)
        log.info(
            "Found %d gap clusters out of %d demand clusters (threshold=%.2f)",
            len(gaps),
            len(self.clusters),
            similarity_threshold,
        )
        return gaps

    def get_cluster(self, cluster_id: int) -> Optional[Cluster]:
        """Retrieve a specific cluster by ID."""
        for c in self.clusters:
            if c.id == cluster_id:
                return c
        return None

    def cluster_similarities(self) -> np.ndarray:
        """
        Compute the pairwise cosine similarity matrix between cluster centroids.

        Returns
        -------
        np.ndarray
            Square matrix of shape (n_clusters, n_clusters).
        """
        self._check_fitted("cluster_similarities")
        centroids = np.array([c.centroid for c in self.clusters])
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        normed = centroids / np.maximum(norms, 1e-10)
        return normed @ normed.T

    # ------------------------------------------------------------------
    # Pipeline stages (private)
    # ------------------------------------------------------------------

    def _embed(self):
        """Stage 1: Embed texts using Sentence-BERT."""
        if self._precomputed_embeddings is not None:
            log.info("Using pre-computed embeddings (%s)", self._precomputed_embeddings.shape)
            self.embeddings = self._precomputed_embeddings
            return

        from sentence_transformers import SentenceTransformer

        log.info("Embedding %d texts with %s", len(self.texts), self.embedder_name)
        model = SentenceTransformer(self.embedder_name)
        self.embeddings = model.encode(
            self.texts,
            show_progress_bar=True,
            batch_size=256,
            convert_to_numpy=True,
        )
        log.info("Embeddings shape: %s", self.embeddings.shape)

    def _reduce(self):
        """Stage 2: Reduce dimensionality with UMAP."""
        import umap

        n_samples = self.embeddings.shape[0]
        # Adjust n_neighbors if dataset is small
        n_neighbors = min(self.umap_n_neighbors, n_samples - 1)
        n_components = min(self.umap_n_components, n_samples - 1)

        log.info(
            "UMAP reduction: %d dims -> %d dims (n_neighbors=%d)",
            self.embeddings.shape[1],
            n_components,
            n_neighbors,
        )
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric=self.umap_metric,
            random_state=self.random_state,
        )
        self.reduced = reducer.fit_transform(self.embeddings)

    def _cluster(self):
        """Stage 3: Cluster with HDBSCAN and build Cluster objects."""
        import hdbscan

        log.info(
            "HDBSCAN clustering (min_cluster_size=%d, min_samples=%d)",
            self.min_cluster_size,
            self.min_samples,
        )
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        self.labels = clusterer.fit_predict(self.reduced)

        # Build Cluster objects
        unique_labels = sorted(set(self.labels))
        self.clusters = []

        for label in unique_labels:
            if label == -1:
                continue  # skip noise

            mask = self.labels == label
            indices = np.where(mask)[0]
            cluster_texts = [self.texts[i] for i in indices]
            cluster_embeddings = self.embeddings[indices]
            cluster_volumes = [self.volumes[i] for i in indices]

            centroid = cluster_embeddings.mean(axis=0)

            # Rank texts by proximity to centroid for top_terms
            centroid_norm = centroid / max(np.linalg.norm(centroid), 1e-10)
            emb_norms = np.linalg.norm(cluster_embeddings, axis=1, keepdims=True)
            emb_normed = cluster_embeddings / np.maximum(emb_norms, 1e-10)
            distances = emb_normed @ centroid_norm
            ranked_indices = np.argsort(-distances)

            top_terms = []
            seen = set()
            for idx in ranked_indices:
                term = cluster_texts[idx].strip().lower()
                if term not in seen:
                    seen.add(term)
                    top_terms.append(cluster_texts[idx].strip())
                if len(top_terms) >= self.top_n_terms:
                    break

            self.clusters.append(
                Cluster(
                    id=int(label),
                    size=int(mask.sum()),
                    top_terms=top_terms,
                    centroid=centroid,
                    texts=cluster_texts,
                    volume=int(sum(cluster_volumes)),
                )
            )

        # Sort clusters by volume descending
        self.clusters.sort(key=lambda c: c.volume, reverse=True)
        log.info(
            "Found %d clusters (noise: %d / %d = %.1f%%)",
            len(self.clusters),
            (self.labels == -1).sum(),
            len(self.labels),
            self.noise_ratio * 100,
        )

    def _build_graph(self):
        """Stage 4: Build a NetworkX graph over cluster centroids."""
        import networkx as nx

        self.graph = nx.Graph()

        # Add cluster nodes
        for cluster in self.clusters:
            self.graph.add_node(
                cluster.id,
                size=cluster.size,
                volume=cluster.volume,
                top_terms=cluster.top_terms[:3],
            )

        # Add edges where cosine similarity exceeds threshold
        sim_matrix = self.cluster_similarities()
        for i, ci in enumerate(self.clusters):
            for j, cj in enumerate(self.clusters):
                if i >= j:
                    continue
                sim = float(sim_matrix[i, j])
                if sim >= self.similarity_threshold:
                    self.graph.add_edge(ci.id, cj.id, weight=sim)

        log.info(
            "Graph: %d nodes, %d edges (threshold=%.2f)",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            self.similarity_threshold,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _check_fitted(self, method_name: str):
        if not self._is_fitted:
            raise RuntimeError(
                f"SemanticGraph.{method_name}() called before fit(). "
                "Call .fit() first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"SemanticGraph(texts={len(self.texts)}, "
            f"embedder='{self.embedder_name}', "
            f"status={status}"
            f"{f', clusters={self.n_clusters}' if self._is_fitted else ''})"
        )
