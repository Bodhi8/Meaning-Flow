"""
Semantic Embeddings Module

Handles text vectorization using transformer models (Sentence-BERT),
dimensionality reduction (UMAP/PCA), and clustering (HDBSCAN/K-Means).
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Literal
from dataclasses import dataclass, field
from tqdm import tqdm


@dataclass
class EmbeddingResult:
    """Container for embedding results with metadata."""
    
    vectors: np.ndarray
    texts: List[str]
    ids: Optional[List[str]] = None
    reduced_vectors: Optional[np.ndarray] = None
    cluster_labels: Optional[np.ndarray] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        data = {
            "text": self.texts,
            "embedding": list(self.vectors),
        }
        if self.ids:
            data["id"] = self.ids
        if self.reduced_vectors is not None:
            for i in range(self.reduced_vectors.shape[1]):
                data[f"dim_{i}"] = self.reduced_vectors[:, i]
        if self.cluster_labels is not None:
            data["cluster"] = self.cluster_labels
        return pd.DataFrame(data)


class SemanticEmbedder:
    """
    Semantic embedding pipeline for text content.
    
    Transforms text into dense vector representations, reduces dimensionality,
    and clusters similar content together.
    
    Parameters
    ----------
    model_name : str
        Sentence-Transformers model name. Default is 'all-MiniLM-L6-v2'.
    device : str, optional
        Device to run model on ('cuda', 'cpu', or 'mps').
        
    Examples
    --------
    >>> embedder = SemanticEmbedder()
    >>> texts = ["How to optimize content", "Content optimization tips"]
    >>> result = embedder.embed(texts)
    >>> result.vectors.shape
    (2, 384)
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        
    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model
    
    def embed(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.
        
        Parameters
        ----------
        texts : List[str]
            Texts to embed.
        ids : List[str], optional
            Identifiers for each text.
        batch_size : int
            Batch size for encoding.
        show_progress : bool
            Whether to show a progress bar.
            
        Returns
        -------
        EmbeddingResult
            Container with vectors and metadata.
        """
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return EmbeddingResult(vectors=vectors, texts=texts, ids=ids)
    
    def reduce_dimensions(
        self,
        result: EmbeddingResult,
        n_components: int = 2,
        method: Literal["umap", "pca"] = "umap",
        **kwargs,
    ) -> EmbeddingResult:
        """
        Reduce embedding dimensionality for visualization and clustering.
        
        Parameters
        ----------
        result : EmbeddingResult
            Embedding result to reduce.
        n_components : int
            Target number of dimensions.
        method : str
            Reduction method: 'umap' or 'pca'.
        **kwargs
            Additional arguments passed to the reducer.
            
        Returns
        -------
        EmbeddingResult
            Updated result with reduced_vectors populated.
        """
        if method == "umap":
            from umap import UMAP
            
            defaults = {
                "n_neighbors": min(15, len(result.vectors) - 1),
                "min_dist": 0.1,
                "metric": "cosine",
                "random_state": 42,
            }
            defaults.update(kwargs)
            reducer = UMAP(n_components=n_components, **defaults)
            
        elif method == "pca":
            from sklearn.decomposition import PCA
            
            reducer = PCA(n_components=n_components, **kwargs)
            
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        reduced = reducer.fit_transform(result.vectors)
        result.reduced_vectors = reduced
        return result
    
    def cluster(
        self,
        result: EmbeddingResult,
        method: Literal["hdbscan", "kmeans", "agglomerative"] = "hdbscan",
        use_reduced: bool = True,
        **kwargs,
    ) -> EmbeddingResult:
        """
        Cluster embeddings into semantic groups.
        
        Parameters
        ----------
        result : EmbeddingResult
            Embedding result to cluster.
        method : str
            Clustering method: 'hdbscan', 'kmeans', or 'agglomerative'.
        use_reduced : bool
            Whether to cluster on reduced dimensions (if available).
        **kwargs
            Additional arguments passed to the clustering algorithm.
            
        Returns
        -------
        EmbeddingResult
            Updated result with cluster_labels populated.
        """
        vectors = (
            result.reduced_vectors 
            if use_reduced and result.reduced_vectors is not None 
            else result.vectors
        )
        
        if method == "hdbscan":
            import hdbscan
            
            defaults = {
                "min_cluster_size": max(2, len(vectors) // 20),
                "min_samples": 1,
                "metric": "euclidean",
            }
            defaults.update(kwargs)
            clusterer = hdbscan.HDBSCAN(**defaults)
            labels = clusterer.fit_predict(vectors)
            
        elif method == "kmeans":
            from sklearn.cluster import KMeans
            
            defaults = {
                "n_clusters": min(10, len(vectors) // 5),
                "random_state": 42,
                "n_init": 10,
            }
            defaults.update(kwargs)
            clusterer = KMeans(**defaults)
            labels = clusterer.fit_predict(vectors)
            
        elif method == "agglomerative":
            from sklearn.cluster import AgglomerativeClustering
            
            defaults = {
                "n_clusters": min(10, len(vectors) // 5),
            }
            defaults.update(kwargs)
            clusterer = AgglomerativeClustering(**defaults)
            labels = clusterer.fit_predict(vectors)
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        result.cluster_labels = labels
        return result
    
    def fit_transform(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        n_components: int = 2,
        reduction_method: Literal["umap", "pca"] = "umap",
        cluster_method: Literal["hdbscan", "kmeans", "agglomerative"] = "hdbscan",
        **kwargs,
    ) -> EmbeddingResult:
        """
        Full pipeline: embed, reduce, and cluster texts.
        
        Parameters
        ----------
        texts : List[str]
            Texts to process.
        ids : List[str], optional
            Identifiers for each text.
        n_components : int
            Dimensions for reduction.
        reduction_method : str
            Method for dimensionality reduction.
        cluster_method : str
            Method for clustering.
        **kwargs
            Additional arguments for sub-methods.
            
        Returns
        -------
        EmbeddingResult
            Complete result with embeddings, reduced vectors, and clusters.
        """
        result = self.embed(texts, ids=ids)
        result = self.reduce_dimensions(result, n_components=n_components, method=reduction_method)
        result = self.cluster(result, method=cluster_method)
        return result


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def pairwise_cosine_similarity(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
    return sklearn_cosine(vectors)
