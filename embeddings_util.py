from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


def _cache_key(texts: Sequence[str], model_name: str) -> str:
    payload = {"texts": [str(t or "") for t in texts], "model": model_name, "schema": "sbert-v1"}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:20]


def embed_texts(
    texts: Sequence[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    cache_dir: str | Path = "cache",
) -> np.ndarray:
    """Return normalized SBERT sentence embeddings with disk cache."""
    texts_list = [str(t or "") for t in texts]
    if not texts_list:
        return np.zeros((0, 0), dtype=np.float32)

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"emb_{_cache_key(texts_list, model_name)}.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        return np.asarray(data["vectors"], dtype=np.float32)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vectors = model.encode(
        texts_list,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    np.savez(cache_path, vectors=vectors)
    return vectors


def semantic_profile(
    texts: Sequence[str],
    polarities: Sequence[float],
    labels: Sequence[str],
    ids: Sequence[str],
    n_clusters: int | None = None,
    cache_dir: str | Path = "cache",
) -> Dict[str, Any]:
    """Cluster texts semantically and summarize sentiment by cluster.

    Adds research value beyond document-level sentiment by identifying semantic
    subcorpora whose affect profiles differ, plus representative exemplars and
    semantic outliers.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances

    texts_list = [str(t or "") for t in texts]
    n = len(texts_list)
    if n < 3:
        return {"available": False, "reason": "need at least 3 documents"}

    vectors = embed_texts(texts_list, cache_dir=cache_dir)
    if vectors.size == 0:
        return {"available": False, "reason": "embedding failed"}

    if n_clusters is None:
        n_clusters = max(2, min(8, int(round(np.sqrt(n / 2)))))
    n_clusters = max(2, min(int(n_clusters), n))

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(vectors)
    dists = pairwise_distances(vectors, km.cluster_centers_, metric="cosine")
    own_dists = np.asarray([dists[i, cluster_ids[i]] for i in range(n)])

    pol = np.asarray([float(x) for x in polarities], dtype=float)
    clusters = []
    for c in range(n_clusters):
        idx = np.where(cluster_ids == c)[0]
        if len(idx) == 0:
            continue
        exemplar_i = int(idx[np.argmin(own_dists[idx])])
        label_counts = {str(k): int(v) for k, v in zip(*np.unique(np.asarray(labels)[idx], return_counts=True))}
        clusters.append({
            "cluster": int(c),
            "size": int(len(idx)),
            "mean_polarity": round(float(np.mean(pol[idx])), 4),
            "median_polarity": round(float(np.median(pol[idx])), 4),
            "label_distribution": label_counts,
            "exemplar_doc_id": str(ids[exemplar_i]),
            "exemplar_preview": texts_list[exemplar_i][:220],
            "exemplar_distance": round(float(own_dists[exemplar_i]), 4),
        })

    outlier_n = min(10, n)
    outlier_idx = np.argsort(own_dists)[::-1][:outlier_n]
    outliers = [
        {
            "doc_id": str(ids[i]),
            "cluster": int(cluster_ids[i]),
            "distance": round(float(own_dists[i]), 4),
            "polarity": round(float(pol[i]), 4),
            "label": str(labels[i]),
            "preview": texts_list[i][:220],
        }
        for i in outlier_idx
    ]

    return {
        "available": True,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "n_clusters": int(n_clusters),
        "clusters": sorted(clusters, key=lambda x: x["mean_polarity"]),
        "semantic_outliers": outliers,
    }
