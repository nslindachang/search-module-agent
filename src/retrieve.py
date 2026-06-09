"""Search + rank: turn a fuzzy query into ranked files with pages to snapshot.

Pipeline (see the README/notebook for the worked example):
  1. similarity search  -> top-K candidate CHUNKS (query vector vs chunk vectors)
  2. optional rerank    -> cross-encoder re-scores (query, chunk) pairs
  3. aggregate by FILE  -> file score = its best chunk (max-pooling) -> rank files
  4. aggregate by PAGE  -> per file, take the top-N pages to render
"""

from __future__ import annotations

import math
from functools import lru_cache

from .config import (
    MAX_FILES,
    MAX_PAGES_PER_FILE,
    RERANK_MODEL,
    TOP_K_CHUNKS,
)
from .index import get_store
from .render import render_pages

_SNIPPET_LEN = 300


@lru_cache(maxsize=1)
def get_reranker():
    """Local cross-encoder reranker, or None if disabled (NUS_RERANK_MODEL='')."""
    if not RERANK_MODEL:
        return None
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text[:_SNIPPET_LEN] + ("..." if len(text) > _SNIPPET_LEN else "")


def search(
    query: str,
    *,
    k: int = TOP_K_CHUNKS,
    max_files: int = MAX_FILES,
    max_pages: int = MAX_PAGES_PER_FILE,
    rerank: bool = True,
) -> list[dict]:
    """Return ranked file matches (without images). See module docstring."""
    store = get_store()
    hits = store.similarity_search_with_score(query, k=k)  # [(Document, distance)]
    if not hits:
        return []

    # --- 1/2. per-chunk score (rerank if available, else cosine) -------------
    reranker = get_reranker() if rerank else None
    if reranker is not None:
        pairs = [[query, doc.page_content] for doc, _ in hits]
        logits = reranker.predict(pairs)
        chunk_scores = [_sigmoid(float(s)) for s in logits]
    else:
        # Chroma cosine "score" is a distance (0 = identical); flip to similarity.
        chunk_scores = [1.0 - float(dist) for _, dist in hits]

    # --- 3/4. aggregate chunks -> pages -> files -----------------------------
    files: dict[str, dict] = {}
    for (doc, _), score in zip(hits, chunk_scores):
        meta = doc.metadata
        src = meta["source"]
        f = files.setdefault(
            src,
            {
                "file_name": meta["file_name"],
                "source": src,
                "pdf_path": meta["pdf_path"],
                "module": meta.get("module", ""),
                "file_type": meta.get("file_type", ""),
                "score": score,
                "_pages": {},  # page_no -> {"score", "snippet"}
            },
        )
        f["score"] = max(f["score"], score)  # file score = best chunk (max-pool)

        page_no = meta["page"]
        page = f["_pages"].get(page_no)
        if page is None or score > page["score"]:
            f["_pages"][page_no] = {
                "score": score,
                "snippet": _snippet(doc.page_content),
            }

    # finalise: pick top pages per file, sort files by score
    results = []
    for f in files.values():
        pages = sorted(
            (
                {"page": p, "score": v["score"], "snippet": v["snippet"]}
                for p, v in f["_pages"].items()
            ),
            key=lambda x: x["score"],
            reverse=True,
        )[:max_pages]
        f.pop("_pages")
        f["pages"] = pages
        results.append(f)

    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:max_files]
    for rank, f in enumerate(results, start=1):
        f["rank"] = rank
    return results


def attach_snapshots(results: list[dict], *, max_pages: int = MAX_PAGES_PER_FILE) -> list[dict]:
    """Render up to `max_pages` page images per file and attach their paths."""
    for f in results:
        pages = [p["page"] for p in f["pages"][:max_pages]]
        f["snapshots"] = render_pages(
            f["pdf_path"], pages, label=f["file_name"]
        )
    return results


def search_and_render(query: str, **kwargs) -> list[dict]:
    """Convenience: search then render snapshots in one call."""
    max_pages = kwargs.get("max_pages", MAX_PAGES_PER_FILE)
    results = search(query, **kwargs)
    return attach_snapshots(results, max_pages=max_pages)
