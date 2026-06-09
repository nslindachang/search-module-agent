"""Build and maintain the local vector index (sentence-transformers + Chroma).

Indexing is incremental: a manifest of {file: mtime} lets us re-embed only new
or changed files, and drop chunks for files that were deleted or edited.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DATA_ROOT,
    EMBED_MODEL,
    MANIFEST_PATH,
    ensure_dirs,
)
from .ingest import discover_files, doc_id, file_to_documents


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Local sentence-transformers embeddings (cached; the model load is slow)."""
    return HuggingFaceEmbeddings(
        model=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},  # unit vectors -> cosine
    )


def get_store(embeddings: HuggingFaceEmbeddings | None = None) -> Chroma:
    """Open (or create) the persistent Chroma collection."""
    ensure_dirs()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def _load_manifest() -> dict[str, int]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(manifest: dict[str, int]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def _delete_source(store: Chroma, source: str) -> None:
    """Remove all chunks belonging to one source file."""
    existing = store.get(where={"source": source})
    ids = existing.get("ids", [])
    if ids:
        store.delete(ids=ids)


def build_index(
    *,
    force: bool = False,
    data_root: str | Path = DATA_ROOT,
    verbose: bool = True,
) -> dict:
    """Index new/changed files under data_root. Returns a small summary dict."""
    ensure_dirs()
    store = get_store()
    manifest = _load_manifest()

    files = discover_files(data_root)
    current: dict[str, int] = {}
    files_indexed = 0
    chunks_added = 0

    for f in files:
        src = str(f.resolve())
        mtime = f.stat().st_mtime_ns
        current[src] = mtime
        if not force and manifest.get(src) == mtime:
            continue  # unchanged since last run

        _delete_source(store, src)  # clear stale chunks (edited file)
        try:
            docs = file_to_documents(f, data_root)
        except Exception as e:  # one bad file shouldn't kill the whole index
            if verbose:
                print(f"  ! skipped {f.name}: {type(e).__name__}: {e}")
            continue
        if docs:
            ids = [doc_id(d.metadata) for d in docs]
            store.add_documents(docs, ids=ids)
            files_indexed += 1
            chunks_added += len(docs)
            if verbose:
                print(f"  + {f.name}: {len(docs)} chunks")

    # Drop files that no longer exist on disk.
    for src in list(manifest):
        if src not in current:
            _delete_source(store, src)
            if verbose:
                print(f"  - removed {Path(src).name}")

    _save_manifest(current)
    summary = {
        "total_files": len(files),
        "files_indexed_this_run": files_indexed,
        "chunks_added_this_run": chunks_added,
    }
    if verbose:
        print(summary)
    return summary
