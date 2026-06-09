"""Central configuration for the NUS module search agent.

Everything tunable lives here and can be overridden with environment variables,
so the rest of the code never hardcodes a path, model, or provider.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Project paths -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load machine-specific settings (data path, API keys) from a local .env if one
# exists. Real environment variables take precedence over .env, and python-dotenv
# is optional — without it, plain env vars still work. Never commit .env.
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ModuleNotFoundError:
    pass

# Root folder that holds the per-module folders (cs5224, cs5228, ...).
# Override with NUS_DATA_ROOT; the default is a neutral path under the home dir.
DATA_ROOT = Path(
    os.environ.get(
        "NUS_DATA_ROOT",
        Path.home() / "nus-courses",
    )
)

CACHE_DIR = Path(os.environ.get("NUS_CACHE_DIR", PROJECT_ROOT / ".cache"))      # Office -> PDF
CHROMA_DIR = Path(os.environ.get("NUS_CHROMA_DIR", PROJECT_ROOT / ".chroma"))   # vector store
SNAPSHOT_DIR = Path(os.environ.get("NUS_SNAPSHOT_DIR", PROJECT_ROOT / "snapshots"))  # PNGs
MANIFEST_PATH = CHROMA_DIR / "manifest.json"  # tracks what's been indexed

# --- File types we index -----------------------------------------------------
SUPPORTED_EXTS = {".pdf", ".docx", ".doc", ".pptx", ".ppt"}
OFFICE_EXTS = {".docx", ".doc", ".pptx", ".ppt"}  # need LibreOffice -> PDF

# --- LibreOffice (Office -> PDF) ---------------------------------------------
SOFFICE_BIN = os.environ.get(
    "SOFFICE_BIN",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
)

# --- Embeddings: local sentence-transformers ---------------------------------
# Swap to a faster/smaller model (e.g. "sentence-transformers/all-MiniLM-L6-v2")
# or a stronger one by changing this one value. Re-index after changing it.
EMBED_MODEL = os.environ.get("NUS_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Optional local cross-encoder reranker. Set NUS_RERANK_MODEL="" to disable.
RERANK_MODEL = os.environ.get("NUS_RERANK_MODEL", "BAAI/bge-reranker-base")

# --- Vector store ------------------------------------------------------------
COLLECTION_NAME = "nus_modules"

# --- LLM provider (pluggable): "ollama" | "anthropic" | "openai" -------------
LLM_PROVIDER = os.environ.get("NUS_LLM_PROVIDER", "ollama")
LLM_MODEL = os.environ.get("NUS_LLM_MODEL", "qwen3:8b")
LLM_TEMPERATURE = float(os.environ.get("NUS_LLM_TEMPERATURE", "0"))

# --- Chunking ----------------------------------------------------------------
# Text is split *within* each page, so every chunk maps back to exactly one page.
# 1000 chars ~= 250 tokens ~= 180 words: well inside bge-small's 512-token window.
# A short slide stays one chunk; a dense page becomes several (same page number).
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Retrieval / ranking defaults --------------------------------------------
TOP_K_CHUNKS = 40          # chunks pulled from the store before aggregation
MAX_FILES = 8              # number of files returned to the user
MAX_PAGES_PER_FILE = 3     # snapshots per file (requirement: up to 3)

# --- Rendering ---------------------------------------------------------------
RENDER_DPI = 130


def ensure_dirs() -> None:
    """Create the working directories if they don't exist yet."""
    for d in (CACHE_DIR, CHROMA_DIR, SNAPSHOT_DIR):
        d.mkdir(parents=True, exist_ok=True)
