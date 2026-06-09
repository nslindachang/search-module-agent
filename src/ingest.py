"""Discover course files, normalise them to PDF, and split into page-tagged chunks.

The key trick: convert every Office file to PDF once (cached), so a single code
path (PyMuPDF) handles per-page text extraction AND page rendering. Every chunk
carries the page number it came from, which is what lets us snapshot the match.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    CACHE_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_ROOT,
    SOFFICE_BIN,
    SUPPORTED_EXTS,
)


def discover_files(data_root: str | Path = DATA_ROOT) -> list[Path]:
    """Recursively find all supported documents under the data root."""
    root = Path(data_root)
    files: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name.startswith("~$") or p.name.startswith("."):
            continue  # Office lock files, hidden files
        if p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
    return files


def _module_of(path: Path, data_root: str | Path) -> str:
    """The top-level module folder a file belongs to (e.g. 'cs5224')."""
    try:
        rel = path.resolve().relative_to(Path(data_root).resolve())
        return rel.parts[0] if len(rel.parts) > 1 else "(root)"
    except ValueError:
        return path.parent.name


def to_pdf(path: str | Path) -> Path:
    """Return a renderable PDF for any supported file.

    PDFs are returned as-is; Office files are converted via LibreOffice and
    cached by (path, mtime) so unchanged files are never re-converted.
    """
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return path

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(
        f"{path.resolve()}:{path.stat().st_mtime_ns}".encode()
    ).hexdigest()
    target = CACHE_DIR / f"{key}.pdf"
    if target.exists():
        return target

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            SOFFICE_BIN,
            "--headless",
            f"-env:UserInstallation=file://{tmp}/lo_profile",
            "--convert-to",
            "pdf",
            "--outdir",
            tmp,
            str(path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        produced = sorted(Path(tmp).glob("*.pdf"))
        if not produced:
            raise RuntimeError(f"LibreOffice produced no PDF for {path}")
        shutil.move(str(produced[0]), target)
    return target


def load_pages(pdf_path: str | Path) -> list[tuple[int, str]]:
    """Extract text per page. Returns [(page_number_1based, text), ...]."""
    pages: list[tuple[int, str]] = []
    with fitz.open(pdf_path) as doc:
        for i in range(doc.page_count):
            pages.append((i + 1, doc.load_page(i).get_text("text")))
    return pages


def file_to_documents(
    path: str | Path, data_root: str | Path = DATA_ROOT
) -> list[Document]:
    """Convert one file into page-tagged, chunk-sized LangChain Documents."""
    path = Path(path)
    pdf_path = to_pdf(path)
    module = _module_of(path, data_root)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    docs: list[Document] = []
    for page_no, text in load_pages(pdf_path):
        text = (text or "").strip()
        if not text:
            continue  # blank or image-only page (no extractable text)
        for ci, chunk in enumerate(splitter.split_text(text)):
            chunk = chunk.strip()
            if len(chunk) < 20:
                continue  # skip tiny fragments (page numbers, stray headers)
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": str(path.resolve()),
                        "file_name": path.name,
                        "module": module,
                        "file_type": path.suffix.lower().lstrip("."),
                        "page": page_no,
                        "pdf_path": str(Path(pdf_path).resolve()),
                        "chunk": ci,
                    },
                )
            )
    return docs


def doc_id(meta: dict) -> str:
    """Deterministic id for a chunk, so re-indexing upserts instead of duplicating."""
    raw = f"{meta['source']}::p{meta['page']}::c{meta['chunk']}"
    return hashlib.md5(raw.encode()).hexdigest()
