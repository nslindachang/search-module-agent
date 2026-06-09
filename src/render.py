"""Render PDF pages (or LibreOffice-converted slides) to PNG snapshots.

Everything that gets here is already a PDF (native, or an Office file converted
to PDF during ingest), so PyMuPDF handles all formats uniformly.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF

from .config import RENDER_DPI, SNAPSHOT_DIR


def _safe(name: str) -> str:
    """Make a filesystem-safe slug from a file stem."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")[:80]


def render_pages(
    pdf_path: str | Path,
    pages: list[int],
    *,
    dpi: int = RENDER_DPI,
    out_dir: str | Path | None = None,
    label: str | None = None,
) -> list[dict]:
    """Render the given 1-based page numbers of a PDF to PNG files.

    Returns a list of {"page": int, "image": str} for pages that exist.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir) if out_dir else SNAPSHOT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = _safe(label or pdf_path.stem)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    results: list[dict] = []
    with fitz.open(pdf_path) as doc:
        for page_no in pages:
            idx = page_no - 1  # fitz pages are 0-based
            if idx < 0 or idx >= doc.page_count:
                continue
            pix = doc.load_page(idx).get_pixmap(matrix=matrix)
            image_path = out_dir / f"{stem}_p{page_no}.png"
            pix.save(image_path)
            results.append({"page": page_no, "image": str(image_path)})
    return results
