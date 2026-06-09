"""Self-test the search pipeline on a synthetic corpus.

Runs ingest -> index -> retrieve -> render end to end WITHOUT touching the real
course files (no Full Disk Access needed) and WITHOUT an LLM (no Ollama needed).
Also exercises the LibreOffice .pptx -> PDF path with one generated slide deck.

Run:  python tests/test_pipeline.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import fitz  # PyMuPDF


def make_pdf(path: Path, pages: list[str]) -> None:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 545, 780), text, fontsize=12)
    doc.save(path)
    doc.close()


def make_pptx(path: Path, slides: list[tuple[str, str]]) -> None:
    from pptx import Presentation

    prs = Presentation()
    for title, body in slides:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
    prs.save(path)


def build_corpus(root: Path) -> None:
    (root / "cs5224").mkdir(parents=True, exist_ok=True)
    (root / "cs5228").mkdir(parents=True, exist_ok=True)
    (root / "cs5346").mkdir(parents=True, exist_ok=True)

    make_pdf(
        root / "cs5224" / "Lecture 10 - Pricing.pdf",
        [
            "Course overview and logistics. Grading, projects, and schedule.",
            "Cloud pricing models. Cloud providers charge for virtual machines "
            "using on-demand instances billed per second, reserved instances for "
            "steady workloads, and spot instances for cheap interruptible capacity.",
            "Total cost of ownership. Capital expenditure versus operational "
            "expenditure when renting compute from a public cloud vendor.",
        ],
    )
    make_pdf(
        root / "cs5224" / "Lecture 05 - Virtualisation.pdf",
        [
            "Virtualisation basics. A hypervisor runs multiple guest virtual "
            "machines on one physical host, enabling multitenancy and isolation.",
            "Containers versus VMs. Containers share the host kernel and are "
            "lighter weight than full virtual machines.",
        ],
    )
    make_pdf(
        root / "cs5228" / "Lecture - Clustering.pdf",
        [
            "Clustering algorithms. K-means partitions points into k clusters by "
            "minimising within-cluster variance. DBSCAN finds density-based clusters.",
            "Evaluating clusters with the silhouette score and the elbow method.",
        ],
    )
    # Office path: a .pptx that must be converted to PDF via LibreOffice.
    make_pptx(
        root / "cs5346" / "Visualisation.pptx",
        [
            ("Bar Charts", "Bar charts compare categorical quantities using rectangular bars."),
            ("Scatter Plots", "Scatter plots reveal correlation between two numeric variables."),
        ],
    )


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="nus_selftest_"))
    data_root = tmp / "corpus"
    data_root.mkdir(parents=True)

    # Point all working dirs at the temp area BEFORE importing the package.
    os.environ["NUS_DATA_ROOT"] = str(data_root)
    os.environ["NUS_CHROMA_DIR"] = str(tmp / "chroma")
    os.environ["NUS_CACHE_DIR"] = str(tmp / "cache")
    os.environ["NUS_SNAPSHOT_DIR"] = str(tmp / "snapshots")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    print(f"corpus: {data_root}")
    build_corpus(data_root)

    from src import index, retrieve

    print("\n--- building index ---")
    summary = index.build_index()
    assert summary["total_files"] == 4, summary
    assert summary["chunks_added_this_run"] > 0, summary

    failures = []

    def check(query, expect_file, expect_module):
        results = retrieve.search_and_render(query)
        print(f"\nQUERY: {query!r}")
        for r in results:
            pages = ", ".join(str(p["page"]) for p in r["pages"])
            print(f'  {r["rank"]}. {r["file_name"]} [{r["module"]}] '
                  f'score={r["score"]:.3f} pages=[{pages}] '
                  f'imgs={len(r.get("snapshots", []))}')
        if not results:
            failures.append(f"{query!r}: no results")
            return
        top = results[0]
        if expect_file not in top["file_name"]:
            failures.append(f"{query!r}: expected {expect_file!r} on top, got {top['file_name']!r}")
        if top["module"] != expect_module:
            failures.append(f"{query!r}: expected module {expect_module}, got {top['module']}")
        for s in top.get("snapshots", []):
            if not Path(s["image"]).exists():
                failures.append(f"{query!r}: snapshot missing on disk: {s['image']}")

    # Fuzzy queries (no exact keyword overlap with the slides).
    check("how do cloud vendors bill you for servers", "Lecture 10 - Pricing", "cs5224")
    check("running many isolated guest OSes on one machine", "Lecture 05 - Virtualisation", "cs5224")
    check("grouping unlabelled data points into k groups", "Lecture - Clustering", "cs5228")
    check("comparing categories with rectangular bars", "Visualisation", "cs5346")  # .pptx path

    print("\n--- incremental re-index (should add 0) ---")
    summary2 = index.build_index()
    assert summary2["chunks_added_this_run"] == 0, summary2

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
