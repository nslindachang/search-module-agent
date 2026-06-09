# NUS Module Search Agent

Find the slide or page you half-remember. You describe what you're looking for in
plain language — *"the slide that illustrates how a transformer encoder works"* —
and the agent searches your NUS course files, ranks the matches, and renders the
matching pages as images so you can confirm them at a glance.

---

## Intent

University courses bury you in lecture PDFs, slide decks, and notes spread across
many modules. You often remember *what* you saw but not *which file* or *which
page*. This tool is a **page locator**, not a chatbot or a summarizer:

- **Input:** a fuzzy, natural-language description of content you're trying to find.
- **Output:** a ranked list of the file(s) and page number(s) that best match,
  plus a rendered PNG of each matching page so you can eyeball it.

It is built as a hands-on study of **agents**: the heavy lifting (similarity
search, ranking, rendering) lives in deterministic Python, and an LLM only
decides *which* tool to call and *how* to phrase the answer. Two agent designs
ship side by side so they can be compared:

| File | Design | Use it to learn |
|---|---|---|
| [`src/agent.py`](src/agent.py) | A single LangChain **tool-calling agent** (`create_agent`) with two tools | The classic single-agent loop |
| [`src/graph.py`](src/graph.py) | A LangGraph **multi-agent supervisor** routing specialist nodes, with a grader → re-query reflection loop | Multi-agent orchestration, shared state, cyclic graphs |

Both sit on the same retrieval/rendering pipeline; only the orchestration differs.

---

## How it works

```
course files (.pdf/.docx/.pptx/...)
      │  ingest.py  ── LibreOffice → PDF (cached) ── PyMuPDF per-page text
      ▼
   chunks tagged with {file, module, page}
      │  index.py   ── sentence-transformers embeddings → Chroma (persistent)
      ▼
   vector store
      │  retrieve.py ── similarity search → optional cross-encoder rerank
      │               → aggregate chunks → pages → files
      ▼
   ranked file matches (with best pages)
      │  render.py  ── PyMuPDF page → PNG snapshot
      ▼
   answer + page images
```

The **multi-agent** variant ([`src/graph.py`](src/graph.py)) wraps that pipeline
in a supervisor graph:

```
START → supervisor → (query | retrieval | render | synthesize)
        query      → supervisor       # fuzzy request → concrete search strings
        retrieval  → grader           # run search, then judge the hits
        grader     → supervisor        #   "weak" → re-query;  "good" → proceed
        render     → supervisor       # render top files to PNG
        synthesize → END              # write the final ranked, cited answer
```

The **grader → re-query** cycle is the point of the multi-agent design: if the
retrieved files don't actually match, the supervisor sends the query agent back
to reformulate (bounded by `MAX_QUERY_ATTEMPTS`).

Key design choices: every Office file is converted to PDF **once** (cached by
path+mtime), so a single code path (PyMuPDF) handles both text extraction and
rendering; every chunk records the **page** it came from, which is what makes
page-level snapshots possible; the index is **incremental** (a manifest of
`{file: mtime}` re-embeds only new/changed files).

---

## Setup

Requires **Python 3.13**, **LibreOffice** (for Office→PDF), and **Ollama** (local
LLM, the default provider).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or requirements.lock.txt for pinned versions
ollama pull qwen3:8b                      # default local LLM
```

Point it at your course folder and build the index:

```bash
export NUS_DATA_ROOT="/path/to/your/course/files"   # default: ~/nus-courses
python -c "from src.index import build_index; build_index()"
```

The data root is expected to hold one folder per module (`cs5224/`, `cs5228/`, …);
the top-level folder name is used as the module tag.

---

## Usage

### Notebook (recommended starting point)

[`notebooks/nus_module_agent.ipynb`](notebooks/nus_module_agent.ipynb) is a guided,
runnable walkthrough of the whole system — and the easiest way to *see* it work,
since matching pages render inline. It steps through each layer in order:

| Section | What it shows |
|---|---|
| 0–1 | Prerequisites; embeddings (text → meaning-vector) |
| 2 | Building the Chroma index |
| 3 | Retrieval + ranking (chunks → files → pages) |
| 4 | Rendering matching pages as inline snapshots |
| 5–6 | The pluggable LLM and the agent's tools |
| 7 | Running the single agent **and** the multi-agent graph (`src.graph.run`) |
| 8 | Notes |

```bash
pip install -r requirements.txt   # includes jupyter + ipykernel
jupyter lab notebooks/nus_module_agent.ipynb
```

The first cell adds the project root to `sys.path`, so `import src...` works from
inside `notebooks/`. Model loads (embeddings, Ollama) are slow on the **first**
cell that uses them and cached thereafter; if you change `NUS_EMBED_MODEL`,
restart the kernel and rebuild the index.

### Scripted

**Single agent** ([`src/agent.py`](src/agent.py)):

```python
from src.agent import build_agent, ask
agent = build_agent()
print(ask(agent, "find the slides about how the cloud charges for compute"))
```

**Multi-agent graph** ([`src/graph.py`](src/graph.py)) — returns the full state, so
you can inspect what each node did:

```python
from src.graph import run
state = run("find the slide that illustrates how a transformer encoder works")
print(state["queries"])                       # what the query agent searched for
print(state["grade"], state["grade_reason"])  # the grader's verdict
print(state["answer"])                         # final ranked answer
```

**Watch the agents route** (smoke tests — they run end-to-end and print output for you to eyeball; they need a live LLM):

```bash
python tests/test_graph_smoke.py   # streams the graph node-by-node
python tests/test_agent_smoke.py   # the single-agent tool-calling loop
```

Rendered page images land in [`snapshots/`](snapshots/) as `<file>_p<page>.png`.

---

## Configuration

Everything tunable lives in [`src/config.py`](src/config.py) and is overridable by
environment variable. The most useful knobs:

| Env var | Default | Purpose |
|---|---|---|
| `NUS_DATA_ROOT` | `~/nus-courses` | Where your course files live |
| `NUS_LLM_PROVIDER` | `ollama` | `ollama` \| `anthropic` \| `openai` |
| `NUS_LLM_MODEL` | `qwen3:8b` | Chat model for the agent |
| `NUS_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model (re-index after changing) |
| `NUS_RERANK_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder reranker; set `""` to disable |
| `SOFFICE_BIN` | macOS LibreOffice path | LibreOffice binary for Office→PDF |

The LLM provider is pluggable: local Ollama is the default for development, and
Claude or OpenAI can be swapped in with one env var (or one argument to
`get_llm`). Cloud providers need their respective API keys in the environment.

---

## Assumptions

- **Folder layout:** the data root contains one folder per module; that top-level
  folder name becomes the `module` tag. Files directly in the root are tagged
  `(root)`.
- **File types:** `.pdf`, `.docx`, `.doc`, `.pptx`, `.ppt` (see `SUPPORTED_EXTS`).
  Office files are converted to PDF via LibreOffice; anything else is ignored.
- **Local tooling present:** LibreOffice is installed for Office conversion, and
  Ollama is running with the configured model pulled (for the default provider).
- **Text-bearing content:** retrieval is over **extracted page text**. Slides are
  assumed to carry enough text (titles, labels, bullet points) to be findable.
- **English content:** the default embedding/rerank models are English (`bge-*`).
- **Local-first:** designed to run entirely offline on a laptop by default; cloud
  models are opt-in.

---

## Limitations

- **Image-only slides are not retrievable.** Because retrieval indexes *extracted
  text*, a slide that is a pure diagram/figure with little or no text won't become
  a search candidate at all. This is a **recall** gap that grading cannot fix
  (the grader only re-ranks what retrieval already found). Addressing it requires
  index-time changes — e.g. vision-model captions, OCR, or image embeddings
  (CLIP/ColPali). Not yet implemented.
- **macOS Full Disk Access.** Reading files under `~/Documents` from inside an
  IDE's extension host (e.g. Cursor) raises `Operation not permitted` until you
  grant **Full Disk Access** to the host app and fully relaunch it (Cmd-Q, reopen
  — TCC only applies to freshly launched processes). The render step skips
  unreadable files rather than crashing, so you'll see `render skipped …` lines if
  this isn't sorted.
- **LibreOffice fidelity.** Office→PDF conversion can reflow layouts; rendered
  pages may differ slightly from how the file looks in PowerPoint/Word.
- **Local-model quality.** Small local models (`qwen3:8b`) can occasionally choose
  weaker search queries or misjudge relevance; swapping in a cloud model improves
  this at the cost of an API key and sending content to the provider.
- **No OCR.** Text baked into images (scanned PDFs, text inside figures) is not
  extracted, so those pages are effectively invisible to search.
- **One language / one corpus.** Tuned for an English, lecture-style corpus; other
  languages or document types may need a different embedding model and chunking.
- **Not a Q&A system.** It locates pages; it does not answer questions *about* the
  content or summarize it.
```
