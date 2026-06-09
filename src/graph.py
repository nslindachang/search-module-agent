"""Multi-agent version of the search assistant, built directly on LangGraph.

This is the same task as `agent.py` (fuzzy query -> ranked files -> page
snapshots), but instead of one `create_agent` loop it uses a **supervisor**
that routes between specialist nodes. The point is to make the moving parts of
a LangGraph multi-agent app visible:

    * a shared, typed ``State`` threaded between nodes
    * a supervisor that *routes* (the hub of a supervisor architecture)
    * a reflection loop: grader -> supervisor -> re-query  (a cyclic edge,
      the thing a single `create_agent` call hides from you)

Node roster (4 LLM nodes + 2 deterministic nodes):

    supervisor   (LLM)  routes to the next worker, or to "synthesize" when done
    query_agent  (LLM)  fuzzy request -> 1-3 concrete search strings
    retrieval    (det.) runs retrieve.search() for each query, merges hits
    grader       (LLM)  do the hits actually answer the request? good / weak
    render       (det.) renders page PNGs for the top files
    synthesize   (LLM)  writes the final ranked, cited answer

Graph shape (workers always return to the supervisor -- the supervisor stays
the orchestrator)::

    START -> supervisor -> (query | retrieval | render | synthesize)
             query     -> supervisor
             retrieval -> grader -> supervisor
             render    -> supervisor
             synthesize -> END

Run it:  python tests/test_graph_smoke.py
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from . import retrieve
from .config import MAX_FILES, MAX_PAGES_PER_FILE
from .llm import get_llm

# How many times the grader is allowed to send us back to reformulate before we
# give up and render whatever we have. Stops the reflection loop from spinning.
MAX_QUERY_ATTEMPTS = 2


# --- Shared state ------------------------------------------------------------
# Every node receives this dict and returns a partial update to it. Keeping the
# state explicit (rather than hidden inside an agent's message history) is the
# whole point of the exercise -- you can print it at any step.
class State(TypedDict, total=False):
    user_request: str          # the original, possibly-vague request
    queries: list[str]         # set by query_agent
    hits: list[dict]           # set by retrieval (ranked file matches)
    grade: str                 # "good" | "weak", set by grader
    grade_reason: str          # why the grader judged it that way
    attempts: int              # how many times query_agent has run
    snapshots: list[dict]      # set by render
    answer: str                # final text, set by synthesize
    route: str                 # supervisor's last routing choice (for tracing)


# --- Structured-output schemas (so local models stay parseable) --------------
class _Route(BaseModel):
    """Which node should run next."""

    next: Literal["query", "retrieval", "render", "synthesize"]


class _Queries(BaseModel):
    queries: list[str] = Field(description="1-3 concise search strings")


class _Grade(BaseModel):
    grade: Literal["good", "weak"]
    reason: str = Field(description="one sentence: why")


# --- Helpers -----------------------------------------------------------------
def _valid_routes(state: State) -> list[str]:
    """The routes that actually make sense given the current state.

    The supervisor LLM chooses among these; constraining the menu keeps a small
    local model from picking a nonsensical next step (and looping forever).
    The same logic could be written as plain conditional edges -- doing it in
    the supervisor is what makes this a *supervisor* architecture.
    """
    if not state.get("queries"):
        return ["query"]
    if not state.get("hits"):
        return ["retrieval"]
    if state.get("grade") == "weak" and state.get("attempts", 0) < MAX_QUERY_ATTEMPTS:
        return ["query"]          # reflect & reformulate
    if not state.get("snapshots"):
        return ["render"]
    return ["synthesize"]


def _hits_digest(hits: list[dict]) -> str:
    """Compact, LLM-readable summary of the current hits."""
    lines = []
    for h in hits:
        pages = ", ".join(str(p["page"]) for p in h.get("pages", []))
        lines.append(
            f'- {h["file_name"]} [{h.get("module", "")}] score={h["score"]:.2f} pages: {pages}'
        )
        for p in h.get("pages", [])[:1]:  # one snippet is enough for grading
            lines.append(f'    {p["snippet"]}')
    return "\n".join(lines) if lines else "(no hits)"


def _merge_hits(per_query: list[list[dict]]) -> list[dict]:
    """Union the results of several queries, keeping each file's best score."""
    merged: dict[str, dict] = {}
    for results in per_query:
        for r in results:
            src = r["source"]
            if src not in merged or r["score"] > merged[src]["score"]:
                merged[src] = r
    out = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:MAX_FILES]
    for rank, r in enumerate(out, start=1):
        r["rank"] = rank
    return out


# --- Nodes -------------------------------------------------------------------
def supervisor(state: State) -> dict:
    """Route to the next worker. The hub of the graph."""
    options = _valid_routes(state)
    if len(options) == 1:
        # Only one sensible move -- no need to spend an LLM call deciding.
        return {"route": options[0]}

    llm = get_llm().with_structured_output(_Route)
    prompt = (
        "You are the supervisor of a document-search assistant. Pick the next "
        "step.\n\n"
        f"User request: {state['user_request']!r}\n"
        f"Have queries: {bool(state.get('queries'))}\n"
        f"Have hits: {bool(state.get('hits'))}\n"
        f"Grade: {state.get('grade', '(ungraded)')}\n"
        f"Rendered snapshots: {bool(state.get('snapshots'))}\n\n"
        f"Valid next steps right now: {options}. Choose exactly one of them."
    )
    choice = llm.invoke(prompt).next
    if choice not in options:        # local model went off-menu -> safe fallback
        choice = options[0]
    return {"route": choice}


def query_agent(state: State) -> dict:
    """Turn the (vague) request into concrete search strings.

    On a re-query (grade == "weak") it also sees the previous queries and the
    grader's complaint, so it can reformulate rather than repeat itself.
    """
    llm = get_llm().with_structured_output(_Queries)
    prompt = (
        "The user is trying to LOCATE the page(s)/slide(s) that best match what "
        "they describe -- they remember the content but not which file or page "
        "it's in (it may be a single slide or several). Convert their "
        "description into 1-3 concise search queries over a set of university "
        "lecture files. Keep the concrete concept/object they described (e.g. a "
        "diagram, formula, or term); strip filler like 'find me the slide that'."
        f"\n\nRequest: {state['user_request']!r}"
    )
    if state.get("grade") == "weak":
        prompt += (
            f"\n\nYour previous queries {state.get('queries')} were judged weak "
            f"because: {state.get('grade_reason')!r}. Try different wording or "
            "more specific terms."
        )
    queries = [q.strip() for q in llm.invoke(prompt).queries if q.strip()][:3]
    # Reset downstream state so the supervisor re-evaluates from scratch.
    return {
        "queries": queries or [state["user_request"]],
        "attempts": state.get("attempts", 0) + 1,
        "hits": [],
        "grade": "",
        "snapshots": [],
    }


def retrieval(state: State) -> dict:
    """Deterministic worker: run the existing search for each query, merge."""
    per_query = [retrieve.search(q) for q in state["queries"]]
    return {"hits": _merge_hits(per_query)}


def grader(state: State) -> dict:
    """Reflection node: do the hits actually answer the request?"""
    if not state.get("hits"):
        return {"grade": "weak", "grade_reason": "no files matched"}
    llm = get_llm().with_structured_output(_Grade)
    prompt = (
        "The user is looking for the page(s)/slide(s) that best match what they "
        "describe (could be one slide or several). Judge whether the retrieved "
        "files plausibly contain matching pages.\n\n"
        f"Looking for: {state['user_request']!r}\n\n"
        f"Retrieved files (best-matching page snippet each):\n"
        f"{_hits_digest(state['hits'])}\n\n"
        'Reply "good" if at least one file plausibly contains a matching page, '
        'else "weak". Note: the target may be a diagram/figure, so a page whose '
        'text only labels the topic can still be a match.'
    )
    result = llm.invoke(prompt)
    return {"grade": result.grade, "grade_reason": result.reason}


def render(state: State) -> dict:
    """Deterministic worker: render PNGs for the top files (reuses retrieve).

    A file that can't be opened (corrupt, or unreadable due to macOS Full Disk
    Access) is skipped rather than crashing the whole graph -- same policy as
    ingest.py. Its snapshots come back empty and synthesize still runs.
    """
    snaps = []
    for f in state["hits"][:3]:  # snapshot only the few best files
        try:
            retrieve.attach_snapshots([f], max_pages=MAX_PAGES_PER_FILE)
            images = f.get("snapshots", [])
        except Exception as e:  # noqa: BLE001 -- one bad file shouldn't abort the run
            print(f"  ! render skipped {f['file_name']}: {type(e).__name__}: {e}")
            images = []
        snaps.append(
            {"file_name": f["file_name"], "module": f.get("module", ""), "images": images}
        )
    return {"snapshots": snaps}


def synthesize(state: State) -> dict:
    """Write the final ranked, cited answer from the gathered state."""
    llm = get_llm()
    prompt = (
        "The user is trying to find the page(s)/slide(s) that best match what "
        "they describe (could be one or several). Using ONLY the files below, "
        "point them to the matching location(s), best first. For each: file "
        "name, module, the exact page number(s) to look at, and one line on why "
        "it matches. Note that snapshots were rendered so they can eyeball the "
        "pages. Never invent file names or pages.\n\n"
        f"Looking for: {state['user_request']!r}\n\n"
        f"Files:\n{_hits_digest(state['hits'])}"
    )
    return {"answer": llm.invoke(prompt).content}


# --- Graph wiring ------------------------------------------------------------
def build_graph():
    """Compile the supervisor graph. Returns a runnable LangGraph app."""
    g = StateGraph(State)

    g.add_node("supervisor", supervisor)
    g.add_node("query", query_agent)
    g.add_node("retrieval", retrieval)
    g.add_node("grader", grader)
    g.add_node("render", render)
    g.add_node("synthesize", synthesize)

    g.add_edge(START, "supervisor")
    # The supervisor's "route" field decides which worker runs next.
    g.add_conditional_edges(
        "supervisor",
        lambda s: s["route"],
        {
            "query": "query",
            "retrieval": "retrieval",
            "render": "render",
            "synthesize": "synthesize",
        },
    )
    # Workers hand control back to the supervisor (retrieval goes via the grader
    # first -- that detour is the reflection loop).
    g.add_edge("query", "supervisor")
    g.add_edge("retrieval", "grader")
    g.add_edge("grader", "supervisor")
    g.add_edge("render", "supervisor")
    g.add_edge("synthesize", END)

    return g.compile()


def run(query: str) -> dict:
    """Run one request through the graph and return the final state.

    The returned dict still has every intermediate field (queries, hits, grade,
    snapshots, answer) so you can inspect what each agent did.
    """
    app = build_graph()
    return app.invoke(
        {"user_request": query, "attempts": 0},
        config={"recursion_limit": 25},  # belt-and-braces against loops
    )
