"""The LangChain agent: an LLM that orchestrates the search + render tools.

This is the "agent" part of the project. The heavy lifting (similarity, ranking,
rendering) lives in deterministic Python tools; the LLM only decides *which* tool
to call and *how* to phrase the final answer. Built with LangChain 1.x's
`create_agent` (a tool-calling agent compiled on top of LangGraph).
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.tools import tool

from . import retrieve
from .index import get_store
from .llm import get_llm
from .render import render_pages


@tool
def search_courses(query: str) -> str:
    """Search the indexed NUS course files for content matching a (possibly fuzzy)
    query. Returns a ranked list of matching files, each with its module, a
    relevance score, the best-matching page numbers, and a short snippet per page.
    May be called more than once in a turn; each call is independent."""
    results = retrieve.search(query)
    if not results:
        return "No matching files found."
    lines = []
    for r in results:
        pages = ", ".join(str(p["page"]) for p in r["pages"])
        lines.append(
            f'{r["rank"]}. {r["file_name"]}  [{r["module"]}]  '
            f'score={r["score"]:.2f}  pages: {pages}'
        )
        for p in r["pages"]:
            lines.append(f'      p{p["page"]}: {p["snippet"]}')
    return "\n".join(lines)


@tool
def render_file_snapshots(
    file_name: str,
    pages: list[int],
    module: str | None = None,
) -> str:
    """Render the given page numbers of one file from the index. Pass the file_name
    and the list of page numbers (ints) you saw in search_courses output.
    Optionally pass the module to disambiguate when two modules contain a file
    with the same name. Returns a text list of the saved image paths."""
    where: dict = {"file_name": file_name}
    if module:
        where = {"$and": [{"file_name": file_name}, {"module": module}]}
    rows = get_store().get(where=where, limit=1)
    metadatas = rows.get("metadatas") or []
    if not metadatas:
        return f"No indexed file named {file_name!r}."
    pdf_path = metadatas[0]["pdf_path"]
    snaps = render_pages(pdf_path, pages, label=file_name)
    if not snaps:
        return f"Could not render pages {pages} of {file_name!r}."
    paths = "\n".join(f"  p{s['page']}: {s['image']}" for s in snaps)
    return f"Rendered {len(snaps)} image(s) for {file_name}:\n{paths}"


SYSTEM_PROMPT = """You are a study assistant that helps the user find content in their NUS course files.
The user's description of what they want may be vague. To help them:
1. Call search_courses with a concise query that captures their intent (call it more than once if the request has distinct parts).
2. For the top 1-3 most relevant files, call render_file_snapshots(file_name, pages, module) using the file_name, page numbers, and module exactly as they appeared in the search output.
3. Reply with a short ranked summary: for each file give its name, its module, one line on why it matches, the page numbers, and note that snapshots were rendered.
Only mention files returned by the tools - never invent file names or pages."""


def build_agent(provider: str | None = None, model: str | None = None):
    """Construct the tool-calling agent with the chosen (pluggable) LLM."""
    llm = get_llm(provider=provider, model=model)
    return create_agent(
        llm,
        tools=[search_courses, render_file_snapshots],
        system_prompt=SYSTEM_PROMPT,
    )


def ask(agent, query: str) -> str:
    """Run one query through the agent and return its final text answer."""
    out = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return out["messages"][-1].content
