"""Smoke test: watch the multi-agent supervisor graph route a request end-to-end.

This is a SMOKE TEST, not a correctness test. Unlike test_agent_smoke.py (the
single create_agent loop), this streams the LangGraph app node-by-node so you can
SEE the supervisor pattern: which worker ran, what the query agent rewrote, what
the grader decided, and whether the reflection loop re-queried. It asserts only
that the run reached a final answer; the routing path and outputs are PRINTED for
you to eyeball, not asserted. Needs a live LLM (and a built index for real hits).

Run:  python tests/test_graph_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph import build_graph  # noqa: E402


def main() -> int:
    app = build_graph()
    user_msg = "find the slides about how the cloud charges for compute"
    print(f"USER asked: {user_msg!r}\n")

    path = []
    final: dict = {}
    # stream() yields {node_name: state_update} after each node runs.
    for step in app.stream(
        {"user_request": user_msg, "attempts": 0},
        config={"recursion_limit": 25},
    ):
        for node, update in step.items():
            path.append(node)
            if node == "supervisor":
                print(f"  [supervisor] -> {update.get('route')}")
            elif node == "query":
                print(f"  [query]       queries={update.get('queries')}")
            elif node == "retrieval":
                print(f"  [retrieval]   {len(update.get('hits', []))} file(s)")
            elif node == "grader":
                print(f"  [grader]      {update.get('grade')}: {update.get('grade_reason')}")
            elif node == "render":
                print(f"  [render]      {len(update.get('snapshots', []))} file(s) snapshotted")
            elif node == "synthesize":
                final = update

    print("\nPATH:", " -> ".join(path))
    print("\nFINAL ANSWER:")
    print((final.get("answer") or "(none)")[:600])

    # Success = the graph actually reached a synthesized answer.
    return 0 if final.get("answer") else 1


if __name__ == "__main__":
    raise SystemExit(main())
