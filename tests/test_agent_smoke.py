"""Smoke test: can the qwen3 agent actually drive the LangChain tools?

This is a SMOKE TEST, not a correctness test. It runs against whatever is in the
index (empty is fine) and asserts only that the LLM -> tool-call -> response loop
runs end-to-end (a tool was called). The query the LLM chose and the final answer
are PRINTED for you to eyeball; their quality is not asserted. Needs a live LLM.

Run:  python tests/test_agent_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import ask, build_agent  # noqa: E402
from src.llm import get_llm  # noqa: E402


def main() -> int:
    print("== 1. LLM smoke test ==")
    llm = get_llm()
    print("reply:", repr(llm.invoke("Reply with exactly the word: ready").content)[:200])

    print("\n== 2. Agent tool-calling loop ==")
    agent = build_agent()
    user_msg = "find the slides about how the cloud charges for compute"
    out = agent.invoke({"messages": [{"role": "user", "content": user_msg}]})

    print(f"\nUSER asked: {user_msg!r}\n")
    saw_tool_call = False
    for m in out["messages"]:
        for tc in getattr(m, "tool_calls", []) or []:
            saw_tool_call = True
            print(f"  TOOL CALL -> {tc['name']}  ARGS={tc['args']}")

    print("\nFINAL ANSWER:")
    print(out["messages"][-1].content[:600])

    print("\n== result ==")
    print("tool was called:", saw_tool_call)
    return 0 if saw_tool_call else 1


if __name__ == "__main__":
    raise SystemExit(main())
