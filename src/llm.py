"""Pluggable chat-model factory.

Local Ollama is the default for development; Claude or OpenAI can be swapped in
later with one argument (or the NUS_LLM_PROVIDER / NUS_LLM_MODEL env vars).
Every provider returns a LangChain BaseChatModel, so the agent code is identical
regardless of which one is active.
"""

from __future__ import annotations

import os

from .config import LLM_PROVIDER, LLM_TEMPERATURE

# Sensible per-provider default model (over_ridden by NUS_LLM_MODEL or arg).
DEFAULT_MODELS = {
    "ollama": "qwen3:8b",
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
}


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    **kwargs,
):
    """Return a LangChain chat model for the chosen provider."""
    provider = (provider or LLM_PROVIDER).lower()
    model = model or os.environ.get("NUS_LLM_MODEL") or DEFAULT_MODELS.get(provider)
    temperature = LLM_TEMPERATURE if temperature is None else temperature

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=temperature, **kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature, **kwargs)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature, **kwargs)

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. Use 'ollama', 'anthropic', or 'openai'."
    )
