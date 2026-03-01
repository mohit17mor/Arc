"""
Arc LLM Providers — Layer 1.

Adapters for different LLM backends.
All providers implement the same LLMProvider interface.

Supported providers:
    - Ollama (local, free)
    - OpenAI, OpenRouter, Groq, Together, LM Studio (OpenAI-compatible)
    - Mock (testing)

Quick usage::

    from arc.llm.factory import create_llm

    llm = create_llm("openrouter", model="anthropic/claude-sonnet-4-20250514", api_key="sk-or-...")
    llm = create_llm("ollama", model="llama3.2")
"""