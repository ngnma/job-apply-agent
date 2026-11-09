"""One place to create the chat LLM, so every part of the project uses the same settings.

Default: a local model in Ollama. Switch with env vars, no code change:
    LLM_PROVIDER=ollama   LLM_MODEL=qwen3:14b      (default)
    LLM_PROVIDER=bedrock  LLM_MODEL=<bedrock model id>   (later, on AWS)
"""

import os

from langchain_core.language_models import BaseChatModel

DEFAULT_MODELS = {"ollama": "qwen3:14b", "bedrock": ""}


def get_llm(model: str | None = None, provider: str | None = None, temperature: float = 0) -> BaseChatModel:
    provider = provider or os.getenv("LLM_PROVIDER", "ollama")
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"Unknown LLM provider {provider!r}. Use one of {list(DEFAULT_MODELS)}")
    model = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider]

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            temperature=temperature,  # 0 = same input -> same output
            num_ctx=16384,            # Ollama's default context is small and silently cuts long input
            reasoning=False,          # no "thinking" mode: faster, plain answers
        )

    from langchain_aws import ChatBedrockConverse  # only needed on AWS

    if not model:
        raise ValueError("Set LLM_MODEL to a Bedrock model id")
    return ChatBedrockConverse(model=model, temperature=temperature)
