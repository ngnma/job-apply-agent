"""Task 5: turn text (chunks and search queries) into vectors.

Default model: "nomic-embed-text" running locally in Ollama (free, private).
Later in production you can switch to AWS Bedrock with one setting, no code change:
    EMBEDDING_PROVIDER=bedrock  EMBEDDING_MODEL=amazon.titan-embed-text-v2:0

Try it:
    ollama pull nomic-embed-text
    uv run python -m job_agent.rag.embeddings data/master_cv.json
"""

import math
import os
import sys
import time

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODELS = {
    "ollama": "nomic-embed-text",
    "bedrock": "amazon.titan-embed-text-v2:0",
}

# Some embedding models are trained with a different prefix for stored text and for
# search queries. Using them gives better search results.
PREFIXES = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
}


class PrefixedEmbeddings(Embeddings):
    """Wraps any LangChain embedding model and adds the model's document/query prefixes."""

    def __init__(self, base: Embeddings, doc_prefix: str = "", query_prefix: str = ""):
        self.base = base
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base.embed_documents([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.base.embed_query(self.query_prefix + text)


def get_embeddings(provider: str | None = None, model: str | None = None) -> Embeddings:
    """Return the embedding model chosen by arguments, or by env vars, or the defaults."""
    provider = provider or os.getenv("EMBEDDING_PROVIDER", DEFAULT_PROVIDER)
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"Unknown embedding provider: {provider!r}. Use one of {list(DEFAULT_MODELS)}")
    model = model or os.getenv("EMBEDDING_MODEL", DEFAULT_MODELS[provider])

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        base: Embeddings = OllamaEmbeddings(model=model)
    else:  # bedrock — imported only when used, so it is not needed locally
        from langchain_aws import BedrockEmbeddings
        base = BedrockEmbeddings(model_id=model)

    doc_prefix, query_prefix = PREFIXES.get(model, ("", ""))
    return PrefixedEmbeddings(base, doc_prefix, query_prefix)


def embedding_dimension(embeddings: Embeddings) -> int:
    """Length of one vector. The vector DB table needs this number (task 6)."""
    return len(embeddings.embed_query("test"))


# ---------- quick check: search the chunks in memory (the real DB comes in task 6) ----------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def top_k(query: str, chunks: list[Document], vectors: list[list[float]],
          embeddings: Embeddings, k: int = 3) -> list[tuple[float, Document]]:
    q = embeddings.embed_query(query)
    scored = [(cosine_similarity(q, v), c) for c, v in zip(chunks, vectors, strict=True)]
    return sorted(scored, key=lambda pair: pair[0], reverse=True)[:k]


def main() -> None:
    from dotenv import load_dotenv

    from job_agent.rag.chunking import chunk_master_cv, load_master_cv

    load_dotenv()  # read settings from .env, if it exists

    path = sys.argv[1] if len(sys.argv) > 1 else "data/master_cv.json"
    chunks = chunk_master_cv(load_master_cv(path))
    embeddings = get_embeddings()

    start = time.perf_counter()
    vectors = embeddings.embed_documents([c.page_content for c in chunks])
    seconds = time.perf_counter() - start
    print(f"Embedded {len(vectors)} chunks in {seconds:.1f}s, vector size = {len(vectors[0])}\n")

    for query in ("Python data pipelines with Airflow", "causal inference", "deploying models on AWS"):
        print(f"Query: {query}")
        for score, chunk in top_k(query, chunks, vectors, embeddings):
            print(f"  {score:.3f}  {chunk.id}")
        print()


if __name__ == "__main__":
    main()
