"""Embedding tests use a fake model, so they run without Ollama (e.g. in CI)."""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from job_agent.rag.embeddings import (
    PrefixedEmbeddings,
    cosine_similarity,
    embedding_dimension,
    get_embeddings,
    top_k,
)


class FakeEmbeddings(Embeddings):
    """Records what it was asked to embed; vector = counts of a few keywords."""

    WORDS = ("python", "airflow", "aws", "causal")

    def __init__(self):
        self.seen: list[str] = []

    def _vec(self, text: str) -> list[float]:
        self.seen.append(text)
        return [float(text.lower().count(w)) for w in self.WORDS]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def test_prefixes_are_added():
    fake = FakeEmbeddings()
    emb = PrefixedEmbeddings(fake, doc_prefix="search_document: ", query_prefix="search_query: ")
    emb.embed_documents(["Built ETL"])
    emb.embed_query("ETL")
    assert fake.seen == ["search_document: Built ETL", "search_query: ETL"]


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_embeddings(provider="nope")


def test_default_ollama_model_gets_nomic_prefixes(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    emb = get_embeddings()  # builds the client only; no request is sent
    assert isinstance(emb, PrefixedEmbeddings)
    assert (emb.doc_prefix, emb.query_prefix) == ("search_document: ", "search_query: ")


def test_env_var_chooses_model(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "mxbai-embed-large")
    emb = get_embeddings(provider="ollama")
    assert emb.base.model == "mxbai-embed-large"
    assert emb.doc_prefix == ""  # no known prefix for this model


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0  # no division by zero


def test_top_k_finds_the_matching_chunk():
    chunks = [
        Document(id="a", page_content="Built a Python pipeline with Airflow"),
        Document(id="b", page_content="Causal inference study"),
        Document(id="c", page_content="Deployed on AWS"),
    ]
    emb = FakeEmbeddings()
    vectors = emb.embed_documents([c.page_content for c in chunks])
    results = top_k("causal", chunks, vectors, emb, k=2)
    assert results[0][1].id == "b"
    assert len(results) == 2


def test_embedding_dimension():
    assert embedding_dimension(FakeEmbeddings()) == 4
