"""Integration tests against a real Postgres + pgvector.

They run when the database is up (docker compose up -d db) and are SKIPPED otherwise,
so `pytest` still passes on a laptop without Docker. A fake embedding model is used,
so Ollama is not needed. Tests use their own table and delete it afterwards.
"""

import math

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from job_agent.rag.vectorstore import (
    count_rows,
    create_hnsw_index,
    create_table,
    database_url,
    get_engine,
    get_vector_store,
    table_exists,
)

TEST_TABLE = "test_cv_chunks"


def _db_is_up() -> bool:
    try:
        with create_engine(database_url()).connect():
            return True
    except OperationalError:
        return False


pytestmark = pytest.mark.skipif(not _db_is_up(), reason="Postgres is not running (docker compose up -d db)")


class FakeEmbeddings(Embeddings):
    """4-number vectors built from keyword counts (normalised for cosine)."""

    WORDS = ("python", "airflow", "aws", "causal")

    def _vec(self, text: str) -> list[float]:
        v = [text.lower().count(w) + 0.01 for w in self.WORDS]
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


DOCS = [
    Document(id="exp-acme-b1", page_content="Built a Python pipeline with Airflow",
             metadata={"type": "experience", "parent_id": "exp-acme", "skills": ["Python", "Airflow"]}),
    Document(id="proj-study-b1", page_content="Causal inference study deployed on AWS",
             metadata={"type": "project", "parent_id": "proj-study", "skills": ["AWS"]}),
    Document(id="skill-python", page_content="Skill: Python (category: Programming Languages)",
             metadata={"type": "skill", "parent_id": "skill-python", "skills": ["Python"]}),
]


@pytest.fixture
def store():
    engine = get_engine()
    create_table(engine, vector_size=4, table_name=TEST_TABLE, overwrite=True)
    vs = get_vector_store(engine, FakeEmbeddings(), table_name=TEST_TABLE)
    yield vs
    with create_engine(database_url()).begin() as conn:  # clean up
        conn.execute(text(f'DROP TABLE IF EXISTS "{TEST_TABLE}"'))


def test_table_is_created(store):
    assert table_exists(table_name=TEST_TABLE)
    assert count_rows(table_name=TEST_TABLE) == 0


def test_add_and_search(store):
    store.add_documents(DOCS, ids=[d.id for d in DOCS])
    assert count_rows(table_name=TEST_TABLE) == 3
    best = store.similarity_search("causal inference", k=1)[0]
    assert best.id == "proj-study-b1"
    assert best.metadata["skills"] == ["AWS"]  # metadata comes back


def test_same_ids_update_instead_of_duplicating(store):
    store.add_documents(DOCS, ids=[d.id for d in DOCS])
    store.add_documents(DOCS, ids=[d.id for d in DOCS])  # run twice
    assert count_rows(table_name=TEST_TABLE) == 3


def test_filter_by_type(store):
    store.add_documents(DOCS, ids=[d.id for d in DOCS])
    results = store.similarity_search("python", k=3, filter={"type": "skill"})
    assert [d.id for d in results] == ["skill-python"]


def test_hnsw_index(store):
    create_hnsw_index(store)
    assert store.is_valid_index()
    create_hnsw_index(store)  # second call does nothing (no error)
