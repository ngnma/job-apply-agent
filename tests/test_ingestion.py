import copy
import json
import math

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from job_agent.rag.ingestion import (
    HASH_KEY,
    content_hash,
    embedding_model_name,
    ingest,
    plan_changes,
)
from job_agent.rag.vectorstore import count_rows, database_url
from tests.test_chunking import FAKE_CV

# ---------- unit tests (no database) ----------


def _doc(id_: str, text_: str = "some text") -> Document:
    doc = Document(id=id_, page_content=text_, metadata={"type": "skill"})
    doc.metadata[HASH_KEY] = content_hash(doc, "model-a")
    return doc


def test_hash_changes_with_text_metadata_or_model():
    doc = Document(id="x", page_content="Built ETL", metadata={"type": "project"})
    base = content_hash(doc, "model-a")
    assert content_hash(doc, "model-a") == base  # same input -> same hash
    assert content_hash(doc, "model-b") != base  # new embedding model
    assert content_hash(Document(id="x", page_content="Built ETL!", metadata={"type": "project"}), "model-a") != base
    assert content_hash(Document(id="x", page_content="Built ETL", metadata={"type": "skill"}), "model-a") != base


def test_plan_changes():
    new, changed, same = _doc("new"), _doc("changed", "edited text"), _doc("same")
    existing = {"changed": "old-hash", "same": same.metadata[HASH_KEY], "removed": "whatever"}
    plan = plan_changes([new, changed, same], existing)
    assert [d.id for d in plan.to_add] == ["new"]
    assert [d.id for d in plan.to_update] == ["changed"]
    assert plan.unchanged == ["same"]
    assert plan.to_delete == ["removed"]


def test_embedding_model_name():
    class Base:
        model = "nomic-embed-text"

    class Wrapper:
        base = Base()

    assert embedding_model_name(Wrapper()) == "nomic-embed-text"


# ---------- integration tests (real Postgres, fake embeddings) ----------

TEST_TABLE = "test_ingestion_chunks"


def _db_is_up() -> bool:
    try:
        with create_engine(database_url()).connect():
            return True
    except OperationalError:
        return False


db = pytest.mark.skipif(not _db_is_up(), reason="Postgres is not running (docker compose up -d db)")


class FakeEmbeddings(Embeddings):
    """Small keyword-count vectors; counts how many texts it embedded."""

    WORDS = ("python", "airflow", "aws", "rag")

    def __init__(self, size: int = 4):
        self.size = size
        self.model = f"fake-{size}"
        self.calls = 0

    def _vec(self, t: str) -> list[float]:
        v = [t.lower().count(w) + 0.01 for w in self.WORDS][: self.size]
        v += [0.01] * (self.size - len(v))
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v]

    def embed_documents(self, texts):
        self.calls += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, t):
        return self._vec(t)


@pytest.fixture
def cv_file(tmp_path):
    path = tmp_path / "master_cv.json"
    path.write_text(json.dumps(FAKE_CV), encoding="utf-8")
    yield path
    with create_engine(database_url()).begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS "{TEST_TABLE}"'))


@db
def test_first_run_adds_everything(cv_file):
    report = ingest(cv_file, FakeEmbeddings(), table_name=TEST_TABLE, rebuild=True)
    assert (report.added, report.updated, report.unchanged, report.deleted) == (10, 0, 0, 0)
    assert count_rows(table_name=TEST_TABLE) == 10


@db
def test_second_run_changes_nothing(cv_file):
    ingest(cv_file, FakeEmbeddings(), table_name=TEST_TABLE, rebuild=True)
    emb = FakeEmbeddings()
    report = ingest(cv_file, emb, table_name=TEST_TABLE)
    assert (report.added, report.updated, report.unchanged, report.deleted) == (0, 0, 10, 0)
    assert emb.calls == 0  # nothing re-embedded


@db
def test_edit_and_remove(cv_file):
    ingest(cv_file, FakeEmbeddings(), table_name=TEST_TABLE, rebuild=True)

    cv = copy.deepcopy(FAKE_CV)
    cv["experiences"][0]["bullets"][0]["text"] = "Built a Python ETL pipeline on AWS with Airflow."  # edited
    cv["skills"] = cv["skills"][:2]                                                              # 1 skill removed
    cv_file.write_text(json.dumps(cv), encoding="utf-8")

    emb = FakeEmbeddings()
    report = ingest(cv_file, emb, table_name=TEST_TABLE)
    assert (report.added, report.updated, report.unchanged, report.deleted) == (0, 1, 8, 1)
    assert emb.calls == 1  # only the edited bullet was re-embedded
    assert count_rows(table_name=TEST_TABLE) == 9


@db
def test_new_embedding_size_needs_rebuild(cv_file):
    ingest(cv_file, FakeEmbeddings(size=4), table_name=TEST_TABLE, rebuild=True)
    with pytest.raises(ValueError, match="--rebuild"):
        ingest(cv_file, FakeEmbeddings(size=3), table_name=TEST_TABLE)
    report = ingest(cv_file, FakeEmbeddings(size=3), table_name=TEST_TABLE, rebuild=True)
    assert report.added == 10
