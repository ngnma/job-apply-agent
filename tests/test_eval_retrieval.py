import json
from pathlib import Path

import pytest
from langchain_core.documents import Document
from sqlalchemy import create_engine, text

from job_agent.evals.retrieval import (
    DATASET,
    hit_at_k,
    load_dataset,
    recall_at_k,
    reciprocal_rank,
    run_mode,
    save_report,
    unknown_ids,
)
from job_agent.rag.retriever import CVRetriever, create_keyword_index
from job_agent.rag.vectorstore import (
    create_table,
    database_url,
    get_engine,
    get_vector_store,
)
from tests.test_vectorstore import FakeEmbeddings, _db_is_up

# ---------- metrics (no database) ----------


def test_hit_at_k():
    assert hit_at_k(["a", "b", "c"], {"c"}, k=3) == 1.0
    assert hit_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0


def test_recall_at_k():
    assert recall_at_k(["a", "x", "b"], {"a", "b", "c", "d"}, k=3) == 0.5
    assert recall_at_k(["a"], set(), k=3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a"], {"a"}) == 0.5
    assert reciprocal_rank(["a"], {"a"}) == 1.0
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_real_dataset_is_valid():
    dataset = load_dataset(Path(DATASET))
    assert len(dataset) >= 15
    for item in dataset:
        assert item["query"].strip()
        assert item["relevant"], f"no relevant ids for: {item['query']}"


# ---------- full run (real Postgres, fake embeddings) ----------

TEST_TABLE = "test_eval_chunks"
db = pytest.mark.skipif(not _db_is_up(), reason="Postgres is not running (docker compose up -d db)")

DOCS = [
    Document(id="a", page_content="Python pipeline with Airflow", metadata={"type": "experience", "parent_id": "a"}),
    Document(id="b", page_content="Causal study on AWS", metadata={"type": "project", "parent_id": "b"}),
    Document(id="c", page_content="Skill: XGBoost", metadata={"type": "skill", "parent_id": "c"}),
]
TINY_DATASET = [
    {"query": "airflow", "relevant": ["a"]},
    {"query": "XGBoost", "relevant": ["c"]},  # only keyword search can find this one
]


@pytest.fixture
def retriever():
    engine = get_engine()
    create_table(engine, vector_size=4, table_name=TEST_TABLE, overwrite=True)
    store = get_vector_store(engine, FakeEmbeddings(), table_name=TEST_TABLE)
    store.add_documents(DOCS, ids=[d.id for d in DOCS])
    create_keyword_index(TEST_TABLE)
    yield CVRetriever(store, table_name=TEST_TABLE)
    with create_engine(database_url()).begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS "{TEST_TABLE}"'))


@db
def test_run_mode_keyword_vs_hybrid(retriever):
    keyword = run_mode(retriever, TINY_DATASET, "keyword", k=1)
    hybrid = run_mode(retriever, TINY_DATASET, "hybrid", k=3)
    assert keyword.hit == 1.0          # both queries contain a word from the right chunk
    assert hybrid.recall == 1.0
    assert hybrid.queries[1].missed == []


@db
def test_unknown_ids(retriever):
    assert unknown_ids([{"query": "q", "relevant": ["a", "zzz"]}], table_name=TEST_TABLE) == ["zzz"]


@db
def test_save_report(retriever, tmp_path):
    path = save_report([run_mode(retriever, TINY_DATASET, "hybrid", k=3)], out_dir=tmp_path)
    saved = json.loads(path.read_text())
    assert saved[0]["mode"] == "hybrid" and len(saved[0]["queries"]) == 2
