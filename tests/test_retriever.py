import pytest
from langchain_core.documents import Document
from sqlalchemy import create_engine, text

from job_agent.rag.retriever import (
    CVRetriever,
    RetrievedChunk,
    build_tsquery,
    create_keyword_index,
    group_by_parent,
    reciprocal_rank_fusion,
)
from job_agent.rag.vectorstore import (
    create_table,
    database_url,
    get_engine,
    get_vector_store,
)
from tests.test_vectorstore import FakeEmbeddings, _db_is_up

# ---------- unit tests (no database) ----------


def test_build_tsquery_uses_or_and_cleans_input():
    assert build_tsquery("Python & Airflow pipelines") == "python | airflow | pipelines"
    assert build_tsquery("Python, python; PYTHON") == "python"  # duplicates removed
    assert build_tsquery("'; DROP TABLE x; --") == "drop | table | x"  # only letters/digits survive
    assert build_tsquery("!!!") == ""


def test_rrf_rewards_chunks_found_by_both_searches():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["c", "d"]], k=60)
    assert max(scores, key=scores.get) == "c"  # 3rd in one list + 1st in the other beats 1st in one list
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)


def test_group_by_parent():
    chunks = [
        RetrievedChunk("exp-a-b2", "", {"parent_id": "exp-a"}, score=0.5),
        RetrievedChunk("proj-b-b1", "", {"parent_id": "proj-b"}, score=0.9),
        RetrievedChunk("exp-a-b1", "", {"parent_id": "exp-a"}, score=0.7),
    ]
    groups = group_by_parent(chunks)
    assert list(groups) == ["proj-b", "exp-a"]  # best group first
    assert [c.id for c in groups["exp-a"]] == ["exp-a-b1", "exp-a-b2"]  # best bullet first


# ---------- integration tests (real Postgres, fake embeddings) ----------

TEST_TABLE = "test_retriever_chunks"
db = pytest.mark.skipif(not _db_is_up(), reason="Postgres is not running (docker compose up -d db)")

DOCS = [
    Document(id="exp-acme-b1", page_content="Built a Python ETL pipeline with Airflow and Spark",
             metadata={"type": "experience", "parent_id": "exp-acme", "skills": []}),
    Document(id="exp-acme-b2", page_content="Ran A/B tests on pricing",
             metadata={"type": "experience", "parent_id": "exp-acme", "skills": []}),
    Document(id="proj-loyalty-b1", page_content="Trained an XGBoost model on 5M purchase records",
             metadata={"type": "project", "parent_id": "proj-loyalty", "skills": ["XGBoost"]}),
    Document(id="skill-python", page_content="Skill: Python (category: Programming Languages)",
             metadata={"type": "skill", "parent_id": "skill-python", "skills": ["Python"]}),
    Document(id="skill-aws", page_content="Skill: AWS (category: MLOps, Deployment & Cloud)",
             metadata={"type": "skill", "parent_id": "skill-aws", "skills": ["AWS"]}),
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
def test_keyword_search_finds_exact_term(retriever):
    # "xgboost" is not one of the fake model's words, so only keyword search can find it
    hits = retriever.keyword_search("experience with XGBoost", k=5)
    assert [h.id for h in hits] == ["proj-loyalty-b1"]
    assert hits[0].type == "project" and hits[0].parent_id == "proj-loyalty"


@db
def test_hybrid_search_combines_both(retriever):
    results = retriever.search("XGBoost and Python", k=3)
    ids = [r.id for r in results]
    assert "proj-loyalty-b1" in ids          # found by keywords
    assert ids[0] in {"exp-acme-b1", "skill-python"}  # matched by both searches -> top
    top = results[0]
    assert top.vector_rank is not None and top.keyword_rank is not None


@db
def test_type_filter(retriever):
    results = retriever.search("Python AWS", k=5, types=["skill"])
    assert {r.id for r in results} == {"skill-python", "skill-aws"}


@db
def test_k_limits_results(retriever):
    assert len(retriever.search("Python", k=2)) == 2


@db
def test_keyword_index_is_created(retriever):
    with create_engine(database_url()).connect() as conn:
        names = conn.execute(text(
            "SELECT indexname FROM pg_indexes WHERE tablename = :t"), {"t": TEST_TABLE}).scalars().all()
    assert f"{TEST_TABLE}_content_fts" in names
