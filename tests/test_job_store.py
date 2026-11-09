"""Job store tests against a real Postgres. They use their own schema ("test_jobs"),
so the real `jobs` table is never touched, and are skipped when the database is down."""

import pytest
from sqlalchemy import create_engine, text

from job_agent.contracts.job import (
    CriterionScore,
    JobPost,
    JobRequirements,
    JobStatus,
    MatchResult,
)
from job_agent.jobs.store import JobStore
from job_agent.rag.vectorstore import database_url
from tests.test_job_contracts import JOB
from tests.test_vectorstore import _db_is_up

pytestmark = pytest.mark.skipif(not _db_is_up(), reason="Postgres is not running (docker compose up -d db)")

SCHEMA = "test_jobs"


@pytest.fixture
def store():
    admin = create_engine(database_url())
    with admin.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {SCHEMA}"))
    engine = create_engine(database_url(), connect_args={"options": f"-csearch_path={SCHEMA}"})
    s = JobStore(engine)
    s.init_db()
    yield s
    engine.dispose()
    with admin.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))


def job(**changes) -> JobPost:
    return JobPost.model_validate({**JOB, **changes})


def test_save_and_get(store):
    report = store.save_jobs([job()])
    assert (report.added, report.updated, report.unchanged) == (1, 0, 0)
    saved = store.get("reed-54321")
    assert saved.title == "Graduate Data Scientist"
    assert saved.salary_max == 38000
    assert store.get_status("reed-54321") == JobStatus.NEW


def test_no_duplicates(store):
    store.save_jobs([job()])
    report = store.save_jobs([job()])
    assert (report.added, report.unchanged) == (0, 1)
    assert store.counts_by_status() == {"new": 1}


def test_changed_advert_goes_back_to_new(store):
    store.save_jobs([job()])
    store.save_requirements("reed-54321", JobRequirements(must_have_skills=["Python"]))
    assert store.get_status("reed-54321") == JobStatus.PREPROCESSED

    report = store.save_jobs([job(description="Now we also want Spark and Airflow.")])
    assert report.updated == 1
    assert store.get_status("reed-54321") == JobStatus.NEW      # must be processed again
    assert store.get_requirements("reed-54321") is None         # old analysis cleared
    assert "Spark" in store.get("reed-54321").description


def test_requirements_and_match_round_trip(store):
    store.save_jobs([job()])
    store.save_requirements("reed-54321", JobRequirements(must_have_skills=["Python", "SQL"], seniority="graduate"))
    assert store.get_requirements("reed-54321").seniority == "graduate"

    match = MatchResult(job_id="reed-54321", stage1_score=0.8, score=82, decision=JobStatus.SELECTED,
                        criteria=[CriterionScore(criterion="skills", score=9, reason="Python, SQL")],
                        evidence_chunk_ids=["exp-snappshop-b10"])
    store.save_match(match)
    assert store.get_status("reed-54321") == JobStatus.SELECTED
    assert store.get_match("reed-54321").evidence_chunk_ids == ["exp-snappshop-b10"]


def test_list_by_status(store):
    store.save_jobs([job(), job(id="reed-2", source_id="2", title="ML Engineer")])
    store.set_status("reed-2", JobStatus.FILTERED_OUT)
    assert [j.id for j in store.list_by_status(JobStatus.NEW)] == ["reed-54321"]
    assert store.counts_by_status() == {"new": 1, "filtered_out": 1}


def test_unknown_job_raises(store):
    with pytest.raises(KeyError):
        store.set_status("reed-nope", JobStatus.REJECTED)
