import pytest
from pydantic import ValidationError

from job_agent.contracts.job import (
    CriterionScore,
    JobPost,
    JobRequirements,
    JobStatus,
    MatchResult,
)

JOB = {
    "id": "reed-54321",
    "source": "reed",
    "source_id": "54321",
    "title": "Graduate Data Scientist",
    "company": "Acme Ltd",
    "location": "London",
    "salary_min": 32000,
    "salary_max": 38000,
    "contract_type": "permanent",
    "full_time": True,
    "url": "https://www.reed.co.uk/jobs/graduate-data-scientist/54321",
    "posted_at": "2026-11-01",
    "description": "We are looking for a graduate data scientist with Python and SQL.",
}


def test_valid_job():
    job = JobPost.model_validate(JOB)
    assert job.currency == "GBP"
    assert job.posted_at.year == 2026
    assert job.fetched_at.tzinfo is not None  # stored with a timezone


def test_id_must_match_source_and_source_id():
    with pytest.raises(ValidationError, match="id must be"):
        JobPost.model_validate({**JOB, "id": "reed-999"})


def test_salary_order():
    with pytest.raises(ValidationError, match="salary_max"):
        JobPost.model_validate({**JOB, "salary_min": 50000, "salary_max": 30000})


def test_unknown_source_rejected():
    with pytest.raises(ValidationError):
        JobPost.model_validate({**JOB, "source": "linkedin", "id": "linkedin-54321"})


def test_requirements_defaults():
    req = JobRequirements(must_have_skills=["Python"])
    assert req.seniority == "unknown"
    assert req.visa_sponsorship == "not_mentioned"


def test_match_result_selected_needs_score():
    with pytest.raises(ValidationError, match="needs a score"):
        MatchResult(job_id="reed-1", stage1_score=0.7, decision=JobStatus.SELECTED)


def test_match_result_filtered_out_without_score():
    m = MatchResult(job_id="reed-1", stage1_score=0.2, decision=JobStatus.FILTERED_OUT)
    assert m.score is None


def test_criteria_unique():
    c = CriterionScore(criterion="skills", score=8, reason="Python, SQL")
    with pytest.raises(ValidationError, match="only once"):
        MatchResult(job_id="reed-1", stage1_score=0.8, score=80, criteria=[c, c], decision=JobStatus.SELECTED)


def test_scores_in_range():
    with pytest.raises(ValidationError):
        CriterionScore(criterion="skills", score=11, reason="too high")
    with pytest.raises(ValidationError):
        MatchResult(job_id="reed-1", stage1_score=1.5, decision=JobStatus.FILTERED_OUT)
