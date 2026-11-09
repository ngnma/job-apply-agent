"""Job schemas (Phase 2).

JobPost         = one job advert, the same shape whatever the source (Reed now, more later).
JobRequirements = what the job asks for, extracted from the description by an LLM (task 5).
MatchResult     = how well the job fits the master CV, and the decision (tasks 6-7).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, HttpUrl, model_validator

from job_agent.contracts.cv import Strict

JobSource = Literal["reed"]  # add "adzuna" etc. later


class JobStatus(StrEnum):
    """Where a job is in the pipeline (stored in the jobs table)."""
    NEW = "new"                    # just fetched
    PREPROCESSED = "preprocessed"  # requirements extracted
    FILTERED_OUT = "filtered_out"  # dropped by the cheap stage-1 filter
    SCORED = "scored"              # LLM judge gave a score
    SELECTED = "selected"          # score >= threshold -> goes to CV tailoring
    REJECTED = "rejected"          # score < threshold


# ---------- the job advert ----------

class JobPost(Strict):
    id: str                        # globally unique: "<source>-<source_id>", e.g. "reed-54321"
    source: JobSource
    source_id: str                 # the ID on the source site
    title: str
    company: str
    location: str
    salary_min: float | None = None   # yearly, in `currency`
    salary_max: float | None = None
    currency: str = "GBP"
    contract_type: Literal["permanent", "contract", "temp"] | None = None
    full_time: bool | None = None
    url: HttpUrl                   # the advert page
    apply_url: HttpUrl | None = None  # employer's own application page, if any (needed for auto-apply)
    posted_at: date | None = None
    expires_at: date | None = None
    description: str = Field(min_length=1)  # full text (HTML removed in task 5)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def check(self) -> Self:
        if self.id != f"{self.source}-{self.source_id}":
            raise ValueError(f"id must be '{self.source}-{self.source_id}'")
        if self.salary_min and self.salary_max and self.salary_max < self.salary_min:
            raise ValueError("salary_max is below salary_min")
        return self


# ---------- what the job asks for (LLM output in task 5) ----------

class JobRequirements(Strict):
    must_have_skills: list[str] = Field(default_factory=list)  # e.g. ["Python", "SQL", "AWS"]
    nice_to_have_skills: list[str] = Field(default_factory=list)
    seniority: Literal["intern", "graduate", "junior", "mid", "senior", "lead", "unknown"] = "unknown"
    min_years_experience: int | None = Field(default=None, ge=0, le=40)
    visa_sponsorship: Literal["yes", "no", "not_mentioned"] = "not_mentioned"
    work_mode: Literal["onsite", "hybrid", "remote", "unknown"] = "unknown"


# ---------- how well it matches the CV (tasks 6-7) ----------

Criterion = Literal["skills", "experience_level", "domain", "location_visa"]


class CriterionScore(Strict):
    """One line of the LLM judge's rubric."""
    criterion: Criterion
    score: int = Field(ge=0, le=10)
    reason: str


class MatchResult(Strict):
    job_id: str
    stage1_score: float = Field(ge=0, le=1)     # cheap embedding coverage score (task 6)
    score: int | None = Field(default=None, ge=0, le=100)  # LLM judge score (task 7); None if filtered out
    criteria: list[CriterionScore] = Field(default_factory=list)
    evidence_chunk_ids: list[str] = Field(default_factory=list)  # master-CV chunks that support the match
    summary: str = ""
    decision: JobStatus

    @model_validator(mode="after")
    def check(self) -> Self:
        if self.decision in (JobStatus.SELECTED, JobStatus.REJECTED, JobStatus.SCORED) and self.score is None:
            raise ValueError(f"decision '{self.decision}' needs a score")
        names = [c.criterion for c in self.criteria]
        if len(names) != len(set(names)):
            raise ValueError("each criterion may appear only once")
        return self
