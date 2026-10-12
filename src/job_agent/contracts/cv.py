"""CV schemas.

MasterCV   = everything about the person (source of truth, stored in RAG).
TailoredCV = what goes into the Word template for one job. Every line cites
             the master-CV IDs it came from, so the grounding guardrail can
             check it.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, EmailStr, Field, HttpUrl, model_validator
from pydantic.json_schema import SkipJsonSchema

# IDs are created by code (not the LLM), so they are hidden from the JSON schema
# the LLM sees. Format: lowercase words joined by "-", e.g. "exp-snappshop-b1".
ID_PATTERN = r"^[a-z0-9]+(-[a-z0-9]+)*$"
ItemId = Annotated[SkipJsonSchema[str], Field(pattern=rf"^$|{ID_PATTERN}")]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------- building blocks ----------

class YearMonth(Strict):
    year: int = Field(ge=1990, le=2100)
    month: int = Field(ge=1, le=12)

    def __str__(self) -> str:
        return date(self.year, self.month, 1).strftime("%b %Y")  # "Oct 2023"


class Period(Strict):
    start: YearMonth
    end: YearMonth | None = None  # None = "Present"

    @model_validator(mode="after")
    def end_after_start(self) -> Self:
        if self.end and (self.end.year, self.end.month) < (self.start.year, self.start.month):
            raise ValueError("end is before start")
        return self

    def __str__(self) -> str:  # "{duration}" in the template
        return f"{self.start} – {self.end or 'Present'}"


class Bullet(Strict):
    id: ItemId = ""
    text: str = Field(min_length=10)
    skills: list[str] = []  # tags, e.g. ["Python", "Airflow"]; help retrieval + ATS


# ---------- master CV sections (one RAG chunk each) ----------

class Contact(Strict):
    name: str
    location: str
    phone: str
    email: EmailStr
    linkedin: HttpUrl
    github: HttpUrl


class Experience(Strict):
    id: ItemId = ""
    position: str
    company: str
    location: str
    period: Period
    bullets: list[Bullet] = Field(min_length=1)  # store ALL bullets, not just 4


class Project(Strict):
    id: ItemId = ""
    title: str
    tools: list[str] = Field(min_length=1)
    link: HttpUrl | None = None  # one repo link, or None
    bullets: list[Bullet] = Field(min_length=1)


class SkillGroup(Strict):
    id: ItemId = ""
    name: str  # e.g. "Machine Learning"
    items: list[str] = Field(min_length=1)


class Education(Strict):
    id: ItemId = ""
    degree: str  # e.g. "MSc Artificial Intelligence"
    institution: str
    location: str
    period: Period
    grade: str | None = None  # e.g. "Distinction (predicted)"
    modules: list[str] = []
    dissertation: str | None = None


class Certification(Strict):
    id: ItemId = ""
    title: str
    provider: str
    year: int = Field(ge=1990, le=2100)
    link: HttpUrl | None = None


class MasterCV(Strict):
    contact: Contact
    # Copied ONLY if the CV already has a summary section; never generated here.
    # The job-specific summary is written later, in TailoredCV.summary.
    summary: str | None = None
    experiences: list[Experience]
    projects: list[Project]
    skills: list[SkillGroup]
    education: list[Education]
    certifications: list[Certification] = []

    @model_validator(mode="after")
    def ids_unique(self) -> Self:
        ids = [x.id for x in self.all_items()]
        ids += [b.id for x in [*self.experiences, *self.projects] for b in x.bullets]
        if "" in ids:
            raise ValueError("every item and bullet needs an id (run assign_ids)")
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate ids: {sorted(dupes)}")
        return self

    def all_items(self) -> list[Experience | Project | SkillGroup | Education | Certification]:
        return [*self.experiences, *self.projects, *self.skills, *self.education, *self.certifications]

    def bullet_index(self) -> dict[str, Bullet]:
        return {b.id: b for x in [*self.experiences, *self.projects] for b in x.bullets}


# ---------- tailored CV (fills the Word template) ----------

class GroundedText(Strict):
    """A rewritten line + the master-CV IDs it is based on."""
    text: str
    source_ids: list[str] = Field(min_length=1)


class TailoredSection(Strict):
    source_id: str  # Experience / Project id; header fields copied from master
    bullets: list[GroundedText] = Field(min_length=1, max_length=4)


class TailoredSkillGroup(Strict):
    name: str
    items: list[str] = Field(min_length=1)  # must exist in master skills/tools


class TailoredCV(Strict):
    job_id: str
    # Always written for the job post. Built from the master summary (if there is
    # one) plus the experiences/projects/skills chosen for this job.
    # source_ids = the master IDs it is based on (use "summary" for MasterCV.summary).
    summary: GroundedText
    experiences: list[TailoredSection] = Field(max_length=2)
    projects: list[TailoredSection] = Field(max_length=3)
    skills: list[TailoredSkillGroup] = Field(max_length=4)
    education_ids: list[str]
    certification_ids: list[str] = []
