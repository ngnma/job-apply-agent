"""CV schemas.

MasterCV   = everything about the person (source of truth, stored in RAG).
TailoredCV = what goes into the Word template for one job. Every line cites
             the master-CV IDs it came from, so the grounding guardrail can
             check it.
"""

from __future__ import annotations

from datetime import date
from typing import Self

from pydantic import BaseModel, ConfigDict, EmailStr, Field, HttpUrl, model_validator

# ID format: lowercase words joined by "-", e.g. "exp-snappshop", "exp-snappshop-b1"
ID_PATTERN = r"^[a-z0-9]+(-[a-z0-9]+)*$"


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
    id: str = Field(pattern=ID_PATTERN)
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
    id: str = Field(pattern=ID_PATTERN)
    position: str
    company: str
    location: str
    period: Period
    bullets: list[Bullet] = Field(min_length=1)  # store ALL bullets, not just 4


class Project(Strict):
    id: str = Field(pattern=ID_PATTERN)
    title: str
    tools: list[str] = Field(min_length=1)
    link: HttpUrl | None = None  # some projects have no repo
    bullets: list[Bullet] = Field(min_length=1)


class SkillGroup(Strict):
    id: str = Field(pattern=ID_PATTERN)
    name: str  # e.g. "Machine Learning"
    items: list[str] = Field(min_length=1)


class Education(Strict):
    id: str = Field(pattern=ID_PATTERN)
    degree: str  # e.g. "MSc Artificial Intelligence"
    institution: str
    location: str
    period: Period
    grade: str | None = None  # e.g. "Distinction (predicted)"
    modules: list[str] = []
    dissertation: str | None = None


class Certification(Strict):
    id: str = Field(pattern=ID_PATTERN)
    title: str
    provider: str
    year: int = Field(ge=1990, le=2100)
    link: HttpUrl | None = None


class MasterCV(Strict):
    contact: Contact
    summary: str  # base summary; rewritten per job in TailoredCV
    experiences: list[Experience]
    projects: list[Project]
    skills: list[SkillGroup]
    education: list[Education]
    certifications: list[Certification] = []

    @model_validator(mode="after")
    def ids_unique(self) -> Self:
        ids = [x.id for x in self.all_items()]
        ids += [b.id for x in [*self.experiences, *self.projects] for b in x.bullets]
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
    summary: GroundedText
    experiences: list[TailoredSection] = Field(max_length=2)
    projects: list[TailoredSection] = Field(max_length=3)
    skills: list[TailoredSkillGroup] = Field(max_length=4)
    education_ids: list[str]
    certification_ids: list[str] = []
