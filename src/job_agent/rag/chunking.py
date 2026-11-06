"""Task 4: turn master_cv.json into chunks (LangChain Documents) ready for embedding.

Chunking rule ("chunk by unit"):
    - Experience / Project -> one chunk PER BULLET, with the job/project header added
      as context. A role with 14 bullets covers many topics (A/B testing, CRM, ETL...);
      one big chunk would blur them, one bullet per chunk keeps search precise.
    - Skill -> one chunk per skill (e.g. "Docker"), so a job requirement can match it directly.
    - Education / Certification -> one chunk per item (already small).
    - Contact details -> never chunked (personal data, useless for search).

Each chunk's ID is the master-CV ID (e.g. "exp-snappshop-b3"), so later the tailored
CV can cite exactly which chunk a line came from.

Preview the chunks:
    uv run python -m job_agent.rag.chunking data/master_cv.json
"""

import sys
from pathlib import Path

from langchain_core.documents import Document

from job_agent.contracts.cv import (
    Bullet,
    Certification,
    Education,
    Experience,
    MasterCV,
    Period,
    Project,
    Skill,
)


def load_master_cv(path: str | Path) -> MasterCV:
    return MasterCV.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _ym(period: Period) -> tuple[str, str | None]:
    """Period -> ("2023-10", "2026-02") for metadata (None end = present)."""
    start = f"{period.start.year}-{period.start.month:02d}"
    end = f"{period.end.year}-{period.end.month:02d}" if period.end else None
    return start, end


# ---------- one function per item type ----------

def _experience_chunks(exp: Experience) -> list[Document]:
    start, end = _ym(exp.period)
    header = f"Experience: {exp.position} at {exp.company} ({exp.location}), {exp.period}"
    return [
        Document(
            id=b.id,
            page_content=f"{header}\n{b.text}",
            metadata={
                "chunk_id": b.id,
                "parent_id": exp.id,
                "type": "experience",
                "title": f"{exp.position} | {exp.company}",
                "skills": b.skills,
                "start": start,
                "end": end,
            },
        )
        for b in exp.bullets
    ]


def _project_chunks(proj: Project) -> list[Document]:
    header = f"Project: {proj.title} | Tools: {', '.join(proj.tools)}"

    def make(b: Bullet) -> Document:
        return Document(
            id=b.id,
            page_content=f"{header}\n{b.text}",
            metadata={
                "chunk_id": b.id,
                "parent_id": proj.id,
                "type": "project",
                "title": proj.title,
                # project tools apply to every bullet; merge them with the bullet's own tags
                "skills": sorted(set(proj.tools) | set(b.skills)),
                "link": str(proj.link) if proj.link else None,
            },
        )

    return [make(b) for b in proj.bullets]


def _skill_chunk(skill: Skill) -> Document:
    return Document(
        id=skill.id,
        page_content=f"Skill: {skill.name} (category: {', '.join(skill.categories)})",
        metadata={
            "chunk_id": skill.id,
            "parent_id": skill.id,
            "type": "skill",
            "title": skill.name,
            "skills": [skill.name],
            "categories": skill.categories,
        },
    )


def _education_chunk(edu: Education) -> Document:
    start, end = _ym(edu.period)
    lines = [f"Education: {edu.degree}, {edu.institution} ({edu.location}), {edu.period}"]
    if edu.grade:
        lines.append(f"Grade: {edu.grade}")
    if edu.modules:
        lines.append(f"Modules: {', '.join(edu.modules)}")
    if edu.dissertation:
        lines.append(f"Dissertation: {edu.dissertation}")
    return Document(
        id=edu.id,
        page_content="\n".join(lines),
        metadata={
            "chunk_id": edu.id,
            "parent_id": edu.id,
            "type": "education",
            "title": edu.degree,
            "skills": edu.modules,
            "start": start,
            "end": end,
        },
    )


def _certification_chunk(cert: Certification) -> Document:
    return Document(
        id=cert.id,
        page_content=f"Certification: {cert.title} - {cert.provider} ({cert.year})",
        metadata={
            "chunk_id": cert.id,
            "parent_id": cert.id,
            "type": "certification",
            "title": cert.title,
            "skills": [],
            "year": cert.year,
        },
    )


# ---------- main entry point ----------

def chunk_master_cv(cv: MasterCV) -> list[Document]:
    chunks: list[Document] = []
    for exp in cv.experiences:
        chunks += _experience_chunks(exp)
    for proj in cv.projects:
        chunks += _project_chunks(proj)
    chunks += [_skill_chunk(s) for s in cv.skills]
    chunks += [_education_chunk(e) for e in cv.education]
    chunks += [_certification_chunk(c) for c in cv.certifications]
    return chunks


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/master_cv.json"
    chunks = chunk_master_cv(load_master_cv(path))

    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.metadata["type"]] = counts.get(c.metadata["type"], 0) + 1
    print(f"{len(chunks)} chunks: {counts}\n")

    for c in chunks[:3]:  # show a few examples
        print(f"--- {c.id} ---\n{c.page_content}\n{c.metadata}\n")


if __name__ == "__main__":
    main()
