"""Task 3: turn the master CV text into structured JSON (MasterCV) with a local LLM.

Run once:
    uv run python -m job_agent.rag.extract data/Master.docx data/master_cv.json

Then open data/master_cv.json, check it by hand, and fix anything wrong.
From then on, that JSON file (not the Word file) is the source of truth.
"""

import argparse
import re
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from job_agent.contracts.cv import (
    Certification,
    Contact,
    Education,
    Experience,
    MasterCV,
    Project,
    SkillGroup,
    Strict,
)
from job_agent.rag.loader import load_cv_text

# ---------- 1. split the CV text into sections ----------

# which "# Heading" belongs to which part of the schema (matched by keyword)
SECTION_KEYWORDS = {
    "contact": ("contact",),
    "summary": ("summary", "profile", "about"),
    "experiences": ("work", "experience", "employment"),
    "projects": ("project",),
    "skills": ("skill",),
    "education": ("education",),
    "certifications": ("certific", "course", "professional development"),
}


def split_sections(text: str) -> dict[str, str]:
    """Group lines under the top-level '# ' heading they belong to.

    Lines before the first known heading (e.g. '# Your Name') go to 'contact'.
    """
    sections: dict[str, list[str]] = {}
    current = "contact"
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].lower()
            match = next((k for k, words in SECTION_KEYWORDS.items() if any(w in title for w in words)), None)
            if match:
                current = match
                continue
        sections.setdefault(current, []).append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


# ---------- 2. one small schema per section (smaller output = fewer LLM mistakes) ----------

class Summary(Strict):
    text: str

class Experiences(Strict):
    items: list[Experience]

class Projects(Strict):
    items: list[Project]

class Skills(Strict):
    items: list[SkillGroup]

class EducationList(Strict):
    items: list[Education]

class Certifications(Strict):
    items: list[Certification]


SECTION_SCHEMAS: dict[str, type[BaseModel]] = {
    "contact": Contact,
    "summary": Summary,
    "experiences": Experiences,
    "projects": Projects,
    "skills": Skills,
    "education": EducationList,
    "certifications": Certifications,
}

# ---------- 3. prompt + LLM ----------

SYSTEM_PROMPT = (
    "You extract one section of a CV into JSON that matches the given schema.\n"
    "Rules:\n"
    "- Copy text EXACTLY as written. Never rephrase, shorten, summarise or add anything.\n"
    "- Each line starting with '- ' is one bullet. Keep every bullet, in order.\n"
    "- Entry headers look like 'Position | Company | Location | MM/YYYY – MM/YYYY' "
    "or 'Project title | tools | GitHub (url)'.\n"
    "- Dates: 'MM/YYYY' -> year and month. If the end is 'Present' or missing, set end to null.\n"
    "- Bullet 'skills': only tools or skills literally named in that bullet; empty list if none.\n"
    "- Lines starting with '##' or '###' are group names (e.g. a skill group name).\n"
    "- If a field is not in the text, leave it empty or null. Never invent data."
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Section: {section}\n\n{text}"),
])


def make_llm(model: str) -> ChatOllama:
    return ChatOllama(
        model=model,
        temperature=0,     # same input -> same output
        num_ctx=16384,     # Ollama's default context is small and silently cuts long input
        reasoning=False,   # turn off "thinking" mode (faster, plain JSON answer)
    )


def extract_section(llm: ChatOllama, section: str, text: str) -> BaseModel:
    chain = PROMPT | llm.with_structured_output(SECTION_SCHEMAS[section])
    # if the output breaks the schema (Pydantic error), ask again — max 3 tries
    return chain.with_retry(stop_after_attempt=3).invoke({"section": section, "text": text})


# ---------- 4. IDs are made by code, not by the LLM (stable + always valid) ----------

def _slug(text: str, max_words: int = 4) -> str:
    return "-".join(re.findall(r"[a-z0-9]+", text.lower())[:max_words])


def assign_ids(cv: MasterCV) -> None:
    used: set[str] = set()

    def unique(base: str) -> str:
        new, n = base, 2
        while new in used:
            new, n = f"{base}-{n}", n + 1
        used.add(new)
        return new

    for exp in cv.experiences:
        exp.id = unique(f"exp-{_slug(exp.company, 2)}")
        for i, b in enumerate(exp.bullets, 1):
            b.id = unique(f"{exp.id}-b{i}")
    for proj in cv.projects:
        proj.id = unique(f"proj-{_slug(proj.title.split('—')[0])}")
        for i, b in enumerate(proj.bullets, 1):
            b.id = unique(f"{proj.id}-b{i}")
    for group in cv.skills:
        group.id = unique(f"skills-{_slug(group.name)}")
    for edu in cv.education:
        edu.id = unique(f"edu-{_slug(edu.degree, 1)}")
    for cert in cv.certifications:
        cert.id = unique(f"cert-{_slug(cert.title)}")


# ---------- 5. quality checks: did the LLM drop or change anything? ----------

def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def check_extraction(cv: MasterCV, sections: dict[str, str]) -> list[str]:
    warnings = []
    for name, items in (("experiences", cv.experiences), ("projects", cv.projects)):
        source = sections.get(name, "")
        expected = sum(1 for line in source.splitlines() if line.startswith("- "))
        got = sum(len(x.bullets) for x in items)
        if got != expected:
            warnings.append(f"{name}: {expected} bullets in the CV, {got} extracted")
        for x in items:
            for b in x.bullets:
                if _norm(b.text) not in _norm(source):
                    warnings.append(f"{b.id}: text was changed by the LLM -> {b.text[:60]}...")
    return warnings


# ---------- 6. run everything ----------

def extract_master_cv(cv_path: str | Path, model: str = "qwen3:8b") -> tuple[MasterCV, list[str]]:
    sections = split_sections(load_cv_text(cv_path))
    llm = make_llm(model)

    parts: dict[str, BaseModel] = {}
    for name, text in sections.items():
        if text.strip():
            print(f"Extracting {name} ...")
            parts[name] = extract_section(llm, name, text)

    cv = MasterCV.model_construct(  # build without validation; IDs are not set yet
        contact=parts["contact"],
        summary=parts["summary"].text if "summary" in parts else None,
        experiences=parts["experiences"].items if "experiences" in parts else [],
        projects=parts["projects"].items if "projects" in parts else [],
        skills=parts["skills"].items if "skills" in parts else [],
        education=parts["education"].items if "education" in parts else [],
        certifications=parts["certifications"].items if "certifications" in parts else [],
    )
    assign_ids(cv)
    cv = MasterCV.model_validate(cv.model_dump())  # full validation, now with IDs
    return cv, check_extraction(cv, sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a CV into master_cv.json")
    parser.add_argument("cv_path", help="e.g. data/Master.docx")
    parser.add_argument("out_path", help="e.g. data/master_cv.json")
    parser.add_argument("--model", default="qwen3:8b", help="Ollama model name")
    args = parser.parse_args()

    cv, warnings = extract_master_cv(args.cv_path, args.model)
    Path(args.out_path).write_text(cv.model_dump_json(indent=2), encoding="utf-8")

    print(f"\nSaved {args.out_path}: {len(cv.experiences)} jobs, {len(cv.projects)} projects, "
          f"{len(cv.skills)} skill groups, {len(cv.education)} degrees, {len(cv.certifications)} certificates")
    for w in warnings:
        print("WARNING:", w)
    if not warnings:
        print("All checks passed.")


if __name__ == "__main__":
    main()
