import json

import pytest

from job_agent.contracts.cv import MasterCV
from job_agent.rag.chunking import chunk_master_cv, load_master_cv

# A small FAKE master CV with the same structure as data/master_cv.json
# (never use the real one in tests)
FAKE_CV = {
    "contact": {
        "name": "Jane Doe",
        "location": "London, UK (Open to relocation)",
        "phone": "+44 7000 000000",
        "email": "jane@example.com",
        "linkedin": "https://www.linkedin.com/in/jane/",
        "github": "http://github.com/jane",
    },
    "summary": None,
    "experiences": [
        {
            "id": "exp-acme",
            "position": "Data Analyst",
            "company": "Acme Ltd",
            "location": "London, UK",
            "period": {"start": {"year": 2024, "month": 1}, "end": {"year": 2025, "month": 6}},
            "bullets": [
                {"id": "exp-acme-b1", "text": "Built a Python ETL pipeline orchestrated with Airflow.", "skills": []},
                {"id": "exp-acme-b2", "text": "Ran 20+ A/B tests across pricing and search.", "skills": []},
            ],
        },
        {
            "id": "exp-globex",
            "position": "Backend Developer Intern",
            "company": "Globex",
            "location": "Manchester, UK",
            "period": {"start": {"year": 2025, "month": 9}, "end": None},  # current job
            "bullets": [{"id": "exp-globex-b1", "text": "Developed microservices with Java and PostgreSQL.", "skills": []}],
        },
    ],
    "projects": [
        {
            "id": "proj-rag-chatbot",
            "title": "RAG Chatbot — LLM Agent over PDFs",
            "tools": ["LangChain", "Python"],
            "link": "https://github.com/jane/rag-chatbot",
            "bullets": [{"id": "proj-rag-chatbot-b1", "text": "Built a RAG chatbot over PDF documents.", "skills": []}],
        },
        {
            "id": "proj-kpi-platform",
            "title": "Unified KPI Data Platform",
            "tools": ["Python", "Spark"],
            "link": None,  # private project, no repo
            "bullets": [{"id": "proj-kpi-platform-b1", "text": "Consolidated KPIs from many source systems.", "skills": []}],
        },
    ],
    "skills": [
        {"id": "skill-python", "name": "Python", "categories": ["Programming Languages"]},
        {"id": "skill-docker", "name": "Docker", "categories": ["MLOps, Deployment & Cloud"]},
        {"id": "skill-langchain", "name": "LangChain", "categories": ["Agentic AI", "Machine Learning Frameworks"]},
    ],
    "education": [{
        "id": "edu-msc",
        "degree": "MSc Artificial Intelligence (Master of Science)",
        "institution": "Uni of Test",
        "location": "London, UK",
        "period": {"start": {"year": 2025, "month": 9}, "end": {"year": 2026, "month": 9}},
        "grade": None,
        "modules": ["Machine Learning", "Deep Learning"],
        "dissertation": None,
    }],
    "certifications": [
        {"id": "cert-aws-cloud-practitioner", "title": "AWS Cloud Practitioner", "provider": "AWS",
         "year": 2025, "link": None},
    ],
}


@pytest.fixture
def chunks():
    return chunk_master_cv(MasterCV.model_validate(FAKE_CV))


def by_id(chunks, chunk_id):
    return next(c for c in chunks if c.id == chunk_id)


def test_one_chunk_per_bullet_or_item(chunks):
    # 3 experience bullets + 2 project bullets + 3 skills + 1 degree + 1 cert
    assert len(chunks) == 10
    assert [c.metadata["type"] for c in chunks] == (
        ["experience"] * 3 + ["project"] * 2 + ["skill"] * 3 + ["education", "certification"]
    )


def test_chunk_ids_match_master_cv_ids(chunks):
    ids = [c.id for c in chunks]
    assert ids[:3] == ["exp-acme-b1", "exp-acme-b2", "exp-globex-b1"]
    assert len(ids) == len(set(ids))  # unique
    assert all(c.id == c.metadata["chunk_id"] for c in chunks)


def test_experience_chunk_has_header_and_dates(chunks):
    c = by_id(chunks, "exp-acme-b1")
    assert c.page_content == (
        "Experience: Data Analyst at Acme Ltd (London, UK), Jan 2024 – Jun 2025\n"
        "Built a Python ETL pipeline orchestrated with Airflow."
    )
    assert c.metadata["parent_id"] == "exp-acme"
    assert (c.metadata["start"], c.metadata["end"]) == ("2024-01", "2025-06")


def test_current_job_has_no_end_date(chunks):
    c = by_id(chunks, "exp-globex-b1")
    assert "Sep 2025 – Present" in c.page_content
    assert c.metadata["end"] is None


def test_project_chunk_uses_tools_as_skills(chunks):
    c = by_id(chunks, "proj-rag-chatbot-b1")
    assert c.page_content.startswith("Project: RAG Chatbot — LLM Agent over PDFs | Tools: LangChain, Python")
    assert c.metadata["skills"] == ["LangChain", "Python"]  # bullet skills are empty -> tools only
    assert c.metadata["link"] == "https://github.com/jane/rag-chatbot"


def test_project_without_link(chunks):
    assert by_id(chunks, "proj-kpi-platform-b1").metadata["link"] is None


def test_one_chunk_per_skill(chunks):
    c = by_id(chunks, "skill-docker")
    assert c.page_content == "Skill: Docker (category: MLOps, Deployment & Cloud)"
    assert c.metadata["skills"] == ["Docker"]
    assert c.metadata["categories"] == ["MLOps, Deployment & Cloud"]


def test_skill_in_two_categories(chunks):
    c = by_id(chunks, "skill-langchain")
    assert c.page_content == "Skill: LangChain (category: Agentic AI, Machine Learning Frameworks)"


def test_education_and_certification_text(chunks):
    edu = by_id(chunks, "edu-msc")
    assert "Modules: Machine Learning, Deep Learning" in edu.page_content
    assert "Grade" not in edu.page_content  # empty fields are left out
    cert = by_id(chunks, "cert-aws-cloud-practitioner")
    assert cert.page_content == "Certification: AWS Cloud Practitioner - AWS (2025)"


def test_no_contact_details_in_chunks(chunks):
    everything = " ".join(c.page_content for c in chunks)
    for private in ("jane@example.com", "+44 7000 000000", "Jane Doe"):
        assert private not in everything


def test_metadata_is_json_serialisable(chunks):
    for c in chunks:
        json.dumps(c.metadata)  # the vector DB stores metadata as JSON


def test_load_master_cv(tmp_path):
    path = tmp_path / "master_cv.json"
    path.write_text(json.dumps(FAKE_CV), encoding="utf-8")
    assert load_master_cv(path).contact.name == "Jane Doe"
