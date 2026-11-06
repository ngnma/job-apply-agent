"""Tests for the code around the LLM call (no Ollama needed)."""

import pytest

from job_agent.contracts.cv import Bullet, Project
from job_agent.rag.extract import (
    SkillGroup,
    _snap_to_source,
    skills_from_groups,
    split_sections,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://github.com/a/b", "https://github.com/a/b"),
        ("GitHub (https://github.com/a/b)", "https://github.com/a/b"),
        ("Frontend GitHub (https://github.com/a/f), Backend GitHub (https://github.com/a/b)",
         "https://github.com/a/f"),  # two links -> keep the first
        ("Github", None),  # link text without a URL
        (None, None),
    ],
)
def test_project_link_keeps_only_the_url(raw, expected):
    project = Project(title="T", tools=["Python"], link=raw, bullets=[{"text": "Built a test thing."}])
    assert (str(project.link) if project.link else None) == expected


def test_bullet_mark_is_removed():
    assert Bullet(text="- Built a classical NLP model.").text == "Built a classical NLP model."


@pytest.mark.parametrize(
    ("llm_text", "expected"),
    [
        ("Built a classical NLP baselinemodel", "Built a classical NLP baseline model"),  # dropped space
        ("A/B TestingFramework", "A/B Testing Framework"),
        ("Built a  classical NLP", "Built a classical NLP"),  # extra space
        ("Something not in the CV", "Something not in the CV"),  # unchanged if no match
    ],
)
def test_snap_to_source_fixes_spacing(llm_text, expected):
    source = "A/B Testing Framework | Stats\n- Built a classical NLP baseline model for reviews."
    assert _snap_to_source(llm_text, source) == expected


def test_split_sections():
    text = "# Jane Doe\n# Contact details\njane@x.com\n# Work History\nAnalyst | Acme\n- Did things well."
    sections = split_sections(text)
    assert sections["contact"] == "# Jane Doe\njane@x.com"
    assert sections["experiences"] == "Analyst | Acme\n- Did things well."


def test_skills_from_groups_one_skill_per_item():
    groups = [
        SkillGroup(name="### Programming Languages", items=["Python, SQL"]),  # comma list in one item
        SkillGroup(name="Agentic AI", items=["LangChain", "Prompt Engineering"]),
        SkillGroup(name="ML Frameworks", items=["PyTorch", "LangChain", "prompt engineering"]),
    ]
    skills = skills_from_groups(groups)
    assert [s.name for s in skills] == ["Python", "SQL", "LangChain", "Prompt Engineering", "PyTorch"]
    assert skills[0].categories == ["Programming Languages"]  # "###" removed
    # listed twice (any letter case) -> one skill with both categories
    assert skills[2].categories == ["Agentic AI", "ML Frameworks"]
    assert skills[3].categories == ["Agentic AI", "ML Frameworks"]
