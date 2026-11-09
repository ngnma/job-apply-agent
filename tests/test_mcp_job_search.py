"""MCP server tests: a real MCP client talks to the server in-process.
Reed is replaced by a fake client, so no network and no API key are needed."""

import asyncio

import pytest
from mcp import Client

from job_agent.mcp_servers import job_search
from tests.test_reed import DETAILS

SEARCH_RESULTS = [
    {"jobId": 1, "jobTitle": "Data Scientist ", "employerName": "Acme", "locationName": "London",
     "minimumSalary": 40000.0, "maximumSalary": 50000.0, "date": "02/11/2026",
     "jobUrl": "https://www.reed.co.uk/jobs/1", "jobDescription": "Python " * 100},
    {"jobId": 2, "jobTitle": "ML Engineer", "employerName": None, "locationName": "Leeds"},
]


class FakeReed:
    def __init__(self):
        self.search_calls: list[dict] = []

    def search(self, keywords, **kwargs):
        self.search_calls.append({"keywords": keywords, **kwargs})
        return SEARCH_RESULTS[: kwargs["max_results"]]

    def get_details(self, job_id):
        return {**DETAILS, "jobId": int(job_id)}


@pytest.fixture
def fake_reed(monkeypatch):
    fake = FakeReed()
    monkeypatch.setattr(job_search, "get_client", lambda: fake)
    return fake


def call(tool: str, args: dict):
    async def run():
        async with Client(job_search.server) as client:
            return await client.call_tool(tool, args)
    return asyncio.run(run())


def test_tools_are_listed_with_descriptions():
    async def run():
        async with Client(job_search.server) as client:
            return await client.list_tools()
    tools = {t.name: t for t in asyncio.run(run()).tools}
    assert set(tools) == {"search_jobs", "get_job_details"}
    assert "Search UK job adverts" in tools["search_jobs"].description
    assert "keywords" in tools["search_jobs"].input_schema["required"]
    assert tools["search_jobs"].annotations.read_only_hint is True


def test_search_jobs(fake_reed):
    result = call("search_jobs", {"keywords": "data scientist", "location": "London", "max_results": 5})
    assert not result.is_error
    jobs = result.structured_content["result"]
    assert [j["id"] for j in jobs] == ["reed-1", "reed-2"]
    assert jobs[0]["title"] == "Data Scientist"
    assert len(jobs[0]["snippet"]) == 300           # shortened
    assert jobs[1]["company"] == "Unknown"           # missing employer handled
    assert fake_reed.search_calls[0]["location"] == "London"


def test_search_jobs_caps_max_results(fake_reed):
    call("search_jobs", {"keywords": "python", "max_results": 5000})
    assert fake_reed.search_calls[0]["max_results"] == 100


def test_search_jobs_rejects_empty_keywords(fake_reed):
    result = call("search_jobs", {"keywords": "  "})
    assert result.is_error
    assert "must not be empty" in result.content[0].text  # the model sees the real reason


def test_get_job_details(fake_reed):
    result = call("get_job_details", {"job_id": "reed-777"})
    assert not result.is_error
    job = result.structured_content
    assert job["id"] == "reed-777"
    assert job["apply_url"] == "https://careers.acme.com/apply/123"


def test_get_job_details_bad_id(fake_reed):
    result = call("get_job_details", {"job_id": "linkedin-abc"})
    assert result.is_error
    assert "Not a Reed job id" in result.content[0].text


def test_reed_auth_error_is_passed_to_the_model(monkeypatch):
    from job_agent.jobs.reed import ReedAuthError

    def no_key():
        raise ReedAuthError("REED_API_KEY is not set (add it to .env)")

    class Broken:
        def get_details(self, job_id):
            no_key()

    monkeypatch.setattr(job_search, "get_client", lambda: Broken())
    result = call("get_job_details", {"job_id": "reed-1"})
    assert result.is_error and "REED_API_KEY" in result.content[0].text
