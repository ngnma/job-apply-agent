"""Reed client tests. All HTTP calls are mocked with respx: no real requests, no API key needed."""

import base64
from datetime import date

import httpx
import pytest
import respx

from job_agent.jobs.reed import BASE_URL, ReedAuthError, ReedClient, to_job_post

DETAILS = {
    "jobId": 54321,
    "employerName": "Acme Ltd",
    "jobTitle": "Graduate Data Scientist ",
    "locationName": "London",
    "minimumSalary": 160.0,            # daily rate...
    "maximumSalary": 200.0,
    "yearlyMinimumSalary": 32000.0,    # ...but the yearly figure is preferred
    "yearlyMaximumSalary": 38000.0,
    "currency": "GBP",
    "contractType": "Permanent",
    "fullTime": True,
    "datePosted": "01/11/2026",
    "expirationDate": "13/12/2026",
    "externalUrl": "https://careers.acme.com/apply/123",
    "jobUrl": "https://www.reed.co.uk/jobs/graduate-data-scientist/54321",
    "jobDescription": "<p>We want <b>Python</b> and SQL.</p>",
}


def summary(job_id: int) -> dict:
    return {"jobId": job_id, "jobTitle": f"Job {job_id}"}


@pytest.fixture
def client():
    return ReedClient(api_key="test-key", min_interval=0)  # no waiting between requests in tests


# ---------- mapping ----------

def test_to_job_post_maps_fields():
    job = to_job_post(DETAILS)
    assert job.id == "reed-54321" and job.source_id == "54321"
    assert job.title == "Graduate Data Scientist"  # trimmed
    assert (job.salary_min, job.salary_max) == (32000.0, 38000.0)  # yearly figures
    assert job.contract_type == "permanent"
    assert job.posted_at == date(2026, 11, 1)
    assert job.expires_at == date(2026, 12, 13)
    assert str(job.apply_url) == "https://careers.acme.com/apply/123"


def test_to_job_post_handles_missing_fields():
    job = to_job_post({"jobId": 1, "jobTitle": "Analyst", "contractType": "Temporary", "externalUrl": ""})
    assert job.company == "Unknown"
    assert job.contract_type == "temp"
    assert job.apply_url is None          # empty string -> None
    assert job.salary_min is None
    assert job.description == "Analyst"   # falls back to the title


# ---------- API calls ----------

@respx.mock
def test_sends_api_key_as_basic_auth(client):
    route = respx.get(f"{BASE_URL}/jobs/1").mock(return_value=httpx.Response(200, json=DETAILS))
    client.get_details(1)
    auth = route.calls.last.request.headers["Authorization"]
    assert auth == "Basic " + base64.b64encode(b"test-key:").decode()  # key as username, empty password


@respx.mock
def test_search_follows_paging(client):
    pages = [
        httpx.Response(200, json={"results": [summary(i) for i in range(100)], "totalResults": 150}),
        httpx.Response(200, json={"results": [summary(i) for i in range(100, 150)], "totalResults": 150}),
    ]
    route = respx.get(f"{BASE_URL}/search").mock(side_effect=pages)
    results = client.search("data scientist", location="London", max_results=500)
    assert len(results) == 150
    assert route.call_count == 2
    assert route.calls[1].request.url.params["resultsToSkip"] == "100"
    assert route.calls[0].request.url.params["locationName"] == "London"


@respx.mock
def test_search_respects_max_results(client):
    respx.get(f"{BASE_URL}/search").mock(
        return_value=httpx.Response(200, json={"results": [summary(i) for i in range(5)], "totalResults": 999}))
    results = client.search("python", max_results=5)
    assert len(results) == 5


@respx.mock
def test_retries_temporary_errors(client, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)  # skip the back-off wait
    route = respx.get(f"{BASE_URL}/jobs/1").mock(side_effect=[
        httpx.Response(503), httpx.Response(429), httpx.Response(200, json=DETAILS)])
    assert client.get_details(1)["jobId"] == 54321
    assert route.call_count == 3


@respx.mock
def test_wrong_key_raises_clear_error(client):
    respx.get(f"{BASE_URL}/jobs/1").mock(return_value=httpx.Response(401))
    with pytest.raises(ReedAuthError, match="rejected the API key"):
        client.get_details(1)


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("REED_API_KEY", raising=False)
    with pytest.raises(ReedAuthError, match="not set"):
        ReedClient()


@respx.mock
def test_fetch_jobs_skips_broken_job(client, capsys):
    respx.get(f"{BASE_URL}/search").mock(
        return_value=httpx.Response(200, json={"results": [summary(1), summary(2)], "totalResults": 2}))
    respx.get(f"{BASE_URL}/jobs/1").mock(return_value=httpx.Response(200, json=DETAILS))
    respx.get(f"{BASE_URL}/jobs/2").mock(return_value=httpx.Response(404))
    jobs = client.fetch_jobs("data")
    assert [j.id for j in jobs] == ["reed-54321"]
    assert "Skipped job 2" in capsys.readouterr().out
