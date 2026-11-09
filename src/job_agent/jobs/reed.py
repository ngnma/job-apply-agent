"""Phase 2, task 2: client for the Reed jobseeker API (https://www.reed.co.uk/developers).

    search()       -> short job summaries (description is cut short), paged 100 at a time
    get_details()  -> one job with the FULL description
    fetch_jobs()   -> search + details, returned as JobPost objects (our own schema)

Auth: HTTP Basic, API key as the username and an empty password.

Try it (needs REED_API_KEY in .env):
    uv run python -m job_agent.jobs.reed "data scientist" --location London --max 5
"""

import argparse
import os
import time
from datetime import date, datetime
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from job_agent.contracts.job import JobPost

BASE_URL = "https://www.reed.co.uk/api/1.0"
PAGE_SIZE = 100  # the most Reed returns per request


class ReedError(Exception):
    """Any problem talking to Reed."""


class ReedAuthError(ReedError):
    """Missing or wrong API key (HTTP 401)."""


def _is_temporary(exc: BaseException) -> bool:
    """Worth retrying: network glitches, rate limiting (429) and server errors (5xx)."""
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and (
        exc.response.status_code == 429 or exc.response.status_code >= 500
    )


class ReedClient:
    def __init__(self, api_key: str | None = None, *, min_interval: float = 0.5,
                 timeout: float = 20.0, http: httpx.Client | None = None):
        api_key = api_key or os.getenv("REED_API_KEY")
        if not api_key:
            raise ReedAuthError("REED_API_KEY is not set (add it to .env)")
        self.min_interval = min_interval  # seconds between requests: Reed publishes no rate limit, so be polite
        self._last_request = 0.0
        self.http = http or httpx.Client(base_url=BASE_URL, auth=(api_key, ""), timeout=timeout)

    # ---------- low-level ----------

    @retry(retry=retry_if_exception(_is_temporary), stop=stop_after_attempt(4),
           wait=wait_exponential(multiplier=1, max=20), reraise=True)
    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

        response = self.http.get(path, params=params)
        if response.status_code == 401:
            raise ReedAuthError("Reed rejected the API key (401). Check REED_API_KEY.")
        response.raise_for_status()
        return response.json()

    # ---------- API calls ----------

    def search(self, keywords: str, *, location: str | None = None, distance_miles: int = 10,
               min_salary: int | None = None, permanent: bool | None = None,
               graduate: bool | None = None, max_results: int = 100) -> list[dict]:
        """Job summaries, following Reed's paging until max_results or no more results."""
        params: dict[str, Any] = {"keywords": keywords, "distanceFromLocation": distance_miles}
        if location:
            params["locationName"] = location
        if min_salary is not None:
            params["minimumSalary"] = min_salary
        if permanent is not None:
            params["permanent"] = str(permanent).lower()
        if graduate is not None:
            params["graduate"] = str(graduate).lower()

        results: list[dict] = []
        while len(results) < max_results:
            take = min(PAGE_SIZE, max_results - len(results))
            page = self._get("/search", {**params, "resultsToTake": take, "resultsToSkip": len(results)})
            batch = page.get("results", [])
            results += batch
            if len(batch) < take or len(results) >= page.get("totalResults", 0):
                break  # last page
        return results[:max_results]

    def get_details(self, job_id: int | str) -> dict:
        return self._get(f"/jobs/{job_id}")

    def fetch_jobs(self, keywords: str, **search_kwargs: Any) -> list[JobPost]:
        """Search, then load each job's full details and convert to JobPost.
        A job that fails to load or convert is skipped, not fatal."""
        jobs: list[JobPost] = []
        for summary in self.search(keywords, **search_kwargs):
            try:
                jobs.append(to_job_post(self.get_details(summary["jobId"])))
            except (ReedError, httpx.HTTPError, ValueError, KeyError) as exc:
                if isinstance(exc, ReedAuthError):
                    raise
                print(f"Skipped job {summary.get('jobId')}: {exc}")
        return jobs

    def close(self) -> None:
        self.http.close()


# ---------- Reed JSON -> our JobPost ----------

def _parse_date(value: str | None) -> date | None:
    """Reed dates look like '25/11/2026'."""
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%d/%m/%Y").date()
    except ValueError:
        return None


CONTRACT_TYPES = {"permanent": "permanent", "contract": "contract", "temporary": "temp", "temp": "temp"}


def to_job_post(d: dict) -> JobPost:
    """Map Reed's job-details JSON to our JobPost schema."""
    job_id = str(d["jobId"])
    return JobPost(
        id=f"reed-{job_id}",
        source="reed",
        source_id=job_id,
        title=d["jobTitle"].strip(),
        company=(d.get("employerName") or "Unknown").strip(),
        location=(d.get("locationName") or "Unknown").strip(),
        # prefer Reed's yearly figures (some adverts give daily/hourly pay)
        salary_min=d.get("yearlyMinimumSalary") or d.get("minimumSalary"),
        salary_max=d.get("yearlyMaximumSalary") or d.get("maximumSalary"),
        currency=d.get("currency") or "GBP",
        contract_type=CONTRACT_TYPES.get(str(d.get("contractType") or "").lower()),
        full_time=d.get("fullTime"),
        url=d.get("jobUrl") or f"https://www.reed.co.uk/jobs/{job_id}",
        apply_url=d.get("externalUrl") or None,
        posted_at=_parse_date(d.get("datePosted") or d.get("date")),
        expires_at=_parse_date(d.get("expirationDate")),
        description=d.get("jobDescription") or d["jobTitle"],
    )


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Search Reed and print the jobs found")
    parser.add_argument("keywords")
    parser.add_argument("--location")
    parser.add_argument("--max", type=int, default=5)
    args = parser.parse_args()

    client = ReedClient()
    try:
        jobs = client.fetch_jobs(args.keywords, location=args.location, max_results=args.max)
    finally:
        client.close()
    for job in jobs:
        salary = f"£{job.salary_min:,.0f}-£{job.salary_max:,.0f}" if job.salary_min and job.salary_max else "salary n/a"
        print(f"{job.id:<14} {job.title[:45]:<45} | {job.company[:25]:<25} | {job.location} | {salary}")
    print(f"\n{len(jobs)} jobs")


if __name__ == "__main__":
    main()
