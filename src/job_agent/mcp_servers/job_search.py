"""Phase 2, task 3: Job search MCP server.

Exposes the Reed client as MCP tools, so any MCP client (our LangGraph agent later,
Claude Desktop, the MCP Inspector) can search UK jobs without knowing about Reed.

Errors we expect (bad id, bad API key) are raised as ToolError, so the calling
model sees the actual message and can correct itself.

Tools:
    search_jobs(keywords, ...)  -> short summaries (cheap: 1 request per 100 jobs)
    get_job_details(job_id)     -> one full job post (1 request)

Run it (stdio transport, the standard for local MCP servers):
    uv run python -m job_agent.mcp_servers.job_search

Inspect it in a browser (needs Node.js):
    npx @modelcontextprotocol/inspector uv run python -m job_agent.mcp_servers.job_search
"""

from datetime import date

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from job_agent.contracts.job import JobPost
from job_agent.jobs.reed import ReedClient, ReedError, _parse_date, to_job_post

server = MCPServer(
    name="job-search",
    instructions=(
        "Search UK job adverts (source: Reed). Use search_jobs to find candidate jobs, "
        "then get_job_details for the full description of the ones worth reading."
    ),
)

# read-only (changes nothing) + open-world (talks to an external website)
READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=True)

_client: ReedClient | None = None


def get_client() -> ReedClient:
    """One Reed client per server process, created on first use (tests replace this)."""
    global _client
    if _client is None:
        from dotenv import load_dotenv

        load_dotenv()
        _client = ReedClient()
    return _client


class JobSummary(BaseModel):
    """A short search result. Call get_job_details(id) for the full advert."""
    id: str = Field(description="Job ID to pass to get_job_details, e.g. 'reed-54321'")
    title: str
    company: str
    location: str
    salary_min: float | None = None
    salary_max: float | None = None
    posted_at: date | None = None
    url: str
    snippet: str = Field(description="First part of the description")


def to_summary(r: dict) -> JobSummary:
    return JobSummary(
        id=f"reed-{r['jobId']}",
        title=r.get("jobTitle", "").strip(),
        company=(r.get("employerName") or "Unknown").strip(),
        location=(r.get("locationName") or "Unknown").strip(),
        salary_min=r.get("minimumSalary"),
        salary_max=r.get("maximumSalary"),
        posted_at=_parse_date(r.get("date")),
        url=r.get("jobUrl") or f"https://www.reed.co.uk/jobs/{r['jobId']}",
        snippet=(r.get("jobDescription") or "")[:300],
    )


@server.tool(annotations=READ_ONLY)
def search_jobs(
    keywords: str,
    location: str | None = None,
    distance_miles: int = 10,
    min_salary: int | None = None,
    permanent: bool | None = None,
    graduate: bool | None = None,
    max_results: int = 20,
) -> list[JobSummary]:
    """Search UK job adverts on Reed.

    Args:
        keywords: Job title or skills, e.g. "data scientist python".
        location: Town or city, e.g. "London". Leave empty for all of the UK.
        distance_miles: Search radius around the location.
        min_salary: Minimum yearly salary in GBP.
        permanent: True for permanent roles only.
        graduate: True for graduate roles only.
        max_results: How many jobs to return (1-100).
    """
    if not keywords.strip():
        raise ToolError("keywords must not be empty")
    max_results = max(1, min(max_results, 100))
    try:
        results = get_client().search(
            keywords, location=location, distance_miles=distance_miles, min_salary=min_salary,
            permanent=permanent, graduate=graduate, max_results=max_results,
        )
    except ReedError as exc:  # e.g. bad API key -> clear message for the model
        raise ToolError(str(exc)) from exc
    return [to_summary(r) for r in results]


@server.tool(annotations=READ_ONLY)
def get_job_details(job_id: str) -> JobPost:
    """Get the full job advert (complete description, salary, dates, apply link).

    Args:
        job_id: The id from search_jobs, e.g. "reed-54321" (a plain "54321" also works).
    """
    source_id = job_id.removeprefix("reed-").strip()
    if not source_id.isdigit():
        raise ToolError(f"Not a Reed job id: {job_id!r}. Use an id from search_jobs, e.g. 'reed-54321'.")
    try:
        return to_job_post(get_client().get_details(source_id))
    except ReedError as exc:
        raise ToolError(str(exc)) from exc


if __name__ == "__main__":
    server.run()  # stdio
