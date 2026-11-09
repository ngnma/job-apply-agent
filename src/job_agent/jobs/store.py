"""Phase 2, task 4: store jobs in Postgres (table `jobs`).

- No duplicates: the job id ("reed-54321") is the primary key, so fetching the same
  job again updates it instead of adding a copy.
- Each job has a status that moves through the pipeline:
      new -> preprocessed -> filtered_out | scored -> selected | rejected
- If an advert's content changes (title, description, salary...), its status goes back
  to "new" so it is processed again.

Try it (database up + REED_API_KEY in .env):
    uv run python -m job_agent.jobs.store fetch "data scientist" --location London --max 20
    uv run python -m job_agent.jobs.store status
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Engine,
    Float,
    String,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from job_agent.contracts.job import JobPost, JobRequirements, JobStatus, MatchResult
from job_agent.rag.vectorstore import database_url

# fields that, when changed, mean the advert must be processed again
CONTENT_FIELDS = ("title", "company", "location", "salary_min", "salary_max", "contract_type", "description")


class Base(DeclarativeBase):
    pass


class JobRow(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True)          # "reed-54321"
    source: Mapped[str] = mapped_column(String, index=True)
    source_id: Mapped[str] = mapped_column(String)
    title: Mapped[str] = mapped_column(Text)
    company: Mapped[str] = mapped_column(Text)
    location: Mapped[str] = mapped_column(Text)
    salary_min: Mapped[float | None] = mapped_column(Float)
    salary_max: Mapped[float | None] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(3))
    contract_type: Mapped[str | None] = mapped_column(String)
    full_time: Mapped[bool | None] = mapped_column(Boolean)
    url: Mapped[str] = mapped_column(Text)
    apply_url: Mapped[str | None] = mapped_column(Text)
    posted_at: Mapped[date | None] = mapped_column(Date)
    expires_at: Mapped[date | None] = mapped_column(Date)
    description: Mapped[str] = mapped_column(Text)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    content_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String, index=True, default=JobStatus.NEW)
    requirements: Mapped[dict | None] = mapped_column(JSONB)   # JobRequirements (task 5)
    match: Mapped[dict | None] = mapped_column(JSONB)          # MatchResult (tasks 6-7)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(),
                                                 onupdate=func.now())


def content_hash(job: JobPost) -> str:
    data = job.model_dump(mode="json", include=set(CONTENT_FIELDS))
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _row_to_job(row: JobRow) -> JobPost:
    return JobPost(
        id=row.id, source=row.source, source_id=row.source_id, title=row.title, company=row.company,
        location=row.location, salary_min=row.salary_min, salary_max=row.salary_max,
        currency=row.currency, contract_type=row.contract_type, full_time=row.full_time,
        url=row.url, apply_url=row.apply_url, posted_at=row.posted_at, expires_at=row.expires_at,
        description=row.description, fetched_at=row.fetched_at,
    )


@dataclass
class SaveReport:
    added: int = 0
    updated: int = 0     # content changed -> status reset to "new"
    unchanged: int = 0

    def __str__(self) -> str:
        return f"added {self.added}, updated {self.updated}, unchanged {self.unchanged}"


class JobStore:
    def __init__(self, engine: Engine | None = None):
        self.engine = engine or create_engine(database_url())

    def init_db(self) -> None:
        """Create the jobs table if it does not exist."""
        Base.metadata.create_all(self.engine)

    # ---------- write ----------

    def save_jobs(self, jobs: list[JobPost]) -> SaveReport:
        report = SaveReport()
        with Session(self.engine) as session, session.begin():
            for job in jobs:
                new_hash = content_hash(job)
                row = session.get(JobRow, job.id)
                if row is None:
                    session.add(JobRow(**job.model_dump(mode="python", exclude={"url", "apply_url"}),
                                       url=str(job.url), apply_url=str(job.apply_url) if job.apply_url else None,
                                       content_hash=new_hash, status=JobStatus.NEW))
                    report.added += 1
                elif row.content_hash != new_hash:
                    for field in (*CONTENT_FIELDS, "full_time", "expires_at", "fetched_at"):
                        setattr(row, field, getattr(job, field))
                    row.apply_url = str(job.apply_url) if job.apply_url else None
                    row.content_hash = new_hash
                    row.status, row.requirements, row.match = JobStatus.NEW, None, None  # re-process
                    report.updated += 1
                else:
                    row.fetched_at = job.fetched_at  # still online: just note when we last saw it
                    report.unchanged += 1
        return report

    def set_status(self, job_id: str, status: JobStatus) -> None:
        with Session(self.engine) as session, session.begin():
            self._get_row(session, job_id).status = status

    def save_requirements(self, job_id: str, requirements: JobRequirements) -> None:
        with Session(self.engine) as session, session.begin():
            row = self._get_row(session, job_id)
            row.requirements = requirements.model_dump(mode="json")
            row.status = JobStatus.PREPROCESSED

    def save_match(self, match: MatchResult) -> None:
        with Session(self.engine) as session, session.begin():
            row = self._get_row(session, match.job_id)
            row.match = match.model_dump(mode="json")
            row.status = match.decision

    # ---------- read ----------

    def get(self, job_id: str) -> JobPost | None:
        with Session(self.engine) as session:
            row = session.get(JobRow, job_id)
            return _row_to_job(row) if row else None

    def get_status(self, job_id: str) -> JobStatus | None:
        with Session(self.engine) as session:
            row = session.get(JobRow, job_id)
            return JobStatus(row.status) if row else None

    def get_requirements(self, job_id: str) -> JobRequirements | None:
        with Session(self.engine) as session:
            row = session.get(JobRow, job_id)
            return JobRequirements.model_validate(row.requirements) if row and row.requirements else None

    def get_match(self, job_id: str) -> MatchResult | None:
        with Session(self.engine) as session:
            row = session.get(JobRow, job_id)
            return MatchResult.model_validate(row.match) if row and row.match else None

    def list_by_status(self, status: JobStatus, limit: int = 100) -> list[JobPost]:
        """Oldest first, so jobs are processed in the order they arrived."""
        with Session(self.engine) as session:
            rows = session.scalars(
                select(JobRow).where(JobRow.status == status).order_by(JobRow.created_at).limit(limit))
            return [_row_to_job(r) for r in rows]

    def counts_by_status(self) -> dict[str, int]:
        with Session(self.engine) as session:
            rows = session.execute(select(JobRow.status, func.count()).group_by(JobRow.status))
            return {status: n for status, n in rows}

    @staticmethod
    def _get_row(session: Session, job_id: str) -> JobRow:
        row = session.get(JobRow, job_id)
        if row is None:
            raise KeyError(f"No job with id {job_id!r}")
        return row


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Fetch jobs from Reed into the database")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="search Reed and save the jobs")
    fetch.add_argument("keywords")
    fetch.add_argument("--location")
    fetch.add_argument("--max", type=int, default=20)
    sub.add_parser("status", help="how many jobs are in each status")
    args = parser.parse_args()

    store = JobStore()
    store.init_db()
    if args.command == "fetch":
        from job_agent.jobs.reed import ReedClient

        client = ReedClient()
        try:
            jobs = client.fetch_jobs(args.keywords, location=args.location, max_results=args.max)
        finally:
            client.close()
        print(f"Fetched {len(jobs)} jobs: {store.save_jobs(jobs)}")
    print(f"Jobs by status ({datetime.now(UTC):%Y-%m-%d %H:%M} UTC): {store.counts_by_status()}")


if __name__ == "__main__":
    main()
