"""Task 6: the vector database (Postgres + pgvector) for CV chunks.

Start the database (Docker):
    docker compose up -d db

Create the table + index once:
    uv run python -m job_agent.rag.vectorstore init

Check it:
    uv run python -m job_agent.rag.vectorstore status
"""

import asyncio
import os
import sys

from langchain_core.embeddings import Embeddings
from langchain_postgres import Column, PGEngine, PGVectorStore
from langchain_postgres.v2.indexes import HNSWIndex
from sqlalchemy import create_engine, inspect, text

# Windows fix: psycopg's async mode cannot run on Windows' default event loop
# (ProactorEventLoop). Use the selector loop instead. Must run before the engine starts.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

DEFAULT_DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/job_agent"
TABLE_NAME = "cv_chunks"

# Metadata keys stored as real table columns (fast to filter on, e.g. "only projects").
# Every other metadata key is kept in one JSON column.
FILTER_COLUMNS = ["type", "parent_id"]


def database_url() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def get_engine(url: str | None = None) -> PGEngine:
    return PGEngine.from_connection_string(url or database_url())


def create_table(engine: PGEngine, vector_size: int, table_name: str = TABLE_NAME,
                 overwrite: bool = False) -> None:
    """Create the chunks table. vector_size must match the embedding model (768 for nomic)."""
    engine.init_vectorstore_table(
        table_name=table_name,
        vector_size=vector_size,
        id_column=Column("langchain_id", "TEXT", nullable=False),  # our IDs are text, e.g. "exp-snappshop-b3"
        metadata_columns=[Column(name, "TEXT") for name in FILTER_COLUMNS],
        overwrite_existing=overwrite,
    )


def get_vector_store(engine: PGEngine, embeddings: Embeddings,
                     table_name: str = TABLE_NAME) -> PGVectorStore:
    """Connect LangChain to the existing table (cosine distance by default)."""
    return PGVectorStore.create_sync(
        engine=engine,
        embedding_service=embeddings,
        table_name=table_name,
        metadata_columns=FILTER_COLUMNS,
    )


def create_hnsw_index(store: PGVectorStore) -> None:
    """HNSW index = fast approximate nearest-neighbour search, instead of scanning every row."""
    if not store.is_valid_index():
        store.apply_vector_index(HNSWIndex())


def table_exists(url: str | None = None, table_name: str = TABLE_NAME) -> bool:
    return inspect(create_engine(url or database_url())).has_table(table_name)


def count_rows(url: str | None = None, table_name: str = TABLE_NAME) -> int:
    engine = create_engine(url or database_url())
    with engine.connect() as conn:
        return conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar_one()


def main() -> None:
    from dotenv import load_dotenv

    from job_agent.rag.embeddings import embedding_dimension, get_embeddings

    load_dotenv()  # read settings from .env, if it exists

    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    engine = get_engine()
    embeddings = get_embeddings()

    if command == "init":  # safe to run again: skips what already exists
        size = embedding_dimension(embeddings)
        if table_exists():
            print(f"Table '{TABLE_NAME}' already exists.")
        else:
            create_table(engine, vector_size=size)
            print(f"Created table '{TABLE_NAME}' (vector size {size}).")
        create_hnsw_index(get_vector_store(engine, embeddings))
        print("HNSW index ready.")
    elif command == "status":
        print(f"Table '{TABLE_NAME}': {count_rows()} rows")
    else:
        sys.exit(f"Unknown command {command!r}. Use: init | status")


if __name__ == "__main__":
    main()
