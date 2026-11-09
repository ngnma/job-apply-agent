"""Task 7: ingestion pipeline — fill the vector DB from master_cv.json.

    master_cv.json -> chunks -> embeddings -> pgvector (table cv_chunks)

Run (needs the database running and Ollama with nomic-embed-text):
    uv run python -m job_agent.rag.ingestion data/master_cv.json

Safe to run again and again ("idempotent"):
    - unchanged chunks are skipped (no new embedding = fast and cheap)
    - changed chunks are re-embedded and updated
    - chunks no longer in the CV are deleted
Use --rebuild to delete everything and start again (e.g. after changing the embedding model).
"""

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy import create_engine, text

from job_agent.rag.chunking import chunk_master_cv, load_master_cv
from job_agent.rag.vectorstore import (
    TABLE_NAME,
    create_hnsw_index,
    create_table,
    database_url,
    get_engine,
    get_vector_store,
    table_exists,
)

HASH_KEY = "content_hash"


# ---------- 1. fingerprint each chunk ----------

def embedding_model_name(embeddings: Embeddings) -> str:
    base = getattr(embeddings, "base", embeddings)  # unwrap PrefixedEmbeddings
    return str(getattr(base, "model", None) or getattr(base, "model_id", None) or type(base).__name__)


def content_hash(chunk: Document, model_name: str) -> str:
    """Changes when the text, the metadata or the embedding model changes."""
    payload = json.dumps(
        {"text": chunk.page_content, "metadata": chunk.metadata, "model": model_name},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------- 2. compare with what is already in the database ----------

@dataclass
class Plan:
    to_add: list[Document] = field(default_factory=list)      # new IDs
    to_update: list[Document] = field(default_factory=list)   # same ID, different content
    unchanged: list[str] = field(default_factory=list)        # same ID, same hash
    to_delete: list[str] = field(default_factory=list)        # in DB but not in the CV any more


def plan_changes(chunks: list[Document], existing: dict[str, str]) -> Plan:
    """chunks must already carry metadata[HASH_KEY]; existing = {id: hash} from the DB."""
    plan = Plan()
    for chunk in chunks:
        old_hash = existing.get(chunk.id)
        if old_hash is None:
            plan.to_add.append(chunk)
        elif old_hash != chunk.metadata[HASH_KEY]:
            plan.to_update.append(chunk)
        else:
            plan.unchanged.append(chunk.id)
    current_ids = {c.id for c in chunks}
    plan.to_delete = sorted(i for i in existing if i not in current_ids)
    return plan


def stored_hashes(table_name: str = TABLE_NAME) -> dict[str, str]:
    """{chunk id: content hash} for every row already in the table."""
    sql = text(f"SELECT langchain_id, langchain_metadata->>'{HASH_KEY}' FROM \"{table_name}\"")
    with create_engine(database_url()).connect() as conn:
        return {row[0]: row[1] for row in conn.execute(sql)}


def stored_vector_size(table_name: str = TABLE_NAME) -> int | None:
    """Vector length of the rows already stored (None if the table is empty)."""
    sql = text(f'SELECT vector_dims(embedding) FROM "{table_name}" LIMIT 1')
    with create_engine(database_url()).connect() as conn:
        return conn.execute(sql).scalar()


# ---------- 3. run ----------

@dataclass
class Report:
    added: int
    updated: int
    unchanged: int
    deleted: int

    def __str__(self) -> str:
        return (f"added {self.added}, updated {self.updated}, "
                f"unchanged {self.unchanged}, deleted {self.deleted}")


def ingest(cv_path: str | Path, embeddings: Embeddings, table_name: str = TABLE_NAME,
           rebuild: bool = False) -> Report:
    engine = get_engine()

    # make sure the table exists (and start from empty if rebuild)
    vector_size = len(embeddings.embed_query("test"))
    if rebuild or not table_exists(table_name=table_name):
        create_table(engine, vector_size=vector_size, table_name=table_name, overwrite=True)
    elif (stored := stored_vector_size(table_name)) not in (None, vector_size):
        raise ValueError(
            f"The table has {stored}-number vectors but the embedding model makes {vector_size}. "
            "You changed the embedding model: run again with --rebuild."
        )
    store = get_vector_store(engine, embeddings, table_name=table_name)

    # chunks + fingerprints
    model = embedding_model_name(embeddings)
    chunks = chunk_master_cv(load_master_cv(cv_path))
    for chunk in chunks:
        chunk.metadata[HASH_KEY] = content_hash(chunk, model)

    plan = plan_changes(chunks, stored_hashes(table_name))

    # write: add_documents with the same IDs updates rows ("upsert"), so add + update are one call
    to_write = plan.to_add + plan.to_update
    if to_write:
        store.add_documents(to_write, ids=[c.id for c in to_write])
    if plan.to_delete:
        store.delete(ids=plan.to_delete)

    create_hnsw_index(store)
    return Report(len(plan.to_add), len(plan.to_update), len(plan.unchanged), len(plan.to_delete))


def main() -> None:
    from dotenv import load_dotenv

    from job_agent.rag.embeddings import get_embeddings

    load_dotenv()
    parser = argparse.ArgumentParser(description="Load master_cv.json into the vector DB")
    parser.add_argument("cv_path", nargs="?", default="data/master_cv.json")
    parser.add_argument("--rebuild", action="store_true", help="drop all rows and re-embed everything")
    args = parser.parse_args()

    report = ingest(args.cv_path, get_embeddings(), rebuild=args.rebuild)
    print(f"Ingestion done: {report}")


if __name__ == "__main__":
    main()
