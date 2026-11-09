"""Task 8: hybrid retriever over the CV chunks in pgvector.

Two searches, then one merged ranking:
    1. vector search  -> similar MEANING   ("data workflows" finds "ETL pipeline")
    2. keyword search -> exact WORDS       ("PyTorch", "GDPR", "XGBoost")
    3. Reciprocal Rank Fusion (RRF) merges both lists: a chunk ranked high in
       either list (or in both) ends up near the top.

Try it (database up + chunks ingested):
    uv run python -m job_agent.rag.retriever "Python data pipelines with Airflow"
    uv run python -m job_agent.rag.retriever "PyTorch" --types project skill --k 5
"""

import argparse
import re
from collections import OrderedDict
from dataclasses import dataclass

from langchain_postgres import PGVectorStore
from sqlalchemy import create_engine, text

from job_agent.rag.vectorstore import TABLE_NAME, database_url

CHUNK_TYPES = ("experience", "project", "skill", "education", "certification")


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict
    score: float                     # fused RRF score (higher = better)
    vector_rank: int | None = None   # position in the vector results (1 = best), None if not found
    keyword_rank: int | None = None  # position in the keyword results

    @property
    def type(self) -> str:
        return self.metadata.get("type", "")

    @property
    def parent_id(self) -> str:
        return self.metadata.get("parent_id", self.id)


# ---------- keyword search (Postgres full-text search) ----------

def build_tsquery(query: str) -> str:
    """'Python & Airflow pipelines' -> 'python | airflow | pipelines'.

    OR (|) instead of AND: a long job-post sentence should still match chunks that
    contain only some of its words. Only letters/digits are kept, so it is safe SQL input.
    """
    words = dict.fromkeys(w.lower() for w in re.findall(r"[A-Za-z0-9]+", query))
    return " | ".join(words)


def keyword_index_sql(table_name: str = TABLE_NAME) -> str:
    return (f'CREATE INDEX IF NOT EXISTS "{table_name}_content_fts" '
            f"ON \"{table_name}\" USING GIN (to_tsvector('english', content))")


def create_keyword_index(table_name: str = TABLE_NAME) -> None:
    """GIN index = fast full-text search (like HNSW, but for words)."""
    with create_engine(database_url()).begin() as conn:
        conn.execute(text(keyword_index_sql(table_name)))


# ---------- merge the two rankings ----------

def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """score(chunk) = sum over lists of 1 / (k + rank). k=60 is the standard value;
    it stops the very first positions from dominating."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


# ---------- the retriever ----------

class CVRetriever:
    def __init__(self, store: PGVectorStore, table_name: str = TABLE_NAME,
                 candidates: int = 20, rrf_k: int = 60):
        self.store = store
        self.table_name = table_name
        self.candidates = candidates  # how many results each search returns before merging
        self.rrf_k = rrf_k
        self._db = create_engine(database_url())

    def vector_search(self, query: str, k: int, types: list[str] | None = None) -> list[RetrievedChunk]:
        filter_ = {"type": {"$in": list(types)}} if types else None
        results = self.store.similarity_search_with_score(query, k=k, filter=filter_)
        return [
            RetrievedChunk(id=doc.id, text=doc.page_content, metadata=doc.metadata,
                           score=1 - distance, vector_rank=rank)  # cosine distance -> similarity
            for rank, (doc, distance) in enumerate(results, start=1)
        ]

    def keyword_search(self, query: str, k: int, types: list[str] | None = None) -> list[RetrievedChunk]:
        tsquery = build_tsquery(query)
        if not tsquery:
            return []
        type_filter = "AND type = ANY(:types)" if types else ""
        sql = text(f"""
            SELECT langchain_id, content, type, parent_id, langchain_metadata,
                   ts_rank_cd(to_tsvector('english', content), q) AS rank
            FROM "{self.table_name}", to_tsquery('english', :tsquery) AS q
            WHERE to_tsvector('english', content) @@ q {type_filter}
            ORDER BY rank DESC
            LIMIT :k
        """)
        params = {"tsquery": tsquery, "k": k, "types": list(types or [])}
        with self._db.connect() as conn:
            rows = conn.execute(sql, params).all()
        return [
            RetrievedChunk(id=r.langchain_id, text=r.content,
                           metadata={**(r.langchain_metadata or {}), "type": r.type, "parent_id": r.parent_id},
                           score=float(r.rank), keyword_rank=rank)
            for rank, r in enumerate(rows, start=1)
        ]

    def search(self, query: str, k: int = 5, types: list[str] | None = None) -> list[RetrievedChunk]:
        """Hybrid search: the top k chunks by fused rank."""
        vector_hits = self.vector_search(query, self.candidates, types)
        keyword_hits = self.keyword_search(query, self.candidates, types)

        fused = reciprocal_rank_fusion(
            [[c.id for c in vector_hits], [c.id for c in keyword_hits]], k=self.rrf_k)

        by_id: dict[str, RetrievedChunk] = {}
        for hit in vector_hits + keyword_hits:
            chunk = by_id.setdefault(hit.id, RetrievedChunk(hit.id, hit.text, hit.metadata, 0.0))
            chunk.vector_rank = chunk.vector_rank or hit.vector_rank
            chunk.keyword_rank = chunk.keyword_rank or hit.keyword_rank
            chunk.score = fused[hit.id]

        return sorted(by_id.values(), key=lambda c: c.score, reverse=True)[:k]


def group_by_parent(chunks: list[RetrievedChunk]) -> "OrderedDict[str, list[RetrievedChunk]]":
    """Put bullets of the same job/project together; the group with the best chunk comes first."""
    groups: OrderedDict[str, list[RetrievedChunk]] = OrderedDict()
    for chunk in sorted(chunks, key=lambda c: c.score, reverse=True):
        groups.setdefault(chunk.parent_id, []).append(chunk)
    return groups


def main() -> None:
    from dotenv import load_dotenv

    from job_agent.rag.embeddings import get_embeddings
    from job_agent.rag.vectorstore import get_engine, get_vector_store

    load_dotenv()
    parser = argparse.ArgumentParser(description="Search the CV chunks")
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--types", nargs="*", choices=CHUNK_TYPES, help="only these chunk types")
    args = parser.parse_args()

    retriever = CVRetriever(get_vector_store(get_engine(), get_embeddings()))
    results = retriever.search(args.query, k=args.k, types=args.types)

    print(f"{'score':>7}  {'vec':>3}  {'kw':>3}  id")
    for c in results:
        print(f"{c.score:7.4f}  {c.vector_rank or '-':>3}  {c.keyword_rank or '-':>3}  {c.id}")
        print(f"{'':19}{c.text.splitlines()[-1][:90]}")


if __name__ == "__main__":
    main()
