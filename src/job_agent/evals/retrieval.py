"""Task 10: retrieval evaluation — how well does the retriever find the right chunks?

Dataset: evals/retrieval_dataset.json  (query + the chunk IDs that SHOULD be returned)

Metrics (all between 0 and 1, higher = better), averaged over all queries:
    hit@k     - did at least one relevant chunk appear in the top k?
    recall@k  - what share of the relevant chunks appeared in the top k?
    MRR       - 1 / position of the first relevant chunk (1st -> 1.0, 2nd -> 0.5, missing -> 0)

It compares three modes: vector only, keyword only and hybrid, so you can see what each adds.

Run (database up + chunks ingested):
    uv run python -m job_agent.evals.retrieval
    uv run python -m job_agent.evals.retrieval --k 5 --min-recall 0.6   # exit code 1 if below -> for CI
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, text

from job_agent.rag.retriever import CVRetriever
from job_agent.rag.vectorstore import TABLE_NAME, database_url

DATASET = Path("evals/retrieval_dataset.json")
RESULTS_DIR = Path("evals/results")
MODES = ("vector", "keyword", "hybrid")


# ---------- metrics for one query ----------

def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for position, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / position
    return 0.0


# ---------- run the eval ----------

@dataclass
class QueryResult:
    query: str
    hit: float
    recall: float
    rr: float
    missed: list[str]


@dataclass
class ModeResult:
    mode: str
    k: int
    hit: float = 0.0
    recall: float = 0.0
    mrr: float = 0.0
    queries: list[QueryResult] = field(default_factory=list)


def load_dataset(path: Path = DATASET) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["queries"]


def unknown_ids(dataset: list[dict], table_name: str = TABLE_NAME) -> list[str]:
    """Relevant IDs that are not in the database (the dataset is out of date with the CV)."""
    with create_engine(database_url()).connect() as conn:
        stored = set(conn.execute(text(f'SELECT langchain_id FROM "{table_name}"')).scalars())
    return sorted({i for q in dataset for i in q["relevant"]} - stored)


def run_mode(retriever: CVRetriever, dataset: list[dict], mode: str, k: int) -> ModeResult:
    search = {"vector": retriever.vector_search,
              "keyword": retriever.keyword_search,
              "hybrid": retriever.search}[mode]
    result = ModeResult(mode=mode, k=k)
    for item in dataset:
        relevant = set(item["relevant"])
        retrieved = [c.id for c in search(item["query"], k=k, types=item.get("types"))]
        result.queries.append(QueryResult(
            query=item["query"],
            hit=hit_at_k(retrieved, relevant, k),
            recall=recall_at_k(retrieved, relevant, k),
            rr=reciprocal_rank(retrieved, relevant),
            missed=sorted(relevant - set(retrieved)),
        ))
    n = len(result.queries) or 1
    result.hit = sum(q.hit for q in result.queries) / n
    result.recall = sum(q.recall for q in result.queries) / n
    result.mrr = sum(q.rr for q in result.queries) / n
    return result


def print_report(results: list[ModeResult], worst: int = 5) -> None:
    k = results[0].k
    print(f"\n{'mode':<9}{'hit@' + str(k):>8}{'recall@' + str(k):>11}{'MRR':>7}")
    for r in results:
        print(f"{r.mode:<9}{r.hit:8.2f}{r.recall:11.2f}{r.mrr:7.2f}")

    hybrid = next(r for r in results if r.mode == "hybrid")
    print(f"\nWorst {worst} queries (hybrid):")
    for q in sorted(hybrid.queries, key=lambda q: (q.recall, q.rr))[:worst]:
        print(f"  recall {q.recall:.2f}  {q.query}")
        if q.missed:
            print(f"               missed: {', '.join(q.missed)}")


def save_report(results: list[ModeResult], out_dir: Path = RESULTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"retrieval_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    return path


def main() -> None:
    from dotenv import load_dotenv

    from job_agent.rag.embeddings import get_embeddings
    from job_agent.rag.vectorstore import get_engine, get_vector_store

    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate the CV retriever")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-recall", type=float, default=None,
                        help="fail (exit code 1) if hybrid recall@k is below this")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    if missing := unknown_ids(dataset):
        print(f"WARNING: {len(missing)} relevant IDs are not in the database "
              f"(update the dataset or re-run ingestion): {', '.join(missing)}")

    retriever = CVRetriever(get_vector_store(get_engine(), get_embeddings()))
    results = [run_mode(retriever, dataset, mode, args.k) for mode in MODES]
    print_report(results)
    print(f"\nSaved {save_report(results)}")

    hybrid_recall = next(r.recall for r in results if r.mode == "hybrid")
    if args.min_recall is not None and hybrid_recall < args.min_recall:
        sys.exit(f"FAILED: hybrid recall@{args.k} = {hybrid_recall:.2f} < {args.min_recall}")


if __name__ == "__main__":
    main()
