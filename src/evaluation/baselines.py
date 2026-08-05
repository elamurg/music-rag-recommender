"""Implementation of the four comparison baselines. the input for each baseline is a query string and k (target number of results),
returns a ranked list of track_ids.

- baseline_random = random k tracks
- baseline_popularity = top k listener_count (no personalisation)
- baseline_bm25 = BM25 keyword retrieval over document_text (sparse)
- baseline_dense = FAISS-only retrieval (dense, no LLM)
- baseline_rag = full RAG: FASS + LLM re-ranking (dense + LLM)"""

import json
import random
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.corpus.db import get_conn
from src.retrieval.query import retrieve, TrackHit
from src.generation.generate import (_call_claude, _parse_response, _filter_hallucinated)
from src.generation.prompt import build_user_prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_bm25_index: BM25Okapi | None = None
_bm25_track_ids: list[int] = []

def _simple_tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer for BM25"""
    import re
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t)>1] #filter single char tokens

def _load_bm25_index() -> tuple[BM25Okapi, list[int]]:
    """To build BM25 index over all track_documents, cached at module level."""
    global _bm25_index, _bm25_track_ids
    if _bm25_index is not None:
        return _bm25_index, _bm25_track_ids
    
    print("baselines building BM25 index over corpus documents...")
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT track_id, document_text FROM track_documents ORDER BY track_id"
        ).fetchall()
        track_ids = [r["track_id"] for r in rows]
        tokenised_docs = [_simple_tokenize(r["document_text"]) for r in rows]
        bm25 = BM25Okapi(tokenised_docs)

        _bm25_index = bm25
        _bm25_track_ids = track_ids
        print(f"BM25 index built over {len(track_ids)} documents.")
        return bm25, track_ids
    
"""Baseline 1: Random"""
def baseline_random(query: str, k: int, seed: int | None = None) -> list[int]:
    """Return k random track_ids from the corpus.Seed can be set for reproducibility across runs, 
    though evaluation typically averages across queries rather than across random seeds. """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT track_id FROM track_documents"
        ).fetchall()
    all_ids = [r["track_id"] for r in rows]

    rng = random.Random(seed)
    if len(all_ids) <= k:
        return all_ids
    return rng.sample(all_ids, k)

"""Baseline 2: Popularity"""
def baseline_popularity(query: str, k: int) -> list[int]:
    """Return k tracks with the highest listener counts."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT t.id 
            FROM tracks t
            JOIN track_documents td on t.id = td.track_id
            WHERE t.listener_count IS NOT NULL
            ORDER BY t.listener_count DESC
            LIMIT ?
            """,
            (k,),
        ).fetchall()
    return [r["id"] for r in rows]

"""Baseline 3: BM25"""
def baseline_bm25(query: str, k: int) -> list[int]:
    """Return k tracks with the highest BM25 scores against tokanised query."""
    bm25, track_ids = _load_bm25_index()
    tokens = _simple_tokenize(query)
    if not tokens:
        return []
    
    scores = bm25.get_scores(tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [track_ids[i] for i in ranked_indices]

"""Baseline 4: Dense-only"""
def baseline_dense(query: str, k: int) -> list[int]:
    """Return k tracks with the highest FAISS similarity scores to the query."""
    hits = retrieve(query, k=k)
    return [h.track_id for h in hits]

"""Baseline 5: RAG"""
def baseline_rag(query: str, k: int, retrieval_breadth: int | None = None) -> list[int]:
    """Return k tracks using the full RAG pipeline.
    
    Args: 
    - query
    - k: number of results to return
    - retrieval_breadth: how many candidates to pull for LLM to consider (defaults to 2*k)
    
    If the LLM call fails or all recommendations are hallucinated it falls back to dense-only ordering """
    if retrieval_breadth is None:
        retrieval_breadth = 2* k 
    hits = retrieve(query, k = retrieval_breadth)
    if not hits:
        return []
    valid_ids = {h.track_id for h in hits}
    user_prompt = build_user_prompt(query, hits, n=k)
    try:
        raw = _call_claude(user_prompt)
    except Exception as e:
        print(f"LLM call failed: {e}")
        return [h.track_id for h in hits[:k]] #fall back to dense-only
    
    parsed = _parse_response(raw)
    if parsed is None:
        print(f"LLM response parsing failed: {raw}")
        return [h.track_id for h in hits[:k]] #another fall back to dense-only
    
    filtered = _filter_hallucinated(parsed, valid_ids)
    if not filtered.recommendations:
        print(f"LLM returned only hallucinated recommendations: {raw}")
        return [h.track_id for h in hits[:k]] #final fall back to dense-only
    
    return [rec.track_id for rec in filtered.recommendations[:k]]

BASELINES = {
    "random": baseline_random,
    "popularity": baseline_popularity,
    "bm25": baseline_bm25,
    "dense": baseline_dense,
    "rag": baseline_rag,
}

def run_baseline(name: str, query: str, k:int) -> list[int]:
    """Dispatch to the named baseline, if name unknown it raises KeyError"""
    if name not in BASELINES:
        raise KeyError(f"Unknown baseline {name}. Valid options: {list(BASELINES.keys())}")
    return BASELINES[name](query, k)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural-language query to retrieve for")
    parser.add_argument(
        "-k", type=int, default=10, help="Number of results (default 10)"
    )
    parser.add_argument(
        "--baseline",
        choices=list(BASELINES.keys()) + ["all"],
        default = "all",
        help = "Which baselines to run",
    )
    args = parser.parse_args()
    names = list(BASELINES.keys()) if args.baseline == "all" else [args.baseline]

    with get_conn() as conn:
        for name in names:
            print(f"Baseline: {name}  (query: {args.query!r})")
            print("-" * 70)
            track_ids = run_baseline(name, args.query, args.k)
            for i, tid in enumerate(track_ids, 1):
                row = conn.execute(
                    "SELECT name, artist_name FROM tracks WHERE id = ?", (tid,)
                ).fetchone()
                if row:
                    print(f"{i:2d}. {row['name']} by {row['artist_name']}")
                else:
                    print(f"{i:2d}. track_id={tid} (not found in tracks table)")
