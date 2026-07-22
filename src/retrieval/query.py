"""Contains the main module for retrieval."""

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.corpus.db import get_conn

#change model name if youre gonna change embeddings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_PATH = PROJECT_ROOT / "data" / "embeddings" / "corpus.faiss"

_model: SentenceTransformer | None = None
_index: faiss.Index | None = None

@dataclass
class TrackHit:
    """One retrieval result with metadata hydrated from the corpus."""
    track_id: int
    score: float
    name: str
    artist_name: str
    album_name: str | None
    document_text: str


def _load_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer model."""
    global _model
    if _model is None:
        print(f"[retrieval] loading model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        actual_dim = _model.get_embedding_dimension()
        assert actual_dim == EMBEDDING_DIM, (
            f"Model produced {actual_dim}-dim vectors; expected {EMBEDDING_DIM}. "
            f"Did the index and this module get out of sync?"
        )
    return _model


def _load_index() -> faiss.Index:
    """Load and cache the FAISS index from disk."""
    global _index
    if _index is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_PATH}. "
                f"Run Phase 6 first: `python -m src.embedding.embed`"
            )
        print(f"[retrieval] loading index: {INDEX_PATH.name}")
        _index = faiss.read_index(str(INDEX_PATH))
    return _index


def _embed_query(query: str) -> np.ndarray:
    """encoding a query string to a unit-normalised vector for FAISS lookup """
    model = _load_model()
    vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    return vec


def _hydrate_hits(
    conn, track_ids: list[int], scores: list[float]
) -> list[TrackHit]:
    """Fetch track metadata + document summary text for a set of retrieved IDs."""
    if not track_ids:
        return []

    placeholders = ",".join("?" for _ in track_ids)
    rows = conn.execute(
        f"""
        SELECT t.id, t.name, t.artist_name, t.album_name, td.document_text
        FROM tracks t
        JOIN track_documents td ON td.track_id = t.id
        WHERE t.id IN ({placeholders})
        """,
        track_ids,
    ).fetchall()

    # index rows by track_id so it rebuilds in FAISS order
    row_by_id = {r["id"]: r for r in rows}

    hits = []
    for track_id, score in zip(track_ids, scores):
        row = row_by_id.get(track_id)
        if row is None:
            continue
        hits.append(
            TrackHit(
                track_id=int(track_id),
                score=float(score),
                name=row["name"],
                artist_name=row["artist_name"],
                album_name=row["album_name"],
                document_text=row["document_text"],
            )
        )
    return hits


def retrieve(query: str, k: int = 20) -> list[TrackHit]:
    """Return the top-k most similar tracks to the query. Maybe add args on here for better explaination"""
    if not query or not query.strip():
        return []

    index = _load_index()
    query_vec = _embed_query(query)

    scores, ids = index.search(query_vec, k=k)

    # FAISS returns 2D arrays (batching muultiple queries)
    track_ids = [int(i) for i in ids[0] if i != -1]
    result_scores = [float(s) for s in scores[0][: len(track_ids)]]

    with get_conn() as conn:
        return _hydrate_hits(conn, track_ids, result_scores)


def format_hit(hit: TrackHit) -> str:
    """Format a hit for human-readable display (debugging, logging)."""
    return f"[{hit.score:.3f}] {hit.name} by {hit.artist_name}"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural-language query to retrieve for")
    parser.add_argument(
        "-k", type=int, default=10, help="Number of results (default 10)"
    )
    parser.add_argument(
        "--show-doc",
        action="store_true",
        help="Print each hit's document text (verbose)",
    )
    args = parser.parse_args()

    hits = retrieve(args.query, k=args.k)

    print(f"\nQuery: {args.query!r}")
    print(f"Retrieved {len(hits)} hits\n")
    for hit in hits:
        print(format_hit(hit))
        if args.show_doc:
            print("  " + "-" * 60)
            for line in hit.document_text.splitlines():
                print(f"  {line}")
            print()