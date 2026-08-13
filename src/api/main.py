"""FastAPI with three endpoints:
POST /recommend - accepts JSON body with string query, returns ranked rec with justification and Spotify URL
GET /health - returns 200 OK with basic system status
GET /stats - returns corpus statistics"""

import os 
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    HealthResponse,
    RecommendRequest,
    RecommendResponse,
    RecommendationItem,
    StatsResponse,
)

from src.corpus.db import get_conn
from src.generation.generate import generate_recommendations
from src.retrieval.query import EMBEDDING_DIM, INDEX_PATH, retrieve

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT/".env")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """preloading and indexing at startup so the first request isn't slow"""
    print("Warming up retrieval model and FAISS index...")
    try:
        _ = retrieve("warmup", k=1)
        print("Warmup complete!")
    except Exception as e:
        print(f"Warmup failed: {e}")
    yield
    #so there is nothing to cleanup at shutdown

app = FastAPI(
    title="SoundRAG API",
    description=(
        "Cold-start music recommendation via retrieval-augmented generation."
        "Accepts natural-language queries and returns grounded, LLM-ranked recommendations with Spotify playback URLs."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

#api endpoints
@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    """Generated ranked recommendation for a natural-language query. 
    
    The pipeline follows: FAISS retrieval -> claude-sonnet-4.5 re-ranking and hallucionation guard
    -> spotify identity resolution -> reranked recommednations with justifications.
    If fails it returns top-N FAISS results with grounded = false."""
    hits = retrieve(payload.query, k = 20)
    if not hits:
        raise HTTPException(
            status_code=500,
            detail="retrieval returned no candidates. Corpus or index may not be ready.",
        )
    response = generate_recommendations(payload.query, hits, n = payload.n)
    hit_by_id = {h.track_id for h in hits}
    items = []
    for rec in response.recommendations:
        hit = hit_by_id.get(rec.track_id)
        if hit is None:
            continue
        items.append(
            RecommendationItem(
                track_id=rec.track_id,
                name=hit.name,
                artist_name=hit.artist_name,
                justification=rec.justification,
                spotify_url=rec.spotify_url,
                spotify_confidence=rec.spotify_confidence,
            )
        )
    return RecommendResponse(
        query=payload.query,
        recommendations=items,
        grounded=response.grounded,
    )

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Checking reachibility of corpus, index and llm."""
    corpus_ready= False
    index_ready= INDEX_PATH.exists()
    llm_ready= bool(os.getenv("ANTHROPIC_API_KEY"))

    try: 
        with get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM tracks").fetchone()
            corpus_ready = row["n"] > 0
    except Exception:
        corpus_ready = False
    
    all_ok = corpus_ready and index_ready and llm_ready
    return HealthResponse(
        status = "ok" if all_ok else "degraded",
        corpus_ready=corpus_ready,
    )

@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    """Returns corpus statistics"""
    try:
        with get_conn() as conn:
            total = conn.execute("SELECT COUNT (*) AS n FROM tracks").fetchone()["n"]
            with_lyrics = conn.execute(
                "SELECT COUNT(*) AS n FROM lyrics WHERE lyrics_text IS NOT NULL"
            ).fetchone()["n"]
            with_wiki = conn.execute(
                "SELECT COUNT(*) AS n FROM tracks WHERE wiki_content IS NOT NULL"
            ).fetchone()["n"]
            unique_tags = conn.execute("SELECT COUNT(*) AS n FROM tags").fetchone()["n"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Corpus query failed: {e}")
    
    index_count = 0
    try:
        from src.retrieval.query import _load_index
        index = _load_index()
        index_count = index.ntotal
    except Exception:
        index_count = 0
    
    return StatsResponse(
        total_tracks=total,
        tracks_with_lyrics=with_lyrics,
        tracks_with_wiki=with_wiki,
        unique_tags=unique_tags,
        index_vector_count=index_count,
        embedding_dim=EMBEDDING_DIM,
    )

@app.get("/")
def root():
    """landing page"""
    return {
        "service": "SoundRAG",
        "docs": "/docs",
        "endpoints": ["/recommend", "/health", "/stats"],
    }