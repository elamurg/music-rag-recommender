"""LLM invocation with retry, response validation and callback logic"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pydantic import ValidationError

from src.generation.prompt import (
    RecommendationResponse,
    Recommendation,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from src.retrieval.query import TrackHit

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL = "claude-sonnet-4-5"  
MAX_TOKENS = 2000  
DEFAULT_N = 5   

MAX_RETRIES = 2
INITIAL_BACKOFF_SEC = 3

_client: anthropic.Anthropic | None = None


@dataclass
class GenerationResponse:
    """The final response returned to the caller."""
    recommendations: list[Recommendation]
    grounded: bool
    raw_llm_output: str | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not found in .env")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _call_claude(user_prompt: str) -> str:
    """Call Claude with retry on failures. Returns raw text response."""
    client = _get_client()
    backoff = INITIAL_BACKOFF_SEC

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as e:
            if attempt >= MAX_RETRIES:
                raise
            print(f"transient error (attempt {attempt + 1}): {e}, retrying in {backoff}s")
            time.sleep(backoff)
            backoff *= 3


def _parse_response(raw: str) -> RecommendationResponse | None:
    """Parse and validate the LLM's JSON response. This also handles common malformations (trailing whitespaces, unexpected keys...). 
    Returns None if not salvaged"""
    text = raw.strip()

    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"JSON parse failed: {e}")
        return None

    try:
        return RecommendationResponse.model_validate(data)
    except ValidationError as e:
        print(f"schema validation failed: {e}")
        return None


def _filter_hallucinated(
    response: RecommendationResponse, valid_track_ids: set[int]) -> RecommendationResponse:
    """Drop any recommendation whose track_id isn't in the retrieved candidates. This helps with handling hallucinations.
    If we filter them early they won't reach the end user"""
    filtered = [
        rec for rec in response.recommendations
        if rec.track_id in valid_track_ids
    ]
    return RecommendationResponse(recommendations=filtered)

def _fallback_from_hits(hits: list[TrackHit], n: int) -> list[Recommendation]:
    """Fallback ranking: use top-N by FAISS score with a generic justification."""
    return [
        Recommendation(
            track_id=hit.track_id,
            justification=(
                f"Retrieved by dense similarity to your query "
                f"(similarity score: {hit.score:.3f}). "
                f"LLM re-ranking failed; returning top FAISS matches directly."
            )
        )
        for hit in hits[:n]
    ]

def generate_recommendations(query: str, hits: list[TrackHit], n: int = DEFAULT_N) -> GenerationResponse:
    """Generate a ranked, grounded set of recommendations from retrieval hits.
    Args as inputs are:
    - query: the user's original query text
    - hits: retrieval results from src.retrieval.query.retrieve()
    - n: how many recommendations to return (automatically set to 5)
    Returns are:
    - GenerationResponse with either LLM-ranked recommendations (grounded=True) or a FAISS-only fallback (grounded=False)."""

    if not hits:
        return GenerationResponse(recommendations=[], grounded=False)

    valid_ids = {h.track_id for h in hits}

    user_prompt = build_user_prompt(query, hits, n=n)

    try:
        raw = _call_claude(user_prompt)
    except Exception as e:
        print(f"LLM call failed: {e}")
        return GenerationResponse(recommendations=_fallback_from_hits(hits, n),grounded=False)

    parsed = _parse_response(raw)
    if parsed is None:
        return GenerationResponse(recommendations=_fallback_from_hits(hits, n), grounded=False, raw_llm_output=raw)

    filtered = _filter_hallucinated(parsed, valid_ids)

    if not filtered.recommendations:
        #fall back if only hallucionated results were returned
        return GenerationResponse(recommendations=_fallback_from_hits(hits, n), grounded=False, raw_llm_output=raw)

    return GenerationResponse(
        recommendations=filtered.recommendations[:n],
        grounded=True,
        raw_llm_output=raw,
    )

if __name__ == "__main__":
    import argparse

    from src.retrieval.query import retrieve
    from src.corpus.db import get_conn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural-language query")
    parser.add_argument("-k", type=int, default=20, help="Retrieval breadth (default 20)")
    parser.add_argument("-n", type=int, default=5, help="Recommendation count (default 5)")
    parser.add_argument("--show-raw", action="store_true", help="Print raw LLM output")
    args = parser.parse_args()

    print(f"\nQuery: {args.query!r}")
    print(f"Retrieving top-{args.k}...")
    hits = retrieve(args.query, k=args.k)
    print(f"Retrieved {len(hits)} candidates")

    print(f"\nGenerating {args.n} recommendations...")
    response = generate_recommendations(args.query, hits, n=args.n)

    grounding = "grounded by LLM" if response.grounded else "FAISS fallback (LLM failed)"
    print(f"\nRecommendations ({grounding}):")
    print("=" * 70)

    with get_conn() as conn:
        for i, rec in enumerate(response.recommendations, 1):
            row = conn.execute(
                "SELECT name, artist_name FROM tracks WHERE id = ?",
                (rec.track_id,),
            ).fetchone()
            title = f"{row['name']} by {row['artist_name']}" if row else f"[unknown track {rec.track_id}]"
            print(f"\n{i}. {title}")
            print(f"   {rec.justification}")

    if args.show_raw and response.raw_llm_output:
        print("\n" + "=" * 70)
        print("Raw LLM output:")
        print(response.raw_llm_output)