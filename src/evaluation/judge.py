"""LLM-as-judge evaluation on a query subset. Judgments are cached in the judge_cache table so repeat evaluation runs
don't repay for the same judgment. Uses Claude Haiku as the cheapest model for the job needed."""
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from src.corpus.db import get_conn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

JUDGE_MODEL = "claude-haiku-4-5-20251001"  
BATCH_SIZE = 10
MAX_TOKENS = 2000
MAX_RETRIES = 2
INITIAL_BACKOFF_SEC = 3

_client: anthropic.Anthropic | None = None


@dataclass
class Judgment:
    track_id: int
    score: int  
    reasoning: str


JUDGE_SYSTEM_PROMPT = """\
You are a music recommendation evaluator. Your job is to rate how well each recommended track fits a user's natural-language music query.

Rate each track on a 3-point scale:
- 2 = HIGHLY RELEVANT. The track strongly matches the query on genre, mood, and context.
- 1 = PARTIALLY RELEVANT. The track matches some elements of the query but not all, or is a plausible but not obvious fit.
- 0 = NOT RELEVANT. The track does not match the query's intent, or matches only on superficial tokens.

Judge based on the track's actual musical content as described in the passage, not just the track name. A track called "Sad Song" that is actually upbeat should score low on a "sad music" query.

Base your judgments only on facts stated in the retrieved passage. Do not invent facts about tracks.

Respond ONLY with valid JSON matching the schema. No preamble."""


BATCH_JUDGE_PROMPT_TEMPLATE = """\
Query: {query}

Rate each of the following tracks (0-2 scale) on how well they fit the query. Return a JSON array with one object per track_id, in the same order they appear below.

Schema: [{{"track_id": <int>, "score": <0|1|2>, "reasoning": "<one sentence>"}}]

Tracks to judge:

{candidates_block}

Return ONLY the JSON array, no other text."""


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not found in .env")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _fetch_cached_judgments(
    query_id: str, track_ids: list[int]
) -> dict[int, Judgment]:
    """Return cached judgments for the (query_id, track_id) pairs"""
    if not track_ids:
        return {}

    placeholders = ",".join("?" for _ in track_ids)
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT track_id, score, reasoning
            FROM judge_cache
            WHERE query_id = ?
              AND model = ?
              AND track_id IN ({placeholders})
            """,
            [query_id, JUDGE_MODEL] + track_ids,
        ).fetchall()

    return {
        r["track_id"]: Judgment(
            track_id=r["track_id"],
            score=r["score"],
            reasoning=r["reasoning"] or "",
        )
        for r in rows
    }


def _cache_judgments(query_id: str, judgments: list[Judgment]) -> None:
    """persist judgments to the cache"""
    if not judgments:
        return

    with get_conn() as conn:
        for j in judgments:
            conn.execute(
                """
                INSERT OR REPLACE INTO judge_cache
                    (query_id, track_id, score, reasoning, model, judged_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (query_id, j.track_id, j.score, j.reasoning, JUDGE_MODEL),
            )


def _fetch_track_documents(track_ids: list[int]) -> dict[int, str]:
    """Fetch document text for the given track_ids."""
    if not track_ids:
        return {}

    placeholders = ",".join("?" for _ in track_ids)
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT track_id, document_text
            FROM track_documents
            WHERE track_id IN ({placeholders})
            """,
            track_ids,
        ).fetchall()

    return {r["track_id"]: r["document_text"] for r in rows}


def _call_judge(user_prompt: str) -> str:
    """call Claude judge with retry on transient failures."""
    client = _get_client()
    backoff = INITIAL_BACKOFF_SEC

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=MAX_TOKENS,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as e:
            if attempt >= MAX_RETRIES:
                raise
            print(f"[judge] transient error (attempt {attempt + 1}): {e}, retrying")
            time.sleep(backoff)
            backoff *= 3


def _parse_judge_response(raw: str) -> list[dict] | None:
    """Parse the judge's JSON array response"""
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
        print(f"[judge] JSON parse failed: {e}")
        return None

    if not isinstance(data, list):
        print(f"[judge] expected JSON array, got {type(data).__name__}")
        return None
    return data


def _judge_batch(
    query: str, batch_candidates: list[tuple[int, str]]
) -> list[Judgment]:
    """Judge a single batch of tracks."""
    candidates_block = "\n\n".join(
        f"-- track_id {tid} --\n{doc}"
        for tid, doc in batch_candidates
    )
    user_prompt = BATCH_JUDGE_PROMPT_TEMPLATE.format(
        query=query,
        candidates_block=candidates_block,
    )

    try:
        raw = _call_judge(user_prompt)
    except Exception as e:
        print(f"[judge] call failed for batch: {e}, marking all as score 0")
        return [
            Judgment(track_id=tid, score=0, reasoning=f"judge call failed: {e}")
            for tid, _ in batch_candidates
        ]

    parsed = _parse_judge_response(raw)
    if parsed is None:
        print(f"[judge] unparseable response, marking all as score 0")
        return [
            Judgment(track_id=tid, score=0, reasoning="judge response unparseable")
            for tid, _ in batch_candidates
        ]

    #index parsed by track_id
    parsed_by_id = {}
    for item in parsed:
        try:
            tid = int(item["track_id"])
            score = int(item["score"])
            if score not in (0, 1, 2):
                score = 0
            reasoning = str(item.get("reasoning", ""))[:500]
            parsed_by_id[tid] = Judgment(
                track_id=tid, score=score, reasoning=reasoning
            )
        except (KeyError, ValueError, TypeError):
            continue

    #emit judgments in input order, to miss the ones that score 0
    results = []
    for tid, _ in batch_candidates:
        if tid in parsed_by_id:
            results.append(parsed_by_id[tid])
        else:
            results.append(
                Judgment(track_id=tid, score=0, reasoning="missing from judge response")
            )
    return results


def judge_recommendations(
    query_id: str, query_text: str, track_ids: list[int]
) -> dict[int, float]:
    """Judge each recommended track. Returns dict mapping track_id -> score (as float).

    Uses cache aggressively. Batches uncached tracks 10 at a time.

    Args:
        query_id: the eval query id (used as cache key)
        query_text: the natural-language query
        track_ids: recommended track IDs to judge

    Returns dict mapping track_id -> relevance score in {0.0, 1.0, 2.0}.
    """
    if not track_ids:
        return {}

    cached = _fetch_cached_judgments(query_id, track_ids)
    uncached_ids = [tid for tid in track_ids if tid not in cached]

    all_judgments: dict[int, Judgment] = dict(cached)

    if uncached_ids:
        docs = _fetch_track_documents(uncached_ids)
        candidates = [(tid, docs.get(tid, "")) for tid in uncached_ids]
        candidates = [(tid, doc) for tid, doc in candidates if doc]

        #batch and call judge
        new_judgments = []
        for i in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[i:i + BATCH_SIZE]
            batch_judgments = _judge_batch(query_text, batch)
            new_judgments.extend(batch_judgments)

        #persist and merge
        _cache_judgments(query_id, new_judgments)
        for j in new_judgments:
            all_judgments[j.track_id] = j

    return {tid: float(j.score) for tid, j in all_judgments.items()}

if __name__ == "__main__":
    import argparse

    from src.evaluation.query import load_queries

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-id", required=True, help="e.g. C01")
    parser.add_argument(
        "--track-ids",
        required=True,
        help="Comma-separated track_ids to judge, e.g. 1234,5678,9012",
    )
    args = parser.parse_args()

    all_queries = {q.id: q for q in load_queries()}
    if args.query_id not in all_queries:
        print(f"Unknown query_id: {args.query_id}")
        exit(1)

    query = all_queries[args.query_id]
    track_ids = [int(x.strip()) for x in args.track_ids.split(",")]

    print(f"\nQuery: {query.query!r}")
    print(f"Judging {len(track_ids)} tracks using {JUDGE_MODEL}...\n")

    scores = judge_recommendations(query.id, query.query, track_ids)

    with get_conn() as conn:
        for tid in track_ids:
            row = conn.execute(
                "SELECT name, artist_name FROM tracks WHERE id = ?", (tid,)
            ).fetchone()
            score = scores.get(tid, 0.0)
            title = f"{row['name']} by {row['artist_name']}" if row else f"[{tid}]"
            marker = {2: "HIGH", 1: "PART", 0: "NONE"}.get(int(score), "?")
            print(f"  [{marker}] score={score:.0f}  {title}")