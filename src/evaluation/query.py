"""in doc.evaluation.queries: Hand-constructed evaluation queries with gold standards. About 100 queries testing: mood, style, reference adn compound.
Load and validate evaluation qieries and compute ground truth relevant track sets"""

import json
from pathlib import Path
from dataclasses import dataclass, field

from src.corpus.db import get_conn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
QUERIES_PATH = PROJECT_ROOT / "docs" / "evaluation" / "queries.json"

@dataclass
class EvalQuery:
    """One evaluation query with its category and ground truth relevant track_ids."""
    id: str
    category: str
    query: str
    ground_truth_strategy: str
    ground_truth_tags: list[str] = field(default_factory=list)
    reference_artist: str | None = None

def load_queries() -> list[EvalQuery]:
    """ Load queries.json into a list of EvalQuery instances"""
    with open(QUERIES_PATH) as f:
        data = json.load(f)
        queries = []
        for q in data["queries"]:
            queries.append(
                EvalQuery(
                    id=q["id"],
                    category=q["category"],
                    query=q["query"],
                    ground_truth_strategy=q["ground_truth_strategy"],
                    ground_truth_tags=q.get("ground_truth_tags", []),
                    reference_artist=q.get("reference_artist"),
                )
            )
        return queries

def queries_by_category(category: str) -> list[EvalQuery]:
    """Retrun only queries in a given category"""
    return [q for q in load_queries() if q.category == category]

def get_relevant_tracks_by_tags(tags: list[str], conn)-> set[int]:
    """Return track_ids whose track_tags or artist_tags inlude any of the given tags"""
    if not tags:
        return set()
    #case-sensitive
    tag_placeholders = ",".join("?" for _ in tags)

    track_level = conn.execute(
        f"""
        SELECT DISTINCT tt.track_id
        FROM track_tags tt
        JOIN tags t ON tt.tag_id = t.id
        WHERE LOWER(t.name) IN ({tag_placeholders})
        """,
        [t.lower() for t in tags],
    ).fetchall()

    artist_level = conn.execute(
        f"""
        SELECT DISTINCT tr.id AS track_id
        FROM tracks tr
        JOIN artists a ON tr.artist_name = a.name
        JOIN artist_tags at ON at.artist_id = a.id
        JOIN tags t ON t.id = at.tag_id
        WHERE LOWER(t.name) IN ({tag_placeholders})
        """,
        [t.lower() for t in tags]
    ).fetchall()

    return {r["track_id"] for r in track_level} | {r["track_id"] for r in artist_level}

def get_relevant_tracks_by_similar_artists(reference_artist: str, conn) -> set[int]:
    """Return track_ids whose artist is similar to the reference_artist"""
    if not reference_artist:
        return set()
    ref_lower = reference_artist.lower()
    #artists that appera as targets in similar_tracks for reference_artist
    similar_from_graph = conn.execute(
        """
        SELECT DISTINCT s.target_artist
        FROM similar_tracks s
        JOIN tracks t ON t.id = s.source_track_id
        WHERE LOWER(t.artist_name) = ?
        """,
        (ref_lower,),
    ).fetchall()
    similar_artist_names = {r["target_artist"] for r in similar_from_graph}

    #artists sharing 2+ tags with reference_artist
    ref_tag_rows = conn.execute(
        """
        SELECT DISTINCT LOWER(t.name) as tag_name
        FROM artist_tags at
        JOIN tags t on t.id = at.tag_id
        JOIN artists a on a.id = at.artist_id
        WHERE LOWER(a.name) = ?
        """,
        (ref_lower,),
    ).fetchall()
    ref_tags = {r["tag_name"] for r in ref_tag_rows}

    if len(ref_tags) >= 2:
        #find other artists sharing 2+ tags
        placeholders = ",".join("?" for _ in ref_tags)
        tag_overlap_rows = conn.execute(
            f"""
            SELECT LOWER(a.name) AS name, COUNT(DISTINCT LOWER(t.name)) AS shared
            FROM artist_tags at
            JOIN artists a ON a.id = at.artist_id
            JOIN tags t ON t.id = at.tag_id
            WHERE LOWER(t.name) IN ({placeholders})
              AND LOWER(a.name) != ?
            GROUP BY LOWER(a.name)
            HAVING COUNT(DISTINCT LOWER(t.name)) >= 2
            """,
            list(ref_tags) + [ref_lower],
        ).fetchall()
        similar_artist_names |= {r["name"] for r in tag_overlap_rows}
    
    similar_artist_names.add(ref_lower)
    if not similar_artist_names:
        return set()
    
    artist_placeholders = ",".join("?" for _ in similar_artist_names)
    track_rows = conn.execute(
        f"""
        SELECT id FROM tracks
        WHERE LOWER(artist_name) in ({artist_placeholders})
        """,
        [name.lower() for name in similar_artist_names]
    ).fetchall()
    return {r["id"] for r in track_rows}

def get_relevant_tracks(eval_query: EvalQuery) -> set[int] | None:
    """Dispatch to the right resolver based on strategy.
    Returns a set of track_ids for tag_match/similar_artist strategy but none of the llm_judge queries"""

    if eval_query.ground_truth_strategy == "llm_judge":
        return None
    with get_conn() as conn:
        if eval_query.ground_truth_strategy == "tag_match":
            return get_relevant_tracks_by_tags(eval_query.ground_truth_tags, conn)
        elif eval_query.ground_truth_strategy == "similar_artists":
            return get_relevant_tracks_by_similar_artists(eval_query.reference_artist, conn)
        else:
            raise ValueError(f"Unknown ground_truth_strategy: {eval_query.ground_truth_strategy}")
        

#sanity checkkk
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--query-id",
        help="Inspect ground truth for a specific query id",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Show ground truth size for a random sample of N queries",
    )
    args = parser.parse_args()

    all_queries = load_queries()
    print(f"Loaded {len(all_queries)} queries from {QUERIES_PATH.name}")

    if args.query_id:
        matches = [q for q in all_queries if q.id == args.query_id]
        if not matches:
            print(f"No query with id = {args.query_id}")
        else:
            q = matches[0]
            print(f"\n{q.id} [{q.category}]: {q.query!r}")
            print(f"Strategy: {q.ground_truth_strategy}")
            if q.ground_truth_tags:
                print(f"Tags: {q.ground_truth_tags}")
            if q.reference_artist:
                print(f"Reference artist: {q.reference_artist}")
            
            relevant = get_relevant_tracks(q)
            if relevant is None:
                print("Ground truth: LLM-judged (no precomputed set)")
            else:
                print(f"Relevant tracks: {len(relevant)}")
                if len(relevant) < 30:
                    print(f"fewer than 30 relevant tracks (noisy metrics)")
            
    elif args.sample:
        import random
        sample = random.sample(all_queries, min(args.sample, len(all_queries)))
        print(f"\nGround truth sizes for {len(sample)} random queries:")
        print("-" * 70)
        for q in sample:
            relevant = get_relevant_tracks(q)
            if relevant is None:
                size_str = "[LLM-judged]"
            else:
                marker = " *" if len(relevant) < 30 else "  "
                size_str = f"{len(relevant):5d}{marker}"
            print(f"  {q.id} [{q.category}]  {size_str}  {q.query[:50]}")
        print("\n* = fewer than 30 relevant tracks (statistically thin)")
    else:
        # summary by category
        from collections import Counter
        cats = Counter(q.category for q in all_queries)
        print("\nQueries by category:")
        for cat, count in cats.items():
            print(f"  {cat}: {count}")
