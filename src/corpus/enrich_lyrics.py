"""fetch lyrics for each enriched track from Genius."""
import time
from pathlib import Path

from tqdm import tqdm

from .db import get_conn
from .genius_client import fetch_lyrics, get_genius_client

#1 request per second is stable
POLITE_DELAY_SEC = 1.0


def fetch_unprocessed_tracks() -> list[dict]:
    """Return enriched tracks that don't yet have a lyrics row."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT t.id, t.name, t.artist_name
            FROM tracks t
            LEFT JOIN lyrics l ON l.track_id = t.id
            WHERE t.enriched_at IS NOT NULL
              AND l.track_id IS NULL
            ORDER BY t.id
            """
        ).fetchall()
    return [dict(r) for r in rows]


def write_lyrics_result(conn, track_id: int, lyrics_data: dict | None) -> None:
    """Write result to the lyrics table."""
    if lyrics_data is None:
        conn.execute(
            """
            INSERT OR REPLACE INTO lyrics
                (track_id, lyrics_text, genius_url, genius_id, retrieved_at)
            VALUES (?, NULL, NULL, NULL, CURRENT_TIMESTAMP)
            """,
            (track_id,),
        )
    else:
        conn.execute(
            """
            INSERT OR REPLACE INTO lyrics
                (track_id, lyrics_text, genius_url, genius_id, retrieved_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                track_id,
                lyrics_data["lyrics_text"],
                lyrics_data["genius_url"],
                lyrics_data["genius_id"],
            ),
        )


def process_one_track(client, track_row: dict) -> tuple[bool, bool, str | None]:
    """fetch and store lyrics for one track"""
    track_id = track_row["id"]
    name = track_row["name"]
    artist = track_row["artist_name"]

    try:
        result = fetch_lyrics(client, name, artist)
    except Exception as e:
        return False, False, f"fetch failed: {e!r}"

    with get_conn() as conn:
        write_lyrics_result(conn, track_id, result)

    return True, result is not None, None


def report_progress() -> None:
    """Summarise lyrics ingestion progress."""
    with get_conn() as conn:
        stats = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM tracks WHERE enriched_at IS NOT NULL) AS enriched_tracks,
                COUNT(*) AS processed,
                SUM(CASE WHEN lyrics_text IS NOT NULL AND lyrics_text != '' THEN 1 ELSE 0 END) AS with_lyrics,
                ROUND(AVG(LENGTH(lyrics_text)), 0) AS avg_lyric_length
            FROM lyrics
            """
        ).fetchone()

    enriched = stats["enriched_tracks"] or 0
    processed = stats["processed"] or 0
    with_lyrics = stats["with_lyrics"] or 0
    avg_len = stats["avg_lyric_length"] or 0
    hit_rate = 100 * with_lyrics / processed if processed else 0

    print(f"\nLyrics ingestion status:")
    print(f"Enriched tracks: {enriched}")
    print(f"Processed:{processed}")
    print(f"With lyrics {with_lyrics} ({hit_rate:.1f}% hit rate)")
    print(f"Average lyric length:{avg_len:.0f} chars")


def main():
    client = get_genius_client()
    unprocessed = fetch_unprocessed_tracks()

    if not unprocessed:
        print("All tracks already processed. Nothing to do.")
        report_progress()
        return

    print(f"Processing {len(unprocessed)} tracks (1 API call each ~= "
          f"{len(unprocessed) * POLITE_DELAY_SEC / 60:.0f} min)")

    hits = 0
    misses = 0
    hard_failures = 0
    with tqdm(unprocessed, desc="lyrics") as pbar:
        for track_row in pbar:
            processed, found, err = process_one_track(client, track_row)
            if not processed:
                hard_failures += 1
                pbar.write(f"[fail] {track_row['artist_name']} - {track_row['name']}: {err}")
            elif found:
                hits += 1
            else:
                misses += 1
            time.sleep(POLITE_DELAY_SEC)
            pbar.set_postfix(hit=hits, miss=misses, fail=hard_failures)

    print(f"\nFinished. Hits: {hits}, misses: {misses}, hard failures: {hard_failures}.")
    report_progress()


if __name__ == "__main__":
    main()

    