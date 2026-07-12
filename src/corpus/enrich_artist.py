"""Enrich each distinct artist in the corpus."""
import os
import time
from pathlib import Path

import pylast
from dotenv import load_dotenv
from tqdm import tqdm

from .db import get_conn
from .lastfm_client import fetch_artist_info, parse_artist_enrichment
from .schemas import ArtistEnrichment

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")
LASTFM_API_SECRET = os.getenv("LASTFM_API_SECRET")

if not LASTFM_API_KEY:
    raise EnvironmentError("LASTFM_API_KEY not found in .env")

POLITE_DELAY_SEC = 0.25


def get_lastfm_client() -> pylast.LastFMNetwork:
    return pylast.LastFMNetwork(
        api_key=LASTFM_API_KEY,
        api_secret=LASTFM_API_SECRET or "",
    )


def fetch_unenriched_artist_names() -> list[str]:
    """Return distinct artist names from tracks that aren't yet in artists,
    or that exist in artists but have enriched_at IS NULL."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT t.artist_name
            FROM tracks t
            LEFT JOIN artists a ON a.name = t.artist_name
            WHERE a.name IS NULL OR a.enriched_at IS NULL
            ORDER BY t.artist_name
            """
        ).fetchall()
    return [r["artist_name"] for r in rows]


def upsert_tag(conn, tag_name: str) -> int:
    """Insert tag if new. Returns the id. Same helper as in enrich.py — could
    be pulled into a shared module later.
    """
    conn.execute(
        "INSERT OR IGNORE INTO tags (name) VALUES (?)",
        (tag_name,),
    )
    row = conn.execute(
        "SELECT id FROM tags WHERE name = ?", (tag_name,)
    ).fetchone()
    return row["id"]


def upsert_artist(conn, artist_name: str) -> int:
    """Insert artist name (without enrichment fields) if not present.
    Returns the artist's id."""
    conn.execute(
        "INSERT OR IGNORE INTO artists (name) VALUES (?)",
        (artist_name,),
    )
    row = conn.execute(
        "SELECT id FROM artists WHERE name = ?", (artist_name,)
    ).fetchone()
    return row["id"]


def write_artist_enrichment(conn, artist_name: str, enrichment: ArtistEnrichment) -> None:
    """Write parsed artist enrichment to artists and artist_tags."""
    artist_id = upsert_artist(conn, artist_name)

    conn.execute(
        """
        UPDATE artists
        SET mbid = COALESCE(?, mbid),
            listener_count = ?,
            playcount = ?,
            bio_summary = ?,
            bio_content = ?,
            enriched_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            enrichment.mbid,
            enrichment.listener_count,
            enrichment.playcount,
            enrichment.bio_summary,
            enrichment.bio_content,
            artist_id,
        ),
    )

    for tag_info in enrichment.tags:
        tag_id = upsert_tag(conn, tag_info.name)
        conn.execute(
            """
            INSERT OR REPLACE INTO artist_tags (artist_id, tag_id, weight)
            VALUES (?, ?, ?)
            """,
            (artist_id, tag_id, tag_info.weight),
        )


def enrich_one_artist(lastfm: pylast.LastFMNetwork, artist_name: str) -> tuple[bool, str | None]:
    """Enrich a single artist. Returns (success, error_message)."""
    lastfm_artist = lastfm.get_artist(artist_name)

    try:
        info_raw = fetch_artist_info(lastfm_artist)
    except pylast.WSError as e:
        return False, f"info failed (permanent): {e.details}"
    except Exception as e:
        return False, f"info failed (transient after retries): {e!r}"

    enrichment = parse_artist_enrichment(info_raw)

    with get_conn() as conn:
        write_artist_enrichment(conn, artist_name, enrichment)

    return True, None


def report_progress() -> None:
    """Summarise the artist enrichment progress."""
    with get_conn() as conn:
        stats = conn.execute(
            """
            SELECT
                (SELECT COUNT(DISTINCT artist_name) FROM tracks) AS total_distinct,
                COUNT(*) AS total_in_artists,
                SUM(CASE WHEN enriched_at IS NOT NULL THEN 1 ELSE 0 END) AS enriched,
                SUM(CASE WHEN mbid IS NOT NULL AND mbid != '' THEN 1 ELSE 0 END) AS with_mbid,
                SUM(CASE WHEN bio_content IS NOT NULL AND bio_content != '' THEN 1 ELSE 0 END) AS with_bio
            FROM artists
            """
        ).fetchone()
        artist_tag_count = conn.execute(
            "SELECT COUNT(*) FROM artist_tags"
        ).fetchone()[0]

    total_distinct = stats["total_distinct"] or 0
    total_in_artists = stats["total_in_artists"] or 0
    enriched = stats["enriched"] or 0
    with_mbid = stats["with_mbid"] or 0
    with_bio = stats["with_bio"] or 0
    pct = 100 * enriched / total_distinct if total_distinct else 0

    print(f"\nArtist enrichment status:")
    print(f"  Distinct artists across tracks: {total_distinct}")
    print(f"  Rows in artists table:          {total_in_artists}")
    print(f"  Artists enriched:               {enriched} ({pct:.1f}%)")
    print(f"  With MBID:                      {with_mbid}")
    print(f"  With bio content:               {with_bio}")
    print(f"  Artist-tag edges:               {artist_tag_count}")


def main():
    lastfm = get_lastfm_client()
    unenriched = fetch_unenriched_artist_names()

    if not unenriched:
        print("All artists already enriched. Nothing to do.")
        report_progress()
        return

    print(f"Enriching {len(unenriched)} artists (1 API call each = "
          f"{len(unenriched) * POLITE_DELAY_SEC / 60:.0f} min)")

    successes = 0
    failures = 0
    with tqdm(unenriched, desc="artists") as pbar:
        for artist_name in pbar:
            ok, err = enrich_one_artist(lastfm, artist_name)
            if ok:
                successes += 1
            else:
                failures += 1
                pbar.write(f"[skip] {artist_name}: {err}")
            time.sleep(POLITE_DELAY_SEC)
            pbar.set_postfix(ok=successes, skip=failures)

    print(f"\nFinished. Enriched {successes} artists, skipped {failures}.")
    report_progress()


if __name__ == "__main__":
    main()