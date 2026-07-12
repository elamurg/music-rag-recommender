"""ENrich each seed track with tags, similar tracks and metadata."""
import os
import time
from pathlib import Path

import pylast
from dotenv import load_dotenv
from tqdm import tqdm

from .db import get_conn
from .lastfm_client import (
    SIMILAR_TRACKS_LIMIT,
    fetch_similar_tracks,
    fetch_track_info,
    parse_enrichment
)
from .schemas import TrackEnrichment

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT/".env")

LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")
LASTFM_API_SECRET = os.getenv("LASTFM_API_SECRET")

if not LASTFM_API_KEY:
    raise EnvironmentError("API Key not found in .env.")

POLITE_DELAY_SEC = 0.25

def get_lastfm_client() -> pylast.LastFMNetwork:
    return pylast.LastFMNetwork(
        api_key=LASTFM_API_KEY,
        api_secret=LASTFM_API_SECRET or "",
    )

def fetch_unenriched_tracks() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, name, artist_name
            FROM tracks 
            WHERE enriched_at IS NULL
            ORDER BY id
            """
        ).fetchall()
    return [dict(r) for r in rows]

def upsert_tag(conn, tag_name: str) -> int:
    """Insert tag if new. It reterns the id."""
    conn.execute(
        "INSERT OR IGNORE INTO tags (name) VALUES (?)",
        (tag_name,),
    )
    row = conn.execute(
        "SELECT id FROM tags WHERE name = ?", (tag_name,)
    ).fetchone()
    return row["id"]

def write_enrichment(conn, track_id: int, enrichment: TrackEnrichment) -> None:
    """ Enrichment data for tracks, track_tags and similar_tracks."""
    conn.execute(
        """
        UPDATE tracks
        SET mbid = COALESCE(?, mbid),
            listener_count = ?,
            playcount = ?,
            wiki_summary = ?,
            wiki_content = ?,
            album_name = ?,
            lastfm_url = ?,
            enriched_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            enrichment.mbid,
            enrichment.listener_count,
            enrichment.playcount,
            enrichment.wiki_summary,
            enrichment.wiki_content,
            enrichment.album_name,
            enrichment.lastfm_url,
            track_id,
        ),
    )

    for tag_info in enrichment.tags:
        tag_id = upsert_tag(conn, tag_info.name)
        conn.execute(
            """
            INSERT OR REPLACE INTO track_tags (track_id, tag_id, weight)
            VALUES (?,?,?)
            """,
            (track_id, tag_id, tag_info.weight),
        )

    for similar in enrichment.similar:
        conn.execute(
            """
            INSERT OR REPLACE INTO similar_tracks
                (source_track_id, target_name, target_artist, similarity)
            VALUES (?,?,?,?)
            """,
            (track_id, similar.target_name, similar.target_artist, similar.similarity),
        )

def enrich_one_track(lastfm: pylast.LastFMNetwork, track_row: dict) -> tuple[bool, str| None]:
    """Enrich a single track and return success or error. Both fetch calls must succeed for it to work."""
    track_id = track_row["id"]
    name = track_row["name"]
    artist = track_row["artist_name"]
    
    lastfm_track = lastfm.get_track(artist, name)

    try:
        info_raw = fetch_track_info(lastfm_track)
    except pylast.WSError as e:
        return False, f"info failed (permanent): {e.details}"
    except Exception as e:
        return False, f"info failed (transient after retries): {e!r}"
    
    try: 
        similar_raw = fetch_similar_tracks(lastfm_track, limit= SIMILAR_TRACKS_LIMIT)
    except pylast.WSError as e:
        return False, f"similar failed (permanent): {e.details}"
    except Exception as e:
        return False, f"similar failed (transient after retries): {e!r}"
    
    enrichment = parse_enrichment(info_raw, similar_raw)
    with get_conn() as conn:
        write_enrichment(conn, track_id, enrichment)
    return True, None

def report_progress() -> None:
    "Summarise the enrichment progress."
    with get_conn() as conn:
        stats = conn.execute(
            """
            SELECT 
              COUNT(*) AS total,
              SUM(CASE WHEN enriched_at IS NOT NULL THEN 1 ELSE 0 END) AS enriched,
              SUM(CASE WHEN mbid IS NOT NULL AND mbid != '' THEN 1 ELSE 0 END) AS with_mbid
            FROM tracks
            """
        ).fetchone()
        tag_count = conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
        track_tag_count = conn.execute("SELECT COUNT(*) FROM track_tags").fetchone()[0]
        similar_count = conn.execute("SELECT COUNT(*) FROM similar_tracks").fetchone()[0] 

    total = stats["total"]
    enriched = stats["enriched"] or 0
    with_mbid = stats['with_mbid'] or 0
    pct = 100*enriched / total if total else 0

    print(f"\nCorpus status:")
    print(f"Total tracks: {total}")
    print(f"Tracks enriched: {enriched}({pct:.1f}%)")
    print(f"Tracks with MBID: {with_mbid} ({100* with_mbid/total if total else 0:.1f}%)")
    print(f"Unique tags: {tag_count}")
    print(f"Track-tag endges: {track_tag_count}")
    print(f"Similar-track edges: {similar_count}")

        
def main():
    lastfm = get_lastfm_client()
    unenriched = fetch_unenriched_tracks()
    if not unenriched:
        print("All tracks already enriched. Nothing to do.")
        report_progress()
        return
    print(f"Enriching {len(unenriched)} tracks (2 API calls each = "
         f"{2* len(unenriched)* POLITE_DELAY_SEC/60:.0f} min)")
    
    sucesses = 0
    failures = 0
    with tqdm(unenriched, desc= "enriching") as pbar:
        for track_row in pbar:
            ok, err = enrich_one_track(lastfm, track_row)
            if ok:
                sucesses += 1
            else:
                failures += 1
                pbar.write(f"[skip] {track_row['artist_name']} - {track_row['name']}: {err}")
            time.sleep(POLITE_DELAY_SEC)
            pbar.set_postfix(ok = sucesses, skip = failures)
    
    print(f"\nFinished. Enriched {sucesses} tracks, skipped {failures}")
    report_progress()

if __name__ == '__main__':
    main()
    