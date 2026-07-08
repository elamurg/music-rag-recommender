import os
import time
from pathlib import Path

import pylast
from dotenv import load_dotenv
from tqdm import tqdm

from .db import init_db, get_conn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")
LASTFM_API_SECRET = os.getenv("LASTFM_API_SECRET")

if not LASTFM_API_KEY:
    raise EnvironmentError("API key not found in .env")

#the aim is to collect around 50 tags with 200 tracks each, a total of 10k tracks, which is 7.5k unique tracks after deduplication
CURATED_TAGS = [
    "rock", "pop", "hip hop", "electronic", "jazz", "clasical",
    "indie", "metal", "punk", "folk", "country", "blues", "reggae", 
    "soul", "r&b", "disco", "funk", "house", "techno", "indie rock", 
    "alternative", "shoegraze", "post-punk", "new wave", "drum and bass",
    "synthwave", "vaporwave", "lo-fi", "ambient", "dream pop", "trip-hop",
    "dubstep", "k-pop", "j-pop", "afrobeat", "latin", "reggaeton", "psychedelic",
    "grunge", "emo", "hardcore", "chill", "melancholy", "energetic", "romantic",
    "dark", "atmospheric", "dreamy", "80s", "90s", "2000s", "2010s", "workout",
    "study", "sleep"
]
TRACKS_PER_TAG = 200
CHART_LIMIT =1000
POLITE_DELAY_SEC = 0.25 #responds to 4 api calls per second

def get_lastfm_client() -> pylast.LastFMNetwork:
    return pylast.LastFMNetwork(
        api_key = LASTFM_API_KEY,
        api_secret = LASTFM_API_SECRET or "",
    )

def insert_seed(conn, name: str, artist: str, source: str) -> bool:
    cur = conn.execute(
        """INSERT OR IGNORE INTO tracks(name, artist_name, seed_source)
        values(?, ?, ?)""",
        (name.strip(), artist.strip(), source),
    )
    return cur.rowcount == 1

def collect_from_tags(lastfm: pylast.LastFMNetwork, tags: list[str], per_tag: int) -> None:
    """For each tag fetch top tracks and insert as seeds.
    Use a fresh db connection per tag so mid-tag crash leaves earlier tags intact. """
    for tag_name in tqdm(tags, desc= "tags"):
        try:
            tag = lastfm.get_tag(tag_name)
            top = tag.get_top_tracks(limit=per_tag)
        except pylast.WSError as e:
            print(f"Error fetching: {tag_name}: {e}")
        except Exception as e:
            print(f"Unexpected error fetching: {tag_name}: {e}")
            continue
        inserted = 0
        with get_conn() as conn:
            for item in top:
                track = item.item
                try:
                    if insert_seed(
                        conn, 
                        name=track.get_name(), 
                        artist=track.get_artist().get_name(), 
                        source=f"tag:{tag_name}",
                    ):
                        inserted += 1
                except Exception as e:
                    print(f"Insertion failed for {track!r}: {e}")
                    continue
        time.sleep(POLITE_DELAY_SEC)

def collect_from_chart(lastfm: pylast.LastFMNetwork, limit: int) -> None:
    """Fetch global tracks from Last.fm's chart."""
    try:
        top = lastfm.get_top_tracks(limit = limit)
    except pylast.WSError as e:
        print(f"Error fetching chart tracks: {e}")
        return
    with get_conn() as conn:
        for item in tqdm(top, desc="Chart"):
            track = item.item
            try:
                insert_seed(
                    conn,
                    name=track.get_name(),
                    artist=track.get_artist().get_name(),
                    source="chart",
                )
            except Exception as e:
                print(f"Insertion failed for {track!r}: {e}")
                continue

def report_totals() -> None:
    """Visualise the total number of unique tracks in the corpus."""
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
        by_source = conn.execute(
            """
            SELECT 
              CASE
                WHEN seed_source LIKE 'tag:%' THEN 'tag-based'
                ELSE seed_source
              END AS source_group,
              COUNT (*) AS n
            FROM tracks
            GROUP BY source_group
            ORDER BY n DESC
            """
        ).fetchall()
    print(f"\n Total unique tracks in corpus: {total}")
    print("By source:")
    for row in by_source:
        print(f" {row['source_group']}: {row['n']}")

def main():
    init_db()
    lastfm = get_lastfm_client()
    print(f"Tag-based seeding: ({len(CURATED_TAGS)} tags x {TRACKS_PER_TAG} tracks)")
    collect_from_tags(lastfm, CURATED_TAGS, TRACKS_PER_TAG)

    print(f"\nChart Seeding (top {CHART_LIMIT} global)")
    collect_from_chart(lastfm, CHART_LIMIT)

    report_totals()

if __name__ == '__main__':
    main()

