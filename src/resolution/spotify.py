"""The seach-and-match logic with catching."""
import os
from dataclasses import dataclass
from pathlib import Path

import spotipy
from dotenv import load_dotenv
from rapidfuzz import fuzz
from spotipy.oauth2 import SpotifyClientCredentials

from src.corpus.db import get_conn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT/".env")

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

CONFIDENCE_THRESHOLD = 85.0
_client: spotipy.Spotify | None = None

@dataclass
class SpotifyMatch:
    """Result of a Spotify resolution attempt"""
    track_id: int
    spotify_track_id: str | None
    spotify_url:str | None
    confidence: float

def _get_client() -> spotipy.Spotify:
    """Load spotify client from credentials"""
    global _client
    if _client is None:
        if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
            raise EnvironmentError(
                "SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET not found in .env"
            )
        auth_manager = SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET,
        )
        _client = spotipy.Spotify(auth_manager=auth_manager)
    return _client

def _fetch_cached(conn, track_id:int) -> SpotifyMatch | None:
    """Return match if it exists other return none"""
    row = conn.execute(
        """
        SELECT track_id, spotify_track_id, spotify_url, confidence
        FROM spotify_matches
        WHERE track_id = ?
        """, 
        (track_id,),
    ).fetchone()
    if row is None:
        return None
    return SpotifyMatch(
        track_id=row["track_id"],
        spotify_track_id=row["spotify_track_id"],
        spotify_url=row["spotify_url"], 
        confidence=row["confidence"],
    )

def _cache_result(conn, match: SpotifyMatch) -> None:
    """persisting a resolution result to the cache"""
    conn.execute(
        """
        INSERT OR REPLACE INTO spotify_matches
            (track_id, spotify_track_id, spotify_url, confidence, resolved_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (
            match.track_id,
            match.spotify_track_id,
            match.spotify_url,
            match.confidence,
        ),
    )

def _score_match(
    query_name: str, query_artist: str,
    hit_name: str, hit_artist: str,
) -> float:
    """Combined similarity score across name and artist."""
    name_score = fuzz.token_set_ratio(query_name.lower(), hit_name.lower())
    artist_score = fuzz.token_set_ratio(query_artist.lower(), hit_artist.lower())
    return (name_score + artist_score) / 2.0
 
 
def _search_spotify(
    client: spotipy.Spotify, name: str, artist: str
) -> tuple[str | None, str | None, float]:
    """Query Spotify search and return (track_id, url, confidence). Returns (None, None, 0.0) if Spotify has no matching results at all.
    Returns (id, url, score) for the top result even if below threshold"""
    query = f"track:{name} artist:{artist}"
    try:
        results = client.search(q=query, type="track", limit=1)
    except Exception as e:
        print(f"[resolution] search failed for '{name}' by '{artist}': {e}")
        return None, None, 0.0
 
    items = results.get("tracks", {}).get("items", [])
    if not items:
        return None, None, 0.0
 
    hit = items[0]
    hit_name = hit["name"]
    hit_artist = hit["artists"][0]["name"] if hit["artists"] else ""
    confidence = _score_match(name, artist, hit_name, hit_artist)
 
    return hit["id"], hit["external_urls"]["spotify"], confidence
 
 
def resolve_track(
    track_id: int, name: str, artist_name: str
) -> SpotifyMatch:
    """Resolve a corpus track to its Spotify URL. Checks the cache first. On cache miss, queries Spotify's search API,
    scores the top hit via fuzzy matching, and caches the result. """
    with get_conn() as conn:
        cached = _fetch_cached(conn, track_id)
        if cached is not None:
            return cached
 
    # cache miss 
    client = _get_client()
    spotify_id, spotify_url, confidence = _search_spotify(client, name, artist_name)
 
    # apply confidence threshold
    if confidence < CONFIDENCE_THRESHOLD:
        match = SpotifyMatch(
            track_id=track_id,
            spotify_track_id=None,
            spotify_url=None,
            confidence=confidence,
        )
    else:
        match = SpotifyMatch(
            track_id=track_id,
            spotify_track_id=spotify_id,
            spotify_url=spotify_url,
            confidence=confidence,
        )
 
    # cache the result 
    with get_conn() as conn:
        _cache_result(conn, match)
 
    return match
 
 
def resolve_batch(
    tracks: list[tuple[int, str, str]]
) -> dict[int, SpotifyMatch]:
    """Resolve multiple tracks efficiently, sharing one connection.
 
    Args:
        tracks: list of (track_id, name, artist_name) tuples
 
    Returns:
        Dict mapping track_id -> SpotifyMatch
    """
    return {
        track_id: resolve_track(track_id, name, artist)
        for track_id, name, artist in tracks
    }
 
if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-id", type=int, help="Resolve a specific track by id")
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="Resolve N random tracks from the corpus (default 5)",
    )
    args = parser.parse_args()
 
    with get_conn() as conn:
        if args.track_id:
            row = conn.execute(
                "SELECT id, name, artist_name FROM tracks WHERE id = ?",
                (args.track_id,),
            ).fetchone()
            if not row:
                print(f"Track {args.track_id} not found")
                exit(1)
            rows = [row]
        else:
            rows = conn.execute(
                """
                SELECT id, name, artist_name FROM tracks
                WHERE enriched_at IS NOT NULL
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (args.sample,),
            ).fetchall()
 
    for row in rows:
        match = resolve_track(row["id"], row["name"], row["artist_name"])
        marker = "OK " if match.spotify_url else "-- "
        print(
            f"{marker}[{match.confidence:5.1f}] "
            f"{row['name']} by {row['artist_name']}"
        )
        if match.spotify_url:
            print(f"       {match.spotify_url}")