"""Genius API client for lyrics ingestion"""
import os
from pathlib import Path

import lyricsgenius
from dotenv import load_dotenv

from .retry import retry

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")

if not GENIUS_ACCESS_TOKEN:
    raise EnvironmentError("GENIUS_ACCESS_TOKEN not found in .env")

UNICODE_SPACES_TO_NORMALISE = (
    "\u2005",  
    "\u2009",  
    "\u200a",  
    "\u2028", 
    "\xa0",    
)


def get_genius_client() -> lyricsgenius.Genius:
    """Configure the lyricsgenius client. Attributes are set post-construction
    because lyricsgenius 3.x removed most of them from __init__.
    """
    client = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN)
    client.verbose = False
    client.remove_section_headers = True
    client.timeout = 15
    client.skip_non_songs = True
    client.excluded_terms = ["(Remix)", "(Live)"]
    return client


def normalise_lyrics(raw: str) -> str:
    """Replace unusual unicode spaces with regular ASCII spaces."""
    for ch in UNICODE_SPACES_TO_NORMALISE:
        raw = raw.replace(ch, " ")
    return raw.strip()


@retry
def fetch_lyrics(
    client: lyricsgenius.Genius,
    track_name: str,
    artist_name: str,
) -> dict | None:
    """Search Genius for a song and return its lyrics, URL, and Genius ID."""
    song = client.search_song(track_name, artist_name)
    if song is None or not song.lyrics:
        return None
    return {
        "lyrics_text": normalise_lyrics(song.lyrics),
        "genius_url": song.url,
        "genius_id": getattr(song, "id", None) or getattr(song, "song_id", None)   }


