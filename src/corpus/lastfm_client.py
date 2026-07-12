import functools
import time
from typing import Callable, TypeVar

import pylast
import requests

from .schemas import TrackEnrichment, TagInfo, SimilarTrackInfo, ArtistEnrichment

T = TypeVar("T")
SIMILAR_TRACKS_LIMIT = 30
MAX_RETRIES = 3
INITIAL_BACKOFF_SEC = 5
BACKOFF_MULTIPLIER = 3
PERMANENT_ERROR_MARKERS= (
    "Tracks not found",
    "Artist not found",
    "Album not found",
    "Invalid parameters"
    "Invalid API key"
)

def is_permanent_error(exc: Exception) -> bool:
    """Checking if the errors are permanent or if there is posibility to retry"""
    if not isinstance(exc, pylast.WSError):
        return False
    details = str(exc.details).lower() if exc.details else ""
    return any(marker in details for marker in PERMANENT_ERROR_MARKERS)

def retry(fn: Callable[..., T]):
    """Retry exponential backoff, the retries limit in MAX_RETRIES doubling the wait time each time.
    Permanent erros raise immediately"""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        backoff = INITIAL_BACKOFF_SEC
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except (requests.RequestException, TimeoutError) as e:
                # always transient
                last_exc = e
            except pylast.WSError as e:
                if is_permanent_error(e):
                    raise 
                last_exc = e
            except Exception as e:
                #unknown exception
                if attempt == MAX_RETRIES:
                    raise
                last_exc = e
 
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER
        #exhausted retries
        raise last_exc 
    return wrapper

@retry
def fetch_track_info(track: pylast.Track) -> dict:
    """Fetches track metadata and returns it as a dictoonary with keys that might be missing or none which will be normalised later"""
    return {
        "mbid": track.get_mbid(),
        "listener_count": track.get_listener_count(),
        "playcount": track.get_playcount(),
        "wiki_content": track.get_wiki_content(),
        "album": track.get_album(),
        "url": track.get_url(),
        "top_tags": track.get_top_tags(limit=20),
    }

@retry
def fetch_similar_tracks(track: pylast.Track, limit: int = SIMILAR_TRACKS_LIMIT) -> list:
    """Fetch top-N similar tracks"""
    return track.get_similar(limit=limit)

def parse_enrichment(info_raw: dict, similar_raw: list) -> TrackEnrichment:
    """COnvert raw pylast responses into a typed TrackEnrichment."""
    enrichment = TrackEnrichment()
    mbid = info_raw.get("mbid")
    enrichment.mbid = mbid if mbid else None

    list_count = info_raw.get("listener_count")
    enrichment.listener_count = int(list_count) if list_count is not None else None

    playcount = info_raw.get("playcount")
    enrichment.playcount = int(playcount) if playcount is not None else None

    wiki_summary = info_raw.get("wiki_summary")
    enrichment.wiki_summary = wiki_summary.strip() if wiki_summary else None

    wiki_content = info_raw.get("wiki_content")
    enrichment.wiki_content = wiki_content.strip() if wiki_content else None

    album = info_raw.get("album")
    if album:
        try:
            enrichment.album_name = album.get_name()
        except Exception:
            enrichment.album_name = None
    
    enrichement_lastfm_url = info_raw.get("url") or None

    for item in info_raw.get("top_tags", []):
        try:
            enrichment.tags.append(
                TagInfo(name = item.item.get_name().strip(), weight = int(item.weight))
            )
        except (AttributeError, ValueError):
            continue
    
    for item in similar_raw:
        try:
            similar_track = item.item
            enrichment.similar.append(
                SimilarTrackInfo(
                    target_name = similar_track.get_name().strip(),
                    target_artist = similar_track.get_artist().get_name().strip(),
                    similarity = float(item.match), 
                )
            )
        except (AttributeError, ValueError):
            continue
    return enrichment

@retry
def fetch_artist_info(artist: pylast.Artist) -> dict:
    """Fetch artist metadata from Last.fm's artist.getInfo."""
    return {
        "mbid": artist.get_mbid(),
        "listener_count": artist.get_listener_count(),
        "playcount": artist.get_playcount(),
        "bio_summary": artist.get_bio_summary(),
        "bio_content": artist.get_bio_content(),
        "top_tags": artist.get_top_tags(limit=20),
    }


def parse_artist_enrichment(info_raw: dict) -> ArtistEnrichment:
    """Convert raw pylast artist responses into a typed ArtistEnrichment"""
    enrichment = ArtistEnrichment()

    mbid = info_raw.get("mbid")
    enrichment.mbid = mbid if mbid else None

    lc = info_raw.get("listener_count")
    enrichment.listener_count = int(lc) if lc is not None else None

    pc = info_raw.get("playcount")
    enrichment.playcount = int(pc) if pc is not None else None

    bs = info_raw.get("bio_summary")
    enrichment.bio_summary = bs.strip() if bs else None

    bc = info_raw.get("bio_content")
    enrichment.bio_content = bc.strip() if bc else None

    for item in info_raw.get("top_tags", []):
        try:
            enrichment.tags.append(
                TagInfo(name=item.item.get_name().strip(), weight=int(item.weight))
            )
        except (AttributeError, ValueError):
            continue

    return enrichment