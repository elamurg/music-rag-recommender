"""Last.fm API client helpers."""
import pylast

from .retry import retry
from .schemas import (
    ArtistEnrichment,
    SimilarTrackInfo,
    TagInfo,
    TrackEnrichment,
)

#how many similar tracks to store per source track
SIMILAR_TRACKS_LIMIT = 30


@retry
def fetch_track_info(track: pylast.Track) -> dict:
    """Fetch track metadata from Last.fm's track.getInfo."""
    return {
        "mbid": track.get_mbid(),
        "listener_count": track.get_listener_count(),
        "playcount": track.get_playcount(),
        "wiki_summary": track.get_wiki_summary(),
        "wiki_content": track.get_wiki_content(),
        "album": track.get_album(),
        "url": track.get_url(),
        "top_tags": track.get_top_tags(limit=20),
    }


@retry
def fetch_similar_tracks(track: pylast.Track, limit: int = SIMILAR_TRACKS_LIMIT) -> list:
    """Fetch top-N similar tracks from Last.fm's track.getSimilar."""
    return track.get_similar(limit=limit)


def parse_enrichment(
    info_raw: dict,
    similar_raw: list,
) -> TrackEnrichment:
    """Convert raw pylast track responses into a typed TrackEnrichment."""
    enrichment = TrackEnrichment()

    mbid = info_raw.get("mbid")
    enrichment.mbid = mbid if mbid else None

    lc = info_raw.get("listener_count")
    enrichment.listener_count = int(lc) if lc is not None else None

    pc = info_raw.get("playcount")
    enrichment.playcount = int(pc) if pc is not None else None

    ws = info_raw.get("wiki_summary")
    enrichment.wiki_summary = ws.strip() if ws else None

    wc = info_raw.get("wiki_content")
    enrichment.wiki_content = wc.strip() if wc else None

    album = info_raw.get("album")
    if album:
        try:
            enrichment.album_name = album.get_name()
        except Exception:
            enrichment.album_name = None

    enrichment.lastfm_url = info_raw.get("url") or None

    for item in info_raw.get("top_tags", []):
        try:
            enrichment.tags.append(
                TagInfo(name=item.item.get_name().strip(), weight=int(item.weight))
            )
        except (AttributeError, ValueError):
            continue

    for item in similar_raw:
        try:
            similar_track = item.item
            enrichment.similar.append(
                SimilarTrackInfo(
                    target_name=similar_track.get_name().strip(),
                    target_artist=similar_track.get_artist().get_name().strip(),
                    similarity=float(item.match),
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
    """Convert raw pylast artist responses into a typed ArtistEnrichment."""
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