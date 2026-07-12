"""These typed data structures for lastfm_client.py and enrich.py"""
from dataclasses import dataclass, field

@dataclass
class TagInfo:
    """One tag applied to an artist ot track"""
    name:str
    weight: int

@dataclass
class SimilarTrackInfo:
    """One similar-track relationship as returned by track.get_similar()"""
    target_name:str
    target_artist: str
    similarity: float

@dataclass
class TrackEnrichment:
    """Everything that extracted from track.get_info() and track.get_similar() for a single track."""
    mbit: str | None = None
    listenet_count : int | None = None
    playcount: int | None = None
    wiki_summary: str | None = None
    wiki_content: str | None = None
    album_name: str | None = None
    lastfm_url: str | None = None
    tags: list[TagInfo] = field(default_factory=list)
    similar: list[SimilarTrackInfo] = field(default_factory=list)
