"""Request and response schemas for the SoundRAG REST API.
the pydantic models validate incoming requests, serialise outgoing responses to JSON, 
auto-generate OpenAI docs."""

from pydantic import BaseModel, Field

class RecommendRequest(BaseModel):
    """input payload for /recommend endpoint."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural language music preference query",
        examples=["sad indie about heartbreak"],
    )
    n: int = Field(
        default=5,
        ge=1, 
        le=10,
        description="Number of recommendations to return (1-10)"
    )

class RecommendationItem(BaseModel):
    """One recommended track in the response"""
    track_id: int = Field(description="Corpus track_id, stable identifier")
    name: str = Field(description="Track name")
    artist_name: str = Field(description="Artist name")
    justification: str = Field(description="two to three sentance explanation grounded in retrieved facts")
    spotify_url: str | None = Field(
        default=None,
        description="Playable Spotify URL if resolved above confidence threshold",
    )
    spotify_confidence: float | None = Field(
        default=None,
        description="Rapu=idFuzz confidence score (0-100) of the Spotify match",
    )

class RecommendResponse(BaseModel):
    "full response from /recommend"
    query: str = Field(description="echo of the input query for client-side caching")
    recommendations: list[RecommendationItem]
    grounded: bool = Field(
        description=(
            "True if recommendation were produced by LLM re-ranking; "
            "False if the system fell back to dense retrieval only"
        )
    )

#helth check
class HealthResponse(BaseModel):
    """Service health check"""
    status: str = Field(description="'ok' if all components reachable")
    corpus_ready: bool
    index_ready: bool
    llm_ready: bool

class StatsResponse(BaseModel):
    """corpus stats for observability"""
    total_tracks: int
    tracks_with_lyrics: int
    tracks_with_wiki: int
    unique_tags: int
    index_vector_count: int
    embedding_dim: int
    
