"""Stores prompt templates and JSON output schemas validated with pydantic.
The prompt follows a specific structure which is a canonical grounded-generation practice
(The system message defines the recommender's role and grounding coonstraints, 
the user message contains query and retrieved candidates and output format)"""

from pydantic import BaseModel, Field
from src.retrieval.query import TrackHit

class Recommendation(BaseModel):
    """One recommended track, with a natural-language justification"""
    track_id: int = Field(
        description = "The track_id, which MIST come from the provided candidates."
    )
    justification: str = Field(
        description = (
            "Two to three sentences explaining why this track fits the query, " \
            "grounded in the facts from the retrieved passage. Do not invent facts." 
        )
    )
    spotify_url: str | None = None
    spotify_confidence: float | None = None

class RecommendationResponse(BaseModel):
    """The full LLM response, containing the ranked list of recommendations."""
    recommendations: list[Recommendation] = Field(
        description = " Ranked list, most relevant first" 
    )

SYSTEM_PROMPT = """
You are a music recommender system that produces grounded, personalised recommendations.

Your rules:
1. You may ONLY recommend tracks whose track_id appears in the provided candidates list. Never invent track_ids or recommend tracks not in the candidates.
2. Every recommendation must include a justification that draws on facts from the retrieved track description. Do not invent facts about tracks — if the description doesn't say something, don't claim it.
3. Rank recommendations by how well they match the user's query. Consider genre, mood, era, energy, and lyrical themes as described in the retrieved passages.
4. If some candidates only weakly match the query, exclude them rather than pad the list. Better to return 3 strong matches than 5 mediocre ones.
5. Respond ONLY with valid JSON matching the provided schema. No preamble, no markdown code fences, no commentary outside the JSON."""

USER_PROMPT_TEMPLATE = """\
User query: {query}
Retrieved candidates (top-{k} dense retrieval):

{candidates_block}

Task: Return the {n} tracks from the candidates above that BEST match the user's query, ranked by relevance. For each, write a two-to-three-sentence justification grounded in the retrieved descriptions.

Respond with a JSON objkect matching this schema:
{schema}

Return only the JSON, no other text."""

def format_candidate(hit: TrackHit) -> str:
    """Format the retrieval hit for inclusion in the prompt"""
    return (
        f"(track_id: {hit.track_id})\n"
        f"{hit.document_text}\n"
    )

def build_user_prompt(query: str, hits: list[TrackHit], n: int) -> str:
    """Formula to assemble the full message prompt from query, hits adn target count"""
    candidates_block = "\n".join(format_candidate(h)for h in hits)
    schema = RecommendationResponse.model_json_schema()
    schema_str = str(schema).replace("'", '"')

    return USER_PROMPT_TEMPLATE.format(
        query = query, 
        k = len(hits),
        n=n,
        candidates_block=candidates_block,
        schema = schema_str,
    )

