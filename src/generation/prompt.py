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

class RecommendationResponse(BaseModel):
    """The full LLM response, containing the ranked list of recommendations."""
    recommendations: list[Recommendation] = Field(
        description = " Ranked list, most relevant first" 
    )

SYSTEM_PROMPT = """
You are a music recommender system that produces grounded, personalised recommendations.

The rules you have to stick to:
1. You may only recommend tracks whose track_id appears in the provided candidate list
2. Every recommendation must included a justification that draws on facts from the summary documnets
3. Rank recommendations by how well they match the user's query. Consider genre, tags and artist before the title
4. If some candidates only weakly match the user's query, exclude them rather than return them back to the user
5. Respond ONLY with valid JSON matching the provided schema"""

USER_PROMPT_TEMPLATE = """\
User query: {query}
Retrieved candidates (top-{k} dense retrieval):

{candidates_block}

Task: Return the {n} tracks from the candidates above that best match the user's query

Respond with a JSON objkect matching this schema:
{schema}

Return only JSON, no other text."""

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

