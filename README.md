# SoundRAG

**Retrieval-Augmented Music Recommendation for Cold-Start Users**

MSc thesis project by **Ela Murgelj**, Queen Mary University of London, School of Electronic Engineering and Computer Science, MSc Computer Science (Conversion), 2025/26.

Supervisor: Dr. Fabrizio Smeraldi.

**Repository:** [https://github.com/elamurg/music-rag-recommender](https://github.com/elamurg/music-rag-recommender)

---

## Overview

SoundRAG is an end-to-end music recommendation pipeline that solves the *user cold-start problem* — the well-documented failure mode of collaborative filtering when a new user arrives with no listening history. Instead of requiring interaction data, SoundRAG accepts natural-language preference queries (e.g. *"sad indie about heartbreak"* or *"chill synthwave for late night driving"*) and returns ranked, grounded recommendations with playable Spotify URLs.

The system is architected as four layers:

1. **Knowledge** — 9,499-track SQLite corpus enriched from Last.fm, Genius and Wikipedia.
2. **Retrieval** — FAISS dense-vector index over sentence-transformer embeddings, returning top-20 candidates in under 10 ms.
3. **Generation** — Claude Sonnet 4.5 re-ranks candidates and produces natural-language justifications, with a hallucination guard that rejects any recommendation not present in the retrieved set.
4. **Identity resolution** — Spotify search + RapidFuzz fuzzy matching maps track names to playable URLs, with results cached in the corpus database.

The full system is deployed as a FastAPI REST service, containerised with Docker, and evaluated across 100 hand-crafted queries against four baselines (Random, Popularity, BM25, Dense-only). Full methodology and results are in the accompanying thesis (`EM_SoundRAG.docx`).

---

## Quick start by Docker

The Docker deployment is the reference executable path per the handbook. It requires only Docker, Docker Compose, the `data/` directory (containing the SQLite corpus and FAISS index), and a `.env` file with API credentials.

```bash
git clone https://github.com/elamurg/music-rag-recommender.git
cd music-rag-recommender

#Provide credentials (see "Environment variables" below)
cp .env.example .env
# ...edit .env with your keys...

#Build the image and start the service
docker compose up --build
```

The first build takes 5–10 minutes as it downloads base images and compiles dependencies (PyTorch, faiss-cpu, sentence-transformers). Subsequent builds use cached layers and complete in ~30 seconds.

Once the service is running:

- **Interactive API docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health check:** `curl http://localhost:8000/health`
- **Example recommendation:**

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "sad indie about heartbreak", "n": 5}'
```

Full API documentation is provided in **Appendix E** of the thesis.

---

## Repository structure

```
music-rag-recommender/
├── src/
│   ├── corpus/              # Layer 1: SQLite corpus + ingestion pipeline
│   │   └── schema_changes/  # Idempotent SQL migrations
│   ├── embedding/           # Layer 2 (offline): text synthesis + FAISS index
│   ├── retrieval/           # Layer 2 (online): query pipeline
│   ├── generation/          # Layer 3: LLM re-ranking + hallucination guard
│   ├── resolution/          # Layer 4: Spotify identity resolution
│   ├── evaluation/          # 100-query evaluation framework
│   └── api/                 # FastAPI REST service
├── docs/
│   ├── architecture.md      # System-level design notes
│   ├── roadmap.md           # Phase-by-phase build log
│   ├── evaluation/
│   │   └── queries.json     # 100 evaluation queries (5 categories)
│   └── example_queries/     # Saved outputs from demonstration queries
├── data/
│   ├── raw/corpus.db        # SQLite corpus (~200 MB, git-ignored)
│   ├── embeddings/          # FAISS index (~14 MB, git-ignored)
│   └── evaluation/          # results.json, summary.json, significance.json, CSV
├── Dockerfile               # Multi-stage build (builder + runtime)
├── docker-compose.yml       # Local deployment
├── .dockerignore
├── .env.example             # Credentials template
├── requirements.txt
├── README.md                # This file
└── EM_SoundRAG.docx         # Accompanying thesis document
```

---

## File-by-file reference

Detailed per-file descriptions organised by the four architectural layers.

### Layer 1 — Knowledge (`src/corpus/`)

The ingestion pipeline that builds the SQLite corpus from external APIs. Each phase is idempotent and can be safely interrupted and resumed.

| File | Description |
|---|---|
| `db.py` | SQLite connection management via a `get_conn()` context manager, with `row_factory` configured for dict-like column access. All other modules consume this rather than opening raw connections. |
| `seeds.py` | Phase 1: seed track collection from Last.fm's tag chart API, iterating over ~55 seed tags to build a genre-balanced starting corpus of ~9,500 tracks. |
| `enrich.py` | Phase 2: per-track enrichment via Last.fm's `track.getInfo` and `track.getSimilar` endpoints, populating tags, listener counts and the similar-tracks graph used for evaluation ground truth. |
| `enrich_artists.py` | Phase 3: artist biography enrichment via the `wikipedia-api` Python library, pulling ~3,174 unique artist biographies to support retrieval documents when track-level context is thin. |
| `enrich_lyrics.py` | Phase 4: lyrics ingestion via the `lyricsgenius` library, with RapidFuzz-based title validation at 85% confidence to filter false-positive matches (achieves 90.3% coverage across the corpus). |
| `lastfm_client.py` | Thin wrapper around `pylast` with exponential-backoff retry, respecting Last.fm's five-requests-per-second policy. |
| `genius_client.py` | Wrapper around `lyricsgenius` adding retry logic and title-match validation before persisting a lyrics row. |
| `retry.py` | Generic retry decorator with exponential backoff, used by all API-facing modules to survive transient failures without losing progress. |
| `schemas.py` | Pydantic dataclasses for external API responses (Last.fm track, Genius song, Wikipedia article), providing type-safe parsing between wire format and the SQLite schema. |

#### Schema migrations (`src/corpus/schema_changes/`)

Idempotent additive migrations. Each script uses `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS` so it can be re-run safely against an existing database.

| File | Description |
|---|---|
| `add_artist_tags.py` | Creates the `artist_tags` many-to-many join table that maps artists to tags with weights, addressing the sparse coverage of track-level tags on Last.fm. |
| `add_track_documents.py` | Creates the `track_documents` table storing the synthesised per-track passages used as inputs to the sentence-transformer encoder. |
| `add_spotify_matches.py` | Creates the `spotify_matches` cache table storing Spotify URL, confidence score and resolved-at timestamp per corpus track_id, so repeat recommendations do not re-query the Spotify API. |
| `add_judge_cache.py` | Creates the `judge_cache` table for the LLM-as-judge evaluation results, keyed on `(query_id, track_id, model)` so re-running evaluation after a code change is free. |

### Layer 2 — Retrieval (`src/embedding/` and `src/retrieval/`)

The retrieval layer straddles offline and online modes. Documents are synthesised and embedded offline; the query is embedded and matched online against the pre-built FAISS index.

| File | Description |
|---|---|
| `embedding/synthesise.py` | Assembles a per-track text document from tags, Wikipedia content, artist biography and lyrics excerpt using the tag-forward template (top-3 tags amplified via repetition), with artist-tag fallback lifting coverage from 27% to 99% of tracks. |
| `embedding/embed.py` | Loads `sentence-transformers/all-MiniLM-L6-v2`, encodes all synthesised documents to 384-dimensional embeddings with L2 normalisation, and builds a FAISS `IndexFlatIP` wrapped in `IndexIDMap2` so each vector carries its corpus track_id. |
| `retrieval/query.py` | Query-time retrieval: the `retrieve(query, k)` function caches the model and FAISS index at module load, encodes the incoming query in the same vector space as the corpus, and returns the top-k `TrackHit` objects with pre-loaded metadata for downstream LLM consumption. |

### Layer 3 — Generation (`src/generation/`)

The LLM re-ranking layer. Takes retrieval candidates and produces the final ranked recommendations with grounded natural-language justifications.

| File | Description |
|---|---|
| `prompt.py` | Prompt templates and Pydantic schemas: `Recommendation` (track_id, justification, spotify_url, spotify_confidence) and `RecommendationResponse` (list of recommendations, grounded flag). The prompt instructs Claude to select tracks only from the retrieved set and to justify each pick using retrieved-passage facts. |
| `generate.py` | Full generation pipeline: calls Claude Sonnet 4.5 via the `anthropic` Python SDK, validates the JSON response against the Pydantic schema, filters out any hallucinated track_ids, and falls back to top-k FAISS results (with `grounded=False`) if the LLM call fails or produces unusable output. |

### Layer 4 — Identity Resolution (`src/resolution/`)

Maps recommended tracks back to playable Spotify URLs. The Spotify API is used solely for identity resolution — never for recommendation itself.

| File | Description |
|---|---|
| `spotify.py` | Spotify Web API search + RapidFuzz `token_set_ratio` fuzzy matching at 85% combined threshold across track and artist names. Results are cached in the `spotify_matches` table so subsequent recommendations of the same track return instantly, and no-match results are also cached to avoid retrying hopeless lookups. |

### Evaluation framework (`src/evaluation/`)

Runs every baseline against every query, computes metrics, aggregates by category, and runs paired significance tests. Full evaluation of 100 queries against 5 baselines takes ~30 minutes.

| File | Description |
|---|---|
| `queries.py` | Loads `docs/evaluation/queries.json` into typed `EvalQuery` instances and dispatches to the correct ground-truth resolver based on strategy (`tag_match` for genre/mood queries, `similar_artists` for reference queries, `llm_judge` deferred to `judge.py` for compound/contextual queries). |
| `baselines.py` | Implements five baseline retrievers with a shared interface. `Random` (metric floor), `Popularity` (top listener_count), `BM25` (sparse Okapi retrieval over synthesised documents with module-level index caching), `Dense-only` (FAISS without LLM re-ranking) and `RAG` (full SoundRAG pipeline). |
| `metrics.py` | Pure-math implementations of Precision@k, Recall@k, NDCG@k (with binary and graded variants), MRR and Hit@k. Includes a `compute_all_metrics()` convenience function that dispatches based on whether ground truth is a relevant set or a graded score dict. |
| `judge.py` | LLM-as-judge for subjective queries using Claude Haiku 4.5 (cheaper than Sonnet and different from the generator to reduce self-evaluation bias). Batches 10 tracks per API call and caches every judgment in `judge_cache` keyed by `(query_id, track_id, model)`. |
| `run.py` | Evaluation orchestrator with `tqdm` progress reporting. Executes every baseline against every query, aggregates metrics per category and overall, runs paired t-tests for key baseline comparisons, and writes JSON, CSV and significance outputs to `data/evaluation/`. |

### API service (`src/api/`)

Thin REST layer wrapping the underlying pipeline. All business logic lives in the layer modules; this layer only translates between HTTP and Python.

| File | Description |
|---|---|
| `main.py` | FastAPI application exposing three endpoints (`POST /recommend`, `GET /health`, `GET /stats`) with a lifespan warmup that pre-loads the retrieval model and FAISS index at startup, avoiding a ~5-second first-request latency spike. |
| `schemas.py` | Pydantic request/response models (`RecommendRequest`, `RecommendResponse`, `RecommendationItem`, `HealthResponse`, `StatsResponse`) driving both incoming validation and the auto-generated OpenAPI documentation at `/docs`. |

### Documentation (`docs/`)

| File | Description |
|---|---|
| `architecture.md` | Formal four-layer architecture document with per-file implementation logs, referenced from the thesis Methodology chapter. |
| `roadmap.md` | Phase-by-phase build log tracking design decisions, tradeoffs considered, and the rationale for each choice. Useful for the viva as a reference to defend design choices. |
| `evaluation/queries.json` | The 100 evaluation queries, stratified as 20 per category across Genre (A), Mood (B), Compound (C), Reference (D) and Contextual (E). Each query specifies its ground-truth strategy (`tag_match`, `similar_artists`, or `llm_judge`) and associated tags or reference artist. |
| `example_queries/` | Saved terminal outputs from the demonstration queries used throughout the thesis (`sad_indie_final.txt`, `synthwave_driving.txt`, etc.), providing reproducible reference outputs. |

### Deployment infrastructure (root)

| File | Description |
|---|---|
| `Dockerfile` | Multi-stage build: a `builder` stage installs Python dependencies into a virtualenv using `build-essential` and gcc; a lean `runtime` stage copies only the virtualenv and application code, keeping the final image at ~600 MB rather than ~2 GB. Runs as a non-root `appuser` for security hardening. |
| `docker-compose.yml` | One-command local deployment that mounts `data/` as a read-only volume, injects credentials from `.env`, exposes port 8000, and configures a health check hitting the `/health` endpoint. The corpus is mounted rather than baked into the image so it can be swapped without rebuilding. |
| `.dockerignore` | Excludes virtual environments, git history, IDE state, and `data/` directory from the Docker build context, keeping build times fast and preventing accidental credential leaks into the image. |
| `.env.example` | Template file listing all required environment variables with placeholder values, checked into git as a starting point for new developers. Real credentials go into `.env` which is git-ignored. |
| `requirements.txt` | Pinned Python dependencies including `fastapi`, `uvicorn`, `anthropic`, `sentence-transformers`, `faiss-cpu`, `spotipy`, `rapidfuzz`, `rank-bm25`, `scipy` and `pydantic`. Pinned to specific versions for reproducibility. |
| `EM_SoundRAG.docx` | The accompanying MSc dissertation research paper containing full methodology, evaluation and discussion. |

---

## Environment variables

The service reads all credentials from a `.env` file at the project root. Copy `.env.example` and fill in your own values.

| Variable                | Purpose                                             | Obtain from                                                                 |
|-------------------------|-----------------------------------------------------|-----------------------------------------------------------------------------|
| `ANTHROPIC_API_KEY`     | Claude Sonnet 4.5 (generation) + Haiku 4.5 (judge)  | [console.anthropic.com](https://console.anthropic.com/) → API Keys          |
| `SPOTIFY_CLIENT_ID`     | Spotify Web API identity resolution                 | [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard/) |
| `SPOTIFY_CLIENT_SECRET` | Spotify Web API secret                              | [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard/) |
| `LASTFM_API_KEY`        | Last.fm API (offline corpus ingestion only)         | [last.fm/api/account/create](https://www.last.fm/api/account/create)        |
| `GENIUS_ACCESS_TOKEN`   | Genius API (offline lyrics ingestion only)          | [genius.com/api-clients](https://genius.com/api-clients)                    |

**For running the deployed API only** (i.e. serving recommendations from the pre-built corpus), you need `ANTHROPIC_API_KEY`, `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET`. The Last.fm and Genius keys are only needed if you want to rebuild the corpus from scratch.

**Approximate API cost per query:** ~$0.005 (Claude Sonnet). Free within Anthropic's initial credit allocation.

---

## Running the service

### Option 1 — Docker (reproducible)

```bash
docker compose up --build       # build + start (foreground)
docker compose up -d            # background mode
docker compose logs -f          # follow logs
docker compose down             # stop and remove
```

### Option 2 — Local Python (development)

For development you can bypass Docker and run directly:

```bash
# Python 3.11 required
python -m venv .venv
source .venv/bin/activate       # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Start the service
uvicorn src.api.main:app --reload --port 8000
```

Warmup takes ~5 seconds while the sentence-transformer model and FAISS index load.

### Testing the endpoints

Interactive Swagger UI is served automatically at [http://localhost:8000/docs](http://localhost:8000/docs). Alternatively:

```bash
# Health check (returns component readiness)
curl http://localhost:8000/health

# Corpus statistics
curl http://localhost:8000/stats

# Recommendation
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "melancholic music for a rainy walk", "n": 5}'
```

Full API contract (request/response schemas, status codes, error handling) is documented in **Appendix E** of the thesis.

---

## Reproducing the evaluation

The full 100-query evaluation from the thesis can be re-run against the pre-built corpus:

```bash
#Verify environment
python -c "from src.corpus.db import get_conn; \
    print(get_conn().execute('SELECT COUNT(*) FROM tracks').fetchone()[0])"
#Expected output: 9506

#Smoke test: 10 random queries across all 5 baselines
python -m src.evaluation.run --sample 10

#Full 100-query evaluation
python -m src.evaluation.run
```

The full run takes approximately 30 minutes and costs ~$2–3 in Anthropic API credits (Claude Haiku judgments are cached in `judge_cache`, so re-runs are free). Output is written to:

```
data/evaluation/
├── results.json          # per-query per-baseline detail
├── summary.json          # aggregated metrics by baseline and category
├── significance.json     # paired t-tests (SoundRAG vs each baseline)
└── results.csv           # pandas-friendly flat file
```

To evaluate a single category:

```bash
python -m src.evaluation.run --category genre
python -m src.evaluation.run --category compound
```

To run individual baselines against a query interactively:

```bash
python -m src.evaluation.baselines "sad indie about heartbreak" -k 5 --baseline all
```

---

## Rebuilding the corpus from scratch (advanced)

If you need to reproduce the corpus rather than use the pre-built one, the four ingestion phases run in sequence. Each is idempotent (safe to interrupt and resume). Total wall-clock time is approximately 40 hours end-to-end due to API rate limits.

```bash
#Phase 1: Seed track collection (~30 seconds)
python -m src.corpus.seeds

#Phase 2: Track enrichment — Last.fm metadata + similar tracks (~17 hours)
python -m src.corpus.enrich

#Phase 3: Artist enrichment — Wikipedia biographies
python -m src.corpus.enrich_artists

#Phase 4: Lyrics ingestion — Genius API (~13.5 hours)
python -m src.corpus.enrich_lyrics

#Text synthesis
python -m src.embedding.synthesise

#FAISS index construction
python -m src.embedding.embed
```

The ingestion is network-bound (respecting Last.fm's five-requests-per-second policy and Genius's rate limits) rather than CPU-bound; it runs comfortably on a laptop but should be kicked off as an overnight job.

---

## Documentation and references

| Document                          | Location                                         |
|-----------------------------------|--------------------------------------------------|
| Thesis (research paper)           | `EM_SoundRAG.docx`                               |
| System architecture               | `docs/architecture.md`                           |
| Phase-by-phase build log          | `docs/roadmap.md`                                |
| 100 evaluation queries            | `docs/evaluation/queries.json`                   |
| Example query outputs             | `docs/example_queries/`                          |
| API endpoint documentation        | Thesis Appendix E                                |
| Corpus schema and ER diagram      | Thesis Appendix B                                |
| Embedding model specification     | Thesis Appendix D                                |
| Generative AI usage declaration   | Thesis Appendix A                                |

---

## Requirements

- Docker + Docker Compose (recommended path) *or* Python 3.11 (local path)
- ~1 GB disk space (corpus + FAISS index + Docker image)
- Internet connection (API calls to Anthropic and Spotify at inference time)
- Anthropic API credit (~$5 covers development and evaluation)

Key Python dependencies (see `requirements.txt` for full list):

- `fastapi`, `uvicorn` — REST service
- `anthropic` — Claude API client
- `sentence-transformers` — MiniLM-L6-v2 embeddings
- `faiss-cpu` — vector index
- `spotipy` — Spotify Web API
- `rapidfuzz` — fuzzy string matching for identity resolution
- `rank-bm25` — sparse retrieval baseline
- `scipy` — paired t-tests for statistical significance
- `pydantic` — schema validation

---

## License

The source code in this repository is released under the MIT License for academic and non-commercial research purposes. Third-party data accessed via API (Last.fm tags and similarity graph, Genius lyrics excerpts, Wikipedia artist biographies, Spotify metadata) is subject to the respective terms of service of each provider. Lyrics excerpts are stored for retrieval-augmentation only and are not displayed verbatim to end users.

---

## Citation

If you build on this work, please cite:

> Murgelj, E. (2026) *SoundRAG: A Retrieval-Augmented Approach to Cold-Start Music Recommendation*. MSc dissertation, Queen Mary University of London.

---

## Contact

For questions about the code, evaluation reproduction or thesis content, please open an issue on the GitHub repository or contact the author through the university.