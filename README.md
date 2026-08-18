# SoundRAG

**Retrieval-Augmented Music Recommendation for Cold-Start Users**

MSc thesis project by **Ela Murgelj** (250057317), Queen Mary University of London, School of Electronic Engineering and Computer Science, MSc Computer Science (Conversion), 2025/26.

Supervisor: Dr. Fabrizio Smeraldi.

**Repository:** [https://github.com/elamurg/music-rag-recommender](https://github.com/elamurg/music-rag-recommender)

---
## Note for Examiners

This README is written to enable an examiner to understand, install and run this project end-to-end with minimal friction. The sections below are as follows:

1. **[Data files required for reproduction](#data-files-required-for-reproduction)** — the pre-built corpus and FAISS index are too large to include in the standard code submission and are hosted separately.
2. **[Running without deployment (offline evaluation of the system)](#running-without-deployment-offline-evaluation-of-the-system)** — if you cannot run the deployment due to API credential requirements, the accompanying video and Appendix G sample outputs provide the evidence you need.

---

## Table of Contents

1. [Overview](#overview)
2. [Project structure at a glance](#project-structure-at-a-glance)
3. [Data files required for reproduction](#data-files-required-for-reproduction)
4. [Quick start (Docker)](#quick-start-docker)
5. [Environment variables and where to obtain them](#environment-variables-and-where-to-obtain-them)
6. [Repository structure — full file-by-file reference](#repository-structure--full-file-by-file-reference)
7. [Running the service](#running-the-service)
8. [Testing the endpoints](#testing-the-endpoints)
9. [Reproducing the evaluation](#reproducing-the-evaluation)
10. [Rebuilding the corpus from scratch](#rebuilding-the-corpus-from-scratch-advanced)
11. [Running without deployment (offline evaluation of the system)](#running-without-deployment-offline-evaluation-of-the-system)
12. [Cross-references to thesis document](#cross-references-to-thesis-document)
13. [Requirements](#requirements)
---

## Overview

SoundRAG is an end-to-end music recommendation pipeline that solves the *user cold-start problem*, the well-documented failure mode of collaborative filtering when a new user arrives with no listening history. Instead of requiring interaction data, SoundRAG accepts natural-language preference queries (e.g. *"sad indie about heartbreak"* or *"chill synthwave for late night driving"*) and returns ranked, grounded recommendations with playable Spotify URLs.

The system is architected as four layers, described fully in the acompanying thesis (Section III):

1. **Knowledge** — 9,499-track SQLite corpus enriched from Last.fm, Genius and Wikipedia.
2. **Retrieval** — FAISS dense-vector index over sentence-transformer embeddings, returning top-20 candidates in under 10 ms.
3. **Generation** — Claude Sonnet 4.5 re-ranks candidates and produces natural-language justifications, with a hallucination guard that rejects any recommendation not present in the retrieved set.
4. **Identity resolution** — Spotify search + RapidFuzz fuzzy matching maps track names to playable URLs, with results cached in the corpus database.

The full system is deployed as a FastAPI REST service, containerised with Docker, and evaluated across 100 hand-crafted queries against four baselines (Random, Popularity, BM25, Dense-only). Full methodology and results are in the accompanying thesis (`EM_SoundRAG.docx`).

---

## Project structure at a glance

```
music-rag-recommender/
├── src/                          Source code (4 architectural layers + evaluation + API)
├── docs/                         Design documents, evaluation queries, sample outputs
├── data/                         Corpus, FAISS index, evaluation results
├── Dockerfile                    Multi-stage container build
├── docker-compose.yml            One-command local deployment
├── requirements.txt              Python dependencies
├── requirements-dev.txt          More project dependencies
├── .env.example                  Credentials template
├── README.md                     This file

```

---

## Data files required for reproduction

Two data files are required for the system to run but are too large for standard code submission:

| File | Size | Purpose |
|------|------|---------|
| `data/raw/corpus.db` | 96 MB | SQLite corpus with 9,499 enriched tracks (metadata, tags, lyrics, similarity graph) |
| `data/embeddings/corpus.faiss` | 14 MB | Pre-built FAISS IndexFlatIP with 9,499 unit-normalised 384-dimensional vectors |

**Where to obtain them:**

**Git clone** The GitHub repository at https://github.com/elamurg/music-rag-recommender uses *Git LFS* to store these files. Cloning the repository fetches both files automatically:

```bash
#Git LFS must be installed first
git lfs install
git clone https://github.com/elamurg/music-rag-recommender.git
cd music-rag-recommender

---

## Quick start by Docker

The Docker deployment is the intended and easiest way to run the system.

```bash
#Create your .env file with API credentials (see next section)
cp .env.example .env
#Edit .env with your keys

#Build the image and start the service (first build ~5-10 min)
docker compose up --build

#Test the deployment (in a separate terminal, once startup is complete)
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "sad indie about heartbreak", "n": 5}'
```

Once running:
- **Interactive API docs:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health
- **Corpus stats:** http://localhost:8000/stats

---

## Environment variables and where to obtain them

The service reads all credentials from a `.env` file at the project root. Copy `.env.example` to `.env` and fill in your own values.

| Variable | Purpose | Obtain from |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Claude Sonnet 4.5 (generation) and Claude Haiku 4.5 (evaluation judge) | https://console.anthropic.com — API Keys section. Requires funded account (approximately $5 covers development and full evaluation) |
| `SPOTIFY_CLIENT_ID` | Spotify Web API identity resolution | https://developer.spotify.com/dashboard — Create a new app, use Client ID |
| `SPOTIFY_CLIENT_SECRET` | Spotify Web API secret (paired with client ID) | Same as above — click "View client secret" |
| `LASTFM_API_KEY` | Last.fm API access (offline corpus rebuild only) | https://www.last.fm/api/account/create |
| `GENIUS_ACCESS_TOKEN` | Genius API access (offline corpus rebuild only) | https://genius.com/api-clients |

**For running the deployed service**, only three credentials are strictly required:
- `ANTHROPIC_API_KEY`
- `SPOTIFY_CLIENT_ID`
- `SPOTIFY_CLIENT_SECRET`

The Last.fm and Genius credentials are needed only if rebuilding the corpus from scratch.

---

## Repository structure

```
music-rag-recommender/
├── .venv/                       (ignored - dev only)
├── data/                        (LFS or external hosting)
    ├── evaluation/ 
    ├── embeddings/
    ├── raw/                     (location of corpus.db)
    └── queries/                    
├── docs/
│   ├── evaluation/
│   └── example_queries/
├── src/
│   ├── api/
│   ├── corpus/
        └── schema_changes/
│   ├── embedding/
│   ├── evaluation/
│   ├── generation/
│   ├── resolution/
│   ├── retrieval/
│   └── smoke_test.py
├── .cache
├── .dockerignore
├── .env                         (never uploaded)
├── .env.example
├── .gitignore
├── docker-compose.yml
├── dockerfile
├── pyproject.toml
├── rag-architecture.pdf
├── README.md
├── requirements-dev.txt
└── requirements.txt
```

---

## File-by-file reference

Every file in the submission with a one-line description of its purpose. Organised by architectural layer (matching Section III of the thesis) and by role.

### Layer 1 — Knowledge (`src/corpus/`)

The ingestion pipeline that builds the SQLite corpus from external APIs. Each phase is idempotent and can be safely interrupted and resumed.

| File | Description |
|------|-------------|
| `src/corpus/db.py` | SQLite connection management via `get_conn()` context manager with `row_factory` configured for dict-like access. All other modules consume this rather than opening raw connections. |
| `src/corpus/seeds.py` | Phase 1 ingestion: seed track collection from Last.fm's tag chart API across approximately 55 seed tags, producing a genre-balanced starting corpus of approximately 9,500 tracks. |
| `src/corpus/enrich.py` | Phase 2: per-track enrichment via Last.fm's `track.getInfo` and `track.getSimilar`, populating tags, listener counts and the similar-tracks graph used for evaluation ground truth. |
| `src/corpus/enrich_artist.py` | Phase 3: artist biography enrichment via the `wikipedia-api` Python library, retrieving approximately 3,174 unique artist biographies. |
| `src/corpus/enrich_lyrics.py` | Phase 4: lyrics ingestion via the `lyricsgenius` library with RapidFuzz-based title validation at 85% confidence, achieving 90.3% lyrics coverage across the corpus. |
| `src/corpus/lastfm_client.py` | Wrapper around `pylast` adding exponential-backoff retry, respecting Last.fm's five-requests-per-second policy. |
| `src/corpus/genius_client.py` | Wrapper around `lyricsgenius` adding retry logic and title-match validation. |
| `src/corpus/retry.py` | Generic retry decorator with exponential backoff, used by all API-facing modules. |
| `src/corpus/schemas.py` | Pydantic dataclasses for external API responses (Last.fm track, Genius song, Wikipedia article). |

#### Schema migrations (`src/corpus/schema_changes/`)

Idempotent additive migrations. Each script uses `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS` so it can be re-run safely against an existing database.

| File | Description |
|------|-------------|
| `src/corpus/schema_changes/add_artist_tags.py` | Creates the `artist_tags` many-to-many join table addressing sparse track-level tag coverage. |
| `src/corpus/schema_changes/add_track_documents.py` | Creates the `track_documents` table storing synthesised per-track embedding documents. |
| `src/corpus/schema_changes/add_spotify_matches.py` | Creates the `spotify_matches` cache table storing resolved Spotify URLs by corpus track_id. |
| `src/corpus/schema_changes/add_judge_cache.py` | Creates the `judge_cache` table for LLM-as-judge evaluation results, keyed on `(query_id, track_id, model)`. |

### Layer 2 — Retrieval (`src/embedding/` and `src/retrieval/`)

The retrieval layer straddles offline and online modes. Documents are synthesised and embedded offline; the query is embedded and matched online against the pre-built FAISS index.

| File | Description |
|------|-------------|
| `src/embedding/synthesise.py` | Assembles a per-track text document (approximately 1,000 characters) from tags, Wikipedia content, artist biography and lyrics excerpt using the tag-forward template with artist-tag fallback. |
| `src/embedding/embed.py` | Loads `sentence-transformers/all-MiniLM-L6-v2`, encodes all documents to 384-dimensional embeddings with L2 normalisation, and builds the FAISS IndexFlatIP wrapped in IndexIDMap2. |
| `src/retrieval/query.py` | Query-time retrieval. The `retrieve(query, k)` function caches model and index at module import, encodes the incoming query, and returns top-k `TrackHit` objects with pre-loaded metadata. |

### Layer 3 — Generation (`src/generation/`)

The LLM re-ranking layer. Takes retrieval candidates and produces the final ranked recommendations with grounded natural-language justifications.

| File | Description |
|------|-------------|
| `src/generation/prompt.py` | Prompt templates and Pydantic schemas: `Recommendation` (track_id, justification, spotify_url, spotify_confidence) and `RecommendationResponse` (list of recommendations, grounded flag). |
| `src/generation/generate.py` | Full generation pipeline: calls Claude Sonnet 4.5 via the `anthropic` Python SDK, validates the JSON response against the Pydantic schema, filters hallucinated track_ids (hallucination guard), and falls back to top-k FAISS results with `grounded=False` on LLM failure. |

### Layer 4 — Identity Resolution (`src/resolution/`)

Maps recommended tracks back to playable Spotify URLs. The Spotify API is used solely for identity resolution, never for recommendation itself.

| File | Description |
|------|-------------|
| `src/resolution/spotify.py` | Spotify Web API search with RapidFuzz `token_set_ratio` fuzzy matching at 85% combined confidence threshold. Results cached in `spotify_matches` table (both hits and misses cached to avoid retries). |

### Evaluation framework (`src/evaluation/`)

Runs every baseline against every query, computes metrics, aggregates by category, and runs paired significance tests. Full evaluation of 100 queries against 5 baselines takes  approximately 30 minutes.

| File | Description |
|------|-------------|
| `src/evaluation/queries.py` | Loads `docs/evaluation/queries.json` into typed `EvalQuery` instances and dispatches to the correct ground-truth resolver based on strategy (tag_match, similar_artists, llm_judge). |
| `src/evaluation/baselines.py` | Implements five baseline retrievers with a shared interface: Random (floor), Popularity (top listener_count), BM25 (sparse Okapi retrieval), Dense-only (FAISS without LLM re-ranking), RAG (full SoundRAG pipeline). |
| `src/evaluation/metrics.py` | Pure-math implementations of Precision@k, Recall@k, NDCG@k (binary and graded variants), MRR and Hit@k. Includes `compute_all_metrics()` dispatcher. |
| `src/evaluation/judge.py` | LLM-as-judge scoring for subjective queries using Claude Haiku 4.5 (Sonnet judges Sonnet is avoided to reduce self-evaluation bias). Batched 10 tracks per API call, results cached in `judge_cache`. |
| `src/evaluation/run.py` | Evaluation orchestrator with tqdm progress. Executes every baseline against every query, aggregates metrics per category, runs paired t-tests, and writes JSON and CSV outputs to `data/evaluation/`. |

### API service (`src/api/`)

Thin REST layer wrapping the underlying pipeline. All business logic lives in the layer modules; this layer only translates between HTTP and Python.

| File | Description |
|------|-------------|
| `src/api/main.py` | FastAPI application exposing three endpoints (`POST /recommend`, `GET /health`, `GET /stats`) with a lifespan warmup that pre-loads the retrieval model and FAISS index at startup. |
| `src/api/schemas.py` | Pydantic request/response models (`RecommendRequest`, `RecommendResponse`, `RecommendationItem`, `HealthResponse`, `StatsResponse`) driving both incoming validation and auto-generated OpenAPI docs at `/docs`. |

### Documentation (`docs/`, `data/`)

| File | Description |
|------|-------------|
| `docs/architecture.md` | Formal four-layer architecture document with per-file implementation notes. |
| `docs/evaluation/queries.json` | The 100 hand-crafted evaluation queries, stratified 20 per category, each with ground-truth strategy metadata. |
| `docs/example_queries/` | Saved JSON outputs from representative queries, as presented in thesis Appendix G. |
| `data/queries/` | JSON-structure end results for queries run in Appendix G. |

### Deployment infrastructure (root)

| File | Description |
|------|-------------|
| `Dockerfile` | Multi-stage build: `builder` stage installs Python dependencies into a virtualenv; `runtime` stage copies the venv and application code into a lean image (approximately 600 MB) running as non-root `appuser`. |
| `docker-compose.yml` | One-command local deployment. Mounts `data/` as read-only volume, injects `.env` credentials, exposes port 8000, configures health check on `/health` endpoint. |
| `.dockerignore` | Excludes virtual environments, git history, IDE state, and `data/` from the Docker build context. |
| `.env.example` | Template of required credentials with source URLs. Real credentials go in `.env` (git-ignored). |
| `requirements.txt` | Python dependencies pinned to specific versions: `fastapi`, `uvicorn`, `anthropic`, `sentence-transformers`, `faiss-cpu`, `spotipy`, `rapidfuzz`, `rank-bm25`, `scipy`, `pydantic`. |

### Data (present after cloning with Git LFS or downloading separately)

| File | Description |
|------|-------------|
| `data/raw/corpus.db` | Full 9,499-track SQLite corpus (96 MB). |
| `data/embeddings/corpus.faiss` | Pre-built FAISS index over the corpus (14 MB). |
| `data/evaluation/summary.json` | Aggregated metrics per baseline (data for Table I). |
| `data/evaluation/significance.json` | Paired t-test results (data for Table III). |
| `data/evaluation/results.json` | Per-query per-baseline detailed results. |
| `data/evaluation/results.csv` | Pandas-friendly flat file of results. |

---

## Running the service

### Docker

```bash
docker compose up --build       #build image and start service (first build 5-10 min)
docker compose up -d            #background mode after initial build
docker compose logs -f          #follow container logs
docker compose down             #stop and remove container
```

The lifespan warmup takes approximately 5 seconds, loading the sentence-transformer model and FAISS index into memory. The service prints `Warmup complete!` when ready to accept requests.

### Local Python (for development)

If you have Python 3.11 installed and want to bypass Docker:

```bash
#Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate       #Windows: .venv\Scripts\activate

#Install dependencies
pip install -r requirements.txt

#Start the service
uvicorn src.api.main:app --reload --port 8000
```

Warmup is the same as Docker (approximately 5 seconds). Server is available at http://localhost:8000/docs.

---

## Testing the endpoints

Interactive Swagger UI is served automatically at http://localhost:8000/docs . This is the easiest way to try the API without writing curl commands.

### Health check (fastest verification)

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "corpus_ready": true,
  "index_ready": true,
  "llm_ready": true
}
```

If `status` is `degraded`, the individual flags will show which component is unavailable.

### Corpus statistics

```bash
curl http://localhost:8000/stats
```

Expected response:
```json
{
  "total_tracks": 9506,
  "tracks_with_lyrics": 8574,
  "tracks_with_wiki": 5031,
  "unique_tags": 5905,
  "index_vector_count": 9499,
  "embedding_dim": 384
}
```

### Recommendation

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "melancholic music for a rainy walk", "n": 5}'
```

Response returns within approximately 3 seconds with five ranked recommendations, each including track_id, name, artist_name, justification, spotify_url, and spotify_confidence. See thesis Appendix F for full endpoint documentation and Appendix G for representative example responses.

---

## Reproducing the evaluation

The full 100-query evaluation reported in Section V of the thesis can be re-run against the pre-built corpus.

```bash
#Verify environment (should print 9506)
python -c "from src.corpus.db import get_conn; \
    print(get_conn().execute('SELECT COUNT(*) FROM tracks').fetchone()[0])"

#Smoke test: 10 random queries across all 5 baselines
python -m src.evaluation.run --sample 10

#Full 100-query evaluation
python -m src.evaluation.run

#Single-category evaluation
python -m src.evaluation.run --category genre
python -m src.evaluation.run --category compound
```

Results are written to `data/evaluation/`:
- `results.json` — per-query per-baseline detail
- `summary.json` — aggregated metrics by baseline and category (data for Table I and Table II)
- `significance.json` — paired t-tests (data for Table III)
- `results.csv` — pandas-friendly flat file

To run individual baselines interactively for a single query:

```bash
python -m src.evaluation.baselines "sad indie about heartbreak" -k 5 --baseline all
```

This prints side-by-side comparison of Random, Popularity, BM25, Dense-only and full SoundRAG output for a chosen query, useful for qualitative inspection.

---

## Rebuilding the corpus from scratch (advanced)

If you need to reproduce the corpus rather than use the pre-built one, the four ingestion phases run in sequence. Each is idempotent and safe to interrupt and resume. **Total wall-clock time is approximately 40 hours** due to API rate limits (Last.fm five-requests-per-second, Genius similar limits).

```bash
#Phase 1: Track collection
python -m src.corpus.seeds

#Phase 2: Track enrichment via Last.fm
python -m src.corpus.enrich

#Phase 3: Artist biography enrichment via Wikipedia 
python -m src.corpus.enrich_artists

#Phase 4: Lyrics ingestion via Genius
python -m src.corpus.enrich_lyrics

#Text synthesis
python -m src.embedding.synthesise

#FAISS index construction
python -m src.embedding.embed
```

---

## Running without deployment (offline evaluation of the system)

If Docker or API credentials are unavailable in your environment, evidence of the system's behaviour is provided in three forms without requiring you to run anything:

**1. The presentation video** (submitted alongside the thesis) demonstrates the full pipeline running end-to-end, including a live demo of the API and side-by-side comparison against all four baselines.

**2. Appendix G of the thesis** contains saved JSON outputs from five representative queries, one per evaluation category, captured directly from the running `/recommend` endpoint. These include the request payload, the full response body with all fields, and analytical commentary.

**3. Saved query outputs in `docs/example_queries/`** contain the raw JSON responses for the same five queries, allowing you to inspect the exact JSON structure returned by the API.

The evaluation results reported in Sections V.A through V.E of the thesis can be verified directly against `data/evaluation/summary.json` and `data/evaluation/significance.json` in this repository without needing to re-run the evaluation.

---

## Cross-references to thesis document

| Thesis section | Repository artefact |
|----------------|---------------------|
| Section III.A (Four-Layer Architecture) | Directory structure of `src/` mirrors the four layers |
| Section III.C (Data Collection) | `src/corpus/enrich*.py` scripts, one per phase |
| Section III.D (Text Synthesis) | `src/embedding/synthesise.py` |
| Section III.E (Embedding Generation) | `src/embedding/embed.py` |
| Section III.F (RAG Pipeline) | `src/generation/generate.py` and `src/retrieval/query.py` |
| Section III.G (Baseline Implementations) | `src/evaluation/baselines.py` |
| Section III.H (Deployment) | `Dockerfile`, `docker-compose.yml`, `src/api/` |
| Section IV (Evaluation Framework) | `src/evaluation/` and `docs/evaluation/queries.json` |
| Section V (Results) | `data/evaluation/summary.json`, `data/evaluation/significance.json` |
| Appendix B (Schema) | `src/corpus/db.py` and `src/corpus/schema_changes/` |
| Appendix C (Infrastructure) | `Dockerfile` and `docker-compose.yml` |
| Appendix D (Embedding Specs) | `src/embedding/embed.py` model constant |
| Appendix E (100 Queries) | `docs/evaluation/queries.json` |
| Appendix F (API Documentation) | `src/api/main.py` and `src/api/schemas.py` |
| Appendix G (Sample Outputs) | `docs/example_queries/*.json` |

---

## Requirements

**Deployment path (Docker):**
- Docker Desktop 4.0 or later (includes Docker Compose 2.0+)
- Approximately 1 GB free disk space
- Internet connection for API calls at inference time
- Anthropic API credit (approximately $5 covers full development and evaluation)

**Development path (local Python):**
- Python 3.11 (other versions untested)
- Approximately 500 MB free disk for Python dependencies
- Same internet and API credit requirements as above

**Key Python dependencies** (full list in `requirements.txt`):

| Package | Purpose |
|---------|---------|
| `fastapi`, `uvicorn` | REST service framework and ASGI server |
| `anthropic` | Claude API client for generation and judging |
| `sentence-transformers` | all-MiniLM-L6-v2 embedding model |
| `faiss-cpu` | Vector index |
| `spotipy` | Spotify Web API client |
| `rapidfuzz` | Fuzzy string matching for identity resolution |
| `rank-bm25` | Sparse retrieval baseline |
| `scipy` | Paired t-tests for statistical significance |
| `pydantic` | Schema validation |
| `pytest`, `pytest-cov`, `httpx` | Testing (dev only) |

