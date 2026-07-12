# Music RAG Recommender

**MSc Computer Science Thesis — Queen Mary University of London**

Addressing the Cold-Start Problem in Spotify Music Recommendation Systems Using Retrieval-Augmented Generation and Natural Language Processing.

Traditional music recommendation systems inherit a user-item vector-space design from collaborative filtering, in which each user and each item are represented by a vector learned from historical interaction data. This design produces a first-class engineering problem known as ***cold-start***: recommendations fail for new users and for new items. Common mitigations, such as popularity fallback, demographic feeling, onboarding questionnaires and content-based feature extraction from audio, each address a subset of the probelm at a cost of degraded personalisation, additional friction or dependence on data sources that are increasingly restricted. 

## System overview

SoundRAG comprises four layers, corresponding to the data flow from ingestion through recommendation delivery.

**Layer 1** — Knowledge. A structured corpus of music metadata, tags, artist information, and lyrics, sourced from Last.fm and Genius and stored in a normalised SQLite database. Built offline via an idempotent, resumable ingestion pipeline.

**Layer 2** — Retrieval. A dense vector index (FAISS) over textual descriptions synthesised from the Layer 1 corpus. At query time, the user's natural-language input is embedded and used to retrieve the top-k most similar item descriptions.

**Layer 3** — Generation. A large language model (Claude or GPT-4) receives the user query and the retrieved descriptions as context, and generates a ranked list of recommendations with grounded justifications drawn from the retrieved text.

**Layer 4** — Identity resolution. Recommended track names and artists are resolved to Spotify identifiers via the (still-permitted) Spotify search endpoint, producing playable URLs for the end user.

## Architecture Overview
The deployed system is a RAG-enhanced recommendation engine that accepts natural language queried from cold-start users and returns ranked, explainable track recommendations. It operates as a containerised FastAPI service backed by a FAISS vector store and an LLM orchastrated via LangChain.

## Layer 1: Knowledge
**Purpose:** produces the corpus of textual descriptions that the retrival and generation layers depend on. Every track available to the recommender must appear in the corpus with sufficient descriptive text to be reasoned about.

## Data sources

-> Last.fm — track metadata, crowdsourced tags with weights, listener statistics, wiki summaries, artist biographies, and pairwise track similarity from collaborative filtering signals.
-> Genius — song lyrics (retrieved via HTML scraping through the lyricsgenius library, since Genius's public API does not expose raw lyrics).
-> MusicBrainz — canonical identifiers (MBIDs) surfaced through Last.fm where available.

## Storage design
The corpus is stored in a normalised SQLite database at `data/raw/corpus.db`. SQLite was chosen over a client-server RDBMS for its zero-configuration deployment, portability, and adequacy at the target corpus scale (approximately 10,000 tracks). The schema comprises six tables:

- `tracks` — one row per unique `(artist_name, name)` pair; holds metadata and enrichment fields (nullable until Phase 2 populates them). Enrichment progress is tracked via nullable timestamp columns (`enriched_at`, `lyrics_processed_at`), which double as boolean flags without requiring a separate state table.
- `tags` — normalised tag names, case-insensitive via `COLLATE NOCASE`.
- `track_tags` — many-to-many junction between tracks and tags, with a weight column recording Last.fm's tag popularity (0–100).
- `artists` — separate table for per-artist enrichment (biographies, listener counts), populated in Phase 3 and soft-linked to `tracks.artist_name` by string rather than foreign key, due to inconsistencies in Last.fm's canonical artist naming.
- `similar_tracks` — Last.fm's collaborative filtering signal, stored with target track name and artist as strings (rather than foreign keys) because target tracks may not be present in the corpus.
- `lyrics` — one-to-optional-one with `tracks`; contains lyrics text, source URL, and retrieval timestamp.

Foreign key constraints with `ON DELETE CASCADE` maintain referential integrity across the junction and dependent tables. The dedup key on `tracks` is `UNIQUE(artist_name, name)` rather than `mbid`, because MBIDs are absent for approximately 50% of Last.fm tracks and a UNIQUE constraint on nullable columns does not enforce uniqueness for null values in SQL.

### Ingestion pipeline

Ingestion is structured as four sequential phases, each independently idempotent and resumable:

1. **Seed collection** — populate `tracks` with names and artists from a curated set of Last.fm tag pages and the global chart. Enrichment fields remain null.
2. **Track enrichment** — for each seeded track, call `track.getInfo` and `track.getSimilar` on Last.fm; populate scalar enrichment fields, insert tag relationships, insert similar-track edges. `enriched_at` is stamped only when both API calls succeed.
3. **Artist enrichment** — for each distinct artist in `tracks`, call `artist.getInfo`; populate `artists`.
4. **Lyrics enrichment** — for each enriched track, query Genius for lyrics; insert into `lyrics`.

Resumability is achieved through a two-part invariant: (1) every insert uses `INSERT OR IGNORE` or `INSERT OR REPLACE` to avoid duplicate-key crashes on re-runs, and (2) enrichment timestamps are stamped only within the same transaction as the corresponding data writes. If a phase is interrupted mid-run, subsequent invocations query for rows with null timestamps and process only those.

### Rate limiting and error handling

Last.fm's stated rate limit is 5 requests per second. The pipeline maintains a 0.25-second polite delay between API calls, corresponding to 4 requests per second — sufficient headroom to avoid throttling under normal network conditions.

Transient failures trigger exponential backoff with three retries at 5s, 15s, and 45s intervals. Permanent failures (HTTP 404, invalid parameters, track-not-found) are recognised via error message classification and skipped immediately without retrying. This distinction is implemented via a decorator (`@retry`) applied to individual API-calling functions.

### Module structure

The Layer 1 codebase lives under `src/corpus/` and comprises the following modules:

**`db.py`** — Database schema definition and connection management. Exports:
- `SCHEMA`: multi-statement SQL string defining all six tables and their indexes.
- `init_db()`: idempotent schema initialisation, safe to invoke on an existing database.
- `get_conn()`: context manager providing a per-transaction SQLite connection with foreign keys enabled and dict-like row access.

The schema is embedded as a string rather than an ORM to preserve legibility and simplify version control. Path resolution uses `pathlib` with `__file__`-relative computation, decoupling the database location from the caller's working directory.

**`seeds.py`** — Phase 1 seed collection. Defines a curated list of 55 tags spanning genres, subgenres, moods, and eras; iterates through them via Last.fm's `tag.getTopTracks`; augments with the global chart via `chart.getTopTracks`. Uses `INSERT OR IGNORE` to make re-runs idempotent. Fresh database connections are opened per tag to minimise lock duration and to ensure a mid-tag crash preserves earlier tags' inserts.

**`schemas.py`** — Typed data containers for parsed API responses. Uses `@dataclass` to define `TagInfo`, `SimilarTrackInfo`, and `TrackEnrichment` without boilerplate. Provides a normalised interface between the API layer and the database writer, avoiding direct dependence on the pylast object model in downstream code.

**`lastfm_client.py`** — Last.fm API access with retry logic and response parsing. Defines the `@retry_on_transient` decorator, which wraps API-calling functions with exponential backoff for transient errors and immediate re-raise for permanent errors. Exports `fetch_track_info` and `fetch_similar_tracks` (decorated), and `parse_enrichment` for converting raw pylast objects into `TrackEnrichment` instances.

**`enrich.py`** — Phase 2 batch orchestration. Queries the database for tracks with null `enriched_at`, iterates through them with a progress bar, and invokes the per-track worker `enrich_one_track` for each. The worker fetches, parses, and writes in a single database transaction, stamping `enriched_at` only on full success. Failures are logged and the track is skipped, remaining unenriched for future runs.

**`__init__.py`** — Marks `src/corpus/` as a Python package; carries a package-level docstring.

### Module dependency structure

```
enrich.py       ─── depends on ──▶  lastfm_client.py, schemas.py, db.py
seeds.py        ─── depends on ──▶  db.py
lastfm_client.py─── depends on ──▶  schemas.py
schemas.py      ─── depends on ──▶  (standard library only)
db.py           ─── depends on ──▶  (standard library only)
```

Dependencies flow in one direction, from orchestration modules toward foundational modules. `db.py` and `schemas.py` have no internal dependencies and are consumed by all other modules. This structure permits testing the API and database layers independently of one another.

## Layer 2: Retrieval
### Purpose

Layer 2 converts the descriptive text produced in Layer 1 into a dense vector representation, and provides fast approximate-nearest-neighbour retrieval over that representation for arbitrary text queries.

### Planned design

Each track will be represented by a single document synthesised from its Layer 1 enrichment fields: tags (concatenated and weighted by frequency), artist biography, wiki summary, and a lyrics excerpt. This document will be embedded via a sentence transformer (candidate models: `sentence-transformers/all-MiniLM-L6-v2` for baseline speed, `BAAI/bge-large-en` for higher retrieval quality) into a fixed-dimensional vector.

Vectors will be stored in a FAISS index, keyed by the track's SQLite `id`. FAISS was selected for its production-grade performance, native support for approximate search (necessary at corpus sizes beyond ~100k), and mature Python bindings. The index will use an inverted-file structure with product quantisation (`IndexIVFPQ`) to balance memory footprint against retrieval accuracy at the target corpus scale.

At query time, the pipeline will embed the user query with the same encoder, query the FAISS index for the top-k nearest neighbours, and issue a batch `SELECT` against SQLite using the returned track IDs to materialise the full descriptive text for each hit.

Hybrid retrieval combining dense (FAISS) and sparse (BM25 via `rank_bm25`) signals will be evaluated as an extension, with reciprocal rank fusion for signal combination.

### Planned modules

- `src/retrieval/embed.py` — offline embedding of the Layer 1 corpus.
- `src/retrieval/index.py` — FAISS index construction and persistence.
- `src/retrieval/query.py` — query-time embedding and nearest-neighbour retrieval.

## Layer 3: Generation
### Purpose

Layer 3 converts a set of retrieved track descriptions into a ranked, justified recommendation. The LLM's role is not to generate recommendations from parametric knowledge — which would risk hallucinated track names and unverifiable claims — but to reason over the passages provided by Layer 2 and select the most appropriate subset for the user's query.

### Planned design

The prompt structure will follow standard grounded-generation practice: a system message defining the recommender's role and constraints (e.g. "only recommend tracks present in the provided context"), the user's query, and the retrieved passages formatted with track identifiers. The model will be instructed to return structured output (JSON) listing recommended track IDs and per-track natural-language justifications derived from the passages.

Candidate models include Claude 3.5 Sonnet, GPT-4o, and (for local-deployment evaluation) Llama 3.1 8B via Ollama. Selection will be driven by retrieval-recommendation alignment metrics and cost per query.

Guard rails against hallucination will include structured output validation (recommended track IDs must exist in the retrieved set) and a fallback path that returns the top-k retrieved tracks unmodified if the LLM's output cannot be validated.

### Planned modules

- `src/generation/prompt.py` — prompt templates and structured output schemas.
- `src/generation/llm.py` — model invocation with retry and validation logic.

---

## Layer 4: Identity resolution

**Status:** Planned.

### Purpose

Layer 4 converts the LLM's recommended track and artist names into Spotify track identifiers, producing URLs that the end user can click to play the recommendation.

### Planned design

Each recommended track will be queried against Spotify's `sp.search()` endpoint with a query string composed from the track name and primary artist name. The top result's URI will be treated as the resolution match, with a confidence threshold based on string similarity to the LLM's output to guard against incorrect matches.

Resolution failures (no Spotify hit, or hit below confidence threshold) will be surfaced in the recommendation output as text-only results, without a playable link.

### Planned modules

- `src/resolution/spotify.py` — Spotify search and URL construction.

---

## API surface

**Status:** Planned.

The complete system will be exposed as a FastAPI application with a single primary endpoint:

- `POST /recommend` — accepts a user query and returns ranked recommendations with justifications and Spotify URLs.

Additional endpoints for corpus statistics, retrieval-only queries, and health checks will support evaluation and monitoring.

Deployment will target Docker (`Dockerfile` and `docker-compose.yml` at repository root) with Kubernetes manifests under `k8s/` for later cloud deployment if required.

---

# Appendix: Implementation Log

The following notes are not part of the formal architecture and are retained as a personal learning record for later review and thesis reflection.

## `db.py`

The schema string was written to be declarative and self-documenting, so that the SQL itself functions as documentation of the data model. Embedding SQL as a Python string rather than using an ORM (SQLAlchemy, etc.) was a deliberate choice to keep the codebase minimal at thesis scale and to preserve one-to-one correspondence between the schema code and the actual database structure.

Points internalised while writing this module:

- SQLite dynamic typing: `TEXT`, `INTEGER`, and `TIMESTAMP` are storage hints, not enforced types. This differs from PostgreSQL and MySQL and requires care when relying on type coercion.
- Foreign keys are disabled by default in SQLite for historical compatibility. The `PRAGMA foreign_keys = ON` statement must be issued per connection, and it is easy to omit — silently allowing orphan rows.
- `INTEGER PRIMARY KEY AUTOINCREMENT` prevents ID reuse across delete operations, which matters for a corpus that may be pruned and re-ingested.
- Composite `UNIQUE` constraints are declared at the table level, not the column level, and treat null values as non-colliding — hence the choice of `(artist_name, name)` over `mbid` as the dedup key.
- The `@contextmanager` decorator from `contextlib` enables clean transactional semantics with substantially less boilerplate than the equivalent class-based `__enter__`/`__exit__` implementation.
- `pathlib.Path` operations (`.parent`, `/`, `.resolve()`) are the modern replacement for `os.path` string manipulation and produce more legible code.

## `seeds.py`

The tag list was hand-curated rather than fetched from Last.fm's `network.get_top_tags()` in order to guarantee reproducibility: a re-run of the ingestion at a future date will produce a corpus drawn from the same tag distribution, which is a prerequisite for reproducible thesis results.

Points internalised while writing this module:

- Rate limiting can be adequately handled with a fixed `time.sleep()` between calls; adaptive backoff is unnecessary for well-behaved APIs at thesis-scale request volumes.
- Fresh database connections per outer-loop iteration are preferred over a single long-lived connection: they minimise the duration of write locks on the SQLite file and localise the blast radius of a mid-loop crash.
- `INSERT OR IGNORE` combined with a `UNIQUE` schema constraint gives idempotent inserts with no application-level dedup logic.
- Named constants at module scope (`TRACKS_PER_TAG`, `POLITE_DELAY_SEC`) improve legibility over inline magic numbers and simplify tuning.
- The `if __name__ == "__main__":` idiom allows a single file to function as both an importable library and a runnable script.

## `schemas.py`

Dataclasses provide a lightweight way to give parsed API responses a stable typed shape without invoking the full class-boilerplate mechanism. This decouples the database write path from the pylast object model, which uses lazy attribute access and can trigger additional network calls if raw pylast objects are held past the initial fetch.

Points internalised while writing this module:

- `@dataclass` auto-generates `__init__`, `__repr__`, and `__eq__` methods from field declarations.
- Mutable default values must be created via `field(default_factory=list)` rather than `= []`, to avoid the classic Python foot-gun of shared mutable state across instances.
- Optional fields are declared as `T | None` (Python 3.10+ union syntax) with default `None`, making it explicit at the type level which fields may be absent.

## `lastfm_client.py`

The retry decorator was written to separate cross-cutting error handling from business logic. Every API-facing function in the module can opt into retry semantics with a single `@retry_on_transient` line, and the retry policy itself is defined once.

Points internalised while writing this module:

- Decorators are functions that take a function and return a modified version of it. The `@decorator` syntax is syntactic sugar for `f = decorator(f)`.
- Closures allow the inner `wrapper` function to reference the outer `fn` argument after the outer function has returned. This is fundamental to how decorators work.
- `functools.wraps(fn)` copies name and docstring metadata from the wrapped function to the wrapper. Without it, tracebacks and introspection tools report `wrapper` for every decorated function, which severely degrades debuggability.
- Distinguishing transient from permanent errors requires domain-specific classification. In Last.fm's case, `pylast.WSError.details` contains parseable error strings ("track not found", etc.) that reliably indicate permanence.
- `Callable[..., T]` type hints (with `T = TypeVar("T")`) allow decorators to preserve the wrapped function's return type in the type checker's view.

## `enrich.py`

The per-track worker (`enrich_one_track`) and the batch orchestrator (`main`) are deliberately separated. The worker takes a single track and returns success or failure; it knows about Last.fm and the database. The orchestrator knows about progress bars, rate limiting, and loop iteration; it does not know anything about Last.fm's error taxonomy. This separation enables independent testing and reduces the surface area of each function.

Points internalised while writing this module:

- The `enriched_at` timestamp is stamped inside the same transaction as the data writes. This is the mechanism that enforces the coarse-resumability invariant: a partial write followed by a crash leaves the timestamp null, and the next run re-processes the track cleanly.
- `INSERT OR REPLACE` on `track_tags` and `similar_tracks` allows a re-run to refresh tag weights and similarity scores if Last.fm's data has updated, without accumulating stale-plus-new duplicates.
- `COALESCE(?, mbid)` in the `UPDATE` statement preserves an existing MBID against a null overwrite from a re-run, protecting good data against a bad refetch.
- `tqdm`'s `set_postfix` method enables live counters on the progress bar without breaking its formatting, which is useful for surfacing skip counts during a long run.
- Structuring the failure path as `(success: bool, error: str | None)` rather than raising exceptions from the worker keeps the orchestrator's control flow linear and avoids the need for try/except around every worker call.