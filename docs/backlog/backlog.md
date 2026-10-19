# SoundRAG — Production Upgrade Backlog

Turning the thesis system into a production RAG. **Prefix:** `SR-`. One ticket per branch, one PR per ticket.

The organising principle: **measure first, then change, then measure again.** The thesis proved the idea works. This proves it keeps working, says what it costs, and tells you when it breaks.

---

## Before you start

### The parity principle

Several tickets here are migrations — different store, different index, different code path. **A migration must not change your numbers.** If NDCG moves when you swap SQLite+FAISS for pgvector, something is wrong, and finding out which thing is the entire value of the exercise.

So: record the baseline before touching anything, and check parity after every structural change.

### pgvector

Two index types. **HNSW** is a graph index: better recall at a given speed, slower to build, more memory. **IVFFlat** partitions the space into lists and searches a few: faster to build, needs tuning of `lists` and `probes`, and must be built after data is loaded. At your corpus size HNSW is the straightforward choice, but know why.

Both are *approximate*. Your current `IndexFlatIP` is exact. So a small recall drop after migration is expected and should be measured, not assumed away.

### Bi-encoders versus cross-encoders

Your current embedding model is a **bi-encoder**: it encodes the query and the document separately, so document vectors can be precomputed and the search is a nearest-neighbour lookup. Fast, scalable, and it never sees the query and document together.

A **cross-encoder** encodes the pair jointly and outputs a relevance score. Far more accurate, and it cannot be precomputed — you run it once per candidate. That's why it only works as a re-ranker over a shortlist, and why it sits naturally between your retrieval and your LLM.

`sentence-transformers` has a `CrossEncoder` class; the `ms-marco-MiniLM` family is the standard starting point.

---

## Sprint 0 — Measurement (nothing changes yet)

**SR-01 · Freeze and document the evaluation set**
Why: every number from here is judged against it, so it cannot move underneath you.
Build: move the 100-query stratified set into `data/eval/queries.jsonl` under version control, with the relevance judgments and the stratification scheme. Write `docs/evaluation-method.md` recording who made the judgments, when, and on what basis.
Done when: loadable, versioned, and the method document states plainly that judgments are single-annotator.
Teaches: a golden set is an artifact with provenance. That honesty is also what a reviewer or interviewer will ask about first.

**SR-02 · Metrics implemented from scratch**
Why: NDCG has several definitions and the one you used is part of your result.
Build: NDCG@k, P@k, MRR, recall@k. Implemented, not imported. State the gain function (linear or exponential) and the log base explicitly in the code.
Done when: unit tested against a hand-computed example, **and it reproduces your thesis figures — NDCG@10 0.875, P@5 0.458, MRR 0.517 — on the same data.**
Teaches: reproducing your own published numbers is the first real test of whether the pipeline is sound. If it doesn't reproduce, that is the most important bug you will find all month.

**SR-03 · Evaluation runner and recorded baseline**
Build: one command runs any configuration over the eval set and writes a timestamped JSON report — metrics, per-query results, latency, and cost.
Done when: `results/baseline-<date>.json` exists and the figures are in the README.
Teaches: reproducible measurement. Everything after this is a delta.

**SR-04 · Diagnose the NDCG–precision gap**
Why: NDCG@10 of 0.875 alongside P@5 of 0.458 is a real tension worth understanding, and it probably connects to the genre problem.
Build: per-query analysis. Which queries score high on NDCG and low on P@5? Is it generous graded relevance, position effects, or one stratum behaving differently from the rest?
Done when: `docs/analysis/ndcg-precision-gap.md` names a cause with concrete examples.
Teaches: reading your own metrics instead of reporting them. This is the single most undervalued skill in applied ML.

**SR-05 · Regression gate in CI**
Build: CI runs the evaluation on every change and fails if NDCG@10 or P@5 drops more than a stated tolerance below baseline.
Done when: a deliberately degraded retriever turns the build red.
Teaches: the thing that makes every later change safe to attempt.

---

## Sprint 1 — One store

**SR-06 · Postgres and pgvector schema**
Build: tracks, enrichment sources, embeddings with a model-version column, and relevance judgments. HNSW index on the embedding column. Alembic migration.
Done when: schema is in version control and a fresh database can be built from migrations alone.

**SR-07 · Migration from SQLite plus FAISS**
Build: a one-off migration script moving the corpus and re-embedding into pgvector. Keep the old path runnable behind a flag until SR-10 passes.
Done when: both paths run and return results for the same query.
Teaches: migrations are reversible until you prove the new path, not after.

**SR-08 · Embedding versioning**
Why: change the embedding model and every stored vector becomes meaningless. Production needs a plan that isn't "rebuild everything over a weekend."
Build: every embedding row carries its model name and version. Queries only compare against vectors from the matching model. A re-embed job can run alongside the live one.
Done when: two embedding versions coexist in the table and the retriever picks the right one.
Teaches: the operational problem nobody anticipates until it bites.

**SR-09 · Incremental ingestion**
Why: `IndexFlatIP` has to be rebuilt from scratch to add one track. That is not a system.
Build: add, update and delete a track without a rebuild. Enrichment from Last.fm, Genius and Wikipedia runs per track with retries and a quarantine for failures.
Done when: adding 50 tracks takes seconds and the evaluation still passes.
Teaches: a corpus that cannot grow is a snapshot, not a product.

**SR-10 · Parity check**
Build: run the full evaluation against pgvector and compare to the SR-03 baseline.
Done when: the difference is within the CI tolerance, **or** you have written down exactly why HNSW's approximation moved it and by how much.
Teaches: approximate indexes trade recall for speed, and you should be able to quantify your own trade rather than hope.

---

## Sprint 2 — The cascade

This sprint is the centrepiece. Your thesis found that LLM re-ranking is the decisive component; this asks what it costs and whether something cheaper recovers most of it.

**SR-11 · Cross-encoder re-ranker**
Build: `rerank/cross_encoder.py` scoring query–document pairs over the top 100 retrieved candidates. Batched. Model pinned.
Done when: it runs standalone and scores on the eval set.

**SR-12 · Cascade orchestration**
Build: retrieve 100 → cross-encoder to 10 → LLM re-ranks those 10. Every stage configurable and independently switchable.
Done when: all four configurations run from config alone — dense only, dense + cross-encoder, dense + LLM, dense + cross-encoder + LLM.
Teaches: making the thing you want to measure a parameter rather than a rewrite.

**SR-13 · Per-tier cost and latency instrumentation**
Build: record wall time and cost for each stage separately. Tokens and money for the LLM; milliseconds for the rest.
Done when: a single query's report shows where its time and money went.

**SR-14 · The cost–accuracy curve**
Build: run all configurations over the eval set. Plot NDCG@10 and P@5 against cost per query and p95 latency.
Done when: `docs/analysis/cost-accuracy.md` has the plot and the table.
Teaches: **this is the most valuable artifact in the whole project.** It is the exact tradeoff AI engineering interviews probe, it is a direct continuation of your own thesis finding, and essentially no portfolio project has one.

**SR-15 · Choose the production configuration, in writing**
Build: an ADR picking one configuration, naming what you gave up, and stating the condition under which you would revisit it.
Done when: someone could disagree with your choice but not with your reasoning.

**SR-16 · Semantic cache**
Build: exact-match cache first, then a similarity cache above a threshold, on both retrieval and re-ranking. Record hit rate.
Done when: the evaluation report shows cost per query with and without cache.
Teaches: where the money actually goes in a RAG system.

---

## Sprint 3 — The genre problem

**SR-17 · Diagnose the cause**
Why: you measured 65% of queries over-indexing on genre and patched it with guardrails. A patch is not a diagnosis.
Build: inspect the embedded documents. What proportion of each document's text is genre tags? Compare embedding similarity for genre-matched versus mood-matched pairs. Test the hypothesis that tag-dominated text produces genre-shaped vectors.
Done when: `docs/analysis/genre-over-indexing.md` states the cause with evidence.
Teaches: the diagnostic habit — that the fix follows from the cause, not from the symptom.

**SR-18 · Metadata filtering for genre**
Why: genre is a structured field. Filtering is exact and free; semantic matching on it is neither.
Build: genre as a filterable column, applied in the same pgvector query as the vector search. Query classification decides when to filter.
Done when: measured against baseline on the eval set, reported per stratum.

**SR-19 · Field-weighted document construction**
Build: build the embedded text with configurable weights per source — tags, lyrics, description, Wikipedia summary. Make the weighting a config value.
Done when: at least three weightings are evaluated and the results are in the analysis document.

**SR-20 · Choose and record**
Build: pick the approach that actually won. Remove the original guardrail logic if the real fix makes it redundant.
Done when: before-and-after numbers are in the README and the 65% figure has a successor.
Teaches: **"I diagnosed the cause of a measured failure mode and fixed it, and here is the before and after"** is one of the strongest sentences you can say in an interview.

---

## Sprint 4 — Production concerns

**SR-21 · Tracing**
Build: OpenTelemetry. One trace per request; spans for retrieve, cross-encode, LLM re-rank, generate. Attributes: query, retrieved IDs with scores, prompt version, tokens, latency, cost, cache hit.
Done when: given one recommendation you can reconstruct exactly why it was made.
Teaches: the answer to the first question anyone asks about a recommender.

**SR-22 · Graceful degradation**
Build: LLM unavailable → serve cross-encoder results. Cross-encoder unavailable → serve dense results. Response states which tier answered.
Done when: killing the LLM provider still returns recommendations, and the evaluation records the degraded score.
Teaches: you now have tiers, so you have something to fall back to. That is the payoff of SR-12.

**SR-23 · Token budget and circuit breaker**
Build: per-request token ceiling, and a breaker that opens after repeated LLM failures and routes to the degraded path.
Done when: a test drives the breaker open, through cooldown, and closed.

**SR-24 · Deploy**
Build: containerised, Postgres with pgvector managed, secrets out of code, health check that verifies the database and the index. Infrastructure in Terraform.
Done when: destroy and rebuild from nothing produces a working service.

**SR-25 · Load test**
Build: realistic query mix, concurrency, measure p50/p95/p99 and cost per thousand queries. Note what breaks first.
Done when: `docs/performance.md` records the numbers and the first bottleneck.

---

## Sprint 5 — The write-up

**SR-26 · Results document**
Build: `docs/results.md` — baseline, every change, what it did to accuracy, cost and latency. Including the changes that made things worse.
Done when: an honest reader can see what the production work bought and what it cost.

**SR-27 · README**
Build: architecture diagram, the cost-accuracy plot, the headline numbers, quickstart, and a stated limitations section.
Done when: a stranger could run it and know what it does and does not do.

**SR-28 · ADRs**
Build: `docs/adr/` covering at least — pgvector over FAISS, the cascade configuration, embedding versioning, the genre fix, metadata filtering over semantic matching, degradation strategy.
Done when: six decisions, five sentences each, alternatives named.

**SR-29 · Public write-up**
Build: a technical post on the cost-accuracy tradeoff in RAG re-ranking, using your numbers.
Done when: published.
Teaches: this is the highest-return hour in the backlog for your job search. Workshop papers get cited by three people; a good post with real numbers gets read by hiring managers.

---

## What this is worth when it's done

You can say, with evidence from your own system: here is what the expensive component actually buys you and what it costs; here is a measured failure mode and the fix that followed from diagnosing it; here is how I know a change didn't break anything; here is what happens when the model provider goes down; here is the cost per thousand queries.

That is a different project from a thesis, and it is the one the AI engineering postings are describing.