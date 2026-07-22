"""iterates through the corpus, synthesises each document, embeds it and writes the vectors to disk incrementally so the process is resumable."""
import json
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.corpus.db import get_conn

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  
BATCH_SIZE = 64

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "embeddings"
INDEX_PATH = INDEX_DIR / "corpus.faiss"
METADATA_PATH = INDEX_DIR / "corpus.faiss.json"


def load_model() -> SentenceTransformer:
    """Load the sentence-transformer."""
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    actual_dim = model.get_sentence_embedding_dimension()
    assert actual_dim == EMBEDDING_DIM, (
        f"Model produced {actual_dim}-dim vectors; expected {EMBEDDING_DIM}. "
        f"Update EMBEDDING_DIM constant if switching models."
    )
    return model


def fetch_documents(conn) -> tuple[list[int], list[str]]:
    """Return parallel lists of track_ids and document texts."""
    rows = conn.execute(
        "SELECT track_id, document_text FROM track_documents ORDER BY track_id"
    ).fetchall()
    track_ids = [r["track_id"] for r in rows]
    documents = [r["document_text"] for r in rows]
    return track_ids, documents


def embed_documents(
    model: SentenceTransformer, documents: list[str]
) -> np.ndarray:
    """Encode all documents in batches.Sentence-transformers handles batching and shows its own progress bar."""
    embeddings = model.encode(
        documents,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    # FAISS requires float32
    return embeddings.astype("float32")


def build_index(
    embeddings: np.ndarray, track_ids: list[int]
) -> faiss.Index:
    """Build an IndexFlatIP wrapped in IndexIDMap2.

    IndexFlatIP: exact inner-product search over all vectors. At 9,500 vectors,
    this is essentially instant (single matrix multiply) and gives guaranteed
    100% recall — the top-k is genuinely the k nearest.

    IndexIDMap2: wrapper that stores an external ID (our track_id) alongside
    each vector. Without it, FAISS returns 0-indexed positions and we'd have
    to maintain a separate position-to-id mapping. With it, retrieval returns
    track_ids directly.
    """
    base_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index = faiss.IndexIDMap2(base_index)
    ids_array = np.array(track_ids, dtype="int64")
    index.add_with_ids(embeddings, ids_array)
    return index


def save_index(index: faiss.Index, doc_count: int) -> None:
    """persist the index and its metadata sidecar to disk"""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    metadata = {
        "model_name": MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "doc_count": doc_count,
        "index_type": "IndexIDMap2(IndexFlatIP)",
        "normalize_embeddings": True,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))


def smoke_test(
    model: SentenceTransformer, index: faiss.Index, conn
) -> None:
    """Verify the index by embedding a known document and checking retrieval"""
    row = conn.execute(
        """
        SELECT td.track_id, td.document_text, t.name, t.artist_name
        FROM track_documents td
        JOIN tracks t ON t.id = td.track_id
        ORDER BY t.listener_count DESC
        LIMIT 1
        """
    ).fetchone()

    if not row:
        print("[warn] no documents to smoke-test")
        return

    query_id = row["track_id"]
    query_doc = row["document_text"]
    query_name = f"{row['name']} by {row['artist_name']}"

    query_vec = model.encode(
        [query_doc],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, ids = index.search(query_vec, k=5)

    print(f"\nSmoke test — query track: {query_name} (id={query_id})")
    print("Top-5 nearest neighbours:")
    for rank, (score, hit_id) in enumerate(zip(scores[0], ids[0]), start=1):
        hit_row = conn.execute(
            "SELECT name, artist_name FROM tracks WHERE id = ?",
            (int(hit_id),),
        ).fetchone()
        if hit_row:
            marker = "  <-- query track" if int(hit_id) == query_id else ""
            print(
                f"  {rank}. [score={score:.4f}] "
                f"{hit_row['name']} by {hit_row['artist_name']}{marker}"
            )

    if int(ids[0][0]) == query_id:
        print("sanity check passed")
    else:
        print(
            " top hit is NOT the query track"
            "check normalization or index construction"
        )


def report_status() -> None:
    """Print index metadata and file size."""
    if not INDEX_PATH.exists():
        print("[warn] index file not found")
        return

    size_mb = INDEX_PATH.stat().st_size / (1024 * 1024)
    metadata = json.loads(METADATA_PATH.read_text())

    print(f"\nIndex status:")
    print(f"  Path:         {INDEX_PATH}")
    print(f"  Size on disk: {size_mb:.2f} MB")
    print(f"  Model:        {metadata['model_name']}")
    print(f"  Dimensions:   {metadata['embedding_dim']}")
    print(f"  Vectors:      {metadata['doc_count']}")
    print(f"  Index type:   {metadata['index_type']}")
    print(f"  Built:        {metadata['built_at']}")


def main():
    model = load_model()

    with get_conn() as conn:
        print("Fetching documents from corpus...")
        track_ids, documents = fetch_documents(conn)

        if not documents:
            print("no documents found.")
            return

        print(f"Loaded {len(documents)} documents")

        print("Embedding documents...")
        start = time.time()
        embeddings = embed_documents(model, documents)
        elapsed = time.time() - start
        rate = len(embeddings) / elapsed if elapsed else 0
        print(
            f"Embedded {len(embeddings)} documents in {elapsed:.1f}s "
            f"({rate:.1f} docs/sec)"
        )

        print("Building FAISS index...")
        index = build_index(embeddings, track_ids)
        save_index(index, doc_count=len(documents))

        smoke_test(model, index, conn)

    report_status()


if __name__ == "__main__":
    main()