"""synthesise per-track text documents for embedding."""
import argparse
from pathlib import Path
import re

from tqdm import tqdm

from src.corpus.db import get_conn

# document layout constants
TARGET_DOC_LENGTH = 1000 
TAG_COUNT = 5                   
WIKI_EXCERPT_LENGTH = 200 
BIO_EXCERPT_LENGTH = 200        
LYRICS_EXCERPT_LENGTH = 500     

HARD_CAP_MULTIPLIER = 1.2

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
LASTFM_MARKERS = (
    "user-contributed text is available",
    "read more on last.fm",
    "creative commons",
)

DISAMBIGUATION_MARKERS = (
    "there is more than one artist with this name",
    "there are multiple artists",
    "there are several artists",
)


def truncate_at_boundary(text: str, max_len: int) -> str:
    """Truncate to max_len, preferring sentence boundaries then word boundaries."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text

    search_start = int(max_len * 0.8)
    truncated = text[:max_len]

    # look for the latest sentence-ending punctuation in the tail
    for i in range(len(truncated) - 1, search_start, -1):
        if truncated[i] in ".!?":
            return truncated[: i + 1]

    if " " in truncated:
        return truncated.rsplit(" ", 1)[0] + "..."
    return truncated + "..."


def flatten_lyrics(text: str) -> str:
    """Collapse newlines in lyrics into ' / ' separators for compactness."""
    return text.replace("\n\n", " / ").replace("\n", " / ").strip()


def clean_wiki_text(text: str) -> str:
    """Strip HTML tags and last.fm's licence boilerplate from wiki content."""
    if not text:
        return ""
    # strip html tags
    cleaned = HTML_TAG_PATTERN.sub("", text)
    # cut off at any boilerplate marker
    lower = cleaned.lower()
    earliest_cut = len(cleaned)
    for marker in LASTFM_MARKERS:
        idx = lower.find(marker)
        if idx != -1 and idx < earliest_cut:
            earliest_cut = idx
    cleaned = cleaned[:earliest_cut]
    return cleaned.strip()


def is_disambiguation_bio(text: str) -> bool:
    """Detect artist bios that are actually disambiguation pages."""
    if not text:
        return False
    lower = text.lower()
    return any(marker in lower for marker in DISAMBIGUATION_MARKERS)

def synthesise_document(track_data: dict) -> str:
    """Build the document string from a track's assembled enrichment data."""
    lines: list[str] = []

    lines.append(f"Title: {track_data['name']} by {track_data['artist_name']}")

    if track_data.get("album_name"):
        lines.append(f"Album: {track_data['album_name']}")

    tags = track_data.get("tags", [])
    if tags:
        lines.append(f"Tags: {', '.join(tags[:TAG_COUNT])}")

    if track_data.get("wiki_content"):
        cleaned = clean_wiki_text(track_data["wiki_content"])
        excerpt = truncate_at_boundary(cleaned, WIKI_EXCERPT_LENGTH)
        if excerpt:
            lines.append(f"About the track: {excerpt}")

    if track_data.get("artist_bio") and not is_disambiguation_bio(track_data["artist_bio"]):
        cleaned = clean_wiki_text(track_data["artist_bio"]) 
        excerpt = truncate_at_boundary(cleaned, BIO_EXCERPT_LENGTH)
        if excerpt:
            lines.append(f"About the artist: {excerpt}")

    if track_data.get("lyrics_text"):
        flat = flatten_lyrics(track_data["lyrics_text"])
        excerpt = truncate_at_boundary(flat, LYRICS_EXCERPT_LENGTH)
        if excerpt:
            lines.append(f"Lyrics: {excerpt}")

    document = "\n".join(lines)

    hard_cap = int(TARGET_DOC_LENGTH * HARD_CAP_MULTIPLIER)
    if len(document) > hard_cap:
        document = truncate_at_boundary(document, hard_cap)

    return document


def fetch_track_data(conn, track_id: int) -> dict | None:
    """Assemble all enrichment data for one track via three queries."""
    row = conn.execute(
        """
        SELECT t.id, t.name, t.artist_name, t.album_name, t.wiki_content,
               a.bio_content AS artist_bio,
               l.lyrics_text
        FROM tracks t
        LEFT JOIN artists a ON a.name = t.artist_name
        LEFT JOIN lyrics l ON l.track_id = t.id
        WHERE t.id = ?
        """,
        (track_id,),
    ).fetchone()

    if not row:
        return None

    tag_rows = conn.execute(
        """
        SELECT tags.name
        FROM tags
        JOIN track_tags tt ON tt.tag_id = tags.id
        WHERE tt.track_id = ?
        ORDER BY tt.weight DESC
        LIMIT ?
        """,
        (track_id, TAG_COUNT),
    ).fetchall()

    return {
        "id": row["id"],
        "name": row["name"],
        "artist_name": row["artist_name"],
        "album_name": row["album_name"],
        "wiki_content": row["wiki_content"],
        "artist_bio": row["artist_bio"],
        "lyrics_text": row["lyrics_text"],
        "tags": [t["name"] for t in tag_rows],
    }


def fetch_track_ids(conn, replace: bool = False) -> list[int]:
    """Return the list of track IDs to process."""
    if replace:
        rows = conn.execute(
            "SELECT id FROM tracks WHERE enriched_at IS NOT NULL ORDER BY id"
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT t.id
            FROM tracks t
            LEFT JOIN track_documents d ON d.track_id = t.id
            WHERE t.enriched_at IS NOT NULL
              AND d.track_id IS NULL
            ORDER BY t.id
            """
        ).fetchall()
    return [r["id"] for r in rows]


def fetch_preview_ids(conn, count: int = 10) -> list[int]:
    """Return a diverse sample of enriched track IDs for preview mode."""
    rows = conn.execute(
        """
        SELECT id FROM tracks
        WHERE enriched_at IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (count,),
    ).fetchall()
    return [r["id"] for r in rows]


def insert_document(conn, track_id: int, document: str) -> None:
    """Insert or replace the document row for one track."""
    conn.execute(
        """
        INSERT OR REPLACE INTO track_documents
            (track_id, document_text, document_length, synthesised_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (track_id, document, len(document)),
    )


def report_progress() -> None:
    """Summarise document synthesis status."""
    with get_conn() as conn:
        stats = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM tracks WHERE enriched_at IS NOT NULL) AS enriched,
                COUNT(*) AS synthesised,
                ROUND(AVG(document_length), 0) AS avg_length,
                MIN(document_length) AS min_length,
                MAX(document_length) AS max_length
            FROM track_documents
            """
        ).fetchone()

    enriched = stats["enriched"] or 0
    synth = stats["synthesised"] or 0
    pct = 100 * synth / enriched if enriched else 0

    print("\nDocument synthesis status:")
    print(f"  Enriched tracks:      {enriched}")
    print(f"  Documents synthesised: {synth} ({pct:.1f}%)")
    if synth > 0:
        print(f"  Avg document length:  {stats['avg_length']:.0f} chars")
        print(f"  Min document length:  {stats['min_length']} chars")
        print(f"  Max document length:  {stats['max_length']} chars")


def run_preview(count: int = 10) -> None:
    """Print sample synthesised documents without writing to db."""
    with get_conn() as conn:
        ids = fetch_preview_ids(conn, count=count)
        if not ids:
            print("No enriched tracks found for preview.")
            return

        print(f"Preview mode: {count} random tracks, no db writes\n")
        for track_id in ids:
            data = fetch_track_data(conn, track_id)
            if not data:
                continue
            doc = synthesise_document(data)
            print("-" * 70)
            print(f"track_id={data['id']}  ({data['name']} by {data['artist_name']})")
            print("-" * 70)
            print(doc)
            print(f"\n[document length: {len(doc)} chars]\n")


def run_batch(replace: bool = False) -> None:
    """Run synthesis on all applicable tracks."""
    with get_conn() as conn:
        track_ids = fetch_track_ids(conn, replace=replace)

    if not track_ids:
        print("All enriched tracks already have documents. Nothing to do.")
        report_progress()
        return

    action = "Regenerating" if replace else "Synthesising"
    print(f"{action} documents for {len(track_ids)} tracks")

    with get_conn() as conn:
        for track_id in tqdm(track_ids, desc="synth"):
            data = fetch_track_data(conn, track_id)
            if not data:
                continue
            doc = synthesise_document(data)
            insert_document(conn, track_id, doc)

    print(f"\nFinished. Synthesised {len(track_ids)} documents.")
    report_progress()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print 10 sample documents without writing to the db",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Regenerate documents for all enriched tracks",
    )
    args = parser.parse_args()

    if args.preview:
        run_preview()
    else:
        run_batch(replace=args.replace)


if __name__ == "__main__":
    main()