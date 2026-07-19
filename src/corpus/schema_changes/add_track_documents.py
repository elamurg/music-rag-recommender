"""Add the track_documents table. Stores the synthesised text document per track that will be fed to the
embedding model"""
from ..db import get_conn


MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS track_documents (
    track_id         INTEGER PRIMARY KEY,
    document_text    TEXT NOT NULL,
    document_length  INTEGER NOT NULL,
    synthesised_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_track_documents_length
    ON track_documents(document_length);
"""


def run() -> None:
    """Apply the migration. Idempotent."""
    with get_conn() as conn:
        conn.executescript(MIGRATION_SQL)
    print("Applied: track_documents table created.")


if __name__ == "__main__":
    run()