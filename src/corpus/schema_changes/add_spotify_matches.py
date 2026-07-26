"""Add the spotify_matches table.Caches resolved Spotify URIs per track so we don't re-hit the Spotify search
API for tracks we've already resolved"""
from ..db import get_conn


MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS spotify_matches (
    track_id           INTEGER PRIMARY KEY,
    spotify_track_id   TEXT,
    spotify_url        TEXT,
    confidence         REAL NOT NULL,
    resolved_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_spotify_matches_confidence
    ON spotify_matches(confidence);
"""


def run() -> None:
    """Apply the migration. Idempotent."""
    with get_conn() as conn:
        conn.executescript(MIGRATION_SQL)
    print("Applied: spotify_matches table created.")


if __name__ == "__main__":
    run()