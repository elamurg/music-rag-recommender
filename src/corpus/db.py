import sqlite3
from pathlib import Path
from contextlib import contextmanager

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT/ "data" / "raw" / "corpus.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mbid TEXT,
    name TEXT NOT NULL,
    artist_name TEXT NOT NULL, 
    listener_count INTEGER,
    playcount INTEGER,
    wiki_summary TEXT,
    wiki_content TEXT, 
    album_name TEXT,
    lastfm_url TEXT,
    -- for tracking purposes
    seed_source TEXT NOT NULL,
    enriched_at TIMESTAMP,
    lyrics_processed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- dedup key
    UNIQUE(artist_name, name)
);
CREATE INDEX IF NOT EXISTS idx_tracks_enriched ON tracks(enriched_at);
CREATE INDEX IF NOT EXISTS idx_tracks_lyrics ON tracks(lyrics_processed_at);
CREATE INDEX IF NOT EXISTS idx_tracks_artist ON tracks(artist_name);
CREATE INDEX IF NOT EXISTS idx_tracks_mbid ON tracks(mbid);

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL COLLATE NOCASE
);

CREATE TABLE IF NOT EXISTS track_tags (
    track_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    weight INTEGER,
    PRIMARY KEY (track_id, tag_id),
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS artists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    mbid TEXT,
    bio_summary TEXT,
    bio_content TEXT,
    listener_count INTEGER,
    playcount INTEGER, 
    enriched_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS similar_tracks (
    source_track_id INTEGER NOT NULL,
    target_name TEXT NOT NULL,
    target_artist TEXT NOT NULL,
    similarity REAL NOT NULL,
    PRIMARY KEY (source_track_id, target_name, target_artist),
    FOREIGN KEY (source_track_id) REFERENCES tracks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS lyrics (
    track_id INTEGER PRIMARY KEY,
    lyrics_text TEXT,
    genius_url TEXT,
    genius_id INTEGER,
    retrieved_at TIMESTAMP,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
);
"""

def init_db(db_path=DB_PATH) -> None:
    db_path.parent.mkdir(parents= True, exist_ok = True)
    connection = sqlite3.connect(db_path)
    connection.executescript(SCHEMA)
    connection.commit()
    connection.close()

@contextmanager
def get_conn(db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

#only run this when invoked directly, not when imported as a module
if __name__ == '__main__':
    init_db()
    print(f"Schema initialised at {DB_PATH}")