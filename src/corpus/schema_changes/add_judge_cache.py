"""Add judge_cache table to cache LLM judgements """
from ..db import get_conn

MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS judge_cache(
  query_id TEXT NOT NULL,
  track_id INTEGER NOT NULL,
  score INTEGER NOT NULL CHECK (score IN (0, 1, 2)),
  reasoning TEXT,
  judged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  model TEXT NOT NULL,
  PRIMARY KEY (query_id, track_id, model),
  FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_judge_cache_query
  ON judge_cache(query_id);
"""
def run() -> None:
    """apply the migration"""
    with get_conn() as conn:
        conn.executescript(MIGRATION_SQL)
    print("Applied: judge_cache table created.")

if __name__ == "__main__":
    run()