"""add artist_tags junction table. Adds the many-to-many relationship between artists and tags."""
from ..db import get_conn


MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS artist_tags (
    artist_id INTEGER NOT NULL,
    tag_id    INTEGER NOT NULL,
    weight    INTEGER,
    PRIMARY KEY (artist_id, tag_id),
    FOREIGN KEY (artist_id) REFERENCES artists(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id)    REFERENCES tags(id)   ON DELETE CASCADE
);
"""


def run() -> None:
    """Apply the migration. Idempotent."""
    with get_conn() as conn:
        conn.executescript(MIGRATION_SQL)
    print("Changes applied: artist_tags table created.")


if __name__ == "__main__":
    run()