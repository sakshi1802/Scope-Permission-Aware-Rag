import sqlite3
from dataclasses import dataclass

import numpy as np

from app.ingest import Chunk
from app.rbac import validate_role

SCHEMA = """
DROP TABLE IF EXISTS chunk_roles;
DROP TABLE IF EXISTS chunks;
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    doc_title TEXT NOT NULL,
    para INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE TABLE chunk_roles (
    chunk_id TEXT NOT NULL REFERENCES chunks(id),
    role TEXT NOT NULL,
    PRIMARY KEY (chunk_id, role)
);
"""


@dataclass(frozen=True)
class Hit:
    id: str
    doc_title: str
    para: int
    text: str
    score: float


class ChunkStore:
    def __init__(self, embedder, db_path: str = ":memory:"):
        self.embedder = embedder
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.executescript(SCHEMA)

    def index(self, chunks: list[Chunk]) -> None:
        vectors = self.embedder.documents([f"{c.doc_title}\n{c.text}" for c in chunks])
        with self.db:
            for chunk, vector in zip(chunks, vectors):
                self.db.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                    (chunk.id, chunk.doc_id, chunk.doc_title, chunk.para, chunk.text,
                     np.asarray(vector, dtype=np.float32).tobytes()),
                )
                self.db.executemany(
                    "INSERT INTO chunk_roles VALUES (?, ?)",
                    [(chunk.id, role) for role in chunk.allowed_roles],
                )

    def docs_for_role(self, role: str) -> list[dict]:
        validate_role(role)
        rows = self.db.execute(
            """SELECT DISTINCT c.doc_id, c.doc_title
               FROM chunks c JOIN chunk_roles r ON r.chunk_id = c.id
               WHERE r.role = ?
               ORDER BY c.doc_title""",
            (role,),
        ).fetchall()
        return [{"id": doc_id, "title": title} for doc_id, title in rows]

    def search(self, question: str, role: str, top_k: int = 4) -> list[Hit]:
        validate_role(role)
        rows = self.db.execute(
            """SELECT c.id, c.doc_title, c.para, c.text, c.embedding
               FROM chunks c JOIN chunk_roles r ON r.chunk_id = c.id
               WHERE r.role = ?""",
            (role,),
        ).fetchall()
        if not rows:
            return []

        query = np.asarray(self.embedder.query(question), dtype=np.float32)
        matrix = np.vstack([np.frombuffer(row[4], dtype=np.float32) for row in rows])
        scores = matrix @ query
        best = np.argsort(-scores)[:top_k]
        return [
            Hit(rows[i][0], rows[i][1], rows[i][2], rows[i][3], float(scores[i]))
            for i in best
        ]