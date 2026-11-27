from dataclasses import dataclass
from pathlib import Path

import yaml

from app.rbac import resolve_allowed_roles


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    doc_title: str
    para: int
    text: str
    allowed_roles: tuple[str, ...]


def parse_doc(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        raise ValueError(f"{path.name} is missing frontmatter")
    _, frontmatter, body = raw.split("---", 2)
    meta = yaml.safe_load(frontmatter) or {}

    doc_id = path.stem
    title = meta.get("title") or doc_id
    roles = tuple(resolve_allowed_roles(meta.get("allowed_roles") or []))

    chunks = []
    for block in body.strip().split("\n\n"):
        block = block.strip()
        if not block or (block.startswith("#") and "\n" not in block):
            continue
        para = len(chunks) + 1
        chunks.append(Chunk(f"{doc_id}#p{para}", doc_id, title, para, block, roles))
    return chunks


def load_chunks(docs_dir: str | Path) -> list[Chunk]:
    paths = sorted(p for p in Path(docs_dir).glob("*.md") if not p.name.startswith("_"))
    if not paths:
        raise FileNotFoundError(f"no docs found in {docs_dir}")
    return [chunk for path in paths for chunk in parse_doc(path)]