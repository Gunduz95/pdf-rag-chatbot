"""Step 3: chunk the extracted text, embed every chunk and store it in ChromaDB."""
from __future__ import annotations

from pathlib import Path

from . import config
from .chunking import chunk_text
from .store import embed, recreate_collection


def build_index(txt_path: Path = config.TXT_PATH) -> int:
    text = txt_path.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    print(f"Chunked into {len(chunks)} pieces.")

    collection = recreate_collection()
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embed(chunk)],
            ids=[f"chunk_{i}"],
            metadatas=[{"chunk_index": i, "char_length": len(chunk)}],
        )
    print(f"Stored {collection.count()} chunks in {config.DB_PATH}")
    return collection.count()
