"""Embeddings (Ollama) and the vector store (ChromaDB).

Heavy libraries are imported inside the functions, so the pure logic
(chunking, prompt building) can be tested without Ollama or ChromaDB.
"""
from __future__ import annotations

from . import config


def embed(text: str) -> list[float]:
    import ollama

    return ollama.embeddings(model=config.EMBED_MODEL, prompt=text)["embedding"]


def _client():
    import chromadb

    return chromadb.PersistentClient(path=config.DB_PATH)


def get_collection():
    return _client().get_collection(name=config.COLLECTION_NAME)


def recreate_collection():
    """Drop the old collection (if any) and create an empty one that uses cosine distance."""
    client = _client()
    try:
        client.delete_collection(name=config.COLLECTION_NAME)
    except Exception:
        pass
    return client.create_collection(name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
