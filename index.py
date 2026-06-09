import chromadb
import ollama
from pathlib import Path

CHUNK_SIZE = 500
OVERLAP = 100
EMBED_MODEL = "bge-m3"
COLLECTION_NAME = "pdf_chunks"
DB_PATH = "./chroma_db"

text = Path("data/source.txt").read_text(encoding="utf-8")

chunks = []
start = 0
while start < len(text):
    chunks.append(text[start:start + CHUNK_SIZE])
    start += CHUNK_SIZE - OVERLAP

print(f"Chunked into {len(chunks)} pieces.\n")

client = chromadb.PersistentClient(path=DB_PATH)

try:
    client.delete_collection(name=COLLECTION_NAME)
except Exception:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},   # add this
)

for i, chunk in enumerate(chunks):
    embedding = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)["embedding"]

    collection.add(
        documents=[chunk],
        embeddings=[embedding],
        ids=[f"chunk_{i}"],
        metadatas=[{"chunk_index": i, "char_length": len(chunk)}],
    )
    print(f"Stored chunk {i}  ({len(chunk)} chars)")

print(f"\nTotal stored in DB: {collection.count()}")
