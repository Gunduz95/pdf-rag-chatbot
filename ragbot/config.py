"""Settings. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

PDF_PATH = Path(os.getenv("RAG_PDF_PATH", "data/source.pdf"))
TXT_PATH = Path(os.getenv("RAG_TXT_PATH", "data/source.txt"))
DB_PATH = os.getenv("RAG_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "pdf_chunks")

EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "bge-m3")
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "qwen2.5:7b")

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
OVERLAP = int(os.getenv("RAG_OVERLAP", "100"))
TOP_K = int(os.getenv("RAG_TOP_K", "3"))
