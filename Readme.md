# PDF RAG Chatbot

A learning project — Japanese-language RAG (Retrieval-Augmented Generation) over PDF documents
using local models (Ollama).

## Architecture
- Text extraction: pypdf
- Chunking: sliding window (500 chars / 100 overlap)
- Embeddings: bge-m3 (multilingual)
- Vector store: ChromaDB (cosine similarity, persistent on disk)
- LLM: qwen2.5:7b (temperature=0)
- Prompt: 5-example few-shot covering positive, negative,
  attribution, open-permission, and prohibition inference patterns

## How to run
1. Install Ollama: https://ollama.com
2. Pull models:
   ollama pull bge-m3
   ollama pull qwen2.5:7b
3. Set up the venv and install dependencies:
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
4. Place your PDF as data/source.pdf
5. Run the pipeline:
   python pdf.py        # extract text
   python index.py      # chunk + embed + store
   python rag.py        # ask questions

## What I learned
- Embeddings & cosine similarity
- Vector databases (ChromaDB)
- Retrieval failure modes (small chunks, multi-topic chunks, vocabulary mismatch)
- Prompt engineering: few-shot for positive/negative/modal inference patterns
- Hallucination control with grounded prompts
