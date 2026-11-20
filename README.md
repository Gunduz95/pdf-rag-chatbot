# PDF RAG Chatbot (Japanese, local models)

![tests](https://github.com/Gunduz95/pdf-rag-chatbot/actions/workflows/tests.yml/badge.svg)

Ask questions about a Japanese PDF (for example terms of service) and get answers **only from the document**.
Everything runs locally with [Ollama](https://ollama.com): no data leaves the machine.

```
PDF ──pypdf──▶ text ──sliding window──▶ chunks ──bge-m3──▶ ChromaDB (cosine)
                                                               │ top-k
question ──bge-m3──▶ query vector ─────────────────────────────┘
                                   ▼
                 grounded few-shot prompt ──qwen2.5:7b (temp 0)──▶ answer
```

## Design choices

| Step | Choice | Why |
|---|---|---|
| Extraction | `pypdf`, one header per page | simple, keeps page boundaries visible |
| Chunking | 500 characters, 100 overlap | Japanese has no spaces; overlap keeps sentences that cross a boundary |
| Embeddings | `bge-m3` | multilingual, works well for Japanese |
| Vector store | ChromaDB, cosine distance, on disk | no server needed |
| LLM | `qwen2.5:7b`, temperature 0 | good Japanese, repeatable answers |
| Prompt | rules + 5 few-shot examples | covers yes, no, "who owns", "can it change" and "is it forbidden" questions; says 「提供された文書からは分かりません」 when the answer is not in the document |

## Evaluation questions

`python -m ragbot demo` runs 5 questions that check the hard cases: a prohibited action, a yes/no question from a positive sentence, a disclaimer ("is safety guaranteed?"), a hidden assumption ("how do I pay?" for a free service) and a question the document does not answer at all. The expected answer for each is written next to it in `ragbot/answer.py`.

## Run it

```bash
ollama pull bge-m3
ollama pull qwen2.5:7b

python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# put your PDF at data/source.pdf
python -m ragbot extract          # PDF -> data/source.txt
python -m ragbot index            # chunk + embed + store
python -m ragbot ask "サービスは無料ですか？"
python -m ragbot demo             # the evaluation questions
```

Settings (models, chunk size, top-k, paths) can be changed with environment variables, see `ragbot/config.py`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover chunking, page extraction and prompt building, and check that `ask` sends the retrieved text to the model (the model is replaced by a fake, so tests run without Ollama). They run on every push with GitHub Actions.

## Project layout

```
ragbot/        extract.py · chunking.py · store.py · indexer.py · answer.py · __main__.py (CLI)
tests/         unit tests
experiments/   small scripts I used while learning (embedding similarity, retrieval debug, chat)
```

## What I learned

- Embeddings and cosine similarity (see `experiments/embedding_similarity.py`)
- Retrieval failure modes: chunks too small, chunks with many topics, different words for the same idea
- Few-shot prompting for yes / no / conditional answers
- Controlling hallucination with a grounded prompt and a clear "I don't know" answer
