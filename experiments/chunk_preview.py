from pathlib import Path

TXT_PATH = Path("data/source.txt")
CHUNK_SIZE = 500
OVERLAP = 100

text = TXT_PATH.read_text(encoding="utf-8")

chunks = []
start = 0
while start < len(text):
    end = start + CHUNK_SIZE
    chunks.append(text[start:end])
    start += CHUNK_SIZE - OVERLAP

print(f"Total chars: {len(text)}")
print(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={OVERLAP})\n")

for i, c in enumerate(chunks):
    preview = c[:120].replace("\n", " ")
    print(f"--- Chunk {i} ({len(c)} chars) ---")
    print(preview + "...\n")


