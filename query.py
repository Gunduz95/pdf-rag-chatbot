import chromadb
import ollama

EMBED_MODEL = "bge-m3"
COLLECTION_NAME = "pdf_chunks"
DB_PATH = "./chroma_db"
TOP_K = 3

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)


# Try changing this question to anything you want to ask the document
question = "ピザの作り方を教えてください。"

# Embed the question with the SAME model used for the chunks
q_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"]

# Ask ChromaDB for the top K most similar chunks
results = collection.query(
    query_embeddings=[q_embedding],
    n_results=TOP_K,
)

print(f"Question: {question}\n")
print(f"Top {TOP_K} retrieved chunks:\n")

for i in range(len(results["documents"][0])):
    doc = results["documents"][0][i]
    dist = results["distances"][0][i]
    meta = results["metadatas"][0][i]
    print(f"=== Rank {i+1} | distance={dist:.4f} | chunk_index={meta['chunk_index']} ===")
    print(doc[:300])
    print()
