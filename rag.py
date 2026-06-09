import chromadb
import ollama

EMBED_MODEL = "bge-m3"
LLM_MODEL = "qwen2.5:7b"
COLLECTION_NAME = "pdf_chunks"
DB_PATH = "./chroma_db"
TOP_K = 3

# --- DB connection ---
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)


def retrieve(question: str, k: int = TOP_K) -> list[str]:
    """Return top-k chunk texts most similar to the question."""
    q_emb = ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"]
    results = collection.query(query_embeddings=[q_emb], n_results=k)
    return results["documents"][0]


def build_prompt(question: str, chunks: list[str]) -> list[dict]:
    """Build the messages list with retrieved context."""
    context = "\n\n---\n\n".join(
        f"[文書{i+1}]\n{c}" for i, c in enumerate(chunks)
    )

    system_prompt = (
        "あなたは、与えられた文書だけを根拠にして質問に答えるアシスタントです。\n\n"
        "ルール:\n"
        "1. 答えが文書に明示されている、または文書から明らかに導ける場合は答えてください。\n"
        "   肯定の場合も、否定の場合も、条件付きの場合も、文書の内容に基づいて答えてください。\n"
        "2. 文書に関連情報が一切なければ「提供された文書からは分かりません」と答えてください。\n"
        "3. 文書に書かれていない事実を勝手に作ってはいけません。\n"
        "4. 簡潔に、日本語で答えてください。\n\n"
        "推論の例:\n"
        "- 文書に「本サービスは無償で提供されます」と書かれている場合、\n"
        "  「料金はいくらですか？」という質問には「無料です」と答えてください。\n\n"
        "- 文書に「著作権は当社に帰属します」と書かれている場合、\n"
        "  「著作権は誰のものですか？」には「当社のものです」と答えてください。\n\n"
        "- 文書に「当社は損害について賠償する責任を負いません」と書かれている場合、\n"
        "  「損害賠償を受けられますか？」には「いいえ、当社は責任を負わないと記載されています」と答えてください。\n\n"
        "- 文書に「当社は利用者の了解を得ることなく本規約を変更できます」と書かれている場合、\n"
        "  「規約はいつ変更されますか？」には「当社の判断でいつでも変更される可能性があります」と答えてください。\n\n"
        "- 文書に「○○の行為はできません」と書かれている場合、\n"
        "  「○○してもいいですか？」には「いいえ、文書では禁止されています」と答えてください。"
    )



    user_message = (
        f"以下の文書を読んで、質問に答えてください。\n\n"
        f"=== 文書 ===\n{context}\n=== 文書ここまで ===\n\n"
        f"質問: {question}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]


def ask(question: str):
    print(f"\n質問: {question}\n")

    chunks = retrieve(question)

    print(f"--- 取得したチャンク数: {len(chunks)} ---")
    for i, c in enumerate(chunks):
        print(f"[文書{i+1}] {c[:80].strip()}...")
    print()

    messages = build_prompt(question, chunks)
    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        options={"temperature": 0},
    )

    print(f"回答: {response['message']['content']}\n")
    print("=" * 60)


if __name__ == "__main__":

    questions = [
        "他の利用者の個人情報を公開してもいいですか？",
        # 1) Tests "prohibition → no" inference (the 5th few-shot example).
        # Section 8 says collecting/publishing others' personal info is prohibited.
        # Should answer: "いいえ、文書では禁止されています"

        "サービスは無料ですか？",
        # 2) Tests yes/no inference from a positive statement.
        # Doc says "無償で提供" — should answer "はい、無料です."
        # Different angle than Q1 last batch (which asked "how much" — open-ended).

        "サービスの安全性は保証されていますか？",
        # 3) Tricky — doc has a disclaimer section saying NO guarantees on
        # safety/accuracy/legality. Tests if model finds the disclaimer and
        # answers "いいえ、保証されていません."

        "利用料金の支払い方法を教えてください。",
        # 4) Implicit-assumption test. The service is free, so there's no payment.
        # Best answer: "サービスは無料なので、支払い方法はありません" or
        # "支払い方法については文書に記載がありません."
        # Watch carefully — the model might hallucinate a payment method.

        "サービスが止まったら、代わりのサイトを紹介してもらえますか？",
        # 5) Pure hypothetical not addressed in doc. Should refuse cleanly:
        # "提供された文書からは分かりません."
    ]


    for q in questions:
        ask(q)
