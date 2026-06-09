import ollama

system_prompt = "You are a careful assistant. If you are not 100% certain about a fact, say 'I'm not sure' instead of guessing. Answer in one sentence."
user_question = "What is RAG in AI engineering?"

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_question},
    ],
)

print(response["message"]["content"])
