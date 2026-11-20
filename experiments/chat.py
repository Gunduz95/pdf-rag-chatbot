import ollama

MODEL = "llama3.2:3b"

system_prompt = (
    "You are a helpful assistant. Be concise. "
    "If you're not sure about a fact, say so."
)

messages = [{"role": "system", "content": system_prompt}]

print("Chat with the model. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in {"quit", "exit"}:
        break
    if not user_input:
        continue

    messages.append({"role": "user", "content": user_input})
    
    print(f"\n[Sending {len(messages)} messages, total chars: {sum(len(m['content']) for m in messages)}]\n")

    response = ollama.chat(
        model=MODEL,
        messages=messages,
        options={"temperature": 0.7},
    )

    reply = response["message"]["content"]
    messages.append({"role": "assistant", "content": reply})

    print(f"\nAssistant: {reply}\n")
