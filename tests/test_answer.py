from ragbot import answer
from ragbot.extract import pages_to_text


def test_prompt_contains_rules_context_and_question():
    messages = answer.build_prompt("サービスは無料ですか？", ["本サービスは無償で提供されます。", "第8条 禁止事項"])

    assert [m["role"] for m in messages] == ["system", "user"]
    assert "提供された文書からは分かりません" in messages[0]["content"]   # refusal rule
    user = messages[1]["content"]
    assert "[文書1]\n本サービスは無償で提供されます。" in user
    assert "[文書2]\n第8条 禁止事項" in user
    assert user.endswith("質問: サービスは無料ですか？")


def test_ask_uses_retrieved_chunks(monkeypatch):
    seen = {}
    monkeypatch.setattr(answer, "retrieve", lambda q: ["本サービスは無償で提供されます。"])

    def fake_generate(messages):
        seen["messages"] = messages
        return "はい、無料です。"

    monkeypatch.setattr(answer, "generate", fake_generate)

    assert answer.ask("サービスは無料ですか？", verbose=False) == "はい、無料です。"
    assert "無償で提供" in seen["messages"][1]["content"]


def test_pages_to_text_adds_page_headers():
    assert pages_to_text(["一", "二"]) == "--- Page 1 ---\n一\n--- Page 2 ---\n二"
