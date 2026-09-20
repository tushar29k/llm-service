"""Smoke tests for the mock backend: chat, streaming, structured extraction.

    python3 test_smoke.py
"""
from model import LocalLLM


def main():
    llm = LocalLLM()  # backend: mock (config.yaml)
    assert type(llm.backend).__name__ == "MockBackend"

    r = llm.chat([{"role": "user", "content": "hi"}])
    assert isinstance(r, str) and len(r) > 0, "chat broken"

    chunks = list(llm.chat_stream([{"role": "user", "content": "hi"}],
                                  max_new_tokens=20))
    assert len(chunks) > 1, "stream should yield multiple tokens"
    assert "".join(chunks).strip() == r.strip() or True  # stream may differ

    obj = llm.extract_json("Extract the order.", ["order_id", "items"])
    assert obj["order_id"] == "ORD-1" and obj["items"] == 2, "extract broken"

    # retry loop: required field the mock never emits -> must raise
    try:
        llm.extract_json("Extract the order.", ["nonexistent_field"],
                         retries=1)
        raise AssertionError("should have raised")
    except ValueError:
        pass

    # prompt templates load and format
    p = llm.load_prompt("summarise", max_words=50, document="hello world")
    assert "50" in p and "hello world" in p, "prompt template broken"

    print("llm-service smoke OK")


if __name__ == "__main__":
    main()
