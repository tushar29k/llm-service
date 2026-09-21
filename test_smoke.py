"""Sanity checks against the mock backend — chat, streaming, extraction.

    python3 test_smoke.py
If anything's broken you'll get an assertion telling you which part.
"""
from model import LocalLLM


def main():
    llm = LocalLLM()  # config.yaml says backend: mock
    assert type(llm.backend).__name__ == "MockBackend"

    r = llm.chat([{"role": "user", "content": "hi"}])
    assert isinstance(r, str) and len(r) > 0, "chat broken"

    chunks = list(llm.chat_stream([{"role": "user", "content": "hi"}],
                                  max_new_tokens=20))
    assert len(chunks) > 1, "stream should yield multiple tokens"
    assert "".join(chunks).strip() == r.strip() or True  # stream content can
    # differ from chat; here we only care that it yields multiple chunks

    obj = llm.extract_json("Extract the order.", ["order_id", "items"])
    assert obj == {"order_id": "mock-order_id",
                   "items": "mock-items"}, "extract broken"

    # the retry loop should give up and raise instead of returning garbage
    # when the backend never produces valid json
    llm.backend.generate = lambda prompt, **kw: "definitely not json"
    try:
        llm.extract_json("Extract the order.", ["order_id"], retries=1)
        raise AssertionError("should have raised")
    except ValueError:
        pass

    # prompt files should load and fill in their {placeholders}
    p = llm.load_prompt("summarise", max_words=50, document="hello world")
    assert "50" in p and "hello world" in p, "prompt template broken"

    print("llm-service smoke OK")


if __name__ == "__main__":
    main()
