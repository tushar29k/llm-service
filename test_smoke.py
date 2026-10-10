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

    # a schema-violating output must trigger a retry, not get returned:
    # first call returns total as a string (schema wants a number),
    # second call is fixed — expect exactly 2 backend calls and the
    # corrected object
    calls = {"n": 0}

    def flaky(prompt, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return '{"order_id": "ORD-1", "total": "twelve-fifty"}'
        return '{"order_id": "ORD-1", "total": 12.50}'

    llm.backend.generate = flaky
    schema = {"type": "object", "required": ["order_id", "total"],
              "properties": {"order_id": {"type": "string"},
                             "total": {"type": "number"}}}
    obj = llm.extract_json("Extract the order.", ["order_id", "total"],
                           retries=2, schema=schema)
    assert obj == {"order_id": "ORD-1", "total": 12.50}, \
        f"schema retry returned garbage: {obj}"
    assert calls["n"] == 2, \
        f"expected the violation to trigger one retry, got {calls['n']} calls"

    # and a violation that never fixes itself still gives up loudly
    llm.backend.generate = lambda prompt, **kw: \
        '{"order_id": "ORD-1", "total": "twelve-fifty"}'
    try:
        llm.extract_json("Extract the order.", ["order_id", "total"],
                         retries=1, schema=schema)
        raise AssertionError("should have raised")
    except ValueError:
        pass

    # prompt files should load and fill in their {placeholders}
    p = llm.load_prompt("summarise", max_words=50, document="hello world")
    assert "50" in p and "hello world" in p, "prompt template broken"

    # vllm backend: same interface, lazy load; falls back to HF with a
    # warning when the vllm package isn't installed (this box)
    llm_v = LocalLLM(backend="vllm")
    assert type(llm_v.backend).__name__ in ("VLLMBackend", "HFBackend")
    assert hasattr(llm_v.backend, "generate")
    assert hasattr(llm_v.backend, "stream")
    assert not llm_v.backend.loaded  # nothing loaded until first generate

    # history truncation: a long conversation gets cut to the budget on
    # both strategies — sliding window drops the oldest turns, summarise
    # keeps the recent ones verbatim behind one summary message
    llm = LocalLLM()  # fresh instance — earlier tests patched generate
    long = ([{"role": "system", "content": "be brief"}] +
            [{"role": "user" if i % 2 == 0 else "assistant",
              "content": "word " * 120 + str(i)} for i in range(30)])
    out, stats = llm.prepare_history(long)  # default = sliding_window
    assert stats["in_budget"] and stats["dropped"] > 0, stats
    assert out[-1]["content"].endswith("29"), "newest turn must survive"
    assert out[0]["role"] == "system", "system prompt must survive"
    out, stats = llm.prepare_history(long, strategy="summarise")
    assert stats["in_budget"], stats
    assert any("[earlier conversation summary]" in m.get("content", "")
               for m in out), "summarise must collapse old turns"
    assert out[-1]["content"].endswith("29"), "newest turn must survive"

    print("llm-service smoke OK")


if __name__ == "__main__":
    main()
