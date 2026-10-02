"""Prompt cache: repeated prompts report hits.  python3 test_cache.py

Done when repeated prompts report hits — the second identical /chat call is
served from the cache ("cached": true) and /info shows hit_rate > 0.
Also proves the optional semantic tier catches near-duplicate prompts.
"""
import os

os.environ.pop("LLM_API_KEY", None)  # mock backend: no network touched

import app as app_mod  # noqa: E402


def _client():
    from fastapi.testclient import TestClient
    return TestClient(app_mod.app)


def test_exact_hit():
    app_mod._prompt_cache.reset()
    client = _client()
    body = {"messages": [{"role": "user",
                          "content": "what is the capital of cache-land"}]}
    r1 = client.post("/chat", json=body)
    assert r1.status_code == 200, r1.text
    assert r1.json().get("cached") is not True, \
        "first call must hit the backend, not the cache"
    r2 = client.post("/chat", json=body)
    assert r2.status_code == 200, r2.text
    assert r2.json().get("cached") is True, \
        f"repeat prompt should be served from cache: {r2.json()}"
    # same answer, no model call involved the second time
    assert r2.json()["response"] == r1.json()["response"]
    print("exact-hit ok")


def test_info_hit_rate():
    client = _client()
    body = client.get("/info").json()
    cache = body.get("cache")
    assert cache is not None, f"/info must carry a cache object: {body}"
    for key in ("enabled", "hits", "misses", "hit_rate", "size"):
        assert key in cache, f"cache stats missing {key!r}: {cache}"
    assert cache["enabled"] is True, cache
    assert cache["hits"] >= 1, cache  # the repeat call above
    assert cache["hit_rate"] > 0, f"repeated prompts must report hits: {cache}"
    assert cache["size"] >= 1, cache
    print(f"info ok: hit_rate={cache['hit_rate']}")


def test_semantic_tier():
    # near-duplicate prompt, semantic tier switched on just for this check
    cache = app_mod._prompt_cache
    old = (cache.sem_enabled, cache.sem_threshold)
    cache.sem_enabled, cache.sem_threshold = True, 0.9
    try:
        client = _client()
        base = "the quick brown fox jumps over the lazy dog near the river"
        r1 = client.post("/chat", json={"messages": [
            {"role": "user", "content": base}]})
        assert r1.status_code == 200, r1.text
        r2 = client.post("/chat", json={"messages": [
            {"role": "user", "content": base + " tonight"}]})
        assert r2.status_code == 200, r2.text
        assert r2.json().get("cached") is True, \
            f"near-duplicate should be a semantic hit: {r2.json()}"
        assert client.get("/info").json()["cache"]["semantic_hits"] >= 1
    finally:
        cache.sem_enabled, cache.sem_threshold = old
    print("semantic-tier ok")


def test_different_prompt_misses():
    client = _client()
    body = {"messages": [{"role": "user",
                          "content": "a completely unrelated prompt qqq"}]}
    r = client.post("/chat", json=body)
    assert r.status_code == 200, r.text
    assert r.json().get("cached") is not True, \
        "a fresh prompt must not be served from cache"
    print("miss ok")


def main():
    test_exact_hit()
    test_info_hit_rate()
    test_semantic_tier()
    test_different_prompt_misses()
    print("cache test OK")


if __name__ == "__main__":
    main()
