"""Rate limiting: hammer a tiny bucket until the API starts 429ing.

    python3 test_rate_limit.py
Done when 429s appear past the limit — each one must carry Retry-After.
"""
import app as app_mod


def main():
    # shrink the bucket so the test finishes in milliseconds
    app_mod.limiter.rate = 0.001     # effectively no refill during the test
    app_mod.limiter.capacity = 2    # burst of two, then empty
    app_mod.limiter.enabled = True

    from fastapi.testclient import TestClient
    client = TestClient(app_mod.app)
    body = {"messages": [{"role": "user", "content": "hi"}]}

    codes = [client.post("/chat", json=body).status_code for _ in range(5)]
    assert codes[:2] == [200, 200], f"burst of 2 should pass, got {codes}"
    assert codes[2:] == [429, 429, 429], \
        f"429s should appear past the limit, got {codes}"

    r = client.post("/chat", json=body)
    assert r.status_code == 429
    assert "retry-after" in r.headers, "a 429 must tell the client when to retry"

    # a different api key gets its own bucket, not the default's bill
    r = client.post("/chat", json=body, headers={"X-API-Key": "someone-else"})
    assert r.status_code == 200, "per-key buckets leaked into each other"

    # /info stays reachable for monitoring even while throttled
    assert client.get("/info").status_code == 200

    print("rate-limit test OK")


if __name__ == "__main__":
    main()
