"""API key auth: keys from env/config, rejected requests get 401.

    python3 test_auth.py
Done when unauthenticated requests fail with 401 while valid keys pass —
and when no keys are configured, the service stays wide open.
"""
import app as app_mod


def main():
    from fastapi.testclient import TestClient
    client = TestClient(app_mod.app)
    body = {"messages": [{"role": "user", "content": "hi"}]}

    # no keys configured -> open access (demo keeps working keyless)
    app_mod._KEYS.clear()
    assert client.post("/chat", json=body).status_code == 200, \
        "open mode should let everything through"

    # keys configured -> missing key gets a 401
    app_mod._KEYS.update({"secret-test-key"})
    r = client.post("/chat", json=body)
    assert r.status_code == 401, f"missing key should 401, got {r.status_code}"
    assert "www-authenticate" in r.headers, \
        "a 401 must tell the client which scheme to use"

    # wrong key gets a 401 too
    r = client.post("/chat", json=body, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401, f"bad key should 401, got {r.status_code}"

    # valid key passes — both the header and the bearer scheme
    assert client.post("/chat", json=body,
                       headers={"X-API-Key": "secret-test-key"}).status_code == 200
    assert client.post("/chat", json=body,
                       headers={"Authorization": "Bearer secret-test-key"}).status_code == 200

    # the stream and openai endpoints are guarded the same way
    assert client.post("/chat/stream", json=body).status_code == 401
    assert client.post("/v1/chat/completions", json=body,
                       headers={"X-API-Key": "secret-test-key"}).status_code == 200

    # / and /info stay public even with keys set — observability, not spend
    assert client.get("/").status_code == 200
    assert client.get("/info").status_code == 200

    app_mod._KEYS.clear()
    print("auth test OK")


if __name__ == "__main__":
    main()
