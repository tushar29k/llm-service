"""GET /health (liveness) and /ready (readiness).  python3 test_health.py

/health is always 200 — even mid-shutdown, so orchestrators don't restart
a container that's draining cleanly. /ready is 200 once the model is
loaded and 503 while a backend is still warming up (lazy hf/vllm backends
report loaded=false until their first _ensure_loaded finishes), so
routers hold traffic until real requests would succeed. both probes
bypass auth and the rate limiter: observability is never a model spend.
"""
import os

os.environ.pop("LLM_API_KEY", None)  # start from the open-access default

from fastapi.testclient import TestClient  # noqa: E402

import app  # noqa: E402

client = TestClient(app.app)


def health_always_ok():
    r = client.get("/health")
    assert r.status_code == 200, f"/health -> {r.status_code}"
    assert r.json() == {"status": "ok"}, r.json()
    print("health ok: liveness 200")


def ready_when_loaded():
    # mock backend is ready the moment it exists
    r = client.get("/ready")
    assert r.status_code == 200, f"/ready -> {r.status_code}"
    assert r.json() == {"status": "ready"}, r.json()
    print("health ok: /ready 200 once the model is loaded")


def ready_503_while_loading():
    # swap in a backend that hasn't finished warming up — exactly what a
    # lazy hf/vllm backend reports between startup and first load
    class _WarmingBackend:
        @property
        def loaded(self):
            return False

    real_backend = app.llm.backend
    app.llm.backend = _WarmingBackend()
    try:
        r = client.get("/ready")
        assert r.status_code == 503, f"want 503 while loading, got {r.status_code}"
        body = r.json()
        assert body["status"] == "loading", body
        assert "Retry-After" in r.headers, "503 should tell routers when to retry"
        # /health stays green the whole time — the process is fine, the
        # model just isn't ready
        assert client.get("/health").status_code == 200
    finally:
        app.llm.backend = real_backend
    assert client.get("/ready").status_code == 200, "ready again after restore"
    print("health ok: /ready 503 during load, back to 200 after")


def probes_bypass_auth_and_limits():
    # with keys enforced, /chat needs one — the probes must not
    app._KEYS = {"test-secret-key"}
    try:
        assert client.post("/chat", json={"messages": []}).status_code == 401
        for path in ("/health", "/ready"):
            r = client.get(path)
            assert r.status_code == 200, f"{path} -> {r.status_code} under auth"
    finally:
        app._KEYS = set()

    # same for the rate limiter: burn the bucket on /chat, probes stay 200
    app.limiter = app._RateLimiter(rate=0.01, capacity=1)
    try:
        assert client.post("/chat", json={"messages": []}).status_code == 200
        r = client.post("/chat", json={"messages": []})
        assert r.status_code == 429, f"bucket should be dry, got {r.status_code}"
        for path in ("/health", "/ready"):
            r = client.get(path)
            assert r.status_code == 200, \
                f"{path} -> {r.status_code} with a dry bucket"
    finally:
        app.limiter = app._RateLimiter(
            app._rate_cfg.get("rate_per_sec", 10.0),
            app._rate_cfg.get("burst", 30),
            app._rate_cfg.get("enabled", True))
    print("health ok: probes bypass auth and rate limiting")


def drain_behavior():
    # during graceful shutdown, liveness stays 200 (don't restart us
    # mid-drain) while readiness flips 503 (send no new traffic here)
    app._draining = True
    try:
        assert client.get("/health").status_code == 200
        r = client.get("/ready")
        assert r.status_code == 503, f"/ready during drain -> {r.status_code}"
    finally:
        app._draining = False
    assert client.get("/ready").status_code == 200
    print("health ok: drain keeps /health 200, /ready 503")


if __name__ == "__main__":
    health_always_ok()
    ready_when_loaded()
    ready_503_while_loading()
    probes_bypass_auth_and_limits()
    drain_behavior()
