"""GET /info reports the real-LLM status honestly.  python3 test_info.py

Runs in mock mode (no key in the environment): real_llm must be False and
provider/model must be None. Never touches a real API.
"""
import os

os.environ.pop("LLM_API_KEY", None)  # the mock path is what we're checking

from fastapi.testclient import TestClient  # noqa: E402

import app  # noqa: E402


def main():
    r = TestClient(app.app).get("/info")
    assert r.status_code == 200, f"/info -> {r.status_code}"
    body = r.json()
    for key in ("real_llm", "provider", "model"):
        assert key in body, f"/info missing {key!r}: {body}"
    assert body["real_llm"] is False, f"no key set, want mock: {body}"
    assert body["provider"] is None, body  # mock mode names no provider
    assert body["last_error"] is None, body  # nothing failed yet
    print("info ok: mock mode reported honestly")


def api_backend_error_surface():
    """APIBackend remembers the last api failure (for /info) and the
    mock fallback still works. Uses a dummy key; the client is stubbed
    so no network is touched."""
    os.environ["LLM_API_KEY"] = "dummy-key-for-test"
    try:
        from model import APIBackend
        from llm_client import FreeLLMError

        be = APIBackend()
        assert be.last_error is None

        def boom(*a, **k):
            raise FreeLLMError("gemini rejected the request (HTTP 429)")
        be.client.generate = boom
        out = be.generate("hello")
        assert "[model unavailable" in out, out  # mock fallback, flagged
        assert be.last_error == \
            "gemini rejected the request (HTTP 429)", be.last_error

        be.client.generate = lambda *a, **k: "real answer"
        assert be.generate("hello") == "real answer"
        assert be.last_error is None  # success clears it
    finally:
        os.environ.pop("LLM_API_KEY", None)
    print("info ok: api failures surface on last_error and clear on recovery")


if __name__ == "__main__":
    main()
    api_backend_error_surface()
