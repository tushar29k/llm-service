"""Retry with backoff + circuit breaker around backend calls.

    python3 test_resilience.py
Done when a flaky backend test recovers — a fake backend that fails twice
then succeeds must come back through retry, and a dead backend must trip
the breaker: further calls fail fast (BreakerOpenError, no retry storm)
and a 503 goes out over HTTP. No network touched, delays are tiny.
"""
import os
import tempfile
import time

os.environ.pop("LLM_API_KEY", None)  # mock backend: no network touched

import yaml  # noqa: E402

from model import LocalLLM, BreakerOpenError  # noqa: E402


def _llm(**overrides):
    # throwaway config: fast retries, hair-trigger breaker
    cfg = {
        "backend": "mock",
        "model_id": "x",
        "prompt_dir": "prompts/v1",
        "retry": {"enabled": True, "max_attempts": 4,
                  "base_delay_seconds": 0.01, "jitter_seconds": 0.0},
        "circuit_breaker": {"enabled": True, "failure_threshold": 2,
                            "cooldown_seconds": 0.2},
    }
    for k, v in overrides.items():
        cfg[k] = v
    with tempfile.NamedTemporaryFile("w", suffix=".yaml",
                                     delete=False) as f:
        yaml.safe_dump(cfg, f)
        path = f.name
    llm = LocalLLM(config_path=path)
    os.unlink(path)
    return llm


class _Flaky:
    # fake backend: raises for the first `fails` generate calls, then works
    def __init__(self, fails):
        self.fails = fails
        self.calls = 0

    def generate(self, prompt, max_new_tokens=256, temperature=0.7):
        self.calls += 1
        if self.calls <= self.fails:
            raise RuntimeError("transient backend boom")
        return "recovered"

    def stream(self, prompt, max_new_tokens=256):
        for word in self.generate(prompt, max_new_tokens).split():
            yield word + " "


def test_retry_recovers():
    # threshold sits above the planned failures, so the breaker stays shut
    # while retry does its job
    llm = _llm(circuit_breaker={"enabled": True, "failure_threshold": 10,
                                "cooldown_seconds": 0.2})
    llm.backend = _Flaky(fails=2)
    t0 = time.monotonic()
    out = llm.generate("hello")
    dt = time.monotonic() - t0
    assert out == "recovered", out
    assert llm.backend.calls == 3, \
        f"expected 3 attempts, got {llm.backend.calls}"
    # backoff actually waited: 0.01 + 0.02 = 0.03s minimum
    assert dt >= 0.03, f"backoff didn't wait: {dt:.3f}s"
    assert llm.breaker_status()["state"] == "closed", \
        "a recovered backend must leave the breaker closed"
    print(f"retry-recovers ok ({llm.backend.calls} attempts, {dt:.2f}s)")


def test_stream_retry_recovers():
    # a stream that dies before its first chunk gets retried too
    llm = _llm(circuit_breaker={"enabled": True, "failure_threshold": 10,
                                "cooldown_seconds": 0.2})
    llm.backend = _Flaky(fails=1)
    chunks = list(llm.stream("hello"))
    assert "".join(chunks).strip() == "recovered", chunks
    print("stream-retry-recovers ok")


def test_breaker_opens_and_fails_fast():
    llm = _llm()
    llm.backend = _Flaky(fails=999)  # dead backend
    try:
        llm.generate("hello")
        raise AssertionError("should have raised")
    except BreakerOpenError:
        pass  # 2 consecutive failures trips the threshold=2 breaker
    assert llm.breaker_status()["state"] == "open", llm.breaker_status()
    calls = llm.backend.calls
    t0 = time.monotonic()
    try:
        llm.generate("hello")
        raise AssertionError("breaker should fail fast")
    except BreakerOpenError as e:
        assert e.retry_after > 0, "fail-fast should say how long to wait"
    dt = time.monotonic() - t0
    assert dt < 0.5, f"fail-fast took too long: {dt:.2f}s"
    assert llm.backend.calls == calls, \
        "an open breaker must not touch the backend at all"
    print(f"breaker-open ok (fail-fast in {dt * 1000:.0f}ms, "
          f"0 backend calls)")


def test_breaker_recovers_after_cooldown():
    llm = _llm()
    flaky = _Flaky(fails=999)
    llm.backend = flaky
    try:
        llm.generate("x")
    except BreakerOpenError:
        pass
    assert llm.breaker_status()["state"] == "open"
    time.sleep(0.25)          # past the 0.2s cooldown -> half-open probe
    flaky.fails = 0          # backend is healthy again
    out = llm.generate("hello")
    assert out == "recovered", out
    assert llm.breaker_status()["state"] == "closed", llm.breaker_status()
    print("breaker-recovery ok (half-open probe closed it)")


def test_breaker_open_is_503_over_http():
    import app as app_mod  # noqa: E402
    from fastapi.testclient import TestClient  # noqa: E402
    real_backend = app_mod.llm.backend
    app_mod.llm.backend = _Flaky(fails=10 ** 9)
    try:
        # app config: threshold 5, max_attempts 3 — two failing requests
        # trip the breaker (3 + 3 attempts = 6 consecutive failures)
        client = TestClient(app_mod.app, raise_server_exceptions=False)
        body = {"messages": [{"role": "user", "content": "breaker-503"}]}
        r1 = client.post("/chat", json=body)
        assert r1.status_code == 500, r1.status_code
        r2 = client.post("/chat", json=body)
        assert r2.status_code in (500, 503), r2.status_code
        r3 = client.post("/chat", json=body)
        assert r3.status_code == 503, \
            f"open breaker must surface as 503, got {r3.status_code}"
        assert "circuit breaker" in r3.json()["detail"].lower(), r3.json()
        assert r3.headers.get("retry-after"), "503 should carry Retry-After"
        info = client.get("/info").json()
        assert info["breaker"]["state"] == "open", info["breaker"]
        print(f"http-503 ok (detail: {r3.json()['detail'][:60]}…)")
    finally:
        app_mod.llm.backend = real_backend
        # leave the shared app llm with a closed breaker for other tests
        app_mod.llm._breaker._success()


def main():
    test_retry_recovers()
    test_stream_retry_recovers()
    test_breaker_opens_and_fails_fast()
    test_breaker_recovers_after_cooldown()
    test_breaker_open_is_503_over_http()
    print("resilience tests all green")


if __name__ == "__main__":
    main()
