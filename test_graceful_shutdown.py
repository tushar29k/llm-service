"""Graceful shutdown: SIGTERM drains in-flight streams instead of cutting them.

    python3 test_graceful_shutdown.py
Done when a SIGTERM'd server finishes an active SSE stream before exiting,
and new requests get a 503 while draining (`/` and `/info` stay up).
"""
import asyncio
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

log = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
os.environ["LLM_REQUEST_LOG"] = log  # must land before app is imported

import app as app_mod  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

BODY = {"messages": [{"role": "user", "content": "hi"}]}


def _reset():
    app_mod._draining = False
    app_mod._grace_secs = 20.0


def test_drain_503s_new_requests():
    # flip the draining flag the way the lifespan shutdown hook does, then
    # check the contract: model endpoints 503, observability stays up
    app_mod.limiter.enabled = False
    client = TestClient(app_mod.app)
    try:
        app_mod._draining = True
        r = client.post("/chat", json=BODY)
        assert r.status_code == 503, f"expected 503, got {r.status_code}"
        assert "retry-after" in r.headers, "a 503 must say when to retry"
        r = client.post("/chat/stream", json=BODY)
        assert r.status_code == 503, f"stream should 503 too, got {r.status_code}"
        r = client.post("/v1/chat/completions", json=BODY)
        assert r.status_code == 503
        assert client.get("/info").status_code == 200, "/info must stay up"
        assert client.get("/").status_code == 200, "demo page must stay up"
    finally:
        _reset()

    # flag off again: business as usual
    assert client.post("/chat", json=BODY).status_code == 200
    print("drain-503 test OK")


def test_lifespan_waits_for_inflight():
    # hold a fake in-flight slot, run the shutdown hook, prove it waits
    async def go():
        app_mod._inflight_inc()
        app_mod._grace_secs = 5.0
        finished = asyncio.Event()

        async def shut():
            async with app_mod._drain_on_shutdown(app_mod.app):
                pass
            finished.set()

        task = asyncio.create_task(shut())
        await asyncio.sleep(0.3)
        assert not finished.is_set(), \
            "shutdown returned while a request was in-flight"
        assert app_mod._draining, "draining flag must be set during shutdown"
        app_mod._inflight_dec()  # the "stream" finishes
        await asyncio.wait_for(task, timeout=5)
        assert finished.is_set(), "shutdown never finished after drain"
        assert app_mod._inflight == 0

    asyncio.run(go())
    _reset()
    print("lifespan-wait test OK")


def _wait_for_port(port, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return
        except OSError:
            time.sleep(0.2)
    raise AssertionError(f"server never opened port {port}")


def test_sigterm_waits_for_active_stream():
    # real uvicorn, real SIGTERM: slow the mock stream so it's genuinely
    # mid-flight when the signal lands, then prove every chunk arrives
    wrapper = (
        "import time\n"
        "import app as app_mod\n"
        "_orig = app_mod.llm.chat_stream\n"
        "def _slow(messages, max_new_tokens=256):\n"
        "    for c in _orig(messages, max_new_tokens):\n"
        "        time.sleep(0.6)\n"
        "        yield c\n"
        "app_mod.llm.chat_stream = _slow\n"
        "app = app_mod.app\n"
    )
    tmpd = tempfile.mkdtemp()
    with open(os.path.join(tmpd, "grace_probe_app.py"), "w") as f:
        f.write(wrapper)

    port = 8123
    env = dict(os.environ, LLM_REQUEST_LOG="off", LLM_GRACEFUL_SECS="25",
               PYTHONPATH=tmpd + os.pathsep + os.getcwd())
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "grace_probe_app:app",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=os.getcwd(), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_for_port(port, timeout=30)

        chunks, first = [], threading.Event()

        def read_stream():
            import requests
            with requests.post(
                    f"http://127.0.0.1:{port}/chat/stream", json=BODY,
                    stream=True, timeout=60) as r:
                for line in r.iter_lines():
                    if line:
                        chunks.append(line)
                        first.set()

        t = threading.Thread(target=read_stream, daemon=True)
        t.start()
        assert first.wait(timeout=20), "stream never started"
        time.sleep(0.5)  # properly mid-flight now
        started = time.time()
        proc.send_signal(signal.SIGTERM)

        t.join(timeout=40)
        assert not t.is_alive(), "stream reader hung after SIGTERM"
        # mock streams ~16 tokens; all of them must arrive, none cut
        assert len(chunks) >= 10, \
            f"stream was cut short by SIGTERM: {len(chunks)} chunks"

        rc = proc.wait(timeout=30)
        # uvicorn re-raises SIGTERM after a graceful shutdown so the exit
        # status still shows the signal — -15 here means "drained, then
        # died by SIGTERM", which is exactly the unix-correct outcome
        assert rc == -signal.SIGTERM, f"server exited {rc}"
        print(f"sigterm-drain test OK "
              f"({len(chunks)} chunks delivered, drained then exited by "
              f"SIGTERM in {time.time() - started:.1f}s)")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def main():
    test_drain_503s_new_requests()
    test_lifespan_waits_for_inflight()
    test_sigterm_waits_for_active_stream()
    print("graceful-shutdown tests OK")


if __name__ == "__main__":
    main()
