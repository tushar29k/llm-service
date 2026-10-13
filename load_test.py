"""Load test: asyncio concurrent clients against /chat, p50/p99 report.

By default it spins up the service locally on the mock backend (no keys,
no GPU, no network) and drives POST /chat at several concurrency levels,
so the numbers measure the HTTP stack + middleware, not a model.

    python3 load_test.py                        # local server, mock backend
    python3 load_test.py --base-url URL          # hit a running service instead
    python3 load_test.py --levels 1,10,50 --requests 100

429s are the token bucket doing its job, not failures — they're counted
separately and excluded from the latency percentiles (which cover 2xx
only). Prompts are distinct per request so the exact-match cache can't
inflate throughput; one warmup request lands before timing starts.
"""
import argparse
import asyncio
import os
import statistics
import subprocess
import sys
import time

import httpx

# 40 distinct prompts — the exact cache keys on (endpoint, backend,
# model, messages, params), so repeats would never touch the backend
PROMPTS = [
    f"load probe {i}: reply with the number {i} and nothing else"
    for i in range(1, 41)
]
PATH = "/chat"
PORT = 8137


def start_server():
    # mock backend from config.yaml; make sure no stray key flips the
    # auto backend to the real api mid-test
    env = dict(os.environ)
    env.pop("LLM_API_KEY", None)
    env["LLM_REQUEST_LOG"] = "off"  # load noise stays out of the jsonl
    env["LLM_ROUTER_LOG"] = "off"
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app",
         "--host", "127.0.0.1", "--port", str(PORT), "--log-level", "warning"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    return proc


async def wait_ready(base, timeout=30):
    # trust_env=False: the sandbox's no_proxy holds unbracketed IPv6
    # entries that httpx misparses into "Invalid port" — localhost
    # traffic needs no proxy anyway
    async with httpx.AsyncClient(base_url=base, timeout=5,
                                 trust_env=False) as c:
        for _ in range(int(timeout * 2)):
            try:
                r = await c.get("/ready")
                if r.status_code == 200:
                    return
            except httpx.ConnectError:
                pass
            await asyncio.sleep(0.5)
    raise SystemExit("server never became ready")


async def one(client, prompt, headers):
    body = {"messages": [{"role": "user", "content": prompt}]}
    t0 = time.perf_counter()
    try:
        r = await client.post(PATH, json=body, headers=headers)
        status = r.status_code
    except httpx.HTTPError:
        status = 0  # connection died mid-flight — a real failure
    return status, (time.perf_counter() - t0) * 1000


async def hammer(base, concurrency, n, mode):
    # the rate limiter buckets per X-API-Key. per-client gives each
    # concurrent slot its own bucket (measures the backend + http stack);
    # shared puts everyone in one bucket (documents the guardrail shedding
    # load past 10 r/s + burst 30). both are honest, they answer different
    # questions — the table header says which one you're looking at.
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=base, timeout=30,
                                 trust_env=False,
                                 limits=httpx.Limits(
                                     max_connections=concurrency + 5)) as c:
        # warmup: one request through before timing (JIT, caches, lazies)
        await one(c, PROMPTS[0], _headers(mode, 0))

        async def guarded(i):
            async with sem:
                return await one(c, PROMPTS[i % len(PROMPTS)],
                                 _headers(mode, i % concurrency))

        t0 = time.perf_counter()
        results = await asyncio.gather(
            *(guarded(i) for i in range(n)))
        wall = time.perf_counter() - t0
    return results, wall


def _headers(mode, slot):
    # auth is open by default (no keys configured) — the header only
    # selects the rate-limit bucket, it doesn't need to be a real key
    key = "load-shared" if mode == "shared" else f"load-client-{slot}"
    return {"X-API-Key": key}


def pct(vals, q):
    if not vals:
        return float("nan")
    return statistics.quantiles(vals, n=100)[int(q) - 1] \
        if len(vals) > 1 else vals[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=None,
                    help="hit a running service instead of spawning one")
    ap.add_argument("--levels", default="1,5,20,50",
                    help="concurrency levels, comma-separated")
    ap.add_argument("--requests", type=int, default=100,
                    help="requests per level")
    ap.add_argument("--mode", default="per-client",
                    choices=("per-client", "shared"),
                    help="per-client: each slot gets its own rate-limit "
                         "bucket (measures the backend); shared: one bucket "
                         "for everyone (documents the guardrail)")
    args = ap.parse_args()

    proc = None
    base = args.base_url
    if base is None:
        base = f"http://127.0.0.1:{PORT}"
        print(f"starting local server on {base} (mock backend, "
              f"mode={args.mode}) ...")
        proc = start_server()
    try:
        asyncio.run(wait_ready(base))
        print(f"{'concurrency':>11s} {'n':>4s} {'2xx':>5s} {'429':>5s} "
              f"{'err':>4s} {'p50_ms':>7s} {'p99_ms':>7s} {'req/s':>7s}")
        for level in [int(x) for x in args.levels.split(",")]:
            results, wall = asyncio.run(
                hammer(base, level, args.requests, args.mode))
            ok = [ms for s, ms in results if s == 200]
            r429 = sum(1 for s, _ in results if s == 429)
            err = sum(1 for s, _ in results if s not in (200, 429))
            print(f"{level:>11d} {args.requests:>4d} {len(ok):>5d} "
                  f"{r429:>5d} {err:>4d} {pct(ok, 50):>7.1f} "
                  f"{pct(ok, 99):>7.1f} {args.requests / wall:>7.1f}")
        print("\n429 = rate limiter shedding past 10 r/s + burst 30 "
              "(the guardrail working); err = real failures (want 0)")
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=10)


if __name__ == "__main__":
    main()
