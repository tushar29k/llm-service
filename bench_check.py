#!/usr/bin/env python3
"""CI bench gate: fail when the mock backend's throughput regresses.

Runs `bench.py --json` on the default (mock) backend — no GPU, no
downloads, no keys — and compares chat/stream tokens-per-second against
the committed bench/baseline.json. Fails when either metric drops more
than TOLERANCE below baseline, or when a baseline-passing extract flips
to failing. GitHub runners are noisy, so the tolerance is generous: this
gate catches real perf cliffs (an accidental sleep, an O(n^2) loop in
the hot path), not 5% wobble.

    python3 bench_check.py
"""
import json
import subprocess
import sys

BASELINE = "bench/baseline.json"
# 30% headroom for runner noise — a genuine regression clears this easily
TOLERANCE = 0.30


def run_bench():
    out = subprocess.run([sys.executable, "bench.py", "--json",
                          "--max-new-tokens", "20"],
                         capture_output=True, text=True, check=True)
    # --json is the last line; the human lines stay for the log
    line = out.stdout.strip().splitlines()[-1]
    return json.loads(line)


def main():
    try:
        current = run_bench()
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"bench gate error: bench.py failed ({e})")
        return 2
    try:
        baseline = json.load(open(BASELINE))
    except (OSError, json.JSONDecodeError) as e:
        print(f"bench gate error: can't read {BASELINE} ({e})")
        return 2

    base = {(r["backend"], r["quant"]): r for r in baseline}
    fails = 0
    print(f"{'backend':10s} {'metric':11s} {'baseline':>10s} {'current':>10s}")
    for r in current:
        key = (r["backend"], r["quant"])
        b = base.get(key)
        if b is None:
            print(f"{r['backend']:10s} {'(new row)':11s} — skipped")
            continue
        for metric in ("chat_tps", "stream_tps"):
            lo = b[metric] * (1 - TOLERANCE)
            mark = "" if r[metric] >= lo else "  REGRESSION"
            print(f"{r['backend']:10s} {metric:11s} {b[metric]:>10.1f} "
                  f"{r[metric]:>10.1f}{mark}")
            if r[metric] < lo:
                fails += 1
        # extract health is binary — a flip from ok to broken always fails
        if b["extract_ok"] and not r["extract_ok"]:
            print(f"{r['backend']:10s} extract_ok   True      False  REGRESSION")
            fails += 1
    if fails:
        print(f"bench gate failed: {fails} metric(s) regressed")
        return 1
    print("bench gate passed: throughput held within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
