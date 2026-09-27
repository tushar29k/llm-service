"""Spot-check extract_json against a fixed 10-case set.

    python3 evals/run_extract_eval.py                 # mock backend (default)
    python3 evals/run_extract_eval.py --backend hf    # real HF model
    LLM_BACKEND=vllm python3 evals/run_extract_eval.py

Against the mock the check is exact equality (the mock echoes each
requested field as "mock-<field>"). Against real backends the check is
looser — all required fields present with the expected types — because a
real model's wording shouldn't fail the suite. Three cases inject bad
first responses to prove the validate-and-retry loop actually recovers
(or gives up loudly when nothing works).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from model import LocalLLM  # noqa: E402

CASES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "extract_cases.jsonl")

TYPE_CHECKS = {
    "str": lambda v: isinstance(v, str),
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "bool": lambda v: isinstance(v, bool),
    "list": lambda v: isinstance(v, list),
    "dict": lambda v: isinstance(v, dict),
}


def load_cases():
    # one json object per line, comments not allowed — keep it parseable
    return [json.loads(line) for line in open(CASES)
            if line.strip()]


def inject(llm, kind):
    # bad-first-response simulation: wrap generate so the first call
    # lies and the rest delegate to the real backend
    real = llm.backend.generate
    calls = {"n": 0}

    def flaky(prompt, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            if kind == "malformed-first":
                return '{"order_id": "oops", broken'  # invalid JSON
            if kind == "missing-field-first":
                return '{"order_id": "only-half"}'    # valid, missing fields
        if kind == "never-valid":
            return "definitely not json"
        return real(prompt, **kw)

    llm.backend.generate = flaky


def check(case, obj, is_mock):
    if is_mock:
        # mock echoes requested fields, so exact match is deterministic
        return obj == case["mock_expected"], (
            f"expected {case['mock_expected']}, got {obj}")
    # real models get a softer check: fields present, types plausible
    missing = [f for f in case["fields"] if f not in obj]
    if missing:
        return False, f"missing fields: {missing}"
    for f, t in case.get("types", {}).items():
        if not TYPE_CHECKS[t](obj[f]):
            return False, f"field {f!r} not {t}: {obj[f]!r}"
    return True, ""


def run_case(case, backend):
    llm = LocalLLM(backend=backend)
    if case.get("inject"):
        inject(llm, case["inject"])
    try:
        obj = llm.extract_json(case["prompt"], case["fields"])
    except ValueError as e:
        # the gives-up case passes by raising — that's the retry loop
        # refusing to return garbage
        if case.get("expect_error"):
            return True, "raised ValueError as expected"
        return False, f"extract_json raised: {e}"
    if case.get("expect_error"):
        return False, "expected ValueError, got a result"
    return check(case, obj, type(llm.backend).__name__ == "MockBackend")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=os.environ.get("LLM_BACKEND"),
                    help="mock | hf | vllm (default: mock, or LLM_BACKEND)")
    args = ap.parse_args()
    cases = load_cases()
    passed, failed = 0, 0
    for case in cases:
        ok, detail = run_case(case, args.backend)
        mark = "PASS" if ok else "FAIL"
        passed, failed = passed + ok, failed + (not ok)
        print(f"[{mark}] {case['name']}" + (f" — {detail}" if not ok else ""))
    print(f"\n{passed}/{len(cases)} passed "
          f"(backend: {args.backend or 'mock'})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
