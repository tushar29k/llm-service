"""Golden prompt regression tests — catches prompt drift, not model drift.

    python3 evals/run_prompt_regression.py            # compare, exit 1 on mismatch
    python3 evals/run_prompt_regression.py --record   # bless new goldens (deliberate)

Every case pins two things against the mock backend: the exact prompt
text the production code sends (templates rendered, inline suffixes
applied) and the backend's output for it. Edit a prompt file
(prompts/v1/...) or the inline extract suffix in model.py and this
script fails with a diff — edit the goldens instead if you *meant*
to change the prompt.

CI note: there is no .github/workflows yet (roadmap v2.4 "CI: smoke +
bench regression on push"). Until that lands, run this in CI by hand:
`python3 evals/run_prompt_regression.py` alongside test_smoke.py.
Goldens are mock-only — real backends are non-deterministic, so
--backend anything-but-mock just prints outputs without comparing.
"""
import argparse
import difflib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from model import LocalLLM  # noqa: E402

GOLDENS_PATH = os.path.join(HERE, "prompt_goldens.json")

# fixed inputs — boring on purpose; the prompt is what's under test
EXTRACT_CASES = [
    {"name": "extract-order",
     "prompt": "Extract the order.",
     "fields": ["order_id", "items"]},
    {"name": "extract-profile",
     "prompt": "Pull the contact details from this email signature.",
     "fields": ["name", "email", "phone"]},
]
SUMMARISE_CASES = [
    {"name": "summarise-doc",
     "document": "The quick brown fox jumps over the lazy dog.",
     "max_words": 20},
]
CHAT_CASES = [
    {"name": "chat-capital",
     "messages": [{"role": "user", "content": "What is the capital of France?"}]},
    {"name": "chat-system",
     "messages": [{"role": "system", "content": "Answer in one word."},
                  {"role": "user", "content": "What is the capital of France?"}]},
]


def _collect(llm):
    # the production prompt paths, verbatim — a wrapper records the
    # exact prompt text extract_json sends, template suffix and all
    cases = []

    orig_generate = llm.backend.generate
    sent = {}

    def recording(prompt, max_new_tokens=256, temperature=0.7):
        sent["prompt"] = prompt
        return orig_generate(prompt, max_new_tokens=max_new_tokens,
                             temperature=temperature)

    llm.backend.generate = recording
    try:
        for c in EXTRACT_CASES:
            sent.clear()
            out = llm.extract_json(c["prompt"], c["fields"])
            cases.append({"name": c["name"], "kind": "extract",
                          "prompt": sent["prompt"], "output": out})
    finally:
        llm.backend.generate = orig_generate

    for c in SUMMARISE_CASES:
        prompt = llm.load_prompt("summarise", document=c["document"],
                                 max_words=c["max_words"])
        out = llm.generate(prompt)
        cases.append({"name": c["name"], "kind": "template",
                      "prompt": prompt, "output": out})

    for c in CHAT_CASES:
        prompt = llm.build_prompt(c["messages"])
        out = llm.generate(prompt)
        cases.append({"name": c["name"], "kind": "chat",
                      "prompt": prompt, "output": out})
    return cases


def _normalise(cases):
    # stable comparison: sorted keys, no stray whitespace in prompts
    return json.loads(json.dumps(
        [{"name": c["name"], "kind": c["kind"],
          "prompt": c["prompt"], "output": c["output"]} for c in cases]))


def _diff(a, b, label):
    a_lines = json.dumps(a, indent=2, sort_keys=True).splitlines()
    b_lines = json.dumps(b, indent=2, sort_keys=True).splitlines()
    return "\n".join(difflib.unified_diff(
        a_lines, b_lines, fromfile="golden", tofile="current", lineterm=""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true",
                    help="rewrite goldens from current prompts (bless a change)")
    ap.add_argument("--backend", default="mock",
                    help="mock (default); others print outputs without comparing")
    args = ap.parse_args()

    llm = LocalLLM(backend=args.backend)
    is_mock = type(llm.backend).__name__ == "MockBackend"
    current = _normalise(_collect(llm))

    if not is_mock:
        # real backends aren't deterministic — show, don't compare
        print(f"backend: {args.backend} (goldens are mock-only, no comparison)")
        for c in current:
            print(f"\n== {c['name']} ==")
            print(c["prompt"])
            print("->", json.dumps(c["output"])[:200])
        return 0

    if args.record:
        with open(GOLDENS_PATH, "w") as f:
            json.dump(current, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"goldens recorded: {GOLDENS_PATH} ({len(current)} cases)")
        return 0

    try:
        goldens = json.load(open(GOLDENS_PATH))
    except FileNotFoundError:
        print(f"FAIL: no goldens at {GOLDENS_PATH} — run with --record first")
        return 1

    failures = 0
    for golden in goldens:
        cur = next((c for c in current if c["name"] == golden["name"]), None)
        if cur is None:
            print(f"FAIL {golden['name']}: case missing from runner")
            failures += 1
            continue
        if cur == golden:
            print(f"ok   {golden['name']}")
        else:
            failures += 1
            print(f"FAIL {golden['name']}: prompt or output drifted")
            print(_diff(golden, cur, golden["name"]))
    cur_names = {c["name"] for c in current}
    for golden in goldens:
        if golden["name"] not in cur_names:
            print(f"FAIL {golden['name']}: case removed without re-recording")
            failures += 1

    if failures:
        print(f"\n{failures} regression(s) — a prompt changed; "
              "re-run with --record only if the change was intended")
        return 1
    print(f"\n{len(goldens)} goldens green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
