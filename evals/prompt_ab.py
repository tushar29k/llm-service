"""Prompt A/B harness: two prompt versions, scored outputs, winner kept.

    python3 evals/prompt_ab.py                 # mock backend (default)
    python3 evals/prompt_ab.py --backend hf    # real HF model

A = the incumbent: prompts/ab/extract_a.txt (the extract prompt suffix
verbatim from model.py's extract_json today; chat runs with no system
prompt, the current behaviour). B = the challenger:
prompts/ab/extract_b.txt (stricter, more explicit) and
prompts/ab/chat_system_b.txt for chat.

spot-check set: the prompt-quality cases from evals/extract_cases.jsonl
(the inject/expect_error ones test the retry loop, not the prompt, so
they're skipped) plus 3 chat spot checks. scoring is deterministic —
format compliance, required fields present, no extra/hallucinated
fields, non-empty values, chat length bounds — so the mock backend can
run it. ties go to the incumbent: a challenger has to beat it outright
to displace production. results land in evals/prompt-ab-results.md and
the winner in prompts/ab/winner.txt.

# SWAP: against a keyed backend, slot an LLM-as-judge call next to the
# deterministic checks below (score 1-5 on a rubric) — the cheap checks
# stay as the gate, the judge breaks the ties the mock can't.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from model import LocalLLM  # noqa: E402

AB_DIR = os.path.join(os.path.dirname(HERE), "prompts", "ab")
RESULTS_MD = os.path.join(HERE, "prompt-ab-results.md")
WINNER_TXT = os.path.join(AB_DIR, "winner.txt")

CHAT_CASES = [
    {"name": "chat-capital",
     "messages_user": "What is the capital of France?",
     "min_chars": 1, "max_chars": 500},
    {"name": "chat-summarise",
     "messages_user": ("Summarise in one sentence: the quick brown fox "
                       "jumps over the lazy dog."),
     "min_chars": 1, "max_chars": 500},
    {"name": "chat-explain",
     "messages_user": "Explain what a vector database is, in plain words.",
     "min_chars": 10, "max_chars": 2000},
]


def load_variants():
    # A is the incumbent — the exact suffix extract_json uses today
    def read(name):
        return open(os.path.join(AB_DIR, name)).read().strip()
    return {
        "A": {"extract": read("extract_a.txt"), "system": None,
              "label": "incumbent (production today)"},
        "B": {"extract": read("extract_b.txt"),
              "system": read("chat_system_b.txt"),
              "label": "challenger (stricter/explicit)"},
    }


def load_extract_cases():
    # the spot-check set: prompt-quality cases only; retry-loop cases
    # belong to the resilience tests, not a prompt comparison
    cases = []
    path = os.path.join(HERE, "extract_cases.jsonl")
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        case = json.loads(line)
        if case.get("inject") or case.get("expect_error"):
            continue
        cases.append(case)
    return cases


def check_extract(case, raw, fields, is_mock):
    # deterministic gates a real model must also pass
    checks = {}
    obj, err = LocalLLM._extract_json(
        raw, {"type": "object", "required": fields})
    checks["valid_json"] = err is None
    checks["required_present"] = (obj is not None
                                  and all(f in obj for f in fields))
    checks["no_extra_fields"] = (obj is not None
                                 and set(obj) <= set(fields))
    checks["values_nonempty"] = (obj is not None
                                 and all(isinstance(obj.get(f), str)
                                         and obj[f].strip()
                                         for f in fields))
    if is_mock:
        # the existing eval's mock metric: the mock echoes each field
        expected = {f: "mock-" + f for f in fields}
        checks["echo_exact"] = obj == expected
    return checks


def check_chat(case, text):
    # length bounds + non-emptiness — the mock can only pass or fail
    # these; real quality needs the judge marked SWAP above
    checks = {}
    checks["non_empty"] = bool(text and text.strip())
    checks["length_ok"] = (case["min_chars"] <= len(text or "")
                           <= case["max_chars"])
    return checks


def run(llm, variants, extract_cases, is_mock):
    # one LocalLLM per backend; variants differ only in prompt text
    rows = []
    for case in extract_cases:
        for vname, v in variants.items():
            text = (case["prompt"] + "\n"
                    + v["extract"].format(fields=", ".join(case["fields"])))
            raw = llm.generate(text)
            checks = check_extract(case, raw, case["fields"], is_mock)
            rows.append((case["name"], "extract", vname, checks))
    for case in CHAT_CASES:
        for vname, v in variants.items():
            messages = ([{"role": "system", "content": v["system"]}]
                        if v["system"] else [])
            messages.append({"role": "user",
                             "content": case["messages_user"]})
            text = llm.chat(messages)
            rows.append((case["name"], "chat", vname,
                         check_chat(case, text)))
    return rows


def score(rows):
    # pass rate per variant across every check on every case
    totals = {"A": [0, 0], "B": [0, 0]}  # [passed, total]
    for _, _, vname, checks in rows:
        passed = sum(1 for ok in checks.values() if ok)
        totals[vname][0] += passed
        totals[vname][1] += len(checks)
    return {v: (p / t if t else 0.0) for v, (p, t) in totals.items()}


def write_results(rows, scores, winner, rationale, backend):
    lines = [f"# Prompt A/B — spot-check results",
             "",
             f"- date: 2026-10-07",
             f"- backend: {backend or 'mock'}",
             f"- A (incumbent): current extract suffix / no system prompt",
             f"- B (challenger): prompts/ab/extract_b.txt + chat_system_b.txt",
             "",
             "## Per-case checks (failures listed)",
             "",
             "| case | type | A | B | A failures | B failures |",
             "|---|---|---|---|---|---|"]
    # group by case so A and B sit on one row
    by_case = {}
    for case, ctype, vname, checks in rows:
        key = (case, ctype)
        by_case.setdefault(key, {})[vname] = checks
    for (case, ctype), vs in by_case.items():
        cells = []
        for vname in ("A", "B"):
            checks = vs[vname]
            ok = sum(1 for v in checks.values() if v)
            fails = [k for k, v in checks.items() if not v]
            cells.append(f"{ok}/{len(checks)}")
            cells.append(", ".join(fails) or "—")
        lines.append(f"| {case} | {ctype} | {cells[0]} | {cells[2]} "
                     f"| {cells[1]} | {cells[3]} |")
    lines += ["",
              "## Totals",
              ""]
    for vname in ("A", "B"):
        lines.append(f"- {vname}: {scores[vname]:.3f} pass rate")
    lines += ["",
              "## Winner",
              "",
              f"**{winner}** — {rationale}",
              "",
              "tie-break rule: the challenger must strictly outscore the "
              "incumbent to displace production; ties keep A.",
              "rerun with `--backend hf` (or another real backend) to "
              "measure differences the mock can't show.",
              "",
              "## Winner kept",
              "",
              "prompts/ab/winner.txt records the verdict. while the winner "
              "is A, production is unchanged — model.py's extract_json "
              "already uses the A suffix inline. to adopt B later, swap "
              "that suffix for prompts/ab/extract_b.txt.",
              ""]
    open(RESULTS_MD, "w").write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=os.environ.get("LLM_BACKEND"),
                    help="mock | hf | vllm (default: mock, or LLM_BACKEND)")
    args = ap.parse_args()
    backend = args.backend or "mock"

    variants = load_variants()
    extract_cases = load_extract_cases()
    llm = LocalLLM(backend=args.backend)
    is_mock = type(llm.backend).__name__ == "MockBackend"

    rows = run(llm, variants, extract_cases, is_mock)
    scores = score(rows)
    if scores["B"] > scores["A"]:
        winner, rationale = ("B", "challenger outscored the incumbent — "
                                "adopt prompts/ab/extract_b.txt")
    else:
        # tie or A ahead: the incumbent stays; B is retained as the
        # documented alternative in prompts/ab/
        tie = abs(scores["B"] - scores["A"]) < 1e-9
        winner, rationale = ("A", "tie on the mock backend — incumbent "
                                 "kept" if tie else
                                 "incumbent outscored the challenger")

    write_results(rows, scores, winner, rationale, args.backend)
    open(WINNER_TXT, "w").write(
        f"winner: {winner}\n"
        f"date: 2026-10-07\n"
        f"backend: {backend}\n"
        f"score: A {scores['A']:.3f} vs B {scores['B']:.3f}\n"
        f"rationale: {rationale}\n"
        f"production: {'unchanged (A suffix already inline in model.py)' if winner == 'A' else 'adopt B — see results md'}\n")

    print(f"A: {scores['A']:.3f}   B: {scores['B']:.3f}   "
          f"winner: {winner} ({rationale})")
    print(f"results -> {RESULTS_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
