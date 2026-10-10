"""History truncation: a 50-turn conversation stays in budget.

    python3 test_history.py
Builds a 50-turn chat, runs both strategies through LocalLLM.prepare_history
(the same path /chat uses), prints the before/after token table, and asserts
the result fits the configured budget. Exit 0 = the roadmap item's
done-when is met.
"""
import os

os.environ.pop("LLM_API_KEY", None)  # mock backend: no network touched

from history import history_tokens
from model import LocalLLM

BUDGET = 4096


def _turn(i, role):
    # ~65 words each — realistic chat turns, comfortably over budget at 49
    words = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do "
             "eiusmod tempor incididunt ut labore et dolore magna aliqua "
             f"turn number {i} keeps the conversation going with a bit of "
             "substance so the token estimate has something to chew on and "
             "the window actually has to drop turns, then it rambles on a "
             "while longer with more filler text to pad the token count "
             "well past the budget line")
    return {"role": role, "content": f"[{role} turn {i}] {words}"}


def _fifty_turns():
    msgs = [{"role": "system",
             "content": "You are a helpful assistant. Be concise."}]
    for i in range(49):
        msgs.append(_turn(i, "user" if i % 2 == 0 else "assistant"))
    return msgs


def _check(name, messages, strategy, llm, keep_recent=10):
    before_msgs, before_toks = len(messages), history_tokens(messages)
    out, stats = llm.prepare_history(messages, strategy=strategy)
    after_toks = history_tokens(out)
    ok = after_toks <= BUDGET and stats["in_budget"]
    print(f"  {name:14s} msgs {before_msgs:3d} -> {len(out):3d}   "
          f"tokens {before_toks:5d} -> {after_toks:5d}   "
          f"budget {BUDGET}   {'IN BUDGET' if ok else 'OVER BUDGET'}")
    assert ok, f"{name}: {after_toks} tokens still over the {BUDGET} budget"
    # the newest turn must survive either strategy — that's the one the
    # user is actually waiting on
    assert out[-1]["content"] == messages[-1]["content"], \
        f"{name}: dropped the newest turn"
    # the system prompt survives too — losing it changes model behaviour
    assert out[0]["role"] == "system" and \
        out[0]["content"] == messages[0]["content"], \
        f"{name}: lost the system prompt"
    return stats


def main():
    llm = LocalLLM()  # config.yaml says backend: mock
    conv = _fifty_turns()
    assert len(conv) == 50, len(conv)
    assert history_tokens(conv) > BUDGET, \
        "the fixture must start over budget or the test proves nothing"
    print(f"50-turn conversation: {history_tokens(conv)} tokens "
          f"({len(conv)} messages), budget {BUDGET}")
    _check("sliding_window", conv, "sliding_window", llm)
    stats = _check("summarise", conv, "summarise", llm)
    # summarise keeps the recent turns verbatim and replaces the older
    # ones with exactly one summary message
    out, _ = llm.prepare_history(conv, strategy="summarise")
    summaries = [m for m in out
                 if "[earlier conversation summary]" in m.get("content", "")]
    assert len(summaries) == 1, \
        f"expected exactly one summary message, got {len(summaries)}"
    recent = [m for m in out if m.get("role") != "system"
              and m not in summaries]
    assert recent == [m for m in conv if m.get("role") != "system"][-10:], \
        "summarise must keep the last 10 turns verbatim"
    # mock-safe fallback: a failing summariser call must not break the chat
    orig = llm._summarise_older
    llm._summarise_older = lambda older: (_ for _ in ()).throw(
        RuntimeError("backend down"))
    out, stats = llm.prepare_history(conv, strategy="summarise")
    assert stats["in_budget"], "fallback path must still land in budget"
    assert any("summary unavailable" in m.get("content", "") for m in out), \
        "expected the labelled excerpt fallback"
    llm._summarise_older = orig
    # an unknown strategy degrades to the safe one instead of 500ing
    out, stats = llm.prepare_history(conv, strategy="nonsense")
    assert stats["strategy"] == "sliding_window" and stats["in_budget"]
    # /chat end to end: the strategy rides in the request body, the stats
    # ride back in the response
    from fastapi.testclient import TestClient
    import app as app_mod
    client = TestClient(app_mod.app)
    r = client.post("/chat", json={"messages": conv,
                                   "history_strategy": "summarise"})
    assert r.status_code == 200, r.text
    hist = r.json().get("history")
    assert hist and hist["in_budget"] and hist["strategy"] == "summarise", \
        f"/chat must report truncation stats: {r.json().keys()}"
    print("history truncation OK — 50 turns fit the budget on both "
          "strategies")


if __name__ == "__main__":
    main()
