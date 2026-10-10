"""Multi-turn history truncation: keep long conversations inside the
context budget before they reach the model.

Two strategies:

- sliding_window: drop the oldest turns first, always keeping system
  messages and the most recent ones. dumb, cheap, never needs a model.
- summarise: keep system messages and the last `keep_recent` turns
  verbatim, collapse everything older into one short summary (written by
  the backend through `summariser`, or a plain excerpt when the model
  can't be reached). keeps the gist without replaying every word.

Token estimates are char-based (~4 chars/token) — no tokenizer needed
offline, and good enough to size the window conservatively. Backends with
a real tokenizer already count properly inside their own generate paths.
"""
import re

CHARS_PER_TOKEN = 4      # rough english average — errs on the high side,
                         # which is the safe direction for a budget
_MSG_OVERHEAD = 4        # role framing the chat template adds per message


def estimate_tokens(text):
    # stdlib-only — the box running this may not have a tokenizer at all
    return max(1, len(str(text)) // CHARS_PER_TOKEN)


def message_tokens(msg):
    # content plus the role tag; dict-shaped messages only, anything else
    # is treated as a bare string
    if isinstance(msg, dict):
        return (_MSG_OVERHEAD + estimate_tokens(msg.get("role", "")) +
                estimate_tokens(msg.get("content", "")))
    return estimate_tokens(msg)


def history_tokens(messages):
    # how much of the budget a list of messages would eat
    return sum(message_tokens(m) for m in messages)


def sliding_window(messages, max_tokens, keep_system=True):
    # newest-first fill: walk back from the latest turn and stop when the
    # next one wouldn't fit. system messages always survive — losing the
    # persona/instructions changes the model's behaviour mid-conversation.
    msgs = [m for m in messages if isinstance(m, dict)]
    system = [m for m in msgs if m.get("role") == "system"] if keep_system else []
    rest = [m for m in msgs if m not in system]
    kept, used = [], history_tokens(system)
    for m in reversed(rest):
        cost = message_tokens(m)
        if kept and used + cost > max_tokens:
            break
        kept.append(m)
        used += cost
    # the newest turn always fits — a single huge message is the caller's
    # problem, not something truncation should solve by eating everything
    if not kept and rest:
        kept.append(rest[-1])
        used += message_tokens(rest[-1])
    return system + list(reversed(kept))


def _fallback_summary(older):
    # no model to hand (or the call failed) — excerpt the first turns
    # verbatim instead of inventing a summary. labelled so nobody mistakes
    # it for a real abstract.
    excerpt = "\n".join(
        f"{m.get('role', '')}: {m.get('content', '')}" for m in older[:4])
    return "[summary unavailable — first turns of the dropped history]\n" + excerpt


def summarise_history(messages, max_tokens, keep_recent=10, summariser=None):
    # the middle of the conversation gets replaced by one summary message
    # sitting right after the system prompt, where context naturally lives
    msgs = [m for m in messages if isinstance(m, dict)]
    if history_tokens(msgs) <= max_tokens:
        return msgs
    system = [m for m in msgs if m.get("role") == "system"]
    rest = [m for m in msgs if m.get("role") != "system"]
    if len(rest) <= keep_recent:
        # nothing sensible to summarise — fall back to the window
        return sliding_window(msgs, max_tokens)
    recent = rest[-keep_recent:]
    older = rest[:-keep_recent]
    try:
        summary = summariser(older) if summariser else _fallback_summary(older)
    except Exception:
        summary = _fallback_summary(older)  # summarising must never 500 a chat
    summary_msg = {"role": "user",
                   "content": "[earlier conversation summary]\n" + summary}
    out = system + [summary_msg] + recent
    if history_tokens(out) > max_tokens:
        # a pathological summary (or huge recent turns) — the window is
        # the backstop, system + summary included in the fill
        return sliding_window(out, max_tokens)
    return out


def truncate(messages, strategy="sliding_window", max_tokens=4096,
             keep_recent=10, summariser=None):
    # one entry point for the service. returns (messages, stats) — stats
    # ride along in the /chat response so callers can see what happened.
    msgs = [m for m in (messages or []) if isinstance(m, dict)]
    before = history_tokens(msgs)
    if before <= max_tokens:
        return msgs, {"strategy": strategy, "in_budget": True,
                      "tokens_before": before, "tokens_after": before,
                      "messages_before": len(msgs),
                      "messages_after": len(msgs), "dropped": 0}
    strat = (strategy or "sliding_window").lower()
    if strat == "summarise":
        out = summarise_history(msgs, max_tokens, keep_recent, summariser)
    else:
        out = sliding_window(msgs, max_tokens)  # unknown strategy -> the safe one
    after = history_tokens(out)
    return out, {"strategy": strat if strat in ("sliding_window", "summarise")
                 else "sliding_window", "in_budget": after <= max_tokens,
                 "tokens_before": before, "tokens_after": after,
                 "messages_before": len(msgs),
                 "messages_after": len(out),
                 "dropped": len(msgs) - len(out)}
