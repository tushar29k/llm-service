"""Quick timings with no HTTP in the loop — talks to the backend directly.

    python3 bench.py
Prints time-to-first-token, total latency and tokens/sec for chat +
streaming, plus one extract_json call so you can see the retry path work.
"""
import time

from model import LocalLLM

MESSAGES = [{"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Explain RAG in one sentence."}]


def main():
    llm = LocalLLM()

    t0 = time.perf_counter()
    resp = llm.chat(MESSAGES, max_new_tokens=60)
    dt = time.perf_counter() - t0
    toks = len(resp.split())
    print(f"chat:   {dt*1000:.0f}ms total, ~{toks} tokens "
          f"({toks/dt:.0f} tok/s)")

    t0 = time.perf_counter()
    first, toks, t_first = None, 0, None
    for tok in llm.chat_stream(MESSAGES, max_new_tokens=60):
        if first is None:
            t_first = time.perf_counter() - t0
            first = True
        toks += len(tok.split())
    dt = time.perf_counter() - t0
    print(f"stream: first token {t_first*1000:.0f}ms, {toks/dt:.0f} tok/s")

    obj = llm.extract_json("Extract the order.", ["order_id", "items"])
    print(f"extract_json: {obj}")


if __name__ == "__main__":
    main()
