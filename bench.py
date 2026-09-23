"""Quick timings with no HTTP in the loop — talks to the backend directly.

    python3 bench.py                  # config.yaml as-is
    python3 bench.py --quant 8bit     # force 8-bit weights for this run
    python3 bench.py --quant compare  # FP16 vs INT8 rows (the quality/speed
                                      # table — needs backend: hf + a real model)
Prints time-to-first-token, total latency and tokens/sec for chat +
streaming, weight memory, a greedy quality spot-check, and one extract_json
call so you can see the retry path work.
"""
import argparse
import gc
import time

from model import LocalLLM

MESSAGES = [{"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Explain RAG in one sentence."}]

# fixed prompt for the quality spot-check — greedy (temperature 0), so both
# backends answer deterministically and the outputs are genuinely comparable
SPOT = [{"role": "system", "content": "You are concise."},
        {"role": "user",
         "content": "In one short sentence, what does the human heart do?"}]


def weight_mb(llm):
    # real bytes held by the weights (int8 layers report 1 byte/param,
    # so quantized models genuinely show up smaller)
    b = llm.backend
    if type(b).__name__ != "HFBackend" or not b.loaded:
        return None
    return sum(p.numel() * p.element_size()
               for p in b.model.parameters()) / 1e6


def bench_one(quant, max_new_tokens=60):
    llm = LocalLLM(quantization=quant)

    t0 = time.perf_counter()
    resp = llm.chat(MESSAGES, max_new_tokens=max_new_tokens)
    dt = time.perf_counter() - t0
    toks = len(resp.split())
    chat_tps = toks / dt

    t0 = time.perf_counter()
    first, toks, t_first = None, 0, None
    for tok in llm.chat_stream(MESSAGES, max_new_tokens=max_new_tokens):
        if first is None:
            t_first = time.perf_counter() - t0
            first = True
        toks += len(tok.split())
    dt = time.perf_counter() - t0
    stream_tps = toks / dt

    sample = llm.chat(SPOT, max_new_tokens=40, temperature=0.0)

    try:
        obj = llm.extract_json("Extract the order.", ["order_id", "items"])
        extract_ok = (isinstance(obj, dict) and "order_id" in obj
                      and "items" in obj)
    except ValueError:
        # a botched extract is a data point, not a bench crash — keep the row
        obj, extract_ok = {}, False

    return {
        # effective quantization: the backend may have fallen back to none
        "quant": getattr(llm.backend, "quantization", quant),
        "hf": type(llm.backend).__name__ == "HFBackend",
        "chat_tps": chat_tps,
        "ttft_ms": t_first * 1000, "stream_tps": stream_tps,
        "weight_mb": weight_mb(llm),
        "sample": sample.strip().replace("\n", " "),
        "extract_ok": extract_ok, "extract": obj,
    }


def label(r):
    if not r["hf"]:
        return "mock"
    return "INT8" if r["quant"] == "8bit" else "FP16"


def print_run(r):
    w = f"{r['weight_mb']:.0f} MB" if r["weight_mb"] else "n/a"
    print(f"[{label(r)}] chat: ~{r['chat_tps']:.0f} tok/s | "
          f"stream: first token {r['ttft_ms']:.0f}ms, "
          f"{r['stream_tps']:.0f} tok/s | weights: {w}")
    print(f"[{label(r)}] spot-check: {r['sample'][:140]}")
    print(f"[{label(r)}] extract_json ok={r['extract_ok']}: {r['extract']}")


def print_table(rows):
    # the quality-vs-speed table this whole item is about
    print()
    print("quant   weights   chat tok/s   stream tok/s   stream TTFT   "
          "extract ok   quality spot-check (greedy, 40 tok)")
    for r in rows:
        w = f"{r['weight_mb']:.0f}MB" if r["weight_mb"] else "n/a"
        print(f"{label(r):7} "
              f"{w:9} {r['chat_tps']:>10.1f} {r['stream_tps']:>13.1f} "
              f"{r['ttft_ms']:>12.0f}ms   {str(r['extract_ok']):11} "
              f"{r['sample'][:70]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default=None,
                    choices=["none", "8bit", "compare"],
                    help="quantization for this run; 'compare' benches "
                         "both and prints the FP16 vs INT8 table")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    args = ap.parse_args()

    probe = LocalLLM()
    is_hf = type(probe.backend).__name__ == "HFBackend"
    del probe
    gc.collect()
    if args.quant == "compare" and not is_hf:
        print("compare needs backend: hf in config.yaml "
              "(mock has no real weights to quantize)")
        return

    quants = ["none", "8bit"] if args.quant == "compare" else [args.quant]
    rows = []
    for q in quants:
        r = bench_one(q, args.max_new_tokens)
        print_run(r)
        rows.append(r)
        gc.collect()  # drop the model before loading the next quantization

    if args.quant == "compare":
        print_table(rows)


if __name__ == "__main__":
    main()
