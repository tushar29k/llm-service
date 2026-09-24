"""Quick timings with no HTTP in the loop — talks to the backend directly.

    python3 bench.py                  # config.yaml as-is
    python3 bench.py --quant 8bit     # force 8-bit weights for this run
    python3 bench.py --quant compare  # FP16 vs INT8 rows (the quality/speed
                                      # table — needs backend: hf + a real model)
    python3 bench.py --backends vllm,hf   # vLLM vs HF rows (needs a CUDA GPU
                                          # for the vllm row; falls back to
                                          # HF with a warning without vllm)
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
    # so quantized models genuinely show up smaller); mock has no
    # weights, and vllm keeps its inside the engine, so both read n/a
    b = llm.backend
    if type(b).__name__ != "HFBackend" or not b.loaded:
        return None
    return sum(p.numel() * p.element_size()
               for p in b.model.parameters()) / 1e6


def bench_one(quant, backend=None, max_new_tokens=60):
    llm = LocalLLM(quantization=quant, backend=backend)

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
        "quant": getattr(llm.backend, "quantization", quant) or "-",
        "backend": type(llm.backend).__name__,
        "requested": backend or llm.cfg.get("backend", "mock"),
        "hf": type(llm.backend).__name__ == "HFBackend",
        "chat_tps": chat_tps,
        "ttft_ms": t_first * 1000, "stream_tps": stream_tps,
        "weight_mb": weight_mb(llm),
        "sample": sample.strip().replace("\n", " "),
        "extract_ok": extract_ok, "extract": obj,
    }


def label(r):
    b = r["backend"]
    if b == "VLLMBackend":
        return "vLLM"
    if b == "HFBackend":
        return "INT8" if r["quant"] == "8bit" else "FP16"
    return "mock"


def print_run(r):
    w = f"{r['weight_mb']:.0f} MB" if r["weight_mb"] else "n/a"
    print(f"[{label(r)}] chat: ~{r['chat_tps']:.0f} tok/s | "
          f"stream: first token {r['ttft_ms']:.0f}ms, "
          f"{r['stream_tps']:.0f} tok/s | weights: {w}")
    print(f"[{label(r)}] spot-check: {r['sample'][:140]}")
    print(f"[{label(r)}] extract_json ok={r['extract_ok']}: {r['extract']}")
    if r["requested"] == "vllm" and r["backend"] != "VLLMBackend":
        # the vllm row fell back — say so next to its numbers, not buried
        # in a warning three screens up
        print(f"[{label(r)}] note: vllm was requested but {r['backend']} "
              f"served (vllm package not installed)")


def print_table(rows):
    # the backend head-to-head this whole item is about
    print()
    print("backend   quant   weights   chat tok/s   stream tok/s   "
          "stream TTFT   extract ok   quality spot-check (greedy, 40 tok)")
    for r in rows:
        w = f"{r['weight_mb']:.0f}MB" if r["weight_mb"] else "n/a"
        print(f"{label(r):9} {r['quant']:7} "
              f"{w:9} {r['chat_tps']:>10.1f} {r['stream_tps']:>13.1f} "
              f"{r['ttft_ms']:>12.0f}ms   {str(r['extract_ok']):11} "
              f"{r['sample'][:70]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default=None,
                    choices=["none", "8bit", "compare"],
                    help="quantization for this run; 'compare' benches "
                         "both and prints the FP16 vs INT8 table")
    ap.add_argument("--backends", default=None,
                    help="comma-separated backends to bench head-to-head, "
                         "e.g. --backends vllm,hf (default: config.yaml)")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    args = ap.parse_args()

    probe = LocalLLM()
    is_hf = type(probe.backend).__name__ == "HFBackend"
    del probe
    gc.collect()
    if args.quant == "compare" and not is_hf and not args.backends:
        print("compare needs backend: hf in config.yaml "
              "(mock has no real weights to quantize)")
        return

    backends = args.backends.split(",") if args.backends else [None]
    quants = ["none", "8bit"] if args.quant == "compare" else [args.quant]
    rows = []
    for b in backends:
        for q in quants:
            r = bench_one(q, backend=b,
                          max_new_tokens=args.max_new_tokens)
            print_run(r)
            rows.append(r)
            gc.collect()  # drop the model before loading the next backend

    if len(rows) > 1:
        print_table(rows)


if __name__ == "__main__":
    main()
