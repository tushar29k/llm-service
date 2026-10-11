"""Structural check for the prompts/library/ system prompt set.

    python3 evals/prompt_library_check.py

Loads each curated system prompt through LocalLLM.load_system_prompt and
asserts the plumbing a real caller depends on: the prompt loads
byte-identical to the file, the three are behaviourally distinct texts,
unknown names fail loudly, build_prompt puts the system message first so
every backend treats it as the governing instruction, and a chat request
with each persona flows end-to-end on the mock backend. Also prints the
prompt size table that docs/system-prompts.md quotes.

Honest scope: the mock backend is prompt-blind (templated answers), so
this checks plumbing, not output quality. Measured behavioural
differences need a real backend — rerun the pattern in
evals/prompt_ab.py with --backend hf or a keyed api backend.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from model import LocalLLM  # noqa: E402

SAMPLE_USER = "Explain what a vector database is, in plain words."
LIBRARY_DIR = os.path.join(os.path.dirname(HERE), "prompts", "library")


def main():
    llm = LocalLLM()  # config.yaml says backend: mock
    assert type(llm.backend).__name__ == "MockBackend"

    loaded = {}
    for name in LocalLLM.SYSTEM_PROMPTS:
        text = llm.load_system_prompt(name)
        # byte-identical to the file — the loader must not mangle text
        assert text == open(f"{LIBRARY_DIR}/{name}.txt").read(), \
            f"{name}: loaded text differs from the file"
        assert text.strip(), f"{name}: prompt is empty"
        loaded[name] = text
    # distinct texts = distinct personas; identical files mean a duplicate
    assert len(set(loaded.values())) == len(loaded), \
        "library prompts must be behaviourally distinct"

    # typos should surface at load time, not as silent wrong behaviour
    try:
        llm.load_system_prompt("no_such_prompt")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown system prompt name should raise")

    print(f"{'prompt':<9} {'chars':>6} {'words':>6} {'built prompt':>12}")
    for name, text in loaded.items():
        messages = [{"role": "system", "content": text},
                    {"role": "user", "content": SAMPLE_USER}]
        built = llm.build_prompt(messages)
        # system first: every chat template and the mock stand-in render
        # messages in order, so ordering is the guarantee the persona wins
        assert built.lstrip().startswith("system:"), \
            f"{name}: system message is not first in the built prompt"
        assert text in built, f"{name}: system text missing from built prompt"
        reply = llm.chat(messages, max_new_tokens=64)
        assert isinstance(reply, str) and reply.strip(), \
            f"{name}: chat returned nothing"
        print(f"{name:<9} {len(text):>6} {len(text.split()):>6} "
              f"{len(built):>12}")

    print("system prompt library: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
