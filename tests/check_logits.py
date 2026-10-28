#!/usr/bin/env python3
"""Compare the engine against Hugging Face transformers (fp32) on the real model.

For each of 20 fixed prompts:
  - final logits after the prompt: max |diff| must be < 1e-3
  - greedy decoding must match token-for-token for 100 tokens

  pip install torch transformers
  python3 tests/check_logits.py build/engine models/SmolLM2-360M

HF results are cached in <model_dir>/hf_reference.npz so reruns only run the engine.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
    "def fibonacci(n):",
    "The theory of relativity states that",
    "Water boils at",
    "In 1969, Neil Armstrong",
    "The best way to learn programming is",
    "A transformer neural network consists of",
    "My favorite food is",
    "The University of Michigan is located in",
    "import numpy as np\n\nx = np.",
    "Photosynthesis is the process by which",
    "The quick brown fox",
    "1, 2, 3, 4, 5,",
    "Dear hiring manager,",
    "The main causes of World War I were",
    "To make a cup of tea, first",
    "SELECT * FROM users WHERE",
    "The mitochondria is",
    "Q: What is 12 times 12?\nA:",
]
N_GEN = 100


def hf_reference(model_dir: Path):
    cache = model_dir / "hf_reference.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return list(z["prompt_ids"]), list(z["logits"]), list(z["greedy"])
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM

    # Tokenize with the `tokenizers` library straight from tokenizer.json (AutoTokenizer would need
    # tokenizer_config.json, and guessing its class can silently add a BOS token).
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    prompt_ids, logits, greedy = [], [], []
    with torch.no_grad():
        for i, p in enumerate(PROMPTS):
            ids = torch.tensor([tok.encode(p).ids])
            res = model(ids, use_cache=True)
            out = res.logits[0, -1].numpy()
            # Plain greedy loop with HF's KV cache (no generate() heuristics; end-of-text allowed).
            gen, past, logit = ids[0].tolist(), res.past_key_values, res.logits[0, -1]
            for _ in range(N_GEN):
                nxt = int(logit.argmax())
                gen.append(nxt)
                res = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
                past, logit = res.past_key_values, res.logits[0, -1]
            prompt_ids.append(ids[0].tolist())
            logits.append(out)
            greedy.append(gen)
            print(f"  hf reference {i + 1}/{len(PROMPTS)}", end="\r", flush=True)
    print()
    np.savez(cache, prompt_ids=np.array(prompt_ids, dtype=object), logits=np.array(logits),
             greedy=np.array(greedy, dtype=object))
    return prompt_ids, logits, greedy


def main():
    engine, model_dir = sys.argv[1], Path(sys.argv[2])
    prompt_ids, ref_logits, ref_greedy = hf_reference(model_dir)
    worst, failures = 0.0, 0
    with tempfile.TemporaryDirectory() as td:
        for i, p in enumerate(PROMPTS):
            lf, idf = Path(td) / "l.bin", Path(td) / "ids.txt"
            subprocess.run([engine, "--model", model_dir / "model.bin", "--tokenizer", model_dir / "tokenizer.json",
                            "--prompt", p, "--max-tokens", str(N_GEN), "--ignore-eos", "--quiet",
                            "--dump-logits", lf, "--ids-out", idf], check=True, capture_output=True)
            logits = np.fromfile(lf, dtype=np.float32)
            ids = [int(x) for x in idf.read_text().split()]
            n_prompt = len(prompt_ids[i])
            diff = float(np.max(np.abs(logits - ref_logits[i])))
            worst = max(worst, diff)
            same_prompt = ids[:n_prompt] == list(prompt_ids[i])
            gen, ref = ids[n_prompt:], list(ref_greedy[i])[n_prompt:]
            match = next((k for k, (a, b) in enumerate(zip(gen, ref)) if a != b), min(len(gen), len(ref)))
            ok = diff < 1e-3 and same_prompt and match >= N_GEN
            failures += not ok
            print(f"{'PASS' if ok else 'FAIL'} {p[:40]!r:44} max|diff| {diff:.2e}  prompt tokens match {same_prompt}"
                  f"  greedy matches {match}/{N_GEN}")
    print(f"worst logit diff {worst:.2e}; {len(PROMPTS) - failures}/{len(PROMPTS)} prompts pass")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
