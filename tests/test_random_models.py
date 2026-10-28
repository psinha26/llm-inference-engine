#!/usr/bin/env python3
"""End-to-end check on small random Llama models (no download needed).

For each config: write a random HF-style checkpoint (bf16 safetensors), convert it, run the
engine, and compare against tests/reference.py:
  - logits after the prompt: max |diff| must be < 1e-3
  - greedy continuation must match token-for-token
  - the --no-kv-cache baseline must produce exactly the same tokens as the cached path

  python3 tests/test_random_models.py build/engine
"""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
import reference  # noqa: E402

CONFIGS = {
    # SmolLM2-like: GQA (3 query heads per KV head), tied embeddings, big rope theta
    "gqa_tied": dict(hidden_size=96, intermediate_size=256, num_hidden_layers=3, num_attention_heads=6,
                     num_key_value_heads=2, vocab_size=300, rope_theta=100000.0, rms_norm_eps=1e-5,
                     tie_word_embeddings=True),
    # Qwen2-like: QKV biases, untied LM head
    "bias_untied": dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=4, vocab_size=257, rope_theta=10000.0, rms_norm_eps=1e-6,
                        tie_word_embeddings=False),
}


def write_safetensors(path, tensors):
    header, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        bits = (arr.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)  # truncate to bf16
        data = bits.tobytes()
        header[name] = {"dtype": "BF16", "shape": list(arr.shape), "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    h = json.dumps(header).encode()
    h += b" " * ((-len(h)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def make_model(d: Path, cfg, seed):
    rng = np.random.default_rng(seed)
    D, H, nh, nkv = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = D // nh
    r = lambda *s: rng.normal(0, 0.5 / np.sqrt(s[-1]), s).astype(np.float32) * 4
    t = {"model.embed_tokens.weight": rng.normal(0, 1, (cfg["vocab_size"], D)).astype(np.float32)}
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        t[p + "input_layernorm.weight"] = rng.uniform(0.5, 1.5, D).astype(np.float32)
        t[p + "self_attn.q_proj.weight"] = r(nh * hd, D)
        t[p + "self_attn.k_proj.weight"] = r(nkv * hd, D)
        t[p + "self_attn.v_proj.weight"] = r(nkv * hd, D)
        if not cfg["tie_word_embeddings"]:  # use the bias variant for the Qwen-like config
            t[p + "self_attn.q_proj.bias"] = rng.normal(0, 0.1, nh * hd).astype(np.float32)
            t[p + "self_attn.k_proj.bias"] = rng.normal(0, 0.1, nkv * hd).astype(np.float32)
            t[p + "self_attn.v_proj.bias"] = rng.normal(0, 0.1, nkv * hd).astype(np.float32)
        t[p + "self_attn.o_proj.weight"] = r(D, nh * hd)
        t[p + "post_attention_layernorm.weight"] = rng.uniform(0.5, 1.5, D).astype(np.float32)
        t[p + "mlp.gate_proj.weight"] = r(H, D)
        t[p + "mlp.up_proj.weight"] = r(H, D)
        t[p + "mlp.down_proj.weight"] = r(D, H)
    t["model.norm.weight"] = rng.uniform(0.5, 1.5, D).astype(np.float32)
    if not cfg["tie_word_embeddings"]:
        t["lm_head.weight"] = r(cfg["vocab_size"], D)
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({**cfg, "max_position_embeddings": 512}))
    write_safetensors(d / "model.safetensors", t)


def run_engine(engine, model_bin, prompt, n, tmp, extra=()):
    logits_f, ids_f = tmp / "logits.bin", tmp / "ids.txt"
    subprocess.run([engine, "--model", model_bin, "--tokens", ",".join(map(str, prompt)), "--max-tokens", str(n),
                    "--quiet", "--ignore-eos", "--dump-logits", logits_f, "--ids-out", ids_f, *extra],
                   check=True, capture_output=True)
    return np.fromfile(logits_f, dtype=np.float32), [int(x) for x in ids_f.read_text().split()]


def main():
    engine = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "build" / "engine")
    failures = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for seed, (name, cfg) in enumerate(CONFIGS.items()):
            d = tmp / name
            make_model(d, cfg, seed)
            subprocess.run([sys.executable, ROOT / "scripts" / "convert.py", d, d / "model.bin"],
                           check=True, capture_output=True)
            rng = np.random.default_rng(100 + seed)
            for trial in range(3):
                prompt = rng.integers(0, cfg["vocab_size"], size=int(rng.integers(1, 12))).tolist()
                n_gen = 24
                logits, ids = run_engine(engine, d / "model.bin", prompt, n_gen, tmp)
                ref_logits = reference.forward(d, prompt)[-1]
                diff = float(np.max(np.abs(logits - ref_logits)))
                ref_ids = reference.greedy(d, prompt, n_gen)
                _, naive_ids = run_engine(engine, d / "model.bin", prompt, n_gen, tmp, ["--no-kv-cache"])
                ok = diff < 1e-3 and ids == ref_ids and naive_ids == ids
                failures += not ok
                print(f"{'PASS' if ok else 'FAIL'} {name} trial {trial}: prompt {len(prompt)} tok, "
                      f"max|logit diff| {diff:.2e}, greedy match {ids == ref_ids}, no-kv-cache match {naive_ids == ids}")
    print("all passed" if not failures else f"{failures} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
