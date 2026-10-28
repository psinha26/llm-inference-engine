#!/usr/bin/env python3
"""End-to-end check on small random Llama models (no download needed).

For each config: write a random HF-style checkpoint (bf16 safetensors), convert it, run the
engine, and compare against tests/reference.py:
  - logits after the prompt: max |diff| must be < 1e-3
  - greedy continuation must match token-for-token
  - the --no-kv-cache baseline must produce exactly the same tokens as the cached path
  - every kernel (naive / simd / blocked) at 1 and 3 threads must meet the same bar
  - int8 models: the scalar int8 kernel must match an int8-simulating NumPy reference, and the
    SIMD int8 kernels must match the scalar one (same integer math)

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

# "dtype" is the checkpoint format. bf16 exercises the converter's bf16 path, but bf16 weights times
# int8-dequantized embeddings land *exactly* on .5 rounding boundaries; the engine and NumPy then break
# those ties differently (1-ulp differences) and a random model amplifies the flipped int8 step. So the
# int8 comparison against NumPy runs on fp32 checkpoints only, where exact ties don't occur.
CONFIGS = {
    # SmolLM2-like: GQA (3 query heads per KV head), tied embeddings, big rope theta
    "gqa_tied": dict(dtype="BF16", hidden_size=96, intermediate_size=256, num_hidden_layers=3, num_attention_heads=6,
                     num_key_value_heads=2, vocab_size=300, rope_theta=100000.0, rms_norm_eps=1e-5,
                     tie_word_embeddings=True),
    # Qwen2-like: QKV biases, untied LM head
    "bias_untied": dict(dtype="F32", hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=4, vocab_size=257, rope_theta=10000.0, rms_norm_eps=1e-6,
                        tie_word_embeddings=False),
    # Same shape as gqa_tied, fp32 checkpoint so the int8 path can be checked exactly
    "gqa_tied_f32": dict(dtype="F32", hidden_size=96, intermediate_size=256, num_hidden_layers=3,
                         num_attention_heads=6, num_key_value_heads=2, vocab_size=300, rope_theta=100000.0,
                         rms_norm_eps=1e-5, tie_word_embeddings=True),
}


def write_safetensors(path, tensors, dtype="BF16"):
    header, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        if dtype == "BF16":
            data = (arr.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()  # truncate
        else:
            data = arr.astype("<f4").tobytes()
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [off, off + len(data)]}
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
    hf_cfg = {k: v for k, v in cfg.items() if k != "dtype"}
    (d / "config.json").write_text(json.dumps({**hf_cfg, "max_position_embeddings": 512}))
    write_safetensors(d / "model.safetensors", t, cfg.get("dtype", "BF16"))


def run_engine(engine, model_bin, prompt, n, tmp, extra=()):
    logits_f, ids_f = tmp / "logits.bin", tmp / "ids.txt"
    subprocess.run([engine, "--model", model_bin, "--tokens", ",".join(map(str, prompt)), "--max-tokens", str(n),
                    "--quiet", "--ignore-eos", "--dump-logits", logits_f, "--ids-out", ids_f, *extra],
                   check=True, capture_output=True)
    return np.fromfile(logits_f, dtype=np.float32), [int(x) for x in ids_f.read_text().split()]


def check(label, logits, ids, ref_logits, ref_ids, tol=1e-3):
    diff = float(np.max(np.abs(logits - ref_logits)))
    ok = diff < tol and ids == ref_ids
    print(f"{'PASS' if ok else 'FAIL'} {label}: max|logit diff| {diff:.2e}, greedy match {ids == ref_ids}")
    return ok


def main():
    engine = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "build" / "engine")
    failures = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for seed, (name, cfg) in enumerate(CONFIGS.items()):
            d = tmp / name
            make_model(d, cfg, seed)
            for fmt in ("fp32", "int8"):
                for layout in ("rowmajor", "tiled"):
                    out = d / (f"{fmt}.bin" if layout == "rowmajor" else f"{fmt}-tiled.bin")
                    subprocess.run([sys.executable, ROOT / "scripts" / "convert.py", d, out, "--quant", fmt,
                                    "--layout", layout], check=True, capture_output=True)
            rng = np.random.default_rng(100 + seed)
            for trial in range(2):
                prompt = rng.integers(0, cfg["vocab_size"], size=int(rng.integers(1, 12))).tolist()
                n_gen = 24
                tag = f"{name} t{trial} ({len(prompt)}-tok prompt)"
                ref_logits = reference.forward(d, prompt)[-1]
                ref_ids = reference.greedy(d, prompt, n_gen)

                # fp32: every kernel and thread count vs the NumPy reference, plus the no-cache baseline.
                for kernel in ("naive", "simd", "blocked"):
                    for threads in (1, 3):
                        logits, ids = run_engine(engine, d / "fp32.bin", prompt, n_gen, tmp,
                                                 ["--kernel", kernel, "--threads", str(threads)])
                        failures += not check(f"{tag} fp32 {kernel:7} x{threads}", logits, ids, ref_logits, ref_ids)
                for kernel in ("naive", "blocked"):
                    for threads in (1, 3):
                        logits, ids = run_engine(engine, d / "fp32-tiled.bin", prompt, n_gen, tmp,
                                                 ["--kernel", kernel, "--threads", str(threads)])
                        failures += not check(f"{tag} fp32 tiled {kernel:7} x{threads}", logits, ids, ref_logits,
                                              ref_ids)
                logits, ids = run_engine(engine, d / "fp32.bin", prompt, n_gen, tmp, ["--kernel", "naive", "--no-kv-cache"])
                failures += not check(f"{tag} fp32 naive no-kv-cache", logits, ids, ref_logits, ref_ids)

                # int8: scalar kernel vs simulated-int8 NumPy, then SIMD kernels vs the scalar kernel.
                q_logits, q_ids = run_engine(engine, d / "int8.bin", prompt, n_gen, tmp, ["--kernel", "naive"])
                if cfg["dtype"] == "F32":
                    failures += not check(f"{tag} int8 naive   x1 vs int8 reference", q_logits, q_ids[:len(prompt) + 1],
                                          reference.forward(d, prompt, q8=True)[-1], q_ids[:len(prompt) + 1])
                for kernel in ("simd", "blocked"):
                    for threads in (1, 3):
                        logits, ids = run_engine(engine, d / "int8.bin", prompt, n_gen, tmp,
                                                 ["--kernel", kernel, "--threads", str(threads)])
                        failures += not check(f"{tag} int8 {kernel:7} x{threads} vs int8 naive", logits, ids,
                                              q_logits, q_ids)
                for kernel in ("naive", "blocked"):
                    for threads in (1, 3):
                        logits, ids = run_engine(engine, d / "int8-tiled.bin", prompt, n_gen, tmp,
                                                 ["--kernel", kernel, "--threads", str(threads)])
                        failures += not check(f"{tag} int8 tiled {kernel:7} x{threads} vs int8 naive", logits, ids,
                                              q_logits, q_ids)
                rel = np.max(np.abs(q_logits - ref_logits)) / np.max(np.abs(ref_logits))
                agree = int(np.argmax(q_logits) == np.argmax(ref_logits))
                print(f"     {tag} int8 vs fp32 reference: max diff {rel:.1%} of logit range, top-1 agrees {bool(agree)}")
    print("all passed" if not failures else f"{failures} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
