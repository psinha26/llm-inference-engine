#!/usr/bin/env python3
"""Write a random checkpoint with the exact shapes of a real model (default: SmolLM2-360M).

Speed depends only on tensor shapes, never on weight values, so this gives the same tokens/sec as
the real model without downloading it. Output is a normal HF-style directory (config.json +
model.safetensors, fp32) that scripts/convert.py accepts. Tensors are streamed to disk one at a
time, so memory stays low.

  python3 bench/make_shaped_model.py out_dir [--preset smollm2-360m]
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

PRESETS = {
    "smollm2-360m": dict(hidden_size=960, intermediate_size=2560, num_hidden_layers=32, num_attention_heads=15,
                         num_key_value_heads=5, vocab_size=49152, rope_theta=100000.0, rms_norm_eps=1e-5,
                         tie_word_embeddings=True, max_position_embeddings=8192),
    "smollm2-135m": dict(hidden_size=576, intermediate_size=1536, num_hidden_layers=30, num_attention_heads=9,
                         num_key_value_heads=3, vocab_size=49152, rope_theta=100000.0, rms_norm_eps=1e-5,
                         tie_word_embeddings=True, max_position_embeddings=8192),
    # Tiny model for smoke-testing the benchmark scripts.
    "tiny": dict(hidden_size=128, intermediate_size=384, num_hidden_layers=2, num_attention_heads=4,
                 num_key_value_heads=2, vocab_size=1000, rope_theta=10000.0, rms_norm_eps=1e-5,
                 tie_word_embeddings=True, max_position_embeddings=512),
}


def shapes(c):
    D, H, nh, nkv = c["hidden_size"], c["intermediate_size"], c["num_attention_heads"], c["num_key_value_heads"]
    hd = D // nh
    yield "model.embed_tokens.weight", (c["vocab_size"], D)
    for i in range(c["num_hidden_layers"]):
        p = f"model.layers.{i}."
        yield p + "input_layernorm.weight", (D,)
        yield p + "self_attn.q_proj.weight", (nh * hd, D)
        yield p + "self_attn.k_proj.weight", (nkv * hd, D)
        yield p + "self_attn.v_proj.weight", (nkv * hd, D)
        yield p + "self_attn.o_proj.weight", (D, nh * hd)
        yield p + "post_attention_layernorm.weight", (D,)
        yield p + "mlp.gate_proj.weight", (H, D)
        yield p + "mlp.up_proj.weight", (H, D)
        yield p + "mlp.down_proj.weight", (D, H)
    yield "model.norm.weight", (D,)


def main():
    out = Path(sys.argv[1])
    preset = sys.argv[sys.argv.index("--preset") + 1] if "--preset" in sys.argv else "smollm2-360m"
    cfg = PRESETS[preset]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    header, off = {}, 0
    for name, shape in shapes(cfg):
        n = int(np.prod(shape)) * 4
        header[name] = {"dtype": "F32", "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    h = json.dumps(header).encode()
    h += b" " * ((-len(h)) % 8)
    rng = np.random.default_rng(0)
    with open(out / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for name, shape in shapes(cfg):
            if len(shape) == 1:
                t = np.ones(shape, dtype="<f4")
            else:
                t = (rng.standard_normal(shape, dtype=np.float32) * 0.02).astype("<f4")
            f.write(t.tobytes())
    print(f"wrote {preset}-shaped random checkpoint to {out} ({off / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
