#!/usr/bin/env python3
"""Independent NumPy implementation of the Llama forward pass, used as a test oracle.

Reads the original Hugging Face safetensors (not the engine's converted file), processes the
whole sequence at once with a causal mask (no KV cache), so it shares no code or structure
with the C++ engine. Used to check the engine on small random models without needing PyTorch.

  python3 tests/reference.py <hf_model_dir> 1,2,3,4  -> prints last-position logits as .npy to stdout path
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from convert import SafeTensors  # noqa: E402  (shared reader; its correctness is checked by the logit match)


def rmsnorm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rope(x, theta):
    # x: [heads, seq, head_dim]; rotate-half convention used by HF Llama.
    seq, hd = x.shape[1], x.shape[2]
    inv_freq = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd))
    ang = np.outer(np.arange(seq), inv_freq)            # [seq, hd/2]
    cos = np.concatenate([np.cos(ang)] * 2, axis=-1)
    sin = np.concatenate([np.sin(ang)] * 2, axis=-1)
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    rot = np.concatenate([-x2, x1], axis=-1)
    return x * cos + rot * sin


def forward(model_dir, tokens):
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    st = SafeTensors(Path(model_dir))
    g = lambda n: st.get(n).astype(np.float64)
    D, nh = cfg["hidden_size"], cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads", nh)
    hd = cfg.get("head_dim") or D // nh
    eps, theta = cfg.get("rms_norm_eps", 1e-5), cfg.get("rope_theta", 10000.0)
    T = len(tokens)

    x = g("model.embed_tokens.weight")[tokens]  # [T, D]
    mask = np.triu(np.full((T, T), -np.inf), k=1)
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        h = rmsnorm(x, g(p + "input_layernorm.weight"), eps)
        q, k, v = (h @ g(p + f"self_attn.{n}_proj.weight").T for n in "qkv")
        if p + "self_attn.q_proj.bias" in st:
            q, k, v = (t + g(p + f"self_attn.{n}_proj.bias") for t, n in zip((q, k, v), "qkv"))
        q = rope(q.reshape(T, nh, hd).transpose(1, 0, 2), theta)
        k = rope(k.reshape(T, nkv, hd).transpose(1, 0, 2), theta)
        v = v.reshape(T, nkv, hd).transpose(1, 0, 2)
        k = np.repeat(k, nh // nkv, axis=0)
        v = np.repeat(v, nh // nkv, axis=0)
        s = q @ k.transpose(0, 2, 1) / np.sqrt(hd) + mask
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = (s @ v).transpose(1, 0, 2).reshape(T, nh * hd)
        x = x + o @ g(p + "self_attn.o_proj.weight").T
        h = rmsnorm(x, g(p + "post_attention_layernorm.weight"), eps)
        gate = h @ g(p + "mlp.gate_proj.weight").T
        up = h @ g(p + "mlp.up_proj.weight").T
        x = x + (gate / (1 + np.exp(-gate)) * up) @ g(p + "mlp.down_proj.weight").T
    x = rmsnorm(x, g("model.norm.weight"), eps)
    head = g("lm_head.weight") if "lm_head.weight" in st else g("model.embed_tokens.weight")
    return x @ head.T  # [T, vocab]


def greedy(model_dir, tokens, n):
    toks = list(tokens)
    for _ in range(n):
        toks.append(int(np.argmax(forward(model_dir, toks)[-1])))
    return toks


if __name__ == "__main__":
    logits = forward(sys.argv[1], [int(t) for t in sys.argv[2].split(",")])
    np.save(sys.stdout.buffer, logits[-1].astype(np.float32))
