#!/usr/bin/env python3
"""Convert a Hugging Face Llama-family checkpoint into the engine's flat binary format.

Reads config.json + *.safetensors from a model directory with nothing but NumPy
(the safetensors format is an 8-byte header length, a JSON header, then raw bytes).

Output layout (little-endian), see src/model/format.h for the C++ side:
  [0, 256)    header: magic, version, 9 x int32 config, rope_theta, norm_eps, zero padding
  then every tensor as contiguous float32, each starting on a 64-byte boundary, in this order:
    token_embedding [vocab, dim]
    per layer:
      attn_norm [dim]
      wq [n_heads*head_dim, dim]  wk [n_kv*head_dim, dim]  wv [n_kv*head_dim, dim]
      (bq, bk, bv if the model has QKV biases, e.g. Qwen2)
      wo [dim, n_heads*head_dim]
      mlp_norm [dim]
      w_gate [hidden, dim]  w_up [hidden, dim]  w_down [dim, hidden]
    final_norm [dim]
    lm_head [vocab, dim]   (omitted when embeddings are tied)

Usage: python3 scripts/convert.py <hf_model_dir> <out.bin>
"""
import json
import mmap
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = 0x454D4C4C  # "LLME"
VERSION = 1
HEADER_BYTES = 256
ALIGN = 64

FLAG_TIED = 1 << 0
FLAG_QKV_BIAS = 1 << 1

_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}


class SafeTensors:
    """Minimal read-only safetensors reader over one or more shards."""

    def __init__(self, model_dir: Path):
        self.index = {}
        self._maps = []
        for path in sorted(model_dir.glob("*.safetensors")):
            f = open(path, "rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self._maps.append((f, mm))
            (hlen,) = struct.unpack("<Q", mm[:8])
            header = json.loads(mm[8 : 8 + hlen])
            base = 8 + hlen
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                self.index[name] = (mm, base, meta)
        if not self.index:
            sys.exit(f"no .safetensors files found in {model_dir}")

    def __contains__(self, name):
        return name in self.index

    def get(self, name) -> np.ndarray:
        mm, base, meta = self.index[name]
        start, end = meta["data_offsets"]
        dtype = meta["dtype"]
        if dtype not in _DTYPES:
            sys.exit(f"unsupported dtype {dtype} for {name}")
        raw = np.frombuffer(mm, dtype=_DTYPES[dtype], count=(end - start) // np.dtype(_DTYPES[dtype]).itemsize,
                            offset=base + start)
        if dtype == "BF16":  # bf16 is the top 16 bits of an fp32
            arr = (raw.astype(np.uint32) << 16).view(np.float32)
        else:
            arr = raw.astype(np.float32)
        return arr.reshape(meta["shape"])


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    model_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    cfg = json.loads((model_dir / "config.json").read_text())
    st = SafeTensors(model_dir)

    dim = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_heads)
    head_dim = cfg.get("head_dim") or dim // n_heads
    n_layers = cfg["num_hidden_layers"]
    vocab = cfg["vocab_size"]
    hidden = cfg["intermediate_size"]
    max_seq = cfg.get("max_position_embeddings", 2048)
    theta = float(cfg.get("rope_theta", 10000.0))
    eps = float(cfg.get("rms_norm_eps", 1e-5))
    tied = bool(cfg.get("tie_word_embeddings", False)) or "lm_head.weight" not in st
    has_bias = "model.layers.0.self_attn.q_proj.bias" in st
    if cfg.get("rope_scaling"):
        print(f"warning: rope_scaling {cfg['rope_scaling']} is ignored by this engine")

    flags = (FLAG_TIED if tied else 0) | (FLAG_QKV_BIAS if has_bias else 0)
    header = struct.pack("<II9iff", MAGIC, VERSION, dim, hidden, n_layers, n_heads, n_kv,
                         vocab, max_seq, head_dim, flags, theta, eps)
    header += b"\0" * (HEADER_BYTES - len(header))

    order = ["model.embed_tokens.weight"]
    for i in range(n_layers):
        p = f"model.layers.{i}."
        order += [p + "input_layernorm.weight",
                  p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"]
        if has_bias:
            order += [p + "self_attn.q_proj.bias", p + "self_attn.k_proj.bias", p + "self_attn.v_proj.bias"]
        order += [p + "self_attn.o_proj.weight", p + "post_attention_layernorm.weight",
                  p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight", p + "mlp.down_proj.weight"]
    order.append("model.norm.weight")
    if not tied:
        order.append("lm_head.weight")

    total = 0
    with open(out_path, "wb") as out:
        out.write(header)
        for name in order:
            pad = (-out.tell()) % ALIGN
            out.write(b"\0" * pad)
            t = np.ascontiguousarray(st.get(name), dtype="<f4")
            out.write(t.tobytes())
            total += t.size
    print(f"wrote {out_path}: {n_layers} layers, dim {dim}, heads {n_heads}/{n_kv}, vocab {vocab}, "
          f"{total / 1e6:.1f}M params, tied={tied}, qkv_bias={has_bias}")


if __name__ == "__main__":
    main()
