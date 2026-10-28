#!/usr/bin/env python3
"""Generate NumPy reference vectors for the per-op unit tests (tests/test_ops.cpp).

Writes tests/data/op_vectors.bin: a sequence of records
    name: 32 bytes, NUL-padded ASCII
    n:    int32 element count
    data: n float32
References are computed in float64 and rounded to float32 at the end.

  python3 tests/gen_op_vectors.py
"""
import struct
from pathlib import Path

import numpy as np

rng = np.random.default_rng(1234)
records = {}


def put(name, arr):
    records[name] = np.asarray(arr, dtype=np.float64).ravel()


# matvec: odd sizes on purpose, so later SIMD kernels must handle tails
W = rng.normal(size=(37, 53)); x = rng.normal(size=53)
put("matvec.W", W); put("matvec.x", x); put("matvec.out", W @ x)

# rmsnorm
x = rng.normal(size=96) * 3; w = rng.uniform(0.5, 1.5, size=96); eps = 1e-5
put("rmsnorm.x", x); put("rmsnorm.w", w); put("rmsnorm.eps", [eps])
put("rmsnorm.out", x / np.sqrt(np.mean(x * x) + eps) * w)

# rope (rotate-half), 3 heads of dim 64 at position 17
heads, hd, pos, theta = 3, 64, 17, 100000.0
x = rng.normal(size=(heads, hd))
inv = 1.0 / theta ** (np.arange(0, hd, 2) / hd)
ang = pos * inv
cos, sin = np.cos(ang), np.sin(ang)
x1, x2 = x[:, : hd // 2], x[:, hd // 2:]
out = np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=1)
put("rope.x", x); put("rope.params", [heads, hd, pos, theta]); put("rope.out", out)

# softmax with large logits (checks the max-subtraction for stability)
x = rng.normal(size=50) * 20 + 500
e = np.exp(x - x.max())
put("softmax.x", x); put("softmax.out", e / e.sum())

# swiglu
a = rng.normal(size=40) * 4; b = rng.normal(size=40)
put("swiglu.a", a); put("swiglu.b", b); put("swiglu.out", a / (1 + np.exp(-a)) * b)

out_path = Path(__file__).resolve().parent / "data" / "op_vectors.bin"
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "wb") as f:
    for name, arr in records.items():
        f.write(name.encode().ljust(32, b"\0"))
        f.write(struct.pack("<i", arr.size))
        f.write(arr.astype("<f4").tobytes())
print(f"wrote {len(records)} arrays to {out_path}")
