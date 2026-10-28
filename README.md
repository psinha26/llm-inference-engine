# llm-inference-engine

A dependency-free C++17 inference engine for Llama-family language models: its own safetensors
converter, BPE tokenizer and transformer forward pass, verified against Hugging Face.

> Status: **naive fp32 baseline** (milestone 2). Speedup ladder (KV cache → threads → SIMD →
> memory layout → int8) in progress. Numbers below will be filled in as each step lands.

## Headline numbers

| Step | Decode tok/s | Speedup vs naive |
|---|---|---|
| 0. Naive (fp32, 1 thread, no KV cache) | _tbd_ | 1.0× |
| 1. KV cache | _tbd_ | |
| 2. Thread pool | _tbd_ | |
| 3. SIMD (NEON / AVX2) | _tbd_ | |
| 4. Cache-friendly layout | _tbd_ | |
| 5. int8 quantization | _tbd_ | |

Machine: _tbd_. Model: SmolLM2-360M. Median of 5 runs.

## Quick start

```bash
./scripts/setup.sh          # downloads SmolLM2-360M, converts it, builds
./build/engine --model models/SmolLM2-360M/model.bin \
               --tokenizer models/SmolLM2-360M/tokenizer.json \
               --prompt "The capital of France is" --max-tokens 64
```

Requires a C++17 compiler, CMake ≥ 3.16 and Python 3 with NumPy (for the converter only).

### CLI options

| Flag | Meaning |
|---|---|
| `--max-tokens N` | tokens to generate (default 256) |
| `--temp T` | sampling temperature; 0 = greedy (default) |
| `--top-k K`, `--top-p P` | top-k / nucleus sampling |
| `--seed S` | RNG seed for reproducible sampling |
| `--no-kv-cache` | naive baseline: recompute the whole sequence for every token |
| `--ignore-eos` | keep generating past end-of-text |

Every run ends with a stats line: prefill tok/s, decode tok/s, peak memory.

## How it works

```
prompt ─► tokenizer (byte-level BPE) ─► token ids
                                         │
            ┌────────────────────────────▼──────────────────────────────┐
            │ embedding                                                 │
            │ N × [ RMSNorm → GQA attention (RoPE, KV cache) → +residual │
            │       RMSNorm → SwiGLU MLP                    → +residual ]│
            │ final RMSNorm → LM head → logits                          │
            └────────────────────────────┬──────────────────────────────┘
                                         ▼
                     sampler (greedy / temperature / top-k / top-p)
```

| Module | What's in it |
|---|---|
| `scripts/convert.py` | safetensors (bf16/fp16/fp32) → flat fp32 file, 64-byte aligned tensors, read with plain NumPy |
| `src/model/` | `mmap` loader and the forward pass with a preallocated per-layer KV cache |
| `src/tokenizer/` | JSON parser, GPT-2 pre-tokenizer (hand-written regex equivalent + generated Unicode tables), BPE merges |
| `src/ops/` | matvec, RMSNorm, RoPE, softmax, SwiGLU — plain loops, the reference for later optimisations |
| `src/sampler.*` | greedy, temperature, top-k, top-p with a seeded RNG |

## Correctness

| Check | Against | Result |
|---|---|---|
| Per-op unit tests (GoogleTest) | NumPy | pass |
| Tokenizer, 1,256 strings (emoji, CJK, code, digits, specials) | HF `tokenizers` | 1256/1256 exact |
| Forward pass on random GQA / QKV-bias models | independent NumPy implementation | max logit diff < 1e-4, greedy identical |
| Real model, 20 prompts | HF `transformers` fp32 | _tbd_ |

```bash
python3 tests/test_random_models.py build/engine            # no download needed
python3 tests/check_tokenizer.py build/tokenize models/SmolLM2-360M/tokenizer.json
python3 tests/check_logits.py build/engine models/SmolLM2-360M   # needs torch + transformers
```

## What I learned

_tbd_
