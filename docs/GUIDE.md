# Guide: understanding, finishing and presenting this project

This is the companion to the README for you, Parth. The README is for recruiters; this is for
you to learn the code well enough to defend every line of it in an interview.

## 1. Finish the last steps (≈ 1 hour, mostly waiting)

Your Mac's disk was full when this was built. Free ~4 GB first.

```bash
cd ~/Parth/Projects/llm-inference-engine
xcode-select --install                    # once, if `clang++ --version` fails
./scripts/setup.sh                        # download model + WikiText-2, convert, build
./scripts/verify.sh                       # real model vs Hugging Face + perplexity -> bench/results/verification.md
BENCH=1 ./scripts/setup.sh                # adds model-tiled.bin for the benchmark
.venv/bin/pip install matplotlib
.venv/bin/python bench/run_bench.py --model-dir models/SmolLM2-360M --machine macbook-air
```

Then:

1. Copy `.github/workflows/ci.yml` from the zip into the repo (the file-copy bridge couldn't write
   into `.github/`). It runs the unit tests, tokenizer check and random-model tests on every push.
2. Replace the VM numbers in the README's headline table and chart links with
   `bench/results/macbook-air.md` and its PNGs (native macOS, real weights).
3. Add the perplexity and HF-match rows from `bench/results/verification.md` to the README's
   Correctness table.
4. `git add -A && git commit` and push to a public GitHub repo.

If `verify.sh` reports a greedy mismatch on a prompt after many matching tokens, look at whether
the two top logits were nearly tied at that step. fp32 summation order differs between any two
implementations, so a near-tie can flip. Say so honestly rather than hiding it; a mismatch at
token 1 would be a real bug.

## 2. Reading order

Read in this order. Each step builds on the last.

1. **`scripts/convert.py`**: what's in a checkpoint (safetensors = 8-byte header length + JSON
   header + raw bytes), bf16 → fp32 (bf16 is just the top 16 bits of an fp32), the file format
   the engine reads, int8 quantization (`quantize_q8`) and tiling (`tile`).
2. **`src/model/model.h`**: `Config`, `Mat`, `RunState`. Note that the KV cache is just two big
   arrays `[layers, ctx, kv_dim]`.
3. **`src/model/model.cpp` → `forward()`**: the whole transformer in ~50 lines. Trace one token
   through it with SmolLM2's numbers: dim 960, 15 query heads, 5 KV heads (so 3 query heads share
   each KV head: grouped-query attention), head_dim 64, MLP 2560, 32 layers.
4. **`src/ops/ops.cpp`**: RMSNorm, RoPE, softmax, SwiGLU as plain loops. These are the math.
5. **`tests/reference.py`**: the same model in NumPy, written differently (whole sequence at
   once, causal mask, no KV cache). If you understand why both give the same logits, you
   understand the KV cache.
6. **`src/ops/kernels.cpp` → `kernels_neon.cpp`**: scalar matvec, then NEON, then 4-row
   blocked, then tiled. Read `bench/bench_kernels.cpp` alongside.
7. **`src/ops/thread_pool.cpp`**: spin-then-sleep, the generation counter, and why the caller
   does chunk 0 itself.
8. **`src/tokenizer/tokenizer.cpp`**: byte-to-unicode mapping, the GPT-2 regex as a state
   machine, BPE merges by rank.
9. **`src/server/server.cpp`**: HTTP parsing, SSE, why bytes are held back while streaming.

## 3. Concepts you must be able to explain

**KV cache.** Attention for token t needs keys/values of tokens 0..t. They never change once
computed, so store them. Without a cache, token t recomputes the full forward pass for all t
previous positions: O(n²) total work. With it, each new token does one position: O(n) per token
for attention, and constant for everything else.

**Grouped-query attention.** SmolLM2 has 15 query heads but only 5 key/value heads; query head
h uses KV head h / 3. This cuts the KV cache 3× with little quality loss.

**RoPE.** Position is encoded by rotating pairs of dimensions in q and k by an angle proportional
to position. HF Llama pairs dimension i with i + head_dim/2 ("rotate half"), not adjacent dims.
Getting this pairing wrong gives a model that runs but produces garbage.

**Why decode is memory bound.** A matvec does 2 FLOPs per weight and reads each weight once.
Apple Silicon can do far more than 2 FLOPs in the time it takes to fetch 4 bytes from DRAM, so
the speed limit is bytes/second. Tokens/sec ≈ bandwidth / model size. That's why int8 (3.5×
fewer bytes) gave the biggest late win and why more threads stop helping.

**Block-wise int8.** Each run of 32 weights stores 32 int8 values plus one fp32 scale
d = max|w| / 127. The activation vector is quantized the same way, so each block's dot product
is pure integer math (`vdotq_s32` / SDOT on ARM does 16 int8 multiply-adds per instruction), then
one float multiply by d_w × d_x. One scale per 32 values (rather than per row) keeps outliers in
one block from destroying precision everywhere else.

**Tiled layout.** See bug 2 below.

## 4. The bugs (your interview stories)

"Tell me about a bug that took you longest to find" — pick one of these. They all really
happened while building this, with numbers.

**Bug 1: the int8 test that failed for the wrong reason.** The int8 engine disagreed with a
NumPy simulation of int8 by up to 1.0 logit, but only for ~7% of tokens. Bisecting: a 0-layer
model matched, a 1-layer model with one token didn't, and zeroing the attention output
projection made it match. Checking the actual quantization inputs showed values landing
*exactly* on x.5 before rounding. Cause: bf16 test weights times int8-dequantized embeddings
produce values with short binary expansions, so ties are common, and the engine and NumPy
compute the same product with 1-ulp differences that break ties in opposite directions. Not an
engine bug; the fix was testing int8 on fp32 test weights, where exact ties don't occur
(then: 3e-6 max difference). Lesson: when a numeric comparison fails, find out *where* the
difference is born before assuming the code is wrong.

**Bug 2: the optimization that made things 2× slower.** Computing 4 rows of the matvec at once
loads each piece of x once and uses it 4 times. In the end-to-end benchmark it was 2× *slower*
than one row at a time. A microbenchmark measuring hot (in-cache) vs cold (streamed from DRAM)
weights showed why: hot, it was faster (62 vs 56 GB/s); cold, it collapsed to 19 GB/s. Reading 4
rows that sit 3,840 bytes apart is 4 interleaved memory streams, which the prefetcher handles
badly. Fix: have the converter interleave each group of 4 rows into small tiles so the kernel
reads one sequential stream. Cold bandwidth went to 55 GB/s, the best of any kernel.

**Bug 3: more threads, less speed.** With 4 threads int8 decode was 87 tok/s, slower than 1
thread. The pool's workers spun 20,000 iterations before sleeping on a condition variable. On
Apple Silicon `yield` is so cheap that 20,000 iterations is a few microseconds, shorter than the
serial work (norms, RoPE) between two parallel regions. So every one of ~250 regions per token
paid a full futex wake-up. Building with 0 and 200,000 spin iterations confirmed the theory;
switching to a time-based spin (500 µs) made 4 threads 125 tok/s.

**Bug 4 (methodology): noisy measurements.** In a VM, run-to-run variance was ±15%, enough to
"show" that `--no-mmap` both helped and hurt. Lesson: report medians of several runs, and don't
claim an effect smaller than the noise.

## 5. Interview questions to rehearse

- Walk me through what happens to one token, from input id to sampled output id.
- Why is the KV cache only for keys and values, not queries?
- What limits your decode speed? How would you know? (bandwidth; tok/s × bytes ≈ GB/s)
- Why did threads help the scalar kernel 3× but the SIMD kernel only 1.2×?
- Why quantize activations too, not just weights? What does that cost in accuracy?
- How does top-p differ from top-k? Why apply temperature before both?
- Why is prefill one token at a time here, and how would you make it faster? (batch prompt
  tokens into a matrix-matrix multiply: weights get reused across tokens, so it becomes
  compute bound)
- How would you serve several users at once? (batched decoding: one weight read serves many
  sequences, which is why production servers batch)
- How do you know your output is correct? (tests at three levels: ops, random models vs
  independent NumPy, real model vs HF)

## 6. Resume bullets

Fill the bracketed numbers from `verification.md` and your native benchmark. Until then, the VM
numbers are real measurements on your Mac:

- Built a dependency-free C++17 inference engine for SmolLM2-360M with a custom BPE tokenizer and
  Llama transformer forward pass, matching Hugging Face output token-for-token [verify.sh]
- Reached 125 tokens/sec decode, 52× a scalar single-threaded baseline, via KV caching, a custom
  `std::thread` pool, NEON SIMD kernels and a tiled weight layout (19 → 55 GB/s streaming from DRAM for the 4-row kernel)
- Cut model memory 3.5× (1,385 → 393 MB) with block-wise int8 quantization (SDOT dot products) at
  [P]% perplexity increase on WikiText-2; added an OpenAI-compatible streaming HTTP server

Use "52× a scalar baseline", not "734×": the 734× includes the no-KV-cache baseline, whose cost
depends on sequence length, and an interviewer may push on it.

## 7. Recording the demo

The README should open with a short GIF of the engine streaming text.

1. Install a terminal recorder: `brew install asciinema agg` (or use macOS's built-in screen
   recording with Cmd+Shift+5 and convert to GIF with `ffmpeg`).
2. Make the terminal ~100×24 with a large font so text is readable when scaled down.
3. Record: `asciinema rec demo.cast`, then run
   `./build/engine --model models/SmolLM2-360M-Instruct/model-int8-tiled.bin --tokenizer models/SmolLM2-360M-Instruct/tokenizer.json --threads 4 --temp 0.7 --top-p 0.9 --max-tokens 120 --prompt $'<|im_start|>user\nExplain what a CPU cache is in two sentences.<|im_end|>\n<|im_start|>assistant\n'`
   (the `$'...'` quoting turns `\n` into real newlines; generation stops at `<|im_end|>`)
   so the stats line shows at the end. The Instruct model (`./scripts/setup.sh
   HuggingFaceTB/SmolLM2-360M-Instruct`) answers questions; the base model only continues text.
4. Optionally, in a second take, show the server: start it, then a `curl -N ... "stream": true`
   streaming into the other half of the screen.
5. `agg demo.cast docs/demo.gif`, keep it under ~10 MB, and replace the comment at the top of the
   README with `![demo](docs/demo.gif)`.
