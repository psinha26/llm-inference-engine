# Verification: SmolLM2-360M

Run on 2026-10-28 in a 4-vCPU Linux VM on a MacBook Air (Apple Silicon), NEON kernels.
Reference: Hugging Face `transformers` 4.49 + PyTorch 2.5.1 (CPU, fp32) and `tokenizers` 0.21.

| Check | Result |
|---|---|
| Downloaded weights | `model.safetensors` SHA-256 `7aaff666…540f`, matches the Hub's LFS hash |
| Tokenizer vs HF `tokenizers`, 1,256 strings (edge cases + random text, emoji, CJK, code, digits, special tokens) | 1256/1256 identical token ids; decode(encode(s)) == s for all |
| Final logits after the prompt, 20 prompts, vs HF fp32 | worst max-abs diff 2.0e-4 (target < 1e-3); typical 3-5e-5 |
| Greedy decoding, 100 tokens × 20 prompts, vs HF | 20/20 identical token-for-token |
| Raw-id sequences (4 random), scalar and NEON kernels | max-abs logit diff ≤ 7.1e-5, argmax identical |
| WikiText-2 test perplexity, fp32 | 18.4442 |
| WikiText-2 test perplexity, int8 (32-value blocks) | 18.5656 (+0.66%) |
| Weight file size | fp32 1,447 MB, int8 407 MB (3.6× smaller); peak RSS 1,385 → 393 MB |

Perplexity: the first 8,192 tokens of `wiki.test.raw` (8,176 scored), non-overlapping 512-token
windows, each starting from an empty KV cache. fp32 was evaluated in two halves (`--ppl-start`)
whose window boundaries coincide with the int8 run's, and the summed negative log-likelihoods
combined exactly.

Reproduce: `./scripts/verify.sh` (all of the above) or
`python3 tests/check_logits.py build/engine models/SmolLM2-360M --threads 4`.
