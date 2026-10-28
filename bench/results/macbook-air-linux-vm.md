Machine: aarch64, 4 cores, 4 GB RAM, Linux 6.8.0-138-generic. 8-token prompt, 16 generated tokens, median of 5 runs (3 for the baseline).

| Step | Decode tok/s | Prefill tok/s | Speedup vs naive | Peak memory |
|---|---|---|---|---|
| 0. Naive: fp32, 1 thread, no KV cache | 0.2 | 2.6 | 1.0× | 1385 MB |
| 1. + KV cache | 2.4 | 2.5 | 14.1× | 1385 MB |
| 2. + thread pool (4 threads) | 7.5 | 7.9 | 43.9× | 1385 MB |
| 3. + SIMD (NEON) | 42.4 | 46.3 | 249.6× | 1385 MB |
| 4. + tiled weight layout | 38.8 | 44.4 | 228.4× | 1385 MB |
| 5. + int8 weights | 124.8 | 137.6 | 734.4× | 393 MB |

Best configuration: int8 + tiled at 4 threads, **124.8 tok/s** decode (734× the naive baseline), 393 MB peak memory vs 1385 MB for fp32.

| Threads | 1 | 2 | 4 |
|---|---|---|---|
| int8 decode tok/s | 95.9 | 115.5 | 124.8 |
| fp32 decode tok/s | 31.4 | 38.7 | 38.8 |
