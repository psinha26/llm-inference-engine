Machine: aarch64, 4 cores, 4 GB RAM, Linux 6.8.0-138-generic. 8-token prompt, 16 generated tokens, median of 5 runs (3 for the baseline).

| Step | Decode tok/s | Prefill tok/s | Speedup vs naive | Peak memory |
|---|---|---|---|---|
| 0. Naive: fp32, 1 thread, no KV cache | 0.2 | 2.2 | 1.0× | 1385 MB |
| 1. + KV cache | 2.4 | 2.4 | 14.2× | 1385 MB |
| 2. + thread pool (4 threads) | 8.8 | 5.9 | 51.8× | 1385 MB |
| 3. + SIMD (NEON) | 36.7 | 9.7 | 215.6× | 1385 MB |
| 4. + tiled weight layout | 34.2 | 9.7 | 201.1× | 1385 MB |
| 5. + int8 weights | 117.1 | 21.4 | 688.9× | 393 MB |

Best configuration: int8 + tiled at 4 threads, **117.1 tok/s** decode (689× the naive baseline), 393 MB peak memory vs 1385 MB for fp32.

| Threads | 1 | 2 | 4 |
|---|---|---|---|
| int8 decode tok/s | 96.1 | 106.0 | 117.1 |
| fp32 decode tok/s | 29.2 | 38.2 | 34.2 |
