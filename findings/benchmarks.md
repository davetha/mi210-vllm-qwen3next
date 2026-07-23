# Benchmark results

Single MI210 (gfx90a, 64 GB HBM2e). Model: Qwen3-Next-80B-A3B-Thinking, 4-bit
compressed-tensors (`int4_w4a16`). `--max-model-len 16384`, `--gpu-memory-utilization 0.90`.
One 64-token warmup request before each measurement (triggers first-request Triton JIT).

Generation throughput = completion tokens / generation wall time (streaming, prompt time
excluded via first-token timing). Aggregate = total completion tokens / wall time across N
concurrent streams, 256 max_tokens each.

## Single-stream

| vLLM | run 1 | run 2 |
|------|------:|------:|
| 0.11.2 (with 2 env workarounds) | 33 t/s | — |
| **0.23 prebuilt** | 52.7 | 52.8 |
| 0.25.2 self-built | 53.1 | 53.1 |

## Concurrent (aggregate tok/s)

| vLLM | 16 streams | 32 streams |
|------|-----------:|-----------:|
| 0.11.2 | ~100 | ~100 |
| **0.23 prebuilt** | 225.6 | 346.2 |
| 0.25.2 self-built | 225.8 | 358.9 |

## Reference points

- `llama.cpp` (ROCm 7.14 build, same model Q5_K_M, GPU-hybrid): ~68 t/s single-stream.
  Faster than vLLM for one stream; vLLM wins decisively under concurrency.

## Notes

- **0.25.2 ≈ 0.23.** The performance PRs for this model (GDN fusion #40711, Fused Shared
  Expert #39280) landed in v0.21, so both images already have them. The real jump was from
  0.11.2 → ≥0.21.
- The vLLM per-window log line (e.g. "Avg generation throughput: 12.7 tokens/s") averages over
  idle time and understates the per-request rate — trust the client-side measurement.
- Attention runs on the Triton paged path (the ROCm custom C++ paged kernel can't take the
  544-token block); aiter is unavailable on gfx90a.
