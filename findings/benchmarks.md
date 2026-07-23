# Benchmark results

All on 2× MI210 (gfx90a, 64 GB each), ROCm 7.14, vLLM 0.25.2 (self-built for gfx90a), PCIe-linked.
Model: Qwen3-Next-80B-A3B-Thinking, int4 compressed-tensors (group_size 32).

Generation throughput = completion tokens / generation wall time (streaming, prompt time excluded).
Aggregate = total completion tokens / wall time across N concurrent streams. One 64-token warmup
before each measurement. Scripts: [`scripts/bench.py`](../scripts/bench.py),
[`bench_concurrent.py`](../scripts/bench_concurrent.py), [`bench-all.sh`](../scripts/bench-all.sh),
[`bench_prefix.py`](../scripts/bench_prefix.py).

> **Note on two harnesses:** early runs used 256-token generations; the tuning matrix used a
> stricter 128-token harness (`bench-all.sh`). Absolute aggregate numbers differ between them —
> compare within a harness, not across.

## Version + parallelism progression (256-token gen)

| Config | single | agg@16 | agg@32 | KV | cards |
|---|--:|--:|--:|--:|--:|
| vLLM 0.11.2 (old ceiling) | 33 | ~100 | ~100 | 10 GB | 1 |
| vLLM 0.23 prebuilt | 52.8 | 225.6 | 346.2 | 10 GB | 1 |
| vLLM 0.25.2 self-built | 53.1 | 225.8 | 358.9 | 10 GB | 1 |
| + hand-tuned MoE config | 51.6 | 248.5 | 425.4 | 10 GB | 1 |
| **TP=2 default** | 57.0 | 318.0 | 570.5 | 32 GB | 2 |
| **TP=2 + tuned MoE config** | 55.1 | — | 620.5 | 32 GB | 2 |
| TP=2 + tuned + prefix caching | 55.1 | — | 517* | 32 GB | 2 |

*prefix caching costs ~17% aggregate on cache **misses**, but gives −67% TTFT on hits (below).

## Tuning matrix (128-token gen, TP=2 base)

| Config | single | agg@32 | agg@128 | agg@256 | verdict |
|---|--:|--:|--:|--:|---|
| **baseline (num_warps=4)** | 55.1 | 433.8 | 1375.8 | 2342.5 | **optimal** |
| fp8 KV cache | 53.3 | 428.0 | 1381.5 | 2336.3 | neutral, +11× capacity |
| num_warps=8 | 55.3 | 353.7 | 943.2 | 1798.7 | worse |
| max-num-batched-tokens 32k | 54.9 | 444.4 | 1347.5 | 2374.9 | neutral |
| fp8 mamba state | — | — | — | — | unsupported |

**Aggregate scales hard with concurrency** (32 GB KV): agg@32 ≈ 434 → agg@128 ≈ 1376 →
**agg@256 ≈ 2342 t/s**. Single-stream (~55) is the weak axis, capped by PCIe TP all-reduce.

## fp8 KV cache — capacity

| | KV capacity | max concurrency @16K | single-stream |
|---|--:|--:|--:|
| bf16 KV (default `auto`) | 388K tokens | 24× | 55.1 |
| **fp8 KV** | **4.4M tokens** | **268×** | 53.3 (−3%) |

Only affects the 12 full-attention layers; the 36 GDN layers keep bf16 state.

## Prefix caching (shared 600-token prefix)

| | TTFT |
|---|--:|
| cold (no cache) | 0.246 s |
| **warm (prefix cached)** | **0.082 s (−67%)** |

40% cache-hit rate observed on the shared-prefix test. Auto-enables experimental mamba `align` mode.

## Reference points

- `llama.cpp` (ROCm 7.14, same model Q5_K_M, GPU-hybrid): **~68 t/s single-stream** — still the
  single-stream winner. vLLM's advantage is concurrency (aggregate scales to 2342+ t/s).
- Aggregate at 256 concurrent (fp8 KV, 128-tok): **~2336 t/s**.

## What didn't move the needle

Expert Parallel (worse on PCIe), NCCL-LL protocol (negative), aiter (dead on gfx90a int4),
TurboQuant (unmerged/MI300-only), int8 requant (no CDNA2 speedup), fp8 mamba state (unsupported),
TunableOp (impractically slow). Full analysis in [dead-ends.md](dead-ends.md).
