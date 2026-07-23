# Optimization journey — Qwen3-Next-80B on 2× MI210 (gfx90a)

The full record of what moved the needle and what didn't, with measured numbers.
Hardware: 2× AMD Instinct MI210 (gfx90a/CDNA2, 64 GB each), ROCm 7.14, vLLM 0.25.2,
**PCIe-linked (no xGMI)**. Model: Qwen3-Next-80B-A3B-Thinking, int4 compressed-tensors
(group_size 32), a Gated-DeltaNet/Mamba + full-attention hybrid MoE (512 experts, 3B active).

## Headline

| Stage | single-stream | agg @32 | notes |
|---|--:|--:|---|
| vLLM 0.11.2 (old, crashed) | 33 t/s | ~100 | needed 2 env workarounds just to generate |
| vLLM 0.23 prebuilt (1 card) | 52.8 | 346 | the version bump was the big unlock |
| + hand-tuned MoE config (1 card) | 51.6 | 425 | +18% aggregate, free |
| **+ TP=2 across both cards** | **55–57** | **570–620** | the biggest lever |
| + fp8 KV cache | 53–55 | ~equal | 11× KV capacity (388K→4.4M tokens) |

*(Aggregate figures use 256-token generations. A second, stricter 128-token harness gives
agg@32≈434, agg@128≈1376, agg@256≈2342 — use those for apples-to-apples across the tuning
matrix below, since generation length shifts the absolute numbers.)*

The two levers that actually mattered: **(1) a newer vLLM** (0.23+, which added the ROCm GDN
fusion + fused-shared-expert kernels — see the main README), and **(2) tensor-parallel across
both MI210s** plus a **hand-written MoE config**. Everything else was neutral or a dead end
(see [dead-ends.md](dead-ends.md)).

## The winning MoE config (free, no autotune)

vLLM warns `Using default MoE config. Performance might be sub-optimal!` because there's no
tuned config for the MI210 at this model's shape. `benchmark_moe.py --tune` generates one but
costs **~17 min per batch size** (Triton compile-bound). We skipped that entirely: for int4
the tuner only searches a tiny space (`matrix_instr_nonkdim`/`kpack` don't apply to int4), so
a **hand-written config** captures nearly all of it. See [configs/](../configs).

Key facts learned the hard way:
- **`SPLIT_K` is required** in the config or the kernel crashes (`TypeError: dynamic_func()
  missing 1 required positional argument: 'SPLIT_K'`). int4 forces `SPLIT_K=1`.
- **`BLOCK_SIZE_K` must divide the K dim** (2048 and 512 here) — `64` works.
- **`num_warps=4` beats `num_warps=8`** on this model — 8 regressed aggregate by ~30% (agg128
  943 vs 1376). More warps hurt the batched MoE regime on gfx90a.
- Under **TP=2 the MoE intermediate N shards 512→256**, so the config filename changes from
  `E=512,N=512,...` (single card) to `E=512,N=256,...` (TP=2). Both are in [configs/](../configs).

Winning per-batch config (both files): `BLOCK_SIZE_M` 16/32/64 by batch bucket, `BLOCK_SIZE_N=64`,
`BLOCK_SIZE_K=64`, `GROUP_SIZE_M=1`, `SPLIT_K=1`, `num_warps=4`, `num_stages=2`, `waves_per_eu=2`.

## TP=2 — the biggest lever (even over PCIe)

MoE is memory-bandwidth-bound; tensor-parallel across both cards aggregates HBM bandwidth for
expert-weight loading. Despite the cards being **PCIe-linked (no xGMI)** — so every layer's
all-reduce pays PCIe latency — TP=2 won on every axis:

| | single | agg@32 | agg@128 | agg@256 | KV cache |
|---|--:|--:|--:|--:|--:|
| 1 card (tuned) | 51.6 | 425 | — | — | 10 GB |
| **TP=2 (tuned)** | **55.1** | **620** | 1376 | 2342 | **32 GB** |

At batch-1 the cross-card traffic is tiny (hidden-state vectors, latency-bound), so bandwidth
aggregation wins net. TP=2 also **triples the KV cache** (10→32 GB) as a side benefit.

## Tuning matrix (128-token harness, TP=2 base)

| Config | single | agg@32 | agg@128 | agg@256 | verdict |
|---|--:|--:|--:|--:|---|
| **baseline (num_warps=4)** | 55.1 | 433.8 | 1375.8 | 2342.5 | **optimal** |
| fp8 KV cache | 53.3 | 428 | 1381 | 2336 | neutral, **+11× capacity** |
| fp8 mamba state | — | — | — | — | **unsupported** (fp16 min) |
| num_warps=8 | 55.3 | 353.7 | 943 | 1799 | **worse** |
| max-num-batched-tokens 32k | 54.9 | 444.4 | 1347.5 | 2374.9 | neutral |
| Expert Parallel (`--enable-expert-parallel`) | 55.3 | 476.6* | — | 811* | **worse** (see dead-ends) |

*EP measured at 256-token gen; still clearly below TP at mid-concurrency.

**Aggregate scales enormously with concurrency** thanks to the 32 GB KV cache:
agg@32 ≈ 434 → agg@128 ≈ 1376 → **agg@256 ≈ 2342 t/s**. The box is a throughput monster at
scale; single-stream (~55) is the weak axis (capped by the PCIe TP all-reduce).

## fp8 KV cache — capacity, not speed

`--kv-cache-dtype fp8` stores attention KV at 8-bit. It only touches the **12 full-attention
layers** (the 36 GDN layers keep fixed-size bf16 state, which can't be quantized — see
dead-ends). Measured effect: **KV capacity 388K → 4.4M tokens, max concurrency 24× → 268×**,
same ~31 GB budget, for a ~3% single-stream cost and no throughput change (we weren't
KV-limited). Worth it for very long context, >256 concurrency, or to reclaim VRAM; otherwise
optional. It's in the final `launch-tp2.sh`.

## Prefix caching — workload-dependent

`--enable-prefix-caching` auto-enables the experimental mamba **`align`** cache mode for
Qwen3-Next. Measured on a 600-token shared prefix: **TTFT 0.246s → 0.082s (−67%)** on cache
hits (40% hit rate observed), but **~17% aggregate cost** at mid-concurrency when there are no
hits (align-mode overhead). Enable for multi-turn / long shared prompts; skip for pure varied
one-shot throughput.

## Reproduce

```bash
scripts/launch-tp2.sh /path/to/qwen3-next-80b-a3b-int4 8000   # TP=2 + tuned config + fp8 KV
scripts/bench.py 8000                                          # single-stream
scripts/bench_concurrent.py 8000 qwen3-next 256 256           # aggregate @256
scripts/bench_prefix.py 8000                                   # prefix-cache TTFT (needs --enable-prefix-caching)
```
