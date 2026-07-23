# llama.cpp vs vLLM on 2× MI210, and the CPU-hybrid frontier path

Two things vLLM can't do that llama.cpp can on this box: (1) win single-stream, and (2) run
frontier-size MoEs (>128 GB) by spilling experts to the 499 GB RAM. Measured below.

## Round 1 — same model, same quant (apples-to-apples)

**Stock** `Qwen3-Next-80B-A3B-Thinking`, ~4-bit both sides (vLLM int4 / llama.cpp Q4_K_M), 16K ctx:

| | single | agg@16 | agg@32 | agg@256 |
|---|--:|--:|--:|--:|
| **llama.cpp** (1 card, Q4_K_M) | **71** | 193 | 182 (flat) | — |
| **vLLM** TP=2 (int4) | 55 | 226 | **620** | **~2340** |

- **llama.cpp wins single-stream by +29%** (71 vs 55) — better single-sequence path, no TP overhead.
- **vLLM wins concurrency decisively** — 3.4× at 32 streams, and it *scales* to ~2340 t/s at 256
  concurrent, while **llama.cpp is flat ~185** regardless of concurrency (weak batching).
- The quant was *not* the confound: llama.cpp Q4 (71) ≈ Q5 (72) — the 3B-active MoE is fast either
  way, and KV-q8_0 dominates. (Earlier round with abliterated-Q5 vs stock-int4 gave the same story;
  abliteration doesn't affect speed.)

**Verdict: one interactive user → llama.cpp; many concurrent → vLLM.**

## Round 2 — 235B frontier via CPU-hybrid (vLLM *cannot run this*)

`Qwen3-235B-A22B-Thinking-2507` Q5_K_M is **155 GB > 128 GB VRAM** — vLLM can't load it. llama.cpp's
`-ncmoe N` keeps attention + N-fewer layers' experts on the GPUs and parks the rest of the experts
in the **499 GB RAM**. Tuning the split (single-stream):

| config | single | note |
|---|--:|---|
| `-ncmoe 60, -ts 1,1` | 12.8 | first fit |
| `-ncmoe 50, -ts 3,2` | 14.4 | rebalanced split (card 1 was overloaded) |
| **`-ncmoe 44, -ts 3,2`** | **15.1** | best (layer split) |

Key tuning notes:
- `-ncmoe` too low → **OOM on one card** (~78 GB); the fix was **rebalancing `-ts` to 3,2** (60/40)
  to unload the overloaded card, which let more experts sit on GPU.
- `--no-mmap` is needed (the log recommends it for CPU tensor overrides; it loads the model into
  RAM properly instead of mmap-faulting the CPU experts).
- **Multi-GPU caveat:** this uses native layer split, which risks the peer-DMA fault under
  concurrency (see [multi-gpu-llamacpp.md](multi-gpu-llamacpp.md)). Single-stream was fine; for
  concurrent serving run the 235B via **RPC** instead.

**The 235B runs at ~15 t/s** — ~4.7× slower than the 80B's 71, but a much more capable model, and a
class of model **vLLM physically cannot serve on this box**. Slow-but-smart for deep single-user
thinking.

## The three-tier picture for this box

| Need | Use |
|---|---|
| Max concurrency / aggregate throughput | **vLLM TP=2** (Qwen3-Next-80B int4) — ~2340 t/s @256 |
| Fastest single-user, model fits in VRAM | **llama.cpp single-card** (80B Q4/Q5) — ~71 t/s |
| Smartest model, exceeds VRAM | **llama.cpp CPU-hybrid** (235B+/DeepSeek/Kimi) — ~10–15 t/s |
