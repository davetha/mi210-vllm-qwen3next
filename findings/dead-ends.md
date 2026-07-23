# Dead ends — what does NOT help Qwen3-Next on MI210 (gfx90a), and why

Every one of these was investigated (often at the source-code level) and/or measured. Recording
them so nobody re-treads the ground. The common theme: **CDNA2 (gfx90a) lacks fp8/fp4 matrix
hardware, and the Gated-DeltaNet (GDN) hybrid architecture breaks tools built for standard KV.**

## aiter (AMD optimized kernels) — comprehensively dead for int4 on gfx90a
See [aiter.md](aiter.md) for the full write-up. Summary of the triangulated verdict:
- vLLM gates all aiter features (except RMSNorm) behind `on_mi3xx()` = **gfx942/gfx950 only**.
- aiter's only int4 MoE kernel is FlyDSL `a16wi4` (group_size 32), tuned for **gfx950 only**;
  its fast int8/fp8 MoE kernels ship as **gfx942/gfx950 asm binaries with no source**.
- The gfx908 fork measured aiter **−6% to −26% vs native Triton** on CDNA and recommends
  `VLLM_ROCM_USE_AITER=0`.
- Empirically: master `AITER=1` gave ~1% (RMSNorm only); forcing `AITER_MOE`/attention was
  neutral-to-negative; forcing `ROCM_AITER_UNIFIED_ATTN` **crashed the engine** on gfx90a.
- **The int8-CK porting path** (requant to int8, patch aiter build maps + vLLM gate) is ~35–45%
  likely to even compile — and would likely still lose, because **CDNA2 int8 MFMA runs at the
  same throughput as bf16** (the 2× int8 advantage is CDNA3+) while doubling memory traffic.

**Keep `VLLM_ROCM_USE_AITER=0`.**

## Expert Parallel (`--enable-expert-parallel`) — worse on PCIe
EP shards the 512 experts (256/card) and uses all-to-all routing instead of TP's all-reduce.
AMD's guidance favors EP for ultra-sparse MoE — **but that assumes xGMI**. Over our PCIe-only
link the all-to-all shuffle costs more than it saves: agg@32 **476 vs 620** (TP), equal at the
extremes. Stick with plain TP=2.

## NCCL/RCCL `LL` protocol — negative
`NCCL_PROTO=LL` (low-latency protocol) to cut the PCIe all-reduce latency: no single-stream
gain and it **capped bandwidth**, hurting mid-concurrency (agg@32 520 vs 620). Reverted.

## Speculative decoding — broken by the GDN architecture
- **ngram / EAGLE / draft-model** all need KV/state rollback on rejection. Qwen3-Next's GDN
  layers advance a recurrent state that vLLM **cannot roll back** → corrupted output
  ([vLLM #39273](https://github.com/vllm-project/vllm/issues/39273)).
- **Native MTP** (`qwen3_next_mtp`) is the only correct method but is documented as a **76%
  latency regression** ([#35387](https://github.com/vllm-project/vllm/issues/35387)).
- This is architecture-level, not gfx90a-specific. Spec-dec would work on a **standard-attention
  MoE** — see [model-alternatives.md](model-alternatives.md). Track the ReplaySSM RFC (#47572).

## int8 / fp8 weight or KV quant — hardware says no
- **fp8 anything**: CDNA2 has **no fp8 matrix unit** (added in CDNA3/gfx942). fp8 KV *storage*
  works (dequantized on read — see optimization-journey), but fp8 *compute* does not.
- **int8 w8a8**: CDNA2 has int8 MFMA, but at the **same throughput as bf16** (no speedup) while
  doubling bytes vs int4 — a net loss at the memory-bound batch-1 regime.
- **int4 MFMA**: does not exist on CDNA2 at all — the current int4 path already dequantizes to
  bf16 before the matmul; the only int4 benefit is halved HBM bytes.

## fp8 Mamba/GDN state — not supported (on purpose)
`--mamba-ssm-cache-dtype` only accepts `auto/bfloat16/float16/float32` — **no fp8**. The GDN
state is a recurrent accumulator; fp8 error would compound across the sequence and corrupt
output. So the 75% of layers that are GDN can't be shrunk below fp16. (float16 = same 2 bytes as
bf16, no win; float32 = bigger.)

## LMCache — fails to start + no gfx90a support
- **Qwen3-Next + TP=2 + LMCache literally fails to start**: *"Hybrid KV cache manager is disabled
  but failed to convert the KV cache specs to one unified type"*
  ([LMCache #2927](https://github.com/LMCache/LMCache/issues/2927) — closed "not planned";
  [vLLM #38700](https://github.com/vllm-project/vllm/issues/38700)). GLM hybrids work, so it's
  Qwen3-Next-architecture-specific.
- **ROCm support is gfx942/gfx950 only** — MI210/gfx90a appears nowhere in LMCache's ROCm docs.
- Better alternative for "use the 499 GB RAM as a bigger cache": vLLM's own **native CPU-offload
  connector** (in-tree, same codebase). Or the **CPU-hybrid frontier-MoE** direction (llama.cpp),
  see [model-alternatives.md](model-alternatives.md).

## TurboQuant — a KV quantizer that barely applies here
- It's a KV-cache quantizer (WHT rotation + Lloyd-Max, 3–8 bit) bundled as an attention backend,
  opt-in via `--kv-cache-dtype turboquant_*`. It was "excluded" in our logs simply because we
  never requested it — not a hardware/model block.
- Not worth forcing: all ROCm kernel work targets **MI300X/MI355X and is unmerged** (PRs
  #40393/#40396/#41597); hybrid bugs are open (#40807 crashes CUDA-graph capture on Qwen3-Next);
  and it only touches the **12/48 attention layers** — the GDN state (bulk of memory) is untouched.
- For KV capacity, plain `--kv-cache-dtype fp8` (which we use) is the lower-risk, already-working
  lever.

## MoE Triton autotune (benchmark_moe.py --tune) — too slow for the payoff
~**17 min per batch size** (Triton compile-bound), search space explodes with batch size
(608 configs @bs16 → 4496 @bs128). The hand-written config captures nearly all of it because
int4's real search space is tiny. Ray also needs patching to run single-GPU on ROCm (see
[../moe-tuning/](../moe-tuning)). Not worth the multi-hour cost.

## torch.compile max-autotune, PyTorch TunableOp — impractical / marginal
- **TunableOp** (`PYTORCH_TUNABLEOP_ENABLED=1`): tunes only dense `torch` GEMMs, not this model's
  Triton MoE/GDN kernels. Its online tuning across the 51 cudagraph-capture sizes ran **>18 min
  and never finished** in testing. Killed.
- **torch.compile `max-autotune`** on ROCm: hang risk, and it only touches the tiny non-Triton
  glue. Marginal.

## Skinny-GEMM — already on
`VLLM_ROCM_USE_SKINNY_GEMM` defaults to `True` on gfx9 — you're already getting it.
