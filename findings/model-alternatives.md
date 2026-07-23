# Model alternatives for 2× MI210 (128 GB VRAM + 499 GB RAM)

Two strategic directions beyond tuning Qwen3-Next-80B in vLLM. Both trade something for the
GDN-hybrid's limitations (no fp8 KV compute, broken spec-dec, no LMCache — see
[dead-ends.md](dead-ends.md)).

Hardware gate that filters everything: **CDNA2/gfx90a has no fp8 or fp4/MXFP4 matrix hardware**,
so MXFP4-native models (e.g. gpt-oss) run degraded/dequantized, and int4 (AWQ/GPTQ/
compressed-tensors) + bf16/int8 are the viable quant paths.

## Direction A — swap to a standard-attention MoE (fits in VRAM, unlocks tooling)

A non-hybrid MoE unlocks fp8 KV *compute*, speculative decoding (EAGLE-3 is confirmed working on
AMD Instinct at **1.69–2×**), mature prefix caching, and LMCache — all of which the GDN hybrid
blocks. Cost: standard attention has a **KV cache that grows with context** (vs GDN's fixed
O(1) state), but with 32–64 GB KV headroom on 2 cards that's affordable.

Ranked for this box (capability × speed × fit):

| Pick | Total/active | Attn | int4 fit | Est. speed | Coding (SWE-bench) |
|---|---|---|---|--:|---|
| **GLM-4.5-Air** | 106B/12B | GQA | ~55 GB — **1 card** | ~15–22 t/s | 57.6% (≈ current) |
| MiniMax-M2-REAP-162B | 162B/10B | full | ~86 GB, TP=2 | ~18–25 t/s | 69–78% (stronger) |
| Qwen3-Coder-30B-A3B | 30B/3B | GQA | ~17 GB — 1 card | ~50–55 t/s | ~50% (keeps speed) |
| Qwen3-235B-A22B-Thinking | 235B/22B | GQA | ~118 GB (no KV room) | ~8 t/s | highest, but slow |

**Ruled out**: gpt-oss (MXFP4, no gfx90a accel), GLM-4.6+/DeepSeek-V3/Kimi (>128 GB at int4),
Mixtral (obsolete), Llama-4 (weak coding).

**Standout = GLM-4.5-Air**: it **fits on one MI210 (TP=1)** → zero PCIe all-reduce penalty (the
exact thing capping our single-stream), and frees the second card for a **draft model** to run
EAGLE/draft speculative decoding. Single-card + spec-dec could plausibly hit **~30–40 t/s
single-stream — faster than Qwen3-Next** despite more active params, purely by escaping the PCIe
TP tax. That's the one path to genuinely better single-user latency on this box.

**Honest caveat**: if fp8-KV / spec-dec / LMCache don't change your workflow, **Qwen3-Next-80B
stays the best raw speed×capability point** — you'd switch for the tooling, not a smarter model.

## Direction B — CPU-expert-offloaded FRONTIER MoE (llama.cpp, not vLLM)

vLLM needs the **whole model in VRAM** (its `--cpu-offload-gb` is dumb weight-streaming). llama.cpp's
`-ncmoe N` / `--override-tensor "exps=CPU"` is purpose-built for the opposite: keep attention +
hot layers on GPU, park the bulk of the (rarely-hit) MoE experts in the **499 GB RAM**. MoE
sparsity (few active experts/token) makes this viable. This unlocks **frontier models vLLM
cannot run on this box**:

| Model | Total/active | Q4 size | Fits 499 GB RAM | Est. single-stream (CPU-hybrid) |
|---|---|---|--:|--:|
| Qwen3-235B-A22B-Thinking | 235B/22B | ~140 GB | ✅ easily | ~15–25 t/s (sweet spot) |
| DeepSeek-V3.1 | 671B/37B | ~380 GB | ✅ | ~8–15 t/s |
| GLM-4.6 | 357B/32B | ~200 GB | ✅ | ~10–18 t/s |
| Kimi-K2 | 1T/32B | ~450 GB (Q3) | ✅ tight | ~8–12 t/s |

With 128 GB VRAM you push a large fraction of experts onto the GPUs (the `-ncmoe` tuning we did
for the 397B earlier), so these run better here than a typical CPU-offload rig.

**Trade-off**: CPU-hybrid = **frontier capability** but CPU-bound single-stream and much lower
concurrency than vLLM full-GPU. Ideal for a single interactive "thinking/coding" user who wants
the smartest answers; poor for high-concurrency serving.

**Tools**: llama.cpp (ROCm-solid, already the single-stream winner at ~68 t/s on the 80B) is the
pragmatic pick. **ktransformers** is specialized for GPU+CPU MoE (AMX/AVX-512, often 2–4× faster)
but is CUDA-first — ROCm support is experimental/uncertain.

**Recommended first test**: Qwen3-235B-A22B-Thinking (~140 GB, 22B active, big fraction fits
on-GPU → likely 15–25 t/s) as the capability-vs-speed sweet spot, or DeepSeek-V3.1 for max
capability at ~10 t/s.

## Deployment idea — run both

The two directions aren't exclusive: vLLM TP=2 Qwen3-Next for **fast/concurrent serving**, and a
llama.cpp CPU-hybrid frontier MoE for **deep single-user thinking** — different roles, same box.
