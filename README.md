# Running Qwen3-Next-80B on AMD Instinct MI210 (gfx90a / CDNA2) with vLLM

Getting **Qwen3-Next-80B-A3B** (Gated DeltaNet linear-attention hybrid MoE, 512 experts)
to run — and run *fast* — on an **AMD Instinct MI210** (`gfx90a` / CDNA2) under ROCm with vLLM.

This repo documents a working setup, benchmarks, the from-source build recipe for a newer
vLLM on gfx90a, and the gotchas found along the way. TL;DR: **newer vLLM just works on the
MI210 now**, and the AMD-prebuilt CDNA image gets you there with no build at all.

> Hardware: 2× MI210 (64 GB HBM2e each), EPYC 74F3, ROCm 7.14. Model served: the
> Qwen3-Next-80B-A3B Thinking variant, 4-bit (compressed-tensors int4_w4a16).

---

## The short version

| Config | single-stream | 32 concurrent (agg) | KV cache | how to get it |
|------|--------------:|--------------------:|--------------------:|---------------|
| 0.11.2 (old) | 33 t/s | ~100 t/s | 10 GB | *crashed on first gen until 2 env workarounds* |
| **0.23** prebuilt (1 card) | 52.8 t/s | 346.2 t/s | 10 GB | `docker pull` the AMD prebuilt CDNA image — no build |
| 0.25.2 self-built (1 card) | 53.1 t/s | 358.9 t/s | 10 GB | build from source on gfx90a (recipe below) |
| + hand-tuned MoE config | 51.6 t/s | 425.4 t/s | 10 GB | free — a hand-written config JSON, no autotune |
| **TP=2 both cards + tuned config** | **55.1 t/s** | **620.5 t/s** | **32 GB** | `scripts/launch-tp2.sh` — the recommended setup |
| **+ fp8 KV cache** | 53.3 t/s | ~neutral | 32 GB | 11× KV capacity (388K→4.4M tokens) for ~3% single-stream |

*Qwen3-Next-80B-A3B-Thinking int4, 16K context. Aggregate scales to **~2340 t/s at 256 concurrent**
with TP=2. See [findings/benchmarks.md](findings/benchmarks.md) for the full matrix.*

### Best deployment (what we landed on)

**TP=2 across both MI210s + hand-tuned MoE config + fp8 KV cache** → `scripts/launch-tp2.sh`.
The two levers that mattered: a **newer vLLM** (≥0.23) and **tensor-parallel across both cards**
(aggregates HBM bandwidth for the memory-bound MoE — wins even over a PCIe-only link). The MoE
config is free (hand-written, no autotune). Everything else — aiter, Expert Parallel, NCCL tuning,
TurboQuant, LMCache, int8/fp8 requant, speculative decoding — was a measured dead end on this
hardware/model; see **[findings/dead-ends.md](findings/dead-ends.md)**.

### Documentation index
- **[findings/optimization-journey.md](findings/optimization-journey.md)** — the full story + tuning matrix
- **[findings/benchmarks.md](findings/benchmarks.md)** — every measured number
- **[findings/dead-ends.md](findings/dead-ends.md)** — what didn't work, and why (aiter, EP, NCCL, TurboQuant, LMCache, spec-dec, int8, fp8-mamba)
- **[findings/aiter.md](findings/aiter.md)** — the aiter deep-dive (triangulated three ways)
- **[findings/model-alternatives.md](findings/model-alternatives.md)** — standard-attention MoE swaps + CPU-hybrid frontier-MoE direction (llama.cpp)
- **[configs/](configs)** — the hand-tuned MoE config JSONs (single-card `N=512` + TP=2 `N=256`)
- **[moe-tuning/](moe-tuning)** — patches to run `benchmark_moe.py --tune` single-GPU on ROCm

**Takeaways**
- The old `ValueError: arange's range must be a power of 2` crash (from the non-power-of-2
  544-token attention block the hybrid Mamba/GDN model forces) is **fixed in vLLM ≥ 0.21**.
- The perf that matters for this model — **GDN linear-attention fusion for ROCm**
  ([vllm#40711](https://github.com/vllm-project/vllm/pull/40711)) and **Fused Shared Expert
  for Qwen3-Next** ([vllm#39280](https://github.com/vllm-project/vllm/pull/39280)) — landed in
  **v0.21**, so any ≥0.21 image has it. **0.25.2 ≈ 0.23**; the jump was leaving 0.11.2.
- vLLM's advantage on the MI210 is **concurrency (~3.6× aggregate)**. Single-stream (≈53 t/s)
  still trails a tuned `llama.cpp` (~68 t/s). One user at a time → llama.cpp; many → vLLM.
- **aiter** (AMD's optimized kernels) now **builds and runs** on gfx90a/ROCm 7.14 (it JIT-compiles
  the core module and self-skips unsupported flags — a real change from the old stack where it
  crashed/wouldn't compile). **But it gives ~no gain for this int4 model**, because its heavy
  kernels don't apply on gfx90a: aiter **MoE** doesn't support the model's int4 compressed-tensors
  (WNA16) quant (it's fp8/fp4), so vLLM keeps the Triton MoE; and aiter **attention** isn't even an
  offered backend on gfx90a (candidates are `ROCM_ATTN`/`TRITON_ATTN` only). Only aiter **RMSNorm**
  engages → ~1%, and forcing the MoE/attention switches slightly *hurts* concurrency. **Keep
  `VLLM_ROCM_USE_AITER=0`.** See [findings/aiter.md](findings/aiter.md). To actually use aiter's MoE
  you'd need an **fp8** model (which for 80B needs ~2× MI210 via tensor-parallel) or MI300-class hardware.

---

## Fastest path — just pull the prebuilt image

AMD publishes prebuilt `rocm/vllm` images per architecture family. The **CDNA** tag covers
gfx90a (MI210/MI250):

```bash
docker pull rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0
```

Then launch (see [`scripts/launch-vllm023.sh`](scripts/launch-vllm023.sh)):

```bash
docker run -d --name vllm --network host \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined --shm-size 16g \
  -e HIP_VISIBLE_DEVICES=0 -e VLLM_ROCM_USE_AITER=0 \
  -v /path/to/models:/models \
  --entrypoint vllm \
  rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0 \
  serve /models/qwen3-next-80b-a3b-int4 \
  --served-model-name qwen3-next --max-model-len 16384 \
  --gpu-memory-utilization 0.90 --port 8000
```

Notes:
- `--group-add video --group-add render` — match your host's `video`/`render` gids
  (e.g. `--group-add 44 --group-add 991`).
- `VLLM_ROCM_USE_AITER=0` is required on gfx90a (aiter's MoE asm crashes and its attention
  won't JIT-compile there).
- The model's hybrid Mamba/GDN layers make vLLM set a **544-token attention block**
  (non-power-of-2). That is handled cleanly in ≥0.21; you'll see it in the logs, not a crash.

---

## Building a newer vLLM for gfx90a (from source)

Only needed if you want a tag newer than the latest prebuilt. The trick that makes this a
**~15-minute build instead of ~3 hours**: build *on top of* the prebuilt image so you **reuse
its torch 2.11 + ROCm 7.14 + aiter stack** and only recompile vLLM's own kernels.

See [`images/Dockerfile.vllm-gfx90a`](images/Dockerfile.vllm-gfx90a):

```dockerfile
FROM rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0
ARG VLLM_TAG=v0.25.1
ENV PYTORCH_ROCM_ARCH=gfx90a VLLM_TARGET_DEVICE=rocm MAX_JOBS=24 CMAKE_BUILD_PARALLEL_LEVEL=24
WORKDIR /build
RUN git clone --depth 1 --branch ${VLLM_TAG} https://github.com/vllm-project/vllm.git
WORKDIR /build/vllm
RUN python use_existing_torch.py                         # keep the image's torch, drop the pin
RUN pip install --no-build-isolation -e . -v             # compiles csrc for gfx90a
# GOTCHA: vLLM 0.25.x downgrades `tensorizer` to 2.10.1, which is broken on Python 3.14
# (KeyError: 'crypto_stream_salsa20_NONCEBYTES'). Restore the working version:
RUN pip install --no-cache-dir 'tensorizer==2.12.1'
RUN python -c "import vllm, tensorizer; print('vllm', vllm.__version__, '| tensorizer', tensorizer.__version__)"
```

Build:

```bash
docker build --build-arg VLLM_TAG=v0.25.1 -t vllm-gfx90a:0.25 images/
```

The compile emits `... warnings generated when compiling for gfx90a` and builds the HIP
objects (including the mamba `selective_scan_fwd` and `skinny_gemms` kernels) — that's success.

---

## Benchmarks

Two tiny, dependency-free scripts (Python stdlib only) against the OpenAI-compatible endpoint:

- [`scripts/bench.py`](scripts/bench.py) — single-stream generation tok/s (streaming, excludes
  prompt time via first-token timing).
- [`scripts/bench_concurrent.py`](scripts/bench_concurrent.py) — aggregate tok/s under N
  concurrent streams.

```bash
python3 scripts/bench.py 8000 qwen3-next 256           # single-stream
python3 scripts/bench_concurrent.py 8000 qwen3-next 32 256   # 32 concurrent
```

Raw numbers in [`findings/benchmarks.md`](findings/benchmarks.md).

---

## MoE tuning — attempted, and why it's usually not worth it

vLLM warns there's no pretuned fused-MoE config for the MI210 at this model's shape
(`E=512, N=512, dtype=int4_w4a16`) and falls back to a generic (sub-optimal) config.
`benchmark_moe.py --tune` can generate one, but on gfx90a it's a **poor trade**:

- **~17 minutes *per batch size*** — dominated by Triton *compiling* a fresh kernel for each
  of hundreds of configs, not the benchmark loop the progress bar shows.
- Search space **explodes with batch size**: ~608 configs at batch-16 → **4496** at batch-128.
- Expected payoff only **≤ ~15%**, and only on MoE-bound (large-batch) throughput — the batch
  sizes that matter for interactive serving are **1–32** (single-stream = 1; decode batch ≈
  concurrency).

Two gotchas make the tuner even run on a single-GPU ROCm box — both provided as patch scripts:

- [`moe-tuning/patch_ray_amd_gpu.py`](moe-tuning/patch_ray_amd_gpu.py) — Ray's AMD accelerator
  manager raises *"Please use HIP_VISIBLE_DEVICES instead of ROCR_VISIBLE_DEVICES"* inside its
  workers; patch self-heals instead of raising.
- [`moe-tuning/patch_benchmark_moe.py`](moe-tuning/patch_benchmark_moe.py) — bypasses Ray
  entirely (no benefit on one GPU) and runs the tuner in-process, keeping `HIP_VISIBLE_DEVICES`
  so torch sees the card. Also swaps `ray_tqdm` for plain `tqdm`.

See [`moe-tuning/README.md`](moe-tuning/README.md) for how to run it if you still want to.

---

## Environment reference

- **Card**: AMD Instinct MI210, `gfx90a` (CDNA2), 64 GB HBM2e
- **ROCm**: 7.14 (via the prebuilt image)
- **torch**: 2.11.0+rocm7.14 · **vLLM**: 0.23.1 (prebuilt) or self-built 0.25.2 · **Python** 3.14
- **Model**: Qwen3-Next-80B-A3B (Thinking), 4-bit compressed-tensors (`int4_w4a16`)

## License

MIT — see [LICENSE](LICENSE). Findings and scripts only; vLLM, Ray, and Qwen are their
respective projects' works under their own licenses.
