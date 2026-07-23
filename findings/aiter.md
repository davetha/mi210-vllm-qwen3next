# aiter on the MI210 (gfx90a) — does it help?

**AMD's aiter kernels are tuned for MI300-class (gfx942) hardware and fp8/fp4 quant.**
On the MI210 (gfx90a) with an int4 (compressed-tensors WNA16) model, the answer is: aiter now
*runs*, but it doesn't move the needle.

## The old vs. new situation

- **Old stack (vLLM 0.11.2 / ROCm 7.0):** aiter's MoE asm kernel hard-crashed and its attention
  wouldn't JIT-compile on gfx90a — hence the long-standing "keep `VLLM_ROCM_USE_AITER=0`" advice.
- **New stack (vLLM 0.23–0.25.2 / ROCm 7.14):** aiter's core module **JIT-builds fine** on gfx90a
  (it even auto-skips an unsupported hipcc flag: `-mllvm -amdgpu-coerce-illegal-types=1`) and imports.
  So aiter is no longer dead on this hardware.

## But the heavy kernels don't engage for an int4 model

Tested on the self-built 0.25.2 image, Qwen3-Next-80B-A3B int4 (compressed-tensors), single MI210:

| Config | single-stream | 32 concurrent (agg) |
|--------|--------------:|--------------------:|
| aiter **off** (baseline) | 53.1 t/s | 358.9 t/s |
| `VLLM_ROCM_USE_AITER=1` (master) | 54.0 | 356.7 |
| `+ AITER_MOE=1 + AITER_FUSION_SHARED_EXPERTS=1` | 54.2 | 328.7 |
| `+ AITER_MHA=1 + AITER_UNIFIED_ATTENTION=1 + AITER_PAGED_ATTN=1` | 53.4 | 327.2 |

What the logs show:
- **MoE**: even with `VLLM_ROCM_USE_AITER_MOE=1` forced, vLLM logs `moe_backend='auto'` and
  `Using CompressedTensorsWNA16MoEMethod` — the **Triton** MoE. aiter's MoE kernels are fp8/fp4;
  they don't support this model's int4 WNA16 quant, so vLLM keeps the Triton path.
- **Attention**: the backend selector reports candidates `['ROCM_ATTN', 'TRITON_ATTN']` — **no aiter
  attention backend is offered on gfx90a** — and further logs `Cannot use ROCm custom paged attention
  kernel, falling back to Triton implementation`.
- **RMSNorm**: the one thing that does engage (`rms_norm=['aiter','native']`) → ~1%, within noise.
- Forcing the MoE/attention switches **slightly reduced** concurrent throughput (overhead without
  engaging a faster kernel).

## Conclusion

For **int4 Qwen3-Next on a single MI210**, aiter is not a useful lever — the quant format and the
hardware are both outside what aiter's fast MoE/attention kernels support. Keep `VLLM_ROCM_USE_AITER=0`.

The realistic ways to actually benefit from aiter's MoE:
1. Run an **fp8** (`fp8_w8a8`) checkpoint — aiter MoE supports it. For an 80B model that's ~80 GB,
   so you'd need **tensor-parallel across 2× MI210** (128 GB) to fit.
2. Run on **MI300-class** (gfx942/gfx950) hardware, aiter's actual target.
