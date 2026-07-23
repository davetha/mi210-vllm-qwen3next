# MoE tuning on the MI210 (gfx90a)

vLLM logs a warning that there's no pretuned fused-MoE config for the MI210 at Qwen3-Next's
shape and uses a generic (sub-optimal) one:

```
Using default MoE config. Performance might be sub-optimal!
Config file not found at .../configs/E=512,N=512,device_name=AMD_Instinct_MI210,dtype=int4_w4a16.json
```

`benchmark_moe.py --tune` generates one. **Whether it's worth it: usually not.** See the cost
analysis in the top-level README — ~17 min *per batch size*, search space explodes with batch
size, ≤15% payoff, and only on large-batch (high-concurrency) throughput.

## Running it

Two patches are needed for the tuner to run at all on a single-GPU ROCm box:

```bash
# inside a container from the vLLM image, with the GPU attached and HIP_VISIBLE_DEVICES set:
python patch_ray_amd_gpu.py    /opt/python/lib/python3.14/site-packages/ray/_private/accelerators/amd_gpu.py
python patch_benchmark_moe.py  /app/vllm/benchmarks/kernels/benchmark_moe.py

python /app/vllm/benchmarks/kernels/benchmark_moe.py \
  --model /models/qwen3-next-80b-a3b-int4 \
  --dtype int4_w4a16 --tp-size 1 --tune \
  --batch-size 1 2 4 8 16 24 32 \
  --save-dir /out
```

- Prefer bind-mounting the patched copies over the originals (read-only) instead of editing
  in place, so the base image stays clean.
- **Restrict `--batch-size` to what you actually serve** (1–32 covers single-stream and up to
  ~32 concurrent). The large batch sizes (128–4096) dominate tune time for near-zero real-world
  benefit — batch 128 alone is ~4496 configs.
- The tuned JSON lands in `--save-dir` as
  `E=512,N=512,device_name=AMD_Instinct_MI210,dtype=int4_w4a16.json`. Drop it into vLLM's
  `.../model_executor/layers/fused_moe/configs/` (bind-mount it — see
  `scripts/launch-vllm-gfx90a.sh`).

## Splitting across two GPUs

The patched tuner is serial (one GPU). To use two cards, run two containers pinned to
different `HIP_VISIBLE_DEVICES`, each with a disjoint `--batch-size` list and its own
`--save-dir`, then merge the two JSONs (each is a `{batch_size: config}` dict — a plain
`dict.update()` merge). Note the search-space imbalance: give the small (fast) batch sizes and
the large (slow) ones a balanced split, not a naive half-and-half.
