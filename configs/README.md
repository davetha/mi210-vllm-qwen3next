# Hand-tuned MoE configs for MI210 (gfx90a)

vLLM warns `Using default MoE config. Performance might be sub-optimal!` for Qwen3-Next on the
MI210 because no tuned config ships for this shape. These are **hand-written** configs (no
`benchmark_moe.py --tune` — that costs ~17 min/batch-size and, for int4, searches almost nothing
that matters). They gave **+18% aggregate** vs the default.

## Files
- `E=512,N=512,device_name=AMD_Instinct_MI210,dtype=int4_w4a16.json` — **single-card** (TP=1).
- `E=512,N=256,device_name=AMD_Instinct_MI210,dtype=int4_w4a16.json` — **TP=2** (the MoE
  intermediate `N` shards 512→256, so vLLM looks up the `N=256` file).

`N` = MoE intermediate size after sharding. `E` = number of experts (512). Check the exact
filename vLLM wants in its "Config file not found at …" warning for your parallelism.

## Install
Bind-mount over vLLM's configs dir (path shown is for the editable self-built image; a wheel
install uses `.../site-packages/vllm/...`):

```bash
docker run ... \
  -v "$PWD/E=512,N=256,device_name=AMD_Instinct_MI210,dtype=int4_w4a16.json:\
/build/vllm/vllm/model_executor/layers/fused_moe/configs/E=512,N=256,device_name=AMD_Instinct_MI210,dtype=int4_w4a16.json:ro" \
  ...
```
Confirm pickup in the log: `Using configuration from …E=512,N=256…json for MoE layer.`

## Tuning notes (learned the hard way)
- **`SPLIT_K` is mandatory** or the kernel crashes (`missing … 'SPLIT_K'`). int4 → `SPLIT_K=1`.
- **`BLOCK_SIZE_K` must divide the K dim** (2048 & 512 here) — `64` works.
- **`num_warps=4` beats `num_warps=8`** (8 regressed aggregate ~30%).
- `matrix_instr_nonkdim`/`kpack` do **not** apply to int4 — don't bother setting them.
- Per batch: `BLOCK_SIZE_M` 16/32/64 by bucket, `BLOCK_SIZE_N=64`, `BLOCK_SIZE_K=64`,
  `GROUP_SIZE_M=1`, `SPLIT_K=1`, `num_warps=4`, `num_stages=2`, `waves_per_eu=2`.
