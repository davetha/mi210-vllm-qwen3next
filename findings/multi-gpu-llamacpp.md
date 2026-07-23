# Multi-GPU with llama.cpp on 2× MI210 (gfx90a) — splitting methods + the peer-DMA fault

Running a model across both MI210s in llama.cpp is a minefield on this hardware. Summary:
**native tensor-split crashes or is unsupported; RPC is the only stable multi-GPU path, at ~24% cost.**

Measured on Qwen3-Next-80B-A3B-Thinking Q4_K_M (fits on one card), FA + q8_0 KV, cont-batching.

## The results

| Method | single | agg@32 | stable? |
|---|--:|--:|---|
| **Single card** (`--split-mode none`, 1 GPU) | **71** | **182** | ✅ fastest + stable |
| Native **layer** split (`--split-mode layer -ts 1,1`) | — | 💥 crash | ❌ GPU memory access fault under load |
| Native **row** split (`--split-mode row`) | — | — | ❌ unsupported |
| **RPC** (`ggml-rpc-server` per card + `--rpc`) | 53.7 | 153.9 | ✅ stable, ~24% slower |

## The three failure/success modes

### 1. Native layer split → GPU memory access fault
`-ngl 999 --split-mode layer -ts 1,1` loads fine and generates for a while, then crashes under
load:
```
amdgpu 0000:c3:00.0: [mmhub0] no-retry page fault (src_id:0 ring:144 vmid:3 ...)
llama-server: .../runtime.cpp:2026: rocr::core::Runtime::VMFaultHandler ... "GPU memory access fault"
```
This is a **multi-GPU peer-DMA software bug** on CDNA2/ROCm — it strikes whichever card is the
secondary peer-reader when there's heavy GPU↔GPU tensor traffic. It surfaces under **concurrent
load** (it crashed during a 32-stream benchmark, after ~150 tokens). Not a hardware fault (cards
recover); it's the peer-copy path.

### 2. Native row/tensor split → unsupported
`--split-mode row` fails immediately at load:
```
llama_model_load: error loading model: device ROCm0 does not support split buffers
```
The ROCm/HIP ggml backend doesn't implement the split-buffer path row-mode needs. Not available
on gfx90a, period.

### 3. RPC → stable (the workaround)
llama.cpp's RPC backend layer-splits over **host TCP sockets** between separate per-card server
processes — **no direct GPU↔GPU peer DMA**, so it sidesteps the fault. It ran the 32-stream
concurrent test with zero crashes. Cost: **~24% slower** than single-card (socket serialization
per layer + the model didn't need two cards anyway).

## RPC setup (reproducible)

The stock llama.cpp build here was compiled **without** `GGML_RPC` (no `--rpc` flag, no
`ggml-rpc-server` binary). Rebuild with it on:
```bash
cd <llama.cpp build dir>
cmake . -DGGML_RPC=ON
cmake --build . --target ggml-rpc-server llama-server -j
```

Then one RPC backend per card (pin with `HIP_VISIBLE_DEVICES`), plus the main server:
```bash
# card 0 backend
HIP_VISIBLE_DEVICES=0 ggml-rpc-server -H 0.0.0.0 -p 50052 &
# card 1 backend
HIP_VISIBLE_DEVICES=1 ggml-rpc-server -H 0.0.0.0 -p 50053 &
# main server (NO local GPU — it delegates to the RPC backends)
llama-server -m model.gguf --rpc 127.0.0.1:50052,127.0.0.1:50053 -ngl 999 \
  -fa on -ctk q8_0 -ctv q8_0 --host 0.0.0.0 --port 8000
```

## Recommendation

- **Model fits on one card → run single-card.** It's both fastest and stable. Splitting a
  fitting model across both cards only adds overhead + the crash risk. (This is the 80B case.)
- **Model needs both cards** (e.g. a >64 GB model, or the 235B CPU-hybrid case) → **use RPC**, not
  native tensor-split, so concurrent load doesn't hit the peer-DMA fault. Accept the ~24% cost.
- **Never use `--split-mode row` on gfx90a** — unsupported.

See also [dead-ends.md](dead-ends.md) (vLLM side) and the session's earlier finding that this same
peer-DMA fault (the `c3 ... SDMA1` VM protection fault) is a software multi-GPU bug, not hardware.
