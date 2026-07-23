#!/usr/bin/env bash
# BEST-THROUGHPUT deployment: Qwen3-Next-80B on 2x MI210 (gfx90a) via vLLM.
#   - tensor-parallel=2 across both cards (aggregates HBM bandwidth for the MoE)
#   - hand-tuned MoE config bind-mounted (see configs/E=512,N=256,...json)
#   - fp8 KV cache (attention KV -> ~2x smaller, ~11x more token capacity here)
#
# Measured on 2x MI210 / ROCm 7.14 / vLLM 0.25.2, Qwen3-Next-80B-A3B-Thinking int4:
#   single-stream ~53-55 t/s, aggregate ~430 t/s @32 conc, ~2340 t/s @256 conc, 32GB KV.
#
# Usage: launch-tp2.sh [/path/to/model] [port]
set -euo pipefail
MODEL="${1:-/models/qwen3-next-80b-a3b-int4}"
PORT="${2:-8000}"
IMG="${VLLM_IMAGE:-vllm-gfx90a:0.25}"   # self-built image (see images/Dockerfile.vllm-gfx90a)

# Under TP=2 the MoE intermediate N shards 512->256, so vLLM looks up an N=256 config.
CFGNAME="E=512,N=256,device_name=AMD_Instinct_MI210,dtype=int4_w4a16.json"
CFG_SRC="$(dirname "$0")/../configs/${CFGNAME}"
# Path inside the *editable-install* self-built image; adjust if using a wheel install:
DST="/build/vllm/vllm/model_executor/layers/fused_moe/configs/${CFGNAME}"
VIDEO_GID="$(getent group video  | cut -d: -f3)"
RENDER_GID="$(getent group render | cut -d: -f3)"

MOUNT=()
[ -f "$CFG_SRC" ] && MOUNT=(-v "$(readlink -f "$CFG_SRC"):${DST}:ro")

docker rm -f vllm-tp2 >/dev/null 2>&1 || true
docker run -d --name vllm-tp2 --network host --restart unless-stopped \
  --device /dev/kfd --device /dev/dri \
  --group-add "${VIDEO_GID}" --group-add "${RENDER_GID}" \
  --security-opt seccomp=unconfined --shm-size 32g \
  -e VLLM_ROCM_USE_AITER=0 \
  -v "$(dirname "${MODEL}")":"$(dirname "${MODEL}")" "${MOUNT[@]}" \
  --entrypoint vllm "${IMG}" \
  serve "${MODEL}" --served-model-name qwen3-next \
  --tensor-parallel-size 2 \
  --kv-cache-dtype fp8 \
  --max-num-seqs 512 \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --port "${PORT}"
echo "launched vllm-tp2 (TP=2 + tuned MoE config + fp8 KV) on port ${PORT}"

# NOTES:
#   * Drop --kv-cache-dtype fp8 for ~3% more single-stream at the cost of ~11x less KV headroom.
#   * Add --enable-prefix-caching for multi-turn / long shared prompts (-67% TTFT on hits,
#     but ~17% aggregate cost on cache misses; auto-enables experimental mamba 'align' mode).
#   * aiter MUST stay disabled on gfx90a (VLLM_ROCM_USE_AITER=0) - see findings/aiter.md.
