#!/usr/bin/env bash
# Launch Qwen3-Next-80B on a SELF-BUILT vLLM image for gfx90a (see images/Dockerfile.vllm-gfx90a).
# Optionally bind-mounts a tuned MoE config if one exists (see moe-tuning/).
#
# Usage: launch-vllm-gfx90a.sh [/path/to/model] [port] [hip_device] [image] [tuned_moe_json]
set -euo pipefail

MODEL="${1:-/models/qwen3-next-80b-a3b-int4}"
PORT="${2:-8000}"
HIP_DEV="${3:-0}"
IMG="${4:-vllm-gfx90a:0.25}"
TUNED_MOE="${5:-}"   # optional path to E=512,N=512,device_name=AMD_Instinct_MI210,dtype=int4_w4a16.json

VIDEO_GID="$(getent group video  | cut -d: -f3)"
RENDER_GID="$(getent group render | cut -d: -f3)"

MOUNT=()
if [[ -n "${TUNED_MOE}" && -f "${TUNED_MOE}" ]]; then
  # editable install lives at /build/vllm inside the self-built image
  DST="/build/vllm/vllm/model_executor/layers/fused_moe/configs/$(basename "${TUNED_MOE}")"
  MOUNT=(-v "${TUNED_MOE}:${DST}:ro")
  echo "[vllm] using tuned MoE config: ${TUNED_MOE}"
fi

docker rm -f vllm >/dev/null 2>&1 || true
docker run -d --name vllm --network host --restart unless-stopped \
  --device /dev/kfd --device /dev/dri \
  --group-add "${VIDEO_GID}" --group-add "${RENDER_GID}" \
  --security-opt seccomp=unconfined --shm-size 16g \
  -e HIP_VISIBLE_DEVICES="${HIP_DEV}" \
  -e VLLM_ROCM_USE_AITER=0 \
  -v "$(dirname "${MODEL}")":"$(dirname "${MODEL}")" \
  "${MOUNT[@]}" \
  --entrypoint vllm "${IMG}" \
  serve "${MODEL}" --served-model-name qwen3-next \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --port "${PORT}"

echo "launched self-built vllm on port ${PORT}, HIP device ${HIP_DEV}"
